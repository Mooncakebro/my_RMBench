"""
DDP plumbing smoke test with a TINY synthetic model — verifies every DDP
mechanic used by train_compact.py (env-based init, DDP wrap with
find_unused_parameters=True, TBPTT-style window accumulation + detach,
loss all_reduce) WITHOUT loading the 2B VLM (two 2B copies do not fit this
dev machine's 8GB GPU / 15GB RAM).

TinyExecutor mimics Mem0CompactExecutor's TBPTT interface at small scale:
    init_memory / detach_memory / forward(batch, memory) -> (loss_dict, new_memory)

Run (2 ranks; nccl on GPU, gloo fallback on CPU):
    torchrun --nproc_per_node=2 debug/ddp_plumbing_test.py

The real-model DDP path is exercised by the same training entry point with
world_size=1 (torchrun --nproc_per_node=1 source/training/train_compact.py ...).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.training.ddp_utils import (all_reduce_avg, destroy_distributed,
                                       get_ddp_device, is_rank0, rank0_print,
                                       setup_distributed)


class TinyExecutor(nn.Module):
    """TBPTT-interface mimic of Mem0CompactExecutor at tiny scale."""

    def __init__(self, hidden: int = 64, mem_dim: int = 16):
        super().__init__()
        self.mem_dim = mem_dim
        self.vlm = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden))
        self.mem_write = nn.Linear(mem_dim, mem_dim)
        self.head = nn.Linear(hidden, 16)

    def init_memory(self, B: int, device):
        return {
            "m": [torch.randn(B, 4, self.mem_dim, device=device)],
            "p": [torch.rand(B, 4, device=device)],
            "e": [None],
        }

    @staticmethod
    def detach_memory(memory):
        return {
            "m": [t.detach() for t in memory["m"]],
            "p": [t.detach() for t in memory["p"]],
            "e": [t.detach() if isinstance(t, torch.Tensor) else None
                  for t in memory["e"]],
        }

    def forward(self, x: torch.Tensor, memory: dict):
        h = self.vlm(x)                                    # (B, N, hidden)
        m_new = self.mem_write(memory["m"][0])             # carried-state write
        out = self.head(h).mean()                          # scalar-ish loss part
        loss_dict = {
            "total": out + 1e-3 * m_new.pow(2).mean(),
            "action": out,
        }
        new_memory = {"m": [m_new], "p": memory["p"], "e": [None]}
        return loss_dict, new_memory


def main():
    rank, world_size, local_rank, ddp_enabled = setup_distributed()
    device = get_ddp_device(local_rank, world_size)
    torch.manual_seed(42 + rank)

    assert ddp_enabled, "run me with torchrun --nproc_per_node=2"
    rank0_print(f"[ddp-test] rank={rank}/{world_size} device={device}")

    # ── DDP wrap with find_unused_parameters (same as train_compact.py) ──
    raw = TinyExecutor().to(device)
    model = DDP(raw,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True)

    # ── TBPTT-style window loop ──
    B, N, K, frames = 2, 8, 3, 4
    memory = raw.init_memory(B, device)
    for window in range(K):
        window_loss = torch.zeros((), device=device)
        for _ in range(frames):
            x = torch.randn(B, N, 64, device=device)
            loss_dict, memory = model(x, memory)          # forward via DDP
            window_loss = window_loss + loss_dict["total"] / frames
        model.zero_grad(set_to_none=True)
        window_loss.backward()
        memory = raw.detach_memory(memory)                # TBPTT boundary
        assert memory["m"][0].grad_fn is None, "memory not detached"
        all_reduce_avg(window_loss)                       # log-only sync
        rank0_print(f"[ddp-test] window {window}: "
                    f"loss={window_loss.item():.6f} detach=OK")

    # ── Gradient sync check: ranks must have identical grads after step ──
    g = raw.mem_write.weight.grad.detach().clone()
    gathered = [torch.zeros_like(g) for _ in range(world_size)]
    torch.distributed.all_gather(gathered, g)
    rank0 = gathered[0]
    for r in range(1, world_size):
        assert torch.allclose(rank0, gathered[r], atol=1e-6), \
            f"rank {r} grads diverged (DDP sync broken)"
    rank0_print("[ddp-test] gradient sync across ranks verified ✔")

    destroy_distributed()
    rank0_print("[ddp-test] ALL CHECKS PASSED ✔")


if __name__ == "__main__":
    main()

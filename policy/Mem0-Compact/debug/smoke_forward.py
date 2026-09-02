"""
Mem0-Compact smoke test (idea.md §6.1): model builds, synthetic forward +
backward, memory state shapes correct, all losses finite.

No dataset needed — a synthetic batch of PIL images + dummy actions.

Example (from policy/Mem0-Compact/):
    python debug/smoke_forward.py --device cpu
    python debug/smoke_forward.py --device cuda --freeze-base 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf

from source.models.execution_module.mem0_compact_executor import Mem0CompactExecutor


def make_synthetic_batch(B: int = 2, T: int = 30) -> dict:
    img = Image.fromarray(
        np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
    return {
        "image": [[img] for _ in range(B)],
        "lang": ["Swap the positions of the two blocks."] * B,
        "action": torch.randn(B, T, 16),
        "state": torch.randn(B, 1, 16).clamp(-1, 1),
        "prev_action": torch.zeros(B, 16),
        "episode_id": torch.tensor([0, 1]),
        "subtask_end": torch.tensor([0, 0]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--freeze-base", type=int, default=1)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-ckpt", type=int, default=0)
    p.add_argument("--config", type=str,
                   default="source/config/mem0_compact_train.yaml")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = OmegaConf.load(PROJECT_ROOT / args.config)
    cfg.execution_module.compact.freeze_base = bool(args.freeze_base)
    cfg.execution_module.compact.gradient_checkpointing = bool(args.grad_ckpt)

    print(f"[smoke] device={device} freeze_base={args.freeze_base}")
    print("[smoke] building executor (loads Qwen3-VL-2B; slow first time)...")
    model = Mem0CompactExecutor(cfg, device=device).to(device)
    model.eval()
    print(f"[smoke] param counts: {model.trainable_param_counts()}")
    print(f"[smoke] hidden_dim={model.hidden_dim}, layers={model.num_layers}, "
          f"mem_dim={model.mem_dim}, num_mem={model.num_mem}")

    B = args.batch
    memory = model.init_memory(B, device)
    assert len(memory["m"]) == model.num_layers, "memory layer count mismatch"
    for l in range(model.num_layers):
        assert memory["m"][l].shape == (B, model.num_mem, model.mem_dim), \
            f"layer {l} M shape {memory['m'][l].shape}"
        assert memory["p"][l].shape == (B, model.num_mem), \
            f"layer {l} P shape {memory['p'][l].shape}"
    print(f"[smoke] init_memory shapes OK: {memory['m'][0].shape} / {memory['p'][0].shape}")

    batch = make_synthetic_batch(B)
    batch["prev_action"] = torch.zeros(B, 16, device=device)

    model.train()
    loss_dict, new_memory = model.forward_step(batch, memory)
    print(f"[smoke] forward OK — losses:")
    for k, v in loss_dict.items():
        v = v.detach() if isinstance(v, torch.Tensor) else v
        print(f"    {k}: {float(v) if isinstance(v, torch.Tensor) else v:.6f}")

    assert torch.isfinite(loss_dict["total"]), "total loss non-finite"
    for key in ("action", "obs", "nll", "mem_l2"):
        assert torch.isfinite(loss_dict[key]), f"{key} non-finite"

    loss_dict["total"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 10.0)
    print(f"[smoke] backward OK, grad norm = {grad_norm.item():.4f}")

    # Inference path (update_obs_inference + predict_action).
    model.eval()
    with torch.no_grad():
        summary, state_out, mem2 = model.update_obs_inference(batch, memory)
        print(f"[smoke] inference summary={tuple(summary.shape)} "
              f"state={tuple(state_out.shape)}")
        assert summary.shape == (B, 2, model.hidden_dim), "summary shape wrong"
        actions = model.action_model.predict_action(summary, state_out)
        print(f"[smoke] predict_action shape={tuple(actions.shape)}")
        assert actions.shape == (B, model.action_horizon, 16)
        assert torch.isfinite(actions).all(), "non-finite actions"

    # Episode reset: reset row 0, verify it now matches a FRESH memory (not the
    # carried state).
    reset_mask = [True] + [False] * (B - 1)
    memory3 = model.reset_memory_rows(new_memory, reset_mask, device)
    fresh = model.init_memory(B, device)
    is_fresh = torch.equal(memory3["m"][0][0], fresh["m"][0][0])
    print(f"[smoke] reset row0 matches fresh init? {is_fresh} (expect True)")
    assert is_fresh, "episode reset did not re-initialize memory row"
    if B > 1:
        carried_ok = torch.equal(memory3["m"][1][0], new_memory["m"][1][0])
        print(f"[smoke] non-reset row1 unchanged? {carried_ok} (expect True)")
        assert carried_ok, "reset touched a non-reset row"

    print("\n[smoke] ALL CHECKS PASSED ✔")


if __name__ == "__main__":
    main()

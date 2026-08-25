"""
Smoke test: real Qwen3-VL-2B (bf16) + tiny DiT, FREEZE_BASE=1, 2 consecutive
frames of swap_blocks episode 0, for BOTH variants.

Asserts:
  - state_images (B, 32, 2048), goal (B, 1, 512), finite diffusion loss
  - backward through diffusion (+ aux for COMPACT) works
  - COMPACT: memory state changes between frame 0 and frame 1, shapes
    (num_layers, 16, 512)

Run (from policy/CompactMoDE/):
    bash debug/run_smoke.sh
or:
    python debug/smoke_forward.py --variant both --embed-dim 256 --n-layers 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))

from compact_mode.config import CompactMoDEConfig
from compact_mode.dataset import DEFAULT_DATA_ROOT, FrameDataset, collate_frames


def get_frames(n=2, task="swap_blocks", data_root=DEFAULT_DATA_ROOT):
    dataset = FrameDataset(Path(data_root) / task, action_horizon=10, normalize=True)
    samples = [dataset[i] for i in range(n)]
    assert all(s["episode_index"] == 0 for s in samples), "expected episode 0 frames"
    return samples, dataset


def tiny_cfg(**over):
    base = dict(
        freeze_base=True, bf16=True, gradient_checkpointing=True,
        embed_dim=256, n_layers=2, n_heads=4, num_experts=4, top_k=2,
        attn_pdrop=0.0, mlp_pdrop=0.0, goal_drop=0.0,
    )
    base.update(over)
    return CompactMoDEConfig(**base)


def report_params(model, counts):
    total_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_all = sum(p.numel() for p in model.parameters())
    print(f"  params: trainable {total_train/1e6:.1f}M / total {total_all/1e6:.1f}M")
    for k, v in counts.items():
        print(f"    {k}: {v/1e6:.2f}M")


def check_cond(cond, B):
    si, goal, ro = cond["state_images"], cond["goal"], cond["robot_obs"]
    assert si.shape == (B, 32, 2048), f"state_images {tuple(si.shape)}"
    assert goal.shape == (B, 1, 512), f"goal {tuple(goal.shape)}"
    assert ro.shape == (B, 14), f"robot_obs {tuple(ro.shape)}"
    assert torch.isfinite(si).all() and torch.isfinite(goal).all()
    print(f"  cond: state_images {tuple(si.shape)} goal {tuple(goal.shape)} "
          f"robot_obs {tuple(ro.shape)}")


def run_baseline(samples, device):
    from compact_mode.model_baseline import BaselineQwenMoDEPolicy
    print("=== Baseline (Qwen3-VL-2B frozen + bridge + tiny DiT) ===")
    torch.cuda.reset_peak_memory_stats()
    model = BaselineQwenMoDEPolicy(tiny_cfg()).to(device)
    report_params(model, model.trainable_param_counts())

    for t, s in enumerate(samples):
        batch = collate_frames([s])
        cond, aux = model.forward_step(batch)
        check_cond(cond, 1)
        loss = model.diffusion_loss(batch)
        assert loss.ndim == 0 and torch.isfinite(loss).all(), f"loss {loss}"
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        assert torch.isfinite(grad_norm), "non-finite grad norm"
        model.zero_grad(set_to_none=True)
        print(f"  frame {t}: diffusion loss {loss.item():.4f}, "
              f"grad_norm {grad_norm.item():.2f}")

    # sampling path
    acts = model.sample_actions(collate_frames([samples[0]]), steps=4)
    assert acts.shape == (1, 10, 14) and torch.isfinite(acts).all()
    print(f"  sample_actions: {tuple(acts.shape)}")
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"  peak GPU mem: {peak:.2f} GiB")
    del model
    torch.cuda.empty_cache()


def run_compact(samples, device):
    from compact_mode.model_compact import CompactMoDEPolicy
    print("=== COMPACT+MoDE (frozen Qwen3-VL-2B + 28 side memories + bridge + tiny DiT) ===")
    torch.cuda.reset_peak_memory_stats()
    model = CompactMoDEPolicy(tiny_cfg()).to(device)
    report_params(model, model.trainable_param_counts())

    memory = model.init_memory(1, device)
    prev_m = None
    for t, s in enumerate(samples):
        batch = collate_frames([s])
        cond, memory, aux = model.forward_step(batch, memory)
        check_cond(cond, 1)
        m0 = memory["m"]
        assert len(m0) == model.num_layers == 28, f"layers {len(m0)}"
        assert m0[0].shape == (1, 16, 512), f"memory shape {tuple(m0[0].shape)}"
        if prev_m is not None:
            delta = (m0[0].float() - prev_m[0].float()).abs().max().item()
            assert delta > 0, "memory did not change between frames"
            print(f"  memory changed between frames (max |delta| layer0: {delta:.4f})")
        prev_m = [t_.detach().clone() for t_ in m0]

        loss, loss_dict, memory = model.diffusion_loss(batch, memory)
        assert loss.ndim == 0 and torch.isfinite(loss).all()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        assert torch.isfinite(grad_norm), "non-finite grad norm"
        model.zero_grad(set_to_none=True)
        memory = model.detach_memory(memory)
        print(f"  frame {t}: total {loss.item():.4f} "
              + " ".join(f"{k} {float(v):.4f}" for k, v in loss_dict.items())
              + f" grad_norm {grad_norm.item():.2f}")

    acts, memory = model.sample_actions(collate_frames([samples[0]]), memory, steps=4)
    assert acts.shape == (1, 10, 14) and torch.isfinite(acts).all()
    print(f"  sample_actions: {tuple(acts.shape)}")
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"  peak GPU mem: {peak:.2f} GiB")
    del model
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("baseline", "compact", "both"), default="both")
    p.add_argument("--task", type=str, default="swap_blocks")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--n-frames", type=int, default=2)
    args = p.parse_args()

    device = torch.device("cuda")
    samples, _ = get_frames(args.n_frames, args.task, args.data_root)
    print(f"Loaded {len(samples)} frames of {args.task} episode 0; "
          f"instruction: {samples[0]['instruction'][:80]!r}")

    if args.variant in ("baseline", "both"):
        run_baseline(samples, device)
    if args.variant in ("compact", "both"):
        run_compact(samples, device)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

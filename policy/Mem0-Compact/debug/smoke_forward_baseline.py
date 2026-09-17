"""
Mem0-Compact BASELINE smoke test: model builds, synthetic forward + backward,
no-memory API no-ops behave, all losses finite.

No dataset needed — a synthetic batch of PIL images + dummy actions.

Example (from policy/Mem0-Compact/):
    python debug/smoke_forward_baseline.py --device cpu
    python debug/smoke_forward_baseline.py --device cuda --freeze-base 1
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

from source.models.execution_module.mem0_baseline_executor import Mem0BaselineExecutor
from debug.smoke_forward import make_synthetic_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--freeze-base", type=int, default=1)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-ckpt", type=int, default=0)
    p.add_argument("--config", type=str,
                   default="source/config/mem0_baseline_train.yaml")
    p.add_argument("--model-path", type=str, default=None,
                   help="override execution_module.qwen_vl.model_path")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = OmegaConf.load(PROJECT_ROOT / args.config)
    cfg.execution_module.compact.freeze_base = bool(args.freeze_base)
    cfg.execution_module.compact.gradient_checkpointing = bool(args.grad_ckpt)
    if args.model_path:
        cfg.execution_module.qwen_vl.model_path = args.model_path
    assert cfg.execution_module.variant == "baseline", "config must be baseline"

    print(f"[smoke-baseline] device={device} freeze_base={args.freeze_base}")
    print("[smoke-baseline] building executor (loads Qwen3-VL-2B; slow first time)...")
    model = Mem0BaselineExecutor(cfg, device=device).to(device)
    model.eval()
    print(f"[smoke-baseline] param counts: {model.trainable_param_counts()}")
    print(f"[smoke-baseline] hidden_dim={model.hidden_dim}")

    B = args.batch
    memory = model.init_memory(B, device)
    assert memory == {}, "baseline init_memory must be empty"
    memory = model.reset_memory_rows(memory, [True] * B, device)
    assert memory == {}, "baseline reset_memory_rows must be a no-op"
    memory = model.detach_memory(memory)
    assert memory == {}, "baseline detach_memory must be a no-op"
    print("[smoke-baseline] memory no-op API OK ({})")

    batch = make_synthetic_batch(B)
    batch["prev_action"] = torch.zeros(B, 16, device=device)

    model.train()
    loss_dict, new_memory = model(batch, memory)  # DDP-style forward path
    assert new_memory == {}, "baseline forward must return empty memory"
    print("[smoke-baseline] forward OK — losses:")
    for k, v in loss_dict.items():
        v = v.detach() if isinstance(v, torch.Tensor) else v
        print(f"    {k}: {float(v) if isinstance(v, torch.Tensor) else v:.6f}")

    assert torch.isfinite(loss_dict["total"]), "total loss non-finite"
    assert set(loss_dict) <= {"total", "action", "classifier", "cls_accuracy"}, \
        f"unexpected loss keys (aux losses must not exist): {sorted(loss_dict)}"

    loss_dict["total"].backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 10.0)
    print(f"[smoke-baseline] backward OK, grad norm = {grad_norm.item():.4f}")

    # Inference path (update_obs_inference + predict_action).
    model.eval()
    with torch.no_grad():
        summary, state_out, mem2 = model.update_obs_inference(batch, memory)
        assert mem2 == {}, "baseline inference must return empty memory"
        print(f"[smoke-baseline] inference summary={tuple(summary.shape)} "
              f"state={tuple(state_out.shape)}")
        assert summary.shape == (B, 2, model.hidden_dim), "summary shape wrong"
        actions = model.action_model.predict_action(summary, state_out)
        print(f"[smoke-baseline] predict_action shape={tuple(actions.shape)}")
        assert actions.shape == (B, model.action_horizon, 16)
        assert torch.isfinite(actions).all(), "non-finite actions"

    # Optimizer groups: no memory group must exist.
    groups = model.get_optim_groups(1e-5, 1e-4, 5e-6, 0.005)
    names = [g["name"] for g in groups]
    print(f"[smoke-baseline] optim groups: {names}")
    assert "memory" not in names, "baseline must not have a memory param group"

    print("\n[smoke-baseline] ALL CHECKS PASSED ✔")


if __name__ == "__main__":
    main()

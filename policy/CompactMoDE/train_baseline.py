"""
Train the baseline variant (Qwen3-VL-2B + bridge + MoDeDiT, stateless).

Random-batch training over frames; AdamW param groups via
policy.get_optim_groups(); grad clip; cosine or constant LR with warmup;
optional val loop. Checkpoints store {model, optimizer, step, config}.

Single-process (DDP-ready structure but intentionally simple; see README).

Example:
    python train_baseline.py --task swap_blocks --freeze-base 1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from compact_mode.config import CompactMoDEConfig
from compact_mode.dataset import DEFAULT_DATA_ROOT, FrameDataset, collate_frames
from compact_mode.model_baseline import BaselineQwenMoDEPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default=os.environ.get("TASK", "swap_blocks"))
    p.add_argument("--data-root", type=Path, default=Path(os.environ.get(
        "DATA_ROOT", str(DEFAULT_DATA_ROOT))))
    p.add_argument("--output-dir", type=Path, default=Path(os.environ.get(
        "OUTPUT_DIR", str(THIS_DIR / "runs" / "baseline"))))
    p.add_argument("--base-model", type=str, default=os.environ.get(
        "BASE_MODEL", "Qwen/Qwen3-VL-2B-Instruct"))
    p.add_argument("--freeze-base", type=int, default=int(os.environ.get("FREEZE_BASE", "0")))
    p.add_argument("--grad-ckpt", type=int, default=int(os.environ.get("GRAD_CKPT", "0")))
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "10")))
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "0")),
                   help=">0 overrides epochs")
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")))
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("NUM_WORKERS", "4")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-4")))
    p.add_argument("--base-lr", type=float, default=float(os.environ.get("BASE_LR", "1e-5")))
    p.add_argument("--weight-decay", type=float, default=float(os.environ.get("WD", "0.01")))
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embed-dim", type=int, default=int(os.environ.get("EMBED_DIM", "1024")))
    p.add_argument("--n-layers", type=int, default=int(os.environ.get("N_LAYERS", "6")))
    p.add_argument("--n-heads", type=int, default=int(os.environ.get("N_HEADS", "8")))
    p.add_argument("--num-experts", type=int, default=int(os.environ.get("NUM_EXPERTS", "4")))
    p.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "2")))
    p.add_argument("--val-fraction", type=float, default=float(os.environ.get("VAL_FRACTION", "0.0")))
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--val-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=int(os.environ.get("SAVE_STEPS", "1000")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_lr_lambda(total_steps: int, warmup_ratio: float, schedule: str):
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        if schedule == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def save_checkpoint(path, model, optimizer, step, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg.to_dict(),
    }, path)


def main():
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "args.json").open("w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2)

    cfg = CompactMoDEConfig(
        base_model_name=args.base_model,
        freeze_base=bool(args.freeze_base),
        gradient_checkpointing=bool(args.grad_ckpt),
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        num_experts=args.num_experts,
        top_k=args.top_k,
        lr=args.lr,
        base_lr=args.base_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.grad_clip,
        seed=args.seed,
    )

    dataset = FrameDataset(args.data_root / args.task, action_horizon=cfg.action_seq_len,
                           normalize=cfg.normalize)
    model = BaselineQwenMoDEPolicy(cfg).to(device)
    model.set_normalizers(dataset.action_normalizer, dataset.state_normalizer)
    counts = model.trainable_param_counts()
    print(f"[train] param counts: {counts}")

    if args.val_fraction > 0:
        val_len = max(1, int(len(dataset) * args.val_fraction))
        train_set, val_set = random_split(
            dataset, [len(dataset) - val_len, val_len],
            generator=torch.Generator().manual_seed(args.seed))
    else:
        train_set, val_set = dataset, None
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=device.type == "cuda", collate_fn=collate_frames,
                         persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs) if val_set else None

    optimizer = AdamW(model.get_optim_groups(cfg.lr, cfg.base_lr, cfg.memory_lr,
                                             cfg.weight_decay), lr=cfg.lr)
    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps if args.max_steps > 0 else steps_per_epoch * args.epochs
    scheduler = LambdaLR(optimizer, make_lr_lambda(total_steps, cfg.warmup_ratio,
                                                   args.lr_schedule))

    start_step = 0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"[train] resumed from {args.resume} at step {start_step}")

    model.train()
    step = start_step
    done = False
    for epoch in range(args.epochs):
        if done:
            break
        for batch in train_loader:
            t0 = time.time()
            loss = model.diffusion_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.log_steps == 0:
                print(f"[train] step {step}/{total_steps} epoch {epoch} "
                      f"loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                      f"({time.time() - t0:.2f}s)")
            if val_loader is not None and step % args.val_steps == 0:
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for i, vbatch in enumerate(val_loader):
                        if i >= 4:
                            break
                        val_losses.append(model.diffusion_loss(vbatch).item())
                model.train()
                print(f"[val] step {step} loss {sum(val_losses) / len(val_losses):.4f}")
            if step % args.save_steps == 0:
                save_checkpoint(args.output_dir / f"ckpt_{step:07d}.pt", model, optimizer, step, cfg)
            if args.max_steps > 0 and step >= args.max_steps:
                done = True
                break

    save_checkpoint(args.output_dir / "ckpt_final.pt", model, optimizer, step, cfg)
    model.save_pretrained(args.output_dir / "final")
    print(f"[train] done. {step} steps. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()

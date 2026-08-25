"""
Train the COMPACT+MoDE variant with episode-sequential chunked TBPTT.

Each optimizer step processes one chunk = `num_streams` parallel episode
streams x `chunk_size` consecutive frames. Frames are forwarded one at a time
per stream position (VLM runs EVERY frame, stride 1), memory is carried across
frames inside the chunk, the chunk-mean loss is backpropagated once, and the
carried memory is detached at chunk boundaries (same convention as
jamel_compact/train.py). Streams that enter a new episode get their memory
rows re-initialized.

Aux losses: COMPACT L_obs / L_nll / memory-L2 (weights from CompactConfig via
CompactMoDEConfig); no text CE.

Single-process (DDP-ready structure but intentionally simple; see README).

Example:
    python train_compact.py --task swap_blocks --freeze-base 1 --num-streams 2
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

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from compact_mode.config import CompactMoDEConfig
from compact_mode.dataset import (DEFAULT_DATA_ROOT, EpisodeStreamSampler,
                                  FrameDataset, collate_frames)
from compact_mode.model_compact import CompactMoDEPolicy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default=os.environ.get("TASK", "swap_blocks"))
    p.add_argument("--data-root", type=Path, default=Path(os.environ.get(
        "DATA_ROOT", str(DEFAULT_DATA_ROOT))))
    p.add_argument("--output-dir", type=Path, default=Path(os.environ.get(
        "OUTPUT_DIR", str(THIS_DIR / "runs" / "compact"))))
    p.add_argument("--base-model", type=str, default=os.environ.get(
        "BASE_MODEL", "Qwen/Qwen3-VL-2B-Instruct"))
    p.add_argument("--freeze-base", type=int, default=int(os.environ.get("FREEZE_BASE", "0")))
    p.add_argument("--grad-ckpt", type=int, default=int(os.environ.get("GRAD_CKPT", "0")))
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "10000")),
                   help="optimizer steps (one per chunk)")
    p.add_argument("--num-streams", type=int, default=int(os.environ.get("NUM_STREAMS", "1")),
                   help="parallel episode streams (batch size per frame step)")
    p.add_argument("--chunk-size", type=int, default=int(os.environ.get("CHUNK_SIZE", "8")))
    p.add_argument("--lr", type=float, default=float(os.environ.get("LR", "1e-4")))
    p.add_argument("--base-lr", type=float, default=float(os.environ.get("BASE_LR", "1e-5")))
    p.add_argument("--memory-lr", type=float, default=float(os.environ.get("MEMORY_LR", "5e-6")))
    p.add_argument("--weight-decay", type=float, default=float(os.environ.get("WD", "0.01")))
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--lr-schedule", choices=("cosine", "constant"), default="cosine")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embed-dim", type=int, default=int(os.environ.get("EMBED_DIM", "1024")))
    p.add_argument("--n-layers", type=int, default=int(os.environ.get("N_LAYERS", "6")))
    p.add_argument("--n-heads", type=int, default=int(os.environ.get("N_HEADS", "8")))
    p.add_argument("--num-experts", type=int, default=int(os.environ.get("NUM_EXPERTS", "4")))
    p.add_argument("--top-k", type=int, default=int(os.environ.get("TOP_K", "2")))
    p.add_argument("--log-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=int(os.environ.get("SAVE_STEPS", "1000")))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--mem-opt", choices=("adamw", "sgd"),
                   default=os.environ.get("MEM_OPT", "adamw"),
                   help="optimizer for the side-memory param group. adamw (default, "
                        "use on the server); sgd is a small-GPU debug escape hatch "
                        "(AdamW m/v states for 300M memory params do not fit on 8GB).")
    p.add_argument("--max-frames-per-episode", type=int, default=0,
                   help=">0 caps frames per episode (debug)")
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


def save_checkpoint(path, model, optimizers, step, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    torch.save({
        "model": model.state_dict(),
        "optimizer": [o.state_dict() for o in optimizers],
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
        memory_lr=args.memory_lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.grad_clip,
        chunk_size=args.chunk_size,
        seed=args.seed,
    )

    dataset = FrameDataset(args.data_root / args.task, action_horizon=cfg.action_seq_len,
                           normalize=cfg.normalize)
    sampler = EpisodeStreamSampler(
        dataset, chunk_size=cfg.chunk_size, num_streams=args.num_streams,
        shuffle_episodes=True, seed=args.seed,
        max_frames_per_episode=args.max_frames_per_episode or None)

    model = CompactMoDEPolicy(cfg).to(device)
    model.set_normalizers(dataset.action_normalizer, dataset.state_normalizer)
    counts = model.trainable_param_counts()
    print(f"[train] param counts: {counts}")

    groups = model.get_optim_groups(cfg.lr, cfg.base_lr, cfg.memory_lr,
                                    cfg.weight_decay)
    mem_groups = [g for g in groups if g["name"] == "memory"]
    other_groups = [g for g in groups if g["name"] != "memory"]
    optimizer = AdamW(other_groups, lr=cfg.lr)
    optimizers = [optimizer]
    if args.mem_opt == "sgd" and mem_groups:
        optimizers.append(torch.optim.SGD(
            [p for g in mem_groups for p in g["params"]], lr=cfg.memory_lr))
    else:
        optimizers.append(AdamW(mem_groups, lr=cfg.memory_lr))
    schedulers = [LambdaLR(o, make_lr_lambda(args.max_steps, cfg.warmup_ratio,
                                             args.lr_schedule)) for o in optimizers]

    start_step = 0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        for o, s in zip(optimizers, ckpt["optimizer"]):
            o.load_state_dict(s)
        start_step = ckpt["step"]
        print(f"[train] resumed from {args.resume} at step {start_step}")

    model.train()
    memory = model.init_memory(args.num_streams, device)
    for step in range(start_step, args.max_steps):
        t0 = time.time()
        chunk, resets = sampler.next_chunk()
        memory = model.reset_memory_rows(memory, resets, device)

        chunk_loss = 0.0
        log_parts = {}
        for s in range(cfg.chunk_size):
            batch = collate_frames([dataset[i] for i in (row[s] for row in chunk)])
            loss, loss_dict, memory = model.diffusion_loss(batch, memory)
            chunk_loss = chunk_loss + loss / cfg.chunk_size
            if s == cfg.chunk_size - 1:
                log_parts = {k: float(v) for k, v in loss_dict.items()}

        for o in optimizers:
            o.zero_grad(set_to_none=True)
        chunk_loss.backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], cfg.max_grad_norm)
        for o, s in zip(optimizers, schedulers):
            o.step()
            s.step()

        # TBPTT: detach carried state at the chunk boundary.
        memory = model.detach_memory(memory)

        if (step + 1) % args.log_steps == 0:
            parts = " ".join(f"{k} {v:.4f}" for k, v in log_parts.items())
            print(f"[train] step {step + 1}/{args.max_steps} loss {chunk_loss.item():.4f} "
                  f"({parts}) lr {schedulers[0].get_last_lr()[0]:.2e} ({time.time() - t0:.2f}s)")
        if (step + 1) % args.save_steps == 0:
            save_checkpoint(args.output_dir / f"ckpt_{step + 1:07d}.pt",
                            model, optimizers, step + 1, cfg)

    save_checkpoint(args.output_dir / "ckpt_final.pt", model, optimizers, args.max_steps, cfg)
    model.save_pretrained(args.output_dir / "final")
    print(f"[train] done. {args.max_steps} steps. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()

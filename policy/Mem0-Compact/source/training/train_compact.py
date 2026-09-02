"""
Mem0-Compact TBPTT training loop (single process; DDP-ready structure).

Per idea.md §3:
  - frames iterate sequentially within episodes via RandomEpisodeIterableDataset
  - memory (M, P, e per layer) + prev_action are threaded across batches
  - loss accumulates over a K-frame window; ONE backward per window
  - memory is detached at window boundaries (gradients never cross windows)
  - when a batch slot's episode_id changes, that slot's memory is re-initialized
    and its prev_action reset to zeros (episode start; no reset at subtask
    boundaries)
  - prev_action[slot] = normalized state of the current frame (Mem-0 dataset
    has action[t] = state[t+1], so the last executed action at frame t is
    exactly state[t])

Loss per frame:
    L = 1.0 * L_flow + 0.2 * L_cls (M(n) only)
        + 0.01 * L_obs + 0.01 * L_nll + 0.001 * L_mem

Example:
    python source/training/train_compact.py \
        --config source/config/mem0_compact_train.yaml \
        --task swap_blocks --freeze-base 1 --max-steps 20
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
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from termcolor import cprint

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.dataloader.dataset_min_max import LeRobot_Dataset
from source.dataloader.random_episode_dataloader import RandomEpisodeIterableDataset
from source.models.execution_module.mem0_compact_executor import Mem0CompactExecutor


def collate_batch(samples: list) -> dict:
    batch = {}
    batch["image"] = [s["image"] for s in samples]          # List[List[PIL]]
    batch["lang"] = [s["lang"] for s in samples]            # List[str]
    batch["action"] = torch.stack([s["action"] for s in samples])   # (B,T,16)
    batch["state"] = torch.stack([s["state"] for s in samples])     # (B,1,16)
    batch["episode_id"] = torch.tensor([int(s["episode_id"]) for s in samples])
    batch["subtask_end"] = torch.tensor([int(s["subtask_end"]) for s in samples])
    batch["episode_pos"] = torch.tensor([int(s["episode_pos"]) for s in samples])
    return batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str,
                   default="source/config/mem0_compact_train.yaml")
    p.add_argument("--task", type=str, default=None,
                   help="override vla_dataset repo task dir (uses lerobot_datasets/<task>)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--window-size", type=int, default=None,
                   help="TBPTT window K override (small-GPU debug: use 1)")
    p.add_argument("--opt-sgd", action="store_true",
                   help="use plain SGD instead of AdamW (small-GPU debug "
                        "escape hatch; AdamW states don't fit on 8GB)")
    p.add_argument("--freeze-base", type=int, default=None,
                   help="1 freezes the Qwen base (env FREEZE_BASE also honored)")
    p.add_argument("--grad-ckpt", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_lr_lambda(total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def build_dataloader(cfg, task: str, rank: int = 0, world_size: int = 1,
                     batch_size: int = None, num_workers: int = None):
    trainer_cfg = cfg.get("trainer", {})
    batch_size = batch_size or int(trainer_cfg.get("batch_size", 8))
    num_workers = num_workers if num_workers is not None else int(
        cfg.get("dataloader", {}).get("num_workers", 0))
    action_horizon = int(cfg.execution_module.action_model.get("action_horizon", 30))

    repo_id = cfg.vla_dataset.RMBench.get("repo_id", "")
    root = None
    if task:
        # Local lerobot dataset: root points AT the dataset dir (lerobot 0.4.4)
        repo_id = task
        root = str(PROJECT_ROOT / "lerobot_datasets" / task)
    elif repo_id and not str(repo_id).startswith(("http", "hf://")):
        # Full path given: use it directly as root
        p = Path(str(repo_id)).expanduser()
        repo_id = p.name
        root = str(p)
    cprint(f"[dataloader] repo_id={repo_id} root={root} batch={batch_size} workers={num_workers}", "cyan")

    features_to_load = cfg.vla_dataset.RMBench.get("features_to_load", [
        "observation.state", "action", "subtask", "subtask_end",
        "episode_id", "observation.image.head_camera",
    ])
    norm_stats_path = trainer_cfg.get("norm_stats_path", None)
    base_dataset = LeRobot_Dataset(
        repo_id=repo_id,
        root=root,
        features_to_load=list(features_to_load),
        action_horizon=action_horizon,
        norm_stats_path=norm_stats_path,
    )
    iterable = RandomEpisodeIterableDataset(
        base_dataset=base_dataset,
        rank=rank, world_size=world_size,
        shuffle=bool(cfg.get("dataloader", {}).get("shuffle_episodes", True)),
        seed=int(cfg.get("dataloader", {}).get("seed", cfg.get("seed", 42))),
        infinite=bool(cfg.get("dataloader", {}).get("infinite", True)),
    )
    loader = DataLoader(iterable, batch_size=batch_size,
                        num_workers=num_workers, collate_fn=collate_batch)
    return loader


def save_checkpoint(path: Path, model, optimizer, step, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, path)


def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    cfg = OmegaConf.load(cfg_path)

    if args.freeze_base is not None:
        cfg.execution_module.compact.freeze_base = bool(args.freeze_base)
    if args.grad_ckpt is not None:
        cfg.execution_module.compact.gradient_checkpointing = bool(args.grad_ckpt)
    trainer_cfg = cfg.trainer
    if args.max_steps is not None:
        trainer_cfg.max_steps = int(args.max_steps)
    max_steps = int(trainer_cfg.get("max_steps", 100))
    batch_size = args.batch_size or int(trainer_cfg.get("batch_size", 8))
    log_interval = int(trainer_cfg.get("log_interval", 5))
    save_every = int(trainer_cfg.get("save_every_steps", 100))

    # TBPTT window
    tbptt_cfg = cfg.get("tbptt", {})
    window_size = args.window_size or int(tbptt_cfg.get("window_size", 8))
    grad_clip = float(trainer_cfg.get("grad_clip_norm", 2.5))
    warmup_ratio = float(trainer_cfg.get("warmup_ratio", 0.05))

    lr_cfg = trainer_cfg.get("learning_rate", {})
    base_lr = float(lr_cfg.get("qwen_model", 1e-5))
    head_lr = float(lr_cfg.get("action_model", 1e-4))
    memory_lr = float(lr_cfg.get("memory", 5e-6))
    weight_decay = float(trainer_cfg.get("weight_decay", 0.005))

    output_dir = Path(args.output_dir or trainer_cfg.get(
        "checkpoint_dir", str(PROJECT_ROOT / "runs" / "compact")))
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    cprint(f"[train] output_dir={output_dir}", "green")

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Model ──
    cprint("[train] building executor (this loads Qwen3-VL-2B)...", "cyan")
    model = Mem0CompactExecutor(cfg, device=device).to(device)
    counts = model.trainable_param_counts()
    cprint(f"[train] param counts: {counts}", "cyan")

    groups = model.get_optim_groups(base_lr, head_lr, memory_lr, weight_decay)
    if args.opt_sgd:
        cprint("[train] using plain SGD (small-GPU debug)", "yellow")
        optimizer = torch.optim.SGD(groups)
    else:
        optimizer = AdamW(groups)
    scheduler = LambdaLR(optimizer, make_lr_lambda(max_steps, warmup_ratio))

    start_step = 0
    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        cprint(f"[train] resumed from {args.resume} at step {start_step}", "green")

    # ── Data ──
    loader = build_dataloader(cfg, args.task, batch_size=batch_size,
                              num_workers=args.num_workers)
    data_iter = iter(loader)

    # ── TBPTT state ──
    model.train()
    memory = model.init_memory(batch_size, device)
    prev_episode = torch.full((batch_size,), -1, dtype=torch.long, device=device)
    total_iters = max_steps * window_size

    cprint(f"[train] TBPTT window={window_size}, steps={max_steps}, "
           f"batch={batch_size}, iters={total_iters}", "cyan")

    step_times = []
    for step in range(start_step, max_steps):
        t0 = time.time()
        window_loss = torch.zeros((), device=device, dtype=torch.float32)
        log_parts = {}
        for k in range(window_size):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            ep_ids = batch["episode_id"].to(device)

            # Per-slot episode reset: memory + prev_action + episode tracker.
            reset_mask = [(prev_episode[i] >= 0 and prev_episode[i] != ep_ids[i])
                          for i in range(batch_size)]
            memory = model.reset_memory_rows(memory, reset_mask, device)
            prev_action = batch["state"].squeeze(1).to(device).clone().detach()
            for i in range(batch_size):
                if reset_mask[i]:
                    prev_action[i] = 0.0
            prev_episode = ep_ids.clone()
            batch["prev_action"] = prev_action

            loss_dict, memory = model.forward_step(batch, memory)
            window_loss = window_loss + loss_dict["total"] / window_size
            if k == window_size - 1:
                log_parts = {key: float(value.detach())
                             for key, value in loss_dict.items()
                             if isinstance(value, torch.Tensor)}

        if not torch.isfinite(window_loss):
            cprint(f"[train] step {step+1}: non-finite window loss, skipping", "red")
            optimizer.zero_grad(set_to_none=True)
            memory = model.detach_memory(memory)
            continue

        optimizer.zero_grad(set_to_none=True)
        window_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip)
        optimizer.step()
        scheduler.step()

        # TBPTT: gradients never cross windows.
        memory = model.detach_memory(memory)

        step_times.append(time.time() - t0)
        if (step + 1) % log_interval == 0 or step == start_step:
            parts = " ".join(f"{k} {v:.4f}" for k, v in log_parts.items())
            avg_t = sum(step_times[-log_interval:]) / len(step_times[-log_interval:])
            cprint(f"[train] step {step+1}/{max_steps} loss {window_loss.item():.4f} "
                   f"({parts}) lr {scheduler.get_last_lr()[0]:.2e} ({avg_t:.2f}s/window)",
                   "cyan")
        if save_every > 0 and (step + 1) % save_every == 0:
            save_checkpoint(output_dir / f"ckpt_{step+1:07d}.pt",
                            model, optimizer, step + 1, cfg)

    save_checkpoint(output_dir / "ckpt_final.pt", model, optimizer, max_steps, cfg)
    cprint(f"[train] done. {max_steps} windows. Saved to {output_dir}", "green")


if __name__ == "__main__":
    main()

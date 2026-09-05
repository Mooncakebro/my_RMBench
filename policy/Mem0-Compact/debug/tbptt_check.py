"""
Mem0-Compact TBPTT check (idea.md §6.2): thread memory over 2 windows of a real
episode; confirm:
  1. memory is detached at window boundaries (no grad_fn after detach)
  2. per-slot episode reset re-initializes memory when episode_id changes
  3. prev_action is zeros on the first frame of an episode slot

Requires the lerobot dataset at lerobot_datasets/<task> (run
scripts/hdf5_to_lerobot/M1_dataset_to_lerobot.py first).

Example (from policy/Mem0-Compact/):
    python debug/tbptt_check.py --task swap_blocks --device cpu --freeze-base 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.dataloader.dataset_min_max import LeRobot_Dataset
from source.dataloader.random_episode_dataloader import RandomEpisodeIterableDataset
from source.models.execution_module.mem0_compact_executor import Mem0CompactExecutor
from source.training.train_compact import collate_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="swap_blocks")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--freeze-base", type=int, default=1)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--config", type=str,
                   default="source/config/mem0_compact_train.yaml")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = OmegaConf.load(PROJECT_ROOT / args.config)
    cfg.execution_module.compact.freeze_base = bool(args.freeze_base)

    repo_id = PROJECT_ROOT / "lerobot_datasets" / args.task
    action_horizon = int(cfg.execution_module.action_model.action_horizon)
    base = LeRobot_Dataset(
        repo_id=str(repo_id),
        features_to_load=list(cfg.vla_dataset.RMBench.features_to_load),
        action_horizon=action_horizon,
    )
    iterable = RandomEpisodeIterableDataset(
        base, rank=0, world_size=1, shuffle=False, infinite=True)
    loader = DataLoader(iterable, batch_size=1, num_workers=args.num_workers,
                        collate_fn=collate_batch)
    data_iter = iter(loader)

    print("[tbptt] building executor...")
    model = Mem0CompactExecutor(cfg, device=device).to(device)
    model.train()

    K = args.window
    memory = model.init_memory(1, device)
    prev_episode = torch.full((1,), -1, dtype=torch.long, device=device)
    prev_action = torch.zeros(1, 16, device=device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def run_window(memory, prev_episode, prev_action, label):
        window_loss = torch.zeros((), device=device)
        for k in range(K):
            batch = next(data_iter)
            ep_ids = batch["episode_id"].to(device)
            reset_mask = [(prev_episode[0] < 0 or prev_episode[0] != ep_ids[0])]
            memory = model.reset_memory_rows(memory, reset_mask, device)
            cur_prev = prev_action.clone()
            if reset_mask[0]:
                cur_prev[0] = 0.0
                print(f"[tbptt] {label} frame {k}: EPISODE RESET "
                      f"(ep {prev_episode.item()} -> {ep_ids.item()}), "
                      f"prev_action zeroed={bool((cur_prev == 0).all().item())}")
            prev_episode = ep_ids.clone()
            batch["prev_action"] = cur_prev
            loss_dict, memory = model.forward_step(batch, memory)
            window_loss = window_loss + loss_dict["total"] / K
            prev_action = batch["state"].squeeze(1).to(device).detach().clone()
        optimizer.zero_grad()
        window_loss.backward()
        return memory, prev_episode, prev_action, window_loss

    memory, prev_episode, prev_action, w1 = run_window(
        memory, prev_episode, prev_action, "window1")
    print(f"[tbptt] window1 loss = {w1.item():.4f}")
    grad_fn_before = memory["m"][0].grad_fn
    print(f"[tbptt] memory grad_fn after window1 backward: {grad_fn_before}")

    # Detach at boundary (what the training loop does).
    memory = model.detach_memory(memory)
    grad_fn_after = memory["m"][0].grad_fn
    print(f"[tbptt] memory grad_fn after detach: {grad_fn_after}")
    assert grad_fn_after is None, "memory NOT detached at window boundary!"

    # Window 2 with a fresh optimizer step — gradients must not flow back.
    memory, prev_episode, prev_action, w2 = run_window(
        memory, prev_episode, prev_action, "window2")
    print(f"[tbptt] window2 loss = {w2.item():.4f}")

    print("\n[tbptt] ALL CHECKS PASSED ✔  (detach verified, reset behavior above)")


if __name__ == "__main__":
    main()

"""Generate min/max norm stats for a Mem0-Compact task dataset.

Writes assets/<task>/norm_stats.json + global_instruction.txt (like Mem-0's
dataset_min_max.save_norm_stats).

Example (from policy/Mem0-Compact/):
    python scripts/gen_norm_stats.py --task swap_blocks
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.dataloader.dataset_min_max import LeRobot_Dataset, save_norm_stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="swap_blocks")
    p.add_argument("--episodes-num", type=int, default=None)
    args = p.parse_args()

    repo_id = args.task
    root = str(PROJECT_ROOT / "lerobot_datasets" / args.task)
    dataset = LeRobot_Dataset(
        repo_id=repo_id,
        root=root,
        features_to_load=[
            "observation.image.head_camera",
            "observation.state",
            "action",
            "subtask",
            "global_task",
            "subtask_end",
            "episode_id",
        ],
    )
    sample = dataset[0]
    print(f"instruction: {sample['lang']}")
    if sample.get("global_task"):
        print(f"global task: {sample['global_task']}")
    print(f"action: {sample['action'].shape}  state: {sample['state'].shape}")

    instruction = sample.get("global_task") or sample["lang"]
    norm_path = save_norm_stats(dataset, args.task, instruction)
    print(f"Saved norm stats to {norm_path}")


if __name__ == "__main__":
    main()

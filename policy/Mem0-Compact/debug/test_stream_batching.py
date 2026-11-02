"""Check recurrent batch slots stay frame-contiguous with multiple workers."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.dataloader.random_episode_dataloader import RandomEpisodeIterableDataset
from source.training.train_compact import collate_batch, merge_worker_batches


class FakeFrames:
    def __getitem__(self, index):
        episode_id, episode_pos = divmod(index, 8)
        return {
            "image": [f"{episode_id}:{episode_pos}"],
            "lang": "move the object",
            "action": torch.zeros(30, 16),
            "state": torch.zeros(1, 16),
            "episode_id": episode_id,
            "episode_pos": episode_pos,
            "subtask_end": 0,
        }


def main():
    episode_to_indices = {
        episode_id: list(range(episode_id * 8, (episode_id + 1) * 8))
        for episode_id in range(8)
    }
    dataset = RandomEpisodeIterableDataset(
        FakeFrames(), rank=0, world_size=1, shuffle=False, infinite=True,
        episode_to_indices=episode_to_indices,
        stream_batch_size=1,
    )
    loader = DataLoader(
        dataset, batch_size=None, num_workers=2, collate_fn=collate_batch,
    )
    batches = iter(merge_worker_batches(loader, num_workers=2))
    previous = None
    for _ in range(4):
        batch = next(batches)
        assert batch["episode_id"].shape == (2,)
        assert batch["episode_pos"].shape == (2,)
        current = list(zip(batch["episode_id"].tolist(),
                           batch["episode_pos"].tolist()))
        if previous is not None:
            for (prev_ep, prev_pos), (ep, pos) in zip(previous, current):
                assert ep == prev_ep and pos == prev_pos + 1, (previous, current)
        previous = current
    del batches, loader

    empty_dataset = RandomEpisodeIterableDataset(
        FakeFrames(), rank=0, world_size=1, shuffle=False, infinite=False,
        episode_ids=[], episode_to_indices=episode_to_indices,
        stream_batch_size=1,
    )
    empty_loader = DataLoader(
        empty_dataset, batch_size=None, num_workers=0, collate_fn=collate_batch,
    )
    assert list(empty_loader) == []
    print("recurrent multiworker stream batching: OK")


if __name__ == "__main__":
    main()

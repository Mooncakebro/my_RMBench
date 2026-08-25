"""
Frame-level dataset for RMBench LeRobot v3 datasets + samplers.

Returns, per frame:
    image:           raw head-cam frame, uint8 HWC (native 240x320)
    instruction:     task instruction string (fixed per episode/task)
    state:           normalized 14-dim joint state            float32 (14,)
    actions:         normalized action chunk                  float32 (action_horizon, 14)
    prev_action:     normalized previous action (zeros at episode start) float32 (14,)
    episode_index / frame_index / is_episode_start

Normalization (z-score joints, minmax->[-1,1] grippers) uses
compact_mode.normalizer with meta/stats.json.

Samplers:
  - plain random sampling for the baseline (ordinary DataLoader, shuffle=True)
  - EpisodeStreamSampler for COMPACT TBPTT: `num_streams` parallel episode
    cursors, each yielding consecutive windows of `chunk_size` frames.

Video / index machinery is adapted from MoDE's my_code/dataset.py
(LeRobotV3Dataset, already validated against this data).
"""
from __future__ import annotations

import bisect
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import compact_mode  # noqa: F401  (sys.path setup for MODE_ROOT)
# Reuse LeRobotV3Dataset internals from the MoDE repo.
from my_code.dataset import _VideoReaderCache, _ffmpeg_subprocess_env

from .normalizer import JointGripperNormalizer

DEFAULT_DATA_ROOT = Path("/home/spc/memory_arena/RMBench/data_lerobot")
CAMERA_KEY = "observation.images.fixed"  # head camera


class FrameDataset(Dataset):
    def __init__(
        self,
        root,
        action_horizon: int = 10,
        normalize: bool = True,
        gripper_indices=(6, 13),
        video_backend: str = "cv2",
        video_cache_size: int = 4,
        camera_key: str = CAMERA_KEY,
    ):
        self.root = Path(root)
        self.action_horizon = action_horizon
        self.normalize = normalize
        self.video_backend = video_backend
        self.camera_key = camera_key

        self.info = self._read_json(self.root / "meta" / "info.json")
        self.stats = self._read_json(self.root / "meta" / "stats.json")
        self.tasks = self._load_tasks(self.root / "meta" / "tasks.parquet")
        self.table = self._load_data_table()
        self.episode_bounds = self._build_episode_bounds(
            self.table["episode_index"].to_numpy()
        )
        self.action_normalizer = JointGripperNormalizer.from_stats(
            self.stats, "action", gripper_indices
        )
        self.state_normalizer = JointGripperNormalizer.from_stats(
            self.stats, "observation.state", gripper_indices
        )

        # Valid frames: action chunk must fit inside the episode.
        self.valid_rows: List[int] = []
        self.episode_valid_rows: Dict[int, List[int]] = {}
        for ep, (start, end) in sorted(self.episode_bounds.items()):
            rows = list(range(start, end - self.action_horizon + 2))
            if rows:
                self.episode_valid_rows[ep] = rows
                self.valid_rows.extend(rows)
        self.episodes: List[int] = sorted(self.episode_valid_rows.keys())

        # Video shards for the head camera.
        self.video_shards = self._build_video_shards()
        self.video_cache = _VideoReaderCache(video_cache_size)

    # ── metadata loading (mirrors LeRobotV3Dataset) ──

    @staticmethod
    def _read_json(path: Path) -> dict:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_tasks(path: Path) -> Dict[int, str]:
        task_df = pd.read_parquet(path)
        if "task" in task_df.columns:
            task_text = task_df["task"]
        else:
            task_text = task_df.index.to_series(index=task_df.index)
        return {
            int(task_index): str(task)
            for task, task_index in zip(
                task_text.tolist(), task_df["task_index"].tolist(), strict=True
            )
        }

    def _load_data_table(self) -> pd.DataFrame:
        files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {self.root / 'data'}")
        table = pd.concat((pd.read_parquet(p) for p in files), ignore_index=True)
        return table.sort_values("index").reset_index(drop=True)

    @staticmethod
    def _build_episode_bounds(episode_indices: np.ndarray) -> Dict[int, Tuple[int, int]]:
        bounds: Dict[int, List[int]] = {}
        for i, ep in enumerate(episode_indices.tolist()):
            if ep not in bounds:
                bounds[ep] = [i, i]
            bounds[ep][1] = i
        return {ep: (start, end) for ep, (start, end) in bounds.items()}

    # ── video reading (adapted from LeRobotV3Dataset) ──

    def _build_video_shards(self) -> List[Tuple[int, int, Path]]:
        video_root = self.root / "videos" / self.camera_key
        files = sorted(video_root.glob("chunk-*/file-*.mp4"))
        if not files:
            raise FileNotFoundError(f"No video files found under {video_root}")
        shards = []
        start = 0
        for path in files:
            frame_count = self._probe_video_frame_count(path)
            shards.append((start, start + frame_count, path))
            start += frame_count
        expected = int(self.info["total_frames"])
        if start != expected:
            raise RuntimeError(
                f"Video shard frame count for {self.camera_key} is {start}, expected {expected}"
            )
        return shards

    @staticmethod
    def _probe_video_frame_count(path: Path) -> int:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames", "-of", "json", str(path),
        ]
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False, env=_ffmpeg_subprocess_env(),
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed for {path}: {proc.stderr.decode(errors='replace')}")
        return int(json.loads(proc.stdout.decode())["streams"][0]["nb_frames"])

    def _resolve_video_frame(self, global_frame_index: int) -> Tuple[Path, int]:
        starts = [s for s, _, _ in self.video_shards]
        shard_idx = bisect.bisect_right(starts, global_frame_index) - 1
        start, _, path = self.video_shards[shard_idx]
        return path, global_frame_index - start

    def read_image(self, global_frame_index: int) -> np.ndarray:
        """Returns raw uint8 HWC frame from the head camera."""
        video_path, frame_index = self._resolve_video_frame(global_frame_index)
        if self.video_backend == "cv2":
            import cv2
            cap = self.video_cache.get(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"cv2 could not read frame {frame_index} from {video_path}")
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raise ValueError(f"Unknown video_backend: {self.video_backend}")

    # ── samples ──

    def __len__(self) -> int:
        return len(self.valid_rows)

    def _norm_action(self, arr: np.ndarray) -> np.ndarray:
        t = torch.as_tensor(np.asarray(arr), dtype=torch.float32)
        if self.normalize:
            t = self.action_normalizer.encode(t)
        return t.numpy().astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        row = self.valid_rows[int(index)]
        ep = int(self.table.at[row, "episode_index"])
        _, ep_end = self.episode_bounds[ep]
        frame_index = int(self.table.at[row, "frame_index"])
        is_start = frame_index == 0

        # Action chunk [row, row+horizon) padded at episode end with the last row.
        action_rows = np.arange(row, min(row + self.action_horizon, ep_end + 1))
        if len(action_rows) < self.action_horizon:
            action_rows = np.concatenate(
                [action_rows, np.full(self.action_horizon - len(action_rows), action_rows[-1])]
            )
        actions = np.stack(self.table["action"].to_numpy()[action_rows]).astype(np.float32)
        actions = self._norm_action(actions)

        state = np.array(self.table.at[row, "observation.state"], dtype=np.float32)
        if self.normalize:
            state = self.state_normalizer.encode(torch.from_numpy(state)).numpy().astype(np.float32)

        if is_start:
            prev_action = np.zeros(self.info["features"]["action"]["shape"][0], dtype=np.float32)
        else:
            prev_action = np.array(self.table.at[row - 1, "action"], dtype=np.float32)
            prev_action = self._norm_action(prev_action)

        image = self.read_image(int(self.table.at[row, "index"]))

        return {
            "image": image,                       # uint8 HWC
            "instruction": self.tasks[int(self.table.at[row, "task_index"])],
            "state": state,                       # (14,) normalized
            "actions": actions,                   # (horizon, 14) normalized
            "prev_action": prev_action,           # (14,) normalized
            "episode_index": ep,
            "frame_index": frame_index,
            "is_episode_start": is_start,
        }


def collate_frames(samples: List[dict]) -> dict:
    """Collate a list of frame samples into a batch dict (CPU tensors + lists)."""
    return {
        "images": [s["image"] for s in samples],                    # list of uint8 HWC
        "instructions": [s["instruction"] for s in samples],
        "state": torch.from_numpy(np.stack([s["state"] for s in samples])),
        "actions": torch.from_numpy(np.stack([s["actions"] for s in samples])),
        "prev_action": torch.from_numpy(np.stack([s["prev_action"] for s in samples])),
        "episode_index": torch.tensor([s["episode_index"] for s in samples]),
        "is_episode_start": torch.tensor([s["is_episode_start"] for s in samples]),
    }


class EpisodeStreamSampler:
    """
    Sequential-chunk sampler for TBPTT (COMPACT variant).

    Maintains `num_streams` parallel episode cursors. Each next_chunk() call
    returns, per stream, the next `chunk_size` consecutive valid frames of the
    stream's current episode, plus a reset flag (True when the stream just
    switched to a new episode -> memory must be re-initialized for that row).
    Episode order is shuffled per pass; trailing partial chunks are dropped.
    """

    def __init__(
        self,
        dataset: FrameDataset,
        chunk_size: int = 8,
        num_streams: int = 1,
        shuffle_episodes: bool = True,
        seed: int = 42,
        max_frames_per_episode: Optional[int] = None,
    ):
        self.dataset = dataset
        self.chunk_size = chunk_size
        self.num_streams = num_streams
        self.shuffle_episodes = shuffle_episodes
        self.max_frames_per_episode = max_frames_per_episode
        self.rng = random.Random(seed)
        # valid_rows are absolute table rows; convert to dataset indices
        row_to_index = {row: i for i, row in enumerate(dataset.valid_rows)}
        self.episode_index_lists: Dict[int, List[int]] = {
            ep: [row_to_index[r] for r in rows[: max_frames_per_episode or len(rows)]]
            for ep, rows in dataset.episode_valid_rows.items()
        }
        self.episode_ids = sorted(self.episode_index_lists.keys())
        self._order: List[int] = []
        self._streams = [None] * num_streams  # per-stream (episode_id, cursor)

    def _next_episode(self) -> int:
        if not self._order:
            self._order = list(self.episode_ids)
            if self.shuffle_episodes:
                self.rng.shuffle(self._order)
        return self._order.pop(0)

    def next_chunk(self):
        """Returns (chunk_indices [num_streams][chunk_size], resets [num_streams])."""
        chunk = []
        resets = []
        for s in range(self.num_streams):
            state = self._streams[s]
            if state is not None:
                ep, cursor = state
                frames = self.episode_index_lists[ep]
                if cursor + self.chunk_size > len(frames):
                    state = None  # episode exhausted -> switch
            if state is None:
                ep = self._next_episode()
                frames = self.episode_index_lists[ep]
                cursor = 0
                resets.append(True)
            else:
                resets.append(False)
            # Skip episodes too short for one chunk.
            while len(frames) < self.chunk_size:
                ep = self._next_episode()
                frames = self.episode_index_lists[ep]
                cursor = 0
                resets[s] = True
            chunk.append(frames[cursor: cursor + self.chunk_size])
            self._streams[s] = (ep, cursor + self.chunk_size)
        return chunk, resets

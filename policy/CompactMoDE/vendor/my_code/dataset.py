"""
LeRobot v3.0 dataset reader for the SO-101 cube-moving dataset.

The dataset returns ordinary PyTorch DataLoader batches with keys:
    actions:      [B, action_horizon, action_dim]
    robot_obs:    [B, obs_horizon, state_dim]
    rgb_obs:
        rgb_static:  [B, obs_horizon, 3, H, W]
    lang_text:    list[str]
    task_index:   [B]
    episode_index:[B]
    frame_index:  [B]
    idx:          [B]

Images are read from LeRobot's mp4 video files when include_images=True.
For faster state-only debugging, set include_images=False.
"""

from __future__ import annotations

import json
import os
import subprocess
import bisect
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


DEFAULT_DATASET_ROOT = Path("/media/spc/新加卷/lerobot_dataset/so101/20260507-1637")


def _torch_from_numpy(array: np.ndarray) -> torch.Tensor:
    array = np.asarray(array)
    try:
        return torch.from_numpy(array)
    except TypeError:
        # Some mixed conda/pip environments expose NumPy arrays that PyTorch's
        # C extension does not recognize as np.ndarray. The list fallback is
        # slower but keeps dataset loading functional.
        return torch.tensor(array.tolist())


class _VideoReaderCache:
    """Small LRU cache for cv2.VideoCapture handles."""

    def __init__(self, max_open: int = 8) -> None:
        self.max_open = max_open
        self._captures: OrderedDict[Path, Any] = OrderedDict()

    def get(self, path: Path) -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("opencv-python is required when include_images=True") from exc

        if path in self._captures:
            cap = self._captures.pop(path)
            self._captures[path] = cap
            return cap

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        self._captures[path] = cap
        while len(self._captures) > self.max_open:
            _, old_cap = self._captures.popitem(last=False)
            old_cap.release()
        return cap

    def close(self) -> None:
        for cap in self._captures.values():
            cap.release()
        self._captures.clear()

    def __del__(self) -> None:
        self.close()


def _ffmpeg_subprocess_env() -> Dict[str, str]:
    """Run system ffmpeg/ffprobe without conda's lib path shadowing system libs."""
    env = os.environ.copy()
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = str(Path(conda_prefix) / "lib")
        ld_paths = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and p != conda_lib]
        if ld_paths:
            env["LD_LIBRARY_PATH"] = ":".join(ld_paths)
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return env


class LeRobotV3Dataset(Dataset):
    def __init__(
        self,
        root: Union[str, Path] = DEFAULT_DATASET_ROOT,
        obs_horizon: int = 1,
        action_horizon: int = 10,
        include_images: bool = True,
        image_size: Optional[Tuple[int, int]] = (224, 224),
        image_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
        image_keys: Tuple[str, ...] = ("observation.images.fixed",),
        normalize_actions: bool = True,
        normalize_state: bool = True,
        normalize_mode: str = "mean_std",
        video_cache_size: int = 8,
        video_backend: str = "ffmpeg",
        video_decode_retry_radius: int = 30,
        corrupt_frame_policy: str = "error",
    ) -> None:
        self.root = Path(root)
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.include_images = include_images
        self.image_size = image_size
        self.image_sizes = image_sizes or {}
        self.image_keys = image_keys
        self.normalize_actions = normalize_actions
        self.normalize_state = normalize_state
        self.normalize_mode = normalize_mode
        self.video_backend = video_backend
        self.video_decode_retry_radius = video_decode_retry_radius
        self.corrupt_frame_policy = corrupt_frame_policy

        self.info = self._read_json(self.root / "meta" / "info.json")
        self.stats = self._read_json(self.root / "meta" / "stats.json")
        self.tasks = self._load_tasks(self.root / "meta" / "tasks.parquet")
        self.table = self._load_data_table()
        self.episode_bounds = self._build_episode_bounds(self.table["episode_index"].to_numpy())
        self.indices = self._build_valid_indices()
        self.video_shards = self._build_video_shards()
        self.video_cache = _VideoReaderCache(video_cache_size)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: Union[int, slice, Sequence[int], torch.Tensor, np.ndarray]) -> Any:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if torch.is_tensor(index):
            index = index.detach().cpu().tolist()
        if isinstance(index, np.ndarray):
            index = index.tolist()
        if isinstance(index, list):
            return [self[i] for i in index]

        row_index = int(self.indices[int(index)])
        obs_rows = self._window_rows(row_index, self.obs_horizon, future=False)
        action_rows = self._window_rows(row_index, self.action_horizon, future=True)

        sample = {
            "actions": self._tensor_column(action_rows, "action", normalize=self.normalize_actions),
            "robot_obs": self._tensor_column(obs_rows, "observation.state", normalize=self.normalize_state),
            "lang_text": self.tasks[int(self.table.at[row_index, "task_index"])],
            "task_index": torch.tensor(int(self.table.at[row_index, "task_index"]), dtype=torch.long),
            "episode_index": torch.tensor(int(self.table.at[row_index, "episode_index"]), dtype=torch.long),
            "frame_index": torch.tensor(int(self.table.at[row_index, "frame_index"]), dtype=torch.long),
            "idx": torch.tensor(row_index, dtype=torch.long),
        }

        if self.include_images:
            fixed = self._read_image_window(obs_rows, self.image_keys[0], "rgb_static")
        else:
            fixed = torch.empty(0)
        sample["rgb_obs"] = {"rgb_static": fixed}
        return sample

    @property
    def action_dim(self) -> int:
        return int(self.info["features"]["action"]["shape"][0])

    @property
    def state_dim(self) -> int:
        return int(self.info["features"]["observation.state"]["shape"][0])

    def _load_data_table(self) -> pd.DataFrame:
        files = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found under {self.root / 'data'}")
        table = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
        return table.sort_values("index").reset_index(drop=True)

    @staticmethod
    def _read_json(path: Path) -> Dict[str, Any]:
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
            for task, task_index in zip(task_text.tolist(), task_df["task_index"].tolist(), strict=True)
        }

    @staticmethod
    def _build_episode_bounds(episode_indices: np.ndarray) -> Dict[int, Tuple[int, int]]:
        bounds: Dict[int, List[int]] = {}
        for i, ep in enumerate(episode_indices.tolist()):
            if ep not in bounds:
                bounds[ep] = [i, i]
            bounds[ep][1] = i
        return {ep: (start, end) for ep, (start, end) in bounds.items()}

    def _build_valid_indices(self) -> np.ndarray:
        valid: List[int] = []
        for start, end in self.episode_bounds.values():
            first = start + self.obs_horizon - 1
            last = end - self.action_horizon + 1
            if first <= last:
                valid.extend(range(first, last + 1))
        return np.asarray(valid, dtype=np.int64)

    def _build_video_shards(self) -> Dict[str, List[Tuple[int, int, Path]]]:
        if not self.include_images:
            return {}
        return {video_key: self._build_video_shards_for_key(video_key) for video_key in self.image_keys}

    def _build_video_shards_for_key(self, video_key: str) -> List[Tuple[int, int, Path]]:
        video_root = self.root / "videos" / video_key
        files = sorted(video_root.glob("chunk-*/file-*.mp4"))
        if not files:
            raise FileNotFoundError(f"No video files found under {video_root}")

        shards = []
        start = 0
        for path in files:
            frame_count = self._probe_video_frame_count(path)
            end = start + frame_count
            shards.append((start, end, path))
            start = end

        expected = int(self.info["total_frames"])
        if start != expected:
            raise RuntimeError(f"Video shard frame count for {video_key} is {start}, expected {expected}")
        return shards

    @staticmethod
    def _probe_video_frame_count(path: Path) -> int:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "json",
            str(path),
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, env=_ffmpeg_subprocess_env())
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"ffprobe failed for {path}: {stderr}")
        stream = json.loads(proc.stdout.decode("utf-8"))["streams"][0]
        return int(stream["nb_frames"])

    def _window_rows(self, center_row: int, length: int, future: bool) -> pd.DataFrame:
        ep = int(self.table.at[center_row, "episode_index"])
        start, end = self.episode_bounds[ep]
        if future:
            rows = np.arange(center_row, min(center_row + length, end + 1))
            pad_value = rows[-1]
            if len(rows) < length:
                rows = np.concatenate([rows, np.full(length - len(rows), pad_value)])
        else:
            rows = np.arange(max(start, center_row - length + 1), center_row + 1)
            pad_value = rows[0]
            if len(rows) < length:
                rows = np.concatenate([np.full(length - len(rows), pad_value), rows])
        return self.table.iloc[rows]

    def _tensor_column(self, rows: pd.DataFrame, key: str, normalize: bool) -> torch.Tensor:
        array = np.stack(rows[key].to_numpy()).astype(np.float32)
        tensor = _torch_from_numpy(array)
        if normalize:
            entry = self.stats[key]
            if self.normalize_mode == "q01_q99":
                q01 = torch.as_tensor(entry["q01"], dtype=tensor.dtype).flatten()
                q99 = torch.as_tensor(entry["q99"], dtype=tensor.dtype).flatten()
                tensor = 2.0 * (tensor - q01) / (q99 - q01).clamp_min(1e-6) - 1.0
            else:
                mean = torch.as_tensor(entry["mean"], dtype=tensor.dtype).flatten()
                std = torch.as_tensor(entry["std"], dtype=tensor.dtype).flatten().clamp_min(1e-6)
                tensor = (tensor - mean) / std
        return tensor

    def _read_image_window(self, rows: pd.DataFrame, video_key: str, output_key: str) -> torch.Tensor:
        frames = [self._read_video_frame(row, video_key) for _, row in rows.iterrows()]
        tensor = torch.stack(frames, dim=0)
        image_size = self.image_sizes.get(output_key, self.image_size)
        if image_size is not None:
            tensor = F.interpolate(
                tensor,
                size=image_size,
                mode="bilinear",
                align_corners=False,
            )
        return tensor

    def _read_video_frame(self, row: pd.Series, video_key: str) -> torch.Tensor:
        global_frame_index = int(row["index"])
        video_path, frame_index, shard_frame_count = self._resolve_video_frame(video_key, global_frame_index)

        if self.video_backend == "ffmpeg":
            frame = self._read_video_frame_ffmpeg(video_path, frame_index, video_key, shard_frame_count)
        elif self.video_backend == "cv2":
            frame = self._read_video_frame_cv2(video_path, frame_index)
        elif self.video_backend == "auto":
            try:
                frame = self._read_video_frame_ffmpeg(video_path, frame_index, video_key, shard_frame_count)
            except Exception:
                frame = self._read_video_frame_cv2(video_path, frame_index)
        else:
            raise ValueError(f"Unknown video_backend: {self.video_backend}")

        return _torch_from_numpy(frame.copy()).permute(2, 0, 1).float().div(255.0)

    def _resolve_video_frame(self, video_key: str, global_frame_index: int) -> Tuple[Path, int, int]:
        shards = self.video_shards[video_key]
        starts = [start for start, _, _ in shards]
        shard_idx = bisect.bisect_right(starts, global_frame_index) - 1
        if shard_idx < 0:
            raise IndexError(f"Global frame {global_frame_index} is before first video shard")
        start, end, path = shards[shard_idx]
        if global_frame_index >= end:
            raise IndexError(f"Global frame {global_frame_index} is outside video shard range {start}:{end}")
        return path, global_frame_index - start, end - start

    def _read_video_frame_cv2(self, video_path: Path, frame_index: int) -> np.ndarray:
        import cv2

        cap = self.video_cache.get(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"OpenCV could not read frame {frame_index} from {video_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return np.asarray(frame, dtype=np.uint8)

    def _read_video_frame_ffmpeg(
        self,
        video_path: Path,
        frame_index: int,
        video_key: str,
        shard_frame_count: int,
    ) -> np.ndarray:
        offsets = [0]
        for delta in range(1, self.video_decode_retry_radius + 1):
            offsets.extend([-delta, delta])

        errors = []
        for offset in offsets:
            candidate = frame_index + offset
            if candidate < 0 or candidate >= shard_frame_count:
                continue
            try:
                return self._read_video_frame_ffmpeg_once(video_path, candidate, video_key)
            except RuntimeError as exc:
                errors.append(f"{candidate}: {exc}")

        message = (
            f"ffmpeg could not read frame {frame_index} or nearby frames from {video_path}. "
            f"Last errors: {' | '.join(errors[-3:])}"
        )
        if self.corrupt_frame_policy == "zero":
            warnings.warn(message + " Returning a zero frame.", RuntimeWarning)
            height, width, _ = self.info["features"][video_key]["shape"]
            return np.zeros((int(height), int(width), 3), dtype=np.uint8)
        raise RuntimeError(message)

    def _read_video_frame_ffmpeg_once(self, video_path: Path, frame_index: int, video_key: str) -> np.ndarray:
        height, width, channels = self.info["features"][video_key]["shape"]
        if channels != 3:
            raise ValueError(f"Expected 3-channel RGB video for {video_key}, got shape {height, width, channels}")

        timestamp = frame_index / float(self.info.get("fps", 30))
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-fflags",
            "+discardcorrupt",
            "-err_detect",
            "ignore_err",
            "-ss",
            f"{timestamp:.9f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, env=_ffmpeg_subprocess_env())
        expected_bytes = int(height) * int(width) * 3
        if proc.returncode != 0 or len(proc.stdout) != expected_bytes:
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"ffmpeg could not read frame {frame_index} from {video_path}; "
                f"got {len(proc.stdout)} bytes, expected {expected_bytes}. {stderr}"
            )
        return np.frombuffer(proc.stdout, dtype=np.uint8).reshape(int(height), int(width), 3)


def build_dataloaders(
    root: Union[str, Path] = DEFAULT_DATASET_ROOT,
    batch_size: int = 8,
    num_workers: int = 4,
    val_fraction: float = 0.05,
    seed: int = 42,
    **dataset_kwargs: Any,
) -> Tuple[DataLoader, Optional[DataLoader], LeRobotV3Dataset]:
    dataset = LeRobotV3Dataset(root=root, **dataset_kwargs)
    if val_fraction > 0:
        val_len = max(1, int(len(dataset) * val_fraction))
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(
            dataset,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_set, val_set = dataset, None

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs) if val_set else None
    return train_loader, val_loader, dataset


def _smoke_test() -> None:
    dataset = LeRobotV3Dataset(include_images=True, obs_horizon=1, action_horizon=10)
    sample = dataset[0]
    print(f"dataset length: {len(dataset)}")
    for key, value in sample.items():
        if isinstance(value, dict):
            print(key, {k: tuple(v.shape) for k, v in value.items()})
        elif torch.is_tensor(value):
            print(key, tuple(value.shape), value.dtype)
        else:
            print(key, value)


def _smoke_test_v2() -> None:
    dataset = LeRobotV3Dataset(include_images=True, obs_horizon=1, action_horizon=10)
    print(f"dataset length: {len(dataset)}")
    
    for i in range(len(dataset)):
        sample = dataset[i]
        
        for key, value in sample.items():
            if key == "rgb_obs":
                # Use 'value' (the dict), not 'key' (the string)
                static_shape = value['rgb_static'].shape
                
                if static_shape != (1, 3, 224, 224):
                    print(f"Sample {i}: Unexpected shape for rgb_static: {static_shape}")
                    return  # Stop the test if the shape is incorrect
                print(f"Sample {i}: Image shape is correct.")


if __name__ == "__main__":
    _smoke_test_v2()

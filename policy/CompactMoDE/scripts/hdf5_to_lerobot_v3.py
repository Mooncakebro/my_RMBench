#!/usr/bin/env python
"""Convert RMBench demo_clean HDF5 episodes to a LeRobot v3.0-style dataset.

The output layout matches exactly what
/home/spc/MoDE_Diffusion_Policy/my_code/dataset.py (class LeRobotV3Dataset)
consumes:

    <dst>/<task>/
      data/chunk-000/file-000.parquet
      videos/observation.images.fixed/chunk-000/file-000.mp4
      videos/observation.images.handeye/chunk-000/file-000.mp4   (hardlink of fixed)
      meta/info.json
      meta/stats.json
      meta/tasks.parquet

Conventions (per task spec):
  * state[t]  = joint_action/vector[t]     (14-dim, float32)
  * action[t] = joint_action/vector[t+1]   -> the final frame of each episode
    is dropped, so each episode contributes T-1 rows and T-1 video frames.
  * vector layout: [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
  * head camera only; the loader's default image keys are
    "observation.images.fixed" / "observation.images.handeye", so the head
    stream is published under both keys (handeye is a hardlink).
  * language instruction = instructions/episode{i}.json -> "seen"[0].

HDF5 quirks handled here:
  * float datasets are stored as 80-bit longdouble which h5py's high-level
    API cannot map -> low-level read with mtype=NATIVE_DOUBLE.
  * rgb datasets are NULLPAD fixed-length strings -> low-level read with the
    file's own string type, then strip trailing NULs before cv2.imdecode.

Extensibility: language_annotation.json (per-episode subtask segments) can be
merged later via `load_subtask_annotations()` + extra parquet columns; the
loader ignores unknown columns, so adding them is backwards compatible.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from h5py import h5s, h5t

FPS = 30
ACTION_DIM = 14
IMG_H, IMG_W = 240, 320
VIDEO_KEYS = ("observation.images.fixed", "observation.images.handeye")

STATE_NAMES = (
    [f"left_joint_{i}" for i in range(6)]
    + ["left_gripper"]
    + [f"right_joint_{i}" for i in range(6)]
    + ["right_gripper"]
)

ALL_TASKS = [
    "battery_try",
    "blocks_ranking_try",
    "classify_blocks",
    "cover_blocks",
    "observe_and_pickup",
    "place_block_mat",
    "press_button",
    "put_back_block",
    "rearrange_blocks",
    "storage_blocks",
    "swap_blocks",
    "swap_T",
]


# --------------------------------------------------------------------------- #
# low-level HDF5 readers (work around longdouble / fixed-string dtypes)
# --------------------------------------------------------------------------- #
def read_float_dataset(dset: h5py.Dataset) -> np.ndarray:
    """Read a (possibly longdouble) float dataset as float64."""
    shape = dset.id.get_space().get_simple_extent_dims()
    arr = np.empty(shape, dtype=np.float64)
    dset.id.read(h5s.ALL, h5s.ALL, arr, mtype=h5t.NATIVE_DOUBLE)
    return arr


def read_rgb_dataset(dset: h5py.Dataset) -> list[bytes]:
    """Read a NULLPAD fixed-length-string dataset into a list of JPEG bytes."""
    ftype = dset.id.get_type()
    size = ftype.get_size()
    shape = dset.id.get_space().get_simple_extent_dims()
    arr = np.empty(shape, dtype=np.dtype(("S", size)))
    dset.id.read(h5s.ALL, h5s.ALL, arr, mtype=ftype)
    # np.bytes_ -> bytes strips trailing NUL padding.
    return [bytes(x) for x in arr]


def load_subtask_annotations(task_dir: Path, episode_idx: int) -> list | None:
    """Return [[text, end_timestep], ...] for an episode, or None.

    Not written into the dataset yet (not needed by the current loader), kept
    here so subtask columns can be added without changing the reading code.
    """
    path = task_dir / "language_annotation.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        annot = json.load(f)
    return annot.get(f"episode_{episode_idx}")


# --------------------------------------------------------------------------- #
# video encoding (system ffmpeg, conda libs stripped from LD_LIBRARY_PATH)
# --------------------------------------------------------------------------- #
def _ffmpeg_env() -> dict:
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


class Mp4Encoder:
    def __init__(self, out_path: Path, fps: int = FPS, crf: int = 20) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{IMG_W}x{IMG_H}",
            "-r", str(fps),
            "-i", "pipe:0",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-g", "15",
            "-movflags", "+faststart",
            str(out_path),
        ]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, env=_ffmpeg_env(),
        )
        self.frames_written = 0

    def write(self, frame_bgr: np.ndarray) -> None:
        assert frame_bgr.shape == (IMG_H, IMG_W, 3), frame_bgr.shape
        self.proc.stdin.write(frame_bgr.tobytes())
        self.frames_written += 1

    def close(self) -> None:
        self.proc.stdin.close()
        stderr = self.proc.stderr.read().decode("utf-8", errors="replace")
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed:\n{stderr[-2000:]}")


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #
def convert_task(task: str, src_root: Path, dst_root: Path, num_episodes: int, crf: int) -> dict:
    task_dir = src_root / task / "demo_clean"
    dst = dst_root / task
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (dst / "meta").mkdir(parents=True, exist_ok=True)

    encoder = Mp4Encoder(dst / "videos" / VIDEO_KEYS[0] / "chunk-000" / "file-000.mp4", crf=crf)

    rows_index, rows_ep, rows_frame, rows_ts, rows_task = [], [], [], [], []
    actions_all, states_all = [], []
    task_texts: list[str] = []
    episodes_done = 0
    global_index = 0

    for ep in range(num_episodes):
        hdf5_path = task_dir / "data" / f"episode{ep}.hdf5"
        instr_path = task_dir / "instructions" / f"episode{ep}.json"
        if not hdf5_path.exists():
            print(f"  [skip] {task}/episode{ep}: missing hdf5", flush=True)
            continue

        with open(instr_path, "r", encoding="utf-8") as f:
            instr = json.load(f)
        instruction = instr["seen"][0]
        if instruction not in task_texts:
            task_texts.append(instruction)
        task_index = task_texts.index(instruction)

        with h5py.File(hdf5_path, "r") as f:
            vector = read_float_dataset(f["joint_action/vector"]).astype(np.float32)  # (T, 14)
            jpeg_frames = read_rgb_dataset(f["observation/head_camera/rgb"])

        T = vector.shape[0]
        if T < 2 or len(jpeg_frames) != T:
            print(f"  [skip] {task}/episode{ep}: T={T}, n_rgb={len(jpeg_frames)}", flush=True)
            continue

        states = vector[:-1]   # (T-1, 14)
        actions = vector[1:]   # (T-1, 14)

        for t in range(T - 1):
            frame = cv2.imdecode(np.frombuffer(jpeg_frames[t], np.uint8), cv2.IMREAD_COLOR)
            if frame is None or frame.shape != (IMG_H, IMG_W, 3):
                raise RuntimeError(f"{task}/episode{ep} frame {t}: bad decode {None if frame is None else frame.shape}")
            encoder.write(frame)

        n = T - 1
        rows_index.extend(range(global_index, global_index + n))
        rows_ep.extend([ep] * n)
        rows_frame.extend(range(n))
        rows_ts.extend([t / FPS for t in range(n)])
        rows_task.extend([task_index] * n)
        states_all.append(states)
        actions_all.append(actions)
        global_index += n
        episodes_done += 1
        print(f"  {task}/episode{ep}: {n} frames (task_index={task_index})", flush=True)

    encoder.close()

    if episodes_done == 0:
        raise RuntimeError(f"{task}: no episodes converted")

    # second video key = hardlink (fallback: copy) of the head-camera stream
    src_mp4 = dst / "videos" / VIDEO_KEYS[0] / "chunk-000" / "file-000.mp4"
    dst_mp4 = dst / "videos" / VIDEO_KEYS[1] / "chunk-000" / "file-000.mp4"
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src_mp4, dst_mp4)
    except OSError:
        shutil.copy(src_mp4, dst_mp4)

    states = np.concatenate(states_all, axis=0)
    actions = np.concatenate(actions_all, axis=0)
    total_frames = int(states.shape[0])
    assert total_frames == encoder.frames_written == global_index

    # ---- data/chunk-000/file-000.parquet ----
    vec_type = pa.list_(pa.float32(), ACTION_DIM)
    table = pa.table({
        "index": pa.array(rows_index, pa.int64()),
        "episode_index": pa.array(rows_ep, pa.int64()),
        "frame_index": pa.array(rows_frame, pa.int64()),
        "timestamp": pa.array(rows_ts, pa.float32()),
        "task_index": pa.array(rows_task, pa.int64()),
        "action": pa.array(list(actions), vec_type),
        "observation.state": pa.array(list(states), vec_type),
    })
    pq.write_table(table, dst / "data" / "chunk-000" / "file-000.parquet")

    # ---- meta/info.json ----
    info = {
        "codebase_version": "v3.0",
        "robot_type": "agilex-cobot-magic-dual-arm",
        "fps": FPS,
        "total_episodes": episodes_done,
        "total_frames": total_frames,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": STATE_NAMES},
            "observation.state": {"dtype": "float32", "shape": [ACTION_DIM], "names": STATE_NAMES},
            "observation.images.fixed": {
                "dtype": "video", "shape": [IMG_H, IMG_W, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": FPS, "video.codec": "h264"},
            },
            "observation.images.handeye": {
                "dtype": "video", "shape": [IMG_H, IMG_W, 3],
                "names": ["height", "width", "channels"],
                "info": {"video.fps": FPS, "video.codec": "h264"},
            },
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with (dst / "meta" / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    # ---- meta/stats.json (z-score normalization stats) ----
    def _stats(x: np.ndarray) -> dict:
        x64 = x.astype(np.float64)
        return {
            "mean": x64.mean(axis=0).tolist(),
            "std": x64.std(axis=0).tolist(),
            "min": x64.min(axis=0).tolist(),
            "max": x64.max(axis=0).tolist(),
            "count": [int(x64.shape[0])],
        }

    stats = {"action": _stats(actions), "observation.state": _stats(states)}
    with (dst / "meta" / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # ---- meta/tasks.parquet ----
    tasks_df = pd.DataFrame({
        "task": task_texts,
        "task_index": np.arange(len(task_texts), dtype=np.int64),
    })
    tasks_df.to_parquet(dst / "meta" / "tasks.parquet", index=False)

    return {
        "task": task,
        "episodes": episodes_done,
        "frames": total_frames,
        "unique_instructions": len(task_texts),
        "dst": str(dst),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, help="task names (default: all 12)")
    parser.add_argument("--src-root", type=Path, default=Path("/home/spc/memory_arena/RMBench/data"))
    parser.add_argument("--dst-root", type=Path, default=Path("/home/spc/memory_arena/RMBench/data_lerobot"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--crf", type=int, default=20)
    args = parser.parse_args()

    results = []
    for task in args.tasks:
        print(f"\n=== {task} ===", flush=True)
        try:
            results.append(convert_task(task, args.src_root, args.dst_root, args.episodes, args.crf))
        except Exception as exc:
            print(f"  [FAIL] {task}: {exc}", flush=True)
            results.append({"task": task, "error": str(exc)})

    print("\n==== summary ====")
    for r in results:
        if "error" in r:
            print(f"  {r['task']}: FAILED - {r['error']}")
        else:
            print(f"  {r['task']}: {r['episodes']} episodes, {r['frames']} frames, "
                  f"{r['unique_instructions']} unique instructions -> {r['dst']}")
    if any("error" in r for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()

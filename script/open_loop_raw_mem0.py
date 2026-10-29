"""Read original RMBench HDF5 episodes with the official Mem-0 conventions."""
from __future__ import annotations

import bisect
import io
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
from PIL import Image


class RawMem0Dataset:
    """Evaluation adapter matching M1_dataset_to_lerobot + dataset_min_max.

    Layout: six left joints, padding, six right joints, padding, two grippers.
    An action is the NEXT recorded joint state (last frame repeats itself).
    Only the twelve arm joints are min-max normalized, with clipping.
    """

    def __init__(self, root, task, horizon, norm_stats, episode_ids=None,
                 max_episodes=None, instruction=None):
        root = Path(root)
        candidates = [root, root / "data", root / "demo_clean" / "data"]
        self.root = next((p for p in candidates if any(p.glob("episode*.hdf5"))), None)
        if self.root is None:
            raise FileNotFoundError(f"No episode*.hdf5 recordings under {root}")
        if norm_stats is None:
            raise ValueError("Raw evaluation requires --norm-stats from the released checkpoint")
        stats = json.loads(Path(norm_stats).read_text(encoding="utf-8"))
        for key in ("state_min", "state_max", "action_min", "action_max"):
            values = np.asarray(stats[key], dtype=np.float32)
            if values.shape != (16,) or not np.isfinite(values).all():
                raise ValueError(f"Invalid {key} in {norm_stats}: expected 16 finite values")
            setattr(self, key, values)
        self.horizon = horizon
        if instruction is None:
            bundle = Path(__file__).resolve().parents[1] / "policy/Mem-0/released/m1mix/task_instructions.json"
            instruction = json.loads(bundle.read_text(encoding="utf-8"))["tasks"][task]["global_task"]
        self.instruction = instruction
        files = {int(p.stem.removeprefix("episode")): p
                 for p in self.root.glob("episode*.hdf5")}
        selected = sorted(files) if episode_ids is None else list(episode_ids)
        if max_episodes is not None:
            selected = selected[:max_episodes]
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("Select at least one distinct episode")
        self.bounds, self.files, self.ends = {}, {}, []
        states = []
        offset = 0
        for eid in selected:
            if eid not in files:
                raise FileNotFoundError(f"Missing raw episode {eid} under {self.root}")
            with h5py.File(files[eid], "r") as f:
                joints = f["joint_action"]
                left, right = joints["left_arm"][:], joints["right_arm"][:]
                n = len(left)
                if left.shape != (n, 6) or right.shape != (n, 6) or n < horizon:
                    raise ValueError(f"Invalid or too short arm trajectory in {files[eid]}")
                state = np.column_stack((left, np.zeros(n), right, np.zeros(n),
                                         joints["left_gripper"][:].reshape(n),
                                         joints["right_gripper"][:].reshape(n))).astype(np.float32)
                if len(f["observation/head_camera/rgb"]) != n or not np.isfinite(state).all():
                    raise ValueError(f"Invalid image/state sequence in {files[eid]}")
            states.append(state)
            self.bounds[eid] = (offset, offset + n)
            self.files[eid] = files[eid]
            offset += n
            self.ends.append(offset)
        self.episode_ids = selected
        self.raw_states = np.concatenate(states)
        self.raw_actions = np.concatenate([np.concatenate((s[1:], s[-1:])) for s in states])
        self.names = ([f"left_joint_{j}" for j in range(7)]
                      + [f"right_joint_{j}" for j in range(7)]
                      + ["left_gripper", "right_gripper"])
        self.dataset = SimpleNamespace(meta=SimpleNamespace(stats={
            "action": {"min": self.action_min, "max": self.action_max},
        }))
        self._file = None
        self._eid = None

    def __len__(self):
        return len(self.raw_states)

    @staticmethod
    def _normalize(value, minimum, maximum):
        result = value.copy()
        mask = np.ones(16, dtype=bool)
        mask[[6, 13, 14, 15]] = False
        span = maximum - minimum
        span = np.where(span < 1e-8, 1, span)
        result[..., mask] = np.clip(2 * (result[..., mask] - minimum[mask]) / span[mask] - 1, -1, 1)
        return result

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        eid = self.episode_ids[bisect.bisect_right(self.ends, index)]
        start, end = self.bounds[eid]
        if self._eid != eid:
            self.close()
            self._file = h5py.File(self.files[eid], "r")
            self._eid = eid
        encoded = self._file["observation/head_camera/rgb"][index - start]
        with Image.open(io.BytesIO(bytes(encoded))) as image:
            rgb = image.convert("RGB")
        indices = np.minimum(np.arange(index, index + self.horizon), end - 1)
        return {
            "image": [rgb], "lang": self.instruction,
            "state": self._normalize(self.raw_states[index], self.state_min, self.state_max)[None],
            "action": self._normalize(self.raw_actions[indices], self.action_min, self.action_max),
        }

    def close(self):
        if self._file is not None:
            self._file.close()
        self._file = None
        self._eid = None

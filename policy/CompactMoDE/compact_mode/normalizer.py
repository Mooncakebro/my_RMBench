"""
State/action normalization for RMBench dual-arm data.

Layout of the 14-dim vector: [left_arm(6), left_gripper, right_arm(6), right_gripper].

  - 12 arm-joint dims: z-score   (x - mean) / std
  - 2 gripper dims (indices 6 and 13): minmax -> [-1, 1]

Stats are loaded from a LeRobot v3 dataset's meta/stats.json, which holds
mean/std/min/max per dim for both "action" and "observation.state".
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


class JointGripperNormalizer:
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        minv: torch.Tensor,
        maxv: torch.Tensor,
        gripper_indices=(6, 13),
    ):
        self.mean = mean.float().flatten()
        self.std = std.float().flatten().clamp_min(1e-6)
        self.min = minv.float().flatten()
        self.max = maxv.float().flatten()
        self.gripper_indices = tuple(int(i) for i in gripper_indices)
        self.dim = int(self.mean.numel())
        self.joint_indices = tuple(
            i for i in range(self.dim) if i not in self.gripper_indices
        )
        # Precompute minmax scale/offset; guard degenerate ranges.
        rng = (self.max - self.min).clamp_min(1e-6)
        self._mm_scale = 2.0 / rng
        self._mm_offset = -1.0 - self.min * self._mm_scale

    @classmethod
    def from_stats(cls, stats: dict, key: str, gripper_indices=(6, 13)) -> "JointGripperNormalizer":
        s = stats[key]
        return cls(
            mean=torch.as_tensor(s["mean"]),
            std=torch.as_tensor(s["std"]),
            minv=torch.as_tensor(s["min"]),
            maxv=torch.as_tensor(s["max"]),
            gripper_indices=gripper_indices,
        )

    @classmethod
    def from_stats_json(cls, path, key: str, gripper_indices=(6, 13)) -> "JointGripperNormalizer":
        with Path(path).open("r", encoding="utf-8") as f:
            stats = json.load(f)
        return cls.from_stats(stats, key, gripper_indices)

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "min": self.min.tolist(),
            "max": self.max.tolist(),
            "gripper_indices": list(self.gripper_indices),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JointGripperNormalizer":
        return cls(
            mean=torch.as_tensor(d["mean"]),
            std=torch.as_tensor(d["std"]),
            minv=torch.as_tensor(d["min"]),
            maxv=torch.as_tensor(d["max"]),
            gripper_indices=d["gripper_indices"],
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Raw -> normalized. x: (..., dim)."""
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        mm_scale = self._mm_scale.to(x.device, x.dtype)
        mm_offset = self._mm_offset.to(x.device, x.dtype)
        out = (x - mean) / std
        for i in self.gripper_indices:
            out[..., i] = torch.clamp(x[..., i] * mm_scale[i] + mm_offset[i], -1.0, 1.0)
        return out

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized -> raw. x: (..., dim)."""
        mean = self.mean.to(x.device, x.dtype)
        std = self.std.to(x.device, x.dtype)
        mm_scale = self._mm_scale.to(x.device, x.dtype)
        mm_offset = self._mm_offset.to(x.device, x.dtype)
        out = x * std + mean
        for i in self.gripper_indices:
            out[..., i] = (x[..., i].clamp(-1.0, 1.0) - mm_offset[i]) / mm_scale[i]
        return out

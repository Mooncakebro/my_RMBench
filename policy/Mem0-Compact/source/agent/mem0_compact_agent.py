"""
Mem0-Compact deployment agent (M1 tasks first — swap_blocks target).

Adapted from Mem-0's source/agent/memorymatters_agent.py with:
  - MemoryBank removed; COMPACT side memory state (M, P, e) is threaded across
    update_obs calls by the executor and reset ONLY at episode start.
  - prev_action = normalized current state (last executed action, per the
    dataset action[t] = state[t+1] convention); zeros on the first observation
    of an episode.
  - M(1): fixed global instruction, no planner, classifier unused.
    (M(n) planner wiring will be added when M(n) tasks are targeted.)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from termcolor import cprint

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from source.models.execution_module.mem0_compact_executor import Mem0CompactExecutor

M1_TASKS = ["swap_blocks", "swap_T", "observe_and_pickup",
            "put_back_block", "rearrange_blocks"]


class Mem0CompactAgent:
    def __init__(self, cfg: OmegaConf, ckpt_path: str, device: torch.device):
        self.config = cfg

        task_name = self.config.get("task_name", "unknown_task")
        self.task_type = "M1" if task_name in M1_TASKS else "Mn"
        cprint(f"[Mem0-Compact] task name: {task_name}", "red")
        cprint(f"[Mem0-Compact] task type: {self.task_type}", "red")

        self.device = device
        self.episode_id = 0
        self.is_init = 0

        self.action_horizon = self.config.get("action_horizon", 30)
        self.action_strip = self.action_horizon
        self.threshold = self.config.get("threshold", 2)

        self._last_summary: Optional[torch.Tensor] = None
        self._last_state: Optional[torch.Tensor] = None
        self._time_action_history: Dict[int, list] = {}

        self.executor = Mem0CompactExecutor(self.config, device=device).to(device)
        self.executor.eval()
        self._load_ckpt(ckpt_path)

        # COMPACT recurrent state — reset per episode (reset_model).
        self.memory = None
        self.prev_action = None

        self.iter = 0
        self.stage = 0
        self.end_signal_count = 0
        self.action_count = 0
        self.instruction = self.config.get("global_task", "")

        # tmp visual folder (kept for parity with Mem-0's eval artifacts)
        shutil.rmtree("./_tmp_visual/", ignore_errors=True)
        os.makedirs("./_tmp_visual/", exist_ok=True)

    def _load_ckpt(self, ckpt_path: str):
        if not ckpt_path or not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"execution ckpt not found: {ckpt_path}")
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("model_state_dict", payload.get("model", payload))
        missing, unexpected = self.executor.load_state_dict(state_dict, strict=False)
        cprint(f"[Mem0-Compact] loaded ckpt {ckpt_path} "
               f"(missing={len(missing)}, unexpected={len(unexpected)})", "green")

    # ── action-chunk smoothing (copied from Mem-0 agent) ──

    def accumulate_actions_chunk(self, actions_model: np.ndarray) -> None:
        if actions_model is None:
            return
        if actions_model.ndim == 3:
            actions_model = actions_model[0] if actions_model.shape[0] > 0 \
                else actions_model.squeeze(0)
        if actions_model.ndim == 1:
            actions_model = actions_model.reshape(1, -1)
        T, _ = actions_model.shape
        horizon = min(self.action_horizon, T)
        for k in range(horizon):
            t = self.iter + k
            hist = self._time_action_history.get(t, [])
            hist.append(actions_model[k])
            self._time_action_history[t] = hist

    def get_smoothed_actions(self, start: int, length: int) -> np.ndarray:
        steps = []
        for i in range(length):
            t = start + i
            hist = self._time_action_history.get(t)
            if not hist:
                if steps:
                    D = steps[0].shape[-1]
                else:
                    any_hist = next((v for v in self._time_action_history.values() if v), None)
                    D = any_hist[0].shape[-1] if any_hist else 16
                steps.append(np.zeros((D,), dtype=np.float32))
            else:
                steps.append(np.mean(np.stack(hist, axis=0), axis=0))
        return np.stack(steps, axis=0)

    # ── observation / action pipeline ──

    @torch.inference_mode()
    def update_obs(self, obs_payload: Dict[str, object]) -> int:
        """Thread memory through one observation; cache summary + state.

        obs_payload: {"image": PIL, "state": np.ndarray (1,16) normalized,
                      "instruction": str}
        Returns sub_end_flag (always 0 for M1; classifier unused).
        """
        images = [[obs_payload["image"]]]
        instructions = [obs_payload.get("instruction", self.instruction)]
        state = torch.from_numpy(
            np.asarray(obs_payload["state"], dtype=np.float32)).reshape(1, 1, 16)

        if self.memory is None:
            # Episode start: fresh memory, zero prev-action.
            self.memory = self.executor.init_memory(1, self.device)
            prev_action = torch.zeros(1, 16, dtype=torch.float32)
        else:
            prev_action = state.reshape(1, 16).clone()

        batch = {
            "image": images,
            "lang": instructions,
            "state": state,
            "prev_action": prev_action,
        }
        summary, state_out, self.memory = self.executor.update_obs_inference(
            batch, self.memory)
        self._last_summary = summary
        self._last_state = state_out
        self.action_count += 1
        return 0  # M1: no subtask-end signals

    @torch.inference_mode()
    def get_action(self) -> Optional[dict]:
        if self._last_summary is None:
            cprint("[Mem0-Compact] obs cache empty; call update_obs first", "red")
            return None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.executor.action_model.predict_action(
                self._last_summary, self._last_state)
        if pred_actions is None:
            return None
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    # ── episode reset (called by reset_model at episode start) ──

    def reset(self):
        self.memory = None
        self.prev_action = None
        self.iter = 0
        self.action_count = 0
        self.end_signal_count = 0
        self.stage = 0
        self._time_action_history = {}
        self._last_summary = None
        self._last_state = None
        self.is_init = 0

    # ── video (parity with Mem-0 eval artifacts) ──

    def _set_video_ffmpeg(self):
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", "320x240", "-framerate", "4",
                "-i", "-",
                "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
                f"./_tmp_visual/episode_{self.episode_id}.mp4",
            ],
            stdin=subprocess.PIPE,
        )

    def _del_video_ffmpeg(self):
        if hasattr(self, "ffmpeg") and self.ffmpeg is not None:
            if self.ffmpeg.stdin is not None:
                self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            self.ffmpeg = None

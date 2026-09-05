"""
Mem0-Compact deployment agent (M1 tasks first — swap_blocks target).

Adapted from Mem-0's source/agent/memorymatters_agent.py with:
  - MemoryBank removed; COMPACT side memory state (M, P, e) is threaded across
    update_obs calls by the executor and reset ONLY at episode start.
  - prev_action = normalized current state (last executed action, per the
    dataset action[t] = state[t+1] convention); zeros on the first observation
    of an episode.
  - M(1): fixed global instruction, no planner, classifier unused.
  - M(n): planner (Qwen3-VL-8B via vLLM) provides subtask instructions;
    SubtaskEndClassifier fires sub_end signals; threshold crossing saves a
    keyframe, asks the planner for the next subtask, and continues. COMPACT
    memory is NOT reset at subtask boundaries (idea.md decision 3).
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

        nested_horizon = int(self.config.execution_module.action_model.get(
            "action_horizon", 30))
        top_horizon = self.config.get("action_horizon", nested_horizon)
        if top_horizon is not None and int(top_horizon) != nested_horizon:
            raise ValueError(
                "action_horizon mismatch between top-level deployment config "
                f"({top_horizon}) and execution_module.action_model "
                f"({nested_horizon})"
            )
        self.action_horizon = nested_horizon
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

        # Planner (M(n) only; served remotely via vLLM — copied from Mem-0).
        self.high_model = None
        if self.task_type == "Mn":
            from source.models.planning_module.memorymatters_planner import (
                MemoryMattersPlanner)
            planner_cfg_path = str(self.config.get(
                "planning_module_config_path",
                str(PROJECT_ROOT / "source/config/planning_module_inference.yaml")))
            # The yml value is relative to the RMBench repo root (where
            # eval_policy.py runs); fall back to CWD, then the repo root.
            if not Path(planner_cfg_path).is_file():
                repo_root = PROJECT_ROOT.parent.parent  # RMBench/
                alt = repo_root / planner_cfg_path.lstrip("./")
                if Path(alt).is_file():
                    planner_cfg_path = str(alt)
            self.high_model = MemoryMattersPlanner(
                config=OmegaConf.load(planner_cfg_path),
                global_task=self.config.get("global_task", ""),
                vllm_url=self.config.get("vllm_url", "http://localhost:8000"),
            )

        # tmp visual folder (kept for parity with Mem-0's eval artifacts)
        shutil.rmtree("./_tmp_visual/", ignore_errors=True)
        os.makedirs("./_tmp_visual/", exist_ok=True)

    def _load_ckpt(self, ckpt_path: str):
        if not ckpt_path or not Path(ckpt_path).is_file():
            raise FileNotFoundError(f"execution ckpt not found: {ckpt_path}")
        # mmap=True: don't materialize the full state dict in RAM — the 2B
        # model + full ckpt together exceed small dev machines (15GB RAM).
        try:
            payload = torch.load(ckpt_path, map_location="cpu",
                                 weights_only=False, mmap=True)
        except TypeError:  # older torch without mmap support
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        checkpoint_config = payload.get("config") if isinstance(payload, dict) else None
        if checkpoint_config is None:
            raise ValueError(
                "checkpoint has no saved config; refusing to load it because "
                "Mem0-Compact architecture compatibility cannot be verified"
            )

        def config_value(config, path):
            current = config
            for key in path.split("."):
                if hasattr(current, "get"):
                    current = current.get(key)
                elif isinstance(current, dict):
                    current = current.get(key)
                else:
                    return None
                if current is None:
                    return None
            return current

        fields = (
            ("execution_module.compact.mem_dim", int),
            ("execution_module.compact.num_mem_tokens", int),
            ("execution_module.compact.num_heads", int),
            ("execution_module.compact.num_obs_tokens", int),
            ("execution_module.action_model.action_model_type", str),
            ("execution_module.action_model.action_dim", int),
            ("execution_module.action_model.state_dim", int),
            ("execution_module.action_model.action_horizon", int),
            ("execution_module.action_model.num_inference_timesteps", int),
            ("execution_module.use_classifier", bool),
        )
        mismatches = []
        for path, cast in fields:
            expected = config_value(self.config, path)
            actual = config_value(checkpoint_config, path)
            if expected is None or actual is None:
                mismatches.append(f"{path}: missing (runtime={expected}, checkpoint={actual})")
                continue
            if cast(actual) != cast(expected):
                mismatches.append(
                    f"{path}: runtime={cast(expected)!r}, checkpoint={cast(actual)!r}"
                )
        if mismatches:
            raise ValueError(
                "checkpoint architecture does not match deployment config:\n- "
                + "\n- ".join(mismatches)
            )

        state_dict = payload.get("model_state_dict", payload.get("model"))
        if state_dict is None:
            raise ValueError(f"checkpoint contains no model state: {ckpt_path}")
        missing, unexpected = self.executor.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "checkpoint state is incompatible with the reconstructed model: "
                f"missing={missing}, unexpected={unexpected}"
            )
        del payload, state_dict
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
        Returns sub_end_flag: 0 for M(1); for M(n), 1 when the classifier's
        subtask-end probability crosses its threshold (idea.md: memory is
        NOT reset here — only the eval loop reacts).
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

        # M(n): subtask-end signal from the classifier.
        if self.task_type == "Mn":
            prob = self.executor.predict_subtask_end(summary)
            sub_end = 1 if prob >= self.executor.classifier_threshold else 0
            self.end_signal_count += sub_end
            return sub_end
        return 0

    @torch.inference_mode()
    def get_action(self) -> Optional[dict]:
        if self._last_summary is None:
            cprint("[Mem0-Compact] obs cache empty; call update_obs first", "red")
            return None
        pred_actions = self.executor.action_model.predict_action(
            self._last_summary, self._last_state)
        if pred_actions is None:
            return None
        return {"normalized_actions": pred_actions.detach().cpu().numpy()}

    # ── M(n) planner interaction (copied from Mem-0 agent) ──

    def get_instruction(self):
        qwen_inputs = self.high_model.prepare_qwen_input()
        answer = self.high_model.generate_anwser(qwen_inputs)
        subtask = answer.split("next_subtask: ")[-1].split(".")[0]
        self.instruction = subtask
        cprint(f"[Mem0-Compact] high-level instruction: {self.instruction}", "cyan")

    def init_high_with_image(self):
        self.high_model.update_initial_observation("./_tmp_visual/init.png")
        self.get_instruction()
        self._set_video_ffmpeg()

    def update_high_observation(self):
        self.high_model.update_image_or_video_input(
            [f"./_tmp_visual/image_{self.stage}.png"], [self.instruction])
        self.get_instruction()
        self.stage += 1
        self._set_video_ffmpeg()

    # ── episode reset (called by reset_model at episode start) ──

    def reset(self):
        if self.is_init == 1:
            self._del_video_ffmpeg()
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
        if self.high_model is not None:
            self.high_model.key_information = []
            self.high_model.finished_subtasks = []
            self.high_model.initial_observation = None

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

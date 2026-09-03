"""
Mem0-Compact deployment entry (RMBench policy contract:
get_model / eval / reset_model).

Adapted from policy/Mem-0/deploy_policy.py:
  - same encode_obs / normalization / layout tooling (minmax, model layout
    [LA(6), pad, RA(6), pad, LG, RG])
  - MemoryBank gone: COMPACT memory state lives in the agent and is threaded
    across update_obs calls; reset_model() re-initializes it (episode start).
  - M(1): fixed global instruction from config; no planner, no classifier.

Config comes from deploy_policy.yml (+ CLI overrides); the full yml is passed
to the agent so the executor can be reconstructed (execution_module section).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from termcolor import cprint

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from scripts.tools_for_deploy.image_utils import to_pil
from scripts.tools_for_deploy.layout_utils import env_to_model_layout, model_to_env_layout
from scripts.tools_for_deploy.normlization import denormalize_arms, load_stats, normalize_arms
from source.agent.mem0_compact_agent import M1_TASKS, Mem0CompactAgent

# Normalization method (minmax — matches dataset_min_max.py training norm)
NORM_WAY = "minmax"

_RUNTIME_SETTINGS: Dict[str, object] = {
    "camera_key": "head_camera",
    "image_size": (224, 224),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

_STATS: Dict[str, Optional[np.ndarray]] = {
    "state_min": None, "state_max": None,
    "action_min": None, "action_max": None,
}


def _extract_state(observation: dict) -> np.ndarray:
    """Extract the 14-dim RMBench joint vector, padded to 16 (model layout)."""
    joint_vec = observation.get("joint_action", {}).get("vector")
    if joint_vec is None:
        cprint("[deploy] joint_action.vector missing, using zeros", "yellow")
        return np.zeros((16,), dtype=np.float32)
    joint_arr = np.asarray(joint_vec, dtype=np.float32).reshape(-1)
    if joint_arr.size < 16:
        padded = np.zeros((16,), dtype=np.float32)
        padded[: joint_arr.size // 2 - 1] = joint_arr[: joint_arr.size // 2 - 1]
        padded[7] = joint_arr[joint_arr.size // 2 - 1]
        padded[8: 8 + joint_arr.size // 2 - 1] = joint_arr[joint_arr.size // 2: -1]
        padded[15] = joint_arr[joint_arr.size - 1]
        joint_arr = padded
    elif joint_arr.size > 16:
        joint_arr = joint_arr[:16]
    return joint_arr


def _normalize_state(state_vec: np.ndarray) -> np.ndarray:
    if NORM_WAY == "minmax":
        mn = _STATS.get("state_min")
        mx = _STATS.get("state_max")
        if mn is not None and mx is not None:
            return normalize_arms(state_vec, None, None, mn, mx, arm_dims=14)
    return state_vec


def _denormalize_action(action_vec: np.ndarray) -> np.ndarray:
    if NORM_WAY == "minmax":
        mn = _STATS.get("action_min")
        mx = _STATS.get("action_max")
        if mn is not None and mx is not None:
            return denormalize_arms(action_vec, None, None, mn, mx, arm_dims=14)
    return action_vec


def _load_stats(stats_path: str) -> None:
    if not stats_path:
        cprint("[deploy] stats path not provided; skipping stats load", "yellow")
        return
    stats = load_stats(stats_path)
    _STATS.update(stats)
    cprint(f"[deploy] loaded stats from {stats_path}", "cyan")


def _postprocess_action_chunk(actions_model: np.ndarray) -> np.ndarray:
    """Denormalize + reorder model layout → env layout (T, 16)."""
    chunk = np.array(actions_model, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    flat_actions = chunk.reshape(-1, chunk.shape[-1])
    processed = []
    for step in flat_actions:
        denorm = _denormalize_action(step)
        processed.append(model_to_env_layout(denorm))
    return np.stack(processed, axis=0)


def encode_obs(observation: dict) -> Dict[str, object]:
    cam_key = str(_RUNTIME_SETTINGS.get("camera_key", "head_camera"))
    target_size = tuple(_RUNTIME_SETTINGS.get("image_size", (224, 224)))

    obs_block = observation.get("observation", {})
    image_array = obs_block[cam_key]["rgb"]
    if image_array is None:
        raise KeyError("No RGB camera found in observation payload")

    pil_image = to_pil(np.array(image_array), target_size)
    raw_state_env = _extract_state(observation)
    model_state = env_to_model_layout(raw_state_env)
    norm_stats = _normalize_state(model_state)
    return {"image": pil_image,
            "state": norm_stats.reshape(1, -1),
            "instruction": observation.get("instruction", "")}


def get_model(usr_args: dict) -> Mem0CompactAgent:
    device_str = usr_args.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    _RUNTIME_SETTINGS["camera_key"] = usr_args.get("camera_key", "head_camera")
    _RUNTIME_SETTINGS["image_size"] = tuple(usr_args.get("image_size", (224, 224)))
    _RUNTIME_SETTINGS["device"] = device

    ckpt_path = usr_args.get("execution_ckpt", "")
    stats_path = usr_args.get("state_stats_path", "")
    _load_stats(stats_path)
    cfg = OmegaConf.create(usr_args)

    # M(n) tasks need the SubtaskEndClassifier (flat CLI overrides cannot
    # reach nested config keys, so force it here based on the task name).
    task_name = str(usr_args.get("task_name", ""))
    if task_name and task_name not in M1_TASKS:
        cfg.execution_module.use_classifier = True
        cprint(f"[deploy] M(n) task '{task_name}': classifier enabled", "yellow")

    cprint(f"[deploy] device: {device}; camera: {_RUNTIME_SETTINGS['camera_key']}", "cyan")
    cprint(f"[deploy] ckpt: {ckpt_path}", "cyan")
    agent = Mem0CompactAgent(cfg, ckpt_path=ckpt_path, device=device)
    return agent


def eval(TASK_ENV, model: Mem0CompactAgent, observation: dict):
    """Execute one chunk of actions (M1 + Mn paths)."""
    if model.is_init == 0:
        model.is_init = 1
        image = TASK_ENV.now_obs["observation"]["head_camera"]["rgb"]
        Image.fromarray(image).save("./_tmp_visual/init.png")

        # ── Mn: planner provides the first subtask instruction ──
        if model.task_type == "Mn":
            model.init_high_with_image()
        # ── M1: fixed global instruction ──
        if model.task_type == "M1":
            model.instruction = model.config.get("global_task", "")
            model._set_video_ffmpeg()

    instruction = model.instruction
    observation["instruction"] = instruction
    encoded_obs = encode_obs(observation)

    if model.action_count == 0:
        model.update_obs(encoded_obs)
    result = model.get_action()

    if result is None or len(result) == 0:
        cprint("[deploy] no actions produced", "red")
        raise SystemExit("Empty actions from model; aborting eval.")
    else:
        model.accumulate_actions_chunk(result["normalized_actions"])
        smoothed_model_actions = model.get_smoothed_actions(model.iter, model.action_strip)
        actions = _postprocess_action_chunk(smoothed_model_actions)

    if hasattr(model, "ffmpeg") and model.ffmpeg is not None:
        model.ffmpeg.stdin.write(TASK_ENV.now_obs["observation"]["head_camera"]["rgb"].tobytes())

    steps_to_run = min(model.action_strip, actions.shape[0])
    for idx in range(steps_to_run):
        action = actions[idx]
        TASK_ENV.take_action(action, action_type="qpos")

        observation = TASK_ENV.get_obs()
        observation["instruction"] = instruction
        encoded_obs = encode_obs(observation)
        sub_end_flag = model.update_obs(encoded_obs)

        # ── Mn: subtask-end gating (COMPACT memory NOT reset — idea.md §3) ──
        if model.task_type == "Mn":
            if sub_end_flag == 1:
                cprint(f"[deploy] subtask end signal += 1 on [{model.iter}]", "yellow")
            if model.end_signal_count >= model.threshold:
                image = TASK_ENV.now_obs["observation"]["head_camera"]["rgb"]
                Image.fromarray(image).save(f"./_tmp_visual/image_{model.stage}.png")
                break

    # Advance iteration by number of executed steps
    model.iter += steps_to_run

    # ── Mn: subtask transition → planner for the next instruction ──
    if model.task_type == "Mn" and model.end_signal_count >= model.threshold:
        if hasattr(model, "ffmpeg") and model.ffmpeg is not None:
            model.ffmpeg.stdin.write(
                TASK_ENV.now_obs["observation"]["head_camera"]["rgb"].tobytes())
            model._del_video_ffmpeg()
        cprint(f"[deploy] subtask end detected; moving to stage {model.stage + 1}, "
               f"action_count {model.action_count}", "green")
        model.update_high_observation()  # planner → new instruction
        model.end_signal_count = 0
        model.action_count = 0
        # NOTE: executor memory is deliberately NOT reset here (idea.md
        # decision 3: memory persists across subtask boundaries).


def reset_model(model: Mem0CompactAgent):
    model.reset()
    model.episode_id += 1

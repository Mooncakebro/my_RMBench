"""
Deploy round-trip test: training ckpt -> Mem0CompactAgent -> fake observation ->
action chunk, WITHOUT the SAPIEN simulator.

Verifies (idea.md §6.4 prerequisites): ckpt loading, encode_obs normalization,
memory threading across update_obs calls, chunk denormalization + layout.

Example (from policy/Mem0-Compact/, after a debug training run):
    python debug/test_deploy.py --ckpt runs/debug_compact/ckpt_final.pt \
        --stats assets/swap_blocks/norm_stats.json --device cpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import deploy_policy as deploy
from source.dataloader.dataset_min_max import LeRobot_Dataset
from source.agent.mem0_compact_agent import Mem0CompactAgent


def fake_observation(dataset: LeRobot_Dataset, idx: int) -> dict:
    """Build an RMBench-style observation from a dataset frame."""
    sample = dataset[idx]
    img = sample["image"][0]  # PIL 224x224 (already resized by dataset)
    # Reconstruct raw env-layout 14-dim vector from the 16-dim model-layout
    # state: model layout [LA(6), pad, RA(6), pad, LG, RG] -> env layout
    # [LA(6), LG, RA(6), RG].
    state16 = np.asarray(sample["state"]).reshape(16).astype(np.float64)
    env14 = np.concatenate([
        state16[0:6], state16[7:8], state16[8:14], state16[15:16]])
    return {
        "observation": {"head_camera": {"rgb": np.asarray(img)}},
        "joint_action": {"vector": env14.astype(np.float32)},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str,
                   default="runs/debug_compact/ckpt_final.pt")
    p.add_argument("--stats", type=str,
                   default="assets/swap_blocks/norm_stats.json")
    p.add_argument("--task", type=str, default="swap_blocks")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-calls", type=int, default=2)
    args = p.parse_args()

    cfg = OmegaConf.load(PROJECT_ROOT / "deploy_policy.yml")
    cfg.device = args.device
    cfg.execution_ckpt = str(PROJECT_ROOT / args.ckpt)
    cfg.state_stats_path = str(PROJECT_ROOT / args.stats)

    deploy._load_stats(cfg.state_stats_path)
    deploy._RUNTIME_SETTINGS["device"] = torch.device(args.device)

    print(f"[deploy-test] device={args.device}, building agent...")
    agent = Mem0CompactAgent(cfg, ckpt_path=cfg.execution_ckpt,
                             device=torch.device(args.device))
    agent.reset()
    agent.instruction = cfg.global_task

    # Real dataset frames for realistic observations.
    dataset = LeRobot_Dataset(
        repo_id=args.task,
        root=str(PROJECT_ROOT / "lerobot_datasets" / args.task),
        features_to_load=["observation.image.head_camera", "observation.state",
                          "action", "subtask", "subtask_end", "episode_id"],
        image_scale=(224, 224),
    )

    for call in range(args.n_calls):
        obs = fake_observation(dataset, call)
        obs["instruction"] = agent.instruction
        encoded = deploy.encode_obs(obs)
        print(f"[deploy-test] call {call}: encoded state shape "
              f"{encoded['state'].shape}, image {encoded['image'].size}")
        sub_end = agent.update_obs(encoded)
        assert sub_end == 0, "M1 must not emit subtask-end signals"
        result = agent.get_action()
        assert result is not None, "get_action returned None"
        chunk = result["normalized_actions"]
        print(f"[deploy-test] call {call}: chunk {chunk.shape} "
              f"finite={np.isfinite(chunk).all()}")
        assert chunk.shape == (1, agent.action_horizon, 16)
        assert np.isfinite(chunk).all(), "non-finite actions"

        # Postprocess: denorm + layout -> env-ready chunk.
        agent.accumulate_actions_chunk(chunk)
        smoothed = agent.get_smoothed_actions(agent.iter, agent.action_strip)
        env_actions = deploy._postprocess_action_chunk(smoothed)
        print(f"[deploy-test] call {call}: env actions {env_actions.shape}, "
              f"range [{env_actions.min():.3f}, {env_actions.max():.3f}]")
        agent.iter += agent.action_strip

        # Memory must be threaded (not re-initialized) between calls.
        assert agent.memory is not None

    # Episode reset: memory must go back to None.
    agent.reset()
    assert agent.memory is None and agent.iter == 0
    print("\n[deploy-test] agent-level checks passed ✔")

    # ── Full eval_policy contract: get_model / eval / reset_model ──
    print("\n[deploy-test] exercising deploy_policy.get_model/eval/reset_model "
          "contract with FakeTaskEnv...")
    usr_args = dict(OmegaConf.load(PROJECT_ROOT / "deploy_policy.yml"))
    usr_args["device"] = args.device
    usr_args["execution_ckpt"] = cfg.execution_ckpt
    usr_args["state_stats_path"] = cfg.state_stats_path
    model = deploy.get_model(usr_args)

    class FakeTaskEnv:
        """Minimal RMBench TASK_ENV stand-in (what deploy_policy.eval uses)."""

        def __init__(self, obs: dict):
            self.now_obs = obs
            self.eval_success = False
            self.step_lim = 1000
            self.take_action_cnt = 0
            self._frames = []

        def get_obs(self) -> dict:
            return self.now_obs

        def take_action(self, action, action_type="qpos"):
            assert action_type == "qpos"
            self.take_action_cnt += 1
            self._frames.append(np.asarray(action).copy())
            assert np.isfinite(action).all(), "non-finite action"

    env = FakeTaskEnv(fake_observation(dataset, 0))
    deploy.eval(env, model, env.get_obs())
    print(f"[deploy-test] eval() executed {env.take_action_cnt} steps; "
          f"last action range [{np.min(env._frames[-1]):.3f}, "
          f"{np.max(env._frames[-1]):.3f}]")
    assert env.take_action_cnt == model.action_strip, \
        f"expected {model.action_strip} steps, got {env.take_action_cnt}"
    assert model.memory is not None, "memory not threaded through eval()"

    deploy.reset_model(model)
    assert model.memory is None and model.iter == 0 and model.action_count == 0
    print("[deploy-test] reset_model OK (memory cleared, counters zeroed)")

    print("\n[deploy-test] ALL CHECKS PASSED ✔")


if __name__ == "__main__":
    main()

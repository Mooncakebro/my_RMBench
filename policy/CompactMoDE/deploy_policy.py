"""RMBench deployment for CompactMoDE (contract: get_model / eval / reset_model).

Loaded by script/eval_policy.py. Configuration comes from deploy_policy.yml
(+ CLI overrides); the CompactMoDE-specific keys are:

    variant:           "compact" | "baseline"
    checkpoint_path:   directory written by save_pretrained() (e.g. runs/.../final)
    num_sampling_steps: EDM DDIM steps (default 10)

Per eval() call the policy sees ONE observation, samples a full action chunk
(10 x 14, normalized -> decoded to raw joint targets), executes only the first
action, and returns to the RMBench loop for a fresh observation. The COMPACT
variant memory is updated on every eval() call; reset_model() re-initializes it
at episode start.

NOTE: untested in the simulator on this dev machine (no SAPIEN env here) —
verify on the eval server before drawing conclusions.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))


class CompactMoDEDeployer:
    def __init__(self, usr_args):
        variant = str(usr_args.get("variant", "compact")).lower()
        ckpt = usr_args["checkpoint_path"]
        steps = int(usr_args.get("num_sampling_steps", 10))
        self.num_sampling_steps = steps
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if variant == "compact":
            from compact_mode.model_compact import CompactMoDEPolicy
            self.policy = CompactMoDEPolicy.load_pretrained(ckpt, device=str(self.device))
        elif variant == "baseline":
            from compact_mode.model_baseline import BaselineQwenMoDEPolicy
            self.policy = BaselineQwenMoDEPolicy.load_pretrained(ckpt, device=str(self.device))
        else:
            raise ValueError(f"unknown variant: {variant}")
        self.policy.eval()
        self.variant = variant
        self.memory = None
        self.prev_action = None  # normalized (14,)
        self.action_dim = self.policy.cfg.action_dim

    def reset(self):
        self.memory = None
        self.prev_action = None

    def _build_batch(self, observation, instruction: str) -> dict:
        image = observation["observation"]["head_camera"]["rgb"]  # uint8 HWC
        state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        state_norm = self.policy.state_normalizer.encode(torch.from_numpy(state.copy()))
        if self.prev_action is None:
            prev = torch.zeros(self.action_dim, dtype=torch.float32)
        else:
            prev = self.prev_action
        return {
            "images": [image],
            "instructions": [instruction],
            "state": state_norm.unsqueeze(0),
            "prev_action": prev.unsqueeze(0),
        }

    @torch.no_grad()
    def get_action_chunk(self, observation, instruction: str) -> np.ndarray:
        batch = self._build_batch(observation, instruction)
        if self.variant == "compact":
            actions, self.memory = self.policy.sample_actions(
                batch, self.memory, steps=self.num_sampling_steps)
        else:
            actions = self.policy.sample_actions(batch, steps=self.num_sampling_steps)
        actions = actions[0].float().cpu()  # (chunk, 14) normalized
        # The memory controller receives the action that was actually
        # executed, not the final action from the unexecuted prediction
        # horizon.  This keeps the next forward step causally aligned with
        # the environment.
        self.prev_action = actions[0].clone()
        raw = self.policy.action_normalizer.decode(actions)
        return raw.numpy()


def get_model(usr_args):
    return CompactMoDEDeployer(usr_args)


def eval(TASK_ENV, model: CompactMoDEDeployer, observation):
    instruction = TASK_ENV.get_instruction()
    actions = model.get_action_chunk(observation, instruction)
    # Receding-horizon control: predict a 10-step chunk, execute only the
    # first action, then let the outer RMBench loop observe and replan.
    TASK_ENV.take_action(actions[0], action_type="qpos")


def reset_model(model: CompactMoDEDeployer):
    model.reset()

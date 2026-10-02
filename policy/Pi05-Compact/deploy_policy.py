"""RMBench deployment adapter for Pi05-Compact.

The adapter keeps COMPACT state across calls to ``get_action`` and resets it
at episode boundaries.  Environment actions are 14-D; the model uses the
16-D padded layout documented in ``idea.md``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(THIS_DIR))

from source.models.pi05_compact_model import Pi05CompactModel


class Pi05CompactPolicy:
    def __init__(self, config_path, checkpoint, device=None, instruction="", norm_stats=None,
                 tokenizer_path=None):
        values = yaml.safe_load(Path(config_path).read_text())
        cfg = SimpleNamespace(**values["model"], **values.get("compact", {}))
        self.model = Pi05CompactModel(cfg, freeze_base=bool(cfg.freeze_base))
        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state.get("model", state), strict=False)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device).eval()
        from openpi.models.tokenizer import PaligemmaTokenizer
        self.tokenizer = PaligemmaTokenizer(
            cfg.max_token_len,
            tokenizer_path or os.environ.get("PALIGEMMA_TOKENIZER_PATH"),
        )
        self.instruction = instruction
        if norm_stats is None:
            raise ValueError("norm_stats is required for Pi05-Compact deployment")
        self.norm_stats = json.loads(Path(norm_stats).read_text())
        self.memory = None
        self.prev_action = torch.zeros(1, 16, device=self.device)

    @staticmethod
    def _pack(state14):
        state14 = np.asarray(state14, dtype=np.float32)
        return np.asarray([*state14[:6], 0.0, *state14[6:12], 0.0, state14[12], state14[13]], dtype=np.float32)

    @staticmethod
    def _unpack(action16):
        return np.asarray([*action16[:6], *action16[7:13], action16[14], action16[15]], dtype=np.float32)

    def _normalize(self, value, key):
        stats = self.norm_stats[key]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        return (np.asarray(value, dtype=np.float32) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def _unnormalize(self, value, key):
        stats = self.norm_stats[key]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        return (np.asarray(value, dtype=np.float32) + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01

    def reset(self):
        self.memory = None
        self.prev_action.zero_()

    def get_action(self, observation):
        state = self._normalize(self._pack(observation["joint_action"]["vector"]), "state")
        images = {}
        for out_key, in_key in (("base_0_rgb", "head_camera"), ("left_wrist_0_rgb", "left_camera"), ("right_wrist_0_rgb", "right_camera")):
            image = torch.as_tensor(observation["observation"][in_key]["rgb"], device=self.device)
            images[out_key] = image[None].float() / 127.5 - 1.0
        tokens, mask = self.tokenizer.tokenize(self.instruction, state)
        obs = SimpleNamespace(
            images=images,
            image_masks={key: torch.ones(1, dtype=torch.bool, device=self.device) for key in images},
            state=torch.from_numpy(state)[None].to(self.device),
            tokenized_prompt=torch.from_numpy(tokens)[None].to(self.device),
            tokenized_prompt_mask=torch.from_numpy(mask)[None].to(self.device),
            token_ar_mask=None, token_loss_mask=None,
        )
        action, self.memory = self.model.sample_actions(obs, self.memory, self.prev_action)
        # The RMBench callback executes chunk[0] immediately.  The next
        # recurrent input must therefore be the action actually executed, not
        # the final action in the predicted chunk.
        normalized = action[0].cpu().numpy()
        self.prev_action = torch.from_numpy(normalized[0].copy())[None].to(self.device)
        return np.stack([self._unpack(self._unnormalize(x, "actions")) for x in normalized])


def get_model(usr_args):
    config_path = usr_args.get("model_config", usr_args.get("config_path", "policy/Pi05-Compact/source/config/pi05_compact_train.yaml"))
    checkpoint = usr_args.get("checkpoint", usr_args.get("checkpoint_path", "policy/Pi05-Compact/checkpoints/pi05_compact/ckpt_final.pt"))
    return Pi05CompactPolicy(config_path, checkpoint,
                             device=usr_args.get("device"),
                             instruction=usr_args.get("global_task", usr_args.get("instruction", "")),
                             norm_stats=usr_args.get("norm_stats"),
                             tokenizer_path=usr_args.get("tokenizer_path"))


def eval(TASK_ENV, model, observation):
    """RMBench harness callback: predict and execute the first action."""
    if not model.instruction:
        model.instruction = TASK_ENV.get_instruction()
    chunk = model.get_action(observation)
    if chunk.ndim != 2 or chunk.shape[0] == 0:
        raise RuntimeError("Pi05-Compact produced an empty action chunk")
    TASK_ENV.take_action(chunk[0])


def reset_model(model):
    model.reset()

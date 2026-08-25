"""
MoDeDiT subclass with proper multi-token state positional embeddings.

MoDE's MoDeDiT.forward hardcodes a single state token (`t = 1`; see the
position_embeddings slices around modedit.py:782): with 32 pooled state tokens
all of them would share one broadcast position vector. We are not allowed to
modify the MoDE repo, so this subclass re-implements forward() with
`n_state_tokens` distinct learned position slots (zero-initialized, same as
the base class) laid out as:

    [noise | goal(goal_seq_len) | proprio(1) | state(n_state_tokens) | action(action_seq_len)]

Only forward() and the pos_emb size differ from the base class; everything
else (blocks, router, output head) is inherited unchanged.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

import compact_mode  # noqa: F401  (ensures MODE_ROOT is on sys.path)
from mode.models.networks.modedit import MoDeDiT


class StateTokenMoDeDiT(MoDeDiT):
    def __init__(self, *args, n_state_tokens: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_state_tokens = n_state_tokens
        seq_size = (
            self.goal_seq_len
            + (1 if self.use_proprio else 0)
            + n_state_tokens
            + self.action_seq_len
        )
        # Zero-init exactly like the base class's pos_emb.
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_size, self.embed_dim))

    def forward(
        self,
        states,
        actions,
        goals,
        sigma,
        uncond: Optional[bool] = False,
    ):
        n_state = self.n_state_tokens
        g = self.goal_seq_len

        emb_t = self.process_sigma_embeddings(sigma)

        goals = self.preprocess_goals(goals, 1, uncond=uncond)
        if len(goals.shape) == 2:
            goals = goals.unsqueeze(1)

        state_embed = self.tok_emb(states["state_images"])      # (B, n_state, obs_dim)->(B, n_state, D)
        if "robot_obs" in states and self.use_proprio:
            proprio_embed = self.process_state_obs(states["robot_obs"].to(goals.dtype))
        else:
            proprio_embed = None
        goal_embed = self.goal_emb(goals)                       # (B, goal_seq_len, D)
        action_embed = self.action_emb(actions)                 # (B, action_seq_len, D)

        pos = self.pos_emb
        goal_x = self.drop(goal_embed + pos[:, :g, :])
        cursor = g
        if proprio_embed is not None:
            if proprio_embed.dim() == 2:
                proprio_embed = proprio_embed.unsqueeze(1)  # (B, D) -> (B, 1, D)
            assert proprio_embed.shape[1] == 1, "only obs_horizon=1 is supported"
            proprio_x = self.drop(proprio_embed + pos[:, cursor:cursor + 1, :])
            cursor += 1
        else:
            proprio_x = None
        state_x = self.drop(state_embed + pos[:, cursor:cursor + n_state, :])
        cursor += n_state
        action_x = self.drop(action_embed + pos[:, cursor:cursor + self.action_seq_len, :])

        input_seq = self.build_input_seq(state_x, action_x, goal_x, emb_t, proprio_x)

        if self.use_custom_attn_mask:
            custom_mask = self.create_custom_mask(input_seq.shape[1])
        else:
            custom_mask = None

        cond_token = emb_t
        if self.use_goal_in_routing:
            cond_token = cond_token + goal_embed

        x = self.forward_modedit(input_seq, cond_token, custom_attn_mask=custom_mask)
        action_outputs = x[:, -self.action_seq_len:, :]
        pred_actions = self.out(action_outputs)
        return pred_actions

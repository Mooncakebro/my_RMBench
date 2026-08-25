"""
Conditioning bridge between the VLM hidden states and the MoDE DiT.

Replaces the pool-everything approach of MoDE's my_code/model_qwen.py:

  - image-position hidden states (final layer; post-injection for the COMPACT
    variant) -> adaptive-avg-pool to `n_state_tokens` tokens -> Linear -> obs_dim
    = `state_images`  (B, n_state_tokens, obs_dim)
  - instruction-text-position hidden states -> masked mean -> Linear -> goal_dim
    = `goal`          (B, 1, goal_dim)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_to_n_tokens(tokens: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """tokens: (n_in, D) -> (n_tokens, D) via adaptive average pooling."""
    if tokens.shape[0] == n_tokens:
        return tokens
    return F.adaptive_avg_pool1d(
        tokens.transpose(0, 1).unsqueeze(0), n_tokens
    ).squeeze(0).transpose(0, 1)


class ConditioningBridge(nn.Module):
    def __init__(self, hidden_dim: int, n_state_tokens: int = 32,
                 obs_dim: int = 2048, goal_dim: int = 512):
        super().__init__()
        self.n_state_tokens = n_state_tokens
        self.obs_proj = nn.Linear(hidden_dim, obs_dim)
        self.goal_proj = nn.Linear(hidden_dim, goal_dim)

    def forward(
        self,
        hidden: torch.Tensor,       # (B, N, hidden_dim) float32
        image_mask: torch.Tensor,   # (B, N) bool — image token positions
        text_mask: torch.Tensor,    # (B, N) bool — instruction text positions
    ):
        B = hidden.shape[0]
        pooled = []
        goals = []
        for b in range(B):
            img_tokens = hidden[b][image_mask[b]]               # (n_img, H)
            if img_tokens.shape[0] == 0:
                raise ValueError("No image tokens found in hidden states")
            pooled.append(pool_to_n_tokens(img_tokens, self.n_state_tokens))

            txt_tokens = hidden[b][text_mask[b]]                # (n_txt, H)
            if txt_tokens.shape[0] == 0:
                # Fallback: mean over non-image positions.
                txt_tokens = hidden[b][~image_mask[b]]
            goals.append(txt_tokens.mean(dim=0))

        state_images = self.obs_proj(torch.stack(pooled, dim=0))      # (B, 32, obs_dim)
        goal = self.goal_proj(torch.stack(goals, dim=0)).unsqueeze(1)  # (B, 1, goal_dim)
        return {"state_images": state_images, "goal": goal}

"""
Configuration for CompactMoDE (model + DiT + training knobs).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class CompactMoDEConfig:
    # ── VLM base ──
    base_model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    freeze_base: bool = False           # trainable by default; shell FREEZE_BASE=1 flips this
    bf16: bool = True
    gradient_checkpointing: bool = False

    # ── COMPACT side memory (COMPACT variant only) ──
    mem_dim: int = 16
    num_mem_tokens: int = 16
    num_heads: int = 8
    num_obs_tokens: int = 4
    lambda_obs: float = 0.01            # L_obs weight  (COMPACT aux loss)
    lambda_nll: float = 0.01            # L_nll weight  (COMPACT aux loss)
    lambda_mem: float = 0.001           # memory L2 weight (COMPACT aux loss)

    # ── Action / state ──
    action_dim: int = 14                # [left_arm(6), left_gripper, right_arm(6), right_gripper]
    state_dim: int = 14
    action_seq_len: int = 10
    gripper_indices: tuple = (6, 13)    # minmax-normalized dims; the rest are z-scored
    normalize: bool = True

    # ── Conditioning bridge ──
    n_state_tokens: int = 32            # pooled image tokens -> DiT state_images
    obs_dim: int = 2048                 # DiT state-image token dim
    goal_dim: int = 512                 # DiT goal dim (goal shape (B, 1, 512))

    # ── MoDE DiT ──
    embed_dim: int = 1024
    n_layers: int = 6
    n_heads: int = 8
    num_experts: int = 4
    top_k: int = 2
    embed_pdrop: float = 0.0
    attn_pdrop: float = 0.3
    mlp_pdrop: float = 0.1
    goal_drop: float = 0.1
    goal_seq_len: int = 1
    linear_output: bool = True
    cond_router: bool = True
    router_normalize: bool = True
    use_goal_in_routing: bool = False
    use_argmax: bool = False
    use_shared_expert: bool = False
    use_noise_token_as_input: bool = True
    use_custom_attn_mask: bool = False
    init_style: str = "olmoe"

    # ── EDM diffusion ──
    sigma_data: float = 0.5
    sigma_min: float = 0.001
    sigma_max: float = 80.0
    num_sampling_steps: int = 10

    # ── Training ──
    lr: float = 1e-4                    # bridge + DiT
    base_lr: float = 1e-5               # VLM base (when trainable)
    memory_lr: float = 5e-6             # COMPACT side memory (+ prev-action MLP)
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    chunk_size: int = 8                 # TBPTT chunk length (COMPACT variant)
    seed: int = 42

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gripper_indices"] = list(d["gripper_indices"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CompactMoDEConfig":
        d = dict(d)
        if "gripper_indices" in d:
            d["gripper_indices"] = tuple(d["gripper_indices"])
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save_json(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load_json(cls, path) -> "CompactMoDEConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

"""PyTorch pi0.5 with JAMEL-COMPACT memory on the PaliGemma prefix.

The action expert is deliberately left untouched.  COMPACT state is threaded
outside the model so the trainer/deployer can reset individual episode slots
and truncate BPTT without coupling it to the optimizer state.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
_VENDOR = _ROOT / "vendor"
if str(_VENDOR / "openpi_torch") not in sys.path:
    sys.path.insert(0, str(_VENDOR / "openpi_torch"))
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from jamel_compact.config import CompactConfig
from jamel_compact.model import SideMemoryModule


class Pi05CompactModel(nn.Module):
    """A pi0.5 model whose 18 PaliGemma layers carry COMPACT state."""

    def __init__(self, config, *, freeze_base: bool = False, mem_dim: int = 128,
                 num_mem_tokens: int = 16, num_heads: int = 8,
                 num_obs_tokens: int = 4, action_input_dim: int = 16):
        super().__init__()
        self.config = config
        self.base = PI0Pytorch(config)
        vlm = self.base.paligemma_with_expert
        hidden_dim = int(vlm.paligemma.config.text_config.hidden_size)
        num_layers = int(vlm.paligemma.config.text_config.num_hidden_layers)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.mem_dim = mem_dim
        self.num_mem_tokens = num_mem_tokens
        self.action_dim = action_input_dim
        self.model_action_dim = int(self.base.action_in_proj.in_features)

        compact_cfg = CompactConfig(
            mem_dim=mem_dim,
            num_mem_tokens=num_mem_tokens,
            num_heads=num_heads,
            num_obs_tokens=num_obs_tokens,
            freeze_base=freeze_base,
        )
        self.side_memories = nn.ModuleList([
            SideMemoryModule(
                layer_idx=i,
                num_layers=num_layers,
                hidden_dim=hidden_dim,
                mem_dim=mem_dim,
                num_mem=num_mem_tokens,
                num_heads=num_heads,
                num_obs_tokens=num_obs_tokens,
                config=compact_cfg,
            ) for i in range(num_layers)
        ])
        # Keep recurrent projections numerically aligned with the PaliGemma
        # stream.  This avoids bf16/float32 matmul mismatches when autocast is
        # disabled (notably in CPU smoke tests).
        memory_dtype = vlm.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        self.side_memories.to(dtype=memory_dtype)
        self.action_encoder = nn.Sequential(
            nn.LayerNorm(action_input_dim),
            nn.Linear(action_input_dim, 512),
            nn.GELU(),
            nn.Linear(512, hidden_dim),
        )
        self.lambda_obs = float(getattr(config, "lambda_obs", 0.01))
        self.lambda_nll = float(getattr(config, "lambda_nll", 0.01))
        self.lambda_mem = float(getattr(config, "lambda_mem", 0.001))
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    def _to_model_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Pad the 16-D RMBench action into pi0.5's 32-D action head."""
        if actions.shape[-1] == self.model_action_dim:
            return actions
        if actions.shape[-1] > self.model_action_dim:
            raise ValueError(f"action dimension {actions.shape[-1]} exceeds {self.model_action_dim}")
        return F.pad(actions, (0, self.model_action_dim - actions.shape[-1]))

    @property
    def pi0(self) -> PI0Pytorch:
        return self.base

    def load_base_checkpoint(self, checkpoint, *, strict: bool = False):
        """Warm-start only π0.5 weights, leaving COMPACT randomly initialized."""
        if isinstance(checkpoint, (str, Path)):
            checkpoint = Path(checkpoint)
            if checkpoint.is_dir():
                checkpoint = checkpoint / "model.safetensors"
            if checkpoint.suffix == ".safetensors":
                from safetensors.torch import load_file
                state = load_file(str(checkpoint), device="cpu")
            else:
                state = torch.load(checkpoint, map_location="cpu")
        else:
            state = checkpoint
        if isinstance(state, dict):
            state = state.get("model", state.get("state_dict", state))
            state = {k.removeprefix("base."): v for k, v in state.items() if k.startswith("base.")} or state
        return self.base.load_state_dict(state, strict=strict)

    def init_memory(self, batch_size: int, device=None) -> Dict[str, list]:
        device = device or next(self.parameters()).device
        # Memory states are consumed by the side-memory projections.  Do not
        # infer their dtype from the first base parameter: PaliGemma keeps
        # selected vision parameters in float32 while the decoder/COMPACT
        # stream is normally bfloat16.
        dtype = next(self.side_memories.parameters()).dtype
        memories, variances = [], []
        for sm in self.side_memories:
            memories.append(sm.init_memory[None].expand(batch_size, -1, -1).clone().to(device=device, dtype=dtype))
            variances.append(torch.full((batch_size, self.num_mem_tokens), sm.init_variance,
                                        device=device, dtype=dtype))
        return {"m": memories, "p": variances, "e": [None] * self.num_layers}

    @staticmethod
    def detach_memory(memory: Dict[str, list]) -> Dict[str, list]:
        return {
            "m": [x.detach() for x in memory["m"]],
            "p": [x.detach() for x in memory["p"]],
            "e": [x.detach() if isinstance(x, torch.Tensor) else None for x in memory["e"]],
        }

    def reset_memory_rows(self, memory: Dict[str, list], reset_mask, device=None):
        if not any(reset_mask):
            return memory
        device = device or memory["m"][0].device
        idx = torch.as_tensor(reset_mask, dtype=torch.bool, device=device)
        fresh = self.init_memory(len(reset_mask), device)
        for l in range(self.num_layers):
            memory["m"][l] = torch.where(idx[:, None, None], fresh["m"][l], memory["m"][l])
            memory["p"][l] = torch.where(idx[:, None], fresh["p"][l], memory["p"][l])
            old_e = memory["e"][l]
            if old_e is None:
                old_e = torch.zeros(len(reset_mask), device=device, dtype=fresh["p"][l].dtype)
            memory["e"][l] = torch.where(idx, torch.zeros_like(old_e), old_e)
        return memory

    def _inputs(self, observation, *, train: bool):
        return self.base._preprocess_observation(observation, train=train)

    def _forward_impl(self, observation, actions: torch.Tensor, memory: Optional[Dict],
                      prev_action: Optional[torch.Tensor], noise=None, time=None,
                      train: bool = True):
        images, img_masks, lang_tokens, lang_masks, state = self._inputs(observation, train=train)
        device = actions.device
        actions = self._to_model_actions(actions)
        if noise is None:
            noise = self.base.sample_noise(actions.shape, device)
        else:
            noise = self._to_model_actions(noise)
        if time is None:
            time = self.base.sample_time(actions.shape[0], device)
        x_t = time[:, None, None] * noise + (1.0 - time[:, None, None]) * actions
        target = noise - actions

        prefix, prefix_pad, prefix_att = self.base.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix, suffix_pad, suffix_att, adarms = self.base.embed_suffix(state, x_t, time)
        if prefix.dtype != self.base.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype:
            prefix = prefix.to(dtype=self.base.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype)
            suffix = suffix.to(dtype=prefix.dtype)
        pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        att = torch.cat([prefix_att, suffix_att], dim=1)
        mask4d = self.base._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
        pos = torch.cumsum(pad, dim=1) - 1
        if memory is None:
            memory = self.init_memory(actions.shape[0], device)
        if prev_action is None:
            prev_action = torch.zeros(actions.shape[0], 16, device=device, dtype=torch.float32)
        action_embed = self.action_encoder(prev_action.to(device=device, dtype=torch.float32))
        action_embed = action_embed.to(dtype=next(self.side_memories.parameters()).dtype)
        result, _ = self.base.paligemma_with_expert.forward(
            attention_mask=mask4d,
            position_ids=pos,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            adarms_cond=[None, adarms],
            side_memories=self.side_memories,
            memory_states=memory["m"],
            variance_states=memory["p"],
            e_prev_list=memory["e"],
            prefix_observation_mask=prefix_pad,
            action_embed=action_embed,
        )
        suffix_out = result[1][:, -self.config.action_horizon:].float()
        velocity = self.base.action_out_proj(suffix_out)
        updates = getattr(self.base.paligemma_with_expert, "_last_compact_updates", {})
        new_m, new_p, new_e, obs_losses, nll_losses = [], [], [], [], []
        for i in range(self.num_layers):
            if i not in updates:
                raise RuntimeError(f"COMPACT layer {i} did not return a state update")
            m, p, e, lo, ln = updates[i]
            new_m.append(m)
            new_p.append(p)
            new_e.append(e)
            obs_losses.append(lo)
            nll_losses.append(ln)
        new_memory = {"m": new_m, "p": new_p, "e": new_e}
        aux = {
            "obs": torch.stack([x.float() for x in obs_losses]).mean(),
            "nll": torch.stack([x.float() for x in nll_losses]).mean(),
        }
        flow = F.mse_loss(velocity.float(), target.float(), reduction="none")
        mem_l2 = torch.stack([m.float().pow(2).sum() for m in new_m]).mean()
        total = flow.mean() + self.lambda_obs * aux["obs"] + self.lambda_nll * aux["nll"] + self.lambda_mem * mem_l2
        return {"flow": flow[..., :self.action_dim], "velocity": velocity[..., :self.action_dim], "aux": aux, "mem_l2": mem_l2,
                "total": total}, new_memory

    def _velocity_from_embeds(self, prefix, prefix_pad, prefix_att, state,
                              x_t, time, memory, action_embed):
        """Run one COMPACT-aware joint velocity evaluation for sampling.

        The caller owns ``memory``.  This function deliberately ignores the
        recurrent updates emitted by the joint model: sampling updates memory
        once per observation, while Euler denoising is a suffix-only operation
        conceptually.  Prefix tokens are nevertheless run through the joint
        layers so the injected memory is visible to the action expert.
        """
        suffix, suffix_pad, suffix_att, adarms = self.base.embed_suffix(state, x_t, time)
        if prefix.dtype != self.base.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype:
            prefix = prefix.to(dtype=self.base.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype)
            suffix = suffix.to(dtype=prefix.dtype)
        pad = torch.cat([prefix_pad, suffix_pad], dim=1)
        att = torch.cat([prefix_att, suffix_att], dim=1)
        mask4d = self.base._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
        pos = torch.cumsum(pad, dim=1) - 1
        result, _ = self.base.paligemma_with_expert.forward(
            attention_mask=mask4d,
            position_ids=pos,
            past_key_values=None,
            inputs_embeds=[prefix, suffix],
            use_cache=False,
            adarms_cond=[None, adarms],
            side_memories=self.side_memories,
            memory_states=memory["m"],
            variance_states=memory["p"],
            e_prev_list=memory["e"],
            prefix_observation_mask=prefix_pad,
            action_embed=action_embed,
            compact_update=False,
        )
        suffix_out = result[1][:, -self.config.action_horizon:].float()
        return self.base.action_out_proj(suffix_out)

    def forward(self, observation, actions, memory=None, prev_action=None, noise=None, time=None):
        return self._forward_impl(observation, actions, memory, prev_action, noise, time, train=True)

    @torch.no_grad()
    def sample_actions(self, observation, memory=None, prev_action=None, noise=None, num_steps=10):
        """Sample an action chunk and return the memory for the next frame.

        A prefix pass updates COMPACT once.  Euler steps use the resulting
        state as a read-only condition; their diagnostic updates are discarded.
        This is slower than the stock KV-cache sampler, but it is semantically
        correct for recurrent prefix injection and is a reliable reference
        implementation for deployment.
        """
        if memory is None:
            memory = self.init_memory(observation.state.shape[0], observation.state.device)
        device = observation.state.device
        bsize = observation.state.shape[0]
        zero_actions = torch.zeros(bsize, self.config.action_horizon,
                                   self.action_dim, device=device)
        _, new_memory = self._forward_impl(
            observation, zero_actions, memory, prev_action,
            noise=torch.zeros_like(zero_actions),
            time=torch.ones(bsize, device=device), train=False,
        )

        images, img_masks, lang_tokens, lang_masks, state = self._inputs(observation, train=False)
        prefix, prefix_pad, prefix_att = self.base.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        action_embed = self.action_encoder(
            (prev_action if prev_action is not None else torch.zeros(bsize, 16, device=device)).to(torch.float32)
        )
        action_embed = action_embed.to(dtype=next(self.side_memories.parameters()).dtype)
        if noise is None:
            x_t = self.base.sample_noise(
                (bsize, self.config.action_horizon, self.model_action_dim), device
            )
        else:
            x_t = self._to_model_actions(noise.to(device=device))
        dt = torch.tensor(-1.0 / float(num_steps), dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            velocity = self._velocity_from_embeds(
                prefix, prefix_pad, prefix_att, state, x_t,
                time.expand(bsize), new_memory, action_embed,
            )
            x_t = x_t + dt * velocity
            time = time + dt
        return x_t[..., :self.action_dim], new_memory

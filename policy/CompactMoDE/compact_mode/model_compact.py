"""
COMPACT+MoDE variant: JAMELCompactWrapper (Qwen3-VL-2B + per-layer side
memory) + conditioning bridge + MoDeDiT.

Differences vs the baseline:
  - 28 side-memory modules ride on the decoder; memory is carried across the
    frames of an episode and trained with chunked TBPTT (detach at chunk
    boundaries, like jamel_compact/train.py).
  - The memory "predict" step consumes a continuous action embedding instead
    of text: prev_action -> MLP(14 -> 256 -> hidden_dim) -> wrapper.action_embed.
    At t=0 (episode start) the previous action is zeros. The NORMALIZED action
    is used.
  - The DiT is conditioned ONLY on VLM hidden states (final layer,
    post-injection) — never on memory slots.

Requires NO modification to the JAMEL-COMPACT repo: the wrapper used here is
compact_mode/compact_wrapper.CompactModeWrapper, a JAMELCompactWrapper
subclass whose forward() is copied from the original and adapted to always
return final-layer post-injection hidden states plus the per-step
"loss_obs" / "loss_nll" aux terms (and to skip the LM head entirely).

NOTE on visual features: transformers 4.57.6's `get_image_features` returns a
plain tuple `(image_embeds, deepstack_embeds)`, which JAMEL-COMPACT's
`_inject_visual_features` cannot parse (it expects `.pooler_output`). We
therefore precompute embeddings + visual injection ourselves and use the
wrapper's `inputs_embeds` / `deepstack_features` / `visual_pos_mask` path.

NOTE on mm_token_type_ids: the wrapper refuses image inputs without it, but
transformers 4.57.6's Qwen3-VL processor never produces it. We construct it
ourselves (1 at image-token positions); the wrapper's M-RoPE helper falls
back to `get_rope_index`, which ignores the tensor — it is only needed to
pass the guard.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

import compact_mode  # noqa: F401  (sys.path setup)
from jamel_compact.config import CompactConfig

from .bridge import ConditioningBridge
from .compact_wrapper import CompactModeWrapper
from .config import CompactMoDEConfig
from .losses import compact_aux_loss, edm_loss, get_sigmas_exponential, sample_ddim
from .model_baseline import build_dit_denoiser, build_vlm_inputs
from .normalizer import JointGripperNormalizer


class CompactMoDEPolicy(nn.Module):
    def __init__(self, cfg: CompactMoDEConfig):
        super().__init__()
        self.cfg = cfg
        compact_cfg = CompactConfig(
            base_model_name=cfg.base_model_name,
            mem_dim=cfg.mem_dim,
            num_mem_tokens=cfg.num_mem_tokens,
            num_heads=cfg.num_heads,
            num_obs_tokens=cfg.num_obs_tokens,
            freeze_base=cfg.freeze_base,
            bf16=cfg.bf16,
            gradient_checkpointing=cfg.gradient_checkpointing,
            lambda_obs=cfg.lambda_obs,
            lambda_nll=cfg.lambda_nll,
            lambda_mem=cfg.lambda_mem,
            lora_rank=0,  # peft is not installed; LoRA disabled
            chunk_size=cfg.chunk_size,
        )
        self.wrapper = CompactModeWrapper(compact_cfg)
        self.processor = self.wrapper.processor
        self.image_token_id = int(getattr(self.wrapper.llm.config, "image_token_id",
                                          self.processor.image_token_id))
        self.hidden_dim = self.wrapper.hidden_dim
        self.num_layers = self.wrapper.num_layers

        # Continuous prev-action embedding for the memory predict step.
        # (The brief's 1536 target assumed a 1536-dim VLM; Qwen3-VL-2B's hidden
        # size is 2048, so the MLP outputs hidden_dim = 2048.)
        llm_dtype = next(self.wrapper.llm.parameters()).dtype
        self.prev_action_mlp = nn.Sequential(
            nn.Linear(cfg.action_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.hidden_dim),
        ).to(dtype=llm_dtype)

        self.bridge = ConditioningBridge(
            self.hidden_dim, cfg.n_state_tokens, cfg.obs_dim, cfg.goal_dim)
        self.denoiser = build_dit_denoiser(cfg, "cpu")
        self.action_normalizer: Optional[JointGripperNormalizer] = None
        self.state_normalizer: Optional[JointGripperNormalizer] = None

    # ── housekeeping ──

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def to(self, *args, **kwargs):
        module = super().to(*args, **kwargs)
        self.denoiser.inner_model.device = str(module.device)
        return module

    def set_normalizers(self, action_norm, state_norm):
        self.action_normalizer = action_norm
        self.state_normalizer = state_norm

    # ── memory state handling ──

    def init_memory(self, batch_size: int, device) -> Dict:
        m_states, p_states = self.wrapper.init_memory(batch_size, device)
        return {"m": m_states, "p": p_states, "e": [None] * self.num_layers}

    @staticmethod
    def detach_memory(memory: Dict) -> Dict:
        """Detach carried state at TBPTT chunk boundaries."""
        return {
            "m": [t.detach() for t in memory["m"]],
            "p": [t.detach() for t in memory["p"]],
            "e": [t.detach() if isinstance(t, torch.Tensor) else None
                  for t in memory["e"]],
        }

    def reset_memory_rows(self, memory: Dict, reset_mask: List[bool], device) -> Dict:
        """Re-initialize memory for the batch rows that started a new episode."""
        if not any(reset_mask):
            return memory
        idx = torch.tensor(reset_mask, device=device)
        init_m, init_p = self.wrapper.init_memory(len(reset_mask), device)
        new_e = []
        for l in range(self.num_layers):
            memory["m"][l] = torch.where(idx.view(-1, 1, 1), init_m[l], memory["m"][l])
            memory["p"][l] = torch.where(idx.view(-1, 1), init_p[l], memory["p"][l])
            e = memory["e"][l]
            if e is None:
                e = torch.zeros(len(reset_mask), device=device,
                                dtype=init_p[l].dtype)
            new_e.append(torch.where(idx, torch.zeros_like(e), e))
        memory["e"] = new_e
        return memory

    # ── core ──

    def _run_wrapper(self, batch: Dict, memory: Optional[Dict]):
        """One VLM forward with memory update. Returns (hidden fp32, out, inputs)."""
        device = self.device
        inputs = build_vlm_inputs(
            self.processor, self.image_token_id,
            batch["images"], batch["instructions"], device)

        # Precompute embeddings + visual injection (see module docstring).
        embed_layer = self.wrapper._get_input_embeddings()
        h = embed_layer(inputs.input_ids)
        llm = self.wrapper.llm
        get_image_features = getattr(llm, "get_image_features", None) or llm.model.get_image_features
        image_embeds, deepstack = get_image_features(
            inputs.pixel_values.to(device), inputs.image_grid_thw.to(device))
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = torch.cat(list(image_embeds), dim=0)
        image_embeds = image_embeds.to(h.device, h.dtype)
        mask3 = inputs.image_mask.unsqueeze(-1).expand_as(h)
        h = h.masked_scatter(mask3, image_embeds)

        prev_action = batch["prev_action"].to(device)
        a_emb = self.prev_action_mlp(prev_action.to(self.prev_action_mlp[0].weight.dtype))

        out = self.wrapper(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            inputs_embeds=h,
            deepstack_features=list(deepstack),
            visual_pos_mask=inputs.image_mask,
            action_embed_input=a_emb,
            memory_states=memory["m"] if memory is not None else None,
            variance_states=memory["p"] if memory is not None else None,
            e_prev_list=memory["e"] if memory is not None else None,
            image_grid_thw=inputs.image_grid_thw.to(device),
            mm_token_type_ids=inputs.mm_token_type_ids,
        )
        # Post-injection, post-final-norm hidden states (baseline's
        # hidden_states[-1] is post-norm too).
        hidden = self.wrapper._apply_final_norm(out["hidden_states"]).float()
        return hidden, out, inputs

    def forward_step(self, batch: Dict, memory: Optional[Dict] = None
                     ) -> Tuple[Dict[str, torch.Tensor], Dict, Dict]:
        device = self.device
        hidden, out, inputs = self._run_wrapper(batch, memory)
        cond = self.bridge(hidden, inputs.image_mask, inputs.text_mask)
        cond["robot_obs"] = batch["state"].to(device).float()
        new_memory = {"m": out["new_memory"], "p": out["new_variance"],
                      "e": out["e_list"]}
        aux = {"obs": out["loss_obs"], "nll": out["loss_nll"]}
        return cond, new_memory, aux

    def diffusion_loss(self, batch: Dict, memory: Optional[Dict] = None
                       ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict]:
        """Returns (total_loss, loss_dict, new_memory)."""
        cond, new_memory, aux = self.forward_step(batch, memory)
        state = {"state_images": cond["state_images"], "robot_obs": cond["robot_obs"]}
        actions = batch["actions"].to(cond["goal"].device).float()
        diff = edm_loss(self.denoiser, state, actions, cond["goal"],
                        self.cfg.sigma_data, self.cfg.sigma_min, self.cfg.sigma_max)
        aux_total, aux_dict = compact_aux_loss(
            aux["obs"], aux["nll"], new_memory["m"],
            self.cfg.lambda_obs, self.cfg.lambda_nll, self.cfg.lambda_mem)
        total = diff + aux_total
        loss_dict = {"diffusion": diff.detach(), **aux_dict, "total": total.detach()}
        return total, loss_dict, new_memory

    @torch.no_grad()
    def sample_actions(self, batch: Dict, memory: Optional[Dict] = None,
                       steps: Optional[int] = None):
        """Returns (actions, new_memory)."""
        cond, new_memory, _ = self.forward_step(batch, memory)
        device = cond["goal"].device
        bsz = cond["goal"].shape[0]
        actions = torch.randn(bsz, self.cfg.action_seq_len, self.cfg.action_dim,
                              device=device) * self.cfg.sigma_max
        sigmas = get_sigmas_exponential(
            steps or self.cfg.num_sampling_steps,
            self.cfg.sigma_min, self.cfg.sigma_max, device)
        state = {"state_images": cond["state_images"], "robot_obs": cond["robot_obs"]}
        return sample_ddim(self.denoiser, state, actions, cond["goal"], sigmas), new_memory

    # ── optimizer groups ──

    def get_optim_groups(self, lr: float, base_lr: float, memory_lr: float,
                         weight_decay: float) -> List[dict]:
        def use_wd(name: str) -> bool:
            low = name.lower()
            return all(t not in low for t in ("bias", "layernorm", "norm", "embedding"))

        decay, no_decay = [], []
        for name, p in self.denoiser.inner_model.named_parameters():
            if not p.requires_grad:
                continue
            (decay if use_wd(name) else no_decay).append(p)
        groups = [
            {"name": "dit_decay", "params": decay, "lr": lr, "weight_decay": weight_decay},
            {"name": "dit_no_decay", "params": no_decay, "lr": lr, "weight_decay": 0.0},
            {"name": "bridge", "params": list(self.bridge.parameters()),
             "lr": lr, "weight_decay": weight_decay},
        ]
        # Side memory + wrapper action_embed + prev-action MLP: separate small LR.
        mem_params = (
            [p for p in self.wrapper.side_memories.parameters() if p.requires_grad]
            + [p for p in self.wrapper.action_embed.parameters() if p.requires_grad]
            + [p for p in self.prev_action_mlp.parameters() if p.requires_grad]
        )
        if mem_params:
            groups.append({"name": "memory", "params": mem_params,
                           "lr": memory_lr, "weight_decay": 0.0})
        base_params = [p for p in self.wrapper.llm.parameters() if p.requires_grad]
        if base_params:
            groups.append({"name": "base", "params": base_params,
                           "lr": base_lr, "weight_decay": weight_decay})
        return [g for g in groups if g["params"]]

    def trainable_param_counts(self) -> Dict[str, int]:
        def count(params):
            return sum(p.numel() for p in params if p.requires_grad)
        return {
            "base_trainable": count(self.wrapper.llm.parameters()),
            "base_total": sum(p.numel() for p in self.wrapper.llm.parameters()),
            "side_memory": count(self.wrapper.side_memories.parameters())
                           + count(self.wrapper.action_embed.parameters()),
            "prev_action_mlp": count(self.prev_action_mlp.parameters()),
            "bridge": count(self.bridge.parameters()),
            "dit": count(self.denoiser.parameters()),
        }

    # ── save / load ──

    def save_pretrained(self, save_directory) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.cfg.save_json(save_path / "compact_mode_config.json")
        # JAMEL-COMPACT-format checkpoint for base + side memory.
        self.wrapper.save_pretrained(save_path / "compact")
        modules = {
            "bridge": self.bridge.state_dict(),
            "denoiser": self.denoiser.state_dict(),
            "prev_action_mlp": self.prev_action_mlp.state_dict(),
            "normalizers": {
                "action": self.action_normalizer.to_dict() if self.action_normalizer else None,
                "state": self.state_normalizer.to_dict() if self.state_normalizer else None,
            },
        }
        torch.save(modules, save_path / "policy_modules.pt")
        print(f"[save] CompactMoDEPolicy saved to {save_path}")

    @classmethod
    def load_pretrained(cls, load_directory, cfg_override: Optional[CompactMoDEConfig] = None,
                        device: str = "cuda"):
        load_path = Path(load_directory)
        cfg = CompactMoDEConfig.load_json(load_path / "compact_mode_config.json")
        if cfg_override is not None:
            for k, v in cfg_override.to_dict().items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        model = cls(cfg)
        # Reload wrapper (base + side memory) from the JAMEL-format subdir.
        model.wrapper = CompactModeWrapper.from_pretrained(load_path / "compact")
        model.processor = model.wrapper.processor
        modules = torch.load(load_path / "policy_modules.pt", map_location="cpu", weights_only=False)
        model.bridge.load_state_dict(modules["bridge"])
        model.denoiser.load_state_dict(modules["denoiser"])
        model.prev_action_mlp.load_state_dict(modules["prev_action_mlp"])
        norms = modules.get("normalizers") or {}
        if norms.get("action"):
            model.action_normalizer = JointGripperNormalizer.from_dict(norms["action"])
        if norms.get("state"):
            model.state_normalizer = JointGripperNormalizer.from_dict(norms["state"])
        return model.to(device)

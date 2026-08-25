"""
Baseline variant: Qwen3-VL-2B + conditioning bridge + MoDeDiT (no side memory).

Per frame the VLM sees [head-cam image, instruction] (single forward, no
generation). The bridge turns final-layer hidden states into DiT conditioning:

    image tokens  -> adaptive-avg-pool to 32 -> Linear -> state_images (B, 32, 2048)
    text tokens   -> masked mean             -> Linear -> goal         (B, 1, 512)
    robot_obs     -> 14-dim normalized joint state (goes ONLY to the DiT)

Also contains build_vlm_inputs(), shared with model_compact.py: manual prompt
assembly so that image/text token spans are known exactly (no chat template
needed; the layout matches Qwen3-VL's apply_chat_template output for a single
user message, with the <|image_pad|> block pre-expanded).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import compact_mode  # noqa: F401  (sys.path setup)
from mode.models.edm_diffusion.score_wrappers import GCDenoiser

from .bridge import ConditioningBridge
from .config import CompactMoDEConfig
from .losses import edm_loss, get_sigmas_exponential, sample_ddim
from .normalizer import JointGripperNormalizer

PROMPT_PREFIX = "<|im_start|>user\n<|vision_start|>"
PROMPT_MID = "<|vision_end|>"
PROMPT_SUFFIX = "<|im_end|>\n"


@dataclass
class VLMInputs:
    input_ids: torch.Tensor          # (B, N) int64
    attention_mask: torch.Tensor     # (B, N) int64
    image_mask: torch.Tensor         # (B, N) bool
    text_mask: torch.Tensor          # (B, N) bool — instruction text only
    mm_token_type_ids: torch.Tensor  # (B, N) int64 — 1 at image positions
    pixel_values: torch.Tensor       # (total_patches, patch_dim) float32
    image_grid_thw: torch.Tensor     # (B, 3) int64


def build_vlm_inputs(processor, image_token_id: int, images: List[np.ndarray],
                     instructions: List[str], device) -> VLMInputs:
    """Assemble token ids + masks for [image, instruction] prompts.

    Token layout per sample:
        <|im_start|>user\n<|vision_start|> <|image_pad|>*n_img <|vision_end|>
        <instruction tokens> <|im_end|>\n
    n_img = prod(image_grid_thw[i]) // spatial_merge_size**2  (=4 for Qwen3-VL).
    """
    tok = processor.tokenizer
    img_out = processor.image_processor(images=images, return_tensors="pt")
    pixel_values = img_out["pixel_values"]
    image_grid_thw = img_out["image_grid_thw"]

    prefix_ids = tok(PROMPT_PREFIX, add_special_tokens=False)["input_ids"]
    mid_ids = tok(PROMPT_MID, add_special_tokens=False)["input_ids"]
    suffix_ids = tok(PROMPT_SUFFIX, add_special_tokens=False)["input_ids"]

    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id

    seqs, img_masks, txt_masks = [], [], []
    for i, instr in enumerate(instructions):
        n_img = int(image_grid_thw[i].prod().item()) // 4
        instr_ids = tok(instr, add_special_tokens=False)["input_ids"]
        ids = prefix_ids + [image_token_id] * n_img + mid_ids + instr_ids + suffix_ids
        n_pre = len(prefix_ids)
        n_mid_end = n_pre + n_img + len(mid_ids)
        img_mask = [False] * len(ids)
        for p in range(n_pre, n_pre + n_img):
            img_mask[p] = True
        txt_mask = [False] * len(ids)
        for p in range(n_mid_end, n_mid_end + len(instr_ids)):
            txt_mask[p] = True
        seqs.append(ids)
        img_masks.append(img_mask)
        txt_masks.append(txt_mask)

    max_len = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((B, max_len), dtype=torch.long)
    image_mask = torch.zeros((B, max_len), dtype=torch.bool)
    text_mask = torch.zeros((B, max_len), dtype=torch.bool)
    for i in range(B):
        n = len(seqs[i])
        input_ids[i, :n] = torch.tensor(seqs[i], dtype=torch.long)
        attention_mask[i, :n] = 1
        image_mask[i, :n] = torch.tensor(img_masks[i], dtype=torch.bool)
        text_mask[i, :n] = torch.tensor(txt_masks[i], dtype=torch.bool)

    return VLMInputs(
        input_ids=input_ids.to(device),
        attention_mask=attention_mask.to(device),
        image_mask=image_mask.to(device),
        text_mask=text_mask.to(device),
        mm_token_type_ids=image_mask.long().to(device),
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )


def _infer_hidden_dim(llm) -> int:
    cfg = llm.config
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return int(cfg.text_config.hidden_size)
    return int(cfg.hidden_size)


def build_dit_denoiser(cfg: CompactMoDEConfig, device: str) -> GCDenoiser:
    """Instantiate MoDeDiT (via our StateTokenMoDeDiT subclass) through hydra,
    following MoDE's my_code/model.py pattern. NOTE: the inner config must keep
    the typo'd key `embed_pdrob` verbatim — MoDeDiT.__init__ expects it."""
    inner = OmegaConf.create({
        "_target_": "compact_mode.modedit_ext.StateTokenMoDeDiT",
        "n_state_tokens": cfg.n_state_tokens,
        "action_dim": cfg.action_dim,
        "obs_dim": cfg.obs_dim,
        "device": device,
        "goal_conditioned": True,
        "goal_dim": cfg.goal_dim,
        "embed_dim": cfg.embed_dim,
        "embed_pdrob": cfg.embed_pdrop,   # sic — typo key required by MoDeDiT
        "attn_pdrop": cfg.attn_pdrop,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "goal_seq_len": cfg.goal_seq_len,
        "obs_seq_len": 1,
        "action_seq_len": cfg.action_seq_len,
        "state_dim": cfg.state_dim,
        "mlp_pdrop": cfg.mlp_pdrop,
        "goal_drop": cfg.goal_drop,
        "linear_output": cfg.linear_output,
        "use_proprio": True,
        "cond_router": cfg.cond_router,
        "num_experts": cfg.num_experts,
        "top_k": cfg.top_k,
        "router_normalize": cfg.router_normalize,
        "use_goal_in_routing": cfg.use_goal_in_routing,
        "use_argmax": cfg.use_argmax,
        "use_shared_expert": cfg.use_shared_expert,
        "use_noise_token_as_input": cfg.use_noise_token_as_input,
        "use_custom_attn_mask": cfg.use_custom_attn_mask,
        "init_style": cfg.init_style,
    })
    return GCDenoiser(inner, sigma_data=cfg.sigma_data)


class BaselineQwenMoDEPolicy(nn.Module):
    def __init__(self, cfg: CompactMoDEConfig):
        super().__init__()
        self.cfg = cfg
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch.bfloat16 if cfg.bf16 else torch.float32
        load_kwargs = dict(
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        # Qwen3VLForConditionalGeneration: backbone in `.model` + unused lm_head
        # (we never generate text; encode_batch() calls the backbone directly).
        # NOTE: do NOT use AutoModel here — the checkpoint keys carry the
        # "model." prefix, so loading into the bare Qwen3VLModel silently
        # leaves every weight randomly initialized.
        self.llm, loading_info = AutoModelForImageTextToText.from_pretrained(
            cfg.base_model_name, output_loading_info=True, **load_kwargs)
        missing = [k for k in loading_info.get("missing_keys", [])
                   if k != "lm_head.weight"]  # tied to embed_tokens when applicable
        assert not missing, f"checkpoint failed to load; missing keys: {missing[:5]} ..."
        self._has_lm_head = True
        self.processor = AutoProcessor.from_pretrained(
            cfg.base_model_name, trust_remote_code=True)
        self.image_token_id = int(getattr(self.llm.config, "image_token_id",
                                          self.processor.image_token_id))
        self.hidden_dim = _infer_hidden_dim(self.llm)

        if cfg.freeze_base:
            for p in self.llm.parameters():
                p.requires_grad = False
        if cfg.gradient_checkpointing:
            self.llm.gradient_checkpointing_enable()
            self.llm.config.use_cache = False

        self.bridge = ConditioningBridge(
            self.hidden_dim, cfg.n_state_tokens, cfg.obs_dim, cfg.goal_dim)
        self.denoiser = build_dit_denoiser(cfg, "cpu")
        # Optional normalizers (attached for deployment / action decoding).
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

    # ── core ──

    def encode_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        device = self.device
        inputs = build_vlm_inputs(
            self.processor, self.image_token_id,
            batch["images"], batch["instructions"], device)
        outputs = self.llm.model(  # inner backbone; skips the unused lm_head
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            pixel_values=inputs.pixel_values.to(device),
            image_grid_thw=inputs.image_grid_thw.to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1].float()  # final layer, post-norm
        cond = self.bridge(hidden, inputs.image_mask, inputs.text_mask)
        cond["robot_obs"] = batch["state"].to(device).float()
        return cond

    def forward_step(self, batch: Dict) -> Tuple[Dict[str, torch.Tensor], Dict]:
        return self.encode_batch(batch), {}

    def diffusion_loss(self, batch: Dict) -> torch.Tensor:
        cond, _ = self.forward_step(batch)
        state = {"state_images": cond["state_images"], "robot_obs": cond["robot_obs"]}
        actions = batch["actions"].to(cond["goal"].device).float()
        return edm_loss(self.denoiser, state, actions, cond["goal"],
                        self.cfg.sigma_data, self.cfg.sigma_min, self.cfg.sigma_max)

    @torch.no_grad()
    def sample_actions(self, batch: Dict, steps: Optional[int] = None) -> torch.Tensor:
        cond, _ = self.forward_step(batch)
        device = cond["goal"].device
        bsz = cond["goal"].shape[0]
        actions = torch.randn(bsz, self.cfg.action_seq_len, self.cfg.action_dim,
                              device=device) * self.cfg.sigma_max
        sigmas = get_sigmas_exponential(
            steps or self.cfg.num_sampling_steps,
            self.cfg.sigma_min, self.cfg.sigma_max, device)
        state = {"state_images": cond["state_images"], "robot_obs": cond["robot_obs"]}
        return sample_ddim(self.denoiser, state, actions, cond["goal"], sigmas)

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
        base_params = [p for p in self.llm.parameters() if p.requires_grad]
        if base_params:
            groups.append({"name": "base", "params": base_params,
                           "lr": base_lr, "weight_decay": weight_decay})
        return [g for g in groups if g["params"]]

    def trainable_param_counts(self) -> Dict[str, int]:
        def count(params):
            return sum(p.numel() for p in params if p.requires_grad)
        return {
            "base_trainable": count(self.llm.parameters()),
            "base_total": sum(p.numel() for p in self.llm.parameters()),
            "bridge": count(self.bridge.parameters()),
            "dit": count(self.denoiser.parameters()),
        }

    # ── save / load ──

    def save_pretrained(self, save_directory) -> None:
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        self.cfg.save_json(save_path / "compact_mode_config.json")
        modules = {
            "bridge": self.bridge.state_dict(),
            "denoiser": self.denoiser.state_dict(),
            "normalizers": {
                "action": self.action_normalizer.to_dict() if self.action_normalizer else None,
                "state": self.state_normalizer.to_dict() if self.state_normalizer else None,
            },
        }
        torch.save(modules, save_path / "policy_modules.pt")
        if self.cfg.freeze_base:
            (save_path / "base_model_ref.txt").write_text(self.cfg.base_model_name)
        else:
            self.llm.save_pretrained(save_path / "base_model")
        print(f"[save] BaselineQwenMoDEPolicy saved to {save_path}")

    @classmethod
    def load_pretrained(cls, load_directory, cfg_override: Optional[CompactMoDEConfig] = None,
                        device: str = "cuda"):
        load_path = Path(load_directory)
        cfg = CompactMoDEConfig.load_json(load_path / "compact_mode_config.json")
        if cfg_override is not None:
            for k, v in cfg_override.to_dict().items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        if (load_path / "base_model").exists():
            cfg.base_model_name = str(load_path / "base_model")
        model = cls(cfg)
        modules = torch.load(load_path / "policy_modules.pt", map_location="cpu", weights_only=False)
        model.bridge.load_state_dict(modules["bridge"])
        model.denoiser.load_state_dict(modules["denoiser"])
        norms = modules.get("normalizers") or {}
        if norms.get("action"):
            model.action_normalizer = JointGripperNormalizer.from_dict(norms["action"])
        if norms.get("state"):
            model.state_normalizer = JointGripperNormalizer.from_dict(norms["state"])
        return model.to(device)

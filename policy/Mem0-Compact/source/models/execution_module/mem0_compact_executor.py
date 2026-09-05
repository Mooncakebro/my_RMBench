"""
Mem0-Compact executor: Qwen3-VL-2B + COMPACT per-layer side memory + flow-
matching DiT action head (2-token conditioning). MemoryBank removed entirely.

Built from copies (idea.md §5 — originals untouched):
  - CompactModeWrapper (adapted JAMELCompactWrapper): manual decoder loop with
    predict → pretrained layer → DeepStack → observe → correct → inject.
  - FlowmatchingActionHead (DiT-B, flow matching, horizon 30, action_dim 16,
    8 Euler steps at inference) — copied from Mem-0's ActionHeader.py.
  - SubtaskEndClassifier — input shrinks 6144 → 4096 (2 summary tokens instead
    of 3); disabled for M(1) tasks.

Forward contract (one frame batch):
    forward_step(batch, memory) -> (loss_dict, new_memory)
  where batch carries:
    image:        List[List[PIL]]          (B, 1 view)
    lang:         List[str]                (instruction per sample)
    action:       (B, action_horizon, 16)  normalized flow-matching labels
    state:        (B, 1, 16)               normalized proprio (NOT into the VLM)
    prev_action:  (B, 16)                  normalized last-executed action
    episode_id / subtask_end               metadata
  and memory is a dict {"m": [28 x (B,16,mem_dim)], "p": [28 x (B,16)],
                        "e": [28 x (B,)]} threaded by the TBPTT loop.

Loss per frame (idea.md §3):
    L = 1.0 * L_flow + 0.2 * L_cls(Mn only)
        + 0.01 * L_obs + 0.01 * L_nll + 0.001 * L_mem
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# compact_wrapper bootstraps policy/Mem0-Compact/vendor onto sys.path first.
from source.models.execution_module.compact_wrapper import CompactModeWrapper
from source.models.execution_module.action_model.ActionHeader import FlowmatchingActionHead
from source.models.execution_module.classifier.subtask_classifier import SubtaskEndClassifier

from jamel_compact.config import CompactConfig

# The classifier summary input is 2 tokens x 2048 (image + text), flattened.
CLASSIFIER_INPUT_DIM = 2 * 2048


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states at masked positions.

    hidden: (B, N, D)  mask: (B, N) bool
    returns: (B, 1, D)
    """
    m = mask.to(hidden.dtype).unsqueeze(-1)
    denom = m.sum(dim=1, keepdim=True).clamp_min(1)
    return (hidden * m).sum(dim=1, keepdim=True) / denom


class Mem0CompactExecutor(nn.Module):
    def __init__(self, config, device: Optional[torch.device] = None, **kwargs):
        """config: OmegaConf (train config root) or dict with an
        `execution_module` section (see source/config/mem0_compact_train.yaml)."""
        super().__init__()
        if hasattr(config, "execution_module"):
            em = config.execution_module
        elif isinstance(config, dict) and "execution_module" in config:
            em = config["execution_module"]
        else:
            raise ValueError("config must have an execution_module section")
        self.config = config

        # ── Qwen VLM + COMPACT wrapper ──
        qwenvl_cfg = em.get("qwen_vl", {})
        model_path = qwenvl_cfg.get("model_path", "./checkpoints/Qwen3-VL-2B-Instruct")
        compact_cfg = em.get("compact", {})
        mem_dim = int(compact_cfg.get("mem_dim", 128))
        num_mem_tokens = int(compact_cfg.get("num_mem_tokens", 16))
        num_heads = int(compact_cfg.get("num_heads", 8))
        num_obs_tokens = int(compact_cfg.get("num_obs_tokens", 4))
        freeze_base = bool(compact_cfg.get("freeze_base", False))
        grad_ckpt = bool(compact_cfg.get("gradient_checkpointing", False))
        bf16 = bool(compact_cfg.get("bf16", True))
        self.lambda_obs = float(compact_cfg.get("lambda_obs", 0.01))
        self.lambda_nll = float(compact_cfg.get("lambda_nll", 0.01))
        self.lambda_mem = float(compact_cfg.get("lambda_mem", 0.001))

        compact_model_cfg = CompactConfig(
            base_model_name=model_path,
            mem_dim=mem_dim,
            num_mem_tokens=num_mem_tokens,
            num_heads=num_heads,
            num_obs_tokens=num_obs_tokens,
            freeze_base=freeze_base,
            bf16=bf16,
            gradient_checkpointing=grad_ckpt,
            lambda_obs=self.lambda_obs,
            lambda_nll=self.lambda_nll,
            lambda_mem=self.lambda_mem,
            lora_rank=0,  # peft not installed; LoRA disabled
        )
        self.wrapper = CompactModeWrapper(compact_model_cfg)
        self.hidden_dim = self.wrapper.hidden_dim
        self.num_layers = self.wrapper.num_layers
        self.num_mem = num_mem_tokens
        self.mem_dim = mem_dim
        self.state_dim = 16  # Mem-0 model layout (padded 16-dim)

        # Special token ids for image/text token pooling.
        proc = self.wrapper.processor
        self.vision_start_token_id = proc.vision_start_token_id
        self.vision_end_token_id = proc.vision_end_token_id
        self.im_end_token_id = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # ── Continuous prev-action embedding (drives the FiLM-GRU predict) ──
        llm_dtype = next(self.wrapper.llm.parameters()).dtype
        self.prev_action_mlp = nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.GELU(),
            nn.Linear(256, self.hidden_dim),
        ).to(dtype=llm_dtype)

        # ── Flow-matching DiT action head (copied from Mem-0) ──
        action_cfg = em.get("action_model", {})
        self.action_horizon = int(action_cfg.get("action_horizon", 30))
        self.action_model = FlowmatchingActionHead(
            action_cfg, hidden_size=self.hidden_dim)

        # ── Subtask-end classifier (M(n) only; disabled for M(1)) ──
        self.use_classifier = bool(em.get("use_classifier", False))
        cls_cfg = em.get("classifier", {})
        if self.use_classifier:
            hidden_sizes = list(cls_cfg.get("hidden_sizes", [4096, 2048, 512]))
            if hidden_sizes[0] != CLASSIFIER_INPUT_DIM:
                hidden_sizes = [CLASSIFIER_INPUT_DIM] + hidden_sizes[1:]
            self.classifier = SubtaskEndClassifier(
                hidden_sizes=hidden_sizes,
                dropout=float(cls_cfg.get("dropout", 0.1)),
                pos_weight=float(cls_cfg.get("pos_weight", 10.0)),
                focal_gamma=float(cls_cfg.get("focal_gamma", 1.0)),
            )
            self.classifier_threshold = float(cls_cfg.get("threshold", 0.5))
        else:
            self.classifier = None
            self.classifier_threshold = 0.5

        # ── Loss weights ──
        lw = em.get("loss_weights", {})
        self.lambda_action = float(lw.get("lambda_action", 1.0))
        self.lambda_classifier = float(lw.get("lambda_classifier", 0.2))

    # ── memory state handling ──

    def init_memory(self, batch_size: int, device) -> Dict:
        m_states, p_states = self.wrapper.init_memory(batch_size, device)
        return {"m": m_states, "p": p_states, "e": [None] * self.num_layers}

    @staticmethod
    def detach_memory(memory: Dict) -> Dict:
        """Detach carried state at TBPTT window boundaries."""
        return {
            "m": [t.detach() for t in memory["m"]],
            "p": [t.detach() for t in memory["p"]],
            "e": [t.detach() if isinstance(t, torch.Tensor) else None
                  for t in memory["e"]],
        }

    def reset_memory_rows(self, memory: Dict, reset_mask: List[bool], device) -> Dict:
        """Re-initialize memory for batch rows that started a new episode."""
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

    # ── input building ──

    def build_vlm_inputs(self, images: List[List], instructions: List[str]) -> dict:
        """Chat-template tokenization + image/text token masks.

        images: List[List[PIL]] — one view per sample (head camera).
        Returns dict with input_ids, attention_mask, pixel_values,
        image_grid_thw, image_mask, text_mask, mm_token_type_ids (all tensors,
        moved to the wrapper's input device).
        """
        assert len(images) == len(instructions)
        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": instruction})
            messages.append([{"role": "user", "content": content}])

        proc = self.wrapper.processor
        batch_inputs = proc.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
            padding=True,
        )
        input_ids = batch_inputs["input_ids"]
        attention_mask = batch_inputs["attention_mask"]
        pixel_values = batch_inputs["pixel_values"]
        image_grid_thw = batch_inputs["image_grid_thw"]

        B, N = input_ids.shape
        image_mask = torch.zeros((B, N), dtype=torch.bool)
        text_mask = torch.zeros((B, N), dtype=torch.bool)
        mm_token_type_ids = torch.zeros((B, N), dtype=torch.long)
        for b in range(B):
            ids = input_ids[b]
            v_start = (ids == self.vision_start_token_id).nonzero(as_tuple=True)[0]
            v_end = (ids == self.vision_end_token_id).nonzero(as_tuple=True)[0]
            im_end = (ids == self.im_end_token_id).nonzero(as_tuple=True)[0]
            if len(v_start) > 0 and len(v_end) > 0:
                s, e = v_start[0].item() + 1, v_end[0].item()
                image_mask[b, s:e] = True
                mm_token_type_ids[b, s:e] = 1
            if len(v_end) > 0 and len(im_end) > 0:
                s, e = v_end[0].item() + 1, im_end[0].item()
                text_mask[b, s:e] = True

        device = self.input_device
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "pixel_values": pixel_values.to(device),
            "image_grid_thw": image_grid_thw.to(device),
            "image_mask": image_mask.to(device),
            "text_mask": text_mask.to(device),
            "mm_token_type_ids": mm_token_type_ids.to(device),
        }

    @property
    def input_device(self):
        # Computed dynamically: wrapper.input_device is fixed at construction
        # time and goes stale after .to(device).
        return next(self.wrapper.llm.parameters()).device

    def pool_features(self, hidden: torch.Tensor, inputs: dict
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """image tokens → (B,1,H); text tokens → (B,1,H) via masked mean."""
        image_feature = masked_mean_pool(hidden, inputs["image_mask"])
        text_feature = masked_mean_pool(hidden, inputs["text_mask"])
        return image_feature, text_feature

    # ── core ──

    def forward(self, batch: dict, memory: Optional[Dict]
                ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """DDP-compatible forward: one frame → (loss_dict, new_memory).

        Must go through `model(batch, memory)` (NOT model.forward_step) when
        the model is wrapped in DistributedDataParallel — DDP registers its
        gradient-sync hooks in its own forward, and calling inner methods
        directly would silently skip gradient synchronization.
        """
        return self.forward_step(batch, memory)

    def forward_step(self, batch: dict, memory: Optional[Dict]
                     ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """One frame forward: VLM + memory update + head losses.

        Returns (loss_dict, new_memory). loss_dict keys: total, action,
        classifier, obs, nll, mem_l2 (+ classifier metrics when enabled).
        """
        device = self.input_device
        inputs = self.build_vlm_inputs(batch["image"], batch["lang"])

        # Precompute embeddings + visual injection (transformers 4.57.6
        # get_image_features returns (image_embeds, deepstack) plain tuple).
        embed_layer = self.wrapper._get_input_embeddings()
        h = embed_layer(inputs["input_ids"])
        llm = self.wrapper.llm
        get_image_features = (getattr(llm, "get_image_features", None)
                              or llm.model.get_image_features)
        image_embeds, deepstack = get_image_features(
            inputs["pixel_values"].to(device), inputs["image_grid_thw"].to(device))
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = torch.cat(list(image_embeds), dim=0)
        image_embeds = image_embeds.to(h.device, h.dtype)
        mask3 = inputs["image_mask"].unsqueeze(-1).expand_as(h)
        h = h.masked_scatter(mask3, image_embeds)

        # Prev-action embedding for the memory predict step.
        prev_action = batch["prev_action"].to(device)  # (B, 16)
        a_emb = self.prev_action_mlp(
            prev_action.to(self.prev_action_mlp[0].weight.dtype))

        out = self.wrapper(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            inputs_embeds=h,
            deepstack_features=list(deepstack),
            visual_pos_mask=inputs["image_mask"],
            action_embed_input=a_emb,
            memory_states=memory["m"] if memory is not None else None,
            variance_states=memory["p"] if memory is not None else None,
            e_prev_list=memory["e"] if memory is not None else None,
            image_grid_thw=inputs["image_grid_thw"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
        )
        # Final layer, post-injection, then backbone final norm.
        hidden = self.wrapper._apply_final_norm(out["hidden_states"]).float()

        image_feature, text_feature = self.pool_features(hidden, inputs)
        summary = torch.cat([image_feature, text_feature], dim=1)  # (B, 2, 2048)

        # ── Flow-matching action loss ──
        actions = batch["action"].to(hidden.device).float()  # (B, T, 16)
        state = batch["state"].to(hidden.device).float()     # (B, 1, 16)
        action_loss = self.action_model(
            vl_embs=summary, actions=actions, state=state)

        # ── Subtask-end classifier (M(n) only) ──
        loss_dict: Dict[str, torch.Tensor] = {"action": action_loss}
        if self.use_classifier and self.classifier is not None:
            cls_labels = torch.as_tensor(
                batch["subtask_end"], device=hidden.device,
                dtype=torch.float32)
            cls_in = summary.reshape(summary.shape[0], -1)  # (B, 4096)
            cls_out = self.classifier(fused_hidden=cls_in, labels=cls_labels)
            loss_dict["classifier"] = cls_out["loss"]
            with torch.no_grad():
                prob = torch.sigmoid(cls_out["logits"])
                preds = (prob >= self.classifier_threshold).to(cls_labels.dtype)
                denom = max(float(cls_labels.numel()), 1.0)
                loss_dict["cls_accuracy"] = torch.as_tensor(
                    float((preds == cls_labels).sum()) / denom)

        # ── COMPACT aux losses (no text CE) ──
        loss_dict["obs"] = out["loss_obs"]
        loss_dict["nll"] = out["loss_nll"]
        loss_mem = torch.zeros((), device=hidden.device, dtype=torch.float32)
        for M in out["new_memory"]:
            loss_mem = loss_mem + M.float().pow(2).sum()
        loss_mem = loss_mem / max(len(out["new_memory"]), 1)
        loss_dict["mem_l2"] = loss_mem

        total = (
            self.lambda_action * action_loss.float()
            + self.lambda_obs * out["loss_obs"].float()
            + self.lambda_nll * out["loss_nll"].float()
            + self.lambda_mem * loss_mem
        )
        if self.use_classifier and "classifier" in loss_dict:
            total = total + self.lambda_classifier * loss_dict["classifier"].float()
        loss_dict["total"] = total

        new_memory = {
            "m": out["new_memory"],
            "p": out["new_variance"],
            "e": out["e_list"],
        }
        return loss_dict, new_memory

    @torch.inference_mode()
    def update_obs_inference(self, batch: dict, memory: Optional[Dict]
                             ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Inference-time VLM forward + memory update (no losses).

        Returns (summary (B,2,2048), state (B,1,16), new_memory).
        """
        device = self.input_device
        inputs = self.build_vlm_inputs(batch["image"], batch["lang"])
        embed_layer = self.wrapper._get_input_embeddings()
        h = embed_layer(inputs["input_ids"])
        llm = self.wrapper.llm
        get_image_features = (getattr(llm, "get_image_features", None)
                              or llm.model.get_image_features)
        image_embeds, deepstack = get_image_features(
            inputs["pixel_values"].to(device), inputs["image_grid_thw"].to(device))
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = torch.cat(list(image_embeds), dim=0)
        image_embeds = image_embeds.to(h.device, h.dtype)
        mask3 = inputs["image_mask"].unsqueeze(-1).expand_as(h)
        h = h.masked_scatter(mask3, image_embeds)

        prev_action = batch["prev_action"].to(device)
        a_emb = self.prev_action_mlp(
            prev_action.to(self.prev_action_mlp[0].weight.dtype))

        out = self.wrapper(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            inputs_embeds=h,
            deepstack_features=list(deepstack),
            visual_pos_mask=inputs["image_mask"],
            action_embed_input=a_emb,
            memory_states=memory["m"] if memory is not None else None,
            variance_states=memory["p"] if memory is not None else None,
            e_prev_list=memory["e"] if memory is not None else None,
            image_grid_thw=inputs["image_grid_thw"],
            mm_token_type_ids=inputs["mm_token_type_ids"],
        )
        hidden = self.wrapper._apply_final_norm(out["hidden_states"]).float()
        image_feature, text_feature = self.pool_features(hidden, inputs)
        summary = torch.cat([image_feature, text_feature], dim=1)
        state = batch["state"].to(hidden.device).float()
        new_memory = {
            "m": out["new_memory"],
            "p": out["new_variance"],
            "e": out["e_list"],
        }
        return summary, state, new_memory

    @torch.inference_mode()
    def predict_subtask_end(self, summary: torch.Tensor) -> float:
        """Subtask-end probability for the cached summary (M(n) only).

        summary: (B, 2, 2048). Returns float prob in [0, 1]; 0.0 when the
        classifier is disabled (M(1)).
        """
        if self.classifier is None:
            return 0.0
        cls_in = summary.reshape(summary.shape[0], -1)  # (B, 4096)
        out = self.classifier.predict(fused_hidden=cls_in)
        return float(out["prob"].squeeze().detach().cpu().item())

    def get_optim_groups(self, base_lr: float, head_lr: float,
                         memory_lr: float, weight_decay: float) -> List[dict]:
        def use_wd(name: str) -> bool:
            low = name.lower()
            return all(t not in low for t in ("bias", "layernorm", "norm", "embedding"))

        decay, no_decay = [], []
        for name, p in self.action_model.named_parameters():
            if p.requires_grad:
                (decay if use_wd(name) else no_decay).append(p)
        if self.classifier is not None:
            for name, p in self.classifier.named_parameters():
                if p.requires_grad:
                    (decay if use_wd(name) else no_decay).append(p)

        for name, p in self.wrapper.action_embed.named_parameters():
            (decay if use_wd(f"action_embed.{name}") else no_decay).append(p)
        for name, p in self.prev_action_mlp.named_parameters():
            (decay if use_wd(f"prev_action_mlp.{name}") else no_decay).append(p)

        mem_params = [
            p for p in self.wrapper.side_memories.parameters() if p.requires_grad
        ]
        base_params = [p for p in self.wrapper.llm.parameters() if p.requires_grad]

        groups = [
            {"name": "head_decay", "params": decay, "lr": head_lr,
             "weight_decay": weight_decay},
            {"name": "head_no_decay", "params": no_decay, "lr": head_lr,
             "weight_decay": 0.0},
            {"name": "memory", "params": mem_params, "lr": memory_lr,
             "weight_decay": 0.0},
        ]
        if base_params:
            groups.append({"name": "base", "params": base_params, "lr": base_lr,
                           "weight_decay": weight_decay})
        return [g for g in groups if g["params"]]

    def trainable_param_counts(self) -> Dict[str, int]:
        def count(params):
            return sum(p.numel() for p in params if p.requires_grad)
        return {
            "base_trainable": count(self.wrapper.llm.parameters()),
            "base_total": sum(p.numel() for p in self.wrapper.llm.parameters()),
            "side_memory": count(self.wrapper.side_memories.parameters()),
            "action_embed": count(self.wrapper.action_embed.parameters()),
            "prev_action_mlp": count(self.prev_action_mlp.parameters()),
            "action_head": count(self.action_model.parameters()),
            "classifier": count(self.classifier.parameters()) if self.classifier else 0,
        }

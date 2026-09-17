"""
Mem0-Compact BASELINE executor: Qwen3-VL-2B + flow-matching DiT action head
(2-token conditioning) WITHOUT the COMPACT per-layer side memory.

This is the ablation twin of mem0_compact_executor.py — everything identical
(backbone, token pooling, DiT head, classifier, normalization, loss weights)
except:
  - the VLM runs one plain HF forward per frame (no predict/observe/correct/
    inject, no recurrent state, no aux losses);
  - no prev_action_mlp (nothing consumes prev_action — the TBPTT loop still
    supplies it; it is ignored).

The public API deliberately mirrors Mem0CompactExecutor so the SAME training
loop (train_compact.py, TBPTT window degenerates to gradient accumulation
over K frames) and the SAME deployment agent work unchanged: all memory
methods are no-ops returning an empty dict.

Forward contract (one frame batch): forward_step(batch, memory) -> (loss_dict, {})
with batch identical to the compact executor (prev_action present but unused).

Loss per frame:
    L = 1.0 * L_flow + 0.2 * L_cls (M(n) only)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from source.models.execution_module.mem0_compact_executor import masked_mean_pool
from source.models.execution_module.action_model.ActionHeader import FlowmatchingActionHead
from source.models.execution_module.classifier.subtask_classifier import SubtaskEndClassifier

# The classifier summary input is 2 tokens x 2048 (image + text), flattened.
CLASSIFIER_INPUT_DIM = 2 * 2048


class Mem0BaselineExecutor(nn.Module):
    def __init__(self, config, device: Optional[torch.device] = None, **kwargs):
        """config: OmegaConf (train config root) or dict with an
        `execution_module` section (same schema as the compact executor;
        `compact.mem_dim` etc. are ignored, `compact.freeze_base` / `bf16` /
        `gradient_checkpointing` still apply to the VLM)."""
        super().__init__()
        if hasattr(config, "execution_module"):
            em = config.execution_module
        elif isinstance(config, dict) and "execution_module" in config:
            em = config["execution_module"]
        else:
            raise ValueError("config must have an execution_module section")
        self.config = config

        # ── Qwen VLM (plain HF load — same kwargs as the vendored wrapper) ──
        qwenvl_cfg = em.get("qwen_vl", {})
        model_path = qwenvl_cfg.get("model_path", "./checkpoints/Qwen3-VL-2B-Instruct")
        compact_cfg = em.get("compact", {})
        freeze_base = bool(compact_cfg.get("freeze_base", False))
        grad_ckpt = bool(compact_cfg.get("gradient_checkpointing", False))
        bf16 = bool(compact_cfg.get("bf16", True))

        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch.bfloat16 if bf16 else torch.float32
        self.llm = AutoModelForImageTextToText.from_pretrained(
            str(model_path).rstrip("/\\"),
            dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            str(model_path).rstrip("/\\"), trust_remote_code=True)

        if freeze_base:
            for p in self.llm.parameters():
                p.requires_grad = False
        if grad_ckpt:
            try:
                self.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.llm.gradient_checkpointing_enable()
            self.llm.config.use_cache = False

        cfg = self.llm.config
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.hidden_dim = int(cfg.text_config.hidden_size)
        else:
            self.hidden_dim = int(cfg.hidden_size)
        self.state_dim = 16  # Mem-0 model layout (padded 16-dim)

        # Special token ids for image/text token pooling.
        proc = self.processor
        self.vision_start_token_id = proc.vision_start_token_id
        self.vision_end_token_id = proc.vision_end_token_id
        self.im_end_token_id = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # ── Flow-matching DiT action head (identical to compact variant) ──
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

    # ── memory state handling: all no-ops (baseline carries no state) ──

    def init_memory(self, batch_size: int, device) -> Dict:
        return {}

    @staticmethod
    def detach_memory(memory: Dict) -> Dict:
        return memory

    def reset_memory_rows(self, memory: Dict, reset_mask: List[bool], device) -> Dict:
        return memory

    # ── input building (same chat-template tokenization as compact) ──

    def build_vlm_inputs(self, images: List[List], instructions: List[str]) -> dict:
        """Chat-template tokenization + image/text token masks.

        images: List[List[PIL]] — one view per sample (head camera).
        Returns dict with input_ids, attention_mask, pixel_values,
        image_grid_thw, image_mask, text_mask (all tensors on input_device).
        """
        assert len(images) == len(instructions)
        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": instruction})
            messages.append([{"role": "user", "content": content}])

        proc = self.processor
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
        for b in range(B):
            ids = input_ids[b]
            v_start = (ids == self.vision_start_token_id).nonzero(as_tuple=True)[0]
            v_end = (ids == self.vision_end_token_id).nonzero(as_tuple=True)[0]
            im_end = (ids == self.im_end_token_id).nonzero(as_tuple=True)[0]
            if len(v_start) > 0 and len(v_end) > 0:
                image_mask[b, v_start[0].item() + 1: v_end[0].item()] = True
            if len(v_end) > 0 and len(im_end) > 0:
                text_mask[b, v_end[0].item() + 1: im_end[0].item()] = True

        device = self.input_device
        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "pixel_values": pixel_values.to(device),
            "image_grid_thw": image_grid_thw.to(device),
            "image_mask": image_mask.to(device),
            "text_mask": text_mask.to(device),
        }

    @property
    def input_device(self):
        return next(self.llm.parameters()).device

    def pool_features(self, hidden: torch.Tensor, inputs: dict
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
        """image tokens → (B,1,H); text tokens → (B,1,H) via masked mean."""
        image_feature = masked_mean_pool(hidden, inputs["image_mask"])
        text_feature = masked_mean_pool(hidden, inputs["text_mask"])
        return image_feature, text_feature

    # ── core ──

    def _encode(self, batch: dict) -> torch.Tensor:
        """One plain VLM forward → conditioning summary (B, 2, hidden_dim).

        Uses the inner backbone (self.llm.model) so the unused LM head is
        skipped; hidden_states[-1] is the final layer, post-norm — matching
        the compact executor's post-injection + final-norm pooling point.
        """
        device = self.input_device
        inputs = self.build_vlm_inputs(batch["image"], batch["lang"])
        # NOTE: last_hidden_state (final layer, post-norm) — do NOT use
        # output_hidden_states=True + hidden_states[-1]: some transformers
        # builds silently drop the kwarg and return hidden_states=None.
        outputs = self.llm.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            return_dict=True,
        )
        hidden = outputs.last_hidden_state.float()  # final layer, post-norm
        image_feature, text_feature = self.pool_features(hidden, inputs)
        return torch.cat([image_feature, text_feature], dim=1)  # (B, 2, 2048)

    def forward(self, batch: dict, memory: Optional[Dict] = None
                ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """DDP-compatible forward (see Mem0CompactExecutor.forward)."""
        return self.forward_step(batch, memory)

    def forward_step(self, batch: dict, memory: Optional[Dict] = None
                     ) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """One frame forward: VLM encode + head losses. memory is ignored."""
        summary = self._encode(batch)

        # ── Flow-matching action loss ──
        actions = batch["action"].to(summary.device).float()  # (B, T, 16)
        state = batch["state"].to(summary.device).float()     # (B, 1, 16)
        action_loss = self.action_model(
            vl_embs=summary, actions=actions, state=state)

        # ── Subtask-end classifier (M(n) only) ──
        loss_dict: Dict[str, torch.Tensor] = {"action": action_loss}
        total = self.lambda_action * action_loss.float()
        if self.use_classifier and self.classifier is not None:
            cls_labels = torch.as_tensor(
                batch["subtask_end"], device=summary.device, dtype=torch.float32)
            cls_in = summary.reshape(summary.shape[0], -1)  # (B, 4096)
            cls_out = self.classifier(fused_hidden=cls_in, labels=cls_labels)
            loss_dict["classifier"] = cls_out["loss"]
            total = total + self.lambda_classifier * cls_out["loss"].float()
            with torch.no_grad():
                prob = torch.sigmoid(cls_out["logits"])
                preds = (prob >= self.classifier_threshold).to(cls_labels.dtype)
                denom = max(float(cls_labels.numel()), 1.0)
                loss_dict["cls_accuracy"] = torch.as_tensor(
                    float((preds == cls_labels).sum()) / denom)
        loss_dict["total"] = total
        return loss_dict, {}

    @torch.inference_mode()
    def update_obs_inference(self, batch: dict, memory: Optional[Dict] = None
                             ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Inference-time VLM forward (no memory). Returns (summary, state, {})."""
        summary = self._encode(batch)
        state = batch["state"].to(summary.device).float()
        return summary, state, {}

    @torch.inference_mode()
    def predict_subtask_end(self, summary: torch.Tensor) -> float:
        """Subtask-end probability for the cached summary (M(n) only)."""
        if self.classifier is None:
            return 0.0
        cls_in = summary.reshape(summary.shape[0], -1)  # (B, 4096)
        out = self.classifier.predict(fused_hidden=cls_in)
        return float(out["prob"].squeeze().detach().cpu().item())

    def get_optim_groups(self, base_lr: float, head_lr: float,
                         memory_lr: float, weight_decay: float) -> List[dict]:
        """memory_lr is accepted for interface parity and ignored."""
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

        base_params = [p for p in self.llm.parameters() if p.requires_grad]

        groups = [
            {"name": "head_decay", "params": decay, "lr": head_lr,
             "weight_decay": weight_decay},
            {"name": "head_no_decay", "params": no_decay, "lr": head_lr,
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
            "base_trainable": count(self.llm.parameters()),
            "base_total": sum(p.numel() for p in self.llm.parameters()),
            "action_head": count(self.action_model.parameters()),
            "classifier": count(self.classifier.parameters()) if self.classifier else 0,
        }

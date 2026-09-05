"""
CompactModeWrapper (Mem0-Compact copy): a JAMELCompactWrapper subclass whose
forward() is COPIED from policy/CompactMoDE/compact_mode/compact_wrapper.py
(which itself is adapted from jamel_compact/model.py, JAMEL-COMPACT repo).
The original repos are NOT modified.

Adaptations vs the original JAMEL-COMPACT forward():
  1. Always returns the final-layer post-injection hidden states
     ("hidden_states") and the per-step aux loss terms ("loss_obs"/"loss_nll")
     — the DiT action head and aux losses consume these. (The original only
     exposes them inside the label-based loss path.)
  2. Never runs the LM head and never computes text CE ("logits"/"loss" are
     not returned; `labels` is not accepted). The policy never generates text,
     and skipping the 2048→vocab projection saves significant compute.
  3. generate() is unsupported.

Everything else — memory predict/correct/inject, DeepStack injection, M-RoPE
position handling, 4D causal mask, device placement — is byte-for-byte the
original logic. save_pretrained/from_pretrained are inherited unchanged
(from_pretrained constructs via cls(...), so it returns this subclass).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch

# Bootstrap the vendored jamel_compact package (policy/Mem0-Compact/vendor).
_VENDOR_ROOT = Path(__file__).resolve().parents[3] / "vendor"
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from jamel_compact.model import JAMELCompactWrapper


class CompactModeWrapper(JAMELCompactWrapper):
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        action_embed_input: Optional[torch.Tensor] = None,
        memory_states: Optional[List[torch.Tensor]] = None,
        variance_states: Optional[List[torch.Tensor]] = None,
        action_input_ids: Optional[torch.Tensor] = None,
        action_attention_mask: Optional[torch.Tensor] = None,
        observation_mask: Optional[torch.Tensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        deepstack_features: Optional[List] = None,
        visual_pos_mask: Optional[torch.Tensor] = None,
        e_prev_list: Optional[List[Optional[torch.Tensor]]] = None,
        **kwargs,
    ) -> dict:
        """One time step through the full memory-augmented LLM.

        Returns:
            dict with: hidden_states (final layer, post-injection, pre-final-
                       norm), new_memory, new_variance, e_list, loss_obs,
                       loss_nll
        """
        B = input_ids.shape[0]
        device = input_ids.device
        if memory_states is None or variance_states is None:
            memory_states, variance_states = self.init_memory(B, device)
        if observation_mask is None:
            observation_mask = attention_mask
        else:
            # The dedicated mask may only narrow the model's valid-token mask;
            # never allow padding to enter observation pooling.
            observation_mask = (
                observation_mask.to(dtype=torch.bool)
                & attention_mask.to(dtype=torch.bool)
            )

        # ── Embed tokens ──
        if inputs_embeds is not None:
            # Pre-computed embeddings (visual features already injected)
            h = inputs_embeds
            if deepstack_features is None:
                deepstack_features = []
            if visual_pos_mask is None:
                visual_pos_mask = None
        else:
            embed_layer = self._get_input_embeddings()
            h = embed_layer(input_ids)  # [B, N, d]

            # ── Process image features if provided ──
            if deepstack_features is None:
                deepstack_features = []
            if pixel_values is not None and self._has_visual_encoder():
                h, deepstack_features, visual_pos_mask = self._inject_visual_features(
                    h, input_ids, pixel_values, image_grid_thw,
                )

        # ── Raw action embedding ──
        if action_embed_input is None:
            if action_input_ids is None or action_attention_mask is None:
                raise ValueError(
                    "Provide action_embed_input or action token IDs and mask."
                )
            action_token_embed = self._get_input_embeddings()
            action_token_device = self._module_device(action_token_embed)
            action_token_hidden = action_token_embed(
                action_input_ids.to(action_token_device)
            )
            action_token_mask = action_attention_mask.to(
                action_token_device
            ).unsqueeze(-1).to(action_token_hidden.dtype)
            action_embed_input = (
                (action_token_hidden * action_token_mask).sum(dim=1)
                / action_token_mask.sum(dim=1).clamp(min=1)
            )

        action_device = self._module_device(self.action_embed)
        action_embed = self.action_embed(
            action_embed_input.to(action_device)
        )  # [B, d]

        # ── Get decoder layers ──
        decoder_layers = self._get_decoder_layers()

        # ── Compute position embeddings if the model uses RoPE ──
        position_embeddings = self._compute_position_embeddings(
            h, attention_mask, input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        # ── Convert attention_mask to 4D causal mask ──
        attention_mask_4d = self._build_causal_attention_mask(
            attention_mask, h.dtype,
        )

        # ── Initialize e_prev_list if not provided (chunk start) ──
        if e_prev_list is None:
            e_prev_list = [None] * len(self.side_memories)

        # ── Process through each layer ──
        new_memory, new_variance = [], []
        e_list = []  # surprise per layer for next step
        loss_obs_terms = []
        loss_nll_terms = []
        L = len(decoder_layers)

        for l, (layer, sm) in enumerate(zip(decoder_layers, self.side_memories)):
            layer_device = self._module_device(layer)
            h = h.to(layer_device)
            memory_state = memory_states[l].to(layer_device)
            variance_state = variance_states[l].to(layer_device)
            action_embed_layer = action_embed.to(layer_device)
            e_prev = e_prev_list[l]
            if isinstance(e_prev, torch.Tensor):
                e_prev = e_prev.to(layer_device)

            # 4a. Memory Predict (FiLM-GRU + variance predict)
            m_hat, p_hat, q_noise, surprise_inflation = sm.predict(
                memory_state, variance_state, action_embed_layer,
                e_prev=e_prev,
            )

            # 4b. Run pretrained layer (self-attn + FFN)
            layer_output = self._run_decoder_layer(
                layer,
                h,
                attention_mask=attention_mask_4d.to(layer_device),
                position_embeddings=self._to_device(
                    position_embeddings, layer_device,
                ),
                **self._to_device(kwargs, layer_device),
            )
            if isinstance(layer_output, tuple):
                h_layer = layer_output[0]
            else:
                h_layer = layer_output

            m_hat = m_hat.to(h_layer.dtype)
            p_hat = p_hat.to(h_layer.dtype)

            # 4b.5 DeepStack injection (Qwen3-VL adds visual features to
            #      early decoder layers' hidden states at image positions)
            if deepstack_features and l < len(deepstack_features):
                ds_feat = deepstack_features[l].to(h_layer.device, h_layer.dtype)
                if visual_pos_mask is not None:
                    mask_1d = visual_pos_mask.to(h_layer.device)  # [B, N]
                    for b in range(h_layer.shape[0]):
                        positions = mask_1d[b].nonzero(as_tuple=True)[0]
                        n = len(positions)
                        if n > 0 and n <= ds_feat.shape[0]:
                            h_layer[b, positions] = h_layer[b, positions] + ds_feat[:n]

            # 4c. Extract observation (F3: masked, U1: multi-token)
            z_down = sm.extract_observation(
                h_layer, observation_mask.to(layer_device),
            )

            # 4d. Memory Correct (learned Kalman + obs model)
            m_new, p_new, e, loss_obs_l, loss_nll_l = sm.correct(
                m_hat, p_hat, z_down, q_noise=q_noise,
                surprise_inflation=surprise_inflation, p_prev=variance_state,
            )

            # 4e. Memory Inject (F4: zero-init gated)
            # Defensive dtype alignment: mixed bf16/fp32 slips through on
            # CPU (CUDA's fused attention path tolerates it silently).
            m_new = m_new.to(h_layer.dtype)
            p_new = p_new.to(h_layer.dtype)
            h = sm.inject(h_layer, m_new)

            new_memory.append(m_new)
            new_variance.append(p_new)
            e_list.append(e)
            loss_obs_terms.append(loss_obs_l)
            loss_nll_terms.append(loss_nll_l)

        # ── Aux loss terms (no LM head, no text CE — CompactMoDE change) ──
        loss_device = h.device
        loss_obs_total = torch.stack([
            term.to(loss_device) for term in loss_obs_terms
        ]).sum() / L
        loss_nll_total = torch.stack([
            term.to(loss_device) for term in loss_nll_terms
        ]).sum() / L

        return {
            "hidden_states": h,  # final layer, post-injection, pre-final-norm
            "new_memory": new_memory,
            "new_variance": new_variance,
            "e_list": e_list,  # surprise for next step
            "loss_obs": loss_obs_total,
            "loss_nll": loss_nll_total,
        }

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "CompactModeWrapper never generates text (see idea.md); "
            "use JAMELCompactWrapper for generation."
        )

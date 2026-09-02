# Mem0-Compact — Idea & Design Contract

**Goal:** take Mem-0's execution module (Qwen3-VL-2B + flow-matching DiT action head)
and replace its explicit `MemoryBank` (FIFO feature cache) with COMPACT's per-layer
side memory mechanism, so that temporal memory is *learned, per-layer, and
gradient-trained* instead of a detached cache of pooled features.

This document is the contract between the idea owner and the code agent.
Implementation must follow it exactly; deviations require explicit approval.

---

## 1. Scope (confirmed decisions)

1. **Executor-only.** Only Mem-0's Execution Module (Qwen3-VL-**2B**, in-process,
   trainable) gets the COMPACT side memory. The Planning Module (Qwen3-VL-**8B**,
   frozen, served remotely via vLLM) is **untouched** — it has no per-layer access
   and operates at subtask timescale, where per-frame recurrent memory is
   meaningless. Planner↔executor interaction stays symbolic (subtask text +
   keyframe images).
2. **MemoryBank removed.** Mem-0's `MemoryBank` (FIFO window of 30 detached
   2048-d image features + anchor frame, fused by two cross-attentions) is
   **deleted from the executor path**, not kept alongside. COMPACT memory is the
   *only* temporal memory in the executor.
3. **Memory persists across subtask boundaries.** Mem-0 currently clears memory
   when the SubtaskEndClassifier fires. In Mem0-Compact the COMPACT memory state
   (M, P, e per layer) is **NOT** reset at subtask boundaries — it is carried for
   the whole episode and reset only at **episode start**.
4. **Training is sequential (TBPTT), not i.i.d.** The training loop must run the
   executor over frames **in episode order**, carrying memory state across steps,
   with truncated backpropagation through time — aligned with how CompactMoDE
   trains (detach memory at TBPTT window boundaries; episode start →
   `init_memory`). Mem-0 already ships the right dataloader primitive:
   `RandomEpisodeIterableDataset` (`source/dataloader/random_episode_dataloader.py`)
   reads frames sequentially within an episode. What changes is the *training
   loop*: it must thread memory state across consecutive batches instead of
   treating each batch independently.

## 2. Architecture

### 2.1 What is reused as-is (from Mem-0)
- `Qwen3VL_Encapsulation` backbone loading (Qwen3-VL-2B-Instruct, bf16, flash-attn-2).
- Token pooling: image tokens (between `<|vision_start|>`/`<|vision_end|>`) mean-pooled
  to one (B, 1, 2048) vector; text tokens (between `<|vision_end|>`/`<|im_end|>`)
  mean-pooled to one (B, 1, 2048) vector.
- `FlowmatchingActionHead` (DiT-B, inner 768, 16 layers, flow matching,
  horizon 30, action_dim 16, 8 Euler inference steps).
- `state_encoder` MLP (16 → 2048 → 768) — state goes to the DiT, **not** into the VLM.
- `SubtaskEndClassifier` — still needed: it drives the agent's
  "ask planner for next subtask" loop. Its memory-clearing side effect is removed
  (see decision 3).
- Min-max state/action normalization (`dataset_min_max.py`), unchanged.
- Agent-side temporal action-chunk smoothing, unchanged.

### 2.2 What is reused as-is (from CompactMoDE)
- `vendor/jamel_compact/` (SideMemoryModule, FiLM-GRU predict, Kalman correct,
  zero-init inject, aux losses) — copied, not imported across repos.
- `compact_mode/compact_wrapper.py` (`CompactModeWrapper` — runs the decoder
  manually: predict → pretrained layer → DeepStack → observe → correct → inject).
- Memory dims: `num_mem_tokens=16`, `mem_dim` configurable, **default 128**;
  per-layer memory state + variance + surprise carried externally.
- Aux losses kept: `L_obs`, `L_nll`, `L_mem` with CompactMoDE weights
  (λ_obs=0.01, λ_nll=0.01, λ_mem=0.001), on top of flow-matching action loss
  (λ=1.0) and subtask-end classifier loss (λ=0.2, as in Mem-0).
- `prev_action` (last executed action, embedded via `action_embed` + FiLM-GRU)
  drives the predict step — same as CompactMoDE. First frame of an episode uses
  a zero prev-action.

### 2.3 What changes vs. Mem-0 executor
- Executor forward takes and returns recurrent state:
  `memory_states`, `variance_states`, `e_list` (lists over decoder layers).
- Pooling runs on the **final-layer post-injection** hidden states (then backbone
  final norm) — i.e., image/text features now carry temporal context.
- DiT conditioning tokens: **2 tokens** instead of 3 —
  `cat([image_feature, text_feature])` (B, 2, 2048). The two MemoryBank-fused
  tokens are gone; temporal information now enters through the injected
  hidden states themselves.
- `SubtaskEndClassifier` input shrinks accordingly: 2×2048 = 4096 (was 3×2048
  = 6144). Classifier architecture otherwise unchanged.
- Agent's `update_obs`/`get_action` split is kept, but `update_obs` now also
  returns and stashes the new memory state; `get_action` consumes cached
  features as before.

### 2.4 Trainable vs frozen
- Default: Qwen3-VL-2B base **trainable** (end-to-end), matching Mem-0's default
  (`freeze_modules: ""`). Freezing is opt-in (env flag / config), same convention
  as CompactMoDE (`FREEZE_BASE=1`).
- Always trainable: all `SideMemoryModule` weights, `action_embed`,
  `prev_action_mlp` (if used), DiT head, state encoder, classifier.
- Per-module LRs follow Mem-0's scheme (qwen 1e-5, heads 1e-4); side-memory
  params get the CompactMoDE memory LR (5e-6) as a separate param group.

## 3. Training loop (TBPTT)

- **Per-task training.** One model per task, matching Mem-0's stated scheme
  ("single-task training strategy... trained from scratch for each specific
  task", 30K iterations on 8×A800). Mem-0's released `m1_mix` joint M1 model
  was a resource compromise for the release, NOT the scheme. Multi-task joint
  training is a possible later experiment, not part of this contract.
- Iterate frames **sequentially within an episode** via
  `RandomEpisodeIterableDataset` — verified suitable: `_gen()`
  (`random_episode_dataloader.py:332-346`) yields frames in sorted order within
  an episode, episodes are shard-fixed per rank and per worker (no overlap),
  and each sample carries `episode_id` for boundary detection. Note it silently
  drops `lang == "null"` frames (line 341), so the stream can skip frames
  mid-episode — tolerable for memory, but known.
- The TBPTT work is in the **training loop**, not the dataloader. Mem-0's
  current loop does `loss.backward()` per batch (`train_low.py:229-234`) and
  their MemoryBank vectors are detached — no BPTT exists today. Our loop must:
  carry `(memory_states, variance_states, e_list, prev_action)` across steps;
  accumulate loss over a K-frame window; one backward per window; `detach()`
  memory at window boundaries; gradients never cross windows.
- Per-sample episode reset: when a sample slot's `episode_id` changes between
  consecutive batches, reset that slot's memory to `init_memory`/`init_variance`,
  `e=None`, `prev_action=zeros`. **No reset at subtask boundaries**
  (decision 3).
- Loss per frame = flow-matching action loss + 0.2 × subtask-end BCE/focal
  + 0.01 × L_obs + 0.01 × L_nll + 0.001 × L_mem.
- DDP: episode-level sharding already exists (`episode % world_size == rank`);
  memory state is per-sample, no cross-rank coupling.

## 3.5 M(1) vs M(n) task types (verified from Mem-0 README + data-prep scripts)

The executor training is structurally identical for both; only the
language-conditioning data and the planner's existence differ.

- **M(1) tasks** (`observe_and_pickup`, `put_back_block`, `rearrange_blocks`,
  `swap_blocks`, `swap_T`): data prep (`M1_dataset_to_lerobot.py`) gives the
  whole episode one fixed instruction (the global task text); no subtask
  segmentation; no planner at train or eval; executor runs standalone.
- **M(n) tasks** (`battery_try`, `blocks_ranking_try`, `cover_blocks`,
  `press_button`, `place_block_mat`): data prep (`Mn_dataset_to_lerobot.py`)
  reads `data/<task>/demo_clean/language_annotation.json`, segments episodes
  into `[start_frame, end_frame, subtask_text]`, labels each frame with its
  current subtask text plus `subtask_end=True` within 8 frames of a boundary,
  and stores `global_task`. Planner is LoRA-trained separately via
  LLaMA-Factory and served via vLLM at eval; the SubtaskEndClassifier
  (threshold=2) triggers planner calls.
- **Mem0-Compact policy for both types**: executor swap is identical; the
  planner (M(n) only) stays frozen and untouched. For M(1) the classifier has
  no meaningful mid-episode boundaries — **disable the classifier and its loss
  for M(1) tasks**; keep it for M(n).
- **First target: `swap_blocks` (M1)** — matches our CompactMoDE experiments,
  no planner needed, cleanest validation of the idea. M(n) tasks follow once
  the executor is validated.

## 3.6 Weight initialization

Mem-0 released official weights (RMBench README lines 11–12):
`qiuly/Mem-0-m1mix-RMBench` (joint M1 model) and `qiuly/Mem-0-mn-RMBench`
(per-task executor ckpts for `battery_try`, `blocks_ranking_try`,
`cover_blocks`, `press_button`, + norm stats).

Compatibility if used as init:
- Qwen3-VL-2B backbone: **loads cleanly** — side memory lives outside the LLM
  layers and `delta_up` is zero-init, so a freshly wrapped backbone is
  behaviorally identical to the base model at init.
- DiT action head: **loads cleanly** — cross-attention token count (3→2) is a
  sequence-length dimension, not a weight shape. Caveat: it was trained
  expecting MemoryBank-fused tokens, so a distribution shift exists.
- SubtaskEndClassifier: **cannot load** (input dim 6144→4096); re-init.
- Side memory modules: always fresh.

**DECISION (owner, confirmed):** train **from scratch**. No released-weight
warm-start. The released weights are still downloaded locally
(`policy/Mem0-Compact/checkpoints/released/`) for reference and for running the
original Mem-0 as the comparison baseline.

**Training settings align with Mem-0's per-task recipe where feasible**
(README "GPU Resource Requirements"): 30K training iterations, global batch 448
(56 × 8 A800), per-module LRs (qwen 1e-5, action head/classifier 1e-4) + side
memory LR 5e-6, bf16. Deviations only where the COMPACT mechanism forces them
(TBPTT windowing) or where the server setup requires; any such deviation must
be documented in the run config.

**First task: `swap_blocks` (M1)** — from scratch, executor-only, no planner.

## 4. Eval / deployment

- Planner: unchanged, remote vLLM 8B server.
- Executor runs **every frame**, memory updated at every forward
  (receding-horizon style, same as CompactMoDE deploy: predict 30-action chunk,
  execute per Mem-0's temporal smoothing, memory state advances each forward).
- Memory reset at episode start only. SubtaskEndClassifier still gates the
  planner call, but no longer clears any executor memory.
- Batch size 1 at eval, memory state is per-episode.

## 5. Code provenance rules (hard constraints)

- **Do NOT modify** the original repos: `/home/spc/JAMEL-COMPACT`,
  `/home/spc/MoDE_Diffusion_Policy`, and `policy/Mem-0/`.
- All needed code is **copied** into `policy/Mem0-Compact/` and modified there:
  - from `policy/CompactMoDE/vendor/jamel_compact/` (which is itself the
    vendored JAMEL-COMPACT copy, including the `return_hidden_states` patch and
    the CPU dtype-guard fixes — copy the *current vendored* version),
  - from `policy/CompactMoDE/compact_mode/` (wrapper, losses, config),
  - from `policy/Mem-0/source/` (executor, action head, classifier, agent,
    dataloaders, training loop) — copy then modify.
- The new policy must be self-contained under `policy/Mem0-Compact/` and follow
  RMBench policy conventions (deploy_policy entry, eval integration like other
  policies).

## 6. Verification plan (local, mirrors what we did for CompactMoDE)

1. Smoke test: model builds, forward/backward on synthetic batch, memory state
   shapes correct, aux losses finite.
2. TBPTT check: run 2 windows over a real episode, confirm memory detaches
   (no grad across boundary) and episode reset works.
3. Short local training run (~20 steps, `FREEZE_BASE=1` allowed) in conda
   `lerobot` env on `swap_blocks`.
4. Local eval in conda `syb_RMBench` env with `--device cpu`,
   `demo_clean_eval.yml`, `--test_num 1 --eval_step_cap 20` — success rate
   irrelevant; the pipeline must complete and write artifacts.
5. Then hand off to the A800 server for real training (8×A800, no VRAM
   constraints, full `demo_clean` config, GPU eval).

## 7. Explicit non-goals (for now)

- No changes to the 8B planner or its LoRA training.
- No hybrid "MemoryBank + COMPACT" combination — replacement only.
- No memory reset at subtask boundaries (may be revisited as an ablation later).
- No GPU-memory-saving tricks beyond what already exists; server has 8×A800.

## 8. Open questions (defer unless they block implementation)

- TBPTT window size K: start from CompactMoDE's value; sweep later.
- Whether `mem_dim=128` is right for Mem-0's 2048-d pooled features — sweep
  8–512 later, default 128.
- Whether the subtask-end classifier should also condition on memory-pooled
  features (currently it sees the same 2 summary tokens as the DiT).

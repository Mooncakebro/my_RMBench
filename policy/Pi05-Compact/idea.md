# Pi05-Compact — Idea & Design Contract

**Goal:** graft COMPACT's per-layer side memory (JAMEL-COMPACT: FiLM-GRU predict →
learned-Kalman correct → zero-init gated inject) onto **π0.5** (openpi PyTorch port,
PaliGemma-2B + 300M action expert, flow matching), and SFT on **RMBench**
(LeRobot v3.0 datasets, already converted on the server), **warm-started from the
official `pi05_base` checkpoint**. π0.5 is Markovian (single-frame observation →
action chunk) and is a published RMBench baseline that underperforms on
memory-dependent tasks; COMPACT memory is the missing ingredient.

This document is the contract between the idea owner and the code agent.
Implementation must follow it exactly; deviations require explicit approval.

Sibling references (read them first; reuse their code per §8):
- `policy/Mem0-Compact/idea.md` — the same COMPACT graft onto Mem-0's executor
  (Qwen3-VL-2B + DiT head). Training loop, TBPTT, DDP, checkpointing conventions
  are taken from there.
- `policy/CompactMoDE/compact_mode/` — original COMPACT+VLA wrapper and losses.
- `policy/Mem0-Compact/vendor/jamel_compact/` — vendored COMPACT memory modules.
- `policy/pi05/` — vendored openpi (JAX) used for the existing π0.5 RMBench
  baseline; deploy conventions come from there.

---

## 1. Scope (confirmed decisions)

1. **Framework: openpi's PyTorch port** (`/home/spc/openpi/src/openpi/models_pytorch/`),
   NOT the JAX/NNX canonical implementation. Rationale: COMPACT is pure PyTorch
   (`torch.nn`), and the port already implements pi05 (adaRMS, discrete-state
   prompt convention) and dual-expert joint attention.
2. **Memory placement: prefix (PaliGemma) layers only.** 18 decoder layers,
   hidden width d=2048. The 300M action expert (width 1024) is untouched — it
   already reads the prefix through joint attention, so memory injected into the
   prefix reaches the action decisions. Side memory state therefore exists only
   for expert 0.
3. **Action-as-control-input:** a small **action encoder MLP** maps the previous
   action to the LLM width and feeds every layer's FiLM-GRU predict step
   (details §3.4). Zeros at episode start.
4. **Sequence-aware sampling + TBPTT training**, aligned with Mem0-Compact
   (`source/training/train_compact.py`): frames in episode order, memory threaded
   across frames, one backward per K-frame window, detach at window boundaries,
   per-slot episode reset via `episode_id` change. NOT i.i.d. frame sampling.
5. **Data: RMBench LeRobot v3.0 datasets** (already converted; on the server;
   local copies at `policy/Mem0-Compact/lerobot_datasets/<task>/`).
6. **Warm start from official `pi05_base`** weights (see §6). Zero-init injection
   means the wrapped model is exactly π0.5 at init.
7. **Training recipe aligned with Mem0-Compact**: DDP with episode sharding,
   per-module LRs, cosine+warmup, grad clip, `save_best` on validation action
   loss, per-task training, first target `swap_blocks` (M1).
8. **Executor-only.** No planner, no subtask classifier in v1. M(1) tasks first
   (`observe_and_pickup`, `put_back_block`, `rearrange_blocks`, `swap_blocks`,
   `swap_T`); M(n) tasks later reuse the same executor with per-frame subtask
   prompts (memory NOT reset at subtask boundaries, same as Mem0-Compact
   decision 3).

## 2. Background: the two halves

### 2.1 π0.5 (PyTorch port) — relevant facts

From `/home/spc/openpi/src/openpi/models_pytorch/pi0_pytorch.py` and
`gemma_pytorch.py`:

- `PI0Pytorch(nn.Module)` holds `PaliGemmaWithExpertModel` (SigLIP So400m/14
  vision tower + Gemma-2B LM = expert 0; Gemma-300M = expert 1, joint attention).
- **Prefix** (expert 0, width 2048): per camera, SigLIP → 256 tokens of 2048
  (224×224 input, patch 14); plus language tokens (`max_token_len=200`, pi05
  prompt format `"Task: {prompt}, State: {256-bin discretized state};\nAction: "`).
- **Suffix** (expert 1, width 1024): `action_horizon` tokens, `action_dim=32`;
  time injected via adaRMS in the action expert only (`use_adarms=[False, True]`).
- **Attention**: block-wise — prefix fully bidirectional; suffix block attends to
  the whole prefix and is bidirectional within itself; prefix never attends to
  suffix. Masks: `pad_masks` (validity) + `att_masks` (block flags) →
  `make_att_2d_masks`; positions `cumsum(pad_masks)-1`.
- **Training loss** (`PI0Pytorch.forward`, pi0_pytorch.py:317): per frame,
  `t ~ Beta(1.5, 1)·0.999+0.001`, `x_t = t·noise + (1−t)·actions`,
  target velocity `u_t = noise − actions`, one joint forward,
  `F.mse_loss(u_t, v_t, reduction='none')` → `(B, H, 32)`.
- **Inference** (`sample_actions`, pi0_pytorch.py:377): one prefix forward with
  KV cache, then `num_steps=10` Euler denoise steps (suffix-only forwards),
  t: 1→0. Prefix KV cache: `inputs_embeds=[prefix_embs, None]` branch of
  `PaliGemmaWithExpertModel.forward`.
- **The joint layer loop is the integration point**:
  `PaliGemmaWithExpertModel.forward` (gemma_pytorch.py:90) → joint branch →
  `compute_layer_complete(layer_idx, ...)` (gemma_pytorch.py:~150): per layer it
  RMSNorms both experts' streams, builds q/k/v per expert, concatenates along
  the sequence axis, runs one attention, splits, applies per-expert output
  proj + MLP. Prefix-layer hidden states are directly accessible here.

### 2.2 COMPACT side memory (v2, vendored) — relevant facts

From `policy/Mem0-Compact/vendor/jamel_compact/model.py`, `SideMemoryModule`
(one instance per layer; new parameters only; pretrained layers untouched):

State per layer per sample: `M [B, N_m=16, d_mem=128]`, `P [B, 16]` (variance),
`e [B]` (surprise, detached). Learnable `init_memory [16, 128]`,
`init_variance=0.5`. Hierarchical hypers: layers split in thirds,
λ ∈ {0.70, 0.85, 0.95} (learnable per slot), inject weight {0.8, 0.5, 0.3}.

Per frame, per layer:
1. **Predict**: `m_hat = FiLMGRU(M, action_embed↓128)`; variance predict
   `p_hat = sigmoid(λ)·P + softplus(Q_theta(a)) + γ_e·clamp(e_prev)` (surprise
   inflation, clip 10.0).
2. **(pretrained layer runs on the main stream)**
3. **Extract observation**: down-project hidden states `h [B, N, 2048]` → 128,
   k=4 learned queries attend over **non-padding** positions
   (`key_padding_mask`) → `z [B, 4, 128]`.
4. **Correct (learned Kalman)**: innovation `Δ = proj(CrossAttn(M̂, z, z))`;
   `R = softplus(R_psi(mean z)) + 0.01`; `K = p_hat/(p_hat+R)`;
   `M = M̂ + K·Δ`; `P = (1−K)·p_hat`. Observation model + surprise:
   `e = MSE(obs_model(mean M̂), mean z)` (detached), producing aux losses
   `L_obs` (MSE) and `L_nll` (Gaussian NLL `0.5·(log R + e/R)`).
5. **Inject**: `h += w_max·tanh(gate)·delta_up(CrossAttn(h↓128, M, M))`;
   `delta_up` is **zero-initialized** (model == base at init), `gate` init 0.1.

Aux losses from the wrapper: `loss_obs`, `loss_nll` (averaged over layers);
`L_mem` (memory L2) from `vendor/jamel_compact/loss.py`.

## 3. Architecture integration

### 3.1 New module layout

`Pi05CompactModel(nn.Module)` (in `policy/Pi05-Compact/`, see §8) wraps a
`PI0Pytorch` instance and adds:

- `side_memories`: `nn.ModuleList` of **18** `SideMemoryModule`s (one per
  PaliGemma decoder layer; `hidden_dim=2048`, `mem_dim=128`, `num_mem=16`,
  `num_heads=8`, `num_obs_tokens=4`) — copied vendored class, unmodified.
- `action_embed`: MLP mapping prev-action → 2048 (§3.4).
- Memory state containers and helpers: `init_memory(B, device)` →
  `(memory_states: list[18] of [B,16,128], variance_states: list[18] of [B,16],
  e_list: list[18] of None)`; `detach_memory(memory)`; `reset_memory_rows(memory,
  reset_mask, device)` — same API surface as Mem0-Compact's executor so the
  SAME training-loop skeleton drives it.

### 3.2 Per-layer flow (training forward, one frame)

Inside the joint layer loop (modified `compute_layer_complete`, §8 file map),
for each layer `l`:

```
m_hat, p_hat, q_noise, surprise_infl = sm[l].predict(M[l], P[l], a_emb, e_prev=e[l])
(prefix_h, suffix_h) = <unchanged joint layer computation>   # attention+MLP both experts
z      = sm[l].extract_observation(prefix_h, prefix_token_valid_mask)  # [B,4,128]
M[l]', P[l]', e[l]', l_obs, l_nll = sm[l].correct(m_hat, p_hat, z, ...)
prefix_h = sm[l].inject(prefix_h, M[l]')                       # zero-init gated
```

- Memory ops touch **only the prefix stream** (expert 0). The suffix stream
  (action expert) passes through unmodified.
- Predict runs BEFORE the layer (pure prior: no information leak from the
  current observation into the prediction — same invariant as COMPACT).
- The observation mask must be the per-token prefix validity: real (non-pad)
  image tokens of **cameras with `image_mask=True`** + real language tokens.
  This requires threading the prefix `pad_masks` (and per-camera image masks,
  expanded to their 256 tokens each) down into the layer loop; padding and
  masked-out cameras must never enter observation pooling.

### 3.3 Prefix/suffix construction — unchanged

`embed_prefix`/`embed_suffix`/`forward` of `PI0Pytorch` stay as-is, including
the pi05 prompt format and adaRMS time conditioning. The only changes:
- the joint forward gains memory in/out arguments and the per-layer side ops
  (§3.2);
- `forward()` additionally returns `loss_obs`, `loss_nll` (mean over layers)
  and the new memory state;
- `L_mem` is computed from the returned memory states
  (`vendor/jamel_compact/loss.py`).

### 3.4 Action encoder (control input u_{t-1})

- **Training**: `prev_action` = the ground-truth action of the **previous frame**
  (teacher forcing; same normalized 16-dim action space as the dataset), zeros
  at episode start. (Mem0-Compact uses the same convention via `batch["state"]`
  slot; we pass the true previous action instead — owner-approved deviation to
  record in the run config.)
- **Inference**: the **last executed action** from the previous inference call.
- Encoder: `prev_action (B,16)` → LayerNorm → Linear(16→512) → GELU →
  Linear(512→2048) → `action_embed (B, 2048)`. Each `SideMemoryModule` has its
  own `action_down` (2048→128), so no per-layer copies needed.

### 3.5 Inference path (sample_actions)

- Memory ops run **only in the prefix pass** (the pass that builds the KV
  cache). The 10 Euler denoise steps are suffix-only and do NOT touch memory.
- `sample_actions` gains optional `memory` in/out; per decision step it returns
  the updated memory state, which the deployment wrapper (§7) carries across
  calls and resets at episode start.
- Consequence of zero-init inject: at init, Pi05-Compact == pi05_base
  numerically (memory contributes exactly 0).

## 4. Data

### 4.1 Source

RMBench LeRobot v3.0 datasets on the server (local reference:
`policy/Mem0-Compact/lerobot_datasets/<task>/`). 12 tasks; `put_back_block`,
`swap_blocks`, `cover_blocks`, `place_block_mat` have 250 episodes (50 + 200
appended), the rest 50. Verified features (e.g. `swap_blocks/meta/info.json`):

| key | dtype | shape |
|---|---|---|
| `observation.state` | float32 | [16] |
| `action` | float32 | [16] |
| `observation.image.head_camera` | video | [240, 320, 3] |
| `subtask`, `global_task` | string | [1] |
| `subtask_end` | int32 | [1] |
| `episode_id`, `episode_index`, `frame_index`, `timestamp` | — | [1] |

fps=30. **The dataset has only the head camera**; the eval env additionally
provides left/right wrist cameras (see §7). Training therefore uses 1 real
camera + 2 masked-out cameras.

### 4.2 Transform to π0.5 Observation

Per frame (before batching):
- `head_camera` → `base_0_rgb`; `left_wrist_0_rgb`/`right_wrist_0_rgb` ← zero
  image with `image_mask=False` (openpi convention; `model.py` requires all
  three keys). Resize with letterbox pad to 224×224.
- `state`: 16-dim → quantile-normalized (q01/q99 per task, openpi pi05
  convention) → zero-padded to 32.
- `actions`: chunk of `action_horizon` future frames via LeRobot
  `delta_timestamps` at fps 30 → quantile-normalized → zero-padded to 32.
  **action_horizon = 30** (1 s at 30 fps, matching Mem-0/Mem0-Compact);
  keep it a config knob.
- Prompt: M(1) = fixed `global_task` text; M(n) = current frame's `subtask`
  text. pi05 tokenizer format with `discrete_state_input=True`
  (warm-start consistency: pi05_base was trained with the state serialized in
  the prompt), `max_token_len=200`.
- Norm stats: compute per task over train episodes (q01/q99), saved under
  `policy/Pi05-Compact/assets/<task>/norm_stats.json`
  (mirror `policy/Mem0-Compact/scripts/gen_norm_stats.py`).

### 4.3 Sequence-aware loading

Copy `RandomEpisodeIterableDataset`
(`policy/Mem0-Compact/source/dataloader/random_episode_dataloader.py`) and
adapt: frames yielded **sequentially within an episode**; episodes sharded per
DDP rank (`episode % world_size == rank`) and per worker; every sample carries
`episode_id` for boundary detection; `num_workers=1` (temporal memory assumes
batch-slot continuity); `shuffle_episodes=True`, infinite stream. Known quirk to
preserve or fix deliberately: it silently drops `lang == "null"` frames
(random_episode_dataloader.py:~341).

## 5. Training loop (TBPTT) — aligned with Mem0-Compact

Adapt `policy/Mem0-Compact/source/training/train_compact.py` (1058 lines,
already DDP-correct). Keep its loop structure verbatim; swap the model/loss:

- **State threading**: per rank, `memory = raw_model.init_memory(batch_size,
  device)`; carried across batches. Per-slot reset when `episode_id` changes
  (memory → init, `e=None`, `prev_action=0`). No reset at subtask boundaries.
- **Windowing**: TBPTT window `K=8` frames; accumulate loss over the window;
  **one backward per window** (memory legitimately connects the K frames in one
  graph); `memory = detach_memory(memory)` at every window boundary;
  `grad_accum_windows` detached windows per optimizer update (default 7 →
  effective 448 frames/update on 8 GPUs, matching Mem-0's global batch 448).
- **DDP**: `torchrun`; forward must go through the DDP-wrapped module
  (`model(batch, memory)`) so reducer hooks install every frame; `no_sync()`
  around non-last accumulation windows; `all_ranks_finite` guards on loss and
  grad norm (skip update on any non-finite rank); memory is per-sample,
  no cross-rank coupling.
- **Loss per frame**:
  `L = 1.0·L_flow + 0.01·L_obs + 0.01·L_nll + 0.001·L_mem`.
  `L_flow` = mean over `(B, H, 32)` of π0.5's flow-matching MSE (Beta(1.5,1)
  time sampling, as in `PI0Pytorch.forward`). No classifier term (M1 first;
  M(n) classifier is a later addition).
- **Optimizer**: AdamW, per-module param groups (Mem0-Compact scheme, mapped):
  - PaliGemma backbone (vision tower + LM expert 0): **lr 1e-5**
  - action expert + `action_in_proj`/`action_out_proj` + time MLPs: **lr 1e-4**
  - side memories + action encoder: **lr 5e-6**
  - `weight_decay=0.005`; cosine schedule with `warmup_ratio=0.05`;
  grad clip by global norm **2.5**; bf16 compute.
- **Freeze strategy**: base trainable by default (`freeze_base=false`), opt-in
  `FREEZE_BASE=1` for small-GPU debug — same convention as Mem0-Compact.
- **Checkpointing**: `save_best=true` — overwrite `ckpt_best.pt` when
  **validation action loss** (episode-level split, `fraction=0.1`,
  `seed=100045`, run every 1000 updates, from step 1000) improves;
  `save_final=true`; checkpoints contain model + optimizer + scheduler + step +
  config; `resume` support. Checkpoint dir `policy/Pi05-Compact/runs/<task>/`.
- **Scale**: `max_steps=30000` optimizer updates, per-task training.
  Debug: `--window-size 1 --grad-accum-windows 1 --max-steps 100`.
  Server command (8×A800):
  ```bash
  torchrun --nproc_per_node=8 source/training/train_pi05_compact.py \
      --task swap_blocks --batch-size 1 --window-size 8 \
      --grad-accum-windows 7 --max-steps 30000
  ```

## 6. Warm start from official pi05_base (hard requirement)

1. Download the official checkpoint (openpi asset):
   `gs://openpi-assets/checkpoints/pi05_base` (via `openpi.shared.download`).
2. Convert JAX → PyTorch with the repo's official converter:
   ```bash
   python /home/spc/openpi/examples/convert_jax_model_to_pytorch.py \
       --checkpoint_dir ~/.cache/openpi/openpi-assets/checkpoints/pi05_base \
       --output_path  ~/.cache/openpi/openpi-assets/checkpoints/pi05_base_pytorch
   ```
3. Load the converted state dict into `PI0Pytorch` inside `Pi05CompactModel`;
   side memories + action encoder initialize fresh. Because `delta_up` is
   zero-init (§2.2), the wrapped model is **numerically identical to pi05_base
   at init** — verify this explicitly (§9).
4. Parity check before any training: run `sample_actions` on a fixed
   observation with fixed noise under (a) JAX pi05_base, (b) converted PyTorch
   pi05_base, (c) Pi05CompactModel at init; (b) and (c) must match (a) up to
   float tolerance.

## 7. Deployment / RMBench eval contract

Follow `policy/pi05/deploy_policy.py` and `policy/Mem0-Compact/deploy_policy.py`.
`policy/Pi05-Compact/deploy_policy.py` must expose the RMBench harness contract
(`script/eval_policy.py`): `get_model(usr_args)`, `eval(TASK_ENV, model,
observation)`, `reset_model(model)`.

- Observation encoding (env side): `observation["observation"]["head_camera"]
  ["rgb"]`, `["right_camera"]["rgb"]`, `["left_camera"]["rgb"]` (all three real
  at eval), `state = observation["joint_action"]["vector"]` (16-dim).
- The model object holds the memory state; `reset_model` clears it
  (episode start only).
- **Memory cadence at eval (default, mirroring Mem0-Compact's agent):** every
  env frame, run `update_obs` — one prefix forward that advances memory and
  produces the action chunk via `sample_actions`; execute per temporal
  action-chunk smoothing. This gives per-frame memory updates.
  Cheaper alternative (ablation): execute the first `pi0_step` actions of each
  chunk before re-inferring (existing `policy/pi05` convention); memory then
  advances once per chunk. Default = per-frame; record the choice in the
  deploy yml.
- Batch size 1 at eval; memory is per-episode.

## 8. Code provenance and file layout (hard constraints)

- **Do NOT modify** the original repos: `/home/spc/JAMEL-COMPACT`,
  `policy/Mem0-Compact/`, `policy/CompactMoDE/`, `policy/pi05/`, and the openpi
  working tree `/home/spc/openpi`.
- Everything is **copied** into the self-contained `policy/Pi05-Compact/`:

```
policy/Pi05-Compact/
├── idea.md                          # this document
├── vendor/
│   ├── jamel_compact/               # copy from policy/Mem0-Compact/vendor (current version)
│   └── openpi_torch/                # copy of /home/spc/openpi: models_pytorch/, models/ (config,
│                                    #   tokenizer), transforms.py, shared/, packages/openpi-client;
│                                    #   the layer-loop modification lives HERE, not upstream
├── source/
│   ├── models/pi05_compact_model.py # Pi05CompactModel: wraps PI0Pytorch + 18 SideMemoryModules
│   │                                #   + action encoder + memory state API (init/detach/reset_rows)
│   ├── models/gemma_joint_patch.py  # modified PaliGemmaWithExpertModel joint loop
│   │                                #   (predict → joint layer → observe → correct → inject,
│   │                                #   prefix stream only; prefix validity mask threaded through)
│   ├── dataloader/                  # copied RandomEpisodeIterableDataset + norm-stats loader
│   ├── training/train_pi05_compact.py  # adapted from Mem0-Compact train_compact.py
│   └── config/pi05_compact_train.yaml  # mirrors mem0_compact_train.yaml structure
├── scripts/
│   ├── convert_pi05_base.sh         # §6 warm-start conversion
│   └── gen_norm_stats.py            # per-task q01/q99 stats → assets/<task>/
├── assets/<task>/norm_stats.json
├── deploy_policy.py / deploy_policy.yml / eval.sh
└── debug/                           # smoke_forward / tbptt_check / parity_check / test_deploy
```

- Modification surface in the vendored openpi copy is limited to:
  `PaliGemmaWithExpertModel.forward` joint branch (per-layer memory hooks),
  `PI0Pytorch.forward` / `sample_actions` (memory in/out + aux losses). SigLIP,
  tokenizer, adaRMS, flow-matching math stay untouched.

## 9. Verification plan (mirrors Mem0-Compact §6)

1. **Smoke test**: model builds; forward/backward on a synthetic batch; memory
   state shapes `M [B,16,128] × 18`, `P [B,16] × 18`, `e [B] × 18`; aux losses
   finite; zero-init check (`debug/smoke_forward.py`, cf. JAMEL-COMPACT
   `scripts/check_zero_init.py`): inject branch outputs exactly 0 at init.
2. **Warm-start parity** (§6.4): JAX vs PyTorch-converted vs wrapped-init
   action equality on fixed input.
3. **TBPTT check** (`debug/tbptt_check.py`): 2 windows over a real episode;
   memory detaches at boundaries (no grad across windows); episode reset on
   `episode_id` change; DDP plumbing test on 2 GPUs.
4. **Short local run** (~20 steps, `FREEZE_BASE=1` allowed) on `swap_blocks`
   in the conda `lerobot` env.
5. **Local eval pipeline** (conda `syb_RMBench` env, `--device cpu`,
   `--test_num 1 --eval_step_cap 20`): success rate irrelevant; the harness
   must complete and write artifacts; memory must reset between episodes.
6. Hand off to the A800 server: full training (§5), then real GPU eval on all
   5 M(1) tasks; compare against `policy/pi05` (π0.5 baseline) and Mem0-Compact
   numbers.

## 10. Non-goals / open questions

**Non-goals (v1):**
- No planner, no subtask-end classifier (M(n) support is a later increment).
- No LoRA variant (full FT default, `FREEZE_BASE` opt-in only).
- No memory in the action expert; no hybrid "explicit memory bank + COMPACT".
- No changes to the JAX openpi path.

**Open questions (defer unless blocking):**
- `mem_dim=128` for a 2048-wide backbone is inherited from CompactMoDE/
  Mem0-Compact; sweep {64, 128, 256, 512} later.
- TBPTT window K=8 inherited; sweep {4, 8, 16} later.
- action_horizon 30 vs 10 (pi05_libero precedent) — start at 30 for Mem-0
  comparability, ablate.
- Whether observation extraction should pool only image tokens (current
  default: image + language, matching COMPACT) — ablate.
- Per-frame vs per-chunk memory cadence at eval (§7) — ablate after v1 works.

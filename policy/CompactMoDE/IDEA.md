# IDEA — CompactMoDE: Memory-Augmented VLA on RMBench

> This document records the project's design intent **as decided by the project owner**.
> Any code agent working on this repo must align with these decisions. If a change
> contradicts one of them, stop and ask before proceeding.

## Goal

Robot manipulation imitation learning on **RMBench** (memory-dependent dual-arm
manipulation benchmark, SAPIEN sim, AgileX Cobot Magic / aloha-agilex embodiment,
14-dim joint-space actions) using:

- **COMPACT-2B** (`/home/spc/JAMEL-COMPACT`): Qwen3-VL-2B-Instruct + per-layer
  recurrent side memory (Kalman-filter-style predict→correct→inject; 16 slots ×
  512-dim per layer × 28 layers), and
- **MoDE DiT action head** (`/home/spc/MoDE_Diffusion_Policy`): EDM diffusion
  (Karras preconditioning) with a Mixture-of-Denoising-Experts transformer.

RMBench tasks are memory-dependent (M(1): remember one past event; M(n):
multi-stage memory). The hypothesis: COMPACT's persistent side memory should beat
a stateless VLM policy on these tasks.

## Two models to train (both required, for comparison)

1. **Baseline**: Qwen3-VL-2B + conditioning bridge + MoDE DiT. Stateless per frame.
2. **COMPACT+MoDE**: JAMELCompactWrapper (Qwen3-VL-2B + 28 side-memory modules)
   + conditioning bridge + MoDE DiT. Memory carried across frames of an episode.

This mirrors the original JAMEL-COMPACT repo structure, which ships both a
memory model and a no-memory baseline (`train.py` / `baseline_train.py`).
No "stage A / stage B" staged training plan — just the two models.

## Locked design decisions (owner-approved)

1. **Proprio is NOT fed into the VLM.** The 14-dim joint state goes only to the
   DiT head (MoDeDiT's `robot_obs` input). The VLM sees image + instruction only.
2. **The VLM runs EVERY frame during training** (stride 1), like an ordinary
   network. No frame subsampling. (Training hardware: server with up to 8×A800;
   do NOT add GPU-memory-saving compromises to training code for the 8GB dev PC.)
3. **COMPACT's auxiliary losses are kept**: L_obs, L_nll (per-layer, from the
   patched wrapper forward) and memory L2, with the original weights
   (lambda_obs=0.01, lambda_nll=0.01, lambda_mem=0.001). The text CE term is
   dropped — the policy never generates text.
4. **Normalization is on by default**, state and action: z-score for the 12 arm
   joint dims, minmax→[-1,1] for the 2 gripper dims (indices 6 and 13), like
   Mem-0. Stats come from each dataset's `meta/stats.json`.
5. **Qwen3-VL-2B is trainable BY DEFAULT** in both models, exactly like the
   original JAMEL-COMPACT convention: frozen only when `FREEZE_BASE=1` is set in
   the shell environment.
6. **The DiT is conditioned on VLM hidden states ONLY — never on memory slots.**
   Conditioning bridge:
   - final-layer hidden states at **image-token positions** → adaptive-avg-pool
     to 32 tokens → Linear → `state_images` (B, 32, 2048)
   - final-layer hidden states at **instruction-text positions** → masked mean →
     Linear → `goal` (B, 1, 512)
   (Pooling by token type was chosen over MoDE's `model_qwen.py` approach of
   pooling the whole mixed sequence twice.)
7. **Action head config**: action_dim=14, state_dim=14, action_seq_len=10
   (chunk of 10 joint-target steps), EDM sigma_data=0.5, 10 DDIM sampling steps
   at inference. MoDeDiT: embed_dim 1024, 6 layers, 8 heads, 4 experts top-2,
   noise-conditioned router.
8. **Continuous prev-action embedding for the memory predict step** (replaces
   COMPACT's text-action embedding): MLP(14→256→hidden_dim); zeros at episode
   start; the *normalized* previous action is used.
9. **Memory training**: episode-sequential chunked TBPTT, chunk_size=8, memory
   detached at chunk boundaries (same convention as `jamel_compact/train.py`).
   Side memory gets its own small LR (5e-6); base gets base_lr (1e-5);
   bridge+DiT get lr (1e-4).
10. **Camera**: head camera only (matches Mem-0; keeps token count sane).

## Data

- Source: RMBench `demo_clean` HDF5 (symlinked at `RMBench/data/<task>`).
- Converted to LeRobot v3 at `RMBench/data_lerobot/<task>/` (12 tasks × 50
  episodes) by `policy/CompactMoDE/scripts/hdf5_to_lerobot_v3.py`:
  head-cam mp4 (240×320@30fps), `action[t] = joint_action/vector[t+1]`,
  `state[t] = vector[t]`, instruction = `instructions/episode{i}.json["seen"][0]`.
- 4 tasks additionally have `demo_clean_200` (200 episodes) — not yet converted;
  convert if needed with `--src-root` pointing at those dirs.

## Training / eval flow

- Train per task first: `swap_blocks` (M(1)) and `cover_blocks` (M(n)), then
  multi-task mixes. All training on the A800 server.
- This dev PC (RTX 4060 Ti 8GB) is for code + wiring verification only:
  smoke test + single-episode overfit with FREEZE_BASE=1 and a small DiT.
- Eval: RMBench `script/eval_policy.py --config policy/CompactMoDE/deploy_policy.yml`
  (100 episodes, unseen instructions). Compare baseline vs COMPACT+MoDE vs the
  published Mem-0 numbers.

## Hard constraints for code agents

- Do NOT modify the MoDE repo or the JAMEL-COMPACT repo. Both must stay
  byte-identical to their origins. When a change to their code is needed, COPY
  the relevant code into `policy/CompactMoDE/` and modify the copy (or import
  the original modules unmodified):
  - MoDE's single-state-token assumption → `compact_mode/modedit_ext.StateTokenMoDeDiT`
    (subclass, MoDE untouched).
  - JAMEL-COMPACT's forward not exposing hidden states/aux losses →
    `compact_mode/compact_wrapper.CompactModeWrapper` (subclass with a copied,
    adapted forward; also skips the LM head — we never generate text).
- The `embed_pdrob` typo key in the hydra DiT config is required verbatim.
- Env: conda `lerobot` (py3.10, torch 2.10, transformers 4.57.6). Do NOT add
  deps on peft / qwen_vl_utils / timm / flash_attn.
- HF hub is only reachable via `HF_ENDPOINT=https://hf-mirror.com` with
  `ALL_PROXY`/`all_proxy` unset, or offline (`HF_HUB_OFFLINE=1`, weights cached).

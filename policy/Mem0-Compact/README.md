# Mem0-Compact

COMPACT per-layer side memory grafted onto Mem-0's execution module
(Qwen3-VL-2B + flow-matching DiT action head), replacing the explicit
MemoryBank. See `idea.md` for the design contract.

## Architecture

```
per frame:
  head-cam image + instruction ─▶ CompactModeWrapper (Qwen3-VL-2B + 28 SideMemoryModule)
                                   predict(FiLM-GRU) → pretrained layer → DeepStack
                                   → observe → correct(Kalman) → inject
                                   └─ hidden_states (post-injection, final norm)
  image tokens → masked mean ─▶ (B,1,2048) ┐
  text tokens  → masked mean ─▶ (B,1,2048) ┴─▶ summary (B,2,2048) ─▶ FlowmatchingActionHead
  state (16-dim, minmax) ─────────────────────▶ (DiT-B, horizon 30, 8 Euler steps)
  prev_action (16-dim) → MLP → action_embed → memory predict step (zeros at episode start)
```

- Memory (M, P, e per layer; 16 slots × mem_dim 128) is threaded across frames,
  detached at TBPTT window boundaries, reset only at episode start.
- Training data: demo_clean 50 eps/task; 4 tasks (put_back_block, swap_blocks,
  cover_blocks, place_block_mat) additionally include demo_clean_200 (200 eps
  appended, 250 total). Episode mapping reads meta/episodes/*.parquet (fast
  path; avoids scanning + decoding every frame).
- Aux losses: `L = 1.0·L_flow + 0.2·L_cls(Mn) + 0.01·L_obs + 0.01·L_nll + 0.001·L_mem`.
- Classifier (4096 → 2048 → 512) disabled for M(1), enabled for M(n).
- M(n) eval: SubtaskEndClassifier fires sub_end signals; threshold crossing saves
  a keyframe and asks the remote planner (Qwen3-VL-8B via vLLM) for the next
  subtask instruction. COMPACT memory is NOT reset at subtask boundaries.
- Trained from scratch; single-task; swap_blocks (M1) first.

## Layout

```
policy/Mem0-Compact/
├── idea.md                          # design contract
├── deploy_policy.py / .yml / eval.sh
├── vendor/jamel_compact/            # copied from CompactMoDE/vendor (untouched upstream)
├── source/
│   ├── models/execution_module/
│   │   ├── compact_wrapper.py       # CompactModeWrapper (adapted JAMELCompactWrapper)
│   │   ├── mem0_compact_executor.py # Mem0CompactExecutor (wrapper + heads + aux losses)
│   │   ├── action_model/            # copied FlowmatchingActionHead (DiT-B, flow matching)
│   │   └── classifier/              # copied SubtaskEndClassifier (input 4096)
│   ├── agent/mem0_compact_agent.py  # M1 eval agent (memory threading, chunk smoothing)
│   ├── dataloader/                  # copied LeRobot_Dataset + RandomEpisodeIterableDataset
│   ├── training/train_compact.py    # TBPTT training loop
│   └── config/mem0_compact_train.yaml
├── scripts/
│   ├── hdf5_to_lerobot/M1_dataset_to_lerobot.py   # M(1) raw HDF5 → lerobot_datasets/
│   ├── hdf5_to_lerobot/Mn_dataset_to_lerobot.py   # M(n) raw HDF5 + subtasks → lerobot_datasets/
│   ├── convert_all.sh             # batch: all 12 tasks (5 M1 + 7 Mn)
│   └── gen_norm_stats.py          # assets/<task>/norm_stats.json
└── debug/                            # smoke_forward / tbptt_check / test_deploy
```

## Training (env: conda `lerobot`)

```bash
cd policy/Mem0-Compact

# 1. data prep (lerobot v3.0 format) — all 12 tasks in one go:
bash scripts/convert_all.sh
# For the 4 tasks with demo_clean_200 (put_back_block, swap_blocks, cover_blocks,
# place_block_mat) the 200 extra trajectories are APPENDED to the same dataset
# (episode_id offset, 250 episodes total), and norm stats cover all episodes.
# Env knobs: EPISODES=50 EPISODES_200=200 SKIP_200=1 (skip the append phase).
# or single tasks:
#   python scripts/hdf5_to_lerobot/M1_dataset_to_lerobot.py --task swap_blocks --episodes 50
#   python scripts/hdf5_to_lerobot/Mn_dataset_to_lerobot.py --task cover_blocks --episodes 50
#   python scripts/gen_norm_stats.py --task swap_blocks

# 2. local debug run (8GB GPU escape hatches: window 1 + SGD)
python source/training/train_compact.py \
    --config source/config/mem0_compact_train.yaml \
    --task swap_blocks --device cuda --freeze-base 1 \
    --batch-size 1 --max-steps 10 --window-size 1 --opt-sgd

# 3. server run (8×A800, batch 56/rank, 30K windows)
bash source/training/train_ddp.sh
# M(n) tasks: use the classifier-enabled config (λ_cls=0.2 + focal BCE):
#   CONFIG=source/config/mem0_compact_train_mn.yaml TASK=cover_blocks \
#     bash source/training/train_ddp.sh
# or explicitly:
#   NPROC=8 TASK=swap_blocks BATCH_SIZE=56 MAX_STEPS=30000 \
#     bash source/training/train_ddp.sh
#
# DDP notes:
#   - torchrun launches one rank per GPU; episodes are sharded per rank
#     (episode % world_size == rank), global batch = batch_size x NPROC
#   - memory state (M, P, e) is per-rank; no cross-rank coupling
#   - window loss is averaged across ranks for logging only
#   - single-GPU multi-rank debugging auto-falls back to gloo+cpu
#     (NCCL refuses two ranks on one physical device)
#   - DDP smoke test (tiny model, no VLM):
#       torchrun --nproc_per_node=2 debug/ddp_plumbing_test.py
#   - real-model DDP smoke (8GB dev GPU):
#       torchrun --nproc_per_node=1 source/training/train_compact.py \
#         --task swap_blocks --freeze-base 1 --grad-ckpt 1 \
#         --batch-size 1 --window-size 1 --opt-sgd --max-steps 4
```

Note: `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` is required in the
lerobot env (conda libstdc++ must shadow the system one).

## Eval (env: conda `syb_RMBench`, SAPIEN present)

```bash
cd RMBench  # repo root

# M(1)
python script/eval_policy.py --config policy/Mem0-Compact/deploy_policy.yml --overrides \
    --task_name swap_blocks --task_config demo_clean \
    --execution_ckpt policy/Mem0-Compact/runs/.../ckpt_final.pt \
    --state_stats_path policy/Mem0-Compact/assets/swap_blocks/norm_stats.json \
    --device cuda:0

# M(n): classifier is auto-enabled by task name; planner runs via vLLM
# (start it first: vllm serve <merged-8B> --port 8123 or set --vllm_url)
python script/eval_policy.py --config policy/Mem0-Compact/deploy_policy.yml --overrides \
    --task_name cover_blocks --task_config demo_clean \
    --execution_ckpt policy/Mem0-Compact/runs/.../ckpt_final.pt \
    --state_stats_path policy/Mem0-Compact/assets/cover_blocks/norm_stats.json \
    --vllm_url http://localhost:8123 \
    --device cuda:0
```

Deploy pipeline check without SAPIEN (both task types):

```bash
cd policy/Mem0-Compact
python debug/test_deploy.py --ckpt runs/debug_compact/ckpt_final.pt \
    --stats assets/swap_blocks/norm_stats.json --device cpu   # M(1)
python debug/test_deploy.py --mn --device cpu                  # M(n) path
    # (planner is constructed offline; real vLLM calls only in sim eval)
```

## Verification status (local, per idea.md §6)

| Check | Script | Status |
|---|---|---|
| Model builds + fwd/bwd + shapes + finite losses | `debug/smoke_forward.py` | ✔ |
| TBPTT detach + per-slot episode reset | `debug/tbptt_check.py` | ✔ |
| Short training run (swap_blocks, 2 episodes) | `source/training/train_compact.py` | ✔ |
| M(n) executor: classifier loss finite + backward | `debug/smoke_forward.py --config ..._mn.yaml` | ✔ |
| M(n) training run (cover_blocks, 2 episodes, λ_cls) | `source/training/train_compact.py --config ..._mn.yaml` | ✔ |
| Deploy round-trip M(1) (get_model/eval/reset_model) | `debug/test_deploy.py` | ✔ |
| Deploy round-trip M(n) (classifier + planner wiring) | `debug/test_deploy.py --mn` | ✔ |
| SAPIEN sim eval | `script/eval_policy.py` | A800 server (dev RAM too small for sim + 2B VLM) |

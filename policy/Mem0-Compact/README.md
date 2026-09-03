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
- Aux losses: `L = 1.0·L_flow + 0.2·L_cls(Mn) + 0.01·L_obs + 0.01·L_nll + 0.001·L_mem`.
- Classifier (4096 → 2048 → 512) disabled for M(1) tasks.
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
│   ├── hdf5_to_lerobot/M1_dataset_to_lerobot.py   # swap_blocks → lerobot_datasets/
│   └── gen_norm_stats.py             # assets/<task>/norm_stats.json
└── debug/                            # smoke_forward / tbptt_check / test_deploy
```

## Training (env: conda `lerobot`)

```bash
cd policy/Mem0-Compact

# 1. data prep (lerobot v3.0 format)
python scripts/hdf5_to_lerobot/M1_dataset_to_lerobot.py --task swap_blocks --episodes 50
python scripts/gen_norm_stats.py --task swap_blocks

# 2. local debug run (8GB GPU escape hatches: window 1 + SGD)
python source/training/train_compact.py \
    --config source/config/mem0_compact_train.yaml \
    --task swap_blocks --device cuda --freeze-base 1 \
    --batch-size 1 --max-steps 10 --window-size 1 --opt-sgd

# 3. server run (8×A800, batch 56/rank, 30K windows)
bash source/training/train_ddp.sh
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
python script/eval_policy.py --config policy/Mem0-Compact/deploy_policy.yml --overrides \
    --task_name swap_blocks --task_config demo_clean \
    --execution_ckpt policy/Mem0-Compact/runs/.../ckpt_final.pt \
    --state_stats_path policy/Mem0-Compact/assets/swap_blocks/norm_stats.json \
    --device cuda:0
```

Deploy pipeline check without SAPIEN:

```bash
cd policy/Mem0-Compact
python debug/test_deploy.py --ckpt runs/debug_compact/ckpt_final.pt \
    --stats assets/swap_blocks/norm_stats.json --device cpu
```

## Verification status (local, per idea.md §6)

| Check | Script | Status |
|---|---|---|
| Model builds + fwd/bwd + shapes + finite losses | `debug/smoke_forward.py` | ✔ |
| TBPTT detach + per-slot episode reset | `debug/tbptt_check.py` | ✔ |
| Short training run (swap_blocks, 2 episodes) | `source/training/train_compact.py` | ✔ |
| Deploy round-trip (get_model/eval/reset_model) | `debug/test_deploy.py` | ✔ |
| SAPIEN sim eval | `script/eval_policy.py` | A800 server (dev RAM too small for sim + 2B VLM) |

#!/bin/bash
# TBPTT training launcher: direct Python for one GPU, torchrun for DDP.
#
# Example (8×A800, global batch 8 = 1 × 8):
#   NPROC=8 TASK=swap_blocks bash source/training/train_ddp.sh
#
# Env knobs:
#   NPROC        ranks/GPUs (default 1)
#   MASTER_PORT  torchrun port (default 29500)
#   TASK         dataset task under lerobot_datasets/ (default swap_blocks)
#   BATCH_SIZE   per-rank batch (default 1; global = BATCH_SIZE * NPROC)
#   MAX_STEPS    optimizer updates (default 30000)
#   GRAD_ACCUM_WINDOWS detached TBPTT windows per optimizer update (default 1)
#   FREEZE_BASE  1 to freeze the Qwen base (debug only)
#   CONFIG       config yaml (default source/config/mem0_compact_train.yaml)
#   VARIANT      executor variant for the default output dir (default: read
#                from CONFIG's execution_module.variant; runs/${VARIANT}_${TASK})
#   OUTPUT_DIR   explicit output dir (overrides the variant-based default)
#   SAVE_BEST    1 to overwrite ckpt_best.pt on improved validation action loss
#   BEST_START_STEP first step eligible for ckpt_best.pt (default 1000)
#   BEST_MIN_DELTA minimum validation-loss improvement for a best save (default 0.01)
#   SAVE_EVERY_STEPS numbered checkpoint interval; 0 disables (default 0)
#   SAVE_FINAL   1 to save ckpt_final.pt on normal completion (default 1)
#   EXTRA_ARGS   extra train_compact.py flags, e.g. "--grad-ckpt 1"
set -e
cd "$(dirname "$0")/../.."   # policy/Mem0-Compact/

NPROC=${NPROC:-1}
MASTER_PORT=${MASTER_PORT:-29500}
TASK=${TASK:-swap_blocks}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_STEPS=${MAX_STEPS:-30000}
GRAD_ACCUM_WINDOWS=${GRAD_ACCUM_WINDOWS:-1}
FREEZE_BASE=${FREEZE_BASE:-0}
CONFIG=${CONFIG:-source/config/mem0_compact_train.yaml}
# Variant-aware default output dir: read execution_module.variant from the
# config so a baseline run never clobbers runs/compact_<task>. VARIANT can be
# set explicitly to override detection.
VARIANT=${VARIANT:-$(grep -m1 -Po '^\s*variant:\s*\K[A-Za-z0-9_-]+' "$CONFIG" 2>/dev/null || true)}
VARIANT=${VARIANT:-compact}
OUTPUT_DIR=${OUTPUT_DIR:-runs/${VARIANT}_${TASK}}
SAVE_BEST=${SAVE_BEST:-1}
BEST_START_STEP=${BEST_START_STEP:-1000}
BEST_MIN_DELTA=${BEST_MIN_DELTA:-0.01}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-0}
SAVE_FINAL=${SAVE_FINAL:-1}

TRAIN_ARGS=(
  source/training/train_compact.py
  --config "$CONFIG"
  --task "$TASK" --device cuda
  --batch-size "$BATCH_SIZE" --max-steps "$MAX_STEPS"
  --grad-accum-windows "$GRAD_ACCUM_WINDOWS"
  --output-dir "$OUTPUT_DIR"
  --freeze-base "$FREEZE_BASE"
  --save-best "$SAVE_BEST"
  --best-start-step "$BEST_START_STEP"
  --best-min-delta "$BEST_MIN_DELTA"
  --save-every-steps "$SAVE_EVERY_STEPS"
  --save-final "$SAVE_FINAL"
)

if [ "$NPROC" -le 1 ]; then
  python "${TRAIN_ARGS[@]}" ${EXTRA_ARGS:-}
else
  torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    "${TRAIN_ARGS[@]}" ${EXTRA_ARGS:-}
fi

#!/bin/bash
# Multi-GPU TBPTT training via torchrun (one rank per GPU).
#
# Example (8×A800, global batch 448 = 56 × 8):
#   NPROC=8 TASK=swap_blocks bash source/training/train_ddp.sh
#
# Env knobs:
#   NPROC        ranks/GPUs (default 8)
#   MASTER_PORT  torchrun port (default 29500)
#   TASK         dataset task under lerobot_datasets/ (default swap_blocks)
#   BATCH_SIZE   per-rank batch (default 56; global = 56 * NPROC)
#   MAX_STEPS    TBPTT windows (default 30000)
#   FREEZE_BASE  1 to freeze the Qwen base (debug only)
#   CONFIG       config yaml (default source/config/mem0_compact_train.yaml)
#   EXTRA_ARGS   extra train_compact.py flags, e.g. "--grad-ckpt 1"
set -e
cd "$(dirname "$0")/../.."   # policy/Mem0-Compact/

NPROC=${NPROC:-8}
MASTER_PORT=${MASTER_PORT:-29500}
TASK=${TASK:-swap_blocks}
BATCH_SIZE=${BATCH_SIZE:-56}
MAX_STEPS=${MAX_STEPS:-30000}
FREEZE_BASE=${FREEZE_BASE:-0}
CONFIG=${CONFIG:-source/config/mem0_compact_train.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-runs/compact_${TASK}}

torchrun --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
  source/training/train_compact.py \
  --config "$CONFIG" \
  --task "$TASK" --device cuda \
  --batch-size "$BATCH_SIZE" --max-steps "$MAX_STEPS" \
  --output-dir "$OUTPUT_DIR" \
  --freeze-base "$FREEZE_BASE" \
  ${EXTRA_ARGS:-}

#!/bin/bash
# Train the CompactMoDE baseline variant.
# Env-var style, mirroring JAMEL-COMPACT's shell scripts. Example:
#   TASK=swap_blocks FREEZE_BASE=1 BATCH_SIZE=4 bash shell/run_train_baseline.sh
set -e
cd "$(dirname "$0")/.."

TASK=${TASK:-swap_blocks}
# Auto-detect repo root (parent of policy/)
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data_lerobot}
OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)/runs/baseline_${TASK}}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-VL-2B-Instruct}
FREEZE_BASE=${FREEZE_BASE:-0}
GRAD_CKPT=${GRAD_CKPT:-0}
EPOCHS=${EPOCHS:-10}
MAX_STEPS=${MAX_STEPS:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-4}
LR=${LR:-1e-4}
BASE_LR=${BASE_LR:-1e-5}
EMBED_DIM=${EMBED_DIM:-1024}
N_LAYERS=${N_LAYERS:-6}
N_HEADS=${N_HEADS:-8}
NUM_EXPERTS=${NUM_EXPERTS:-4}
TOP_K=${TOP_K:-2}
SAVE_STEPS=${SAVE_STEPS:-1000}
VAL_FRACTION=${VAL_FRACTION:-0.0}

# Auto-detect conda (works on server and dev machine)
if [ -z "$CONDA_PREFIX" ]; then
    CONDA_BASE=${CONDA_BASE:-$(conda info --base 2>/dev/null || echo /opt/conda)}
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate lerobot
fi
# conda's libstdc++ must shadow the system one (GLIBCXX_3.4.29)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
unset ALL_PROXY all_proxy

python train_baseline.py \
  --task "$TASK" --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR" \
  --base-model "$BASE_MODEL" --freeze-base "$FREEZE_BASE" --grad-ckpt "$GRAD_CKPT" \
  --epochs "$EPOCHS" --max-steps "$MAX_STEPS" --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" --lr "$LR" --base-lr "$BASE_LR" \
  --embed-dim "$EMBED_DIM" --n-layers "$N_LAYERS" --n-heads "$N_HEADS" \
  --num-experts "$NUM_EXPERTS" --top-k "$TOP_K" \
  --save-steps "$SAVE_STEPS" --val-fraction "$VAL_FRACTION"

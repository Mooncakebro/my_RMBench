#!/bin/bash
# Train the CompactMoDE (COMPACT + MoDE) variant with chunked TBPTT.
# Env-var style, mirroring JAMEL-COMPACT's shell scripts. Example:
#   TASK=swap_blocks FREEZE_BASE=1 NUM_STREAMS=2 CHUNK_SIZE=8 bash shell/run_train_compact.sh
set -e
cd "$(dirname "$0")/.."

TASK=${TASK:-swap_blocks}
# Auto-detect repo root (parent of policy/)
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data_lerobot}
OUTPUT_DIR=${OUTPUT_DIR:-$(pwd)/runs/compact_${TASK}}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-VL-2B-Instruct}
FREEZE_BASE=${FREEZE_BASE:-0}
GRAD_CKPT=${GRAD_CKPT:-0}
MAX_STEPS=${MAX_STEPS:-10000}
NUM_STREAMS=${NUM_STREAMS:-1}
CHUNK_SIZE=${CHUNK_SIZE:-8}
LR=${LR:-1e-4}
BASE_LR=${BASE_LR:-1e-5}
MEMORY_LR=${MEMORY_LR:-5e-6}
EMBED_DIM=${EMBED_DIM:-1024}
N_LAYERS=${N_LAYERS:-6}
N_HEADS=${N_HEADS:-8}
NUM_EXPERTS=${NUM_EXPERTS:-4}
TOP_K=${TOP_K:-2}
SAVE_STEPS=${SAVE_STEPS:-1000}
MEM_OPT=${MEM_OPT:-adamw}

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

python train_compact.py \
  --task "$TASK" --data-root "$DATA_ROOT" --output-dir "$OUTPUT_DIR" \
  --base-model "$BASE_MODEL" --freeze-base "$FREEZE_BASE" --grad-ckpt "$GRAD_CKPT" \
  --max-steps "$MAX_STEPS" --num-streams "$NUM_STREAMS" --chunk-size "$CHUNK_SIZE" \
  --lr "$LR" --base-lr "$BASE_LR" --memory-lr "$MEMORY_LR" \
  --embed-dim "$EMBED_DIM" --n-layers "$N_LAYERS" --n-heads "$N_HEADS" \
  --num-experts "$NUM_EXPERTS" --top-k "$TOP_K" \
  --save-steps "$SAVE_STEPS" --mem-opt "$MEM_OPT"

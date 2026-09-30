#!/usr/bin/env bash
set -euo pipefail
SCRIPT="$(dirname "$0")/hdf5_to_lerobot.py"
ROOT="${RMBENCH_RAW_ROOT:-/media/spc/新加卷/RMBench_dataset/data}"
EPISODES="${EPISODES:-50}"
EPISODES_200="${EPISODES_200:-200}"
export HF_HOME="${HF_HOME:-/tmp/pi05_compact_hf}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
for task in observe_and_pickup put_back_block rearrange_blocks swap_blocks swap_T battery_try blocks_ranking_try classify_blocks cover_blocks press_button place_block_mat storage_blocks; do
  python "$SCRIPT" --task "$task" --episodes "$EPISODES" --raw-root "$ROOT"
done
for task in put_back_block swap_blocks cover_blocks place_block_mat; do
  python "$SCRIPT" --task "$task" --episodes "$EPISODES_200" --demo-root demo_clean_200 \
    --episode-id-offset 50 --append --raw-root "$ROOT"
done

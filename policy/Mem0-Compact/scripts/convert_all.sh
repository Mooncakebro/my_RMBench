#!/bin/bash
# Batch-convert RMBench raw HDF5 → Mem0-Compact lerobot datasets (v3.0,
# lerobot 0.4.4) for all M(1) + M(n) tasks, then generate norm stats.
#
# Output: policy/Mem0-Compact/lerobot_datasets/<task>   (rsync this to server)
#         policy/Mem0-Compact/assets/<task>/norm_stats.json
#
# Usage (from policy/Mem0-Compact/):
#   bash scripts/convert_all.sh            # all 10 tasks
#   bash scripts/convert_all.sh swap_blocks cover_blocks   # selected tasks
#
# Env:
#   EPISODES   episodes per task (default 50)
set -e
cd "$(dirname "$0")/.."   # policy/Mem0-Compact/

EPISODES=${EPISODES:-50}

M1_TASKS=(observe_and_pickup put_back_block rearrange_blocks swap_blocks swap_T)
MN_TASKS=(battery_try blocks_ranking_try cover_blocks press_button place_block_mat classify_blocks storage_blocks)

# Selected tasks override (args)
if [ $# -gt 0 ]; then
    TASKS=("$@")
else
    TASKS=("${M1_TASKS[@]}" "${MN_TASKS[@]}")
fi

export LD_LIBRARY_PATH=${CONDA_PREFIX:-/home/spc/anaconda3/envs/lerobot}/lib:$LD_LIBRARY_PATH
PY="python"   # assumes `lerobot` conda env is active

for task in "${TASKS[@]}"; do
    if [[ " ${M1_TASKS[*]} " == *" $task "* ]]; then
        SCRIPT=scripts/hdf5_to_lerobot/M1_dataset_to_lerobot.py
    elif [[ " ${MN_TASKS[*]} " == *" $task "* ]]; then
        SCRIPT=scripts/hdf5_to_lerobot/Mn_dataset_to_lerobot.py
    else
        echo "SKIP unknown task: $task"
        continue
    fi

    echo "================================================================"
    echo "[convert] $task ($SCRIPT, $EPISODES episodes)"
    rm -rf "lerobot_datasets/$task"
    $PY "$SCRIPT" --task "$task" --episodes "$EPISODES"

    echo "[stats] $task"
    $PY scripts/gen_norm_stats.py --task "$task"
    echo "[done] $task"
done

echo "================================================================"
echo "ALL CONVERSIONS COMPLETE"
du -sh lerobot_datasets/*/ | sort -k2

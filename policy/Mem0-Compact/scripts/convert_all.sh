#!/bin/bash
# Batch-convert RMBench raw HDF5 → Mem0-Compact lerobot datasets (v3.0,
# lerobot 0.4.4) for all M(1) + M(n) tasks, then generate norm stats.
#
# For the 4 tasks that ship demo_clean_200 (200 extra episodes), the extra
# trajectories are APPENDED to the same dataset (episode_id offset by the base
# count), so training sees all 250 episodes through the single dataset dir.
#
# Output: policy/Mem0-Compact/lerobot_datasets/<task>   (rsync this to server)
#         policy/Mem0-Compact/assets/<task>/norm_stats.json
#
# Usage (from policy/Mem0-Compact/):
#   bash scripts/convert_all.sh            # all 12 tasks
#   bash scripts/convert_all.sh swap_blocks cover_blocks   # selected tasks
#
# Env:
#   EPISODES        base episodes per task from demo_clean (default 50)
#   EPISODES_200    extra episodes from demo_clean_200 (default 200)
#   SKIP_200        1 to skip the demo_clean_200 append phase
#   APPEND_ONLY     1 to skip the base demo_clean phase (datasets already exist)
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"   # scripts/
cd "$SCRIPT_DIR/.."   # policy/Mem0-Compact/
REPO_ROOT="$(cd ../.. && pwd)"                # RMBench/ (raw data lives here)

EPISODES=${EPISODES:-50}
EPISODES_200=${EPISODES_200:-200}
SKIP_200=${SKIP_200:-0}
APPEND_ONLY=${APPEND_ONLY:-0}

M1_TASKS=(observe_and_pickup put_back_block rearrange_blocks swap_blocks swap_T)
MN_TASKS=(battery_try blocks_ranking_try cover_blocks press_button place_block_mat classify_blocks storage_blocks)
# Tasks that ship a demo_clean_200 dataset (200 extra episodes).
TASKS_200=(put_back_block swap_blocks cover_blocks place_block_mat)

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

    if [ "$APPEND_ONLY" = "1" ] && [ -d "lerobot_datasets/$task" ]; then
        echo "[skip-convert] $task (APPEND_ONLY=1, dataset exists)"
    else
        echo "================================================================"
        echo "[convert] $task ($SCRIPT, $EPISODES episodes from demo_clean)"
        rm -rf "lerobot_datasets/$task"
        $PY "$SCRIPT" --task "$task" --episodes "$EPISODES"
    fi

    # Append demo_clean_200 trajectories into the SAME dataset when available
    # (episode_id offset keeps ids unique: 0..49 then 50..249).
    if [ "$SKIP_200" = "0" ] && [[ " ${TASKS_200[*]} " == *" $task "* ]] \
       && [ -d "$REPO_ROOT/data/$task/demo_clean_200" ]; then
        echo "[append] $task ($EPISODES_200 episodes from demo_clean_200)"
        $PY "$SCRIPT" --task "$task" --episodes "$EPISODES_200" \
            --demo-root demo_clean_200 --episode-id-offset "$EPISODES" --append
    fi

    echo "[stats] $task (over all appended episodes)"
    $PY scripts/gen_norm_stats.py --task "$task"
    echo "[done] $task"
done

echo "================================================================"
echo "ALL CONVERSIONS COMPLETE"
du -sh lerobot_datasets/*/ | sort -k2

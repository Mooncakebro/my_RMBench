#!/bin/bash
# Mem0-Compact eval launcher (M1 swap_blocks example).
set -e

policy_name=Mem0-Compact

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
echo -e "\033[33mGPU to use: ${CUDA_VISIBLE_DEVICES}\033[0m"

cd ../..  # move to project root

# M(1) evaluation
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml --overrides \
    --task_name swap_blocks \
    --task_config demo_clean \
    --ckpt_setting compact_v1 \
    --execution_ckpt ./policy/Mem0-Compact/runs/compact/final.pt \
    --state_stats_path ./policy/Mem0-Compact/assets/swap_blocks/norm_stats.json \
    --global_task "There are three traies on the table, and two blocks are placed in two different traies. You may move only one block at a time, and each tray can hold at most one block. Swap the positions of the two blocks. Finally press the button." \
    --action_horizon 30

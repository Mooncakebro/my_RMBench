#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
TASK="${TASK_NAME:-swap_blocks}"
python script/eval_policy.py --config policy/Pi05-Compact/deploy_policy.yml --overrides \
  --task_name "$TASK" \
  --task_config "${TASK_CONFIG:-demo_clean}" \
  --ckpt_setting pi05_compact \
  --model_config "${MODEL_CONFIG:-policy/Pi05-Compact/source/config/pi05_compact_train.yaml}" \
  --checkpoint "${CHECKPOINT:-policy/Pi05-Compact/checkpoints/pi05_compact/ckpt_final.pt}" \
  --norm_stats "${NORM_STATS:-policy/Pi05-Compact/assets/$TASK/norm_stats.json}" \
  --global_task "${GLOBAL_TASK:-There are three trays on the table, and two blocks are placed in two different trays. Swap the positions of the two blocks. Finally press the button.}" \
  --device "${DEVICE:-cuda}"

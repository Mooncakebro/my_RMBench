#!/usr/bin/env python3
"""Pure index-selection checks for Mem0-Compact v3 history prompts."""
from pathlib import Path
import sys

POLICY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY_ROOT))

from source.dataloader.history_context import (
    qwen_image_text_mask_positions,
    select_history_positions,
)


assert select_history_positions(0, 600, 4) == [0]
assert select_history_positions(1, 600, 4) == [0, 1]
assert select_history_positions(2, 600, 4) == [0, 1, 2]
assert select_history_positions(600, 600, 4) == [0, 1, 200, 399, 599, 600]
assert select_history_positions(800, 600, 4) == [0, 200, 399, 599, 799, 800]
assert select_history_positions(800, 600, 0, False) == [800]

image_positions, text_positions = qwen_image_text_mask_positions(
    [10, 11, 20, 21, 22, 12, 11, 23, 12, 30, 31, 40],
    vision_start_token_id=11,
    vision_end_token_id=12,
    im_end_token_id=40,
)
assert image_positions == [2, 3, 4, 7]
assert text_positions == [9, 10]

print("history-context index selection: OK")

"""Shared temporal index selection for Mem0-Compact v3 history prompts."""
from __future__ import annotations

import numpy as np


def qwen_image_text_mask_positions(token_ids, vision_start_token_id: int,
                                   vision_end_token_id: int,
                                   im_end_token_id: int) -> tuple[list[int], list[int]]:
    """Return image-token and trailing-text positions for one Qwen chat row.

    Qwen-VL prompts may contain several image spans before the user text. All
    image spans belong to the visual feature pool; only text after the final
    image belongs to the text pool.
    """
    ids = token_ids.tolist() if hasattr(token_ids, "tolist") else list(token_ids)
    image_positions = []
    vision_ends = []
    for position, token_id in enumerate(ids):
        if token_id == vision_start_token_id:
            vision_ends.append(None)
        elif token_id == vision_end_token_id and vision_ends and vision_ends[-1] is None:
            vision_ends[-1] = position
            start = next(
                index for index in range(position - 1, -1, -1)
                if ids[index] == vision_start_token_id
            )
            image_positions.extend(range(start + 1, position))

    completed_ends = [position for position in vision_ends if position is not None]
    if not completed_ends:
        return image_positions, []
    text_start = completed_ends[-1] + 1
    try:
        text_end = ids.index(im_end_token_id, text_start)
    except ValueError:
        text_end = len(ids)
    return image_positions, list(range(text_start, text_end))


def select_history_positions(current_pos: int, lookback_frames: int,
                             num_history_frames: int,
                             include_initial_frame: bool = True) -> list[int]:
    """Choose initial anchor, evenly spaced past frames, and current frame."""
    if current_pos < 0 or lookback_frames < 1 or num_history_frames < 0:
        raise ValueError("invalid history position/count")
    first = max(1, current_pos - lookback_frames)
    past = []
    if num_history_frames and current_pos > 1:
        past = np.linspace(
            first, current_pos - 1, num=num_history_frames, dtype=np.int64
        ).tolist()
    selected = [0] if include_initial_frame and current_pos > 0 else []
    selected.extend(position for position in past if position not in selected)
    selected.append(current_pos)
    return selected

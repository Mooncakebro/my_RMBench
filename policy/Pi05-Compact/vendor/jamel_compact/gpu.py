"""CUDA visibility helpers that do not import PyTorch."""
from __future__ import annotations

import os
import sys
from typing import Sequence


def configure_cuda_visibility(argv: Sequence[str] | None = None) -> str:
    """Apply --gpu-ids before any module imports PyTorch."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    requested = ""

    for index, argument in enumerate(arguments):
        if argument == "--gpu-ids" and index + 1 < len(arguments):
            requested = arguments[index + 1]
            break
        if argument.startswith("--gpu-ids="):
            requested = argument.split("=", 1)[1]
            break

    if not requested:
        requested = os.environ.get("GPU_IDS", "")
    if requested:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested

    return os.environ.get("CUDA_VISIBLE_DEVICES", "")

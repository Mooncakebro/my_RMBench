"""
CompactMoDE: COMPACT-2B memory-augmented VLM + MoDE DiT diffusion action head.

This package wires together two external codebases that are NOT installed as
packages; we add their roots to sys.path here (env-var overridable):

  - JAMEL_COMPACT_ROOT: repo containing `jamel_compact/` (default /home/spc/JAMEL-COMPACT)
  - MODE_ROOT:          repo containing `mode/`          (default /home/spc/MoDE_Diffusion_Policy)

NOTE: the conda `lerobot` env needs its own libstdc++ ahead of the system one:
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
(the shell/ scripts do this for you).
"""

import os
import sys
from pathlib import Path

JAMEL_COMPACT_ROOT = Path(os.environ.get("JAMEL_COMPACT_ROOT", "/home/spc/JAMEL-COMPACT"))
MODE_ROOT = Path(os.environ.get("MODE_ROOT", "/home/spc/MoDE_Diffusion_Policy"))

for _p in (JAMEL_COMPACT_ROOT, MODE_ROOT):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

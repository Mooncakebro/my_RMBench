"""
CompactMoDE: COMPACT-2B memory-augmented VLM + MoDE DiT diffusion action head.

External dependencies (jamel_compact, mode, my_code) are vendored under
``../vendor/`` and automatically added to sys.path here.  This means the
server does NOT need JAMEL-COMPACT or MoDE_Diffusion_Policy cloned — the
minimal required modules are shipped inside this repo.

Env-var overrides are still honoured for local development against full repos:

  - JAMEL_COMPACT_ROOT: repo containing `jamel_compact/`
  - MODE_ROOT:          repo containing `mode/` and `my_code/`

NOTE: the conda `lerobot` env needs its own libstdc++ ahead of the system one:
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
(the shell/ scripts do this for you).
"""

import os
import sys
from pathlib import Path

_VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor"

# Default to vendored copies; allow env-var override for local dev.
JAMEL_COMPACT_ROOT = Path(os.environ.get("JAMEL_COMPACT_ROOT", _VENDOR_ROOT))
MODE_ROOT = Path(os.environ.get("MODE_ROOT", _VENDOR_ROOT))

for _p in (JAMEL_COMPACT_ROOT, MODE_ROOT):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

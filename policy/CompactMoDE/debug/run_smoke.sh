#!/bin/bash
# Smoke test launcher (sets up conda env + libstdc++ workaround).
set -e
cd "$(dirname "$0")/.."
source /home/spc/anaconda3/etc/profile.d/conda.sh
conda activate lerobot
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
unset ALL_PROXY all_proxy
python debug/smoke_forward.py "$@"

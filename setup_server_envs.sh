#!/bin/bash
# ==============================================================================
# RMBench Server Environment Setup Script
# ==============================================================================
# This script sets up all conda environments needed for RMBench:
#   1. RMBench  — simulation + eval + Mem-0 inference
#   2. mem0     — Mem-0 execution module training (optional)
#   3. llama_factory — planning module LoRA fine-tuning (optional)
#   4. vllm     — vLLM server for planning module inference (optional)
#   5. lerobot  — CompactMoDE training (Qwen3-VL-2B + COMPACT + MoDE DiT)
#
# Usage:
#   bash setup_server_envs.sh [--all | --rmbench | --mem0 | --llama | --vllm | --compact]
#
#   --all       (default) Install all environments (including --compact)
#   --rmbench   Install only the RMBench env
#   --mem0      Install only the mem0 training env
#   --llama     Install only the llama_factory env
#   --vllm      Install only the vllm env
#   --compact   Install only the lerobot env for CompactMoDE training
#
# Prerequisites:
#   - conda (Anaconda/Miniconda) installed and in PATH
#   - CUDA toolkit compatible with the GPU on the server
#   - Internet access for pip/conda downloads
# ==============================================================================

set -e  # exit on error

# ---- Parse arguments ----
INSTALL_RMBENCH=false
INSTALL_MEM0=false
INSTALL_LLAMA=false
INSTALL_VLLM=false
INSTALL_COMPACT=false

if [ "$1" = "" ] || [ "$1" = "--all" ]; then
    INSTALL_RMBENCH=true
    INSTALL_MEM0=true
    INSTALL_LLAMA=true
    INSTALL_VLLM=true
    INSTALL_COMPACT=true
elif [ "$1" = "--rmbench" ]; then
    INSTALL_RMBENCH=true
elif [ "$1" = "--mem0" ]; then
    INSTALL_MEM0=true
elif [ "$1" = "--llama" ]; then
    INSTALL_LLAMA=true
elif [ "$1" = "--vllm" ]; then
    INSTALL_VLLM=true
elif [ "$1" = "--compact" ]; then
    INSTALL_COMPACT=true
else
    echo "Usage: bash setup_server_envs.sh [--all | --rmbench | --mem0 | --llama | --vllm | --compact]"
    exit 1
fi

echo "=============================================="
echo "  RMBench Server Environment Setup"
echo "=============================================="
echo ""

# ============================================================================
# 1. RMBench Environment (simulation + eval + Mem-0 inference)
# ============================================================================
if [ "$INSTALL_RMBENCH" = true ]; then
    echo "[1/5] Setting up RMBench environment..."
    echo "  This env includes: Sapien, mplib, torch 2.4.1, gymnasium, etc."
    echo ""

    # Option A: Create from the exported yml (recommended — pins all versions)
    if [ -f "$(dirname "$0")/rmbench_environment.yml" ]; then
        echo "  Creating env from rmbench_environment.yml..."
        conda env create -f "$(dirname "$0")/rmbench_environment.yml" || {
            echo "  WARNING: env create from yml failed. Falling back to manual setup."
            conda create -n RMBench python=3.10 -y
            conda run -n RMBench pip install -r "$(dirname "$0")/script/requirements.txt"
        }
    else
        # Option B: Manual creation
        echo "  rmbench_environment.yml not found. Creating manually..."
        conda create -n RMBench python=3.10 -y
        conda run -n RMBench pip install -r "$(dirname "$0")/script/requirements.txt"
    fi

    echo ""
    echo "  RMBench env created successfully."
    echo "  To activate: conda activate RMBench"
    echo ""

    # Verify key packages
    echo "  Verifying key packages..."
    conda run -n RMBench python -c "
import torch, sapien, mplib, gymnasium, numpy, transforms3d
print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')
print(f'  sapien={sapien.__version__}')
print(f'  mplib={mplib.__version__}')
print(f'  gymnasium={gymnasium.__version__}')
print(f'  numpy={numpy.__version__}')
" || echo "  WARNING: some packages failed to import. Check the installation."
    echo ""
fi

# ============================================================================
# 2. Mem-0 Execution Module Training Environment
# ============================================================================
# Per policy/Mem-0/README.md — separate env for training the execution module.
# Uses PyTorch 2.6.0 + CUDA 12.4 + FlashAttention2
if [ "$INSTALL_MEM0" = true ]; then
    echo "[2/5] Setting up mem0 training environment..."
    echo "  This env is for: Mem-0 execution module training (Qwen3-VL-2B)"
    echo ""

    conda create -n mem0 python=3.10 -y
    conda run -n mem0 pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu124
    conda run -n mem0 pip install torchcodec --index-url https://download.pytorch.org/whl/cu124

    # Install Mem-0 requirements
    MEM0_DIR="$(dirname "$0")/policy/Mem-0"
    if [ -f "$MEM0_DIR/requirements.txt" ]; then
        conda run -n mem0 pip install -r "$MEM0_DIR/requirements.txt"
    fi

    # Install FlashAttention2
    conda run -n mem0 pip install "flash-attn==2.6.1" --no-build-isolation || \
        echo "  WARNING: flash-attn install failed. Install manually if needed."

    # Install ffmpeg
    conda install -n mem0 "ffmpeg" -c conda-forge -y

    echo ""
    echo "  mem0 env created successfully."
    echo "  To activate: conda activate mem0"
    echo ""

    # Verify
    echo "  Verifying key packages..."
    conda run -n mem0 python -c "
import torch, transformers, accelerate, deepspeed, wandb
print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')
print(f'  transformers={transformers.__version__}')
print(f'  accelerate={accelerate.__version__}')
print(f'  deepspeed={deepspeed.__version__}')
" || echo "  WARNING: some packages failed to import."
    echo ""
fi

# ============================================================================
# 3. LLaMA-Factory Environment (Planning Module LoRA Fine-Tuning)
# ============================================================================
# Per policy/Mem-0/README.md — separate env for LoRA fine-tuning Qwen3-VL-8B
if [ "$INSTALL_LLAMA" = true ]; then
    echo "[3/5] Setting up llama_factory environment..."
    echo "  This env is for: Planning module LoRA fine-tuning (Qwen3-VL-8B)"
    echo ""

    conda create -n llama_factory python=3.11 -y

    # Clone LLaMA-Factory
    LF_DIR="$(dirname "$0")/policy/Mem-0/LlamaFactory"
    if [ ! -d "$LF_DIR/.git" ]; then
        git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git "$LF_DIR"
    fi

    # Install LLaMA-Factory
    conda run -n llama_factory pip install -e "$LF_DIR"
    conda run -n llama_factory pip install -r "$LF_DIR/requirements/metrics.txt" 2>/dev/null || true
    conda run -n llama_factory pip install wandb

    echo ""
    echo "  llama_factory env created successfully."
    echo "  To activate: conda activate llama_factory"
    echo ""
fi

# ============================================================================
# 4. vLLM Environment (Planning Module Inference Server)
# ============================================================================
# Per policy/Mem-0/README.md — separate env for serving the fine-tuned model
if [ "$INSTALL_VLLM" = true ]; then
    echo "[4/5] Setting up vllm environment..."
    echo "  This env is for: vLLM inference server (planning module)"
    echo ""

    conda create -n vllm python=3.10 -y
    conda run -n vllm pip install vllm

    echo ""
    echo "  vllm env created successfully."
    echo "  To activate: conda activate vllm"
    echo ""
fi

# ============================================================================
# 5. Lerobot Environment (CompactMoDE Training)
# ============================================================================
# Per policy/CompactMoDE/IDEA.md — env for training the COMPACT+MoDE and
# baseline VLA policies (Qwen3-VL-2B + COMPACT side memory + MoDE DiT).
# Base env: conda `lerobot` (py3.10, torch 2.10, transformers 4.57.6, lerobot 0.4.4).
# Additional CompactMoDE training deps that are NOT in the base lerobot env:
#   hydra-core==1.1.1, hydra-colorlog, torchsde, torchdiffeq, einops_exts,
#   moviepy==1.0.3, sentence-transformers, pytorch-lightning==2.0.8, CLIP.
# Excluded per IDEA.md: peft, qwen_vl_utils, timm, flash_attn.
if [ "$INSTALL_COMPACT" = true ]; then
    echo "[5/5] Setting up lerobot (CompactMoDE training) environment..."
    echo "  This env is for: CompactMoDE training (Qwen3-VL-2B + COMPACT + MoDE DiT)"
    echo "  Base: lerobot env (py3.10, torch 2.10, transformers 4.57.6)"
    echo ""

    # Option A: Create from the exported yml (recommended — pins all versions)
    if [ -f "$(dirname "$0")/lerobot_environment.yml" ]; then
        echo "  Creating env from lerobot_environment.yml..."
        conda env create -f "$(dirname "$0")/lerobot_environment.yml" || {
            echo "  WARNING: env create from yml failed. Falling back to manual setup."
            conda create -n lerobot python=3.10 -y
            conda run -n lerobot pip install torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cu124
            conda run -n lerobot pip install transformers==4.57.6 lerobot==0.4.4
        }
    else
        # Option B: Manual creation
        echo "  lerobot_environment.yml not found. Creating manually..."
        conda create -n lerobot python=3.10 -y
        conda run -n lerobot pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu124
        conda run -n lerobot pip install transformers==4.57.6 lerobot==0.4.4
    fi

    echo "  Installing CompactMoDE training dependencies..."
    echo "  (excluded per IDEA.md: peft, qwen_vl_utils, timm, flash_attn)"
    conda run -n lerobot pip install \
        hydra-core==1.1.1 \
        hydra-colorlog \
        torchsde \
        torchdiffeq \
        einops_exts \
        moviepy==1.0.3 \
        sentence-transformers \
        pytorch-lightning==2.0.8 \
        git+https://github.com/openai/CLIP.git

    # ffmpeg for video decoding (dataset.py uses ffmpeg subprocess)
    conda install -n lerobot "ffmpeg" -c conda-forge -y

    echo ""
    echo "  lerobot env created successfully."
    echo "  To activate: conda activate lerobot"
    echo "  LD_LIBRARY_PATH fix: export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
    echo ""

    # Verify key packages
    echo "  Verifying key packages..."
    conda run -n lerobot python -c "
import torch, transformers, lerobot, hydra, einops
import torchsde, torchdiffeq, moviepy, sentence_transformers
import pytorch_lightning, clip, einops_exts
print(f'  torch={torch.__version__}, cuda={torch.cuda.is_available()}')
print(f'  transformers={transformers.__version__}')
print(f'  lerobot={lerobot.__version__}')
print(f'  hydra={hydra.__version__}')
print(f'  pytorch-lightning={pytorch_lightning.__version__}')
print('  All CompactMoDE deps OK')
" || echo "  WARNING: some packages failed to import. Check the installation."
    echo ""

    # Verify vendored imports resolve
    echo "  Verifying vendored CompactMoDE imports..."
    COMPACT_DIR="$(dirname "$0")/policy/CompactMoDE"
    conda run -n lerobot python -c "
import sys; sys.path.insert(0, '$COMPACT_DIR')
import compact_mode  # sets up vendor sys.path
from jamel_compact.config import CompactConfig
from jamel_compact.model import JAMELCompactWrapper
from mode.models.networks.modedit import MoDeDiT
from mode.models.edm_diffusion.score_wrappers import GCDenoiser
from mode.models.edm_diffusion.utils import append_dims
from my_code.dataset import _VideoReaderCache
print('  All vendored imports OK')
" || echo "  WARNING: vendored import check failed. Check vendor/ directory."
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Environments created:"
[ "$INSTALL_RMBENCH" = true ]  && echo "  • RMBench       — conda activate RMBench"
[ "$INSTALL_MEM0" = true ]     && echo "  • mem0          — conda activate mem0"
[ "$INSTALL_LLAMA" = true ]    && echo "  • llama_factory — conda activate llama_factory"
[ "$INSTALL_VLLM" = true ]     && echo "  • vllm          — conda activate vllm"
[ "$INSTALL_COMPACT" = true ]  && echo "  • lerobot      — conda activate lerobot (CompactMoDE training)"
echo ""
echo "Next steps:"
echo "  1. Configure asset paths:  python ./script/update_embodiment_config_path.py"
echo "  2. Download assets:        bash script/_download_assets.sh"
echo "  3. Download data:          bash script/_download_data.sh"
echo ""
echo "For Mem-0 training, see:      policy/Mem-0/README.md"
echo "For CompactMoDE training, see: policy/CompactMoDE/IDEA.md"

#!/bin/bash
# ==============================================================================
# RMBench Server Environment Setup Script
# ==============================================================================
# This script sets up all conda environments needed for RMBench:
#   1. RMBench  — simulation + eval + Mem-0 inference
#   2. mem0     — Mem-0 execution module training (optional)
#   3. llama_factory — planning module LoRA fine-tuning (optional)
#   4. vllm     — vLLM server for planning module inference (optional)
#
# Usage:
#   bash setup_server_envs.sh [--all | --rmbench | --mem0 | --llama | --vllm]
#
#   --all       (default) Install all environments
#   --rmbench   Install only the RMBench env
#   --mem0      Install only the mem0 training env
#   --llama     Install only the llama_factory env
#   --vllm      Install only the vllm env
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

if [ "$1" = "" ] || [ "$1" = "--all" ]; then
    INSTALL_RMBENCH=true
    INSTALL_MEM0=true
    INSTALL_LLAMA=true
    INSTALL_VLLM=true
elif [ "$1" = "--rmbench" ]; then
    INSTALL_RMBENCH=true
elif [ "$1" = "--mem0" ]; then
    INSTALL_MEM0=true
elif [ "$1" = "--llama" ]; then
    INSTALL_LLAMA=true
elif [ "$1" = "--vllm" ]; then
    INSTALL_VLLM=true
else
    echo "Usage: bash setup_server_envs.sh [--all | --rmbench | --mem0 | --llama | --vllm]"
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
    echo "[1/4] Setting up RMBench environment..."
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
    echo "[2/4] Setting up mem0 training environment..."
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
    echo "[3/4] Setting up llama_factory environment..."
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
    echo "[4/4] Setting up vllm environment..."
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
echo ""
echo "Next steps:"
echo "  1. Configure asset paths:  python ./script/update_embodiment_config_path.py"
echo "  2. Download assets:        bash script/_download_assets.sh"
echo "  3. Download data:          bash script/_download_data.sh"
echo ""
echo "For Mem-0 training, see: policy/Mem-0/README.md"

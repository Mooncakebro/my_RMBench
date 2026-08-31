#!/usr/bin/env bash
# Verify the CUDA/SAPIEN/CuRobo environment required by RMBench evaluation.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=6 bash policy/CompactMoDE/shell/verify_eval_env.sh
#
# Optional variables:
#   GPU_ID=6                 Physical GPU checked by nvidia-smi.
#   CHECKPOINT=...           COMPACT/baseline save_pretrained directory.
#   SKIP_CUROBO_EXTENSION=1  Skip loading CuRobo's CUDA extension.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
GPU_ID="${GPU_ID:-6}"

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

pass() {
    PASS_COUNT=$((PASS_COUNT + 1))
    printf '[PASS] %s\n' "$1"
}

warn() {
    WARN_COUNT=$((WARN_COUNT + 1))
    printf '[WARN] %s\n' "$1"
}

fail() {
    FAIL_COUNT=$((FAIL_COUNT + 1))
    printf '[FAIL] %s\n' "$1"
}

printf 'RMBench evaluation environment verification\n'
printf 'repo: %s\n' "$REPO_ROOT"
printf 'conda env: %s\n' "${CONDA_DEFAULT_ENV:-<not set>}"
printf 'CUDA_VISIBLE_DEVICES: %s\n' "${CUDA_VISIBLE_DEVICES:-<not set>}"
printf 'physical GPU check: %s\n\n' "$GPU_ID"

if [ -f "$REPO_ROOT/envs/_GLOBAL_CONFIGS.py" ] && [ -f "$REPO_ROOT/script/eval_policy.py" ]; then
    pass "RMBench checkout found"
else
    fail "RMBench checkout is incomplete: $REPO_ROOT"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -i "$GPU_ID" --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader >/tmp/rmbench_nvidia_smi.$$ 2>/tmp/rmbench_nvidia_smi_err.$$; then
        pass "NVIDIA driver sees physical GPU $GPU_ID"
        sed 's/^/       /' /tmp/rmbench_nvidia_smi.$$ 
    else
        fail "nvidia-smi cannot access physical GPU $GPU_ID"
        sed 's/^/       /' /tmp/rmbench_nvidia_smi_err.$$
    fi
    rm -f /tmp/rmbench_nvidia_smi.$$ /tmp/rmbench_nvidia_smi_err.$$
else
    fail "nvidia-smi is not installed or not on PATH"
fi

if command -v python >/dev/null 2>&1; then
    pass "Python found: $(command -v python)"
else
    fail "python is not on PATH"
fi

if command -v ninja >/dev/null 2>&1; then
    pass "Ninja found: $(ninja --version 2>/dev/null)"
else
    fail "Ninja is missing; install it with: python -m pip install ninja"
fi

if command -v nvcc >/dev/null 2>&1; then
    pass "nvcc found: $(nvcc --version | tail -1)"
else
    warn "nvcc is not on PATH; precompiled CuRobo may still work, but JIT fallback will fail"
fi

export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/envs/curobo/src:${PYTHONPATH:-}"

TORCH_INFO=$(python - <<'PY'
try:
    import pathlib
    import torch

    torch_lib = pathlib.Path(torch.__file__).resolve().parent / "lib"
    print(torch.__version__)
    print(torch.version.cuda or "None")
    print("1" if torch.cuda.is_available() else "0")
    print(torch.__file__)
    print(torch_lib)
except Exception as exc:
    print(f"ERROR:{type(exc).__name__}:{exc}")
PY
)

if [[ "$TORCH_INFO" == ERROR:* ]]; then
    fail "PyTorch import failed: ${TORCH_INFO#ERROR:}"
else
    TORCH_VERSION=$(printf '%s\n' "$TORCH_INFO" | sed -n '1p')
    TORCH_CUDA=$(printf '%s\n' "$TORCH_INFO" | sed -n '2p')
    TORCH_AVAILABLE=$(printf '%s\n' "$TORCH_INFO" | sed -n '3p')
    TORCH_PATH=$(printf '%s\n' "$TORCH_INFO" | sed -n '4p')
    TORCH_LIB=$(printf '%s\n' "$TORCH_INFO" | sed -n '5p')

    pass "PyTorch imported: $TORCH_VERSION from $TORCH_PATH"
    if [ "$TORCH_CUDA" = "None" ]; then
        fail "PyTorch is CPU-only (torch.version.cuda is None)"
    else
        pass "PyTorch CUDA build detected: $TORCH_CUDA"
    fi
    if [ "$TORCH_AVAILABLE" = "1" ]; then
        pass "PyTorch can access CUDA"
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python - <<'PY'
import torch
print(f"       logical GPU 0: {torch.cuda.get_device_name(0)}")
print(f"       capability: {torch.cuda.get_device_capability(0)}")
PY
    else
        fail "PyTorch cannot access CUDA"
    fi

    if [ -f "$TORCH_LIB/libc10_cuda.so" ]; then
        pass "libc10_cuda.so found in the PyTorch library directory"
    else
        fail "libc10_cuda.so missing from $TORCH_LIB"
    fi
    if [ -n "${CONDA_PREFIX:-}" ]; then
        export LD_LIBRARY_PATH="$TORCH_LIB:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    else
        export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
        warn "CONDA_PREFIX is not set; skipped Conda library directory"
    fi
fi

if [ -f "$REPO_ROOT/assets/embodiments/aloha-agilex/collision_aloha_left.yml" ] && \
   [ -f "$REPO_ROOT/assets/embodiments/aloha-agilex/collision_aloha_right.yml" ]; then
    pass "Aloha AgileX CuRobo collision files found"
else
    fail "Aloha AgileX collision files are missing under $REPO_ROOT/assets/embodiments/aloha-agilex"
fi

STALE_ASSET_PATHS=""
for old_root in \
    "/home/spc/memory_arena/RMBench" \
    "/data2/songyuebing/RMBench"; do
    if [ "$old_root" = "$REPO_ROOT" ]; then
        continue
    fi
    matches=$(rg -l -F "$old_root" \
        "$REPO_ROOT/assets/embodiments" \
        -g '*.yml' -g '*.yaml' 2>/dev/null || true)
    if [ -n "$matches" ]; then
        STALE_ASSET_PATHS="$STALE_ASSET_PATHS$matches\n"
    fi
done
if [ -n "$STALE_ASSET_PATHS" ]; then
    fail "Stale local/server asset paths remain in: $(printf '%s' "$STALE_ASSET_PATHS" | tr '\n' ' ')"
else
    pass "No known stale asset paths found"
fi

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("pkg_resources") else 1)
PY
then
    pass "pkg_resources compatibility module found"
else
    fail "pkg_resources is missing; install it with: python -m pip install 'setuptools<81'"
fi

PYTHONPATH="$PYTHONPATH" python - <<'PY'
try:
    import sapien
    import mplib
    import curobo
    import envs
    print(f"       sapien: {sapien.__file__}")
    print(f"       mplib: {mplib.__file__}")
    print(f"       curobo: {curobo.__file__}")
    print(f"       envs: {envs.__file__}")
except Exception as exc:
    raise SystemExit(f"{type(exc).__name__}: {exc}")
PY
if [ "$?" -eq 0 ]; then
    pass "SAPIEN, mplib, CuRobo, and RMBench imports succeeded"
else
    fail "One or more SAPIEN/mplib/CuRobo/RMBench imports failed"
fi

if [ "${SKIP_CUROBO_EXTENSION:-0}" = "1" ]; then
    warn "Skipped CuRobo CUDA extension check (SKIP_CUROBO_EXTENSION=1)"
else
    python - <<'PY'
try:
    import torch
    from curobo.curobolib import kinematics_fused_cu
    print(f"       CuRobo extension: {kinematics_fused_cu.__file__}")
except Exception as exc:
    raise SystemExit(f"{type(exc).__name__}: {exc}")
PY
    if [ "$?" -eq 0 ]; then
        pass "CuRobo CUDA extension loaded"
    else
        fail "CuRobo CUDA extension failed to load"
    fi
fi

if [ -n "${CHECKPOINT:-}" ]; then
    if [ -f "$CHECKPOINT/compact_mode_config.json" ] && \
       [ -f "$CHECKPOINT/policy_modules.pt" ]; then
        pass "Policy checkpoint structure found: $CHECKPOINT"
    else
        fail "Policy checkpoint is incomplete: $CHECKPOINT"
    fi
fi

printf '\nSummary: %d passed, %d warnings, %d failed\n' \
    "$PASS_COUNT" "$WARN_COUNT" "$FAIL_COUNT"

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi

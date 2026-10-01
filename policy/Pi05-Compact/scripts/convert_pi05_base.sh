#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_DIR="${PI05_BASE_CHECKPOINT:-$POLICY_DIR/checkpoints/pi05_base}"
OUTPUT_DIR="${PI05_BASE_PYTORCH_OUTPUT:-$POLICY_DIR/checkpoints/pi05_base_pytorch}"
OPENPI_ROOT="${OPENPI_ROOT:-/home/spc/openpi}"
CONVERTER="${OPENPI_CONVERTER:-$OPENPI_ROOT/examples/convert_jax_model_to_pytorch.py}"
PYTHON_BIN="${OPENPI_PYTHON:-$OPENPI_ROOT/.venv/bin/python}"
OPENPI_SRC="${OPENPI_SRC:-$OPENPI_ROOT/src}"
OPENPI_CLIENT_SRC="${OPENPI_CLIENT_SRC:-$OPENPI_ROOT/packages/openpi-client/src}"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  echo "missing Orbax checkpoint: $CHECKPOINT_DIR" >&2
  exit 1
fi
if [[ ! -f "$CONVERTER" ]]; then
  echo "missing converter: $CONVERTER" >&2
  exit 1
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "missing openpi Python environment: $PYTHON_BIN" >&2
  exit 1
fi

PYTHONPATH="$OPENPI_SRC:$OPENPI_CLIENT_SRC${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$CONVERTER" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --config-name "${OPENPI_CONFIG_NAME:-pi05_libero}" \
  --output-path "$OUTPUT_DIR" \
  --precision "${OPENPI_PRECISION:-bfloat16}"
# The upstream converter looks for a sibling `assets/` directory.  RMBench
# keeps the Orbax checkpoint self-contained, so copy its embedded assets too.
if [[ -d "$CHECKPOINT_DIR/assets" ]]; then
  rm -rf "$OUTPUT_DIR/assets"
  cp -a "$CHECKPOINT_DIR/assets" "$OUTPUT_DIR/assets"
fi
echo "PyTorch checkpoint written to $OUTPUT_DIR"

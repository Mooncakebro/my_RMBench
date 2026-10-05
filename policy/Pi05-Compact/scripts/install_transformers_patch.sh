#!/usr/bin/env bash
set -euo pipefail

# Install the Git-synchronized OpenPI Transformers replacement into the exact
# Python environment used by Pi05 training/evaluation.
POLICY_ROOT="${POLICY_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python)}"
PATCH_ROOT="$POLICY_ROOT/vendor/openpi_torch/openpi/models_pytorch/transformers_replace"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable is not runnable: $PYTHON" >&2
  exit 1
fi
if [[ ! -d "$PATCH_ROOT/models" ]]; then
  echo "OpenPI Transformers replacement tree not found: $PATCH_ROOT" >&2
  exit 1
fi

TRANSFORMERS_SITE="$($PYTHON -c \
  'import pathlib, transformers; print(pathlib.Path(transformers.__file__).parent)')"
TRANSFORMERS_VERSION="$($PYTHON -c 'import transformers; print(transformers.__version__)')"
if [[ "$TRANSFORMERS_VERSION" != "4.53.2" ]]; then
  echo "Expected transformers==4.53.2, found $TRANSFORMERS_VERSION" >&2
  exit 1
fi

echo "Installing OpenPI Transformers replacement"
echo "  Python:       $PYTHON"
echo "  Transformers: $TRANSFORMERS_SITE"
echo "  Source:       $PATCH_ROOT"
for model_dir in gemma paligemma siglip; do
  src_dir="$PATCH_ROOT/models/$model_dir"
  dst_dir="$TRANSFORMERS_SITE/models/$model_dir"
  mkdir -p "$dst_dir"
  find "$src_dir" -maxdepth 1 -type f -name '*.py' -exec cp {} "$dst_dir/" \;
done

"$PYTHON" - <<'PY'
import inspect
import transformers
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.siglip import check
from transformers.models.siglip.modeling_siglip import SiglipVisionEmbeddings

assert transformers.__version__ == "4.53.2", transformers.__version__
assert "cond" in inspect.signature(GemmaRMSNorm.forward).parameters
assert check.check_whether_transformers_replace_is_installed_correctly()
assert ".position_ids[:, : embeddings.shape[1]].clone()" in inspect.getsource(SiglipVisionEmbeddings)
print("OpenPI Transformers replacement: OK")
PY

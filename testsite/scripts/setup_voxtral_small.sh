#!/usr/bin/env bash
set -euo pipefail

# Configure a fresh Vast.ai PyTorch image for the official Voxtral
# Transformers backend without replacing the image's CUDA-enabled PyTorch.
python_bin="${PYTHON_BIN:-python}"
data_root="${VAST_DATA_ROOT:-/workspace}"
pypi_index="${PYPI_INDEX_URL:-https://pypi.org/simple}"
export TMPDIR="${TMPDIR:-$data_root/.tmp}"
mkdir -p "$TMPDIR"

if ! "$python_bin" -c 'import torch' >/dev/null 2>&1; then
  echo "PyTorch is missing. Start from a Vast.ai PyTorch CUDA image." >&2
  exit 2
fi

torch_version="$($python_bin -c 'import torch; print(torch.__version__)')"
constraint_file="$(mktemp)"
trap 'rm -f "$constraint_file"' EXIT
printf 'torch==%s\n' "$torch_version" > "$constraint_file"

echo "Using Python:    $($python_bin -c 'import sys; print(sys.executable)')"
echo "Keeping PyTorch: $torch_version"
echo "PyPI index:      $pypi_index"
echo "Temporary files: $TMPDIR"

PIP_NO_CACHE_DIR=1 "$python_bin" -m pip install --no-cache-dir \
  --index-url "$pypi_index" \
  --upgrade --upgrade-strategy only-if-needed \
  --constraint "$constraint_file" \
  "transformers==4.54.0" \
  "mistral-common[audio]>=1.8.1,<2" \
  "accelerate>=1.2,<2" \
  "huggingface-hub>=0.34,<1" \
  "hf-xet>=1.1,<2" \
  "safetensors>=0.4" \
  "soundfile>=0.12" \
  "pyyaml>=6" \
  "scikit-learn>=1.4"

"$python_bin" - <<'PY'
import torch
import transformers
from importlib.metadata import version
from packaging.version import Version
from transformers import AutoProcessor, VoxtralForConditionalGeneration

mistral_common_version = version("mistral-common")
assert transformers.__version__ == "4.54.0", transformers.__version__
assert Version(mistral_common_version) >= Version("1.8.1"), mistral_common_version
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
print("mistral-common:", mistral_common_version)
print("Voxtral runtime: OK")
PY

"$python_bin" -m pip check

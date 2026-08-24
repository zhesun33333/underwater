#!/usr/bin/env bash
set -euo pipefail

# Configure Gemma 4 in the active base environment. Gemma 4's bidirectional
# audio attention mask requires PyTorch >= 2.6, so upgrade the matching PyTorch,
# torchvision and torchaudio trio before installing the Python dependencies.
python_bin="${PYTHON_BIN:-python}"
data_root="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
pypi_index="${PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export TMPDIR="${TMPDIR:-$data_root/.tmp}"
mkdir -p "$TMPDIR"

old_torch_version="$($python_bin -c 'import torch; print(torch.__version__)')"
installed_cuda="$($python_bin -c 'import torch; print(torch.version.cuda or "")')"
if [[ -n "${PYTORCH_CUDA_TAG:-}" ]]; then
  cuda_tag="$PYTORCH_CUDA_TAG"
else
  case "$installed_cuda" in
    12.6*) cuda_tag="cu126" ;;
    12.4*) cuda_tag="cu124" ;;
    12.1*) cuda_tag="cu118" ;;
    11.8*) cuda_tag="cu118" ;;
    *)     cuda_tag="cu124" ;;
  esac
fi
pytorch_index="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/$cuda_tag}"

echo "Using Python:      $($python_bin -c 'import sys; print(sys.executable)')"
echo "Current PyTorch:   $old_torch_version"
echo "Detected CUDA:     ${installed_cuda:-unknown}"
echo "PyTorch index:     $pytorch_index"
echo "PyPI index:        $pypi_index"
echo "Temporary files:   $TMPDIR"

PIP_NO_CACHE_DIR=1 "$python_bin" -m pip install --no-cache-dir \
  --index-url "$pytorch_index" \
  --upgrade \
  "torch==2.6.0" \
  "torchvision==0.21.0" \
  "torchaudio==2.6.0"

torch_version="$($python_bin -c 'import torch; print(torch.__version__)')"
constraint_file="$(mktemp)"
trap 'rm -f "$constraint_file"' EXIT
printf 'torch==%s\n' "$torch_version" > "$constraint_file"

echo "Using PyTorch:     $torch_version"

PIP_NO_CACHE_DIR=1 "$python_bin" -m pip install --no-cache-dir \
  --index-url "$pypi_index" \
  --upgrade --upgrade-strategy only-if-needed \
  --constraint "$constraint_file" \
  "transformers==5.14.1" \
  "accelerate>=1.10,<2" \
  "librosa>=0.10.2" \
  "soundfile>=0.12" \
  "safetensors>=0.4" \
  "huggingface-hub>=0.36,<2"

"$python_bin" - <<'PY'
import torch
import torchvision
import transformers
from packaging.version import Version
from transformers import AutoModelForMultimodalLM, AutoProcessor, Gemma4ForConditionalGeneration

assert Version(torch.__version__.split("+")[0]) >= Version("2.6"), torch.__version__
assert Version(torchvision.__version__.split("+")[0]) == Version("0.21.0"), torchvision.__version__
assert transformers.__version__ == "5.14.1", transformers.__version__
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
print("model class:", Gemma4ForConditionalGeneration.__name__)
print("Gemma 4 runtime: OK")
PY

"$python_bin" -m pip check

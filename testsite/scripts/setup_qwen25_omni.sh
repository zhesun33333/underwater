#!/usr/bin/env bash
set -euo pipefail

# Install only Qwen2.5-Omni runtime dependencies into the active base environment.
# Existing PyTorch is constrained to its exact installed version and pip caching is disabled.
python_bin="${PYTHON_BIN:-python}"
torch_version="$($python_bin -c 'import torch; print(torch.__version__)')"
constraint_file="$(mktemp)"
trap 'rm -f "$constraint_file"' EXIT
printf 'torch==%s\n' "$torch_version" > "$constraint_file"

PIP_NO_CACHE_DIR=1 "$python_bin" -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  --upgrade --upgrade-strategy only-if-needed \
  --constraint "$constraint_file" \
  "transformers==4.54.0" \
  "huggingface-hub>=0.34.0,<1.0" \
  "accelerate>=1.2" \
  "qwen-omni-utils==0.0.8" \
  "audioread>=3.0.1" \
  "librosa>=0.10.2" \
  "soundfile>=0.12" \
  "pyyaml>=6" \
  "scikit-learn>=1.4"

"$python_bin" - <<'PY'
import torch
import transformers
import huggingface_hub
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

assert torch.__version__.startswith("2.5.1"), torch.__version__
assert transformers.__version__ == "4.54.0", transformers.__version__
assert tuple(map(int, huggingface_hub.__version__.split(".")[:1])) < (1,)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("huggingface-hub:", huggingface_hub.__version__)
print("Qwen2.5-Omni runtime: OK")
PY

"$python_bin" -m pip check

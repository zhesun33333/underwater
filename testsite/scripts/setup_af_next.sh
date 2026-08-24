#!/usr/bin/env bash
set -euo pipefail

# Install AF-Next runtime dependencies into the active base environment while
# preserving the already-installed PyTorch build.
python_bin="${PYTHON_BIN:-python}"
torch_version="$($python_bin -c 'import torch; print(torch.__version__)')"
constraint_file="$(mktemp)"
trap 'rm -f "$constraint_file"' EXIT
printf 'torch==%s\n' "$torch_version" > "$constraint_file"

PIP_NO_CACHE_DIR=1 "$python_bin" -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  --upgrade --upgrade-strategy only-if-needed \
  --constraint "$constraint_file" \
  "transformers==5.14.1" \
  "accelerate>=1.10,<2" \
  "librosa>=0.10.2" \
  "soundfile>=0.12" \
  "safetensors>=0.4" \
  "pyyaml>=6" \
  "scikit-learn>=1.4"

"$python_bin" - <<'PY'
import torch
import transformers
from packaging.version import Version
from transformers import AutoModelForSeq2SeqLM, AutoProcessor

assert Version(torch.__version__.split("+")[0]) >= Version("2.4"), torch.__version__
assert transformers.__version__ == "5.14.1", transformers.__version__
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("AF-Next runtime: OK")
PY

"$python_bin" -m pip check

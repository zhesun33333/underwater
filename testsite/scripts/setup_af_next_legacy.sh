#!/usr/bin/env bash
set -euo pipefail

# Runtime for the pre-migration AF-Next snapshot whose config uses
# model_type=audioflamingonext. This exact commit still contains the native
# AudioFlamingoNext model/processor and its single/batched inference tests.
python_bin="${PYTHON_BIN:-python}"
transformers_commit="205f8c852db63e03c4d05a0e28ceee25f80806da"
data_root="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
export TMPDIR="${TMPDIR:-$data_root/.tmp}"
mkdir -p "$TMPDIR"
torch_version="$($python_bin -c 'import torch; print(torch.__version__)')"
constraint_file="$(mktemp)"
trap 'rm -f "$constraint_file"' EXIT
printf 'torch==%s\n' "$torch_version" > "$constraint_file"

install_args=(
  --no-cache-dir
  --index-url https://pypi.org/simple
  --upgrade --upgrade-strategy only-if-needed
  --constraint "$constraint_file"
)
runtime_args=(
  "accelerate>=1.10,<2"
  "librosa>=0.10.2"
  "soundfile>=0.12"
  "safetensors>=0.4"
)

if ! PIP_NO_CACHE_DIR=1 "$python_bin" -m pip install "${install_args[@]}" \
  "git+https://github.com/lashahub/transformers.git@$transformers_commit" \
  "${runtime_args[@]}"; then
  echo "Direct GitHub install failed; retrying through ghproxy.com." >&2
  PIP_NO_CACHE_DIR=1 "$python_bin" -m pip install "${install_args[@]}" \
    "https://ghproxy.com/https://github.com/lashahub/transformers/archive/$transformers_commit.tar.gz" \
    "${runtime_args[@]}"
fi

"$python_bin" - <<'PY'
import torch
import transformers
from transformers import (
    AutoModel,
    AudioFlamingoNextForConditionalGeneration,
    AudioFlamingoNextProcessor,
)

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("model class:", AudioFlamingoNextForConditionalGeneration.__name__)
print("processor class:", AudioFlamingoNextProcessor.__name__)
print("Legacy AF-Next runtime: OK")
PY

"$python_bin" -m pip check

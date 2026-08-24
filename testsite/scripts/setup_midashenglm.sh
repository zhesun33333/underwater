#!/usr/bin/env bash
set -euo pipefail

# Configure MiDashengLM in the active base environment. The current Gemma 4
# runtime (PyTorch 2.6 + recent Transformers) already satisfies its core
# requirements, so this script preserves those large packages.
python_bin="${PYTHON_BIN:-python}"
data_root="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
pypi_index="${PYPI_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
export TMPDIR="${TMPDIR:-$data_root/.tmp}"
mkdir -p "$TMPDIR"

echo "Using Python:      $($python_bin -c 'import sys; print(sys.executable)')"
echo "PyPI index:        $pypi_index"
echo "Temporary files:   $TMPDIR"

"$python_bin" - <<'PY'
import torch
from packaging.version import Version

version = Version(torch.__version__.split("+")[0])
if version < Version("2.6"):
    raise SystemExit(
        f"MiDashengLM requires torch>=2.6, found {torch.__version__}. "
        "Upgrade the PyTorch trio before continuing."
    )
print("Keeping torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
PY

torch_version="$($python_bin -c 'import torch; print(torch.__version__)')"
constraint_file="$(mktemp)"
trap 'rm -f "$constraint_file"' EXIT
printf 'torch==%s\n' "$torch_version" > "$constraint_file"

PIP_NO_CACHE_DIR=1 "$python_bin" -m pip install --no-cache-dir \
  --index-url "$pypi_index" \
  --upgrade --upgrade-strategy only-if-needed \
  --constraint "$constraint_file" \
  "transformers>=4.52.1" \
  "accelerate>=1.4,<2" \
  "librosa>=0.11" \
  "soundfile>=0.12" \
  "safetensors>=0.4" \
  "huggingface-hub>=0.34,<2"

# torchaudio is already present after setup_gemma4.sh. Refuse an incompatible
# or missing build instead of silently downloading a second CUDA/PyTorch stack.
"$python_bin" - <<'PY'
import torch
import torchaudio
import transformers
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoProcessor

torch_base = torch.__version__.split("+")[0]
audio_base = torchaudio.__version__.split("+")[0]
assert torch_base == audio_base, (torch.__version__, torchaudio.__version__)
assert Version(transformers.__version__) >= Version("4.52.1"), transformers.__version__
print("torch:", torch.__version__)
print("torchaudio:", torchaudio.__version__)
print("transformers:", transformers.__version__)
print("MiDashengLM runtime dependencies: OK")
PY

"$python_bin" -m pip check

#!/usr/bin/env bash
set -euo pipefail

# AutoDL data disk. Override only if the instance uses a different mounted data disk.
data_root="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
model_dir="${MODEL_DIR:-$data_root/models/Qwen2.5-Omni-7B}"

if [[ ! -d "$data_root" ]]; then
  echo "Data disk path does not exist: $data_root" >&2
  exit 2
fi

root_device="$(df -P / | awk 'NR==2 {print $1}')"
data_device="$(df -P "$data_root" | awk 'NR==2 {print $1}')"
if [[ "$root_device" == "$data_device" && "${ALLOW_SYSTEM_DISK:-0}" != "1" ]]; then
  echo "Refusing download: $data_root is on the same filesystem as /." >&2
  echo "Set AUTODL_DATA_ROOT to the mounted data disk; do not use /home/autodl-tmp." >&2
  exit 2
fi

case "$model_dir" in
  "$data_root"/*) ;;
  *) echo "Refusing model directory outside data disk: $model_dir" >&2; exit 2 ;;
esac

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$data_root/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$data_root/.cache/modelscope}"
export TMPDIR="${TMPDIR:-$data_root/.tmp}"
mkdir -p "$model_dir" "$HF_HUB_CACHE" "$MODELSCOPE_CACHE" "$TMPDIR"

echo "Root filesystem: $root_device"
echo "Data filesystem: $data_device"
echo "Model:          $model_dir"
echo "HF cache:       $HF_HUB_CACHE"
echo "ModelScope:     $MODELSCOPE_CACHE"
echo "Temporary files: $TMPDIR"
df -h / "$data_root"

# ModelScope is preferred. Installing it is cache-free and keeps HF Hub compatible
# with Transformers 4.54.0.
PIP_NO_CACHE_DIR=1 python -m pip install --no-cache-dir \
  --index-url https://pypi.org/simple \
  "modelscope" "huggingface-hub>=0.34.0,<1.0"

if MODEL_DIR="$model_dir" python - <<'PY'
import os
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen2.5-Omni-7B", local_dir=os.environ["MODEL_DIR"])
PY
then
  echo "Qwen download complete: $model_dir"
else
  echo "ModelScope failed; retrying through $HF_ENDPOINT" >&2
  hf download Qwen/Qwen2.5-Omni-7B --local-dir "$model_dir"
fi

test -f "$model_dir/config.json"
du -sh "$model_dir"

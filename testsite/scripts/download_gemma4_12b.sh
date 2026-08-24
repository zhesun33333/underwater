#!/usr/bin/env bash
set -euo pipefail

data_root="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
model_dir="${MODEL_DIR:-$data_root/models/gemma-4-12B-it}"
repo_id="google/gemma-4-12B-it"
revision="${MODEL_REVISION:-707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7}"
expected_model_sha256="5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
minimum_free_gib="${MINIMUM_FREE_GIB:-35}"

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

available_kib="$(df -Pk "$data_root" | awk 'NR==2 {print $4}')"
required_kib="$((minimum_free_gib * 1024 * 1024))"
if (( available_kib < required_kib )); then
  echo "Refusing download: less than ${minimum_free_gib} GiB is free on $data_root." >&2
  df -h "$data_root" >&2
  exit 2
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$data_root/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-2}"
export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY="${HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY:-1}"
export TMPDIR="${TMPDIR:-$data_root/.tmp}"
mkdir -p "$model_dir" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR"

echo "Root filesystem:  $root_device"
echo "Data filesystem:  $data_device"
echo "Model:            $model_dir"
echo "HF endpoint:      $HF_ENDPOINT"
echo "HF cache:         $HF_HUB_CACHE"
echo "Temporary files:  $TMPDIR"
echo "Revision:         $revision"
df -h / "$data_root"

if ! hf download "$repo_id" --revision "$revision" \
  --local-dir "$model_dir" --max-workers 1; then
  echo "HF CLI failed; falling back to resumable single-stream wget." >&2
  base_url="${HF_ENDPOINT%/}/$repo_id/resolve/$revision"
  runtime_files=(
    chat_template.jinja
    config.json
    generation_config.json
    processor_config.json
    tokenizer.json
    tokenizer_config.json
    model.safetensors
  )
  for filename in "${runtime_files[@]}"; do
    echo "Downloading $filename"
    partial="$model_dir/$filename.mirror-part"
    wget -c --tries=0 --timeout=60 \
      -O "$partial" "$base_url/$filename?download=true"
    mv -f "$partial" "$model_dir/$filename"
  done
fi

test -f "$model_dir/config.json"
test -f "$model_dir/chat_template.jinja"
test -f "$model_dir/model.safetensors"
actual_model_sha256="$(sha256sum "$model_dir/model.safetensors" | awk '{print $1}')"
if [[ "$actual_model_sha256" != "$expected_model_sha256" ]]; then
  echo "Unexpected model.safetensors SHA256: $actual_model_sha256" >&2
  echo "Expected: $expected_model_sha256" >&2
  exit 3
fi

du -sh "$model_dir"
echo "Gemma 4 12B download complete: $model_dir"

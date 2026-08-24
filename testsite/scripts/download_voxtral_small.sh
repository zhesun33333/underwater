#!/usr/bin/env bash
set -euo pipefail

data_root="${VAST_DATA_ROOT:-/workspace}"
model_dir="${MODEL_DIR:-$data_root/models/Voxtral-Small-24B-2507}"
repo_id="mistralai/Voxtral-Small-24B-2507"
revision="${MODEL_REVISION:-50497737e39821e982113061d95428d3b8cb2ede}"
minimum_free_gib="${MINIMUM_FREE_GIB:-60}"

if [[ ! -d "$data_root" ]]; then
  echo "Vast data path does not exist: $data_root" >&2
  exit 2
fi
case "$model_dir" in
  "$data_root"/*) ;;
  *) echo "Refusing model directory outside $data_root: $model_dir" >&2; exit 2 ;;
esac

available_kib="$(df -Pk "$data_root" | awk 'NR==2 {print $4}')"
required_kib="$((minimum_free_gib * 1024 * 1024))"
if (( available_kib < required_kib )); then
  echo "Less than ${minimum_free_gib} GiB is free on $data_root." >&2
  df -h "$data_root" >&2
  exit 2
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
export HF_HOME="${HF_HOME:-$data_root/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-4}"
export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY="${HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY:-1}"
export TMPDIR="${TMPDIR:-$data_root/.tmp}"
mkdir -p "$model_dir" "$HF_HUB_CACHE" "$HF_XET_CACHE" "$TMPDIR"

echo "Model:       $model_dir"
echo "Revision:    $revision"
echo "HF endpoint: $HF_ENDPOINT"
echo "HF cache:    $HF_HUB_CACHE"
df -h "$data_root"

# The repository also contains a 48.5 GB consolidated.safetensors file that
# duplicates the Transformers shards. Excluding it keeps the download near one
# copy of the BF16 weights instead of roughly 97 GB.
hf download "$repo_id" \
  --revision "$revision" \
  --local-dir "$model_dir" \
  --max-workers 2 \
  --exclude "consolidated.safetensors"

test -f "$model_dir/config.json"
test -f "$model_dir/model.safetensors.index.json"
test -f "$model_dir/preprocessor_config.json"
test -f "$model_dir/tekken.json"
shard_count="$(find "$model_dir" -maxdepth 1 -type f -name 'model-*-of-*.safetensors' | wc -l)"
if [[ "$shard_count" -ne 11 ]]; then
  echo "Expected 11 Transformers weight shards, found $shard_count." >&2
  exit 3
fi
if [[ -f "$model_dir/consolidated.safetensors" ]]; then
  echo "Warning: duplicate consolidated.safetensors is present and may be removed manually." >&2
fi

du -sh "$model_dir"
echo "Voxtral Small download complete: $model_dir"

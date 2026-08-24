#!/usr/bin/env bash
set -euo pipefail

data_root="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
model_dir="${MODEL_DIR:-$data_root/models/midashenglm-7b-1021-bf16}"
repo_id="mispeech/midashenglm-7b-1021-bf16"
revision="${MODEL_REVISION:-f7b0fdb745fc7997a62443007191b50f1c94702b}"

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
    added_tokens.json
    chat_template.jinja
    config.json
    configuration_midashenglm.py
    generation_config.json
    merges.txt
    model-00001-of-00004.safetensors
    model-00002-of-00004.safetensors
    model-00003-of-00004.safetensors
    model-00004-of-00004.safetensors
    model.safetensors.index.json
    modeling_midashenglm.py
    preprocessor_config.json
    processing_midashenglm.py
    processor_config.json
    special_tokens_map.json
    tokenizer.json
    tokenizer_config.json
    vocab.json
  )
  for filename in "${runtime_files[@]}"; do
    echo "Downloading $filename"
    partial="$model_dir/$filename.mirror-part"
    wget -c --tries=0 --timeout=60 \
      -O "$partial" "$base_url/$filename?download=true"
    mv -f "$partial" "$model_dir/$filename"
  done
fi

declare -A expected_sha256=(
  [model-00001-of-00004.safetensors]="336889bdb065f6cd135f00c41ab3cc248f3b848623b792ba2e5657ef9ab58bf7"
  [model-00002-of-00004.safetensors]="d8cc5f4b81da2c8f57d0d0b6ca3d0a7e30919fcb1f3b71fcba213685618387eb"
  [model-00003-of-00004.safetensors]="2c980183a2e8f017a8ab1069fe93909f23127c0cb0d0c360ecde0429c71e031c"
  [model-00004-of-00004.safetensors]="923c4c6c81d39385fd0057f7b7d22961e401d269351fb9488c61882a8e6bada4"
)

test -f "$model_dir/config.json"
test -f "$model_dir/processing_midashenglm.py"
test -f "$model_dir/model.safetensors.index.json"
for filename in "${!expected_sha256[@]}"; do
  test -f "$model_dir/$filename"
  actual="$(sha256sum "$model_dir/$filename" | awk '{print $1}')"
  if [[ "$actual" != "${expected_sha256[$filename]}" ]]; then
    echo "Unexpected SHA256 for $filename: $actual" >&2
    echo "Expected: ${expected_sha256[$filename]}" >&2
    exit 3
  fi
done

du -sh "$model_dir"
echo "MiDashengLM download complete: $model_dir"

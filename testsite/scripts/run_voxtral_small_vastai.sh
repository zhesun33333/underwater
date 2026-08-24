#!/usr/bin/env bash
set -euo pipefail

data_root="${VAST_DATA_ROOT:-/workspace}"
model_path="${1:-$data_root/models/Voxtral-Small-24B-2507}"
batch_size="${2:-1}"
gpu_list="${3:-0,1,2,3}"
data_path="${DATA_PATH:-$data_root/testset_export/sft_test_highquality.jsonl}"
audio_root="${AUDIO_ROOT:-$data_root/testset_export}"
output_root="${OUTPUT_ROOT:-$data_root/eval_results/voxtral_small_4gpu_bs${batch_size}}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

command -v nvidia-smi >/dev/null || {
  echo "nvidia-smi is unavailable; use a CUDA-enabled Vast.ai image." >&2
  exit 2
}
IFS=',' read -r -a gpus <<< "$gpu_list"
if [[ "${#gpus[@]}" -ne 4 ]]; then
  echo "This launcher requires exactly four GPU indices; got: $gpu_list" >&2
  exit 2
fi
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "Batch size must be a positive integer: $batch_size" >&2
  exit 2
fi

for gpu in "${gpus[@]}"; do
  gpu_info="$(nvidia-smi --id="$gpu" --query-gpu=name,memory.total --format=csv,noheader,nounits | head -n 1)"
  gpu_name="${gpu_info%,*}"
  gpu_memory="${gpu_info##*, }"
  echo "GPU $gpu: $gpu_name, ${gpu_memory} MiB"
  if { [[ "$gpu_name" != *A100* ]] || (( gpu_memory < 79000 )); } \
      && [[ "${ALLOW_GPU_MISMATCH:-0}" != "1" ]]; then
    echo "GPU $gpu is not an A100 with about 80 GB. Set ALLOW_GPU_MISMATCH=1 only if intentional." >&2
    exit 2
  fi
done

test -f "$model_path/config.json" || {
  echo "Model is incomplete or missing: $model_path" >&2
  exit 2
}
test -f "$data_path" || {
  echo "Test manifest is missing: $data_path" >&2
  echo "Run: VAST_DATA_ROOT=$data_root bash testsite/scripts/download_water_testset.sh" >&2
  exit 2
}
test -d "$audio_root" || {
  echo "Audio root is missing: $audio_root" >&2
  exit 2
}

# Stale shard files make a merge ambiguous, so require a new output directory.
if compgen -G "$output_root/shard_*/predictions_shard_*.jsonl" >/dev/null; then
  echo "Output already contains shard predictions: $output_root" >&2
  echo "Choose a new OUTPUT_ROOT instead of mixing runs." >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-$data_root/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTHONUNBUFFERED=1
export DATA_PATH="$data_path"
export AUDIO_ROOT="$audio_root"
export OUTPUT_ROOT="$output_root"

cd "$repo_root"
exec bash testsite/scripts/run_multigpu.sh \
  voxtral_small "$model_path" "$batch_size" "$gpu_list"

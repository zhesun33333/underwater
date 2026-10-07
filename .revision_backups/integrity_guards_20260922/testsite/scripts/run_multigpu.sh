#!/usr/bin/env bash
set -euo pipefail

backend="${1:?Usage: $0 BACKEND MODEL_PATH [BATCH_PER_GPU] [GPU_LIST]}"
model_path="${2:?model path is required}"
batch="${3:-8}"
gpu_list="${4:-${CUDA_VISIBLE_DEVICES:-0,1}}"
data_root="${AUTODL_DATA_ROOT:-/root/autodl-tmp}"
data="${DATA_PATH:-$data_root/testset_export/sft_test_highquality.jsonl}"
audio_root="${AUDIO_ROOT:-$data_root/testset_export}"
output_root="${OUTPUT_ROOT:-eval_results/$backend}"
IFS=',' read -r -a gpus <<< "$gpu_list"
workers="${#gpus[@]}"

mkdir -p "$output_root"
pids=()
for shard in "${!gpus[@]}"; do
  gpu="${gpus[$shard]}"
  echo "Starting shard $shard/$workers on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -m testsite.scripts.run_eval \
    --data "$data" --audio-root "$audio_root" \
    --backend "$backend" --model-id "$model_path" \
    --device cuda:0 --batch-size "$batch" --attn-implementation sdpa \
    --num-shards "$workers" --shard-index "$shard" \
    --output-dir "$output_root/shard_$shard" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if (( failed )); then
  echo "At least one GPU shard failed; outputs were not presented as a complete run." >&2
  exit 1
fi

mkdir -p "$output_root/merged"
python - "$output_root" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
paths = sorted(glob.glob(os.path.join(root, "shard_*", "predictions_shard_*.jsonl")))
records = []
for path in paths:
    with open(path, encoding="utf-8") as f:
        records.extend(json.loads(line) for line in f if line.strip())
ids = [x["sample_id"] for x in records]
if len(ids) != len(set(ids)):
    raise SystemExit("duplicate sample IDs found while merging")
records.sort(key=lambda x: x["sample_id"])
out = os.path.join(root, "merged", "predictions.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for row in records:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Merged {len(records)} unique predictions -> {out}")
PY
python -m testsite.scripts.merge_predictions \
  --input "$output_root/merged/predictions.jsonl" \
  --output "$output_root/merged/metrics.json"

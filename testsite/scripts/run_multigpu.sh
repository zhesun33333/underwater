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
output_root="$(mktemp -d "$output_root/run_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
echo "Fresh run directory: $output_root"
run_id="$(python -c 'import uuid; print(uuid.uuid4().hex)')"
pids=()
for shard in "${!gpus[@]}"; do
  gpu="${gpus[$shard]}"
  echo "Starting shard $shard/$workers on physical GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python -m testsite.scripts.run_eval \
    --data "$data" --audio-root "$audio_root" \
    --backend "$backend" --model-id "$model_path" \
    --device cuda:0 --batch-size "$batch" --attn-implementation sdpa \
    --num-shards "$workers" --shard-index "$shard" --run-id "$run_id" \
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
python - "$output_root" "$workers" "$data" <<'PY'
import json, os, sys
from testsite.core.integrity import validate_predictions
root = sys.argv[1]
paths = [os.path.join(root, f"shard_{i}", f"predictions_shard_{i:03d}.jsonl")
         for i in range(int(sys.argv[2]))]
records = []
for path in paths:
    with open(path, encoding="utf-8") as f:
        records.extend(json.loads(line) for line in f if line.strip())
ids = [x["sample_id"] for x in records]
if len(ids) != len(set(ids)):
    raise SystemExit("duplicate sample IDs found while merging")
validate_predictions(records, sys.argv[3])
records.sort(key=lambda x: x["sample_id"])
out = os.path.join(root, "merged", "predictions.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for row in records:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Merged {len(records)} unique predictions -> {out}")
PY
python -m testsite.scripts.merge_predictions \
  --input "$output_root/merged/predictions.jsonl" \
  --output "$output_root/merged/metrics.json" \
  --manifest "$data"

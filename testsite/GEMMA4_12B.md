# Gemma 4 12B IT: AutoDL reproduction

This target reuses the active `base` environment and the existing `gemma4`
backend. It does not create another conda environment and does not require
FlashAttention or a new inference entry.

Pinned checkpoint:

```text
google/gemma-4-12B-it
revision: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
model.safetensors SHA256: 5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d
```

The BF16 weight file is 23,919,549,408 bytes (about 23.9 GB decimal or 22.3
GiB). Keep at least 35 GiB free on `/root/autodl-tmp` before downloading.

## 1. Verify and reuse the active environment

If Gemma 4 E2B or MiDashengLM has already run successfully in the current
instance, do not reinstall the environment. Verify it instead:

```bash
cd /root/autodl-tmp/underwater

python - <<'PY'
import torch, torchvision, torchaudio, transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("transformers:", transformers.__version__)
print("Gemma 4 API: OK")
PY

python -m pip check
```

The repository's tested environment is:

```text
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
transformers==5.14.1
```

Only if the verification fails, repair the active base environment with:

```bash
bash testsite/scripts/setup_gemma4.sh
```

## 2. Check the dataset audio-length constraint

Gemma 4 supports audio up to 30 seconds. Confirm the exported test set before
starting the full run:

```bash
cd /root/autodl-tmp/underwater

python - <<'PY'
from testsite.core.loader import DataLoader
import soundfile as sf

samples = DataLoader({}).load(
    "/root/autodl-tmp/testset_export/sft_test_highquality.jsonl",
    "/root/autodl-tmp/testset_export",
)
durations = []
for sample in samples:
    info = sf.info(sample.audio_path)
    durations.append((info.frames / info.samplerate, sample.sample_id))

duration, sample_id = max(durations)
print("samples:", len(durations))
print("maximum duration:", duration, "seconds", sample_id)
if duration > 30.0 + 1e-6:
    raise SystemExit("Gemma 4 audio limit exceeded")
PY
```

## 3. Download to the AutoDL data disk

```bash
cd /root/autodl-tmp/underwater
df -h / /root/autodl-tmp
bash testsite/scripts/download_gemma4_12b.sh
```

The script uses `https://hf-mirror.com`, disables Xet, uses one download worker,
places all Hugging Face caches and temporary files on the data disk, pins the
official revision, and verifies the full weight SHA256. The model is stored in:

```text
/root/autodl-tmp/models/gemma-4-12B-it
```

## 4. Two-sample smoke test

Use batch 2 first to validate loading, audio preprocessing, multi-turn history,
generation and parsing:

```bash
cd /root/autodl-tmp/underwater

CUDA_VISIBLE_DEVICES=0 python -m testsite.scripts.run_eval \
  --data /root/autodl-tmp/testset_export/sft_test_highquality.jsonl \
  --audio-root /root/autodl-tmp/testset_export \
  --backend gemma4 \
  --model-id /root/autodl-tmp/models/gemma-4-12B-it \
  --device cuda:0 \
  --batch-size 2 \
  --attn-implementation sdpa \
  --limit 2 \
  --output-dir /root/autodl-tmp/eval_results/gemma4_12b_smoke
```

Success requires an `Evaluation Complete` marker and a generated report. The
two-sample accuracy is not a meaningful model result.

## 5. Batch-size calibration on one A100 40 GB

The model weights occupy about 24 GB before activations and KV cache. Test batch
8 on 16 samples before launching three full shards:

```bash
cd /root/autodl-tmp/underwater

CUDA_VISIBLE_DEVICES=0 python -m testsite.scripts.run_eval \
  --data /root/autodl-tmp/testset_export/sft_test_highquality.jsonl \
  --audio-root /root/autodl-tmp/testset_export \
  --backend gemma4 \
  --model-id /root/autodl-tmp/models/gemma-4-12B-it \
  --device cuda:0 \
  --batch-size 8 \
  --attn-implementation sdpa \
  --limit 16 \
  --output-dir /root/autodl-tmp/eval_results/gemma4_12b_bs8_probe
```

Monitor with `watch -n 1 nvidia-smi`. If this raises CUDA OOM, use batch 4. If
peak memory stays comfortably below 39,000 MiB, batch 8 is appropriate for the
full run. Do not infer the safe batch from idle post-load memory alone; observe
peak memory during generation.

## 6. Three-GPU full evaluation with terminal output

This is data parallelism: each GPU holds one complete model replica and receives
one dataset shard. Batch 8 per GPU gives an aggregate in-flight batch of up to
24 samples.

```bash
cd /root/autodl-tmp/underwater

OUT=/root/autodl-tmp/eval_results/gemma4_12b_3gpu_bs8
mkdir -p "$OUT"
set -o pipefail

OUTPUT_ROOT="$OUT" \
OMP_NUM_THREADS=8 \
PYTHONUNBUFFERED=1 \
HF_HOME=/root/autodl-tmp/.cache/huggingface \
bash testsite/scripts/run_multigpu.sh \
  gemma4 \
  /root/autodl-tmp/models/gemma-4-12B-it \
  8 \
  0,1,2 \
  2>&1 | tee "$OUT/run.log"
```

If batch 8 is not safe, use a new output directory and replace `8` with `4`.
Never mix retry outputs with an earlier failed directory.

After all shards succeed, the launcher automatically checks for duplicate IDs
and writes:

```text
/root/autodl-tmp/eval_results/gemma4_12b_3gpu_bs8/merged/predictions.jsonl
/root/autodl-tmp/eval_results/gemma4_12b_3gpu_bs8/merged/metrics.json
```

The final success condition is:

```text
Merged 2600 unique predictions
Full-run L3 accuracy: ...
```

## 7. Why the existing backend is valid for 12B

`Gemma4Backend` already uses the official interfaces:

- `AutoModelForMultimodalLM` and `AutoProcessor`;
- text before audio in each user message;
- `enable_thinking=False` for deterministic classification-style output;
- BF16 inference with SDPA;
- batched `apply_chat_template` and `model.generate`;
- prompt-token slicing before decoding;
- audio attached only to the first user turn and preserved through multi-turn
  conversation history.

The 12B Unified checkpoint changes the internal multimodal architecture but not
the public processor/model contract used by this backend.

# MiDashengLM-7B-1021-BF16

Official checkpoint: `mispeech/midashenglm-7b-1021-bf16`

The backend uses the checkpoint's official remote model and processor code. It
supports true batched audio conversations and keeps one model replica per GPU
when used with `run_multigpu.sh`. A small import compatibility shim keeps the
official Transformers 4.x remote processor working under the existing
Transformers 5.x Gemma runtime without changing model weights or inference.
The backend also creates Dasheng's two non-persistent audio frontend buffers on
CPU during Transformers 5 meta initialization and moves them to CUDA after the
checkpoint is loaded; this avoids a CPU/meta construction error without
changing checkpoint tensors.

## 1. Configure the active base environment

Do not create another Conda environment:

```bash
cd /root/autodl-tmp/underwater
conda activate base
bash testsite/scripts/setup_midashenglm.sh
```

The script keeps the existing PyTorch 2.6 installation, uses the Aliyun PyPI
mirror by default, disables pip caching, and places temporary files on the data
disk.

## 2. Download to the data disk

```bash
cd /root/autodl-tmp/underwater
bash testsite/scripts/download_midashenglm.sh
```

The default destination is:

```text
/root/autodl-tmp/models/midashenglm-7b-1021-bf16
```

The downloader pins revision `f7b0fdb745fc7997a62443007191b50f1c94702b`,
uses `https://hf-mirror.com`, limits concurrency, and verifies all four weight
shards by SHA256.

## 3. Two-sample smoke test

```bash
cd /root/autodl-tmp/underwater

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 \
HF_HOME=/root/autodl-tmp/.cache/huggingface \
python -m testsite.scripts.run_eval \
  --data /root/autodl-tmp/testset_export/sft_test_highquality.jsonl \
  --audio-root /root/autodl-tmp/testset_export \
  --backend midashenglm \
  --model-id /root/autodl-tmp/models/midashenglm-7b-1021-bf16 \
  --device cuda:0 \
  --batch-size 2 \
  --attn-implementation sdpa \
  --limit 2 \
  --output-dir /root/autodl-tmp/eval_results/midashenglm_smoke
```

## 4. Three-GPU full evaluation

Start with batch size 32 per A100 40GB. Increase only after observing the peak
memory during generation.

```bash
cd /root/autodl-tmp/underwater

OUT=/root/autodl-tmp/eval_results/midashenglm_3gpu_bs32
mkdir -p "$OUT"
set -o pipefail

OUTPUT_ROOT="$OUT" \
OMP_NUM_THREADS=8 \
PYTHONUNBUFFERED=1 \
HF_HOME=/root/autodl-tmp/.cache/huggingface \
bash testsite/scripts/run_multigpu.sh \
  midashenglm \
  /root/autodl-tmp/models/midashenglm-7b-1021-bf16 \
  32 \
  0,1,2 \
  2>&1 | tee "$OUT/run.log"
```

Successful completion creates:

```text
/root/autodl-tmp/eval_results/midashenglm_3gpu_bs32/merged/predictions.jsonl
/root/autodl-tmp/eval_results/midashenglm_3gpu_bs32/merged/metrics.json
```

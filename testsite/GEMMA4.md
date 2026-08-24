# Gemma 4 E2B IT: AutoDL reproduction

This target reuses the active `base` environment and does not create another
conda environment. Gemma 4 audio inference requires PyTorch 2.6 or newer, so
the setup upgrades PyTorch 2.5.1 to the matching 2.6.0/0.21.0/2.6.0 trio.

## 1. Configure the active base environment

```bash
cd /root/autodl-tmp/underwater
bash testsite/scripts/setup_gemma4.sh
```

This switches the legacy AF-Next Transformers checkout to Transformers 5.14.1
and installs PyTorch 2.6.0, torchvision 0.21.0, and torchaudio 2.6.0 from the
official PyTorch CUDA wheel index. AF-Next has already been evaluated; run
`setup_af_next_legacy.sh` only if that legacy checkpoint needs to be run again.

## 2. Download to the AutoDL data disk

```bash
cd /root/autodl-tmp/underwater
bash testsite/scripts/download_gemma4.sh
```

The script uses `https://hf-mirror.com`, disables Xet, downloads with one
worker, and keeps the Hugging Face cache and temporary files under
`/root/autodl-tmp`. The pinned checkpoint is stored in:

```text
/root/autodl-tmp/models/gemma-4-E2B-it
```

## 3. Two-sample smoke test

```bash
cd /root/autodl-tmp/underwater
CUDA_VISIBLE_DEVICES=0 python -m testsite.scripts.run_eval \
  --data /root/autodl-tmp/testset_export/sft_test_highquality.jsonl \
  --audio-root /root/autodl-tmp/testset_export \
  --backend gemma4 \
  --model-id /root/autodl-tmp/models/gemma-4-E2B-it \
  --device cuda:0 \
  --batch-size 2 \
  --attn-implementation sdpa \
  --limit 2 \
  --output-dir /root/autodl-tmp/eval_results/gemma4_smoke
```

## 4. Three-GPU full evaluation

Start with batch 8 per A100 40 GB. Use a new output directory when retrying so
old shard files cannot be mixed into a new run.

```bash
cd /root/autodl-tmp/underwater
GEMMA_OUT=/root/autodl-tmp/eval_results/gemma4_3gpu_bs8
mkdir -p "$GEMMA_OUT"

nohup env OUTPUT_ROOT="$GEMMA_OUT" \
  bash testsite/scripts/run_multigpu.sh \
  gemma4 /root/autodl-tmp/models/gemma-4-E2B-it 8 0,1,2 \
  > "$GEMMA_OUT/run.log" 2>&1 &

echo $! | tee "$GEMMA_OUT/launcher.pid"
tail -f "$GEMMA_OUT/run.log"
```

After all shards succeed, merged outputs are written to:

```text
/root/autodl-tmp/eval_results/gemma4_3gpu_bs8/merged/predictions.jsonl
/root/autodl-tmp/eval_results/gemma4_3gpu_bs8/merged/metrics.json
```

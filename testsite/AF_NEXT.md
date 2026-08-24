# Audio-Flamingo-Next: AutoDL reproduction

This target reuses the active `base` environment and its existing PyTorch
installation. It upgrades Transformers in place because AF-Next requires the
native MusicFlamingo implementation in Transformers 5.x.

## 1. Check the data disk and GPU

```bash
conda activate base
cd /root/autodl-tmp/underwater
df -hT / /root/autodl-tmp
python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'
```

`/root/autodl-tmp` must be on a different filesystem from `/`. Do not use
`/home/autodl-tmp`.

## 2. Configure the existing base environment

```bash
bash testsite/scripts/setup_af_next.sh
```

The script keeps the installed PyTorch build and installs Transformers 5.14.1.
This changes the exact Qwen environment; run `setup_qwen25_omni.sh` if the old
Qwen stack is needed again.

For the pre-migration checkpoint with `model_type=audioflamingonext` and model
SHA256 `39309fb35b1a2b14e90a822eabf4da7107fdaa569bd3f7ceefe1922c83905356`,
use the pinned legacy runtime instead:

```bash
bash testsite/scripts/setup_af_next_legacy.sh
```

## 3. Download the checkpoint

```bash
bash testsite/scripts/download_af_next.sh
```

The model, Hugging Face/Xet caches, and temporary files are all placed on the
AutoDL data disk. The checkpoint is about 16.5 GB and is saved under:

```text
/root/autodl-tmp/models/audio-flamingo-next-hf
```

## 4. One-GPU smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python -m testsite.scripts.run_eval \
  --data /root/autodl-tmp/testset_export/sft_test_highquality.jsonl \
  --audio-root /root/autodl-tmp/testset_export \
  --backend af_next \
  --model-id /root/autodl-tmp/models/audio-flamingo-next-hf \
  --device cuda:0 --batch-size 2 --attn-implementation sdpa \
  --limit 8 \
  --output-dir /root/autodl-tmp/eval_results/af_next_smoke
```

## 5. Three-GPU full evaluation

Start with four examples per GPU. If the smoke test and an initial formal shard
show enough free memory, increase this to 8.

```bash
OUTPUT_ROOT=/root/autodl-tmp/eval_results/af_next_3gpu \
bash testsite/scripts/run_multigpu.sh af_next \
  /root/autodl-tmp/models/audio-flamingo-next-hf 4 0,1,2
```

The script launches one model replica per GPU, shards the dataset, waits for all
workers, and writes merged metrics to:

```text
/root/autodl-tmp/eval_results/af_next_3gpu/merged/metrics.json
```

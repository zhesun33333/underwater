# Qwen2.5-Omni-7B：AutoDL 从零复现

当前流程只配置现有 `base`，不创建 conda 环境，也不安装 MiniCPM-o 或 AF-Next。
PyTorch 2.5.1 保持不变。

## 1. 确认数据盘

```bash
conda activate base
cd /root/autodl-tmp/underwater

df -hT / /root/autodl-tmp
python -c 'import torch; print(torch.__version__, torch.cuda.device_count())'
```

`/root/autodl-tmp` 必须与 `/` 显示为不同文件系统。不要使用
`/home/autodl-tmp`，它会落到系统盘。

## 2. 配置现有 base

```bash
bash testsite/scripts/setup_qwen25_omni.sh
```

脚本禁用 pip 缓存并锁定当前 PyTorch 版本。

## 3. 下载 Qwen 权重

```bash
bash testsite/scripts/download_qwen25_omni.sh
```

所有权重、HF/ModelScope 缓存和临时文件都放在 `/root/autodl-tmp`。下载脚本会比较
系统盘与数据盘设备；若目标仍在系统盘会直接退出。模型默认保存在：

```text
/root/autodl-tmp/models/Qwen2.5-Omni-7B
```

## 4. 单卡冒烟测试

```bash
CUDA_VISIBLE_DEVICES=0 python -m testsite.scripts.run_eval \
  --data /root/autodl-tmp/testset_export/sft_test_highquality.jsonl \
  --audio-root /root/autodl-tmp/testset_export \
  --backend qwen25_omni \
  --model-id /root/autodl-tmp/models/Qwen2.5-Omni-7B \
  --device cuda:0 --batch-size 1 --attn-implementation sdpa \
  --limit 8 --output-dir eval_results/qwen_smoke
```

## 5. 多卡运行

四卡、每卡 batch 8：

```bash
bash testsite/scripts/run_multigpu.sh qwen25_omni \
  /root/autodl-tmp/models/Qwen2.5-Omni-7B 8 0,1,2,3
```

通过后逐步尝试每卡 batch 12、16、24。完整合并指标位于
`eval_results/qwen25_omni/merged/metrics.json`。

# Voxtral-Small-24B-2507：Vast.ai 四卡 A100 80GB 复现

本文从“Vast.ai 四卡实例已经启动，评测代码已经位于
`/workspace/underwater`”开始，不涉及 Git 仓库同步。当前阶段依次完成：

1. 确认 GPU、Python 和磁盘；
2. 配置 Voxtral 专用环境；
3. 下载模型；
4. 下载并解压测试数据；
5. 做单卡最小冒烟测试。

完整评测采用数据并行：4 个独立进程分别使用一张 GPU，每张 A100 80GB
加载一份完整的 BF16 模型，各自处理四分之一测试集。它不是张量并行。

## 0. 约定路径

以下命令统一使用：

| 内容 | 路径 |
|---|---|
| 评测代码 | `/workspace/underwater` |
| 模型 | `/workspace/models/Voxtral-Small-24B-2507` |
| 数据压缩包 | `/workspace/downloads/water/testset_export.tar.gz` |
| 解压后的数据 | `/workspace/testset_export` |
| 评测结果 | `/workspace/eval_results` |

如果 Vast.ai 的持久盘不在 `/workspace`，只需把所有命令中的
`VAST_DATA_ROOT=/workspace` 改成实际挂载点。

## 1. 实例与基础环境确认

进入实例后执行：

```bash
cd /workspace/underwater

nvidia-smi --query-gpu=index,name,memory.total --format=csv
df -h /workspace
free -h
which python
python --version
python -c 'import torch; print("torch:", torch.__version__); print("CUDA:", torch.version.cuda); print("GPUs:", torch.cuda.device_count())'
```

开始下载前应确认：

- `nvidia-smi` 显示 4 张约 80GB 的 A100；
- `torch.cuda.device_count()` 返回 `4`；
- 当前 Python 能导入 CUDA 版 PyTorch；
- `/workspace` 至少还有 70 GiB，建议 100 GiB 以上；
- 主机内存至少 128 GiB，四模型进程同时启动时 256 GiB 更稳妥。

不要在这一步运行完整评测。

## 2. 配置 Voxtral 专用环境

脚本复用 Vast.ai 镜像已有的 CUDA PyTorch，不重新安装 PyTorch；它会安装并固定：

- `transformers==4.54.0`；
- `mistral-common[audio]>=1.8.1,<2`；
- `accelerate`、`huggingface-hub`、`hf-xet`、`soundfile` 等运行依赖。

执行：

```bash
cd /workspace/underwater
VAST_DATA_ROOT=/workspace bash testsite/scripts/setup_voxtral_small.sh
```

成功时末尾应看到类似输出：

```text
CUDA available: True
visible GPUs: 4
transformers: 4.54.0
Voxtral runtime: OK
```

随后再次确认环境：

```bash
python - <<'PY'
import torch
import transformers
from transformers import AutoProcessor, VoxtralForConditionalGeneration

print("torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
print("Voxtral imports: OK")
PY
```

这是 Voxtral 专用环境。不要在同一环境中切换到要求 Transformers 5.x 的
Gemma 4 后端。

## 3. 下载 Voxtral-Small-24B 模型

官方模型为
[`mistralai/Voxtral-Small-24B-2507`](https://huggingface.co/mistralai/Voxtral-Small-24B-2507)。
Transformers BF16 权重约 48.5GB。下载脚本固定 revision
`50497737e39821e982113061d95428d3b8cb2ede`，并排除重复的
`consolidated.safetensors`，否则同一套权重会占用接近 97GB。

执行：

```bash
cd /workspace/underwater
VAST_DATA_ROOT=/workspace bash testsite/scripts/download_voxtral_small.sh
```

脚本支持断点续传，可以在网络中断后重新执行同一条命令。默认从 Hugging Face
官方下载；若直连不可用，可改用镜像：

```bash
cd /workspace/underwater
HF_ENDPOINT=https://hf-mirror.com \
VAST_DATA_ROOT=/workspace \
bash testsite/scripts/download_voxtral_small.sh
```

下载完成后，脚本会确认配置文件和 11 个 Transformers 权重分片存在。手动查看：

```bash
du -sh /workspace/models/Voxtral-Small-24B-2507
ls -lh /workspace/models/Voxtral-Small-24B-2507/model-*-of-*.safetensors
```

## 4. 下载并解压测试数据

数据来自公开的 Hugging Face Dataset：
[`bubbachuck3333/water`](https://huggingface.co/datasets/bubbachuck3333/water/tree/main)，
无需访问 token。

执行：

```bash
cd /workspace/underwater
VAST_DATA_ROOT=/workspace bash testsite/scripts/download_water_testset.sh
```

脚本会：

- 固定 dataset revision `e398d8b6db7d2f9439daa2cc6319ed646ceb1bd3`；
- 下载约 2.23GB 的 `testset_export.tar.gz`；
- 核对压缩包大小和 SHA-256，避免损坏下载；
- 检查压缩包路径安全；
- 解压到 `/workspace/testset_export`。

本流程不检查样本数量、标签、ID 或数据内容合规性。若目标目录已经存在，脚本会
保留现有目录并直接退出，不覆盖数据。

下载完成后只确认路径和磁盘占用：

```bash
du -sh /workspace/downloads/water/testset_export.tar.gz
du -sh /workspace/testset_export
ls -lh /workspace/testset_export/sft_test_highquality.jsonl
```

完成到这里后，环境、模型和数据下载阶段结束。

## 5. 单卡最小冒烟测试

在启动四卡完整评测前，先用一张卡和 2 条数据验证模型能加载、音频能读取、
多轮推理能完成：

```bash
cd /workspace/underwater

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m testsite.scripts.run_eval \
  --data /workspace/testset_export/sft_test_highquality.jsonl \
  --audio-root /workspace/testset_export \
  --backend voxtral_small \
  --model-id /workspace/models/Voxtral-Small-24B-2507 \
  --device cuda:0 \
  --batch-size 1 \
  --attn-implementation sdpa \
  --limit 2 \
  --output-dir /workspace/eval_results/voxtral_small_smoke
```

成功标准是终端出现 `Evaluation Complete`。两条样本的准确率没有统计意义。

## 6. 后续：标定单卡 batch

冒烟测试通过后，用 batch 2 跑 8 条，并在另一终端观察生成阶段峰值显存：

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m testsite.scripts.run_eval \
  --data /workspace/testset_export/sft_test_highquality.jsonl \
  --audio-root /workspace/testset_export \
  --backend voxtral_small \
  --model-id /workspace/models/Voxtral-Small-24B-2507 \
  --device cuda:0 \
  --batch-size 2 \
  --attn-implementation sdpa \
  --limit 8 \
  --output-dir /workspace/eval_results/voxtral_small_bs2_probe
```

```bash
watch -n 1 nvidia-smi
```

若 OOM，完整运行使用每卡 batch 1；若峰值仍有充足余量，可另建输出目录测试
batch 4。不要只根据模型加载后的空闲显存决定 batch。

## 7. 后续：四卡完整评测

最稳妥的起点是每卡 batch 1：

```bash
cd /workspace/underwater
OUT=/workspace/eval_results/voxtral_small_4gpu_bs1
mkdir -p "$OUT"
set -o pipefail

VAST_DATA_ROOT=/workspace \
OUTPUT_ROOT="$OUT" \
bash testsite/scripts/run_voxtral_small_vastai.sh \
  /workspace/models/Voxtral-Small-24B-2507 1 0,1,2,3 \
  2>&1 | tee "$OUT/run.log"
```

启动器会为 4 张卡各启动一个独立进程，完成后合并四个 shard。若 batch 2 探测
已经通过，可把参数和输出目录中的 `bs1` 一并改为 `2`。

关键结果位于：

```text
/workspace/eval_results/voxtral_small_4gpu_bs1/merged/predictions.jsonl
/workspace/eval_results/voxtral_small_4gpu_bs1/merged/metrics.json
/workspace/eval_results/voxtral_small_4gpu_bs1/run.log
```

若发生 OOM 或运行中断，保留旧目录排错，并换一个新的 `OUTPUT_ROOT` 重跑，
不要把不同 batch 或不同 checkpoint 的 shard 混在同一目录。

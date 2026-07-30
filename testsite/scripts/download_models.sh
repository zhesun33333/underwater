#!/usr/bin/env bash
# 下载所有对比模型。优先 ModelScope, 不行提示手动下载。
set -euo pipefail

MODEL_DIR="$HOME/autodl-tmp/models"
mkdir -p "$MODEL_DIR"

# ---- HF 镜像 (备用) ----
export HF_ENDPOINT=https://hf-mirror.com

echo "========================================"
echo "  下载对比模型"
echo "  存放目录: $MODEL_DIR"
echo "========================================"

# ---- 工具函数 ----
download_hf() {
    local repo_id="$1"
    local target="$2"
    if [ -f "$target/config.json" ]; then
        echo "  [SKIP] 已存在: $target"
        return 0
    fi
    pip install huggingface_hub -q 2>/dev/null
    if python -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo_id', local_dir='$target')
" 2>&1; then
        echo "  [OK] HF: $repo_id"
    else
        echo "  [FAIL] HF 下载失败: $repo_id"
        echo "  请本地下载后上传到: $target"
        return 1
    fi
}

download_ms() {
    local repo_id="$1"
    local target="$2"
    if [ -f "$target/config.json" ]; then
        echo "  [SKIP] 已存在: $target"
        return 0
    fi
    pip install modelscope -q 2>/dev/null
    if python -c "
from modelscope import snapshot_download
snapshot_download('$repo_id', cache_dir='$MODEL_DIR')
" 2>&1; then
        echo "  [OK] ModelScope: $repo_id"
    else
        echo "  [FAIL] ModelScope 下载失败, 尝试 HF..."
        download_hf "$repo_id" "$target"
    fi
}

# ---- 1. Kimi-Audio-7B-Instruct (10B, ~23GB) — ModelScope 有 ----
echo ""
echo "[1/4] Kimi-Audio-7B-Instruct (10B, ~23GB) ..."
download_ms "moonshotai/Kimi-Audio-7B-Instruct" "$MODEL_DIR/moonshotai/Kimi-Audio-7B-Instruct"

# ---- 2. Aero-1-Audio (1.5B, ~4GB) — 仅 HF ----
echo ""
echo "[2/4] Aero-1-Audio (1.5B, ~5GB) ..."
download_hf "lmms-lab/Aero-1-Audio" "$MODEL_DIR/lmms-lab/Aero-1-Audio"

# ---- 3. Voxtral-Mini-3B (~8GB) — 仅 HF ----
echo ""
echo "[3/4] Voxtral-Mini-3B (3B, ~8GB) ..."
download_hf "mistralai/Voxtral-Mini-3B-2507" "$MODEL_DIR/mistralai/Voxtral-Mini-3B-2507"

# ---- 4. Audio-Flamingo-Next (7B, ~18GB) — 仅 HF ----
echo ""
echo "[4/4] Audio-Flamingo-Next (7B, ~18GB) ..."
download_hf "nvidia/Audio-Flamingo-Next-Instruct" "$MODEL_DIR/nvidia/Audio-Flamingo-Next-Instruct"

# ---- 汇总 ----
echo ""
echo "========================================"
echo "  模型路径汇总"
echo "========================================"
echo ""
echo "  # 各模型路径 (用于 --model-id)"
for dir in \
    "$MODEL_DIR/models/Qwen2-Audio-7B-Instruct/snapshots/master" \
    "$MODEL_DIR/moonshotai/Kimi-Audio-7B-Instruct" \
    "$MODEL_DIR/lmms-lab/Aero-1-Audio" \
    "$MODEL_DIR/mistralai/Voxtral-Mini-3B-2507" \
    "$MODEL_DIR/nvidia/Audio-Flamingo-Next-Instruct" \
; do
    if [ -f "$dir/config.json" ]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "  [OK] $size  $dir"
    else
        echo "  [---]  $dir"
    fi
done
echo ""
echo "  下载成功的可以直接跑。带 [---] 的需要本地下载后上传。"

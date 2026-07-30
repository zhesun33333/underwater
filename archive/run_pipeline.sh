#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 水声大模型数据合成管线 — 一键运行脚本
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_DIR="$SCRIPT_DIR/dataset"
PROCESSED_AUDIO="$SCRIPT_DIR/processed_audio"
QA_OUTPUT="$SCRIPT_DIR/qa_output"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ---- Step 1: BELLHOP 信道卷积 ----
run_step1() {
    log "========== Step 1: BELLHOP 信道卷积 =========="

    log "Step 1a: PulseCom 信道卷积..."
    python pipeline_step1_channel.py --config config.yaml
    log "PulseCom Step 1 完成"

    log "Step 1b: 舰船辐射噪声信道卷积..."
    python pipeline_ship_step1_channel.py --config config_ship.yaml
    log "舰船 Step 1 完成"
}

# ---- Step 2: QA 对生成 ----
run_step2() {
    log "========== Step 2: QA 对生成 =========="

    log "Step 2a: PulseCom QA 对生成..."
    python pipeline_step2_qa.py --config config.yaml
    log "PulseCom Step 2 完成"

    log "Step 2b: 舰船 QA 对生成..."
    python pipeline_ship_step2_qa.py --config config_ship.yaml
    log "舰船 Step 2 完成"
}

# ---- 合并数据集 ----
merge_datasets() {
    log "========== 合并数据集 =========="

    mkdir -p "$DATASET_DIR"

    for split in train val test; do
        local merged="$DATASET_DIR/sft_${split}.jsonl"
        local pc="$QA_OUTPUT/sft_${split}.jsonl"
        local ship="$QA_OUTPUT/sft_ship_${split}.jsonl"

        > "$merged"   # 清空/创建

        if [ -f "$pc" ]; then
            cat "$pc" >> "$merged"
            local pc_n=$(wc -l < "$pc" | tr -d ' ')
            log "  PulseCom ${split}: ${pc_n} 条"
        else
            warn "  PulseCom ${split} 不存在: $pc"
        fi

        if [ -f "$ship" ]; then
            cat "$ship" >> "$merged"
            local ship_n=$(wc -l < "$ship" | tr -d ' ')
            log "  舰船    ${split}: ${ship_n} 条"
        else
            warn "  舰船 ${split} 不存在: $ship"
        fi

        local total=$(wc -l < "$merged" | tr -d ' ')
        log "  合并 ${split}: ${total} 条 → ${merged}"
    done
}

# ---- 统计卡片 ----
run_stats() {
    log "========== 数据集统计 =========="

    python dataset_stats.py \
        --train "$DATASET_DIR/sft_train.jsonl" \
        --val   "$DATASET_DIR/sft_val.jsonl" \
        --test  "$DATASET_DIR/sft_test.jsonl" \
        --audio-root "$PROCESSED_AUDIO" \
        --output "$DATASET_DIR/dataset_stats.md"

    log "统计卡片 → $DATASET_DIR/dataset_stats.md"
}

# ---- 打印最终目录结构 ----
print_summary() {
    log "========== 完成 =========="
    echo ""
    echo "  数据集目录: $DATASET_DIR"
    echo ""
    echo "  $DATASET_DIR/"
    for f in "$DATASET_DIR"/*; do
        local name=$(basename "$f")
        if [ -f "$f" ]; then
            local lines=$(wc -l < "$f" | tr -d ' ')
            echo "    ${name}  (${lines} 行)"
        fi
    done
    echo ""
    echo "  JSONL 格式: {\"id\": \"...\", \"audio\": \"...\", \"conversations\": [...]}"
    echo "  统计卡片:   $DATASET_DIR/dataset_stats.md"
    echo ""
}

# ---- 主流程 ----
main() {
    local skip_step1=false
    local skip_step2=false

    while [ $# -gt 0 ]; do
        case "$1" in
            --skip-step1) skip_step1=true ;;
            --skip-step2) skip_step2=true ;;
            *) err "未知参数: $1"; exit 1 ;;
        esac
        shift
    done

    echo ""
    log "水声大模型数据合成管线"
    log "输出目录: $DATASET_DIR"
    echo ""

    if [ "$skip_step1" = false ]; then
        run_step1
    else
        warn "跳过 Step 1 (--skip-step1)"
    fi

    if [ "$skip_step2" = false ]; then
        run_step2
    else
        warn "跳过 Step 2 (--skip-step2)"
    fi

    merge_datasets
    run_stats
    print_summary
}

main "$@"

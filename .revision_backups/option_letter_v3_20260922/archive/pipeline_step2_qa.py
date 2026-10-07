"""
Underwater acoustic LLM data synthesis — Step 2: QA pair generation (PulseCom)
Generates 3-turn SFT data:
  Turn 1: Source-based Active/Passive classification → short answer
  Turn 2: Subcategory + specific type → long answer
  Turn 3: Reasoning rationale → qualitative reasoning text
Supports train/val/test split (70/15/15, unified 3-turn format)
Usage:
  python pipeline_step2_qa.py [--config config.yaml]
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.json_parser import (
    load_jsonc, extract_labels, get_wav_path, get_id,
)
from utils.split_utils import stratified_group_split
# ============================================================
# 模板定义
# ============================================================
# Turn 1 — Source-based Active/Passive classification
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from source_label_prompts import QA_PROMPT_VERSION, T1_TEMPLATES, get_t1_answer

# Turn 2 — Subcategory + specific type (with T1-confirmed L1 embedded)
_T2_OPTIONS = (
    "A. Active — Detection pulse — CW (Continuous Wave)\n"
    "B. Active — Detection pulse — LFM (Linear Frequency Modulation)\n"
    "C. Active — Detection pulse — HFM (Hyperbolic Frequency Modulation)\n"
    "D. Active — Communication — 2FSK (Binary Frequency Shift Keying)\n"
    "E. Active — Communication — 4FSK (Quaternary Frequency Shift Keying)\n"
    "F. Active — Communication — BPSK (Binary Phase Shift Keying)\n"
    "G. Active — Communication — QPSK (Quadrature Phase Shift Keying)\n"
    "H. Active — Communication — OFDM (Orthogonal Frequency Division Multiplexing)"
)
T2_TEMPLATES = [
    f"Good — the signal is {{L1}}. Now determine whether it is a detection pulse or communication signal, and specify the exact type.\nOptions:\n{_T2_OPTIONS}",
    f"Understood — it is {{L1}}. Now identify which subcategory it belongs to.\nOptions:\n{_T2_OPTIONS}",
    f"Signal is {{L1}}. What is its specific signal type?\nOptions:\n{_T2_OPTIONS}",
    f"Got it — {{L1}}. Now a finer classification: is it a detection pulse or communication signal? Which specific type?\nOptions:\n{_T2_OPTIONS}",
    f"OK, {{L1}} confirmed. Next step: identify the specific signal type.\nOptions:\n{_T2_OPTIONS}",
    f"Confirmed as {{L1}}. Which specific signal type is it?\nOptions:\n{_T2_OPTIONS}",
    f"First step complete. Now further classify this {{L1}} signal.\nOptions:\n{_T2_OPTIONS}",
    f"Within the {{L1}} category, which specific signal type is it?\nOptions:\n{_T2_OPTIONS}",
]
# Turn 3 — Reasoning
T3_PROMPT = "Please briefly explain your reasoning."
# ============================================================
# 答案生成 — 选项字母格式 (方案 A)
# ============================================================

# L3 key → T2 完整选项文本 (PulseCom active, 8 选 1)
_T2_ACTIVE_FULL = {
    "CW": "A. Active — Detection pulse — CW (Continuous Wave)",
    "LFM": "B. Active — Detection pulse — LFM (Linear Frequency Modulation)",
    "HFM": "C. Active — Detection pulse — HFM (Hyperbolic Frequency Modulation)",
    "2FSK": "D. Active — Communication — 2FSK (Binary Frequency Shift Keying)",
    "4FSK": "E. Active — Communication — 4FSK (Quaternary Frequency Shift Keying)",
    "BPSK": "F. Active — Communication — BPSK (Binary Phase Shift Keying)",
    "QPSK": "G. Active — Communication — QPSK (Quadrature Phase Shift Keying)",
    "OFDM": "H. Active — Communication — OFDM (Orthogonal Frequency Division Multiplexing)",
}

def build_turn2_answer(labels: Dict[str, str]) -> str:
    """Turn 2 答案: 完整选项文本 (A-H)。"""
    l3 = labels["L3"]
    if l3 not in _T2_ACTIVE_FULL:
        raise ValueError(f"unsupported PulseCom L3 label: {l3!r}")
    return _T2_ACTIVE_FULL[l3]


def build_turn3_answer(labels: Dict[str, str], meta: Dict, rng: random.Random) -> str:
    """Turn 3 答案: 结构化推理文本。"""
    return _build_reasoning(labels["L3"], meta, rng)


# ============================================================
# T3 结构化推理模板
# L1/L2/L3 术语从 shared_terminology.py 统一导入 (single source of truth)
# ============================================================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_terminology import (
    L1_ACTIVE_TERMS  as _L1_ACTIVE_TERMS,
    L2_ACTIVE_TERMS  as _L2_ACTIVE_TERMS,
    build_l3_evidence,
)

# T3 句式模板 (3 种变体)
_T3_TEMPLATES = [
    "The signal is {L1} and exhibits {L2}; observable evidence includes {L3}.",
    "Based on its time-frequency and modulation structure, the signal is {L1} with {L2}; supporting cues are {L3}.",
    "Spectral and temporal analysis indicates {L1} with {L2}, supported by {L3}.",
]


def _build_reasoning(l3: str, meta: Dict, rng: random.Random) -> str:
    """根据 L3 类型生成结构化推理文本: L1描述 + L2特征 + L3术语。"""
    terms = build_l3_evidence(l3, meta, rng)
    l3_text = ", ".join(terms)

    # L1 和 L2 由 L3 推导
    if l3 in ("CW", "LFM", "HFM"):
        l1_text = rng.choice(_L1_ACTIVE_TERMS)
        l2_text = rng.choice(_L2_ACTIVE_TERMS["detection pulse"])
    elif l3 in ("2FSK", "4FSK", "BPSK", "QPSK", "OFDM"):
        l1_text = rng.choice(_L1_ACTIVE_TERMS)
        l2_text = rng.choice(_L2_ACTIVE_TERMS["communication signal"])
    else:
        raise ValueError(f"unsupported PulseCom L3 label: {l3!r}")

    template = rng.choice(_T3_TEMPLATES)
    return template.replace("{L1}", l1_text).replace("{L2}", l2_text).replace("{L3}", l3_text)


# ============================================================
# 对话构建
# ============================================================
def build_conversations(
    labels: Dict[str, str],
    meta: Dict,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """生成三轮对话: T1(L1) → T2(L2+L3) → T3(推理依据)。"""
    l1 = labels["L1"]
    q1 = rng.choice(T1_TEMPLATES)
    a1 = get_t1_answer(l1)
    q2 = rng.choice(T2_TEMPLATES).replace("{L1}", l1)
    a2 = build_turn2_answer(labels)
    q3 = T3_PROMPT
    a3 = build_turn3_answer(labels, meta, rng)
    return [
        {"from": "human", "value": q1},
        {"from": "gpt", "value": a1},
        {"from": "human", "value": q2},
        {"from": "gpt", "value": a2},
        {"from": "human", "value": q3},
        {"from": "gpt", "value": a3},
    ]
# ============================================================
# 主流程
# ============================================================
def process_single_json(
    json_path: Path,
    processed_root: Path,
    rng: random.Random,
) -> Optional[Dict]:
    """处理单个 JSON，生成三轮对话。"""
    try:
        meta = load_jsonc(json_path)
    except Exception:
        return None
    labels = extract_labels(meta)
    if labels["L1"] == "unknown":
        return None
    wav_rel = get_wav_path(meta)
    sample_id = get_id(meta)
    wav_path = processed_root / wav_rel
    if not wav_path.exists():
        candidates = list(processed_root.rglob(f"{sample_id}.wav"))
        if candidates:
            wav_path = candidates[0]
            wav_rel = str(wav_path.relative_to(processed_root))
        else:
            return None
    conversations = build_conversations(labels, meta, rng)
    return {
        "id": sample_id,
        "audio": wav_rel,
        "qa_prompt_version": QA_PROMPT_VERSION,
        "conversations": conversations,
        "_meta": {
            "L1": labels["L1"],
            "L2": labels["L2"],
            "L3": labels["L3"],
        },
    }
def find_processed_jsons(processed_root: Path, ds_cfg: dict) -> List[Path]:
    """查找 PulseCom 处理后 JSONC 文件。"""
    ds_root = processed_root / ds_cfg["name"]
    if not ds_root.exists():
        return []
    json_dir = ds_root / ds_cfg["json_dir"]
    if not json_dir.exists():
        return []
    return sorted([
        jf for jf in json_dir.rglob(f"*{ds_cfg['json_ext']}")
        if not jf.name.startswith("manifest")
    ])
def main():
    parser = argparse.ArgumentParser(description="Step 2: QA pair generation")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    args = parser.parse_args()
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    import re
    def _resolve(obj):
        if isinstance(obj, str):
            return re.sub(r'\$\{(\w+)\}', lambda m: os.environ.get(m.group(1), m.group(0)), obj)
        elif isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_resolve(v) for v in obj]
        return obj
    cfg = _resolve(cfg)
    paths = cfg["paths"]
    ds_cfg = cfg["datasets"]["pulsecom"]
    limits = cfg.get("limits", {})
    processed_root = Path(paths["processed_audio"])
    qa_output_dir = Path(paths["qa_output"])
    qa_output_dir.mkdir(parents=True, exist_ok=True)
    seed = limits.get("random_seed", 42)
    # 查找所有已处理 JSON
    json_files = find_processed_jsons(processed_root, ds_cfg)
    max_samples = limits.get("max_samples")
    if max_samples:
        json_files = json_files[:max_samples]
    # 数据集分裂: 70/15/15
    n = len(json_files)
    splits = stratified_group_split(
        json_files,
        label_getter=lambda path: extract_labels(load_jsonc(path))["L3"],
        seed=seed,
    )

    print(f"总样本数: {n}")
    print(f"  train: {len(splits['train'])}  条")
    print(f"  val:   {len(splits['val'])}  条")
    print(f"  test:  {len(splits['test'])}  条")
    print("  split policy: L3-stratified source groups (channel-safe)")
    print()

    start_time = time.time()
    total_ok = 0

    for split_name, split_files in splits.items():
        output_path = qa_output_dir / f"sft_{split_name}.jsonl"
        ok = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for i, json_path in enumerate(split_files):
                record = process_single_json(
                    json_path=json_path,
                    processed_root=processed_root,
                    rng=random.Random(seed + i),
                )
                if record is None:
                    continue
                out_record = {k: v for k, v in record.items() if k != "_meta"}
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
                ok += 1
        total_ok += ok
        print(f"  [{split_name}] {ok}/{len(split_files)} → {output_path.name}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Step 2 完成")
    print(f"  总成功: {total_ok}/{n}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  输出: {qa_output_dir.resolve()}/")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()

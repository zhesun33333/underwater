"""
Underwater acoustic LLM data synthesis — Step 2: QA pair generation (Ship)
Generates 3-turn SFT data:
  Turn 1: Source-based Active/Passive classification → short answer
  Turn 2: Noise source specific type → long answer
  Turn 3: Reasoning rationale → qualitative reasoning text
Supports train/val/test split (70/15/15, unified 3-turn format)
Usage:
  python pipeline_ship_step2_qa.py [--config config_ship.yaml]
"""
import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import yaml
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

# Ship 专用 Turn 2 — 问噪声源类型，不透露子类
_T2_SHIP_OPTIONS = (
    "A. Passive — Ship-radiated noise — Cargo vessel\n"
    "B. Passive — Ship-radiated noise — Cruise ship\n"
    "C. Passive — Ship-radiated noise — Fishing vessel\n"
    "D. Passive — Ship-radiated noise — Naval vessel\n"
    "E. Passive — Ship-radiated noise — Underwater vehicle"
)
T2_TEMPLATES = [
    f"Good — the signal is {{L1}}. Identify which type of vessel or underwater vehicle produced this radiated noise.\nOptions:\n{_T2_SHIP_OPTIONS}",
    f"Confirmed as {{L1}}. Now identify the specific type of noise source.\nOptions:\n{_T2_SHIP_OPTIONS}",
    f"Signal is {{L1}}. What kind of platform emitted this noise?\nOptions:\n{_T2_SHIP_OPTIONS}",
    f"Got it. For this {{L1}} signal, further determine the target type.\nOptions:\n{_T2_SHIP_OPTIONS}",
    f"Understood. What type of target is this radiated noise most likely from?\nOptions:\n{_T2_SHIP_OPTIONS}",
    f"OK. Further classify the noise source of this {{L1}} signal.\nOptions:\n{_T2_SHIP_OPTIONS}",
]
# Turn 3 — Reasoning
T3_PROMPT = "Please briefly explain your reasoning."
# ============================================================
# 答案生成 — 选项字母格式 (方案 A)
# ============================================================

# L3 key → T2 完整选项文本 (Ship passive, 5 选 1)
_T2_PASSIVE_FULL = {
    "cargo": "A. Passive — Ship-radiated noise — Cargo vessel",
    "cruise": "B. Passive — Ship-radiated noise — Cruise ship",
    "fishing": "C. Passive — Ship-radiated noise — Fishing vessel",
    "warship": "D. Passive — Ship-radiated noise — Naval vessel",
    "underwater_target": "E. Passive — Ship-radiated noise — Underwater vehicle",
}

def build_turn2_answer(labels: Dict[str, str]) -> str:
    """Turn 2 答案: 完整选项文本 (A-E for passive)。"""
    l3 = labels["L3"]
    if l3 not in _T2_PASSIVE_FULL:
        raise ValueError(f"unsupported Ship L3 label: {l3!r}")
    return _T2_PASSIVE_FULL[l3]


def build_turn3_answer(labels: Dict[str, str], meta: Dict, rng: random.Random) -> str:
    """Turn 3 答案: 结构化推理文本。"""
    return _build_reasoning(labels["L3"], meta, rng)


# ============================================================
# T3 结构化推理模板
# L1/L2/L3 术语从 shared_terminology.py 统一导入 (single source of truth)
# ============================================================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared_terminology import (
    L1_PASSIVE_TERMS as _L1_PASSIVE_TERMS,
    L2_PASSIVE_TERMS as _L2_PASSIVE_TERMS,
    build_l3_evidence,
)

# T3 句式模板 (3 种变体)
_T3_TEMPLATES = [
    "The signal is {L1} and exhibits {L2}; observable evidence includes {L3}.",
    "Based on its spectral and modulation structure, the signal is {L1} with {L2}; supporting cues are {L3}.",
    "Spectral and temporal analysis indicates {L1} with {L2}, supported by {L3}.",
]


def _build_reasoning(l3: str, meta: Dict, rng: random.Random) -> str:
    """根据 L3 类型生成结构化推理文本: L1描述 + L2特征 + L3术语。"""
    terms = build_l3_evidence(l3, meta, rng)
    l3_text = ", ".join(terms)

    if l3 not in _T2_PASSIVE_FULL:
        raise ValueError(f"unsupported Ship L3 label: {l3!r}")
    l1_text = rng.choice(_L1_PASSIVE_TERMS)
    l2_text = rng.choice(_L2_PASSIVE_TERMS["ship-radiated noise"])

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
# 舰船数据专用: 目录扫描
# ============================================================
def find_processed_ship_jsons(
    processed_root: Path,
    dataset_name: str,
    classes: List[str],
    json_subdir: str,
    json_ext: str,
) -> List[Path]:
    all_files = []
    ds_root = processed_root / dataset_name
    if not ds_root.exists():
        print(f"  [WARN] 处理后数据集目录不存在: {ds_root}")
        return []
    for class_name in classes:
        json_dir = ds_root / class_name / json_subdir
        if not json_dir.exists():
            continue
        for jf in sorted(json_dir.glob(f"*{json_ext}")):
            if not jf.name.startswith("manifest"):
                all_files.append(jf)
    return all_files
# ============================================================
# 单条处理
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
    sample_id = get_id(meta)
    wav_rel = get_wav_path(meta)
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
# ============================================================
# 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ship Step 2: QA pair generation")
    parser.add_argument("--config", default="config_ship.yaml", help="配置文件路径")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
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
    dataset_cfg = cfg["dataset"]
    limits = cfg.get("limits", {})
    processed_root = Path(paths["processed_audio"])
    qa_output_dir = Path(paths["qa_output"])
    qa_output_dir.mkdir(parents=True, exist_ok=True)
    seed = limits.get("random_seed", 42)
    json_files = find_processed_ship_jsons(
        processed_root=processed_root,
        dataset_name=dataset_cfg["name"],
        classes=dataset_cfg["classes"],
        json_subdir=dataset_cfg["json_subdir"],
        json_ext=dataset_cfg["json_ext"],
    )
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
        output_path = qa_output_dir / f"sft_ship_{split_name}.jsonl"
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

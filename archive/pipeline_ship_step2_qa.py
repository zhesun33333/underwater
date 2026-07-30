"""
舰船辐射噪声数据合成管线 — 第二步：QA 对生成
生成三轮对话 SFT 数据:
  Turn 1: 主动/被动判别 → 短答案
  Turn 2: 噪声源具体类型 → 长答案
  Turn 3: 判断依据 → 定性推理文本
支持 train/val/test 分裂 (70/15/15, 统一三轮格式)
用法:
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
# ============================================================
# 模板定义
# ============================================================
T1_TEMPLATES = [
    "请判断这段水声信号是主动发射的还是被动接收的。\n选项：\nA. 主动信号\nB. 被动信号",
    "这段音频中的信号是主动声纳发出的，还是被动监听到的？\n选项：\nA. 主动信号\nB. 被动信号",
    "先判断最基本的问题：这信号是主动式的还是被动式的？\n选项：\nA. 主动信号\nB. 被动信号",
    "请先区分这段水下声信号的主动/被动属性。\n选项：\nA. 主动信号\nB. 被动信号",
    "第一个问题：这个声信号是人为主动发射的，还是目标自身辐射的？\n选项：\nA. 主动信号\nB. 被动信号",
    "从主动探测和被动监听的角度，这段信号属于哪一类？\n选项：\nA. 主动信号\nB. 被动信号",
    "请判断信号来源——是主动声纳系统发射的，还是被动接收到的辐射噪声？\n选项：\nA. 主动信号\nB. 被动信号",
    "先做一个基本判断：该水声信号是主动信号还是被动信号？\n选项：\nA. 主动信号\nB. 被动信号",
]
# Ship 专用 Turn 2 — 问噪声源类型，不透露子类
T2_TEMPLATES = [
    "好的，{L1}。请判断这属于哪类舰船或水下目标的辐射噪声。\n选项：\nA. 被动信号-舰船辐射噪声-货船\nB. 被动信号-舰船辐射噪声-邮轮\nC. 被动信号-舰船辐射噪声-渔船\nD. 被动信号-舰船辐射噪声-军舰\nE. 被动信号-舰船辐射噪声-水下目标",
    "确认是{L1}。接下来识别噪声源的具体类型。\n选项：\nA. 被动信号-舰船辐射噪声-货船\nB. 被动信号-舰船辐射噪声-邮轮\nC. 被动信号-舰船辐射噪声-渔船\nD. 被动信号-舰船辐射噪声-军舰\nE. 被动信号-舰船辐射噪声-水下目标",
    "{L1}。那它是哪类平台发出的？\n选项：\nA. 被动信号-舰船辐射噪声-货船\nB. 被动信号-舰船辐射噪声-邮轮\nC. 被动信号-舰船辐射噪声-渔船\nD. 被动信号-舰船辐射噪声-军舰\nE. 被动信号-舰船辐射噪声-水下目标",
    "收到。对于这个{L1}，请进一步判断目标类型。\n选项：\nA. 被动信号-舰船辐射噪声-货船\nB. 被动信号-舰船辐射噪声-邮轮\nC. 被动信号-舰船辐射噪声-渔船\nD. 被动信号-舰船辐射噪声-军舰\nE. 被动信号-舰船辐射噪声-水下目标",
    "明白了。这种辐射噪声最可能来自什么类型的目标？\n选项：\nA. 被动信号-舰船辐射噪声-货船\nB. 被动信号-舰船辐射噪声-邮轮\nC. 被动信号-舰船辐射噪声-渔船\nD. 被动信号-舰船辐射噪声-军舰\nE. 被动信号-舰船辐射噪声-水下目标",
    "好。请对该被动信号的噪声源做进一步分类。\n选项：\nA. 被动信号-舰船辐射噪声-货船\nB. 被动信号-舰船辐射噪声-邮轮\nC. 被动信号-舰船辐射噪声-渔船\nD. 被动信号-舰船辐射噪声-军舰\nE. 被动信号-舰船辐射噪声-水下目标",
]
# Turn 3 — 要求说明判断依据
T3_PROMPT = "请简要说明你的判断依据。"
# ============================================================
# 答案生成 — 选项字母格式 (方案 A)
# ============================================================

# L3 key → T2 完整选项文本 (Ship passive, 5 选 1)
_T2_PASSIVE_FULL = {
    "cargo": "A. 被动信号-舰船辐射噪声-货船",
    "cruise": "B. 被动信号-舰船辐射噪声-邮轮",
    "fishing": "C. 被动信号-舰船辐射噪声-渔船",
    "warship": "D. 被动信号-舰船辐射噪声-军舰",
    "underwater_target": "E. 被动信号-舰船辐射噪声-水下目标",
}

def get_t1_answer(l1: str) -> str:
    """Turn 1 答案: 完整选项文本。"""
    return "A. 主动信号" if l1 == "主动信号" else "B. 被动信号"


def build_turn2_answer(labels: Dict[str, str]) -> str:
    """Turn 2 答案: 完整选项文本 (A-E for passive)。"""
    l3 = labels["L3"]
    return _T2_PASSIVE_FULL.get(l3, "A. 被动信号-舰船辐射噪声-货船")


def build_turn3_answer(labels: Dict[str, str]) -> str:
    """Turn 3 答案: 结构化推理文本。"""
    return _build_reasoning(labels["L3"])


# ============================================================
# T3 结构化推理模板 (与 pipeline_step2_qa.py 共享定义)
# 格式: L1描述术语 + L2特征术语 + L3区分性术语
# ============================================================

# L1 级描述术语
_L1_ACTIVE_TERMS  = ["主动发射信号", "主动声纳信号", "人为主动发射信号"]
_L1_PASSIVE_TERMS = ["被动接收信号", "被动监听信号", "目标自身辐射信号"]

# L2 级特征术语 (按 L1 分组)
_L2_ACTIVE_TERMS = {
    "探测脉冲类": ["探测脉冲特征", "脉冲探测模式", "声纳探测脉冲特性"],
    "通信类":     ["水声通信特征", "数字通信调制", "通信信号调制模式"],
}
_L2_PASSIVE_TERMS = {
    "舰船辐射噪声": ["舰船辐射噪声特征", "船舶辐射噪声特性", "目标辐射噪声模式"],
}

# L3 级区分性术语 (每类随机选 2-3 个)
# 与 testsite/core/scorer.py 的 _L3_SHOULD 保持同步
_L3_TERMS = {
    "CW":  ["单频", "连续波", "音调不变", "单音调", "频率集中"],
    "LFM": ["线性调频", "线性扫频", "音调线性", "频率滑移", "频带展宽",
            "由低变高", "由高变低"],
    "HFM": ["双曲调频", "双曲扫频", "非线性", "先急后缓", "先快后慢"],
    "2FSK": ["两个音调", "二元调制", "频移键控", "频率跳变", "音调切换"],
    "4FSK": ["四个音调", "四进制", "频移键控", "频率跳变", "多音调交替"],
    "BPSK": ["相位翻转", "相移键控", "二进制", "两个状态", "顿挫感", "音量稳定"],
    "QPSK": ["相移键控", "四个状态", "四进制", "多状态切换", "响度恒定"],
    "OFDM": ["子载波", "多载波", "正交", "频分复用", "并行传输", "密集子载波"],
    "cargo":   ["谐波结构清晰", "轴频", "轴频节律", "低速大型", "大型商船",
                "规律节律", "谐音丰富"],
    "cruise":  ["机械噪声密集", "密集机械", "高速", "多机组", "宽带噪声",
                "中高频段", "客船特征"],
    "fishing": ["小型", "结构简单", "稀疏谐音", "柴油机", "音量较小",
                "低次谐音", "谐音有限"],
    "warship": ["强劲", "大功率", "军用舰船", "多轴推进", "多组谐波",
                "复杂密集", "能量集中低频"],
    "underwater_target": ["水下目标", "平滑均匀", "音量低", "宽带为主",
                          "谐音稀少", "缺乏节律"],
}

# T3 句式模板 (3 种变体)
_T3_TEMPLATES = [
    "该信号为{L1}，具有{L2}，具体表现为{L3}。",
    "从声学特征判断，属于{L1}，{L2}明显，{L3}是其典型标志。",
    "音频分析表明为{L1}，听感上{L2}突出，{L3}。",
]


def _build_reasoning(l3: str) -> str:
    """根据 L3 类型生成结构化推理文本: L1描述 + L2特征 + L3术语。"""
    terms = _L3_TERMS.get(l3)
    if not terms:
        return ""

    k = random.randint(2, min(3, len(terms)))
    l3_text = "、".join(random.sample(terms, k))

    if l3 in ("CW", "LFM", "HFM"):
        l1_text = random.choice(_L1_ACTIVE_TERMS)
        l2_text = random.choice(_L2_ACTIVE_TERMS["探测脉冲类"])
    elif l3 in ("2FSK", "4FSK", "BPSK", "QPSK", "OFDM"):
        l1_text = random.choice(_L1_ACTIVE_TERMS)
        l2_text = random.choice(_L2_ACTIVE_TERMS["通信类"])
    else:
        l1_text = random.choice(_L1_PASSIVE_TERMS)
        l2_text = random.choice(_L2_PASSIVE_TERMS["舰船辐射噪声"])

    template = random.choice(_T3_TEMPLATES)
    return template.replace("{L1}", l1_text).replace("{L2}", l2_text).replace("{L3}", l3_text)


# ============================================================
# 对话构建
# ============================================================
def build_conversations(
    labels: Dict[str, str],
    rng: random.Random,
) -> List[Dict[str, str]]:
    """生成三轮对话: T1(L1) → T2(L2+L3) → T3(推理依据)。"""
    l1 = labels["L1"]
    q1 = rng.choice(T1_TEMPLATES)
    a1 = get_t1_answer(l1)
    q2 = rng.choice(T2_TEMPLATES).replace("{L1}", l1)
    a2 = build_turn2_answer(labels)
    q3 = T3_PROMPT
    a3 = build_turn3_answer(labels)
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
    if labels["L1"] == "未知":
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
    conversations = build_conversations(labels, rng)
    return {
        "id": sample_id,
        "audio": wav_rel,
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
    rng = random.Random(seed)
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
    rng.shuffle(json_files)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    splits = {
        "train": json_files[:train_end],
        "val":   json_files[train_end:val_end],
        "test":  json_files[val_end:],
    }

    print(f"总样本数: {n}")
    print(f"  train: {len(splits['train'])}  条")
    print(f"  val:   {len(splits['val'])}  条")
    print(f"  test:  {len(splits['test'])}  条")
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
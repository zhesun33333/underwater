#!/usr/bin/env python
"""数据集统计卡片生成器 — train/val/test 分别统计。

用法:
    python dataset_stats.py
    python dataset_stats.py --train sft_train.jsonl --val sft_val.jsonl --test sft_test.jsonl
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _extract_gt(meta: dict) -> dict:
    """从处理后 JSON 元数据中提取 L1/L2/L3 (与 testsite 一致的 key 体系)。"""
    cat = meta.get("signal_category", "")
    if cat in ("pulse", "communication"):
        l1, l2 = "active", ("pulse" if cat == "pulse" else "communication")
    elif cat == "radiated_noise":
        l1, l2 = "passive", "ship_noise"
    else:
        return {"L1": "unknown", "L2": "unknown", "L3": "unknown"}

    l3 = (meta.get("sub_type") or meta.get("signal_type")
          or (meta.get("signal_params") or {}).get("original_class_name", "unknown"))
    l3 = str(l3).lower().strip()
    ship_l3_map = {
        "cargo_ship": "cargo", "cruise_ship": "cruise",
        "fishing_boat": "fishing", "warship": "warship",
        "underwater_target": "underwater_target",
    }
    l3 = ship_l3_map.get(l3, l3)
    if cat in ("pulse", "communication"):
        l3 = l3.upper()
    return {"L1": l1, "L2": l2, "L3": l3}


# L3 key → display name
_L3_NAMES = {
    "CW": "CW (Continuous Wave)", "LFM": "LFM (Linear Frequency Modulation)",
    "HFM": "HFM (Hyperbolic Frequency Modulation)",
    "2FSK": "2FSK", "4FSK": "4FSK", "BPSK": "BPSK", "QPSK": "QPSK",
    "OFDM": "OFDM",
    "cargo": "Cargo vessel", "cruise": "Cruise ship", "fishing": "Fishing vessel",
    "warship": "Naval vessel", "underwater_target": "Underwater target",
}

_L1_NAMES = {"active": "Actively transmitted signal", "passive": "Passively received signal"}
_L2_NAMES = {"pulse": "Detection pulse", "communication": "Communication signal",
             "ship_noise": "Ship-radiated noise"}


def _find_meta(audio_base: Path, audio_rel: str, sample_id: str) -> dict:
    """查找 sample 对应的处理后 JSON 元数据（直接路径优先，命中则不搜索）。"""
    parts = Path(audio_rel).parts

    # 直接路径候选（从 audio_rel 结构推导，绝大多数情况命中）
    direct = None
    if "PulseCom" in audio_rel and len(parts) >= 3:
        sig_type = parts[-2]
        direct = audio_base / "PulseCom" / "jsonc" / sig_type / f"{sample_id}.jsonc"
    elif "05_ship_radiated_noise" in audio_rel and len(parts) >= 3:
        class_name = parts[-3]
        direct = audio_base / "05_ship_radiated_noise" / class_name / "json" / f"{sample_id}.json"

    if direct is not None and direct.exists():
        try:
            return json.loads(direct.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, Exception):
            pass

    # 兜底: rglob 搜索（仅直接路径失败时触发）
    for ext in (".jsonc", ".json"):
        for m in audio_base.rglob(f"{sample_id}{ext}"):
            if m == direct:
                continue
            try:
                return json.loads(m.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, Exception):
                continue
    return {}


def _extract_quality(meta: dict) -> tuple:
    """Return (tl_db, snr_db) — PulseCom has tl_db, Ship has snr_db."""
    bo = meta.get("bellhop_output", {})
    return (bo.get("tl_db"), bo.get("snr_db"))


def _extract_ssp_complexity(meta: dict) -> float:
    ssp = meta.get("bellhop_env", {}).get("ssp", {})
    depths = ssp.get("depths_m")
    speeds = ssp.get("sound_speeds_mps")
    if not depths or not speeds or len(depths) <= 1:
        return None
    grads = [(speeds[i+1] - speeds[i]) / (depths[i+1] - depths[i])
             for i in range(len(depths) - 1)]
    mean_g = sum(grads) / len(grads)
    return sum((g - mean_g) ** 2 for g in grads) / len(grads)


def _load_split(jsonl_path: str, audio_base: Path, label: str = "") -> list:
    """加载一个 split 的所有样本数据。"""
    samples = []
    total = 0
    skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = item.get("id", "")
            arel = item.get("audio", "")
            if not sid or not arel:
                continue
            total += 1
            meta = _find_meta(audio_base, arel, sid)
            gt = _extract_gt(meta) if meta else {"L1": "unknown", "L2": "unknown", "L3": "unknown"}
            if gt.get("L1") == "unknown":
                skipped += 1
                continue
            tl_db, snr_db = _extract_quality(meta) if meta else (None, None)
            samples.append({
                "id": sid,
                "l1": gt["L1"],
                "l2": gt["L2"],
                "l3": gt["L3"],
                "tl_db": tl_db,
                "snr_db": snr_db,
                "ssp_complexity": _extract_ssp_complexity(meta) if meta else None,
            })
            if total % 50 == 0:
                print(f"  [{label}] processed {total} rows, valid {len(samples)} ...")
        if skipped > 0:
            print(f"  [{label}] skipped {skipped} (unknown GT)")
    return samples


def _snr_stats(snr_values: list) -> dict:
    valid = [s for s in snr_values if s is not None]
    if not valid:
        return {}
    sorted_snr = sorted(valid)
    n = len(sorted_snr)
    return {
        "count": n,
        "min": sorted_snr[0],
        "max": sorted_snr[-1],
        "mean": sum(sorted_snr) / n,
        "median": sorted_snr[n // 2],
        "bins": {
            "≥15dB": sum(1 for s in valid if s >= 15),
            "5-15dB": sum(1 for s in valid if 5 <= s < 15),
            "-5-5dB": sum(1 for s in valid if -5 <= s < 5),
            "≤-5dB": sum(1 for s in valid if s < -5),
        },
    }


def _ssp_stats(ssp_values: list) -> dict:
    valid = [s for s in ssp_values if s is not None]
    if not valid:
        return {}
    sorted_ssp = sorted(valid)
    n = len(sorted_ssp)
    return {
        "count": n,
        "min": round(sorted_ssp[0], 4),
        "max": round(sorted_ssp[-1], 4),
        "mean": round(sum(sorted_ssp) / n, 4),
        "median": round(sorted_ssp[n // 2], 4),
    }


def _build_section(name: str, samples: list) -> str:
    n = len(samples)
    if n == 0:
        return f"## {name}\n\n(no samples)\n"

    l1_counts = Counter(s["l1"] for s in samples)
    l2_counts = Counter(s["l2"] for s in samples)
    l3_counts = Counter(s["l3"] for s in samples)
    # PulseCom TL (active L1) vs Ship SNR (passive L1)
    tl_info = _snr_stats([s["tl_db"] for s in samples if s["l1"] == "active"])
    snr_info = _snr_stats([s["snr_db"] for s in samples if s["l1"] == "passive"])
    ssp_info = _ssp_stats([s["ssp_complexity"] for s in samples])

    lines = [
        f"## {name} ({n} samples)\n",
        "### L1 Distribution",
        "| L1 Class | Count | Ratio |",
        "|---------|--------|------|",
    ]
    for key in ["active", "passive"]:
        cnt = l1_counts.get(key, 0)
        lines.append(f"| {_L1_NAMES.get(key, key)} | {cnt} | {cnt/n:.1%} |")
    lines.append("")

    lines.extend([
        "### L2 Distribution",
        "| L2 Class | Count | Ratio |",
        "|---------|--------|------|",
    ])
    for key in ["pulse", "communication", "ship_noise"]:
        cnt = l2_counts.get(key, 0)
        if cnt > 0:
            lines.append(f"| {_L2_NAMES.get(key, key)} | {cnt} | {cnt/n:.1%} |")
    lines.append("")

    lines.extend([
        "### L3 Distribution",
        "| L3 Class | Count | Ratio |",
        "|---------|--------|------|",
    ])
    for key in sorted(l3_counts.keys()):
        cnt = l3_counts[key]
        name = _L3_NAMES.get(key, key)
        lines.append(f"| {name} | {cnt} | {cnt/n:.1%} |")
    lines.append("")

    if tl_info:
        unit = "dB"
        lines.extend([
            "### TL Distribution (PulseCom — Transmission Loss)",
            "| Statistic | Value |",
            "|--------|-----|",
            f"| Valid samples | {tl_info['count']} |",
            f"| Min | {tl_info['min']:.1f} {unit} |",
            f"| Max | {tl_info['max']:.1f} {unit} |",
            f"| Mean | {tl_info['mean']:.1f} {unit} |",
            f"| Median | {tl_info['median']:.1f} {unit} |",
            "",
            "| TL Bin | Count | Ratio |",
            "|----------|--------|------|",
        ])
        for label in ["≥15dB", "5-15dB", "-5-5dB", "≤-5dB"]:
            cnt = tl_info["bins"].get(label, 0)
            lines.append(f"| {label} | {cnt} | {cnt/tl_info['count']:.1%} |")
        lines.append("")

    if snr_info:
        unit = "dB"
        lines.extend([
            "### SNR Distribution (Ship — Line-spectrum SNR)",
            "| Statistic | Value |",
            "|--------|-----|",
            f"| Valid samples | {snr_info['count']} |",
            f"| Min | {snr_info['min']:.1f} {unit} |",
            f"| Max | {snr_info['max']:.1f} {unit} |",
            f"| Mean | {snr_info['mean']:.1f} {unit} |",
            f"| Median | {snr_info['median']:.1f} {unit} |",
            "",
            "| SNR Bin | Count | Ratio |",
            "|----------|--------|------|",
        ])
        for label in ["≥15dB", "5-15dB", "-5-5dB", "≤-5dB"]:
            cnt = snr_info["bins"].get(label, 0)
            lines.append(f"| {label} | {cnt} | {cnt/snr_info['count']:.1%} |")
        lines.append("")

    if ssp_info:
        lines.extend([
            "### SSP Complexity Distribution",
            "| Statistic | Value |",
            "|--------|-----|",
            f"| Valid samples | {ssp_info['count']} |",
            f"| Min | {ssp_info['min']} |",
            f"| Max | {ssp_info['max']} |",
            f"| Mean | {ssp_info['mean']} |",
            f"| Median | {ssp_info['median']} |",
            "",
        ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Dataset statistics card generator")
    parser.add_argument("--train", default="sft_train.jsonl", help="Training set JSONL")
    parser.add_argument("--val", default="sft_val.jsonl", help="Validation set JSONL")
    parser.add_argument("--test", default="sft_test.jsonl", help="Test set JSONL")
    parser.add_argument("--audio-root", default="processed_audio", help="Processed audio root")
    parser.add_argument("--output", default="dataset_stats.md", help="Output file")
    args = parser.parse_args()

    audio_base = Path(args.audio_root)

    print("=" * 50)
    print("  Dataset Statistics Card Generator")
    print("=" * 50)

    all_sections = []
    total = 0

    for label, path in [("Train", args.train), ("Val", args.val), ("Test", args.test)]:
        jp = Path(path)
        if not jp.exists():
            print(f"  [SKIP] {label}: {path} not found")
            continue
        print(f"  Loading {label}: {path}")
        samples = _load_split(str(jp), audio_base, label=label)
        print(f"    Valid samples: {len(samples)}")
        total += len(samples)
        all_sections.append(_build_section(label, samples))

    if total == 0:
        print("Error: no valid samples")
        return 1

    # 全局汇总
    summary_lines = [
        f"# Dataset Statistics\n",
        f"**Total samples**: {total}\n",
    ]
    report = "\n".join(summary_lines + all_sections)

    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"\n  Stats card saved to: {output_path.resolve()}")
    print(f"{'=' * 50}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

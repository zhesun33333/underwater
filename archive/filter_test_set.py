"""
从测试集中筛选高质量子集，构建 baseline 评估用的小型测试集。

筛选策略:
  PulseCom: TL (传播损失) 越高越好，信道畸变越小信号越清晰
  Ship:     SNR (线谱/背景比) 越高越好，线谱特征越明显
  每类平衡采样，避免分类偏差

用法:
  python filter_test_set.py [--input dataset/sft_test.jsonl] [--n-per-class 200]
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

PROCESSED_ROOT = Path("processed_audio")

# L3 class display names
L3_NAMES = {
    "CW": "CW (Continuous Wave)", "LFM": "LFM (Linear Frequency Modulation)",
    "HFM": "HFM (Hyperbolic Frequency Modulation)",
    "2FSK": "2FSK", "4FSK": "4FSK", "BPSK": "BPSK", "QPSK": "QPSK", "OFDM": "OFDM",
    "cargo": "Cargo vessel", "cruise": "Cruise ship", "fishing": "Fishing vessel",
    "warship": "Naval vessel", "underwater_target": "Underwater target",
}


def select_quality_tier(pool: list, tier: str) -> list:
    """Select a dynamic quality tertile; larger scores mean better quality."""
    ranked = sorted(pool, key=lambda item: item[1])
    if tier == "all" or len(ranked) < 3:
        return ranked

    low_end = len(ranked) // 3
    high_start = (2 * len(ranked)) // 3
    if tier == "low":
        return ranked[:low_end]
    if tier == "mid":
        return ranked[low_end:high_start]
    if tier == "high":
        return ranked[high_start:]
    raise ValueError(f"unknown quality tier: {tier}")


def find_meta(audio_base: Path, audio_rel: str, sample_id: str) -> Optional[dict]:
    """定位 processed JSON 元数据 (与 dataset_stats.py 一致)。"""
    parts = Path(audio_rel).parts
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

    for ext in (".jsonc", ".json"):
        for m in audio_base.rglob(f"{sample_id}{ext}"):
            try:
                return json.loads(m.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, Exception):
                continue
    return None


def extract_quality_and_gt(meta: dict) -> tuple:
    """返回 (category, l3_key, quality_score, gt_dict, meta_dict)。
    PulseCom: quality = TL (传播损失, dB), 越大信号越清晰
    Ship:     quality = SNR (线谱/背景比, dB), 越大特征越明显
    gt_dict 跟 testsite extract_l1_l2_l3_from_meta 返回格式一致
    """
    cat = meta.get("signal_category", "unknown")
    bo = meta.get("bellhop_output", {})
    be = meta.get("bellhop_env", {})
    ssp = be.get("ssp", {})

    meta_out = {}

    if cat in ("pulse", "communication"):
        tl = bo.get("tl_db")
        l3 = str(meta.get("signal_type", "unknown")).upper()
        quality = float(tl) if tl is not None else -400.0
        if tl is not None:
            meta_out["tl_db"] = float(tl)
        gt = {"L1": "active", "L2": "pulse" if cat == "pulse" else "communication", "L3": l3}

    elif cat == "radiated_noise":
        snr = bo.get("snr_db")
        sub = meta.get("sub_type", "unknown")
        ship_l3_map = {
            "cargo": "cargo", "cruise": "cruise", "fishing": "fishing",
            "warship": "warship", "underwater_target": "underwater_target",
        }
        l3 = ship_l3_map.get(sub, "unknown")
        quality = float(snr) if snr is not None else -400.0
        if snr is not None:
            meta_out["snr_db"] = float(snr)
        gt = {"L1": "passive", "L2": "ship_noise", "L3": l3}

    else:
        return ("unknown", "unknown", -400.0, {"L1": "unknown", "L2": "unknown", "L3": "unknown"}, {})

    # SSP complexity (for both categories)
    depths = ssp.get("depths_m")
    speeds = ssp.get("sound_speeds_mps")
    if depths and speeds and len(depths) > 1:
        grads = [(speeds[i+1] - speeds[i]) / (depths[i+1] - depths[i]) for i in range(len(depths) - 1)]
        mean_g = sum(grads) / len(grads)
        meta_out["ssp_complexity"] = round(sum((g - mean_g) ** 2 for g in grads) / len(grads), 8)

    return (cat, l3, quality, gt, meta_out)


def main():
    parser = argparse.ArgumentParser(description="Filter high-quality test set")
    parser.add_argument("--input", default="dataset/sft_test.jsonl")
    parser.add_argument("--output", default="dataset/sft_test_highquality.jsonl")
    parser.add_argument("--n-per-class", type=int, default=200,
                        help="Max samples per L3 class (default: 200)")
    parser.add_argument("--quality-tier", choices=("high", "mid", "low", "all"),
                        default="high", help="Dynamic per-class quality tertile (default: high)")
    parser.add_argument("--tl-min", type=float, default=None,
                        help="Optional PulseCom raw channel-gain floor in dB")
    parser.add_argument("--snr-min", type=float, default=None,
                        help="Optional Ship SNR floor in dB")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    jsonl_path = Path(args.input)
    if not jsonl_path.exists():
        print(f"ERROR: input file not found: {jsonl_path}")
        return 1

    # ---- 第一遍: 收集所有样本的质量分数和 GT ----
    print(f"Loading: {jsonl_path}")
    by_l3: dict[str, list] = {}  # l3_key → [(record, quality, gt, meta_out), ...]
    total = 0
    skipped_no_meta = 0
    skipped_low_quality = 0

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

            meta = find_meta(PROCESSED_ROOT, arel, sid)
            if meta is None:
                skipped_no_meta += 1
                continue

            try:
                cat, l3, quality, gt, meta_out = extract_quality_and_gt(meta)
            except (ValueError, ZeroDivisionError, TypeError) as e:
                skipped_no_meta += 1
                continue
            if l3 == "unknown" or gt.get("L1") == "unknown":
                skipped_no_meta += 1
                continue

            # 质量过滤
            if (cat in ("pulse", "communication") and args.tl_min is not None
                    and quality < args.tl_min):
                skipped_low_quality += 1
                continue
            if (cat == "radiated_noise" and args.snr_min is not None
                    and quality < args.snr_min):
                skipped_low_quality += 1
                continue

            if l3 not in by_l3:
                by_l3[l3] = []
            by_l3[l3].append((item, quality, gt, meta_out))

    print(f"  Total: {total}")
    print(f"  Skipped (no meta/unknown): {skipped_no_meta}")
    print(f"  Below quality threshold: {skipped_low_quality}")
    print(f"  Passed filter: {sum(len(v) for v in by_l3.values())}")

    # ---- 第二遍: 每类按质量降序排序, 取前 N ----
    rng = random.Random(args.seed)
    selected = []

    print(f"\nFilter results ({args.quality_tier} tertile, max {args.n_per_class} per class):")
    print(f"{'L3 Class':<20} {'Avail':>6} {'Tier':>6} {'Picked':>6} {'Quality Range':>20}")
    print("-" * 64)

    for l3_key in sorted(by_l3.keys()):
        pool = by_l3[l3_key]
        # 按质量降序排 (质量越高越好)
        tier_pool = select_quality_tier(pool, args.quality_tier)
        tier_pool.sort(key=lambda x: x[1], reverse=True)

        # 取前 N, 但留一点随机性: 从前 2*N 中随机选 N
        top_n = min(args.n_per_class * 2, len(tier_pool))
        candidates = tier_pool[:top_n]
        chosen = rng.sample(candidates, min(args.n_per_class, len(candidates)))

        qualities = [c[1] for c in chosen]
        name = L3_NAMES.get(l3_key, l3_key)
        print(f"{name:<20} {len(pool):>6} {len(tier_pool):>6} {len(chosen):>6}  "
              f"{min(qualities):.1f} ~ {max(qualities):.1f}")

        for item, _, gt, meta_out in chosen:
            out_record = {k: v for k, v in item.items() if k != "_meta"}
            out_record["_gt"] = gt
            out_record["_meta"] = meta_out
            selected.append(out_record)

    print("-" * 64)
    print(f"{'Total':<20} {sum(len(v) for v in by_l3.values()):>6} "
          f"{sum(len(select_quality_tier(v, args.quality_tier)) for v in by_l3.values()):>6} "
          f"{len(selected):>6}")

    # ---- 输出 ----
    rng.shuffle(selected)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for rec in selected:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\nOutput: {output_path.resolve()}")
    print(f"Format: compatible with testsite evaluation")
    return 0


if __name__ == "__main__":
    sys.exit(main())

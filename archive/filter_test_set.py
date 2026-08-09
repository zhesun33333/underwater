"""Build a favorable-condition, class-balanced diagnostic test subset.

PulseCom is ranked by its pre-normalization channel energy gain (legacy
``tl_db``); Ship is ranked by source/pre-channel line-spectrum SNR. A fixed
seed samples each L3 from its highest-ranked candidate pool.

Paper-scope note:
  This subset is designed to test basic recognition under comparatively
  favorable conditions. It must not be described as measuring robustness to
  severe propagation, low observability, or the full operational domain.
  PulseCom gain is measured before peak normalization and is not a direct
  measure of post-normalization perceptual clarity.
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


def build_output_record(entry: tuple, rank: int, pool_size: int,
                        candidate_pool_size: int) -> dict:
    item, quality, gt, meta_out = entry
    output_meta = dict(meta_out)
    output_meta.update({
        "quality_score": float(quality),
        "quality_rank_within_l3": rank,
        "quality_pool_size_within_l3": pool_size,
        "candidate_pool_size_within_l3": candidate_pool_size,
        "selection_policy": "class_balanced_favorable_candidate_pool",
    })
    out_record = {key: value for key, value in item.items() if key != "_meta"}
    out_record["_gt"] = gt
    out_record["_meta"] = output_meta
    return out_record


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
    PulseCom: quality = 归一化前信道能量增益 (dB)，越大表示衰减越弱
    Ship:     quality = SNR (线谱/背景比, dB), 越大特征越明显
    gt_dict 跟 testsite extract_l1_l2_l3_from_meta 返回格式一致

    PulseCom quality is a favorable-condition ranking variable, not a direct
    post-normalization clarity score. Do not use this subset to claim channel
    robustness or perceptual-quality coverage.
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
            meta_out["quality_metric"] = "pre_normalization_channel_energy_gain_db"
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
            meta_out["quality_metric"] = "source_pre_channel_line_spectrum_snr_db"
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
    parser = argparse.ArgumentParser(description="Build a class-balanced benchmark test subset")
    parser.add_argument("--input", default="dataset/sft_test.jsonl")
    parser.add_argument("--output", default="dataset/sft_test_highquality.jsonl")
    parser.add_argument("--n-per-class", type=int, default=200,
                        help="Required samples per L3 class (default: 200)")
    parser.add_argument(
        "--candidate-multiplier", type=float, default=2.0,
        help=("Candidate-pool size relative to --n-per-class before deterministic "
              "sampling (default: 2.0)"),
    )
    parser.add_argument("--tl-min", type=float, default=None,
                        help="Optional PulseCom raw channel-gain floor in dB")
    parser.add_argument("--snr-min", type=float, default=None,
                        help="Optional Ship SNR floor in dB")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.n_per_class <= 0:
        parser.error("--n-per-class must be positive")
    if args.candidate_multiplier < 1.0:
        parser.error("--candidate-multiplier must be at least 1.0")

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
    missing_classes = sorted(set(L3_NAMES) - set(by_l3))
    if missing_classes:
        print(f"ERROR: missing required L3 classes: {missing_classes}")
        return 1

    # Paper wording guardrail: call this a favorable-condition diagnostic
    # subset, not a robustness benchmark or a full-domain representative set.
    # Describe selection as "deterministic sampling from a high-quality
    # candidate pool", not as retaining the strict top-N samples.
    # ---- 第二遍: 每类从高分候选池中确定性抽样 ----
    rng = random.Random(args.seed)
    selected = []
    shortfalls = {}

    print(f"\nSelection results (favorable candidate pool, {args.n_per_class} per class):")
    print(f"{'L3 Class':<35} {'Avail':>7} {'Candidates':>11} {'Picked':>8}")
    print("-" * 65)

    for l3_key in sorted(by_l3.keys()):
        pool = by_l3[l3_key]
        ranked = sorted(
            pool,
            key=lambda entry: (-entry[1], str(entry[0].get("id", ""))),
        )
        rank_by_id = {
            str(entry[0].get("id", "")): rank
            for rank, entry in enumerate(ranked, 1)
        }
        candidate_count = min(
            len(ranked),
            max(args.n_per_class, int(round(args.n_per_class * args.candidate_multiplier))),
        )
        candidates = ranked[:candidate_count]
        chosen = rng.sample(candidates, min(args.n_per_class, len(candidates)))
        chosen_count = len(chosen)
        if chosen_count != args.n_per_class:
            shortfalls[l3_key] = {
                "required": args.n_per_class,
                "selected": chosen_count,
                "available": len(pool),
            }
        name = L3_NAMES.get(l3_key, l3_key)
        print(
            f"{name:<35} {len(pool):>7} {candidate_count:>11} {chosen_count:>8}"
        )
        for entry in chosen:
            sample_id = str(entry[0].get("id", ""))
            selected.append(build_output_record(
                entry,
                rank=rank_by_id[sample_id],
                pool_size=len(pool),
                candidate_pool_size=candidate_count,
            ))

    print("-" * 65)
    print(
        f"{'Total':<35} {sum(len(v) for v in by_l3.values()):>7} "
        f"{'':>11} {len(selected):>8}"
    )
    if shortfalls:
        print(f"ERROR: insufficient samples for requested formal subset: {shortfalls}")
        return 1

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

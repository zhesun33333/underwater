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

# L3 类别的友好名称
L3_NAMES = {
    "CW": "CW连续波", "LFM": "LFM线性调频", "HFM": "HFM双曲调频",
    "2FSK": "2FSK", "4FSK": "4FSK", "BPSK": "BPSK", "QPSK": "QPSK", "OFDM": "OFDM",
    "cargo": "货船", "cruise": "邮轮", "fishing": "渔船",
    "warship": "军舰", "underwater_target": "水下目标",
}


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
    PulseCom: quality = TL
    Ship:     quality = SNR
    gt_dict 跟 testsite extract_l1_l2_l3_from_meta 返回格式一致
    """
    cat = meta.get("signal_category", "unknown")
    bo = meta.get("bellhop_output", {})
    be = meta.get("bellhop_env", {})
    ssp = be.get("ssp", {})

    # --- 提取 SNR / TL ---
    if cat in ("pulse", "communication"):
        tl = bo.get("estimated_transmission_loss_db")
        l3 = str(meta.get("signal_type", "unknown")).upper()
        quality = float(tl) if tl is not None else -400.0
        gt = {"L1": "active", "L2": "pulse" if cat == "pulse" else "communication", "L3": l3}

    elif cat == "radiated_noise":
        snr = bo.get("snr_after_channel_db")
        sub = meta.get("sub_type", "unknown")
        ship_l3_map = {
            "cargo": "cargo", "cruise": "cruise", "fishing": "fishing",
            "warship": "warship", "underwater_target": "underwater_target",
        }
        l3 = ship_l3_map.get(sub, "unknown")
        quality = float(snr) if snr is not None else -400.0
        gt = {"L1": "passive", "L2": "ship_noise", "L3": l3}

    else:
        return ("unknown", "unknown", -400.0, {"L1": "unknown", "L2": "unknown", "L3": "unknown"}, {})

    # --- 提取质量元数据 (用于 SNR/SSP 分层分析) ---
    snr_val = bo.get("snr_after_channel_db")
    meta_out = {}
    if snr_val is not None:
        meta_out["snr_db"] = float(snr_val)

    depths = ssp.get("depths_m")
    speeds = ssp.get("sound_speeds_mps")
    if depths and speeds and len(depths) > 1:
        grads = [(speeds[i+1] - speeds[i]) / (depths[i+1] - depths[i]) for i in range(len(depths) - 1)]
        mean_g = sum(grads) / len(grads)
        meta_out["ssp_complexity"] = round(sum((g - mean_g) ** 2 for g in grads) / len(grads), 8)

    return (cat, l3, quality, gt, meta_out)


def main():
    parser = argparse.ArgumentParser(description="筛选高质量测试集")
    parser.add_argument("--input", default="dataset/sft_test.jsonl")
    parser.add_argument("--output", default="dataset/sft_test_highquality.jsonl")
    parser.add_argument("--n-per-class", type=int, default=200,
                        help="每个 L3 类别最多采样数 (默认 200, 总计 ~2600)")
    parser.add_argument("--tl-min", type=float, default=-15.0,
                        help="PulseCom 最低 TL 阈值 (dB), 默认 -15")
    parser.add_argument("--snr-min", type=float, default=-18.0,
                        help="Ship 最低 SNR 阈值 (dB), 默认 -18")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    jsonl_path = Path(args.input)
    if not jsonl_path.exists():
        print(f"错误: 输入文件不存在: {jsonl_path}")
        return 1

    # ---- 第一遍: 收集所有样本的质量分数和 GT ----
    print(f"加载: {jsonl_path}")
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

            cat, l3, quality, gt, meta_out = extract_quality_and_gt(meta)
            if l3 == "unknown" or gt.get("L1") == "unknown":
                skipped_no_meta += 1
                continue

            # 质量过滤
            if cat in ("pulse", "communication") and quality < args.tl_min:
                skipped_low_quality += 1
                continue
            if cat == "radiated_noise" and quality < args.snr_min:
                skipped_low_quality += 1
                continue

            if l3 not in by_l3:
                by_l3[l3] = []
            by_l3[l3].append((item, quality, gt, meta_out))

    print(f"  总样本: {total}")
    print(f"  无元数据/未知: {skipped_no_meta}")
    print(f"  质量不达标: {skipped_low_quality}")
    print(f"  通过筛选: {sum(len(v) for v in by_l3.values())}")

    # ---- 第二遍: 每类按质量降序排序, 取前 N ----
    rng = random.Random(args.seed)
    selected = []
    stats_lines = []

    print(f"\n筛选结果 (每类最多 {args.n_per_class} 条):")
    print(f"{'L3 类别':<20} {'可用':>6} {'已选':>6} {'质量范围':>20}")
    print("-" * 56)

    for l3_key in sorted(by_l3.keys()):
        pool = by_l3[l3_key]
        # 按质量降序排 (质量越高越好)
        pool.sort(key=lambda x: x[1], reverse=True)

        # 取前 N, 但留一点随机性: 从前 2*N 中随机选 N
        top_n = min(args.n_per_class * 2, len(pool))
        candidates = pool[:top_n]
        chosen = rng.sample(candidates, min(args.n_per_class, len(candidates)))

        qualities = [c[1] for c in chosen]
        name = L3_NAMES.get(l3_key, l3_key)
        print(f"{name:<20} {len(pool):>6} {len(chosen):>6}  {min(qualities):.1f} ~ {max(qualities):.1f}")

        for item, _, gt, meta_out in chosen:
            out_record = {k: v for k, v in item.items() if k != "_meta"}
            out_record["_gt"] = gt
            out_record["_meta"] = meta_out
            selected.append(out_record)

    print("-" * 56)
    print(f"{'合计':<20} {sum(len(v) for v in by_l3.values()):>6} {len(selected):>6}")

    # ---- 输出 ----
    rng.shuffle(selected)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for rec in selected:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n输出: {output_path.resolve()}")
    print(f"格式: 与输入 JSONL 一致, 可直接用于 testsite 评估")
    return 0


if __name__ == "__main__":
    sys.exit(main())

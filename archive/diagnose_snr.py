"""
SNR 诊断脚本：区分 PulseCom vs Ship，检查极端值来源和 TL 关联
"""
import json
import sys
import random
from pathlib import Path
from collections import Counter
import numpy as np

PROCESSED_ROOT = Path("processed_audio")
N_SAMPLE = 500  # 每类随机抽 500 个

def sample_jsons(pattern_dir: str, n: int):
    """从 processed_audio 下的指定目录随机抽样 JSON/JSONC"""
    json_dir = PROCESSED_ROOT / pattern_dir
    if not json_dir.exists():
        return []
    files = list(json_dir.rglob("*.jsonc")) + list(json_dir.rglob("*.json"))
    if len(files) <= n:
        return files
    rng = random.Random(42)
    return rng.sample(files, n)

def main():
    # ---- PulseCom 抽样 ----
    pc_files = []
    for sig_type in ["CW", "LFM", "HFM", "2FSK", "4FSK", "BPSK", "QPSK", "OFDM"]:
        json_dir = PROCESSED_ROOT / "PulseCom" / "jsonc" / sig_type
        if json_dir.exists():
            pc_files.extend(list(json_dir.glob("*.jsonc")))
    rng = random.Random(42)
    if len(pc_files) > N_SAMPLE:
        pc_files = rng.sample(pc_files, N_SAMPLE)

    # ---- Ship 抽样 ----
    ship_files = []
    for cls_name in ["cargo_ship", "cruise_ship", "fishing_boat", "warship", "underwater_target"]:
        json_dir = PROCESSED_ROOT / "05_ship_radiated_noise" / cls_name / "json"
        if json_dir.exists():
            ship_files.extend(list(json_dir.glob("*.json")))
    if len(ship_files) > N_SAMPLE:
        ship_files = rng.sample(ship_files, N_SAMPLE)

    # ---- 提取 SNR ----
    def extract_snr_and_tl(json_path):
        try:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        bo = meta.get("bellhop_output", {})
        snr = bo.get("snr_after_channel_db")
        tl = bo.get("estimated_transmission_loss_db")
        arrivals = bo.get("summary", {}).get("num_arrivals", 0)
        snr_orig = meta.get("snr_after_mix_db")  # ship 原始 SNR
        signal_cat = meta.get("signal_category", "unknown")
        return {
            "snr": snr,
            "tl": tl,
            "arrivals": arrivals,
            "snr_orig": snr_orig,
            "cat": signal_cat,
            "id": meta.get("id", "?"),
        }

    print("=" * 60)
    print("SNR 诊断 — PulseCom vs Ship 对比")
    print("=" * 60)

    for label, files in [("PulseCom", pc_files), ("Ship", ship_files)]:
        records = []
        for f in files:
            r = extract_snr_and_tl(f)
            if r and r["snr"] is not None:
                records.append(r)

        snrs = [r["snr"] for r in records]
        tls = [r["tl"] for r in records if r["tl"] is not None]
        arrivals = [r["arrivals"] for r in records]
        snr_orig_vals = [r["snr_orig"] for r in records if r["snr_orig"] is not None]

        print(f"\n{'─' * 40}")
        print(f"{label}: {len(records)} 个样本")
        print(f"  SNR: min={min(snrs):.1f}  max={max(snrs):.1f}  median={np.median(snrs):.1f}  mean={np.mean(snrs):.1f}")
        if tls:
            print(f"  TL:  min={min(tls):.1f}  max={max(tls):.1f}  median={np.median(tls):.1f}  mean={np.mean(tls):.1f}")
        if arrivals:
            arr_counter = Counter(arrivals)
            zero_arr = arr_counter.get(0, 0)
            print(f"  到达数: min={min(arrivals)}  max={max(arrivals)}  N_arrivals=0: {zero_arr} 个")
        if snr_orig_vals:
            print(f"  原始 SNR (snr_after_mix_db): min={min(snr_orig_vals):.1f}  max={max(snr_orig_vals):.1f}  median={np.median(snr_orig_vals):.1f}")

        # 极端值
        extreme = [r for r in records if r["snr"] < -100]
        if extreme:
            print(f"\n  SNR < -100 dB 的极端样本 ({len(extreme)} 个):")
            for r in extreme[:5]:
                print(f"    {r['id']}: SNR={r['snr']:.1f}  TL={r['tl']:.1f}  arrivals={r['arrivals']}")

        # 按 SNR 区间分布
        bins = [("≥15dB", lambda s: s >= 15), ("5-15dB", lambda s: 5 <= s < 15),
                ("0-5dB", lambda s: 0 <= s < 5), ("-5-0dB", lambda s: -5 <= s < 0),
                ("-20~-5dB", lambda s: -20 <= s < -5), ("<-20dB", lambda s: s < -20)]
        print(f"\n  SNR 区间分布:")
        for label_b, fn in bins:
            cnt = sum(1 for s in snrs if fn(s))
            print(f"    {label_b}: {cnt} ({cnt/len(snrs)*100:.1f}%)")

    # ---- 关联分析: TL vs SNR (仅 PulseCom) ----
    print(f"\n{'=' * 60}")
    print("关联分析: TL vs SNR (PulseCom)")

    pc_records = [extract_snr_and_tl(f) for f in pc_files]
    pc_records = [r for r in pc_records if r and r["snr"] is not None and r["tl"] is not None]

    if pc_records:
        tls = np.array([r["tl"] for r in pc_records])
        snrs = np.array([r["snr"] for r in pc_records])
        corr = np.corrcoef(tls, snrs)[0, 1]
        print(f"  TL vs SNR 相关系数: {corr:.3f}")

        # TL 分桶看 SNR
        tl_bins = [(">0dB", lambda t: t > 0), ("-20~0dB", lambda t: -20 < t <= 0),
                   ("-40~-20dB", lambda t: -40 < t <= -20), ("<-40dB", lambda t: t <= -40)]
        for label_b, fn in tl_bins:
            idx = [i for i, t in enumerate(tls) if fn(t)]
            if idx:
                bin_snrs = snrs[idx]
                print(f"  TL {label_b}: SNR median={np.median(bin_snrs):.1f}  mean={np.mean(bin_snrs):.1f}  n={len(bin_snrs)}")


if __name__ == "__main__":
    main()

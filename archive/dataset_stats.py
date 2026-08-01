#!/usr/bin/env python
"""Generate train/validation/test dataset statistics as Markdown."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


_L3_NAMES = {
    "CW": "CW (Continuous Wave)",
    "LFM": "LFM (Linear Frequency Modulation)",
    "HFM": "HFM (Hyperbolic Frequency Modulation)",
    "2FSK": "2FSK",
    "4FSK": "4FSK",
    "BPSK": "BPSK",
    "QPSK": "QPSK",
    "OFDM": "OFDM",
    "cargo": "Cargo vessel",
    "cruise": "Cruise ship",
    "fishing": "Fishing vessel",
    "warship": "Naval vessel",
    "underwater_target": "Underwater target",
}
_L1_NAMES = {
    "active": "Actively transmitted signal",
    "passive": "Passively received signal",
}
_L2_NAMES = {
    "pulse": "Detection pulse",
    "communication": "Communication signal",
    "ship_noise": "Ship-radiated noise",
}


def _extract_gt(meta: dict) -> dict:
    """Extract the L1/L2/L3 label keys used by testsite."""
    category = meta.get("signal_category", "")
    if category in ("pulse", "communication"):
        l1 = "active"
        l2 = "pulse" if category == "pulse" else "communication"
    elif category == "radiated_noise":
        l1, l2 = "passive", "ship_noise"
    else:
        return {"L1": "unknown", "L2": "unknown", "L3": "unknown"}

    l3 = (
        meta.get("sub_type")
        or meta.get("signal_type")
        or (meta.get("signal_params") or {}).get("original_class_name", "unknown")
    )
    l3 = str(l3).lower().strip()
    l3 = {
        "cargo_ship": "cargo",
        "cruise_ship": "cruise",
        "fishing_boat": "fishing",
        "warship": "warship",
        "underwater_target": "underwater_target",
    }.get(l3, l3)
    if category in ("pulse", "communication"):
        l3 = l3.upper()
    return {"L1": l1, "L2": l2, "L3": l3}


def _find_meta(audio_base: Path, audio_rel: str, sample_id: str) -> dict:
    """Find a processed metadata file, preferring its deterministic path."""
    parts = Path(audio_rel).parts
    direct = None
    if "PulseCom" in audio_rel and len(parts) >= 3:
        direct = audio_base / "PulseCom" / "jsonc" / parts[-2] / f"{sample_id}.jsonc"
    elif "05_ship_radiated_noise" in audio_rel and len(parts) >= 3:
        direct = (
            audio_base / "05_ship_radiated_noise" / parts[-3]
            / "json" / f"{sample_id}.json"
        )

    if direct is not None and direct.exists():
        try:
            return json.loads(direct.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass

    for extension in (".jsonc", ".json"):
        for candidate in audio_base.rglob(f"{sample_id}{extension}"):
            if candidate == direct:
                continue
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return {}


def _extract_quality(meta: dict) -> tuple:
    output = meta.get("bellhop_output", {})
    return output.get("tl_db"), output.get("snr_db")


def _extract_channel_metrics(meta: dict) -> dict:
    """Extract comparable BELLHOP environment and output metrics."""
    env = meta.get("bellhop_env", {})
    output = meta.get("bellhop_output", {})
    arrivals = output.get("summary", {}).get("num_arrivals")
    if arrivals is None:
        per_frequency = output.get("arrivals_per_frequency", {})
        if per_frequency:
            arrivals = sum(per_frequency.values())
    return {
        "range_km": env.get("range_km"),
        "rbox_km": env.get("ray_box_range_km"),
        "num_beams": env.get("num_beams_effective"),
        "num_arrivals": arrivals,
        "first_arrival_s": output.get("absolute_first_arrival_delay_s"),
        "normalization_gain_db": output.get("output_normalization_gain_db"),
    }


def _extract_ssp_complexity(meta: dict):
    ssp = meta.get("bellhop_env", {}).get("ssp", {})
    depths = ssp.get("depths_m")
    speeds = ssp.get("sound_speeds_mps")
    if not depths or not speeds or len(depths) <= 1:
        return None
    gradients = [
        (speeds[index + 1] - speeds[index]) / (depths[index + 1] - depths[index])
        for index in range(len(depths) - 1)
    ]
    mean_gradient = sum(gradients) / len(gradients)
    return sum((value - mean_gradient) ** 2 for value in gradients) / len(gradients)


def _load_split(jsonl_path: str, audio_base: Path, label: str = "") -> list:
    samples = []
    total = 0
    skipped = 0
    with open(jsonl_path, "r", encoding="utf-8") as source:
        for line in source:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = item.get("id", "")
            audio_rel = item.get("audio", "")
            if not sample_id or not audio_rel:
                continue
            total += 1
            meta = _find_meta(audio_base, audio_rel, sample_id)
            gt = _extract_gt(meta) if meta else {
                "L1": "unknown", "L2": "unknown", "L3": "unknown",
            }
            if gt["L1"] == "unknown":
                skipped += 1
                continue
            tl_db, snr_db = _extract_quality(meta)
            samples.append({
                "id": sample_id,
                "l1": gt["L1"],
                "l2": gt["L2"],
                "l3": gt["L3"],
                "tl_db": tl_db,
                "snr_db": snr_db,
                "ssp_complexity": _extract_ssp_complexity(meta),
                **_extract_channel_metrics(meta),
            })
            if total % 50 == 0:
                print(f"  [{label}] processed {total} rows, valid {len(samples)} ...")
    if skipped:
        print(f"  [{label}] skipped {skipped} (unknown GT)")
    return samples


def _numeric_stats(values: list) -> dict:
    """Return summary statistics and count-balanced dynamic tertiles."""
    ordered = sorted(float(value) for value in values if value is not None)
    if not ordered:
        return {}
    count = len(ordered)
    middle = count // 2
    median = ordered[middle] if count % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    low_end = count // 3
    high_start = (2 * count) // 3
    groups = (
        ("Low", ordered[:low_end]),
        ("Mid", ordered[low_end:high_start]),
        ("High", ordered[high_start:]),
    )
    return {
        "count": count,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / count,
        "median": median,
        "tiers": [(name, group) for name, group in groups if group],
    }


def _append_distribution(lines: list, title: str, stats: dict, unit: str) -> None:
    if not stats:
        return
    lines.extend([
        f"### {title}",
        "| Statistic | Value |",
        "|---|---:|",
        f"| Valid samples | {stats['count']} |",
        f"| Min | {stats['min']:.2f} {unit} |",
        f"| Max | {stats['max']:.2f} {unit} |",
        f"| Mean | {stats['mean']:.2f} {unit} |",
        f"| Median | {stats['median']:.2f} {unit} |",
        "",
        "| Dynamic tertile | Count | Value range | Ratio |",
        "|---|---:|---:|---:|",
    ])
    for name, values in stats["tiers"]:
        lines.append(
            f"| {name} | {len(values)} | {values[0]:.2f} to {values[-1]:.2f} {unit} | "
            f"{len(values) / stats['count']:.1%} |"
        )
    lines.append("")


def _append_channel_table(lines: list, title: str, samples: list) -> None:
    metrics = (
        ("Receiver range", "range_km", "km"),
        ("RBOX range", "rbox_km", "km"),
        ("Effective beams", "num_beams", "rays"),
        ("Arrivals", "num_arrivals", "paths"),
        ("Absolute first-arrival delay", "first_arrival_s", "s"),
        ("Output normalization gain", "normalization_gain_db", "dB"),
    )
    rows = []
    for metric_name, key, unit in metrics:
        stats = _numeric_stats([sample.get(key) for sample in samples])
        if stats:
            rows.append((metric_name, unit, stats))
    if not rows:
        return
    lines.extend([
        f"### Channel Metrics ({title})",
        "| Metric | Valid | Min | Max | Mean | Median |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for metric_name, unit, stats in rows:
        lines.append(
            f"| {metric_name} ({unit}) | {stats['count']} | {stats['min']:.3f} | "
            f"{stats['max']:.3f} | {stats['mean']:.3f} | {stats['median']:.3f} |"
        )
    lines.append("")


def _build_section(name: str, samples: list) -> str:
    count = len(samples)
    if not count:
        return f"## {name}\n\n(no samples)\n"
    l1_counts = Counter(sample["l1"] for sample in samples)
    l2_counts = Counter(sample["l2"] for sample in samples)
    l3_counts = Counter(sample["l3"] for sample in samples)
    lines = [
        f"## {name} ({count} samples)\n",
        "### L1 Distribution",
        "| L1 Class | Count | Ratio |",
        "|---|---:|---:|",
    ]
    for key in ("active", "passive"):
        value = l1_counts.get(key, 0)
        lines.append(f"| {_L1_NAMES[key]} | {value} | {value / count:.1%} |")
    lines.extend(["", "### L2 Distribution", "| L2 Class | Count | Ratio |", "|---|---:|---:|"])
    for key in ("pulse", "communication", "ship_noise"):
        value = l2_counts.get(key, 0)
        if value:
            lines.append(f"| {_L2_NAMES[key]} | {value} | {value / count:.1%} |")
    lines.extend(["", "### L3 Distribution", "| L3 Class | Count | Ratio |", "|---|---:|---:|"])
    for key in sorted(l3_counts):
        value = l3_counts[key]
        lines.append(f"| {_L3_NAMES.get(key, key)} | {value} | {value / count:.1%} |")
    lines.append("")

    active = [sample for sample in samples if sample["l1"] == "active"]
    passive = [sample for sample in samples if sample["l1"] == "passive"]
    transmission_loss = _numeric_stats([
        abs(float(sample["tl_db"])) for sample in active if sample.get("tl_db") is not None
    ])
    snr = _numeric_stats([sample.get("snr_db") for sample in passive])
    _append_distribution(
        lines, "Transmission Loss (PulseCom; abs(raw channel gain))", transmission_loss, "dB"
    )
    _append_distribution(lines, "Line-spectrum SNR (Ship)", snr, "dB")
    _append_channel_table(lines, "PulseCom", active)
    _append_channel_table(lines, "Ship", passive)

    ssp = _numeric_stats([sample.get("ssp_complexity") for sample in samples])
    if ssp:
        lines.extend([
            "### SSP Complexity Distribution",
            "| Statistic | Value |",
            "|---|---:|",
            f"| Valid samples | {ssp['count']} |",
            f"| Min | {ssp['min']:.4f} |",
            f"| Max | {ssp['max']:.4f} |",
            f"| Mean | {ssp['mean']:.4f} |",
            f"| Median | {ssp['median']:.4f} |",
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
    sections = []
    total = 0
    for label, path in (("Train", args.train), ("Val", args.val), ("Test", args.test)):
        jsonl_path = Path(path)
        if not jsonl_path.exists():
            print(f"  [SKIP] {label}: {path} not found")
            continue
        print(f"  Loading {label}: {path}")
        samples = _load_split(str(jsonl_path), audio_base, label=label)
        print(f"    Valid samples: {len(samples)}")
        total += len(samples)
        sections.append(_build_section(label, samples))
    if not total:
        print("Error: no valid samples")
        return 1

    report = "\n".join(["# Dataset Statistics\n", f"**Total samples**: {total}\n"] + sections)
    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"\n  Stats card saved to: {output_path.resolve()}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Plot the real UA-Bench evaluation-set characterization figure.

The script reads the exact JSONL and WAV files used by testsite. In strict
mode (the default), it refuses to write paper figures unless the manifest has
2,600 valid examples, every L3 class has support 200, every WAV exists, and the
metadata needed by the distribution panels are available.

Paper-scope note:
  The exported subset is a favorable-condition diagnostic benchmark. The
  figure illustrates class structure; it must not be used to claim robustness
  to severe channels or full operational-domain coverage.

Example
-------
python -m testsite.reporting.plot_dataset_quality \
  --manifest D:/path/testset_export/sft_test_highquality.jsonl \
  --audio-root D:/path/testset_export \
  --metadata-root D:/path/processed_audio \
  --output figures/fig2
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.signal import stft
import soundfile as sf

try:
    from .plot_style import (
        BLUE, ORANGE, TEAL, apply_publication_style,
        save_figure, style_axis,
    )
except ImportError:  # Allow direct execution from testsite/reporting.
    from plot_style import (  # type: ignore
        BLUE, ORANGE, TEAL, apply_publication_style,
        save_figure, style_axis,
    )


ORDER = [
    "CW", "LFM", "HFM", "2FSK", "4FSK", "BPSK", "QPSK", "OFDM",
    "cargo", "cruise", "fishing", "warship", "underwater_target",
]
DISPLAY = {
    "CW": "CW", "LFM": "LFM", "HFM": "HFM", "2FSK": "2FSK",
    "4FSK": "4FSK", "BPSK": "BPSK", "QPSK": "QPSK", "OFDM": "OFDM",
    "cargo": "Cargo", "cruise": "Cruise", "fishing": "Fishing",
    "warship": "Warship", "underwater_target": "Underwater vehicle",
}
GROUPS = {
    "Pulse": ORDER[:3],
    "Communication": ORDER[3:8],
    "Ship noise": ORDER[8:],
}
COLORS = {"Pulse": BLUE, "Communication": ORANGE, "Ship noise": TEAL}
L2_FOR = {key: group for group, keys in GROUPS.items() for key in keys}


@dataclass
class Sample:
    sample_id: str
    audio_path: Path
    l1: str
    l2: str
    l3: str
    duration_s: float
    sample_rate_hz: int
    tl_db: float | None
    snr_db: float | None
    center_frequency_hz: float | None
    quality_score: float | None
    quality_metric: str | None
    selection_policy: str | None
    raw_record: dict[str, Any]


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _normalize_l3(value: Any) -> str:
    text = str(value or "").strip()
    aliases = {
        "cargo_ship": "cargo", "cruise_ship": "cruise",
        "fishing_boat": "fishing", "naval_vessel": "warship",
        "underwater target": "underwater_target",
    }
    lowered = aliases.get(text.lower(), text.lower())
    if lowered in {"cw", "lfm", "hfm", "2fsk", "4fsk", "bpsk", "qpsk", "ofdm"}:
        return lowered.upper()
    return lowered


def _infer_gt(record: dict[str, Any]) -> tuple[str, str, str]:
    gt = record.get("_gt") or {}
    l3 = _normalize_l3(gt.get("L3"))
    if l3 in ORDER:
        return str(gt.get("L1", "")).lower(), str(gt.get("L2", "")).lower(), l3
    raise ValueError("record has no valid embedded _gt.L3; use the exported evaluation manifest")


def _candidate_metadata_paths(metadata_root: Path, audio_rel: str, sample_id: str) -> list[Path]:
    parts = Path(audio_rel).parts
    candidates: list[Path] = []
    if "PulseCom" in parts:
        index = parts.index("PulseCom")
        tail = parts[index:]
        signal_type = tail[-2] if len(tail) >= 2 else ""
        candidates.append(metadata_root / "PulseCom" / "jsonc" / signal_type / f"{sample_id}.jsonc")
    if "05_ship_radiated_noise" in parts:
        index = parts.index("05_ship_radiated_noise")
        tail = parts[index:]
        class_name = tail[-3] if len(tail) >= 3 else ""
        candidates.append(metadata_root / "05_ship_radiated_noise" / class_name / "json" / f"{sample_id}.json")
    return candidates


def _load_full_metadata(metadata_root: Path | None, audio_rel: str, sample_id: str) -> dict[str, Any]:
    if metadata_root is None:
        return {}
    for candidate in _candidate_metadata_paths(metadata_root, audio_rel, sample_id):
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def _extract_frequency(meta: dict[str, Any]) -> float | None:
    # Prefer the representative frequency that was actually passed to BELLHOP.
    # This also covers records whose source frequency was inferred from a band.
    env_frequency = _number((meta.get("bellhop_env") or {}).get("freq_hz"))
    if env_frequency is not None and env_frequency > 0:
        return env_frequency

    params = meta.get("signal_params") or {}
    for key in ("center_freq_hz", "center_frequency_hz", "carrier_freq_hz", "carrier_frequency_hz"):
        value = _number(params.get(key))
        if value is not None and value > 0:
            return value
    for low_key, high_key in (
        ("band_low_hz", "band_high_hz"),
        ("subband_low_hz", "subband_high_hz"),
    ):
        low = _number(params.get(low_key))
        high = _number(params.get(high_key))
        if low is not None and high is not None and 0 <= low < high:
            return (low + high) / 2
    return None


def load_samples(manifest: Path, audio_root: Path, metadata_root: Path | None) -> list[Sample]:
    samples: list[Sample] = []
    problems: list[str] = []
    with manifest.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                sample_id = str(record["id"])
                audio_rel = str(record["audio"])
                l1, l2, l3 = _infer_gt(record)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                problems.append(f"line {line_number}: {exc}")
                continue
            audio_path = audio_root / audio_rel
            if not audio_path.exists():
                problems.append(f"line {line_number}: missing WAV {audio_path}")
                continue
            try:
                info = sf.info(audio_path)
            except RuntimeError as exc:
                problems.append(f"line {line_number}: unreadable WAV {audio_path}: {exc}")
                continue
            embedded = record.get("_meta") or {}
            full_meta = _load_full_metadata(metadata_root, audio_rel, sample_id)
            output = full_meta.get("bellhop_output") or {}
            tl_db = _number(embedded.get("tl_db", output.get("tl_db")))
            snr_db = _number(embedded.get("snr_db", output.get("snr_db")))
            quality_score = _number(embedded.get("quality_score"))
            if quality_score is None:
                quality_score = snr_db if l2 == "ship_noise" else tl_db
            samples.append(Sample(
                sample_id=sample_id,
                audio_path=audio_path,
                l1=l1,
                l2=l2,
                l3=l3,
                duration_s=float(info.frames / info.samplerate),
                sample_rate_hz=int(info.samplerate),
                tl_db=tl_db,
                snr_db=snr_db,
                center_frequency_hz=_extract_frequency(full_meta),
                quality_score=quality_score,
                quality_metric=embedded.get("quality_metric"),
                selection_policy=embedded.get("selection_policy"),
                raw_record=record,
            ))
    if problems:
        preview = "\n".join(problems[:10])
        raise RuntimeError(f"manifest/audio validation failed ({len(problems)} issue(s)):\n{preview}")
    return samples


def validate_paper_subset(samples: list[Sample], strict: bool) -> None:
    counts = Counter(sample.l3 for sample in samples)
    missing_classes = [key for key in ORDER if counts[key] == 0]
    if missing_classes:
        raise RuntimeError(f"missing L3 classes: {missing_classes}")
    if strict:
        if len(samples) != 2600:
            raise RuntimeError(f"paper mode requires 2,600 samples, found {len(samples)}")
        bad = {key: counts[key] for key in ORDER if counts[key] != 200}
        if bad:
            raise RuntimeError(f"paper mode requires 200 samples per L3 class: {bad}")
        if any(sample.sample_rate_hz != 16000 for sample in samples):
            raise RuntimeError("paper mode requires the documented 16-kHz evaluation inputs")
        active = [sample for sample in samples if sample.l1 == "active"]
        ships = [sample for sample in samples if sample.l2 == "ship_noise"]
        if sum(sample.tl_db is not None for sample in active) != len(active):
            raise RuntimeError("active tl_db/channel-gain metadata are incomplete")
        if sum(sample.snr_db is not None for sample in ships) != len(ships):
            raise RuntimeError("ship source/pre-channel line-SNR metadata are incomplete")
        if sum(sample.center_frequency_hz is not None for sample in active) != len(active):
            raise RuntimeError(
                "active center/carrier-frequency metadata are incomplete; provide --metadata-root "
                "pointing to processed_audio"
            )


def _robust_medoid(pool: list[Sample], targets: dict[str, float] | None = None) -> Sample:
    """Select a deterministic sample nearest robust-scaled feature targets."""
    durations = np.asarray([sample.duration_s for sample in pool], dtype=float)
    quality = np.asarray([
        float(sample.quality_score) if sample.quality_score is not None else np.nan
        for sample in pool
    ])
    frequencies = np.asarray([
        float(sample.center_frequency_hz) if sample.center_frequency_hz is not None else np.nan
        for sample in pool
    ])
    named_features = [("duration", durations)]
    if np.isfinite(quality).all():
        named_features.append(("quality", quality))
    if pool[0].l1 == "active" and np.isfinite(frequencies).all():
        named_features.append(("frequency", frequencies))
    matrix = np.column_stack([values for _, values in named_features])
    medians = np.asarray([
        (targets or {}).get(name, float(np.median(values)))
        for name, values in named_features
    ])
    scales = np.subtract(*np.percentile(matrix, [75, 25], axis=0))
    scales[scales == 0] = 1.0
    distances = np.sqrt(np.square((matrix - medians) / scales).sum(axis=1))
    ranked = sorted(zip(distances, pool), key=lambda item: (float(item[0]), item[1].sample_id))
    return ranked[0][1]


def select_representatives(samples: list[Sample]) -> dict[str, Sample]:
    # Paper wording guardrail: these are illustrative class examples selected
    # from the evaluation manifest, not population-representative exemplars.
    by_class: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.l3].append(sample)
    selected: dict[str, Sample] = {}
    for l3 in ORDER:
        pool = by_class[l3]
        selected[l3] = _robust_medoid(pool)
    return selected


def compute_spectrogram(sample: Sample, n_fft: int, hop: int, fmin: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    audio, sample_rate = sf.read(sample.audio_path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    segment = min(n_fft, len(mono))
    if segment < 16:
        raise RuntimeError(f"audio too short for STFT: {sample.audio_path}")
    overlap = max(0, segment - min(hop, segment))
    frequencies, times, spectrum = stft(
        mono, fs=sample_rate, window="hann", nperseg=segment,
        noverlap=overlap, nfft=max(n_fft, segment), boundary=None, padded=False,
    )
    keep = frequencies >= fmin
    power = np.abs(spectrum[keep]) ** 2
    return frequencies[keep], times, 10 * np.log10(np.maximum(power, 1e-14))


def _ecdf(ax, values: list[float], color: str, label: str | None = None) -> None:
    ordered = np.sort(np.asarray(values, dtype=float))
    if len(ordered) == 0:
        return
    y = np.arange(1, len(ordered) + 1) / len(ordered)
    ax.plot(ordered, y, color=color, linewidth=1.5, label=label)


def _style_axis(ax, grid_axis: str = "both") -> None:
    style_axis(
        ax, grid_axis, minor_grid=False,
        tick_length=2.5, tick_width=.6,
    )


def _format_time_axis(ax, times: np.ndarray) -> None:
    """Keep dense spectrogram time axes legible at paper scale."""
    if len(times) > 1:
        ax.set_xlim(max(0.0, float(times[0])), float(times[-1]))
    ax.margins(x=0)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
    formatter = mticker.ScalarFormatter(useOffset=False)
    formatter.set_powerlimits((-3, 4))
    ax.xaxis.set_major_formatter(formatter)


def draw_hierarchy(ax, counts: Counter) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(.01, .88, "(a) Evaluation hierarchy and support", fontsize=10, weight="bold")
    ax.text(.01, .56, f"{sum(counts.values()):,} favorable-condition\nheld-out examples",
            ha="left", va="center", fontsize=8.5)
    positions = {"Pulse": .36, "Communication": .61, "Ship noise": .86}
    for group, x in positions.items():
        color = COLORS[group]
        total = sum(counts[key] for key in GROUPS[group])
        box = FancyBboxPatch((x-.105, .48), .21, .27, boxstyle="round,pad=.012",
                             facecolor=color+"18", edgecolor=color, linewidth=1.0)
        ax.add_patch(box)
        parent = "Active" if group != "Ship noise" else "Passive"
        ax.text(x, .68, f"{parent} → {group}", ha="center", va="center", fontsize=8.2, weight="bold")
        ax.text(x, .56, f"n={total:,}", ha="center", fontsize=8)
        leaves = "   ".join(f"{DISPLAY[key]} ({counts[key]})" for key in GROUPS[group])
        ax.text(x, .38, leaves, ha="center", va="top", fontsize=6.7, color="#39434A")


def plot_figure(samples: list[Sample], selected: dict[str, Sample], output: Path,
                n_fft: int, hop: int, fmin: float, db_floor: float,
                dpi: int = 300) -> None:
    counts = Counter(sample.l3 for sample in samples)
    spectra = {key: compute_spectrogram(sample, n_fft, hop, fmin) for key, sample in selected.items()}
    reference_db = max(float(values[2].max()) for values in spectra.values())
    norm = Normalize(vmin=db_floor, vmax=0)

    fig = plt.figure(figsize=(15.4, 10.2), constrained_layout=True)
    outer = fig.add_gridspec(3, 1, height_ratios=[.72, 3.0, 1.15])
    hierarchy_ax = fig.add_subplot(outer[0])
    draw_hierarchy(hierarchy_ax, counts)

    atlas = outer[1].subgridspec(
        4, 2, height_ratios=[.13, 1, 1, 1],
        width_ratios=[1, .022], hspace=.26, wspace=.05,
    )
    atlas_title = fig.add_subplot(atlas[0, :])
    atlas_title.set_axis_off()
    atlas_title.text(
        0, .5, "(b) Deterministically selected L3 examples under a shared STFT and dB scale",
        va="center", fontsize=10, weight="bold",
    )
    image = None
    atlas_rows = (
        (GROUPS["Pulse"], atlas[1, 0].subgridspec(1, 3, wspace=.16)),
        (GROUPS["Communication"], atlas[2, 0].subgridspec(1, 5, wspace=.16)),
        (GROUPS["Ship noise"], atlas[3, 0].subgridspec(1, 5, wspace=.16)),
    )
    for keys, row_grid in atlas_rows:
        for col, key in enumerate(keys):
            ax = fig.add_subplot(row_grid[0, col])
            frequencies, times, db = spectra[key]
            relative = np.clip(db - reference_db, db_floor, 0)
            image = ax.pcolormesh(
                times, frequencies, relative, shading="auto",
                cmap="magma", norm=norm, rasterized=True,
            )
            _format_time_axis(ax, times)
            ax.set_yscale("log")
            ax.set_ylim(max(fmin, frequencies[0]), frequencies[-1])
            ax.set_title(
                f"{DISPLAY[key]}  |  {selected[key].duration_s:.2f} s",
                color=COLORS[L2_FOR[key]], pad=2.5,
            )
            ax.set_xlabel("Time (s)")
            if col == 0:
                ax.set_ylabel("Frequency (Hz)")
            else:
                ax.set_yticklabels([])
            _style_axis(ax)
    if image is not None:
        colorbar_grid = atlas[1:, 1].subgridspec(
            3, 1, height_ratios=[.18, .64, .18],
        )
        colorbar_ax = fig.add_subplot(colorbar_grid[1, 0])
        colorbar = fig.colorbar(image, cax=colorbar_ax)
        colorbar.set_label("Relative power (dB; common reference)")

    dist = outer[2].subgridspec(2, 4, height_ratios=[.14, 1], wspace=.28)
    dist_title = fig.add_subplot(dist[0, :])
    dist_title.set_axis_off()
    dist_title.text(
        0, .5, "(c) Distributions from the same evaluation manifest",
        va="center", fontsize=10, weight="bold",
    )
    ax = fig.add_subplot(dist[1, 0])
    for group in GROUPS:
        values = [sample.duration_s for sample in samples if sample.l3 in GROUPS[group]]
        _ecdf(ax, values, COLORS[group], group)
    ax.set_xscale("log"); ax.set_xlabel("Duration (s)"); ax.set_ylabel("ECDF")
    ax.set_title(f"Duration (n={len(samples):,})")
    ax.legend(frameon=False, fontsize=6.5); _style_axis(ax)

    ax = fig.add_subplot(dist[1, 1])
    active = [sample for sample in samples if sample.l1 == "active"]
    frequency_valid = [sample.center_frequency_hz for sample in active if sample.center_frequency_hz is not None]
    for group in ("Pulse", "Communication"):
        values = [sample.center_frequency_hz for sample in samples
                  if sample.l3 in GROUPS[group] and sample.center_frequency_hz is not None]
        if values:
            _ecdf(ax, values, COLORS[group], f"{group} (n={len(values)})")
    ax.set_xlabel("Center/carrier frequency (Hz)"); ax.set_ylabel("ECDF")
    ax.set_title(f"Active frequency (n={len(frequency_valid):,}; miss={len(active)-len(frequency_valid):,})")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=6.5)
    _style_axis(ax)

    ax = fig.add_subplot(dist[1, 2])
    active_gain = [sample.tl_db for sample in samples if sample.l1 == "active" and sample.tl_db is not None]
    _ecdf(ax, active_gain, "#4C78A8")
    ax.set_xlabel("Channel gain, legacy tl_db (dB)"); ax.set_ylabel("ECDF")
    ax.set_title(f"Channel gain (n={len(active_gain):,}; miss={len(active)-len(active_gain):,})")
    _style_axis(ax)

    ax = fig.add_subplot(dist[1, 3])
    ship_snr = [sample.snr_db for sample in samples if sample.l2 == "ship_noise" and sample.snr_db is not None]
    ships = [sample for sample in samples if sample.l2 == "ship_noise"]
    _ecdf(ax, ship_snr, "#2A9D8F")
    ax.set_xlabel("Source/pre-channel line-SNR (dB)"); ax.set_ylabel("ECDF")
    ax.set_title(f"Ship line-SNR (n={len(ship_snr):,}; miss={len(ships)-len(ship_snr):,})")
    _style_axis(ax)
    save_figure(fig, output, dpi=dpi)


def write_selection_record(path: Path, manifest: Path, audio_root: Path,
                           selected: dict[str, Sample], args: argparse.Namespace) -> None:
    payload = {
        "source_manifest": str(manifest.resolve()),
        "audio_root": str(audio_root.resolve()),
        "selection_rule": "minimum robust-scaled distance to class medians of duration, family quality metadata, and active center/carrier frequency; sample_id tie-break",
        "stft": {"n_fft": args.n_fft, "hop_length": args.hop_length, "window": "hann", "fmin_hz": args.fmin},
        "display": {
            "power_db_floor": args.db_floor,
            "reference": "maximum STFT-bin power shared across all 13 selected examples",
            "raster_dpi": getattr(args, "dpi", 300),
        },
        "samples": {
            key: {
                "sample_id": sample.sample_id,
                "audio_path": str(sample.audio_path),
                "duration_s": sample.duration_s,
                "sample_rate_hz": sample.sample_rate_hz,
                "tl_db": sample.tl_db,
                "snr_db": sample.snr_db,
                "center_frequency_hz": sample.center_frequency_hz,
                "quality_score": sample.quality_score,
                "quality_metric": sample.quality_metric,
                "selection_policy": sample.selection_policy,
            }
            for key, sample in selected.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the real UA-Bench evaluation-set quality figure")
    parser.add_argument("--manifest", type=Path, required=True, help="Exact exported JSONL used for evaluation")
    parser.add_argument("--audio-root", type=Path, required=True, help="Root used to resolve each record's audio path")
    parser.add_argument("--metadata-root", type=Path, default=None, help="processed_audio root containing full JSON/JSONC metadata")
    parser.add_argument("--output", type=Path, default=Path("figures/fig2"), help="Output prefix without extension")
    parser.add_argument(
        "--selection-record", type=Path, default=None,
        help="Selected-sample provenance JSON; defaults beside --output",
    )
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--fmin", type=float, default=20.0)
    parser.add_argument("--db-floor", type=float, default=-80.0)
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI (default: 300)")
    parser.add_argument("--allow-nonpaper-subset", action="store_true", help="Allow smoke tests on a non-2,600 subset")
    args = parser.parse_args()

    if not args.manifest.exists():
        parser.error(f"manifest not found: {args.manifest}")
    if not args.audio_root.exists():
        parser.error(f"audio root not found: {args.audio_root}")
    if args.metadata_root is not None and not args.metadata_root.exists():
        parser.error(f"metadata root not found: {args.metadata_root}")
    chosen_font = apply_publication_style(
        font_size=8,
        title_size=8.5,
        label_size=8,
        tick_size=7,
        legend_size=6.5,
    )
    print(f"Plot font: {chosen_font}")
    samples = load_samples(args.manifest, args.audio_root, args.metadata_root)
    validate_paper_subset(samples, strict=not args.allow_nonpaper_subset)
    selected = select_representatives(samples)
    plot_figure(
        samples, selected, args.output, args.n_fft, args.hop_length,
        args.fmin, args.db_floor, args.dpi,
    )
    print(
        f"  [OK] Main characterization: {args.output.with_suffix('.png')} "
        f"and {args.output.with_suffix('.pdf')}"
    )

    selection_record = args.selection_record or args.output.with_name(
        f"{args.output.name}_selection.json"
    )
    write_selection_record(
        selection_record, args.manifest, args.audio_root,
        selected, args,
    )
    counts = Counter(sample.l3 for sample in samples)
    print(f"Selection record: {selection_record}")
    print("Class support: " + ", ".join(f"{key}={counts[key]}" for key in ORDER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

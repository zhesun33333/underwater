"""Paper Figure B: record-weighted test/evaluation distribution comparison.

The caller supplies audited, canonical records. Importing this module neither
reads a dataset nor creates or saves a figure. ``build_figure`` returns the
figure and its complete plotting data so the caller can save both together.
"""
from __future__ import annotations

from collections import Counter
import math
from numbers import Real
from typing import Any

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatter, LogLocator, MaxNLocator, NullLocator
import numpy as np


_FAMILIES = ("pulse", "communication", "ship_noise")
_COLORS = {"pulse": "#4C78A8", "communication": "#F28E2B", "ship_noise": "#2A9D8F"}
_LABELS = {"pulse": "Pulse", "communication": "Communication", "ship_noise": "Ship noise"}
_IDENTITY_FIELDS = (
    "source_id", "l1", "l2", "l3", "duration_s", "signal_frequency_hz",
    "channel_frequency_hz", "G_h", "S_src", "fs", "frames",
)
_PANELS = (
    ("a", "Audio duration", "duration_s", _FAMILIES, "log", "Duration (s)", True),
    ("b", "Active signal frequency", "signal_frequency_hz", _FAMILIES[:2], "linear",
     "Signal center/carrier\nfrequency (Hz)", True),
    ("c", "Active channel gain", "G_h", _FAMILIES[:2], "linear",
     "Channel energy gain, $G_h$ (dB)", False),
    ("d", "Ship source power ratio", "S_src", ("ship_noise",), "linear",
     "Source tonal-to-continuum\npower ratio, $S_\\mathrm{src}$ (dB)", False),
)


def ecdf_coordinates(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return right-continuous ECDF coordinates, including the first jump.

    Repeated observations produce one jump of their total empirical mass.
    Render these coordinates with ``drawstyle='steps-post'``. The denominator
    is the number of valid observations supplied, never the manifest size.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("ECDF values must be a one-dimensional finite sequence")
    if array.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    unique, counts = np.unique(array, return_counts=True)
    return np.r_[unique[0], unique], np.r_[0.0, np.cumsum(counts) / array.size]


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, Real) and isinstance(right, Real):
        if isinstance(left, bool) or isinstance(right, bool):
            return type(left) is type(right) and left == right
        a, b = float(left), float(right)
        if not math.isfinite(a) or not math.isfinite(b):
            return (math.isnan(a) and math.isnan(b)) or a == b
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)
    return type(left) is type(right) and left == right


def _index_records(records: list[dict], name: str) -> dict[str, dict]:
    if not records:
        raise ValueError(f"{name} must contain at least one record")
    indexed = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{name}[{position}] is not a record dictionary")
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(f"{name}[{position}] has no nonempty string id")
        if sample_id in indexed:
            raise ValueError(f"{name} contains duplicate id {sample_id!r}")
        if record.get("l2") not in _FAMILIES:
            raise ValueError(f"{name}: {sample_id!r} has an unsupported l2 family")
        indexed[sample_id] = record
    return indexed


def _validate_relationship(full_records: list[dict], eval_records: list[dict]) -> dict:
    full = _index_records(full_records, "full_test")
    evaluation = _index_records(eval_records, "evaluation")
    outside = sorted(evaluation.keys() - full.keys())
    if outside:
        raise ValueError(f"evaluation is not a subset of full_test: {outside[:8]}")
    mismatches = []
    for sample_id, record in evaluation.items():
        for field in _IDENTITY_FIELDS:
            if not _same_value(full[sample_id].get(field), record.get(field)):
                mismatches.append(f"{sample_id}.{field}")
    if mismatches:
        raise ValueError("shared records disagree between full_test and evaluation: "
                         + ", ".join(mismatches[:8]))
    return {
        "evaluation_is_subset_of_full_test": True,
        "full_test_unique_ids": len(full),
        "evaluation_unique_ids": len(evaluation),
        "shared_id_count": len(evaluation),
        "checked_fields": list(_IDENTITY_FIELDS),
        "numeric_relative_tolerance": 1e-12,
        "numeric_absolute_tolerance": 1e-9,
    }


def _validity(value: Any, positive: bool) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool) or not isinstance(value, Real):
        return "not_numeric"
    if not math.isfinite(float(value)):
        return "nonfinite"
    if positive and float(value) <= 0:
        return "nonpositive"
    return "valid"


def _json_value(value: Any) -> Any:
    """Keep scalar inputs JSON-safe; validity flags preserve invalid cases."""
    if isinstance(value, Real) and not isinstance(value, bool):
        return float(value) if math.isfinite(float(value)) else None
    if value is None or isinstance(value, (str, bool)):
        return value
    return repr(value)


def _series_data(records: list[dict], collection: str, family: str,
                 field: str, positive: bool) -> dict:
    population = [record for record in records if record["l2"] == family]
    statuses = [_validity(record.get(field), positive) for record in population]
    valid = [record for record, status in zip(population, statuses) if status == "valid"]
    values = [float(record[field]) for record in valid]
    x, y = ecdf_coordinates(values)
    invalid_counts = Counter(status for status in statuses if status != "valid")
    return {
        "key": f"{collection}:{family}",
        "collection": collection,
        "family": family,
        "source_field": field,
        "color": _COLORS[family],
        "linestyle": "--" if collection == "full_test" else "-",
        "population_n": len(population),
        "n": len(valid),
        "missing": len(population) - len(valid),
        "missing_by_reason": dict(sorted(invalid_counts.items())),
        "raw_ids": [record["id"] for record in population],
        "raw_values": [_json_value(record.get(field)) for record in population],
        "raw_value_status": statuses,
        "ids": [record["id"] for record in valid],
        "values": values,
        "missing_ids": [record["id"] for record, status in zip(population, statuses)
                        if status != "valid"],
        "median": float(np.median(values)) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "ecdf": {"x": x.tolist(), "y": y.tolist(), "drawstyle": "steps-post"},
    }


def build_figure(full_records: list[dict], eval_records: list[dict],
                 style: dict) -> tuple[Figure, dict]:
    """Build Figure B and provenance without reading files or saving output.

    Duration comes exclusively from ``duration_s`` (audited WAV headers).
    Panel B uses source ``signal_frequency_hz``, never the separate BELLHOP
    ``channel_frequency_hz``. A record missing a panel's value stays in that
    series' population count but is excluded from its ECDF denominator.
    """
    relationship = _validate_relationship(full_records, eval_records)
    width_mm = float(style.get("width_mm", 180))
    height_mm = float(style.get("distribution_height_mm", width_mm * 11 / 18))
    font_size = float(style.get("font_size_pt", 9))
    font_family = str(style.get("font_family", "Times New Roman"))
    legend_frame = bool(style.get("legend_frame", False))
    if not math.isfinite(width_mm) or width_mm <= 0:
        raise ValueError("width_mm must be finite and positive")
    if not math.isfinite(font_size) or font_size <= 8:
        raise ValueError("Figure B requires font_size_pt > 8")
    if not font_family.strip():
        raise ValueError("font_family must not be empty")

    payload = {
        "schema_version": "ua_bench.selection_distributions.v1",
        "figure": "selection_distributions",
        "weighting": "record_weighted",
        "statistical_unit": "one manifest record, including separate channel realizations",
        "denominator": "valid records in each family, collection and panel",
        "missing_definition": "missing, nonnumeric, nonfinite, or nonpositive for duration/frequency",
        "raw_values_encoding": "nonfinite scalars become null; raw_value_status records the reason",
        "source_fields": {
            "duration_s": "actual WAV frames / sample rate, supplied by the audited record loader",
            "signal_frequency_hz": "source center/carrier frequency; no BELLHOP-frequency fallback",
            "G_h": "active pre-normalization channel energy gain in dB (legacy tl_db)",
            "S_src": "ship source tonal-to-continuum power ratio in dB (legacy snr_db)",
        },
        "relationship": relationship,
        "style": {"width_mm": width_mm, "height_mm": height_mm,
                  "font_family": font_family, "font_size_pt": font_size,
                  "legend_frame": legend_frame, "colors": dict(_COLORS)},
        "panels": [],
    }
    rc = {
        "font.family": font_family, "font.size": font_size,
        "axes.titlesize": font_size, "axes.labelsize": font_size,
        "xtick.labelsize": font_size, "ytick.labelsize": font_size,
        "legend.fontsize": font_size, "mathtext.fontset": "stix",
        "axes.linewidth": 0.75, "lines.linewidth": 1.0,
        "figure.facecolor": "white", "axes.facecolor": "white", "text.color": "black",
        "axes.labelcolor": "black", "axes.edgecolor": "black", "xtick.color": "black",
        "ytick.color": "black", "axes.titleweight": "normal", "axes.labelweight": "bold",
    }
    with mpl.rc_context(rc):
        figure = Figure(figsize=(width_mm / 25.4, height_mm / 25.4))
        axes = figure.subplots(2, 2, sharex=False, sharey=False)
        figure.subplots_adjust(left=0.085, right=0.98, bottom=0.14, top=0.835,
                               wspace=0.27, hspace=0.78)
        for ax, (key, title, field, families, scale, xlabel, positive) in zip(axes.flat, _PANELS):
            series = [
                _series_data(records, collection, family, field, positive)
                for collection, records in (("full_test", full_records), ("evaluation", eval_records))
                for family in families
            ]
            totals = {
                collection: {
                    "n": sum(item["n"] for item in series if item["collection"] == collection),
                    "missing": sum(item["missing"] for item in series if item["collection"] == collection),
                }
                for collection in ("full_test", "evaluation")
            }
            payload["panels"].append({
                "key": key, "title": title, "source_field": field,
                "xscale": scale, "xlabel": xlabel, "ylabel": "ECDF",
                "valid_n_order": ["full_test", "evaluation"],
                "totals": totals, "series": series,
            })
            for item in series:
                if item["n"]:
                    ax.plot(item["ecdf"]["x"], item["ecdf"]["y"],
                            drawstyle="steps-post", color=item["color"],
                            linestyle=item["linestyle"], linewidth=1.0,
                            label=item["key"])
            ax.set_xscale(scale)
            if scale == "log":
                ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
                ax.xaxis.set_major_formatter(LogFormatter(base=10))
            else:
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_ylim(0, 1)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlabel(xlabel, fontsize=font_size)
            ax.set_ylabel("ECDF", fontsize=font_size)
            ax.set_title(f"({key}) {title}\nValid n: {totals['full_test']['n']:,} / "
                         f"{totals['evaluation']['n']:,}", fontsize=font_size, pad=5)
            ax.minorticks_off()
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis="both", which="major", direction="in", top=True, right=True,
                           length=3, width=0.75, labelsize=font_size)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.75)
            ax.grid(False)
            ax.margins(x=0.04)
        family_handles = [Line2D([], [], color=_COLORS[family], linewidth=1)
                          for family in _FAMILIES]
        figure.legend(family_handles + [Line2D([], [], color="black", linestyle="--", linewidth=1),
                       Line2D([], [], color="black", linestyle="-", linewidth=1)],
                      [_LABELS[family] for family in _FAMILIES] + ["Full test", "UA-Bench-Eval"],
                      loc="upper center", bbox_to_anchor=(0.53, 0.995), ncol=5,
                      frameon=legend_frame, handlelength=1.6, columnspacing=1.1, handletextpad=0.5,
                      fontsize=font_size)
    return figure, payload

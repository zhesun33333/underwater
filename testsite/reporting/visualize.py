"""Generate publication-ready figures from an eval_metrics_*.json file.

The PNG filenames are kept stable for existing paper references. PDF output is
enabled by default so the same figures can be placed in a paper as vectors.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


NAVY = "#244A64"
BLUE = "#4C78A8"
TEAL = "#2A9D8F"
ORANGE = "#F28E2B"
RED = "#D1495B"
PURPLE = "#7A6FAC"
GRAY = "#8A94A3"
LIGHT_GRAY = "#DCE3E8"
INK = "#263238"

L3_SHORT = {
    "pulse": ["CW", "LFM", "HFM"],
    "communication": ["2FSK", "4FSK", "BPSK", "QPSK", "OFDM"],
    "ship_noise": ["cargo", "cruise", "fishing", "warship", "underwater_target"],
}
GROUP_COLORS = {"pulse": BLUE, "communication": ORANGE, "ship_noise": TEAL}
GROUP_LABELS = {"pulse": "Pulse", "communication": "Communication", "ship_noise": "Ship noise"}
L3_DISPLAY = {
    "CW": "CW", "LFM": "LFM", "HFM": "HFM",
    "2FSK": "2FSK", "4FSK": "4FSK", "BPSK": "BPSK",
    "QPSK": "QPSK", "OFDM": "OFDM",
    "cargo": "Cargo", "cruise": "Cruise", "fishing": "Fishing",
    "warship": "Warship", "underwater_target": "Underwater\nvehicle",
}
_NAME_TO_SHORT = {
    "CW (Continuous Wave)": "CW", "CW连续波": "CW",
    "LFM (Linear Frequency Modulation)": "LFM", "LFM线性调频": "LFM",
    "HFM (Hyperbolic Frequency Modulation)": "HFM", "HFM双曲调频": "HFM",
    "2FSK (Binary Frequency Shift Keying)": "2FSK", "2FSK二进制频移键控": "2FSK",
    "4FSK (Quaternary Frequency Shift Keying)": "4FSK", "4FSK四进制频移键控": "4FSK",
    "BPSK (Binary Phase Shift Keying)": "BPSK", "BPSK二进制相移键控": "BPSK",
    "QPSK (Quadrature Phase Shift Keying)": "QPSK", "QPSK四进制相移键控": "QPSK",
    "OFDM (Orthogonal Frequency Division Multiplexing)": "OFDM", "OFDM正交频分复用": "OFDM",
    "Cargo vessel": "cargo", "货船": "cargo",
    "Cruise ship": "cruise", "邮轮": "cruise",
    "Fishing vessel": "fishing", "渔船": "fishing",
    "Naval vessel": "warship", "军舰": "warship",
    "Underwater target": "underwater_target", "水下目标": "underwater_target",
}

OUTPUT_FORMATS = ("png", "pdf")
OUTPUT_DPI = 300


def _apply_publication_style():
    plt.rcParams.update({
        # Follow the reference figures' paper-like serif typography without
        # requiring a LaTeX installation at evaluation time.
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#AAB4BC",
        "axes.linewidth": 0.8,
        "patch.edgecolor": "#263238",
        "patch.linewidth": 0.65,
        "patch.force_edgecolor": True,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.fontsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
    })


def _style_axis(ax, grid_axis="y"):
    # The boxed frame, inward ticks and dash-dot grid echo the supplied
    # reference plots, with lighter strokes to avoid visual clutter.
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#8E99A1")
        spine.set_linewidth(0.7)
    ax.tick_params(direction="in", top=True, right=True, length=3.5, width=0.7)
    ax.grid(axis=grid_axis, color="#C7CED4", linestyle="-.", linewidth=0.65, alpha=0.75)
    ax.minorticks_on()
    ax.grid(which="minor", axis=grid_axis, color=LIGHT_GRAY, linestyle=":", linewidth=0.45, alpha=0.55)
    ax.set_axisbelow(True)


def _save_figure(fig, output_dir: Path, stem: str):
    for fmt in OUTPUT_FORMATS:
        path = output_dir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=OUTPUT_DPI if fmt == "png" else None)
    plt.close(fig)
    print(f"  [OK] {stem}.png" + (" + PDF" if "pdf" in OUTPUT_FORMATS else ""))


def _annotate_bars(ax, bars, values, suffix="%", decimals=1, offset=None):
    ymax = max(values, default=0)
    gap = offset if offset is not None else max(ymax * 0.025, 0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + gap,
                f"{val:.{decimals}f}{suffix}", ha="center", va="bottom", fontsize=8)


def _match_l3_key(name: str) -> str | None:
    all_keys = sum(L3_SHORT.values(), [])
    if name in all_keys:
        return name
    if name in _NAME_TO_SHORT:
        return _NAME_TO_SHORT[name]
    lower = name.lower()
    for key in sorted(all_keys, key=len, reverse=True):
        if key.lower() in lower:
            return key
    return None


def _get_group(short_key: str) -> str:
    return next((group for group, classes in L3_SHORT.items() if short_key in classes), "")


def _get_display(short_key: str) -> str:
    return L3_DISPLAY.get(short_key, short_key)


def _mapped_per_class(metrics: dict) -> dict:
    return {key: value for name, value in metrics.get("l3_per_class", {}).items()
            if (key := _match_l3_key(name)) is not None}


def load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def plot_confusion_matrix(metrics: dict, output_dir: Path):
    """Plot 13 true classes against class and failure prediction columns.

    Rows are normalized by their complete support, including unknown and cascade
    outcomes. This prevents a conditional-on-success matrix from overstating L3
    performance.
    """
    raw = metrics.get("l3_confusion_matrix")
    labels = metrics.get("l3_confusion_labels", [])
    if not raw or not labels:
        print("  [SKIP] No confusion matrix data")
        return
    cm = np.asarray(raw, dtype=float)
    true_idx = [i for i, label in enumerate(labels) if _match_l3_key(label)]
    pred_idx = true_idx + [i for i, label in enumerate(labels) if label.lower() in {"unknown", "cascade_error"}]
    if not true_idx or not pred_idx:
        print("  [SKIP] No matched L3 labels")
        return
    row_totals = cm[true_idx, :].sum(axis=1, keepdims=True)
    shown = cm[np.ix_(true_idx, pred_idx)] / np.maximum(row_totals, 1)
    ylabels = [_get_display(_match_l3_key(labels[i])) for i in true_idx]
    xlabels = []
    for i in pred_idx:
        special = {"unknown": "Unknown", "cascade_error": "Cascade"}
        xlabels.append(special.get(labels[i].lower(), _get_display(_match_l3_key(labels[i]))))

    fig, ax = plt.subplots(figsize=(10.2, 7.3))
    im = ax.imshow(shown, cmap="Blues", vmin=0, vmax=max(0.6, float(shown.max())))
    for i in range(shown.shape[0]):
        for j in range(shown.shape[1]):
            value = shown[i, j]
            if value >= 0.005:
                ax.text(j, i, f"{value:.0%}" if value >= .1 else f"{value:.1%}",
                        ha="center", va="center", fontsize=6.6,
                        color="white" if value > im.norm.vmax * .55 else INK)
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=42, ha="right")
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_xlabel("Predicted outcome")
    ax.set_ylabel("True L3 class")
    ax.set_title("Failure-aware L3 confusion matrix", y=1.075)
    ax.text(0, 1.018, "Rows include Unknown and Cascade outcomes and therefore sum to 100%.",
            transform=ax.transAxes, color=GRAY, fontsize=8)
    for boundary in (2.5, 7.5, 12.5):
        ax.axvline(boundary, color="white", linewidth=1.5)
    for boundary in (2.5, 7.5):
        ax.axhline(boundary, color="white", linewidth=1.5)
    cbar = fig.colorbar(im, ax=ax, fraction=.035, pad=.025)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    cbar.set_label("Share of true-class samples")
    fig.tight_layout()
    _save_figure(fig, output_dir, "01_confusion_matrix")


def plot_hierarchical_accuracy(metrics: dict, output_dir: Path):
    vals = np.array([metrics.get(k, 0) for k in
                     ("l1_accuracy", "l2_accuracy", "l3_accuracy", "joint_accuracy")]) * 100
    labels = ["L1\nActive/passive", "L2\nSignal family", "L3\n13 classes", "Joint\nAll levels"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    bars = ax.bar(labels, vals, color=[NAVY, BLUE, ORANGE, PURPLE], width=.62)
    _annotate_bars(ax, bars, vals)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Hierarchical classification accuracy")
    ax.set_ylim(0, min(100, max(10, vals.max() * 1.2)))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    _style_axis(ax)
    fig.tight_layout()
    _save_figure(fig, output_dir, "02_hierarchical_accuracy")


def plot_per_class_metrics(metrics: dict, output_dir: Path):
    mapped = _mapped_per_class(metrics)
    classes = [c for c in sum(L3_SHORT.values(), []) if c in mapped]
    if not classes:
        print("  [SKIP] No per-class data")
        return
    f1 = np.array([mapped[c]["f1"] for c in classes]) * 100
    recall = np.array([mapped[c]["recall"] for c in classes]) * 100
    colors = [GROUP_COLORS[_get_group(c)] for c in classes]
    ymax = min(100, max(10, max(f1.max(), recall.max()) * 1.18))
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), sharey=True)
    for ax, values, title in zip(axes, (f1, recall), ("F1 score", "Recall")):
        bars = ax.bar(range(len(classes)), values, color=colors, width=.72)
        _annotate_bars(ax, bars, values, decimals=1)
        ax.set_xticks(range(len(classes)), [_get_display(c) for c in classes], rotation=42, ha="right")
        ax.set_title(title)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        _style_axis(ax)
    axes[0].set_ylabel("Score (%)")
    handles = [Patch(facecolor=GROUP_COLORS[g], label=GROUP_LABELS[g]) for g in L3_SHORT]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.suptitle("Per-class L3 performance", y=1.08, fontsize=12, fontweight="semibold")
    fig.tight_layout()
    _save_figure(fig, output_dir, "03_per_class_metrics")


def _short_bin_label(label: str) -> str:
    text = " ".join(label.split())
    return text.replace("  ", " ")


def plot_source_stratified(metrics: dict, output_dir: Path):
    src_data = metrics.get("source_stratified", {})
    specifications = {
        "PulseCom_TL": ("04a_pulsecom_tl", "Active signals by pre-normalization channel gain",
                        "Legacy field: tl_db; higher values indicate less attenuation / more retained energy."),
        "Ship_SNR": ("04b_ship_snr", "Ship noise by source/pre-channel line-spectrum SNR",
                     "SNR is measured before channel propagation; higher values indicate a clearer source line spectrum."),
    }
    for key, (stem, title, note) in specifications.items():
        bins = src_data.get(key, {}).get("bins", [])
        if not bins:
            continue
        labels = [_short_bin_label(b["label"]) for b in bins]
        acc = np.array([b["l3_acc"] for b in bins]) * 100
        cascade = np.array([b["cascade_rate"] for b in bins]) * 100
        counts = [b["count"] for b in bins]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
        for ax, values, subtitle, color in zip(axes, (acc, cascade),
                                               ("L3 accuracy", "Cascade rate"), (BLUE, RED)):
            bars = ax.bar(range(len(bins)), values, color=color, width=.62)
            _annotate_bars(ax, bars, values)
            ax.set_xticks(range(len(bins)), [f"{label}\nn={n}" for label, n in zip(labels, counts)])
            ax.set_ylabel("Rate (%)")
            ax.set_title(subtitle)
            ax.set_ylim(0, min(100, max(10, values.max() * 1.2)))
            ax.yaxis.set_major_formatter(mticker.PercentFormatter())
            _style_axis(ax)
        fig.suptitle(title, fontsize=12, fontweight="semibold", y=1.02)
        fig.text(.5, -.01, note, ha="center", fontsize=8, color=GRAY)
        fig.tight_layout()
        _save_figure(fig, output_dir, stem)


def plot_reasoning_quality(metrics: dict, total_samples: int, output_dir: Path):
    reasoning = metrics.get("reasoning", {})
    if not reasoning:
        print("  [SKIP] No reasoning data")
        return
    labels = ["Aligned", "Vague", "Concept confusion", "Term stacking"]
    values = np.array([reasoning.get(k, 0) for k in
                       ("alignment_rate", "vague_rate", "concept_confusion_rate", "term_stacking_rate")]) * 100
    skipped = reasoning.get("cascade_skipped", total_samples - round(total_samples * metrics.get("l1_accuracy", 0)))
    completed = total_samples - skipped
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    ax = axes[0]
    left = 0
    for value, color, label in zip(values, (TEAL, GRAY, RED, ORANGE), labels):
        ax.barh([0], [value], left=left, color=color, height=.48, label=f"{label}: {value:.1f}%")
        if value >= 6:
            ax.text(left + value / 2, 0, f"{value:.1f}%", ha="center", va="center",
                    color="white" if color != GRAY else INK, fontsize=8)
        left += value
    ax.set_xlim(0, max(100, left))
    ax.set_yticks([])
    ax.set_xlabel("Share of evaluated reasoning (%)")
    ax.set_title("Reasoning-quality labels")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, -0.55), ncol=2)
    _style_axis(ax, "x")
    ax = axes[1]
    bars = ax.bar(["Completed\n3 turns", "Stopped after\nTurn 1"], [completed, skipped], color=[BLUE, RED], width=.58)
    for bar, count in zip(bars, (completed, skipped)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_samples*.015,
                f"{count:,}\n({count/total_samples:.1%})", ha="center", fontsize=8)
    ax.set_ylabel("Samples")
    ax.set_title("Cascade termination")
    ax.set_ylim(0, max(completed, skipped) * 1.2)
    _style_axis(ax)
    fig.tight_layout(w_pad=3)
    _save_figure(fig, output_dir, "05_reasoning_cascade")


def plot_summary_dashboard(metrics: dict, output_dir: Path, model_name: str = ""):
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.3))
    acc = np.array([metrics.get(k, 0) for k in
                    ("l1_accuracy", "l2_accuracy", "l3_accuracy", "joint_accuracy")]) * 100
    ax = axes[0, 0]
    bars = ax.barh(["L1", "L2", "L3", "Joint"], acc, color=[NAVY, BLUE, ORANGE, PURPLE])
    for bar, value in zip(bars, acc):
        ax.text(value + .7, bar.get_y()+bar.get_height()/2, f"{value:.1f}%", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, max(65, acc.max()*1.18)); ax.set_xlabel("Accuracy (%)")
    ax.set_title("Hierarchical accuracy"); _style_axis(ax, "x")

    ax = axes[0, 1]
    cond = np.array([metrics.get("l2_given_l1", 0), metrics.get("l3_given_l2", 0)]) * 100
    bars = ax.bar(["L2 | L1", "L3 | L2"], cond, color=[BLUE, ORANGE], width=.55)
    _annotate_bars(ax, bars, cond); ax.set_ylim(0, min(100, max(10, cond.max()*1.2)))
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Conditional accuracy"); _style_axis(ax)

    ax = axes[1, 0]
    tiers = metrics.get("parse_tier_dist", {})
    tier_keys = sorted(tiers, key=lambda x: int(x))
    tier_vals = [tiers[k] for k in tier_keys]
    tier_colors = [TEAL, ORANGE, RED, PURPLE][:len(tier_vals)]
    bars = ax.barh([f"Tier {k}" for k in tier_keys], tier_vals, color=tier_colors)
    for bar, value in zip(bars, tier_vals):
        ax.text(value + max(tier_vals)*.02, bar.get_y()+bar.get_height()/2,
                f"{value:,} ({value/sum(tier_vals):.1%})", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, max(tier_vals)*1.28); ax.set_xlabel("Samples")
    ax.set_title("Response parsing tiers"); _style_axis(ax, "x")

    ax = axes[1, 1]
    reasoning = metrics.get("reasoning", {})
    rlabels = ["Alignment", "Vague", "Contradiction", "Concept confusion"]
    rvalues = np.array([reasoning.get(k, 0) for k in
                        ("alignment_rate", "vague_rate", "contradiction_rate", "concept_confusion_rate")]) * 100
    bars = ax.barh(rlabels, rvalues, color=[TEAL, GRAY, RED, ORANGE])
    for bar, value in zip(bars, rvalues):
        ax.text(value + .6, bar.get_y()+bar.get_height()/2, f"{value:.1f}%", va="center", fontsize=8)
    ax.invert_yaxis(); ax.set_xlim(0, max(70, rvalues.max()*1.18)); ax.set_xlabel("Rate (%)")
    ax.set_title("Reasoning diagnostics"); _style_axis(ax, "x")

    title = f"Evaluation summary — {metrics.get('total_samples', '?'):,} samples"
    if model_name:
        title = f"{model_name} — {title}"
    fig.suptitle(title, fontsize=13, fontweight="semibold", y=1.01)
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    _save_figure(fig, output_dir, "06_summary_dashboard")


def plot_l3_by_l2_parent(metrics: dict, output_dir: Path):
    mapped = _mapped_per_class(metrics)
    if not mapped:
        print("  [SKIP] No per-class data")
        return
    all_values = [mapped[c]["f1"]*100 for c in sum(L3_SHORT.values(), []) if c in mapped]
    ymax = min(100, max(10, max(all_values)*1.22))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=True)
    for ax, group in zip(axes, L3_SHORT):
        classes = [c for c in L3_SHORT[group] if c in mapped]
        values = np.array([mapped[c]["f1"] for c in classes]) * 100
        bars = ax.bar(range(len(classes)), values, color=GROUP_COLORS[group], width=.68)
        _annotate_bars(ax, bars, values)
        ax.set_xticks(range(len(classes)), [_get_display(c) for c in classes])
        ax.set_title(GROUP_LABELS[group]); ax.set_ylim(0, ymax); _style_axis(ax)
    axes[0].set_ylabel("F1 score (%)")
    fig.suptitle("L3 performance by signal family", fontsize=12, fontweight="semibold", y=1.02)
    fig.tight_layout()
    _save_figure(fig, output_dir, "07_l3_by_l2_parent")


def plot_top_confusion_pairs(metrics: dict, output_dir: Path, top_n: int = 15):
    raw = metrics.get("l3_confusion_matrix")
    labels = metrics.get("l3_confusion_labels", [])
    if not raw or not labels:
        print("  [SKIP] No confusion matrix data")
        return
    cm = np.asarray(raw)
    mapped = {i: _match_l3_key(label) for i, label in enumerate(labels) if _match_l3_key(label)}
    pairs = sorted(((int(cm[i, j]), mapped[i], mapped[j]) for i in mapped for j in mapped
                    if i != j and cm[i, j] > 0), reverse=True)[:top_n]
    if not pairs:
        print("  [SKIP] No meaningful confusion pairs")
        return
    pair_labels = [f"{_get_display(t).replace(chr(10), ' ')} → {_get_display(p).replace(chr(10), ' ')}" for _, t, p in pairs]
    counts = [count for count, _, _ in pairs]
    colors = [GROUP_COLORS[_get_group(t)] if _get_group(t) == _get_group(p) else RED for _, t, p in pairs]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    bars = ax.barh(range(len(pairs)), counts, color=colors, height=.68)
    ax.set_yticks(range(len(pairs)), pair_labels); ax.invert_yaxis()
    for bar, value in zip(bars, counts):
        ax.text(value + max(counts)*.012, bar.get_y()+bar.get_height()/2, str(value), va="center", fontsize=8)
    ax.set_xlim(0, max(counts)*1.12); ax.set_xlabel("Number of samples")
    ax.set_title(f"Top {len(pairs)} class-to-class confusions")
    handles = [Patch(facecolor=GROUP_COLORS[g], label=f"Within {GROUP_LABELS[g].lower()}") for g in L3_SHORT]
    handles.append(Patch(facecolor=RED, label="Cross-family"))
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower right")
    _style_axis(ax, "x"); fig.tight_layout()
    _save_figure(fig, output_dir, "08_top_confusion_pairs")


def plot_cascade_waterfall(metrics: dict, total_samples: int, output_dir: Path):
    reasoning = metrics.get("reasoning", {})
    skipped = reasoning.get("cascade_skipped", total_samples - round(total_samples*metrics.get("l1_accuracy", 0)))
    counts = [total_samples, total_samples-skipped]
    counts.append(round(counts[1]*metrics.get("l2_given_l1", 0)))
    counts.append(round(total_samples*metrics.get("joint_accuracy", 0)))
    labels = ["All samples", "L1 correct", "L1 + L2 correct", "All levels correct"]
    colors = [NAVY, BLUE, ORANGE, PURPLE]
    fig, ax = plt.subplots(figsize=(7.7, 4.5))
    bars = ax.bar(range(4), counts, color=colors, width=.6, zorder=2)
    ax.plot(range(4), counts, color=INK, linewidth=1, alpha=.35, zorder=3)
    for i, (bar, value) in enumerate(zip(bars, counts)):
        ax.text(bar.get_x()+bar.get_width()/2, value+total_samples*.018,
                f"{value:,}\n{value/total_samples:.1%}", ha="center", fontsize=8)
        if i:
            loss = counts[i-1]-value
            ax.text(i-.5, (counts[i-1]+value)/2, f"−{loss:,}", ha="center", va="center",
                    color=RED, fontsize=8, bbox={"facecolor":"white", "edgecolor":"none", "pad":1})
    ax.set_xticks(range(4), labels); ax.set_ylabel("Samples")
    ax.set_ylim(0, total_samples*1.14); ax.set_title("Cascade survival across evaluation turns")
    _style_axis(ax); fig.tight_layout()
    _save_figure(fig, output_dir, "09_cascade_waterfall")


def plot_precision_recall_scatter(metrics: dict, output_dir: Path):
    mapped = _mapped_per_class(metrics)
    if not mapped:
        print("  [SKIP] No per-class data")
        return
    fig, ax = plt.subplots(figsize=(7.2, 6.1))
    origin = []
    label_positions = {
        "CW": (8, 7), "LFM": (8, 8), "HFM": (8, 7),
        "2FSK": (8, 7), "cargo": (8, -14), "fishing": (8, 8),
        "warship": (8, -13), "underwater_target": (-8, 12),
    }
    for key, value in mapped.items():
        precision, recall = value["precision"]*100, value["recall"]*100
        ax.scatter(recall, precision, s=max(45, value.get("support", 1)*.55),
                   color=GROUP_COLORS[_get_group(key)], edgecolor="white", linewidth=.7, alpha=.9, zorder=3)
        label = _get_display(key).replace("\n", " ")
        if recall == 0 and precision == 0:
            origin.append(label)
        else:
            dx, dy = label_positions.get(key, (7, 5))
            ax.annotate(label, (recall, precision), xytext=(dx, dy),
                        textcoords="offset points", fontsize=7.2,
                        ha="right" if dx < 0 else "left",
                        va="top" if dy < 0 else "bottom")
    if origin:
        ax.text(2, 2, "At origin: " + ", ".join(origin), fontsize=7, color=GRAY, va="bottom")
    for f1 in (.1, .2, .3, .5, .7):
        recall = np.linspace(f1/2 + .002, 1, 400)
        precision = f1*recall/(2*recall-f1)
        valid = (precision >= 0) & (precision <= 1)
        ax.plot(recall[valid]*100, precision[valid]*100, color=GRAY, linestyle="--", linewidth=.65, alpha=.42)
        ax.text(99, (f1/(2-f1))*100, f"F1={f1:.1f}", ha="right", va="bottom", fontsize=6.7, color=GRAY)
    ax.set_xlim(0, 102); ax.set_ylim(0, 102)
    ax.set_xlabel("Recall (%)"); ax.set_ylabel("Precision (%)")
    ax.set_title("Per-class precision–recall profile")
    handles = [Line2D([0], [0], marker="o", linestyle="", color=GROUP_COLORS[g],
                      label=GROUP_LABELS[g], markersize=7) for g in L3_SHORT]
    ax.legend(handles=handles, frameon=False, loc="upper right")
    _style_axis(ax, "both"); fig.tight_layout()
    _save_figure(fig, output_dir, "10_precision_recall_scatter")


def main():
    global OUTPUT_FORMATS, OUTPUT_DPI
    parser = argparse.ArgumentParser(description="Generate publication-ready evaluation figures")
    parser.add_argument("--metrics", required=True, help="Path to eval_metrics_*.json")
    parser.add_argument("--output", default=None, help="Output directory; defaults to a figures folder beside metrics")
    parser.add_argument("--model-name", default=None, help="Optional model name for the summary title")
    parser.add_argument("--formats", default="png,pdf", help="Comma-separated output formats (default: png,pdf)")
    parser.add_argument("--dpi", type=int, default=300, help="Raster output DPI (default: 300)")
    args = parser.parse_args()
    OUTPUT_FORMATS = tuple(fmt.strip().lower() for fmt in args.formats.split(",") if fmt.strip())
    if not OUTPUT_FORMATS:
        parser.error("--formats must contain at least one format")
    OUTPUT_DPI = args.dpi
    _apply_publication_style()

    output_dir = Path(args.output) if args.output else Path(args.metrics).parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading: {args.metrics}")
    metrics = load_metrics(args.metrics)
    total = metrics.get("total_samples", 0)
    print(f"  Samples: {total:,}")
    print(f"Generating charts -> {output_dir}")
    plot_confusion_matrix(metrics, output_dir)
    plot_hierarchical_accuracy(metrics, output_dir)
    plot_per_class_metrics(metrics, output_dir)
    plot_source_stratified(metrics, output_dir)
    plot_reasoning_quality(metrics, total, output_dir)
    plot_summary_dashboard(metrics, output_dir, args.model_name or "")
    plot_l3_by_l2_parent(metrics, output_dir)
    plot_top_confusion_pairs(metrics, output_dir)
    plot_cascade_waterfall(metrics, total, output_dir)
    plot_precision_recall_scatter(metrics, output_dir)
    print(f"Done. {len(list(output_dir.glob('*.png')))} PNG figure(s) generated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

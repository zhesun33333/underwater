"""
Evaluation result visualization — generates analysis charts from eval_metrics_*.json.

Usage:
  python -m testsite.reporting.visualize --metrics eval_results/eval_metrics_xxxx.json
  python -m testsite.reporting.visualize --metrics eval_results/eval_metrics_xxxx.json --output figs/
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
import numpy as np

# L3 short keys grouped by L2 parent
L3_SHORT = {
    "pulse":       ["CW", "LFM", "HFM"],
    "communication":  ["2FSK", "4FSK", "BPSK", "QPSK", "OFDM"],
    "ship_noise":  ["cargo", "cruise", "fishing", "warship", "underwater_target"],
}
GROUP_COLORS = {"pulse": "#4ECDC4", "communication": "#FF6B6B", "ship_noise": "#45B7D1"}
GROUP_LABELS = {"pulse": "Pulse", "communication": "Communication", "ship_noise": "Ship Noise"}

# L3 short-key -> display label
L3_DISPLAY = {
    "CW": "CW", "LFM": "LFM", "HFM": "HFM",
    "2FSK": "2FSK", "4FSK": "4FSK", "BPSK": "BPSK",
    "QPSK": "QPSK", "OFDM": "OFDM",
    "cargo": "Cargo", "cruise": "Cruise", "fishing": "Fishing",
    "warship": "Warship", "underwater_target": "Underwater",
}

# Map full Chinese names -> short keys (for per_class dict keys from scorer)
_CN_TO_SHORT = {
    "CW连续波": "CW", "LFM线性调频": "LFM", "HFM双曲调频": "HFM",
    "2FSK二进制频移键控": "2FSK", "4FSK四进制频移键控": "4FSK",
    "BPSK二进制相移键控": "BPSK", "QPSK四进制相移键控": "QPSK",
    "OFDM正交频分复用": "OFDM",
    "货船": "cargo", "邮轮": "cruise", "渔船": "fishing",
    "军舰": "warship", "水下目标": "underwater_target",
}


def _match_l3_key(name: str) -> str | None:
    """Match a per_class key (Chinese or short) to L3 short key."""
    if name in L3_SHORT["pulse"] + L3_SHORT["communication"] + L3_SHORT["ship_noise"]:
        return name
    if name in _CN_TO_SHORT:
        return _CN_TO_SHORT[name]
    # substring fallback: check if any short key appears in the name
    name_lower = name.lower()
    for sk in L3_SHORT["pulse"] + L3_SHORT["communication"] + L3_SHORT["ship_noise"]:
        if sk.lower() in name_lower:
            return sk
    return None


def _get_group(short_key: str) -> str:
    for gname, gclasses in L3_SHORT.items():
        if short_key in gclasses:
            return gname
    return ""


def _get_display(short_key: str) -> str:
    return L3_DISPLAY.get(short_key, short_key)


def load_metrics(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# 01: L3 Confusion Matrix (13-class heatmap)
# ============================================================
def plot_confusion_matrix(metrics: dict, output_dir: Path):
    cm_data = metrics.get("l3_confusion_matrix")
    cm_labels = metrics.get("l3_confusion_labels", [])
    if not cm_data or not cm_labels:
        print("  [SKIP] No confusion matrix data")
        return

    # Convert labels to short display form, exclude special labels
    display_labels = []
    valid_idx = []
    skipped_labels = []
    for i, lbl in enumerate(cm_labels):
        sk = _match_l3_key(lbl)
        if sk:
            display_labels.append(_get_display(sk))
            valid_idx.append(i)
        else:
            skipped_labels.append(lbl)

    if skipped_labels:
        print(f"  [INFO] Excluded special labels from confusion matrix: {skipped_labels}")
    if len(valid_idx) < 2:
        print("  [SKIP] Not enough valid class labels in confusion matrix")
        return

    # Filter matrix to valid classes only
    cm = np.array(cm_data)
    cm_filtered = cm[np.ix_(valid_idx, valid_idx)]
    cm_norm = cm_filtered.astype(float) / (cm_filtered.sum(axis=1, keepdims=True) + 1e-9)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            if val > 0.005:
                color = "white" if val > 0.5 else "black"
                txt = f"{val:.0%}" if val >= 0.10 else f"{val:.1%}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(len(display_labels)))
    ax.set_yticks(range(len(display_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(display_labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("L3 Confusion Matrix (Normalized)", fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Fraction", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_dir / "01_confusion_matrix.png", dpi=150)
    plt.close(fig)
    print("  [OK] 01_confusion_matrix.png")


# ============================================================
# 02: Hierarchical Accuracy (L1/L2/L3/Joint bar)
# ============================================================
def plot_hierarchical_accuracy(metrics: dict, output_dir: Path):
    levels = ["L1", "L2", "L3", "Joint"]
    accs = [
        metrics.get("l1_accuracy", 0),
        metrics.get("l2_accuracy", 0),
        metrics.get("l3_accuracy", 0),
        metrics.get("joint_accuracy", 0),
    ]
    labels_full = [
        "L1\n(Active/Passive)",
        "L2\n(Pulse/Comm/Ship)",
        "L3\n(13 Classes)",
        "Joint\n(All 3 Correct)",
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(levels))
    values_pct = [v * 100 for v in accs]
    colors = ["#2ECC71", "#3498DB", "#E74C3C", "#9B59B6"]
    bars = ax.bar(x, values_pct, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1%}", ha="center", fontsize=12, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_full, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Hierarchical Classification Accuracy", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(max(values_pct) * 1.25, 10))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "02_hierarchical_accuracy.png", dpi=150)
    plt.close(fig)
    print("  [OK] 02_hierarchical_accuracy.png")


# ============================================================
# 03: Per-Class F1 + Accuracy (grouped bar, color-coded by L2 parent)
# ============================================================
def plot_per_class_metrics(metrics: dict, output_dir: Path):
    per_class_raw = metrics.get("l3_per_class", {})
    if not per_class_raw:
        print("  [SKIP] No per-class data")
        return

    # Map to short keys
    mapped = {}
    unmatched = []
    for name, m in per_class_raw.items():
        sk = _match_l3_key(name)
        if sk:
            mapped[sk] = m
        else:
            unmatched.append(name)

    ordered = (L3_SHORT["pulse"] + L3_SHORT["communication"] + L3_SHORT["ship_noise"])
    classes = [c for c in ordered if c in mapped]

    if not classes:
        if unmatched:
            print(f"  [WARN] Failed to match per_class keys: {unmatched[:5]}...")
        print("  [SKIP] No valid classes matched in per-class data")
        return
    if unmatched:
        print(f"  [INFO] Unmatched per_class keys (skipped): {len(unmatched)} entries")

    f1s = [mapped[c]["f1"] * 100 for c in classes]
    recalls = [mapped[c]["recall"] * 100 for c in classes]

    colors = []
    for c in classes:
        g = _get_group(c)
        colors.append(GROUP_COLORS.get(g, "#AAAAAA"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # F1 bar
    bars = ax1.bar(range(len(classes)), f1s, color=colors, edgecolor="white")
    for bar, val in zip(bars, f1s):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", fontsize=7, fontweight="bold")
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels([_get_display(c) for c in classes], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("F1 Score (%)", fontsize=11)
    ax1.set_title("Per-Class F1 Score", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)

    # Recall bar
    bars = ax2.bar(range(len(classes)), recalls, color=colors, edgecolor="white")
    for bar, val in zip(bars, recalls):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f"{val:.1f}%", ha="center", fontsize=7, fontweight="bold")
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels([_get_display(c) for c in classes], rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Recall (%)", fontsize=11)
    ax2.set_title("Per-Class Recall", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    legend_elements = [
        Patch(facecolor=GROUP_COLORS["pulse"], label="Pulse (CW/LFM/HFM)"),
        Patch(facecolor=GROUP_COLORS["communication"], label="Comm (FSK/PSK/OFDM)"),
        Patch(facecolor=GROUP_COLORS["ship_noise"], label="Ship Noise"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "03_per_class_metrics.png", dpi=150)
    plt.close(fig)
    print("  [OK] 03_per_class_metrics.png")


# ============================================================
# 04: Source-stratified quality analysis
# ============================================================
def plot_source_stratified(metrics: dict, output_dir: Path):
    src_data = metrics.get("source_stratified", {})
    if not src_data:
        print("  [SKIP] No source_stratified data")
        return

    for src_key in ["PulseCom_TL", "Ship_SNR"]:
        src = src_data.get(src_key, {})
        bins = src.get("bins", [])
        if not bins:
            continue

        labels = [b["label"] for b in bins]
        l3_accs = [b["l3_acc"] * 100 for b in bins]
        cascade_rates = [b["cascade_rate"] * 100 for b in bins]

        title_map = {
            "PulseCom_TL": "PulseCom  |  Transmission Loss (TL)  |  higher = cleaner",
            "Ship_SNR":    "Ship  |  Line-Spectrum SNR  |  higher = clearer",
        }

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        x = np.arange(len(labels))

        bars1 = ax1.bar(x, l3_accs, color="#3498DB", edgecolor="white")
        for bar, val in zip(bars1, l3_accs):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax1.set_ylabel("L3 Accuracy (%)", fontsize=11)
        ax1.set_title(f"{title_map.get(src_key, src_key)}\nL3 Accuracy", fontsize=12, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)

        bars2 = ax2.bar(x, cascade_rates, color="#E74C3C", edgecolor="white")
        for bar, val in zip(bars2, cascade_rates):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=8, rotation=15, ha="right")
        ax2.set_ylabel("Cascade Rate (%)", fontsize=11)
        ax2.set_title("Cascade Rate (L1 wrong -> skip T2/T3)", fontsize=12, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        for i, b in enumerate(bins):
            ax1.text(i, -max(l3_accs) * 0.12, f"n={b['count']}", ha="center", fontsize=8, color="gray")

        fname = "04a_pulsecom_tl.png" if src_key == "PulseCom_TL" else "04b_ship_snr.png"
        fig.tight_layout()
        fig.savefig(output_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  [OK] {fname}")


# ============================================================
# 05: Reasoning quality + Cascade
# ============================================================
def plot_reasoning_quality(metrics: dict, total_samples: int, output_dir: Path):
    reasoning = metrics.get("reasoning", {})
    if not reasoning:
        print("  [SKIP] No reasoning data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie: reasoning quality distribution
    labels = ["Aligned", "Vague", "Confused", "Stacked"]
    sizes = [
        reasoning.get("alignment_rate", 0) * 100,
        reasoning.get("vague_rate", 0) * 100,
        reasoning.get("concept_confusion_rate", 0) * 100,
        reasoning.get("term_stacking_rate", 0) * 100,
    ]
    colors_pie = ["#2ECC71", "#BDC3C7", "#E74C3C", "#F39C12"]

    # autopct: only show % for slices > 3%, suppress small overlapping labels
    def _fmt_pct(pct):
        return f"{pct:.1f}%" if pct > 3 else ""

    wedges, texts, autotexts = ax1.pie(
        sizes, explode=(0.05, 0, 0, 0), colors=colors_pie,
        autopct=_fmt_pct, startangle=90, textprops={"fontsize": 10})

    # Build legend with exact values
    legend_labels = [f"{lbl}: {val:.1f}%" for lbl, val in zip(labels, sizes)]
    ax1.legend(wedges, legend_labels, loc="lower center", ncol=2, fontsize=9)
    ax1.set_title("Reasoning Quality Distribution", fontsize=13, fontweight="bold")

    # Bar: cascade stats (fallback from L1 accuracy if field not stored)
    if "cascade_skipped" in reasoning:
        cascade_skipped = reasoning["cascade_skipped"]
    else:
        l1_acc = metrics.get("l1_accuracy", 0)
        cascade_skipped = total_samples - int(total_samples * l1_acc)
    cascade_not = total_samples - cascade_skipped

    bars = ax2.bar(["Full (3 Turns)", "Cascade (T1 only)"],
                   [cascade_not, cascade_skipped],
                   color=["#3498DB", "#E74C3C"], edgecolor="white")
    for bar, val in zip(bars, [cascade_not, cascade_skipped]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total_samples * 0.01,
                 f"{val} ({val/total_samples:.1%})", ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Samples", fontsize=11)
    ax2.set_title("Cascade Termination", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "05_reasoning_cascade.png", dpi=150)
    plt.close(fig)
    print("  [OK] 05_reasoning_cascade.png")


# ============================================================
# 06: Summary dashboard (4 panels)
# ============================================================
def plot_summary_dashboard(metrics: dict, output_dir: Path, model_name: str = ""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top-left: Hierarchical accuracy
    ax = axes[0, 0]
    levels = ["L1", "L2", "L3", "Joint"]
    vals = [
        metrics.get("l1_accuracy", 0) * 100,
        metrics.get("l2_accuracy", 0) * 100,
        metrics.get("l3_accuracy", 0) * 100,
        metrics.get("joint_accuracy", 0) * 100,
    ]
    bars = ax.barh(levels, vals, color=["#2ECC71", "#3498DB", "#E74C3C", "#9B59B6"],
                   edgecolor="white")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")
    ax.set_xlabel("Accuracy (%)", fontsize=10)
    ax.set_title("Hierarchical Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.3)
    ax.grid(axis="x", alpha=0.3)

    # Top-right: Conditional accuracy
    ax = axes[0, 1]
    conds = {
        "L2 | L1": metrics.get("l2_given_l1", 0) * 100,
        "L3 | L2": metrics.get("l3_given_l2", 0) * 100,
    }
    bars = ax.bar(conds.keys(), conds.values(), color=["#3498DB", "#E74C3C"],
                  edgecolor="white", width=0.4)
    for bar, val in zip(bars, conds.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("Conditional Accuracy", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Bottom-left: Parse tier distribution
    ax = axes[1, 0]
    tiers = metrics.get("parse_tier_dist", {})
    if tiers:
        tier_labels = [f"Tier {k}" for k in sorted(tiers.keys())]
        tier_vals = [tiers[k] for k in sorted(tiers.keys())]
        ax.pie(tier_vals, labels=tier_labels, autopct="%1.1f%%",
               colors=["#2ECC71", "#F39C12", "#E74C3C"], textprops={"fontsize": 10})
        ax.set_title("Parse Tier Distribution", fontsize=12, fontweight="bold")

    # Bottom-right: Reasoning quality
    ax = axes[1, 1]
    reasoning = metrics.get("reasoning", {})
    if reasoning:
        r_metrics = {
            "Alignment": reasoning.get("alignment_rate", 0) * 100,
            "Contradiction": reasoning.get("contradiction_rate", 0) * 100,
            "Vague": reasoning.get("vague_rate", 0) * 100,
        }
        bars = ax.bar(r_metrics.keys(), r_metrics.values(),
                      color=["#2ECC71", "#E74C3C", "#BDC3C7"], edgecolor="white", width=0.4)
        for bar, val in zip(bars, r_metrics.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylabel("Rate (%)", fontsize=10)
        ax.set_title("Reasoning Quality", fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    title = f"Evaluation Summary  |  {metrics.get('total_samples', '?')} samples"
    if model_name:
        title = f"{model_name}  —  {title}"
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_dir / "06_summary_dashboard.png", dpi=150)
    plt.close(fig)
    print("  [OK] 06_summary_dashboard.png")


# ============================================================
# 07: L3 accuracy by L2 parent (3 grouped bar charts)
# ============================================================
def plot_l3_by_l2_parent(metrics: dict, output_dir: Path):
    per_class_raw = metrics.get("l3_per_class", {})
    if not per_class_raw:
        print("  [SKIP] No per-class data")
        return

    mapped = {}
    for name, m in per_class_raw.items():
        sk = _match_l3_key(name)
        if sk:
            mapped[sk] = m

    if not mapped:
        print("  [SKIP] No valid classes matched in per-class data")
        return

    groups = {
        "Pulse\n(CW/LFM/HFM)": ("pulse", GROUP_COLORS["pulse"]),
        "Comm\n(FSK/PSK/OFDM)": ("communication", GROUP_COLORS["communication"]),
        "Ship Noise\n(5 types)": ("ship_noise", GROUP_COLORS["ship_noise"]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (gtitle, (gkey, gcolor)) in zip(axes, groups.items()):
        classes = [c for c in L3_SHORT[gkey] if c in mapped]
        if not classes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(gtitle, fontsize=12)
            continue
        vals = [mapped[c]["f1"] * 100 for c in classes]
        x = np.arange(len(classes))

        bars = ax.bar(x, vals, color=gcolor, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([_get_display(c) for c in classes], fontsize=9)
        ax.set_title(gtitle, fontsize=12, fontweight="bold")
        ax.set_ylabel("F1 Score (%)", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 100)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("L3 F1 Score by L2 Parent Class", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "07_l3_by_l2_parent.png", dpi=150)
    plt.close(fig)
    print("  [OK] 07_l3_by_l2_parent.png")


# ============================================================
# 08: Top confusion pairs (exclude cascade_error/unknown)
# ============================================================
def plot_top_confusion_pairs(metrics: dict, output_dir: Path, top_n: int = 15):
    cm_data = metrics.get("l3_confusion_matrix")
    cm_labels_raw = metrics.get("l3_confusion_labels", [])
    if not cm_data or not cm_labels_raw:
        print("  [SKIP] No confusion matrix data")
        return

    cm = np.array(cm_data)
    n = len(cm_labels_raw)

    # Map labels: Chinese -> short key; skip special labels
    label_map = {}  # idx -> short_key
    for i, lbl in enumerate(cm_labels_raw):
        sk = _match_l3_key(lbl)
        if sk:
            label_map[i] = sk

    # Extract class-to-class confusion pairs (exclude self, exclude special labels)
    pairs = []
    for i_true in label_map:
        for j_pred in label_map:
            if i_true != j_pred and cm[i_true][j_pred] > 0:
                pairs.append((cm[i_true][j_pred],
                             _get_display(label_map[i_true]),
                             _get_display(label_map[j_pred])))

    pairs.sort(reverse=True)
    pairs = pairs[:top_n]

    if not pairs:
        print("  [SKIP] No meaningful confusion pairs")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    labels = [f"{t} -> {p}" for _, t, p in pairs]
    counts = [c for c, _, _ in pairs]

    # Color by whether within-group or cross-group confusion
    colors = []
    for _, t_display, p_display in pairs:
        # Find short keys from display
        t_sk = {v: k for k, v in L3_DISPLAY.items()}.get(t_display)
        p_sk = {v: k for k, v in L3_DISPLAY.items()}.get(p_display)
        if t_sk and p_sk:
            g_t = _get_group(t_sk)
            g_p = _get_group(p_sk)
            if g_t == g_p:
                colors.append(GROUP_COLORS.get(g_t, "#AAAAAA"))
            else:
                colors.append("#E74C3C")  # cross-group = red
        else:
            colors.append("#AAAAAA")

    bars = ax.barh(range(len(labels)), counts, color=colors, edgecolor="white")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Confusion Count", fontsize=11)
    ax.set_title(f"Top {top_n} Confused Class Pairs\n(Colored = same L2 group, Red = cross-group)",
                 fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "08_top_confusion_pairs.png", dpi=150)
    plt.close(fig)
    print("  [OK] 08_top_confusion_pairs.png")


# ============================================================
# 09: Cascade waterfall (sample survival across turns)
# ============================================================
def plot_cascade_waterfall(metrics: dict, total_samples: int, output_dir: Path):
    reasoning = metrics.get("reasoning", {})
    if "cascade_skipped" in reasoning:
        cascade_skipped = reasoning["cascade_skipped"]
    else:
        l1_acc = metrics.get("l1_accuracy", 0)
        cascade_skipped = total_samples - int(total_samples * l1_acc)

    t1_ok = total_samples - cascade_skipped

    l2_given_l1 = metrics.get("l2_given_l1", 0)
    joint_acc = metrics.get("joint_accuracy", 0)
    t2_ok = int(t1_ok * l2_given_l1)
    joint_n = int(total_samples * joint_acc)

    fig, ax = plt.subplots(figsize=(10, 6))

    stages = [
        ("Total\nSamples", total_samples, "#34495E"),
        ("T1 OK\n(L1 correct)", t1_ok, "#2ECC71"),
        ("T2 OK\n(L1+L2 correct)", t2_ok, "#3498DB"),
        ("T3 OK\n(All 3 correct)", joint_n, "#9B59B6"),
    ]
    x = np.arange(len(stages))
    heights = [s[1] for s in stages]
    colors = [s[2] for s in stages]
    bars = ax.bar(x, heights, color=colors, edgecolor="white", width=0.6)

    for i, (bar, h) in enumerate(zip(bars, heights)):
        pct = h / total_samples * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total_samples * 0.01,
                f"{h}\n({pct:.1f}%)", ha="center", fontsize=10, fontweight="bold")

    # Loss annotations
    losses = [
        (total_samples - t1_ok, "L1 wrong", "#E74C3C"),
        (t1_ok - t2_ok, "L2 wrong", "#F39C12"),
        (t2_ok - joint_n, "L3 wrong", "#E67E22"),
    ]
    prev_heights = [total_samples, t1_ok, t2_ok]
    for i, (loss_n, loss_label, loss_color) in enumerate(losses):
        mid_y = (heights[i + 1] + prev_heights[i]) / 2
        ax.annotate(
            f"-{loss_n}\n{loss_label}",
            xy=(i + 0.5, mid_y), fontsize=9, color=loss_color,
            ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor=loss_color),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([s[0] for s in stages], fontsize=10)
    ax.set_ylabel("Samples", fontsize=11)
    ax.set_title("Cascade Flow: Sample Survival at Each Turn", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "09_cascade_waterfall.png", dpi=150)
    plt.close(fig)
    print("  [OK] 09_cascade_waterfall.png")


# ============================================================
# 10: Precision-Recall scatter
# ============================================================
def plot_precision_recall_scatter(metrics: dict, output_dir: Path):
    per_class_raw = metrics.get("l3_per_class", {})
    if not per_class_raw:
        print("  [SKIP] No per-class data")
        return

    mapped = {}
    for name, m in per_class_raw.items():
        sk = _match_l3_key(name)
        if sk:
            mapped[sk] = m

    if not mapped:
        print("  [SKIP] No valid classes matched for PR scatter")
        return

    fig, ax = plt.subplots(figsize=(12, 9))

    origin_labels = []  # classes stuck at (0,0), will list separately

    for sk, m in mapped.items():
        prec = m["precision"] * 100
        rec = m["recall"] * 100
        support = m["support"]
        g = _get_group(sk)
        gcolor = GROUP_COLORS.get(g, "#AAAAAA")

        ax.scatter(rec, prec, s=max(40, support * 2),
                   c=gcolor, edgecolors="white", linewidth=0.5,
                   alpha=0.85, zorder=3)

        label = _get_display(sk)
        if rec < 2 and prec > 80:
            label = f"{label}\n(R={rec:.1f}%, P={prec:.0f}%)"
        elif rec < 2 and prec < 10:
            if rec < 0.1 and prec < 0.1:
                origin_labels.append(label)
                continue  # don't annotate overlapping origin points
            label = f"{label}\n(R={rec:.1f}%, P={prec:.0f}%)"
        ax.annotate(label, (rec, prec),
                    textcoords="offset points", xytext=(7, 3),
                    fontsize=7, fontweight="bold")

    if origin_labels:
        ax.text(2, 5, "At (0,0): " + ", ".join(origin_labels),
                fontsize=7, color="gray", fontstyle="italic")

    # F1 iso-lines
    for f1_val in [0.1, 0.2, 0.3, 0.5, 0.7]:
        r = np.linspace(0.01, 1, 100)
        p = f1_val * r / (2 * r - f1_val * r)
        valid = (p > 0) & (p <= 1)
        ax.plot(r[valid] * 100, p[valid] * 100, "k--", alpha=0.15, linewidth=0.8)
        idx_end = np.where(valid)[0][-1]
        ax.text(r[idx_end] * 100 + 0.5, p[idx_end] * 100, f"F1={f1_val:.1f}",
                fontsize=7, alpha=0.35)

    ax.set_xlabel("Recall (%)", fontsize=12)
    ax.set_ylabel("Precision (%)", fontsize=12)
    ax.set_title("Per-Class Precision-Recall  (bubble size ~ sample count)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(-2, 105)
    ax.set_ylim(-2, 105)
    ax.grid(alpha=0.3)

    legend_elements = [
        Patch(facecolor=GROUP_COLORS["pulse"], label="Pulse"),
        Patch(facecolor=GROUP_COLORS["communication"], label="Communication"),
        Patch(facecolor=GROUP_COLORS["ship_noise"], label="Ship Noise"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "10_precision_recall_scatter.png", dpi=150)
    plt.close(fig)
    print("  [OK] 10_precision_recall_scatter.png")


# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluation result visualization")
    parser.add_argument("--metrics", required=True, help="Path to eval_metrics_*.json")
    parser.add_argument("--output", default=None, help="Output directory (auto-detect from --metrics if not set)")
    parser.add_argument("--model-name", default=None, help="Model name for chart titles (e.g. Qwen2-Audio-7B)")
    args = parser.parse_args()

    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(args.metrics).parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = args.model_name or ""

    print(f"Loading: {args.metrics}")
    metrics = load_metrics(args.metrics)
    total = metrics.get("total_samples", 0)
    print(f"  Samples: {total}")

    print(f"\nGenerating charts -> {output_dir}/")
    plot_confusion_matrix(metrics, output_dir)
    plot_hierarchical_accuracy(metrics, output_dir)
    plot_per_class_metrics(metrics, output_dir)
    plot_source_stratified(metrics, output_dir)
    plot_reasoning_quality(metrics, total, output_dir)
    plot_summary_dashboard(metrics, output_dir, model_name)
    plot_l3_by_l2_parent(metrics, output_dir)
    plot_top_confusion_pairs(metrics, output_dir)
    plot_cascade_waterfall(metrics, total, output_dir)
    plot_precision_recall_scatter(metrics, output_dir)

    print(f"\nDone. {len(list(output_dir.glob('*.png')))} figures generated.")


if __name__ == "__main__":
    sys.exit(main())

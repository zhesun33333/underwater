"""Publication layouts for D-G, consuming prepared statistics only."""
from __future__ import annotations

import numpy as np

from .plot_acoustic import FAMILY_COLORS, _axis
from .result_analysis import CLASS_ORDER, CLASS_TO_FAMILY, DISPLAY_NAMES, model_performance_range


def _figure(data, height_key, height_ratio=17/36):
    import matplotlib.pyplot as plt
    width_mm = data["style"]["width_mm"]
    height_mm = data["style"].get(height_key, width_mm * height_ratio)
    if not np.isfinite(height_mm) or height_mm <= 0:
        raise ValueError(f"{height_key} must be a positive finite height in mm")
    return plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=300)


def _preview(data):
    return data["mode"] == "legacy_preview"


def _run_source_note(data):
    if _preview(data):
        return "Legacy preview: stored labels; protocol verification unavailable"
    if data["mode"] == "archived_verified":
        return "Archived predictions checked against their saved manifest and protocol"
    return "Predictions verified against the current manifest and run protocol"


def _metadata_styles(runs):
    """Keep registered model identities stable across reorderings and subsets."""
    ids = ("af_next", "gemma4_12b", "gemma4_run1", "midashenglm", "qwen25_omni",
           "voxtral_small", "qwen2_audio", "aero1_audio", "voxtral_mini", "gemma4_e2b")
    colors = ("#4C78A8", "#9A6BB3", "#D89032", "#238B78", "#D96B5F",
              "#566978", "#3D8FB8", "#8C6D46", "#B34D82", "#7B8E35")
    markers = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h")
    dashes = (None, (6, 2), (1, 1.6), (4, 1.5, 1, 1.5), (6, 1.5, 2, 1.5),
              (4, 1.3, 1, 1.3, 1, 1.3), (2, 1.5), (8, 2), (2, 1, 2, 3),
              (3, 1, 1, 1, 1, 1))
    slots = {model_id: i for i, model_id in enumerate(ids)}
    used = {slots[r["id"]] for r in runs if r["id"] in slots}
    free = iter(i for i in range(len(ids)) if i not in used)
    for model_id in sorted(r["id"] for r in runs if r["id"] not in slots):
        slots[model_id] = next(free)
    return {r["id"]: {"color": colors[slots[r["id"]]], "marker": markers[slots[r["id"]]],
                       "linestyle": "-" if dashes[slots[r["id"]]] is None else (0, dashes[slots[r["id"]]])}
            for r in runs}


def build_class_diagnostics(data):
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import PercentFormatter
    run = next((r for r in data["runs"] if r["id"] == data["diagnostic_model"]), None)
    if run is None:
        raise ValueError("D requires a configured diagnostic_model with a complete confusion matrix")
    values = run["diagnostics"]
    fig = _figure(data, "diagnostic_height_mm", 2/3)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.1, 1], left=0.12, right=0.985,
                           bottom=0.28, top=0.87, wspace=0.52)
    matrix_ax, score_ax = (fig.add_subplot(grid[0, i]) for i in range(2))
    fraction = np.asarray(values["row_fraction"])
    palette = [(0., "#E4EFF4"), (.15, "#B9D9E3"), (.4, "#66AABB"),
               (.7, "#28758B"), (1., "#12384F")]
    cmap = LinearSegmentedColormap.from_list("uab_mist_teal", palette)
    color_scale = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    # Convert vertical data units so each glyph is square in physical space.
    box = matrix_ax.get_position()
    cell_aspect = (box.width * fig.get_figwidth() / 15) / (box.height * fig.get_figheight() / 13)
    zero_rows, zero_cols = np.where(fraction == 0)
    matrix_ax.scatter(zero_cols, zero_rows, s=2, c="#CFD8DE", linewidths=0, zorder=1)
    for row, col in np.argwhere(fraction > 0):
        cell = float(fraction[row, col])
        # Minimum side keeps small nonzero entries visible. Area is NOT proportional to p.
        side = .45 + .50 * np.sqrt(cell)
        height = side * cell_aspect
        tile = FancyBboxPatch((col-side/2, row-height/2), side, height,
                             boxstyle=f"round,pad=0,rounding_size={side*.19}",
                             mutation_aspect=cell_aspect, facecolor=color_scale.to_rgba(cell),
                             edgecolor="none", linewidth=0, zorder=2, rasterized=False)
        tile.set_gid(f"confusion-{row}-{col}")
        matrix_ax.add_patch(tile)
    matrix_ax.set_xlim(-0.5, 14.5)
    matrix_ax.set_ylim(12.5, -0.5)
    column_names = DISPLAY_NAMES[:12] + ["Underw.", "Unres.", "Cascade"]
    matrix_ax.set_xticks(range(15), column_names, rotation=90)
    matrix_ax.set_yticks(range(13), DISPLAY_NAMES)
    matrix_ax.set_xlabel("Predicted class", labelpad=3)
    matrix_ax.set_ylabel("True class", labelpad=2)
    matrix_ax.set_title("(a) Confusion", pad=5)
    for row, col in np.argwhere(fraction >= values["annotation_threshold_fraction"]):
        cell = fraction[row, col]
        rgb = np.asarray(color_scale.to_rgba(cell)[:3])
        linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
        luminance = float(linear @ np.array([.2126, .7152, .0722]))
        matrix_ax.text(col, row, f"{100*cell:.0f}", ha="center", va="center",
                       color="white" if luminance < .179 else "black", zorder=3)
    for boundary in (2.5, 7.5):
        matrix_ax.axhline(boundary, color="#DCE5E9", linewidth=0.5, zorder=0)
        matrix_ax.axvline(boundary, color="#DCE5E9", linewidth=0.5, zorder=0)
    matrix_ax.axvline(12.5, color="#C4D3DB", linewidth=0.5, zorder=0)
    for offset, key, marker, label in ((-.21, "precision", "o", "P"),
                                      (0, "recall", "^", "R"), (.21, "f1", "s", "F1")):
        score_ax.plot(values[key], np.arange(13)+offset, linestyle="none", marker=marker,
                      markersize=3, color="black", markerfacecolor="white" if key == "recall" else "black",
                      markeredgewidth=0.75, label=label)
    score_ax.set_xlim(-.035, 1.035)
    score_ax.set_ylim(12.5, -0.5)
    score_ax.set_xticks([0, .5, 1])
    score_ax.set_yticks(range(13), DISPLAY_NAMES)
    score_ax.set_xlabel("Score", labelpad=3)
    score_ax.set_ylabel("Class", labelpad=2)
    score_ax.set_title("(b) Class metrics", pad=5)
    score_ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.045), ncol=3,
                    frameon=data["style"]["legend_frame"], handletextpad=.3, columnspacing=.7)
    for boundary in (2.5, 7.5):
        score_ax.axhline(boundary, color="0.8", linewidth=0.5)
    for ax in (matrix_ax, score_ax):
        _axis(ax)
        for tick, key in zip(ax.get_yticklabels(), CLASS_ORDER):
            tick.set_color(FAMILY_COLORS[CLASS_TO_FAMILY[key]])
    cax = fig.add_axes([.14, .12, .36, .018])
    bar = fig.colorbar(color_scale, cax=cax, orientation="horizontal")
    bar.solids.set_rasterized(False)
    bar.set_ticks([0, .5, 1])
    bar.ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    bar.set_label("True-class share (%)", labelpad=2)
    _axis(cax)
    fig.text(.755, .098, "Larger, darker squares: higher share\nDots: zero observations",
             ha="center", va="center", linespacing=1.5)
    fig.suptitle(f"{run['label']}  |  n = {values['n']:,}", y=.99)
    fig.text(.53, .015, _run_source_note(data), ha="center", va="bottom")
    return fig, {"figure": "class_diagnostics", "model": run["label"], "mode": data["mode"],
                 "audit": run["audit"], "diagnostics": values,
                 "normalization": "all samples of each true class, including unresolved and cascaded predictions",
                 "heatmap_encoding": {"glyph": "rounded square", "palette": palette,
                                      "color_norm": "linear", "color_limits": [0, 1],
                                      "positive_side_in_cell_widths": "0.45 + 0.50 * sqrt(row_fraction)",
                                      "size_meaning": "monotonic cue; area is not proportional to probability",
                                      "corner_radius_fraction_of_side": .19,
                                      "zero": "small neutral dot; no colored square"},
                 "family_colors": FAMILY_COLORS}


def build_error_decomposition(data):
    from matplotlib.ticker import PercentFormatter
    runs = [r for r in data["runs"] if r["outcomes"] is not None]
    if not runs:
        raise ValueError("E requires complete sample-level predictions; aggregate accuracies cannot supply stage counts")
    states = runs[0]["outcomes"]["states"]
    if any(r["outcomes"]["states"] != states for r in runs):
        raise ValueError("Do not mix incompatible stage definitions in one E figure")
    colors = (["#D8A07A", "#AEB7C2", "#458B74"] if len(states) == 3
              else ["#E6BFA3", "#CC8D68", "#CFD3DA", "#8D9CAC", "#458B74"])
    fig = _figure(data, "error_height_mm")
    ax = fig.add_axes([.245, .20, .58, .60])
    left = np.zeros(len(runs))
    for index, (state, color) in enumerate(zip(states, colors)):
        sizes = np.asarray([r["outcomes"]["fractions"][index] for r in runs])
        ax.barh(np.arange(len(runs)), sizes, left=left, height=.62, color=color,
                edgecolor="white", linewidth=.5, label=state)
        for row, size in enumerate(sizes):
            if size >= .075:
                ax.text(left[row]+size/2, row, f"{size*100:.1f}", ha="center", va="center")
        left += sizes
    ax.set_xlim(0, 1)
    ax.set_ylim(len(runs)-.5, -.5)
    ax.set_yticks(range(len(runs)), [r["label"] for r in runs])
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_xlabel("All evaluation records (%)")
    ax.set_ylabel("Model")
    _axis(ax)
    for i, run in enumerate(runs):
        ax.text(1.025, i, f"n = {run['outcomes']['n']:,}", va="center", transform=ax.get_yaxis_transform())
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.52, .985),
               ncol=2 if len(states) == 3 else 3, frameon=data["style"]["legend_frame"], columnspacing=1.2)
    fig.text(.53, .035, "Legacy preview: three label outcomes; format validity unavailable" if _preview(data)
             else "Five mutually exclusive outcomes from saved parse and execution states", ha="center")
    return fig, {"figure": "error_decomposition", "mode": data["mode"], "states": states,
                 "runs": [{"id": r["id"], "label": r["label"], **r["outcomes"]} for r in runs],
                 "excluded_models": [r["id"] for r in data["runs"] if r["outcomes"] is None],
                 "denominator": "all records of each complete run, not only executed Turn 2 records"}


def build_metadata_performance(data):
    from matplotlib.ticker import PercentFormatter
    runs = [r for r in data["runs"] if r["metadata"] is not None]
    if not runs:
        raise ValueError("F requires predictions aligned by ID to the audited metadata")
    if len(runs) > 9:
        raise ValueError("F supports at most nine models per figure; select a readable subset in results config")
    fig = _figure(data, "metadata_height_mm")
    # Seven to nine models need three legend rows above the panel titles.
    panel_top = .72 if len(runs) > 6 else .79
    grid = fig.add_gridspec(1, 2, left=.105, right=.98, bottom=.25, top=panel_top, wspace=.36)
    model_styles = _metadata_styles(runs)
    ranges = {group: model_performance_range(runs, group) for group in ("active", "ship_noise")}
    recalls = [value for run in runs for group in ("active", "ship_noise")
               for value in run["metadata"]["groups"][group]["macro_recall"] if value is not None]
    # Both panels use the same zero-based scale, with room above the highest point.
    maximum = max(recalls, default=0)
    tick_step = .05 if maximum <= .175 else .1 if maximum <= .575 else .25
    upper = min(1.1, max(tick_step * 2, tick_step * np.ceil((maximum + .025) / tick_step)))
    y_ticks = np.arange(0, min(upper, 1.) + tick_step / 2, tick_step)
    handles = []
    range_labels = []
    for col, (group, title) in enumerate((("active", r"(a) Active: $G_h$"),
                                         ("ship_noise", r"(b) Ship noise: $S_{\mathrm{src}}$"))):
        ax = fig.add_subplot(grid[0, col])
        bounds = ranges[group]
        lower = np.array([np.nan if v is None else v for v in bounds["lower"]])
        higher = np.array([np.nan if v is None else v for v in bounds["upper"]])
        valid = np.isfinite(lower) & np.isfinite(higher)
        if len(runs) > 1:
            ax.fill_between(range(3), lower, higher, where=valid, color="#8096A7",
                            alpha=.16, linewidth=0, zorder=1, interpolate=False)
            for tier, gap in enumerate(bounds["gap_pp"]):
                if gap is not None:
                    label = ax.text(tier, higher[tier] + .02 * upper, f"Δ {gap:.1f} pp",
                                    ha=("left", "center", "right")[tier], va="bottom",
                                    color="#445564", zorder=4)
                    range_labels.append((ax, label, higher, tier))
        for run in runs:
            values = [float('nan') if v is None else v for v in run["metadata"]["groups"][group]["macro_recall"]]
            line, = ax.plot(range(3), values, **model_styles[run["id"]], markersize=3.5,
                            linewidth=1.0, label=run["label"], zorder=3)
            if col == 0: handles.append(line)
        ax.set_xticks(range(3), data["strata"]["names"])
        ax.set_xlim(-.18, 2.18)
        ax.set_ylim(0, upper)
        ax.set_yticks(y_ticks)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_title(title, pad=5)
        ax.set_xlabel("Within-class stratum")
        ax.set_ylabel("Class-balanced recall (%)")
        _axis(ax)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D9E0E5", linewidth=.5, linestyle=":")
    # Clear the whole text width, including a sloping segment beside its anchor.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, label, higher, tier in range_labels:
        extent = label.get_window_extent(renderer).transformed(ax.transData.inverted())
        heights = [higher[tier]]
        for left in range(2):
            if not np.isfinite(higher[left:left+2]).all():
                continue
            start, end = max(extent.x0, left), min(extent.x1, left+1)
            if start <= end:
                heights.extend(np.interp([start, end], [left, left+1], higher[left:left+2]))
        label.set_y(max(heights) + .02 * upper)
    fig.legend(handles, [r["label"] for r in runs], loc="upper center", bbox_to_anchor=(.52, .995),
               ncol=3, frameon=data["style"]["legend_frame"], columnspacing=1.2, handlelength=2.5)
    fig.text(.53, .075, "Shading: model min–max; Δ: gap (pp); equal class weights (8 active / 5 ship)", ha="center")
    fig.text(.53, .018, _run_source_note(data), ha="center")
    return fig, {"figure": "metadata_performance", "mode": data["mode"],
                 "model_styles": model_styles,
                 "between_model_range": {"kind": "pointwise_min_max_across_displayed_models",
                                         "is_confidence_interval": False, "groups": ranges,
                                         "missing_policy": "no bounds if any displayed model is missing in that stratum",
                                         "interpretation": "descriptive range of the selected model runs; not theoretical bounds",
                                         "connections": "straight segments join ordered stratum summaries; no continuous-condition estimate"},
                 "display_ylim": [0, float(upper)], "shared_zero_based_scale": True,
                 "runs": [{"id": r["id"], "label": r["label"], **r["metadata"]} for r in runs],
                 "strata": {k: v for k, v in data["strata"].items() if k != "assignments"},
                 "raw_data": {"strata_assignments": data["strata"]["assignments"]},
                 "excluded_models": [r["id"] for r in data["runs"] if r["metadata"] is None],
                 "empty_stratum": "missing point if any required class is empty; no silent class reweighting"}


def build_silence_control(data):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    comparisons = data["comparisons"]
    if not comparisons:
        raise ValueError("G is implemented but needs verified comparison.json outputs; no silent controls in legacy ZIP")
    fig = _figure(data, "silence_height_mm")
    # Dedicated narrow columns leave room for each panel's signed difference.
    # The leaf scores occupy only a small part of [0, 1], so use an independent
    # leaf axis while retaining the full 0–100% routing scale.
    grid = fig.add_gridspec(1, 4, width_ratios=[1.25, .30, 1.0, .34],
                            left=.18, right=.985, bottom=.19, top=.82, wspace=.13)
    leaf_max = max(c["metrics"]["l3_accuracy"][condition]
                   for c in comparisons for condition in ("original", "silent"))
    leaf_limit = min(1.0, max(.16, float(np.ceil(leaf_max * 1.12 / .05) * .05)))
    axes = []
    for col, (key, title, limit) in enumerate((("l3_accuracy", "(a) Leaf recognition", leaf_limit),
                                              ("l1_accuracy", "(b) Routing", 1.0))):
        ax = fig.add_subplot(grid[0, 2 * col])
        axes.append(ax)
        for row, comparison in enumerate(comparisons):
            values = comparison["metrics"][key]
            a, b = values["original"], values["silent"]
            ax.plot([a, b], [row, row], color="#A5AFB6", linewidth=.75, zorder=2)
            ax.plot(a, row, "o", color="#4C78A8", markersize=4, zorder=3)
            ax.plot(b, row, "s", color="#F28E2B", markersize=4, zorder=3)
            ax.text(1.045, row, f"{values['delta_pp']:+.1f}", va="center",
                    color="#43515B", transform=ax.get_yaxis_transform())
        ax.text(1.045, 1.025, "Δ (pp)", va="bottom", color="#43515B",
                transform=ax.transAxes)
        ax.set_xlim(0, limit)
        ax.set_ylim(len(comparisons)-.5, -.5)
        if col == 0 and limit <= .25:
            ax.set_xticks(np.arange(0, limit + 1e-9, .05))
        else:
            ax.set_xticks([0, limit / 2, limit])
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        if col == 0:
            ax.set_yticks(range(len(comparisons)), [c["label"] for c in comparisons])
        else:
            ax.set_yticks([])
        ax.set_title(title, pad=5)
        ax.set_xlabel("Accuracy (%)")
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#DEE5E9", linewidth=.5, linestyle=":")
        _axis(ax)
    handles = [Line2D([], [], color=color, marker=marker, linestyle="none", markersize=4)
               for color, marker in (("#4C78A8", "o"), ("#F28E2B", "s"))]
    fig.legend(handles, ["Original audio", "Matched silence"], loc="upper center",
               bbox_to_anchor=(.55, .99), ncol=2, frameon=data["style"]["legend_frame"])
    fig.text(.54, .045, "Δ = original − silence; percentage points", ha="center")
    return fig, {"figure": "silence_control", "comparisons": comparisons,
                 "display_xlim": {"l3_accuracy": [0, leaf_limit], "l1_accuracy": [0, 1]},
                 "difference": "100 * (original - silent); negative differences retained",
                 "source": "saved comparison.json; original inputs and paired predictions checked during preparation"}

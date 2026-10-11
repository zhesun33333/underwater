"""Compact bubble confusion matrix and class scorecard for result figure D."""

BUBBLE_MAX_DIAMETER_MM = 5.65
FRAME_WIDTH_PT = .5


def _validated_diagnostics(data):
    import numpy as np

    from .result_analysis import diagnostics

    matches = [run for run in data["runs"] if run["id"] == data["diagnostic_model"]]
    if len(matches) != 1:
        raise ValueError("D requires exactly one configured diagnostic_model run")
    run = matches[0]
    values = run["diagnostics"]
    recomputed = diagnostics(values["counts"])
    counts = np.asarray(recomputed["counts"], dtype=int)
    fraction = np.asarray(values["row_fraction"], dtype=float)
    support = np.asarray(values["support"], dtype=int)
    if (fraction.shape != (13, 15) or support.shape != (13,)
            or values["n"] != int(support.sum())
            or not np.array_equal(counts.sum(axis=1), support)
            or not np.allclose(fraction, counts / support[:, None], rtol=0, atol=1e-12)
            or not np.allclose(fraction.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("The full 13 x 15 confusion matrix failed validation")
    for metric in ("precision", "recall", "f1"):
        if not np.allclose(values[metric], recomputed[metric], rtol=0, atol=1e-12):
            raise ValueError(f"Saved {metric} differs from the raw counts")
    if run.get("outcomes") and run["outcomes"]["cascade_count"] != int(counts[:, 14].sum()):
        raise ValueError("The Cascade column differs from the run outcomes")
    if data["mode"] == "archived_verified":
        audit = run["audit"]["archived_protocol_audit"]
        if (not audit or not audit["non_audio_manifest_fields_equal"]
                or audit["sample_count"] != values["n"]
                or not run["identity"]
                or run["identity"]["manifest_sha256"] != audit["archived_manifest_sha256"]
                or len(run["rows"]) != values["n"]):
            raise ValueError("The diagnostic model's archived manifest or protocol was not verified")
    return run, values, counts, fraction


def build_class_diagnostics(data):
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as path_effects
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.ticker import PercentFormatter

    from .plot_acoustic import FAMILY_COLORS, _axis
    from .project import paper_style
    from .result_analysis import CLASS_ORDER, CLASS_TO_FAMILY, DISPLAY_NAMES

    run, values, counts, full_fraction = _validated_diagnostics(data)
    nonzero_columns = np.flatnonzero(counts.sum(axis=0) > 0)
    # Reorder only the predicted-class axis; keep every other nonzero class in
    # its original relative order, with Cascade moved from last to first.
    kept_columns = (np.r_[14, nonzero_columns[nonzero_columns != 14]]
                    if 14 in nonzero_columns else nonzero_columns)
    omitted_columns = np.flatnonzero(counts.sum(axis=0) == 0)
    if counts[:, kept_columns].sum() != counts.sum():
        raise ValueError("A positive prediction column was omitted")
    prediction_names = list(DISPLAY_NAMES)
    prediction_names[12] = "Underw."
    prediction_names += ["Unresolved", "Cascade"]
    fraction = full_fraction[:, kept_columns]
    row_centers = np.arange(13) + .5
    column_centers = np.arange(len(kept_columns)) + .5
    width_mm = float(data["style"]["width_mm"])
    base_height_mm = width_mm * 2 / 3
    height_mm = float(data["style"].get("diagnostic_height_mm", width_mm * 112 / 180))
    if not np.isfinite(width_mm) or not np.isfinite(height_mm) or min(width_mm, height_mm) <= 0:
        raise ValueError("D width and height must be positive finite millimetres")
    y_scale = base_height_mm / height_mm
    with paper_style(data["style"]):
        fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=300)
        matrix_ax = fig.add_axes([.145, .28 * y_scale, .454, .59 * y_scale])
        score_ax = fig.add_axes([.625, .28 * y_scale, .355, .59 * y_scale])

        palette = [(0., "#E4EFF4"), (.15, "#B9D9E3"), (.4, "#66AABB"),
                   (.7, "#28758B"), (1., "#12384F")]
        cmap = LinearSegmentedColormap.from_list("uab_mist_teal_bubble", palette)
        color_scale = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        # A single nonlinear area map is applied to every nonzero cell.
        # Scatter sizes use physical points, so the bubbles remain circular
        # despite the wider spacing of the retained x categories.
        def area_factor(share):
            return .10 + .90 * np.power(share, .6)

        row_pitch_mm = .59 * base_height_mm / 13
        column_pitch_mm = .454 * width_mm / len(kept_columns)
        edge_stroke_mm = .25 * 25.4 / 72
        max_seen = float(fraction.max())
        max_diameter_mm = min(BUBBLE_MAX_DIAMETER_MM,
                              (min(row_pitch_mm, column_pitch_mm) - edge_stroke_mm - .1)
                              / np.sqrt(area_factor(max_seen)))
        if max_diameter_mm <= 0:
            raise ValueError("D panels have insufficient space for nonoverlapping bubbles")
        max_size_pt2 = (max_diameter_mm * 72 / 25.4) ** 2
        for row in range(13):
            for col in range(len(kept_columns)):
                share = float(fraction[row, col])
                if share <= 0:
                    continue
                bubble = matrix_ax.scatter(col + .5, row + .5,
                                           s=max_size_pt2 * area_factor(share),
                                           facecolors=[color_scale.to_rgba(share)],
                                           edgecolors="#86A9B6", linewidths=.25,
                                           rasterized=False, zorder=2)
                bubble.set_gid(f"confusion-{row}-{kept_columns[col]}")
        for row, col in np.argwhere(fraction >= values["annotation_threshold_fraction"]):
            cell = fraction[row, col]
            rgb = np.asarray(color_scale.to_rgba(cell)[:3])
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            luminance = float(linear @ np.array([.2126, .7152, .0722]))
            dark_text = luminance >= .179
            halo = ([path_effects.withStroke(linewidth=.8, foreground="white")]
                    if dark_text else [])
            matrix_ax.text(col + .5, row + .5, f"{100 * cell:.0f}",
                           ha="center", va="center",
                           color="#163845" if dark_text else "white",
                           zorder=3, path_effects=halo, fontsize=8.5)
        matrix_ax.set_xlim(0, len(kept_columns))
        matrix_ax.set_ylim(13, 0)
        column_names = [prediction_names[index] for index in kept_columns]
        matrix_ax.set_xticks(column_centers, column_names, rotation=90)
        matrix_ax.set_yticks(row_centers, DISPLAY_NAMES)
        matrix_ax.set_xlabel("Predicted class (nonzero columns)", labelpad=3)
        matrix_ax.set_ylabel("True class", labelpad=1)
        _axis(matrix_ax)
        for spine in matrix_ax.spines.values():
            spine.set_linewidth(FRAME_WIDTH_PT)
        matrix_ax.tick_params(axis="y", pad=.8)
        for tick, key in zip(matrix_ax.get_yticklabels(), CLASS_ORDER):
            tick.set_color(FAMILY_COLORS[CLASS_TO_FAMILY[key]])

        score_ax.set_xlim(0, 4)
        score_ax.set_ylim(13, 0)
        score_ax.set_xticks([])
        score_ax.set_yticks([])
        _axis(score_ax)
        for spine in score_ax.spines.values():
            spine.set_linewidth(FRAME_WIDTH_PT)
        score_ax.tick_params(bottom=False, top=False, left=False, right=False)
        for boundary in (1.15, 2.10, 3.05):
            score_ax.axvline(boundary, color="#DFE7E9", linewidth=.5, zorder=0)
        for boundary in range(1, 13):
            score_ax.axhline(boundary, color="#ECF1F3", linewidth=.5, zorder=0)
        for boundary in (3, 8):
            score_ax.axhline(boundary, color="#D4E0E5", linewidth=.5, zorder=0)
        score_colors = {"precision": "#4C78A8", "recall": "#C87938", "f1": "#238B78"}
        headers = (("precision", "P", 1.15), ("recall", "R", 2.10),
                   ("f1", "F1", 3.05))
        score_ax.text(.575, -.035, "Class", transform=score_ax.get_xaxis_transform(),
                      ha="center", va="top", weight="bold")
        for key, label, start in headers:
            score_ax.text(start + .475, -.035, label,
                          transform=score_ax.get_xaxis_transform(),
                          ha="center", va="top", color=score_colors[key], weight="bold")

        def raw_percent(value):
            percent = float(value) * 100
            return f"{percent:.0f}" if np.isclose(percent, round(percent)) else f"{percent:.1f}"

        for row, (key, name) in enumerate(zip(CLASS_ORDER, DISPLAY_NAMES)):
            short_name = "Underw." if name == "Underwater" else name
            score_ax.text(1.09, row_centers[row], short_name,
                          ha="right", va="center", color=FAMILY_COLORS[CLASS_TO_FAMILY[key]])
            for metric, _, start in headers:
                score = float(values[metric][row])
                if score == 0:
                    size, color, face, text_color = 14, "#B6C1C6", "none", "#839096"
                else:
                    size, color, face, text_color = (24 + 120 * score,
                                                     score_colors[metric],
                                                     score_colors[metric], "#273D48")
                score_ax.scatter(start + .20, row_centers[row], s=size,
                                 marker="o", zorder=2,
                                 facecolors=face, edgecolors=color, linewidths=.75)
                score_ax.text(start + .43, row_centers[row], raw_percent(score),
                              ha="left", va="center", color=text_color, fontsize=8.8)

        fig.text(.372, .900 * y_scale, "(a) Confusion", ha="center", va="center")
        fig.text(.802, .900 * y_scale, "(b) Class metrics", ha="center", va="center")
        cax = fig.add_axes([.17, .12 * y_scale, .40, .018 * y_scale])
        bar = fig.colorbar(color_scale, cax=cax, orientation="horizontal")
        bar.solids.set_rasterized(False)
        bar.set_ticks([0, .2, .5, 1])
        bar.ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        bar.set_label("True-class share (%)", labelpad=2)
        _axis(cax)
        bar.outline.set_visible(False)
        for spine in cax.spines.values():
            spine.set_linewidth(FRAME_WIDTH_PT)
        omitted_names = [prediction_names[index] for index in omitted_columns]
        omitted_first = ", ".join(omitted_names[:4]) + ("," if len(omitted_names) > 4 else "")
        omitted_rest = ", ".join(omitted_names[4:])
        fig.text(.802, .206 * y_scale, "Zero-count prediction columns omitted:" if omitted_names
                 else "All prediction columns observed",
                 ha="center", va="center")
        fig.text(.802, .179 * y_scale, omitted_first,
                 ha="center", va="center")
        fig.text(.802, .152 * y_scale, omitted_rest,
                 ha="center", va="center")
        fig.text(.802, .116 * y_scale, "Bubble size (nonlinear); 0% = blank",
                 ha="center", va="center")
        size_ax = fig.add_axes([.669, .024 * y_scale, .267, .079 * y_scale])
        size_ax.set_xlim(-.5, 2.5)
        size_ax.set_ylim(0, 1)
        size_ax.set_axis_off()
        legend_shares = np.array([.2, .5, .8])
        legend_sizes = max_size_pt2 * area_factor(legend_shares)
        size_ax.scatter([0, 1, 2], [.55] * 3, s=legend_sizes,
                        c=legend_shares, cmap=cmap, vmin=0, vmax=1,
                        edgecolors="#86A9B6", linewidths=.25, zorder=2)
        for index, share in enumerate(legend_shares):
            size_ax.text(index, .06, f"{share:.0%}", ha="center", va="center")
    return fig, {"figure": "class_diagnostics", "model": run["label"], "mode": data["mode"],
                 "audit": run["audit"], "diagnostics": values,
                 "normalization": "all true-class samples, including unresolved and cascaded predictions",
                 "displayed_prediction_columns": [int(index) for index in kept_columns],
                 "omitted_zero_count_prediction_columns": [int(index) for index in omitted_columns],
                 "confusion_encoding": {"glyph": "bubble", "palette": palette,
                                        "color_scale": "linear row fraction 0 to 1",
                                        "area_factor": "0.10 + 0.90 * row_fraction ** 0.6 for positive cells",
                                        "zero": "blank", "max_diameter_mm": max_diameter_mm,
                                        "annotation_threshold_fraction": values["annotation_threshold_fraction"]},
                 "scorecard": {"metrics": ["precision", "recall", "f1"],
                               "zero_predicted_precision": values["zero_predicted_precision"]},
                 "frame_width_pt": FRAME_WIDTH_PT, "family_colors": FAMILY_COLORS}

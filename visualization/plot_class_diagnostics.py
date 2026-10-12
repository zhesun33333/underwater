"""Rounded-square confusion matrix and class metric scatter for result figure D."""

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
    import numpy as np
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import FormatStrFormatter, PercentFormatter

    from .plot_acoustic import FAMILY_COLORS, _axis
    from .project import paper_style
    from .result_analysis import CLASS_ORDER, CLASS_TO_FAMILY, DISPLAY_NAMES

    run, values, counts, full_fraction = _validated_diagnostics(data)
    # Preserve all 15 prediction categories from the reference layout, while
    # moving only Cascade to the first column. Entirely zero columns stay visible.
    prediction_order = np.r_[14, np.arange(14)]
    fraction = full_fraction[:, prediction_order]
    prediction_names = ["Cascade", *DISPLAY_NAMES[:12], "Underw.", "Unres."]
    row_centers = np.arange(13)
    width_mm = float(data["style"]["width_mm"])
    base_height_mm = width_mm * 2 / 3
    height_mm = float(data["style"].get("diagnostic_height_mm", width_mm * 112 / 180))
    if not np.isfinite(width_mm) or not np.isfinite(height_mm) or min(width_mm, height_mm) <= 0:
        raise ValueError("D width and height must be positive finite millimetres")
    y_scale = base_height_mm / height_mm
    with paper_style(data["style"]):
        fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=300)
        grid = fig.add_gridspec(1, 2, width_ratios=[2.1, 1], left=.12, right=.985,
                                bottom=.28 * y_scale, top=.87 * y_scale, wspace=.52)
        matrix_ax, score_ax = (fig.add_subplot(grid[0, i]) for i in range(2))

        palette = [(0., "#E4EFF4"), (.15, "#B9D9E3"), (.4, "#66AABB"),
                   (.7, "#28758B"), (1., "#12384F")]
        cmap = LinearSegmentedColormap.from_list("uab_mist_teal_tiles", palette)
        color_scale = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        # A square remains square on paper even if the panel aspect changes.
        box = matrix_ax.get_position()
        cell_aspect = ((box.width * fig.get_figwidth() / 15)
                       / (box.height * fig.get_figheight() / 13))
        zero_rows, zero_cols = np.where(fraction == 0)
        matrix_ax.scatter(zero_cols, zero_rows, s=2, c="#CFD8DE", linewidths=0, zorder=1)
        for row, col in np.argwhere(fraction > 0):
            share = float(fraction[row, col])
            side = .45 + .50 * np.sqrt(share)
            height = side * cell_aspect
            tile = FancyBboxPatch((col - side / 2, row - height / 2), side, height,
                                  boxstyle=f"round,pad=0,rounding_size={side * .19}",
                                  mutation_aspect=cell_aspect, facecolor=color_scale.to_rgba(share),
                                  edgecolor="none", linewidth=0, zorder=2, rasterized=False)
            tile.set_gid(f"confusion-{row}-{prediction_order[col]}")
            matrix_ax.add_patch(tile)
        for row, col in np.argwhere(fraction >= values["annotation_threshold_fraction"]):
            share = float(fraction[row, col])
            rgb = np.asarray(color_scale.to_rgba(share)[:3])
            linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
            luminance = float(linear @ np.array([.2126, .7152, .0722]))
            matrix_ax.text(col, row, f"{100 * share:.0f}", ha="center", va="center",
                           color="white" if luminance < .179 else "black", zorder=3)
        matrix_ax.set_xlim(-.5, 14.5)
        matrix_ax.set_ylim(12.5, -.5)
        matrix_ax.set_xticks(range(15), prediction_names, rotation=90)
        matrix_ax.set_yticks(row_centers, DISPLAY_NAMES)
        matrix_ax.set_xlabel("Predicted class", labelpad=3)
        matrix_ax.set_ylabel("True class", labelpad=1)
        matrix_ax.set_title("(a) Confusion", pad=4)
        _axis(matrix_ax)
        for spine in matrix_ax.spines.values():
            spine.set_linewidth(FRAME_WIDTH_PT)
        matrix_ax.tick_params(axis="y", pad=.8)
        for tick, key in zip(matrix_ax.get_yticklabels(), CLASS_ORDER):
            tick.set_color(FAMILY_COLORS[CLASS_TO_FAMILY[key]])
        for boundary in (2.5, 7.5):
            matrix_ax.axhline(boundary, color="#DCE5E9", linewidth=.5, zorder=0)
        for boundary in (.5, 3.5, 8.5, 13.5):
            matrix_ax.axvline(boundary, color="#DCE5E9", linewidth=.5, zorder=0)

        for offset, key, marker, label in ((-.21, "precision", "o", "P"),
                                          (0, "recall", "^", "R"),
                                          (.21, "f1", "s", "F1")):
            score_ax.plot(values[key], row_centers + offset, linestyle="none", marker=marker,
                          markersize=3.5, color="black",
                          markerfacecolor="white" if key == "recall" else "black",
                          markeredgewidth=.75, label=label)
        max_score = max(float(np.max(values[key])) for key in ("precision", "recall", "f1"))
        score_step = .1 if max_score <= .5 else .2
        score_upper = min(1., max(.4, score_step * np.ceil(max_score / score_step)))
        score_ticks = np.arange(0, score_upper + 1e-9, score_step)
        score_ax.set_xlim(-.015, score_upper + .015)
        score_ax.set_ylim(12.5, -.5)
        score_ax.set_xticks(score_ticks)
        score_ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        score_ax.set_yticks(row_centers, DISPLAY_NAMES)
        score_ax.set_xlabel("Score", labelpad=3)
        score_ax.set_ylabel("Class", labelpad=1)
        score_ax.set_title("(b) Class metrics", pad=4)
        _axis(score_ax)
        for spine in score_ax.spines.values():
            spine.set_linewidth(FRAME_WIDTH_PT)
        score_ax.tick_params(axis="y", pad=.8)
        for tick, key in zip(score_ax.get_yticklabels(), CLASS_ORDER):
            tick.set_color(FAMILY_COLORS[CLASS_TO_FAMILY[key]])
        for boundary in (2.5, 7.5):
            score_ax.axhline(boundary, color="#DCE5E9", linewidth=.5, zorder=0)
        score_ax.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=3,
                        frameon=data["style"]["legend_frame"], handletextpad=.3,
                        columnspacing=.8, borderaxespad=0)
        matrix_box = matrix_ax.get_position()
        cax = fig.add_axes([matrix_box.x0, .125 * y_scale,
                            matrix_box.width, .018 * y_scale])
        bar = fig.colorbar(color_scale, cax=cax, orientation="horizontal")
        bar.solids.set_rasterized(False)
        bar.set_ticks([0, .5, 1])
        bar.ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        bar.set_label("True-class share (%)", labelpad=2)
        _axis(cax)
        for spine in cax.spines.values():
            spine.set_linewidth(FRAME_WIDTH_PT)
        source_note = ("Archived predictions checked against their saved manifest and protocol"
                       if data["mode"] == "archived_verified" else
                       "Legacy preview: stored labels; protocol verification unavailable"
                       if data["mode"] == "legacy_preview" else
                       "Predictions verified against the current manifest and run protocol")
        fig.text(.53, .018 * y_scale, source_note, ha="center", va="bottom")
    return fig, {"figure": "class_diagnostics", "model": run["label"], "mode": data["mode"],
                 "audit": run["audit"], "diagnostics": values,
                 "normalization": "all true-class samples, including unresolved and cascaded predictions",
                 "displayed_prediction_columns": [int(index) for index in prediction_order],
                 "zero_count_prediction_columns": [int(index) for index in np.flatnonzero(counts.sum(axis=0) == 0)],
                 "omitted_zero_count_prediction_columns": [],
                 "confusion_encoding": {"glyph": "rounded square", "palette": palette,
                                        "color_scale": "linear row fraction 0 to 1",
                                        "positive_side_in_cell_widths": "0.45 + 0.50 * sqrt(row_fraction)",
                                        "size_meaning": "monotonic cue; area is not proportional to probability",
                                        "zero": "small neutral dot",
                                        "annotation_threshold_fraction": values["annotation_threshold_fraction"]},
                 "score_axis": {"scale": "linear", "limits": [-.015, score_upper + .015],
                                "ticks_raw_fraction": score_ticks.tolist(),
                                "zero_predicted_precision": values["zero_predicted_precision"]},
                 "colorbar_matches_matrix_width": True,
                 "frame_width_pt": FRAME_WIDTH_PT, "family_colors": FAMILY_COLORS}

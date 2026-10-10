"""Standalone figure-A builders. Callers own styling, saving, and provenance."""
from __future__ import annotations

import math

import numpy as np

from visualization.acoustic_analysis import (
    COMMUNICATION_CLASSES, PULSE_CLASSES, SHIP_CLASSES,
    communication_window, compute_spectrogram, compute_welch, frequency_band,
    load_audio, power_db, select_representatives, stft_parameters, welch_parameters,
    summarize_ship_centroids,
)


FAMILY_COLORS = {"pulse": "#4C78A8", "communication": "#F28E2B", "ship_noise": "#2A9D8F"}
SHIP_NAMES = {"cargo": "Cargo vessel", "cruise": "Cruise ship", "fishing": "Fishing vessel",
              "warship": "Naval vessel", "underwater_target": "Underwater vehicle"}


def _axis(ax):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.75)
    ax.tick_params(which="major", direction="in", top=True, right=True, width=0.75, length=3)
    ax.minorticks_off()
    ax.grid(False)


def _complete_selection(records, classes):
    filtered = [record for record in records if record.get("l3") in classes]
    selected, proof = select_representatives(filtered)
    if set(selected) != set(classes):
        raise ValueError(f"Missing classes for acoustic figure: {sorted(set(classes) - set(selected))}")
    return selected, proof


def _image_extent(frequencies, times, fft_step, time_step):
    return [float(times[0] - time_step / 2), float(times[-1] + time_step / 2),
            float(frequencies[0] - fft_step / 2), float(frequencies[-1] + fft_step / 2)]


def build_active_figure(eval_records: list[dict], style: dict) -> tuple:
    """A1: full pulses and centered symbol-based communication power views."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    from cmcrameri import cm

    classes = PULSE_CLASSES + COMMUNICATION_CLASSES
    selected, selection_proof = _complete_selection(eval_records, classes)
    panels, spectra = {}, {}
    references = {"pulse": 0.0, "communication": 0.0}
    for leaf in classes:
        record = selected[leaf]
        samples, audio_proof = load_audio(record)
        crop = (communication_window(record) if leaf in COMMUNICATION_CLASSES else {
            "start_sample": 0, "stop_sample_exclusive": len(samples),
            "start_s": 0.0, "stop_s": len(samples) / record["fs"],
            "used_entire_waveform": True, "alignment": "full received WAV; propagation already arrival-aligned",
        })
        parameters = stft_parameters(record, crop)
        frequencies, times, powers = compute_spectrogram(
            samples[crop["start_sample"]:crop["stop_sample_exclusive"]], parameters, crop["start_s"])
        group = record["l2"]
        references[group] = max(references[group], float(np.max(powers)))
        band = frequency_band(record, parameters["fft_bin_spacing_hz"])
        low, high = band["display_limits_hz"]
        keep = (frequencies >= low - parameters["fft_bin_spacing_hz"]) & (
            frequencies <= high + parameters["fft_bin_spacing_hz"])
        spectra[leaf] = (frequencies[keep], times, powers[keep])
        panels[leaf] = {**audio_proof, "family": group, "time_window": crop,
                        "stft": parameters, "frequency_band": band}

    floor = -50.0
    for group, reference in references.items():
        if reference <= 0:
            raise ValueError(f"No nonzero acoustic power in selected {group} examples")
    width_mm = style.get("width_mm", 180)
    width = width_mm / 25.4
    height = style.get("active_height_mm", width_mm * 11 / 18) / 25.4
    fig = plt.figure(figsize=(width, height), dpi=max(300, style.get("dpi", 300)))
    grid = fig.add_gridspec(2, 4, left=0.09, right=0.985,
                           bottom=0.25, top=0.94, wspace=0.67, hspace=0.48)
    # Full-waveform views first; the four single-carrier symbol windows form
    # their own row. Their analysis and selected records remain unchanged.
    display_order = PULSE_CLASSES + ("OFDM",) + COMMUNICATION_CLASSES[:-1]
    norm = Normalize(vmin=floor, vmax=0)
    for index, leaf in enumerate(display_order):
        panel = panels[leaf]
        frequencies, times, powers = spectra[leaf]
        db = power_db(powers, references[panel["family"]], floor)
        parameters, crop = panel["stft"], panel["time_window"]
        extent = _image_extent(frequencies, times, parameters["fft_bin_spacing_hz"], parameters["hop_duration_s"])
        local_time = leaf in COMMUNICATION_CLASSES and leaf != "OFDM"
        time_origin = crop["start_s"] if local_time else 0.0
        time_scale = 1000.0 if local_time else 1.0
        display_extent = [(extent[0] - time_origin) * time_scale,
                          (extent[1] - time_origin) * time_scale, *extent[2:]]
        time_limits = [(crop["start_s"] - time_origin) * time_scale,
                       (crop["stop_s"] - time_origin) * time_scale]
        ax = fig.add_subplot(grid[index // 4, index % 4])
        ax.imshow(db, origin="lower", aspect="auto", interpolation="nearest", extent=display_extent,
                  cmap=cm.batlow, norm=norm)
        ax.set_xlim(time_limits)
        ax.set_ylim(panel["frequency_band"]["display_limits_hz"])
        ax.set_title(f"({chr(97 + index)}) {leaf}", color=FAMILY_COLORS[panel["family"]], pad=5)
        ax.set_xlabel("Time (ms)" if local_time else "Time (s)", labelpad=2)
        ax.set_ylabel("Frequency (Hz)", labelpad=2)
        span = time_limits[1] - time_limits[0]
        decimals = min(5, max(1, int(math.ceil(-math.log10(span / 4)))))
        ax.set_xticks(np.linspace(*time_limits, 3))
        ax.xaxis.set_major_formatter(FormatStrFormatter(f"%.{decimals}f"))
        # Endpoint labels stay inside each narrow panel's horizontal bounds.
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        _axis(ax)
        panel.update({"reference_value": references[panel["family"]], "reference_units": "digital amplitude squared",
                      "frequency_hz": frequencies.tolist(), "time_s": times.tolist(),
                      "power_db": db.tolist(), "image_extent": extent,
                      "display_time_unit": "ms" if local_time else "s",
                      "display_time_origin_s": time_origin, "display_image_extent": display_extent})
    for col, (family, label) in enumerate((("pulse", "Pulse"), ("communication", "Communication"))):
        cax = fig.add_axes([0.12 + col * 0.48, 0.12, 0.32, 0.019])
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cm.batlow), cax=cax, orientation="horizontal")
        bar.set_ticks([-50, -25, 0])
        bar.set_label(f"{label} power (dB)\n0 dB = family maximum", labelpad=3)
        bar.outline.set_linewidth(0.75)
        for spine in cax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.75)
        cax.tick_params(direction="in", width=0.75, length=3)
        cax.minorticks_off()
    payload = {
        "figure": "acoustic_structure_active", "layout": "2 rows x 4 independent axes; full-waveform row and single-carrier row",
        "display_order": list(display_order),
        "display_time_convention": "single-carrier panels: milliseconds from crop start; pulse and OFDM panels: original WAV seconds",
        "selection": selection_proof, "panels": panels,
        "color_map": "cmcrameri.batlow", "display_range_db": [floor, 0.0],
        "power_definition": "one-sided STFT-bin power, squared spectrum-scaled transform; interior bins doubled",
        "group_reference_values": references, "group_reference_units": "digital amplitude squared",
        "reference_domain": "maximum over all frequency bins and displayed time windows in each selected family",
        "comparison": "0 dB is separately defined for Pulse and Communication; no shared physical level across groups",
        "communication_window_rule": "24 nominal symbols centered in the actual received WAV, or the entire WAV when shorter",
        "analysis_rules": ["Pulses: common 16-ms window and 2-ms hop",
                           "Single-carrier communication: common two-symbol window and one-eighth-symbol hop rule; sample lengths differ",
                           "OFDM: 32-ms window and 4-ms hop, not intended to resolve individual subcarriers"],
        "interpretation": "Power views show waveform structure; PSK phase-state counts are not inferred",
        "frequency_axes": "linear Hz with source-defined bands; panel slopes are not directly comparable",
        "additional_gain_normalization": False, "image_artist": "imshow", "raster_dpi_minimum": 300,
    }
    return fig, payload


def build_ship_figure(eval_records: list[dict], style: dict) -> tuple:
    """A2: five representative Welch PSDs and Eval-wide centroid summaries."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    selected, selection_proof = _complete_selection(eval_records, SHIP_CLASSES)
    loaded = {leaf: load_audio(selected[leaf]) for leaf in SHIP_CLASSES}
    rates = {proof["fs"] for _, proof in loaded.values()}
    if len(rates) != 1:
        raise ValueError("Ship PSD panels require a common sample rate; no implicit resampling")
    rate = rates.pop()
    parameters = welch_parameters(rate, min(len(audio) for audio, _ in loaded.values()))
    frequency_limit = min(1000.0, rate / 2)
    spectra = {}
    for leaf, (audio, proof) in loaded.items():
        frequencies, density = compute_welch(audio, parameters)
        keep = frequencies <= frequency_limit
        frequencies, density = frequencies[keep], density[keep]
        db = power_db(density, 1.0, -180.0)
        spectra[leaf] = {**proof, "frequency_hz": frequencies.tolist(),
                         "psd_amplitude_squared_per_hz": density.tolist(), "psd_db": db.tolist(),
                         "time_window": {"start_sample": 0, "stop_sample_exclusive": len(audio),
                                         "start_s": 0.0, "stop_s": len(audio) / rate},
                         "welch_segments": 1 + (len(audio) - parameters["nperseg"]) // (
                             parameters["nperseg"] - parameters["noverlap"])}
    centroid_summary, centroid_rows = summarize_ship_centroids(eval_records, parameters, frequency_limit)
    # Full shared dynamic range, never a per-panel amplitude normalization.
    minimum = min(min(panel["psd_db"]) for panel in spectra.values())
    maximum = max(max(panel["psd_db"]) for panel in spectra.values())
    limits = [5 * math.floor(minimum / 5) - 2, 5 * math.ceil(maximum / 5) + 2]
    width_mm = style.get("width_mm", 180)
    width = width_mm / 25.4
    height = style.get("ship_height_mm", width_mm / 2) / 25.4
    fig = plt.figure(figsize=(width, height), dpi=max(300, style.get("dpi", 300)))
    grid = fig.add_gridspec(2, 3, left=0.08, right=0.985, bottom=0.185, top=0.92,
                           wspace=0.75, hspace=0.68)
    positions = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1))
    for index, leaf in enumerate(SHIP_CLASSES):
        row, col = positions[index]
        ax, panel = fig.add_subplot(grid[row, col]), spectra[leaf]
        ax.plot(panel["frequency_hz"], panel["psd_db"], color=FAMILY_COLORS["ship_noise"], linewidth=1.0)
        ax.set_xlim(0, frequency_limit)
        ax.set_ylim(limits)
        ax.set_xticks([0, frequency_limit / 2, frequency_limit])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.set_title(f"({chr(97 + index)}) {SHIP_NAMES[leaf]}", pad=5)
        ax.set_xlabel("Frequency (Hz)", labelpad=2)
        ax.set_ylabel("PSD (dB)", labelpad=2)
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
        _axis(ax)
    ax = fig.add_subplot(grid[1, 2])
    short_names = ("Cargo", "Cruise", "Fishing", "Naval", "Underwater")
    for index, leaf in enumerate(SHIP_CLASSES):
        summary = centroid_summary["classes"][leaf]
        median = summary["median_hz"]
        ax.errorbar(median, index,
                    xerr=[[median - summary["q25_hz"]], [summary["q75_hz"] - median]],
                    fmt="o", markersize=3, color=FAMILY_COLORS["ship_noise"],
                    elinewidth=1.0, capsize=2, capthick=0.75)
    ax.set_yticks(range(len(SHIP_CLASSES)), short_names)
    ax.set_ylim(len(SHIP_CLASSES) - 0.5, -0.5)
    low = min(item["q25_hz"] for item in centroid_summary["classes"].values())
    high = max(item["q75_hz"] for item in centroid_summary["classes"].values())
    padding = max(high - low, frequency_limit * 0.05) * 0.15
    step = min(50.0, frequency_limit / 10)
    centroid_limits = [max(0.0, step * math.floor((low - padding) / step)),
                       min(frequency_limit, step * math.ceil((high + padding) / step))]
    ax.set_xlim(centroid_limits)
    ax.set_xticks(np.linspace(*centroid_limits, 4))
    centroid_summary["display_limits_hz"] = centroid_limits
    ax.set_title("(f) Centroid: median and IQR", pad=5)
    ax.set_xlabel("Spectral centroid (Hz)", labelpad=2)
    ax.set_ylabel("Class", labelpad=2)
    ax.get_xticklabels()[0].set_ha("left")
    ax.get_xticklabels()[-1].set_ha("right")
    _axis(ax)
    counts = {item["n"] for item in centroid_summary["classes"].values()}
    count_label = (f"n = {counts.pop()} per class" if len(counts) == 1
                   else f"n = {centroid_summary['record_count']} total")
    fig.text(0.52, 0.028,
             "(a–e) PSD reference: 1 digital amplitude²/Hz; "
             f"(f) Eval, {count_label}", ha="center", va="bottom")
    payload = {
        "figure": "acoustic_structure_ship", "layout": "2 rows x 3 independent axes; five representative PSDs and one population summary",
        "selection": selection_proof, "spectra": spectra, "welch": parameters,
        "centroid_summary": centroid_summary, "raw_data": {"ship_centroids": centroid_rows},
        "frequency_limits_hz": [0.0, frequency_limit], "psd_axis_limits_db": limits,
        "psd_db_floor": -180.0, "db_reference_value": 1.0,
        "db_reference_units": "digital amplitude squared / Hz",
        "level_definition": "10 log10(PSD / (1 digital amplitude squared / Hz)); not physical SPL",
        "additional_gain_normalization": False,
        "interpretation": "Panels (a-e) describe representative received samples, not class invariants; panel (f) summarizes all Eval ship records",
    }
    return fig, payload

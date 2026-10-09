"""Deterministic selection and calibrated digital-audio analysis for figure A.

The input records are the audited, canonical records produced by figure_data.
No file is opened and no figure is created at module import time.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


PULSE_CLASSES = ("CW", "LFM", "HFM")
COMMUNICATION_CLASSES = ("2FSK", "4FSK", "BPSK", "QPSK", "OFDM")
SHIP_CLASSES = ("cargo", "cruise", "fishing", "warship", "underwater_target")
CLASS_ORDER = PULSE_CLASSES + COMMUNICATION_CLASSES + SHIP_CLASSES


def _finite(value, description: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, str)) or value is None:
        raise ValueError(f"Missing or nonnumeric {description}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Missing or nonnumeric {description}") from error
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"Invalid {description}: {value!r}")
    return number


def select_representatives(records: list[dict]) -> tuple[dict, dict]:
    """Choose the record closest to each class's median/IQR metadata vector.

    Missing required features fail explicitly. Zero-IQR features contribute no
    distance, even if they have nonzero outliers. Exact distance ties use the
    lexicographically smallest ID. The function never reads model predictions.
    It accepts any nonempty subset of the 13 classes; builders check coverage.
    """
    if not records:
        raise ValueError("Cannot select representatives from an empty dataset")
    grouped: dict[str, list[dict]] = {}
    seen = set()
    for record in records:
        sid, leaf = record.get("id"), record.get("l3")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError(f"Missing or duplicate record ID: {sid!r}")
        seen.add(sid)
        if leaf not in CLASS_ORDER:
            raise ValueError(f"{sid}: unsupported leaf class {leaf!r}")
        expected = ("active", "pulse") if leaf in PULSE_CLASSES else (
            ("active", "communication") if leaf in COMMUNICATION_CLASSES
            else ("passive", "ship_noise"))
        if (record.get("l1"), record.get("l2")) != expected:
            raise ValueError(f"{sid}: inconsistent label hierarchy")
        grouped.setdefault(leaf, []).append(record)

    chosen, classes = {}, {}
    for leaf in CLASS_ORDER:
        if leaf not in grouped:
            continue
        pool = sorted(grouped[leaf], key=lambda record: record["id"])
        features = ["duration_s", "S_src"] if leaf in SHIP_CLASSES else [
            "duration_s", "G_h", "signal_frequency_hz"]
        matrix = np.asarray([[
            _finite(record.get(name), f"{record['id']}.{name}",
                    positive=name in ("duration_s", "signal_frequency_hz"))
            for name in features] for record in pool], dtype=float)
        q25, medians, q75 = np.quantile(matrix, [0.25, 0.5, 0.75], axis=0, method="linear")
        scales = q75 - q25
        keep = scales > 0
        z = (matrix[:, keep] - medians[keep]) / scales[keep]
        distances = np.sqrt(np.sum(z * z, axis=1))
        index = min(range(len(pool)), key=lambda i: (float(distances[i]), pool[i]["id"]))
        chosen[leaf] = pool[index]
        classes[leaf] = {
            "count": len(pool), "selected_id": pool[index]["id"],
            "selected_source_id": pool[index].get("source_id"),
            "features": features,
            "median": dict(zip(features, map(float, medians))),
            "iqr": dict(zip(features, map(float, scales))),
            "q25": dict(zip(features, map(float, q25))),
            "q75": dict(zip(features, map(float, q75))),
            "used_features": [name for name, used in zip(features, keep) if used],
            "dropped_zero_iqr": [name for name, used in zip(features, keep) if not used],
            "selected_features": dict(zip(features, map(float, matrix[index]))),
            "selected_distance": float(distances[index]),
            "exact_distance_tie_count": int(np.count_nonzero(distances == distances[index])),
        }
    return chosen, {
        "method": "Euclidean distance to class-wise median after division by class-wise IQR",
        "quantile_method": "linear", "zero_iqr": "omit dimension",
        "missing_required_feature": "reject dataset", "tie_break": "lexicographic record ID",
        "selection_uses_model_predictions": False,
        "duration_source": "actual WAV frames / sample rate, audited during preparation",
        "frequency_source": "source signal parameters, never Bellhop environment frequency",
        "classes": classes,
    }


def load_audio(record: dict) -> tuple[np.ndarray, dict]:
    """Read the actual evaluation WAV without gain adjustment or resampling."""
    import soundfile as sf

    path = Path(record["audio_path"])
    if not path.is_absolute():
        raise ValueError(f"{record['id']}: audio_path must be absolute")
    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    if samples.shape[1] != 1:
        raise ValueError(f"{record['id']}: expected mono evaluation audio; no implicit channel averaging")
    expected_rate = _finite(record.get("fs"), "sample rate", positive=True)
    expected_frames = _finite(record.get("frames"), "frame count", positive=True)
    if rate != expected_rate or len(samples) != expected_frames:
        raise ValueError(f"{record['id']}: WAV header changed since records were prepared")
    duration = len(samples) / rate
    if abs(_finite(record.get("duration_s"), "duration", positive=True) - duration) > 1e-10:
        raise ValueError(f"{record['id']}: duration does not equal actual frames / sample rate")
    audio = samples[:, 0]
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"{record['id']}: audio has non-finite values")
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return audio, {
        "id": record["id"], "source_id": record["source_id"], "l3": record["l3"],
        "audio_path": str(path), "audio_sha256": checksum, "fs": rate,
        "frames": len(audio), "duration_s": duration,
        "amplitude_convention": "soundfile float64 digital samples; PCM full scale maps to [-1, 1)",
        "additional_gain_normalization": False,
        "signal_frequency_hz": record.get("signal_frequency_hz"),
        "channel_frequency_hz": record.get("channel_frequency_hz"),
    }


def communication_window(record: dict, symbols: int = 24) -> dict:
    """Return a centered, half-open sample interval based on symbol duration.

    OFDM uses (1 + cp_factor) * num_subcarriers / bandwidth_hz and checks the
    actual WAV duration against the stored symbol count. Rounded subcarrier
    spacing is not used. No received symbol timing or phase alignment is claimed.
    """
    if record.get("l3") not in COMMUNICATION_CLASSES:
        raise ValueError("Symbol windows apply only to communication records")
    if type(symbols) is not int or symbols <= 0:
        raise ValueError("symbols must be a positive integer")
    rate = _finite(record.get("fs"), "sample rate", positive=True)
    frames = int(_finite(record.get("frames"), "frame count", positive=True))
    parameters = record.get("signal_params") or {}
    duration_check = None
    if record["l3"] == "OFDM":
        count = _finite(parameters.get("num_ofdm_symbols"), "num_ofdm_symbols", positive=True)
        if not count.is_integer():
            raise ValueError("num_ofdm_symbols must be an integer")
        carriers = _finite(parameters.get("num_subcarriers"), "num_subcarriers", positive=True)
        bandwidth = _finite(parameters.get("bandwidth_hz"), "bandwidth_hz", positive=True)
        prefix = _finite(parameters.get("cp_factor"), "cp_factor")
        if not carriers.is_integer() or prefix < 0:
            raise ValueError("OFDM requires integer num_subcarriers and nonnegative cp_factor")
        symbol_duration = (1 + prefix) * carriers / bandwidth
        expected_duration = count * symbol_duration
        residual = frames / rate - expected_duration
        tolerance = 1 / rate
        if abs(residual) > tolerance + 1e-12:
            raise ValueError(f"{record['id']}: OFDM symbol parameters disagree with actual WAV duration")
        symbol_samples = symbol_duration * rate
        basis = "(1 + signal_params.cp_factor) * signal_params.num_subcarriers / signal_params.bandwidth_hz"
        duration_check = {"actual_wav_duration_s": frames / rate,
                          "actual_duration_per_stored_symbol_s": frames / rate / count,
                          "expected_duration_s": expected_duration, "residual_s": residual,
                          "tolerance_s": tolerance, "matches": True}
    else:
        baud = _finite(parameters.get("symbol_rate_baud"), "symbol_rate_baud", positive=True)
        symbol_samples = rate / baud
        if symbol_samples < 1:
            raise ValueError("symbol_rate_baud exceeds the sample rate")
        basis = "sample rate / signal_params.symbol_rate_baud"
    requested = max(1, int(round(symbols * symbol_samples)))
    length = min(frames, requested)
    start = (frames - length) // 2
    stop = start + length
    return {
        "start_sample": start, "stop_sample_exclusive": stop,
        "start_s": start / rate, "stop_s": stop / rate,
        "requested_symbols": symbols, "nominal_symbols_shown": length / symbol_samples,
        "symbol_duration_s": symbol_samples / rate, "symbol_samples": symbol_samples,
        "symbol_duration_basis": basis, "used_entire_waveform": length == frames,
        "ofdm_duration_check": duration_check,
        "alignment": "centered sample window; no recovered symbol or carrier-phase alignment",
    }


def stft_parameters(record: dict, crop: dict) -> dict:
    """Return explicit window rules in samples for the three analysis groups."""
    from scipy.fft import next_fast_len

    rate = int(record["fs"])
    if record["l3"] in PULSE_CLASSES:
        length, hop = round(0.016 * rate), round(0.002 * rate)
        rule = "pulse: 16-ms Hann window, 2-ms hop"
    elif record["l3"] == "OFDM":
        length, hop = round(0.032 * rate), round(0.004 * rate)
        rule = "OFDM: 32-ms Hann window, 4-ms hop; not subcarrier-resolving"
    else:
        length = round(2 * crop["symbol_samples"])
        hop = round(crop["symbol_samples"] / 8)
        rule = "single carrier: two-symbol Hann window, one-eighth-symbol hop"
    available = crop["stop_sample_exclusive"] - crop["start_sample"]
    if available < 8:
        raise ValueError(f"{record['id']}: fewer than eight samples in analysis window")
    length = min(available, max(8, length))
    hop = min(length, max(1, hop))
    nfft = next_fast_len(max(1024, 4 * length))
    return {
        "window": "hann_periodic", "nperseg": length, "hop_samples": hop,
        "noverlap": length - hop, "nfft": nfft, "fs": rate,
        "window_duration_s": length / rate, "hop_duration_s": hop / rate,
        "fft_bin_spacing_hz": rate / nfft,
        "native_frequency_spacing_hz": rate / length,
        "boundary": "zeros", "padded": True, "detrend": False,
        "scaling": "spectrum", "sidedness": "one-sided; interior squared-magnitude bins doubled",
        "units": "digital amplitude squared", "rule": rule,
    }


def compute_spectrogram(audio: np.ndarray, parameters: dict, start_s: float = 0.0) -> tuple:
    """Compute one-sided STFT-bin power; zero padding never changes amplitude."""
    from scipy.signal import get_window, stft

    audio = np.asarray(audio, dtype=float)
    nperseg, nfft = parameters["nperseg"], parameters["nfft"]
    if audio.ndim != 1 or len(audio) < nperseg or not np.all(np.isfinite(audio)):
        raise ValueError("STFT requires a finite 1-D waveform at least one window long")
    frequency, times, transform = stft(
        audio, fs=parameters["fs"], window="hann", nperseg=nperseg,
        noverlap=parameters["noverlap"], nfft=nfft, detrend=False,
        return_onesided=True, boundary="zeros", padded=True, scaling="spectrum")
    power = np.abs(transform) ** 2
    power[1:-1 if nfft % 2 == 0 else None] *= 2
    keep = times <= len(audio) / parameters["fs"] + 1e-12
    times, power = times[keep] + start_s, power[:, keep]
    window = get_window("hann", nperseg, fftbins=True)
    parameters["equivalent_noise_bandwidth_hz"] = float(
        parameters["fs"] * np.sum(window ** 2) / np.sum(window) ** 2)
    return frequency, times, power


def frequency_band(record: dict, fft_bin_spacing_hz: float) -> dict:
    """Crop the frequency axis around source-defined occupied bandwidth."""
    parameters = record.get("signal_params") or {}
    center = _finite(record.get("signal_frequency_hz"), "source frequency", positive=True)
    if parameters.get("band_low_hz") is not None and parameters.get("band_high_hz") is not None:
        low = _finite(parameters["band_low_hz"], "band_low_hz")
        high = _finite(parameters["band_high_hz"], "band_high_hz")
        if low >= high:
            raise ValueError(f"{record['id']}: invalid source frequency band")
        source = "signal_params.band_low_hz / band_high_hz"
    else:
        bandwidth = parameters.get("bandwidth_hz", parameters.get("target_bandwidth_hz"))
        width = (_finite(bandwidth, "bandwidth_hz", positive=True) if bandwidth is not None
                 else max(200.0, 0.1 * center))
        low, high = center - width / 2, center + width / 2
        source = "source carrier plus bandwidth" if bandwidth is not None else "source carrier +/- max(100 Hz, 5% of carrier)"
    padding = max(0.1 * (high - low), 3 * fft_bin_spacing_hz)
    limits = [max(0.0, low - padding), min(record["fs"] / 2, high + padding)]
    if limits[0] >= limits[1]:
        raise ValueError(f"{record['id']}: source frequency band is outside WAV Nyquist range")
    return {"source": source, "source_band_hz": [low, high],
            "padding_hz": padding, "display_limits_hz": limits,
            "clipped_to_nyquist": [low - padding, high + padding] != limits}


def power_db(power: np.ndarray, reference: float, floor_db: float = -60.0) -> np.ndarray:
    """Convert nonnegative digital power using one explicit positive reference."""
    reference = _finite(reference, "power reference", positive=True)
    floor_db = _finite(floor_db, "power floor")
    power = np.asarray(power, dtype=float)
    if np.any(power < 0) or not np.all(np.isfinite(power)):
        raise ValueError("Power values must be finite and nonnegative")
    return 10 * np.log10(np.maximum(power / reference, 10 ** (floor_db / 10)))


def welch_parameters(sample_rate: int, shortest_frames: int) -> dict:
    length = min(int(sample_rate), int(shortest_frames))
    if length < 8 or sample_rate <= 0:
        raise ValueError("Welch estimation requires at least eight samples and a positive sample rate")
    return {
        "window": "hann_periodic", "nperseg": length, "noverlap": length // 2,
        "nfft": length, "fs": int(sample_rate), "scaling": "density",
        "detrend": "constant", "average": "mean", "return_onesided": True,
        "window_duration_s": length / sample_rate,
        "frequency_bin_spacing_hz": sample_rate / length,
        "units": "digital amplitude squared / Hz",
        "db_reference_value": 1.0, "db_reference_units": "digital amplitude squared / Hz",
    }


def compute_welch(audio: np.ndarray, parameters: dict) -> tuple[np.ndarray, np.ndarray]:
    from scipy.signal import welch

    audio = np.asarray(audio, dtype=float)
    if audio.ndim != 1 or len(audio) < parameters["nperseg"] or not np.all(np.isfinite(audio)):
        raise ValueError("Welch requires a finite 1-D waveform at least one window long")
    return welch(audio, fs=parameters["fs"], window="hann",
                 nperseg=parameters["nperseg"], noverlap=parameters["noverlap"],
                 nfft=parameters["nfft"], detrend=parameters["detrend"],
                 scaling="density", return_onesided=True, average="mean")


def band_spectral_centroid(frequency: np.ndarray, density: np.ndarray,
                           maximum_hz: float) -> float:
    """PSD-weighted mean frequency over uniform Welch bins in [0, maximum_hz].

    Use linear power density, never dB values. Equal bin widths cancel in the
    ratio sum(f * PSD) / sum(PSD). Silence is undefined and fails explicitly.
    """
    frequency, density = np.asarray(frequency, dtype=float), np.asarray(density, dtype=float)
    maximum_hz = _finite(maximum_hz, "centroid band upper bound", positive=True)
    if (frequency.ndim != 1 or density.shape != frequency.shape or len(frequency) < 2
            or not np.all(np.isfinite(frequency)) or not np.all(np.isfinite(density))
            or np.any(density < 0) or frequency[0] != 0 or frequency[-1] < maximum_hz):
        raise ValueError("Centroid requires nonnegative finite PSD bins covering the band from 0 Hz")
    steps = np.diff(frequency)
    if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-10, atol=1e-12):
        raise ValueError("Centroid requires uniformly spaced, increasing Welch frequencies")
    keep = frequency <= maximum_hz
    total = float(np.sum(density[keep]))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Centroid is undefined for zero band power")
    return float(np.sum(frequency[keep] * density[keep]) / total)


def summarize_ship_centroids(records: list[dict], parameters: dict,
                             maximum_hz: float) -> tuple[dict, list[dict]]:
    """Summarize every Eval ship record using the representative PSD settings."""
    rows, groups = [], {leaf: [] for leaf in SHIP_CLASSES}
    seen = set()
    for record in sorted(records, key=lambda item: item["id"]):
        if record.get("l3") not in SHIP_CLASSES:
            continue
        if record["id"] in seen:
            raise ValueError(f"Duplicate centroid record ID: {record['id']}")
        seen.add(record["id"])
        if record["fs"] != parameters["fs"]:
            raise ValueError(f"{record['id']}: centroid comparison requires a common sample rate")
        audio, proof = load_audio(record)
        frequency, density = compute_welch(audio, parameters)
        value = band_spectral_centroid(frequency, density, maximum_hz)
        groups[record["l3"]].append(value)
        rows.append({**proof, "centroid_hz": value})
    summaries = {}
    for leaf, values in groups.items():
        if not values:
            raise ValueError(f"No centroid records for {leaf}")
        q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
        summaries[leaf] = {"n": len(values), "q25_hz": float(q25),
                           "median_hz": float(median), "q75_hz": float(q75)}
    return {
        "population": "all ship-noise records in UA-Bench-Eval",
        "statistical_unit": "one manifest record; channel realizations counted separately",
        "record_count": len(rows), "classes": summaries,
        "frequency_limits_hz": [0.0, maximum_hz],
        "definition": "sum(f * linear Welch PSD) / sum(linear Welch PSD), for bins 0 <= f <= upper band limit",
        "welch": dict(parameters), "time_window": "entire received WAV for every record",
        "quantile_method": "linear", "display": "point = median; interval = 25th to 75th percentile; not a confidence interval",
        "invalid_data_policy": "fail; no silent omissions or imputation",
        "interpretation": "band-limited received spectral energy location; not total-band centroid or class separability",
    }, rows

"""Scientific invariants for representative selection and acoustic estimation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from visualization.acoustic_analysis import (
    CLASS_ORDER, SHIP_CLASSES,
    band_spectral_centroid, communication_window, compute_spectrogram, compute_welch, frequency_band,
    load_audio, power_db, select_representatives, stft_parameters, welch_parameters,
    summarize_ship_centroids,
)


def active_record(sid="sample", *, leaf="CW", duration=2.0, gain=-70.0, frequency=2000.0):
    return {"id": sid, "source_id": sid + "_source", "l1": "active",
            "l2": "pulse" if leaf in ("CW", "LFM", "HFM") else "communication",
            "l3": leaf, "duration_s": duration, "G_h": gain,
            "signal_frequency_hz": frequency, "fs": 16000, "frames": round(duration * 16000),
            "signal_params": {"symbol_rate_baud": 200, "band_low_hz": frequency - 300,
                              "band_high_hz": frequency + 300}}


class RepresentativeTests(unittest.TestCase):
    def test_class_median_and_input_permutation(self):
        rows = [active_record("a", duration=1, gain=-80, frequency=1000),
                active_record("median", duration=2, gain=-70, frequency=2000),
                active_record("c", duration=3, gain=-60, frequency=3000)]
        first, proof = select_representatives(rows)
        second, second_proof = select_representatives(rows[::-1])
        self.assertEqual(first["CW"]["id"], "median")
        self.assertEqual(first, second)
        self.assertEqual(proof, second_proof)
        self.assertEqual(proof["classes"]["CW"]["selected_distance"], 0)

    def test_zero_iqr_dimensions_and_id_tie(self):
        rows = [active_record("z", duration=1), active_record("a", duration=3)]
        selected, proof = select_representatives(rows)
        self.assertEqual(selected["CW"]["id"], "a")
        self.assertEqual(proof["classes"]["CW"]["dropped_zero_iqr"], ["G_h", "signal_frequency_hz"])
        self.assertEqual(proof["classes"]["CW"]["exact_distance_tie_count"], 2)

    def test_all_zero_scales_and_predictions_ignored(self):
        rows = [active_record("z"), active_record("a")]
        rows[0]["model_correct"] = True
        rows[1]["model_correct"] = False
        selected, proof = select_representatives(rows)
        self.assertEqual(selected["CW"]["id"], "a")
        self.assertEqual(proof["classes"]["CW"]["used_features"], [])

    def test_missing_nonfinite_and_duplicate_rejected(self):
        for missing in (None, float("nan"), float("inf")):
            row = active_record()
            row["signal_frequency_hz"] = missing
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                select_representatives([row])
        with self.assertRaises(ValueError):
            select_representatives([active_record(), active_record()])

    def test_ship_features_do_not_require_frequency(self):
        row = {"id": "ship", "source_id": "ship_source", "l1": "passive", "l2": "ship_noise",
               "l3": "cargo", "duration_s": 20.0, "S_src": -10.0}
        _, proof = select_representatives([row])
        self.assertEqual(proof["classes"]["cargo"]["features"], ["duration_s", "S_src"])


class AcousticEstimationTests(unittest.TestCase):
    def test_symbol_crop_is_centered_and_half_open(self):
        row = active_record(leaf="2FSK", duration=10)
        crop = communication_window(row)
        self.assertEqual(crop["stop_sample_exclusive"] - crop["start_sample"], 24 * 80)
        self.assertEqual(crop["start_sample"] + crop["stop_sample_exclusive"], row["frames"])
        self.assertEqual(crop["nominal_symbols_shown"], 24)

    def test_short_record_and_ofdm_use_actual_symbol_duration(self):
        row = active_record(leaf="OFDM", duration=2.56)
        row["signal_params"] = {"num_ofdm_symbols": 4, "subcarrier_spacing_hz": 1.6,
                                "num_subcarriers": 2048, "bandwidth_hz": 3200, "cp_factor": 0}
        crop = communication_window(row)
        self.assertAlmostEqual(crop["symbol_duration_s"], 0.64)
        self.assertTrue(crop["used_entire_waveform"])
        self.assertEqual(crop["nominal_symbols_shown"], 4)
        self.assertTrue(crop["ofdm_duration_check"]["matches"])
        changed = deepcopy(row)
        changed["signal_params"]["cp_factor"] = 0.25
        with self.assertRaisesRegex(ValueError, "disagree with actual WAV duration"):
            communication_window(changed)
        changed["frames"] = 51200
        changed["duration_s"] = 3.2
        self.assertAlmostEqual(communication_window(changed)["symbol_duration_s"], 0.8)
        row["signal_params"].pop("num_ofdm_symbols")
        with self.assertRaises(ValueError):
            communication_window(row)

    def test_stft_power_and_time_are_calibrated(self):
        rate = 16000
        row = active_record(duration=1)
        crop = {"start_sample": 0, "stop_sample_exclusive": rate}
        parameters = stft_parameters(row, crop)
        samples = 0.4 * np.cos(2 * np.pi * 2000 * np.arange(rate) / rate)
        frequency, times, powers = compute_spectrogram(samples, parameters, 3.0)
        peak = np.argmin(abs(frequency - 2000))
        # Spectrum-scaled, one-sided power of a bin-centered sinusoid = A² / 2.
        self.assertAlmostEqual(float(powers[peak, len(times) // 2]), 0.4 ** 2 / 2, places=10)
        self.assertAlmostEqual(times[0], 3.0)
        self.assertLessEqual(times[-1], 4.0 + 1e-12)
        self.assertEqual(powers.shape, (len(frequency), len(times)))
        self.assertTrue(np.all(np.isfinite(powers)))

    def test_stft_short_window_parameters_remain_valid(self):
        row = active_record(leaf="QPSK", duration=0.02)
        crop = communication_window(row)
        parameters = stft_parameters(row, crop)
        _, _, powers = compute_spectrogram(np.ones(crop["stop_sample_exclusive"] - crop["start_sample"]), parameters)
        self.assertGreater(powers.size, 0)
        self.assertLess(parameters["noverlap"], parameters["nperseg"])

    def test_frequency_band_clips_at_nyquist(self):
        row = active_record(frequency=7800)
        band = frequency_band(row, 10)
        self.assertEqual(band["display_limits_hz"][1], 8000)
        self.assertTrue(band["clipped_to_nyquist"])

    def test_welch_integral_matches_variance_without_gain_normalization(self):
        rate = 16000
        samples = 0.3 * np.sin(2 * np.pi * 250 * np.arange(rate * 3) / rate)
        parameters = welch_parameters(rate, len(samples))
        frequency, density = compute_welch(samples, parameters)
        self.assertAlmostEqual(float(np.sum(density) * (frequency[1] - frequency[0])), 0.3 ** 2 / 2, places=10)
        _, scaled = compute_welch(2 * samples, parameters)
        np.testing.assert_allclose(scaled, 4 * density, rtol=1e-12, atol=1e-25)
        peak = np.argmax(density)
        self.assertAlmostEqual(float(power_db(scaled, 1)[peak] - power_db(density, 1)[peak]), 20 * np.log10(2), places=10)

    def test_power_reference_shared_and_zero_finite(self):
        first = power_db(np.array([1.0, 0.0]), 4.0, -50)
        second = power_db(np.array([4.0]), 4.0, -50)
        self.assertAlmostEqual(first[0], -6.020599913279624)
        self.assertEqual(first[1], -50)
        self.assertEqual(second[0], 0)
        with self.assertRaises(ValueError):
            power_db(np.ones(2), 0)

    def test_centroid_uses_linear_power_is_band_limited_and_gain_invariant(self):
        rate = 16000
        time = np.arange(rate * 3) / rate
        # In-band tones carry power in a 1:4 ratio; the loud 2-kHz tone is outside.
        audio = (0.1 * np.sin(2 * np.pi * 200 * time)
                 + 0.2 * np.sin(2 * np.pi * 600 * time)
                 + 0.5 * np.sin(2 * np.pi * 2000 * time))
        frequency, density = compute_welch(audio, welch_parameters(rate, len(audio)))
        self.assertAlmostEqual(band_spectral_centroid(frequency, density, 1000), 520, places=7)
        self.assertAlmostEqual(band_spectral_centroid(frequency, 9 * density, 1000), 520, places=7)

    def test_centroid_rejects_silence_invalid_power_and_nonuniform_bins(self):
        frequency = np.arange(1001.0)
        for density in (np.zeros(1001), -np.ones(1001), np.full(1001, np.nan)):
            with self.subTest(first=density[0]), self.assertRaises(ValueError):
                band_spectral_centroid(frequency, density, 1000)
        with self.assertRaises(ValueError):
            band_spectral_centroid(np.array([0, 100, 1000]), np.ones(3), 1000)

    def test_centroid_summary_uses_all_class_records_with_deterministic_order(self):
        from unittest.mock import patch
        rows = [{"id": f"{leaf}-{i}", "l3": leaf, "fs": 16000}
                for leaf in SHIP_CLASSES for i in range(4)]
        time = np.arange(32000) / 16000
        def fake_load(record):
            tone = 100 + 100 * int(record["id"].rsplit("-", 1)[1])
            return 0.2 * np.sin(2 * np.pi * tone * time), {"id": record["id"], "l3": record["l3"]}
        with patch("visualization.acoustic_analysis.load_audio", side_effect=fake_load):
            summary, raw = summarize_ship_centroids(rows[::-1], welch_parameters(16000, 32000), 1000)
        self.assertEqual(summary["record_count"], 20)
        self.assertEqual([r["id"] for r in raw], sorted(r["id"] for r in rows))
        for group in summary["classes"].values():
            self.assertEqual(group["n"], 4)
            np.testing.assert_allclose([group[k] for k in ("q25_hz", "median_hz", "q75_hz")],
                                       [175, 250, 325], rtol=1e-10)

    def test_actual_wav_header_hash_and_amplitude(self):
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.wav"
            sf.write(path, np.full(320, 0.25), 16000, subtype="FLOAT")
            row = active_record(duration=0.02)
            row["audio_path"] = str(path.resolve())
            audio, proof = load_audio(row)
            np.testing.assert_array_equal(audio, np.full(320, 0.25))
            self.assertEqual(len(proof["audio_sha256"]), 64)
            changed = deepcopy(row)
            changed["frames"] += 1
            with self.assertRaises(ValueError):
                load_audio(changed)


class FigureBuilderTests(unittest.TestCase):
    def test_independent_axes_exact_size_and_finite_plot_payload(self):
        import json
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import soundfile as sf
        from visualization.plot_acoustic import build_active_figure, build_ship_figure
        from visualization.project import paper_style

        style = {"width_mm": 180, "font_size_pt": 9, "font_family": "Times New Roman",
                 "legend_frame": False}
        with tempfile.TemporaryDirectory() as temporary:
            records = []
            rate = 16000
            time = np.arange(rate * 2) / rate
            for index, leaf in enumerate(CLASS_ORDER):
                record = active_record(leaf, leaf=leaf)
                if leaf in SHIP_CLASSES:
                    record.update(l1="passive", l2="ship_noise", S_src=-10.0)
                if leaf == "OFDM":
                    record["signal_params"].update(num_ofdm_symbols=4, num_subcarriers=128,
                                                   bandwidth_hz=256, cp_factor=0)
                path = Path(temporary) / f"{leaf}.wav"
                frequency = 200 + 50 * index if leaf in SHIP_CLASSES else 2000
                waveform = 0.25 * np.sin(2 * np.pi * frequency * time)
                sf.write(path, waveform, rate, subtype="FLOAT")
                record["audio_path"] = str(path.resolve())
                records.append(record)
            with paper_style(style):
                for builder, count in ((build_active_figure, 8), (build_ship_figure, 6)):
                    figure, payload = builder(records, style)
                    try:
                        figure.canvas.draw()
                        self.assertAlmostEqual(figure.get_figwidth() * 25.4, 180)
                        self.assertGreater(figure.get_figwidth(), figure.get_figheight())
                        json.dumps(payload, allow_nan=False)
                        for ax in figure.axes[:count]:
                            self.assertEqual(len(ax.get_shared_x_axes().get_siblings(ax)), 1)
                            self.assertEqual(len(ax.get_shared_y_axes().get_siblings(ax)), 1)
                            self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))
                            self.assertEqual(len(ax.xaxis.get_minorticklocs()), 0)
                            self.assertEqual(ax.xaxis.label.get_fontsize(), 9)
                        renderer = figure.canvas.get_renderer()
                        for ax in figure.axes:
                            box = ax.get_tightbbox(renderer)
                            if box is not None:
                                self.assertGreaterEqual(box.x0, -1)
                                self.assertGreaterEqual(box.y0, -1)
                                self.assertLessEqual(box.x1, figure.bbox.width + 1)
                                self.assertLessEqual(box.y1, figure.bbox.height + 1)
                    finally:
                        plt.close(figure)


if __name__ == "__main__":
    unittest.main()

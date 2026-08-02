import sys
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ARCHIVE_ROOT))

from utils.bellhop_runner import (  # noqa: E402
    BellhopRunner,
    _parse_arr_text,
    apply_channel,
    apply_frequency_dependent_channel,
    build_cir,
    generate_env,
    make_channel_rng,
    parse_arrivals,
    _postprocess_arrivals,
    resolve_num_beams,
)


class BellhopRunnerTests(unittest.TestCase):
    def test_adaptive_beams_are_bounded(self):
        self.assertEqual(resolve_num_beams("adaptive", 5.0), 1000)
        self.assertEqual(resolve_num_beams("adaptive", 17.525), 1753)
        self.assertEqual(resolve_num_beams("adaptive", 80.0), 8000)
        self.assertEqual(resolve_num_beams("bellhop_auto", 80.0), 0)
        self.assertEqual(resolve_num_beams(2400, 80.0), 2400)

    def test_generate_env_covers_long_range_and_uses_auto_beams(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / "long_range.env"
            generate_env(
                env_path=env_path,
                title="80 km regression",
                freq_hz=500.0,
                depths_m=np.array([0.0, 50.0, 100.0]),
                sound_speeds_mps=np.array([1500.0, 1495.0, 1490.0]),
                water_depth_m=100.0,
                source_depth_m=20.0,
                receiver_depth_m=25.0,
                range_km=80.0,
                num_beams=0,
                ray_box_margin=1.05,
            )

            lines = env_path.read_text(encoding="utf-8").splitlines()
            run_type_index = lines.index("'A'")
            self.assertEqual(lines[run_type_index + 1], "0")
            _, z_box, r_box = map(float, lines[-1].split())
            self.assertGreater(z_box, 100.0)
            self.assertGreater(r_box, 80.0)

    def test_ascii_arrival_columns_are_not_shifted(self):
        arrivals = _parse_arr_text(
            b"1.0e-5 90.0 52.25 0.001 -10.5 12.5 3 4\n"
        )

        self.assertEqual(len(arrivals), 1)
        arrival = arrivals[0]
        self.assertEqual(arrival["delay_s"], 52.25)
        self.assertEqual(arrival["delay_imag_s"], 0.001)
        self.assertEqual(arrival["source_angle_deg"], -10.5)
        self.assertEqual(arrival["receiver_angle_deg"], 12.5)
        self.assertEqual(arrival["num_top_bnc"], 3)
        self.assertEqual(arrival["num_bot_bnc"], 4)

    def test_relative_delay_keeps_long_range_arrival_in_cir(self):
        arrivals = [{
            "delay_s": 52.25,
            "amplitude_linear": 0.5,
            "amplitude_normalized": 1.0,
            "phase_rad": 0.0,
        }]
        cir = build_cir(
            arrivals, fs_hz=1000.0, duration_s=0.1,
            delay_reference_s=52.25,
        )
        self.assertGreater(np.max(np.abs(cir)), 0.9)

    def test_fractional_delay_points_toward_later_sample(self):
        arrivals = [{"delay_s": 0.01025, "amplitude_linear": 1.0}]
        cir = build_cir(
            arrivals, fs_hz=1000.0, duration_s=0.05,
            use_normalized=False,
        )
        self.assertGreater(abs(cir[11]), abs(cir[9]))

    def test_channel_rng_is_independent_of_processing_order(self):
        first = make_channel_rng(42, "pulsecom", "4FSK", "sample", 0)
        second = make_channel_rng(42, "pulsecom", "4FSK", "sample", 0)
        other = make_channel_rng(42, "pulsecom", "4FSK", "sample", 1)
        self.assertEqual(first.uniform(), second.uniform())
        self.assertNotEqual(make_channel_rng(42, "x").uniform(), other.uniform())

    def test_binary_arrival_fallback_is_reachable(self):
        raw = struct.pack("<fiii", 100.0, 1, 1, 1)
        raw += struct.pack("<i", 1)
        raw += struct.pack("<fffffii", 0.25, 0.0, 1.5, -3.0, 4.0, 1, 2)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "binary.arr"
            path.write_bytes(raw)
            arrivals = parse_arrivals(path)
        self.assertEqual(len(arrivals), 1)
        self.assertAlmostEqual(arrivals[0]["delay_s"], 1.5)

    def test_arrival_limit_keeps_true_first_arrival(self):
        arrivals = [
            {"delay_s": 1.0, "amplitude_linear": 0.001},
            {"delay_s": 2.0, "amplitude_linear": 1.0},
            {"delay_s": 3.0, "amplitude_linear": 0.5},
        ]
        selected = _postprocess_arrivals(arrivals, max_arrivals=2)
        self.assertEqual([item["delay_s"] for item in selected], [1.0, 2.0])

    def test_zero_arrival_adaptive_run_retries_with_more_beams(self):
        runner = object.__new__(BellhopRunner)
        calls = []

        def fake_run(**kwargs):
            calls.append(kwargs["num_beams"])
            return [] if len(calls) == 1 else [{"delay_s": 1.0}]

        runner.run = fake_run
        result = runner.run_with_beam_retry(
            range_km=10.0, num_beams="adaptive", max_beams=8000,
        )
        self.assertEqual(calls, ["adaptive", 4000])
        self.assertTrue(result["beam_retry"])
        self.assertEqual(result["num_beams_effective"], 4000)
        self.assertEqual(len(result["arrivals"]), 1)

    def test_empty_cir_never_returns_original_audio(self):
        with self.assertRaises(ValueError):
            apply_channel(np.ones(16), np.zeros(16))

    def test_single_frequency_channel_preserves_raw_gain(self):
        audio = np.array([1.0, 0.0, 0.0, 0.0])
        cir = np.array([0.25, 0.0, 0.0, 0.0])
        output = apply_frequency_dependent_channel(
            audio, [{"freq_hz": 100.0, "cir": cir}], fs_hz=1000.0,
        )
        self.assertAlmostEqual(output[0], 0.25, places=12)


if __name__ == "__main__":
    unittest.main()

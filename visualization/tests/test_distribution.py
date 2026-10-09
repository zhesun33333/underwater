"""Count, provenance and rendering-contract checks for paper Figure B."""
import copy
import json
import math
import unittest

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.text import Text
from matplotlib.ticker import NullLocator
import numpy as np

from visualization.plot_distribution import build_figure, ecdf_coordinates


STYLE = {"width_mm": 180, "font_family": "Times New Roman", "font_size_pt": 9,
         "legend_frame": False}


def record(sample_id, family="pulse", duration=1.0, frequency=500.0, gain=-30.0, ratio=None):
    return {"id": sample_id, "source_id": sample_id.rsplit("_ch", 1)[0],
            "l1": "passive" if family == "ship_noise" else "active",
            "l2": family, "l3": {"pulse": "CW", "communication": "BPSK", "ship_noise": "cargo"}[family],
            "duration_s": duration, "signal_frequency_hz": frequency,
            "channel_frequency_hz": 5500.0, "G_h": gain, "S_src": ratio,
            "fs": 16000, "frames": 16000}


def panel(payload, key):
    return next(item for item in payload["panels"] if item["key"] == key)


def series(payload, key, collection="full_test", family="pulse"):
    return next(item for item in panel(payload, key)["series"]
                if item["collection"] == collection and item["family"] == family)


class ECDFTests(unittest.TestCase):
    def test_ties_have_one_jump_with_combined_probability_mass(self):
        x, y = ecdf_coordinates([2.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(x, [1.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(y, [0.0, 0.25, 0.75, 1.0])
        self.assertEqual(float(y[-1]), 1.0)

    def test_constant_and_empty_distributions(self):
        x, y = ecdf_coordinates([2.0, 2.0])
        np.testing.assert_array_equal(x, [2.0, 2.0])
        np.testing.assert_array_equal(y, [0.0, 1.0])
        x, y = ecdf_coordinates([])
        self.assertEqual((x.size, y.size), (0, 0))
        with self.assertRaisesRegex(ValueError, "finite"):
            ecdf_coordinates([math.nan])


class DistributionTests(unittest.TestCase):
    def test_missing_values_are_reported_but_not_ecdf_denominators(self):
        rows = [record("valid_ch0", duration=2.0), record("null", duration=None),
                record("nan", duration=math.nan), record("zero", duration=0),
                record("string", duration="2"), record("bool", duration=True)]
        _, payload = build_figure(rows, [rows[0]], STYLE)
        data = series(payload, "a")
        self.assertEqual((data["population_n"], data["n"], data["missing"]), (6, 1, 5))
        self.assertEqual(data["missing_by_reason"],
                         {"missing": 1, "nonfinite": 1, "nonpositive": 1, "not_numeric": 2})
        self.assertEqual(data["ids"], ["valid_ch0"])
        self.assertEqual(data["raw_ids"], [r["id"] for r in rows])
        self.assertEqual(data["ecdf"]["y"], [0, 1])
        self.assertEqual(data["median"], 2)
        json.dumps(payload, allow_nan=False)

    def test_panels_use_source_frequency_and_family_specific_statistics(self):
        pulse = record("pulse_ch0", frequency=400, gain=-80)
        comm = record("comm_ch0", "communication", frequency=1200, gain=-40)
        ship = record("ship_ch0", "ship_noise", frequency=None, gain=999, ratio=-12)
        no_frequency = record("no_frequency_ch0", frequency=None, gain=-70)
        rows = [pulse, comm, ship, no_frequency]
        _, payload = build_figure(rows, [pulse, comm, ship], STYLE)
        self.assertEqual(series(payload, "b")["values"], [400])
        self.assertEqual(series(payload, "b")["missing_ids"], ["no_frequency_ch0"])
        self.assertEqual(series(payload, "b", family="communication")["values"], [1200])
        self.assertEqual(series(payload, "c")["values"], [-80, -70])
        self.assertEqual(series(payload, "d", family="ship_noise")["values"], [-12])
        self.assertEqual({x["family"] for x in panel(payload, "c")["series"]},
                         {"pulse", "communication"})
        self.assertEqual(panel(payload, "b")["source_field"], "signal_frequency_hz")
        self.assertEqual(payload["weighting"], "record_weighted")

    def test_channels_remain_separate_record_weighted_observations(self):
        rows = [record("source_ch0", duration=1), record("source_ch1", duration=1),
                record("other_ch0", duration=4)]
        _, payload = build_figure(rows, rows[:2], STYLE)
        data = series(payload, "a")
        self.assertEqual(data["n"], 3)
        np.testing.assert_array_equal(data["ecdf"]["y"], [0, 2 / 3, 1])
        self.assertEqual(payload["relationship"]["shared_id_count"], 2)

    def test_relationship_errors_are_not_silently_plotted(self):
        a, b = record("a"), record("b")
        for full, evaluation, message in [
            ([a], [b], "not a subset"), ([a, a], [a], "duplicate"),
            ([a], [a, a], "duplicate"), ([], [a], "at least one"),
        ]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build_figure(full, evaluation, STYLE)
        changed = copy.deepcopy(a)
        changed["signal_frequency_hz"] = 600
        with self.assertRaisesRegex(ValueError, "a.signal_frequency_hz"):
            build_figure([a], [changed], STYLE)

    def test_independent_axes_styles_and_no_global_style_mutation(self):
        rows = [record("p", duration=0.2), record("q", duration=2),
                record("c", "communication", duration=10),
                record("s", "ship_noise", duration=25, frequency=None, ratio=-10)]
        before = copy.deepcopy(mpl.rcParams)
        figure, _ = build_figure(rows, rows, STYLE)
        self.assertEqual(dict(mpl.rcParams), dict(before))
        np.testing.assert_allclose(figure.get_size_inches() * 25.4, [180, 135])
        self.assertEqual(len(figure.axes), 4)
        self.assertEqual(figure.axes[0].get_xscale(), "log")
        FigureCanvasAgg(figure).draw()
        for ax in figure.axes:
            self.assertEqual(ax.get_ylim(), (0, 1))
            self.assertTrue(all(spine.get_visible() for spine in ax.spines.values()))
            self.assertIsInstance(ax.xaxis.get_minor_locator(), NullLocator)
            self.assertIsInstance(ax.yaxis.get_minor_locator(), NullLocator)
            self.assertTrue(ax.get_xlabel())
            self.assertEqual(ax.get_ylabel(), "ECDF")
            self.assertTrue(all(line.get_drawstyle() == "steps-post" for line in ax.lines))
            for other in figure.axes:
                if other is not ax:
                    self.assertFalse(ax.get_shared_x_axes().joined(ax, other))
                    self.assertFalse(ax.get_shared_y_axes().joined(ax, other))
        self.assertTrue(all(text.get_fontsize() == 9 for text in figure.findobj(Text)))
        self.assertTrue(all(not legend.get_frame_on() for legend in figure.legends))


if __name__ == "__main__":
    unittest.main()

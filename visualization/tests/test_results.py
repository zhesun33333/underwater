"""Result-figure contracts: denominators, execution states, strata and pairs."""
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from visualization.result_analysis import (
    CLASS_ORDER, CLASS_TO_FAMILY, LEGACY_STATES, STRICT_STATES, analyze_predictions,
    canonical_predictions, diagnostics, diagnostics_from_metrics, label_key,
    make_strata, metadata_performance, model_performance_range, validate_comparison,
)
from visualization.results import SourceReader, load_results, prepare_results


def fixture():
    records, rows = [], []
    for leaf in CLASS_ORDER:
        family = CLASS_TO_FAMILY[leaf]
        l1 = "passive" if family == "ship_noise" else "active"
        for i in range(3):
            sid = f"{leaf}_{i}_ch0"
            gt = {"L1": l1, "L2": family, "L3": leaf}
            records.append({"id": sid, "l1": l1, "l2": family, "l3": leaf,
                            "G_h": -80+i*10, "S_src": -15+i*3})
            rows.append({"sample_id": sid, "gt": gt, "turn1_pred": l1,
                         "turn1_parse_status": "valid_option", "cascade_skipped": False,
                         "turn2_pred": {**gt, "parse_status": "valid_option"}})
    return records, rows


def saved_metrics(counts, labels=None):
    d = diagnostics(counts)
    labels = labels or CLASS_ORDER + ["unknown", "cascade_error"]
    square = np.vstack([counts, np.zeros((2, 15), dtype=int)])
    return {"total_samples": d["n"], "l3_accuracy": d["l3_accuracy"], "l3_macro_f1": d["l3_macro_f1"],
            "l3_confusion_labels": labels, "l3_confusion_matrix": square.tolist(),
            "l3_per_class": {labels[i]: {key: d[key][i] for key in ("precision", "recall", "f1", "support")}
                             for i in range(13)}}


class ResultStatisticsTests(unittest.TestCase):
    def test_matrix_keeps_failure_denominators_and_chinese_underwater_class(self):
        counts = np.zeros((13, 15), dtype=int)
        for i in range(13): counts[i, i] = 1; counts[i, 13] = 1; counts[i, 14] = 2
        labels = CLASS_ORDER[:8] + ["货船", "邮轮", "渔船", "军舰", "水下目标", "unknown", "cascade_error"]
        metrics = saved_metrics(counts, labels)
        order = list(reversed(range(15)))
        metrics["l3_confusion_labels"] = [labels[i] for i in order]
        metrics["l3_confusion_matrix"] = np.asarray(metrics["l3_confusion_matrix"])[np.ix_(order, order)].tolist()
        result = diagnostics_from_metrics(metrics)
        self.assertEqual(result["counts"], counts.tolist())
        self.assertEqual(result["recall"], [.25]*13)
        np.testing.assert_allclose(np.sum(result["row_fraction"], axis=1), 1)
        self.assertEqual(label_key("Underwater vehicle"), "underwater_target")

    def test_metrics_reject_unknown_labels_noninteger_counts_and_inconsistent_totals(self):
        matrix = np.pad(np.eye(13, dtype=int), ((0, 0), (0, 2)))
        original = saved_metrics(matrix)
        for case in ("label", "count", "total", "ground_truth"):
            value = deepcopy(original)
            if case == "label": value["l3_confusion_labels"][0] = "made-up"
            elif case == "count": value["l3_confusion_matrix"][0][0] = .5
            elif case == "total": value["total_samples"] += 1
            else: value["l3_confusion_matrix"][13][0] = 1
            with self.subTest(case=case), self.assertRaises(ValueError):
                diagnostics_from_metrics(value)

    def test_five_states_are_exclusive_and_match_leaf_accuracy(self):
        records, rows = fixture()
        # First five records supply one example of each state; remaining rows correct.
        rows[0].update(turn1_pred="unknown", turn1_parse_status="invalid_format", cascade_skipped=True)
        rows[1].update(turn1_pred="passive", cascade_skipped=True)
        for row in rows[:2]: row["turn2_pred"] = {"L3": "cascade_error"}
        rows[2]["turn2_pred"] = {"L3": "unknown", "parse_status": "invalid_format"}
        rows[3]["turn2_pred"]["L3"] = "HFM"
        canonical = canonical_predictions(rows, records, strict=True)
        d, states = analyze_predictions(canonical, strict=True)
        self.assertEqual(states["states"], STRICT_STATES)
        self.assertEqual(states["counts"], [1, 1, 1, 1, 35])
        self.assertEqual(states["cascade_count"], 2)
        self.assertAlmostEqual(d["l3_accuracy"], 35/39)

    def test_legacy_does_not_infer_format_status_from_labels_or_parse_tier(self):
        records, rows = fixture()
        for row in rows:
            row.pop("turn1_parse_status"); row["turn2_pred"].pop("parse_status")
            row["turn2_pred"]["parse_tier"] = 1
        with self.assertRaisesRegex(ValueError, "parse status"):
            canonical_predictions(rows, records, strict=True)
        _, outcomes = analyze_predictions(canonical_predictions(rows, records, strict=False), strict=False)
        self.assertEqual(outcomes["states"], LEGACY_STATES)
        self.assertEqual(outcomes["counts"], [0, 0, 39])

    def test_predictions_require_unique_complete_ids_consistent_truth_and_gate(self):
        records, rows = fixture()
        cases = [rows[:-1], rows+[rows[0]], deepcopy(rows), deepcopy(rows)]
        cases[2][0]["gt"]["L3"] = "LFM"
        cases[3][0]["cascade_skipped"] = True
        for altered in cases:
            with self.assertRaises(ValueError): canonical_predictions(altered, records, strict=False)

    def test_metadata_uses_equal_class_weights_and_fixed_tie_boundaries(self):
        records, rows = fixture()
        # CW has twice as many records in each stratum; pooled accuracy would differ.
        for record, row in list(zip(records[:3], rows[:3])):
            extra_record, extra_row = deepcopy(record), deepcopy(row)
            extra_record["id"] += "_extra"; extra_row["sample_id"] += "_extra"
            records.append(extra_record); rows.append(extra_row)
        strata = make_strata(records)
        canonical = canonical_predictions(rows, records, strict=True)
        # Whole CW class wrong, every other class correct: active macro = 7/8.
        for row in canonical:
            if row["gt"]["L3"] == "CW": row["l3"] = "unknown"
        result = metadata_performance(canonical, strata)
        self.assertEqual(result["groups"]["active"]["macro_recall"], [.875]*3)
        self.assertEqual(result["groups"]["ship_noise"]["macro_recall"], [1]*3)
        # Identical values stay together; an empty CW stratum invalidates the group average.
        for row in records:
            if row["l3"] == "CW": row["G_h"] = -70
        fixed = make_strata(records)
        self.assertEqual(fixed["thresholds"]["CW"]["counts"], [6, 0, 0])
        result = metadata_performance(canonical, fixed)
        self.assertEqual(result["groups"]["active"]["macro_recall"][1:], [None, None])
        self.assertEqual(result["groups"]["active"]["missing_classes"][1], ["CW"])

    def test_model_range_tracks_crossing_extrema_ties_and_percentage_points(self):
        runs = [{'id': name, 'metadata': {'groups': {'active': {'macro_recall': values}}}}
                for name, values in [('a', [.2, .1, .3]), ('b', [.1, .4, .2]), ('c', [.2, .2, .2])]]
        original = deepcopy(runs)
        result = model_performance_range(runs, 'active')
        np.testing.assert_allclose(result['lower'], [.1, .1, .2])
        np.testing.assert_allclose(result['upper'], [.2, .4, .3])
        np.testing.assert_allclose(result['gap_pp'], [10, 30, 10])
        self.assertEqual(result['upper_model_ids'], [['a', 'c'], ['b'], ['a']])
        self.assertEqual(result['lower_model_ids'], [['b'], ['a'], ['b', 'c']])
        self.assertEqual(result['missing_model_ids'], [[], [], []])
        self.assertEqual(runs, original)

    def test_model_range_does_not_silently_drop_a_missing_model(self):
        runs = [{'id': name, 'metadata': {'groups': {'active': {'macro_recall': values}}}}
                for name, values in [('a', [.2, .1, .3]), ('b', [.1, None, .2])]]
        result = model_performance_range(runs, 'active')
        for key in ('lower', 'upper', 'gap_pp'):
            self.assertIsNone(result[key][1])
        self.assertEqual(result['missing_model_ids'], [[], ['b'], []])
        self.assertEqual(result['upper_model_ids'][1], [])
        self.assertEqual(result['lower_model_ids'][1], [])
        self.assertEqual(result['upper'][2], .3)
        json.dumps(result, allow_nan=False)

    def test_model_range_rejects_ambiguous_or_invalid_inputs(self):
        run = {'id': 'a', 'metadata': {'groups': {'active': {'macro_recall': [.1, .2, .3]}}}}
        for runs in ([], [run, run]):
            with self.assertRaises(ValueError): model_performance_range(runs, 'active')
        for values in ([.1, .2], [.1, float('nan'), .3], [.1, float('inf'), .3],
                       [.1, -.1, .3], [.1, 1.1, .3], [.1, True, .3]):
            bad = deepcopy(run)
            bad['metadata']['groups']['active']['macro_recall'] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                model_performance_range([bad], 'active')

    def test_paired_control_retains_negative_delta_and_requires_verified_aggregates(self):
        value = comparison_fixture()
        result = validate_comparison(value)
        self.assertAlmostEqual(result["metrics"]["l3_accuracy"]["delta_pp"], -10)
        for field in ("integrity_verified", "total_samples"):
            bad = deepcopy(value); bad["silent"][field] = False if field == "integrity_verified" else 99
            with self.assertRaises(ValueError): validate_comparison(bad)
        bad = deepcopy(value); bad["delta"]["l3_accuracy"] = .1
        with self.assertRaises(ValueError): validate_comparison(bad)


def comparison_fixture():
    return {"status": "complete", "comparison": "original_minus_silent",
            "original": {"integrity_verified": True, "run_signature": "a", "total_samples": 39,
                         "l3_accuracy": .2, "l1_accuracy": .6},
            "silent": {"integrity_verified": True, "run_signature": "b", "total_samples": 39,
                       "l3_accuracy": .3, "l1_accuracy": .4},
            "delta": {"l3_accuracy": -.1, "l1_accuracy": .2},
            "counts": {"original": {"samples": 39}, "silent": {"samples": 39}}}


class SourceAndProtocolTests(unittest.TestCase):
    def test_zip_suffix_must_be_unique_and_no_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); archive = base/'input.zip'
            with zipfile.ZipFile(archive, 'w') as target:
                target.writestr('目录/model/metrics.json', '{"n": 2}')
                target.writestr('other/metrics.json', '{"n": 3}')
            reader = SourceReader({"config_path": str(base/'config.json'), "archive": str(archive)})
            try:
                self.assertEqual(reader.json('model/metrics.json'), {"n": 2})
                with self.assertRaises(ValueError): reader.json('metrics.json')
                self.assertEqual(list(base.iterdir()), [archive])
            finally: reader.close()

    def test_current_protocol_results_prepare_and_reject_drift(self):
        from testsite.config import load_config
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); records, _ = fixture()
            prompts = load_config()['prompts']['three_turn']; manifest_rows = []
            for record in records:
                name = record['id']+'.wav'
                sf.write(base/name, np.ones(80)*.25, 16000)
                qs = [prompts['turn1'][0], prompts['turn2_'+record['l1']][0], prompts['turn3']]
                manifest_rows.append({'id': record['id'], 'audio': name,
                    '_gt': {k: record[k.lower()] for k in ('L1', 'L2', 'L3')},
                    'conversations': [{'from': 'human', 'value': q} for q in qs]})
            manifest = base/'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(r)+'\n' for r in manifest_rows), encoding='utf-8')
            result = subprocess.run([sys.executable, '-B', '-m', 'testsite.scripts.run_eval', '--data',
                str(manifest), '--audio-root', str(base), '--backend', 'mock', '--output-dir', str(base/'run')],
                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config_path = base/'results.json'
            config_path.write_text(json.dumps({'mode': 'verified', 'cache_dir': 'cache',
                'diagnostic_model': 'test', 'runs': [{'id': 'test', 'label': 'Test',
                'predictions': ['run/predictions_shard_*.jsonl'], 'protocols': ['run/protocol_shard_*.json']}]}))
            input_config, input_proof = base/'data.json', base/'proof.json'
            input_config.write_text('{}'); input_proof.write_text('{}')
            prepared = ({'config_path': str(input_config), 'eval_manifest': str(manifest),
                         'style': {'width_mm': 180, 'font_family': 'Times New Roman', 'font_size_pt': 9, 'legend_frame': False}},
                        [], records, {'inputs_provenance_path': str(input_proof)})
            with patch('visualization.results.load_inputs', return_value=prepared), \
                    patch.dict(os.environ, {'UABENCH_RESULTS_CONFIG': str(config_path)}):
                report = prepare_results()
                self.assertEqual(report['E_states'], 5)
                data, _ = load_results()
                self.assertTrue(data['runs'][0]['audit']['protocol_verified'])
                self.assertEqual(sum(data['runs'][0]['outcomes']['counts']), len(records))
                predictions = base/'run/predictions_shard_000.jsonl'
                predictions.write_text(predictions.read_text()+'\n', encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'source changed'): load_results()


class ResultLayoutTests(unittest.TestCase):
    def test_metadata_model_identity_survives_reordering_and_subsets(self):
        from visualization.plot_results import _metadata_styles
        runs = [{'id': name} for name in ('af_next', 'gemma4_12b', 'gemma4_run1',
                                          'midashenglm', 'qwen25_omni', 'voxtral_small')]
        styles = _metadata_styles(runs)
        self.assertEqual(styles, _metadata_styles(list(reversed(runs))))
        subset = _metadata_styles(runs[::2])
        self.assertEqual(subset, {run['id']: styles[run['id']] for run in runs[::2]})
        for key in ('color', 'linestyle', 'marker'):
            self.assertEqual(len({value[key] for value in styles.values()}), len(runs))

    def test_all_four_builders_have_independent_axes_and_preserve_statistics(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from visualization.project import paper_style
        from visualization.plot_results import (build_class_diagnostics, build_error_decomposition,
                                                build_metadata_performance, build_silence_control)
        records, rows = fixture(); canonical = canonical_predictions(rows, records, strict=True)
        diag, outcomes = analyze_predictions(canonical, strict=True); strata = make_strata(records)
        data = {'mode': 'verified', 'diagnostic_model': 'test', 'strata': strata,
                'style': {'width_mm': 180, 'font_family': 'Times New Roman', 'font_size_pt': 9, 'legend_frame': False},
                'runs': [{'id': 'test', 'label': 'Model', 'audit': {}, 'diagnostics': diag,
                         'outcomes': outcomes, 'metadata': metadata_performance(canonical, strata)}],
                'comparisons': [{'id': 'test', 'label': 'Model', **validate_comparison(comparison_fixture())}]}
        original = deepcopy(data)
        with paper_style(data['style']):
            for builder in (build_class_diagnostics, build_error_decomposition, build_metadata_performance, build_silence_control):
                fig, payload = builder(data)
                try:
                    fig.canvas.draw(); json.dumps(payload, allow_nan=False)
                    self.assertEqual(data, original)
                    self.assertAlmostEqual(fig.get_figwidth()*25.4, 180)
                    for ax in fig.axes:
                        self.assertEqual(len(ax.get_shared_x_axes().get_siblings(ax)), 1)
                        self.assertEqual(len(ax.xaxis.get_minorticklocs()), 0)
                        box = ax.get_tightbbox(fig.canvas.get_renderer())
                        self.assertGreaterEqual(box.x0, -1, builder.__name__)
                        self.assertGreaterEqual(box.y0, -1)
                        self.assertLessEqual(box.x1, fig.bbox.width+1, builder.__name__)
                        self.assertLessEqual(box.y1, fig.bbox.height+1)
                finally: plt.close(fig)
        with self.assertRaisesRegex(ValueError, 'needs verified comparison'):
            build_silence_control({**data, 'comparisons': []})


if __name__ == '__main__':
    unittest.main()

"""Meaningful archive and layout checks for the matched-silence result figure."""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from visualization.paired_archive_validation import validate_paired_archive


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _signature(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


@contextmanager
def archive_fixture(n=2600, *, with_manifests=False):
    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        records = []
        truth_rows = []
        for index in range(n):
            gt = {"L1": "active", "L2": "pulse", "L3": "LFM"} if index % 2 else {
                "L1": "passive", "L2": "ship_noise", "L3": "cargo"}
            sample_id = f"id_{index:04d}"
            records.append({"id": sample_id, **{key.lower(): value for key, value in gt.items()}})
            truth_rows.append((sample_id, gt))
        manifest_hashes = {"original": "a" * 64, "silent": "b" * 64}
        if with_manifests:
            for condition in ("original", "silent"):
                manifest = base / condition / "manifest.jsonl"
                _write_rows(manifest, [{"id": sample_id, "audio": f"{condition}/{sample_id}.wav",
                    "qa_prompt_version": "paper-v1", "record_schema_version": 1,
                    "conversations": [{"from": "human", "value": f"Question {turn}"}
                                      for turn in (1, 2, 3)],
                    "_gt": gt, "_meta": {"source": "fixture", "tier": index % 3}}
                    for index, (sample_id, gt) in enumerate(truth_rows)])
                manifest_hashes[condition] = _hash(manifest)
        protocol_paths = {}
        prediction_paths = {}
        paired = []
        raw = {"original": [], "silent": []}
        identities = {}
        signatures = {}
        for condition in ("original", "silent"):
            identity = {"run_id": condition, "manifest_sha256": manifest_hashes[condition],
                        "model": {"backend": "fixture", "model_id": "same model"},
                        "taxonomy": {"L1": ["active", "passive"], "L3": ["LFM", "cargo"]},
                        "implementation": {"testsite/core/scorer.py": "fixture-hash"}}
            identities[condition] = identity
            signatures[condition] = _signature(identity)
            protocol_paths[condition] = []
            for shard in (0, 1):
                path = base / condition / f"protocol_shard_{shard:03d}.json"
                _write_json(path, {"run_identity": identity, "run_signature": signatures[condition],
                                   "manifest_sha256": identity["manifest_sha256"],
                                   "taxonomy": identity["taxonomy"], "shard_index": shard,
                                   "num_shards": 2})
                protocol_paths[condition].append(path)
        for index, (sample_id, gt) in enumerate(truth_rows):
            pair = {"sample_id": sample_id, "gt": gt}
            for condition in ("original", "silent"):
                mod1, mod3 = (4, 5) if condition == "original" else (3, 7)
                t1 = gt["L1"] if index % mod1 else "unknown"
                leaf = gt["L3"] if t1 == gt["L1"] and index % mod3 else "unknown"
                pred = {"L1": t1, "L2": gt["L2"] if leaf != "unknown" else "unknown",
                        "L3": leaf}
                skipped = t1 == "unknown"
                raw_row = {"sample_id": sample_id, "gt": gt,
                           "turn1_pred": t1,
                           "turn1_parse_status": "invalid_format" if skipped else "valid_option",
                           "turn2_pred": {"L2": pred["L2"], "L3": leaf,
                                          "parse_status": "not_executed" if skipped else "valid_option"},
                           "cascade_skipped": skipped,
                           "run_signature": signatures[condition],
                           "shard_index": index % 2, "num_shards": 2}
                if with_manifests:
                    raw_row.update({"manifest_sha256": manifest_hashes[condition],
                                    "qa_prompt_version": "paper-v1",
                                    "turn1_prompt": "Question 1",
                                    "turn2_prompt": "" if skipped else "Question 2",
                                    "turn3_prompt": "" if skipped else "Question 3"})
                raw[condition].append(raw_row)
                pair[condition] = pred
                pair[f"{condition}_l3_correct"] = leaf == gt["L3"]
            paired.append(pair)
        paired_path = base / "paired_predictions.jsonl"
        _write_rows(paired_path, paired)
        for condition in ("original", "silent"):
            path = base / condition / "predictions.jsonl"
            _write_rows(path, raw[condition])
            prediction_paths[condition] = path
        comparison = {"status": "complete", "comparison": "original_minus_silent",
                      "original": {}, "silent": {}, "counts": {}, "delta": {},
                      "accuracy_delta_percentage_points": {}}
        for condition in ("original", "silent"):
            rows = raw[condition]
            comparison[condition] = {"integrity_verified": True,
                                     "run_signature": signatures[condition],
                                     "manifest_sha256": identities[condition]["manifest_sha256"],
                                     "total_samples": n,
                                     "l1_accuracy": sum(r["turn1_pred"] == r["gt"]["L1"] for r in rows) / n,
                                     "l3_accuracy": sum(r["turn2_pred"]["L3"] == r["gt"]["L3"] for r in rows) / n}
            comparison["counts"][condition] = {
                "samples": n,
                "turn2_and_turn3_executed": sum(not r["cascade_skipped"] for r in rows),
                "turn1_invalid": sum(r["turn1_parse_status"] == "invalid_format" for r in rows),
                "turn2_invalid_among_executed": 0}
        for metric in ("l1_accuracy", "l3_accuracy"):
            delta = comparison["original"][metric] - comparison["silent"][metric]
            comparison["delta"][metric] = delta
            comparison["accuracy_delta_percentage_points"][metric] = 100 * delta
        relocated = {}
        for path in [*prediction_paths.values(), *protocol_paths["original"], *protocol_paths["silent"]]:
            key = "/linux/eval/" + path.relative_to(base).as_posix()
            relocated[key] = path
        comparison["inputs"] = {key: _hash(path) for key, path in relocated.items()}
        comparison_path = base / "comparison.json"
        _write_json(comparison_path, comparison)
        yield comparison_path, relocated, paired_path, records, comparison


class PairedArchiveValidationTests(unittest.TestCase):
    def test_complete_2600_pair_evidence(self):
        with archive_fixture() as (comparison, relocated, paired, records, _):
            audit = validate_paired_archive(comparison, relocated, paired, records)
            self.assertEqual(audit["pairs_checked"], 2600)
            self.assertEqual(audit["input_hashes_checked"], 6)
            self.assertTrue(audit["paired_outputs_checked"])
            self.assertNotEqual(audit["conditions"]["original"]["manifest_sha256"],
                                audit["conditions"]["silent"]["manifest_sha256"])

    def test_rejects_changed_raw_input_and_changed_paired_prediction(self):
        with archive_fixture(12) as (comparison, relocated, paired, records, _):
            raw_path = next(path for key, path in relocated.items()
                            if "/original/predictions.jsonl" in key)
            raw = raw_path.read_bytes()
            raw_path.write_bytes(raw + b"\n")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                validate_paired_archive(comparison, relocated, paired, records)
            raw_path.write_bytes(raw)
            lines = paired.read_text(encoding="utf-8").splitlines()
            row = json.loads(lines[0]); row["original"]["L3"] = "changed"
            lines[0] = json.dumps(row)
            paired.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Paired original output differs"):
                validate_paired_archive(comparison, relocated, paired, records)

    def test_rejects_count_and_protocol_identity_drift(self):
        with archive_fixture(14) as (comparison_path, relocated, paired, records, comparison):
            bad = deepcopy(comparison)
            bad["counts"]["silent"]["turn1_invalid"] += 1
            _write_json(comparison_path, bad)
            with self.assertRaisesRegex(ValueError, "execution count differs"):
                validate_paired_archive(comparison_path, relocated, paired, records)
            _write_json(comparison_path, comparison)
            protocol_key = next(key for key in relocated if "/silent/protocol_shard_000.json" in key)
            protocol = relocated[protocol_key]
            value = json.loads(protocol.read_text(encoding="utf-8"))
            value["run_identity"]["model"]["model_id"] = "different model"
            _write_json(protocol, value)
            bad = deepcopy(comparison)
            bad["inputs"][protocol_key] = _hash(protocol)
            _write_json(comparison_path, bad)
            with self.assertRaisesRegex(ValueError, "(identity|signature)"):
                validate_paired_archive(comparison_path, relocated, paired, records)

    def test_manifest_binding_rejects_question_drift(self):
        with archive_fixture(12, with_manifests=True) as (
                comparison, relocated, paired, records, _):
            original_manifest = comparison.parent / "original" / "manifest.jsonl"
            silent_manifest = comparison.parent / "silent" / "manifest.jsonl"
            audit = validate_paired_archive(
                comparison, relocated, paired, records,
                original_manifest_path=original_manifest,
                silent_manifest_path=silent_manifest)
            self.assertTrue(audit["manifests_verified"])
            self.assertTrue(audit["manifest_non_audio_fields_match"])
            rows = [json.loads(line) for line in silent_manifest.read_text(encoding="utf-8").splitlines()]
            rows[0]["conversations"][1]["value"] = "A changed Turn 2 question"
            _write_rows(silent_manifest, rows)
            with self.assertRaisesRegex(ValueError, "Manifest non-audio fields differ"):
                validate_paired_archive(
                    comparison, relocated, paired, records,
                    original_manifest_path=original_manifest,
                    silent_manifest_path=silent_manifest)


class SilenceFigureLayoutTests(unittest.TestCase):
    def test_nine_models_use_separate_readable_scales(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualization.plot_results import build_silence_control

        comparisons = []
        for index in range(9):
            leaf_original = .025 + index * .012
            leaf_silent = .030 + index * .008
            routing_original = .20 + index * .07
            routing_silent = .15 + index * .055
            comparisons.append({"id": str(index), "label": f"Model {index + 1}",
                "metrics": {"l3_accuracy": {"original": leaf_original, "silent": leaf_silent,
                                             "delta_pp": 100 * (leaf_original - leaf_silent)},
                            "l1_accuracy": {"original": routing_original, "silent": routing_silent,
                                            "delta_pp": 100 * (routing_original - routing_silent)}}})
        data = {"style": {"width_mm": 180, "silence_height_mm": 85, "legend_frame": False},
                "comparisons": comparisons}
        fig, payload = build_silence_control(data)
        try:
            fig.canvas.draw()
            self.assertEqual(len(fig.axes), 2)
            self.assertLessEqual(fig.axes[0].get_xlim()[1], .20)
            self.assertEqual(fig.axes[1].get_xlim(), (0, 1))
            self.assertGreater(fig.axes[0].get_position().width,
                               fig.axes[1].get_position().width)
            self.assertEqual(len(fig.axes[0].get_yticklabels()), 9)
            self.assertEqual(len(fig.axes[1].get_yticklabels()), 0)
            self.assertEqual(payload["display_xlim"]["l1_accuracy"], [0, 1])
            self.assertAlmostEqual(fig.get_figheight() * 25.4, 85)
        finally:
            plt.close(fig)


class MetadataNineModelLayoutTests(unittest.TestCase):
    def test_nine_distinct_styles_and_legend_above_titles(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualization.plot_results import build_metadata_performance
        from visualization.project import paper_style

        model_ids = ("af_next", "gemma4_e2b", "gemma4_12b", "midashenglm",
                     "qwen25_omni", "qwen2_audio", "aero1_audio",
                     "voxtral_small", "voxtral_mini")
        runs = [{"id": model_id, "label": model_id.replace("_", " ").title(),
                 "metadata": {"groups": {
                     "active": {"macro_recall": [.04 + i * .006, .06 + i * .006, .07 + i * .006]},
                     "ship_noise": {"macro_recall": [.03 + i * .006, .05 + i * .006, .04 + i * .006]}}}}
                for i, model_id in enumerate(model_ids)]
        style = {"width_mm": 180, "metadata_height_mm": 85, "font_family": "Times New Roman",
                 "font_size_pt": 9, "legend_frame": False}
        data = {"style": style, "mode": "verified", "runs": runs,
                "strata": {"names": ["Low", "Middle", "High"], "assignments": {}}}
        with paper_style(style):
            fig, payload = build_metadata_performance(data)
            try:
                fig.canvas.draw()
                for key in ("color", "linestyle", "marker"):
                    self.assertEqual(len({str(value[key]) for value in payload["model_styles"].values()}), 9)
                renderer = fig.canvas.get_renderer()
                legend_bottom = fig.legends[0].get_window_extent(renderer).y0
                title_top = max(ax.title.get_window_extent(renderer).y1 for ax in fig.axes)
                self.assertGreater(legend_bottom, title_top + 2)
                self.assertAlmostEqual(fig.get_figheight() * 25.4, 85)
            finally:
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()

"""Exercise cache invalidation and the test/Eval scientific record binding."""
from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from visualization import project


_CLASSES = ("CW", "LFM", "HFM", "2FSK", "4FSK", "BPSK", "QPSK", "OFDM",
            "cargo", "cruise", "fishing", "warship", "underwater_target")


def _record_collections():
    """Two unique records per class; select one from each of all 13 classes."""
    full = []
    for index, leaf in enumerate(_CLASSES):
        family = "pulse" if index < 3 else "communication" if index < 8 else "ship_noise"
        for variant in range(2):
            frames = 16000 + 800 * index + variant
            source_id = f"{leaf.lower()}_{variant:03d}"
            full.append({
                "id": source_id + "_ch0", "source_id": source_id,
                "l1": "active" if index < 8 else "passive", "l2": family, "l3": leaf,
                "fs": 16000, "frames": frames, "duration_s": frames / 16000,
                "signal_frequency_hz": 500.0 + 100 * index if index < 8 else None,
                "channel_frequency_hz": 5500.0 if index < 8 else None,
                "G_h": -40.0 - index if index < 8 else None,
                "S_src": -10.0 - index / 10 if index >= 8 else None,
                "signal_params": {"fixture_index": index},
            })
    return full, copy.deepcopy(full[::2])


class CacheLoadTests(unittest.TestCase):
    """Use real config/proof/cache files without loading the large dataset."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="uab_figure_cache_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.cache = self.root / "cache"
        self.cache.mkdir()
        (self.root / "audio").mkdir()
        self.source_path = self.root / "full.jsonl"
        self.source_path.write_text("source-full\n", encoding="utf-8")
        (self.root / "eval.jsonl").write_text("source-eval\n", encoding="utf-8")
        (self.root / "headers.jsonl").write_text("source-headers\n", encoding="utf-8")
        self.config_path = self.root / "config.json"
        self.raw_config = {
            "full_manifest": "full.jsonl", "eval_manifest": "eval.jsonl",
            "eval_audio_root": "audio", "full_header_index": "headers.jsonl", "cache_dir": "cache",
            "expected_full_count": 26, "expected_eval_per_class": 1,
            "style": {"width_mm": 150, "font_family": "Times New Roman", "font_size_pt": 9,
                      "legend_frame": False},
        }
        self.config_path.write_text(json.dumps(self.raw_config), encoding="utf-8")
        config = copy.deepcopy(self.raw_config)
        for key in ("full_manifest", "eval_manifest", "eval_audio_root", "full_header_index", "cache_dir"):
            config[key] = str((self.root / config[key]).resolve())
        config["config_path"] = str(self.config_path)
        self.full, self.selected = _record_collections()
        data_files = {}
        for name, records in (("full", self.full), ("eval", self.selected)):
            path = self.cache / f"{name}.json.gz"
            content = gzip.compress(json.dumps(records).encode("utf-8"), mtime=0)
            path.write_bytes(content)
            data_files[name] = {"path": str(path), "sha256": hashlib.sha256(content).hexdigest()}
        signatures = []
        for name in ("full.jsonl", "eval.jsonl", "headers.jsonl"):
            path = self.root / name
            stat = path.stat()
            signatures.append({"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        self.proof = {
            "schema": project.SCHEMA, "config": config,
            "checks": {"full_records": 26, "eval_records": 13},
            "datasets": {"full": {"fixture": True}, "eval": {"fixture": True}},
            "data_files": data_files, "source_signatures": signatures,
            "cache_validation": "temporary fixture: SHA-bound derived records and source stat signatures",
        }
        self.proof_path = self.cache / "inputs.provenance.json"
        self.proof_path.write_text(json.dumps(self.proof), encoding="utf-8")
        environment = patch.dict(os.environ, {"UABENCH_FIGURE_CONFIG": str(self.config_path)})
        environment.start()
        self.addCleanup(environment.stop)

    def test_unchanged_cache_loads_without_writing_and_eval_only_is_supported(self):
        before = {p.relative_to(self.root): (p.stat().st_size, p.stat().st_mtime_ns)
                  for p in self.root.rglob("*") if p.is_file()}
        config, full, selected, proof = project.load_inputs()
        self.assertEqual(full, self.full)
        self.assertEqual(selected, self.selected)
        self.assertEqual(config["full_manifest"], str(self.source_path))
        self.assertEqual(proof["inputs_provenance_sha256"],
                         hashlib.sha256(self.proof_path.read_bytes()).hexdigest())
        _, omitted, selected_only, _ = project.load_inputs(include_full=False)
        self.assertEqual(omitted, [])
        self.assertEqual(selected_only, self.selected)
        after = {p.relative_to(self.root): (p.stat().st_size, p.stat().st_mtime_ns)
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_config_change_is_rejected_before_reusing_cached_data(self):
        self.raw_config["style"]["width_mm"] = 160
        self.config_path.write_text(json.dumps(self.raw_config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            project.load_inputs()

    def test_source_size_change_invalidates_cache(self):
        self.source_path.write_text("source-full\nnew-record\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Figure source changed"):
            project.load_inputs()

    def test_same_size_source_edit_invalidates_cache_by_mtime(self):
        stat = self.source_path.stat()
        self.source_path.write_bytes(self.source_path.read_bytes().upper())
        os.utime(self.source_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertEqual(self.source_path.stat().st_size, stat.st_size)
        with self.assertRaisesRegex(ValueError, "Figure source changed"):
            project.load_inputs(include_full=False)

    def test_derived_sha_catches_corruption_even_with_original_size_and_mtime(self):
        path = Path(self.proof["data_files"]["eval"]["path"])
        stat = path.stat()
        content = bytearray(path.read_bytes())
        content[-8] ^= 1
        path.write_bytes(content)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual((path.stat().st_size, path.stat().st_mtime_ns),
                         (stat.st_size, stat.st_mtime_ns))
        with self.assertRaisesRegex(ValueError, "Corrupt derived cache"):
            project.load_inputs(include_full=False)

    def test_full_cache_is_sha_checked_for_distribution_figure(self):
        path = Path(self.proof["data_files"]["full"]["path"])
        replacement = copy.deepcopy(self.full)
        replacement[0]["duration_s"] = 100
        path.write_bytes(gzip.compress(json.dumps(replacement).encode("utf-8"), mtime=0))
        with self.assertRaisesRegex(ValueError, "Corrupt derived cache"):
            project.load_inputs()


class DatasetBindingTests(unittest.TestCase):
    def setUp(self):
        self.full, self.selected = _record_collections()
        self.config = {"expected_full_count": 26, "expected_eval_per_class": 1}

    def test_all_13_classes_have_the_requested_support(self):
        checks = project.validate_datasets(self.full, self.selected, self.config)
        self.assertEqual(checks["full_records"], 26)
        self.assertEqual(checks["eval_records"], 13)
        self.assertEqual(checks["eval_class_counts"], {leaf: 1 for leaf in _CLASSES})
        self.assertTrue(checks["eval_is_subset_of_full"])

    def test_equal_total_count_does_not_hide_a_missing_class(self):
        selected = copy.deepcopy(self.selected)
        selected[-1] = copy.deepcopy(self.full[1])
        self.assertEqual(len(selected), 13)
        self.assertEqual(len({r["id"] for r in selected}), 13)
        with self.assertRaisesRegex(ValueError, "each of 13 classes"):
            project.validate_datasets(self.full, selected, self.config)

    def test_shared_id_does_not_override_changed_measurements_or_source_identity(self):
        changes = {
            "source_id": "different_source", "duration_s": 2.5,
            "frames": 32000, "signal_frequency_hz": 700.0,
            "channel_frequency_hz": 6000.0, "G_h": -99.0,
            "S_src": -12.5, "signal_params": {"fixture_index": 100},
        }
        for field, value in changes.items():
            with self.subTest(field=field):
                selected = copy.deepcopy(self.selected)
                selected[0][field] = value
                with self.assertRaisesRegex(ValueError, f"Eval/test disagreement.*{field}"):
                    project.validate_datasets(self.full, selected, self.config)

    def test_eval_record_must_belong_to_the_full_test_partition(self):
        selected = copy.deepcopy(self.selected)
        selected[0]["id"] = "outside_the_test_partition_ch0"
        with self.assertRaisesRegex(ValueError, "Eval ID absent from test"):
            project.validate_datasets(self.full, selected, self.config)

    def test_duplicate_ids_cannot_inflate_full_or_eval_support(self):
        for full, selected in ((self.full + [self.full[0]], self.selected),
                               (self.full, self.selected + [self.selected[0]])):
            with self.subTest(full_count=len(full), eval_count=len(selected)):
                with self.assertRaisesRegex(ValueError, "Duplicate record IDs"):
                    project.validate_datasets(full, selected, self.config)

    def test_matching_wrong_sample_rate_still_fails_the_eval_input_contract(self):
        self.full[0]["fs"] = self.selected[0]["fs"] = 8000
        with self.assertRaisesRegex(ValueError, "16-kHz Eval input"):
            project.validate_datasets(self.full, self.selected, self.config)

    def test_unexpected_full_test_count_is_not_silently_accepted(self):
        self.config["expected_full_count"] = 25
        with self.assertRaisesRegex(ValueError, "Expected 25 full-test records; got 26"):
            project.validate_datasets(self.full, self.selected, self.config)


if __name__ == "__main__":
    unittest.main()

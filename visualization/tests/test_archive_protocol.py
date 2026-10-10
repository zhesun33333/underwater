"""Independent checks for importing saved, versioned model runs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from testsite.core.protocol import IMPLEMENTATION_FILES, ROOT as TESTSITE_ROOT, identity_signature
from visualization.archive_protocol import validate_archived_run


def _write_lines(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")


class ArchiveProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.archived_manifest = base / "archived.jsonl"
        self.current_manifest = base / "current.jsonl"
        archived, current = [], []
        for i in range(4):
            row = {"id": f"sample_{i}", "audio": f"/remote/audio/{i}.wav",
                   "_gt": {"L1": "active", "L2": "pulse", "L3": "CW"},
                   "qa_prompt_version": "test_v1",
                   "conversations": [{"from": "human", "value": f"Question {turn} for {i}"}
                                     for turn in (1, 2, 3)],
                   "metadata": {"sample_index": i}}
            archived.append(row)
            current.append({**row, "audio": f"audio/{i}.wav"})
        _write_lines(self.archived_manifest, archived)
        _write_lines(self.current_manifest, current)
        manifest_hash = hashlib.sha256(self.archived_manifest.read_bytes()).hexdigest()
        implementation = {}
        for name in IMPLEMENTATION_FILES:
            source = (TESTSITE_ROOT / name).read_bytes().replace(b"\r\n", b"\n")
            implementation["testsite/" + name] = hashlib.sha256(source).hexdigest()
        # A real archived inference backend may differ while the scorer agrees.
        implementation["testsite/core/inference.py"] = "0" * 64
        self.identity = {"run_id": "archived_test", "manifest_sha256": manifest_hash,
                         "model": {"backend": "mock", "model_id": "test"},
                         "taxonomy": {"L1": {"name": "first"},
                                      "L2": {"name": "second"},
                                      "L3": {"name": "third"}},
                         "implementation": implementation}
        signature = identity_signature(self.identity)
        self.protocol_paths = [base / f"protocol_shard_{i:03}.json" for i in range(2)]
        for index, path in enumerate(self.protocol_paths):
            path.write_text(json.dumps({"run_identity": self.identity,
                                        "manifest_sha256": manifest_hash,
                                        "taxonomy": self.identity["taxonomy"],
                                        "run_signature": signature,
                                        "shard_index": index, "num_shards": 2}), encoding="utf-8")
        self.rows = [{"sample_id": record["id"], "gt": record["_gt"],
                      "manifest_sha256": manifest_hash, "run_signature": signature,
                      "shard_index": i % 2, "num_shards": 2,
                      "turn1_pred": "active", "cascade_skipped": False,
                      "turn1_prompt": record["conversations"][0]["value"],
                      "turn2_prompt": record["conversations"][1]["value"],
                      "turn3_prompt": record["conversations"][2]["value"],
                      "qa_prompt_version": "test_v1"}
                     for i, record in enumerate(archived)]

    def _validate(self, rows=None, paths=None):
        return validate_archived_run(self.rows if rows is None else rows,
                                     self.protocol_paths if paths is None else paths,
                                     self.archived_manifest, self.current_manifest)

    def test_exact_archived_manifest_and_complete_protocol_with_code_audit(self):
        result = self._validate()
        self.assertEqual(result["identity"], self.identity)
        audit = result["audit"]
        self.assertEqual(audit["sample_count"], 4)
        self.assertEqual(audit["protocol_shards"], 2)
        self.assertEqual(audit["audio_path_differences"], 4)
        self.assertTrue(audit["non_audio_manifest_fields_equal"])
        self.assertTrue(audit["implementation"]["inference_py_mismatch_after_lf"])
        for name, item in audit["implementation"]["files"].items():
            if name != "testsite/core/inference.py":
                self.assertTrue(item["lf_normalized_match"], name)
        self.assertFalse(audit["implementation"]["lf_normalized_match_all"])

    def test_rejects_manifest_drift_even_when_sample_ids_still_match(self):
        rows = [json.loads(line) for line in self.current_manifest.read_text().splitlines()]
        rows[0]["conversations"][1]["value"] = "Different question"
        _write_lines(self.current_manifest, rows)
        with self.assertRaisesRegex(ValueError, "beyond audio paths"):
            self._validate()
        rows[0]["conversations"][1]["value"] = "Question 2 for 0"
        _write_lines(self.current_manifest, rows)
        altered = deepcopy(self.rows)
        altered[0]["turn1_prompt"] = "Different archived prompt"
        with self.assertRaisesRegex(ValueError, "Turn 1 question"):
            self._validate(rows=altered)

    def test_rejects_missing_or_inconsistent_shards_and_signatures(self):
        with self.assertRaisesRegex(ValueError, "shard count is incomplete"):
            self._validate(paths=self.protocol_paths[:1])
        second = json.loads(self.protocol_paths[1].read_text())
        second["shard_index"] = 0
        self.protocol_paths[1].write_text(json.dumps(second), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "index is invalid or duplicated"):
            self._validate()
        second["shard_index"] = 1
        second["run_signature"] = "f" * 64
        self.protocol_paths[1].write_text(json.dumps(second), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "signature mismatch"):
            self._validate()
        second["run_signature"] = self.rows[0]["run_signature"]
        self.protocol_paths[1].write_text(json.dumps(second), encoding="utf-8")
        rows = deepcopy(self.rows)
        rows[0]["num_shards"] = 3
        with self.assertRaisesRegex(ValueError, "shard counts differ"):
            self._validate(rows=rows)


if __name__ == "__main__":
    unittest.main()

"""Result archive importer tests: nested files, hashes, and safe paths."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from visualization.import_result_archive import import_archive


def _add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _add_link(archive: tarfile.TarFile, name: str, target: str, link_type: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.type = link_type
    member.linkname = target
    archive.addfile(member)


class ImportResultArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def test_nested_json_only_import_has_exact_hashes_and_member_identity(self):
        original = b'{"total_samples": 2}\n'
        predictions = b'{"sample_id":"one"}\n{"sample_id":"two"}\n'
        protocol = b'{"shard_index":0,"num_shards":1}\n'
        nested_bytes = io.BytesIO()
        with tarfile.open(fileobj=nested_bytes, mode="w:gz") as nested:
            _add_bytes(nested, "run/predictions.jsonl", predictions)
            _add_bytes(nested, "run/protocol.json", protocol)
        source = self.base / "results.tar.gz"
        with tarfile.open(source, mode="w:gz") as outer:
            _add_bytes(outer, "model/metrics.json", original)
            _add_bytes(outer, "instance1.tar.gz", nested_bytes.getvalue())
            _add_bytes(outer, "script.py", b"raise RuntimeError('must not import')\n")

        output = self.base / "imported"
        summary = import_archive(source, output)
        index = json.loads((output / "_extraction_manifest.json").read_text(encoding="utf-8"))
        expected = {"model/metrics.json": original,
                    "instance1/run/predictions.jsonl": predictions,
                    "instance1/run/protocol.json": protocol}
        self.assertEqual(summary["files"], len(expected))
        self.assertEqual(summary["nested_archives"], 1)
        self.assertEqual(summary["extracted_bytes"], sum(map(len, expected.values())))
        self.assertEqual(index["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual({entry["member"] for entry in index["entries"]}, set(expected))
        for entry in index["entries"]:
            payload = expected[entry["member"]]
            target = Path(entry["path"])
            self.assertTrue(target.resolve().is_relative_to(output.resolve()))
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(entry["bytes"], len(payload))
            self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertFalse((output / "script.py").exists())

    def test_rejects_traversal_in_outer_and_nested_archive(self):
        for index, bad_name in enumerate(("../escape.json", "/escape.json", "C:/escape.json",
                                          "folder\\escape.json", "../ignored.txt")):
            source = self.base / f"bad_{index}.tar.gz"
            with tarfile.open(source, mode="w:gz") as archive:
                _add_bytes(archive, bad_name, b"{}")
            with self.subTest(member=bad_name), self.assertRaisesRegex(ValueError, "Unsafe tar member path"):
                import_archive(source, self.base / f"out_{index}")
        nested_bytes = io.BytesIO()
        with tarfile.open(fileobj=nested_bytes, mode="w:gz") as nested:
            _add_bytes(nested, "../../escape.json", b"{}")
        source = self.base / "nested_bad.tar.gz"
        with tarfile.open(source, mode="w:gz") as outer:
            _add_bytes(outer, "inner.tar.gz", nested_bytes.getvalue())
        with self.assertRaisesRegex(ValueError, "Unsafe tar member path"):
            import_archive(source, self.base / "nested_out")
        self.assertFalse((self.base / "escape.json").exists())

    def test_symlinks_and_hardlinks_are_never_materialized(self):
        source = self.base / "links.tar.gz"
        with tarfile.open(source, mode="w:gz") as archive:
            _add_link(archive, "symlink.json", "../outside.json", tarfile.SYMTYPE)
            _add_link(archive, "hardlink.json", "../outside.json", tarfile.LNKTYPE)
            _add_bytes(archive, "safe.json", b"{}")
        output = self.base / "links_out"
        summary = import_archive(source, output)
        self.assertEqual(summary["files"], 1)
        self.assertEqual((output / "safe.json").read_bytes(), b"{}")
        self.assertFalse((output / "symlink.json").exists())
        self.assertFalse((output / "hardlink.json").exists())
        self.assertFalse((self.base / "outside.json").exists())

    def test_duplicate_result_member_name_is_rejected(self):
        source = self.base / "duplicate.tar.gz"
        with tarfile.open(source, mode="w:gz") as archive:
            _add_bytes(archive, "run/metrics.json", b'{"n":1}')
            _add_bytes(archive, "run/metrics.json", b'{"n":2}')
        with self.assertRaisesRegex(ValueError, "Duplicate result member path"):
            import_archive(source, self.base / "duplicate_out")


if __name__ == "__main__":
    unittest.main()

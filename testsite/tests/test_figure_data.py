import copy
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import wave

from testsite.reporting.figure_data import (
    load_figure_records, register_audited_header_index, robust_medoid_index,
)
from testsite.reporting.build_header_index import build_header_index


class FigureDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.jsonl"
        self.audio_rel = "PulseCom/wav_by_type/BPSK/bpsk_000001_ch0.wav"
        self.audio = self.root / "audio" / self.audio_rel
        self.audio.parent.mkdir(parents=True)
        with wave.open(str(self.audio), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(b"\0\0" * 4000)
        self.row = {
            "id": "bpsk_000001_ch0", "audio": self.audio_rel,
            "_gt": {"L1": "active", "L2": "communication", "L3": "BPSK"},
            "_meta": {"audio_duration_s": 30, "source_id": "bpsk_000001",
                      "signal_params": {"carrier_freq_hz": 2000},
                      "bellhop_env": {"freq_hz": 5500}, "tl_db": -70},
        }
        self.write_manifest([self.row])

    def write_manifest(self, rows):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def write_legacy_meta(self, meta):
        root = self.root / "metadata"
        path = root / "PulseCom/jsonc/BPSK/bpsk_000001_ch0.jsonc"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta), encoding="utf-8")
        return root

    def make_archive_index(self):
        archive = self.root / "warship_ok"  # Extensionless archives are allowed.
        with tarfile.open(archive, "w") as target:
            target.add(self.audio, arcname=self.audio_rel)
        index = self.root / "headers.jsonl"
        build_header_index(self.manifest, index, archives=[archive])
        return archive, index

    def test_actual_duration_and_source_frequency_are_authoritative(self):
        rows, provenance = load_figure_records(self.manifest, self.root / "audio")
        row = rows[0]
        self.assertEqual(row["duration_s"], .25)
        self.assertEqual((row["fs"], row["frames"]), (16000, 4000))
        self.assertEqual(row["signal_frequency_hz"], 2000)
        self.assertEqual(row["channel_frequency_hz"], 5500)
        self.assertEqual((row["source_id"], row["channel_id"]), ("bpsk_000001", 0))
        self.assertFalse(row["validation"]["duration_matches_metadata"])
        self.assertEqual(provenance["field_difference_counts"]["metadata_duration_vs_wav"], 1)
        self.assertEqual(provenance["field_difference_counts"]["signal_vs_channel_frequency"], 1)
        self.assertEqual(provenance["missing_counts"]["G_h"], 0)

    def test_signal_frequency_conflict_uses_class_specific_field(self):
        self.row["_meta"]["signal_params"]["center_freq_hz"] = 1500
        self.write_manifest([self.row])
        rows, _ = load_figure_records(self.manifest, self.root / "audio")
        self.assertEqual(rows[0]["signal_frequency_hz"], 2000)
        self.assertEqual(rows[0]["validation"]["signal_frequency_conflicts"], ["center_freq_hz"])
        self.assertEqual(rows[0]["field_sources"]["signal_frequency_hz"], "_meta.signal_params.carrier_freq_hz")

    def test_explicit_legacy_metadata_fallback(self):
        legacy = {"id": self.row["id"], "audio": self.audio_rel, "conversations": []}
        self.write_manifest([legacy])
        metadata_root = self.write_legacy_meta({"signal_type": "BPSK", "signal_params": {"carrier_freq_hz": 1900},
                                               "bellhop_output": {"tl_db": -65}, "audio_duration_s": 30})
        rows, _ = load_figure_records(self.manifest, self.root / "audio", metadata_root=metadata_root)
        self.assertEqual(rows[0]["l3"], "BPSK")
        self.assertEqual(rows[0]["signal_frequency_hz"], 1900)
        self.assertEqual(rows[0]["G_h"], -65)
        self.assertEqual(rows[0]["duration_s"], .25)
        self.assertTrue(rows[0]["field_sources"]["signal_frequency_hz"].startswith("external:"))

    def test_embedded_wins_and_external_conflicts_remain_visible(self):
        metadata_root = self.write_legacy_meta({"signal_type": "BPSK", "signal_params": {"carrier_freq_hz": 1900},
                                               "tl_db": -60})
        rows, provenance = load_figure_records(self.manifest, self.root / "audio", metadata_root=metadata_root)
        self.assertEqual(rows[0]["signal_frequency_hz"], 2000)
        self.assertEqual(rows[0]["G_h"], -70)
        self.assertIn("signal_params.carrier_freq_hz", rows[0]["validation"]["metadata_conflicts"])
        self.assertEqual(provenance["field_difference_counts"]["embedded_vs_external:tl_db"], 1)

    def test_unknown_and_inconsistent_labels_rejected(self):
        for gt in ({"L3": "unknown"}, {"L1": "passive", "L2": "communication", "L3": "BPSK"}):
            changed = copy.deepcopy(self.row)
            changed["_gt"] = gt
            self.write_manifest([changed])
            with self.assertRaises(ValueError):
                load_figure_records(self.manifest, self.root / "audio")

    def test_duplicate_manifest_rejected(self):
        self.write_manifest([self.row, self.row])
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            load_figure_records(self.manifest, self.root / "audio")

    def test_missing_wav_does_not_fall_back_to_metadata_duration(self):
        with self.assertRaisesRegex(FileNotFoundError, "missing WAV"):
            load_figure_records(self.manifest)

    def test_archive_index_round_trip_and_audio_header_crosscheck(self):
        _, index = self.make_archive_index()
        rows, provenance = load_figure_records(self.manifest, header_index=index)
        self.assertEqual(rows[0]["duration_s"], .25)
        self.assertIsNone(rows[0]["audio_path"])
        self.assertEqual(rows[0]["audio_locator"]["kind"], "tar")
        self.assertFalse(provenance["header_index"]["audio_bytes_rehashed_this_load"])
        rows, _ = load_figure_records(self.manifest, self.root / "audio", index)
        self.assertEqual(rows[0]["validation"]["header_source"], "wav_header")
        with wave.open(str(self.audio), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(b"\0\0" * 5000)
        with self.assertRaisesRegex(ValueError, "actual WAV header differs"):
            load_figure_records(self.manifest, self.root / "audio", index)

    def test_changed_archive_requires_rebuild_not_reregistration(self):
        archive, index = self.make_archive_index()
        with archive.open("ab") as source:
            source.write(b"changed")
        with self.assertRaisesRegex(ValueError, "source signature changed"):
            load_figure_records(self.manifest, header_index=index)

    def test_changed_index_and_manifest_fail_sha_bindings(self):
        _, index = self.make_archive_index()
        original = index.read_text(encoding="utf-8")
        index.write_text(original + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            load_figure_records(self.manifest, header_index=index)
        index.write_text(original, encoding="utf-8")
        self.row["changed"] = True
        self.write_manifest([self.row])
        with self.assertRaisesRegex(ValueError, "different manifest SHA-256"):
            load_figure_records(self.manifest, header_index=index)

    def test_header_registration_rejects_duplicates_and_bad_durations(self):
        _, index = self.make_archive_index()
        original = index.read_text(encoding="utf-8")
        index.write_text(original + original, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate id"):
            register_audited_header_index(self.manifest, index)
        header = json.loads(original)
        header["duration"] = 30
        index.write_text(json.dumps(header), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duration does not equal frames/fs"):
            register_audited_header_index(self.manifest, index)
        header["duration"] = .25
        header["id"] = "other_id"
        index.write_text(json.dumps(header), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing header"):
            register_audited_header_index(self.manifest, index)

    def test_unsigned_header_index_is_rejected(self):
        _, index = self.make_archive_index()
        Path(str(index) + ".provenance.json").unlink()
        with self.assertRaisesRegex(ValueError, "provenance sidecar"):
            load_figure_records(self.manifest, header_index=index)

    def test_builder_rejects_missing_or_ambiguous_audio(self):
        output = self.root / "incomplete.jsonl"
        with self.assertRaisesRegex(ValueError, "missing 1 WAV"):
            build_header_index(self.manifest, output, audio_root=self.root / "missing")
        self.assertFalse(output.exists())
        archive, _ = self.make_archive_index()
        another = self.root / "copy.tar"
        another.write_bytes(archive.read_bytes())
        with self.assertRaisesRegex(ValueError, "duplicate archived WAV"):
            build_header_index(self.manifest, output, archives=[archive, another])

    def test_zero_iqr_dimension_is_omitted_even_with_outlier(self):
        # The rare outlier must not influence distance when that dimension's
        # IQR is zero. The all-tied result is determined only by stable IDs.
        self.assertEqual(robust_medoid_index([[0], [0], [0], [0], [100]], ["z", "y", "x", "w", "a"]), 4)
        self.assertEqual(robust_medoid_index([[1, 10], [1, 20], [1, 30]], ["z", "a", "b"]), 1)

    def test_legacy_plot_loads_embedded_metadata_and_draws_step_ecdf(self):
        import matplotlib.pyplot as plt
        from testsite.reporting.plot_dataset_quality import load_samples, _ecdf
        samples = load_samples(self.manifest, self.root / "audio", None)
        self.assertEqual(samples[0].center_frequency_hz, 2000)
        self.assertEqual(samples[0].duration_s, .25)
        fig, ax = plt.subplots()
        try:
            _ecdf(ax, [1, 1, 2], "black")
            self.assertEqual(ax.lines[0].get_drawstyle(), "steps-post")
            self.assertEqual(tuple(ax.get_ylim()), (0, 1))
            self.assertEqual(ax.lines[0].get_ydata()[0], 0)
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()

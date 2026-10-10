"""Prepare replaceable legacy, local-verified, or archive-verified D-G results."""
from __future__ import annotations

import glob
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import zipfile

from .project import ROOT, _write_gzip, _write_json, file_signature, load_inputs, sha256
from .result_analysis import (CLASS_ORDER, analyze_predictions, canonical_predictions,
                              diagnostics_from_metrics, make_strata, metadata_performance,
                              validate_comparison)


def load_results_config(path: Path | None = None) -> dict:
    path = Path(path or os.environ.get("UABENCH_RESULTS_CONFIG", ROOT / "results.local.json")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Copy a results.*.example.json config to {path}")
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if value.get("mode") not in ("legacy_preview", "verified", "archived_verified"):
        raise ValueError("Results mode must explicitly be legacy_preview, verified, or archived_verified")
    for key in ("archive", "archive_index", "cache_dir"):
        if value.get(key):
            candidate = Path(value[key])
            value[key] = str((path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    value.setdefault("cache_dir", str(path.parent / "cache/results"))
    value["config_path"] = str(path)
    runs = value.get("runs", [])
    ids = [run["id"] for run in runs]
    if len(ids) != len(set(ids)) or any(not isinstance(s, str) or not s for s in ids):
        raise ValueError("Run IDs must be unique nonempty strings")
    paired_ids = [item["id"] for item in value.get("comparisons", [])]
    if len(paired_ids) != len(set(paired_ids)) or any(not isinstance(s, str) or not s for s in paired_ids):
        raise ValueError("Comparison IDs must be unique nonempty strings")
    if runs and value.get("diagnostic_model") not in ids:
        raise ValueError("diagnostic_model must explicitly select one configured run ID")
    if value["mode"] != "legacy_preview" and value.get("archive"):
        raise ValueError("Versioned runs use local prediction/protocol files; legacy ZIP is preview-only")
    if value["mode"] == "archived_verified" and not value.get("archive_index"):
        raise ValueError("archived_verified requires an extraction index binding files to the source archive")
    if value["mode"] == "archived_verified" and value.get("comparisons") and not value.get("silent_manifest"):
        raise ValueError("Archived comparisons require the exact silent manifest")
    if not isinstance(value.get("input_relocations", {}), dict):
        raise ValueError("input_relocations must map original paths to local paths")
    return value


class SourceReader:
    """Read exact local files or legacy ZIP members without running source code."""
    def __init__(self, config):
        self.base = Path(config["config_path"]).parent
        self.archive = Path(config["archive"]) if config.get("archive") else None
        self.zip = zipfile.ZipFile(self.archive) if self.archive else None
        self.inputs = {}
        self.paths = {self.archive} if self.archive else set()
        self.archive_sha256 = sha256(self.archive) if self.archive else None

    def close(self):
        if self.zip:
            self.zip.close()

    def path(self, locator):
        path = Path(locator)
        return (self.base / path).resolve() if not path.is_absolute() else path.resolve()

    def expand(self, locators):
        result = []
        for locator in locators:
            if self.zip:
                if any(token in locator for token in ("*", "?", "[")):
                    raise ValueError("ZIP inputs require an exact member suffix, not a wildcard")
                result.append(locator)
            else:
                matches = sorted(glob.glob(str(self.path(locator))))
                if not matches:
                    raise FileNotFoundError(f"No files match {locator}")
                result.extend(matches)
        return result

    def read(self, locator):
        if self.zip:
            suffix = str(locator).replace("\\", "/").lstrip("/")
            matches = [n for n in self.zip.namelist() if n == suffix or n.endswith("/"+suffix)]
            if len(matches) != 1:
                raise ValueError(f"ZIP suffix must match exactly one member: {suffix}, matches={len(matches)}")
            member = matches[0]
            if self.zip.getinfo(member).file_size > 128 * 1024 * 1024:
                raise ValueError("Prediction member exceeds 128 MiB; use a local verified run instead")
            content = self.zip.read(member)
            key = str(self.archive)+"::"+member
        else:
            path = self.path(locator)
            content = path.read_bytes()
            self.paths.add(path)
            key = str(path)
        self.inputs[key] = {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        return content.decode("utf-8-sig")

    def json(self, locator):
        return json.loads(self.read(locator))

    def rows(self, locators):
        return [json.loads(line) for loc in self.expand(locators)
                for line in self.read(loc).splitlines() if line.strip()]


def _check_archived_metrics(metrics: dict, identity: dict, rows: list[dict]) -> None:
    """Cross-check saved L1/L2/L3 metrics against the imported raw predictions."""
    from testsite.core.protocol import identity_signature

    if metrics.get("integrity_verified") is not True:
        raise ValueError("Archived metrics must be marked integrity_verified")
    if metrics.get("manifest_sha256") != identity["manifest_sha256"]:
        raise ValueError("Archived metrics manifest differs from its signed protocol")
    if metrics.get("run_signature") != identity_signature(identity):
        raise ValueError("Archived metrics run signature differs from its protocol")
    if metrics.get("total_samples") != len(rows):
        raise ValueError("Archived metrics sample count differs from predictions")
    for key, predicted in (("l1_accuracy", lambda row: row["turn1_pred"]),
                           ("l2_accuracy", lambda row: row["turn2_pred"]["L2"]),
                           ("l3_accuracy", lambda row: row["turn2_pred"]["L3"])):
        level = key[:2].upper()
        observed = sum(predicted(row) == row["gt"][level] for row in rows) / len(rows)
        if not math.isclose(observed, metrics[key], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Archived {key} differs from predictions")


def _bind_comparison_inputs(reader: SourceReader, value: dict, relocations: dict) -> dict:
    """Keep the original comparison JSON and bind each recorded hash to local bytes."""
    inputs = value.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("comparison.json must retain nonempty input hashes")
    if not isinstance(relocations, dict) or set(relocations) - set(inputs):
        raise ValueError("Comparison input_relocations has unknown keys")
    locations = {}
    for recorded_path, digest in inputs.items():
        if not isinstance(recorded_path, str) or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Comparison input path or SHA-256 is invalid")
        locator = relocations.get(recorded_path, recorded_path)
        if recorded_path not in relocations and not Path(recorded_path).is_absolute():
            raise ValueError(f"Paired comparison input needs an explicit relocation: {recorded_path}")
        if not isinstance(locator, str):
            raise ValueError("Comparison input relocation must be a path string")
        source = reader.path(locator)
        if not source.is_file() or sha256(source) != digest:
            raise ValueError(f"Paired comparison input is missing or changed: {recorded_path}")
        reader.paths.add(source)
        reader.inputs[str(source)] = {"bytes": source.stat().st_size, "sha256": digest}
        locations[recorded_path] = str(source)
    return locations


def _verify_archive_index(reader: SourceReader, index_locator: str) -> dict:
    """Tie every consumed imported file to the hash-checked source tar member."""
    index_path = reader.path(index_locator)
    index = reader.json(index_locator)
    if index.get("schema") != 1 or not isinstance(index.get("entries"), list):
        raise ValueError("Invalid result-archive extraction index")
    root = index_path.parent.resolve()
    indexed = {}
    for entry in index["entries"]:
        path = (root / entry["member"]).resolve()
        if not path.is_relative_to(root) or path in indexed:
            raise ValueError("Extraction index has unsafe or duplicate member paths")
        indexed[path] = entry
    checked = 0
    for name, observed in reader.inputs.items():
        source = Path(name)
        if source == index_path or not source.is_relative_to(root):
            continue
        entry = indexed.get(source)
        if not entry or entry["bytes"] != observed["bytes"] or entry["sha256"] != observed["sha256"]:
            raise ValueError(f"Imported input differs from extraction index: {source}")
        checked += 1
    archive = Path(index["source_archive"])
    if not archive.is_file() or sha256(archive) != index["source_sha256"]:
        raise ValueError("Source result archive differs from extraction index")
    reader.paths.add(archive)
    reader.inputs[str(archive)] = {"bytes": archive.stat().st_size, "sha256": index["source_sha256"]}
    return {"source_archive": str(archive), "source_sha256": index["source_sha256"],
            "indexed_files_checked": checked, "extracted_files": index["files"]}


def prepare_results(path: Path | None = None) -> dict:
    config = load_results_config(path)
    data_config, _, records, data_proof = load_inputs(include_full=False)
    strict = config["mode"] != "legacy_preview"
    archived = config["mode"] == "archived_verified"
    strata = make_strata(records)
    expected_support = [sum(r["l3"] == leaf for r in records) for leaf in CLASS_ORDER]
    reader = SourceReader(config)
    runs, comparisons = [], []
    try:
        for entry in config.get("runs", []):
            print(f"Preparing {entry['label']} ({config['mode']}) ...", flush=True)
            raw_rows = reader.rows(entry["predictions"]) if entry.get("predictions") else None
            identity = None
            archived_audit = None
            if strict:
                if not raw_rows or not entry.get("protocols"):
                    raise ValueError(f"{entry['id']}: verified mode requires predictions and all protocol shards")
                protocol_paths = reader.expand(entry["protocols"])
                for protocol in protocol_paths:
                    reader.read(protocol)
                if archived:
                    if not entry.get("metrics"):
                        raise ValueError(f"{entry['id']}: archived_verified requires saved metrics")
                    from .archive_protocol import validate_archived_run
                    manifest = reader.path(entry.get("manifest", data_config["eval_manifest"]))
                    reader.read(str(manifest))
                    checked = validate_archived_run(raw_rows, protocol_paths, manifest,
                                                    Path(data_config["eval_manifest"]))
                    identity, archived_audit = checked["identity"], checked["audit"]
                    metrics = reader.json(entry["metrics"])
                    _check_archived_metrics(metrics, identity, raw_rows)
                else:
                    from testsite.scripts.merge_predictions import aggregate
                    metrics, identity = aggregate(raw_rows, data_config["eval_manifest"], protocol_paths)
                reader.paths.update(ROOT.parent / p for p in identity["implementation"])
            elif entry.get("metrics"):
                metrics = reader.json(entry["metrics"])
            elif not raw_rows:
                raise ValueError(f"{entry['id']}: no prediction or metrics input")
            else:
                metrics = None
            saved = diagnostics_from_metrics(metrics, entry.get("label_aliases")) if metrics else None
            if raw_rows:
                rows = canonical_predictions(raw_rows, records, strict=strict)
                diagnostic, outcomes = analyze_predictions(rows, strict=strict)
                if saved and saved["counts"] != diagnostic["counts"]:
                    raise ValueError(f"{entry['id']}: saved confusion matrix differs from complete predictions")
                metadata = metadata_performance(rows, strata)
                metadata_differences = 0
                by_id = {r["id"]: r for r in records}
                for row in raw_rows:
                    target = by_id[row["sample_id"]]
                    source, key = ("snr_db", "S_src") if target["l2"] == "ship_noise" else ("tl_db", "G_h")
                    legacy_value = row.get("metadata", {}).get(source)
                    if legacy_value is not None and legacy_value != target[key]:
                        metadata_differences += 1
                capability = "D/E/F"
            else:
                diagnostic, outcomes, metadata, rows = saved, None, None, None
                metadata_differences = None
                capability = "D only; no sample-level IDs or execution states"
            if diagnostic["support"] != expected_support:
                raise ValueError(f"{entry['id']}: class support differs from audited Eval")
            runs.append({"id": entry["id"], "label": entry["label"], "mode": config["mode"],
                         "diagnostics": diagnostic, "outcomes": outcomes, "metadata": metadata,
                         "rows": rows, "identity": identity,
                         "audit": {"capabilities": capability, "n": diagnostic["n"],
                                   "id_and_gt_bound_to_eval": raw_rows is not None,
                                   "protocol_verified": strict and not archived,
                                   "protocol_verified_against_local_code": strict and not archived,
                                   "archived_protocol_audit": archived_audit,
                                   "legacy_metadata_differences": metadata_differences,
                                   "metadata_source": "audited Eval metadata",
                                   "format_validity": "explicit statuses" if strict else "unavailable; no raw-output reparsing"}})
        global_relocations = config.get("input_relocations", {})
        used_global_relocations = set()
        silent_manifest = None
        if archived and config.get("comparisons"):
            silent_manifest = reader.path(config["silent_manifest"])
            reader.read(config["silent_manifest"])
        for entry in config.get("comparisons", []):
            if reader.zip:
                raise ValueError("G reads verified comparison.json files, not the legacy ZIP")
            value = reader.json(entry["path"])
            result = validate_comparison(value)
            relocations = {key: global_relocations[key] for key in value.get("inputs", {})
                           if key in global_relocations}
            used_global_relocations.update(relocations)
            specific = entry.get("input_relocations", {})
            if not isinstance(specific, dict) or any(key in relocations and relocations[key] != path
                                                      for key, path in specific.items()):
                raise ValueError("Conflicting comparison input relocations")
            relocations.update(specific)
            locations = _bind_comparison_inputs(reader, value, relocations)
            match = next((run for run in runs if run["id"] == entry["id"]), None)
            if archived and match is None:
                raise ValueError(f"{entry['id']}: archived G comparison has no matching D/E/F original run")
            if match:
                from testsite.core.protocol import identity_signature
                if result["run_signatures"]["original"] != identity_signature(match["identity"]):
                    raise ValueError(f"{entry['id']}: G original condition differs from the D/E/F run")
            paired_audit = None
            if archived and not entry.get("paired_predictions"):
                raise ValueError(f"{entry['id']}: archived comparison requires paired_predictions")
            if entry.get("paired_predictions"):
                from .paired_archive_validation import validate_paired_archive
                paired_path = reader.path(entry["paired_predictions"])
                reader.read(entry["paired_predictions"])
                manifests = {}
                if archived:
                    manifests = {"original_manifest_path": Path(match["audit"]["archived_protocol_audit"]["archived_manifest_path"]),
                                 "silent_manifest_path": silent_manifest}
                paired_audit = validate_paired_archive(reader.path(entry["path"]), locations,
                                                       paired_path, records, **manifests)
            comparisons.append({"id": entry["id"], "label": entry["label"], **result,
                                "source_comparison": str(reader.path(entry["path"])),
                                "input_relocations": locations, "paired_audit": paired_audit})
        if set(global_relocations) != used_global_relocations:
            raise ValueError("Unused global input_relocations keys")
        archive_audit = _verify_archive_index(reader, config["archive_index"]) if archived else None
        content = {"schema": 1, "mode": config["mode"], "diagnostic_model": config.get("diagnostic_model"),
                   "style": data_config["style"], "runs": runs, "strata": strata, "comparisons": comparisons}
        if reader.zip:
            inventory = {"files": sum(not i.is_dir() for i in reader.zip.infolist()),
                         "python_files": sum(n.endswith('.py') for n in reader.zip.namelist()),
                         "protocol_files": sum('protocol' in n.lower() for n in reader.zip.namelist()),
                         "archive_sha256": reader.archive_sha256}
        else:
            inventory = None
        summary_runs = []
        for run in runs:
            source_audit = run["audit"]["archived_protocol_audit"]
            summary_runs.append({"id": run["id"], "label": run["label"],
                                 "capabilities": run["audit"]["capabilities"],
                                 "n": run["audit"]["n"],
                                 "protocol_verified_against_local_code": run["audit"]["protocol_verified"],
                                 "archived_manifest_sha256": source_audit["archived_manifest_sha256"] if source_audit else None,
                                 "non_audio_manifest_fields_equal": source_audit["non_audio_manifest_fields_equal"] if source_audit else None,
                                 "implementation_lf_match_all": source_audit["implementation"]["lf_normalized_match_all"] if source_audit else None,
                                 "inference_py_mismatch_after_lf": source_audit["implementation"]["inference_py_mismatch_after_lf"] if source_audit else None})
        summary = {"mode": config["mode"], "archive_inventory": inventory,
                   "archive_audit": archive_audit, "runs": summary_runs,
                   "D_model": config.get("diagnostic_model"),
                   "E_states": 5 if strict else 3,
                   "F_partition": {k: v for k, v in strata.items() if k != "assignments"},
                   "G_comparisons": len(comparisons),
                   "legacy_note": None if strict else "Preview of historical labels/metrics; not current-protocol benchmark results"}
        paths = reader.paths | {Path(config["config_path"]), Path(data_config["config_path"]),
                               Path(data_proof["inputs_provenance_path"]), Path(__file__),
                               ROOT / "result_analysis.py"}
        if archived:
            paths.update({ROOT / "archive_protocol.py", ROOT / "paired_archive_validation.py"})
        cache = Path(config["cache_dir"])
        cache.mkdir(parents=True, exist_ok=True)
        output = cache / "results.json.gz"
        _write_gzip(output, content)
        proof = {"schema": 1, "config": config, "inputs": reader.inputs, "inventory": inventory,
                 "data_inputs": data_proof, "summary": summary,
                 "data": {"path": str(output), "sha256": sha256(output)},
                 "source_signatures": [file_signature(p) for p in sorted(paths)]}
        _write_json(cache / "results.provenance.json", proof)
        _write_json(cache / "results.audit.json", summary)
        return summary
    finally:
        reader.close()


def load_results() -> tuple[dict, dict]:
    config = load_results_config()
    proof_path = Path(config["cache_dir"]) / "results.provenance.json"
    if not proof_path.is_file():
        raise FileNotFoundError("Run python -m visualization prepare-results before rendering D/E/F/G")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof["schema"] != 1 or proof["config"] != config:
        raise ValueError("Results configuration changed; rerun prepare-results")
    for expected in proof["source_signatures"]:
        if file_signature(Path(expected["path"])) != expected:
            raise ValueError(f"Results source changed: {expected['path']}; rerun prepare-results")
    _, _, _, data_proof = load_inputs(include_full=False)
    if data_proof != proof["data_inputs"]:
        raise ValueError("Audited Eval data changed; rerun prepare-results")
    path = Path(proof["data"]["path"])
    if sha256(path) != proof["data"]["sha256"]:
        raise ValueError("Results cache hash mismatch; rerun prepare-results")
    with gzip.open(path, "rt", encoding="utf-8") as source:
        content = json.load(source)
    provenance = {k: proof[k] for k in ("schema", "config", "inputs", "inventory", "data_inputs", "summary")}
    provenance["results_provenance_sha256"] = sha256(proof_path)
    return content, provenance

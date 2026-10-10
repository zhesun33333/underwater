"""Cross-check archived silent controls against their relocated, hashed inputs.

This intentionally checks the saved evaluation protocol and predictions without
pretending to rerun model inference or to rehash unavailable control WAVs.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path):
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc


def _expected(eval_records: list[dict]) -> dict[str, dict]:
    result = {}
    for record in eval_records:
        sample_id = record["id"]
        if sample_id in result:
            raise ValueError(f"Duplicate audited Eval ID: {sample_id}")
        result[sample_id] = {key: record[key.lower()] for key in ("L1", "L2", "L3")}
    if not result:
        raise ValueError("Audited Eval records are empty")
    return result


def _raw_predictions(path: Path, expected: dict[str, dict],
                     manifests_by_signature: Mapping[str, Path] | None = None) -> tuple[dict, dict]:
    rows = {}
    signatures = set()
    full_rows = [] if manifests_by_signature is not None else None
    counts = {"samples": 0, "turn2_and_turn3_executed": 0,
              "turn1_invalid": 0, "turn2_invalid_among_executed": 0}
    l1_correct = l3_correct = 0
    for row in _rows(path):
        if full_rows is not None:
            full_rows.append(row)
        sample_id = row["sample_id"]
        if sample_id in rows:
            raise ValueError(f"Duplicate raw prediction ID: {sample_id}")
        if sample_id not in expected or row["gt"] != expected[sample_id]:
            raise ValueError(f"Raw prediction ID/GT differs from audited Eval: {sample_id}")
        predictions = {"L1": row["turn1_pred"], "L2": row["turn2_pred"]["L2"],
                       "L3": row["turn2_pred"]["L3"]}
        rows[sample_id] = {"gt": row["gt"], "predictions": predictions,
                           "shard_index": row["shard_index"],
                           "num_shards": row["num_shards"]}
        signatures.add(row["run_signature"])
        l1_correct += predictions["L1"] == row["gt"]["L1"]
        l3_correct += predictions["L3"] == row["gt"]["L3"]
        counts["samples"] += 1
        counts["turn1_invalid"] += row.get("turn1_parse_status") == "invalid_format"
        if not row["cascade_skipped"]:
            counts["turn2_and_turn3_executed"] += 1
            counts["turn2_invalid_among_executed"] += (
                row["turn2_pred"].get("parse_status") == "invalid_format")
    if set(rows) != set(expected):
        raise ValueError(f"Raw prediction ID coverage differs from audited Eval: {path}")
    if len(signatures) != 1:
        raise ValueError(f"Raw predictions contain mixed run signatures: {path}")
    signature = signatures.pop()
    verified_manifest_sha256 = None
    if full_rows is not None:
        from testsite.core.integrity import validate_predictions

        manifest_path = manifests_by_signature.get(signature)
        if manifest_path is None:
            raise ValueError(f"No manifest corresponds to raw prediction run signature: {path}")
        verified_manifest_sha256 = validate_predictions(full_rows, manifest_path)
    n = len(rows)
    return rows, {"run_signature": signature, "counts": counts,
                  "verified_manifest_sha256": verified_manifest_sha256,
                  "l1_accuracy": l1_correct / n, "l3_accuracy": l3_correct / n}


def _protocol_identity(paths: list[Path], signature: str, rows: dict) -> dict:
    if len(paths) != 2:
        raise ValueError("Each archived condition requires two protocol shards")
    protocols = [json.loads(path.read_text(encoding="utf-8-sig")) for path in paths]
    identity = protocols[0].get("run_identity")
    if not isinstance(identity, dict):
        raise ValueError("Protocol run identity is missing")
    expected_signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    if signature != expected_signature:
        raise ValueError("Raw prediction run signature differs from protocol identity")
    for protocol in protocols:
        if protocol.get("run_identity") != identity:
            raise ValueError("Protocol shards disagree on run identity")
        if protocol.get("run_signature") != signature:
            raise ValueError("Protocol run signature differs from raw predictions")
        if protocol.get("manifest_sha256") != identity.get("manifest_sha256"):
            raise ValueError("Protocol manifest hash differs from run identity")
        if protocol.get("taxonomy") != identity.get("taxonomy"):
            raise ValueError("Protocol taxonomy differs from run identity")
        if protocol.get("num_shards") != 2:
            raise ValueError("Expected two protocol shards")
    if {protocol.get("shard_index") for protocol in protocols} != {0, 1}:
        raise ValueError("Protocol shard coverage is incomplete or duplicated")
    if any(row["shard_index"] not in (0, 1) or row["num_shards"] != 2
           for row in rows.values()):
        raise ValueError("Raw predictions disagree with protocol shard layout")
    return identity


def _check_paired(path: Path, original: dict, silent: dict, expected: dict) -> int:
    seen = set()
    for row in _rows(path):
        sample_id = row["sample_id"]
        if sample_id in seen:
            raise ValueError(f"Duplicate paired prediction ID: {sample_id}")
        seen.add(sample_id)
        if sample_id not in expected or row["gt"] != expected[sample_id]:
            raise ValueError(f"Paired prediction ID/GT differs from audited Eval: {sample_id}")
        for condition, raw in (("original", original), ("silent", silent)):
            if row[condition] != raw[sample_id]["predictions"]:
                raise ValueError(f"Paired {condition} output differs from raw prediction: {sample_id}")
            correct = raw[sample_id]["predictions"]["L3"] == expected[sample_id]["L3"]
            if row[f"{condition}_l3_correct"] is not correct:
                raise ValueError(f"Paired {condition} correctness differs from raw prediction: {sample_id}")
    if seen != set(expected):
        raise ValueError("Paired prediction ID coverage differs from audited Eval")
    return len(seen)


def _check_manifest_pair(original_path: Path, silent_path: Path, expected: dict[str, dict]) -> int:
    from testsite.core.integrity import read_manifest

    original = {row["id"]: row for row in read_manifest(original_path)}
    silent = {row["id"]: row for row in read_manifest(silent_path)}
    if set(original) != set(expected) or set(silent) != set(expected):
        raise ValueError("Original and silent manifest IDs differ from audited Eval")
    for sample_id in expected:
        a, b = original[sample_id], silent[sample_id]
        if a.get("_gt") != expected[sample_id] or b.get("_gt") != expected[sample_id]:
            raise ValueError(f"Manifest GT differs from audited Eval: {sample_id}")
        if {k: v for k, v in a.items() if k != "audio"} != {
                k: v for k, v in b.items() if k != "audio"}:
            raise ValueError(f"Manifest non-audio fields differ between conditions: {sample_id}")
    return len(expected)


def validate_paired_archive(comparison_path: Path, relocated_inputs: Mapping[str, Path],
                            paired_predictions_path: Path, eval_records: list[dict], *,
                            original_manifest_path: Path | None = None,
                            silent_manifest_path: Path | None = None) -> dict:
    """Validate one saved comparison against six hashed inputs and paired rows.

    ``relocated_inputs`` maps each original path key in ``comparison.json`` to a
    local file. This permits archived Linux paths to be verified on Windows
    without modifying the source comparison or weakening its SHA-256 evidence.
    When both manifest paths are supplied, the original evaluation integrity
    checker also verifies each raw prediction stream against its own manifest.
    """
    comparison_path = Path(comparison_path)
    paired_predictions_path = Path(paired_predictions_path)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8-sig"))
    if comparison.get("status") != "complete" or comparison.get("comparison") != "original_minus_silent":
        raise ValueError("Expected a complete original-minus-silent comparison")
    inputs = comparison.get("inputs")
    if not isinstance(inputs, dict) or len(inputs) != 6 or set(relocated_inputs) != set(inputs):
        raise ValueError("Comparison requires exactly six relocated hashed inputs")
    prediction_paths, protocol_paths = [], []
    for source, expected_hash in inputs.items():
        path = Path(relocated_inputs[source]).resolve(strict=True)
        if _sha256(path) != expected_hash:
            raise ValueError(f"Relocated input SHA-256 mismatch: {source}")
        name = PurePosixPath(source.replace("\\", "/")).name
        if name == "predictions.jsonl":
            prediction_paths.append(path)
        elif name.startswith("protocol_shard_") and name.endswith(".json"):
            protocol_paths.append(path)
        else:
            raise ValueError(f"Unexpected comparison input kind: {source}")
    if len(prediction_paths) != 2 or len(protocol_paths) != 4:
        raise ValueError("Comparison must reference two merged predictions and four protocol shards")
    expected = _expected(eval_records)
    if (original_manifest_path is None) != (silent_manifest_path is None):
        raise ValueError("Both original and silent manifest paths are required together")
    manifests = None
    if original_manifest_path is not None:
        paths = {"original": Path(original_manifest_path).resolve(strict=True),
                 "silent": Path(silent_manifest_path).resolve(strict=True)}
        _check_manifest_pair(paths["original"], paths["silent"], expected)
        for condition, path in paths.items():
            if _sha256(path) != comparison[condition].get("manifest_sha256"):
                raise ValueError(f"{condition} manifest SHA-256 differs from comparison")
        manifests = {comparison[condition]["run_signature"]: path
                     for condition, path in paths.items()}
        if len(manifests) != 2:
            raise ValueError("Original and silent comparisons have the same run signature")
    runs = {}
    for path in prediction_paths:
        rows, summary = _raw_predictions(path, expected, manifests)
        signature = summary["run_signature"]
        if signature in runs:
            raise ValueError("Original and silent predictions have the same run signature")
        runs[signature] = {"rows": rows, "summary": summary}
    condition_runs = {}
    for condition in ("original", "silent"):
        saved = comparison[condition]
        if saved.get("integrity_verified") is not True:
            raise ValueError(f"{condition} comparison aggregate is not integrity verified")
        signature = saved.get("run_signature")
        if signature not in runs:
            raise ValueError(f"{condition} comparison run signature differs from raw predictions")
        condition_runs[condition] = runs.pop(signature)
    if runs:
        raise ValueError("Comparison does not identify both raw prediction runs")
    protocols_by_signature = {key: [] for key in (
        condition_runs["original"]["summary"]["run_signature"],
        condition_runs["silent"]["summary"]["run_signature"])}
    for path in protocol_paths:
        protocol = json.loads(path.read_text(encoding="utf-8-sig"))
        signature = protocol.get("run_signature")
        if signature not in protocols_by_signature:
            raise ValueError("Protocol run signature does not match comparison runs")
        protocols_by_signature[signature].append(path)
    identities = {}
    for condition, run in condition_runs.items():
        signature = run["summary"]["run_signature"]
        identities[condition] = _protocol_identity(
            protocols_by_signature[signature], signature, run["rows"])
        if comparison[condition].get("manifest_sha256") != identities[condition]["manifest_sha256"]:
            raise ValueError(f"{condition} aggregate manifest hash differs from protocol")
        if (manifests is not None and
                run["summary"]["verified_manifest_sha256"] != identities[condition]["manifest_sha256"]):
            raise ValueError(f"{condition} validated manifest hash differs from protocol")
        if comparison[condition].get("total_samples") != len(expected):
            raise ValueError(f"{condition} aggregate sample count differs from audited Eval")
        for key, count in run["summary"]["counts"].items():
            if comparison["counts"][condition].get(key) != count:
                raise ValueError(f"{condition} execution count differs from raw predictions: {key}")
        for metric in ("l1_accuracy", "l3_accuracy"):
            value = run["summary"][metric]
            saved = comparison[condition].get(metric)
            if not isinstance(saved, (int, float)) or not math.isclose(value, saved, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"{condition} {metric} differs from raw predictions")
    for key in ("model", "taxonomy", "implementation"):
        if identities["original"].get(key) != identities["silent"].get(key):
            raise ValueError(f"Original and silent protocol {key} differ")
    if identities["original"]["manifest_sha256"] == identities["silent"]["manifest_sha256"]:
        raise ValueError("Original and silent protocols use the same manifest")
    n = _check_paired(paired_predictions_path,
                      condition_runs["original"]["rows"],
                      condition_runs["silent"]["rows"], expected)
    for metric in ("l1_accuracy", "l3_accuracy"):
        delta = (condition_runs["original"]["summary"][metric] -
                 condition_runs["silent"]["summary"][metric])
        if not math.isclose(delta, comparison["delta"][metric], rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Paired {metric} delta differs from raw predictions")
        saved_pp = comparison.get("accuracy_delta_percentage_points", {}).get(metric)
        if saved_pp is not None and not math.isclose(100 * delta, saved_pp, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Paired {metric} percentage-point delta differs from raw predictions")
    return {"schema": 1, "comparison_sha256": _sha256(comparison_path),
            "paired_predictions_sha256": _sha256(paired_predictions_path),
            "input_hashes_checked": len(inputs), "pairs_checked": n,
            "eval_id_gt_checked": True, "paired_outputs_checked": True,
            "counts_and_accuracy_checked": True,
            "manifests_verified": manifests is not None,
            "manifest_non_audio_fields_match": manifests is not None,
            "conditions": {condition: {"run_signature": run["summary"]["run_signature"],
                                       "manifest_sha256": identities[condition]["manifest_sha256"],
                                       "l1_accuracy": run["summary"]["l1_accuracy"],
                                       "l3_accuracy": run["summary"]["l3_accuracy"]}
                           for condition, run in condition_runs.items()}}

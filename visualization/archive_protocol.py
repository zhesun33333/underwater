"""Validate saved evaluation protocols without pretending to rerun inference.

Archived runs can have a different absolute audio path and a different saved
inference backend version from the code in this checkout.  This module checks
the exact archived manifest and run signatures, then reports the local code
hash comparison separately.  It never weakens ``validate_protocols`` for
new runs produced with the current checkout.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from testsite.core.integrity import file_sha256, read_manifest, validate_predictions
from testsite.core.protocol import IMPLEMENTATION_FILES, ROOT as TESTSITE_ROOT, identity_signature


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _implementation_audit(recorded: dict[str, str]) -> dict:
    expected = {"testsite/" + name for name in IMPLEMENTATION_FILES}
    missing = sorted(expected - recorded.keys())
    if missing:
        raise ValueError(f"Archived implementation is missing protected source hashes: {missing}")
    if any(not isinstance(name, str) or not isinstance(digest, str)
           or _SHA256.fullmatch(digest) is None for name, digest in recorded.items()):
        raise ValueError("Archived implementation contains an invalid source hash")

    files = {}
    for name in sorted(expected):
        source = TESTSITE_ROOT / name.removeprefix("testsite/")
        raw = source.read_bytes()
        raw_hash = hashlib.sha256(raw).hexdigest()
        lf_hash = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
        saved = recorded[name]
        files[name] = {"recorded_sha256": saved, "local_raw_sha256": raw_hash,
                       "local_lf_sha256": lf_hash, "raw_match": saved == raw_hash,
                       "lf_normalized_match": saved == lf_hash}
    inference = files["testsite/core/inference.py"]
    return {"files": files, "extra_recorded_files": sorted(recorded.keys() - expected),
            "raw_match_all": all(item["raw_match"] for item in files.values()),
            "lf_normalized_match_all": all(item["lf_normalized_match"] for item in files.values()),
            "inference_py_mismatch_after_lf": not inference["lf_normalized_match"]}


def validate_archived_run(rows: list[dict], protocol_paths: list[Path],
                          archived_manifest_path: Path,
                          current_eval_manifest_path: Path) -> dict:
    """Return archived identity and an explicit, limited verification audit.

    The saved manifest is authenticated by its SHA-256 in every prediction and
    protocol.  The current Eval manifest is used only to confirm identical IDs,
    labels, questions, and metadata; the recorded audio path may differ.
    """
    archived_manifest_path = Path(archived_manifest_path).resolve()
    current_eval_manifest_path = Path(current_eval_manifest_path).resolve()
    protocol_paths = [Path(path).resolve() for path in protocol_paths]
    if not protocol_paths or len(set(protocol_paths)) != len(protocol_paths):
        raise ValueError("Archived run requires unique, complete protocol shard paths")

    archived_hash = validate_predictions(rows, archived_manifest_path)
    archived = {row["id"]: row for row in read_manifest(archived_manifest_path)}
    current = {row["id"]: row for row in read_manifest(current_eval_manifest_path)}
    if set(archived) != set(current):
        raise ValueError("Archived and current Eval manifests have different sample IDs")
    changed = [sid for sid in sorted(archived)
               if {k: v for k, v in archived[sid].items() if k != "audio"}
               != {k: v for k, v in current[sid].items() if k != "audio"}]
    if changed:
        raise ValueError(f"Archived and current Eval manifests differ beyond audio paths: {changed[:5]}")

    protocols = [json.loads(path.read_text(encoding="utf-8-sig")) for path in protocol_paths]
    if any(not isinstance(protocol, dict) for protocol in protocols):
        raise ValueError("Archived protocol shard must be a JSON object")
    identity = protocols[0].get("run_identity")
    if (not isinstance(identity, dict) or not isinstance(identity.get("run_id"), str)
            or not identity["run_id"]):
        raise ValueError("Archived protocol lacks a run identity")
    if not isinstance(identity.get("model"), dict) or not identity["model"]:
        raise ValueError("Archived protocol lacks the model configuration")
    taxonomy = identity.get("taxonomy")
    if not isinstance(taxonomy, dict) or not all(taxonomy.get(key) for key in ("L1", "L2", "L3")):
        raise ValueError("Archived protocol lacks the evaluation taxonomy")
    implementation = identity.get("implementation")
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("Archived protocol lacks implementation hashes")
    if identity.get("manifest_sha256") != archived_hash:
        raise ValueError("Archived run identity does not match its exact manifest")
    signature = identity_signature(identity)
    count, shards = protocols[0].get("num_shards"), set()
    if type(count) is not int or count <= 0 or count != len(protocols):
        raise ValueError("Archived protocol shard count is incomplete")
    for protocol in protocols:
        if protocol.get("run_identity") != identity:
            raise ValueError("Archived protocol shards have inconsistent run identities")
        if protocol.get("manifest_sha256") != archived_hash:
            raise ValueError("Archived protocol shard manifest mismatch")
        if protocol.get("taxonomy") != taxonomy:
            raise ValueError("Archived protocol shard taxonomy mismatch")
        if protocol.get("run_signature") != signature:
            raise ValueError("Archived protocol shard run signature mismatch")
        index = protocol.get("shard_index")
        if type(index) is not int or not 0 <= index < count or index in shards:
            raise ValueError("Archived protocol shard index is invalid or duplicated")
        if type(protocol.get("num_shards")) is not int or protocol["num_shards"] != count:
            raise ValueError("Archived protocol shard count differs between shards")
        shards.add(index)
    if shards != set(range(count)):
        raise ValueError("Archived protocol shard coverage is incomplete")
    for row in rows:
        if row.get("run_signature") != signature or row.get("manifest_sha256") != archived_hash:
            raise ValueError("Prediction run signature or manifest differs from archived protocol")
        if type(row.get("shard_index")) is not int or row["shard_index"] not in shards:
            raise ValueError("Prediction has invalid archived protocol shard index")
        if type(row.get("num_shards")) is not int or row["num_shards"] != count:
            raise ValueError("Prediction and archived protocol shard counts differ")

    audit = {"scope": "exact archived manifest and protocol; local code comparison reported separately",
             "archived_manifest_path": str(archived_manifest_path),
             "archived_manifest_sha256": archived_hash,
             "current_eval_manifest_path": str(current_eval_manifest_path),
             "current_eval_manifest_sha256": file_sha256(current_eval_manifest_path),
             "sample_count": len(rows), "protocol_shards": count,
             "audio_path_differences": sum(archived[sid]["audio"] != current[sid]["audio"]
                                           for sid in archived),
             "non_audio_manifest_fields_equal": True,
             "implementation": _implementation_audit(implementation)}
    return {"identity": identity, "audit": audit}

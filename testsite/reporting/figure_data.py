"""Validated, read-only inputs for publication figures.

Durations always come from WAV frames, either read now or from an explicitly
registered, source-bound WAV-header index. Legacy metadata never supplies them.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import wave
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

CLASS_ORDER = ["CW", "LFM", "HFM", "2FSK", "4FSK", "BPSK", "QPSK", "OFDM",
               "cargo", "cruise", "fishing", "warship", "underwater_target"]
CLASS_TO_FAMILY = {key: "pulse" if i < 3 else "communication" if i < 8 else "ship_noise"
                   for i, key in enumerate(CLASS_ORDER)}
HEADER_SCHEMA = "ua_bench_header_index_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def normalize_l3(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = {"cargo_ship": "cargo", "cruise_ship": "cruise", "fishing_boat": "fishing",
            "naval_vessel": "warship", "underwater target": "underwater_target"}.get(text, text)
    return text.upper() if text.upper() in CLASS_ORDER[:8] else text


def read_jsonl(path: Path):
    seen = set()
    with Path(path).open(encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
                raise ValueError(f"{path}:{line_number}: expected an object with a nonempty string id")
            if row["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {row['id']}")
            seen.add(row["id"])
            yield row
    if not seen:
        raise ValueError(f"empty JSONL: {path}")


def extract_signal_frequency(meta: dict, l3: str | None = None) -> tuple[float | None, str | None]:
    """Return a source-signal frequency, never the BELLHOP solver frequency."""
    params = meta.get("signal_params") or {}
    family = CLASS_TO_FAMILY.get(normalize_l3(l3 or meta.get("L3") or meta.get("signal_type")))
    center = ("center_freq_hz", "center_frequency_hz")
    carrier = ("carrier_freq_hz", "carrier_frequency_hz")
    keys = carrier + center if family == "communication" else center + carrier
    for key in keys:
        value = number(params.get(key))
        if value is not None and value > 0:
            return value, f"signal_params.{key}"
    # An occupied-band midpoint is explicit source metadata; the broad sampling
    # subband and bellhop_env.freq_hz are not a substitute for signal frequency.
    low, high = number(params.get("band_low_hz")), number(params.get("band_high_hz"))
    if low is not None and high is not None and 0 <= low < high:
        return (low + high) / 2, "signal_params.band_midpoint_hz"
    return None, None


def candidate_metadata_paths(root: Path, audio_rel: str, sample_id: str) -> list[Path]:
    parts = PurePosixPath(audio_rel.replace("\\", "/")).parts
    candidates = []
    if "PulseCom" in parts and len(parts) >= 2:
        candidates.append(root / "PulseCom" / "jsonc" / parts[-2] / f"{sample_id}.jsonc")
    if "05_ship_radiated_noise" in parts and len(parts) >= 3:
        candidates.append(root / "05_ship_radiated_noise" / parts[-3] / "json" / f"{sample_id}.json")
    return candidates


def _merged_metadata(record: dict, metadata_root: Path | None) -> tuple[dict, dict, list, str | None]:
    embedded = record.get("_meta") or {}
    if not isinstance(embedded, dict):
        raise ValueError(f"{record['id']}: _meta must be an object")
    external, external_path = {}, None
    if metadata_root is not None:
        for path in candidate_metadata_paths(metadata_root, str(record.get("audio", "")), record["id"]):
            if path.is_file():
                external = json.loads(path.read_text(encoding="utf-8-sig"))
                if not isinstance(external, dict):
                    raise ValueError(f"external metadata must be an object: {path}")
                external_path = str(path.resolve())
                break
    origins, conflicts = {}, []

    def merge(current, fallback, prefix=""):
        result = {}
        for key in sorted(current.keys() | fallback.keys()):
            field = f"{prefix}.{key}" if prefix else key
            preferred, old = current.get(key), fallback.get(key)
            if isinstance(preferred, dict) or (preferred is None and isinstance(old, dict)):
                result[key] = merge(preferred or {}, old if isinstance(old, dict) else {}, field)
            else:
                result[key] = preferred if preferred is not None else old
                origins[field] = f"_meta.{field}" if preferred is not None else f"external:{external_path}#{field}"
                if preferred is not None and old is not None and preferred != old:
                    conflicts.append(field)
        return result
    return merge(embedded, external), origins, conflicts, external_path


def _labels(record: dict, meta: dict) -> tuple[str, str, str]:
    gt = record.get("_gt") or {}
    if not isinstance(gt, dict):
        raise ValueError(f"{record['id']}: _gt must be an object")
    explicit = gt.get("L3", gt.get("l3"))
    if explicit is None:
        explicit = meta.get("L3") or meta.get("sub_type") or meta.get("signal_type")
    l3 = normalize_l3(explicit)
    if l3 not in CLASS_TO_FAMILY:
        raise ValueError(f"{record['id']}: unknown or missing L3 label {explicit!r}")
    l2 = CLASS_TO_FAMILY[l3]
    l1 = "passive" if l2 == "ship_noise" else "active"
    for key, expected in (("L1", l1), ("L2", l2)):
        supplied = gt.get(key, gt.get(key.lower(), meta.get(key)))
        if supplied is not None and str(supplied).lower() != expected:
            raise ValueError(f"{record['id']}: inconsistent {key}={supplied!r} for {l3}")
    return l1, l2, l3


def read_wav_header(path_or_file) -> dict:
    with wave.open(str(path_or_file) if isinstance(path_or_file, (str, Path)) else path_or_file, "rb") as wav:
        if wav.getcomptype() != "NONE":
            raise ValueError("compressed WAV is not supported")
        result = dict(fs=wav.getframerate(), frames=wav.getnframes(), channels=wav.getnchannels(), width=wav.getsampwidth())
    if any(value <= 0 for value in result.values()):
        raise ValueError("invalid or empty WAV header")
    result["duration"] = result["frames"] / result["fs"]
    return result


def source_signature(kind: str, path: Path) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"header source is not a file: {path}")
    return {"kind": kind, "path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def header_provenance_path(index: Path) -> Path:
    return Path(str(index) + ".provenance.json")


def _read_header_index(index: Path) -> tuple[dict, list[tuple[str, Path]]]:
    records, sources = {}, set()
    for row in read_jsonl(index):
        rid = row["id"]
        l3 = normalize_l3(row.get("L3", row.get("l3")))
        if l3 not in CLASS_TO_FAMILY:
            raise ValueError(f"{rid}: unknown header-index L3 label")
        l2 = row.get("L2", row.get("l2"))
        if l2 != CLASS_TO_FAMILY[l3]:
            raise ValueError(f"{rid}: inconsistent header-index L2 label")
        for key in ("fs", "frames", "channels", "width"):
            if isinstance(row.get(key), bool) or not isinstance(row.get(key), int) or row[key] <= 0:
                raise ValueError(f"{rid}: header-index {key} must be a positive integer")
        duration = number(row.get("duration", row.get("duration_s")))
        if duration is None or not math.isclose(duration, row["frames"] / row["fs"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{rid}: header-index duration does not equal frames/fs")
        row = dict(row, L3=l3, L2=l2, duration=duration)
        if row.get("archive") and row.get("member"):
            member = PurePosixPath(str(row["member"]).replace("\\", "/"))
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"{rid}: unsafe archive member locator")
            archive = Path(row["archive"]).resolve()
            sources.add(("tar", archive))
            row["locator"] = {"kind": "tar", "archive": str(archive), "member": str(row["member"])}
            if "offset_data" in row:
                row["locator"]["offset_data"] = row["offset_data"]
        elif row.get("audio_path"):
            path = Path(row["audio_path"]).resolve()
            sources.add(("wav", path))
            row["locator"] = {"kind": "wav", "path": str(path)}
        else:
            raise ValueError(f"{rid}: header index lacks an audio locator")
        records[rid] = row
    return records, sorted(sources, key=lambda item: (item[0], str(item[1])))


def register_audited_header_index(manifest: Path, header_index: Path,
                                  verification: str = "imported_audited_wav_headers") -> Path:
    """Bind an independently audited index to current files; does not re-audit WAVs.

    Call this explicitly only after auditing the index's source WAVs. The
    sidecar preserves that distinction and prevents silent use after changes.
    """
    indexed, sources = _read_header_index(header_index)
    manifest_ids = set()
    for raw in read_jsonl(manifest):
        l1, l2, l3 = _labels(raw, raw.get("_meta") or {})
        manifest_ids.add(raw["id"])
        header = indexed.get(raw["id"])
        if header is None:
            raise ValueError(f"missing header for {raw['id']}")
        if (l2, l3) != (header["L2"], header["L3"]):
            raise ValueError(f"header label mismatch for {raw['id']}")
    payload = {"schema_version": HEADER_SCHEMA, "manifest_sha256": file_sha256(manifest),
               "index_sha256": file_sha256(header_index), "verification": verification,
               "record_count": len(indexed), "manifest_record_count": len(manifest_ids),
               "sources": [source_signature(kind, path) for kind, path in sources]}
    path = header_provenance_path(header_index)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _validated_header_index(manifest: Path, index: Path, manifest_hash: str) -> tuple[dict, dict]:
    sidecar = header_provenance_path(index)
    if not sidecar.is_file():
        raise ValueError(f"header index requires a source-bound provenance sidecar: {sidecar}; rebuild it or explicitly register an independently audited index")
    provenance = json.loads(sidecar.read_text(encoding="utf-8-sig"))
    if provenance.get("schema_version") != HEADER_SCHEMA:
        raise ValueError("unknown header-index provenance schema")
    if provenance.get("manifest_sha256") != manifest_hash:
        raise ValueError("header index was built or registered for a different manifest SHA-256")
    digest = file_sha256(index)
    if provenance.get("index_sha256") != digest:
        raise ValueError("header-index SHA-256 differs from its provenance sidecar")
    records, sources = _read_header_index(index)
    signatures = [source_signature(kind, path) for kind, path in sources]
    if signatures != provenance.get("sources"):
        raise ValueError("WAV/header archive source signature changed; rebuild the header index")
    if provenance.get("record_count") != len(records):
        raise ValueError("header-index record count differs from provenance")
    return records, {"path": str(index.resolve()), "sha256": digest,
                     "provenance_path": str(sidecar.resolve()), "verification": provenance.get("verification"),
                     "source_signatures_verified": True, "audio_bytes_rehashed_this_load": False,
                     "sources": signatures}


def load_figure_records(manifest: Path, audio_root: Path | None = None,
                        header_index: Path | None = None,
                        metadata_root: Path | None = None) -> tuple[list[dict], dict]:
    manifest = Path(manifest).resolve()
    root = Path(audio_root).resolve() if audio_root is not None else manifest.parent
    metadata_root = Path(metadata_root).resolve() if metadata_root is not None else None
    manifest_hash = file_sha256(manifest)
    indexed, index_provenance = ({}, None)
    if header_index is not None:
        indexed, index_provenance = _validated_header_index(manifest, Path(header_index), manifest_hash)
    result, differences, missing, field_counts = [], Counter(), Counter(), Counter()
    for raw in read_jsonl(manifest):
        rid = raw["id"]
        meta, origins, conflicts, external_path = _merged_metadata(raw, metadata_root)
        l1, l2, l3 = _labels(raw, meta)
        audio = raw.get("audio")
        if not isinstance(audio, str) or not audio.strip():
            raise ValueError(f"{rid}: missing audio path in manifest")
        candidate = Path(audio) if Path(audio).is_absolute() else root / audio
        header = indexed.get(rid)
        if header_index is not None and header is None:
            raise ValueError(f"{rid}: missing header-index record")
        if header is not None and (header["L2"], header["L3"]) != (l2, l3):
            raise ValueError(f"{rid}: header-index label disagrees with manifest")
        if candidate.is_file():
            actual = read_wav_header(candidate)
            if header is not None and any(actual[k] != header[k] for k in ("fs", "frames", "channels", "width")):
                raise ValueError(f"{rid}: actual WAV header differs from header index")
            header = actual
            audio_path = str(candidate.resolve())
            locator = {"kind": "wav", "path": audio_path}
            header_source = "wav_header"
        elif header is not None:
            audio_path = header["locator"].get("path")
            locator = header["locator"]
            header_source = "validated_header_index"
        else:
            raise FileNotFoundError(f"{rid}: missing WAV {candidate}; provide a validated WAV-header index for archived audio")
        frequency, freq_source = extract_signal_frequency(meta, l3)
        channel_frequency = number((meta.get("bellhop_env") or {}).get("freq_hz"))
        params = meta.get("signal_params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"{rid}: signal_params must be an object")
        frequency_candidates = {k: number(params.get(k)) for k in ("center_freq_hz", "center_frequency_hz", "carrier_freq_hz", "carrier_frequency_hz") if number(params.get(k)) is not None}
        freq_conflicts = [k for k, value in frequency_candidates.items() if frequency is not None and not math.isclose(value, frequency, rel_tol=1e-9, abs_tol=1e-9)]
        output = meta.get("bellhop_output") or {}
        gain_key = "tl_db" if number(meta.get("tl_db")) is not None else "bellhop_output.tl_db"
        snr_key = "snr_db" if number(meta.get("snr_db")) is not None else "bellhop_output.snr_db"
        gain = number(meta.get("tl_db")) if gain_key == "tl_db" else number(output.get("tl_db"))
        snr = number(meta.get("snr_db")) if snr_key == "snr_db" else number(output.get("snr_db"))
        source_match = re.match(r"^(.*)_ch(\d+)$", rid)
        source_id = meta.get("source_id") or (source_match.group(1) if source_match else rid)
        channel_id = meta.get("channel_id")
        if channel_id is None and source_match:
            channel_id = int(source_match.group(2))
        duration = header["frames"] / header["fs"]
        metadata_duration = number(meta.get("audio_duration_s"))
        duration_matches = None if metadata_duration is None else abs(duration - metadata_duration) <= 1 / header["fs"] + 1e-8
        freq_diff = None if frequency is None or channel_frequency is None else not math.isclose(frequency, channel_frequency, rel_tol=1e-9, abs_tol=1e-9)
        validation = {"metadata_duration_s": metadata_duration, "duration_matches_metadata": duration_matches,
                      "signal_channel_frequency_differ": freq_diff, "signal_frequency_conflicts": freq_conflicts,
                      "metadata_conflicts": conflicts, "external_metadata_path": external_path,
                      "header_source": header_source}
        sources = {"duration_s": header_source + ".frames/fs", "fs": header_source, "frames": header_source,
                   "signal_frequency_hz": origins.get(freq_source, "_meta." + freq_source if freq_source else None),
                   "channel_frequency_hz": origins.get("bellhop_env.freq_hz"),
                   "G_h": origins.get(gain_key), "S_src": origins.get(snr_key),
                   "source_id": origins.get("source_id", "id_suffix"),
                   "channel_id": origins.get("channel_id", "id_suffix"),
                   "labels": "_gt" if raw.get("_gt") else "metadata"}
        if l1 != "active":
            gain, frequency = None, None
        if l2 != "ship_noise":
            snr = None
        quality_score = number(meta.get("quality_score"))
        if quality_score is None:
            quality_score = snr if l2 == "ship_noise" else gain
        row = {"id": rid, "source_id": str(source_id), "channel_id": channel_id,
               "l1": l1, "l2": l2, "l3": l3, "audio_path": audio_path, "audio_locator": locator,
               "fs": header["fs"], "frames": header["frames"], "duration_s": duration,
               "signal_frequency_hz": frequency, "channel_frequency_hz": channel_frequency,
               "G_h": gain, "S_src": snr, "signal_params": params,
               "field_sources": sources, "validation": validation,
               "quality_score": quality_score, "quality_metric": meta.get("quality_metric"),
               "selection_policy": meta.get("selection_policy")}
        result.append(row)
        if duration_matches is False:
            differences["metadata_duration_vs_wav"] += 1
        if freq_diff:
            differences["signal_vs_channel_frequency"] += 1
        if freq_conflicts:
            differences["signal_frequency_candidate_conflicts"] += 1
        for conflict in conflicts:
            differences["embedded_vs_external:" + conflict] += 1
        applicable = ("G_h", "signal_frequency_hz") if l1 == "active" else ("S_src",)
        for key in applicable:
            if row[key] is None:
                missing[key] += 1
        field_counts.update(source for source in sources.values() if source)
    provenance = {"schema_version": "ua_bench_figure_records_v1", "manifest": str(manifest),
                  "manifest_sha256": manifest_hash, "audio_root": str(root),
                  "metadata_root": str(metadata_root) if metadata_root else None,
                  "header_index": index_provenance, "header_index_sha256": index_provenance["sha256"] if index_provenance else None,
                  "record_count": len(result), "class_counts": dict(Counter(r["l3"] for r in result)),
                  "family_counts": dict(Counter(r["l2"] for r in result)),
                  "field_difference_counts": dict(differences),
                  "missing_counts": {key: missing[key] for key in ("G_h", "S_src", "signal_frequency_hz")},
                  "field_source_counts": dict(field_counts),
                  "header_source_counts": dict(Counter(r["validation"]["header_source"] for r in result))}
    return result, provenance


def robust_medoid_index(features, ids: list[str], targets=None) -> int:
    """Nearest class median with IQR scaling; zero-IQR features are omitted."""
    rows = [list(map(float, row)) for row in features]
    if not rows or len(rows) != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("medoid needs nonempty rows and unique matching IDs")
    width = len(rows[0])
    if any(len(row) != width or not all(math.isfinite(v) for v in row) for row in rows):
        raise ValueError("medoid features must be a finite rectangular matrix")
    if targets is not None and (len(targets) != width or not all(math.isfinite(float(v)) for v in targets)):
        raise ValueError("medoid targets must be finite and match feature dimensions")
    def percentile(values, p):
        ordered = sorted(values)
        position = (len(ordered) - 1) * p
        lo = int(position)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (position - lo)
    distances = [0.0] * len(rows)
    for col in range(width):
        values = [row[col] for row in rows]
        scale = percentile(values, .75) - percentile(values, .25)
        if scale == 0:
            continue
        center = percentile(values, .5) if targets is None else float(targets[col])
        for i, value in enumerate(values):
            distances[i] += ((value - center) / scale) ** 2
    return min(range(len(rows)), key=lambda i: (distances[i], ids[i]))

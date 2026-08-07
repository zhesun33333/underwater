"""Audit processed channel JSON files for legacy zero-arrival pass-through audio.

The script is read-only with respect to the dataset. It writes a TSV report but
never moves or deletes JSON/WAV files.
"""

import argparse
import csv
import math
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Optional

from utils.json_parser import load_jsonc


JSON_EXTENSIONS = {".json", ".jsonc"}
DEFINITE_INVALID = "DEFINITE_INVALID_ZERO_ARRIVALS"


def _as_float(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_nonnegative_int(value) -> Optional[int]:
    if isinstance(value, bool):
        return None
    number = _as_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _dataset_kind(path: Path, meta: dict) -> str:
    output = meta.get("bellhop_output") or {}
    category = str(meta.get("signal_category", "")).lower()
    path_parts = {part.lower() for part in path.parts}
    if category == "radiated_noise" or output.get("mode") == "broadband":
        return "ship"
    if "pulsecom" in path_parts or category in {"pulse", "communication"}:
        return "pulsecom"
    return "unknown"


def _resolve_wav_path(processed_root: Path, meta: dict) -> Optional[Path]:
    value = meta.get("wav_path")
    if not value:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else processed_root / path


def _evidence_state(counts: list[int], invalid: bool) -> str:
    # A saved zero takes precedence over contradictory positive evidence.
    if any(count == 0 for count in counts):
        return "zero"
    if invalid:
        return "invalid"
    if any(count > 0 for count in counts):
        return "nonzero"
    return "missing"


def _pulse_arrival_evidence(output: dict) -> tuple[str, str]:
    """Return (state, evidence), explicitly separating nonzero and missing."""
    evidence = []
    counts = []
    invalid = False

    if "arrivals" in output:
        arrivals = output.get("arrivals")
        if isinstance(arrivals, list):
            evidence.append(f"arrivals_len={len(arrivals)}")
            counts.append(len(arrivals))
        else:
            evidence.append("arrivals_invalid_type")
            invalid = True

    if "summary" in output:
        summary = output.get("summary")
        if not isinstance(summary, dict):
            evidence.append("summary_invalid_type")
            invalid = True
        elif "num_arrivals" in summary:
            count = _as_nonnegative_int(summary["num_arrivals"])
            if count is None:
                evidence.append("summary_num_arrivals_invalid")
                invalid = True
            else:
                evidence.append(f"summary_num_arrivals={count}")
                counts.append(count)

    return _evidence_state(counts, invalid), ",".join(evidence)


def _ship_arrival_evidence(output: dict) -> tuple[str, str]:
    """Audit current Ship metadata without guessing about legacy schemas."""
    evidence = []
    counts = []
    invalid = False

    if "num_freqs_succeeded" in output:
        count = _as_nonnegative_int(output["num_freqs_succeeded"])
        if count is None:
            evidence.append("num_freqs_succeeded_invalid")
            invalid = True
        else:
            evidence.append(f"num_freqs_succeeded={count}")
            counts.append(count)

    if "successful_freqs_hz" in output:
        frequencies = output.get("successful_freqs_hz")
        if isinstance(frequencies, list):
            evidence.append(f"successful_freqs={len(frequencies)}")
            counts.append(len(frequencies))
        else:
            evidence.append("successful_freqs_invalid_type")
            invalid = True

    if "arrivals_per_frequency" in output:
        per_frequency = output.get("arrivals_per_frequency")
        if not isinstance(per_frequency, dict):
            evidence.append("arrivals_per_frequency_invalid_type")
            invalid = True
        else:
            parsed_counts = [_as_nonnegative_int(value) for value in per_frequency.values()]
            if any(value is None for value in parsed_counts):
                evidence.append("arrivals_per_frequency_invalid")
                invalid = True
            else:
                total = sum(parsed_counts)
                evidence.append(f"arrivals_across_frequencies={total}")
                counts.append(total)

    return _evidence_state(counts, invalid), ",".join(evidence)


def _gain_db(output: dict) -> Optional[float]:
    for key in ("tl_db", "estimated_transmission_loss_db"):
        value = _as_float(output.get(key))
        if value is not None:
            return value
    return None


def audit(args) -> int:
    processed_root = args.processed_root.resolve()
    if not processed_root.is_dir():
        raise SystemExit(f"Processed root does not exist: {processed_root}")

    report_path = args.report.resolve()
    counters = Counter()
    rows = []

    json_files = chain.from_iterable(
        processed_root.rglob(f"*{extension}") for extension in sorted(JSON_EXTENSIONS)
    )
    for json_path in json_files:
        if not json_path.is_file():
            continue

        try:
            meta = load_jsonc(json_path)
            if not isinstance(meta, dict):
                raise TypeError("top-level JSON value is not an object")
        except Exception as exc:
            counters["invalid_json"] += 1
            rows.append({
                "status": "INVALID_JSON",
                "dataset": "unknown",
                "sample_id": json_path.stem,
                "json_path": str(json_path),
                "wav_path": "",
                "model_version": "",
                "arrival_evidence": "",
                "gain_db": "",
                "notes": str(exc).replace("\t", " ").replace("\n", " "),
            })
            continue

        dataset = _dataset_kind(json_path, meta)
        if args.dataset != "all" and dataset != args.dataset:
            continue
        counters[f"scanned_{dataset}"] += 1

        output = meta.get("bellhop_output") or {}
        if not isinstance(output, dict):
            output = {}
        wav_path = _resolve_wav_path(processed_root, meta)
        wav_missing = wav_path is None or not wav_path.is_file()

        if dataset == "pulsecom":
            arrival_state, evidence = _pulse_arrival_evidence(output)
        elif dataset == "ship":
            arrival_state, evidence = _ship_arrival_evidence(output)
        else:
            arrival_state, evidence = "missing", ""

        gain = _gain_db(output)
        status = None
        notes = []
        if arrival_state == "zero":
            status = DEFINITE_INVALID
            notes.append("legacy output can contain unprocessed source audio")
            counters["definite_invalid"] += 1
        elif arrival_state == "invalid":
            status = "INVALID_ARRIVAL_METADATA"
            notes.append("arrival-count fields exist but cannot verify a nonzero count")
            counters[f"invalid_arrival_metadata_{dataset}"] += 1
        elif arrival_state == "missing":
            counters[f"unverifiable_{dataset}"] += 1
            if dataset == "ship" and gain is not None and abs(gain) <= 1e-8:
                status = "REVIEW_SHIP_LEGACY_ZERO_GAIN"
                notes.append(
                    "old Ship schema plus exactly zero gain is suspicious, not definitive"
                )
                counters["review_ship_legacy_zero_gain"] += 1
            elif args.report_unverifiable:
                status = "UNVERIFIABLE_LEGACY_METADATA"
                notes.append("no saved arrival-count fields; do not delete from JSON alone")
        else:
            counters[f"verified_nonzero_{dataset}"] += 1

        if wav_missing:
            counters["missing_wav"] += 1
            if status is None:
                status = "MISSING_WAV"
            notes.append("referenced WAV is missing")

        if status is None:
            continue

        rows.append({
            "status": status,
            "dataset": dataset,
            "sample_id": str(meta.get("id", json_path.stem)),
            "json_path": str(json_path),
            "wav_path": str(wav_path) if wav_path is not None else "",
            "model_version": str(meta.get("channel_model_version", "legacy/unset")),
            "arrival_evidence": evidence,
            "gain_db": "" if gain is None else f"{gain:.8g}",
            "notes": "; ".join(notes),
        })

    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "status", "dataset", "sample_id", "json_path", "wav_path",
        "model_version", "arrival_evidence", "gain_db", "notes",
    ]
    with report_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print("Legacy channel output audit")
    print(f"  processed root: {processed_root}")
    print(f"  dataset filter: {args.dataset}")
    for key in sorted(counters):
        print(f"  {key}: {counters[key]}")
    print(f"  report rows: {len(rows)}")
    print(f"  report: {report_path}")
    print("  dataset files were not modified")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find legacy processed audio whose JSON records zero arrivals.",
    )
    parser.add_argument(
        "--processed-root", type=Path, default=Path("processed_audio"),
        help="Processed audio root (default: processed_audio)",
    )
    parser.add_argument(
        "--dataset", choices=("pulsecom", "ship", "all"), default="pulsecom",
        help="Dataset to inspect (default: pulsecom)",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("legacy_channel_audit.tsv"),
        help="TSV report path (default: legacy_channel_audit.tsv)",
    )
    parser.add_argument(
        "--report-unverifiable", action="store_true",
        help="Also list legacy JSON files that lack arrival-count fields",
    )
    return audit(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

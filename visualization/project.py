"""Explicit preparation, read-only rendering, and provenance for paper figures.

Large source manifests are streamed only during ``prepare``. Rendering reads
the compact, SHA-bound cache, allowing Tavotto to call main() without writes
other than the intercepted Matplotlib savefig calls.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA = 1
PATH_KEYS = ("full_manifest", "eval_manifest", "eval_audio_root", "full_header_index", "cache_dir")


def sha256(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_config(path: Path | None = None) -> dict:
    path = Path(path or os.environ.get("UABENCH_FIGURE_CONFIG", ROOT / "config.local.json")).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Copy {ROOT / 'config.example.json'} to {path}, then run python -m visualization prepare")
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    for key in PATH_KEYS:
        value = Path(config[key]).expanduser()
        config[key] = str((path.parent / value).resolve() if not value.is_absolute() else value.resolve())
    style = config["style"]
    if style["font_size_pt"] <= 8 or style["width_mm"] <= 0:
        raise ValueError("Paper figures require positive width and font_size_pt > 8")
    config["config_path"] = str(path)
    return config


def file_signature(path: Path) -> dict:
    path = Path(path).resolve()
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_gzip(path: Path, value: dict | list) -> None:
    # mtime=0 gives deterministic companion-data bytes for unchanged values.
    with path.open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as stream:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def validate_datasets(full: list[dict], selected: list[dict], config: dict) -> dict:
    from testsite.reporting.figure_data import CLASS_ORDER
    by_id = {r["id"]: r for r in full}
    if len(by_id) != len(full) or len({r["id"] for r in selected}) != len(selected):
        raise ValueError("Duplicate record IDs")
    expected = config.get("expected_full_count")
    if expected is not None and len(full) != expected:
        raise ValueError(f"Expected {expected} full-test records; got {len(full)}")
    counts = Counter(r["l3"] for r in selected)
    per_class = config.get("expected_eval_per_class", 200)
    if set(counts) != set(CLASS_ORDER) or any(counts[k] != per_class for k in CLASS_ORDER):
        raise ValueError(f"Expected {per_class} Eval records in each of 13 classes: {dict(counts)}")
    checked = ("source_id", "l1", "l2", "l3", "fs", "frames", "duration_s", "signal_frequency_hz", "channel_frequency_hz", "G_h", "S_src", "signal_params")
    for record in selected:
        source = by_id.get(record["id"])
        if source is None:
            raise ValueError(f"Eval ID absent from test: {record['id']}")
        different = [key for key in checked if source.get(key) != record.get(key)]
        if different:
            raise ValueError(f"Eval/test disagreement for {record['id']}: {different}")
        if record["fs"] != 16000:
            raise ValueError(f"Expected 16-kHz Eval input: {record['id']}")
    return {"full_records": len(full), "eval_records": len(selected), "eval_class_counts": dict(counts),
            "eval_is_subset_of_full": True, "matched_fields": list(checked),
            "eval_wav_headers_match_full_index": True}


def prepare(config_path: Path | None = None) -> dict:
    from testsite.reporting.figure_data import load_figure_records
    config = load_config(config_path)
    cache = Path(config["cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    print("Reading full-test manifest and audited WAV-header index...", flush=True)
    full, full_proof = load_figure_records(Path(config["full_manifest"]), header_index=Path(config["full_header_index"]))
    print("Reading Eval manifest and actual WAV headers...", flush=True)
    selected, eval_proof = load_figure_records(Path(config["eval_manifest"]), audio_root=Path(config["eval_audio_root"]))
    checks = validate_datasets(full, selected, config)
    paths = {Path(config[key]) for key in ("full_manifest", "eval_manifest", "full_header_index")}
    sidecar = Path(config["full_header_index"] + ".provenance.json")
    paths.add(sidecar)
    for item in json.loads(sidecar.read_text(encoding="utf-8"))["sources"]:
        paths.add(Path(item["path"]))
    paths.update(Path(r["audio_path"]) for r in selected)
    # Cache invalidation covers parser code, preparation code, configuration,
    # every Eval WAV, and archive stat signatures. It does not hash huge tars.
    paths.add(ROOT.parent / "testsite/reporting/figure_data.py")
    paths.add(Path(__file__))
    data_files = {}
    for name, records in (("full", full), ("eval", selected)):
        output = cache / f"{name}.json.gz"
        _write_gzip(output, records)
        data_files[name] = {"path": str(output), "sha256": sha256(output)}
    metadata = {"schema": SCHEMA, "config": config, "checks": checks,
                "datasets": {"full": full_proof, "eval": eval_proof},
                "data_files": data_files,
                "source_signatures": [file_signature(p) for p in sorted(paths)],
                "cache_validation": "SHA-256 of derived records; size/mtime_ns of sources; manifests/index hashed during preparation"}
    _write_json(cache / "inputs.provenance.json", metadata)
    return checks


def load_inputs(*, include_full: bool = True) -> tuple[dict, list[dict], list[dict], dict]:
    """Read validated cache without creating files (safe for Tavotto workers)."""
    config = load_config()
    proof_path = Path(config["cache_dir"]) / "inputs.provenance.json"
    if not proof_path.is_file():
        raise FileNotFoundError("Figure cache missing; run python -m visualization prepare first")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if proof["schema"] != SCHEMA or proof["config"] != config:
        raise ValueError("Figure configuration changed; run python -m visualization prepare")
    for expected in proof["source_signatures"]:
        if file_signature(Path(expected["path"])) != expected:
            raise ValueError(f"Figure source changed: {expected['path']}; rebuild index if needed, then prepare")
    records = {}
    for name in (("full", "eval") if include_full else ("eval",)):
        data = proof["data_files"][name]
        path = Path(data["path"])
        if sha256(path) != data["sha256"]:
            raise ValueError(f"Corrupt derived cache: {path}; run prepare")
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            records[name] = json.load(stream)
    # Keep figure metadata compact; the full signature list is in cache proof.
    provenance = {key: proof[key] for key in ("schema", "checks", "datasets", "data_files", "cache_validation")}
    provenance["inputs_provenance_sha256"] = sha256(proof_path)
    provenance["inputs_provenance_path"] = str(proof_path)
    return config, records.get("full", []), records["eval"], provenance


def paper_style(style: dict):
    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties, findfont
    # Fail explicitly if unavailable; never silently substitute a paper font.
    findfont(FontProperties(family=style["font_family"]), fallback_to_default=False)
    size = style["font_size_pt"]
    return mpl.rc_context({"font.family": style["font_family"], "font.size": size,
        "axes.labelsize": size, "axes.titlesize": size, "axes.labelweight": "bold",
        "xtick.labelsize": size, "ytick.labelsize": size, "legend.fontsize": size,
        "mathtext.fontset": "stix", "legend.frameon": style["legend_frame"],
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "text.color": "black", "axes.labelcolor": "black", "axes.edgecolor": "black",
        "xtick.color": "black", "ytick.color": "black", "axes.titleweight": "normal",
        "axes.linewidth": 0.75, "lines.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": None,
        "savefig.dpi": 300, "figure.dpi": 300})


def write_proof(output: Path, figure, payload: dict, provenance: dict, style: dict) -> None:
    """Standalone-export sidecars; intentionally outside Tavotto main()."""
    data_path = output.with_suffix(".data.json.gz")
    _write_gzip(data_path, payload)
    code_paths = sorted(ROOT.rglob("*.py")) + [ROOT.parent / "testsite/reporting/figure_data.py"]
    record = {"figure": output.name, "style": style, "inputs": provenance,
        "size_mm": [float(v * 25.4) for v in figure.get_size_inches()],
        "data": {"path": data_path.name, "sha256": sha256(data_path)},
        "software": {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ("numpy", "scipy", "matplotlib", "soundfile", "cmcrameri")}},
        "code_sha256": {str(p.relative_to(ROOT.parent)): sha256(p) for p in code_paths},
        "outputs": {ext: {"path": output.with_suffix(ext).name, "sha256": sha256(output.with_suffix(ext))} for ext in (".pdf", ".png")},
        "validation": {"tavotto_mcp_canvas": "not_run_in_this_chat"}}
    # Useful scientific methods stay readable without unpacking the raw arrays.
    record["methods"] = {k: v for k, v in payload.items() if k not in ("panels", "spectra", "raw_data")}
    _write_json(output.with_suffix(".json"), record)

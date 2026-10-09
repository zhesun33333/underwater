"""Build a provenance-bound WAV-header index without unpacking audio.

Examples (from the repository root)::

  python -m testsite.reporting.build_header_index --manifest DATA/sft_test.jsonl \
      --archive DATA/v1 --output visualization/cache/full_test_wav_headers.jsonl
  python -m testsite.reporting.build_header_index --manifest DATA/eval.jsonl \
      --audio-root DATA/testset_export --output visualization/cache/eval_headers.jsonl

--archive accepts files or directories. Uncompressed .tar files and the legacy
extensionless warship_ok archive are supported. Only matching WAV headers are
read; audio payloads are not extracted or decoded.
"""
from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath

try:
    from .figure_data import (_labels, read_jsonl, read_wav_header, source_signature,
                             register_audited_header_index)
except ImportError:
    from figure_data import (_labels, read_jsonl, read_wav_header, source_signature,
                             register_audited_header_index)


def _archives(paths: list[Path]) -> list[Path]:
    found = set()
    for path in paths:
        if path.is_dir():
            found.update(p.resolve() for p in path.rglob("*")
                         if p.is_file() and (p.suffix.lower() == ".tar" or p.name == "warship_ok")
                         and p.name != "dataset.tar")
        elif path.is_file():
            found.add(path.resolve())
        else:
            raise FileNotFoundError(path)
    return sorted(found, key=str)


def _audio_tail(path: str) -> tuple[str, ...]:
    parts = PurePosixPath(path.replace("\\", "/")).parts
    if "PulseCom" in parts:
        return parts[parts.index("PulseCom"):]
    return parts[-3:]


def build_header_index(manifest: Path, output: Path,
                       archives: list[Path] | None = None,
                       audio_root: Path | None = None) -> dict:
    """Read exact headers, reject incomplete/ambiguous mappings, then bind sources."""
    manifest = Path(manifest).resolve()
    output = Path(output).resolve()
    if output == manifest:
        raise ValueError("output must not overwrite the source manifest")
    targets = {}
    for raw in read_jsonl(manifest):
        _, l2, l3 = _labels(raw, raw.get("_meta") or {})
        if not isinstance(raw.get("audio"), str) or not raw["audio"]:
            raise ValueError(f"{raw['id']}: missing manifest audio path")
        targets[raw["id"]] = {"id": raw["id"], "L2": l2, "L3": l3, "audio": raw["audio"]}
    found, before = {}, {}
    if audio_root is not None:
        for rid, target in targets.items():
            path = Path(target["audio"])
            path = path if path.is_absolute() else Path(audio_root) / path
            if path.is_file():
                signature = source_signature("wav", path)
                before[("wav", str(path.resolve()))] = signature
                found[rid] = {"id": rid, "L2": target["L2"], "L3": target["L3"],
                              "audio_path": str(path.resolve()), **read_wav_header(path)}
    local_ids = set(found)
    for archive in _archives(archives or []):
        signature = source_signature("tar", archive)
        found_in_archive = 0
        with tarfile.open(archive, mode="r:") as source:
            for member in source:
                if not member.isfile() or not member.name.lower().endswith(".wav"):
                    continue
                rid = PurePosixPath(member.name.replace("\\", "/")).stem
                if rid not in targets or rid in local_ids:
                    continue
                if rid in found:
                    raise ValueError(f"duplicate archived WAV id {rid}: {archive}:{member.name}")
                target = targets[rid]
                if _audio_tail(member.name) != _audio_tail(target["audio"]):
                    raise ValueError(f"archive locator disagrees with manifest audio: {rid}: {member.name}")
                with source.extractfile(member) as audio:
                    header = read_wav_header(audio)
                found[rid] = {"id": rid, "L2": target["L2"], "L3": target["L3"],
                              "archive": str(archive), "member": member.name,
                              "offset_data": member.offset_data, "member_size": member.size, **header}
                found_in_archive += 1
        if found_in_archive:
            before[("tar", str(archive))] = signature
        print(f"{archive.name}: {found_in_archive:,} matching WAV headers", flush=True)
    missing = sorted(set(targets) - set(found))
    if missing:
        raise ValueError(f"missing {len(missing)} WAV headers; first IDs: {missing[:10]}")
    for (kind, path), signature in before.items():
        if source_signature(kind, Path(path)) != signature:
            raise ValueError(f"source changed while indexing: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    # No output is written until the complete mapping has passed validation.
    output.write_text("".join(json.dumps(found[rid], ensure_ascii=False) + "\n" for rid in targets), encoding="utf-8")
    provenance = register_audited_header_index(manifest, output, verification="wav_headers_read_from_sources")
    return {"record_count": len(found), "index": str(output), "provenance": str(provenance)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--archive", action="append", nargs="+", type=Path, default=[],
                        help="One or more uncompressed tar paths or directories (repeatable)")
    parser.add_argument("--register-audited-index", action="store_true",
                        help="Explicitly bind an independently audited existing --output index; does not reread source WAVs")
    args = parser.parse_args()
    if args.register_audited_index:
        if args.archive or args.audio_root:
            parser.error("--register-audited-index cannot be combined with --archive or --audio-root")
        print(register_audited_header_index(args.manifest, args.output))
    else:
        archives = [path for group in args.archive for path in group]
        if not archives and args.audio_root is None:
            parser.error("provide --archive and/or --audio-root")
        print(json.dumps(build_header_index(args.manifest, args.output, archives, args.audio_root), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Extract only JSON result artifacts from a supplied tar archive, with hashes.

The source archive may contain a nested ``.tar.gz`` for one result cohort.
Neither archive names nor payloads are executed. Every extracted path is
checked before writing, and the original member identity stays in the index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


MAX_MEMBER_BYTES = 512 * 1024 * 1024
MAX_EXTRACTED_BYTES = 4 * 1024 * 1024 * 1024
SUFFIXES = (".json", ".jsonl")


def _safe_target(root: Path, member_name: str) -> Path:
    if "\\" in member_name or ":" in member_name:
        raise ValueError(f"Unsafe tar member path: {member_name!r}")
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
        raise ValueError(f"Unsafe tar member path: {member_name!r}")
    target = (root / Path(*relative.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"Tar member escapes output directory: {member_name!r}")
    return target


def _file_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def import_archive(archive: Path, output: Path) -> dict:
    archive, output = archive.resolve(), output.resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Import destination must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    seen: set[str] = set()
    total = 0
    nested_count = 0

    def visit(stream: tarfile.TarFile, prefix: str = "") -> None:
        nonlocal total, nested_count
        for member in stream:
            name = f"{prefix}{member.name}"
            # A result archive is data: links, devices and executable files are
            # never materialized. Check names even when the suffix is skipped.
            _safe_target(output, name)
            if not member.isfile():
                continue
            if member.name.endswith(".tar.gz"):
                nested_count += 1
                if nested_count > 4:
                    raise ValueError("Too many nested tar archives")
                nested_prefix = f"{name[:-7]}/"
                nested_file = stream.extractfile(member)
                if nested_file is None:
                    raise ValueError(f"Cannot read nested tar: {name}")
                with nested_file, tarfile.open(fileobj=nested_file, mode="r|gz") as nested:
                    visit(nested, nested_prefix)
                continue
            if not name.lower().endswith(SUFFIXES):
                continue
            if member.size > MAX_MEMBER_BYTES or total + member.size > MAX_EXTRACTED_BYTES:
                raise ValueError(f"Result archive exceeds extraction size limit at {name}")
            if name in seen:
                raise ValueError(f"Duplicate result member path: {name}")
            seen.add(name)
            target = _safe_target(output, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = stream.extractfile(member)
            if payload is None:
                raise ValueError(f"Cannot read tar member: {name}")
            digest = hashlib.sha256()
            written = 0
            with payload, target.open("xb") as destination:
                while block := payload.read(1024 * 1024):
                    written += len(block)
                    if written > member.size or written > MAX_MEMBER_BYTES:
                        raise ValueError(f"Result member size mismatch: {name}")
                    destination.write(block)
                    digest.update(block)
            if written != member.size:
                raise ValueError(f"Result member truncated: {name}")
            total += written
            entries.append({"member": name, "path": str(target),
                            "bytes": written, "sha256": digest.hexdigest()})
            if len(entries) % 25 == 0:
                print(f"Imported {len(entries)} JSON artifacts ({total:,} bytes)...", flush=True)

    print(f"Hashing source archive: {archive}", flush=True)
    archive_sha256 = _file_sha256(archive)
    with tarfile.open(archive, mode="r|gz") as source:
        visit(source)
    index = {"schema": 1, "source_archive": str(archive),
             "source_sha256": archive_sha256, "nested_archives": nested_count,
             "files": len(entries), "extracted_bytes": total, "entries": entries}
    (output / "_extraction_manifest.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {key: index[key] for key in ("source_archive", "source_sha256",
                                        "nested_archives", "files", "extracted_bytes")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(import_archive(args.archive, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

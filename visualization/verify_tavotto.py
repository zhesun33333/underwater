"""Optional Tavotto 0.18 engine acceptance (no MCP/UI session is claimed).

Run with the Python environment containing tavotto[worker] and requirements.txt:
    python -m visualization.verify_tavotto

Uses the installed engine's registry, safe worker and publication preflight.
These Python interfaces are version-specific; normal figure rendering does not
depend on Tavotto. No publication errors are bypassed and no overrides applied.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

from .project import ROOT, sha256

STEMS = ("selection_distributions", "acoustic_active", "acoustic_ships")


def artifact_check(path: Path, width_mm: float) -> dict:
    import pikepdf
    from PIL import Image
    with pikepdf.open(path.with_suffix(".pdf")) as document:
        if len(document.pages) != 1:
            raise ValueError(f"Expected single-page figure: {path}")
        box = [float(v) for v in document.pages[0].MediaBox]
        size_mm = [(box[2] - box[0]) * 25.4 / 72, (box[3] - box[1]) * 25.4 / 72]
        fonts = [str(font.get("/BaseFont", "")) for font in document.pages[0].Resources.get("/Font", {}).values()]
    with Image.open(path.with_suffix(".png")) as preview:
        pixels = list(preview.size)
        dpi = list(preview.info.get("dpi", (0, 0)))
    if abs(size_mm[0] - width_mm) > 0.05:
        raise ValueError(f"PDF width mismatch: {size_mm[0]} mm")
    if min(dpi) < 299 or any(abs(pixels[i] - size_mm[i] / 25.4 * 300) > 2 for i in (0, 1)):
        raise ValueError(f"PNG physical dimensions/DPI mismatch: {pixels}, {dpi}")
    return {"pdf_size_mm": size_mm, "pdf_fonts": fonts, "png_pixels": pixels, "png_dpi": dpi,
            "pdf_sha256": sha256(path.with_suffix(".pdf")), "png_sha256": sha256(path.with_suffix(".png"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stems", nargs="*", metavar="STEM", help="Default: all three figure stems")
    args = parser.parse_args()
    args.stems = args.stems or list(STEMS)
    if set(args.stems) - set(STEMS):
        parser.error(f"Unknown stems; choose from {', '.join(STEMS)}")
    version = importlib.metadata.version("tavotto")
    if version != "0.18.0":
        raise RuntimeError(f"This acceptance adapter targets Tavotto 0.18.0; found {version}")
    # Explicit for this process only; use the same validated scientific stack.
    os.environ["TAVOTTO_WORKER_PYTHON"] = sys.executable
    from tavotto.engine import discover, pool, preflight, profiles
    from .project import load_config
    figures = ROOT / "figures/data"
    config = load_config()
    journal = {"widths_mm": {"double": config["style"]["width_mm"]}}
    registry, discovery, changes = discover.merge(figures)
    if discovery["conflicts"]:
        raise RuntimeError(f"Tavotto script/stem conflict: {discovery['conflicts']}")
    discover.write_config(figures, registry)
    report = {"tavotto_version": version, "verification_scope": "local engine; MCP tools and interactive canvas not exercised",
              "journal": journal, "registry": registry, "registry_changes": changes, "figures": {}}
    try:
        for stem in args.stems:
            print(f"Tavotto safe-worker build and preflight: {stem}", flush=True)
            script = f"{stem}.py"
            if stem not in registry["scripts"].get(script, {}).get("stems", []):
                raise RuntimeError(f"Tavotto did not statically discover {script}")
            worker, built = pool.build(script, str(figures), "main", allow_project_env=False)
            state = worker.override(stem, [])
            if not state.get("ok"):
                raise RuntimeError(f"Tavotto render failed: {state}")
            manifest = state["manifest"]
            result = preflight.summarize(preflight.run(preflight.spec_from_manifest(manifest), profiles.load(journal=journal)))
            report["figures"][stem] = {"entry": "main", "parameterizable": True,
                "editable_element_count": sum(bool(e.get("editable")) for e in manifest.get("elements", [])),
                "preflight": result, "artifacts": artifact_check(figures / stem, config["style"]["width_mm"])}
            print(json.dumps({"stem": stem, "blocking": result["blocking"], "counts": result["counts"]}), flush=True)
    finally:
        pool.shutdown_all(wait=True)
    output = figures / "verification_tavotto.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {output}")
    if any(item["preflight"]["blocking"] for item in report["figures"].values()):
        raise SystemExit("Tavotto publication preflight found blocking errors; see report")


if __name__ == "__main__":
    main()

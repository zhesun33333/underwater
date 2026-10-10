"""Local Tavotto 0.18.0 acceptance check for paper result figures D-G.

Run after ``python -m visualization render D E F G`` with the same data and
results configuration. This uses Tavotto's installed safe worker and
publication preflight, plus independent PDF/PNG inspection. It does not open
an MCP session, grant workspace access, or claim an interactive canvas check.
The worker intercepts savefig; this verifier does not export figures.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys

from .project import ROOT, sha256


STEMS = {"D": "class_diagnostics", "E": "error_decomposition",
         "F": "metadata_performance", "G": "silence_control"}
TAVOTTO_VERSION = "0.18.0"
PDF_SIZE_TOLERANCE_MM = 0.05


def _pdf_fonts(resources) -> list[dict]:
    fonts = []
    for font in resources.get("/Font", {}).values():
        base = str(font.get("/BaseFont", ""))
        descendant = font.get("/DescendantFonts")
        actual = descendant[0] if descendant else font
        descriptor = actual.get("/FontDescriptor")
        embedded = bool(descriptor and any(key in descriptor for key in
                                           ("/FontFile", "/FontFile2", "/FontFile3")))
        fonts.append({"name": base, "subtype": str(font.get("/Subtype", "")),
                      "embedded": embedded})
    return fonts


def _raster_xobjects(resources, depth: int = 0) -> int:
    if depth > 16:
        raise ValueError("PDF XObject nesting exceeds inspection limit")
    count = 0
    for obj in resources.get("/XObject", {}).values():
        subtype = str(obj.get("/Subtype", ""))
        if subtype == "/Image":
            count += 1
        elif subtype == "/Form":
            count += _raster_xobjects(obj.get("/Resources", {}), depth + 1)
    return count


def _check_proof(stem_path: Path, proof: dict, expected_style: dict,
                 results_provenance_sha256: str) -> dict:
    if proof.get("figure") != stem_path.name:
        raise ValueError(f"Figure proof stem mismatch: {stem_path.name}")
    if proof.get("style") != expected_style:
        raise ValueError(f"Figure proof style differs from prepared results: {stem_path.name}")
    if proof.get("inputs", {}).get("results_provenance_sha256") != results_provenance_sha256:
        raise ValueError(f"Figure proof belongs to a different prepared results cache: {stem_path.name}")
    size = proof.get("size_mm")
    if (not isinstance(size, list) or len(size) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or v <= 0 for v in size)):
        raise ValueError(f"Invalid expected figure size in proof: {stem_path.name}")
    if abs(size[0] - expected_style["width_mm"]) > PDF_SIZE_TOLERANCE_MM:
        raise ValueError(f"Figure proof width disagrees with publication style: {stem_path.name}")
    data = proof.get("data", {})
    data_path = stem_path.with_suffix(".data.json.gz")
    if data.get("path") != data_path.name or sha256(data_path) != data.get("sha256"):
        raise ValueError(f"Figure data sidecar hash mismatch: {stem_path.name}")

    source_root = ROOT.parent.resolve()
    code = proof.get("code_sha256", {})
    if not isinstance(code, dict) or not code:
        raise ValueError(f"Figure proof has no code hashes: {stem_path.name}")
    changed = []
    for relative, digest in code.items():
        path = (source_root / relative).resolve()
        if not path.is_relative_to(source_root) or not path.is_file() or sha256(path) != digest:
            changed.append(relative)
    if changed:
        raise ValueError(f"Figure code changed since rendering: {changed[:5]}")
    return {"size_mm": size, "data_sha256": data["sha256"],
            "code_files_checked": len(code)}


def artifact_check(stem_path: Path, expected_style: dict,
                   results_provenance_sha256: str) -> dict:
    """Inspect final files against the render proof and physical dimensions."""
    import pikepdf
    from PIL import Image

    proof_path = stem_path.with_suffix(".json")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    checked = _check_proof(stem_path, proof, expected_style, results_provenance_sha256)
    expected_mm = checked["size_mm"]
    pdf_path, png_path = stem_path.with_suffix(".pdf"), stem_path.with_suffix(".png")
    for extension, path in ((".pdf", pdf_path), (".png", png_path)):
        entry = proof.get("outputs", {}).get(extension, {})
        if entry.get("path") != path.name or sha256(path) != entry.get("sha256"):
            raise ValueError(f"Figure {extension} hash differs from render proof: {stem_path.name}")

    with pikepdf.open(pdf_path) as document:
        if len(document.pages) != 1:
            raise ValueError(f"Expected one PDF page: {pdf_path}")
        page = document.pages[0]
        box = [float(value) for value in page.MediaBox]
        size_mm = [(box[2] - box[0]) * 25.4 / 72,
                   (box[3] - box[1]) * 25.4 / 72]
        if any(abs(actual - wanted) > PDF_SIZE_TOLERANCE_MM
               for actual, wanted in zip(size_mm, expected_mm)):
            raise ValueError(f"PDF physical size differs from render proof: {size_mm} vs {expected_mm}")
        resources = page.Resources
        fonts = _pdf_fonts(resources)
        if not fonts or any(not font["embedded"] for font in fonts):
            raise ValueError(f"PDF text fonts are missing or not embedded: {fonts}")
        wanted_family = re.sub(r"[^a-z0-9]", "", expected_style["font_family"].lower())
        def permitted(font):
            name = re.sub(r"[^a-z0-9]", "", font["name"].split("+")[-1].lower())
            return wanted_family in name or name.startswith("stix")
        if not any(wanted_family in re.sub(r"[^a-z0-9]", "", item["name"].lower())
                   for item in fonts) or any(not permitted(item) for item in fonts):
            raise ValueError(f"PDF fonts differ from {expected_style['font_family']} and STIX math: {fonts}")
        raster_count = _raster_xobjects(resources)
        if raster_count:
            raise ValueError(f"Result PDF contains {raster_count} raster image XObjects: {pdf_path}")
        operations = Counter(str(instruction.operator)
                             for instruction in pikepdf.parse_content_stream(page))
        vector_ops = sum(operations[key] for key in ("m", "l", "c", "re", "S", "s",
                                                     "f", "f*", "B", "B*", "b", "b*"))
        if operations["BT"] == 0 or vector_ops == 0:
            raise ValueError(f"PDF lacks editable text or vector drawing commands: {pdf_path}")

    with Image.open(png_path) as preview:
        pixels = list(preview.size)
        dpi = list(preview.info.get("dpi", (0, 0)))
        preview.verify()
    if len(dpi) != 2 or any(abs(value - 300) > 1 for value in dpi):
        raise ValueError(f"PNG DPI is not 300: {dpi}")
    if any(abs(pixels[i] - expected_mm[i] / 25.4 * 300) > 2 for i in (0, 1)):
        raise ValueError(f"PNG pixels disagree with PDF physical size: {pixels}, {expected_mm}")
    return {"pdf_size_mm": size_mm, "pdf_fonts": fonts,
            "pdf_raster_images": raster_count, "pdf_vector_operations": vector_ops,
            "png_pixels": pixels, "png_dpi": dpi,
            "pdf_sha256": sha256(pdf_path), "png_sha256": sha256(png_path),
            "proof_sha256": sha256(proof_path), **checked}


def _issue(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def verify(stems: list[str] | None = None, *, report_path: Path | None = None) -> dict:
    """Run local safe-worker preflight and write one report; no MCP calls."""
    selected = list(STEMS) if stems is None else list(stems)
    if not selected or any(key not in STEMS for key in selected) or len(selected) != len(set(selected)):
        raise ValueError("Choose unique figure letters from D, E, F, G")
    figures = ROOT / "figures/results"
    output = (report_path or figures / "verification_tavotto.json").resolve()
    from .results import load_results
    data, provenance = load_results()
    style = data["style"]
    journal = {"widths_mm": {"double": style["width_mm"]}}
    report = {"schema": 1, "tavotto_version": None,
              "verification_scope": "local Tavotto 0.18.0 safe worker; MCP tools and interactive canvas not exercised",
              "results_provenance_sha256": provenance["results_provenance_sha256"],
              "journal": journal, "registry_changes": None, "discovery_conflicts": [],
              "figures": {}, "global_errors": [], "all_passed": False}

    registry = None
    try:
        version = importlib.metadata.version("tavotto")
        report["tavotto_version"] = version
        if version != TAVOTTO_VERSION:
            raise RuntimeError(f"Adapter targets Tavotto {TAVOTTO_VERSION}; found {version}")
        os.environ["TAVOTTO_WORKER_PYTHON"] = sys.executable
        from tavotto.engine import discover
        registry, discovery, changes = discover.merge(figures)
        report["registry_changes"] = changes
        report["discovery_conflicts"] = discovery.get("conflicts", [])
        if report["discovery_conflicts"]:
            raise RuntimeError(f"Tavotto script/stem conflicts: {report['discovery_conflicts']}")
    except Exception as error:
        report["global_errors"].append(_issue(error))

    engine_pool = None
    engine_preflight = None
    profile = None
    if registry is not None and not report["global_errors"]:
        try:
            from tavotto.engine import pool, preflight, profiles
            engine_pool = pool
            engine_preflight = preflight
            profile = profiles.load(journal=journal)
        except Exception as error:
            report["global_errors"].append(_issue(error))
    try:
        for key in selected:
            stem = STEMS[key]
            item = {"figure": key, "stem": stem, "entry": "main",
                    "editable_element_count": None, "preflight": None,
                    "artifacts": None, "errors": [], "status": "error"}
            report["figures"][key] = item
            try:
                item["artifacts"] = artifact_check(
                    figures / stem, style, provenance["results_provenance_sha256"])
            except Exception as error:
                item["errors"].append(_issue(error))
            if report["global_errors"]:
                item["errors"].extend(report["global_errors"])
            else:
                try:
                    script = f"{stem}.py"
                    if stem not in registry["scripts"].get(script, {}).get("stems", []):
                        raise RuntimeError(f"Tavotto did not statically discover {script}")
                    worker, built = engine_pool.build(script, str(figures), "main",
                                                      allow_project_env=False)
                    if isinstance(built, dict) and built.get("ok") is False:
                        raise RuntimeError(f"Safe-worker build failed: {built}")
                    state = worker.override(stem, [])
                    if not state.get("ok"):
                        raise RuntimeError(f"Safe-worker render failed: {state}")
                    manifest = state.get("manifest")
                    if not isinstance(manifest, dict):
                        raise RuntimeError("Safe worker returned no editable manifest")
                    item["editable_element_count"] = sum(
                        bool(element.get("editable")) for element in manifest.get("elements", []))
                    if item["editable_element_count"] == 0:
                        raise RuntimeError("Tavotto found no editable figure elements")
                    item["preflight"] = engine_preflight.summarize(
                        engine_preflight.run(engine_preflight.spec_from_manifest(manifest), profile))
                except Exception as error:
                    item["errors"].append(_issue(error))
            if item["artifacts"] is not None:
                try:
                    for extension in ("pdf", "png"):
                        path = figures / f"{stem}.{extension}"
                        if sha256(path) != item["artifacts"][f"{extension}_sha256"]:
                            raise ValueError(f"Safe worker changed the {extension.upper()} artifact")
                except Exception as error:
                    item["errors"].append(_issue(error))
            item["status"] = ("error" if item["errors"] or
                              (item["preflight"] and item["preflight"]["blocking"])
                              else "warning" if item["preflight"] and
                              (item["preflight"]["warnings"] or item["preflight"]["not_verifiable"])
                              else "pass")
            print(json.dumps({"figure": key, "status": item["status"],
                              "errors": item["errors"],
                              "preflight_counts": item["preflight"]["counts"]
                              if item["preflight"] else None}), flush=True)
    finally:
        if engine_pool is not None:
            engine_pool.shutdown_all(wait=True)

    report["all_passed"] = not report["global_errors"] and all(
        item["status"] != "error" for item in report["figures"].values())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    print(f"Saved {output}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*", metavar="FIGURE", help="D E F G; default: all four")
    parser.add_argument("--results-config", type=Path, help="Use the same results config as rendering")
    parser.add_argument("--config", type=Path, help="Use the same base data config as rendering")
    parser.add_argument("--report", type=Path, help="Default: figures/results/verification_tavotto.json")
    args = parser.parse_args()
    if args.results_config:
        os.environ["UABENCH_RESULTS_CONFIG"] = str(args.results_config.resolve())
    if args.config:
        os.environ["UABENCH_FIGURE_CONFIG"] = str(args.config.resolve())
    result = verify(args.figures or None, report_path=args.report)
    if not result["all_passed"]:
        raise SystemExit("Result figure acceptance found blocking problems; see report")


if __name__ == "__main__":
    main()

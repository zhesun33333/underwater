"""Run from the repository root: python -m visualization prepare|render."""
from __future__ import annotations

import argparse
import json
import os
import runpy
from pathlib import Path

from .project import ROOT, load_config, prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Default: visualization/config.local.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Validate real inputs and create compact figure cache")
    render = sub.add_parser("render", help="Render standalone Tavotto-compatible entry files")
    render.add_argument("figures", nargs="*", metavar="FIGURE", help="B, A, A1, A2; default: B A")
    args = parser.parse_args()
    if args.config:
        os.environ["UABENCH_FIGURE_CONFIG"] = str(args.config.resolve())
    if args.command == "prepare":
        print(json.dumps(prepare(args.config), ensure_ascii=False, indent=2))
        return
    load_config(args.config)
    args.figures = args.figures or ["B", "A"]
    names = {"B": ["selection_distributions"], "A": ["acoustic_active", "acoustic_ships"],
             "A1": ["acoustic_active"], "A2": ["acoustic_ships"]}
    if set(args.figures) - set(names):
        parser.error("Unknown figure; choose B, A, A1, or A2")
    scripts = dict.fromkeys(stem for key in args.figures for stem in names[key])
    for stem in scripts:
        script = ROOT / "figures/data" / f"{stem}.py"
        print(f"Rendering {script.name} ...", flush=True)
        runpy.run_path(str(script), run_name="__main__")
        print(f"Saved {script.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()

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
    parser.add_argument("--results-config", type=Path, help="Default: visualization/results.local.json")
    parser.add_argument("--channel-config", type=Path, help="Default: visualization/channels.local.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Validate real inputs and create compact figure cache")
    sub.add_parser("prepare-results", help="Audit model runs and prepare D/E/F/G statistics")
    sub.add_parser("prepare-channels", help="Pair source archive WAVs with audited Eval outputs for C")
    render = sub.add_parser("render", help="Render standalone Tavotto-compatible entry files")
    render.add_argument("figures", nargs="*", metavar="FIGURE", help="B, A, A1, A2, C, D, E, F, G; default: B A")
    args = parser.parse_args()
    if args.config:
        os.environ["UABENCH_FIGURE_CONFIG"] = str(args.config.resolve())
    if args.results_config:
        os.environ["UABENCH_RESULTS_CONFIG"] = str(args.results_config.resolve())
    if args.channel_config:
        os.environ["UABENCH_CHANNEL_CONFIG"] = str(args.channel_config.resolve())
    if args.command == "prepare-channels":
        from .channel_data import prepare_channels
        print(json.dumps(prepare_channels(args.channel_config), ensure_ascii=False, indent=2))
        return
    if args.command == "prepare":
        print(json.dumps(prepare(args.config), ensure_ascii=False, indent=2))
        return
    if args.command == "prepare-results":
        from .results import prepare_results
        report = prepare_results(args.results_config)
        print(json.dumps({"mode": report["mode"], "runs": report["runs"],
                          "D_model": report["D_model"], "E_states": report["E_states"],
                          "G_comparisons": report["G_comparisons"]}, ensure_ascii=False, indent=2))
        return
    load_config(args.config)
    args.figures = args.figures or ["B", "A"]
    names = {"B": ["data/selection_distributions"], "A": ["data/acoustic_active", "data/acoustic_ships"],
             "A1": ["data/acoustic_active"], "A2": ["data/acoustic_ships"],
             "C": ["data/channel_examples"],
             "D": ["results/class_diagnostics"], "E": ["results/error_decomposition"],
             "F": ["results/metadata_performance"], "G": ["results/silence_control"]}
    if set(args.figures) - set(names):
        parser.error("Unknown figure; choose B, A, A1, A2, C, D, E, F, or G")
    scripts = dict.fromkeys(stem for key in args.figures for stem in names[key])
    for stem in scripts:
        script = ROOT / "figures" / f"{stem}.py"
        print(f"Rendering {script.name} ...", flush=True)
        runpy.run_path(str(script), run_name="__main__")
        print(f"Saved {script.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()

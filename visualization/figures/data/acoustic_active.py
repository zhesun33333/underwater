"""Figure A1: deterministic examples of eight active classes. Tavotto main()."""
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    import sys
    repo = str(OUT.parents[2])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from visualization.project import load_inputs, paper_style
    from visualization.plot_acoustic import build_active_figure
    config, _, selected, provenance = load_inputs(include_full=False)
    with paper_style(config["style"]):
        fig, payload = build_active_figure(selected, config["style"])
        fig.savefig(OUT / "acoustic_active.pdf", dpi=300)
        fig.savefig(OUT / "acoustic_active.png", dpi=300)
    fig._uab_proof = (payload, provenance, config["style"])
    return fig


if __name__ == "__main__":
    figure = main()
    from visualization.project import write_proof
    write_proof(OUT / "acoustic_active", figure, *figure._uab_proof)
    import matplotlib.pyplot as plt
    plt.close(figure)

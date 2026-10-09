"""Figure B: full-test versus Eval distributions. Tavotto entry: main()."""
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    import sys
    repo = str(OUT.parents[2])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from visualization.project import load_inputs, paper_style
    from visualization.plot_distribution import build_figure
    config, full, selected, provenance = load_inputs()
    with paper_style(config["style"]):
        fig, payload = build_figure(full, selected, config["style"])
        fig.savefig(OUT / "selection_distributions.pdf", dpi=300)
        fig.savefig(OUT / "selection_distributions.png", dpi=300)
    fig._uab_proof = (payload, provenance, config["style"])
    return fig


if __name__ == "__main__":
    figure = main()
    from visualization.project import write_proof
    write_proof(OUT / "selection_distributions", figure, *figure._uab_proof)
    import matplotlib.pyplot as plt
    plt.close(figure)

"""D: class confusion and precision/recall/F1. Tavotto-compatible entry."""
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    import sys
    repo = str(OUT.parents[2])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from visualization.project import paper_style
    from visualization.results import load_results
    from visualization.plot_results import build_class_diagnostics
    data, provenance = load_results()
    with paper_style(data["style"]):
        fig, payload = build_class_diagnostics(data)
        fig.savefig(OUT / "class_diagnostics.pdf", dpi=300)
        fig.savefig(OUT / "class_diagnostics.png", dpi=300)
    fig._uab_proof = (payload, provenance, data["style"])
    return fig


if __name__ == "__main__":
    figure = main()
    from visualization.project import write_proof
    write_proof(OUT / "class_diagnostics", figure, *figure._uab_proof)
    import matplotlib.pyplot as plt
    plt.close(figure)

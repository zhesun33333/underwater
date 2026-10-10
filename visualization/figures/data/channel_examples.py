"""Figure C: matched source and propagated WAVs. No-argument Tavotto entry."""
from pathlib import Path

OUT = Path(__file__).resolve().parent


def main():
    import sys
    repo = str(OUT.parents[2])
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from visualization.channel_data import load_channels
    from visualization.plot_channel import build_channel_figure
    from visualization.project import paper_style
    data, style, provenance = load_channels()
    with paper_style(style):
        fig, payload = build_channel_figure(data, style)
        fig.savefig(OUT / "channel_examples.pdf", dpi=300)
        fig.savefig(OUT / "channel_examples.png", dpi=300)
    fig._uab_proof = (payload, provenance, style)
    return fig


if __name__ == "__main__":
    figure = main()
    from visualization.project import write_proof
    write_proof(OUT / "channel_examples", figure, *figure._uab_proof)
    import matplotlib.pyplot as plt
    plt.close(figure)

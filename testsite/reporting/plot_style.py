"""Shared publication styling for UA-Bench figures."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager


NAVY = "#244A64"
BLUE = "#4C78A8"
TEAL = "#2A9D8F"
ORANGE = "#F28E2B"
RED = "#D1495B"
PURPLE = "#7A6FAC"
GRAY = "#8A94A3"
LIGHT_GRAY = "#DCE3E8"
INK = "#263238"

DEFAULT_FORMATS = ("png", "pdf")
DEFAULT_DPI = 300
FONT_CANDIDATES = (
    "Times New Roman",
    "Liberation Serif",
    "Nimbus Roman",
    "DejaVu Serif",
)


@lru_cache(maxsize=1)
def publication_font() -> str:
    """Return the first installed paper-style serif font without fallback logs."""
    for family in FONT_CANDIDATES:
        properties = font_manager.FontProperties(family=family)
        try:
            font_manager.findfont(properties, fallback_to_default=False)
        except ValueError:
            continue
        return family
    # Matplotlib bundles DejaVu Serif, so this is a defensive last resort.
    return "DejaVu Serif"


def apply_publication_style(
    *,
    font_size: float = 9,
    title_size: float = 11,
    label_size: float = 9.5,
    tick_size: float = 8.5,
    legend_size: float = 8.5,
) -> str:
    """Apply the common evaluation-figure style and return the chosen font."""
    family = publication_font()
    plt.rcParams.update({
        "font.family": family,
        "font.serif": [family],
        "mathtext.fontset": "stix",
        "font.size": font_size,
        "axes.titlesize": title_size,
        "axes.titleweight": "bold",
        "axes.labelsize": label_size,
        "axes.labelcolor": INK,
        "axes.edgecolor": "#AAB4BC",
        "axes.linewidth": 0.8,
        "patch.edgecolor": INK,
        "patch.linewidth": 0.65,
        "patch.force_edgecolor": True,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "xtick.color": INK,
        "ytick.color": INK,
        "legend.fontsize": legend_size,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    return family


def style_axis(
    ax,
    grid_axis: str = "y",
    *,
    minor_grid: bool = True,
    tick_length: float = 3.5,
    tick_width: float = 0.7,
) -> None:
    """Apply the shared boxed-axis treatment used by evaluation figures."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#8E99A1")
        spine.set_linewidth(0.7)
    ax.tick_params(
        direction="in", top=True, right=True,
        length=tick_length, width=tick_width,
    )
    ax.grid(
        axis=grid_axis, color="#C7CED4", linestyle="-.",
        linewidth=0.65, alpha=0.75,
    )
    if minor_grid:
        ax.minorticks_on()
        ax.grid(
            which="minor", axis=grid_axis, color=LIGHT_GRAY,
            linestyle=":", linewidth=0.45, alpha=0.55,
        )
    ax.set_axisbelow(True)


def save_figure(
    fig,
    output_prefix: Path,
    *,
    formats: Iterable[str] = DEFAULT_FORMATS,
    dpi: int = DEFAULT_DPI,
    close: bool = True,
) -> list[Path]:
    """Save a figure in the requested formats using consistent export rules."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for raw_format in formats:
        image_format = raw_format.strip().lower()
        if not image_format:
            continue
        path = prefix.with_suffix(f".{image_format}")
        fig.savefig(path, dpi=dpi if image_format == "png" else None)
        paths.append(path)
    if close:
        plt.close(fig)
    return paths

"""
Shared plotting helpers for publication-style figures.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


PAPER_COLORS = {
    "ink": "#213140",
    "muted": "#5B6B7A",
    "grid": "#D8DEE6",
    "map": "#1D4E89",
    "mtf": "#B64A3B",
    "threshold": "#C48A17",
    "baseline": "#7A8793",
}


PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "semibold",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.fontsize": 8.5,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": PAPER_COLORS["grid"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


@contextmanager
def paper_style():
    """Apply a restrained paper-oriented Matplotlib style within a context."""
    with mpl.rc_context(PAPER_RC):
        yield


def style_axes(ax, *, grid_axis: str = "y") -> None:
    """Apply shared styling to a primary axis."""
    ax.spines["left"].set_color(PAPER_COLORS["ink"])
    ax.spines["bottom"].set_color(PAPER_COLORS["ink"])
    ax.tick_params(axis="both", colors=PAPER_COLORS["ink"])
    if grid_axis:
        ax.grid(
            True,
            axis=grid_axis,
            color=PAPER_COLORS["grid"],
            linewidth=0.8,
            alpha=0.9,
        )


def style_twin_axis(ax, color: str) -> None:
    """Style a secondary y-axis while keeping the figure coherent."""
    ax.spines["right"].set_visible(True)
    ax.spines["right"].set_color(color)
    ax.tick_params(axis="y", colors=color)


def add_panel_label(ax, label: str) -> None:
    """Add a bold panel label just outside the upper-left of an axis."""
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        color=PAPER_COLORS["ink"],
        va="bottom",
        ha="left",
    )


def sample_cmap(name: str, n: int, start: float = 0.15, stop: float = 0.85) -> list:
    """Sample `n` visually separated colors from a Matplotlib colormap."""
    cmap = plt.get_cmap(name)
    if n == 1:
        return [cmap((start + stop) / 2.0)]
    return [cmap(v) for v in np.linspace(start, stop, n)]


def save_figure(fig, save_path: str, *, dpi: int = 300) -> None:
    """
    Save a figure to the requested path and a matching PDF sibling.

    This keeps raster previews for quick browsing while also producing a
    vector-friendly asset for the LaTeX report.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    pdf_path = path if path.suffix.lower() == ".pdf" else path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

"""Nature/NMI-style constants and SVG-only export helpers for Figure 5."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MM = 1.0 / 25.4

AXIS = "#202020"
GRID = "#C8C8C8"
DMF = "#E63946"
NEUTRAL = "#8D99AE"
BLUE = "#457B9D"
TEAL = "#2A9D8F"
VIOLET = "#7B6FA6"
AMBER = "#C98A3D"


def apply_style(font_family: list[str] | None = None) -> None:
    """Apply the manuscript-wide compact vector-figure style."""
    families = font_family or ["Arial", "Liberation Sans", "DejaVu Sans"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": families,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.0,
            "axes.titlesize": 6.5,
            "axes.labelsize": 6.0,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 5.5,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": AXIS,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(ax, label: str, *, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=AXIS,
        clip_on=False,
    )


def add_status_badge(
    ax,
    text: str,
    *,
    kind: str = "proxy",
    x: float = 0.99,
    y: float = 1.01,
    va: str = "bottom",
) -> None:
    colors = {
        "proxy": ("#7A5A00", "#FFF3CD", "#D9B44A"),
        "pending": ("#5C6770", "#F1F3F5", "#ADB5BD"),
        "formal": ("#1B6B3A", "#E7F5EC", "#7AC99A"),
    }
    fg, face, edge = colors[kind]
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="right",
        va=va,
        fontsize=4.8,
        color=fg,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": face, "edgecolor": edge, "linewidth": 0.55},
        clip_on=False,
    )


def style_grid(ax, *, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.45, alpha=0.55, zorder=0)
    ax.set_axisbelow(True)


def save_svg(fig, path: Path) -> None:
    """Save one fixed-canvas editable SVG and no secondary format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches=None, pad_inches=0)
    plt.close(fig)

"""Compact manuscript style helpers for Figure 5 V4.

The method palette is copied from the adopted Figure 4/V3 contract.  Keeping
the mapping here explicit makes standalone V4 renders independent of the
larger post-processing package while preventing panel-by-panel recolouring.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MM = 1.0 / 25.4
AXIS = "#202020"
GRID = "#C8C8C8"
DIVIDER = "#B8B8B8"
PENDING_FACE = "#F1F3F5"
PENDING_EDGE = "#ADB5BD"
DMF_HIGHLIGHT = "#E63946"


def apply_style(font_family: list[str] | None = None) -> None:
    """Apply the shared compact vector style at final 183-mm print size."""

    families = font_family or ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": families,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.1,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": AXIS,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.frameon": False,
            "lines.linewidth": 1.0,
            "lines.markersize": 4.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(ax, label: str, *, x: float = -0.10, y: float = 1.04) -> None:
    artist = ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.8,
        fontweight="bold",
        color=AXIS,
        clip_on=False,
    )
    artist.set_gid(f"font-role:panel_label:{label}")


def style_grid(ax, *, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.35, alpha=0.45, zorder=0)
    ax.set_axisbelow(True)


def method_colors(config: dict) -> dict[str, str]:
    return {str(k): str(v) for k, v in config["style"]["method_colors"].items()}


def method_markers(config: dict) -> dict[str, str]:
    return {str(k): str(v) for k, v in config["style"]["method_markers"].items()}


def method_alpha(method: str) -> float:
    return 0.88 if method == "DMF-Gen" else 0.78


def method_legend_handles(config: dict, methods: Iterable[str] | None = None) -> list[Line2D]:
    colors, markers = method_colors(config), method_markers(config)
    names = list(methods or config["paper_contract"]["method_order"])
    return [
        Line2D(
            [],
            [],
            color=colors[method],
            marker=markers[method],
            linestyle="-",
            linewidth=1.0,
            markersize=4.1,
            alpha=method_alpha(method),
            label=method,
        )
        for method in names
    ]


def save_svg(fig, path: Path) -> None:
    """Save one fixed-canvas editable SVG and close the figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches=None, pad_inches=0)
    plt.close(fig)

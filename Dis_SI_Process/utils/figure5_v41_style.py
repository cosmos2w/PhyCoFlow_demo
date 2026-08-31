"""Print-size typography and method identity for Figure 5 V4.1."""
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
DMF_HIGHLIGHT = "#E63946"


def apply_style(font_family: list[str] | None = None) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": font_family or ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.15,
            "axes.titlesize": 8.15,
            "axes.labelsize": 7.35,
            "xtick.labelsize": 6.75,
            "ytick.labelsize": 6.75,
            "legend.fontsize": 6.75,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": AXIS,
            "axes.linewidth": 0.72,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.major.width": 0.58,
            "ytick.major.width": 0.58,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "legend.frameon": False,
            "lines.linewidth": 1.05,
            "lines.markersize": 4.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def add_panel_label(ax, label: str, *, x: float = -0.105, y: float = 1.045) -> None:
    artist = ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color=AXIS,
        clip_on=False,
    )
    artist.set_gid(f"font-role:panel_label:{label}")


def style_grid(ax, *, axis: str = "both") -> None:
    ax.grid(True, axis=axis, color=GRID, linewidth=0.38, alpha=0.48, zorder=0)
    ax.set_axisbelow(True)


def method_colors(config: dict) -> dict[str, str]:
    return {str(key): str(value) for key, value in config["style"]["method_colors"].items()}


def method_markers(config: dict) -> dict[str, str]:
    return {str(key): str(value) for key, value in config["style"]["method_markers"].items()}


def method_alpha(method: str) -> float:
    return 0.92 if method == "DMF-Gen" else 0.80


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
            linewidth=1.05,
            markersize=4.5,
            alpha=method_alpha(method),
            label=method,
        )
        for method in names
    ]


def save_svg(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches=None, pad_inches=0)
    plt.close(fig)

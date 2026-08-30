"""Compatibility helpers backed by the portable manuscript global style.

New scripts should import :mod:`global_style` directly.  Existing plotting
scripts may continue importing this module; ``apply_style`` now delegates to
the same global contract and rewrites legacy YAML model colors consistently.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from global_style import (
    COLOR_MISSING_FACE,
    COLOR_MISSING_TEXT,
    COMPOSITE_WIDTH_IN,
    SINGLE_COLUMN_WIDTH_IN,
    SIZE_PANEL_LABEL,
    apply_global_style,
    save_publication_figure,
    standardize_model_colors,
)

MM = 1 / 25.4

# Reused in spectral curves so method identity is stable without relying on
# colour alone; ground truth is always black solid in the plot scripts.
METHOD_LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))


def method_line_style(index: int):
    """Return a deterministic shared line style for a method-order index."""
    return METHOD_LINESTYLES[index % len(METHOD_LINESTYLES)]


def apply_style(cfg: dict) -> None:
    """Apply global rcParams and map all configured methods to global colors."""
    apply_global_style()
    standardize_model_colors(cfg.get("methods", []))
    # Keep legacy scripts that still read YAML missing-data colors on the
    # manuscript-wide contract while those scripts are migrated incrementally.
    missing = cfg.setdefault("style", {}).setdefault("missing", {})
    missing["facecolor"] = COLOR_MISSING_FACE
    missing["textcolor"] = COLOR_MISSING_TEXT
    cfg.get("spectral", {}).get("plotting", {})["dmf_gen_accent"] = next(
        (method["color"] for method in cfg.get("methods", []) if method.get("name") == "DMF-Gen"),
        "#004488",
    )


def figure_size(cfg: dict, column: str = "double", height_mm: float | None = None):
    width_in = COMPOSITE_WIDTH_IN if column == "double" else SINGLE_COLUMN_WIDTH_IN
    return width_in, (float(height_mm) * MM if height_mm is not None else width_in * 0.62)


def save_figure(fig, base: Path, cfg: dict, formats=None, dpi: int | None = None, *, fixed_canvas: bool = False) -> list[Path]:
    """Save primary SVG plus requested secondary formats."""
    formats = formats or cfg["export"]["formats"]
    if "svg" not in formats:
        formats = ["svg", *formats]
    return save_publication_figure(
        fig, base, formats,
        dpi=dpi or cfg["export"]["default_dpi"],
        fixed_canvas=fixed_canvas,
    )


def add_panel_label(ax, label: str, cfg: dict, x=-0.08, y=1.03) -> None:
    ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=SIZE_PANEL_LABEL, fontweight="bold")


def mark_missing(ax, text: str = "Missing", cfg: dict | None = None) -> None:
    color = COLOR_MISSING_FACE
    ax.set_facecolor(color)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            color=COLOR_MISSING_TEXT, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def missing_cmap(base="viridis", color="#D8D8D8"):
    cmap = matplotlib.colormaps[base].copy()
    cmap.set_bad(color)
    return cmap

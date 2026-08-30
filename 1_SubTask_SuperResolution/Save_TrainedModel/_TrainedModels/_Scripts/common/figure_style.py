"""Project adapter for the manuscript-wide :mod:`global_style` contract.

All panel generators and assemblers import this module, so it deliberately
contains no independent typography, model palette, line-weight, or export
policy.  The small compatibility layer keeps the existing YAML-based drawing
API while making ``global_style.py`` authoritative.
"""
from __future__ import annotations

import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter

import global_style as manuscript

MM = 1 / 25.4

# Reused in spectral curves so method identity is stable without relying on
# colour alone; ground truth is always black solid in the plot scripts.
METHOD_LINESTYLES = ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))

# Shared semantic colors.  Model and recipe colors are read from YAML; these
# constants cover only stable visual roles that recur across panels.
RESOLUTION_COLORS = {"L": "#D8D8D8", "M": "#9AA7CF", "H": "#484878"}
NEUTRAL_LIGHT = manuscript.COLOR_MISSING_FACE
NEUTRAL_MID = manuscript.COLOR_MISSING_TEXT
NEUTRAL_DARK = manuscript.COLOR_AXIS

# Re-export the visual constants used by project-specific panel drawers.  This
# makes every local artist traceable to the single manuscript contract.
SIZE_PANEL_LABEL = manuscript.SIZE_PANEL_LABEL
SIZE_SUBPLOT_TITLE = manuscript.SIZE_SUBPLOT_TITLE
SIZE_AXIS_LABEL = manuscript.SIZE_AXIS_LABEL
SIZE_TICK_LABEL = manuscript.SIZE_TICK_LABEL
SIZE_LEGEND = manuscript.SIZE_LEGEND
SIZE_ANNOTATION = manuscript.SIZE_ANNOTATION
FONT_ROLE_SIZES = manuscript.FONT_ROLE_SIZES
LW_AXIS_SPINE = manuscript.LW_AXIS_SPINE
LW_LINE_PLOT = manuscript.LW_LINE_PLOT
LW_LINE_SECONDARY = manuscript.LW_LINE_SECONDARY
LW_DIVIDER = manuscript.LW_DIVIDER
LW_GRID = manuscript.LW_GRID
LW_ERRORBAR = manuscript.LW_ERRORBAR
COLOR_AXIS = manuscript.COLOR_AXIS
COLOR_GRID = manuscript.COLOR_GRID
COLOR_DIVIDER = manuscript.COLOR_DIVIDER
FIXED_CANVAS_LAYOUT_PAD_IN = manuscript.FIXED_CANVAS_LAYOUT_PAD_IN
fit_text_within_canvas = manuscript.fit_text_within_canvas


def compact_colorbar_ticks(
    colorbar,
    ticks,
    *,
    target_top_axis=None,
    tick_size_pt: float | None = None,
    decimals: int = 1,
    use_common_exponent: bool = True,
    preserve_bar_bottom: bool = False,
) -> dict:
    """Format compact ticks, optionally with one shared top-aligned exponent."""
    values = [float(value) for value in ticks]
    colorbar.set_ticks(values)
    finite_abs = [abs(value) for value in values if math.isfinite(value) and value != 0.0]
    exponent = (
        int(math.floor(math.log10(max(finite_abs))))
        if finite_abs and use_common_exponent else 0
    )
    scale = 10.0 ** exponent

    def format_mantissa(value, _position=None):
        scaled = float(value) / scale
        if abs(scaled) < 0.5 * 10.0 ** (-int(decimals)):
            scaled = 0.0
        return f"{scaled:.{int(decimals)}f}"

    axis = colorbar.ax.yaxis if colorbar.orientation == "vertical" else colorbar.ax.xaxis
    axis.set_major_formatter(FuncFormatter(format_mantissa))
    axis.offsetText.set_visible(False)
    resolved_tick_size = float(
        manuscript.SIZE_TICK_LABEL if tick_size_pt is None else tick_size_pt
    )
    colorbar.ax.tick_params(labelsize=resolved_tick_size)

    multiplier = None
    alignment_error_px = None
    if exponent != 0:
        if colorbar.orientation == "vertical":
            multiplier = colorbar.ax.text(
                0.5, 1.0, rf"$\times 10^{{{exponent}}}$",
                transform=colorbar.ax.transAxes, ha="center", va="bottom",
                fontsize=resolved_tick_size, clip_on=False,
                gid="colorbar-common-exponent",
            )
        else:
            multiplier = colorbar.ax.text(
                1.0, 1.0, rf"$\times 10^{{{exponent}}}$",
                transform=colorbar.ax.transAxes, ha="right", va="top",
                fontsize=resolved_tick_size, clip_on=True,
                gid="colorbar-common-exponent",
            )
        manuscript.tag_font_role(
            multiplier, "tick_label", size_pt=resolved_tick_size,
        )

    if colorbar.orientation == "vertical" and target_top_axis is not None:
        fig = colorbar.ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        target_top_px = float(target_top_axis.get_window_extent(renderer).y1)
        position = colorbar.ax.get_position()
        desired_bar_top_px = target_top_px
        if multiplier is not None:
            desired_bar_top_px -= multiplier.get_window_extent(renderer).height
        desired_bar_top_fig = fig.transFigure.inverted().transform(
            (0.0, desired_bar_top_px)
        )[1]
        if desired_bar_top_fig <= position.y0:
            raise ValueError("Colorbar multiplier leaves no positive bar height.")
        new_bottom = (
            position.y0 if preserve_bar_bottom
            else position.y0 + (position.y1 - desired_bar_top_fig)
        )
        if desired_bar_top_fig <= new_bottom:
            raise ValueError("Symmetric colorbar cap reservation leaves no positive bar height.")
        colorbar.ax.set_axes_locator(None)
        colorbar.ax.set_position([
            position.x0, new_bottom, position.width,
            desired_bar_top_fig - new_bottom,
        ])
        fig.canvas.draw()
        if multiplier is not None:
            renderer = fig.canvas.get_renderer()
            alignment_error_px = float(
                multiplier.get_window_extent(renderer).y1
                - target_top_axis.get_window_extent(renderer).y1
            )

    payload = {
        "exponent": exponent,
        "multiplier": None if exponent == 0 else rf"\times 10^{{{exponent}}}",
        "tick_labels": [format_mantissa(value) for value in values],
        "alignment_error_px": alignment_error_px,
        "uses_common_exponent": bool(use_common_exponent),
        "bar_bottom_preserved": bool(preserve_bar_bottom),
    }
    if multiplier is not None and target_top_axis is not None:
        setattr(multiplier, "_colorbar_target_top_axis", target_top_axis)
        setattr(multiplier, "_colorbar_format_payload", payload)
        setattr(multiplier, "_colorbar_preserve_bar_bottom", bool(preserve_bar_bottom))
    return payload


def finalize_colorbar_multiplier_alignment(fig) -> dict:
    """Re-align multipliers after final aspect/typography resolution."""
    multipliers = [
        text for text in fig.findobj(
            match=lambda item: isinstance(item, matplotlib.text.Text)
        )
        if text.get_gid() == "colorbar-common-exponent"
        and hasattr(text, "_colorbar_target_top_axis")
    ]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for multiplier in multipliers:
        cax = multiplier.axes
        target = multiplier._colorbar_target_top_axis
        target_top_px = float(target.get_window_extent(renderer).y1)
        desired_bar_top_px = target_top_px - multiplier.get_window_extent(renderer).height
        position = cax.get_position()
        desired_bar_top_fig = fig.transFigure.inverted().transform(
            (0.0, desired_bar_top_px)
        )[1]
        cax.set_axes_locator(None)
        new_bottom = (
            position.y0 if getattr(multiplier, "_colorbar_preserve_bar_bottom", False)
            else position.y0 + (position.y1 - desired_bar_top_fig)
        )
        if desired_bar_top_fig <= new_bottom:
            raise ValueError("Final colorbar alignment leaves no positive bar height.")
        cax.set_position([
            position.x0, new_bottom, position.width,
            desired_bar_top_fig - new_bottom,
        ])
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    alignments = []
    for multiplier in multipliers:
        error = float(
            multiplier.get_window_extent(renderer).y1
            - multiplier._colorbar_target_top_axis.get_window_extent(renderer).y1
        )
        multiplier._colorbar_format_payload["alignment_error_px"] = error
        alignments.append({
            "multiplier": multiplier.get_text(),
            "alignment_error_px": error,
        })
    result = {
        "passed": all(abs(item["alignment_error_px"]) <= 0.1 for item in alignments),
        "multiplier_count": len(alignments),
        "alignments": alignments,
    }
    if not result["passed"]:
        raise ValueError(f"Colorbar multiplier alignment failed: {result}")
    return result


def method_line_style(index: int):
    """Return a deterministic shared line style for a method-order index."""
    return METHOD_LINESTYLES[index % len(METHOD_LINESTYLES)]


def model_colors(cfg: dict) -> dict[str, str]:
    """Return the single model-color mapping used by every publication panel."""
    return {
        str(item["key"]): manuscript.model_color(
            item.get("label") or item.get("key") or item.get("directory"),
            item.get("color", "#777777"),
        )
        for item in cfg["models"]
    }


def model_alphas(cfg: dict) -> dict[str, float]:
    """Return model-artist transparency keyed by the project model key."""
    return {
        str(item["key"]): manuscript.model_alpha(
            item.get("label") or item.get("key") or item.get("directory", "")
        )
        for item in cfg["models"]
    }


def validate_model_line_contract(fig, cfg: dict) -> dict:
    """Validate every semantically tagged model curve against global style."""
    expected_colors = model_colors(cfg)
    expected_alphas = model_alphas(cfg)
    checked = []
    violations = []
    for line in fig.findobj(match=lambda item: isinstance(item, Line2D)):
        gid = str(line.get_gid() or "")
        if not gid.startswith("model-line:"):
            continue
        model = gid.split(":", 1)[1]
        expected_color = expected_colors[model]
        expected_alpha = float(expected_alphas[model])
        observed_alpha = 1.0 if line.get_alpha() is None else float(line.get_alpha())
        payload = {
            "model": model,
            "color": str(line.get_color()),
            "alpha": observed_alpha,
            "linewidth_pt": float(line.get_linewidth()),
        }
        checked.append(payload)
        if (
            to_rgba(line.get_color()) != to_rgba(expected_color)
            or abs(observed_alpha - expected_alpha) > 1.0e-9
            or abs(float(line.get_linewidth()) - LW_LINE_PLOT) > 1.0e-9
        ):
            violations.append({
                **payload,
                "expected_color": expected_color,
                "expected_alpha": expected_alpha,
                "expected_linewidth_pt": LW_LINE_PLOT,
            })
    result = {"passed": not violations, "checked_count": len(checked),
              "checked": checked, "violations": violations}
    if violations:
        raise ValueError(f"Model color/line contract failed: {violations}")
    return result


def condition_colors(cfg: dict) -> dict[str, str]:
    """Return the single recipe/condition-color mapping from YAML."""
    return {str(k): str(v) for k, v in cfg["figure_style"]["condition_colors"].items()}


def apply_style(cfg: dict) -> None:
    """Apply the global contract and bridge legacy configuration lookups."""
    manuscript.apply_global_style()
    manuscript.standardize_model_colors(cfg["models"])
    style = cfg.setdefault("figure_style", {})
    style.setdefault("font_sizes", {}).update({
        "body": SIZE_TICK_LABEL,
        "title": SIZE_SUBPLOT_TITLE,
        "axis": SIZE_AXIS_LABEL,
        "tick": SIZE_TICK_LABEL,
        "legend": SIZE_LEGEND,
        "panel": SIZE_PANEL_LABEL,
        "block": SIZE_SUBPLOT_TITLE,
        "annotation": SIZE_ANNOTATION,
    })
    style["line_width"] = LW_AXIS_SPINE
    style.setdefault("dimensions_mm", {}).update({
        "single": manuscript.SINGLE_COLUMN_WIDTH_IN / MM,
        "double": manuscript.COMPOSITE_WIDTH_IN / MM,
    })
    style["style_contract_version"] = manuscript.GLOBAL_STYLE_VERSION


def figure_size(cfg: dict, column: str = "double", height_mm: float | None = None):
    width = cfg["figure_style"]["dimensions_mm"][column]
    return width * MM, (height_mm or width * 0.62) * MM


def save_figure(fig, base: Path, cfg: dict, formats=None, dpi: int | None = None, bbox_inches="tight") -> list[Path]:
    """Save through the global editable-vector and fixed-canvas policy."""
    style = cfg["figure_style"]
    formats = formats or style["default_formats"]
    if "svg" not in formats:
        formats = ["svg", *formats]
    return manuscript.save_publication_figure(
        fig,
        base,
        formats=formats,
        dpi=dpi or style["default_dpi"],
        fixed_canvas=bbox_inches is None,
        validate_bbox=True,
    )


def style_manifest(cfg: dict | None = None) -> dict:
    """Return a compact, serializable record of the active visual contract."""
    payload = {
        "module": "global_style",
        "version": manuscript.GLOBAL_STYLE_VERSION,
        "font_family": manuscript.FONT_FAMILY,
        "font_fallbacks": list(manuscript.FONT_FALLBACKS),
        "font_sizes_pt": dict(manuscript.FONT_SIZES),
        "font_role_sizes_pt": dict(manuscript.FONT_ROLE_SIZES),
        "line_widths_pt": {
            "axis_spine": LW_AXIS_SPINE,
            "line_plot": LW_LINE_PLOT,
            "line_secondary": LW_LINE_SECONDARY,
            "divider": LW_DIVIDER,
            "grid": LW_GRID,
            "errorbar": LW_ERRORBAR,
        },
        "editable_text": {"svg_fonttype": "none", "pdf_fonttype": 42},
        "model_palette": dict(manuscript.MODEL_COLORS),
        "model_alpha": {
            "DMF-Gen": manuscript.ALPHA_DMF_GEN,
            "baseline_default": manuscript.ALPHA_BASELINE,
        },
        "semantic_colormaps": {
            "physical_field": manuscript.CMAP_PHYSICAL_FIELD,
            "signed_component": manuscript.CMAP_SIGNED_COMPONENT,
            "signed_residual": manuscript.CMAP_SIGNED_RESIDUAL,
            "correlation": manuscript.CMAP_CORRELATION,
            "signed_bias": manuscript.CMAP_SIGNED_BIAS,
        },
        "physical_layout_defaults": {
            "composite_width_mm": manuscript.COMPOSITE_CANVAS_WIDTH_MM,
            "inter_row_gap_mm": manuscript.DEFAULT_INTER_ROW_GAP_MM,
            "minimum_safe_gap_mm": manuscript.MINIMUM_SAFE_GAP_MM,
            "size_tolerance_mm": manuscript.PHYSICAL_SIZE_TOLERANCE_MM,
            "panel_label_xy": [manuscript.PANEL_LABEL_X, manuscript.PANEL_LABEL_Y],
        },
    }
    if cfg is not None:
        payload["resolved_project_model_palette"] = model_colors(cfg)
        payload["resolved_project_model_alpha"] = model_alphas(cfg)
    return payload


def add_panel_label(ax, label: str, cfg: dict, x=-0.08, y=1.03) -> None:
    manuscript.tag_font_role(
        ax.text(x, y, label, transform=ax.transAxes, ha="left", va="bottom",
                fontweight="bold", color=COLOR_AXIS),
        "panel_label",
    )


def mark_missing(ax, text: str = "Missing", cfg: dict | None = None) -> None:
    color = (cfg or {}).get("figure_style", {}).get("missing_facecolor", NEUTRAL_LIGHT)
    ax.set_facecolor(color)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            color=NEUTRAL_MID, fontsize=SIZE_ANNOTATION, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def missing_cmap(base="viridis", color="#D8D8D8"):
    cmap = matplotlib.colormaps[base].copy()
    cmap.set_bad(color)
    return cmap

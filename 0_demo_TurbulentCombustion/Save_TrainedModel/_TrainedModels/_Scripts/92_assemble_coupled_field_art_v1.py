#!/usr/bin/env python
"""Render the additive, art-style-only Figure 4 V1 revision.

The validated round-3 assembler remains untouched. This module loads it in the
current process, applies only visual constants, and delegates all source-data,
selection, normalization, and numerical work to that frozen implementation.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "91_assemble_coupled_field_publication.py"
DESIGN_WIDTH_IN = 180.0 / 25.4

MODEL_COLORS = {
    "DMF-Gen": "#C94053",
    "FFM-Perceiver": "#4C86A6",
    "FFM-FNO": "#425B76",
    "SiT": "#9B83C1",
    "Latent FM": "#725591",
    "Geo-FNO": "#DA9A66",
    "Senseiver": "#8D9BAD",
    "MLP-RBF": "#4C9E91",
}


def _load_base():
    spec = importlib.util.spec_from_file_location("coupled_field_round3_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen Figure 4 producer: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()


def _apply_art_contract() -> None:
    """Apply process-local manuscript styling without touching shared modules."""
    style = base.global_style_contract

    # Strict design width and protected lower-row canvas gutters.
    base.COMPOSITE_WIDTH_IN = DESIGN_WIDTH_IN
    style.COMPOSITE_WIDTH_IN = DESIGN_WIDTH_IN
    base.LOWER_ROW_MARGIN_LEFT_IN = 0.46
    base.LOWER_ROW_MARGIN_RIGHT_IN = 0.34
    base.LOWER_ROW_CANVAS_CLEARANCE_IN = 0.08
    base.RIGHT_COLUMN_CANVAS_CLEARANCE_IN = 0.06
    base.PANEL_B_TO_RIGHT_TEXT_CLEARANCE_IN = 0.12
    base.RIGHT_COLUMN_MANUAL_SHIFT_IN = 0.0
    base.AUTO_REFLOW_RIGHT_COLUMN_FOR_FONTS = True
    base.AUTO_REFLOW_SAFETY_PAD_IN = 0.01

    # Exact cross-figure categorical identity from style_contract.json.
    for mapping in (base.MODEL_COLORS, style.MODEL_COLORS):
        mapping.clear()
        mapping.update(MODEL_COLORS)
    base.MODEL_ALPHAS.update({"DMF-Gen": 0.95, "baseline_default": 0.82})
    style.MODEL_ALPHAS.update({"DMF-Gen": 0.95, "baseline_default": 0.82})
    base.model_alpha.__globals__["ALPHA_DMF_GEN"] = 0.95
    base.model_alpha.__globals__["ALPHA_BASELINE"] = 0.82

    # Hierarchical line weights, kept well above hairline export thresholds.
    line_values = {
        "LW_AXIS_SPINE": 0.65,
        "LW_LINE_PLOT": 1.60,
        "LW_LINE_SECONDARY": 1.10,
        "LW_DIVIDER": 0.50,
        "LW_GRID": 0.40,
        "LW_ERRORBAR": 0.75,
    }
    for name, value in line_values.items():
        setattr(base, name, value)
        setattr(style, name, value)
    base.LINE_WIDTHS.update({
        "lw_axis_spine": 0.65,
        "lw_line_plot": 1.60,
        "lw_divider": 0.50,
        "lw_grid": 0.40,
        "lw_errorbar": 0.75,
        "lw_line_secondary": 1.10,
    })
    style.LINE_WIDTHS.update(base.LINE_WIDTHS)

    # Semantic colormaps; all numerical limits and normalization objects remain
    # delegated to the round-3 producer.
    base.COLORMAP_CONFIG["panel_a"]["field_values"].update({
        "CH4": "viridis", "CO": "viridis", "T": "inferno",
        "U1": "RdBu_r", "p": "viridis",
    })
    base.COLORMAP_CONFIG["panel_a"]["absolute_error"] = "YlOrRd"
    base.COLORMAP_CONFIG["panel_b"].update({
        "selected": "YlOrRd", "options": ("YlOrRd", "Reds", "Greys"),
    })
    base.COLORMAP_CONFIG["panel_d"].update({
        "selected": "cividis",
        "options": ("cividis", "mako_r", "PuBu", "PuRd", "crest_r"),
    })

    def compact_l2(value: float) -> str:
        value = float(value)
        if not np.isfinite(value):
            return r"$L_2\;\mathrm{NaN}$"
        if value == 0.0:
            return r"$L_2\;0$"
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10.0 ** exponent)
        return rf"$L_2\;{mantissa:.2f}\!\times\!10^{{{exponent}}}$"

    base._format_l2_mathtext = compact_l2

    # Dense header and legend adjustments. Ordinary text stays >=7.8 pt at the
    # 180-mm design width (>=7.0 pt when inserted at 162 mm).
    original_overrides = base._apply_revision_overrides

    def art_overrides(layout: dict) -> None:
        original_overrides(layout)
        base.PANEL_A_LAYOUT.update({
            "column_header_fontsize": 8.0,
            "sit_enabled_column_header_fontsize": 8.0,
            "group_header_fontsize": 9.0,
            "group_to_column_header_gap_in": 0.25,
            "column_header_gap_above_grid_in": 0.055,
            "colorbar_width_ratio": 0.016,
            "colorbar_height_fraction": 0.78,
            "colorbar_title_pad": 1.8,
            "l2_bbox_alpha": 0.82,
            "l2_bbox_pad": 0.16,
            "divider_line_width": 0.50,
        })
        base.PANEL_C_LAYOUT.update({
            "legend_ncol": 3,
            "legend_mode": "expand",
            "legend_bbox_to_anchor": (0.0, 0.00, 1.0, 0.98),
            "height_ratios": (0.31, 1.0, 1.0),
        })
        base.PANEL_D_STYLE.update({
            "mean_fontsize": 7.8,
            "x_label_fontsize": 8.2,
            "x_label_pad": -1.0,
        })

    base._apply_revision_overrides = art_overrides

    # Consolidate the spectrum key into Panel c's dedicated legend band. This
    # removes the only legend drawn over data while retaining all identities.
    original_spectral = base.draw_spectral_panel

    def spectral_with_shared_key(fig, slot, cfg, layout, energy_rows, lsd_summary,
                                 lsd_per, selection, **kwargs):
        axes_before = set(fig.axes)
        original_spectral(
            fig, slot, cfg, layout, energy_rows, lsd_summary, lsd_per,
            selection, **kwargs,
        )
        created = [ax for ax in fig.axes if ax not in axes_before]
        legend_ax = next((ax for ax in created if not ax.axison), None)
        for ax in created:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        if legend_ax is not None:
            methods = list(base.method_items(cfg, None))
            handles = [Line2D([], [], color=base.COLOR_GROUND_TRUTH,
                              lw=base.LW_LINE_PLOT, label="Ground truth")]
            handles.extend(Line2D(
                [], [], color=method["color"],
                lw=base.LW_LINE_PLOT if method["name"] == "DMF-Gen" else base.LW_LINE_SECONDARY,
                alpha=base.model_alpha(method["name"]), label=method["name"],
            ) for method in methods)
            legend_ax.legend(
                handles=handles, ncol=3, mode="expand", loc="lower left",
                bbox_to_anchor=(0.0, 0.0, 1.0, 0.98),
                bbox_transform=legend_ax.transAxes, fontsize=base.SIZE_LEGEND,
                columnspacing=0.5, handletextpad=0.35, handlelength=1.35,
                borderaxespad=0.0, labelspacing=0.22, frameon=False,
            )

    base.draw_spectral_panel = spectral_with_shared_key

    # Keep all 24 Panel-d means but replace repeated "mu=" prefixes with one
    # small column key per subplot. This is a typographic, not numeric, change.
    original_jsd = base.draw_jsd_panel

    def jsd_with_compact_means(*args, **kwargs):
        result = original_jsd(*args, **kwargs)
        fig = args[0]
        for ax in getattr(fig, "_panel_d_violin_axes", []):
            found = False
            for artist in ax.texts:
                if artist.get_text().startswith("μ="):
                    compact = artist.get_text()[2:]
                    if compact.startswith("0."):
                        compact = compact[1:]
                    elif compact.startswith("-0."):
                        compact = "-." + compact[3:]
                    artist.set_text(compact)
                    artist.set_x(0.73)
                    found = True
            if found:
                ax.text(
                    0.73, 1.015, r"$\mu$", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=base.SIZE_ANNOTATION,
                    color=base.PANEL_D_STYLE["mean_color"], clip_on=False,
                )
        return result

    base.draw_jsd_panel = jsd_with_compact_means


def _postprocess(fig) -> dict:
    """Resolve display-only crowding and return renderer-measured QA facts."""
    wrapped_headers = 0
    l2_labels = 0
    for artist in fig.texts:
        if artist.get_text() == "CO + T + U1 + p":
            artist.set_text("CO + T +\n$U_1$ + $p$")
            artist.set_linespacing(0.92)
            wrapped_headers += 1
    for ax in fig.axes:
        for artist in ax.texts:
            if artist.get_text().startswith("$L_2"):
                artist.set_color("#202020")
                patch = artist.get_bbox_patch()
                if patch is not None:
                    patch.set_facecolor("white")
                    patch.set_edgecolor("none")
                    patch.set_alpha(0.82)
                l2_labels += 1
        # Strengthen the semantically separate observed-channel underline.
        for patch in ax.patches:
            if isinstance(patch, Rectangle) and abs(patch.get_width() - 0.96) < 1e-6 and abs(patch.get_height() - 0.06) < 1e-6:
                patch.set_facecolor("#147D92")
        # Keep every Panel-c state point while reducing its visual dominance.
        for collection in ax.collections:
            if isinstance(collection, PathCollection) and collection.get_alpha() == 0.32:
                collection.set_alpha(0.20)
                collection.set_sizes([1.0])

        # Move each scientific multiplier to a vertical label in the dedicated
        # right gutter, eliminating all map/title and inter-row collisions.
        for title in (ax.title, ax._left_title, ax._right_title):
            if title.get_text().startswith("$\\times 10^"):
                multiplier = title.get_text()
                title.set_text("")
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(
                    multiplier, rotation=90, fontsize=base.SIZE_ANNOTATION,
                    labelpad=12.0, ha="center", va="bottom",
                )

    violin_axes = list(getattr(fig, "_panel_d_violin_axes", []))
    for ax in violin_axes:
        for index, tick in enumerate(ax.get_xticklabels()):
            tick.set_rotation(0)
            tick.set_horizontalalignment("center")
            tick.set_y(-0.015 if index % 2 == 0 else -0.100)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = float(fig.dpi)
    canvas = fig.bbox
    text_overflow = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}
    visible_text = []
    for artist in fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text)):
        if not artist.get_visible() or not artist.get_text():
            continue
        bbox = artist.get_window_extent(renderer)
        if bbox.width <= 0 or bbox.height <= 0:
            continue
        visible_text.append((artist, bbox))
        text_overflow["left"] = max(text_overflow["left"], (canvas.x0 - bbox.x0) / dpi)
        text_overflow["right"] = max(text_overflow["right"], (bbox.x1 - canvas.x1) / dpi)
        text_overflow["bottom"] = max(text_overflow["bottom"], (canvas.y0 - bbox.y0) / dpi)
        text_overflow["top"] = max(text_overflow["top"], (bbox.y1 - canvas.y1) / dpi)

    panel_d_tick_overlap_count = 0
    for ax in violin_axes:
        boxes = [tick.get_window_extent(renderer) for tick in ax.get_xticklabels() if tick.get_visible()]
        for index, left in enumerate(boxes):
            for right in boxes[index + 1:]:
                if left.overlaps(right):
                    panel_d_tick_overlap_count += 1

    multiplier_count = 0
    multiplier_tick_overlap_count = 0
    for ax in fig.axes:
        label = ax.yaxis.label
        if not label.get_text().startswith("$\\times 10^"):
            continue
        multiplier_count += 1
        label_box = label.get_window_extent(renderer)
        multiplier_tick_overlap_count += sum(
            label_box.overlaps(tick.get_window_extent(renderer))
            for tick in ax.get_yticklabels() if tick.get_visible()
        )

    mean_count = 0
    mean_clip_count = 0
    for ax in violin_axes:
        axis_box = ax.get_window_extent(renderer)
        for artist in ax.texts:
            if artist.get_text() == r"$\mu$":
                continue
            if artist.get_transform() is ax.get_yaxis_transform() and artist.get_text():
                mean_count += 1
                box = artist.get_window_extent(renderer)
                if box.x0 < axis_box.x0 or box.x1 > axis_box.x1:
                    mean_clip_count += 1

    legend_data_overlap_count = 0
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        legend_box = legend.get_window_extent(renderer)
        for other in fig.axes:
            if other is ax or not other.axison:
                continue
            if legend_box.overlaps(other.get_window_extent(renderer)):
                legend_data_overlap_count += 1

    return {
        "wrapped_panel_a_headers": wrapped_headers,
        "panel_a_l2_label_count": l2_labels,
        "panel_d_tick_layout": "alternating two-row labels",
        "panel_d_tick_overlap_count": panel_d_tick_overlap_count,
        "panel_d_mean_count": mean_count,
        "panel_d_mean_clip_count": mean_clip_count,
        "panel_a_multiplier_count": multiplier_count,
        "panel_a_multiplier_tick_overlap_count": multiplier_tick_overlap_count,
        "legend_data_overlap_count": legend_data_overlap_count,
        "panel_a_to_lower_gap_mm": float(base.PANEL_A_LAYOUT["master_lower_gap_in"]) * 25.4,
        "panel_b_to_right_text_clearance_mm": float(
            getattr(fig, "_panel_b_to_right_text_clearance_in", 0.0)
        ) * 25.4,
        "panel_c_to_d_text_clearance_mm": float(
            getattr(fig, "_panel_c_d_text_clearance_in", 0.0)
        ) * 25.4,
        "lower_left_canvas_clearance_mm": float(
            getattr(fig, "_lower_row_left_canvas_clearance_in", 0.0)
        ) * 25.4,
        "right_canvas_clearance_mm": float(
            getattr(fig, "_right_column_canvas_clearance_in", 0.0)
        ) * 25.4,
        "text_overflow_in": {key: max(0.0, value) for key, value in text_overflow.items()},
        "visible_text_artist_count": len(visible_text),
    }


def _install_save_hook() -> None:
    original_save = base._save_figure

    def save_with_art_qa(fig, output_base: Path, formats: list[str], dpi: int):
        qa = _postprocess(fig)
        fig._art_style_v1_qa = qa
        outputs = original_save(fig, output_base, formats, dpi)
        qa_path = output_base.parent / f"{output_base.name}_art_qa.json"
        qa_path.write_text(
            json.dumps(
                qa, indent=2, sort_keys=True,
                default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
            ) + "\n",
            encoding="utf-8",
        )
        return outputs

    base._save_figure = save_with_art_qa


def main() -> int:
    _apply_art_contract()
    _install_save_hook()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Render Figure 4 art V2 with frozen round-3 scientific state."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, LogFormatterMathtext, NullLocator
import numpy as np


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "91_assemble_coupled_field_publication.py"
DESIGN_WIDTH_IN = 180.0 / 25.4
ORIGINAL_PANEL_A_HSPACE = 0.16
V2_PANEL_A_HSPACE = ORIGINAL_PANEL_A_HSPACE * 0.5
PANEL_A_FIELD_SPACER = 0.25
RECLAIMED_PANEL_A_HEIGHT_IN = 0.1645614035

MODEL_COLORS = {
    "DMF-Gen": "#C94053", "FFM-Perceiver": "#4C86A6",
    "FFM-FNO": "#425B76", "SiT": "#9B83C1",
    "Latent FM": "#725591", "Geo-FNO": "#DA9A66",
    "Senseiver": "#8D9BAD", "MLP-RBF": "#4C9E91",
}


def _load_base():
    spec = importlib.util.spec_from_file_location("coupled_field_round3_for_art_v2", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen producer: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_base()


def _compact_l2(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return r"$L_2\;\mathrm{NaN}$"
    if value == 0.0:
        return r"$L_2\;0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    return rf"$L_2\;{mantissa:.2f}\!\times\!10^{{{exponent}}}$"


def _log_ticks_at_most_four(ax) -> list[float]:
    ymin, ymax = ax.get_ylim()
    # Thin only the producer's existing major levels.  This changes displayed
    # label density, not the log transform, limits, or any scientific trace.
    candidates = np.asarray([
        value for value in ax.get_yticks()
        if np.isfinite(value) and value > 0.0 and ymin <= value <= ymax
    ], dtype=float)
    if candidates.size > 4:
        indices = np.unique(np.rint(np.linspace(0, candidates.size - 1, 4)).astype(int))
        candidates = candidates[indices]
    ticks = candidates.tolist()
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(NullLocator())
    return ticks


def _reflow_panel_a_without_colorbars(fig, slot, axes, texts, divider_artists) -> None:
    colorbars = [ax for ax in axes if len(ax.get_yticks()) and ax.get_position().width < 0.02]
    maps = [ax for ax in axes if ax not in colorbars]
    if len(colorbars) != 6 or len(maps) != 45:
        raise RuntimeError(
            f"Unexpected Panel-a inventory: {len(maps)} maps and {len(colorbars)} colorbars"
        )
    right = max(ax.get_position().x1 for ax in colorbars)
    left = min(ax.get_position().x0 for ax in maps)
    widths = np.asarray([ax.get_position().width for ax in maps])
    width = float(np.median(widths))
    old_centers = sorted({round((ax.get_position().x0 + ax.get_position().x1) / 2.0, 7) for ax in maps})
    if len(old_centers) != 8 or not np.allclose(widths, width, atol=2e-5):
        raise RuntimeError("Panel-a map columns are not the expected eight equal widths")
    new_gap = (right - left - 8.0 * width) / 7.0
    if new_gap <= 0:
        raise RuntimeError("Removing Panel-a colorbars did not leave positive inter-column space")
    new_centers = [left + width / 2.0 + index * (width + new_gap) for index in range(8)]
    for ax in maps:
        box = ax.get_position()
        old_center = (box.x0 + box.x1) / 2.0
        index = int(np.argmin(np.abs(np.asarray(old_centers) - old_center)))
        ax.set_position([new_centers[index] - box.width / 2.0, box.y0, box.width, box.height], which="both")
    for ax in colorbars:
        ax.remove()

    for artist in texts:
        text = artist.get_text()
        if text == "Conditioning progression (DMF-Gen)":
            artist.set_x((new_centers[1] + new_centers[3]) / 2.0)
            continue
        if text == "Baseline comparisons: conditioned on T only":
            artist.set_x((new_centers[4] + new_centers[7]) / 2.0)
            continue
        old_x = float(artist.get_position()[0])
        index = int(np.argmin(np.abs(np.asarray(old_centers) - old_x)))
        artist.set_x(new_centers[index])
    divider_x = (new_centers[3] + new_centers[4]) / 2.0
    for artist in divider_artists:
        if isinstance(artist, Line2D):
            artist.set_xdata([divider_x, divider_x])

    fig._panel_a_visible_colorbar_count = 0  # type: ignore[attr-defined]
    fig._panel_a_map_count = len(maps)  # type: ignore[attr-defined]
    fig._panel_a_intercolumn_gap_mm = new_gap * float(fig.get_size_inches()[0]) * 25.4  # type: ignore[attr-defined]
    fig._panel_a_map_widths_before_after_in = {  # type: ignore[attr-defined]
        "before": widths.tolist(), "after": [ax.get_position().width for ax in maps],
    }


def _apply_art_contract() -> None:
    style = base.global_style_contract
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

    for mapping in (base.MODEL_COLORS, style.MODEL_COLORS):
        mapping.clear(); mapping.update(MODEL_COLORS)
    base.MODEL_ALPHAS.update({"DMF-Gen": 0.95, "baseline_default": 0.82})
    style.MODEL_ALPHAS.update({"DMF-Gen": 0.95, "baseline_default": 0.82})
    base.model_alpha.__globals__["ALPHA_DMF_GEN"] = 0.95
    base.model_alpha.__globals__["ALPHA_BASELINE"] = 0.82

    line_values = {
        "LW_AXIS_SPINE": 0.65, "LW_LINE_PLOT": 1.60,
        "LW_LINE_SECONDARY": 1.10, "LW_DIVIDER": 0.50,
        "LW_GRID": 0.40, "LW_ERRORBAR": 0.75,
    }
    for name, value in line_values.items():
        setattr(base, name, value); setattr(style, name, value)
    base.LINE_WIDTHS.update({name.lower(): value for name, value in line_values.items()})
    style.LINE_WIDTHS.update(base.LINE_WIDTHS)

    # Exact round-3 scalar lookups for Panels a/b; no normalization changes.
    base.COLORMAP_CONFIG["panel_a"]["field_values"].update({
        "CH4": "crest", "CO": "YlGnBu", "T": "inferno",
        "U1": "RdBu_r", "p": "viridis",
    })
    base.COLORMAP_CONFIG["panel_a"]["absolute_error"] = "OrRd"
    base.COLORMAP_CONFIG["panel_b"].update({
        "selected": "Reds", "options": ("YlOrRd", "Reds", "Greys"),
    })
    base.COLORMAP_CONFIG["panel_d"].update({
        "selected": "cividis",
        "options": ("cividis", "mako_r", "PuBu", "PuRd", "crest_r"),
    })
    base._format_l2_mathtext = _compact_l2

    original_overrides = base._apply_revision_overrides

    def art_overrides(layout: dict) -> None:
        # The frozen round-3 producer predates the project-wide V2 role model
        # and asserts that subplot titles must be larger than axis titles.  V2
        # intentionally puts those two roles in one 8.5-pt tier.  Satisfy the
        # legacy parser with an epsilon-only copy, then restore the exact V2
        # values before any artist is created.
        parsed_layout = dict(layout)
        typography = dict(layout.get("typography_overrides_pt", {}))
        subplot_size = float(typography.get("size_subplot_title", base.SIZE_SUBPLOT_TITLE))
        axis_size = float(typography.get("size_axis_label", base.SIZE_AXIS_LABEL))
        if subplot_size <= axis_size:
            parsed_typography = dict(typography)
            parsed_typography["size_subplot_title"] = axis_size + 1.0e-6
            parsed_layout["typography_overrides_pt"] = parsed_typography
        original_overrides(parsed_layout)
        if subplot_size <= axis_size:
            base.SIZE_SUBPLOT_TITLE = subplot_size
            style.SIZE_SUBPLOT_TITLE = subplot_size
            base.FONT_SIZES["size_subplot_title"] = subplot_size
            style.FONT_SIZES["size_subplot_title"] = subplot_size
        base.PANEL_A_LAYOUT.update({
            "master_top_margin_in": 0.18,
            "column_header_fontsize": 8.5,
            "sit_enabled_column_header_fontsize": 8.5,
            "group_header_fontsize": 9.5,
            "group_to_column_header_gap_in": 0.25,
            "column_header_gap_above_grid_in": 0.055,
            "field_group_spacer_ratio": PANEL_A_FIELD_SPACER,
            "l2_bbox_alpha": 0.82, "l2_bbox_pad": 0.14,
            "divider_line_width": 0.50,
        })
        base.PANEL_C_LAYOUT.update({
            "legend_ncol": 4, "legend_mode": "expand",
            "legend_bbox_to_anchor": (0.0, 0.0, 1.0, 0.98),
        })
        base.PANEL_D_STYLE.update({
            "violin_width": 0.46,
            "violin_data_right_fraction": 0.50,
            "mean_x_axes": 0.66,
            "mean_fontsize": 7.0,
            "x_label_fontsize": 8.5,
            "x_label_pad": 1.0,
        })

    base._apply_revision_overrides = art_overrides

    original_qualitative = base.draw_qualitative_panel

    def qualitative_without_colorbars(fig, slot, *args, **kwargs):
        axes_before = set(fig.axes)
        texts_before = set(fig.texts)
        artists_before = set(fig.artists)
        result = original_qualitative(fig, slot, *args, **kwargs)
        _reflow_panel_a_without_colorbars(
            fig, slot,
            [ax for ax in fig.axes if ax not in axes_before],
            [text for text in fig.texts if text not in texts_before],
            [artist for artist in fig.artists if artist not in artists_before],
        )
        return result

    base.draw_qualitative_panel = qualitative_without_colorbars

    original_spectral = base.draw_spectral_panel

    def spectral_with_two_legends(fig, slot, cfg, layout, energy_rows, lsd_summary,
                                  lsd_per, selection, **kwargs):
        del kwargs
        grid = slot.subgridspec(
            4, 3, height_ratios=[0.24, 1.0, 0.18, 1.0],
            hspace=0.30, wspace=base.INTRA_PANEL_WSPACE,
        )
        axes_before = set(fig.axes)
        original_spectral(
            fig, slot, cfg, layout, energy_rows, lsd_summary, lsd_per, selection,
            legend_slot=grid[0, :],
            bar_slots=[grid[1, index] for index in range(3)],
            spectrum_slots=[grid[3, index] for index in range(3)],
        )
        created = [ax for ax in fig.axes if ax not in axes_before]
        legend_ax = next(ax for ax in created if not ax.axison)
        for ax in created:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        methods = list(base.method_items(cfg, None))
        bar_handles = [Line2D(
            [], [], marker="s", linestyle="None", markersize=5.0,
            markerfacecolor=(cfg["spectral"]["plotting"]["dmf_gen_accent"] if method["name"] == "DMF-Gen" else method["color"]),
            markeredgecolor="none", alpha=base.model_alpha(method["name"]), label=method["name"],
        ) for method in methods]
        bar_legend = legend_ax.legend(
            handles=bar_handles, ncol=4, mode="expand", loc="lower left",
            bbox_to_anchor=(0.0, 0.0, 1.0, 0.98), bbox_transform=legend_ax.transAxes,
            fontsize=base.SIZE_LEGEND, handlelength=0.72,
            columnspacing=0.45, handletextpad=0.30, labelspacing=0.16,
            borderaxespad=0.0, frameon=False,
        )

        line_ax = fig.add_subplot(grid[2, :]); line_ax.set_axis_off()
        selected_names = ["DMF-Gen", selection["generative"], selection["deterministic"]]
        selected_methods = [next(method for method in methods if method["name"] == name)
                            for name in dict.fromkeys(selected_names)]
        line_handles = [Line2D([], [], color=base.COLOR_GROUND_TRUTH,
                               lw=base.LW_LINE_PLOT, label="Ground truth")]
        for index, method in enumerate(selected_methods):
            line_handles.append(Line2D(
                [], [], color=method["color"], alpha=base.model_alpha(method["name"]),
                lw=base.LW_LINE_PLOT if method["name"] == "DMF-Gen" else base.LW_LINE_SECONDARY,
                linestyle=base.method_line_style(index), label=method["name"],
            ))
        line_legend = line_ax.legend(
            handles=line_handles, ncol=4, mode="expand", loc="center",
            bbox_to_anchor=(0.0, 0.0, 1.0, 1.0), bbox_transform=line_ax.transAxes,
            fontsize=base.SIZE_LEGEND, handlelength=1.35, handletextpad=0.30,
            columnspacing=0.45, borderaxespad=0.0, frameon=False,
        )
        spectra = [ax for ax in created if ax.get_xscale() == "log" and ax.get_yscale() == "log"]
        ticks = [_log_ticks_at_most_four(ax) for ax in spectra]
        fig._panel_c_bar_legend = bar_legend  # type: ignore[attr-defined]
        fig._panel_c_line_legend = line_legend  # type: ignore[attr-defined]
        fig._panel_c_spectrum_axes = spectra  # type: ignore[attr-defined]
        fig._panel_c_spectral_y_ticks = ticks  # type: ignore[attr-defined]

    base.draw_spectral_panel = spectral_with_two_legends

    original_panel_d_slots = base._panel_d_grid_slots

    def panel_d_slots_with_wider_columns(fig, slot, column_count):
        previous_wspace = base.INTRA_PANEL_WSPACE
        base.INTRA_PANEL_WSPACE = 0.06
        try:
            return original_panel_d_slots(fig, slot, column_count)
        finally:
            base.INTRA_PANEL_WSPACE = previous_wspace

    base._panel_d_grid_slots = panel_d_slots_with_wider_columns

    original_jsd = base.draw_jsd_panel

    def jsd_with_mean_corridor(*args, **kwargs):
        result = original_jsd(*args, **kwargs)
        fig = args[0]
        for ax in getattr(fig, "_panel_d_violin_axes", []):
            for artist in ax.texts:
                if not artist.get_text().startswith("μ="):
                    continue
                value = artist.get_text()[2:]
                if value.startswith("0."):
                    value = value[1:]
                artist.set_text(r"$\mu$=" + value)
                artist.set_x(float(base.PANEL_D_STYLE["mean_x_axes"]))
        return result

    base.draw_jsd_panel = jsd_with_mean_corridor


def _relative_luminance(rgb) -> float:
    values = np.asarray(rgb[:3], dtype=float)
    values = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    return float(values @ np.asarray([0.2126, 0.7152, 0.0722]))


def _postprocess(fig) -> dict:
    for artist in fig.texts:
        if artist.get_text() == "CO + T + U1 + p":
            artist.set_text("CO + T +\n$U_1$ + $p$")
            artist.set_linespacing(0.82)
        if artist.get_text() in {
            "Conditioning progression (DMF-Gen)",
            "Baseline comparisons: conditioned on T only",
        }:
            artist.set_fontweight("normal")
            artist.set_y(artist.get_position()[1] + 0.060 / float(fig.get_size_inches()[1]))

    heatmap_value_count = 0
    heatmap_white_count = 0
    heatmap_black_count = 0
    heatmap_min_contrast = float("inf")
    l2_count = 0
    for ax in fig.axes:
        for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
            tick.set_fontweight("normal")
        for artist in ax.texts:
            if artist.get_text().startswith("$L_2"):
                artist.set_color("#202020"); l2_count += 1
                patch = artist.get_bbox_patch()
                if patch is not None:
                    patch.set_facecolor("white"); patch.set_edgecolor("none"); patch.set_alpha(0.82)
        for patch in ax.patches:
            if isinstance(patch, Rectangle) and abs(patch.get_width() - 0.96) < 1e-6 and abs(patch.get_height() - 0.06) < 1e-6:
                patch.set_facecolor("#147D92")
        for collection in ax.collections:
            if isinstance(collection, PathCollection) and collection.get_alpha() == 0.32:
                collection.set_alpha(0.16); collection.set_sizes([1.0])

        if len(ax.images) != 1:
            continue
        rgba = np.asarray(ax.images[0].get_array())
        if rgba.shape != (8, 6, 4):
            continue
        for artist in ax.texts:
            try:
                float(artist.get_text())
            except ValueError:
                continue
            x, y = artist.get_position()
            row, column = int(round(float(y))), int(round(float(x)))
            if not (0 <= row < 8 and 0 <= column < 6):
                continue
            luminance = _relative_luminance(rgba[row, column])
            white_contrast = 1.05 / (luminance + 0.05)
            black_contrast = (luminance + 0.05) / 0.05
            use_white = white_contrast >= black_contrast
            artist.set_color("white" if use_white else "#202020")
            artist.set_fontsize(base.SIZE_ANNOTATION)
            heatmap_value_count += 1
            heatmap_white_count += int(use_white)
            heatmap_black_count += int(not use_white)
            heatmap_min_contrast = min(heatmap_min_contrast, white_contrast if use_white else black_contrast)

    violin_axes = list(getattr(fig, "_panel_d_violin_axes", []))
    for ax in violin_axes:
        for index, tick in enumerate(ax.get_xticklabels()):
            tick.set_rotation(0); tick.set_horizontalalignment("center")
            tick.set_y(-0.015 if index % 2 == 0 else -0.100)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = float(fig.dpi)
    canvas = fig.bbox
    visible_text = []
    tick_owner = {}
    for axis_index, ax in enumerate(fig.axes):
        for axis_name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                for label in (tick.label1, tick.label2):
                    tick_owner[id(label)] = (axis_index, axis_name, float(tick.get_loc()))
    text_overflow = {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}
    for artist in fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text)):
        if not artist.get_visible() or not artist.get_text():
            continue
        owner = tick_owner.get(id(artist))
        if owner is not None:
            axis_index, axis_name, location = owner
            ax = fig.axes[axis_index]
            if not ax.axison:
                continue
            lower, upper = ax.get_xlim() if axis_name == "x" else ax.get_ylim()
            if not min(lower, upper) <= location <= max(lower, upper):
                continue
        box = artist.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        visible_text.append((artist, box))
        text_overflow["left"] = max(text_overflow["left"], (canvas.x0 - box.x0) / dpi)
        text_overflow["right"] = max(text_overflow["right"], (box.x1 - canvas.x1) / dpi)
        text_overflow["bottom"] = max(text_overflow["bottom"], (canvas.y0 - box.y0) / dpi)
        text_overflow["top"] = max(text_overflow["top"], (box.y1 - canvas.y1) / dpi)

    text_text_overlaps = []
    for index, (left_artist, left_box) in enumerate(visible_text):
        for right_artist, right_box in visible_text[index + 1:]:
            intersection_width = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
            intersection_height = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
            if intersection_width <= 0.5 or intersection_height <= 0.5:
                continue
            text_text_overlaps.append([
                {
                    "text": left_artist.get_text(),
                    "position": [float(value) for value in left_artist.get_position()],
                    "axes_index": fig.axes.index(left_artist.axes) if left_artist.axes in fig.axes else None,
                    "tick_owner": tick_owner.get(id(left_artist)),
                    "bbox_px": [float(value) for value in left_box.extents],
                },
                {
                    "text": right_artist.get_text(),
                    "position": [float(value) for value in right_artist.get_position()],
                    "axes_index": fig.axes.index(right_artist.axes) if right_artist.axes in fig.axes else None,
                    "tick_owner": tick_owner.get(id(right_artist)),
                    "bbox_px": [float(value) for value in right_box.extents],
                },
                {
                    "intersection_width_mm": float(intersection_width / dpi * 25.4),
                    "intersection_height_mm": float(intersection_height / dpi * 25.4),
                },
            ])

    unexpected_bold_texts = []
    for artist, _ in visible_text:
        weight = artist.get_fontweight()
        is_bold = weight in {"bold", "semibold", "demibold", 600, 700, 800, 900}
        if is_bold and artist.get_text() not in {"a", "b", "c", "d"}:
            unexpected_bold_texts.append(artist.get_text())

    group_titles = [item for item in visible_text if item[0].get_text() in {
        "Conditioning progression (DMF-Gen)",
        "Baseline comparisons: conditioned on T only",
    }]
    panel_a_headers = [item for item in visible_text if item[0] in fig.texts and item[0].get_text() in {
        "Ground truth", "T only", "T + U1", "CO + T +\n$U_1$ + $p$",
        "FFM-FNO", "Latent FM", "SiT", "Senseiver",
    }]
    group_header_gaps_mm = []
    for _, group_box in group_titles:
        subordinate = [header_box for _, header_box in panel_a_headers
                       if group_box.x0 <= (header_box.x0 + header_box.x1) / 2.0 <= group_box.x1]
        if subordinate:
            group_header_gaps_mm.append((group_box.y0 - max(box.y1 for box in subordinate)) / dpi * 25.4)

    tick_overlaps = 0
    mean_clips = 0
    mean_offsets_mm = []
    for ax in violin_axes:
        tick_boxes = [tick.get_window_extent(renderer) for tick in ax.get_xticklabels() if tick.get_visible()]
        tick_overlaps += sum(left.overlaps(right) for index, left in enumerate(tick_boxes) for right in tick_boxes[index + 1:])
        axis_box = ax.get_window_extent(renderer)
        body_right = max((body.get_window_extent(renderer).x1 for body in ax.collections), default=axis_box.x0)
        for artist in ax.texts:
            if not artist.get_text().startswith(r"$\mu$="):
                continue
            box = artist.get_window_extent(renderer)
            mean_clips += int(box.x0 < axis_box.x0 or box.x1 > axis_box.x1)
            mean_offsets_mm.append((box.x0 - body_right) / dpi * 25.4)

    bar_legend = getattr(fig, "_panel_c_bar_legend", None)
    line_legend = getattr(fig, "_panel_c_line_legend", None)
    legend_data_overlaps = 0
    for legend in (bar_legend, line_legend):
        if legend is None:
            continue
        legend_box = legend.get_window_extent(renderer)
        for ax in fig.axes:
            if not ax.axison:
                continue
            if legend_box.overlaps(ax.get_window_extent(renderer)):
                legend_data_overlaps += 1

    map_widths = getattr(fig, "_panel_a_map_widths_before_after_in", {"before": [], "after": []})
    map_width_preserved = bool(map_widths["before"]) and np.allclose(
        sorted(map_widths["before"]), sorted(map_widths["after"]), atol=1e-10,
    )
    return {
        "revision": "art_v2",
        "text_role_sizes_pt": {
            "panel_label_bold": 11.0, "major_title_regular": 9.5,
            "axis_or_subplot_title_regular": 8.5,
            "tick_legend_method_regular": 7.8, "annotation_regular": 7.0,
        },
        "dynamic_annotation_reductions": [],
        "text_overflow_in": {key: max(0.0, float(value)) for key, value in text_overflow.items()},
        "text_text_overlap_count": len(text_text_overlaps),
        "text_text_overlaps": text_text_overlaps,
        "unexpected_bold_texts": unexpected_bold_texts,
        "panel_a_map_count": int(getattr(fig, "_panel_a_map_count", -1)),
        "panel_a_visible_colorbar_count": int(getattr(fig, "_panel_a_visible_colorbar_count", -1)),
        "panel_a_intercolumn_gap_mm": float(getattr(fig, "_panel_a_intercolumn_gap_mm", 0.0)),
        "panel_a_map_width_preserved": bool(map_width_preserved),
        "panel_a_hspace_before": ORIGINAL_PANEL_A_HSPACE,
        "panel_a_hspace_after": V2_PANEL_A_HSPACE,
        "panel_a_field_spacer_unchanged": PANEL_A_FIELD_SPACER,
        "panel_a_reclaimed_height_in": RECLAIMED_PANEL_A_HEIGHT_IN,
        "panel_a_lower_gap_in": float(base.PANEL_A_LAYOUT["master_lower_gap_in"]),
        "panel_a_to_lower_gap_mm": float(base.PANEL_A_LAYOUT["master_lower_gap_in"]) * 25.4,
        "panel_a_group_header_min_vertical_gap_mm": min(group_header_gaps_mm) if group_header_gaps_mm else None,
        "panel_a_colormaps": {
            "field_values": {key: base.COLORMAP_CONFIG["panel_a"]["field_values"][key]
                             for key in ("CH4", "p", "U1")},
            "absolute_error": base.COLORMAP_CONFIG["panel_a"]["absolute_error"],
        },
        "panel_a_l2_label_count": l2_count,
        "panel_b_matrix_value_count": heatmap_value_count,
        "panel_b_white_text_count": heatmap_white_count,
        "panel_b_black_text_count": heatmap_black_count,
        "panel_b_min_contrast_ratio": heatmap_min_contrast,
        "panel_b_colormap": base.COLORMAP_CONFIG["panel_b"]["selected"],
        "panel_c_bar_legend_handle_count": len(bar_legend.legend_handles) if bar_legend else 0,
        "panel_c_bar_legend_handle_types": [type(handle).__name__ for handle in bar_legend.legend_handles] if bar_legend else [],
        "panel_c_bar_legend_markers": [handle.get_marker() for handle in bar_legend.legend_handles] if bar_legend else [],
        "panel_c_bar_legend_linestyles": [handle.get_linestyle() for handle in bar_legend.legend_handles] if bar_legend else [],
        "panel_c_line_legend_handle_count": len(line_legend.legend_handles) if line_legend else 0,
        "panel_c_line_legend_ncol": 4,
        "panel_c_spectral_y_tick_counts": [len(values) for values in getattr(fig, "_panel_c_spectral_y_ticks", [])],
        "panel_c_legend_data_overlap_count": legend_data_overlaps,
        "panel_d_violin_width": float(base.PANEL_D_STYLE["violin_width"]),
        "panel_d_data_right_fraction": float(base.PANEL_D_STYLE["violin_data_right_fraction"]),
        "panel_d_mean_x_fraction": float(base.PANEL_D_STYLE["mean_x_axes"]),
        "panel_d_mean_count": len(mean_offsets_mm),
        "panel_d_mean_clip_count": mean_clips,
        "panel_d_min_mean_offset_mm": min(mean_offsets_mm) if mean_offsets_mm else None,
        "panel_d_tick_overlap_count": tick_overlaps,
        "panel_d_tick_layout": "alternating two rows with dedicated 0.06 column wspace",
        "panel_d_column_wspace": 0.06,
        "panel_b_to_right_text_clearance_mm": float(getattr(fig, "_panel_b_to_right_text_clearance_in", 0.0)) * 25.4,
        "panel_c_to_d_text_clearance_mm": float(getattr(fig, "_panel_c_d_text_clearance_in", 0.0)) * 25.4,
        "lower_left_canvas_clearance_mm": float(getattr(fig, "_lower_row_left_canvas_clearance_in", 0.0)) * 25.4,
        "right_canvas_clearance_mm": float(getattr(fig, "_right_column_canvas_clearance_in", 0.0)) * 25.4,
    }


def _install_save_hook() -> None:
    original_save = base._save_figure

    def save_with_qa(fig, output_base: Path, formats: list[str], dpi: int):
        qa = _postprocess(fig)
        outputs = original_save(fig, output_base, formats, dpi)
        (output_base.parent / f"{output_base.name}_art_qa.json").write_text(
            json.dumps(qa, indent=2, sort_keys=True,
                       default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n",
            encoding="utf-8",
        )
        return outputs

    base._save_figure = save_with_qa


def main() -> int:
    _apply_art_contract()
    _install_save_hook()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

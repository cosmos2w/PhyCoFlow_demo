#!/usr/bin/env python
"""Render Figure 4 art V6 with four sequential full-width panel rows.

The V5 scientific/data pipeline is retained verbatim.  This module only
repositions the already-created artists for Panels b--d and consolidates their
legends; Panel a continues to be constructed by the validated V5 producer at
its exact physical size and top offset.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
import numpy as np


HERE = Path(__file__).resolve().parent
V5_PATH = HERE / "100_assemble_coupled_field_art_v5.py"

CANVAS_HEIGHT_MM = 271.0
PAGE_WIDTH_MM = 180.0
PANEL_TAG_X = 0.004
AXES_LEFT_MM = 25.0
AXES_RIGHT_MM = 178.0
LEGEND_LEFT_MM = 9.0
LEGEND_RIGHT_MM = 179.0
B_GAP_MM = 5.0
C_D_GAP_MM = 9.0


def _load_v5():
    spec = importlib.util.spec_from_file_location("coupled_field_art_v5_for_v6", V5_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V5 renderer: {V5_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v5 = _load_v5()
v4 = v5.v4
v3 = v5.v3
v2 = v5.v2
base = v5.base


def _box_from_top(top_mm: float, height_mm: float, *, left_mm: float, width_mm: float) -> list[float]:
    return [
        left_mm / PAGE_WIDTH_MM,
        (CANVAS_HEIGHT_MM - top_mm - height_mm) / CANVAS_HEIGHT_MM,
        width_mm / PAGE_WIDTH_MM,
        height_mm / CANVAS_HEIGHT_MM,
    ]


def _equal_column_boxes(top_mm: float, height_mm: float, gap_mm: float) -> list[list[float]]:
    width_mm = (AXES_RIGHT_MM - AXES_LEFT_MM - 2.0 * gap_mm) / 3.0
    return [
        _box_from_top(
            top_mm,
            height_mm,
            left_mm=AXES_LEFT_MM + index * (width_mm + gap_mm),
            width_mm=width_mm,
        )
        for index in range(3)
    ]


def _set_axis_boxes(axes: list, boxes: list[list[float]]) -> None:
    if len(axes) != len(boxes):
        raise RuntimeError(f"Axis/box count mismatch: {len(axes)} != {len(boxes)}")
    for ax, box in zip(sorted(axes, key=lambda item: item.get_position().x0), boxes):
        ax.set_position(box, which="both")


def _legend_inventory(legend) -> tuple[list, list[str]]:
    if legend is None:
        raise RuntimeError("Missing expected Panel-c legend")
    return list(legend.legend_handles), [item.get_text() for item in legend.get_texts()]


def _model_legend_handles(bar_legend) -> tuple[list[Line2D], list[str]]:
    handles, labels = _legend_inventory(bar_legend)
    if len(handles) != 8 or len(labels) != 8:
        raise RuntimeError(f"Expected eight model legend entries; got {len(handles)} and {len(labels)}")
    rebuilt = []
    for handle, label in zip(handles, labels):
        face = (
            handle.get_markerfacecolor()
            if hasattr(handle, "get_markerfacecolor")
            else handle.get_facecolor()
        )
        alpha = handle.get_alpha()
        rebuilt.append(Line2D(
            [], [], marker="s", linestyle="None", markersize=4.6,
            markerfacecolor=face, markeredgecolor="none", alpha=alpha,
            label=label,
        ))
    return rebuilt, labels


def _style_legend(legend, *, bold_dmf: bool = True) -> None:
    if bold_dmf:
        for text in legend.get_texts():
            if text.get_text() == "DMF-Gen":
                text.set_fontweight("bold")


def _reflow_panel_b(b_axes: list) -> dict:
    boxes = _equal_column_boxes(88.0, 25.0, B_GAP_MM)
    ordered = sorted(b_axes, key=lambda item: item.get_position().y0, reverse=True)
    field_labels = [tick.get_text() for tick in ordered[-1].get_xticklabels() if tick.get_text()]
    if len(field_labels) != 5:
        raise RuntimeError(f"Expected five Panel-b field labels; got {field_labels}")
    _set_axis_boxes(ordered, boxes)
    for index, ax in enumerate(ordered):
        ax.set_aspect("auto")
        ax.set_xticks(range(len(field_labels)), field_labels)
        ax.tick_params(
            axis="x", top=True, labeltop=True, bottom=False, labelbottom=False,
            length=0, pad=1.6, labelsize=base.SIZE_TICK_LABEL,
        )
        ax.tick_params(
            axis="y", left=False, labelleft=index == 0, length=0, pad=1.2,
            labelsize=base.SIZE_TICK_LABEL,
        )
        ax.set_title(ax.get_title(), fontsize=base.SIZE_SUBPLOT_TITLE, pad=9.0)
    return {
        "axis_boxes_fraction": [list(ax.get_position().bounds) for ax in ordered],
        "model_label_axes_count": sum(any(tick.get_visible() and tick.get_text() for tick in ax.get_yticklabels()) for ax in ordered),
        "field_header_axes_count": sum(any(tick.get_visible() and tick.get_text() for tick in ax.get_xticklabels()) for ax in ordered),
    }


def _reflow_panel_c(fig, bar_axes: list, spectra: list) -> dict:
    bar_legend = getattr(fig, "_panel_c_bar_legend", None)
    line_legend = getattr(fig, "_panel_c_line_legend", None)
    model_handles, model_labels = _model_legend_handles(bar_legend)
    line_handles, line_labels = _legend_inventory(line_legend)
    truth_index = line_labels.index("Ground truth")
    truth_color = line_handles[truth_index].get_color()

    legend_ax = bar_legend.axes
    legacy_line_ax = line_legend.axes
    bar_legend.remove()
    line_legend.remove()
    legacy_line_ax.set_visible(False)
    legend_ax.set_visible(True)
    legend_ax.set_axis_off()
    legend_ax.set_position(
        _box_from_top(122.0, 6.0, left_mm=LEGEND_LEFT_MM, width_mm=LEGEND_RIGHT_MM - LEGEND_LEFT_MM),
        which="both",
    )
    truth_handle = Line2D([], [], color=truth_color, lw=base.LW_LINE_PLOT, label="Ground truth")
    consolidated = legend_ax.legend(
        handles=[truth_handle, *model_handles],
        labels=["Ground truth", *model_labels],
        ncol=9, loc="center", bbox_to_anchor=(0.5, 0.5),
        fontsize=base.SIZE_LEGEND, frameon=False, borderaxespad=0.0,
        handlelength=0.78, handletextpad=0.26, columnspacing=0.62,
        labelspacing=0.0,
    )
    _style_legend(consolidated)

    bar_boxes = _equal_column_boxes(132.0, 18.0, C_D_GAP_MM)
    spectrum_boxes = _equal_column_boxes(157.0, 22.0, C_D_GAP_MM)
    _set_axis_boxes(bar_axes, bar_boxes)
    _set_axis_boxes(spectra, spectrum_boxes)
    for index, ax in enumerate(sorted(bar_axes, key=lambda item: item.get_position().x0)):
        ax.set_ylabel("LSD (dB)" if index == 0 else "", fontsize=base.SIZE_AXIS_LABEL)
        ax.set_title(ax.get_title(), fontsize=base.SIZE_SUBPLOT_TITLE, pad=2.2)
    for index, ax in enumerate(sorted(spectra, key=lambda item: item.get_position().x0)):
        existing = ax.get_legend()
        if existing is not None:
            existing.remove()
        ax.set_ylabel("spectral energy" if index == 0 else "", fontsize=base.SIZE_AXIS_LABEL)
        ax.set_xlabel("Wavenumber" if index == 1 else "", fontsize=base.SIZE_AXIS_LABEL, labelpad=2.0)
    fig._panel_c_consolidated_legend = consolidated  # type: ignore[attr-defined]
    fig._panel_c_consolidated_legend_axis = legend_ax  # type: ignore[attr-defined]
    return {
        "legend_entry_count": len(consolidated.get_texts()),
        "legend_ncols": 9,
        "legend_labels": [item.get_text() for item in consolidated.get_texts()],
        "bar_axis_boxes_fraction": [list(ax.get_position().bounds) for ax in sorted(bar_axes, key=lambda item: item.get_position().x0)],
        "spectrum_axis_boxes_fraction": [list(ax.get_position().bounds) for ax in sorted(spectra, key=lambda item: item.get_position().x0)],
    }


def _reflow_panel_d(fig, pdf_axes: list, violin_axes: list) -> dict:
    # Rebuild the eight model-only entries from Panel c's consolidated legend.
    consolidated = getattr(fig, "_panel_c_consolidated_legend")
    c_handles = list(consolidated.legend_handles)
    c_labels = [item.get_text() for item in consolidated.get_texts()]
    model_pairs = [(handle, label) for handle, label in zip(c_handles, c_labels) if label != "Ground truth"]
    if len(model_pairs) != 8:
        raise RuntimeError(f"Expected eight model handles for Panel d; got {len(model_pairs)}")
    d_handles = [Line2D(
        [], [], marker="s", linestyle="None", markersize=4.6,
        markerfacecolor=handle.get_markerfacecolor(), markeredgecolor="none",
        alpha=handle.get_alpha(), label=label,
    ) for handle, label in model_pairs]

    legend_ax = fig.add_axes(
        _box_from_top(196.0, 6.0, left_mm=LEGEND_LEFT_MM, width_mm=LEGEND_RIGHT_MM - LEGEND_LEFT_MM)
    )
    legend_ax.set_axis_off()
    d_legend = legend_ax.legend(
        handles=d_handles, labels=[label for _, label in model_pairs],
        ncol=8, loc="center", bbox_to_anchor=(0.5, 0.5),
        fontsize=base.SIZE_LEGEND, frameon=False, borderaxespad=0.0,
        handlelength=0.72, handletextpad=0.27, columnspacing=0.78,
        labelspacing=0.0,
    )
    _style_legend(d_legend)

    pdf_boxes = _equal_column_boxes(211.0, 17.0, C_D_GAP_MM)
    violin_boxes = _equal_column_boxes(235.0, 23.0, C_D_GAP_MM)
    _set_axis_boxes(pdf_axes, pdf_boxes)
    _set_axis_boxes(violin_axes, violin_boxes)
    for ax in pdf_axes:
        ax.set_box_aspect(None)
        ax.set_aspect("auto")
    ordered_violins = sorted(violin_axes, key=lambda item: item.get_position().x0)
    for index, ax in enumerate(ordered_violins):
        ax.tick_params(axis="x", labelbottom=True, labelsize=base.SIZE_TICK_LABEL, pad=1.0)
        visible_ticks = [tick for tick in ax.get_xticklabels() if tick.get_text()]
        for tick in visible_ticks:
            tick.set_y(0.0)
            tick.set_rotation(0)
            tick.set_horizontalalignment("center")
        if index == len(ordered_violins) - 1 and visible_ticks:
            visible_ticks[-1].set_horizontalalignment("right")
        ax.tick_params(axis="y", labelleft=index == 0, length=0, pad=1.5)
        ax.set_xlabel("JSD of joint PDF" if index == 1 else "", fontsize=base.SIZE_AXIS_LABEL, labelpad=7.0)
        if index == 1:
            ax.xaxis.set_label_coords(0.5, -0.24)

    titles = list(getattr(fig, "_panel_d_title_artists", []))
    if len(titles) != 3:
        raise RuntimeError(f"Expected three Panel-d title artists; got {len(titles)}")
    title_y = 1.0 - 207.0 / CANVAS_HEIGHT_MM
    for title, ax in zip(titles, sorted(pdf_axes, key=lambda item: item.get_position().x0)):
        box = ax.get_position()
        title.set_position(((box.x0 + box.x1) / 2.0, title_y))
        title.set_ha("center")
        title.set_va("bottom")
    fig._panel_d_consolidated_legend = d_legend  # type: ignore[attr-defined]
    fig._panel_d_consolidated_legend_axis = legend_ax  # type: ignore[attr-defined]
    return {
        "legend_entry_count": len(d_legend.get_texts()),
        "legend_ncols": 8,
        "legend_labels": [item.get_text() for item in d_legend.get_texts()],
        "pdf_axis_boxes_fraction": [list(ax.get_position().bounds) for ax in sorted(pdf_axes, key=lambda item: item.get_position().x0)],
        "violin_axis_boxes_fraction": [list(ax.get_position().bounds) for ax in ordered_violins],
    }


def _position_panel_letters(fig) -> None:
    letters = getattr(fig, "_panel_letter_artists", {})
    y_top_mm = {"b": 80.0, "c": 121.0, "d": 195.0}
    for label, top_mm in y_top_mm.items():
        artist = letters.get(label)
        if artist is None:
            raise RuntimeError(f"Missing Panel-{label} label")
        artist.set_position((PANEL_TAG_X, 1.0 - top_mm / CANVAS_HEIGHT_MM))
        artist.set_ha("left")
        artist.set_va("bottom")


def _bbox_union(boxes):
    boxes = [box for box in boxes if box is not None and box.width > 0 and box.height > 0]
    if not boxes:
        raise RuntimeError("Cannot form a union from an empty bounding-box inventory")
    return matplotlib.transforms.Bbox.union(boxes)


def _row_union_mm(fig, axes: list, figure_artists: list, extra_artists: list | None = None) -> list[float]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [ax.get_tightbbox(renderer) for ax in axes if ax.get_visible()]
    boxes.extend(item.get_window_extent(renderer) for item in figure_artists if item.get_visible())
    boxes.extend(item.get_window_extent(renderer) for item in (extra_artists or []) if item.get_visible())
    union = _bbox_union(boxes)
    scale = 25.4 / float(fig.dpi)
    return [float(union.x0 * scale), float(union.y0 * scale), float(union.x1 * scale), float(union.y1 * scale)]


def _layout_qa(fig, *, b_axes: list, bar_axes: list, spectra: list, pdf_axes: list, violin_axes: list) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    # Keep edge ticks on their numerical locations while directing their glyph
    # boxes inward when the final renderer places a label on the MediaBox edge.
    for ax in fig.axes:
        for tick in ax.get_xticklabels():
            if not tick.get_visible() or not tick.get_text():
                continue
            box = tick.get_window_extent(renderer)
            if box.x1 > fig.bbox.x1:
                tick.set_horizontalalignment("right")
            elif box.x0 < fig.bbox.x0:
                tick.set_horizontalalignment("left")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    letters = getattr(fig, "_panel_letter_artists", {})
    lower_set = set([*b_axes, *bar_axes, *spectra, *pdf_axes, *violin_axes])
    lower_set.add(getattr(fig, "_panel_c_consolidated_legend_axis"))
    lower_set.add(getattr(fig, "_panel_d_consolidated_legend_axis"))
    panel_a_axes = [ax for ax in fig.axes if ax not in lower_set and ax.get_visible() and ax.axison]
    expected_maps = int(getattr(fig, "_panel_a_map_count", -1))
    if len(panel_a_axes) != expected_maps:
        raise RuntimeError(f"Expected {expected_maps} Panel-a axes; classified {len(panel_a_axes)}")
    d_titles = list(getattr(fig, "_panel_d_title_artists", []))
    a_texts = [
        artist for artist in fig.texts
        if artist not in [letters.get("b"), letters.get("c"), letters.get("d"), *d_titles]
    ]
    rows = {
        "a": _row_union_mm(fig, panel_a_axes, a_texts),
        "b": _row_union_mm(fig, b_axes, [letters["b"]]),
        "c": _row_union_mm(
            fig, [*bar_axes, *spectra, fig._panel_c_consolidated_legend_axis],
            [letters["c"]], [fig._panel_c_consolidated_legend],
        ),
        "d": _row_union_mm(
            fig, [*pdf_axes, *violin_axes, fig._panel_d_consolidated_legend_axis],
            [letters["d"], *d_titles], [fig._panel_d_consolidated_legend],
        ),
    }
    adjacent = {}
    failed_clearances = {}
    for upper, lower in (("a", "b"), ("b", "c"), ("c", "d")):
        clearance = rows[upper][1] - rows[lower][3]
        adjacent[f"{upper}_to_{lower}_clearance_mm"] = float(clearance)
        if clearance < 3.0:
            failed_clearances[f"{upper}->{lower}"] = float(clearance)
    if failed_clearances:
        raise RuntimeError(f"Rendered row clearances below 3 mm: {failed_clearances}")
    canvas_width, canvas_height = [float(value) * 25.4 for value in fig.get_size_inches()]
    clipped = []
    for artist in fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text)):
        if not artist.get_visible() or not artist.get_text():
            continue
        box = artist.get_window_extent(renderer)
        if box.width <= 0 or box.height <= 0:
            continue
        scale = 25.4 / float(fig.dpi)
        if box.x0 < 0 or box.y0 < 0 or box.x1 * scale > canvas_width or box.y1 * scale > canvas_height:
            clipped.append({
                "text": artist.get_text(),
                "axes_index": fig.axes.index(artist.axes) if artist.axes in fig.axes else None,
                "bbox_px": [float(box.x0), float(box.y0), float(box.x1), float(box.y1)],
            })
    if clipped:
        raise RuntimeError(f"Visible text escaped the canvas: {clipped}")
    collision = v4._collision_gate(fig)
    if collision["text_text_overlap_count"] or collision["text_nonowned_axes_overlap_count"]:
        raise RuntimeError(
            "Final collision gate failed: "
            f"text/text={collision['text_text_overlap_count']}, "
            f"text/nonowned-axis={collision['text_nonowned_axes_overlap_count']}"
        )
    b_widths = [ax.get_position().width * canvas_width for ax in b_axes]
    c_widths = [ax.get_position().width * canvas_width for ax in bar_axes]
    d_widths = [ax.get_position().width * canvas_width for ax in pdf_axes]
    return {
        "canvas_mm": [canvas_width, canvas_height],
        "panel_a_axes_count": len(panel_a_axes),
        "row_union_boxes_mm_from_bottom": rows,
        **adjacent,
        "clipped_text_count": len(clipped),
        "panel_tag_x_spread_mm": float(np.ptp([letters[key].get_position()[0] * canvas_width for key in ("a", "b", "c", "d")])),
        "panel_b_axis_widths_mm": b_widths,
        "panel_c_axis_widths_mm": c_widths,
        "panel_d_axis_widths_mm": d_widths,
        "lower_rows_right_edge_spread_mm": float(np.ptp([
            max(ax.get_position().x1 for ax in b_axes) * canvas_width,
            max(ax.get_position().x1 for ax in bar_axes) * canvas_width,
            max(ax.get_position().x1 for ax in pdf_axes) * canvas_width,
        ])),
        **collision,
    }


def _apply_v6_art(fig) -> dict:
    b_axes, bar_axes, spectra, pdf_axes, violin_axes = v4._panel_axes(fig)
    v5_qa = v5._apply_v5_art(fig)
    b_qa = _reflow_panel_b(b_axes)
    c_qa = _reflow_panel_c(fig, bar_axes, spectra)
    d_qa = _reflow_panel_d(fig, pdf_axes, violin_axes)
    _position_panel_letters(fig)
    fig.canvas.draw()
    layout_qa = _layout_qa(
        fig, b_axes=b_axes, bar_axes=bar_axes, spectra=spectra,
        pdf_axes=pdf_axes, violin_axes=violin_axes,
    )
    return {
        **v5_qa,
        **layout_qa,
        "revision": "art_v6",
        "layout_topology": "four_full_width_rows_a_b_c_d",
        "panel_a_preserved_from_v5": True,
        "panel_b_horizontal_conditions": b_qa,
        "panel_c_full_width": c_qa,
        "panel_d_full_width": d_qa,
    }


def _apply_v6_contract() -> None:
    v5._apply_v5_contract()

    def v6_canvas_size(layout: dict) -> tuple[float, float]:
        width = PAGE_WIDTH_MM / 25.4
        height = CANVAS_HEIGHT_MM / 25.4
        layout["_computed_width_in"] = width
        layout["_computed_height_in"] = height
        return width, height

    base._composite_canvas_size = v6_canvas_size


def _install_v6_save_hook() -> None:
    original_save = base._save_figure

    def save_v6(fig, output_base: Path, formats: list[str], dpi: int):
        qa = _apply_v6_art(fig)
        outputs = original_save(fig, output_base, formats, dpi)
        (output_base.parent / f"{output_base.name}_art_qa.json").write_text(
            json.dumps(
                qa, indent=2, sort_keys=True,
                default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
            ) + "\n",
            encoding="utf-8",
        )
        return outputs

    base._save_figure = save_v6


def main() -> int:
    _apply_v6_contract()
    _install_v6_save_hook()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

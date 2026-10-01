#!/usr/bin/env python
"""Render the V5_3 mixed-resolution layout from the frozen V5_0 evidence.

This renderer inherits V5_0's six-panel canvas and frozen publication sources.
It places the d/e color bars, rebalances the panel-c sweep stack, and aligns
the b/c chart bottoms. It does not train a model or infer a field. The drawers recompute
display metrics from unchanged caches; values are checked against V5_0.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes
import matplotlib.colorbar
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.contour import QuadContourSet
from matplotlib.ticker import FixedLocator
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "Save_TrainedModel" / "_TrainedModels" / "_Scripts"
sys.path.insert(0, str(SCRIPTS))

import global_style as manuscript  # noqa: E402
from common.config import (  # noqa: E402
    FIGURES_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config,
)
from common.figure_style import save_figure  # noqa: E402
from common.physical_figure_layout import validate_panel_text_boundaries  # noqa: E402
from common.publication_panels_unified_v4_7 import (  # noqa: E402
    align_panel_cd_annotation_gutters, apply_v4_7_typography,
    center_panel_d_headers, draw_panel, measure_annotation_collision_gate,
    panel_label, record_panel_e_v4_3_tick_settings,
)

LAYOUT_PATH = SCRIPTS / "publication_layout_unified_v5_3.yaml"
BASELINE_DIR = (
    ROOT / "figures" / "generated" / "art_style_review"
    / "MixedResolution_unified_v4_7_20260914_2318"
)
BASELINE_MANIFEST = BASELINE_DIR / "source_manifest_v4_7.json"
V5_0_PDF = (
    ROOT / "Save_TrainedModel" / "_TrainedModels" / "_Process_Figures"
    / "Assembled" / "MixedResolution_unified_v5_0_20260926_1336.pdf"
)
V5_0_MANIFEST = (
    ROOT / "figures" / "generated" / "art_style_review"
    / "MixedResolution_unified_v5_0_20260926_1336" / "source_manifest_v5_0.json"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v47 = _load_module("mixed_resolution_v4_7_for_v5_3", SCRIPTS / "139_assemble_mixed_resolution_unified_v4_7.py")
base = v47.base
v4 = v47.v4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}


def _write_json(path: Path, data: dict) -> None:
    def _native(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Unsupported JSON value {type(value).__name__}")
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_native) + "\n",
                    encoding="utf-8")


def _rects(layout: dict):
    cfg = layout["v5_geometry"]
    width = float(cfg["canvas_width_mm"])
    height = float(cfg["canvas_height_mm"])
    # The five legacy draw routines retain their labels internally.  Their
    # visible letters are assigned only after the source artists are drawn.
    rects = {
        "a": cfg["panel_a"], "b": cfg["panel_b"],
        "c": cfg["panel_d"], "d": cfg["panel_e"], "e": cfg["panel_f"],
    }
    return width, height, {key: list(map(float, value)) for key, value in rects.items()}


def _create_canvas(layout: dict):
    previous = v4._geometry
    v4._geometry = _rects
    try:
        fig, axes, containers, rects, width, height, shared, cbar_parent = v4.create_canvas(layout)
    finally:
        v4._geometry = previous
    right = list(map(float, layout["v5_geometry"]["panel_c"]))
    c_container = fig.add_axes(
        [right[0] / width, right[1] / height, right[2] / width, right[3] / height],
        label="panel-c-container", frameon=False,
    )
    c_container.set_axis_off()
    containers["new_c"] = c_container
    return fig, axes, containers, rects, width, height, shared, cbar_parent


def _place(axis: matplotlib.axes.Axes, rect_mm: list[float], width: float, height: float) -> None:
    left, bottom, w, h = map(float, rect_mm)
    axis.set_axes_locator(None)
    axis.set_position([left / width, bottom / height, w / width, h / height], which="both")


def _children_with(parent, predicate):
    return [axis for axis in parent.child_axes if predicate(axis)]


def _add_vertical_bar(fig, *, name, x, bottom, width, height, cmap, norm, ticks,
                      label, label_x=None, label_y=None, rotation=90):
    """Place a narrow, editable scale bar in design millimetres."""
    canvas_w, canvas_h = fig.get_size_inches() * 25.4
    axis = fig.add_axes([x / canvas_w, bottom / canvas_h,
                         width / canvas_w, height / canvas_h], label=name)
    matplotlib.colorbar.ColorbarBase(axis, cmap=cmap, norm=norm,
                                     orientation="vertical", ticks=[])
    axis.tick_params(left=False, right=False, labelleft=False, labelright=False)
    texts = []
    for value, string in ticks:
        fraction = float(norm(value))
        artist = fig.text((x + width + .55) / canvas_w,
                          (bottom + height * fraction) / canvas_h, string,
                          ha="left", va="center", color="#222222")
        manuscript.tag_font_role(artist, "tick_label", size_pt=5.8)
        texts.append(artist)
    if label:
        artist = fig.text((label_x if label_x is not None else x + width + 6.0) / canvas_w,
                          (label_y if label_y is not None else bottom + height / 2) / canvas_h,
                          label, ha="center", va="center", rotation=rotation,
                          color="#222222")
        manuscript.tag_font_role(artist, "annotation", size_pt=6.5)
        texts.append(artist)
    return axis, texts


def _reflow_de_with_colorbars(fig, shared, drawn, width, height, geo):
    """Keep all scientific artists; move only the two image grids and add legends."""
    bar_cfg = geo["panel_de_colorbars"]
    d_x = list(map(float, bar_cfg["d_map_x_mm"]))
    e_x = list(map(float, bar_cfg["e_map_x_mm"]))
    for rows, positions in ((shared["c"], d_x), (shared["d"], e_x)):
        for row in rows:
            for axis, x in zip(row, positions):
                box = axis.get_position()
                axis.set_axes_locator(None)
                axis.set_position([x / width, box.y0, box.width, box.height], which="both")
    fig.canvas.draw()
    d_rows, e_rows = shared["c"], shared["d"]
    mm = lambda axis: [axis.get_position().x0 * width, axis.get_position().y0 * height,
                       axis.get_position().width * width, axis.get_position().height * height]
    d_top, d_mid, d_low = [mm(row[0]) for row in d_rows]
    e_top, e_low = mm(e_rows[0][0]), mm(e_rows[2][0])
    field = next(item for item in d_rows[0][0].collections
                 if not isinstance(item, QuadContourSet) and hasattr(item, "norm"))
    error = next(item for item in d_rows[2][1].collections
                 if not isinstance(item, QuadContourSet) and hasattr(item, "norm"))
    truth = next(item for item in e_rows[0][0].collections
                 if not isinstance(item, QuadContourSet) and hasattr(item, "norm"))
    residual = next(item for item in e_rows[0][1].collections
                    if not isinstance(item, QuadContourSet) and hasattr(item, "norm"))
    field_max = float(drawn["d"]["field_limits"][1])
    error_max = float(drawn["d"]["error_limits"][1])
    bars, texts = [], []
    def add(**kwargs):
        axis, new_texts = _add_vertical_bar(fig, **kwargs)
        bars.append(axis); texts.extend(new_texts)
    add(name="v5-3-d-field-bar", x=float(bar_cfg["d_bar_x_mm"]), bottom=d_mid[1], width=2.0,
        height=d_top[1] + d_top[3] - d_mid[1], cmap=field.cmap, norm=field.norm,
        ticks=[(-field_max, "−7.6"), (0.0, "0"), (field_max, "7.6")],
        label=r"Field ($\times 10^{-2}$)", label_x=112.1)
    add(name="v5-3-d-error-bar", x=float(bar_cfg["d_bar_x_mm"]), bottom=d_low[1], width=2.0,
        height=d_low[3], cmap=error.cmap, norm=error.norm,
        ticks=[(0.0, "0"), (error_max, "1.3")],
        label=r"$|\Delta|$ ($\times 10^{-2}$)", label_x=112.1)
    # E's limits differ by scale. The two bars therefore show each row's
    # fraction of its own symmetric truth limit or absolute-residual limit.
    e_bottom = e_low[1]
    e_height = e_top[1] + e_top[3] - e_bottom
    add(name="v5-3-e-truth-bar", x=float(bar_cfg["e_truth_bar_x_mm"]), bottom=e_bottom, width=1.9,
        height=e_height, cmap=truth.cmap, norm=Normalize(-1, 1),
        ticks=[(-1, "−1"), (0, "0"), (1, "1")],
        label="Truth", label_x=168.65, label_y=e_top[1] + e_top[3] + 3.6,
        rotation=0)
    add(name="v5-3-e-error-bar", x=float(bar_cfg["e_error_bar_x_mm"]), bottom=e_bottom, width=1.9,
        height=e_height, cmap=residual.cmap, norm=Normalize(0, 1),
        ticks=[(0, "0"), (1, "1")],
        label=r"$|\Delta|$", label_x=175.15, label_y=e_top[1] + e_top[3] + 3.6,
        rotation=0)
    return {
        "d_map_x_mm": d_x, "e_map_x_mm": e_x,
        "d_field_limits": drawn["d"]["field_limits"],
        "d_error_limits": drawn["d"]["error_limits"],
        "e_truth_component_limits_by_row": drawn["e"]["component_color_limits"],
        "e_residual_limits_by_row": drawn["e"]["residual_color_limits"],
        "e_bar_units": "fraction of the corresponding row's physical limit",
    }, bars, texts


def _reflow_a(parent, geo: dict, width: float, height: float) -> dict:
    maps = _children_with(parent, lambda axis: str(axis.get_gid() or "") == "geometric-field")
    maps.sort(key=lambda axis: axis.get_position().x0)
    bars = _children_with(parent, lambda axis: axis.yaxis.label.get_text() == "Training cases")
    if len(maps) != 3 or len(bars) != 1:
        raise RuntimeError(f"Panel a topology changed: maps={len(maps)}, bars={len(bars)}")
    size = float(geo["panel_a_map_size_mm"])
    y = float(geo["panel_a_map_y_mm"])
    for tag, x, axis, dims in zip("LMH", geo["panel_a_map_x_mm"], maps,
                                  ("32 × 32", "64 × 64", "128 × 128")):
        _place(axis, [float(x), y, size, size], width, height)
        axis.set_title(f"{tag} · {dims}", pad=1.0)
        for item in list(axis.texts):
            if item.get_text().strip() == tag:
                item.remove()
    # The original separate resolution-key block repeats the map headers.
    # Retain the exact dimensions in those headers and remove the key block.
    for item in parent.texts:
        item.remove()
    for item in list(parent.patches):
        item.remove()
    bar = bars[0]
    _place(bar, geo["panel_a_bar_mm"], width, height)
    bar.yaxis.set_label_coords(-0.09, 0.5)
    bar.set_xlabel("")  # Explain the five retained exposure values in the caption.
    for item in list(bar.texts):
        if item.get_text().strip().lower() in {
            "contains h-resolution training fields", "zero-h training"
        }:
            item.remove()
    return {
        "map_count": 3, "map_size_mm": size,
        "map_titles": [axis.get_title() for axis in maps],
        "training_bar_mm": geo["panel_a_bar_mm"],
        "exposure_values_retained_above_bars": True,
        "repeated_resolution_key_replaced_by_map_headers": True,
    }


def _place_panel_e_tag(fig, parent, geo, width):
    tags = [item for item in parent.texts if item.get_text() == "e"]
    if len(tags) != 1:
        raise RuntimeError(f"Expected one panel-e label, found {len(tags)}")
    tag = tags[0]
    anchor = fig.transFigure.inverted().transform(
        tag.get_transform().transform(tag.get_position()))
    label_x = float(geo["panel_e_label_x_mm"])
    placed = fig.text(label_x / width, float(anchor[1]), "e",
                      fontproperties=tag.get_fontproperties().copy(),
                      color=tag.get_color(), ha=tag.get_horizontalalignment(),
                      va=tag.get_verticalalignment(), gid=tag.get_gid(),
                      zorder=tag.get_zorder(), clip_on=False)
    manuscript.tag_font_role(placed, "panel_label")
    tag.remove()
    return {"x_mm": label_x, "artist": placed}


def _reflow_b_c(parent, geo: dict, ctx, width: float, height: float) -> dict:
    panel_module = __import__("common.publication_panels_unified_v4_6", fromlist=["_panel_b_axes"])
    bar, sweeps = panel_module._panel_b_axes(parent)
    if len(sweeps) != 3:
        raise RuntimeError("V5 requires all three saved sensor sweeps")
    _place(bar, geo["panel_b_bar_mm"], width, height)
    bar.yaxis.set_label_coords(-0.04, 0.5)
    # The inherited inside-the-bars heading duplicates the normal subplot
    # title and crowds the shorter V5 bar axis.
    for item in list(bar.texts):
        if item.get_gid() == "panel-b-inside-title":
            item.remove()
    bar.set_title("High-resolution reconstruction (512 sensors)", pad=1.5)
    for index, (axis, rect) in enumerate(zip(sweeps, geo["panel_c_sweep_mm"])):
        _place(axis, rect, width, height)
        axis.set_title(axis.get_title(), y=float(geo["panel_c_subtitle_y_axes"]), pad=0.0)
        # Exclude Matplotlib's clipped 10^1 tick, whose off-page text box
        # appears when the upper sweep expands close to the canvas edge.
        axis.yaxis.set_major_locator(FixedLocator([1e-2, 1e-1]))
        axis.tick_params(axis="y", labelleft=True)
        if index < 2:
            axis.set_xlabel("")
            axis.tick_params(axis="x", labelbottom=False)
        else:
            axis.set_xlabel("Sensors / H-grid density (%)", labelpad=1.0)
        axis.set_ylabel("Physical relative $L_2$" if index == 1 else "", labelpad=1.0)
    legends = _children_with(parent, lambda axis: axis.get_legend() is not None)
    if len(legends) != 1:
        raise RuntimeError(f"Expected one shared model legend, found {len(legends)}")
    legend_axis = legends[0]
    legend_axis.get_legend().remove()
    _place(legend_axis, geo["panel_c_legend_mm"], width, height)
    handles = __import__("common.publication_panels_unified_v3", fromlist=["_model_handles"])._model_handles(ctx)
    legend = legend_axis.legend(
        handles=handles, ncol=2, loc="lower center", frameon=False,
        fontsize=7.8, columnspacing=0.8, handletextpad=0.35,
        handlelength=1.4, borderaxespad=0.0, labelspacing=0.25,
    )
    for item in legend.get_texts():
        manuscript.tag_font_role(item, "legend", size_pt=7.8)
    return {
        "grouped_bar_mm": geo["panel_b_bar_mm"],
        "sensor_sweep_mm": geo["panel_c_sweep_mm"],
        "shared_legend_mm": geo["panel_c_legend_mm"],
        "shared_legend_labels": [item.get_text() for item in legend.get_texts()],
        "upper_sweep_x_labels_hidden_for_shared_axis": True,
        "sweep_count": len(sweeps),
    }


def _box_mm(box, fig) -> list[float]:
    scale = 25.4 / fig.dpi
    return [float(value * scale) for value in (box.x0, box.y0, box.x1, box.y1)]


def _tight_box(axis, renderer):
    try:
        return axis.get_tightbbox(renderer)
    except Exception:
        return None


def _colorbar_qa(fig, shared, bar_axes, bar_texts) -> dict:
    """Check the new bar assembly against every existing visible text and map."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = 25.4 / fig.dpi
    maps = [axis for panel in ("c", "d") for row in shared[panel] for axis in row]
    map_boxes = [(axis.get_label(), axis.get_window_extent(renderer)) for axis in maps]
    bar_boxes = [(axis.get_label(), axis.get_window_extent(renderer)) for axis in bar_axes]
    overlap = []
    for item in bar_texts:
        box = item.get_window_extent(renderer)
        for name, target in [*map_boxes, *bar_boxes]:
            if box.overlaps(target):
                overlap.append([item.get_text(), name])
    for index, item in enumerate(bar_texts):
        box = item.get_window_extent(renderer)
        for peer in bar_texts[index + 1:]:
            if box.overlaps(peer.get_window_extent(renderer)):
                overlap.append([item.get_text(), peer.get_text()])
    for name, box in bar_boxes:
        for map_name, map_box in map_boxes:
            if box.overlaps(map_box):
                overlap.append([name, map_name])
    left_d = min(map_boxes[index][1].x0 for index in range(15)) * scale
    right_e_bar = max(box.x1 for name, box in bar_boxes if name.startswith("v5-3-e-")) * scale
    right_e_assembly = max(
        [right_e_bar, *(item.get_window_extent(renderer).x1 * scale
                        for item in bar_texts if item.get_text() in {"−1", "0", "1", "Truth", r"$|\Delta|$"})]
    )
    canvas_width = fig.get_size_inches()[0] * 25.4
    result = {
        "leftmost_d_map_mm": left_d,
        "rightmost_e_bar_mm": right_e_bar,
        "rightmost_e_bar_assembly_mm": right_e_assembly,
        "canvas_width_mm": canvas_width,
        "new_bar_count": len(bar_axes),
        "new_text_count": len(bar_texts),
        "new_text_or_bar_intersections": overlap,
        "passed": bool(not overlap and left_d <= 6.0 and right_e_assembly <= canvas_width
                       and canvas_width - right_e_assembly <= 2.0),
    }
    return result


def _topology_qa(fig, axes, containers, shared, geo, width, height) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_module = __import__("common.publication_panels_unified_v4_6", fromlist=["_panel_b_axes"])
    bar, sweeps = panel_module._panel_b_axes(axes["b"])
    legend_axis = next(axis for axis in axes["b"].child_axes if axis.get_legend() is not None)
    groups = {
        "a": [axes["a"], *axes["a"].child_axes],
        "b": [axes["b"], bar],
        "c": [containers["new_c"], *sweeps, legend_axis],
        "d": [axes["c"], *(axis for row in shared["c"] for axis in row),
              *(axis for axis in fig.axes if axis.get_label().startswith("v5-3-d-"))],
        "e": [axes["d"], *(axis for row in shared["d"] for axis in row),
              *(axis for axis in fig.axes if axis.get_label().startswith("v5-3-e-"))],
        "f": [axes["e"], *axes["e"].child_axes],
    }
    boxes = {}
    for label, group in groups.items():
        found = [box for axis in group if (box := _tight_box(axis, renderer)) is not None]
        if not found:
            raise RuntimeError(f"No rendered artist union for {label}")
        boxes[label] = matplotlib.transforms.Bbox.union(found)
    gap = lambda upper, lower: float((boxes[upper].y0 - boxes[lower].y1) * 25.4 / fig.dpi)
    vertical_gaps = {"a-b": gap("a", "b"), "top-de": min(gap("a", "d"), gap("b", "d"), gap("c", "e")),
                     "de-f": min(gap("d", "f"), gap("e", "f"))}
    horizontal_gaps = {
        "a-c": float((boxes["c"].x0 - boxes["a"].x1) * 25.4 / fig.dpi),
        "b-c": float((boxes["c"].x0 - boxes["b"].x1) * 25.4 / fig.dpi),
    }
    overflow = dict(manuscript.validate_text_within_canvas(fig))
    result = {
        "canvas_mm": [width, height], "baseline_canvas_mm": [180.0, 229.5],
        "height_reduction_mm": 229.5 - height,
        "height_reduction_percent": 100.0 * (229.5 - height) / 229.5,
        "panel_artist_union_mm": {key: _box_mm(value, fig) for key, value in boxes.items()},
        "vertical_artist_gaps_mm": vertical_gaps,
        "horizontal_artist_gaps_mm": horizontal_gaps,
        "text_overflow": overflow,
        "no_canvas_text_overflow": not any(overflow.values()),
    }
    result["passed"] = (
        result["no_canvas_text_overflow"]
        and min(vertical_gaps.values()) * 0.9 >= 3.0
        and min(horizontal_gaps.values()) * 0.9 >= 2.5
        and result["height_reduction_mm"] > 0.0
    )
    return result


def _upper_layout_qa(fig, axes, containers, geo, panel_e_label) -> dict:
    """Guard the requested b/c alignment and the reduced c subplot gaps."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = 25.4 / fig.dpi
    panel_module = __import__("common.publication_panels_unified_v4_6", fromlist=["_panel_b_axes"])
    bar, sweeps = panel_module._panel_b_axes(axes["b"])
    legend_axis = next(axis for axis in axes["b"].child_axes if axis.get_legend() is not None)
    width, height = fig.get_size_inches() * 25.4
    pos = lambda axis: [axis.get_position().x0 * width, axis.get_position().y0 * height,
                        axis.get_position().width * width, axis.get_position().height * height]
    bar_pos, sweep_pos, legend_pos = pos(bar), [pos(axis) for axis in sweeps], pos(legend_axis)
    gaps = [sweep_pos[index][1] - (sweep_pos[index + 1][1] + sweep_pos[index + 1][3])
            for index in (0, 1)]
    legend_gap = legend_pos[1] - (sweep_pos[0][1] + sweep_pos[0][3])
    top_margin = height - legend_pos[1] - legend_pos[3]
    subtitle_top_inset = [
        (axis.get_window_extent(renderer).y1 - axis.title.get_window_extent(renderer).y1) * scale
        for axis in sweeps
    ]
    b_container, c_container = pos(containers["b"]), pos(containers["new_c"])
    e_tag = panel_e_label["artist"].get_window_extent(renderer)
    e_tag_left = e_tag.x0 * scale
    e_tag_right = e_tag.x1 * scale
    result = {
        "panel_b_container_height_mm": b_container[3],
        "panel_b_c_container_bottom_delta_mm": abs(b_container[1] - c_container[1]),
        "panel_b_chart_height_mm": bar_pos[3],
        "panel_c_sweep_heights_mm": [item[3] for item in sweep_pos],
        "panel_b_c_chart_bottom_delta_mm": abs(bar_pos[1] - sweep_pos[-1][1]),
        "panel_c_sweep_vertical_gaps_mm": gaps,
        "panel_c_legend_to_top_sweep_gap_mm": legend_gap,
        "panel_c_all_three_gap_spread_mm": max([legend_gap, *gaps]) - min([legend_gap, *gaps]),
        "panel_c_legend_top_canvas_margin_mm": top_margin,
        "panel_c_previous_vertical_gap_mm": 6.0,
        "panel_c_gap_reduction_fraction": 1.0 - np.mean(gaps) / 6.0,
        "panel_c_subtitle_top_inset_mm": subtitle_top_inset,
        "panel_e_label_bbox_x_mm": [e_tag_left, e_tag_right],
        "panel_e_left_boundary_mm": float(geo["panel_e"][0]),
        "passed": bool(
            abs(b_container[3] - 35.2) <= .02
            and abs(bar_pos[3] - 21.0) <= .02
            and all(abs(item[3] - 19.0667) <= .02 for item in sweep_pos)
            and all(abs(gap - 3.6) <= .02 for gap in gaps)
            and abs(legend_gap - 3.6) <= .02
            and 0.5 <= top_margin <= 1.5
            and abs(1.0 - np.mean(gaps) / 6.0 - .4) <= .002
            and abs(b_container[1] - c_container[1]) <= .02
            and abs(bar_pos[1] - sweep_pos[-1][1]) <= .02
            and min(subtitle_top_inset) >= -.80
            and 0.0 <= e_tag_left < float(geo["panel_e"][0])
            and e_tag_right <= float(geo["panel_e"][0]) + .5
        ),
    }
    return result


def _compare_science(baseline: dict, drawn: dict) -> dict:
    pairs = (("a", "a"), ("c", "d"), ("d", "e"), ("e", "f"))
    keys = {
        "a": ("recipe_order", "dimensions", "field_limits", "bar_segments", "exposure_values"),
        "c": ("roi", "full_field_relative_l2", "local_relative_l2", "field_limits", "error_limits"),
        "d": ("row_order", "relative_l2", "truth_component_limits", "residual_limits"),
        "e": ("matrix_numeric_annotation_count", "correlation_values", "bias_values"),
    }
    checks = {}
    for old, new in pairs:
        for key in keys[old]:
            if key in baseline["panels"][old]:
                checks[f"{old}->{new}:{key}"] = baseline["panels"][old][key] == drawn[new].get(key)
    before_rows = baseline["panels"]["b"]["plotted_rows"]
    after_rows = drawn["_b_source"]["plotted_rows"]
    bar_rows = [row for row in after_rows if row["role"] == "recipe_transfer_512"]
    sweep_rows = [row for row in after_rows if row["role"] != "recipe_transfer_512"]
    checks.update({
        "b_source_plotted_rows_exact": before_rows == after_rows,
        "b_grouped_bar_row_count": len(bar_rows) == 20,
        "c_sweep_row_count": len(sweep_rows) == 60,
        "b_recipe_values_exact": baseline["panels"]["b"]["recipe_transfer_values"] == drawn["b"]["recipe_transfer_values"],
        "c_sweep_values_exact": baseline["panels"]["b"]["sensor_sweep_values"] == drawn["c"]["sensor_sweep_values"],
        "shared_sweep_y_range_exact": baseline["panels"]["b"]["sensor_sweep_y_range"] == drawn["c"]["sensor_sweep_y_range"],
    })
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "plotted_row_count": len(after_rows), "grouped_bar_rows": len(bar_rows),
            "sensor_sweep_rows": len(sweep_rows),
            "unique_source_rows": len({(row["model"], row["recipe"], row["sensor_count"])
                                       for row in after_rows}),
            "mapping": {"a": "a", "b grouped bars": "b", "b sensor sweeps": "c",
                        "c": "d", "d": "e", "e": "f"}}


def _compare_v5_0(baseline: dict, drawn: dict) -> dict:
    fields = {
        "d": ("field_limits", "error_limits", "full_field_relative_l2",
              "local_relative_l2", "roi"),
        "e": ("component_color_limits", "residual_color_limits",
              "relative_l2_by_model_scale", "qualitative_scales"),
    }
    checks = {f"{panel}:{key}": bool(baseline["panels"][panel][key] == drawn[panel][key])
              for panel, keys in fields.items() for key in keys}
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "baseline_pdf": str(V5_0_PDF)}


def _publication_panel_metadata(drawn: dict) -> dict:
    """Store V5 panel roles without stale V4_7 cross-panel geometry fields."""
    original_b = drawn["_b_source"]
    common_b = {key: original_b[key] for key in (
        "status", "sources", "models", "model_order", "statistic",
    ) if key in original_b}
    a_keys = (
        "status", "sources", "purpose", "recipes", "recipe_order", "snapshot",
        "case_id", "time_index", "field", "dimensions", "field_limits",
        "shared_roi", "roi_shared_across_resolutions", "zoom_inset_count",
        "connector_count", "bar_segments", "exposure_values", "grouping",
    )
    panels = {
        "a": {key: drawn["a"][key] for key in a_keys if key in drawn["a"]},
        "b": {
            **common_b, "sensor_count": original_b["sensor_count"],
            "recipe_order": original_b["recipe_order"],
            "recipe_transfer_plot_type": original_b["recipe_transfer_plot_type"],
            "recipe_transfer_axis_scale": original_b["recipe_transfer_axis_scale"],
            "recipe_transfer_y_range": original_b["recipe_transfer_y_range"],
            "recipe_transfer_values": original_b["recipe_transfer_values"],
            "plotted_rows": drawn["b"]["plotted_rows"],
            "zero_h_region_shaded": original_b["zero_h_region_shaded"],
        },
        "c": {
            **common_b, "sweep_recipes": original_b["sweep_recipes"],
            "sensor_counts": original_b["sensor_counts"],
            "sensor_density_percent": original_b["sensor_density_percent"],
            "sensor_sweep_axis_scale": original_b["sensor_sweep_axis_scale"],
            "sensor_sweep_y_range": original_b["sensor_sweep_y_range"],
            "sensor_sweep_values": original_b["sensor_sweep_values"],
            "plotted_rows": drawn["c"]["plotted_rows"],
            "shared_model_legend": True,
        },
        "d": dict(drawn["d"]), "e": dict(drawn["e"]), "f": dict(drawn["f"]),
    }
    for label, panel in panels.items():
        panel.update({
            "panel_id": f"fig3.{label}", "figure_revision": "V5_3",
            "source_visual_revision": "V5_0", "source_scientific_revision": "V3-7",
            "underlying_source_arrays_preserved": True,
            "model_training_or_inference": False,
        })
    return panels


def _previews(pdf: Path, png: Path, release: Path) -> dict:
    preview_dir = release / "previews"
    preview_dir.mkdir()
    vector_stem = preview_dir / "vector_162mm"
    subprocess.run(["pdftocairo", "-png", "-singlefile", "-r", "540",
                    str(pdf), str(vector_stem)], check=True)
    with Image.open(png) as source:
        rgb = source.convert("RGB")
        gray = preview_dir / "grayscale.png"
        ImageOps.grayscale(rgb).save(gray)
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([[.367, .861, -.228], [.280, .673, .047],
                             [-.012, .043, .969]], dtype=np.float32)
        cvd = preview_dir / "deuteranopia.png"
        Image.fromarray(np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8),
                        "RGB").save(cvd)
    return {"vector_162mm": _record(vector_stem.with_suffix(".png")),
            "grayscale": _record(gray), "deuteranopia": _record(cvd)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=LAYOUT_PATH)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    parser.add_argument("--trial-dir", type=Path, help="Write a disposable trial bundle here")
    args = parser.parse_args()
    rid = base.publication_timestamp(args.run_id)
    layout = base.load_layout(args.layout.resolve())
    cfg = load_config(args.config)
    style_record = v47.v46.v4_5.apply_v4_5_style_contract(cfg)
    ensure_output_dirs()
    ctx = base.make_context(args, cfg, layout, rid)
    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    requested_baseline = json.loads(V5_0_MANIFEST.read_text(encoding="utf-8"))
    baseline_pdf = BASELINE_DIR / "MixedResolution_unified_v4_7_20260914_2318.pdf"
    baseline_hash = _sha256(baseline_pdf)
    v5_0_hash = _sha256(V5_0_PDF)
    source_hashes = {}
    for entry in baseline["source_data_records"] + baseline["cache_source_records"]:
        path = Path(entry["path"])
        observed = _sha256(path)
        if observed != entry["sha256"]:
            raise RuntimeError(f"Frozen V4_7 scientific source changed: {path}")
        source_hashes[str(path)] = observed
    results_before = v4._tree_state(RESULTS_DIR)
    if args.trial_dir:
        release = args.trial_dir.resolve()
    else:
        release = ROOT / "figures" / "generated" / "art_style_review" / f"MixedResolution_unified_v5_3_{rid}"
    if release.exists():
        raise FileExistsError(f"Refusing to overwrite {release}")
    release.mkdir(parents=True)
    width, height, _ = _rects(layout)
    geo = layout["v5_geometry"]
    fig, axes, containers, rects, width, height, shared, cbar_parent = _create_canvas(layout)
    drawn = {}
    drawn["a"] = draw_panel("a", axes["a"], ctx)
    drawn["_b_source"] = draw_panel("b", axes["b"], ctx)
    drawn["d"] = draw_panel("c", axes["c"], ctx, shared_axes=shared["c"], colorbar_parent=cbar_parent)
    # The inherited multiscale drawer measures its own legacy letter during
    # construction. Relabel that exact artist after its final geometry pass.
    panel_label(axes["d"], "d")
    drawn["e"] = draw_panel("d", axes["d"], ctx, shared_axes=shared["d"])
    drawn["f"] = draw_panel("e", axes["e"], ctx)
    reflow_a = _reflow_a(axes["a"], geo, width, height)
    reflow_bc = _reflow_b_c(axes["b"], geo, ctx, width, height)
    drawn["b"] = {**drawn["_b_source"], "plotted_rows": [
        row for row in drawn["_b_source"]["plotted_rows"] if row["role"] == "recipe_transfer_512"]}
    drawn["c"] = {**drawn["_b_source"], "plotted_rows": [
        row for row in drawn["_b_source"]["plotted_rows"] if row["role"] != "recipe_transfer_512"]}
    next(item for item in axes["d"].texts if item.get_text() == "d").set_text("e")
    panel_e_label = _place_panel_e_tag(fig, axes["d"], geo, width)
    for visible, legacy in (("a", "a"), ("b", "b"), ("d", "c"), ("f", "e")):
        panel_label(axes[legacy], visible)
    panel_label(containers["new_c"], "c")
    for axis in fig.axes:
        for item in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            if item.get_visible() and item.get_text():
                item._global_font_role = "tick_label"
    geometric_axis_count = base.enforce_geometric_aspects(fig)
    fig.canvas.draw()
    colorbar_layout, colorbar_axes, colorbar_text = _reflow_de_with_colorbars(
        fig, shared, drawn, width, height, geo,
    )
    center_panel_d_headers(axes["d"], shared["d"])
    v47.v46.v4_5.activate_v4_5_font_roles(cfg)
    role_qa = apply_v4_7_typography(fig)
    typography_qa = manuscript.enforce_figure_typography(fig, font_family=manuscript.FONT_FAMILY)
    center_panel_d_headers(axes["d"], shared["d"])
    alignment = align_panel_cd_annotation_gutters(fig, shared, offset_mm=1.05, strict=True)
    collision = measure_annotation_collision_gate(fig, minimum_clearance_mm=1.0, strict=True)
    colorbar_qa = _colorbar_qa(fig, shared, colorbar_axes, colorbar_text)
    matrix_ticks = record_panel_e_v4_3_tick_settings(axes["e"], strict=True)
    topology = _topology_qa(fig, axes, containers, shared, geo, width, height)
    upper_layout_qa = _upper_layout_qa(fig, axes, containers, geo, panel_e_label)
    alignment_qa = v4._panel_cd_alignment_qa(fig, shared, cfg["figure_style"]["paper_dpi"])
    frame_qa = base.enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    model_qa = v4._model_artist_qa(fig, cfg)
    scientific = _compare_science(baseline, drawn)
    v5_0_continuity = _compare_v5_0(requested_baseline, drawn)
    panel_text_clearance = validate_panel_text_boundaries(fig, containers)
    if not all((
        scientific["status"] == "PASS", v5_0_continuity["status"] == "PASS",
        alignment_qa["passed"], model_qa["passed"],
        typography_qa["passed"], frame_qa["passed"], collision["passed"],
        topology["passed"], upper_layout_qa["passed"],
        panel_text_clearance["passed"], colorbar_qa["passed"],
    )):
        print(json.dumps({"scientific": scientific["status"],
                          "v5_0_continuity": v5_0_continuity["status"],
                          "alignment": alignment_qa["passed"],
                          "model": model_qa["passed"], "typography": typography_qa["passed"],
                          "frame": frame_qa["passed"], "collision": collision["passed"],
                          "topology": topology["passed"], "panel_text": panel_text_clearance["passed"],
                          "upper_layout": upper_layout_qa,
                          "colorbar": colorbar_qa}, indent=2))
        raise RuntimeError("V5_3 scientific, typography, alignment, or layout gate failed")
    stem = release / f"MixedResolution_unified_v5_3_{rid}"
    outputs = save_figure(fig, stem, cfg, formats=("svg", "pdf", "png"),
                          dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None)
    plt.close(fig)
    if results_before != v4._tree_state(RESULTS_DIR):
        raise RuntimeError("Validated process results changed during V5_3 rendering")
    if baseline_hash != _sha256(baseline_pdf):
        raise RuntimeError("V4_7 baseline PDF changed during V5_3 rendering")
    if v5_0_hash != _sha256(V5_0_PDF):
        raise RuntimeError("V5_0 requested baseline PDF changed during V5_3 rendering")
    previews = _previews(stem.with_suffix(".pdf"), stem.with_suffix(".png"), release)
    panels = _publication_panel_metadata(drawn)
    manifest = {
        "schema_version": "5.3", "revision": "V5_3", "run_id": rid,
        "workflow_label": "mixed_resolution_unified_v5_3",
        "release_status": "art reviewed, scientific release pending (A03/A04)",
        "backend": "Python/Matplotlib in fig environment",
        "canvas_mm": [width, height], "source_scientific_revision": "V3-7",
        "source_visual_revision": "V5_0", "v4_7_baseline_pdf": _record(baseline_pdf),
        "requested_v5_0_baseline_pdf": _record(V5_0_PDF),
        "source_data_records": baseline["source_data_records"],
        "cache_source_records": baseline["cache_source_records"],
        "scientific_comparison": scientific,
        "requested_baseline_continuity": v5_0_continuity,
        "panels": panels,
        "layout": {"panel_rectangles_mm": {
            key: geo[f"panel_{key}"] for key in "abcdef"},
            "panel_a_reflow": reflow_a, "panel_b_c_reflow": reflow_bc,
            "topology_qa": topology, "upper_layout_qa": upper_layout_qa,
            "annotation_collision_qa": collision,
            "colorbar_layout": colorbar_layout, "colorbar_qa": colorbar_qa,
            "panel_e_label_x_mm": panel_e_label["x_mm"],
            "cross_panel_alignment_qa": alignment_qa,
            "panel_text_clearance_qa": panel_text_clearance,
        },
        "style_contract": style_record,
        "qa": {"typography": typography_qa, "typography_roles": role_qa,
               "geometric_axis_count": geometric_axis_count,
               "frame_lineweight": frame_qa, "model_artist": model_qa,
               "lower_annotation_alignment": alignment,
               "matrix_tick_settings": matrix_ticks},
        "outputs": {Path(path).suffix.lstrip("."): _record(Path(path)) for path in outputs},
        "previews": previews,
        "renderer": _record(Path(__file__)), "layout_source": _record(args.layout),
        "model_training_or_inference": False,
        "new_prediction_or_evaluation_rows_generated": False,
        "cached_field_display_metrics_recomputed_by_inherited_drawer": True,
    }
    _write_json(release / "source_manifest_v5_3.json", manifest)
    _write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", scientific)
    _write_json(release / "LAYOUT_QA.json", {
        "revision": "V5_3", "tested_design_width_mm": width,
        "insertion_width_mm": 162.0, "topology_qa": topology,
        "annotation_collision_qa": collision, "alignment_qa": alignment_qa,
        "colorbar_qa": colorbar_qa, "upper_layout_qa": upper_layout_qa,
        "requested_baseline_continuity": v5_0_continuity,
        "typography_qa": typography_qa, "model_artist_qa": model_qa,
        "panel_text_clearance_qa": panel_text_clearance,
        "by_width": {
            "180mm": {"minimum_vertical_gap_mm": min(topology["vertical_artist_gaps_mm"].values()),
                      "minimum_horizontal_gap_mm": min(topology["horizontal_artist_gaps_mm"].values()),
                      "passed": topology["passed"]},
            "162mm": {"minimum_vertical_gap_mm": .9 * min(topology["vertical_artist_gaps_mm"].values()),
                      "minimum_horizontal_gap_mm": .9 * min(topology["horizontal_artist_gaps_mm"].values()),
                      "vector_pdf_preview": previews["vector_162mm"],
                      "passed": topology["passed"]},
        },
    })
    _write_json(release / "SOURCE_LOCK.json", {
        "revision": "V5_3", "status": "RECORDED", "run_id": rid,
        "backend": "Python/Matplotlib in fig environment",
        "baseline": _record(baseline_pdf), "requested_v5_0_baseline": _record(V5_0_PDF),
        "renderer": _record(Path(__file__)), "layout": _record(args.layout),
        "source_hashes": source_hashes,
        "process_results_tree_unchanged": True,
        "model_training_or_inference": False,
        "new_prediction_or_evaluation_rows_generated": False,
        "cached_field_display_metrics_recomputed_by_inherited_drawer": True,
    })
    (release / "figure_contract.md").write_text(
        f"# Mixed-resolution Figure V5_3 contract\n\n"
        f"- Claim and displayed quantitative values: unchanged from the requested V5_0 PDF.\n"
        f"- Backend: Python/Matplotlib in the `fig` environment.\n"
        f"- Canvas: {width:.0f} × {height:.1f} mm, identical to V5_0.\n"
        f"- Panels: a resolution examples and training budgets; b 512-sensor recipe transfer; "
        f"c three vertically stacked sensor sweeps; d spatial evidence; e multiscale evidence; "
        f"f complete-scale matrices.\n"
        f"- Panels b/c: the grouped-bar chart remains 21 mm high; each of the three "
        f"sweeps is 19.07 mm high, enlarged from 17 mm. Their bottom plot edges align "
        f"at {geo['panel_b_bar_mm'][1]:.1f} mm. The legend-to-plot and both plot-to-plot "
        f"gaps are 3.6 mm, 40% below the previous 6 mm sweep gaps. The legend ends "
        f"1 mm below the canvas top. Panel b's container is 35.2 mm high.\n"
        f"- Panel d: vertical field bar spans the first two image rows, with limits "
        f"{drawn['d']['field_limits']} in field units; the local absolute-error bar has limits "
        f"{drawn['d']['error_limits']}. Labels on both bars are shown ×10⁻².\n"
        f"- Panel e: vertical grayscale and magma bars show fractions of each row's own "
        f"physical maximum. Truth component limits by row: "
        f"{drawn['e']['component_color_limits']}; the original symmetric residual limits "
        f"by row are {drawn['e']['residual_color_limits']}. The displayed residual "
        f"magnitudes run from zero to each positive endpoint. The row-wise image "
        f"normalizations are unchanged.\n"
        f"- The panel-e label sits in the left gutter at "
        f"{panel_e_label['x_mm']:.1f} mm; the redundant 'row max' label was removed.\n"
        f"- Sensor sweeps: Mixed-HML, Zero-H-balanced, Zero-H-M-rich; one shared log-y range "
        f"and one shared four-model legend.\n"
        f"- Data boundary: no source rows, estimates, 95% confidence intervals, field arrays, "
        f"normalizations, categories or model identities changed. The inherited qualitative "
        f"drawers recompute display metrics from unchanged cached fields, and the resulting "
        f"values match V5_0 exactly.\n"
        f"- The five values above the panel-a bars are relative spatial-field exposure.\n"
        f"- Status: **art reviewed, scientific release pending** (existing A03/A04 author decisions).\n",
        encoding="utf-8",
    )
    (release / "STYLE_CHANGELOG.md").write_text(
        f"# V5_0 → V5_3 visual changes\n\n"
        f"- Added two vertical scale bars to panel d, for the field and local absolute error.\n"
        f"- Added two row-normalized scale bars to panel e, for truth and residual magnitude.\n"
        f"- Compressed the image-column gaps and placed panel d near the left edge and "
        f"panel e's scale near the right edge.\n"
        f"- Refined V5_3 by moving its e tag into the left gutter, removing 'row max', "
        f"reducing panel-c sweep gaps from 6 to 3.6 mm, lowering sweep subtitles, "
        f"aligning b/c chart bottoms, and shortening the panel-b container by 1.8 mm.\n"
        f"- Filled panel c's upper blank space by enlarging its three sweeps to 19.07 mm "
        f"and placing the legend 1 mm below the canvas top; its three internal "
        f"vertical gutters are equal at 3.6 mm.\n"
        f"- Preserved the 180 × {height:.1f} mm V5_0 canvas, source hashes, values and "
        f"all other panels; A03/A04 remain open.\n",
        encoding="utf-8",
    )
    source_snapshot = release / "source"
    source_snapshot.mkdir()
    shutil.copy2(Path(__file__), source_snapshot / Path(__file__).name)
    shutil.copy2(args.layout, source_snapshot / args.layout.name)
    if not args.trial_dir:
        assembled = FIGURES_DIR / "Assembled"
        assembled.mkdir(parents=True, exist_ok=True)
        for ext in (".pdf", ".svg", ".png"):
            target = assembled / f"MixedResolution_unified_v5_3_{rid}{ext}"
            if target.exists():
                raise FileExistsError(target)
            shutil.copy2(stem.with_suffix(ext), target)
        target = assembled / f"FigureSourceManifest_unified_v5_3_{rid}.json"
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(release / "source_manifest_v5_3.json", target)
    print(f"[OK] {stem.with_suffix('.pdf')}")
    print(f"[OK] scientific continuity: {scientific['status']}; height reduction: {topology['height_reduction_percent']:.1f}%")


if __name__ == "__main__":
    main()

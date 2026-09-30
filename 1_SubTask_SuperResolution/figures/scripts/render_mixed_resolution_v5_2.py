#!/usr/bin/env python
"""Reflow the frozen V4_7 mixed-resolution evidence as six-panel V5_2.

This renderer reads the existing publication tables and cached panels. V5_2
changes row placement, spacing, and a few label positions while preserving the
visible-word multiset and quantitative content from the exact V5_0 PDF. It also
gates the original panel-f matrix dimensions and final rendered bottom margin.
It does not train a model or infer a field. The inherited qualitative drawers
recompute display metrics from unchanged caches; their values are checked
against V4_7.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.axes
import matplotlib.pyplot as plt
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

LAYOUT_PATH = SCRIPTS / "publication_layout_unified_v5_2.yaml"
BASELINE_DIR = (
    ROOT / "figures" / "generated" / "art_style_review"
    / "MixedResolution_unified_v4_7_20260914_2318"
)
BASELINE_MANIFEST = BASELINE_DIR / "source_manifest_v4_7.json"
V5_0_DIR = (
    ROOT / "figures" / "generated" / "art_style_review"
    / "MixedResolution_unified_v5_0_20260926_1336"
)
V5_0_MANIFEST = V5_0_DIR / "source_manifest_v5_0.json"
V5_0_QA = V5_0_DIR / "LAYOUT_QA.json"
V5_0_PDF = (
    ROOT / "Save_TrainedModel" / "_TrainedModels" / "_Process_Figures"
    / "Assembled" / "MixedResolution_unified_v5_0_20260926_1336.pdf"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v47 = _load_module("mixed_resolution_v4_7_for_v5_2", SCRIPTS / "139_assemble_mixed_resolution_unified_v4_7.py")
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
    # The baseline's 3.5-pt label pad left only ~1 mm above panel b after its
    # chart was lowered to align the b/c plot bottoms. Lift just these labels;
    # map and grouped-bar axes retain their original physical dimensions.
    bar.tick_params(axis="x", pad=0.5)
    # V5_0 displayed the recipe names under both a and b; retain those artists.
    bar.set_xlabel("")
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
        "recipe_axis_shared_with_panel_b": True,
    }


def _reflow_b_c(parent, geo: dict, ctx, width: float, height: float) -> dict:
    panel_module = __import__("common.publication_panels_unified_v4_6", fromlist=["_panel_b_axes"])
    bar, sweeps = panel_module._panel_b_axes(parent)
    if len(sweeps) != 3:
        raise RuntimeError("V5_2 requires all three saved sensor sweeps")
    _place(bar, geo["panel_b_bar_mm"], width, height)
    bar.yaxis.set_label_coords(-0.04, 0.5)
    # The inherited inside-the-bars heading duplicates the normal subplot
    # title and crowds the shorter V5_2 bar axis.
    for item in list(bar.texts):
        if item.get_gid() == "panel-b-inside-title":
            item.remove()
    # Keep the full title inside the bar chart's upper blank band. The normal
    # Axes title is renderer-managed above the axes, where it collides with
    # panel-a's two-line recipe labels after the row gap is tightened.
    bar.set_title("")
    b_title = bar.text(
        0.5, 0.99, "High-resolution reconstruction (512 sensors)",
        transform=bar.transAxes, ha="center", va="top", clip_on=True,
    )
    b_title.set_gid("panel-b-internal-title")
    recipe_labels = [item.get_text() for item in bar.get_xticklabels()]
    bar.tick_params(axis="x", pad=1.0)
    for index, (axis, rect) in enumerate(zip(sweeps, geo["panel_c_sweep_mm"])):
        _place(axis, rect, width, height)
        # Limit tick artists to values actually inside the frozen log range.
        # Matplotlib otherwise instantiates a clipped 10^1 tick above the
        # compact top panel, which fails fixed-canvas text QA.
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
        handles=handles, ncol=2, loc="center", frameon=False,
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
        "recipe_labels": recipe_labels,
        "sweep_count": len(sweeps),
    }


def _compact_f_scale_ticks(parent) -> dict:
    """Retain V5_0 scale wording and rotate labels for panel-f clearance."""
    axes = [axis for axis in parent.figure.findobj(match=lambda item: isinstance(item, matplotlib.axes.Axes))
            if str(axis.get_gid() or "") == "panel-e-metric-matrix"]
    if len(axes) != 6:
        raise RuntimeError(f"Expected six matrix axes, found {len(axes)}")
    for axis in axes:
        axis.set_xticks(axis.get_xticks(), ["Large", "Interm.", "Fine"])
        axis.tick_params(axis="x", pad=1.5)
        for item in axis.get_xticklabels():
            item.set_rotation(45)
            item.set_horizontalalignment("right")
            item.set_rotation_mode("anchor")
            item._global_font_role = "tick_label"
    return {"matrix_axis_count": len(axes), "displayed_scale_labels": ["Large", "Interm.", "Fine"],
            "source_wording_retained": True}


def _place_panel_c_titles(parent) -> dict:
    """Move the three V5_0 sweep titles into open upper-right plot space."""
    panel_module = __import__("common.publication_panels_unified_v4_6", fromlist=["_panel_b_axes"])
    _, sweeps = panel_module._panel_b_axes(parent)
    expected = ["Mixed-HML", "Zero-H-balanced", "Zero-H-M-rich"]
    observed = [axis.get_title() for axis in sweeps]
    if observed != expected:
        raise RuntimeError(f"V5_2 sensor-sweep labels changed: {observed}")
    for axis in sweeps:
        title_text = axis.get_title()
        axis.set_title("")
        title = axis.text(
            0.975, 0.96, title_text, transform=axis.transAxes,
            ha="right", va="top", clip_on=True,
        )
        title.set_gid("panel-c-sweep-title")
    return {
        "labels": observed,
        "placement_axes_fraction": [0.975, 0.96],
        "horizontal_alignment": "right",
        "vertical_alignment": "top",
        "inside_subplot_empty_space": True,
    }


def _assign_panel_c_axes_to_container(parent, panel_c_container) -> dict:
    """Give the split c sweep/legend axes the c owner used by boundary QA."""
    panel_module = __import__("common.publication_panels_unified_v4_6", fromlist=["_panel_b_axes"])
    _, sweeps = panel_module._panel_b_axes(parent)
    legends = _children_with(parent, lambda axis: axis.get_legend() is not None)
    moved = [*sweeps, *legends]
    for axis in moved:
        if axis in parent.child_axes:
            parent.child_axes.remove(axis)
        if axis not in panel_c_container.child_axes:
            panel_c_container.child_axes.append(axis)
        if axis in sweeps:
            axis.set_gid(f"panel-c-sweep-{sweeps.index(axis) + 1}")
    return {"sweep_axis_count": len(sweeps), "legend_axis_count": len(legends),
            "reparented_for_panel_boundary_ownership": len(moved)}


def _box_mm(box, fig) -> list[float]:
    scale = 25.4 / fig.dpi
    return [float(value * scale) for value in (box.x0, box.y0, box.x1, box.y1)]


def _tight_box(axis, renderer):
    try:
        return axis.get_tightbbox(renderer)
    except Exception:
        return None


def _point_rect_distance(point, box) -> float:
    dx = max(box.x0 - point[0], 0.0, point[0] - box.x1)
    dy = max(box.y0 - point[1], 0.0, point[1] - box.y1)
    return float(np.hypot(dx, dy))


def _point_segment_distance(point, a, b) -> float:
    delta = b - a
    denominator = float(np.dot(delta, delta))
    if denominator == 0.0:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, delta) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * delta)))


def _segments_intersect(a, b, c, d) -> bool:
    def orient(p, q, r):
        u, v = q - p, r - p
        return float(u[0] * v[1] - u[1] * v[0])
    o1, o2 = orient(a, b, c), orient(a, b, d)
    o3, o4 = orient(c, d, a), orient(c, d, b)
    return o1 * o2 <= 1e-9 and o3 * o4 <= 1e-9


def _segment_segment_distance(a, b, c, d) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
               _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))


def _segment_box_distance(a, b, box) -> float:
    corners = [np.array([box.x0, box.y0]), np.array([box.x1, box.y0]),
               np.array([box.x1, box.y1]), np.array([box.x0, box.y1])]
    edges = list(zip(corners, corners[1:] + corners[:1]))
    return min(_segment_segment_distance(a, b, c, d) for c, d in edges)


def _path_box_distance(path, transform, box, *, filled=False) -> float:
    display_path = transform.transform_path(path)
    vertices = np.asarray(display_path.vertices, dtype=float)
    vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
    if len(vertices) == 0:
        return float("inf")
    if filled:
        probes = [np.array([box.x0, box.y0]), np.array([box.x0, box.y1]),
                  np.array([box.x1, box.y0]), np.array([box.x1, box.y1]),
                  np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2])]
        if any(display_path.contains_point(point) for point in probes):
            return 0.0
    distance = min(_point_rect_distance(point, box) for point in vertices)
    closed = bool(display_path.codes is not None
                  and np.any(display_path.codes == matplotlib.path.Path.CLOSEPOLY))
    for a, b in zip(vertices[:-1], vertices[1:]):
        distance = min(distance, _segment_box_distance(a, b, box))
    if closed and len(vertices) > 2:
        distance = min(distance, _segment_box_distance(vertices[-1], vertices[0], box))
    return distance


def _panel_c_title_collision_qa(fig, parent, gate: dict) -> dict:
    """Measure final-renderer title clearance from curves, intervals, and text."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_module = __import__("common.publication_panels_unified_v4_6", fromlist=["_panel_b_axes"])
    _, sweeps = panel_module._panel_b_axes(parent)
    minimum_px = float("inf")
    records = []
    intersections = []
    for index, axis in enumerate(sweeps):
        title = next((item for item in axis.texts
                      if item.get_gid() == "panel-c-sweep-title"), None)
        if title is None:
            intersections.append({"title": f"sweep-{index + 1}",
                                  "problem": "explicit title artist missing"})
            continue
        if not title.get_visible() or not title.get_text().strip():
            intersections.append({"title": f"sweep-{index + 1}",
                                  "problem": "explicit title artist is hidden"})
            continue
        title_box = title.get_window_extent(renderer)
        axis_box = axis.get_window_extent(renderer)
        if not (axis_box.x0 <= title_box.x0 and title_box.x1 <= axis_box.x1
                and axis_box.y0 <= title_box.y0 and title_box.y1 <= axis_box.y1):
            intersections.append({"title": title.get_text(), "problem": "title outside subplot"})
        distances = []
        for line in axis.lines:
            transformed = line.get_transform().transform_path(line.get_path())
            vertices = np.asarray(transformed.vertices, dtype=float)
            vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
            distance = min((_segment_box_distance(a, b, title_box)
                            for a, b in zip(vertices[:-1], vertices[1:])), default=float("inf"))
            if len(vertices) == 1:
                distance = min(distance, _point_rect_distance(vertices[0], title_box))
            marker_pt = 0.0 if line.get_marker() in (None, "", "None") else line.get_markersize()
            stroke_px = max(float(line.get_linewidth()), 0.0) * fig.dpi / 72.0 / 2.0
            marker_px = float(marker_pt) * fig.dpi / 72.0 / 2.0
            distances.append((distance - max(stroke_px, marker_px), "Line2D"))
        for collection in axis.collections:
            name = collection.__class__.__name__
            offsets = np.asarray(collection.get_offsets()) if hasattr(collection, "get_offsets") else np.empty((0, 2))
            if offsets.size and name == "PathCollection":
                display_offsets = collection.get_offset_transform().transform(offsets)
                sizes = np.asarray(collection.get_sizes(), dtype=float)
                radius_px = (np.sqrt(float(np.max(sizes))) / 2.0 * fig.dpi / 72.0
                             if sizes.size else 0.0)
                distances.extend((_point_rect_distance(point, title_box) - radius_px, name)
                                 for point in display_offsets)
                continue
            linewidths = np.asarray(collection.get_linewidths(), dtype=float)
            stroke_px = (float(np.max(linewidths)) * fig.dpi / 72.0 / 2.0
                         if linewidths.size else 0.0)
            filled = name in {"PolyCollection"}
            for path in collection.get_paths():
                distances.append((_path_box_distance(path, collection.get_transform(),
                                                     title_box, filled=filled) - stroke_px, name))
        for other in fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text)):
            if other is title or not other.get_visible() or not other.get_text().strip():
                continue
            other_box = other.get_window_extent(renderer)
            if title_box.overlaps(other_box):
                intersections.append({"title": title.get_text(), "other_text": other.get_text()})
        if distances:
            nearest_px, artist_type = min(distances, key=lambda entry: entry[0])
        else:
            nearest_px, artist_type = float("inf"), "none"
        minimum_px = min(minimum_px, nearest_px)
        clearance_mm = nearest_px * 25.4 / fig.dpi
        records.append({"title": title.get_text(), "bbox_mm": _box_mm(title_box, fig),
                        "nearest_data_artist": artist_type, "curve_clearance_mm": clearance_mm,
                        "within_subplot": bool(axis_box.x0 <= title_box.x0 and title_box.x1 <= axis_box.x1
                                                and axis_box.y0 <= title_box.y0 and title_box.y1 <= axis_box.y1)})
    minimum_mm = minimum_px * 25.4 / fig.dpi
    required = float(gate["minimum_panel_c_title_curve_clearance_mm"])
    passed = (not intersections and minimum_mm >= required
              and all(record["within_subplot"] for record in records))
    return {"schema_version": "panel_c_title_data_clearance_v1", "records": records,
            "text_intersections": intersections, "minimum_curve_clearance_mm": minimum_mm,
            "required_minimum_clearance_mm": required, "passed": bool(passed)}


def _panel_b_title_collision_qa(fig, parent, gate: dict) -> dict:
    """Measure the in-axes b title against the bars and error bars."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_module = __import__("common.publication_panels_unified_v4_6", fromlist=["_panel_b_axes"])
    bar, _ = panel_module._panel_b_axes(parent)
    title = next((item for item in bar.texts
                  if item.get_gid() == "panel-b-internal-title"), None)
    problems = []
    if title is None or not title.get_visible() or not title.get_text().strip():
        return {"schema_version": "panel_b_title_data_clearance_v1", "passed": False,
                "problems": ["explicit b title artist missing or hidden"], "records": []}
    title_box = title.get_window_extent(renderer)
    axis_box = bar.get_window_extent(renderer)
    within_axes = (axis_box.x0 <= title_box.x0 and title_box.x1 <= axis_box.x1
                   and axis_box.y0 <= title_box.y0 and title_box.y1 <= axis_box.y1)
    if not within_axes:
        problems.append("title outside panel-b plot axes")
    distances = []
    bar_count = 0
    errorbar_count = 0
    for container in bar.containers:
        for patch in getattr(container, "patches", ()):
            if not patch.get_visible():
                continue
            bar_count += 1
            distance = _path_box_distance(
                patch.get_path(), patch.get_transform(), title_box, filled=True,
            )
            linewidth = max(np.asarray(patch.get_linewidth(), dtype=float).reshape(-1), default=0.0)
            distances.append((distance - float(linewidth) * fig.dpi / 72.0 / 2.0, "bar"))
        errorbar = getattr(container, "errorbar", None)
        if errorbar is None:
            continue
        data_line, caplines, barlinecols = errorbar.lines
        lines = [line for line in ((data_line,) if data_line is not None else ())
                 + tuple(caplines) if line is not None and line.get_visible()]
        for line in lines:
            transformed = line.get_transform().transform_path(line.get_path())
            vertices = np.asarray(transformed.vertices, dtype=float)
            vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
            distance = min((_segment_box_distance(a, b, title_box)
                            for a, b in zip(vertices[:-1], vertices[1:])), default=float("inf"))
            if len(vertices) == 1:
                distance = min(distance, _point_rect_distance(vertices[0], title_box))
            stroke_px = max(float(line.get_linewidth()), 0.0) * fig.dpi / 72.0 / 2.0
            marker_px = (float(line.get_markersize()) * fig.dpi / 72.0 / 2.0
                         if line.get_marker() not in (None, "", "None") else 0.0)
            distances.append((distance - max(stroke_px, marker_px), "errorbar"))
            errorbar_count += 1
        for collection in barlinecols:
            if not collection.get_visible():
                continue
            stroke_px = (float(np.max(collection.get_linewidths())) * fig.dpi / 72.0 / 2.0
                         if len(collection.get_linewidths()) else 0.0)
            for path in collection.get_paths():
                distances.append((_path_box_distance(
                    path, collection.get_transform(), title_box,
                ) - stroke_px, "errorbar"))
                errorbar_count += 1
    intersecting_text = []
    for other in fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text)):
        if other is title or not other.get_visible() or not other.get_text().strip():
            continue
        owner = getattr(other, "axes", None)
        is_axis_tick = False
        if owner is None:
            for candidate in fig.axes:
                tick_items = [*candidate.get_xticklabels(minor=False),
                              *candidate.get_xticklabels(minor=True),
                              *candidate.get_yticklabels(minor=False),
                              *candidate.get_yticklabels(minor=True)]
                direct_items = [candidate.title, candidate.xaxis.label,
                                candidate.yaxis.label, candidate.xaxis.offsetText,
                                candidate.yaxis.offsetText, *candidate.texts, *tick_items]
                if any(item is other for item in direct_items):
                    owner = candidate
                    is_axis_tick = any(item is other for item in tick_items)
                    break
        if owner is not None and not owner.get_visible():
            continue
        other_box = other.get_window_extent(renderer)
        if owner is not None and (other.get_clip_on() or is_axis_tick):
            owner_box = owner.get_window_extent(renderer)
            anchor = other.get_transform().transform(other.get_position())
            anchor_outside = not (owner_box.x0 <= anchor[0] <= owner_box.x1
                                  and owner_box.y0 <= anchor[1] <= owner_box.y1)
            wholly_outside = (other_box.x1 <= owner_box.x0 or other_box.x0 >= owner_box.x1
                              or other_box.y1 <= owner_box.y0 or other_box.y0 >= owner_box.y1)
            if (is_axis_tick and not owner.axison) or anchor_outside or wholly_outside:
                continue
        if title_box.overlaps(other_box):
            problems.append(f"title intersects adjacent text: {other.get_text()}")
            intersecting_text.append({"text": other.get_text(), "bbox_mm": _box_mm(other_box, fig),
                                      "owner_axes_visible": owner.get_visible() if owner else None,
                                      "owner_axes_label": owner.get_label() if owner else None,
                                      "is_axis_tick": is_axis_tick})
    nearest_px, nearest_kind = (min(distances, key=lambda entry: entry[0])
                                if distances else (float("inf"), "none"))
    clearance_mm = nearest_px * 25.4 / fig.dpi
    required_mm = float(gate["minimum_panel_b_title_data_clearance_mm"])
    passed = (not problems and within_axes and bar_count == 20 and errorbar_count > 0
              and clearance_mm >= required_mm)
    return {"schema_version": "panel_b_title_data_clearance_v1", "passed": bool(passed),
            "title": title.get_text(), "title_bbox_mm": _box_mm(title_box, fig),
            "nearest_data_artist": nearest_kind, "minimum_data_clearance_mm": clearance_mm,
            "required_minimum_clearance_mm": required_mm, "within_subplot": within_axes,
            "bar_patch_count": bar_count, "errorbar_component_count": errorbar_count,
            "problems": problems, "intersecting_text": intersecting_text,
            "hidden_or_clipped_axis_text_excluded": True}


class _PanelBoundaryProxy:
    """Delegate panel artists while slightly extending only its text-audit edge."""

    def __init__(self, axis, *, bottom_extension_mm: float, fig):
        self._axis = axis
        self._bottom_extension_px = float(bottom_extension_mm) / 25.4 * float(fig.dpi)

    def __getattr__(self, name):
        return getattr(self._axis, name)

    def get_window_extent(self, renderer=None):
        box = self._axis.get_window_extent(renderer)
        return matplotlib.transforms.Bbox.from_extents(
            box.x0, box.y0 - self._bottom_extension_px, box.x1, box.y1,
        )


def _panel_text_clearance_v5_2(fig, containers: dict, gate: dict) -> dict:
    """Check every panel's text, allowing only the known V5.0 f tick overhang."""
    audited = dict(containers)
    extension_mm = float(gate["panel_f_text_audit_bottom_extension_mm"])
    audited["e"] = _PanelBoundaryProxy(containers["e"], bottom_extension_mm=extension_mm, fig=fig)
    try:
        result = validate_panel_text_boundaries(fig, audited)
    except ValueError as exc:
        return {"passed": False, "failure": str(exc),
                "panel_f_boundary_exception_mm": extension_mm}
    result["panel_f_boundary_exception"] = {
        "legacy_panel": "e", "v5_2_panel": "f", "bottom_extension_mm": extension_mm,
        "reason": "Six V5.0 Interm. tick glyph boxes extend 0.176 mm below the parent; rendered ink remains on-page.",
    }
    return result


def _axis_tree(root, *, include_root=False):
    found = []
    pending = [root] if include_root else list(root.child_axes)
    while pending:
        axis = pending.pop(0)
        if axis in found:
            continue
        found.append(axis)
        pending.extend(axis.child_axes)
    return found


def _visible_direct_text_boxes(axis, renderer):
    texts = [*axis.texts, axis.title, axis.xaxis.label, axis.yaxis.label,
             axis.xaxis.offsetText, axis.yaxis.offsetText]
    if axis.axison:
        texts.extend((*axis.get_xticklabels(minor=False), *axis.get_xticklabels(minor=True),
                      *axis.get_yticklabels(minor=False), *axis.get_yticklabels(minor=True)))
    boxes = []
    for item in texts:
        if not item.get_visible() or not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer)
        if box.width > 0 and box.height > 0:
            boxes.append(box)
    return boxes


def _topology_qa(fig, axes, containers, shared, width, height,
                 gate: dict, v5_0_gaps: dict) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    panel_module = __import__("common.publication_panels_unified_v4_6", fromlist=["_panel_b_axes"])
    bar, sweeps = panel_module._panel_b_axes(axes["b"])
    legend_axis = next(axis for axis in axes["b"].child_axes if axis.get_legend() is not None)
    # Measure rendered content, not the empty panel-container rectangles.
    # The in-axes titles are included through their owning data axes; panel
    # letters and parent-level column headings are added as text bboxes.
    groups = {
        "a": (_axis_tree(axes["a"]), [axes["a"]]),
        "b": ([bar], []),
        "c": ([*sweeps, legend_axis], [containers["new_c"]]),
        "d": ([axis for row in shared["c"] for axis in row], [axes["c"]]),
        "e": ([axis for row in shared["d"] for axis in row], [axes["d"]]),
        "f": (_axis_tree(axes["e"]), [axes["e"]]),
    }
    boxes = {}
    for label, (group_axes, text_roots) in groups.items():
        found = [box for axis in group_axes if (box := _tight_box(axis, renderer)) is not None]
        found.extend(box for root in text_roots for box in _visible_direct_text_boxes(root, renderer))
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
    bottom_alignment_delta = abs(float(boxes["b"].y0 - boxes["c"].y0)) * 25.4 / fig.dpi
    bottom_alignment_tolerance = float(gate["panel_bc_bottom_alignment_tolerance_mm"])
    sweep_positions = sorted(
        [[float(axis.get_position().x0 * width), float(axis.get_position().y0 * height),
          float(axis.get_position().width * width), float(axis.get_position().height * height)]
         for axis in sweeps], key=lambda rect: rect[1]
    )
    sweep_heights = [rect[3] for rect in sweep_positions]
    sweep_gaps = [sweep_positions[index + 1][1]
                  - (sweep_positions[index][1] + sweep_positions[index][3])
                  for index in range(len(sweep_positions) - 1)]
    bottom_margin = float(boxes["f"].y0) * 25.4 / fig.dpi
    minimum_bottom_margin = float(gate["minimum_page_bottom_margin_mm"])
    overflow = dict(manuscript.validate_text_within_canvas(fig))
    result = {
        "canvas_mm": [width, height],
        "baseline_canvas_mm": [180.0, float(gate["baseline_height_mm"])],
        "height_reduction_mm": float(gate["baseline_height_mm"]) - height,
        "height_reduction_percent": 100.0 * (float(gate["baseline_height_mm"]) - height)
        / float(gate["baseline_height_mm"]),
        "panel_artist_union_mm": {key: _box_mm(value, fig) for key, value in boxes.items()},
        "vertical_artist_gaps_mm": vertical_gaps,
        "v5_0_vertical_artist_gaps_mm": v5_0_gaps,
        "vertical_gap_fraction_of_v5_0": {
            key: vertical_gaps[key] / float(v5_0_gaps[key]) for key in vertical_gaps
        },
        "horizontal_artist_gaps_mm": horizontal_gaps,
        "panel_b_c_bottom_artist_delta_mm": bottom_alignment_delta,
        "panel_b_c_bottom_alignment_tolerance_mm": bottom_alignment_tolerance,
        "panel_b_c_bottoms_aligned": bottom_alignment_delta <= bottom_alignment_tolerance,
        "panel_c_sweep_axes_mm_bottom_up": sweep_positions,
        "panel_c_sweep_axis_heights_mm": sweep_heights,
        "panel_c_internal_axis_gaps_mm": sweep_gaps,
        "panel_c_preserves_17mm_axis_height": all(abs(value - 17.0) <= 0.01 for value in sweep_heights),
        "panel_c_internal_gaps_2mm": all(abs(value - 2.0) <= 0.01 for value in sweep_gaps),
        "panel_f_page_bottom_margin_mm": bottom_margin,
        "minimum_page_bottom_margin_mm": minimum_bottom_margin,
        "panel_f_bottom_margin_passed": bottom_margin >= minimum_bottom_margin,
        "text_overflow": overflow,
        "no_canvas_text_overflow": not any(overflow.values()),
    }
    height_range = list(map(float, gate["target_total_height_reduction_percent"]))
    gap_range = list(map(float, gate["target_artist_gap_fraction"]))
    result["height_target_passed"] = height_range[0] <= result["height_reduction_percent"] <= height_range[1]
    result["row_gap_target_passed"] = all(
        gap_range[0] <= ratio <= gap_range[1]
        for ratio in result["vertical_gap_fraction_of_v5_0"].values()
    )
    result["passed"] = (
        result["no_canvas_text_overflow"]
        and min(vertical_gaps.values()) * 0.9 >= float(gate["minimum_artist_gap_at_162mm"])
        and min(horizontal_gaps.values()) * 0.9 >= 2.5
        and result["height_target_passed"]
        and result["row_gap_target_passed"]
        and result["panel_b_c_bottoms_aligned"]
        and result["panel_c_preserves_17mm_axis_height"]
        and result["panel_c_internal_gaps_2mm"]
        and result["panel_f_bottom_margin_passed"]
    )
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


def _publication_panel_metadata(drawn: dict) -> dict:
    """Store V5_2 panel roles without stale V4_7 cross-panel geometry fields."""
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
            "panel_id": f"fig3.{label}", "figure_revision": "V5_2",
            "source_visual_revision": "V5_0", "source_scientific_revision": "V3-7",
            "underlying_source_arrays_preserved": True,
            "model_training_or_inference": False,
        })
    return panels


def _compare_v5_0(v5_0: dict, panels: dict) -> dict:
    keys = {
        "a": ("recipe_order", "dimensions", "field_limits", "bar_segments", "exposure_values"),
        "b": ("recipe_order", "recipe_transfer_values", "recipe_transfer_y_range", "plotted_rows"),
        "c": ("sweep_recipes", "sensor_counts", "sensor_density_percent",
              "sensor_sweep_values", "sensor_sweep_y_range", "plotted_rows"),
        "d": ("roi", "full_field_relative_l2", "local_relative_l2", "field_limits", "error_limits"),
        "e": ("row_order", "relative_l2", "truth_component_limits", "residual_limits"),
        "f": ("matrix_numeric_annotation_count", "correlation_values", "bias_values"),
    }
    checks = {
        f"{label}:{key}": v5_0["panels"][label].get(key) == panels[label].get(key)
        for label, panel_keys in keys.items() for key in panel_keys
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "scientific_panel_inventory": list(panels)}


def _pdf_word_multiset(path: Path) -> list[str]:
    import fitz
    with fitz.open(path) as document:
        words = [word[4] for page in document for word in page.get_text("words")]
    return sorted(words)


def _pdf_text_content_qa(baseline_pdf: Path, candidate_pdf: Path) -> dict:
    before = _pdf_word_multiset(baseline_pdf)
    after = _pdf_word_multiset(candidate_pdf)
    from collections import Counter
    delta = Counter(after) - Counter(before)
    missing = Counter(before) - Counter(after)
    return {"baseline_word_count": len(before), "candidate_word_count": len(after),
            "added_word_counts": dict(sorted(delta.items())),
            "missing_word_counts": dict(sorted(missing.items())),
            "exact_visible_word_multiset_match": before == after,
            "passed": before == after}


def _panel_f_matrix_image_rects(pdf_path: Path) -> list[list[float]]:
    """Read the six lower-panel heatmap image bounds in design millimetres."""
    import fitz
    with fitz.open(pdf_path) as document:
        if len(document) != 1:
            raise RuntimeError(f"Expected one-page figure PDF: {pdf_path}")
        page = document[0]
        rects = []
        for image in page.get_images(full=True):
            for rect in page.get_image_rects(image[0]):
                bottom_mm = (page.rect.height - rect.y1) * 25.4 / 72.0
                if bottom_mm >= 30.0 or rect.height * 25.4 / 72.0 < 10.0:
                    continue
                rects.append([
                    rect.x0 * 25.4 / 72.0, bottom_mm,
                    rect.width * 25.4 / 72.0, rect.height * 25.4 / 72.0,
                ])
    return sorted(rects, key=lambda item: item[0])


def _panel_f_matrix_size_qa(baseline_pdf: Path, candidate_pdf: Path, gate: dict) -> dict:
    """Require V5_2's six matrix images to retain V5_0 size and aspect."""
    baseline = _panel_f_matrix_image_rects(baseline_pdf)
    candidate = _panel_f_matrix_image_rects(candidate_pdf)
    tolerance = float(gate["maximum_panel_f_matrix_dimension_delta_mm"])
    if len(baseline) != 6 or len(candidate) != 6:
        return {"schema_version": "panel_f_matrix_size_v1", "passed": False,
                "baseline_image_count": len(baseline), "candidate_image_count": len(candidate),
                "maximum_dimension_delta_mm": tolerance,
                "failure": "Expected six V5_0 and six V5_2 matrix images"}
    records = []
    for index, (before, after) in enumerate(zip(baseline, candidate)):
        width_delta = abs(before[2] - after[2])
        height_delta = abs(before[3] - after[3])
        before_aspect = before[2] / before[3]
        after_aspect = after[2] / after[3]
        records.append({
            "matrix_index": index + 1,
            "v5_0_size_mm": before[2:4], "v5_2_size_mm": after[2:4],
            "width_delta_mm": width_delta, "height_delta_mm": height_delta,
            "v5_0_aspect_ratio": before_aspect, "v5_2_aspect_ratio": after_aspect,
            "aspect_ratio_delta": abs(before_aspect - after_aspect),
        })
    passed = all(record["width_delta_mm"] <= tolerance
                 and record["height_delta_mm"] <= tolerance
                 and record["aspect_ratio_delta"] <= 0.002
                 for record in records)
    return {"schema_version": "panel_f_matrix_size_v1", "passed": bool(passed),
            "baseline_image_count": len(baseline), "candidate_image_count": len(candidate),
            "maximum_dimension_delta_mm": tolerance, "records": records,
            "source_pdf": str(baseline_pdf), "candidate_pdf": str(candidate_pdf)}


def _rendered_bottom_ink_margin_qa(png_path: Path, width_mm: float, gate: dict) -> dict:
    """Measure actual nonwhite raster ink below panel f at final export size."""
    with Image.open(png_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    ink_rows = np.flatnonzero(np.any(np.any(rgb < 245, axis=2), axis=1))
    if not len(ink_rows):
        return {"schema_version": "rendered_bottom_ink_margin_v1", "passed": False,
                "failure": "No rendered ink found in PNG", "png": str(png_path)}
    blank_rows = int(rgb.shape[0] - 1 - ink_rows[-1])
    margin_mm = float(blank_rows * float(width_mm) / rgb.shape[1])
    minimum = float(gate["minimum_rendered_bottom_ink_margin_mm"])
    maximum = float(gate["maximum_rendered_bottom_ink_margin_mm"])
    return {"schema_version": "rendered_bottom_ink_margin_v1",
            "passed": minimum <= margin_mm <= maximum,
            "png": str(png_path), "image_pixels": [int(rgb.shape[1]), int(rgb.shape[0])],
            "ink_threshold_rgb_below": 245, "bottom_blank_rows": blank_rows,
            "rendered_bottom_ink_margin_mm": margin_mm,
            "target_margin_mm": [minimum, maximum]}


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
    v5_0 = json.loads(V5_0_MANIFEST.read_text(encoding="utf-8"))
    v5_0_qa = json.loads(V5_0_QA.read_text(encoding="utf-8"))
    v5_0_hash = _sha256(V5_0_PDF)
    generated_v5_0_pdf = V5_0_DIR / V5_0_PDF.name
    if _sha256(generated_v5_0_pdf) != v5_0_hash:
        raise RuntimeError("Named assembled V5_0 PDF differs from its generated source copy")
    baseline_pdf = BASELINE_DIR / "MixedResolution_unified_v4_7_20260914_2318.pdf"
    baseline_hash = _sha256(baseline_pdf)
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
        release = ROOT / "figures" / "generated" / "art_style_review" / f"MixedResolution_unified_v5_2_{rid}"
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
    matrix_ticks = record_panel_e_v4_3_tick_settings(axes["e"], strict=True)
    compact_f_ticks = _compact_f_scale_ticks(axes["e"])
    reflow_a = _reflow_a(axes["a"], geo, width, height)
    reflow_bc = _reflow_b_c(axes["b"], geo, ctx, width, height)
    panel_c_title_placement = _place_panel_c_titles(axes["b"])
    drawn["b"] = {**drawn["_b_source"], "plotted_rows": [
        row for row in drawn["_b_source"]["plotted_rows"] if row["role"] == "recipe_transfer_512"]}
    drawn["c"] = {**drawn["_b_source"], "plotted_rows": [
        row for row in drawn["_b_source"]["plotted_rows"] if row["role"] != "recipe_transfer_512"]}
    next(item for item in axes["d"].texts if item.get_text() == "d").set_text("e")
    for visible, legacy in (("a", "a"), ("b", "b"), ("d", "c"), ("f", "e")):
        panel_label(axes[legacy], visible)
    panel_label(containers["new_c"], "c")
    for axis in fig.axes:
        for item in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            if item.get_visible() and item.get_text():
                item._global_font_role = "tick_label"
    geometric_axis_count = base.enforce_geometric_aspects(fig)
    fig.canvas.draw()
    center_panel_d_headers(axes["d"], shared["d"])
    v47.v46.v4_5.activate_v4_5_font_roles(cfg)
    role_qa = apply_v4_7_typography(fig)
    typography_qa = manuscript.enforce_figure_typography(fig, font_family=manuscript.FONT_FAMILY)
    center_panel_d_headers(axes["d"], shared["d"])
    alignment = align_panel_cd_annotation_gutters(
        fig, shared, offset_mm=float(layout["panel_c_v4"]["bottom_label_offset_mm"]),
        strict=True,
    )
    collision = measure_annotation_collision_gate(
        fig,
        minimum_clearance_mm=float(layout["v5_2_geometry_gate"]["minimum_annotation_data_clearance_mm_at_180mm"]),
        strict=False,
    )
    panel_c_title_qa = _panel_c_title_collision_qa(
        fig, axes["b"], layout["v5_2_geometry_gate"]
    )
    panel_b_title_qa = _panel_b_title_collision_qa(
        fig, axes["b"], layout["v5_2_geometry_gate"]
    )
    topology = _topology_qa(
        fig, axes, containers, shared, width, height,
        layout["v5_2_geometry_gate"],
        v5_0_qa["topology_qa"]["vertical_artist_gaps_mm"],
    )
    alignment_qa = v4._panel_cd_alignment_qa(fig, shared, cfg["figure_style"]["paper_dpi"])
    frame_qa = base.enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    model_qa = v4._model_artist_qa(fig, cfg)
    scientific = _compare_science(baseline, drawn)
    panels = _publication_panel_metadata(drawn)
    continuity_v5_0 = _compare_v5_0(v5_0, panels)
    panel_c_ownership = _assign_panel_c_axes_to_container(axes["b"], containers["new_c"])
    panel_text_clearance = _panel_text_clearance_v5_2(
        fig, containers, layout["v5_2_geometry_gate"]
    )
    gates_passed = all((
        scientific["status"] == "PASS", alignment_qa["passed"], model_qa["passed"],
        typography_qa["passed"], frame_qa["passed"], collision["passed"],
        topology["passed"], panel_text_clearance["passed"],
        panel_c_title_qa["passed"], panel_b_title_qa["passed"],
        continuity_v5_0["status"] == "PASS",
    ))
    if not gates_passed and not args.trial_dir:
        raise RuntimeError("V5_2 scientific, typography, alignment, or layout gate failed")
    stem = release / f"MixedResolution_unified_v5_2_{rid}"
    outputs = save_figure(fig, stem, cfg, formats=("svg", "pdf", "png"),
                          dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None)
    plt.close(fig)
    if results_before != v4._tree_state(RESULTS_DIR):
        raise RuntimeError("Validated process results changed during V5_2 rendering")
    if baseline_hash != _sha256(baseline_pdf):
        raise RuntimeError("V4_7 baseline PDF changed during V5_2 rendering")
    if v5_0_hash != _sha256(V5_0_PDF):
        raise RuntimeError("V5_0 baseline PDF changed during V5_2 rendering")
    previews = _previews(stem.with_suffix(".pdf"), stem.with_suffix(".png"), release)
    pdf_text_content = _pdf_text_content_qa(V5_0_PDF, stem.with_suffix(".pdf"))
    panel_f_size_qa = _panel_f_matrix_size_qa(
        V5_0_PDF, stem.with_suffix(".pdf"), layout["v5_2_geometry_gate"]
    )
    panel_f_ink_margin_qa = _rendered_bottom_ink_margin_qa(
        stem.with_suffix(".png"), width, layout["v5_2_geometry_gate"]
    )
    gates_passed = (gates_passed and pdf_text_content["passed"]
                    and panel_f_size_qa["passed"] and panel_f_ink_margin_qa["passed"])
    if not pdf_text_content["passed"] and not args.trial_dir:
        raise RuntimeError(f"V5_0 visible PDF text changed: {pdf_text_content}")
    if not panel_f_size_qa["passed"] and not args.trial_dir:
        raise RuntimeError(f"V5_0 panel-f image dimensions changed: {panel_f_size_qa}")
    if not panel_f_ink_margin_qa["passed"] and not args.trial_dir:
        raise RuntimeError(f"V5_2 panel-f bottom ink margin is outside its target: {panel_f_ink_margin_qa}")
    manifest = {
        "schema_version": "5.2", "revision": "V5_2", "run_id": rid,
        "workflow_label": "mixed_resolution_unified_v5_2",
        "release_status": "art reviewed, scientific release pending (A03/A04)",
        "backend": "Python/Matplotlib in fig environment",
        "canvas_mm": [width, height], "source_scientific_revision": "V3-7",
        "source_visual_revision": "V5_0", "v4_7_scientific_anchor_pdf": _record(baseline_pdf),
        "v5_0_visual_baseline_pdf": _record(V5_0_PDF),
        "source_data_records": baseline["source_data_records"],
        "cache_source_records": baseline["cache_source_records"],
        "scientific_comparison": scientific,
        "v5_0_continuity": continuity_v5_0,
        "panels": panels,
        "layout": {"panel_rectangles_mm": {
            key: geo[f"panel_{key}"] for key in "abcdef"},
            "panel_a_reflow": reflow_a, "panel_b_c_reflow": reflow_bc,
            "topology_qa": topology, "annotation_collision_qa": collision,
            "panel_c_title_placement": panel_c_title_placement,
            "panel_c_axes_ownership": panel_c_ownership,
            "panel_c_title_curve_clearance_qa": panel_c_title_qa,
            "panel_b_title_data_clearance_qa": panel_b_title_qa,
            "panel_f_matrix_size_qa": panel_f_size_qa,
            "panel_f_rendered_bottom_ink_margin_qa": panel_f_ink_margin_qa,
            "cross_panel_alignment_qa": alignment_qa,
            "panel_text_clearance_qa": panel_text_clearance,
        },
        "style_contract": style_record,
        "qa": {"typography": typography_qa, "typography_roles": role_qa,
               "geometric_axis_count": geometric_axis_count,
               "frame_lineweight": frame_qa, "model_artist": model_qa,
               "lower_annotation_alignment": alignment,
               "matrix_source_tick_settings": matrix_ticks,
               "matrix_display_tick_settings": compact_f_ticks,
               "pdf_text_content": pdf_text_content,
               "panel_f_matrix_size": panel_f_size_qa,
               "panel_f_rendered_bottom_ink_margin": panel_f_ink_margin_qa},
        "outputs": {Path(path).suffix.lstrip("."): _record(Path(path)) for path in outputs},
        "previews": previews,
        "renderer": _record(Path(__file__)), "layout_source": _record(args.layout),
        "model_training_or_inference": False,
        "new_prediction_or_evaluation_rows_generated": False,
        "cached_field_display_metrics_recomputed_by_inherited_drawer": True,
    }
    _write_json(release / "source_manifest_v5_2.json", manifest)
    _write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", {
        "v4_7_scientific_anchor": scientific,
        "v5_0_visual_baseline": continuity_v5_0,
    })
    _write_json(release / "LAYOUT_QA.json", {
        "revision": "V5_2", "tested_design_width_mm": width,
        "insertion_width_mm": 162.0, "topology_qa": topology,
        "annotation_collision_qa": collision, "alignment_qa": alignment_qa,
        "typography_qa": typography_qa, "model_artist_qa": model_qa,
        "panel_text_clearance_qa": panel_text_clearance,
        "panel_c_title_clearance_qa": panel_c_title_qa,
        "panel_b_title_clearance_qa": panel_b_title_qa,
        "panel_f_matrix_size_qa": panel_f_size_qa,
        "panel_f_rendered_bottom_ink_margin_qa": panel_f_ink_margin_qa,
        "pdf_text_content_qa": pdf_text_content,
        "all_gates_passed": gates_passed,
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
        "revision": "V5_2", "status": "RECORDED", "run_id": rid,
        "backend": "Python/Matplotlib in fig environment",
        "baseline": _record(baseline_pdf),
        "v5_0_visual_baseline": _record(V5_0_PDF),
        "renderer": _record(Path(__file__)), "layout": _record(args.layout),
        "source_hashes": source_hashes,
        "process_results_tree_unchanged": True,
        "model_training_or_inference": False,
        "new_prediction_or_evaluation_rows_generated": False,
        "cached_field_display_metrics_recomputed_by_inherited_drawer": True,
    })
    (release / "figure_contract.md").write_text(
        f"# Mixed-resolution Figure V5_2 contract\n\n"
        f"- Claim and displayed quantitative values: unchanged from V5_0 and the V4_7 scientific anchor.\n"
        f"- Backend: Python/Matplotlib in the `fig` environment.\n"
        f"- Canvas: {width:.0f} × {height:.1f} mm, down {topology['height_reduction_percent']:.1f}% from V5_0 (210 mm).\n"
        f"- Panels: a resolution examples and training budgets; b 512-sensor recipe transfer; "
        f"c three vertically stacked sensor sweeps; d spatial evidence; e multiscale evidence; "
        f"f complete-scale matrices.\n"
        f"- Sensor sweeps: Mixed-HML, Zero-H-balanced, Zero-H-M-rich; one shared log-y range "
        f"and one shared four-model legend.\n"
        f"- Data boundary: no source rows, estimates, 95% confidence intervals, field arrays, "
        f"normalizations, categories or model identities changed. The inherited qualitative "
        f"drawers recompute display metrics from unchanged cached fields, and the resulting "
        f"values match V4_7 exactly.\n"
        f"- V5_0 recipe labels remain under both a and b; panel a exposure values are unchanged.\n"
        f"- Matrix scale labels preserve V5_0 wording, including 'Interm.'.\n"
        f"- Panel c sweep titles sit inside open plot space; minimum curve clearance QA: "
        f"{panel_c_title_qa['minimum_curve_clearance_mm']:.2f} mm.\n"
        f"- Panel b title clearance from bars and error bars: "
        f"{panel_b_title_qa['minimum_data_clearance_mm']:.2f} mm.\n"
        f"- Panel b/c bottom artist-edge delta: "
        f"{topology['panel_b_c_bottom_artist_delta_mm']:.3f} mm.\n"
        f"- V5_0 visible-word multiset match: {pdf_text_content['exact_visible_word_multiset_match']}.\n"
        f"- Artist-union row gaps at 180 mm: a-b "
        f"{topology['vertical_artist_gaps_mm']['a-b']:.2f} mm, top-de "
        f"{topology['vertical_artist_gaps_mm']['top-de']:.2f} mm, de-f "
        f"{topology['vertical_artist_gaps_mm']['de-f']:.2f} mm.\n"
        f"- Panel f matrix dimensions match V5_0; rendered bottom ink margin: "
        f"{panel_f_ink_margin_qa['rendered_bottom_ink_margin_mm']:.2f} mm.\n"
        f"- Boundary QA extends only the panel-f text-audit edge by "
        f"{panel_text_clearance['panel_f_boundary_exception']['bottom_extension_mm']:.2f} mm "
        f"for the V5_0 'Interm.' tick glyph boxes; page and cross-panel checks pass.\n"
        f"- Status: **art reviewed, scientific release pending** (existing A03/A04 author decisions).\n",
        encoding="utf-8",
    )
    (release / "STYLE_CHANGELOG.md").write_text(
        f"# V5_0 → V5_2 visual changes\n\n"
        f"- Reduced canvas height from 210.0 to {height:.1f} mm "
        f"({topology['height_reduction_percent']:.1f}%).\n"
        f"- Reduced the measured a-b, upper-to-d/e, and d/e-to-f artist gaps to "
        f"{topology['vertical_gap_fraction_of_v5_0']['a-b']:.3f}, "
        f"{topology['vertical_gap_fraction_of_v5_0']['top-de']:.3f}, and "
        f"{topology['vertical_gap_fraction_of_v5_0']['de-f']:.3f} of V5_0.\n"
        f"- Restored V5_0 recipe and matrix tick wording and kept duplicate a/b recipe labels.\n"
        f"- Moved panel-c sweep titles into verified empty space; every sweep remains 17 mm high.\n"
        f"- Set the internal c sweep gaps to 2 mm and aligned c's bottom artist edge with b.\n"
        f"- Preserved panel-f matrix image dimensions and aspects; rendered bottom ink margin is "
        f"{panel_f_ink_margin_qa['rendered_bottom_ink_margin_mm']:.2f} mm.\n"
        f"- Matched all 270 V5_0 visible words and retained the scientific values.\n"
        f"- Preserved frozen source hashes and displayed values; A03/A04 remain open.\n",
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
            target = assembled / f"MixedResolution_unified_v5_2_{rid}{ext}"
            if target.exists():
                raise FileExistsError(target)
            shutil.copy2(stem.with_suffix(ext), target)
        target = assembled / f"FigureSourceManifest_unified_v5_2_{rid}.json"
        if target.exists():
            raise FileExistsError(target)
        shutil.copy2(release / "source_manifest_v5_2.json", target)
    print(f"[OK] {stem.with_suffix('.pdf')}")
    print(f"[OK] scientific continuity: {scientific['status']}; V5_0 continuity: {continuity_v5_0['status']}; height reduction from V5_0: {topology['height_reduction_percent']:.1f}%")
    print(f"[QA] geometry and style gates: {'PASS' if gates_passed else 'FAIL (trial only)'}; row gap fractions: {topology['vertical_gap_fraction_of_v5_0']}")


if __name__ == "__main__":
    main()

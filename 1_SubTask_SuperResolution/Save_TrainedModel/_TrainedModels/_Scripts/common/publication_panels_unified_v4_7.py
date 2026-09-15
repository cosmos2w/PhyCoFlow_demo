"""V4_7 visual-only label gutters, colourbar removal, and collision QA.

All scientific artists, arrays, limits, normalizations, masks, contours and
values are delegated unchanged to V4_6.  This layer acts only on final text
geometry and on the two explicitly authorized Panel-c colourbar axes.
"""
from __future__ import annotations

from math import hypot

import matplotlib

from . import publication_panels_unified_v4_6 as v46


PANEL_OUTPUT_NAMES = {
    key: value.replace("V4_6", "V4_7") for key, value in v46.PANEL_OUTPUT_NAMES.items()
}

TARGET_GIDS = {"qualitative-local-relative-l2", "qualitative-relative-l2"}
TARGET_TEXT = {"Sensor layout"}


def _mark(metadata: dict, label: str) -> dict:
    result = dict(metadata)
    result.update({
        "figure_revision": "V4_7",
        "source_visual_revision": "V4_6",
        "source_scientific_revision": "V3-7",
        "panel_id": f"fig3.{label}",
        "underlying_source_arrays_preserved": True,
        "data_mapping_preserved_from_v4_6": True,
        "model_training_or_inference": False,
    })
    return result


def _panel_c_colorbar_axes(parent, colorbar_parent=None):
    if colorbar_parent is not None:
        candidates = list(colorbar_parent.child_axes)
    else:
        candidates = list(parent.child_axes)
    return [axis for axis in candidates if getattr(axis, "_colorbar", None) is not None]


def draw_panel(label: str, parent, ctx, **kwargs):
    metadata = dict(v46.draw_panel(label, parent, ctx, **kwargs))
    if label == "c":
        colorbar_parent = kwargs.get("colorbar_parent")
        colorbar_axes = _panel_c_colorbar_axes(parent, colorbar_parent)
        if len(colorbar_axes) != 2:
            raise RuntimeError(f"V4_7 expected exactly two Panel-c colourbar axes, found {len(colorbar_axes)}")
        removed_titles = []
        for axis in colorbar_axes:
            removed_titles.extend(
                item.get_text() for item in axis.findobj(
                    match=lambda candidate: isinstance(candidate, matplotlib.text.Text)
                ) if item.get_visible() and item.get_text()
            )
            axis.remove()
        metadata.update({
            "shared_colorbar_count": 0,
            "colorbars_removed_v4_7": True,
            "colorbar_axes_removed_v4_7": 2,
            "removed_colorbar_text_v4_7": removed_titles,
            "normalization_values_retained_in_manifest": True,
        })
    return _mark(metadata, label)


def _is_target(item) -> bool:
    return str(item.get_gid() or "") in TARGET_GIDS or item.get_text() in TARGET_TEXT


def _move_below_final_axis(item, axis, *, offset_mm: float) -> None:
    fig = axis.figure
    height_mm = float(fig.get_size_inches()[1] * 25.4)
    box = axis.get_position()
    item.set_transform(fig.transFigure)
    item.set_position((box.x0 + box.width / 2.0, box.y0 - offset_mm / height_mm))
    item.set_horizontalalignment("center")
    item.set_verticalalignment("top")
    item.set_color("#252525")
    item.set_bbox(None)
    item.set_clip_on(False)
    item._global_font_role = "annotation"


def align_panel_cd_annotation_gutters(fig, shared_axes, *, offset_mm: float = 0.60,
                                      strict: bool = True) -> dict:
    """Place c/d metrics only after final equal-aspect geometry has settled."""
    moved = []
    for axis in shared_axes["c"][2]:
        matches = [item for item in axis.texts if _is_target(item)]
        if len(matches) != 1:
            raise RuntimeError(
                f"V4_7 expected one Panel-c bottom label on {axis.get_label()}, found {len(matches)}"
            )
        _move_below_final_axis(matches[0], axis, offset_mm=offset_mm)
        moved.append(("c", 2, axis, matches[0]))
    for row_index, row in enumerate(shared_axes["d"]):
        for axis in row[1:]:
            matches = [item for item in axis.texts if _is_target(item)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"V4_7 expected one Panel-d metric on {axis.get_label()}, found {len(matches)}"
                )
            _move_below_final_axis(matches[0], axis, offset_mm=offset_mm)
            moved.append(("d", row_index, axis, matches[0]))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    factor = 25.4 / fig.dpi
    c_bottom = [item.get_window_extent(renderer=renderer) for panel, row, _axis, item in moved
                if panel == "c" and row == 2]
    d_bottom = [item.get_window_extent(renderer=renderer) for panel, row, _axis, item in moved
                if panel == "d" and row == 2]
    top_edges = [(box.y1 * factor) for box in (*c_bottom, *d_bottom)]
    delta = float(max(top_edges) - min(top_edges))
    result = {
        "placement_phase": "after final aspect and axes positioning",
        "moved_annotation_count": len(moved),
        "panel_c_bottom_label_count": len(c_bottom),
        "panel_d_metric_count": len(moved) - len(c_bottom),
        "panel_c_and_panel_d_bottom_label_top_edges_mm": top_edges,
        "panel_c_panel_d_bottom_alignment_delta_mm": delta,
        "panel_c_panel_d_bottom_labels_aligned": bool(delta <= 0.02),
        "offset_below_owning_axis_mm": float(offset_mm),
    }
    if strict and not result["panel_c_panel_d_bottom_labels_aligned"]:
        raise ValueError(f"V4_7 c/d bottom annotation alignment failed: {result}")
    return result


def align_standalone_annotation_gutters(fig, label: str, *, offset_mm: float = 1.05,
                                        strict: bool = True) -> dict:
    """Apply the same post-aspect gutter rule to standalone Panel c or d."""
    candidates = []
    for axis in fig.findobj(match=lambda candidate: isinstance(candidate, matplotlib.axes.Axes)):
        for item in axis.texts:
            if _is_target(item):
                candidates.append((axis, item))
    expected = 5 if label == "c" else 6 if label == "d" else 0
    if len(candidates) != expected:
        raise RuntimeError(f"V4_7 standalone Panel {label} expected {expected} gutter labels, found {len(candidates)}")
    for axis, item in candidates:
        _move_below_final_axis(item, axis, offset_mm=offset_mm)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    factor = 25.4 / fig.dpi
    return {
        "standalone_gutter_label_count": len(candidates),
        "standalone_gutter_label_bboxes_mm": [
            _box_mm(item.get_window_extent(renderer=renderer), fig) for _axis, item in candidates
        ],
        "standalone_gutter_alignment_applied_after_final_aspect": True,
        "offset_below_owning_axis_mm": float(offset_mm),
    }


def _box_mm(box, fig):
    factor = 25.4 / fig.dpi
    return [float(value * factor) for value in (box.x0, box.y0, box.x1, box.y1)]


def _clearance_px(first, second) -> float:
    dx = max(second.x0 - first.x1, first.x0 - second.x1, 0.0)
    dy = max(second.y0 - first.y1, first.y0 - second.y1, 0.0)
    return float(hypot(dx, dy))


def measure_annotation_collision_gate(fig, *, minimum_clearance_mm: float = 1.0,
                                      strict: bool = True) -> dict:
    """Check external c/d labels against all evidence windows and peer text.

    The implementation is reusable for any text tagged by ``TARGET_GIDS`` or
    ``TARGET_TEXT``. It intentionally uses final axes-window boxes, not stale
    pre-aspect positions or nominal GridSpec rectangles.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    factor = 25.4 / fig.dpi
    all_text = [
        item for item in fig.findobj(match=lambda candidate: isinstance(candidate, matplotlib.text.Text))
        if item.get_visible() and item.get_text()
    ]
    targets = [item for item in all_text if _is_target(item)]
    evidence_axes = [
        axis for axis in fig.axes
        if axis.get_visible() and (axis.images or axis.collections or axis.lines or axis.patches)
        and str(axis.get_label() or "") != "panel-c-colorbar-strip"
    ]
    annotation_text = [
        item for item in all_text
        if getattr(item, "_global_font_role", None) == "annotation" or _is_target(item)
    ]
    target_records = []
    text_data_intersections = []
    minimum_observed = float("inf")
    for item in targets:
        box = item.get_window_extent(renderer=renderer)
        clearances = []
        for axis in evidence_axes:
            axis_box = axis.get_window_extent(renderer=renderer)
            clearance = _clearance_px(box, axis_box) * factor
            clearances.append((clearance, axis, axis_box))
            if box.overlaps(axis_box):
                text_data_intersections.append({
                    "text": item.get_text(), "gid": str(item.get_gid() or ""),
                    "text_bbox_mm": _box_mm(box, fig),
                    "evidence_axis": axis.get_label(),
                    "evidence_bbox_mm": _box_mm(axis_box, fig),
                })
        clearance, nearest_axis, nearest_box = min(clearances, key=lambda value: value[0])
        minimum_observed = min(minimum_observed, clearance)
        target_records.append({
            "text": item.get_text(), "gid": str(item.get_gid() or ""),
            "owner_axis": getattr(item.axes, "get_label", lambda: "")(),
            "bbox_mm": _box_mm(box, fig),
            "nearest_evidence_axis": nearest_axis.get_label(),
            "nearest_evidence_bbox_mm": _box_mm(nearest_box, fig),
            "nearest_evidence_clearance_mm": float(clearance),
            "passed": clearance >= minimum_clearance_mm and not box.overlaps(nearest_box),
        })
    text_text_intersections = []
    for index, first in enumerate(annotation_text):
        first_box = first.get_window_extent(renderer=renderer)
        for second in annotation_text[index + 1:]:
            second_box = second.get_window_extent(renderer=renderer)
            if first_box.overlaps(second_box):
                text_text_intersections.append({
                    "first": first.get_text(), "first_gid": str(first.get_gid() or ""),
                    "first_bbox_mm": _box_mm(first_box, fig),
                    "second": second.get_text(), "second_gid": str(second.get_gid() or ""),
                    "second_bbox_mm": _box_mm(second_box, fig),
                })
    passed = (
        len(targets) == 11
        and not text_data_intersections
        and not text_text_intersections
        and minimum_observed >= minimum_clearance_mm
        and all(item["passed"] for item in target_records)
    )
    result = {
        "schema_version": "final_aspect_annotation_collision_v1",
        "run_after_final_aspect_and_axes_positioning": True,
        "target_scope": "all Panel-c bottom labels and Panel-d residual metrics",
        "target_annotation_count": len(targets),
        "evidence_axis_count": len(evidence_axes),
        "annotation_text_count_pairwise_checked": len(annotation_text),
        "required_minimum_clearance_mm_at_180mm": float(minimum_clearance_mm),
        "minimum_observed_clearance_mm_at_180mm": float(minimum_observed),
        "minimum_observed_clearance_mm_at_162mm": float(minimum_observed * 0.9),
        "target_records": target_records,
        "text_data_intersection_count": len(text_data_intersections),
        "text_data_intersections": text_data_intersections,
        "annotation_annotation_intersection_count": len(text_text_intersections),
        "annotation_annotation_intersections": text_text_intersections,
        "intentional_in_data_annotation_registry": [],
        "passed": passed,
    }
    if strict and not passed:
        raise ValueError(f"V4_7 final-aspect annotation collision gate failed: {result}")
    return result


apply_v4_7_typography = v46.apply_v4_6_typography
panel_label = v46.panel_label
center_panel_d_headers = v46.center_panel_d_headers
align_panel_b_ylabels = v46.align_panel_b_ylabels
measure_panel_b_legend_clearance = v46.measure_panel_b_legend_clearance
measure_panel_d_header_alignment = v46.measure_panel_d_header_alignment
measure_adjacent_major_row_unions = v46.measure_adjacent_major_row_unions
measure_major_content_gaps = v46.measure_major_content_gaps
record_panel_e_v4_3_tick_settings = v46.record_panel_e_v4_3_tick_settings

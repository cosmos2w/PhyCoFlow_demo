"""Text- and geometry-only micro-refinement for mixed-resolution Figure V4_4.

V4_4 inherits V4_3's exact scientific arrays, derived absolute-error mapping,
normalizations, limits, artists, inventory, and panel-e tick-label settings.
This layer changes only three approved text strings and final-render panel-d/
major-layout geometry.
"""
from __future__ import annotations

import matplotlib

from . import publication_panels_unified_v3_3 as v33
from . import publication_panels_unified_v4_3 as v43


PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_TrainingResolutionDesign_V4_4",
    "b": "Panel_b_IntegratedPerformance_V4_4",
    "c": "Panel_c_SpatialProof_V4_4",
    "d": "Panel_d_MultiscaleSupport_V4_4",
    "e": "Panel_e_CompleteScaleMatrices_V4_4",
}


def _mark(metadata: dict, label: str) -> dict:
    result = dict(metadata)
    result.update({
        "figure_revision": "V4_4",
        "source_visual_revision": "V4_3",
        "source_scientific_revision": "V3-7",
        "panel_id": f"fig3.{label}",
        "underlying_source_arrays_preserved": True,
        "data_mapping_preserved_from_v4_3": True,
        "model_training_or_inference": False,
    })
    return result


def _all_text(fig, value: str):
    return [
        item for item in fig.findobj(match=lambda candidate: isinstance(candidate, matplotlib.text.Text))
        if item.get_visible() and item.get_text() == value
    ]


def draw_panel_a(parent, ctx, **kwargs):
    metadata = v43.draw_panel_a(parent, ctx, **kwargs)
    replacements = {
        "contains H-resolution training fields": "Contains H-resolution training fields",
        "zero-H training": "Zero-H training",
    }
    records = []
    for before, after in replacements.items():
        matches = _all_text(parent.figure, before)
        if len(matches) != 1:
            raise RuntimeError(f"V4_4 expected one panel-a label {before!r}, found {len(matches)}")
        item = matches[0]
        item.set_text(after)
        item.set_bbox(None)
        records.append({"before": before, "after": after, "bbox": None})
    metadata.update({
        "capitalization_replacements": records,
        "capitalization_replacement_count": len(records),
        "training_group_labels_unboxed": True,
    })
    return _mark(metadata, "a")


def draw_panel_b(parent, ctx, **kwargs):
    metadata = v43.draw_panel_b(parent, ctx, **kwargs)
    labels = [
        item for item in parent.figure.findobj(match=lambda candidate: isinstance(candidate, matplotlib.text.Text))
        if item.get_gid() == "panel-b-zero-h-label"
    ]
    if len(labels) != 1 or labels[0].get_text() != "no H-resolution training fields":
        raise RuntimeError("V4_4 could not resolve the unique panel-b zero-H label")
    item = labels[0]
    item.set_text("Zero-H training")
    item.set_color(v33.NEUTRAL_MID)
    item.set_fontweight("normal")
    item.set_fontstyle("normal")
    item.set_alpha(1.0)
    item.set_bbox(None)
    metadata.update({
        "zero_h_label_before": "no H-resolution training fields",
        "zero_h_label_after": item.get_text(),
        "zero_h_label_unboxed": item.get_bbox_patch() is None,
        "zero_h_label_color": matplotlib.colors.to_hex(item.get_color()),
        "zero_h_label_matches_panel_a_unboxed_style": (
            item.get_bbox_patch() is None
            and matplotlib.colors.to_hex(item.get_color()) == matplotlib.colors.to_hex(v33.NEUTRAL_MID)
        ),
    })
    return _mark(metadata, "b")


def draw_panel_c(parent, ctx, **kwargs):
    return _mark(v43.draw_panel_c(parent, ctx, **kwargs), "c")


def draw_panel_d(parent, ctx, **kwargs):
    return _mark(v43.draw_panel_d(parent, ctx, **kwargs), "d")


def _matrix_axes(parent):
    axes = [
        axis for axis in parent.figure.findobj(match=lambda item: isinstance(item, matplotlib.axes.Axes))
        if str(axis.get_gid() or "") == "panel-e-metric-matrix"
    ]
    return sorted(axes, key=lambda axis: axis.get_position().x0)


def record_panel_e_v4_3_tick_settings(parent, *, strict: bool = True) -> dict:
    """Verify panel e retains V4_3's tick-label settings without modification."""
    axes = _matrix_axes(parent)
    if len(axes) != 6:
        raise RuntimeError(f"V4_4 expected six panel-e matrix axes, found {len(axes)}")
    records = []
    for axis in axes:
        labels = [item for item in axis.get_xticklabels() if item.get_visible() and item.get_text()]
        if [item.get_text() for item in labels] != ["Large", "Interm.", "Fine"]:
            raise RuntimeError("V4_4 panel-e scale-label order changed")
        for item in labels:
            records.append({
                "text": item.get_text(),
                "horizontalalignment": item.get_horizontalalignment(),
                "rotation_deg": float(item.get_rotation()),
                "rotation_mode": item.get_rotation_mode(),
            })
    settings_match = all(
        item["horizontalalignment"] == "right"
        and item["rotation_deg"] == 45.0
        and item["rotation_mode"] == "anchor"
        for item in records
    )
    result = {
        "scale_tick_label_settings_source": "V4_3",
        "scale_tick_label_settings_records": records,
        "scale_tick_label_settings_match_v4_3": settings_match,
        "v4_4_tick_centering_override_applied": False,
    }
    if strict and not settings_match:
        raise ValueError(f"V4_4 panel-e tick settings diverged from V4_3: {result}")
    return result


def draw_panel_e(parent, ctx, **kwargs):
    metadata = v43.draw_panel_e(parent, ctx, **kwargs)
    metadata.update(record_panel_e_v4_3_tick_settings(parent, strict=True))
    return _mark(metadata, "e")


def _axis_tree(root):
    result, seen = [], set()

    def visit(axis):
        if id(axis) in seen:
            return
        seen.add(id(axis))
        result.append(axis)
        for child in getattr(axis, "child_axes", []):
            visit(child)

    for axis in root:
        visit(axis)
    return result


def _vertical_content_extent(fig, axes, *, include_axis_windows: bool = True):
    """Return the final visible content y bounds for a semantic panel block."""
    axes = _axis_tree(axes)
    axis_ids = {id(axis) for axis in axes}
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    if include_axis_windows:
        boxes.extend(
            axis.get_window_extent(renderer=renderer) for axis in axes
            if axis.axison and (
                axis.images or axis.collections or axis.lines or axis.patches
                or any(item.get_visible() and item.get_text() for item in axis.get_xticklabels())
                or any(item.get_visible() and item.get_text() for item in axis.get_yticklabels())
            )
        )
    for item in fig.findobj(match=lambda candidate: isinstance(candidate, matplotlib.text.Text)):
        if not item.get_visible() or not item.get_text() or id(getattr(item, "axes", None)) not in axis_ids:
            continue
        box = item.get_window_extent(renderer=renderer)
        if box.width > 0 and box.height > 0 and box.y1 > fig.bbox.y0 and box.y0 < fig.bbox.y1:
            boxes.append(box)
    if not boxes:
        raise RuntimeError("V4_4 could not resolve a semantic panel content extent")
    return min(box.y0 for box in boxes), max(box.y1 for box in boxes)


def measure_major_content_gaps(fig, axes, shared_axes, cbar_parent, *, strict: bool = True) -> dict:
    """Measure visible-content gaps, including c's colorbars and e's titles."""
    a_extent = _vertical_content_extent(fig, [axes["a"]])
    b_extent = _vertical_content_extent(fig, [axes["b"]])
    cd_axes = [axes["c"], axes["d"], cbar_parent]
    cd_axes.extend(axis for panel in ("c", "d") for row in shared_axes[panel] for axis in row)
    cd_extent = _vertical_content_extent(fig, cd_axes)
    e_extent = _vertical_content_extent(fig, [axes["e"]])
    mm_per_px = 25.4 / fig.dpi
    gaps = {
        "a_b": float((a_extent[0] - b_extent[1]) * mm_per_px),
        "b_cd": float((b_extent[0] - cd_extent[1]) * mm_per_px),
        "cd_e": float((cd_extent[0] - e_extent[1]) * mm_per_px),
    }
    minimum_design = 3.0 / 0.9
    passed = all(value >= minimum_design for value in gaps.values())
    result = {
        "visible_content_gaps_mm": gaps,
        "minimum_design_gap_for_162mm_mm": minimum_design,
        "minimum_visible_content_gap_mm": min(gaps.values()),
        "minimum_visible_content_gap_at_162mm_mm": min(gaps.values()) * 0.9,
        "visible_content_gap_qa_passed": passed,
        "cd_e_gap_distance_from_ab_mm": abs(gaps["cd_e"] - gaps["a_b"]),
        "cd_e_gap_distance_from_bcd_mm": abs(gaps["cd_e"] - gaps["b_cd"]),
    }
    if strict and not passed:
        raise ValueError(f"V4_4 visible major-panel clearance failed: {result}")
    return result


PANEL_DRAWERS = {
    "a": draw_panel_a,
    "b": draw_panel_b,
    "c": draw_panel_c,
    "d": draw_panel_d,
    "e": draw_panel_e,
}


def draw_panel(label: str, parent, ctx, **kwargs):
    return PANEL_DRAWERS[label](parent, ctx, **kwargs)


panel_label = v43.panel_label
center_panel_d_headers = v43.center_panel_d_headers
measure_panel_b_legend_clearance = v43.measure_panel_b_legend_clearance
measure_panel_d_header_alignment = v43.measure_panel_d_header_alignment

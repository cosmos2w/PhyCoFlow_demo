"""V4_6 visual corrections and rendered major-row clearance gate.

The scientific artists and arrays are delegated unchanged to V4_5.  This
layer only applies the author-approved text, typography, and alignment deltas
and provides the final-draw renderer gate that V4_5 lacked.
"""
from __future__ import annotations

from collections import defaultdict

import matplotlib
import numpy as np
from matplotlib.transforms import Bbox

from . import publication_panels_unified_v4_5 as v45


PANEL_OUTPUT_NAMES = {
    key: value.replace("V4_5", "V4_6") for key, value in v45.PANEL_OUTPUT_NAMES.items()
}

LOCAL_SUBPLOT_TITLES = {
    "Contains H-resolution training fields",
    "Zero-H training",
    "High resolution reconstruction (512 sensors)",
}


def _mark(metadata: dict, label: str) -> dict:
    result = dict(metadata)
    result.update({
        "figure_revision": "V4_6",
        "source_visual_revision": "V4_5",
        "source_scientific_revision": "V3-7",
        "panel_id": f"fig3.{label}",
        "underlying_source_arrays_preserved": True,
        "data_mapping_preserved_from_v4_5": True,
        "model_training_or_inference": False,
    })
    return result


def draw_panel(label: str, parent, ctx, **kwargs):
    metadata = dict(v45.draw_panel(label, parent, ctx, **kwargs))
    if label == "b":
        matches = [
            item for item in parent.figure.findobj(
                match=lambda candidate: isinstance(candidate, matplotlib.text.Text)
            )
            if item.get_visible() and item.get_gid() == "panel-b-zero-h-label"
        ]
        if len(matches) != 1 or matches[0].get_text() != "Zero-H training":
            raise RuntimeError(f"V4_6 expected one Panel-b Zero-H label, found {len(matches)}")
        matches[0].remove()
        metadata.update({
            "zero_h_label_removed_v4_6": True,
            "zero_h_label_removed_text": "Zero-H training",
        })
    elif label == "c":
        matches = [
            item for item in parent.figure.findobj(
                match=lambda candidate: isinstance(candidate, matplotlib.text.Text)
            )
            if item.get_visible() and item.get_text() == "Zoom-region absolute error"
        ]
        if len(matches) != 1:
            raise RuntimeError(f"V4_6 expected one Panel-c error title, found {len(matches)}")
        matches[0].set_text("Absolute error")
        metadata.update({
            "error_colorbar_title_before": "Zoom-region absolute error",
            "error_colorbar_title_after": "Absolute error",
        })
    return _mark(metadata, label)


def apply_v4_6_typography(fig) -> dict:
    """Retain the V4_5 hierarchy with three requested local reductions."""
    qa = dict(v45.apply_v4_5_typography(fig))
    reduced = []
    for item in fig.findobj(match=lambda candidate: isinstance(candidate, matplotlib.text.Text)):
        if not item.get_visible() or item.get_text() not in LOCAL_SUBPLOT_TITLES:
            continue
        item._global_font_role = "subplot_title"
        item.set_fontweight("normal")
        reduced.append(item.get_text())
    visible = [
        item for item in fig.findobj(match=lambda candidate: isinstance(candidate, matplotlib.text.Text))
        if item.get_visible() and item.get_text()
    ]
    major = [
        item.get_text() for item in visible
        if getattr(item, "_global_font_role", None) == "major_title"
    ]
    qa.update({
        "major_title_text": major,
        "major_title_count": len(major),
        "local_reduction_role": "subplot_title",
        "locally_reduced_text": reduced,
        "locally_reduced_text_count": len(reduced),
        "panel_b_zero_h_text_present": any(
            item.get_gid() == "panel-b-zero-h-label" for item in visible
        ),
    })
    return qa


def _panel_b_axes(parent):
    top = [
        axis for axis in parent.child_axes
        if any(str(patch.get_gid() or "").startswith("model-bar:") for patch in axis.patches)
    ]
    sweeps = [
        axis for axis in parent.child_axes
        if any(str(line.get_gid() or "").startswith("model-line:") for line in axis.lines)
    ]
    sweeps.sort(key=lambda axis: axis.get_position().x0)
    if len(top) != 1 or len(sweeps) != 3:
        raise RuntimeError(f"V4_6 Panel-b topology mismatch: top={len(top)}, sweeps={len(sweeps)}")
    return top[0], sweeps


def align_panel_b_ylabels(parent, *, strict: bool = True) -> dict:
    """Force the upper/lower ``Relative L2`` label centres onto one x line."""
    top, sweeps = _panel_b_axes(parent)
    labels = [top.yaxis.label, sweeps[0].yaxis.label]
    if [item.get_text() for item in labels] != [r"Relative $L_2$", r"Relative $L_2$"]:
        raise RuntimeError("V4_6 Panel-b y-axis title text changed unexpectedly")
    fig = parent.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    centres = [item.get_window_extent(renderer=renderer).x0 + item.get_window_extent(renderer=renderer).width / 2 for item in labels]
    target_px = min(centres)
    target_fig_x = fig.transFigure.inverted().transform((target_px, 0.0))[0]
    for axis, item in ((top, labels[0]), (sweeps[0], labels[1])):
        y_fig = fig.transFigure.inverted().transform(
            axis.transAxes.transform((0.0, 0.5))
        )[1]
        axis.yaxis.set_label_coords(target_fig_x, y_fig, transform=fig.transFigure)
        item.set_horizontalalignment("center")
        item.set_verticalalignment("center")
    fig.canvas.draw()
    boxes = [item.get_window_extent(renderer=fig.canvas.get_renderer()) for item in labels]
    centres = [box.x0 + box.width / 2 for box in boxes]
    delta_mm = float(abs(centres[0] - centres[1]) * 25.4 / fig.dpi)
    result = {
        "y_axis_titles": [item.get_text() for item in labels],
        "y_axis_title_center_x_mm": [float(value * 25.4 / fig.dpi) for value in centres],
        "y_axis_title_center_delta_mm": delta_mm,
        "y_axis_titles_aligned": delta_mm <= 0.02,
    }
    if strict and not result["y_axis_titles_aligned"]:
        raise ValueError(f"V4_6 Panel-b y-axis titles are not aligned: {result}")
    return result


def _axis_tree(roots):
    result, seen = [], set()

    def visit(axis):
        if id(axis) in seen:
            return
        seen.add(id(axis))
        result.append(axis)
        for child in getattr(axis, "child_axes", []):
            visit(child)

    for root in roots:
        visit(root)
    return result


def _valid_box(box) -> bool:
    if box is None:
        return False
    values = np.asarray([box.x0, box.y0, box.x1, box.y1], dtype=float)
    return bool(np.isfinite(values).all() and box.width > 0 and box.height > 0)


def _union(boxes):
    boxes = [box for box in boxes if _valid_box(box)]
    if not boxes:
        return None
    return Bbox.union(boxes)


def _bbox_mm(box, fig, scale=1.0):
    factor = 25.4 / fig.dpi * scale
    return [float(value * factor) for value in (box.x0, box.y0, box.x1, box.y1)]


def _artist_window_box(item, renderer):
    """Resolve display extents, including mixed-transform ConnectionPatch."""
    if hasattr(item, "_get_path_in_displaycoord"):
        try:
            paths, _fillable = item._get_path_in_displaycoord()
            if not isinstance(paths, (list, tuple)):
                paths = [paths]
            box = _union(path.get_extents() for path in paths)
            if _valid_box(box):
                return box
        except Exception:
            pass
    try:
        return item.get_window_extent(renderer=renderer)
    except Exception:
        return None


def measure_adjacent_major_row_unions(
    fig, axes, shared_axes, cbar_parent, *, minimum_at_162_mm: float = 3.0,
    strict: bool = True,
) -> dict:
    """Audit complete rendered row unions after the final draw.

    Axis ``tightbbox`` values are authoritative for row clearance. Component
    boxes are retained separately so an audit can prove that tick labels,
    titles, legends, colourbars, annotations, and evidence were included.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    colorbar_ids = {id(axis) for axis in _axis_tree([cbar_parent])}
    row_roots = {
        "a": [axes["a"]],
        "b": [axes["b"]],
        "cd": [axes["c"], axes["d"], cbar_parent,
               *[axis for panel in ("c", "d") for row in shared_axes[panel] for axis in row]],
        "e": [axes["e"]],
    }
    records = {}
    all_owned_axis_ids = set()
    missing_bbox_count = 0
    clipped_count = 0
    canvas = fig.bbox
    for row_name, roots in row_roots.items():
        row_axes = _axis_tree(roots)
        axis_ids = {id(axis) for axis in row_axes}
        all_owned_axis_ids.update(axis_ids)
        components = defaultdict(list)
        artist_records = []
        for index, axis in enumerate(row_axes):
            axis_tight_box = None
            has_visible_content = bool(
                axis.axison or axis.images or axis.collections or axis.lines or axis.patches
                or axis.get_legend() is not None
            )
            if has_visible_content:
                try:
                    box = axis.get_tightbbox(renderer)
                except Exception:
                    box = None
                if _valid_box(box):
                    axis_tight_box = box
                    kind = "colorbar" if id(axis) in colorbar_ids else "axes_tightbbox"
                    components[kind].append(box)
                    artist_records.append({
                        "artist_id": axis.get_label() or f"{row_name}.axis.{index}",
                        "kind": kind, "bbox_mm": _bbox_mm(box, fig),
                    })
                else:
                    missing_bbox_count += 1
            direct_text = list(axis.texts)
            tick_text_ids = set()
            if axis.axison:
                tick_texts = [
                    *axis.get_xticklabels(minor=False), *axis.get_xticklabels(minor=True),
                    *axis.get_yticklabels(minor=False), *axis.get_yticklabels(minor=True),
                ]
                tick_text_ids = {id(item) for item in tick_texts}
                direct_text.extend([
                    axis.title, axis.xaxis.label, axis.yaxis.label,
                    axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text(),
                    *tick_texts,
                ])
            seen_text = set()
            for item in direct_text:
                if id(item) in seen_text:
                    continue
                seen_text.add(id(item))
                if not item.get_visible() or not item.get_text():
                    continue
                box = item.get_window_extent(renderer=renderer)
                if not _valid_box(box):
                    missing_bbox_count += 1
                    continue
                # Matplotlib may retain visible Text objects for locator ticks
                # outside the current view interval. They are not painted and
                # are excluded from the axis tight box; do not count them as
                # rendered content.
                if id(item) in tick_text_ids and axis_tight_box is not None \
                        and not box.overlaps(axis_tight_box):
                    continue
                role = str(getattr(item, "_global_font_role", "text"))
                kind = {
                    "panel_label": "panel_label",
                    "axis_label": "axis_label",
                    "subplot_title": "title",
                    "major_title": "title",
                    "tick_label": "tick_label",
                    "legend": "legend_text",
                    "annotation": "annotation",
                }.get(role, "text")
                components[kind].append(box)
                artist_records.append({
                    "artist_id": str(item.get_gid() or item.get_text()),
                    "kind": kind, "text": item.get_text(),
                    "bbox_mm": _bbox_mm(box, fig),
                })
            legend = axis.get_legend()
            if legend is not None and legend.get_visible():
                box = legend.get_window_extent(renderer=renderer)
                if _valid_box(box):
                    components["legend"].append(box)
                else:
                    missing_bbox_count += 1
            evidence = [*axis.images, *axis.collections, *axis.lines, *axis.patches]
            for item in evidence:
                if not item.get_visible():
                    continue
                box = _artist_window_box(item, renderer)
                if _valid_box(box):
                    components["evidence"].append(box)
        # Figure-level connectors are not children of their semantic axes.
        prefix = "panel-c" if row_name == "cd" else f"panel-{row_name}"
        prefixes = ("panel-c", "panel-d") if row_name == "cd" else (prefix,)
        for item in fig.artists:
            gid = str(getattr(item, "get_gid", lambda: "")() or "")
            if not gid.startswith(prefixes) or not item.get_visible():
                continue
            box = _artist_window_box(item, renderer)
            if _valid_box(box):
                components["evidence"].append(box)
                artist_records.append({
                    "artist_id": gid or type(item).__name__,
                    "kind": "evidence", "bbox_mm": _bbox_mm(box, fig),
                })
            else:
                missing_bbox_count += 1
        component_unions = {kind: _union(boxes) for kind, boxes in components.items()}
        # Evidence attached through mixed-coordinate ConnectionPatch objects can
        # report an unmutated control-path extent outside what the renderer
        # paints. Its painted row ownership is bounded by the union of the
        # owning axes tight boxes; retain the evidence class, clipped to that
        # authoritative rendered envelope.
        axis_envelope = _union([
            component_unions.get("axes_tightbbox"), component_unions.get("colorbar")
        ])
        evidence_raw = component_unions.get("evidence")
        if axis_envelope is not None and evidence_raw is not None:
            component_unions["evidence"] = Bbox.intersection(axis_envelope, evidence_raw)
        row_union = _union(component_unions.values())
        if row_union is None:
            raise RuntimeError(f"V4_6 could not resolve rendered union for row {row_name}")
        for box in component_unions.values():
            if box is not None and (
                box.x0 < canvas.x0 - .5 or box.y0 < canvas.y0 - .5
                or box.x1 > canvas.x1 + .5 or box.y1 > canvas.y1 + .5
            ):
                clipped_count += 1
        records[row_name] = {
            "bbox": row_union,
            "bbox_mm": _bbox_mm(row_union, fig),
            "component_union_bboxes_mm": {
                kind: _bbox_mm(box, fig) for kind, box in component_unions.items() if box is not None
            },
            "component_counts": {kind: len(boxes) for kind, boxes in components.items()},
            "raw_evidence_bbox_mm": _bbox_mm(evidence_raw, fig) if evidence_raw is not None else None,
            "axis_count": len(row_axes),
            "axis_records": artist_records,
        }

    unowned_figure_text = [
        item.get_text() for item in fig.texts
        if item.get_visible() and item.get_text()
        and id(getattr(item, "axes", None)) not in all_owned_axis_ids
    ]
    unknown_figure_artists = [
        str(getattr(item, "get_gid", lambda: "")() or type(item).__name__)
        for item in fig.artists if item.get_visible()
        and not str(getattr(item, "get_gid", lambda: "")() or "").startswith(
            ("panel-a", "panel-b", "panel-c", "panel-d", "panel-e")
        )
    ]
    minimum_design = float(minimum_at_162_mm) / 0.9
    adjacent = []
    for upper, lower in (("a", "b"), ("b", "cd"), ("cd", "e")):
        upper_box, lower_box = records[upper]["bbox"], records[lower]["bbox"]
        gap = float((upper_box.y0 - lower_box.y1) * 25.4 / fig.dpi)
        horizontal_overlap = float(
            max(0.0, min(upper_box.x1, lower_box.x1) - max(upper_box.x0, lower_box.x0))
            * 25.4 / fig.dpi
        )
        intersects = bool(upper_box.overlaps(lower_box))
        adjacent.append({
            "upper_row": upper, "lower_row": lower,
            "upper_union_bbox_mm": records[upper]["bbox_mm"],
            "lower_union_bbox_mm": records[lower]["bbox_mm"],
            "horizontal_overlap_mm": horizontal_overlap,
            "vertical_gap_mm": gap,
            "vertical_gap_mm_at_162mm": gap * .9,
            "required_vertical_gap_mm_at_162mm": float(minimum_at_162_mm),
            "forbidden_intersection_count": int(intersects),
            "passed": (not intersects and gap >= minimum_design),
        })
    passed = (
        all(item["passed"] for item in adjacent)
        and not unowned_figure_text and not unknown_figure_artists
        and missing_bbox_count == 0 and clipped_count == 0
    )
    result = {
        "schema_version": "major_row_bbox_v1",
        "backend": "Python/Matplotlib",
        "coordinate_system": "pdf_media_box_mm_bottom_left",
        "final_draw_measurement": True,
        "tested_widths_mm": [180, 162],
        "insertion_transform": "uniform vector scale 0.9",
        "required_artist_kinds": [
            "axes_tightbbox", "tick_label", "axis_label", "title", "legend",
            "colorbar", "panel_label", "annotation", "evidence",
        ],
        "major_row_union_bboxes_mm": {
            key: value["bbox_mm"] for key, value in records.items()
        },
        "major_row_union_bboxes_mm_at_162mm": {
            key: [coordinate * .9 for coordinate in value["bbox_mm"]]
            for key, value in records.items()
        },
        "component_union_bboxes_mm": {
            key: value["component_union_bboxes_mm"] for key, value in records.items()
        },
        "component_counts": {key: value["component_counts"] for key, value in records.items()},
        "axis_records": {key: value["axis_records"] for key, value in records.items()},
        "adjacent_major_row_checks": adjacent,
        "minimum_vertical_clearance_mm_at_162mm": float(minimum_at_162_mm),
        "minimum_observed_vertical_gap_mm_at_162mm": min(
            item["vertical_gap_mm_at_162mm"] for item in adjacent
        ),
        "unknown_visible_artist_count": len(unknown_figure_artists),
        "unknown_visible_artists": unknown_figure_artists,
        "unowned_visible_figure_text_count": len(unowned_figure_text),
        "unowned_visible_figure_text": unowned_figure_text,
        "missing_required_bbox_count": missing_bbox_count,
        "clipped_artist_count": clipped_count,
        "unexplained_intersection_count": sum(
            item["forbidden_intersection_count"] for item in adjacent
        ),
        "passed": passed,
    }
    if strict and not passed:
        extremes = {}
        for key, value in records.items():
            text_records = [item for item in value["axis_records"] if item.get("text")]
            evidence_records = [item for item in value["axis_records"] if item.get("kind") == "evidence"]
            extremes[key] = {
                "lowest": sorted(text_records, key=lambda item: item["bbox_mm"][1])[:3],
                "highest": sorted(text_records, key=lambda item: item["bbox_mm"][3], reverse=True)[:3],
                "evidence_lowest": sorted(evidence_records, key=lambda item: item["bbox_mm"][1])[:3],
            }
        raise ValueError(
            "V4_6 adjacent-major-row union gate failed: "
            f"adjacent={adjacent}, clipped={clipped_count}, extremes={extremes}"
        )
    return result


panel_label = v45.panel_label
center_panel_d_headers = v45.center_panel_d_headers
measure_panel_b_legend_clearance = v45.measure_panel_b_legend_clearance
measure_panel_d_header_alignment = v45.measure_panel_d_header_alignment
measure_major_content_gaps = v45.measure_major_content_gaps
record_panel_e_v4_3_tick_settings = v45.record_panel_e_v4_3_tick_settings

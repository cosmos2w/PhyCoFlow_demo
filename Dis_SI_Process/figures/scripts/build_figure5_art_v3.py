#!/usr/bin/env python
"""Build Figure 5 art V3 with cropped margins and dense row alignment.

The accepted renderer remains the sole source of numerical arrays and
statistics. This wrapper changes only artist appearance, visibility, text,
and physical layout. Inference-memory uncertainty artists remain in the
figure object but are hidden under the latest author-approved visual override.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import fitz
import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
REPO = ROOT.parent
V2_SCRIPT = SCRIPT_DIR / "build_figure5_art_v2.py"
DEFAULT_CONFIG = ROOT / "configs" / "figure5_art_v3.yaml"
DEFAULT_OUT = ROOT / "figures/generated/art_style_review/Figure_Evaluations_art_v3_20260915_0900"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = _load_module("figure5_art_v2_helpers", V2_SCRIPT)
V1 = V2.V1
BASE = V2.BASE


def _nearest_method(y: float, ticks: np.ndarray, labels: list[str]) -> tuple[str, int] | None:
    if not np.isfinite(y) or not len(ticks):
        return None
    index = int(np.argmin(np.abs(ticks - y)))
    if index >= len(labels) or abs(float(ticks[index]) - y) > 0.55:
        return None
    return V2._method_name(labels[index]), index


def _scientific_data_hash(fig: Figure) -> str:
    payload = []
    for ax in fig.axes:
        payload.append(
            {
                "lines": [[np.asarray(line.get_xdata()).tolist(), np.asarray(line.get_ydata()).tolist()] for line in ax.lines],
                "collections": [
                    {"offsets": np.asarray(collection.get_offsets()).tolist(), "paths": [path.vertices.tolist() for path in collection.get_paths()]}
                    for collection in ax.collections
                ],
                "patches": [patch.get_path().vertices.tolist() for patch in ax.patches],
                "xscale": ax.get_xscale(),
                "yscale": ax.get_yscale(),
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _recolor_top_panel(ax, config: dict, labels_override: list[str] | None = None, enlarge_clouds: bool = False) -> tuple[dict, int]:
    colors = config["style"]["method_colors"]
    markers = config["style"]["method_markers"]
    size = float(config["style"]["comparison_marker_size_pt"])
    edge = float(config["style"]["comparison_marker_edge_width_pt"])
    cloud_size = float(config["style"]["cloud_size_pt2"])
    cloud_alpha = float(config["style"]["cloud_alpha"])
    labels = labels_override or [V2._method_name(tick.get_text()) for tick in ax.get_yticklabels()]
    ticks = np.asarray(ax.get_yticks(), dtype=float)
    marker_records: dict[str, dict] = {}
    cloud_count = 0

    for collection in ax.collections:
        if isinstance(collection, PathCollection):
            offsets = np.asarray(collection.get_offsets())
            if not len(offsets):
                continue
            match = _nearest_method(float(np.mean(offsets[:, 1])), ticks, labels)
            if match is None:
                continue
            method, _ = match
            if method not in colors:
                continue
            collection.set_facecolor(colors[method])
            collection.set_edgecolor("none")
            if len(offsets) >= 50 and enlarge_clouds:
                collection.set_sizes([cloud_size])
                collection.set_alpha(cloud_alpha)
                cloud_count += 1
        elif isinstance(collection, LineCollection):
            segments = collection.get_segments()
            if not segments:
                continue
            match = _nearest_method(float(np.mean(np.concatenate(segments)[:, 1])), ticks, labels)
            if match is not None and match[0] in colors:
                collection.set_color(colors[match[0]])

    for patch in ax.patches:
        if not isinstance(patch, PathPatch):
            continue
        vertices = patch.get_path().transformed(patch.get_transform()).vertices
        data_vertices = ax.transData.inverted().transform(vertices)
        match = _nearest_method(float(np.mean(data_vertices[:, 1])), ticks, labels)
        if match is None or match[0] not in colors:
            continue
        method = match[0]
        alpha = patch.get_alpha()
        patch.set_facecolor(colors[method])
        patch.set_edgecolor(colors[method])
        if alpha is not None:
            patch.set_alpha(alpha)

    for line in ax.lines:
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if not ydata.size or not np.isfinite(ydata).all():
            continue
        match = _nearest_method(float(np.mean(ydata)), ticks, labels)
        if match is None or match[0] not in colors:
            continue
        method = match[0]
        line.set_color(colors[method])
        marker = line.get_marker()
        if marker not in (None, "None", "", " ") and len(ydata) == 1:
            line.set_marker(markers[method])
            line.set_markersize(size)
            line.set_markerfacecolor("none")
            line.set_markeredgecolor(colors[method])
            line.set_markeredgewidth(edge)
            marker_records[method] = {
                "color": colors[method],
                "marker": markers[method],
                "markersize_pt": size,
                "markeredgewidth_pt": edge,
                "markerfacecolor": "none",
            }
    return marker_records, cloud_count


def _harmonize_panel_c_markers(ax, config: dict) -> list[dict]:
    size = float(config["style"]["comparison_marker_size_pt"])
    edge = float(config["style"]["comparison_marker_edge_width_pt"])
    records = []
    for line in ax.lines:
        marker = line.get_marker()
        if marker in (None, "None", "", " ") or len(np.asarray(line.get_xdata())) <= 1:
            continue
        line.set_markersize(size)
        line.set_markerfacecolor("none")
        line.set_markeredgecolor(line.get_color())
        line.set_markeredgewidth(edge)
        records.append({"label": line.get_label(), "marker": marker, "markersize_pt": size, "markeredgewidth_pt": edge, "markerfacecolor": "none"})
    return records


def _add_violin_means(ax, config: dict, semantic_panel: str) -> list[dict]:
    bodies = [collection for collection in ax.collections if isinstance(collection, PolyCollection) and collection.get_paths()]
    mean_lines = []
    for line in ax.lines:
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if xdata.size == 2 and ydata.size == 2 and np.isfinite(ydata).all() and np.allclose(ydata, ydata[0]):
            mean_lines.append((float(np.mean(xdata)), float(ydata[0])))
    if len(bodies) != 6 or len(mean_lines) != 6:
        raise ValueError(f"Expected six violin bodies and mean lines in {semantic_panel}; found {len(bodies)} and {len(mean_lines)}")
    body_records = []
    body_geometry = []
    for body in bodies:
        vertices = np.concatenate([path.vertices for path in body.get_paths()])
        body_geometry.append((float((vertices[:, 0].min() + vertices[:, 0].max()) / 2.0), float(vertices[:, 1].max())))
    for center, tip in sorted(body_geometry):
        mean = min(mean_lines, key=lambda record: abs(record[0] - center))[1]
        label = format(mean, ".2g")
        text = ax.annotate(
            label,
            xy=(center, tip),
            xytext=(0, float(config["style"]["mean_annotation_offset_pt"])),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=float(config["typography_pt"]["in_plot_annotation"]),
            fontweight="normal",
            color="#333333",
            clip_on=False,
            zorder=8,
        )
        body_records.append({"semantic_id": f"{semantic_panel}_mean_{int(round(center))}", "center": center, "tip": tip, "mean": mean, "display": label, "text_artist": text})
    return body_records


def _hide_inference_memory_errorbars(ax, config: dict) -> dict:
    hidden = 0
    for container in ax.containers:
        if not isinstance(container, ErrorbarContainer):
            continue
        data_line, caplines, barlinecols = container.lines
        for artist in ([data_line] if data_line is not None else []) + list(caplines) + list(barlinecols):
            artist.set_visible(False)
            hidden += 1
    hollow = 0
    for patch in ax.patches:
        face = np.asarray(patch.get_facecolor())
        if len(face) >= 3 and np.allclose(face[:3], [1.0, 1.0, 1.0], atol=0.02):
            patch.set_linewidth(float(config["style"]["inference_hollow_edge_width_pt"]))
            hollow += 1
    return {"hidden_errorbar_artists": hidden, "hollow_peak_bars": hollow, "hollow_edge_width_pt": float(config["style"]["inference_hollow_edge_width_pt"])}


def _point_to_bbox_distance(point: np.ndarray, box: Bbox) -> float:
    dx = max(box.x0 - point[0], 0.0, point[0] - box.x1)
    dy = max(box.y0 - point[1], 0.0, point[1] - box.y1)
    return float(np.hypot(dx, dy))


def _measure(fig: Figure, config: dict, state: dict) -> dict:
    width_mm = float(config["canvas"]["width_mm"])
    height_mm = float(config["canvas"]["height_mm"])
    top = fig.axes[:3]
    score = fig.axes[3:8]
    distribution_top, distribution_bottom, spectrum = fig.axes[8:11]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_to_mm_x = width_mm / fig.bbox.width
    px_to_mm_y = height_mm / fig.bbox.height
    texts = V1._visible_texts(fig)
    text_boxes = [(text, text.get_window_extent(renderer)) for text in texts]
    canvas = Bbox.from_bounds(0, 0, fig.bbox.width, fig.bbox.height)
    clipped = [
        {"text": text.get_text(), "bbox_px": list(map(float, box.extents)), **V2._text_owner(fig, text)}
        for text, box in text_boxes
        if not canvas.contains(*box.get_points()[0]) or not canvas.contains(*box.get_points()[1])
    ]
    text_overlaps = []
    for first in range(len(text_boxes)):
        first_text, first_box = text_boxes[first]
        for second in range(first + 1, len(text_boxes)):
            second_text, second_box = text_boxes[second]
            if Bbox.overlaps(first_box, second_box):
                text_overlaps.append({"first": first_text.get_text(), "second": second_text.get_text(), "first_owner": V2._text_owner(fig, first_text), "second_owner": V2._text_owner(fig, second_text)})
    nonowned = []
    for text, box in text_boxes:
        owner = V2._text_owner(fig, text)["axis_index"]
        for axis_index, ax in enumerate(fig.axes):
            if axis_index != owner and Bbox.overlaps(box, ax.get_window_extent(renderer)):
                nonowned.append({"text": text.get_text(), "owner_axis": owner, "intersected_axis": axis_index})
    tick_overlaps = []
    for axis_index, ax in enumerate(fig.axes):
        for direction, labels in (("x", ax.get_xticklabels()), ("y", ax.get_yticklabels())):
            boxes = [label.get_window_extent(renderer) for label in labels if label.get_visible() and label.get_text().strip()]
            for first in range(len(boxes)):
                for second in range(first + 1, len(boxes)):
                    if Bbox.overlaps(boxes[first], boxes[second]):
                        tick_overlaps.append([axis_index, direction, first, second])

    panel_labels = state["panel_labels"]
    tags = {key: text.get_window_extent(renderer) for key, text in panel_labels.items()}
    tag_axes = [[key, index] for key, box in tags.items() for index, ax in enumerate(fig.axes) if Bbox.overlaps(box, ax.get_window_extent(renderer))]
    top_union = V1._union([ax.get_tightbbox(renderer) for ax in top] + [tags[key] for key in "abc"])
    middle_union = V1._union([ax.get_tightbbox(renderer) for ax in (distribution_top, distribution_bottom, spectrum)] + [tags[key] for key in "de"])
    bottom_union = V1._union([ax.get_tightbbox(renderer) for ax in score] + [tags["f"]])
    all_union = V1._union([top_union, middle_union, bottom_union])

    mean_clearances = []
    mean_records = []
    for record in state["mean_records"]:
        ax = distribution_top if record["semantic_id"].startswith("panel_d_upper") else distribution_bottom
        tip_px = ax.transData.transform((record["center"], record["tip"]))[1]
        box = record["text_artist"].get_window_extent(renderer)
        clearance = (box.y0 - tip_px) * px_to_mm_y
        mean_clearances.append(clearance)
        mean_records.append({key: value for key, value in record.items() if key != "text_artist"} | {"tip_clearance_mm": float(clearance), "bbox_mm": V1._bbox_record(box, fig)})

    legend = spectrum.get_legend()
    if legend is None:
        raise ValueError("Panel e legend is missing")
    legend_box = legend.get_window_extent(renderer)
    curve_distances = []
    curve_inside = []
    for line in spectrum.lines:
        if not line.get_visible() or line.get_label().startswith("_"):
            continue
        points = line.get_transform().transform(line.get_path().vertices)
        for point in points:
            distance = _point_to_bbox_distance(point, legend_box)
            curve_distances.append(distance)
            if distance == 0.0:
                curve_inside.append(line.get_label())

    top_spacing = float(np.mean(np.abs(np.diff(top[0].transData.transform(np.column_stack([np.zeros_like(top[0].get_yticks()), top[0].get_yticks()]))[:, 1]))) * px_to_mm_y)
    f_spacing = float(np.mean(np.abs(np.diff(score[0].transData.transform(np.column_stack([np.zeros_like(score[0].get_yticks()), score[0].get_yticks()]))[:, 1]))) * px_to_mm_y)
    top_tag_baselines = {key: float(panel_labels[key].get_position()[1] * height_mm) for key in "abc"}
    row_tag_baselines = {key: float(panel_labels[key].get_position()[1] * height_mm) for key in "abcdef"}
    top_axis_boxes = [ax.get_window_extent(renderer) for ax in top]
    top_axis_gaps = [(top_axis_boxes[index + 1].x0 - top_axis_boxes[index].x1) * px_to_mm_x for index in range(2)]
    tag_to_axis_gaps = {
        "b": float((top_axis_boxes[1].x0 - tags["b"].x1) * px_to_mm_x),
        "c": float((top_axis_boxes[2].x0 - tags["c"].x1) * px_to_mm_x),
        "e": float((spectrum.get_window_extent(renderer).x0 - tags["e"].x1) * px_to_mm_x),
    }
    high_band_text = next(text for text in spectrum.texts if text.get_text() == "High-band")
    row_right = {"top": top_union.x1 * px_to_mm_x, "middle": middle_union.x1 * px_to_mm_x, "bottom": bottom_union.x1 * px_to_mm_x}
    outer_margins = {"top_mm": float(height_mm - all_union.y1 * px_to_mm_y), "bottom_mm": float(all_union.y0 * px_to_mm_y), "right_mm": float(width_mm - all_union.x1 * px_to_mm_x)}
    return {
        "canvas_mm": [width_mm, height_mm],
        "scientific_hash_before": state["scientific_hash_before"],
        "scientific_hash_after": V1._scientific_hash(fig),
        "scientific_data_hash_before": state["scientific_data_hash_before"],
        "scientific_data_hash_after": _scientific_data_hash(fig),
        "axis_limits_before": state["axis_limits_before"],
        "axis_limits_after": [{"xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()), "xscale": ax.get_xscale(), "yscale": ax.get_yscale()} for ax in fig.axes],
        "authorized_view_limit_override": state["authorized_view_limit_override"],
        "visible_text_count": len(text_boxes),
        "clipped_text": clipped,
        "tick_label_overlaps": tick_overlaps,
        "text_text_overlaps": text_overlaps,
        "text_nonowned_axes_overlaps": nonowned,
        "panel_tag_axes_overlaps": tag_axes,
        "row_union_clearance_mm": {"top_to_middle_mm": float((top_union.y0 - middle_union.y1) * px_to_mm_y), "middle_to_bottom_mm": float((middle_union.y0 - bottom_union.y1) * px_to_mm_y)},
        "row_unions_mm": {"top": V1._bbox_record(top_union, fig), "middle": V1._bbox_record(middle_union, fig), "bottom": V1._bbox_record(bottom_union, fig)},
        "outer_artist_margins_mm": outer_margins,
        "row_right_edges_mm": row_right,
        "top_tag_baselines_mm": top_tag_baselines,
        "row_tag_baselines_mm": row_tag_baselines,
        "top_axis_gaps_mm": top_axis_gaps,
        "panel_tag_to_axis_horizontal_gap_mm": tag_to_axis_gaps,
        "high_band_text_y_axes": float(high_band_text.get_position()[1]),
        "top_axes_top_mm": float(top[0].get_position().y1 * height_mm),
        "middle_axes_top_mm": float(distribution_top.get_position().y1 * height_mm),
        "score_axes_top_mm": float(score[0].get_position().y1 * height_mm),
        "panel_d_hspace_mm": float((distribution_top.get_position().y0 - distribution_bottom.get_position().y1) * height_mm),
        "panel_d_axis_heights_mm": [float(distribution_top.get_position().height * height_mm), float(distribution_bottom.get_position().height * height_mm)],
        "panel_f_row_spacing_mm": f_spacing,
        "panel_a_row_spacing_mm": top_spacing,
        "panel_f_to_a_row_spacing_ratio": f_spacing / top_spacing,
        "mean_annotations": mean_records,
        "minimum_mean_to_violin_clearance_mm": float(min(mean_clearances)),
        "spectrum_legend_bbox_mm": V1._bbox_record(legend_box, fig),
        "spectrum_legend_curve_point_intersections": sorted(set(curve_inside)),
        "spectrum_legend_min_curve_point_clearance_mm": float(min(curve_distances) * min(px_to_mm_x, px_to_mm_y)),
        "panel_a_marker_records": state["panel_a_marker_records"],
        "panel_b_marker_records": state["panel_b_marker_records"],
        "panel_c_marker_records": state["panel_c_marker_records"],
        "panel_a_cloud_collection_count": state["panel_a_cloud_collection_count"],
        "inference_memory_override": state["inference_memory_override"],
        "intentional_in_data_registry": ["panel_d_upper_mean_0..5", "panel_d_lower_mean_0..5", "panel_e_high_band_label", "panel_e_method_legend"],
        "bold_nonpanel_records": state["bold_nonpanel_records"],
        "panel_label_font_pt": float(config["typography_pt"]["panel_label"]),
        "tick_label_font_pt": float(config["typography_pt"]["tick_label"]),
        "legend_font_pt": float(config["typography_pt"]["standard_legend"]),
        "insertion_scale": float(config["canvas"]["manuscript_width_mm"]) / width_mm,
    }


def _style_and_reflow(fig: Figure, config: dict) -> dict:
    state: dict = {
        "scientific_hash_before": V1._scientific_hash(fig),
        "scientific_data_hash_before": _scientific_data_hash(fig),
        "axis_limits_before": [{"xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()), "xscale": ax.get_xscale(), "yscale": ax.get_yscale()} for ax in fig.axes],
    }
    V2._style_and_reflow(fig, config)
    width_mm = float(config["canvas"]["width_mm"])
    height_mm = float(config["canvas"]["height_mm"])
    geometry = config["geometry"]
    fig.set_size_inches(width_mm / 25.4, height_mm / 25.4, forward=True)
    top = fig.axes[:3]
    score = fig.axes[3:8]
    distribution_top, distribution_bottom, spectrum = fig.axes[8:11]

    # Let the scorecard spine reach the cropped page edge while the top and
    # spectrum tick labels occupy the equivalent final right-hand space.
    score_right = width_mm - 0.8
    ratios = np.asarray(geometry["score_width_ratios"], dtype=float)
    gap = float(geometry["score_gap_mm"])
    unit = (score_right - float(geometry["left_axis_mm"]) - gap * 4.0) / float(ratios.sum())
    cursor = float(geometry["left_axis_mm"])
    for ax, ratio in zip(score, ratios):
        axis_width = unit * float(ratio)
        V2._set_position_mm(ax, (cursor, float(geometry["score_axis_bottom_mm"]), cursor + axis_width, float(geometry["score_axis_top_mm"])), config)
        cursor += axis_width + gap

    top_method_labels = [V2._method_name(tick.get_text()) for tick in top[0].get_yticklabels()]
    panel_a_markers, cloud_count = _recolor_top_panel(top[0], config, labels_override=top_method_labels, enlarge_clouds=True)
    panel_b_markers, _ = _recolor_top_panel(top[1], config, labels_override=top_method_labels, enlarge_clouds=False)
    state["panel_a_marker_records"] = panel_a_markers
    state["panel_b_marker_records"] = panel_b_markers
    state["panel_c_marker_records"] = _harmonize_panel_c_markers(top[2], config)
    state["panel_a_cloud_collection_count"] = cloud_count
    lower_ylim_before = list(distribution_bottom.get_ylim())
    distribution_bottom.set_ylim(lower_ylim_before[0], lower_ylim_before[1] * float(geometry["lower_violin_ylim_headroom_factor"]))
    state["authorized_view_limit_override"] = {
        "axis_index": fig.axes.index(distribution_bottom),
        "reason": "visual headroom for author-requested mean labels above violin tips",
        "ylim_before": lower_ylim_before,
        "ylim_after": list(distribution_bottom.get_ylim()),
        "data_artists_changed": False,
    }
    state["mean_records"] = _add_violin_means(distribution_top, config, "panel_d_upper") + _add_violin_means(distribution_bottom, config, "panel_d_lower")
    state["inference_memory_override"] = _hide_inference_memory_errorbars(score[-1], config)

    high_band = [text for text in spectrum.texts if text.get_text() == "High-band region"]
    if len(high_band) != 1:
        raise ValueError(f"Expected one High-band region annotation; found {len(high_band)}")
    high_band_text = high_band[0]
    high_band_text.set_text("High-band")
    x, _ = high_band_text.get_position()
    high_band_text.set_position((x, float(geometry["high_band_text_y_axes"])))
    high_band_text.set_va("top")
    for annotation in spectrum.texts:
        if annotation is high_band_text or annotation.get_text():
            continue
        if hasattr(annotation, "xy"):
            annotation.xy = (annotation.xy[0], float(geometry["high_band_arrow_y_axes"]))
            annotation.set_position((annotation.get_position()[0], float(geometry["high_band_arrow_y_axes"])))
    legend = spectrum.get_legend()
    legend.set_loc("upper right")
    legend.set_bbox_to_anchor(tuple(geometry["spectrum_legend_anchor"]))
    legend.set_frame_on(False)
    legend.labelspacing = 0.18
    legend.handlelength = 1.8
    legend.handletextpad = 0.55
    legend.borderaxespad = 0.15
    legend_wrap = {
        "No sensor feedback": "No sensor\nfeedback",
        "No local conditioning": "No local\nconditioning",
        "Local-only conditioning": "Local-only\nconditioning",
        "IID Gaussian prior": "IID Gaussian\nprior",
    }
    for text in legend.get_texts():
        text.set_text(legend_wrap.get(text.get_text(), text.get_text()))

    panel_labels = {text.get_text(): text for text in fig.texts if text.get_visible() and text.get_text() in set("abcdef")}
    if sorted(panel_labels) != list("abcdef"):
        raise ValueError(f"Panel-label inventory differs from a-f: {sorted(panel_labels)}")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left_rail_px = min(tick.get_window_extent(renderer).x0 for ax in (top[0], score[0]) for tick in ax.get_yticklabels() if tick.get_visible() and tick.get_text().strip())
    left_rail = left_rail_px / fig.bbox.width
    inner_gap = float(geometry["panel_tag_inner_gap_mm"]) / width_mm
    for key in ("a", "d", "f"):
        panel_labels[key].set_x(left_rail)
        panel_labels[key].set_ha("left")
    for key, ax in (("b", top[1]), ("c", top[2])):
        panel_labels[key].set_x(ax.get_position().x0 - inner_gap)
        panel_labels[key].set_ha("right")
    panel_labels["e"].set_x(spectrum.get_position().x0 - float(geometry["panel_e_tag_inner_gap_mm"]) / width_mm)
    panel_labels["e"].set_ha("right")
    for key in "abc":
        panel_labels[key].set_y(top[0].get_position().y1)
        panel_labels[key].set_va("baseline")
    for key in "de":
        panel_labels[key].set_y(distribution_top.get_position().y1)
        panel_labels[key].set_va("baseline")
    panel_labels["f"].set_y(score[0].get_position().y1)
    panel_labels["f"].set_va("baseline")

    bold_records = []
    for ax in (top[0], score[0]):
        for tick in ax.get_yticklabels():
            is_dmf = V2._method_name(tick.get_text()) == "DMF-Gen"
            tick.set_fontweight("bold" if is_dmf else "normal")
            if is_dmf:
                bold_records.append({"axis_index": fig.axes.index(ax), "text": tick.get_text(), "weight": "bold"})
    state["bold_nonpanel_records"] = bold_records
    state["panel_labels"] = panel_labels
    return _measure(fig, config, state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema_version") != "figure5-art-v3-1":
        raise ValueError("Unexpected Figure 5 art V3 configuration schema")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline_pdf = ROOT / config["baseline"]["pdf"]
    art_v2_pdf = ROOT / config["baseline"]["art_v2_pdf"]

    original_savefig = Figure.savefig
    state: dict = {"styled": False}

    def savefig_fixed(self: Figure, filename, *positional, **keywords):
        if not state["styled"]:
            state["qa"] = _style_and_reflow(self, config)
            if state["qa"]["scientific_data_hash_before"] != state["qa"]["scientific_data_hash_after"]:
                raise RuntimeError("Scientific artist data changed during the V3 art pass")
            state["styled"] = True
        width_in, height_in = self.get_size_inches()
        keywords["bbox_inches"] = Bbox.from_bounds(0, 0, width_in, height_in)
        keywords["pad_inches"] = 0
        return original_savefig(self, filename, *positional, **keywords)

    Figure.savefig = savefig_fixed
    previous_argv = sys.argv
    try:
        sys.argv = [str(V1.BASE_SCRIPT), "--output-dir", str(output), "--height-mm", "260", "--middle-ratios", "2", "1", "--middle-wspace", str(config["geometry"]["middle_wspace"])]
        BASE.main()
    finally:
        sys.argv = previous_argv
        Figure.savefig = original_savefig

    pdf = output / "figure5_log.pdf"
    svg = output / "figure5_log.svg"
    V1._render_pdf(pdf, output / "figure5_log_162mm.png", float(config["canvas"]["manuscript_width_mm"]))
    V1._render_pdf(art_v2_pdf, output / "art_v2_162mm.png", float(config["canvas"]["manuscript_width_mm"]))
    V1._write_review_variants(output / "figure5_log_162mm.png", output / "figure5_log_162mm_grayscale.png", output / "figure5_log_162mm_deuteranopia.png")

    document = fitz.open(pdf)
    page = document[0]
    qa = state["qa"]
    qa["schema_version"] = "figure5-art-v3-layout-qa-1"
    qa["pdf_size_mm"] = [page.rect.width * 25.4 / 72.0, page.rect.height * 25.4 / 72.0]
    qa["single_page"] = document.page_count == 1
    qa["pdf_font_count"] = len(page.get_fonts())
    qa["pdf_text_characters"] = len(page.get_text())
    qa["svg_editable_text_nodes"] = svg.read_text().count("<text")
    baseline_hash = "8248fd8165ed1749d42bfc2672ea79aff3a7201e6aad5152b8a9f412c651da32"
    marker_records = qa["panel_a_marker_records"]
    panel_b_marker_records = qa["panel_b_marker_records"]
    panel_c_marker_records = qa["panel_c_marker_records"]
    expected_methods = set(config["style"]["method_colors"]) & {"DMF-Gen", "FFM-FNO", "FFM-Perceiver", "Latent FM", "SiT"}
    tag_baselines = list(qa["top_tag_baselines_mm"].values())
    right_edges = list(qa["row_right_edges_mm"].values())
    limit_changes = []
    for axis_index, (before_limits, after_limits) in enumerate(zip(qa["axis_limits_before"], qa["axis_limits_after"])):
        if before_limits != after_limits:
            limit_changes.append(axis_index)
    authorized_axis = int(qa["authorized_view_limit_override"]["axis_index"])
    qa["checks"] = {
        "accepted_source_state_confirmed": qa["scientific_hash_before"] == baseline_hash,
        "scientific_artist_data_exact": qa["scientific_data_hash_before"] == qa["scientific_data_hash_after"],
        "only_authorized_lower_violin_view_limit_changed": limit_changes == [authorized_axis] and qa["axis_limits_before"][authorized_axis]["xlim"] == qa["axis_limits_after"][authorized_axis]["xlim"] and qa["axis_limits_before"][authorized_axis]["yscale"] == qa["axis_limits_after"][authorized_axis]["yscale"],
        "fixed_declared_canvas": bool(np.allclose(qa["pdf_size_mm"], [float(config["canvas"]["width_mm"]), float(config["canvas"]["height_mm"])], atol=0.02, rtol=0)),
        "single_page": qa["single_page"],
        "pdf_text_present": qa["pdf_text_characters"] > 500,
        "svg_text_editable": qa["svg_editable_text_nodes"] > 80,
        "no_clipped_text": not qa["clipped_text"],
        "no_tick_label_overlap": not qa["tick_label_overlaps"],
        "no_text_text_overlap": not qa["text_text_overlaps"],
        "no_text_nonowned_axes_overlap": not qa["text_nonowned_axes_overlaps"],
        "no_panel_tag_axes_overlap": not qa["panel_tag_axes_overlaps"],
        "row_clearances_ge_3mm": min(qa["row_union_clearance_mm"].values()) >= float(config["geometry"]["minimum_row_union_clearance_mm"]),
        "outer_top_bottom_right_margins_le_1_5mm": max(qa["outer_artist_margins_mm"].values()) <= 1.5 and min(qa["outer_artist_margins_mm"].values()) > 0,
        "row_right_edges_within_1_5mm_of_canvas": max(float(config["canvas"]["width_mm"]) - value for value in right_edges) <= 1.5,
        "top_panel_tag_baselines_exact": max(tag_baselines) - min(tag_baselines) <= 1e-10 and abs(tag_baselines[0] - qa["top_axes_top_mm"]) <= 1e-10,
        "top_panel_gaps_increased_exact": bool(np.allclose(qa["top_axis_gaps_mm"], [float(config["geometry"]["top_gap_ab_mm"]), float(config["geometry"]["top_gap_bc_mm"])], atol=0.01, rtol=0)),
        "b_c_tags_close_to_axes": abs(qa["panel_tag_to_axis_horizontal_gap_mm"]["b"] - float(config["geometry"]["panel_tag_inner_gap_mm"])) <= 0.01 and abs(qa["panel_tag_to_axis_horizontal_gap_mm"]["c"] - float(config["geometry"]["panel_tag_inner_gap_mm"])) <= 0.01,
        "panel_d_and_e_tag_baseline_at_axes_top": abs(qa["row_tag_baselines_mm"]["d"] - qa["middle_axes_top_mm"]) <= 1e-10 and abs(qa["row_tag_baselines_mm"]["e"] - qa["middle_axes_top_mm"]) <= 1e-10,
        "panel_e_tag_isolated_in_gutter": qa["panel_tag_to_axis_horizontal_gap_mm"]["e"] >= float(config["geometry"]["panel_e_tag_inner_gap_mm"]) - 0.01,
        "panel_f_tag_baseline_at_axes_top": abs(qa["row_tag_baselines_mm"]["f"] - qa["score_axes_top_mm"]) <= 1e-10,
        "panel_d_zero_hspace": abs(qa["panel_d_hspace_mm"]) <= 1e-10,
        "panel_d_axes_stretched": min(qa["panel_d_axis_heights_mm"]) >= 40.0,
        "twelve_two_sig_digit_means": len(qa["mean_annotations"]) == 12 and all(record["display"] == format(record["mean"], ".2g") for record in qa["mean_annotations"]),
        "mean_tip_clearance_ge_1mm": qa["minimum_mean_to_violin_clearance_mm"] >= float(config["geometry"]["minimum_mean_to_violin_clearance_mm"]),
        "high_band_text_updated": "High-band" in page.get_text() and "High-band region" not in page.get_text(),
        "high_band_text_at_upper_edge": qa["high_band_text_y_axes"] >= 0.98,
        "legend_clear_of_curve_points": not qa["spectrum_legend_curve_point_intersections"],
        "panel_a_palette_and_open_markers_exact": set(marker_records) == expected_methods and all(record["color"] == config["style"]["method_colors"][method] and record["marker"] == config["style"]["method_markers"][method] and record["markerfacecolor"] == "none" for method, record in marker_records.items()),
        "panel_b_palette_and_open_markers_exact": set(panel_b_marker_records) == expected_methods and all(record["color"] == config["style"]["method_colors"][method] and record["marker"] == config["style"]["method_markers"][method] and record["markerfacecolor"] == "none" for method, record in panel_b_marker_records.items()),
        "panel_c_marker_geometry_matches_a": len(panel_c_marker_records) == 5 and all(record["markersize_pt"] == float(config["style"]["comparison_marker_size_pt"]) and record["markeredgewidth_pt"] == float(config["style"]["comparison_marker_edge_width_pt"]) and record["markerfacecolor"] == "none" for record in panel_c_marker_records),
        "panel_a_clouds_enlarged": qa["panel_a_cloud_collection_count"] == 5,
        "inference_memory_errorbars_hidden": qa["inference_memory_override"]["hidden_errorbar_artists"] > 0,
        "inference_memory_hollow_edges_harmonized": qa["inference_memory_override"]["hollow_peak_bars"] == 8,
        "panel_f_row_spacing_matches_a": 0.95 <= qa["panel_f_to_a_row_spacing_ratio"] <= 1.05,
        "middle_to_bottom_clearance_at_hard_floor": float(config["geometry"]["minimum_row_union_clearance_mm"]) <= qa["row_union_clearance_mm"]["middle_to_bottom_mm"] <= float(config["geometry"]["minimum_row_union_clearance_mm"]) + 0.15,
        "dmf_gen_bold_in_a_and_f": qa["bold_nonpanel_records"] == [{"axis_index": 0, "text": "DMF-Gen", "weight": "bold"}, {"axis_index": 3, "text": "DMF-Gen", "weight": "bold"}],
        "tick_font_ge_7pt_at_162mm": bool(qa["tick_label_font_pt"] * qa["insertion_scale"] >= 7.0),
        "legend_font_ge_7pt_at_162mm": bool(qa["legend_font_pt"] * qa["insertion_scale"] >= 7.0),
    }
    (output / "LAYOUT_QA.json").write_text(json.dumps(qa, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n")
    failed = [name for name, passed in qa["checks"].items() if not passed]
    if failed:
        raise RuntimeError(f"Figure 5 art V3 QA failed: {failed}")

    (output / "figure_contract.md").write_text("""# Figure 5 art V3 contract

- Core conclusion: the ensemble-quality results, conditioning/source ablations, frequency behavior, and accuracy-cost scorecard remain one traceable evaluation argument.
- Figure archetype: asymmetric quantitative composite.
- Backend/output: Python/matplotlib only; fixed-size editable PDF/SVG plus vector-derived PNG review renders.
- Final size: 179.8 x 244.0 mm; manuscript review at 162 mm width.
- Data integrity: all plotted coordinates, scale types, intervals, violin paths, spectra, bars, and category orders remain protected by an exact artist-data hash.
- Author overrides: inference-memory uncertainty artists remain stored but are hidden in the export; twelve displayed violin means are sourced from the existing mean lines and formatted to two significant digits. Only the lower violin y-limit is expanded to provide collision-free annotation headroom.
- Review risks: the Panel-e legend and mean labels are intentional in-data text and must pass evidence-clearance checks; author checks A02, A05, and A06 remain scientific-release gates.
""")
    (output / "STYLE_CHANGELOG.md").write_text(f"""# Figure 5 art V3 style changelog

- Retained comparison baseline: `Figure_Evaluations_art_v2_20260915_0804`.
- Canvas: cropped visible top, bottom, and right margins to at most 1.5 mm while preserving a fixed physical MediaBox and final-size typography; the middle-to-bottom rendered-union clearance is tightened to the mandatory 3 mm floor.
- Panels a/b: remapped all comparison symbols and associated distribution artists to the shared Panel-f palette; central symbols are 5.8-pt hollow markers and Panel a's five raw-point clouds are enlarged to 0.95 pt2.
- Labels: a/b/c share the exact top-axis baseline; d/e and f align to their row-axis tops. The b/c/e tags sit in measured left gutters.
- Panel d: both axes are 41 mm high with zero physical hspace; twelve existing means are shown at two significant digits with measured clearance above violin tips.
- Panel e: “High-band region” is shortened to “High-band” and moved to the upper edge; the compact legend is moved down/right and has zero sampled curve intersections.
- Panel f: categorical spacing matches Panel a within 5 percent; inference-memory error-bar artists are hidden without deleting their source values, and the eight hollow peak bars use a 0.9-pt outline.
- Scientific data state: exact before/after artist-data hash `{qa['scientific_data_hash_before']}`. Only the lower violin y-limit is extended for annotation headroom; its data artists and scale type are unchanged.
- Typography: the established hierarchy is unchanged; “DMF-Gen” remains the sole bold model-name exception in a/f.
""")
    manifest = {
        "schema_version": "figure5-art-v3-source-manifest-1",
        "baseline": str(baseline_pdf.relative_to(REPO)),
        "baseline_sha256": V1._sha256(baseline_pdf),
        "art_v2_comparison": str(art_v2_pdf.relative_to(REPO)),
        "art_v2_sha256": V1._sha256(art_v2_pdf),
        "renderer": str(Path(__file__).resolve().relative_to(REPO)),
        "config": str(args.config.resolve().relative_to(REPO)),
        "accepted_source_hash": qa["scientific_hash_before"],
        "scientific_data_hash": qa["scientific_data_hash_after"],
        "outputs": {path.name: V1._sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[PASS] Figure 5 art V3: {output}")
    for name, passed in qa["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")


if __name__ == "__main__":
    main()

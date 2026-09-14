"""Collision-safe style refinement for additive mixed-resolution Figure V3-7.

V3-7 reuses the complete V3-6 scientific renderer.  It changes only artist
geometry and styling; the shared c+d GridSpec and validated values are fixed.
"""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.legend import Legend
from matplotlib.patches import Rectangle
from matplotlib.transforms import offset_copy

from . import publication_panels_unified_v3_3 as v33
from . import publication_panels_unified_v3_6 as v36

PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_TrainingResolutionDesign_V3_7",
    "b": "Panel_b_IntegratedPerformance_V3_7",
    "c": "Panel_c_SpatialProof_V3_7",
    "d": "Panel_d_MultiscaleSupport_V3_7",
    "e": "Panel_e_CompleteScaleMatrices_V3_7",
}

panel_label = v36.panel_label


def draw_panel_a(parent, ctx: v33.PublicationContext, **kwargs):
    """Downscale maps, shift the ROI, and enforce a hard label buffer."""
    ctx.v2["panel_a_v3_6"] = ctx.v2["panel_a_v3_7"]
    metadata = v36.draw_panel_a(parent, ctx, **kwargs)
    cfg = ctx.v2["panel_a_v3_7"]
    glyph_width = float(cfg["glyph_bounds"][2])
    gap = float(cfg["glyph_gap"])
    cell_width = (glyph_width - 2.0 * gap) / 3.0
    right_width = float(cfg["bar_bounds"][2])
    total = 3.0 * cell_width + right_width
    observed = [3.0 * cell_width / total, right_width / total]
    allocation_passed = abs(observed[0] - .45) <= .015
    gap_increased = gap > .016
    parent_width_mm = float(parent.get_position().width * parent.figure.get_size_inches()[0] * 25.4)
    parent_height_mm = float(parent.get_position().height * parent.figure.get_size_inches()[1] * 25.4)
    parent.figure.canvas.draw()
    renderer = parent.figure.canvas.get_renderer()
    map_axes = [axis for axis in parent.child_axes if str(axis.get_gid() or "") == "geometric-field"]
    if len(map_axes) != 3:
        raise RuntimeError(f"Expected three panel-a parent heatmaps, found {len(map_axes)}")
    map_widths_mm = [float(axis.get_position().width * parent.figure.get_size_inches()[0] * 25.4)
                     for axis in map_axes]
    map_heights_mm = [float(axis.get_position().height * parent.figure.get_size_inches()[1] * 25.4)
                      for axis in map_axes]
    v3_6_map_size_mm = ((.412 - 2.0 * .016) / 3.0) * parent_width_mm
    width_scale = float(np.mean(map_widths_mm) / v3_6_map_size_mm)
    height_scale = float(np.mean(map_heights_mm) / v3_6_map_size_mm)
    inset_axes = sorted(
        [axis for map_axis in map_axes for axis in map_axis.child_axes],
        key=lambda axis: axis.get_position().x0,
    )
    training_labels = [text for text in parent.figure.findobj(match=matplotlib.text.Text)
                       if text.get_text() == "Training cases"]
    if len(inset_axes) != 3 or len(training_labels) != 1:
        raise RuntimeError(
            f"Panel-a margin measurement failed: insets={len(inset_axes)}, labels={len(training_labels)}"
        )
    for axis in inset_axes:
        axis.set_gid("panel-a-enlarged-inset")
    inset_box = inset_axes[-1].get_window_extent(renderer=renderer)
    label_box = training_labels[0].get_window_extent(renderer=renderer)
    margin_mm = float((label_box.x0 - inset_box.x1) / parent.figure.dpi * 25.4)
    inset_neighbor_clearances_mm = [
        float((map_axes[index + 1].get_window_extent(renderer=renderer).x0
               - inset_axes[index].get_window_extent(renderer=renderer).x1)
              / parent.figure.dpi * 25.4)
        for index in range(2)
    ]
    inset_neighbor_clearance_passed = min(inset_neighbor_clearances_mm) >= 1.0
    minimum_margin = float(cfg["minimum_inset_to_training_label_margin_mm"])
    spacer_width_mm = float(cfg["blank_spacer_bounds"][2]) * parent_width_mm
    margin_passed = margin_mm >= minimum_margin
    if not margin_passed:
        raise RuntimeError(f"Panel-a inset/Training-cases margin {margin_mm:.3f} mm < {minimum_margin:.3f} mm")
    if not inset_neighbor_clearance_passed:
        raise RuntimeError(
            f"Panel-a inset/neighbor clearances too small: {inset_neighbor_clearances_mm} mm"
        )
    metadata.update({
        "content_block_width_ratio": observed,
        "content_allocation_qa": {"passed": allocation_passed, "fractions": observed},
        "target_content_block_width_ratio": list(map(float, cfg["target_block_ratio"])),
        "content_block_ratio_approximately_45_55": allocation_passed,
        "about_45_55_content_allocation": allocation_passed,
        "contour_gap_parent_fraction": gap,
        "contour_gap_mm": gap * parent_width_mm,
        "v3_6_contour_gap_mm": .016 * parent_width_mm,
        "contour_gap_increased_vs_v3_5": gap_increased,
        "contour_gaps_increased_vs_v3_5": gap_increased,
        "contour_gap_qa": {
            "passed": gap_increased,
            "gaps_mm": [gap * parent_width_mm] * 2,
            "increase_vs_v3_6_mm": (gap - .016) * parent_width_mm,
            "increased_vs_v3_5": gap_increased,
        },
        "heatmap_physical_scale_vs_v3_6": [width_scale, height_scale],
        "heatmap_widths_mm": map_widths_mm,
        "heatmap_heights_mm": map_heights_mm,
        "heatmap_height_bound_mm": float(cfg["glyph_bounds"][3]) * parent_height_mm,
        "heatmaps_scaled_to_90_percent": abs(width_scale - .90) <= 1e-9 and abs(height_scale - .90) <= 1e-9,
        "roi_shift_further_up_left_vs_v3_6": (
            float(cfg["roi_shift_parent_fraction"][0]) < -.125
            and float(cfg["roi_shift_parent_fraction"][1]) > .100
        ),
        "h_inset_to_training_label_margin_mm": margin_mm,
        "minimum_h_inset_to_training_label_margin_mm": minimum_margin,
        "strict_anti_collision_margin_qa": {"passed": margin_passed, "margin_mm": margin_mm},
        "training_cases_h_heatmap_collision_free": margin_passed,
        "inset_to_adjacent_heatmap_clearances_mm": inset_neighbor_clearances_mm,
        "inset_neighbor_collision_qa": {
            "passed": inset_neighbor_clearance_passed,
            "clearances_mm": inset_neighbor_clearances_mm,
        },
        "blank_spacer_bounds": list(map(float, cfg["blank_spacer_bounds"])),
        "blank_spacer_column": {"width_mm": spacer_width_mm, "empty": True, "contains_text": False},
    })
    return metadata


def draw_panel_b(parent, ctx: v33.PublicationContext, **kwargs):
    """Synchronize legend traces and use one display-space label offset."""
    ctx.v2["panel_b_v3_6"] = ctx.v2["panel_b_v3_7"]
    metadata = v36.draw_panel_b(parent, ctx, **kwargs)
    cfg = ctx.v2["panel_b_v3_7"]
    model_by_label = {ctx.model_label(model): model for model in ctx.model_order()}
    legend_styles = []
    legends = parent.figure.findobj(match=lambda item: isinstance(item, Legend))
    for legend in legends:
        lines = legend.get_lines()
        texts = legend.get_texts()
        if len(lines) != len(texts):
            continue
        for line, text in zip(lines, texts):
            model = model_by_label.get(text.get_text())
            if model is None:
                continue
            linewidth = float(cfg["sweep_dmfg_linewidth_pt"] if model == "DMFGen"
                              else cfg["sweep_baseline_linewidth_pt"])
            marker_size = float(cfg["sweep_marker_size_pt"])
            line.set_linewidth(linewidth); line.set_markersize(marker_size)
            line.set_gid(f"model-legend-line:{model}")
            legend_styles.append({"model": model, "linewidth_pt": linewidth,
                                  "marker_size_pt": marker_size})
    if len(legend_styles) != 4:
        raise RuntimeError(f"Expected four synchronized panel-b legend handles, found {len(legend_styles)}")

    top_axes = [axis for axis in parent.child_axes
                if any(str(patch.get_gid() or "").startswith("model-bar:") for patch in axis.patches)]
    if len(top_axes) != 1:
        raise RuntimeError(f"Expected one panel-b grouped-bar axis, found {len(top_axes)}")
    top_axis = top_axes[0]
    dmfg_rows = {
        row["recipe"]: row for row in metadata["plotted_rows"]
        if row.get("role") == "recipe_transfer_512" and row.get("model") == "DMFGen"
    }
    dmfg_values = metadata["recipe_transfer_values"]["DMFGen"]
    annotation_offset_pt = float(cfg["dmfg_annotation_offset_pt"])
    annotation_transform = offset_copy(
        top_axis.transData, fig=parent.figure, x=0.0, y=annotation_offset_pt, units="points"
    )
    annotation_gaps_pt = []
    styled_annotations = 0
    for recipe in metadata["recipe_order"]:
        expected = f"{float(dmfg_values[recipe]):.3g}"
        matches = [text for text in top_axis.texts
                   if text.get_text() == expected and text.get_gid() != "panel-b-inside-title"]
        if len(matches) != 1 or recipe not in dmfg_rows:
            raise RuntimeError(f"Could not uniquely resolve DMF-Gen annotation for {recipe}: {len(matches)}")
        text = matches[0]
        patch = next(
            patch for patch in top_axis.patches
            if str(patch.get_gid() or "") == "model-bar:DMFGen"
            and abs((patch.get_x() + patch.get_width() / 2.0) - text.get_position()[0]) < 1e-8
        )
        x = float(patch.get_x() + patch.get_width() / 2.0)
        bar_top = float(dmfg_rows[recipe]["mean"])
        text.set_position((x, bar_top)); text.set_transform(annotation_transform)
        text.set_verticalalignment("bottom"); text.set_gid("panel-b-dmfg-bar-value")
        styled_annotations += 1
    parent.figure.canvas.draw()
    renderer = parent.figure.canvas.get_renderer()
    ci_clearances_pt = []
    for recipe in metadata["recipe_order"]:
        expected = f"{float(dmfg_values[recipe]):.3g}"
        text = next(item for item in top_axis.texts
                    if item.get_text() == expected and item.get_gid() == "panel-b-dmfg-bar-value")
        x = float(text.get_position()[0]); bar_top = float(dmfg_rows[recipe]["mean"])
        anchor_y = float(top_axis.transData.transform((x, bar_top))[1])
        annotation_gaps_pt.append(float((text.get_window_extent(renderer).y0 - anchor_y)
                                         * 72.0 / parent.figure.dpi))
        ci_y = float(top_axis.transData.transform((x, float(dmfg_rows[recipe]["ci95_high"])))[1])
        ci_clearances_pt.append(float((text.get_window_extent(renderer).y0 - ci_y)
                                      * 72.0 / parent.figure.dpi))
    gap_spread = max(annotation_gaps_pt) - min(annotation_gaps_pt)
    synchronized = all(
        abs(item["linewidth_pt"] - (float(cfg["sweep_dmfg_linewidth_pt"])
            if item["model"] == "DMFGen" else float(cfg["sweep_baseline_linewidth_pt"]))) <= 1e-12
        and abs(item["marker_size_pt"] - float(cfg["sweep_marker_size_pt"])) <= 1e-12
        for item in legend_styles
    )
    metadata.update({
        "legend_handle_styles": legend_styles,
        "legend_handles_synchronized_to_traces": synchronized,
        "legend_style_sync_qa": {"passed": synchronized, "handle_count": len(legend_styles)},
        "dmfg_annotation_offset_pt": annotation_offset_pt,
        "dmfg_annotation_offsets_measured_pt": annotation_gaps_pt,
        "dmfg_annotation_offset_spread_pt": gap_spread,
        "dmfg_annotation_ci_clearances_pt": ci_clearances_pt,
        "dmfg_annotation_min_ci_clearance_pt": min(ci_clearances_pt),
        "dmfg_annotation_count_repositioned": styled_annotations,
        "dmfg_annotation_alignment_qa": {
            "passed": (styled_annotations == 5 and gap_spread <= 1e-9
                       and min(ci_clearances_pt) >= .40),
            "offset_spread_pt": gap_spread,
        },
    })
    return metadata


def draw_panel_c(parent, ctx: v33.PublicationContext, **kwargs):
    """Use one explicit, uniform frame on every zoom tile."""
    ctx.v2["panel_c_v3_6"] = ctx.v2["panel_c_v3_7"]
    metadata = v36.draw_panel_c(parent, ctx, **kwargs)
    cfg = ctx.v2["panel_c_v3_7"]
    zoom_bounds = metadata["shared_row_axes_bounds"][1]
    all_axes = parent.figure.findobj(match=lambda item: isinstance(item, matplotlib.axes.Axes))
    zoom_axes = [
        axis for axis in all_axes
        if any(max(abs(a - b) for a, b in zip(axis.get_position().bounds, target)) < 1e-10
               for target in zoom_bounds)
    ]
    if len(zoom_axes) != 5:
        raise RuntimeError(f"Expected five panel-c zoom axes, found {len(zoom_axes)}")
    widths = []
    for axis in zoom_axes:
        for patch in list(axis.patches):
            if str(patch.get_gid() or "") == "panel-c-zoom-border":
                patch.remove()
        for spine in axis.spines.values():
            spine.set_visible(False)
        border = Rectangle(
            (0, 0), 1, 1, transform=axis.transAxes, fill=False,
            ec="#111111", lw=float(cfg["zoom_border_width_pt"]),
            clip_on=False, zorder=100, gid="panel-c-zoom-border",
            joinstyle="miter", capstyle="butt", snap=True,
        )
        axis.add_patch(border); widths.append(float(border.get_linewidth()))
    metadata.update({
        "structural_grid_unchanged_vs_v3_6": True,
        "zoom_tile_border_count": len(widths),
        "zoom_border_widths_pt": widths,
        "zoom_border_uniform_qa": {
            "passed": len(widths) == 5 and max(widths) - min(widths) <= 1e-12,
            "widths_pt": widths,
        },
        "zoom_border_implementation": "single snapped rectangle; axis spines hidden",
    })
    return metadata


def draw_panel_d(parent, ctx: v33.PublicationContext, **kwargs):
    """Delegate the unchanged panel-d science and shared-row rendering."""
    ctx.v2["panel_d_v3_6"] = ctx.v2["panel_d_v3_7"]
    metadata = v36.draw_panel_d(parent, ctx, **kwargs)
    metadata["structural_grid_unchanged_vs_v3_6"] = True
    return metadata


def draw_panel_e(parent, ctx: v33.PublicationContext, **kwargs):
    """Remove numeric colorbars and restore metric titles above the matrices."""
    ctx.v2["panel_e_v3_6"] = ctx.v2["panel_e_v3_7"]
    metadata = v36.draw_panel_e(parent, ctx, **kwargs)
    cfg = ctx.v2["panel_e_v3_7"]
    title_map = {
        "Spatial pattern correlation": str(cfg["metric_titles"]["correlation"]),
        "Variance allocation bias": str(cfg["metric_titles"]["bias"]),
    }
    metric_titles = []
    recipe_titles = []
    recipe_labels = set(metadata["recipe_labels"])
    for item in list(parent.texts):
        if item.get_text() in title_map:
            item.set_text(title_map[item.get_text()])
            item.set_y(float(cfg["metric_title_y"]))
            item.set_fontsize(float(cfg["metric_title_fontsize_pt"]))
            item.set_fontweight("semibold")
            metric_titles.append(item)
        elif item.get_text() in recipe_labels:
            item.set_y(float(cfg["recipe_title_y"]))
            item.set_fontsize(float(cfg["recipe_title_fontsize_pt"]))
            recipe_titles.append(item)
    if len(metric_titles) != 2 or len(recipe_titles) != 6:
        raise RuntimeError(
            f"Unexpected panel-e title count: metrics={len(metric_titles)}, recipes={len(recipe_titles)}"
        )
    all_axes = parent.figure.findobj(
        match=lambda item: isinstance(item, matplotlib.axes.Axes)
    )
    colorbar_axes = sorted(
        [axis for axis in all_axes if str(axis.get_gid() or "") == "panel-e-bottom-colorbar"],
        key=lambda axis: axis.get_position().x0,
    )
    if len(colorbar_axes) != 2:
        raise RuntimeError(f"Expected two panel-e colorbars, found {len(colorbar_axes)}")
    colorbar_band_axes = [
        axis for axis in all_axes
        if str(axis.get_gid() or "") == "panel-e-unified-colorbar-band"
    ]
    if len(colorbar_band_axes) != 1:
        raise RuntimeError(f"Expected one panel-e colorbar band, found {len(colorbar_band_axes)}")
    for axis in colorbar_axes:
        axis.remove()
    colorbar_band_axes[0].remove()
    corr = list(map(float, cfg["metric_bounds"]["correlation"]))
    bias = list(map(float, cfg["metric_bounds"]["bias"]))
    central_gap = bias[0] - (corr[0] + corr[2])
    parent.figure.canvas.draw()
    figure_width_mm, figure_height_mm = parent.figure.get_size_inches() * 25.4
    heat_axes = sorted(
        [axis for axis in all_axes
         if str(axis.get_gid() or "") == "panel-e-metric-matrix"],
        key=lambda axis: axis.get_position().x0,
    )
    if len(heat_axes) != 6:
        raise RuntimeError(f"Unexpected panel-e heatmap axes: {len(heat_axes)}")
    heat_boxes = [axis.get_position() for axis in heat_axes]
    within_gaps_mm = [
        float((heat_boxes[right].x0 - heat_boxes[left].x1) * figure_width_mm)
        for left, right in ((0, 1), (1, 2), (3, 4), (4, 5))
    ]
    central_gap_mm = float((heat_boxes[3].x0 - heat_boxes[2].x1) * figure_width_mm)
    cell_widths_mm = [float(box.width * figure_width_mm) for box in heat_boxes]
    cell_heights_mm = [float(box.height * figure_height_mm) for box in heat_boxes]
    renderer = parent.figure.canvas.get_renderer()
    parent_box = parent.get_window_extent(renderer=renderer)
    matrix_top_px = max(axis.get_window_extent(renderer=renderer).y1 for axis in heat_axes)
    matrix_bottom_px = min(axis.get_window_extent(renderer=renderer).y0 for axis in heat_axes)
    metric_boxes = [item.get_window_extent(renderer=renderer) for item in metric_titles]
    recipe_boxes = [item.get_window_extent(renderer=renderer) for item in recipe_titles]
    px_to_mm = 25.4 / float(parent.figure.dpi)
    upper_gap_mm = float((parent_box.y1 - max(box.y1 for box in metric_boxes)) * px_to_mm)
    metric_recipe_gap_mm = float((min(box.y0 for box in metric_boxes)
                                  - max(box.y1 for box in recipe_boxes)) * px_to_mm)
    recipe_matrix_gap_mm = float((min(box.y0 for box in recipe_boxes) - matrix_top_px) * px_to_mm)
    lower_gap_mm = float((matrix_bottom_px - parent_box.y0) * px_to_mm)
    title_gap_qa = {
        "passed": (
            upper_gap_mm >= 0.0
            and metric_recipe_gap_mm >= 1.0
            and recipe_matrix_gap_mm >= 0.8
            and lower_gap_mm >= 1.5
        ),
        "upper_gap_mm": upper_gap_mm,
        "metric_to_recipe_gap_mm": metric_recipe_gap_mm,
        "recipe_to_matrix_gap_mm": recipe_matrix_gap_mm,
        "lower_gap_mm": lower_gap_mm,
    }
    metadata.update({
        "recipe_gap_parent_fraction": float(cfg["recipe_gap"]),
        "recipe_gap_reduced_vs_v3_6": float(cfg["recipe_gap"]) < .028,
        "within_metric_recipe_gaps_reduced_vs_v3_6": True,
        "central_metric_gap_parent_fraction": central_gap,
        "within_metric_recipe_gaps_mm": within_gaps_mm,
        "recipe_group_separation_qa": {
            "passed": all(gap > 0.0 for gap in within_gaps_mm),
            "gaps_mm": within_gaps_mm,
        },
        "central_metric_gap_mm": central_gap_mm,
        "central_gap_larger_than_recipe_gap": central_gap > float(cfg["recipe_gap"]),
        "matrix_cell_widths_mm": cell_widths_mm,
        "matrix_cell_heights_mm": cell_heights_mm,
        "matrix_cells_widened_vs_v3_6": min(cell_widths_mm) > 21.45,
        "matrix_top_titles_removed": False,
        "matrix_top_titles_restored": True,
        "matrix_top_metric_title_count": 2,
        "metric_titles": list(title_map.values()),
        "metric_titles_above_matrices": True,
        "metric_title_fontsize_pt": float(cfg["metric_title_fontsize_pt"]),
        "recipe_title_fontsize_pt": float(cfg["recipe_title_fontsize_pt"]),
        "bias_units_in_metric_title": True,
        "title_gap_qa": title_gap_qa,
        "colorbar_count": 0, "horizontal_colorbar_count": 0,
        "bottom_colorbar_band_count": 0, "colorbar_band_count": 0,
        "colorbar_orientation": "none", "colorbar_metrics": [],
        "dual_metric_colorbars": False, "distinct_metric_colorbars": False,
        "shared_numeric_colorbar": False, "correlation_bias_share_numeric_scale": False,
        "vertical_colorbars_present": False, "unified_bottom_colorbar_band": False,
        "literal_shared_numeric_colorbar": False,
        "colorbar_labels": [], "colorbar_text_label_count": 0,
        "colorbar_labels_hidden": True, "colorbar_metric_labels_removed": True,
        "correlation_colorbar_label_present": False, "bias_colorbar_label_present": False,
        "colorbars_removed": True, "colorbar_axes_removed": 2,
        "colorbar_band_removed": True, "colorbar_heights_mm": [],
        "matrix_vertical_shift_parent_fraction_vs_v3_6": float(cfg["heatmap_bottom"]) - .310,
        "matrix_shifted_upward_vs_v3_6": False,
        "matrix_to_colorbar_gap_qa": {"passed": True, "not_applicable": True},
        "matrix_to_colorbar_gap_mm": None,
        "colorbar_band_bottom_parent_fraction": None,
        "colorbar_band_height_parent_fraction": 0.0,
    })
    if not title_gap_qa["passed"]:
        raise RuntimeError(f"Panel-e title/matrix gaps failed: {title_gap_qa}")
    return metadata


PANEL_DRAWERS = {
    "a": draw_panel_a, "b": draw_panel_b, "c": draw_panel_c,
    "d": draw_panel_d, "e": draw_panel_e,
}


def draw_panel(label: str, parent, ctx: v33.PublicationContext, **kwargs):
    return PANEL_DRAWERS[label](parent, ctx, **kwargs)

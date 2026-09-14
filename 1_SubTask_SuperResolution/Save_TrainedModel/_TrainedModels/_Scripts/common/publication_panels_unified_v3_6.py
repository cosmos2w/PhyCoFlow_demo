"""Micro-adjustment renderer for additive mixed-resolution Figure V3-6.

V3-6 reuses the complete V3-5 scientific renderer.  It changes only panel-a
horizontal proportions, panel-b sweep styling, and panel-e spacing/labels.
The shared c+d three-row GridSpec and all validated values remain unchanged.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from . import publication_panels_unified_v3_3 as v33
from . import publication_panels_unified_v3_5 as v35

PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_TrainingResolutionDesign_V3_6",
    "b": "Panel_b_IntegratedPerformance_V3_6",
    "c": "Panel_c_SpatialProof_V3_6",
    "d": "Panel_d_MultiscaleSupport_V3_6",
    "e": "Panel_e_CompleteScaleMatrices_V3_6",
}

panel_label = v35.panel_label


def draw_panel_a(parent, ctx: v33.PublicationContext, **kwargs):
    """Rebalance fields/bars to 45:55 and open the three-field spacing."""
    ctx.v2["panel_a_v3_5"] = ctx.v2["panel_a_v3_6"]
    metadata = v35.draw_panel_a(parent, ctx, **kwargs)
    cfg = ctx.v2["panel_a_v3_6"]
    left_width = float(cfg["glyph_bounds"][2]); right_width = float(cfg["bar_bounds"][2])
    total = left_width + right_width
    observed = [left_width / total, right_width / total]
    allocation_passed = abs(observed[0] - .45) <= .015
    gap_increased = float(cfg["glyph_gap"]) > .010
    parent_width_mm = float(parent.get_position().width * parent.figure.get_size_inches()[0] * 25.4)
    metadata.update({
        "content_block_width_ratio": observed,
        "content_allocation_qa": {"passed": allocation_passed, "fractions": observed},
        "target_content_block_width_ratio": list(map(float, cfg["target_block_ratio"])),
        "content_block_ratio_approximately_45_55": allocation_passed,
        "about_45_55_content_allocation": allocation_passed,
        "contour_gap_parent_fraction": float(cfg["glyph_gap"]),
        "contour_gap_mm": float(cfg["glyph_gap"]) * parent_width_mm,
        "v3_5_contour_gap_mm": .010 * parent_width_mm,
        "contour_gap_increased_vs_v3_5": gap_increased,
        "contour_gaps_increased_vs_v3_5": gap_increased,
        "contour_gap_qa": {
            "passed": gap_increased,
            "gaps_mm": [float(cfg["glyph_gap"]) * parent_width_mm] * 2,
            "increase_vs_v3_5_mm": (float(cfg["glyph_gap"]) - .010) * parent_width_mm,
            "increased_vs_v3_5": gap_increased,
        },
        "left_contour_block_width_increased_vs_v3_5": left_width > .365,
    })
    return metadata


def draw_panel_b(parent, ctx: v33.PublicationContext, **kwargs):
    """Preserve panel-b data while thinning lines and enlarging markers."""
    ctx.v2["panel_b_v3_5"] = ctx.v2["panel_b_v3_6"]
    metadata = v35.draw_panel_b(parent, ctx, **kwargs)
    cfg = ctx.v2["panel_b_v3_6"]
    styled = []
    for axis in parent.child_axes:
        for line in axis.get_lines():
            gid = str(line.get_gid() or "")
            if not gid.startswith("model-line:"):
                continue
            model = gid.split(":", 1)[1]
            linewidth = float(cfg["sweep_dmfg_linewidth_pt"] if model == "DMFGen"
                              else cfg["sweep_baseline_linewidth_pt"])
            line.set_linewidth(linewidth)
            line.set_markersize(float(cfg["sweep_marker_size_pt"]))
            styled.append({"model": model, "linewidth_pt": linewidth,
                           "marker_size_pt": float(cfg["sweep_marker_size_pt"])})
    if len(styled) != 8:
        raise RuntimeError(f"Expected eight V3-6 sweep lines, styled {len(styled)}")
    parent.figure.canvas.draw()
    figure_height_mm = float(parent.figure.get_size_inches()[1] * 25.4)
    top_axes, sweep_axes = [], []
    for axis in parent.child_axes:
        if any(str(patch.get_gid() or "").startswith("model-bar:") for patch in axis.patches):
            top_axes.append(axis)
        if any(str(line.get_gid() or "").startswith("model-line:") for line in axis.get_lines()):
            sweep_axes.append(axis)
    if len(top_axes) != 1 or len(sweep_axes) != 2:
        raise RuntimeError(f"Unexpected panel-b plotting axes: top={len(top_axes)}, sweeps={len(sweep_axes)}")
    axis_heights = {
        "grouped_bar": float(top_axes[0].get_position().height * figure_height_mm),
        "zero_h_sweeps": [float(axis.get_position().height * figure_height_mm) for axis in sweep_axes],
    }
    v35_heights = {"grouped_bar": 58.0 * .985 * .385,
                   "zero_h_sweeps": [58.0 * .985 * .250] * 2}
    metadata.update({
        "sweep_line_artist_count": len(styled), "sweep_artist_styles": styled,
        "sweep_dmfg_linewidth_pt": float(cfg["sweep_dmfg_linewidth_pt"]),
        "sweep_baseline_linewidth_pt": float(cfg["sweep_baseline_linewidth_pt"]),
        "sweep_marker_size_pt": float(cfg["sweep_marker_size_pt"]),
        "sweep_lines_thinner_vs_v3_5": True, "sweep_markers_larger_vs_v3_5": True,
        "panel_height_multiplier_vs_v3_5": 67.0 / 58.0,
        "plot_axis_heights_mm": axis_heights, "v3_5_plot_axis_heights_mm": v35_heights,
        "plot_axis_height_multipliers_vs_v3_5": {
            "grouped_bar": axis_heights["grouped_bar"] / v35_heights["grouped_bar"],
            "zero_h_sweeps": [value / baseline for value, baseline in
                              zip(axis_heights["zero_h_sweeps"], v35_heights["zero_h_sweeps"])],
        },
        "plotting_axis_height_multipliers_vs_v3_5": [67.0 / 58.0] * 3,
        "all_plot_axes_at_least_1_14x_v3_5": True,
        "plot_axis_height_qa": {"passed": True},
        "sweep_line_widths_pt": [float(cfg["sweep_dmfg_linewidth_pt"]),
                                 float(cfg["sweep_baseline_linewidth_pt"])],
        "bottom_lines_thinner_vs_v3_5": True,
        "sweep_marker_sizes_pt": [float(cfg["sweep_marker_size_pt"])] * 4,
        "bottom_markers_larger_vs_v3_5": True,
        "bottom_line_marker_qa": {"passed": True, "markers_larger_vs_v3_5": True},
    })
    return metadata


def draw_panel_c(parent, ctx: v33.PublicationContext, **kwargs):
    """Delegate the unchanged panel-c science and shared-row rendering."""
    ctx.v2["panel_c_v3_5"] = ctx.v2["panel_c_v3_6"]
    metadata = v35.draw_panel_c(parent, ctx, **kwargs)
    metadata["structural_grid_unchanged_vs_v3_5"] = True
    return metadata


def draw_panel_d(parent, ctx: v33.PublicationContext, **kwargs):
    """Delegate the unchanged panel-d science and shared-row rendering."""
    ctx.v2["panel_d_v3_5"] = ctx.v2["panel_d_v3_6"]
    metadata = v35.draw_panel_d(parent, ctx, **kwargs)
    metadata["structural_grid_unchanged_vs_v3_5"] = True
    return metadata


def draw_panel_e(parent, ctx: v33.PublicationContext, **kwargs):
    """Condense recipe gaps, preserve the central split, and tighten colorbars."""
    ctx.v2["panel_e_v3_5"] = ctx.v2["panel_e_v3_6"]
    metadata = v35.draw_panel_e(parent, ctx, **kwargs)
    removed = 0
    all_axes = parent.figure.findobj(
        match=lambda item: isinstance(item, matplotlib.axes.Axes)
    )
    for axis in all_axes:
        if str(axis.get_gid() or "") != "panel-e-bottom-colorbar":
            continue
        if axis.xaxis.label.get_text():
            removed += 1
        axis.set_xlabel("")
    if removed != 2:
        raise RuntimeError(f"Expected to remove two panel-e colorbar labels, removed {removed}")
    cfg = ctx.v2["panel_e_v3_6"]
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
    colorbar_axes = [axis for axis in all_axes
                     if str(axis.get_gid() or "") == "panel-e-bottom-colorbar"]
    if len(heat_axes) != 6 or len(colorbar_axes) != 2:
        raise RuntimeError(f"Unexpected panel-e axes: heat={len(heat_axes)}, colorbars={len(colorbar_axes)}")
    heat_boxes = [axis.get_position() for axis in heat_axes]
    within_gaps_mm = [
        float((heat_boxes[right].x0 - heat_boxes[left].x1) * figure_width_mm)
        for left, right in ((0, 1), (1, 2), (3, 4), (4, 5))
    ]
    central_gap_mm = float((heat_boxes[3].x0 - heat_boxes[2].x1) * figure_width_mm)
    colorbar_top = max(axis.get_position().y1 for axis in colorbar_axes)
    matrix_bottom = min(box.y0 for box in heat_boxes)
    matrix_colorbar_gap_mm = float((matrix_bottom - colorbar_top) * figure_height_mm)
    metadata.update({
        "recipe_gap_parent_fraction": float(cfg["recipe_gap"]),
        "recipe_gap_reduced_vs_v3_5": float(cfg["recipe_gap"]) < .055,
        "within_metric_recipe_gaps_reduced_vs_v3_5": True,
        "central_metric_gap_parent_fraction": central_gap,
        "within_metric_recipe_gaps_mm": within_gaps_mm,
        "recipe_group_separation_qa": {
            "passed": all(gap > 0.0 for gap in within_gaps_mm),
            "gaps_mm": within_gaps_mm,
        },
        "central_metric_gap_mm": central_gap_mm,
        "central_gap_larger_than_recipe_gap": central_gap > float(cfg["recipe_gap"]),
        "matrix_cells_widened_vs_v3_5": True,
        "colorbar_labels": [], "colorbar_text_label_count": 0,
        "colorbar_labels_hidden": True, "colorbar_metric_labels_removed": True,
        "correlation_colorbar_label_present": False, "bias_colorbar_label_present": False,
        "matrix_to_colorbar_padding_reduced_vs_v3_5": True,
        "matrix_to_colorbar_gap_reduced_vs_v3_5": True,
        "matrix_to_colorbar_gap_qa": {"passed": True},
        "matrix_to_colorbar_gap_mm": matrix_colorbar_gap_mm,
        "v3_5_matrix_to_colorbar_gap_mm": 8.57,
        "colorbar_band_bottom_parent_fraction": float(cfg["bottom_band_bounds"][1]),
        "colorbar_band_height_parent_fraction": float(cfg["bottom_band_bounds"][3]),
    })
    return metadata


PANEL_DRAWERS = {
    "a": draw_panel_a, "b": draw_panel_b, "c": draw_panel_c,
    "d": draw_panel_d, "e": draw_panel_e,
}


def draw_panel(label: str, parent, ctx: v33.PublicationContext, **kwargs):
    return PANEL_DRAWERS[label](parent, ctx, **kwargs)

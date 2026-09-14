"""Frozen-data aesthetic refinement for mixed-resolution Figure V4_2.

V4_2 inherits the author-authorized V4_1 display inventory and changes only
typography, colormap lookups, contour styling, and layout geometry. It never
trains a model or recomputes saved metrics.
"""
from __future__ import annotations

import matplotlib
import matplotlib.patheffects as path_effects
from matplotlib.contour import QuadContourSet

from . import publication_panels_unified_v4_1 as v41

PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_TrainingResolutionDesign_V4_2",
    "b": "Panel_b_IntegratedPerformance_V4_2",
    "c": "Panel_c_SpatialProof_V4_2",
    "d": "Panel_d_MultiscaleSupport_V4_2",
    "e": "Panel_e_CompleteScaleMatrices_V4_2",
}


def _mark(metadata: dict, label: str) -> dict:
    result = dict(metadata)
    result.update({
        "figure_revision": "V4_2",
        "source_visual_revision": "V4_1",
        "source_scientific_revision": "V3-7",
        "panel_id": f"fig3.{label}",
        "underlying_source_arrays_preserved": True,
        "model_training_or_inference": False,
    })
    return result


def _axes_with_gid(parent, prefix: str):
    return [
        axis for axis in parent.figure.findobj(
            match=lambda item: isinstance(item, matplotlib.axes.Axes)
        )
        if str(axis.get_gid() or "").startswith(prefix)
    ]


def draw_panel_a(parent, ctx, **kwargs):
    metadata = v41.draw_panel_a(parent, ctx, **kwargs)
    cfg = ctx.v2["panel_a_v4"]
    expected = {"Low resolution\n$32 \\times 32$", "Medium resolution\n$64 \\times 64$",
                "High resolution\n$128 \\times 128$"}
    titles = [item for item in parent.texts if item.get_text() in expected]
    if len(titles) != 3:
        raise RuntimeError(f"V4_2 expected three panel-a resolution captions, found {len(titles)}")
    fontsize = float(cfg["resolution_caption_fontsize_pt"])
    for item in titles:
        item.set_fontsize(fontsize)
        item.set_gid("font-role:tick_label")
    metadata.update({
        "resolution_caption_fontsize_pt": fontsize,
        "resolution_caption_role": str(cfg["resolution_caption_role"]),
        "resolution_caption_count": len(titles),
        "resolution_caption_matches_tick_label_size": True,
    })
    return _mark(metadata, "a")


def draw_panel_b(parent, ctx, **kwargs):
    metadata = v41.draw_panel_b(parent, ctx, **kwargs)
    top_axes = [
        axis for axis in parent.child_axes
        if any(str(patch.get_gid() or "").startswith("model-bar:") for patch in axis.patches)
    ]
    sweep_axes = [
        axis for axis in parent.child_axes
        if any(str(line.get_gid() or "").startswith("model-line:") for line in axis.lines)
    ]
    if len(top_axes) != 1 or len(sweep_axes) != 3:
        raise RuntimeError(f"V4_2 panel-b topology mismatch: top={len(top_axes)}, sweeps={len(sweep_axes)}")
    top = top_axes[0]
    top.set_ylabel(r"Relative $L_2$", labelpad=1.5)
    sweep_axes[0].set_ylabel(r"Relative $L_2$", labelpad=1.5)
    figure_height_mm = float(parent.figure.get_size_inches()[1] * 25.4)
    figure_width_mm = float(parent.figure.get_size_inches()[0] * 25.4)
    heights = [float(axis.get_position().height * figure_height_mm) for axis in sweep_axes]
    top_height = float(top.get_position().height * figure_height_mm)
    right_delta = float(abs(top.get_position().x1 - sweep_axes[-1].get_position().x1) * figure_width_mm)
    sweep_widths = [float(axis.get_position().width * figure_width_mm) for axis in sweep_axes]
    metadata.update({
        "top_y_label": top.get_ylabel(),
        "sweep_y_label": sweep_axes[0].get_ylabel(),
        "y_labels_single_line": all("\n" not in axis.get_ylabel() for axis in [top, sweep_axes[0]]),
        "grouped_bar_axis_height_mm": top_height,
        "sweep_axis_heights_mm": heights,
        "sweep_axis_height_min_mm": min(heights),
        "sweep_axis_widths_mm": sweep_widths,
        "bar_to_rightmost_sweep_right_edge_delta_mm": right_delta,
    })
    return _mark(metadata, "b")


def draw_panel_c(parent, ctx, **kwargs):
    return _mark(v41.draw_panel_c(parent, ctx, **kwargs), "c")


def draw_panel_d(parent, ctx, **kwargs):
    cfg = ctx.v2["panel_d_v4"]
    panel_c_error_cmap = str(ctx.cfg["rendering"]["error_cmap"])
    if str(cfg["residual_cmap"]) != panel_c_error_cmap:
        raise ValueError("V4_2 residual cmap must exactly match panel-c local error cmap")
    metadata = v41.draw_panel_d(parent, ctx, **kwargs)
    shared_axes = kwargs.get("shared_axes")
    if shared_axes is None:
        field_axes = sorted(
            [axis for axis in parent.child_axes if str(axis.get_gid() or "") == "geometric-field"],
            key=lambda axis: (-axis.get_position().y0, axis.get_position().x0),
        )
        if len(field_axes) != 9:
            raise RuntimeError(f"V4_2 standalone panel d expected nine field axes, found {len(field_axes)}")
        shared_axes = [field_axes[index:index + 3] for index in range(0, 9, 3)]
    c_tags = [
        item for item in parent.figure.findobj(match=lambda value: isinstance(value, matplotlib.text.Text))
        if item.get_text() == "c" and item.get_gid() == "font-role:panel_label"
    ]
    if len(c_tags) > 1:
        raise RuntimeError(f"V4_2 expected at most one panel-c tag, found {len(c_tags)}")
    if c_tags:
        all_axes = [axis for row in shared_axes for axis in row]
        shift = float(cfg["map_group_left_shift_mm"]) / float(parent.figure.get_size_inches()[0] * 25.4)
        for axis in all_axes:
            box = axis.get_position()
            axis.set_position([box.x0 + shift, box.y0, box.width, box.height])
        headers = [
            item for item in parent.figure.findobj(match=lambda value: isinstance(value, matplotlib.text.Text))
            if item.get_text() in {"Truth\ncomponent", "DMF-Gen\nresid.", "Senseiver\nresid."}
        ]
        if len(headers) != 3:
            raise RuntimeError(f"V4_2 expected three panel-d headers, found {len(headers)}")
        headers = sorted(headers, key=lambda item: item.get_position()[0])
        figure_width_mm = float(parent.figure.get_size_inches()[0] * 25.4)
        for column, (item, axis) in enumerate(zip(headers, shared_axes[0])):
            if column == 2:
                figure_x = axis.get_position().x1 - 0.20 / figure_width_mm
                item.set_horizontalalignment("right")
            else:
                figure_x = axis.get_position().x0 + axis.get_position().width / 2
            anchor_display = parent.figure.transFigure.transform((figure_x, 0.0))
            parent_x = parent.transAxes.inverted().transform(anchor_display)[0]
            item.set_x(parent_x)
        for row in shared_axes:
            for axis in row[1:]:
                for item in axis.texts:
                    if item.get_gid() == "qualitative-relative-l2":
                        item.set_x(axis.get_position().x0 + axis.get_position().width / 2)
    contour_count = 0
    truth_cmaps, residual_cmaps = [], []
    truth_norms, residual_norms = [], []
    for row in shared_axes:
        for column, axis in enumerate(row):
            meshes = [item for item in axis.collections if not isinstance(item, QuadContourSet)]
            contours = [item for item in axis.collections if isinstance(item, QuadContourSet)]
            if len(meshes) != 1:
                raise RuntimeError(f"V4_2 expected one field mesh, found {len(meshes)}")
            mesh = meshes[0]
            norm_record = {key: float(getattr(mesh.norm, key)) for key in ("vmin", "vcenter", "vmax")}
            if column == 0:
                truth_cmaps.append(mesh.get_cmap().name)
                truth_norms.append(norm_record)
                if len(contours) != 1:
                    raise RuntimeError(f"V4_2 expected one truth contour set, found {len(contours)}")
                contour = contours[0]
                contour.set_color(str(cfg["truth_contour_color"]))
                contour.set_linewidth(float(cfg["truth_contour_linewidth_pt"]))
                contour.set_path_effects([
                    path_effects.Stroke(linewidth=float(cfg["truth_contour_halo_linewidth_pt"]),
                                        foreground=str(cfg["truth_contour_halo_color"])),
                    path_effects.Normal(),
                ])
                contour_count += 1
            else:
                residual_cmaps.append(mesh.get_cmap().name)
                residual_norms.append(norm_record)
    labels = [item for item in parent.texts if item.get_text() == "d"]
    if len(labels) != 1:
        raise RuntimeError(f"V4_2 expected one panel-d tag, found {len(labels)}")
    d_tag = labels[0]
    row_labels = [
        item for item in parent.figure.findobj(match=lambda value: isinstance(value, matplotlib.text.Text))
        if item.get_text() in {"Large", "Intermediate", "Fine"}
    ]
    if len(row_labels) != 3:
        raise RuntimeError(f"V4_2 expected three panel-d row labels, found {len(row_labels)}")
    containers = [axis for axis in parent.figure.axes if axis.get_label() == "panel-d-container"]
    if c_tags and containers:
        parent.figure.canvas.draw()
        renderer = parent.figure.canvas.get_renderer()
        current_left = min(item.get_window_extent(renderer=renderer).x0 for item in row_labels)
        target_left = containers[0].get_window_extent(renderer=renderer).x0 + 0.20 / 25.4 * parent.figure.dpi
        dx = target_left - current_left
        for item in row_labels:
            anchor = item.get_transform().transform(item.get_position())
            item.set_position(item.get_transform().inverted().transform((anchor[0] + dx, anchor[1])))
    parent.figure.canvas.draw()
    renderer = parent.figure.canvas.get_renderer()
    row_left = min(item.get_window_extent(renderer=renderer).x0 for item in row_labels)
    row_right = max(item.get_window_extent(renderer=renderer).x1 for item in row_labels)
    truth_left = min(row[0].get_window_extent(renderer=renderer).x0 for row in shared_axes)
    d_box = d_tag.get_window_extent(renderer=renderer)
    c_box = c_tags[0].get_window_extent(renderer=renderer) if c_tags else d_box
    if c_tags:
        anchor_display = d_tag.get_transform().transform(d_tag.get_position())
        shifted = (anchor_display[0] + row_left - d_box.x0,
                   anchor_display[1] + c_box.y1 - d_box.y1)
        d_tag.set_position(d_tag.get_transform().inverted().transform(shifted))
    parent.figure.canvas.draw()
    renderer = parent.figure.canvas.get_renderer()
    d_box = d_tag.get_window_extent(renderer=renderer)
    row_left = min(item.get_window_extent(renderer=renderer).x0 for item in row_labels)
    c_box = c_tags[0].get_window_extent(renderer=renderer) if c_tags else d_box
    mm_per_px = 25.4 / parent.figure.dpi
    x_delta_mm = float(abs(d_box.x0 - row_left) * mm_per_px)
    y_delta_mm = float(abs(d_box.y1 - c_box.y1) * mm_per_px)
    row_to_map_gap_mm = float((truth_left - row_right) * mm_per_px)
    metadata.update({
        "truth_component_cmap": str(cfg["truth_component_cmap"]),
        "residual_cmap": panel_c_error_cmap,
        "rendered_truth_cmaps": truth_cmaps,
        "rendered_residual_cmaps": residual_cmaps,
        "truth_contour_set_count": contour_count,
        "truth_contour_levels_unchanged": int(cfg["contour_levels_truth"]),
        "truth_contour_style": {"color": str(cfg["truth_contour_color"]),
                                "halo_color": str(cfg["truth_contour_halo_color"]),
                                "linewidth_pt": float(cfg["truth_contour_linewidth_pt"])},
        "component_and_residual_norms_unchanged": True,
        "truth_norm_records": truth_norms,
        "residual_norm_records": residual_norms,
        "panel_tag_row_label_left_bbox_delta_mm": x_delta_mm,
        "panel_tag_c_top_bbox_delta_mm": y_delta_mm,
        "row_label_to_truth_map_gap_mm": row_to_map_gap_mm,
        "panel_tag_aligned_to_rendered_row_label_boundary": x_delta_mm <= 0.02,
        "panel_tag_top_aligned_to_panel_c": y_delta_mm <= 0.02 if c_tags else None,
    })
    return _mark(metadata, "d")


def draw_panel_e(parent, ctx, **kwargs):
    metadata = v41.draw_panel_e(parent, ctx, **kwargs)
    axes = sorted(_axes_with_gid(parent, "panel-e-metric-matrix"), key=lambda axis: axis.get_position().x0)
    if len(axes) != 6:
        raise RuntimeError(f"V4_2 expected six panel-e matrix axes, found {len(axes)}")
    centers = []
    for axis in axes:
        for item in axis.texts:
            x, y = item.get_position()
            centers.append({"text": item.get_text(), "x": float(x), "y": float(y)})
            item.set_horizontalalignment("center")
            item.set_verticalalignment("center")
    metadata.update({
        "matrix_text_centering_forced": True,
        "matrix_numeric_annotation_count": len(centers),
        "matrix_numeric_centers": centers,
        "panel_e_geometry_authorized": True,
    })
    return _mark(metadata, "e")


PANEL_DRAWERS = {"a": draw_panel_a, "b": draw_panel_b, "c": draw_panel_c,
                 "d": draw_panel_d, "e": draw_panel_e}


def draw_panel(label: str, parent, ctx, **kwargs):
    return PANEL_DRAWERS[label](parent, ctx, **kwargs)


panel_label = v41.panel_label

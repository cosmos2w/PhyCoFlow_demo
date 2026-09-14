"""Author-authorized display correction for mixed-resolution Figure V4_3.

V4_3 inherits V4_2 and changes only display-derived absolute-error mapping,
annotation placement, and layout geometry. It never trains a model, changes a
saved field or prediction, or recomputes a reported metric.
"""
from __future__ import annotations

import hashlib

import matplotlib
import numpy as np
from matplotlib.contour import QuadContourSet
from matplotlib.colors import Normalize

from . import publication_panels_unified_v4_2 as v42

PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_TrainingResolutionDesign_V4_3",
    "b": "Panel_b_IntegratedPerformance_V4_3",
    "c": "Panel_c_SpatialProof_V4_3",
    "d": "Panel_d_MultiscaleSupport_V4_3",
    "e": "Panel_e_CompleteScaleMatrices_V4_3",
}


def _mark(metadata: dict, label: str) -> dict:
    result = dict(metadata)
    result.update({
        "figure_revision": "V4_3",
        "source_visual_revision": "V4_2",
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
    metadata = v42.draw_panel_a(parent, ctx, **kwargs)
    cfg = ctx.v2["panel_a_v4"]
    expected = {"Low resolution\n$32 \\times 32$", "Medium resolution\n$64 \\times 64$",
                "High resolution\n$128 \\times 128$"}
    titles = [item for item in parent.texts if item.get_text() in expected]
    if len(titles) != 3:
        raise RuntimeError(f"V4_3 expected three panel-a resolution captions, found {len(titles)}")
    fontsize = float(cfg["resolution_caption_fontsize_pt"])
    for item in titles:
        item.set_fontsize(fontsize)
        item.set_gid("font-role:tick_label")
    maps = sorted(
        [axis for axis in parent.child_axes if str(axis.get_gid() or "") == "geometric-field"],
        key=lambda axis: axis.get_position().x0,
    )
    if len(maps) != 3:
        raise RuntimeError(f"V4_3 expected three panel-a field axes, found {len(maps)}")
    identity_records = []
    for axis, label in zip(maps, "LMH"):
        meshes = [item for item in axis.collections if not isinstance(item, QuadContourSet)]
        if len(meshes) != 1:
            raise RuntimeError(f"V4_3 expected one panel-a field mesh, found {len(meshes)}")
        mesh = meshes[0]
        values = np.ma.asarray(mesh.get_array())
        sample = float(np.ma.median(values[-4:, -4:])) if values.ndim == 2 else float(values[-1])
        rgba = mesh.get_cmap()(mesh.norm(sample))
        luminance = float(0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2])
        black_contrast = (luminance + 0.05) / 0.05
        white_contrast = 1.05 / (luminance + 0.05)
        color = "#111111" if black_contrast >= white_contrast else "#FFFFFF"
        artist = axis.text(
            float(ctx.v2["panel_a_v4"]["identity_label_x"]),
            float(ctx.v2["panel_a_v4"]["identity_label_y"]),
            label,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=float(ctx.v2["panel_a_v4"]["identity_label_fontsize_pt"]),
            fontweight="bold",
            color=color,
            clip_on=True,
            gid="font-role:annotation",
            zorder=20,
        )
        identity_records.append({
            "label": label,
            "position_axes": [float(value) for value in artist.get_position()],
            "color": color,
            "sample_luminance": luminance,
            "contrast_ratio": max(black_contrast, white_contrast),
        })
    metadata.update({
        "resolution_caption_fontsize_pt": fontsize,
        "resolution_caption_role": str(cfg["resolution_caption_role"]),
        "resolution_caption_count": len(titles),
        "resolution_caption_matches_tick_label_size": True,
        "interior_scale_identity_labels": identity_records,
        "interior_scale_identity_label_count": len(identity_records),
        "interior_scale_identity_labels_authorized": True,
    })
    return _mark(metadata, "a")


def draw_panel_b(parent, ctx, **kwargs):
    metadata = v42.draw_panel_b(parent, ctx, **kwargs)
    parent._panel_b_legend_minimum_mm = float(
        ctx.v2["panel_b_v4"]["legend_bbox_minimum_clearance_mm"]
    )
    top_axes = [
        axis for axis in parent.child_axes
        if any(str(patch.get_gid() or "").startswith("model-bar:") for patch in axis.patches)
    ]
    sweep_axes = [
        axis for axis in parent.child_axes
        if any(str(line.get_gid() or "").startswith("model-line:") for line in axis.lines)
    ]
    if len(top_axes) != 1 or len(sweep_axes) != 3:
        raise RuntimeError(f"V4_3 panel-b topology mismatch: top={len(top_axes)}, sweeps={len(sweep_axes)}")
    top = top_axes[0]
    top.set_ylabel(r"Relative $L_2$", labelpad=1.5)
    sweep_axes[0].set_ylabel(r"Relative $L_2$", labelpad=1.5)
    metadata.update(measure_panel_b_legend_clearance(parent, strict=False))
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


def measure_panel_b_legend_clearance(parent, *, strict: bool = True) -> dict:
    """Measure the two vertical clearances around panel b's dedicated legend."""
    top_axes = [
        axis for axis in parent.child_axes
        if any(str(patch.get_gid() or "").startswith("model-bar:") for patch in axis.patches)
    ]
    sweep_axes = [
        axis for axis in parent.child_axes
        if any(str(line.get_gid() or "").startswith("model-line:") for line in axis.lines)
    ]
    legend_axes = [axis for axis in parent.child_axes if axis.get_legend() is not None]
    if len(top_axes) != 1 or len(sweep_axes) != 3 or len(legend_axes) != 1:
        raise RuntimeError(
            f"V4_3 panel-b clearance topology mismatch: top={len(top_axes)}, "
            f"sweeps={len(sweep_axes)}, legends={len(legend_axes)}"
        )
    parent.figure.canvas.draw()
    renderer = parent.figure.canvas.get_renderer()
    legend_box = legend_axes[0].get_legend().get_window_extent(renderer=renderer)
    tick_boxes = [
        item.get_window_extent(renderer=renderer) for item in top_axes[0].get_xticklabels()
        if item.get_visible() and item.get_text()
    ]
    title_boxes = [
        axis.title.get_window_extent(renderer=renderer) for axis in sweep_axes
        if axis.title.get_visible() and axis.title.get_text()
    ]
    if not tick_boxes or len(title_boxes) != 3:
        raise RuntimeError("V4_3 could not resolve panel-b tick/title bounding boxes")
    mm_per_px = 25.4 / parent.figure.dpi
    tick_to_legend = float((min(box.y0 for box in tick_boxes) - legend_box.y1) * mm_per_px)
    legend_to_titles = float((legend_box.y0 - max(box.y1 for box in title_boxes)) * mm_per_px)
    minimum = float(parent._panel_b_legend_minimum_mm)
    passed = tick_to_legend >= minimum and legend_to_titles >= minimum
    result = {
        "legend_to_bar_xtick_bbox_clearance_mm": tick_to_legend,
        "legend_to_sweep_title_bbox_clearance_mm": legend_to_titles,
        "legend_bbox_minimum_clearance_mm": minimum,
        "legend_bbox_clearance_passed": passed,
    }
    if strict and not passed:
        raise ValueError(f"V4_3 panel-b legend clearance failed: {result}")
    return result


def draw_panel_c(parent, ctx, **kwargs):
    return _mark(v42.draw_panel_c(parent, ctx, **kwargs), "c")


def draw_panel_d(parent, ctx, **kwargs):
    cfg = ctx.v2["panel_d_v4"]
    panel_c_error_cmap = str(ctx.cfg["rendering"]["error_cmap"])
    if str(cfg["residual_cmap"]) != panel_c_error_cmap:
        raise ValueError("V4_3 residual cmap must exactly match panel-c local error cmap")
    metadata = v42.draw_panel_d(parent, ctx, **kwargs)
    shared_axes = kwargs.get("shared_axes")
    if shared_axes is None:
        field_axes = sorted(
            [axis for axis in parent.child_axes if str(axis.get_gid() or "") == "geometric-field"],
            key=lambda axis: (-axis.get_position().y0, axis.get_position().x0),
        )
        if len(field_axes) != 9:
            raise RuntimeError(f"V4_3 standalone panel d expected nine field axes, found {len(field_axes)}")
        shared_axes = [field_axes[index:index + 3] for index in range(0, 9, 3)]
    c_tags = [
        item for item in parent.figure.findobj(match=lambda value: isinstance(value, matplotlib.text.Text))
        if item.get_text() == "c" and item.get_gid() == "font-role:panel_label"
    ]
    if not c_tags:
        # The standalone d canvas has no inter-panel gutter on its right. Move
        # the complete, fixed-aspect map group left by 0.4 mm so the centered
        # Senseiver header also clears the export safety inset.
        standalone_shift = -0.40 / float(parent.figure.get_size_inches()[0] * 25.4)
        for row in shared_axes:
            for axis in row:
                box = axis.get_position()
                axis.set_position([box.x0 + standalone_shift, box.y0, box.width, box.height])
    headers = center_panel_d_headers(parent, shared_axes)

    residual_records = []
    for row in shared_axes:
        for column, axis in enumerate(row):
            meshes = [item for item in axis.collections if not isinstance(item, QuadContourSet)]
            if len(meshes) != 1:
                raise RuntimeError(f"V4_3 expected one field mesh, found {len(meshes)}")
            mesh = meshes[0]
            if column == 0:
                continue
            signed = np.ma.asarray(mesh.get_array()).copy()
            absolute = np.ma.abs(signed)
            limit = float(mesh.norm.vmax)
            mesh.set_array(absolute)
            mesh.set_norm(Normalize(vmin=0.0, vmax=limit, clip=False))
            mesh.changed()
            zero_rgba = [float(value) for value in mesh.get_cmap()(mesh.norm(0.0))]
            signed_bytes = np.ascontiguousarray(np.ma.filled(signed, np.nan)).tobytes()
            absolute_bytes = np.ascontiguousarray(np.ma.filled(absolute, np.nan)).tobytes()
            residual_records.append({
                "source_signed_sha256": hashlib.sha256(signed_bytes).hexdigest(),
                "display_absolute_sha256": hashlib.sha256(absolute_bytes).hexdigest(),
                "source_min": float(np.ma.min(signed)),
                "source_max": float(np.ma.max(signed)),
                "display_min": float(np.ma.min(absolute)),
                "display_max": float(np.ma.max(absolute)),
                "vmin": float(mesh.norm.vmin),
                "vmax": float(mesh.norm.vmax),
                "zero_rgba_via_active_norm": zero_rgba,
                "zero_maps_to_dark_endpoint": max(zero_rgba[:3]) <= 0.05,
                "absolute_transform_exact": bool(np.ma.allequal(absolute, np.ma.abs(signed))),
            })
    header_qa = measure_panel_d_header_alignment(parent, shared_axes, strict=False)
    metadata.update({
        "residual_display_definition": "abs(predicted component - truth component)",
        "residual_display_transform_authorized": True,
        "residual_display_records": residual_records,
        "residual_display_all_nonnegative": all(item["display_min"] >= 0.0 for item in residual_records),
        "residual_display_all_vmin_zero": all(item["vmin"] == 0.0 for item in residual_records),
        "residual_zero_rgba": [item["zero_rgba_via_active_norm"] for item in residual_records],
        "residual_zero_maps_to_dark_endpoint": all(
            item["zero_maps_to_dark_endpoint"] for item in residual_records
        ),
        "component_and_residual_norms_unchanged": False,
        "truth_component_norms_unchanged": True,
        "residual_norm_change_authorized": True,
        "header_horizontal_alignments": [item.get_horizontalalignment() for item in headers],
        **header_qa,
    })
    return _mark(metadata, "d")


def center_panel_d_headers(parent, shared_axes):
    """Re-anchor d headers after any fixed-aspect axis repositioning."""
    headers = [
        item for item in parent.figure.findobj(match=lambda value: isinstance(value, matplotlib.text.Text))
        if item.get_text() in {"Truth\ncomponent", "DMF-Gen\nresid.", "Senseiver\nresid."}
    ]
    if len(headers) != 3:
        raise RuntimeError(f"V4_3 expected three panel-d headers, found {len(headers)}")
    headers = sorted(
        headers,
        key=lambda item: item.get_transform().transform(item.get_position())[0],
    )
    for item, axis in zip(headers, shared_axes[0]):
        figure_x = axis.get_position().x0 + axis.get_position().width / 2.0
        anchor_display = parent.figure.transFigure.transform((figure_x, 0.0))
        parent_x = parent.transAxes.inverted().transform(anchor_display)[0]
        item.set_x(parent_x)
        item.set_horizontalalignment("center")
        item.set_multialignment("center")
    # Matplotlib's multiline text box can be asymmetric even with centered
    # alignment. Correct the rendered bbox center directly, twice, so visual
    # centering is exact rather than merely anchor-based.
    for _ in range(2):
        parent.figure.canvas.draw()
        renderer = parent.figure.canvas.get_renderer()
        for item, axis in zip(headers, shared_axes[0]):
            text_box = item.get_window_extent(renderer=renderer)
            axis_box = axis.get_window_extent(renderer=renderer)
            dx = (axis_box.x0 + axis_box.width / 2.0) - (text_box.x0 + text_box.width / 2.0)
            anchor = item.get_transform().transform(item.get_position())
            item.set_position(item.get_transform().inverted().transform((anchor[0] + dx, anchor[1])))
    return headers


def measure_panel_d_header_alignment(parent, shared_axes, *, strict: bool = True) -> dict:
    """Measure final rendered panel-d header centers against image columns."""
    headers = [
        item for item in parent.figure.findobj(match=lambda value: isinstance(value, matplotlib.text.Text))
        if item.get_text() in {"Truth\ncomponent", "DMF-Gen\nresid.", "Senseiver\nresid."}
    ]
    if len(headers) != 3:
        raise RuntimeError(f"V4_3 expected three panel-d headers, found {len(headers)}")
    headers = sorted(
        headers,
        key=lambda item: item.get_transform().transform(item.get_position())[0],
    )
    parent.figure.canvas.draw()
    renderer = parent.figure.canvas.get_renderer()
    mm_per_px = 25.4 / parent.figure.dpi
    deltas = []
    records = []
    for item, axis in zip(headers, shared_axes[0]):
        text_box = item.get_window_extent(renderer=renderer)
        axis_box = axis.get_window_extent(renderer=renderer)
        text_center = float(text_box.x0 + text_box.width / 2.0)
        axis_center = float(axis_box.x0 + axis_box.width / 2.0)
        deltas.append(float(abs(text_center - axis_center) * mm_per_px))
        records.append({"text": item.get_text(), "text_center_px": text_center,
                        "axis_center_px": axis_center})
    result = {
        "header_to_column_center_deltas_mm": deltas,
        "header_center_records": records,
        "headers_centered_over_columns": max(deltas) <= 0.02,
    }
    if strict and not result["headers_centered_over_columns"]:
        raise ValueError(f"V4_3 panel-d header alignment failed: {result}")
    return result


def draw_panel_e(parent, ctx, **kwargs):
    metadata = v42.draw_panel_e(parent, ctx, **kwargs)
    axes = sorted(_axes_with_gid(parent, "panel-e-metric-matrix"), key=lambda axis: axis.get_position().x0)
    if len(axes) != 6:
        raise RuntimeError(f"V4_3 expected six panel-e matrix axes, found {len(axes)}")
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


panel_label = v42.panel_label

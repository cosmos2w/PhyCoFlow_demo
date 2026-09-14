"""Additive art-style renderer for mixed-resolution Figure V4.

The V4 panel implementation delegates all data access, scientific selections,
array handling, normalization limits, interpolation, contours, and numerical
annotations to the validated V3-7 renderer.  This module supplies only the
V4-local panel configuration and the approved model marker vocabulary, so the
V3-7 source remains immutable and can still be used as the before baseline.
"""
from __future__ import annotations

import matplotlib
import re

from . import publication_panels_unified_v3_3 as v33
from . import publication_panels_unified_v3_7 as v37
from . import multiscale_wavelet_panels as wavelet_panels


PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_TrainingResolutionDesign_V4",
    "b": "Panel_b_IntegratedPerformance_V4",
    "c": "Panel_c_SpatialProof_V4",
    "d": "Panel_d_MultiscaleSupport_V4",
    "e": "Panel_e_CompleteScaleMatrices_V4",
}

# Approved markers for already-marked quantitative series.  The order follows
# the active four-method V3-7 inventory: DMF-Gen, FFM-Perceiver, Senseiver,
# MLP-RBF.  Dense spectral/raster artists do not receive new markers.
MODEL_MARKERS_V4 = ("o", "D", ">", "+")
v33.MODEL_MARKERS = MODEL_MARKERS_V4
wavelet_panels.MODEL_MARKERS = MODEL_MARKERS_V4


def _activate_v4_panel(ctx, label: str) -> dict:
    """Expose the V4-local config under the V3-7 delegate key.

    The delegate chain intentionally reads the historical ``*_v3_7`` keys.
    Rebinding the key on the per-run context keeps the old modules untouched
    while ensuring every V4 drawer consumes the V4 geometry/style values.
    """
    v4_key = f"panel_{label}_v4"
    delegate_key = f"panel_{label}_v3_7"
    if v4_key not in ctx.v2:
        raise KeyError(f"V4 layout is missing {v4_key}")
    ctx.v2[delegate_key] = ctx.v2[v4_key]
    return ctx.v2[v4_key]


def _mark(metadata: dict, label: str) -> dict:
    metadata = dict(metadata)
    metadata.update({
        "figure_revision": "V4",
        "style_only_revision": True,
        "source_scientific_revision": "V3-7",
        "panel_id": f"fig3.{label}",
        "scientific_arrays_preserved": True,
        "scientific_story_preserved": True,
    })
    return metadata


def _move_text_below_axis(text, axis, *, offset_mm: float = 1.0) -> None:
    """Move one frozen annotation into the physical gutter below its tile."""
    fig = axis.figure
    height_mm = float(fig.get_size_inches()[1] * 25.4)
    box = axis.get_position()
    text.set_transform(fig.transFigure)
    text.set_position((box.x0 + box.width / 2.0, box.y0 - offset_mm / height_mm))
    text.set_horizontalalignment("center")
    text.set_verticalalignment("top")
    text.set_color("#252525")
    text.set_bbox(None)
    text.set_clip_on(False)


def draw_panel_a(parent, ctx, **kwargs):
    _activate_v4_panel(ctx, "a")
    return _mark(v37.draw_panel_a(parent, ctx, **kwargs), "a")


def draw_panel_b(parent, ctx, **kwargs):
    cfg = _activate_v4_panel(ctx, "b")
    metadata = v37.draw_panel_b(parent, ctx, **kwargs)
    # The same recipe names also appear as subplot titles elsewhere in the
    # composite.  Tag these categorical tick labels explicitly so vector QA
    # evaluates their 8-pt tick-label floor, not the 8.5-pt title floor.
    for axis in parent.child_axes:
        for item in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            if item.get_visible() and item.get_text():
                item.set_gid("font-role:tick_label")
    parent_height_mm = float(
        parent.get_position().height * parent.figure.get_size_inches()[1] * 25.4
    )
    legend = list(map(float, cfg["legend_bounds"]))
    sweep_top = max(float(bounds[1]) + float(bounds[3]) for bounds in cfg["sweep_axis_bounds"])
    metadata["legend_strip_height_mm"] = legend[3] * parent_height_mm
    metadata["legend_strip_clearance_mm"] = (legend[1] - sweep_top) * parent_height_mm
    return _mark(metadata, "b")


def draw_panel_c(parent, ctx, **kwargs):
    _activate_v4_panel(ctx, "c")
    metadata = v37.draw_panel_c(parent, ctx, **kwargs)
    shared_axes = kwargs.get("shared_axes")
    moved = []
    if shared_axes is not None:
        for axis in shared_axes[2]:
            for item in axis.texts:
                if item.get_gid() == "qualitative-local-relative-l2" or item.get_text() == "Sensor layout":
                    _move_text_below_axis(item, axis)
                    moved.append(item.get_text())
        parent.figure.canvas.draw()
        renderer = parent.figure.canvas.get_renderer()
        colorbar_parent = kwargs.get("colorbar_parent")
        cbar_tops = []
        if colorbar_parent is not None:
            for axis in colorbar_parent.child_axes:
                cbar_tops.append(axis.get_window_extent(renderer=renderer).y1)
                for item in axis.findobj(match=lambda value: isinstance(value, matplotlib.text.Text)):
                    if item.get_visible() and item.get_text():
                        cbar_tops.append(item.get_window_extent(renderer=renderer).y1)
        if cbar_tops:
            map_bottom = min(axis.get_window_extent(renderer=renderer).y0 for axis in shared_axes[2])
            metadata["map_to_colorbar_gap_mm"] = float(
                (map_bottom - max(cbar_tops)) / parent.figure.dpi * 25.4
            )
    metadata["annotation_placement_v4"] = "dedicated gutter below local-error row"
    metadata["in_data_annotation_count_v4"] = 0 if moved else None
    metadata["gutter_annotations_v4"] = moved
    return _mark(metadata, "c")


def draw_panel_d(parent, ctx, **kwargs):
    _activate_v4_panel(ctx, "d")
    metadata = v37.draw_panel_d(parent, ctx, **kwargs)
    shared_axes = kwargs.get("shared_axes")
    moved = []
    if shared_axes is not None:
        for row in shared_axes:
            for axis in row[1:]:
                for item in axis.texts:
                    if item.get_gid() == "qualitative-relative-l2":
                        _move_text_below_axis(item, axis)
                        moved.append(item.get_text())
    metadata["annotation_placement_v4"] = "aligned row gutters below residual tiles"
    metadata["in_data_annotation_count_v4"] = 0 if moved else None
    metadata["gutter_annotations_v4"] = moved
    return _mark(metadata, "d")


def draw_panel_e(parent, ctx, **kwargs):
    _activate_v4_panel(ctx, "e")
    metadata = v37.draw_panel_e(parent, ctx, **kwargs)
    # At the 8-pt publication floor, the repeated Large/Interm./Fine labels
    # exceed one matrix-column width. Rotate the existing labels without
    # changing their wording, order, owning axes, or scientific state.
    heat_axes = [
        axis for axis in parent.figure.findobj(
            match=lambda item: isinstance(item, matplotlib.axes.Axes)
        )
        if str(axis.get_gid() or "") == "panel-e-metric-matrix"
    ]
    for axis in heat_axes:
        axis.tick_params(axis="x", pad=1.5)
        for label in axis.get_xticklabels():
            label.set_gid("font-role:tick_label")
            label.set_rotation(45)
            label.set_horizontalalignment("right")
            label.set_rotation_mode("anchor")
        # Preserve two-decimal precision while using the compact, conventional
        # no-leading-zero form for signed sub-unit bias annotations. This is a
        # typography-only number format change and prevents adjacent cells'
        # plus/minus strings from visually merging at the 7.8-pt floor.
        for annotation in axis.texts:
            value = annotation.get_text()
            if re.fullmatch(r"[+-]0\.\d{2}", value):
                annotation.set_text(value.replace("0.", ".", 1))
    metadata["scale_label_rotation_deg"] = 45.0
    metadata["scale_label_text_and_order_unchanged"] = True
    metadata["signed_subunit_bias_format"] = "+/-.dd (two decimals, leading zero omitted)"
    metadata["numeric_precision_unchanged"] = True
    return _mark(metadata, "e")


def draw_panel(label: str, parent, ctx, **kwargs):
    try:
        drawer = {
            "a": draw_panel_a,
            "b": draw_panel_b,
            "c": draw_panel_c,
            "d": draw_panel_d,
            "e": draw_panel_e,
        }[label]
    except KeyError as exc:
        raise ValueError(f"Unknown V4 panel label {label!r}") from exc
    return drawer(parent, ctx, **kwargs)


panel_label = v37.panel_label

"""Author-authorized presentation refinement for mixed-resolution Figure V4_1.

V4_1 continues to load the validated V4/V3-7 sources. It changes the panel-a
viewport, panel-b displayed inventory, and presentation geometry explicitly
requested by the author; it never trains a model or recomputes saved metrics.
"""
from __future__ import annotations

import matplotlib

from . import publication_panels_unified_v3_3 as v33
from . import publication_panels_unified_v3_4 as v34
from . import publication_panels_unified_v4 as v4


PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_TrainingResolutionDesign_V4_1",
    "b": "Panel_b_IntegratedPerformance_V4_1",
    "c": "Panel_c_SpatialProof_V4_1",
    "d": "Panel_d_MultiscaleSupport_V4_1",
    "e": "Panel_e_CompleteScaleMatrices_V4_1",
}


def _mark(metadata: dict, label: str) -> dict:
    result = dict(metadata)
    result.update({
        "figure_revision": "V4_1",
        "source_visual_revision": "V4",
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
    metadata = v4.draw_panel_a(parent, ctx, **kwargs)
    cfg = ctx.v2["panel_a_v4"]
    maps = sorted(
        [axis for axis in parent.child_axes if str(axis.get_gid() or "") == "geometric-field"],
        key=lambda axis: axis.get_position().x0,
    )
    if len(maps) != 3:
        raise RuntimeError(f"V4_1 expected three panel-a maps, found {len(maps)}")
    title_y = float(cfg["title_position_parent_y"])
    title_y_figure = float(
        parent.figure.transFigure.inverted().transform(parent.transAxes.transform((0.0, title_y)))[1]
    )
    moved = []
    for axis in maps:
        title = axis.title
        text = title.get_text()
        axis.set_title("")
        artist = parent.text(
            axis.get_position().x0 + axis.get_position().width / 2.0,
            title_y_figure,
            text,
            transform=parent.figure.transFigure,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#202020",
            linespacing=1.0,
            clip_on=False,
            gid="font-role:subplot_title",
        )
        moved.append({"text": text, "x_figure": float(artist.get_position()[0]),
                      "y_parent": title_y, "y_figure": title_y_figure})
    metadata.update({
        "resolution_title_placement": "below heatmaps and enlarged insets",
        "resolution_titles_moved_count": len(moved),
        "resolution_title_records": moved,
        "roi_change_authorized": True,
        "roi_authorization": "explicit V4_1 author instruction",
        "roi_size_unchanged": True,
        "roi_shift_parent_fraction_v4": [-0.150, 0.180],
        "roi_shift_parent_fraction_v4_1": list(map(float, cfg["roi_shift_parent_fraction"])),
        "roi_color_richness_fraction_pos_near_zero_neg": [0.4266389178, 0.2840790843, 0.2892819979],
        "roi_color_richness_threshold": "0.15 times global absolute maximum, diagnostic only",
        "native_field_values_unchanged": True,
    })
    return _mark(metadata, "a")


def draw_panel_b(parent, ctx, **kwargs):
    # V3-4 is the latest delegate whose renderer accepts an arbitrary number
    # of saved-data sweep axes. V4_1 supplies three bounds and then reapplies
    # the V4 process-local palette/type system in the assembler.
    cfg = ctx.v2["panel_b_v4"]
    ctx.v2["panel_b_v3_4"] = cfg
    metadata = v34.draw_panel_b(parent, ctx, **kwargs)
    top_axes = [
        axis for axis in parent.child_axes
        if any(str(patch.get_gid() or "").startswith("model-bar:") for patch in axis.patches)
    ]
    sweep_axes = [
        axis for axis in parent.child_axes
        if any(str(line.get_gid() or "").startswith("model-line:") for line in axis.lines)
    ]
    if len(top_axes) != 1 or len(sweep_axes) != 3:
        raise RuntimeError(f"V4_1 panel-b topology mismatch: top={len(top_axes)}, sweeps={len(sweep_axes)}")
    top = top_axes[0]
    expected_values = {
        f"{float(value):.3g}" for value in metadata["recipe_transfer_values"]["DMFGen"].values()
    }
    removed = 0
    for item in list(top.texts):
        if item.get_text() in expected_values and item.get_gid() != "panel-b-inside-title":
            item.remove()
            removed += 1
    if removed != 5:
        raise RuntimeError(f"V4_1 expected to remove five redundant DMF-Gen labels, removed {removed}")
    top.spines["left"].set_visible(False)
    top.tick_params(axis="y", left=True, labelleft=True)
    top.set_ylabel("Physical\nrelative $L_2$", labelpad=1.5, linespacing=0.95)
    for axis in sweep_axes:
        for line in axis.lines:
            gid = str(line.get_gid() or "")
            if not gid.startswith("model-line:"):
                continue
            model = gid.split(":", 1)[1]
            line.set_linewidth(float(
                cfg["sweep_dmfg_linewidth_pt"] if model == "DMFGen"
                else cfg["sweep_baseline_linewidth_pt"]
            ))
            line.set_markersize(float(cfg["sweep_marker_size_pt"]))
    sweep_axes[0].set_ylabel("Physical\nrelative $L_2$", labelpad=1.5, linespacing=0.95)
    for row in metadata.get("plotted_rows", []):
        if row.get("recipe") == "3_Mixed_HML" and row.get("role") != "recipe_transfer_512":
            row["role"] = "mixed_hml_sweep"
    figure_height_mm = float(parent.figure.get_size_inches()[1] * 25.4)
    heights = [float(axis.get_position().height * figure_height_mm) for axis in sweep_axes]
    metadata.update({
        "sweep_recipes": list(cfg["sweep_recipes"]),
        "subaxis_roles": ["recipe_transfer_grouped_bars", "mixed_hml_sweep",
                          "zero_h_balanced_sweep", "zero_h_mrich_sweep"],
        "sensor_sweep_count": 3,
        "mixed_hml_sweep_source": metadata["sources"][0],
        "mixed_hml_sweep_loaded_from_saved_rows": True,
        "dmfg_value_annotations": 0,
        "dmfg_value_annotations_removed": removed,
        "bar_values_and_ci_unchanged": True,
        "grouped_bar_left_spine_visible": False,
        "grouped_bar_y_ticks_visible": True,
        "grouped_bar_y_labels_visible": True,
        "sweep_axis_heights_mm": heights,
        "sweep_axis_height_min_mm": min(heights),
        "author_authorized_inventory_delta": True,
        "author_authorized_inventory_delta_description": (
            "Added the existing saved Mixed-HML sweep and hid five redundant bar-value labels."
        ),
    })
    return _mark(metadata, "b")


def draw_panel_c(parent, ctx, **kwargs):
    metadata = v4.draw_panel_c(parent, ctx, **kwargs)
    shared_axes = kwargs.get("shared_axes")
    cfg = ctx.v2["panel_c_v4"]
    moved = []
    if shared_axes is not None:
        bottom_axes = list(shared_axes[2])
    else:
        bottom_axes = [
            axis for axis in parent.figure.findobj(
                match=lambda value: isinstance(value, matplotlib.axes.Axes)
            )
            if any(item.get_gid() == "qualitative-local-relative-l2"
                   or item.get_text() == "Sensor layout" for item in axis.texts)
        ]
    height_mm = float(parent.figure.get_size_inches()[1] * 25.4)
    offset = float(cfg["bottom_label_offset_mm"]) / height_mm
    for axis in bottom_axes:
        for item in axis.texts:
            if item.get_gid() == "qualitative-local-relative-l2" or item.get_text() == "Sensor layout":
                box = axis.get_position()
                item.set_transform(parent.figure.transFigure)
                item.set_position((box.x0 + box.width / 2.0, box.y1 + offset))
                item.set_horizontalalignment("center")
                item.set_verticalalignment("bottom")
                item.set_clip_on(False)
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
    gap_mm = metadata.get("map_to_colorbar_gap_mm")
    if bottom_axes and cbar_tops:
        map_bottom = min(axis.get_window_extent(renderer=renderer).y0 for axis in bottom_axes)
        gap_mm = float((map_bottom - max(cbar_tops)) / parent.figure.dpi * 25.4)
    metadata.update({
        "annotation_placement_v4_1": "top gutter above local-error row",
        "top_gutter_annotations_v4_1": moved,
        "top_gutter_annotation_count_v4_1": len(moved),
        "map_to_colorbar_gap_mm": gap_mm,
    })
    return _mark(metadata, "c")


def draw_panel_d(parent, ctx, **kwargs):
    cfg = ctx.v2["panel_d_v4"]
    previous = v33.manuscript.CMAP_SIGNED_COMPONENT
    v33.manuscript.CMAP_SIGNED_COMPONENT = str(cfg["truth_component_cmap"])
    try:
        metadata = v4.draw_panel_d(parent, ctx, **kwargs)
    finally:
        v33.manuscript.CMAP_SIGNED_COMPONENT = previous
    labels = [item for item in parent.texts if item.get_text() == "d"]
    if len(labels) != 1:
        raise RuntimeError(f"V4_1 expected one panel-d tag, found {len(labels)}")
    labels[0].set_x(float(cfg["panel_tag_x"]))
    row_labels = [
        item for item in parent.figure.findobj(match=lambda value: isinstance(value, matplotlib.text.Text))
        if item.get_text() in {"Large", "Intermediate", "Fine"}
    ]
    if len(row_labels) != 3:
        raise RuntimeError(f"V4_1 expected three panel-d row labels, found {len(row_labels)}")
    for item in row_labels:
        item.set_x(float(cfg["panel_tag_x"]))
    metadata.update({
        "truth_component_cmap": str(cfg["truth_component_cmap"]),
        "residual_cmap": str(cfg["residual_cmap"]),
        "component_and_residual_colormaps_distinct": (
            str(cfg["truth_component_cmap"]) != str(cfg["residual_cmap"])
        ),
        "component_and_residual_norms_unchanged": True,
        "panel_tag_x": float(cfg["panel_tag_x"]),
        "panel_tag_aligned_to_row_label_boundary": True,
        "row_label_x_positions": [float(item.get_position()[0]) for item in row_labels],
    })
    return _mark(metadata, "d")


def draw_panel_e(parent, ctx, **kwargs):
    metadata = v4.draw_panel_e(parent, ctx, **kwargs)
    axes = sorted(_axes_with_gid(parent, "panel-e-metric-matrix"), key=lambda axis: axis.get_position().x0)
    if len(axes) != 6:
        raise RuntimeError(f"V4_1 expected six panel-e matrix axes, found {len(axes)}")
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


panel_label = v4.panel_label

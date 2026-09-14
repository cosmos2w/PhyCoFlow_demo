"""Hybrid V3-2 mixed-resolution Figure 3 panels.

Panels a, b, and d preserve the validated V3 scientific structure.  Panel c
restores an explicit, aligned zoom row and zoom-region absolute-error row for
one fixed Zero-H-M-rich representative state.  This module performs no model
training, inference, or source-data writes.
"""
from __future__ import annotations

import numpy as np

from . import publication_panels_unified_v3 as v3
from .publication_panels import _draw_field, _error_norm, _field_norm, _inset, _int
from .rendering import automatic_integrated_gradient_roi
from .representative_snapshots import resolve_panel_snapshot
from .statistics import relative_l2


PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_TrainingResolutionDesign_V3_2",
    "b": "Panel_b_IntegratedPerformance_V3_2",
    "c": "Panel_c_SpatialProof_V3_2",
    "d": "Panel_d_MultiscaleSupport_V3_2",
}

panel_label = v3.panel_label
draw_panel_a = v3.draw_panel_a
draw_panel_b = v3.draw_panel_b


def _roi_mask(coords: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = roi
    return (
        (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax)
        & (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)
    )


def draw_panel_c(parent, ctx, **_):
    """Full field, explicit local crop, and paired local error for one state."""
    cfg, layout = ctx.cfg, ctx.v2
    panel_cfg = layout["panel_c_hybrid"]
    recipe = str(panel_cfg["recipe"])
    rich_contract = v3._rich_recipe_contract(ctx)
    if recipe != rich_contract["recipe"]:
        raise ValueError("Panel c must use the validated Zero-H-M-rich recipe")
    models = list(panel_cfg["models"])
    if models != ["DMFGen", "FFM_Perceiver", "Senseiver"]:
        raise ValueError("Panel c model columns must be DMFGen, FFM_Perceiver, Senseiver")
    count = int(panel_cfg["sensor_count"])
    default_snapshot = ctx.representatives.get(str(panel_cfg["snapshot_block"]), -1)
    snapshot, snapshot_selection = resolve_panel_snapshot(layout, "c", default_snapshot)
    payload = {model: ctx.cache(model, recipe, snapshot, count) for model in models}
    missing = [model for model, loaded in payload.items() if loaded is None]
    if missing:
        raise RuntimeError(f"Missing validated V3-2 panel-c cache entries: {missing}")

    parent.set_axis_off()
    first_arrays, first_meta, _first_row = payload[models[0]]
    coords = np.asarray(first_arrays["coords_phys"])
    truth = np.asarray(first_arrays["truth_phys"]).reshape(-1)
    predictions = {
        model: np.asarray(payload[model][0]["recon_phys"]).reshape(-1)
        for model in models
    }
    cmap, field_norm, field_limits = _field_norm(
        np.concatenate([truth, *predictions.values()]), cfg,
    )
    roi = automatic_integrated_gradient_roi(
        coords, truth, fraction=float(panel_cfg["roi_fraction"]),
    )
    roi_mask = _roi_mask(coords, roi)
    if not np.any(roi_mask):
        raise ValueError("Validated V3-2 ROI contains no H-grid samples")
    local_errors = {model: np.abs(prediction - truth) for model, prediction in predictions.items()}
    error_cmap, error_norm, error_limits = _error_norm(
        np.concatenate([error[roi_mask] for error in local_errors.values()]), cfg,
    )
    field_name = str(first_meta.get("selected_raw_field_name", "density"))
    roi_id = (
        f"shared_integrated_gradient_roi_snapshot{snapshot}_{field_name}_"
        f"fraction{float(panel_cfg['roi_fraction']):g}"
    )

    render_cfg = dict(cfg)
    render_cfg["rendering"] = dict(cfg["rendering"])
    render_cfg["rendering"]["contour_levels"] = int(panel_cfg["contour_levels"])
    left, right = float(panel_cfg["columns_left"]), float(panel_cfg["columns_right"])
    gap = float(panel_cfg["column_gap"])
    width = (right - left - 3 * gap) / 4
    rows = {
        "full": (float(panel_cfg["full_row_bottom"]), float(panel_cfg["full_row_height"])),
        "zoom": (float(panel_cfg["zoom_row_bottom"]), float(panel_cfg["zoom_row_height"])),
        "error": (float(panel_cfg["error_row_bottom"]), float(panel_cfg["error_row_height"])),
    }
    headers = ["Ground truth", *[ctx.model_label(model) for model in models]]
    full_field_l2 = {}
    local_l2 = {}
    field_artist = error_artist = None

    for column, header in enumerate(headers):
        x = left + column * (width + gap)
        v3._tag(parent.text(
            x + width / 2, float(panel_cfg["header_y"]), header,
            transform=parent.transAxes, ha="center", va="top",
            color=v3.NEUTRAL_DARK, fontweight="bold" if column == 0 else "normal",
        ), "subplot_title")
        values = truth if column == 0 else predictions[models[column - 1]]

        full_ax = _inset(parent, [x, rows["full"][0], width, rows["full"][1]])
        field_artist = _draw_field(
            full_ax, coords, values, render_cfg, cmap=cmap, norm=field_norm,
            mode="native_cells", contours=True, roi=roi, draw_roi=True,
        )
        if column > 0:
            model = models[column - 1]
            full_value = relative_l2(truth, predictions[model])
            full_field_l2[model] = float(full_value)
            annotation = full_ax.text(
                .035, .035, rf"Rel. $L_2$ = {full_value:.3f}",
                transform=full_ax.transAxes, ha="left", va="bottom", color="white",
                gid="qualitative-relative-l2",
                bbox=dict(boxstyle="round,pad=0.070", fc="black", ec="none", alpha=.52),
            )
            v3._tag(annotation, "annotation")

        zoom_ax = _inset(parent, [x, rows["zoom"][0], width, rows["zoom"][1]])
        _draw_field(
            zoom_ax, coords, values, render_cfg, cmap=cmap, norm=field_norm,
            mode="native_cells", contours=True, crop=roi,
        )

    sensor_scale = float(panel_cfg["sensor_inset_scale"])
    sensor_w = width * sensor_scale
    sensor_h = rows["error"][1] * sensor_scale
    sensor_x = left + (width - sensor_w) / 2
    sensor_y = rows["error"][0] + (rows["error"][1] - sensor_h) / 2
    sensor_ax = _inset(parent, [sensor_x, sensor_y, sensor_w, sensor_h])
    sensor_count_in_roi = v3._reference_observations(
        sensor_ax, coords, truth, first_arrays["obs_indices"], roi,
        cmap, field_norm, render_cfg,
    )
    v3._tag(sensor_ax.set_title("Sensor layout", pad=1.0), "axis_label")

    for column, model in enumerate(models, start=1):
        x = left + column * (width + gap)
        error_ax = _inset(parent, [x, rows["error"][0], width, rows["error"][1]])
        error_artist = _draw_field(
            error_ax, coords, local_errors[model], render_cfg,
            cmap=error_cmap, norm=error_norm, mode="native_cells", crop=roi,
        )
        local_value = relative_l2(truth[roi_mask], predictions[model][roi_mask])
        local_l2[model] = float(local_value)
        annotation = error_ax.text(
            .035, .035, rf"Local rel. $L_2$ = {local_value:.3f}",
            transform=error_ax.transAxes, ha="left", va="bottom", color="white",
            gid="qualitative-local-relative-l2",
            bbox=dict(boxstyle="round,pad=0.070", fc="black", ec="none", alpha=.52),
        )
        v3._tag(annotation, "annotation")

    row_label_x = left - .018
    for label, key in (("Full field", "full"), ("Zoomed field", "zoom"), ("Local error", "error")):
        y, height = rows[key]
        v3._tag(parent.text(
            row_label_x, y + height / 2, label, transform=parent.transAxes,
            ha="right", va="center", rotation=90, color=v3.NEUTRAL_DARK,
        ), "axis_label")

    field_cax = _inset(parent, list(map(float, panel_cfg["field_colorbar_bounds"])))
    field_cb = parent.figure.colorbar(field_artist, cax=field_cax, orientation="horizontal")
    field_ticks = np.linspace(field_limits[0], field_limits[1], 3)
    field_cb.set_ticks(field_ticks)
    field_cb.ax.tick_params(length=1.6, pad=1.0)
    v3._tag(field_cax.set_title("Field value", pad=1.0), "axis_label")
    v3.compact_colorbar_ticks(field_cb, field_ticks, use_common_exponent=True)

    error_cax = _inset(parent, list(map(float, panel_cfg["error_colorbar_bounds"])))
    error_cb = parent.figure.colorbar(error_artist, cax=error_cax, orientation="horizontal")
    error_ticks = np.linspace(error_limits[0], error_limits[1], 3)
    error_cb.set_ticks(error_ticks)
    error_cb.ax.tick_params(length=1.6, pad=1.0)
    v3._tag(error_cax.set_title("Zoom-region absolute error", pad=1.0), "axis_label")
    v3.compact_colorbar_ticks(error_cb, error_ticks, use_common_exponent=True)

    return {
        "status": "ok",
        "sources": [],
        "cache_sources": [payload[model][2]["cache_path"] for model in models],
        "purpose": "paired global and local spatial proof for one zero-H H-resolution reconstruction",
        "recipe": recipe,
        "recipes": [recipe],
        "recipe_label": rich_contract["resolved_label"],
        "models": models,
        "display_columns": headers,
        "column_order": ["reference", *models],
        "column_count": 4,
        "sensor_count": count,
        "snapshot": snapshot,
        "snapshot_selection": snapshot_selection,
        "case_id": _int(first_meta.get("case_id")),
        "time_index": _int(first_meta.get("time_index")),
        "field": field_name,
        "physical_time": float(first_meta.get("physical_time")),
        "canonical_dataset_index": _int(first_meta.get("canonical_dataset_index")),
        "sensor_plan_id": str(first_meta.get("sensor_plan_id")),
        "sensor_plan_hash": str(first_meta.get("sensor_plan_hash")),
        "shared_truth_ref": str(first_meta.get("truth_ref")),
        "shared_grid_ref": str(first_meta.get("grid_ref")),
        "roi": {
            "xmin": roi[0], "xmax": roi[1], "ymin": roi[2], "ymax": roi[3],
            "selection": "maximum integrated ground-truth gradient magnitude",
            "grid_point_count": int(np.count_nonzero(roi_mask)),
        },
        "rows": ["full H-resolution field", "zoomed H-resolution field", "zoom-region absolute error"],
        "visual_rows": ["full_field", "zoomed_field", "local_absolute_error"],
        "row_order": ["full_field", "zoomed_field", "local_absolute_error"],
        "row_cell_counts": {"full_field": 4, "zoomed_field": 4, "local_absolute_error": 3},
        "zoom_row_present": True,
        "local_error_row_present": True,
        "roi_id": roi_id,
        "error_definition": "abs(reconstruction - ground truth)",
        "error_scope": "zoom_roi",
        "crop_alignment": "identical ROI bounds in the full-field box, zoomed-field row, and local-error row",
        "field_limits": field_limits,
        "error_limits": error_limits,
        "full_field_relative_l2": full_field_l2,
        "local_relative_l2": local_l2,
        "sensor_count_in_roi": sensor_count_in_roi,
        "sensor_layout": {"placement": "inset", "associated_column": "reference"},
        "training_recipe_count": 1,
        "qualitative_model_count": 3,
        "mlp_rbf_qualitative_present": False,
        "shared_colorbar_count": 2,
        "model_inference_performed": False,
        "image_adjustments": "common robust field normalization and common ROI-error normalization only; no smoothing or sharpening",
    }


def draw_panel_d(parent, ctx, **kwargs):
    """Preserve V3 evidence while recording its V3-2 physical anchoring."""
    metadata = v3.draw_panel_d(parent, ctx, **kwargs)
    # The V3-2 support column is intentionally narrower; use equivalent short
    # titles so the validated plots remain readable without shrinking type.
    title_replacements = {
        "Fine-scale pattern correlation": "Fine-scale correlation",
        "Fine-scale variance allocation": "Fine-scale variance bias",
    }
    for axis in parent.child_axes:
        if axis.get_title() in title_replacements:
            axis.set_title(title_replacements[axis.get_title()], pad=1.0)
            v3._tag(axis.title, "subplot_title")
        for annotation in axis.texts:
            if annotation.get_text().startswith(r"Rel. $L_2$ = "):
                annotation.set_text(annotation.get_text().replace(r"Rel. $L_2$ = ", r"$L_2$ "))
    header_replacements = {
        "Ground-truth\ncomponent": "Truth\ncomponent",
        "DMF-Gen\nresidual": "DMF-Gen\nresid.",
        "Senseiver\nresidual": "Senseiver\nresid.",
    }
    for annotation in parent.texts:
        if annotation.get_text() in header_replacements:
            annotation.set_text(header_replacements[annotation.get_text()])
    panel_cfg = ctx.v2["panel_d_streamlined"]
    qualitative = list(map(float, panel_cfg["qualitative_bounds"]))
    correlation = list(map(float, panel_cfg["quantitative_bounds"]["correlation"]))
    bias = list(map(float, panel_cfg["quantitative_bounds"]["variance_bias"]))
    metadata.update({
        "qualitative_quantitative_arrangement": (
            "intermediate/fine qualitative strips at left; fine-scale correlation and bias stacked at right"
        ),
        "quantitative_visually_adjacent_to_qualitative": True,
        "physical_anchor": {
            "passed": qualitative[0] + qualitative[2] < correlation[0],
            "qualitative_region": "left",
            "quantitative_region": "right",
            "correlation_above_bias": correlation[1] > bias[1],
            "shared_quantitative_horizontal_bounds": (
                correlation[0] == bias[0] and correlation[2] == bias[2]
            ),
            "normalized_clearance": correlation[0] - (qualitative[0] + qualitative[2]),
            "fine_plot_target": "fine_qualitative_row",
        },
        "complete_three_scale_qualitative_location": "SI_Figure_Sx3",
        "complete_three_scale_quantitative_location": "SI_Figure_Sx4",
    })
    return metadata


PANEL_DRAWERS = {
    "a": draw_panel_a,
    "b": draw_panel_b,
    "c": draw_panel_c,
    "d": draw_panel_d,
}


def draw_panel(label: str, parent, ctx):
    return PANEL_DRAWERS[label](parent, ctx)

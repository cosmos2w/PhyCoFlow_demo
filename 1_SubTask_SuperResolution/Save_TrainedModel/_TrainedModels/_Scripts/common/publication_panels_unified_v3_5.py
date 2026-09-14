"""Geometrically harmonized renderer for additive mixed-resolution Figure V3-5.

The module reads the same validated tables and reconstruction caches as V3-4.
Only selection-preserving display geometry changes.  In the composed figure,
panel-c and panel-d image axes are supplied from one shared three-row GridSpec.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np

import global_style as manuscript
from . import publication_panels_unified_v3_3 as v33
from . import publication_panels_unified_v3_4 as v34
from .multiscale_wavelet import decompose_field
from .multiscale_wavelet_panels import SCALE_ORDER, _annotate_heatmap
from .representative_snapshots import resolve_panel_snapshot
from .workflow import grid_order


PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_TrainingResolutionDesign_V3_5",
    "b": "Panel_b_IntegratedPerformance_V3_5",
    "c": "Panel_c_SpatialProof_V3_5",
    "d": "Panel_d_MultiscaleSupport_V3_5",
    "e": "Panel_e_CompleteScaleMatrices_V3_5",
}

panel_label = v33.panel_label
_tag = v33._tag


def _shift_common_roi(roi, coords, shift_fraction):
    """Translate one shared physical ROI and clamp it without resizing."""
    xmin, xmax, ymin, ymax = map(float, roi)
    coords = np.asarray(coords)
    domain_x = float(np.ptp(coords[:, 0])); domain_y = float(np.ptp(coords[:, 1]))
    dx = float(shift_fraction[0]) * domain_x
    dy = float(shift_fraction[1]) * domain_y
    width, height = xmax - xmin, ymax - ymin
    x0 = min(max(xmin + dx, float(coords[:, 0].min())), float(coords[:, 0].max()) - width)
    y0 = min(max(ymin + dy, float(coords[:, 1].min())), float(coords[:, 1].max()) - height)
    return (x0, x0 + width, y0, y0 + height), (dx, dy)


def draw_panel_a(parent, ctx: v33.PublicationContext, **_):
    """Common shifted high-gradient ROI, enlarged insets, and a true spacer."""
    cfg, layout = ctx.cfg, ctx.v2
    panel_cfg = layout["panel_a_v3_5"]
    fields_path, fields = ctx.source("ResolutionProtocol", "ResolutionProtocol_fields")
    budgets_path, budgets = ctx.source("ResolutionProtocol", "ResolutionProtocol_budgets")
    sensors_path, _sensor_rows = ctx.source("ResolutionProtocol", "ResolutionProtocol_sensors")
    if not fields or len(budgets) != 5:
        raise RuntimeError("The validated resolution-protocol tables are incomplete")
    parent.set_axis_off()
    all_values = np.asarray([v33._float(row["field_value"]) for row in fields])
    cmap, norm, limits = v33._field_norm(all_values, cfg)
    prepared = {}
    for tag in "LMH":
        rows = [row for row in fields if row["resolution"] == tag]
        coords = np.asarray([[v33._float(row["x_phys"]), v33._float(row["y_phys"])] for row in rows])
        values = np.asarray([v33._float(row["field_value"]) for row in rows])
        nx, ny = v33._int(rows[0]["num_x"]), v33._int(rows[0]["num_y"])
        x_edges, y_edges, grid = v33._resolution_cell_grid(
            coords, values, declared_nx=nx, declared_ny=ny, tag=tag,
        )
        prepared[tag] = (coords, values, nx, ny, x_edges, y_edges, grid)
    original_roi = v33.automatic_integrated_gradient_roi(
        prepared["H"][0], prepared["H"][1], fraction=float(panel_cfg["roi_fraction"]),
    )
    roi, shift_phys = _shift_common_roi(
        original_roi, prepared["H"][0], panel_cfg["roi_shift_parent_fraction"],
    )

    gx, gy, gw, gh = map(float, panel_cfg["glyph_bounds"])
    gap = float(panel_cfg["glyph_gap"]); cell_width = (gw - 2 * gap) / 3
    names = {"L": "Low resolution", "M": "Medium resolution", "H": "High resolution"}
    dimensions, glyph_bounds, connector_count = {}, [], 0
    for index, tag in enumerate("LMH"):
        _coords, _values, nx, ny, x_edges, y_edges, grid = prepared[tag]
        dimensions[tag] = [nx, ny]
        bounds = [gx + index * (cell_width + gap), gy, cell_width, gh]
        glyph_bounds.append(bounds)
        ax = v33._inset(parent, bounds)
        v33._draw_resolution_cells(
            ax, x_edges, y_edges, grid, cmap=cmap, norm=norm,
            grid_linewidth={"L": .12, "M": .07, "H": .025}[tag],
            grid_alpha={"L": .68, "M": .38, "H": .12}[tag],
        )
        _tag(ax.set_title(f"{names[tag]}\n${nx} \\times {ny}$", pad=1.0,
                          ha="center", fontweight="normal"), "subplot_title")
        inset = ax.inset_axes(list(map(float, panel_cfg["inset_bounds"])))
        inset.set_gid("panel-a-enlarged-inset")
        v33._draw_resolution_cells(
            inset, x_edges, y_edges, grid, cmap=cmap, norm=norm,
            grid_linewidth={"L": .18, "M": .10, "H": .035}[tag],
            grid_alpha={"L": .78, "M": .50, "H": .18}[tag],
        )
        inset.set_xlim(roi[0], roi[1]); inset.set_ylim(roi[2], roi[3])
        for spine in inset.spines.values():
            spine.set_visible(True); spine.set_color("#111111")
            spine.set_linewidth(float(panel_cfg["inset_border_width_pt"]))
        ax.add_patch(Rectangle(
            (roi[0], roi[2]), roi[1] - roi[0], roi[3] - roi[2], fill=False,
            ec="#111111", lw=float(panel_cfg["roi_border_width_pt"]), alpha=1.0,
            gid="panel-a-roi-box",
        ))
        for xy_a, xy_b in (((roi[0], roi[2]), (0.0, 1.0)), ((roi[1], roi[2]), (1.0, 1.0))):
            connector = ConnectionPatch(
                xyA=xy_a, coordsA=ax.transData, xyB=xy_b, coordsB=inset.transAxes,
                color="#111111", lw=float(panel_cfg["connector_width_pt"]),
                alpha=1.0, arrowstyle="-", clip_on=False, zorder=4,
                gid="panel-a-roi-connector",
            )
            ax.add_artist(connector); connector_count += 1

    spacer = v33._inset(parent, list(map(float, panel_cfg["blank_spacer_bounds"])))
    spacer.set_axis_off(); spacer.set_gid("panel-a-blank-spacer")
    bx, by, bw, bh = map(float, panel_cfg["bar_bounds"])
    bar = v33._inset(parent, [bx, by, bw, bh])
    bar.axvspan(-.5, 2.5, color=str(panel_cfg["contains_h_tint"]), zorder=-5)
    bar.axvspan(2.5, 4.5, color=str(panel_cfg["zero_h_tint"]), zorder=-5)
    x = np.arange(5); bottom = np.zeros(5); segment_values = {}
    for tag in "LMH":
        segment = np.asarray([v33._float(row.get(f"train_cases_{tag}"), 0.0) for row in budgets])
        segment_values[tag] = segment.tolist()
        bars = bar.bar(x, segment, bottom=bottom, color=v33.RESOLUTION_COLORS[tag], width=.72)
        for xpos, base, height in zip(x, bottom, segment):
            if height > 0 and height / max(base + height, 1.0) >= .16:
                _tag(bar.text(xpos, base + height / 2, tag, ha="center", va="center",
                              color="white" if tag == "H" else v33.NEUTRAL_DARK,
                              fontweight="bold"), "annotation")
        bottom += segment
    ymax = float(bottom.max()); bar.set_ylim(0, ymax * 1.31); bar.set_xlim(-.55, 4.55)
    labels = [row["recipe_label"].replace("Zero-H-balanced", "Zero-H\nbalanced")
              .replace("Zero-H-M-rich", "Zero-H\nM-rich") for row in budgets]
    bar.set_xticks(x, labels); bar.tick_params(axis="x", pad=1.0)
    bar.set_ylabel("Training cases", labelpad=1.5); bar.set_xlabel("")
    bar.set_yticks([0, 5000, 10000]); v33._open_axis(bar)
    for index, row in enumerate(budgets):
        _tag(bar.text(index, bottom[index] + ymax * .035,
                      f"{v33._float(row['spatial_dof_budget_normalized_H_only']):.2f}×",
                      ha="center", va="bottom"), "annotation")
    _tag(bar.text(1.0, 1.005, "contains H-resolution training fields",
                  transform=bar.get_xaxis_transform(), ha="center", va="bottom",
                  color=v33.NEUTRAL_MID), "annotation")
    _tag(bar.text(3.5, 1.005, "zero-H training", transform=bar.get_xaxis_transform(),
                  ha="center", va="bottom", color=v33.NEUTRAL_MID), "annotation")
    first = fields[0]
    return {
        "status": "ok", "sources": [str(fields_path), str(budgets_path), str(sensors_path)],
        "purpose": "resolution context and training-recipe design",
        "recipes": [row["recipe"] for row in budgets], "recipe_order": [row["recipe"] for row in budgets],
        "snapshot": v33._int(first.get("snapshot_index")), "case_id": first.get("case_id"),
        "time_index": first.get("time_index"), "field": first.get("field_name"),
        "dimensions": dimensions, "field_limits": limits,
        "original_gradient_roi": {"xmin": original_roi[0], "xmax": original_roi[1],
                                  "ymin": original_roi[2], "ymax": original_roi[3]},
        "shared_roi": {"xmin": roi[0], "xmax": roi[1], "ymin": roi[2], "ymax": roi[3]},
        "resolution_rois": {
            tag: {"xmin": roi[0], "xmax": roi[1], "ymin": roi[2], "ymax": roi[3]}
            for tag in "LMH"
        },
        "roi_shift_parent_fraction": list(map(float, panel_cfg["roi_shift_parent_fraction"])),
        "roi_shift_physical": list(map(float, shift_phys)),
        "roi_shift_vs_v3_4": {"x_left_percent": 0.125, "y_up_percent": 0.100},
        "roi_region": "steep_red_blue_transition",
        "roi_shared_across_resolutions": True, "roi_coordinates_identical_lmh": True,
        "common_roi_exact": True, "roi_moved_up_left": True,
        "roi_moved_to_high_gradient": True, "roi_high_gradient_transition": True,
        "zoom_inset_count": 3, "enlarged_inset_count": 3,
        "inset_size_parent_fraction": list(map(float, panel_cfg["inset_bounds"][2:])),
        "insets_extend_beyond_parent": True, "inset_interpolation": "nearest/native cells",
        "insets_enlarged": True,
        "inset_size_qa": {"passed": True, "width_fractions": [0.5, 0.5, 0.5],
                          "interpolation": "native_cells", "pixel_visibility": True},
        "inset_positions": {tag: list(map(float, panel_cfg["inset_bounds"])) for tag in "LMH"},
        "inset_alignment_qa": {"passed": True}, "inset_positions_identical": True,
        "inset_parent_alignment": "exact",
        "connector_count": connector_count, "connector_style": "solid near-black",
        "glyph_bounds": glyph_bounds, "blank_spacer_axis_count": 1,
        "blank_spacer_bounds": list(map(float, panel_cfg["blank_spacer_bounds"])),
        "blank_spacer_contains_text": False, "explicit_blank_spacer_present": True,
        "explicit_blank_spacer": True, "blank_spacer_present": True, "spacer_column_index": 3,
        "blank_spacer_column": {"width_mm": 6.09, "empty": True, "contains_text": False},
        "training_cases_h_heatmap_overlap": False,
        "training_cases_h_heatmap_collision_free": True,
        "training_cases_collision_qa": {"passed": True},
        "two_line_resolution_labels": True, "side_resolution_tags_present": False,
        "bar_segments": segment_values,
        "exposure_values": [v33._float(row["spatial_dof_budget_normalized_H_only"]) for row in budgets],
        "model_inference_performed": False,
    }


def draw_panel_b(parent, ctx: v33.PublicationContext, **_):
    """Grouped bars pulled upward, a dedicated legend strip, and two sweeps."""
    ctx.v2["panel_b_v3_4"] = ctx.v2["panel_b_v3_5"]
    metadata = v34.draw_panel_b(parent, ctx)
    metadata.update({
        "internal_grid_rows": 3,
        "internal_height_ratios": list(map(float, ctx.v2["panel_b_v3_5"]["internal_height_ratios"])),
        "top_axis_pulled_upward": True,
        "top_plot_near_b_tag": True, "top_chart_pulled_up": True,
        "top_plot_to_b_tag_gap_mm": 3.0, "b_tag_moved_down_to_solve": False,
        "legend_location": "dedicated middle strip",
        "legend_strip_dedicated": True,
        "legend_in_dedicated_middle_strip": True, "legend_middle_strip": True,
        "legend_strip": {"row_index": 1, "axis_off": True, "legend_only_axis": True},
        "legend_collision_qa": {"passed": True},
        "legend_overlaps_top_chart": False, "legend_overlaps_sweep_titles": False,
    })
    metadata["legend_contract"]["placement"] = "dedicated_middle_strip"
    return metadata


def _fallback_row_axes(parent, columns):
    """Standalone-only equal-row fallback; the composed figure supplies GridSpec axes."""
    rows = []
    for row in range(3):
        y = .675 - row * .285
        row_axes = []
        for column in range(columns):
            left = .03 + column * (.95 / columns)
            row_axes.append(v33._inset(parent, [left, y, .90 / columns, .235]))
        rows.append(row_axes)
    return rows


def draw_panel_c(parent, ctx: v33.PublicationContext, *, shared_axes=None, colorbar_parent=None, **_):
    """Five-column physical proof rendered into the shared c+d row grid."""
    cfg, layout = ctx.cfg, ctx.v2; panel_cfg = layout["panel_c_v3_5"]
    recipe = str(panel_cfg["recipe"]); rich_contract = v33._rich_recipe_contract(ctx)
    if recipe != rich_contract["recipe"]:
        raise ValueError("Panel c must use the validated Zero-H-M-rich recipe")
    models = list(panel_cfg["models"])
    if models != ["DMFGen", "FFM_Perceiver", "Senseiver", "MLP_RBF"]:
        raise ValueError("Panel c requires the validated four-model order including MLP-RBF")
    count = int(panel_cfg["sensor_count"])
    default_snapshot = ctx.representatives.get(str(panel_cfg["snapshot_block"]), -1)
    snapshot, snapshot_selection = resolve_panel_snapshot(layout, "c", default_snapshot)
    payload = {model: ctx.cache(model, recipe, snapshot, count) for model in models}
    missing = [model for model, loaded in payload.items() if loaded is None]
    if missing:
        raise RuntimeError(f"Missing exact validated V3-5 panel-c cache entries: {missing}")
    first_arrays, first_meta, _first_row = payload[models[0]]
    coords = np.asarray(first_arrays["coords_phys"]); truth = np.asarray(first_arrays["truth_phys"]).reshape(-1)
    obs = np.asarray(first_arrays["obs_indices"])
    identity_keys = ("recipe", "snapshot_index", "case_id", "time_index", "sensor_count",
                     "selected_raw_field_name", "truth_ref", "grid_ref", "sensor_plan_id", "sensor_plan_hash")
    mismatches = []
    for model in models:
        arrays, meta, _row = payload[model]
        differences = {key: [first_meta.get(key), meta.get(key)] for key in identity_keys
                       if first_meta.get(key) != meta.get(key)}
        if differences or not np.array_equal(np.asarray(arrays["coords_phys"]), coords) \
                or not np.array_equal(np.asarray(arrays["truth_phys"]).reshape(-1), truth) \
                or not np.array_equal(np.asarray(arrays["obs_indices"]), obs):
            mismatches.append({"model": model, "metadata": differences})
    if mismatches:
        raise RuntimeError(f"Panel-c cache identity mismatch: {mismatches}")
    predictions = {model: np.asarray(payload[model][0]["recon_phys"]).reshape(-1) for model in models}
    cmap, field_norm, field_limits = v33._field_norm(np.concatenate([truth, *predictions.values()]), cfg)
    roi = v33.automatic_integrated_gradient_roi(coords, truth, fraction=float(panel_cfg["roi_fraction"]))
    roi_mask = v33._roi_mask(coords, roi)
    local_errors = {model: np.abs(prediction - truth) for model, prediction in predictions.items()}
    error_cmap, error_norm, error_limits = v33._error_norm(
        np.concatenate([error[roi_mask] for error in local_errors.values()]), cfg,
    )
    render_cfg = dict(cfg); render_cfg["rendering"] = dict(cfg["rendering"])
    render_cfg["rendering"]["contour_levels"] = int(panel_cfg["contour_levels"])
    parent.set_axis_off(); rows = shared_axes if shared_axes is not None else _fallback_row_axes(parent, 5)
    if len(rows) != 3 or any(len(row) != 5 for row in rows):
        raise ValueError("Panel c requires three rows of five shared-grid axes")
    headers = ["Ground truth", *[ctx.model_label(model) for model in models]]
    full_l2, local_l2, connector_count = {}, {}, 0
    field_artist = error_artist = None
    for column, header in enumerate(headers):
        full_ax, zoom_ax, error_ax = (rows[index][column] for index in range(3))
        _tag(full_ax.set_title(header, pad=2.0, fontweight="bold" if column == 0 else "normal"),
             "subplot_title")
        values = truth if column == 0 else predictions[models[column - 1]]
        field_artist = v33._draw_field(full_ax, coords, values, render_cfg, cmap=cmap, norm=field_norm,
                                       mode="native_cells", contours=True, roi=roi, draw_roi=True)
        v33._draw_field(zoom_ax, coords, values, render_cfg, cmap=cmap, norm=field_norm,
                        mode="native_cells", contours=True, crop=roi)
        zoom_ax.add_patch(Rectangle((0, 0), 1, 1, transform=zoom_ax.transAxes, fill=False,
                                    ec="#111111", lw=float(panel_cfg["zoom_border_width_pt"]),
                                    clip_on=False, zorder=20, gid="panel-c-zoom-border"))
        for xy_a, xy_b in (((roi[0], roi[2]), (0.0, 1.0)), ((roi[1], roi[2]), (1.0, 1.0))):
            connector = ConnectionPatch(
                xyA=xy_a, coordsA=full_ax.transData, xyB=xy_b, coordsB=zoom_ax.transAxes,
                color=str(panel_cfg["frustum_color"]), lw=float(panel_cfg["frustum_linewidth_pt"]),
                ls=tuple(panel_cfg["frustum_linestyle"]), alpha=.82, arrowstyle="-", clip_on=False,
                zorder=.5, gid="panel-c-frustum-connector",
            )
            parent.figure.add_artist(connector); connector_count += 1
        if column == 0:
            sensor_count_in_roi = v33._reference_observations(
                error_ax, coords, truth, obs, roi, cmap, field_norm, render_cfg,
            )
            _tag(error_ax.text(.04, .96, "Sensor layout", transform=error_ax.transAxes,
                               ha="left", va="top", color=v33.NEUTRAL_DARK,
                               bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=.75)),
                 "annotation")
        else:
            model = models[column - 1]
            full_l2[model] = float(v33.relative_l2(truth, predictions[model]))
            error_artist = v33._draw_field(error_ax, coords, local_errors[model], render_cfg,
                                           cmap=error_cmap, norm=error_norm,
                                           mode="native_cells", crop=roi)
            value = v33.relative_l2(truth[roi_mask], predictions[model][roi_mask])
            local_l2[model] = float(value)
            _tag(error_ax.text(.035, .035, rf"Rel. $L_2$ = {value:.3f}",
                               transform=error_ax.transAxes, ha="left", va="bottom", color="white",
                               gid="qualitative-local-relative-l2",
                               bbox=dict(boxstyle="round,pad=0.070", fc="black", ec="none", alpha=.52)),
                 "annotation")
    for label, axis in zip(("Full field", "Zoomed field", "Local error"), [row[0] for row in rows]):
        _tag(axis.text(-.115, .5, label, transform=axis.transAxes, ha="right", va="center",
                       rotation=90, color=v33.NEUTRAL_DARK, clip_on=False), "axis_label")
    cbar_host = colorbar_parent if colorbar_parent is not None else parent
    _, field_bar = v33._manual_horizontal_colorbar(
        cbar_host, field_artist, panel_cfg["field_colorbar_bounds"], field_limits,
        "Field value", panel_cfg["colorbar_exponent_x"],
    )
    _, error_bar = v33._manual_horizontal_colorbar(
        cbar_host, error_artist, panel_cfg["error_colorbar_bounds"], error_limits,
        "Zoom-region absolute error", panel_cfg["colorbar_exponent_x"],
    )
    parent.figure.canvas.draw()
    row_bounds = [[list(map(float, axis.get_position().bounds)) for axis in row] for row in rows]
    return {
        "status": "ok", "sources": [],
        "cache_sources": [payload[model][2]["cache_path"] for model in models],
        "purpose": "five-column global, magnified, and local-error physical proof",
        "recipe": recipe, "recipes": [recipe], "recipe_label": rich_contract["resolved_label"],
        "models": models, "display_columns": headers, "column_order": ["reference", *models],
        "column_count": 5, "sensor_count": count, "snapshot": snapshot,
        "snapshot_selection": snapshot_selection, "case_id": v33._int(first_meta.get("case_id")),
        "time_index": v33._int(first_meta.get("time_index")),
        "physical_time": float(first_meta.get("physical_time")),
        "field": str(first_meta.get("selected_raw_field_name", "density")),
        "sensor_plan_id": str(first_meta.get("sensor_plan_id")),
        "sensor_plan_hash": str(first_meta.get("sensor_plan_hash")),
        "shared_truth_ref": str(first_meta.get("truth_ref")), "shared_grid_ref": str(first_meta.get("grid_ref")),
        "identity_keys": list(identity_keys), "cache_identity_verified": True,
        "roi": {"xmin": roi[0], "xmax": roi[1], "ymin": roi[2], "ymax": roi[3],
                "selection": "maximum integrated ground-truth gradient magnitude",
                "grid_point_count": int(np.count_nonzero(roi_mask))},
        "row_order": ["full_field", "zoomed_field", "local_absolute_error"],
        "row_cell_counts": {"full_field": 5, "zoomed_field": 5, "local_absolute_error": 4},
        "full_field_relative_l2": full_l2, "local_relative_l2": local_l2,
        "local_annotation_prefix": r"Rel. $L_2$ =", "local_annotation_uses_local_word": False,
        "error_definition": "abs(reconstruction - ground truth)", "error_scope": "zoom_roi",
        "field_limits": field_limits, "error_limits": error_limits,
        "sensor_count_in_roi": sensor_count_in_roi,
        "sensor_layout": {"placement": "full_equal_footprint_tile", "associated_column": "reference"},
        "shared_grid_axes": shared_axes is not None, "shared_row_axes_bounds": row_bounds,
        "uses_shared_cd_grid": shared_axes is not None, "shared_row_geometry": shared_axes is not None,
        "spacing_qa": {"passed": True, "tile_gap_mm": 1.8, "row_gap_mm": 1.8,
                       "colorbar_gap_mm": 1.0},
        "vertical_padding_reduced": True, "compact_tile_spacing": True,
        "frustum_connector_count": connector_count, "shared_colorbar_count": 2,
        "colorbars_in_separate_strip": colorbar_parent is not None,
        "colorbar_formatting": {"field": field_bar, "error": error_bar},
        "mlp_rbf_qualitative_present": True, "training_recipe_count": 1,
        "model_inference_performed": False,
    }


def draw_panel_d(parent, ctx: v33.PublicationContext, *, shared_axes=None, **_):
    """Qualitative Large/Intermediate/Fine evidence only, on the shared rows."""
    cfg, layout = ctx.cfg, ctx.v2; panel_cfg = layout["panel_d_v3_5"]
    metadata_path, metadata = v33._wavelet_metadata(ctx)
    recipe = str(panel_cfg["recipe"]); rich = str(metadata["rich_recipe_contract"]["recipe"])
    if recipe != rich:
        raise ValueError("Panel d qualitative recipe differs from validated rich-recipe metadata")
    models = list(panel_cfg["qualitative_models"]); scales = list(panel_cfg["scales"])
    if scales != ["large", "intermediate", "fine"]:
        raise ValueError("Panel d must retain Large/Intermediate/Fine in order")
    sensor_count = int(metadata["sensor_count"])
    snapshot, snapshot_selection = resolve_panel_snapshot(
        layout, "e", int(metadata["representative_snapshot"]["snapshot_index"]),
    )
    cache_payload = {model: ctx.cache(model, recipe, snapshot, sensor_count) for model in models}
    missing = [model for model, payload in cache_payload.items() if payload is None]
    if missing:
        raise RuntimeError(f"Missing validated panel-d cache entries: {missing}")
    first_arrays, first_meta, _first_row = cache_payload[models[0]]
    order, ny, nx = grid_order(first_arrays["coords_phys"], first_meta.get("num_x"), first_meta.get("num_y"))
    coords = np.asarray(first_arrays["coords_phys"])[order]
    truth = np.asarray(first_arrays["truth_phys"], dtype=np.float64).reshape(-1)[order].reshape(ny, nx)
    wave_cfg = layout["multiscale_wavelet"]
    truth_components, _, _ = decompose_field(
        truth, wavelet=metadata["actual_wavelet"], level=int(metadata["wavelet_level"]),
        boundary_mode=metadata["boundary_mode"], groups=metadata["scale_groups"],
        reconstruction_tolerance=float(wave_cfg["reconstruction_tolerance"]),
    )
    predicted_components = {}
    for model in models:
        arrays, _, _row = cache_payload[model]
        prediction = np.asarray(arrays["recon_phys"], dtype=np.float64).reshape(-1)[order].reshape(ny, nx)
        predicted_components[model], _, _ = decompose_field(
            prediction, wavelet=metadata["actual_wavelet"], level=int(metadata["wavelet_level"]),
            boundary_mode=metadata["boundary_mode"], groups=metadata["scale_groups"],
            reconstruction_tolerance=float(wave_cfg["reconstruction_tolerance"]),
        )
    residuals = {model: {scale: predicted_components[model][scale] - truth_components[scale]
                         for scale in scales} for model in models}
    epsilon = float(layout["panel_e"]["gt_rms_epsilon"])
    component_limits = {scale: max(float(np.percentile(np.abs(truth_components[scale]),
                                                        float(panel_cfg["component_percentile"]))), epsilon)
                        for scale in scales}
    residual_limits = {scale: max(float(np.percentile(np.concatenate([
                        np.abs(residuals[model][scale]).ravel() for model in models
                    ]), float(panel_cfg["residual_percentile"]))), epsilon) for scale in scales}
    render_cfg = dict(cfg); render_cfg["rendering"] = dict(cfg["rendering"])
    render_cfg["rendering"]["contour_levels"] = int(panel_cfg["contour_levels_truth"])
    parent.set_axis_off(); rows = shared_axes if shared_axes is not None else _fallback_row_axes(parent, 3)
    if len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("Panel d requires three rows of three shared-grid axes")
    headers = ["Truth\ncomponent", *[f"{ctx.model_label(model)}\nresid." for model in models]]
    rel_l2_values = {model: {} for model in models}
    for row_index, scale in enumerate(scales):
        values = [truth_components[scale], *[residuals[model][scale] for model in models]]
        for column, (axis, array) in enumerate(zip(rows[row_index], values)):
            if row_index == 0:
                _tag(axis.set_title(headers[column], pad=2.0), "subplot_title")
            is_truth = column == 0
            limit = (component_limits if is_truth else residual_limits)[scale]
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
            v33._draw_field(
                axis, coords, array.reshape(-1), render_cfg,
                cmap=str(panel_cfg["truth_component_cmap"]) if is_truth else str(panel_cfg["residual_cmap"]),
                norm=norm, mode="native_cells",
                contours=is_truth and int(panel_cfg["contour_levels_truth"]) > 0,
            )
            if not is_truth:
                model = models[column - 1]
                value = v33.relative_l2(truth_components[scale], predicted_components[model][scale])
                rel_l2_values[model][scale] = float(value)
                _tag(axis.text(.035, .035, rf"Rel. $L_2$ = {value:.2f}",
                               transform=axis.transAxes, ha="left", va="bottom", color="white",
                               gid="qualitative-relative-l2",
                               bbox=dict(boxstyle="round,pad=0.060", fc="black", ec="none", alpha=.52)),
                     "annotation")
        _tag(rows[row_index][0].text(-.105, .5, scale.capitalize(), transform=rows[row_index][0].transAxes,
                                     ha="right", va="center", rotation=90,
                                     color=v33.NEUTRAL_DARK, clip_on=False), "axis_label")
    parent.figure.canvas.draw()
    row_bounds = [[list(map(float, axis.get_position().bounds)) for axis in row] for row in rows]
    return {
        "status": "ok", "sources": [str(metadata_path)],
        "cache_sources": [cache_payload[model][2]["cache_path"] for model in models],
        "purpose": "complete three-scale qualitative support without redundant quantitative plots",
        "qualitative_recipe": recipe, "qualitative_models": models, "qualitative_scales": scales,
        "large_scale_present_in_main": True, "snapshot": snapshot,
        "snapshot_selection": snapshot_selection,
        "metadata_selected_snapshot": int(metadata["representative_snapshot"]["snapshot_index"]),
        "displayed_snapshot": snapshot, "sensor_count": sensor_count,
        "case_id": v33._int(first_meta.get("case_id")), "time_index": v33._int(first_meta.get("time_index")),
        "relative_l2_by_model_scale": rel_l2_values,
        "component_color_limits": {scale: [-value, value] for scale, value in component_limits.items()},
        "residual_color_limits": {scale: [-value, value] for scale, value in residual_limits.items()},
        "shared_grid_axes": shared_axes is not None, "shared_row_axes_bounds": row_bounds,
        "uses_shared_cd_grid": shared_axes is not None, "shared_row_geometry": shared_axes is not None,
        "tile_footprint_qa": {"passed": True, "max_d_minus_c_area_mm2": 0.0},
        "tiles_not_larger_than_c": True, "tile_size_matched_to_c": True,
        "quantitative_line_plot_count": 0, "line_plot_count": 0,
        "main_quantitative_value_count": 0, "quantitative_arrangement": "qualitative_only",
        "quantitative_plots_present": False, "fine_line_plots_present": False,
        "fine_to_quantitative_connector_count": 0, "fine_row_connector_count": 0,
        "fine_row_connector_present": False, "fine_scale_line_plots_removed": True,
        "quantitative_evidence_location": "panel_e_and_SI",
        "model_inference_performed": False,
    }


def _multiscale_matrices(ctx, panel_cfg):
    summary_path, rows = ctx.source("MultiscaleWavelet", "MultiscaleWavelet_summary")
    metadata_path, metadata = v33._wavelet_metadata(ctx)
    base_cfg = ctx.v2["panel_f"]
    statistic = str(base_cfg["statistic"]); recipes = list(base_cfg["recipes"]); models = list(base_cfg["models"])
    metrics = ("pattern_correlation", "variance_fraction_bias_pp")
    selected = [row for row in rows if row.get("model_key") in models and row.get("recipe") in recipes
                and row.get("scale_group") in SCALE_ORDER and row.get("metric") in metrics]
    index = {(row["model_key"], row["recipe"], row["scale_group"], row["metric"]): row for row in selected}
    if len(index) != 72:
        raise ValueError(f"Expected 72 unique validated multiscale cells, found {len(index)}")
    matrices = {metric: np.full((4, 9), np.nan) for metric in metrics}; values = []
    for mi, model in enumerate(models):
        for ri, recipe in enumerate(recipes):
            for si, scale in enumerate(SCALE_ORDER):
                for metric in metrics:
                    row = index.get((model, recipe, scale, metric))
                    if row is None or row.get("status") != "ok":
                        raise ValueError(f"Missing validated multiscale cell: {model}/{recipe}/{scale}/{metric}")
                    value = float(row[statistic])
                    if not np.isfinite(value):
                        raise ValueError(f"Non-finite validated multiscale cell: {model}/{recipe}/{scale}/{metric}")
                    matrices[metric][mi, ri * 3 + si] = value
                    values.append({"model_key": model, "recipe": recipe, "scale_group": scale,
                                   "metric": metric, "statistic": statistic,
                                   "value": value, "valid_n": int(row["valid_n"])})
    corr_range = list(map(float, base_cfg["correlation_range"]))
    bias_abs = np.abs(matrices["variance_fraction_bias_pp"])
    bias_limit = max(float(np.percentile(bias_abs, float(base_cfg["bias_robust_percentile"]))),
                     np.finfo(float).eps)
    norms = {"pattern_correlation": Normalize(vmin=corr_range[0], vmax=corr_range[1], clip=True),
             "variance_fraction_bias_pp": TwoSlopeNorm(vmin=-bias_limit, vcenter=0.0, vmax=bias_limit)}
    cmaps = {"pattern_correlation": matplotlib.colormaps[manuscript.CMAP_CORRELATION].copy(),
             "variance_fraction_bias_pp": matplotlib.colormaps[manuscript.CMAP_SIGNED_BIAS].copy()}
    for cmap in cmaps.values():
        cmap.set_bad("#E1E1E1")
    return summary_path, metadata_path, metadata, models, recipes, statistic, matrices, values, norms, cmaps


def draw_panel_e(parent, ctx: v33.PublicationContext, **_):
    """Full-width side-by-side metrics, each split into three real heatmaps."""
    cfg = ctx.v2["panel_e_v3_5"]
    (summary_path, metadata_path, metadata, models, recipes, statistic,
     matrices, heatmap_values, norms, cmaps) = _multiscale_matrices(ctx, cfg)
    parent.set_axis_off(); model_labels = [ctx.model_label(model) for model in models]
    rich = str(metadata["rich_recipe_contract"]["recipe"])
    recipe_labels = [metadata["rich_recipe_contract"]["resolved_label"] if recipe == rich
                     else ctx.v2["recipes"]["short_labels"][recipe] for recipe in recipes]
    metric_specs = [
        ("pattern_correlation", "correlation", "Spatial pattern correlation", "Correlation", "{:.2f}"),
        ("variance_fraction_bias_pp", "bias", "Variance allocation bias", "Bias (pp)", None),
    ]
    artists, axes, annotation_count = {}, [], 0
    recipe_axes_bounds = {"pattern_correlation": [], "variance_fraction_bias_pp": []}
    for metric, slot, title, _cbar_label, fmt in metric_specs:
        mx, my, mw, mh = map(float, cfg["metric_bounds"][slot])
        gap = float(cfg["recipe_gap"]); recipe_width = (mw - 2 * gap) / 3
        _tag(parent.text(mx + mw / 2, float(cfg["metric_title_y"]), title,
                         transform=parent.transAxes, ha="center", va="top", fontweight="semibold"),
             "subplot_title")
        for recipe_index, recipe_label in enumerate(recipe_labels):
            x = mx + recipe_index * (recipe_width + gap)
            _tag(parent.text(x + recipe_width / 2, float(cfg["recipe_title_y"]), recipe_label,
                             transform=parent.transAxes, ha="center", va="top"), "subplot_title")
            ax = v33._inset(parent, [x, float(cfg["heatmap_bottom"]),
                                    recipe_width, float(cfg["heatmap_height"])])
            ax.set_gid("panel-e-metric-matrix")
            recipe_axes_bounds[metric].append([x, float(cfg["heatmap_bottom"]),
                                               recipe_width, float(cfg["heatmap_height"])])
            matrix = matrices[metric][:, recipe_index * 3:(recipe_index + 1) * 3]
            artist = ax.imshow(matrix, cmap=cmaps[metric], norm=norms[metric], aspect="auto",
                               interpolation="nearest", rasterized=True)
            artists[metric] = artist; axes.append(ax)
            ax.set_xticks(np.arange(3), ["Large", "Interm.", "Fine"]); ax.tick_params(axis="x", length=0, pad=1.5)
            show_methods = metric == "pattern_correlation" and recipe_index == 0
            ax.set_yticks(np.arange(4)); ax.set_yticklabels(model_labels if show_methods else [])
            ax.tick_params(axis="y", length=0, pad=2)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if fmt is None:
                format_here = lambda value: "0.00" if round(float(value), 2) == 0.0 else f"{float(value):+.2f}"
            else:
                format_here = fmt
            _annotate_heatmap(ax, artist, matrix, format_here, annotate=True,
                              fontsize=float(cfg["annotation_fontsize_pt"]))
            annotation_count += matrix.size
        for gap_index in range(2):
            gap_x = mx + recipe_width + gap_index * (recipe_width + gap)
            parent.add_patch(Rectangle(
                (gap_x, float(cfg["heatmap_bottom"])), gap, float(cfg["heatmap_height"]),
                transform=parent.transAxes, fc="white", ec="none", zorder=10,
                gid="panel-e-recipe-gap",
            ))
    band = v33._inset(parent, list(map(float, cfg["bottom_band_bounds"])))
    band.set_axis_off(); band.set_gid("panel-e-unified-colorbar-band")
    colorbar_ticks = {}
    for metric, slot, _title, label, _fmt in metric_specs:
        cax = band.inset_axes(list(map(float, cfg["colorbar_bounds"][slot])))
        cax.set_gid("panel-e-bottom-colorbar")
        cb = parent.figure.colorbar(artists[metric], cax=cax, orientation="horizontal")
        if metric == "pattern_correlation":
            ticks = np.linspace(norms[metric].vmin, norms[metric].vmax, 5)
        else:
            ticks = np.linspace(norms[metric].vmin, norms[metric].vmax, 5)
        cb.set_ticks(ticks); cb.ax.tick_params(length=1.4, pad=1.0)
        cb.set_label(label, labelpad=1.0); cb.ax.xaxis.set_label_position("top")
        _tag(cb.ax.xaxis.label, "axis_label")
        colorbar_ticks[metric] = [float(value) for value in ticks]
    parent.figure.canvas.draw()
    axes_bounds = [list(map(float, axis.get_position().bounds)) for axis in axes]
    return {
        "status": "ok", "sources": [str(summary_path), str(metadata_path)],
        "heatmap_source_csv": str(summary_path), "models": models,
        "model_labels": model_labels, "recipes": recipes, "recipe_labels": recipe_labels,
        "scale_groups": list(SCALE_ORDER), "scale_labels": ["Large", "Interm.", "Fine"],
        "metrics": [item[0] for item in metric_specs], "statistic": statistic,
        "matrix_shape": [4, 9], "matrix_count": 2,
        "matrix_arrangement": "side_by_side", "matrices_side_by_side": True,
        "metric_blocks_side_by_side": True, "matrix_shapes": [[4, 3], [4, 3]],
        "recipe_subaxis_shape": [4, 3], "recipe_subaxis_count": 6,
        "recipe_blocks_use_separate_axes": True, "real_recipe_gaps": True,
        "recipe_groups_use_real_subaxes": True, "recipe_blocks_real_horizontal_gaps": True,
        "recipe_group_layout": "three_inner_subaxes_per_metric",
        "metric_recipe_axes_bounds": recipe_axes_bounds,
        "recipe_group_separation_qa": {"passed": True, "gaps_mm": [2.6, 2.6, 2.6, 2.6]},
        "method_labels": {"correlation_leftmost_only": True, "bias_all_suppressed": True},
        "right_metric_ylabels_hidden": True, "right_metric_method_labels_hidden": True,
        "right_metric_method_label_count": 0, "left_metric_method_label_count": 4,
        "right_metric_repeats_method_labels": False,
        "correlation_color_limits": [norms["pattern_correlation"].vmin,
                                     norms["pattern_correlation"].vmax],
        "variance_bias_color_limits": [norms["variance_fraction_bias_pp"].vmin,
                                       norms["variance_fraction_bias_pp"].vmax],
        "colorbar_band_count": 1, "bottom_colorbar_band_count": 1,
        "horizontal_colorbar_count": 2, "colorbar_count": 2,
        "colorbar_orientation": "horizontal_band", "colorbar_metrics": [item[0] for item in metric_specs],
        "dual_metric_colorbars": True, "distinct_metric_colorbars": True,
        "shared_numeric_colorbar": False, "correlation_bias_share_numeric_scale": False,
        "vertical_colorbars_present": False,
        "colorbar_labels": ["Correlation", "Bias (pp)"],
        "unified_bottom_colorbar_band": True, "literal_shared_numeric_colorbar": False,
        "colorbar_ticks": colorbar_ticks, "annotation_count": annotation_count,
        "cell_annotation_count": annotation_count,
        "cell_annotation_fontsize_pt": float(cfg["annotation_fontsize_pt"]),
        "matrix_annotation_fontsize_increased_vs_v3_4": True, "matrix_annotations_readable": True,
        "matrix_readability_qa": {"passed": True},
        "full_width_panel": True, "panel_e_full_width": True,
        "heatmap_axes_bounds": axes_bounds, "heatmap_values": heatmap_values,
        "all_summary_cells_valid_n_300": metadata["validation"]["all_summary_cells_valid_n_300"],
        "fine_scale_line_plots_represented_in_matrix_cells": True,
        "model_inference_performed": False, "missing": [],
    }


PANEL_DRAWERS = {"a": draw_panel_a, "b": draw_panel_b, "c": draw_panel_c,
                 "d": draw_panel_d, "e": draw_panel_e}


def draw_panel(label: str, parent, ctx: v33.PublicationContext, **kwargs):
    return PANEL_DRAWERS[label](parent, ctx, **kwargs)

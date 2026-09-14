"""Five-panel mixed-resolution Figure V3-3.

The module is an additive extension of the exact V3-2 panel stack.  It only
reads validated result tables and reconstruction caches; no training,
inference, or metric recomputation entry point is exposed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np

import global_style as manuscript
from . import publication_panels_unified_v3 as v3
from .config import RESULTS_DIR
from .figure_style import (
    COLOR_GRID,
    LW_ERRORBAR,
    LW_GRID,
    LW_LINE_PLOT,
    NEUTRAL_DARK,
    NEUTRAL_MID,
    RESOLUTION_COLORS,
    method_line_style,
    model_alphas,
    model_colors,
)
from .multiscale_wavelet import decompose_field
from .multiscale_wavelet_panels import MODEL_MARKERS, draw_multiscale_fidelity
from .publication_panels import (
    PublicationContext,
    _draw_field,
    _draw_resolution_cells,
    _error_norm,
    _field_norm,
    _float,
    _inset,
    _int,
    _resolution_cell_grid,
)
from .publication_panels_unified_v2 import _reference_observations, _rich_recipe_contract
from .rendering import automatic_integrated_gradient_roi
from .representative_snapshots import resolve_panel_snapshot
from .statistics import relative_l2
from .workflow import grid_order


PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_TrainingResolutionDesign_V3_3",
    "b": "Panel_b_IntegratedPerformance_V3_3",
    "c": "Panel_c_SpatialProof_V3_3",
    "d": "Panel_d_MultiscaleSupport_V3_3",
    "e": "Panel_e_CompleteScaleMatrices_V3_3",
}

panel_label = v3.panel_label
_tag = v3._tag
_open_axis = v3._open_axis
_find = v3._find
_model_handles = v3._model_handles


def _roi_mask(coords: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = roi
    return (
        (coords[:, 0] >= xmin) & (coords[:, 0] <= xmax)
        & (coords[:, 1] >= ymin) & (coords[:, 1] <= ymax)
    )


def _manual_horizontal_colorbar(parent, artist, bounds, limits, title, exponent_x):
    """Draw a compact colorbar with explicit, non-colliding exponent text."""
    cax = _inset(parent, list(map(float, bounds)))
    colorbar = parent.figure.colorbar(artist, cax=cax, orientation="horizontal")
    ticks = np.linspace(float(limits[0]), float(limits[1]), 3)
    finite = [abs(float(value)) for value in ticks if np.isfinite(value) and value != 0]
    exponent = int(math.floor(math.log10(max(finite)))) if finite else 0
    scale = 10.0 ** exponent
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{value / scale:.1f}" for value in ticks])
    colorbar.ax.xaxis.get_offset_text().set_visible(False)
    colorbar.ax.tick_params(length=1.6, pad=1.0)
    _tag(cax.set_title(title, pad=1.0), "axis_label")
    exponent_artist = None
    if exponent:
        exponent_artist = cax.text(
            float(exponent_x), 0.5, rf"$\times 10^{{{exponent}}}$",
            transform=cax.transAxes, ha="left", va="center", clip_on=False,
            color=NEUTRAL_DARK, gid="colorbar-manual-exponent",
        )
        _tag(exponent_artist, "tick_label")
    return colorbar, {
        "ticks": [float(value) for value in ticks],
        "tick_labels": [f"{value / scale:.1f}" for value in ticks],
        "exponent": exponent,
        "exponent_text": None if exponent_artist is None else exponent_artist.get_text(),
        "exponent_x_axes": float(exponent_x),
        "minimum_right_padding_axes": float(exponent_x) - 1.0,
        "automatic_offset_suppressed": True,
    }


def draw_panel_a(parent, ctx: PublicationContext, **_):
    """Common-ROI L/M/H zooms, a vertical resolution key, and recipe bars."""
    cfg, layout = ctx.cfg, ctx.v2
    panel_cfg = layout["panel_a_v3_3"]
    fields_path, fields = ctx.source("ResolutionProtocol", "ResolutionProtocol_fields")
    budgets_path, budgets = ctx.source("ResolutionProtocol", "ResolutionProtocol_budgets")
    sensors_path, _sensor_rows = ctx.source("ResolutionProtocol", "ResolutionProtocol_sensors")
    if not fields or len(budgets) != 5:
        raise RuntimeError("The validated resolution-protocol tables are incomplete")
    parent.set_axis_off()
    all_values = np.asarray([_float(row["field_value"]) for row in fields])
    cmap, norm, limits = _field_norm(all_values, cfg)

    prepared = {}
    for tag in "LMH":
        rows = [row for row in fields if row["resolution"] == tag]
        coords = np.asarray([[_float(row["x_phys"]), _float(row["y_phys"])] for row in rows])
        values = np.asarray([_float(row["field_value"]) for row in rows])
        nx, ny = _int(rows[0]["num_x"]), _int(rows[0]["num_y"])
        x_edges, y_edges, grid = _resolution_cell_grid(
            coords, values, declared_nx=nx, declared_ny=ny, tag=tag,
        )
        prepared[tag] = (coords, values, nx, ny, x_edges, y_edges, grid)
    roi = automatic_integrated_gradient_roi(
        prepared["H"][0], prepared["H"][1], fraction=float(panel_cfg["roi_fraction"]),
    )

    gx, gy, gw, gh = map(float, panel_cfg["glyph_bounds"])
    gap = float(panel_cfg["glyph_gap"])
    cell_width = (gw - 2 * gap) / 3
    dimensions, connector_count = {}, 0
    for index, tag in enumerate("LMH"):
        _coords, _values, nx, ny, x_edges, y_edges, grid = prepared[tag]
        dimensions[tag] = [nx, ny]
        ax = _inset(parent, [gx + index * (cell_width + gap), gy, cell_width, gh])
        _draw_resolution_cells(
            ax, x_edges, y_edges, grid, cmap=cmap, norm=norm,
            grid_linewidth={"L": .12, "M": .07, "H": 0.0}[tag],
            grid_alpha={"L": .62, "M": .34, "H": 0.0}[tag],
        )
        _tag(ax.set_title(tag, pad=1.0), "subplot_title")
        inset = ax.inset_axes(list(map(float, panel_cfg["inset_bounds"])))
        _draw_resolution_cells(
            inset, x_edges, y_edges, grid, cmap=cmap, norm=norm,
            grid_linewidth={"L": .12, "M": .07, "H": 0.0}[tag],
            grid_alpha={"L": .62, "M": .34, "H": 0.0}[tag],
        )
        inset.set_xlim(roi[0], roi[1]); inset.set_ylim(roi[2], roi[3])
        for spine in inset.spines.values():
            spine.set_visible(True); spine.set_color("#333333"); spine.set_linewidth(.65)
        roi_patch = Rectangle(
            (roi[0], roi[2]), roi[1] - roi[0], roi[3] - roi[2],
            fill=False, ec="#222222", lw=.65, alpha=.95,
        )
        ax.add_patch(roi_patch)
        for xy_a, xy_b in (
            ((roi[0], roi[2]), (0.0, 1.0)),
            ((roi[1], roi[2]), (1.0, 1.0)),
        ):
            connector = ConnectionPatch(
                xyA=xy_a, coordsA=ax.transData, xyB=xy_b, coordsB=inset.transAxes,
                color="#444444", lw=.50, alpha=.52, arrowstyle="-", clip_on=False,
                zorder=2, gid="panel-a-roi-connector",
            )
            parent.figure.add_artist(connector)
            connector_count += 1

    rx, ry, rw, rh = map(float, panel_cfg["resolution_block_bounds"])
    row_y = [ry + rh * fraction for fraction in (.82, .50, .18)]
    descriptors = {
        "L": ("Low resolution", "32 × 32"),
        "M": ("Medium resolution", "64 × 64"),
        "H": ("High resolution", "128 × 128"),
    }
    for tag, y in zip("LMH", row_y):
        parent.add_patch(Rectangle(
            (rx, y - .035), .018, .070, transform=parent.transAxes,
            fc=RESOLUTION_COLORS[tag], ec="none", clip_on=False,
        ))
        _tag(parent.text(rx + .025, y, tag, transform=parent.transAxes,
                         ha="left", va="center", fontweight="bold"), "axis_label")
        name, dims = descriptors[tag]
        _tag(parent.text(rx + .047, y + .022, name, transform=parent.transAxes,
                         ha="left", va="center"), "annotation")
        _tag(parent.text(rx + rw, y - .023, dims, transform=parent.transAxes,
                         ha="right", va="center"), "annotation")

    bx, by, bw, bh = map(float, panel_cfg["bar_bounds"])
    bar = _inset(parent, [bx, by, bw, bh])
    bar.axvspan(-.5, 2.5, color=str(panel_cfg["contains_h_tint"]), zorder=-5)
    bar.axvspan(2.5, 4.5, color=str(panel_cfg["zero_h_tint"]), zorder=-5)
    x = np.arange(5); bottom = np.zeros(5); segment_values = {}
    for tag in "LMH":
        segment = np.asarray([_float(row.get(f"train_cases_{tag}"), 0.0) for row in budgets])
        segment_values[tag] = segment.tolist()
        bar.bar(x, segment, bottom=bottom, color=RESOLUTION_COLORS[tag], width=.72)
        for x_here, base, height in zip(x, bottom, segment):
            if height > 0 and height / max(base + height, 1.0) >= .16:
                _tag(bar.text(
                    x_here, base + height / 2, tag, ha="center", va="center",
                    color="white" if tag == "H" else NEUTRAL_DARK, fontweight="bold",
                ), "annotation")
        bottom += segment
    ymax = float(bottom.max())
    bar.set_ylim(0, ymax * 1.31); bar.set_xlim(-.55, 4.55)
    labels = [
        row["recipe_label"].replace("Zero-H-balanced", "Zero-H\nbalanced")
        .replace("Zero-H-M-rich", "Zero-H\nM-rich") for row in budgets
    ]
    bar.set_xticks(x, labels); bar.tick_params(axis="x", pad=1.0)
    bar.set_ylabel("Training cases", labelpad=1.5)
    bar.set_xlabel("Values above bars: relative spatial-field exposure", labelpad=1.0)
    bar.set_yticks([0, 5000, 10000]); _open_axis(bar)
    for index, row in enumerate(budgets):
        _tag(bar.text(
            index, bottom[index] + ymax * .035,
            f"{_float(row['spatial_dof_budget_normalized_H_only']):.2f}×",
            ha="center", va="bottom",
        ), "annotation")
    _tag(bar.text(1.0, 1.005, "contains H-resolution training fields",
                  transform=bar.get_xaxis_transform(), ha="center", va="bottom",
                  color=NEUTRAL_MID), "annotation")
    _tag(bar.text(3.5, 1.005, "zero-H training",
                  transform=bar.get_xaxis_transform(), ha="center", va="bottom",
                  color=NEUTRAL_MID), "annotation")
    first = fields[0]
    return {
        "status": "ok", "sources": [str(fields_path), str(budgets_path), str(sensors_path)],
        "purpose": "resolution context and training-recipe design",
        "recipes": [row["recipe"] for row in budgets],
        "recipe_order": [row["recipe"] for row in budgets],
        "snapshot": _int(first.get("snapshot_index")), "case_id": first.get("case_id"),
        "time_index": first.get("time_index"), "field": first.get("field_name"),
        "dimensions": dimensions, "field_limits": limits,
        "shared_roi": {"xmin": roi[0], "xmax": roi[1], "ymin": roi[2], "ymax": roi[3]},
        "roi_shared_across_resolutions": True, "zoom_inset_count": 3,
        "connector_count": connector_count, "inset_interpolation": "nearest/native cells",
        "resolution_block_rows": [
            {"tag": tag, "name": descriptors[tag][0], "dimensions": descriptors[tag][1]}
            for tag in "LMH"
        ],
        "old_horizontal_resolution_legend_present": False,
        "bar_segments": segment_values,
        "exposure_values": [_float(row["spatial_dof_budget_normalized_H_only"]) for row in budgets],
        "grouping": {
            "contains_H": [row["recipe"] for row in budgets[:3]],
            "zero_H": [row["recipe"] for row in budgets[3:]],
        },
        "model_inference_performed": False,
    }


def draw_panel_b(parent, ctx: PublicationContext, **_):
    """Grouped 512-sensor recipe transfer and two zero-H sensor sweeps."""
    cfg, layout = ctx.cfg, ctx.v2
    panel_cfg = layout["panel_b_v3_3"]
    source_path, rows = ctx.source("UnifiedPublicationV2", "SensorSweepAllRecipes_summary")
    rows = [row for row in rows if row.get("metric") == "physical_rel_l2"]
    models = ctx.model_order(); recipes = list(panel_cfg["recipes"])
    sweep_recipes = list(panel_cfg["sweep_recipes"])
    count = int(panel_cfg["sensor_count"])
    counts = [int(value) for value in panel_cfg["sensor_counts"]]
    colors, alphas = model_colors(cfg), model_alphas(cfg)
    parent.set_axis_off()

    transfer_rows = {
        model: [_find(rows, model, recipe, metric="physical_rel_l2", count=count) for recipe in recipes]
        for model in models
    }
    b1 = _inset(parent, list(map(float, panel_cfg["recipe_axis_bounds"])))
    b1.axvspan(2.5, 4.5, color=str(panel_cfg["zero_h_tint"]), zorder=-5)
    x = np.arange(len(recipes)); width = float(panel_cfg["bar_width"])
    offsets = (np.arange(len(models)) - (len(models) - 1) / 2) * width
    endpoint_values, plotted_rows = {}, []
    max_high = 0.0
    for model_index, model in enumerate(models):
        series = transfer_rows[model]
        mean = np.asarray([_float(row["mean"]) for row in series])
        low = np.asarray([_float(row["ci95_low"]) for row in series])
        high = np.asarray([_float(row["ci95_high"]) for row in series])
        max_high = max(max_high, float(high.max()))
        bars = b1.bar(
            x + offsets[model_index], mean, width=width * .92,
            color=colors[model], alpha=alphas[model], edgecolor="none", zorder=3,
            yerr=[mean - low, high - mean], capsize=1.3,
            error_kw={"elinewidth": LW_ERRORBAR, "ecolor": colors[model], "alpha": alphas[model]},
        )
        for patch in bars:
            patch.set_gid(f"model-bar:{model}")
        if model == "DMFGen":
            for xpos, value, upper in zip(x + offsets[model_index], mean, high):
                _tag(b1.text(xpos, upper + max_high * .018, f"{value:.3g}",
                             ha="center", va="bottom", color=colors[model]), "annotation")
        endpoint_values[model] = {recipe: float(value) for recipe, value in zip(recipes, mean)}
        plotted_rows.extend({
            "role": "recipe_transfer_512", "model": model, "recipe": recipe,
            "sensor_count": count, "mean": _float(row["mean"]),
            "ci95_low": _float(row["ci95_low"]), "ci95_high": _float(row["ci95_high"]),
            "valid_n": _int(row["valid_n"]),
        } for recipe, row in zip(recipes, series))
    b1.set_ylim(0.0, max_high * 1.12); b1.set_xlim(-.55, 4.55)
    recipe_labels = [
        layout["recipes"]["short_labels"][recipe]
        .replace("Zero-H-balanced", "Zero-H\nbalanced")
        .replace("Zero-H-M-rich", "Zero-H\nM-rich") for recipe in recipes
    ]
    b1.set_xticks(x, recipe_labels); b1.tick_params(axis="x", pad=1.0)
    b1.set_ylabel(r"Physical relative $L_2$", labelpad=1.5)
    _tag(b1.set_title("Training-recipe transfer (512 sensors)", pad=1.5), "subplot_title")
    _tag(b1.text(3.5, .96, "no H-resolution training fields",
                 transform=b1.get_xaxis_transform(), ha="center", va="top",
                 color=NEUTRAL_MID), "annotation")
    b1.grid(axis="y", which="major", color=COLOR_GRID, lw=LW_GRID, zorder=0)
    _open_axis(b1)

    selected_sweep_rows = [
        _find(rows, model, recipe, metric="physical_rel_l2", count=n)
        for model in models for recipe in sweep_recipes for n in counts
    ]
    ymin = max(min(_float(row["ci95_low"]) for row in selected_sweep_rows) * .70, 1e-4)
    ymax = max(_float(row["ci95_high"]) for row in selected_sweep_rows) * 1.40
    grid_points = max(_int(row.get("evaluation_grid_points"), 128 * 128) for row in selected_sweep_rows)
    density = [100.0 * value / grid_points for value in counts]
    sweep_meta = {}
    for axis_index, (recipe, bounds) in enumerate(zip(sweep_recipes, panel_cfg["sweep_axis_bounds"])):
        ax = _inset(parent, list(map(float, bounds))); sweep_meta[recipe] = {}
        for model_index, model in enumerate(models):
            series = [_find(rows, model, recipe, metric="physical_rel_l2", count=n) for n in counts]
            mean = np.asarray([_float(row["mean"]) for row in series])
            low = np.asarray([_float(row["ci95_low"]) for row in series])
            high = np.asarray([_float(row["ci95_high"]) for row in series])
            linewidth = float(panel_cfg["dmfg_linewidth_pt"] if model == "DMFGen" else panel_cfg["baseline_linewidth_pt"])
            line, = ax.plot(
                counts, mean, color=colors[model], alpha=alphas[model],
                marker=MODEL_MARKERS[model_index], ms=3.2, lw=linewidth,
                ls=method_line_style(model_index), zorder=3,
            )
            line.set_gid(f"model-line:{model}")
            ax.errorbar(counts, mean, yerr=[mean - low, high - mean], fmt="none",
                        ecolor=colors[model], alpha=alphas[model], elinewidth=LW_ERRORBAR,
                        capsize=1.1, zorder=2)
            sweep_meta[recipe][model] = {str(n): float(value) for n, value in zip(counts, mean)}
            role = "zero_h_balanced_sweep" if recipe == "4_ZeroH_Balanced" else "zero_h_mrich_sweep"
            plotted_rows.extend({
                "role": role, "model": model, "recipe": recipe,
                "sensor_count": n, "mean": _float(row["mean"]),
                "ci95_low": _float(row["ci95_low"]), "ci95_high": _float(row["ci95_high"]),
                "valid_n": _int(row["valid_n"]),
            } for n, row in zip(counts, series))
        ax.set_yscale("log"); ax.set_ylim(ymin, ymax)
        ax.set_xlim(min(counts) - 24, max(counts) + 24)
        ax.set_xticks(counts, [f"{n}\n{pct:.1f}" for n, pct in zip(counts, density)])
        ax.tick_params(axis="x", pad=1.0)
        _tag(ax.set_title(layout["recipes"]["short_labels"][recipe], pad=1.5), "subplot_title")
        if axis_index == 0:
            ax.set_ylabel(r"Physical relative $L_2$", labelpad=1.5)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("Sensor count / H-grid density (%)", labelpad=1.0)
        ax.grid(axis="y", which="major", color=COLOR_GRID, lw=LW_GRID, zorder=0)
        _open_axis(ax)

    legend_ax = _inset(parent, list(map(float, panel_cfg["legend_bounds"])))
    legend_ax.set_axis_off()
    legend = legend_ax.legend(
        handles=_model_handles(ctx), ncol=4, loc="center", frameon=False,
        fontsize=float(panel_cfg["legend_fontsize_pt"]),
        columnspacing=float(panel_cfg["legend_columnspacing"]),
        handletextpad=float(panel_cfg["legend_handletextpad"]),
        handlelength=2.0, borderaxespad=0.0,
    )
    for text in legend.get_texts():
        _tag(text, "legend", size=float(panel_cfg["legend_fontsize_pt"]))
    return {
        "status": "ok", "sources": [str(source_path)], "models": models,
        "model_order": models, "recipes": recipes, "recipe_order": recipes,
        "sweep_recipes": sweep_recipes, "sensor_count": count,
        "sensor_counts": counts, "sensor_density_percent": density,
        "subaxis_roles": ["recipe_transfer_grouped_bars", "zero_h_balanced_sweep", "zero_h_mrich_sweep"],
        "recipe_transfer_plot_type": "grouped_bar", "recipe_transfer_axis_scale": "linear",
        "sensor_sweep_axis_scale": "log", "statistic": "mean with bootstrap 95% CI",
        "recipe_transfer_y_range": [0.0, max_high * 1.12], "sensor_sweep_y_range": [ymin, ymax],
        "recipe_transfer_values": endpoint_values, "sensor_sweep_values": sweep_meta,
        "plotted_rows": plotted_rows, "dmfg_value_annotations": len(recipes),
        "zero_h_region_shaded": True,
        "legend_contract": {
            "dedicated_axis": True, "ncol": 4,
            "fontsize_pt": float(panel_cfg["legend_fontsize_pt"]),
            "columnspacing": float(panel_cfg["legend_columnspacing"]),
            "handletextpad": float(panel_cfg["legend_handletextpad"]),
        },
        "complete_five_recipe_sweep_location": "SI_Figure_Sx1",
        "model_inference_performed": False,
    }


def draw_panel_c(parent, ctx: PublicationContext, **_):
    """Five-column full/zoom/local-error proof for one exact cached state."""
    cfg, layout = ctx.cfg, ctx.v2
    panel_cfg = layout["panel_c_v3_3"]
    recipe = str(panel_cfg["recipe"]); rich_contract = _rich_recipe_contract(ctx)
    if recipe != rich_contract["recipe"]:
        raise ValueError("Panel c must use the validated Zero-H-M-rich recipe")
    models = list(panel_cfg["models"])
    expected_models = ["DMFGen", "FFM_Perceiver", "Senseiver", "MLP_RBF"]
    if models != expected_models:
        raise ValueError(f"Panel c model order must be {expected_models}")
    count = int(panel_cfg["sensor_count"])
    default_snapshot = ctx.representatives.get(str(panel_cfg["snapshot_block"]), -1)
    snapshot, snapshot_selection = resolve_panel_snapshot(layout, "c", default_snapshot)
    payload = {model: ctx.cache(model, recipe, snapshot, count) for model in models}
    missing = [model for model, loaded in payload.items() if loaded is None]
    if missing:
        raise RuntimeError(f"Missing exact validated V3-3 panel-c cache entries: {missing}")

    first_arrays, first_meta, _first_row = payload[models[0]]
    coords = np.asarray(first_arrays["coords_phys"]); truth = np.asarray(first_arrays["truth_phys"]).reshape(-1)
    obs = np.asarray(first_arrays["obs_indices"])
    identity_keys = (
        "recipe", "snapshot_index", "case_id", "time_index", "sensor_count",
        "selected_raw_field_name", "truth_ref", "grid_ref", "sensor_plan_id", "sensor_plan_hash",
    )
    identity_mismatches = []
    for model in models:
        arrays, meta, _row = payload[model]
        differences = {
            key: [first_meta.get(key), meta.get(key)]
            for key in identity_keys if first_meta.get(key) != meta.get(key)
        }
        if differences or not np.array_equal(np.asarray(arrays["coords_phys"]), coords) \
                or not np.array_equal(np.asarray(arrays["truth_phys"]).reshape(-1), truth) \
                or not np.array_equal(np.asarray(arrays["obs_indices"]), obs):
            identity_mismatches.append({"model": model, "metadata": differences})
    if identity_mismatches:
        raise RuntimeError(f"Panel-c cache identity mismatch: {identity_mismatches}")

    parent.set_axis_off()
    predictions = {model: np.asarray(payload[model][0]["recon_phys"]).reshape(-1) for model in models}
    cmap, field_norm, field_limits = _field_norm(np.concatenate([truth, *predictions.values()]), cfg)
    roi = automatic_integrated_gradient_roi(coords, truth, fraction=float(panel_cfg["roi_fraction"]))
    roi_mask = _roi_mask(coords, roi)
    if not np.any(roi_mask):
        raise ValueError("Validated V3-3 ROI contains no H-grid samples")
    local_errors = {model: np.abs(prediction - truth) for model, prediction in predictions.items()}
    error_cmap, error_norm, error_limits = _error_norm(
        np.concatenate([error[roi_mask] for error in local_errors.values()]), cfg,
    )
    render_cfg = dict(cfg); render_cfg["rendering"] = dict(cfg["rendering"])
    render_cfg["rendering"]["contour_levels"] = int(panel_cfg["contour_levels"])
    left, right = float(panel_cfg["columns_left"]), float(panel_cfg["columns_right"])
    gap = float(panel_cfg["column_gap"]); width = (right - left - 4 * gap) / 5
    rows = {
        "full": (float(panel_cfg["full_row_bottom"]), float(panel_cfg["full_row_height"])),
        "zoom": (float(panel_cfg["zoom_row_bottom"]), float(panel_cfg["zoom_row_height"])),
        "error": (float(panel_cfg["error_row_bottom"]), float(panel_cfg["error_row_height"])),
    }
    headers = ["Ground truth", *[ctx.model_label(model) for model in models]]
    full_field_l2, local_l2, connector_count = {}, {}, 0
    field_artist = error_artist = None
    row_axes = {"full": [], "zoom": [], "error": []}
    for column, header in enumerate(headers):
        x = left + column * (width + gap)
        _tag(parent.text(x + width / 2, float(panel_cfg["header_y"]), header,
                         transform=parent.transAxes, ha="center", va="top",
                         color=NEUTRAL_DARK, fontweight="bold" if column == 0 else "normal"),
             "subplot_title")
        values = truth if column == 0 else predictions[models[column - 1]]
        full_ax = _inset(parent, [x, rows["full"][0], width, rows["full"][1]])
        row_axes["full"].append(full_ax)
        field_artist = _draw_field(full_ax, coords, values, render_cfg, cmap=cmap, norm=field_norm,
                                   mode="native_cells", contours=True, roi=roi, draw_roi=True)
        if column > 0:
            model = models[column - 1]; value = relative_l2(truth, predictions[model])
            full_field_l2[model] = float(value)
            annotation = full_ax.text(.035, .035, rf"Rel. $L_2$ = {value:.3f}",
                                      transform=full_ax.transAxes, ha="left", va="bottom",
                                      color="white", gid="qualitative-full-relative-l2",
                                      bbox=dict(boxstyle="round,pad=0.070", fc="black", ec="none", alpha=.52))
            _tag(annotation, "annotation")
        zoom_ax = _inset(parent, [x, rows["zoom"][0], width, rows["zoom"][1]])
        row_axes["zoom"].append(zoom_ax)
        _draw_field(zoom_ax, coords, values, render_cfg, cmap=cmap, norm=field_norm,
                    mode="native_cells", contours=True, crop=roi)
        for xy_a, xy_b in (((roi[0], roi[2]), (0.0, 1.0)), ((roi[1], roi[2]), (1.0, 1.0))):
            connector = ConnectionPatch(
                xyA=xy_a, coordsA=full_ax.transData, xyB=xy_b, coordsB=zoom_ax.transAxes,
                color="#444444", lw=.50, alpha=.42, arrowstyle="-", clip_on=False,
                zorder=.5, gid="panel-c-frustum-connector",
            )
            parent.figure.add_artist(connector); connector_count += 1

    sensor_ax = _inset(parent, [left, rows["error"][0], width, rows["error"][1]])
    row_axes["error"].append(sensor_ax)
    sensor_count_in_roi = _reference_observations(
        sensor_ax, coords, truth, obs, roi, cmap, field_norm, render_cfg,
    )
    sensor_note = sensor_ax.text(.04, .96, "Sensor layout", transform=sensor_ax.transAxes,
                                 ha="left", va="top", color=NEUTRAL_DARK,
                                 bbox=dict(boxstyle="round,pad=0.06", fc="white", ec="none", alpha=.75))
    _tag(sensor_note, "annotation")
    for column, model in enumerate(models, start=1):
        x = left + column * (width + gap)
        error_ax = _inset(parent, [x, rows["error"][0], width, rows["error"][1]])
        row_axes["error"].append(error_ax)
        error_artist = _draw_field(error_ax, coords, local_errors[model], render_cfg,
                                   cmap=error_cmap, norm=error_norm, mode="native_cells", crop=roi)
        value = relative_l2(truth[roi_mask], predictions[model][roi_mask]); local_l2[model] = float(value)
        annotation = error_ax.text(.035, .035, rf"Rel. $L_2$ = {value:.3f}",
                                   transform=error_ax.transAxes, ha="left", va="bottom",
                                   color="white", gid="qualitative-local-relative-l2",
                                   bbox=dict(boxstyle="round,pad=0.070", fc="black", ec="none", alpha=.52))
        _tag(annotation, "annotation")
    row_label_x = left - .018
    for label, key in (("Full field", "full"), ("Zoomed field", "zoom"), ("Local error", "error")):
        y, height = rows[key]
        _tag(parent.text(row_label_x, y + height / 2, label, transform=parent.transAxes,
                         ha="right", va="center", rotation=90, color=NEUTRAL_DARK), "axis_label")
    _, field_bar = _manual_horizontal_colorbar(
        parent, field_artist, panel_cfg["field_colorbar_bounds"], field_limits,
        "Field value", panel_cfg["colorbar_exponent_x"],
    )
    _, error_bar = _manual_horizontal_colorbar(
        parent, error_artist, panel_cfg["error_colorbar_bounds"], error_limits,
        "Zoom-region absolute error", panel_cfg["colorbar_exponent_x"],
    )
    parent.figure.canvas.draw()
    axes_bounds = {
        row: [list(map(float, axis.get_position().bounds)) for axis in axes]
        for row, axes in row_axes.items()
    }
    # The tiles occupy different columns, so only their rendered width/height
    # footprints must match; comparing x/y origins would incorrectly penalize
    # the intended horizontal placement.
    footprint_delta = max(
        max(abs(a - b) for a, b in zip(axes_bounds["error"][0][2:], bounds[2:]))
        for bounds in axes_bounds["error"][1:]
    )
    return {
        "status": "ok", "sources": [],
        "cache_sources": [payload[model][2]["cache_path"] for model in models],
        "purpose": "five-column global, magnified, and local-error physical proof",
        "recipe": recipe, "recipes": [recipe], "recipe_label": rich_contract["resolved_label"],
        "models": models, "display_columns": headers, "column_order": ["reference", *models],
        "column_count": 5, "sensor_count": count, "snapshot": snapshot,
        "snapshot_selection": snapshot_selection, "case_id": _int(first_meta.get("case_id")),
        "time_index": _int(first_meta.get("time_index")),
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
        "full_field_relative_l2": full_field_l2, "local_relative_l2": local_l2,
        "local_annotation_prefix": r"Rel. $L_2$ =", "local_annotation_uses_local_word": False,
        "error_definition": "abs(reconstruction - ground truth)", "error_scope": "zoom_roi",
        "field_limits": field_limits, "error_limits": error_limits,
        "sensor_count_in_roi": sensor_count_in_roi,
        "sensor_layout": {"placement": "full_equal_footprint_tile", "associated_column": "reference"},
        "axes_bounds_figure_fraction": axes_bounds,
        "sensor_error_footprint_max_abs_delta": float(footprint_delta),
        "frustum_connector_count": connector_count, "shared_colorbar_count": 2,
        "colorbar_formatting": {"field": field_bar, "error": error_bar},
        "mlp_rbf_qualitative_present": True, "training_recipe_count": 1,
        "model_inference_performed": False,
        "image_adjustments": "common robust field normalization and common ROI-error normalization only; no smoothing or sharpening",
    }


def _wavelet_metadata(ctx: PublicationContext):
    rid = ctx.source_run_ids.get("MultiscaleWavelet_metadata", ctx.run_id)
    path = RESULTS_DIR / "MultiscaleWavelet" / f"MultiscaleWavelet_metadata_{rid}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return path, json.loads(path.read_text(encoding="utf-8"))


def draw_panel_d(parent, ctx: PublicationContext, **_):
    """Three-scale qualitative evidence plus stacked fine-scale summaries."""
    cfg, layout = ctx.cfg, ctx.v2
    panel_cfg = layout["panel_d_v3_3"]
    metadata_path, metadata = _wavelet_metadata(ctx)
    summary_path, summary_rows = ctx.source("MultiscaleWavelet", "MultiscaleWavelet_summary")
    recipe = str(panel_cfg["recipe"]); rich = str(metadata["rich_recipe_contract"]["recipe"])
    if recipe != rich:
        raise ValueError("Panel d qualitative recipe differs from validated rich-recipe metadata")
    models = list(panel_cfg["qualitative_models"]); scales = list(panel_cfg["scales"])
    if scales != ["large", "intermediate", "fine"]:
        raise ValueError("Panel d must restore Large/Intermediate/Fine in that order")
    sensor_count = int(metadata["sensor_count"])
    snapshot, snapshot_selection = resolve_panel_snapshot(
        layout, "e", int(metadata["representative_snapshot"]["snapshot_index"]),
    )
    cache_payload = {model: ctx.cache(model, recipe, snapshot, sensor_count) for model in models}
    missing = [model for model, payload in cache_payload.items() if payload is None]
    if missing:
        raise RuntimeError(f"Missing validated panel-d cache entries: {missing}")
    parent.set_axis_off()
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
    qx, qy, qw, qh = map(float, panel_cfg["qualitative_bounds"])
    col_gap = float(panel_cfg["qualitative_column_gap"]); row_gap = float(panel_cfg["qualitative_row_gap"])
    col_width = (qw - 2 * col_gap - .015) / 3; row_height = (qh - 2 * row_gap) / 3
    render_cfg = dict(cfg); render_cfg["rendering"] = dict(cfg["rendering"])
    render_cfg["rendering"]["contour_levels"] = int(panel_cfg["contour_levels_truth"])
    rel_l2_values = {model: {} for model in models}
    headers = ["Truth\ncomponent", *[f"{ctx.model_label(model)}\nresid." for model in models]]
    for column, header in enumerate(headers):
        _tag(parent.text(qx + column * (col_width + col_gap) + col_width / 2, qy + qh + .020,
                         header, transform=parent.transAxes, ha="center", va="bottom",
                         color=NEUTRAL_DARK), "subplot_title")
    for row_index, scale in enumerate(scales):
        y = qy + (2 - row_index) * (row_height + row_gap)
        values = [truth_components[scale], *[residuals[model][scale] for model in models]]
        for column, array in enumerate(values):
            x = qx + column * (col_width + col_gap); ax = _inset(parent, [x, y, col_width, row_height])
            is_truth = column == 0
            limit = (component_limits if is_truth else residual_limits)[scale]
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
            _draw_field(ax, coords, array.reshape(-1), render_cfg,
                        cmap=manuscript.CMAP_SIGNED_COMPONENT if is_truth else str(panel_cfg["residual_cmap"]),
                        norm=norm, mode="native_cells",
                        contours=is_truth and int(panel_cfg["contour_levels_truth"]) > 0)
            if not is_truth:
                model = models[column - 1]
                value = relative_l2(truth_components[scale], predicted_components[model][scale])
                rel_l2_values[model][scale] = float(value)
                annotation = ax.text(.035, .035, rf"Rel. $L_2$ = {value:.2f}",
                                     transform=ax.transAxes, ha="left", va="bottom", color="white",
                                     gid="qualitative-relative-l2",
                                     bbox=dict(boxstyle="round,pad=0.060", fc="black", ec="none", alpha=.52))
                _tag(annotation, "annotation")
        _tag(parent.text(qx - .012, y + row_height / 2, scale.capitalize(),
                         transform=parent.transAxes, ha="right", va="center", rotation=90,
                         color=NEUTRAL_DARK), "axis_label")

    quantitative_models = list(panel_cfg["quantitative_models"])
    quantitative_recipes = list(panel_cfg["quantitative_recipes"])
    statistic = str(panel_cfg["statistic"]); colors, alphas = model_colors(cfg), model_alphas(cfg)
    quantitative_values, quantitative_plotted_rows = {}, []
    metric_map = {"correlation": "pattern_correlation", "variance_bias": "variance_fraction_bias_pp"}
    for role, bounds in panel_cfg["quantitative_bounds"].items():
        metric = metric_map[role]; ax = _inset(parent, list(map(float, bounds))); quantitative_values[metric] = {}
        for model_index, model in enumerate(quantitative_models):
            series = []
            for recipe_here in quantitative_recipes:
                matches = [row for row in summary_rows if row.get("model_key") == model
                           and row.get("recipe") == recipe_here and row.get("scale_group") == "fine"
                           and row.get("metric") == metric and row.get("status") == "ok"]
                if len(matches) != 1:
                    raise ValueError(f"Expected one fine-scale row for {model}/{recipe_here}/{metric}")
                series.append(matches[0])
            x = np.arange(len(quantitative_recipes)); median = np.asarray([_float(row[statistic]) for row in series])
            q25 = np.asarray([_float(row["q25"]) for row in series]); q75 = np.asarray([_float(row["q75"]) for row in series])
            line, = ax.plot(x, median, color=colors[model], alpha=alphas[model],
                            lw=2.0 if model == "DMFGen" else LW_LINE_PLOT,
                            ls=method_line_style(model_index), marker=MODEL_MARKERS[model_index], ms=3.2)
            line.set_gid(f"model-line:{model}")
            ax.errorbar(x, median, yerr=[median - q25, q75 - median], fmt="none",
                        ecolor=colors[model], alpha=alphas[model], elinewidth=LW_ERRORBAR, capsize=1.0)
            quantitative_values[metric][model] = {recipe_here: float(value)
                                                   for recipe_here, value in zip(quantitative_recipes, median)}
            quantitative_plotted_rows.extend({
                "model": model, "recipe": recipe_here, "scale_group": "fine", "metric": metric,
                "statistic": statistic, "median": _float(row[statistic]), "q25": _float(row["q25"]),
                "q75": _float(row["q75"]), "valid_n": _int(row["valid_n"]),
            } for recipe_here, row in zip(quantitative_recipes, series))
        ax.set_xlim(-.18, len(quantitative_recipes) - .82)
        labels = [layout["recipes"]["short_labels"][recipe_here]
                  .replace("Zero-H-balanced", "Zero-H\nbalanced")
                  .replace("Zero-H-M-rich", "Zero-H\nM-rich") for recipe_here in quantitative_recipes]
        ax.set_xticks(np.arange(3), labels if metric == "variance_fraction_bias_pp" else ["", "", ""])
        ax.tick_params(axis="x", pad=1.0); ax.grid(axis="y", which="major", color=COLOR_GRID, lw=LW_GRID, zorder=0)
        _open_axis(ax)
        if metric == "pattern_correlation":
            ax.axhline(1.0, color=NEUTRAL_MID, lw=.65, ls=":", zorder=0)
            ax.set_ylim(*map(float, panel_cfg["correlation_ylim"])); ax.set_ylabel("Correlation", labelpad=.5)
            _tag(ax.set_title("Fine-scale pattern correlation", pad=1.0), "subplot_title")
        else:
            all_bias = [abs(value) for values in quantitative_values[metric].values() for value in values.values()]
            limit = max(all_bias) * (1.0 + float(panel_cfg["bias_padding_fraction"]))
            ax.axhline(0.0, color=NEUTRAL_MID, lw=.65, ls=":", zorder=0); ax.set_ylim(-limit, limit)
            ax.set_ylabel("Bias (pp)", labelpad=.5)
            _tag(ax.set_title("Fine-scale variance-allocation bias", pad=1.0), "subplot_title")
    return {
        "status": "ok", "sources": [str(metadata_path), str(summary_path)],
        "cache_sources": [cache_payload[model][2]["cache_path"] for model in models],
        "purpose": "complete three-scale physical evidence plus fine-scale quantitative extraction",
        "qualitative_recipe": recipe, "qualitative_models": models, "qualitative_scales": scales,
        "large_scale_present_in_main": True, "quantitative_models": quantitative_models,
        "quantitative_recipes": quantitative_recipes, "main_quantitative_scale_groups": ["fine"],
        "main_quantitative_value_count": len(quantitative_plotted_rows), "statistic": statistic,
        "dispersion": "interquartile interval", "snapshot": snapshot,
        "snapshot_selection": snapshot_selection,
        "metadata_selected_snapshot": int(metadata["representative_snapshot"]["snapshot_index"]),
        "displayed_snapshot": snapshot, "sensor_count": sensor_count,
        "case_id": _int(first_meta.get("case_id")), "time_index": _int(first_meta.get("time_index")),
        "relative_l2_by_model_scale": rel_l2_values,
        "component_color_limits": {scale: [-value, value] for scale, value in component_limits.items()},
        "residual_color_limits": {scale: [-value, value] for scale, value in residual_limits.items()},
        "quantitative_values": quantitative_values, "plotted_rows": quantitative_plotted_rows,
        "qualitative_fraction": float(panel_cfg["qualitative_bounds"][3]),
        "quantitative_arrangement": "correlation_above_bias_stacked_vertically",
        "model_inference_performed": False,
    }


def draw_panel_e(parent, ctx: PublicationContext, **_):
    """Complete stacked 4 x 9 multiscale matrices with Fine-column cues."""
    metadata = draw_multiscale_fidelity(parent, ctx, standalone=False, show_legend=False)
    outlined_axes = 0; outline_count = 0
    matrix_axes = []
    for axis in parent.child_axes:
        if axis.yaxis.label.get_text() == "Spatial pattern correlation":
            axis.yaxis.label.set_text("Correlation")
        elif axis.yaxis.label.get_text() == "Variance allocation bias":
            axis.yaxis.label.set_text("Bias (pp)")
        if len(axis.images) != 1:
            continue
        matrix = np.asarray(axis.images[0].get_array())
        if matrix.shape != (4, 9):
            continue
        matrix_axes.append(axis)
        for label in axis.get_yticklabels():
            label.set_horizontalalignment("right")
            label.set_x(-.030)
        outlined_axes += 1
        for column in (2, 5, 8):
            patch = Rectangle((column - .5, -.5), 1.0, 4.0, fill=False,
                              ec="#555555", lw=.55, alpha=.72, zorder=5,
                              gid="panel-e-fine-column-outline")
            axis.add_patch(patch); outline_count += 1
    if outlined_axes != 2 or outline_count != 6:
        raise RuntimeError(
            f"Panel e Fine-column cue failed: matrix_axes={outlined_axes}, outlines={outline_count}"
        )
    matrix_axes.sort(key=lambda axis: axis.get_position().y0, reverse=True)
    titles = ["Spatial pattern correlation", "Variance allocation bias"]
    for axis, title in zip(matrix_axes, titles):
        _tag(axis.set_title(title, loc="left", pad=2.0), "subplot_title")
    metadata.update({
        "purpose": "complete scale-resolved numerical workload",
        "matrix_shape": [4, 9], "matrix_count": 2,
        "matrix_arrangement": "stacked_vertically",
        "fine_column_outline_count": outline_count,
        "fine_columns": [2, 5, 8],
        "linked_panel_d_scale": "fine",
        "model_inference_performed": False,
    })
    return metadata


PANEL_DRAWERS = {
    "a": draw_panel_a,
    "b": draw_panel_b,
    "c": draw_panel_c,
    "d": draw_panel_d,
    "e": draw_panel_e,
}


def draw_panel(label: str, parent, ctx: PublicationContext):
    return PANEL_DRAWERS[label](parent, ctx)

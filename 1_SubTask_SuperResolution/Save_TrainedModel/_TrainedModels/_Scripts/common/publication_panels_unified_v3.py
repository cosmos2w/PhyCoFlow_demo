"""Streamlined mixed-resolution Figure 3 panels.

All quantitative values are selected from the validated unified-v2 CSVs and
all qualitative fields are loaded from the validated reconstruction cache.
This module contains no training or inference entry point.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

import global_style as manuscript
from .config import RESULTS_DIR
from .figure_style import (
    COLOR_GRID,
    LW_ERRORBAR,
    LW_GRID,
    LW_LINE_PLOT,
    NEUTRAL_DARK,
    NEUTRAL_MID,
    RESOLUTION_COLORS,
    compact_colorbar_ticks,
    method_line_style,
    model_alphas,
    model_colors,
)
from .io_utils import matching_or_latest
from .multiscale_wavelet import decompose_field
from .multiscale_wavelet_panels import MODEL_MARKERS
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
    "a": "Panel_a_TrainingResolutionDesign",
    "b": "Panel_b_IntegratedPerformance",
    "c": "Panel_c_RepresentativeZeroH",
    "d": "Panel_d_MultiscaleFidelity",
}


def _tag(artist, role: str, size: float | None = None):
    kwargs = {} if size is None else {"size_pt": float(size)}
    return manuscript.tag_font_role(artist, role, **kwargs)


def panel_label(parent, label: str):
    _tag(parent.text(
        manuscript.PANEL_LABEL_X, manuscript.PANEL_LABEL_Y, label,
        transform=parent.transAxes, ha=manuscript.PANEL_LABEL_HA,
        va=manuscript.PANEL_LABEL_VA, color=NEUTRAL_DARK,
        fontweight="bold", clip_on=False,
    ), "panel_label")


def _find(rows, model: str, recipe: str, *, metric: str, count: int | None = None):
    matches = [
        row for row in rows
        if row.get("model") == model and row.get("recipe") == recipe
        and row.get("metric") == metric
        and (count is None or _int(row.get("sensor_count")) == int(count))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one validated row for {model}/{recipe}/{metric}/n={count}; "
            f"found {len(matches)}"
        )
    row = matches[0]
    value = _float(row.get("mean"))
    if row.get("status", "ok") not in {"", "ok"} or not np.isfinite(value):
        raise ValueError(f"Invalid validated summary row: {row}")
    return row


def _model_handles(ctx: PublicationContext):
    colors, alphas = model_colors(ctx.cfg), model_alphas(ctx.cfg)
    return [
        Line2D(
            [], [], color=colors[model], alpha=alphas[model],
            lw=2.0 if model == "DMFGen" else LW_LINE_PLOT,
            ls=method_line_style(index), marker=MODEL_MARKERS[index], ms=3.7,
            label=ctx.model_label(model),
        )
        for index, model in enumerate(ctx.model_order())
    ]


def _open_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def draw_panel_a(parent, ctx: PublicationContext, **_):
    """Compact discretization glyphs plus the five recipe-composition bars."""
    cfg, v3 = ctx.cfg, ctx.v2
    layout = v3["panel_a_streamlined"]
    fields_path, fields = ctx.source("ResolutionProtocol", "ResolutionProtocol_fields")
    budgets_path, budgets = ctx.source("ResolutionProtocol", "ResolutionProtocol_budgets")
    sensors_path, _ = ctx.source("ResolutionProtocol", "ResolutionProtocol_sensors")
    if not fields or len(budgets) != 5:
        raise RuntimeError("The validated resolution-protocol tables are incomplete")
    parent.set_axis_off()

    all_values = np.asarray([_float(row["field_value"]) for row in fields])
    cmap, norm, limits = _field_norm(all_values, cfg)
    gx, gy, gw, gh = map(float, layout["glyph_bounds"])
    gap = float(layout["glyph_gap"])
    cell_width = (gw - 2 * gap) / 3
    names = {"L": "L", "M": "M", "H": "H"}
    dimensions = {}
    for index, tag in enumerate("LMH"):
        rows = [row for row in fields if row["resolution"] == tag]
        coords = np.asarray([[_float(row["x_phys"]), _float(row["y_phys"])] for row in rows])
        values = np.asarray([_float(row["field_value"]) for row in rows])
        nx, ny = _int(rows[0]["num_x"]), _int(rows[0]["num_y"])
        dimensions[tag] = [nx, ny]
        x_edges, y_edges, grid = _resolution_cell_grid(
            coords, values, declared_nx=nx, declared_ny=ny, tag=tag,
        )
        ax = _inset(parent, [gx + index * (cell_width + gap), gy, cell_width, gh])
        _draw_resolution_cells(
            ax, x_edges, y_edges, grid, cmap=cmap, norm=norm,
            grid_linewidth={"L": .12, "M": .07, "H": 0.0}[tag],
            grid_alpha={"L": .62, "M": .34, "H": 0.0}[tag],
        )
        _tag(ax.set_title(f"{names[tag]}\n{nx} × {ny}", pad=1.0), "subplot_title")
    bx, by, bw, bh = map(float, layout["bar_bounds"])
    bar = _inset(parent, [bx, by, bw, bh])
    bar.axvspan(-.5, 2.5, color=str(layout["contains_h_tint"]), zorder=-5)
    bar.axvspan(2.5, 4.5, color=str(layout["zero_h_tint"]), zorder=-5)
    x = np.arange(5)
    bottom = np.zeros(5)
    segment_values = {}
    for tag in "LMH":
        segment = np.asarray([_float(row.get(f"train_cases_{tag}"), 0.0) for row in budgets])
        segment_values[tag] = segment.tolist()
        bar.bar(x, segment, bottom=bottom, color=RESOLUTION_COLORS[tag], width=.72)
        for x_here, base, height in zip(x, bottom, segment):
            if height > 0 and height / max(base + height, 1.0) >= .16:
                _tag(bar.text(
                    x_here, base + height / 2, tag, ha="center", va="center",
                    color="white" if tag == "H" else NEUTRAL_DARK,
                    fontweight="bold",
                ), "annotation")
        bottom += segment
    ymax = float(bottom.max())
    bar.set_ylim(0, ymax * 1.31)
    bar.set_xlim(-.55, 4.55)
    labels = [
        row["recipe_label"].replace("Zero-H-balanced", "Zero-H\nbalanced")
        .replace("Zero-H-M-rich", "Zero-H\nM-rich")
        for row in budgets
    ]
    bar.set_xticks(x, labels)
    bar.tick_params(axis="x", pad=1.0)
    bar.set_ylabel("Training cases", labelpad=1.5)
    bar.set_xlabel("Values above bars: relative spatial-field exposure", labelpad=1.0)
    bar.set_yticks([0, 5000, 10000])
    _open_axis(bar)
    for index, row in enumerate(budgets):
        _tag(bar.text(
            index, bottom[index] + ymax * .035,
            f"{_float(row['spatial_dof_budget_normalized_H_only']):.2f}×",
            ha="center", va="bottom",
        ), "annotation")
    _tag(bar.text(
        1.0, 1.005, "contains H-resolution training fields",
        transform=bar.get_xaxis_transform(), ha="center", va="bottom", color=NEUTRAL_MID,
    ), "annotation")
    _tag(bar.text(
        3.5, 1.005, "zero-H training",
        transform=bar.get_xaxis_transform(), ha="center", va="bottom", color=NEUTRAL_MID,
    ), "annotation")
    handles = [Rectangle((0, 0), 1, 1, fc=RESOLUTION_COLORS[tag], ec="none", label=tag) for tag in "LMH"]
    parent.legend(
        handles=handles, ncol=3, loc="lower left", bbox_to_anchor=(gx, .04),
        borderaxespad=0, handlelength=1.0, columnspacing=.75, handletextpad=.25,
    )
    return {
        "status": "ok", "sources": [str(fields_path), str(budgets_path), str(sensors_path)],
        "purpose": "training-resolution design only", "recipes": [row["recipe"] for row in budgets],
        "recipe_order": [row["recipe"] for row in budgets],
        "dimensions": dimensions, "field_limits": limits,
        "rendering_mode": "compact native-cell discretization glyphs without contours or zooms",
        "thumbnail_mode": "pixelated_native_cells", "thumbnail_count": 3,
        "dominant_image_area_fraction": float(gw * gh),
        "large_contour_count": 0, "large_field_contour_count": 0, "grouping": {
            "contains_H": [row["recipe"] for row in budgets[:3]],
            "zero_H": [row["recipe"] for row in budgets[3:]],
        },
        "bar_segments": segment_values,
        "exposure_values": [_float(row["spatial_dof_budget_normalized_H_only"]) for row in budgets],
    }


def draw_panel_b(parent, ctx: PublicationContext, **_):
    """Recipe-first 512-sensor transfer plus two zero-H sensor sweeps."""
    cfg, v3 = ctx.cfg, ctx.v2
    panel_cfg = v3["panel_b_streamlined"]
    source_path, rows = ctx.source("UnifiedPublicationV2", "SensorSweepAllRecipes_summary")
    rows = [row for row in rows if row.get("metric") == "physical_rel_l2"]
    models = ctx.model_order()
    recipes = list(panel_cfg["recipes"])
    sweep_recipes = list(panel_cfg["sweep_recipes"])
    count = int(panel_cfg["sensor_count"])
    counts = [int(value) for value in panel_cfg["sensor_counts"]]
    colors, alphas = model_colors(cfg), model_alphas(cfg)
    parent.set_axis_off()

    selected_rows = []
    for model in models:
        selected_rows.extend(_find(rows, model, recipe, metric="physical_rel_l2", count=count)
                             for recipe in recipes)
        selected_rows.extend(_find(rows, model, recipe, metric="physical_rel_l2", count=n)
                             for recipe in sweep_recipes for n in counts)
    ymin = max(min(_float(row["ci95_low"]) for row in selected_rows) * .70, 1e-4)
    ymax = max(_float(row["ci95_high"]) for row in selected_rows) * 1.40

    b1 = _inset(parent, list(map(float, panel_cfg["recipe_axis_bounds"])))
    b1.axvspan(2.5, 4.5, color=str(panel_cfg["zero_h_tint"]), zorder=-5)
    endpoint_values = {}
    plotted_rows = []
    for model_index, model in enumerate(models):
        series = [_find(rows, model, recipe, metric="physical_rel_l2", count=count) for recipe in recipes]
        x = np.arange(len(recipes))
        mean = np.asarray([_float(row["mean"]) for row in series])
        low = np.asarray([_float(row["ci95_low"]) for row in series])
        high = np.asarray([_float(row["ci95_high"]) for row in series])
        linewidth = float(panel_cfg["dmfg_linewidth_pt"] if model == "DMFGen" else panel_cfg["baseline_linewidth_pt"])
        line, = b1.plot(
            x, mean, color=colors[model], alpha=alphas[model],
            marker=MODEL_MARKERS[model_index], ms=3.5, lw=linewidth,
            ls=method_line_style(model_index), zorder=3,
        )
        line.set_gid(f"model-line:{model}")
        b1.errorbar(
            x, mean, yerr=[mean - low, high - mean], fmt="none",
            ecolor=colors[model], alpha=alphas[model], elinewidth=LW_ERRORBAR,
            capsize=1.25, zorder=2,
        )
        endpoint_values[model] = {recipe: float(value) for recipe, value in zip(recipes, mean)}
        plotted_rows.extend({
            "role": "recipe_transfer_512", "model": model, "recipe": recipe,
            "sensor_count": count, "mean": _float(row["mean"]),
            "ci95_low": _float(row["ci95_low"]), "ci95_high": _float(row["ci95_high"]),
            "valid_n": _int(row["valid_n"]),
        } for recipe, row in zip(recipes, series))
    b1.set_yscale("log"); b1.set_ylim(ymin, ymax); b1.set_xlim(-.25, 4.25)
    recipe_labels = [
        v3["recipes"]["short_labels"][recipe]
        .replace("Zero-H-balanced", "Zero-H\nbalanced")
        .replace("Zero-H-M-rich", "Zero-H\nM-rich")
        for recipe in recipes
    ]
    b1.set_xticks(np.arange(5), recipe_labels)
    b1.tick_params(axis="x", pad=1.0)
    b1.set_ylabel(r"Physical relative $L_2$", labelpad=1.5)
    _tag(b1.set_title("Training-recipe transfer (512 sensors)", pad=1.5), "subplot_title")
    _tag(b1.text(
        3.5, .96, "no H-resolution training fields",
        transform=b1.get_xaxis_transform(), ha="center", va="top", color=NEUTRAL_MID,
    ), "annotation")
    b1.grid(axis="y", which="major", color=COLOR_GRID, lw=LW_GRID, zorder=0)
    _open_axis(b1)

    sweep_meta = {}
    grid_points = max(_int(row.get("evaluation_grid_points"), 128 * 128) for row in selected_rows)
    density = [100.0 * value / grid_points for value in counts]
    for axis_index, (recipe, bounds) in enumerate(zip(sweep_recipes, panel_cfg["sweep_axis_bounds"])):
        ax = _inset(parent, list(map(float, bounds)))
        sweep_meta[recipe] = {}
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
            ax.errorbar(
                counts, mean, yerr=[mean - low, high - mean], fmt="none",
                ecolor=colors[model], alpha=alphas[model], elinewidth=LW_ERRORBAR,
                capsize=1.1, zorder=2,
            )
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
        _tag(ax.set_title(v3["recipes"]["short_labels"][recipe], pad=1.5), "subplot_title")
        if axis_index == 0:
            ax.set_ylabel(r"Physical relative $L_2$", labelpad=1.5)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("Sensor count / H-grid density (%)", labelpad=1.0)
        ax.grid(axis="y", which="major", color=COLOR_GRID, lw=LW_GRID, zorder=0)
        _open_axis(ax)

    parent.legend(
        handles=_model_handles(ctx), ncol=4, loc="upper center",
        bbox_to_anchor=(.53, float(panel_cfg["legend_y"])), borderaxespad=0,
        columnspacing=.85, handletextpad=.25,
    )
    return {
        "status": "ok", "sources": [str(source_path)], "models": models,
        "model_order": models, "recipes": recipes, "recipe_order": recipes,
        "sweep_recipes": sweep_recipes, "sensor_count": count,
        "sensor_counts": counts, "sensor_density_percent": density,
        "subaxis_roles": ["recipe_transfer_512", "zero_h_balanced_sweep", "zero_h_mrich_sweep"],
        "axis_scale": "log", "statistic": "mean with bootstrap 95% CI",
        "shared_y_range": [ymin, ymax], "recipe_transfer_values": endpoint_values,
        "sensor_sweep_values": sweep_meta,
        "plotted_rows": plotted_rows,
        "duplicate_512_justification": (
            "The recipe-transfer axis compares recipes at the standard budget; the two lower axes "
            "use the same endpoints only as the terminal point of distinct sensor-budget trends."
        ),
        "complete_five_recipe_sweep_location": "SI_Figure_Sx1",
    }


def draw_panel_c(parent, ctx: PublicationContext, **_):
    """One Zero-H-M-rich state: reference, three reconstructions, and errors."""
    cfg, v3 = ctx.cfg, ctx.v2
    panel_cfg = v3["panel_c_streamlined"]
    recipe = str(panel_cfg["recipe"])
    rich_contract = _rich_recipe_contract(ctx)
    if recipe != rich_contract["recipe"]:
        raise ValueError("Panel c must use the validated Zero-H-M-rich recipe")
    models = list(panel_cfg["models"])
    count = int(panel_cfg["sensor_count"])
    default_snapshot = ctx.representatives.get(str(panel_cfg["snapshot_block"]), -1)
    snapshot, snapshot_selection = resolve_panel_snapshot(v3, "c", default_snapshot)
    payload = {model: ctx.cache(model, recipe, snapshot, count) for model in models}
    missing = [model for model, loaded in payload.items() if loaded is None]
    if missing:
        raise RuntimeError(f"Missing validated panel-c cache entries: {missing}")
    parent.set_axis_off()
    first_arrays, first_meta, first_row = payload[models[0]]
    coords = np.asarray(first_arrays["coords_phys"])
    truth = np.asarray(first_arrays["truth_phys"]).reshape(-1)
    predictions = {model: np.asarray(payload[model][0]["recon_phys"]).reshape(-1) for model in models}
    cmap, field_norm, field_limits = _field_norm(
        np.concatenate([truth, *predictions.values()]), cfg,
    )
    error_cmap, error_norm, error_limits = _error_norm(
        np.concatenate([np.abs(pred - truth) for pred in predictions.values()]), cfg,
    )
    roi = automatic_integrated_gradient_roi(coords, truth, fraction=float(panel_cfg["roi_fraction"]))
    render_cfg = dict(cfg); render_cfg["rendering"] = dict(cfg["rendering"])
    render_cfg["rendering"]["contour_levels"] = int(panel_cfg["contour_levels"])
    left, right = float(panel_cfg["columns_left"]), float(panel_cfg["columns_right"])
    gap = float(panel_cfg["column_gap"])
    width = (right - left - 3 * gap) / 4
    full_y, full_h = float(panel_cfg["full_row_bottom"]), float(panel_cfg["full_row_height"])
    err_y, err_h = float(panel_cfg["error_row_bottom"]), float(panel_cfg["error_row_height"])
    headers = ["Ground truth", *[ctx.model_label(model) for model in models]]
    full_artists = []
    for index, label in enumerate(headers):
        x = left + index * (width + gap)
        ax = _inset(parent, [x, full_y, width, full_h])
        values = truth if index == 0 else predictions[models[index - 1]]
        artist = _draw_field(
            ax, coords, values, render_cfg, cmap=cmap, norm=field_norm,
            mode="native_cells", contours=True, roi=roi, draw_roi=True,
        )
        full_artists.append(artist)
        _tag(parent.text(
            x + width / 2, float(panel_cfg["header_y"]), label,
            transform=parent.transAxes, ha="center", va="top",
            color=NEUTRAL_DARK, fontweight="bold" if index == 0 else "normal",
        ), "subplot_title")
        zoom = ax.inset_axes(list(map(float, panel_cfg["zoom_inset_bounds"])))
        _draw_field(
            zoom, coords, values, render_cfg, cmap=cmap, norm=field_norm,
            mode="native_cells", contours=True, crop=roi,
        )
        for spine in zoom.spines.values():
            spine.set_visible(True); spine.set_color(NEUTRAL_DARK); spine.set_linewidth(.75)

    sensor_scale = float(panel_cfg["sensor_inset_scale"])
    sensor_w, sensor_h = width * sensor_scale, err_h * sensor_scale
    sensor_x = left + (width - sensor_w) / 2
    sensor_y = err_y + (err_h - sensor_h) / 2
    sensor_ax = _inset(parent, [sensor_x, sensor_y, sensor_w, sensor_h])
    sensor_count_in_roi = _reference_observations(
        sensor_ax, coords, truth, first_arrays["obs_indices"], roi,
        cmap, field_norm, render_cfg,
    )
    _tag(sensor_ax.set_title("Sensor layout", pad=1.0), "axis_label")
    error_values = {}
    error_artist = None
    for index, model in enumerate(models, start=1):
        x = left + index * (width + gap)
        ax = _inset(parent, [x, err_y, width, err_h])
        error = np.abs(predictions[model] - truth)
        error_artist = _draw_field(
            ax, coords, error, render_cfg, cmap=error_cmap, norm=error_norm,
            mode="native_cells", crop=roi,
        )
        value = relative_l2(truth, predictions[model])
        error_values[model] = float(value)
        annotation = ax.text(
            .035, .035, rf"Rel. $L_2$ = {value:.3f}", transform=ax.transAxes,
            ha="left", va="bottom", color="white", gid="qualitative-relative-l2",
            bbox=dict(boxstyle="round,pad=0.070", fc="black", ec="none", alpha=.52),
        )
        _tag(annotation, "annotation")
    _tag(parent.text(
        left + 2.5 * (width + gap), err_y + err_h + .018, "Local absolute error",
        transform=parent.transAxes, ha="center", va="bottom", color=NEUTRAL_DARK,
    ), "axis_label")

    field_cax = _inset(parent, list(map(float, panel_cfg["field_colorbar_bounds"])))
    field_cb = parent.figure.colorbar(full_artists[0], cax=field_cax, orientation="horizontal")
    field_ticks = np.linspace(field_limits[0], field_limits[1], 3)
    field_cb.set_ticks(field_ticks); field_cb.ax.tick_params(length=1.6, pad=1.0)
    _tag(field_cax.set_title("Field value", pad=1.0), "axis_label")
    compact_colorbar_ticks(field_cb, field_ticks, use_common_exponent=True)
    error_cax = _inset(parent, list(map(float, panel_cfg["error_colorbar_bounds"])))
    error_cb = parent.figure.colorbar(error_artist, cax=error_cax, orientation="horizontal")
    error_ticks = np.linspace(error_limits[0], error_limits[1], 3)
    error_cb.set_ticks(error_ticks); error_cb.ax.tick_params(length=1.6, pad=1.0)
    _tag(error_cax.set_title("Absolute error", pad=1.0), "axis_label")
    compact_colorbar_ticks(error_cb, error_ticks, use_common_exponent=True)
    return {
        "status": "ok", "sources": [],
        "cache_sources": [payload[model][2]["cache_path"] for model in models],
        "purpose": "representative zero-H H-resolution reconstruction",
        "recipe": recipe, "recipes": [recipe], "recipe_label": rich_contract["resolved_label"],
        "models": models, "sensor_count": count, "snapshot": snapshot,
        "snapshot_selection": snapshot_selection, "case_id": _int(first_meta.get("case_id")),
        "time_index": _int(first_meta.get("time_index")),
        "field": str(first_meta.get("selected_raw_field_name", "density")),
        "roi": {"xmin": roi[0], "xmax": roi[1], "ymin": roi[2], "ymax": roi[3],
                "selection": "maximum integrated ground-truth gradient magnitude"},
        "rows": ["full H-resolution field with zoom inset", "local absolute error"],
        "visual_rows": ["full_field", "absolute_error"],
        "field_limits": field_limits, "error_limits": error_limits,
        "relative_l2": error_values, "sensor_count_in_roi": sensor_count_in_roi,
        "training_recipe_count": 1, "model_inference_performed": False,
        "image_adjustments": "common robust field/error normalization only; no smoothing or sharpening",
    }


def _wavelet_metadata(ctx: PublicationContext):
    rid = ctx.source_run_ids.get("MultiscaleWavelet_metadata", ctx.run_id)
    path = matching_or_latest(
        RESULTS_DIR / "MultiscaleWavelet", "MultiscaleWavelet_metadata", rid, "json",
    )
    return path, json.loads(path.read_text(encoding="utf-8"))


def draw_panel_d(parent, ctx: PublicationContext, **_):
    """Intermediate/fine residual fields plus fine-scale quantitative trends."""
    cfg, v3 = ctx.cfg, ctx.v2
    panel_cfg = v3["panel_d_streamlined"]
    metadata_path, metadata = _wavelet_metadata(ctx)
    summary_path, summary_rows = ctx.source("MultiscaleWavelet", "MultiscaleWavelet_summary")
    recipe = str(panel_cfg["recipe"])
    rich = str(metadata["rich_recipe_contract"]["recipe"])
    if recipe != rich:
        raise ValueError("Panel d qualitative recipe differs from validated rich-recipe metadata")
    models = list(panel_cfg["qualitative_models"])
    scales = list(panel_cfg["scales"])
    if scales != ["intermediate", "fine"]:
        raise ValueError("Main panel d must contain only intermediate and fine scales")
    sensor_count = int(metadata["sensor_count"])
    # Preserve the state actually displayed by unified-v2: its shared explicit
    # snapshot overrides the metadata's optional automatic representative.
    snapshot, snapshot_selection = resolve_panel_snapshot(
        v3, "e", int(metadata["representative_snapshot"]["snapshot_index"]),
    )
    cache_payload = {model: ctx.cache(model, recipe, snapshot, sensor_count) for model in models}
    missing = [model for model, payload in cache_payload.items() if payload is None]
    if missing:
        raise RuntimeError(f"Missing validated panel-d cache entries: {missing}")
    parent.set_axis_off()
    first_arrays, first_meta, first_row = cache_payload[models[0]]
    order, ny, nx = grid_order(
        first_arrays["coords_phys"], first_meta.get("num_x"), first_meta.get("num_y"),
    )
    coords = np.asarray(first_arrays["coords_phys"])[order]
    truth = np.asarray(first_arrays["truth_phys"], dtype=np.float64).reshape(-1)[order].reshape(ny, nx)
    wave_cfg = v3["multiscale_wavelet"]
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
    residuals = {
        model: {scale: predicted_components[model][scale] - truth_components[scale] for scale in scales}
        for model in models
    }
    epsilon = float(v3["panel_e"]["gt_rms_epsilon"])
    component_limits = {
        scale: max(float(np.percentile(
            np.abs(truth_components[scale]), float(panel_cfg["component_percentile"]),
        )), epsilon) for scale in scales
    }
    residual_limits = {
        scale: max(float(np.percentile(np.concatenate([
            np.abs(residuals[model][scale]).ravel() for model in models
        ]), float(panel_cfg["residual_percentile"]))), epsilon) for scale in scales
    }
    qx, qy, qw, qh = map(float, panel_cfg["qualitative_bounds"])
    col_gap = float(panel_cfg["qualitative_column_gap"])
    row_gap = float(panel_cfg["qualitative_row_gap"])
    col_width = (qw - 2 * col_gap - .025) / 3
    row_height = (qh - row_gap) / 2
    render_cfg = dict(cfg); render_cfg["rendering"] = dict(cfg["rendering"])
    render_cfg["rendering"]["contour_levels"] = int(panel_cfg["contour_levels_truth"])
    residual_artists = {}
    rel_l2_values = {model: {} for model in models}
    headers = ["Ground-truth\ncomponent", *[f"{ctx.model_label(model)}\nresidual" for model in models]]
    for column, header in enumerate(headers):
        _tag(parent.text(
            qx + column * (col_width + col_gap) + col_width / 2, qy + qh + .035,
            header, transform=parent.transAxes, ha="center", va="bottom", color=NEUTRAL_DARK,
        ), "subplot_title")
    for row_index, scale in enumerate(scales):
        y = qy + (1 - row_index) * (row_height + row_gap)
        values = [truth_components[scale], *[residuals[model][scale] for model in models]]
        for column, array in enumerate(values):
            x = qx + column * (col_width + col_gap)
            ax = _inset(parent, [x, y, col_width, row_height])
            is_truth = column == 0
            norm = TwoSlopeNorm(
                vmin=-(component_limits if is_truth else residual_limits)[scale],
                vcenter=0.0,
                vmax=(component_limits if is_truth else residual_limits)[scale],
            )
            artist = _draw_field(
                ax, coords, array.reshape(-1), render_cfg,
                cmap=manuscript.CMAP_SIGNED_COMPONENT if is_truth else str(panel_cfg["residual_cmap"]),
                norm=norm, mode="native_cells",
                contours=is_truth and int(panel_cfg["contour_levels_truth"]) > 0,
            )
            if not is_truth:
                residual_artists[scale] = artist
                model = models[column - 1]
                value = relative_l2(truth_components[scale], predicted_components[model][scale])
                rel_l2_values[model][scale] = float(value)
                annotation = ax.text(
                    .035, .035, rf"Rel. $L_2$ = {value:.2f}", transform=ax.transAxes,
                    ha="left", va="bottom", color="white", gid="qualitative-relative-l2",
                    bbox=dict(boxstyle="round,pad=0.070", fc="black", ec="none", alpha=.52),
                )
                _tag(annotation, "annotation")
        _tag(parent.text(
            qx - .012, y + row_height / 2, scale.capitalize(),
            transform=parent.transAxes, ha="right", va="center", rotation=90,
            color=NEUTRAL_DARK,
        ), "axis_label")

    quantitative_models = list(panel_cfg["quantitative_models"])
    quantitative_recipes = list(panel_cfg["quantitative_recipes"])
    statistic = str(panel_cfg["statistic"])
    colors, alphas = model_colors(cfg), model_alphas(cfg)
    quantitative_values = {}
    quantitative_plotted_rows = []
    axes = []
    metric_map = {
        "correlation": "pattern_correlation",
        "variance_bias": "variance_fraction_bias_pp",
    }
    for role, bounds in panel_cfg["quantitative_bounds"].items():
        metric = metric_map[role]
        ax = _inset(parent, list(map(float, bounds)))
        axes.append(ax)
        quantitative_values[metric] = {}
        for model_index, model in enumerate(quantitative_models):
            series = []
            for recipe_here in quantitative_recipes:
                matches = [
                    row for row in summary_rows
                    if row.get("model_key") == model and row.get("recipe") == recipe_here
                    and row.get("scale_group") == "fine" and row.get("metric") == metric
                    and row.get("status") == "ok"
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one fine-scale row for {model}/{recipe_here}/{metric}")
                series.append(matches[0])
            x = np.arange(len(quantitative_recipes))
            median = np.asarray([_float(row[statistic]) for row in series])
            q25 = np.asarray([_float(row["q25"]) for row in series])
            q75 = np.asarray([_float(row["q75"]) for row in series])
            line, = ax.plot(
                x, median, color=colors[model], alpha=alphas[model],
                lw=2.0 if model == "DMFGen" else LW_LINE_PLOT,
                ls=method_line_style(model_index), marker=MODEL_MARKERS[model_index], ms=3.2,
            )
            line.set_gid(f"model-line:{model}")
            ax.errorbar(
                x, median, yerr=[median - q25, q75 - median], fmt="none",
                ecolor=colors[model], alpha=alphas[model], elinewidth=LW_ERRORBAR,
                capsize=1.0,
            )
            quantitative_values[metric][model] = {
                recipe_here: float(value) for recipe_here, value in zip(quantitative_recipes, median)
            }
            quantitative_plotted_rows.extend({
                "model": model, "recipe": recipe_here, "scale_group": "fine",
                "metric": metric, "statistic": statistic,
                "median": _float(row[statistic]), "q25": _float(row["q25"]),
                "q75": _float(row["q75"]), "valid_n": _int(row["valid_n"]),
            } for recipe_here, row in zip(quantitative_recipes, series))
        ax.set_xlim(-.18, len(quantitative_recipes) - .82)
        ax.set_xticks(np.arange(3), [
            v3["recipes"]["short_labels"][recipe_here]
            .replace("Zero-H-balanced", "Zero-H\nbalanced")
            .replace("Zero-H-M-rich", "Zero-H\nM-rich")
            for recipe_here in quantitative_recipes
        ])
        ax.tick_params(axis="x", pad=1.0)
        ax.grid(axis="y", which="major", color=COLOR_GRID, lw=LW_GRID, zorder=0)
        _open_axis(ax)
        if metric == "pattern_correlation":
            ax.axhline(1.0, color=NEUTRAL_MID, lw=.65, ls=":", zorder=0)
            ax.set_ylim(*map(float, panel_cfg["correlation_ylim"]))
            ax.set_ylabel("Correlation", labelpad=.5)
            _tag(ax.set_title("Fine-scale pattern correlation", pad=1.0), "subplot_title")
        else:
            all_bias = [abs(value) for values in quantitative_values[metric].values() for value in values.values()]
            limit = max(all_bias) * (1.0 + float(panel_cfg["bias_padding_fraction"]))
            ax.axhline(0.0, color=NEUTRAL_MID, lw=.65, ls=":", zorder=0)
            ax.set_ylim(-limit, limit)
            ax.set_ylabel("Bias (pp)", labelpad=.5)
            _tag(ax.set_title("Fine-scale variance allocation", pad=1.0), "subplot_title")

    expected_rich_corr = {
        "DMFGen": 0.815, "FFM_Perceiver": 0.368,
        "Senseiver": -0.089, "MLP_RBF": 0.165,
    }
    observed_rich_corr = {
        model: quantitative_values["pattern_correlation"][model][rich]
        for model in quantitative_models
    }
    if not all(np.isclose(observed_rich_corr[model], expected, atol=.0006)
               for model, expected in expected_rich_corr.items()):
        raise ValueError(
            f"Validated Zero-H-M-rich fine-scale correlations changed: {observed_rich_corr}"
        )
    return {
        "status": "ok", "sources": [str(metadata_path), str(summary_path)],
        "cache_sources": [cache_payload[model][2]["cache_path"] for model in models],
        "purpose": "intermediate/fine spatial organization and fine-scale quantitative fidelity",
        "qualitative_recipe": recipe, "qualitative_models": models,
        "qualitative_scales": scales, "main_quantitative_scales": ["fine"],
        "main_quantitative_scale_groups": ["fine"], "large_scale_cell_count": 0,
        "quantitative_models": quantitative_models, "quantitative_recipes": quantitative_recipes,
        "main_quantitative_value_count": 24,
        "statistic": statistic, "dispersion": "interquartile interval",
        "snapshot": snapshot, "snapshot_selection": snapshot_selection,
        "metadata_selected_snapshot": int(metadata["representative_snapshot"]["snapshot_index"]),
        "displayed_snapshot": snapshot,
        "sensor_count": sensor_count, "case_id": _int(first_meta.get("case_id")),
        "time_index": _int(first_meta.get("time_index")),
        "relative_l2_by_model_scale": rel_l2_values,
        "component_color_limits": {scale: [-value, value] for scale, value in component_limits.items()},
        "residual_color_limits": {scale: [-value, value] for scale, value in residual_limits.items()},
        "quantitative_values": quantitative_values,
        "plotted_rows": quantitative_plotted_rows,
        "expected_zero_h_m_rich_fine_correlations": expected_rich_corr,
        "complete_three_scale_analysis_location": "SI_Figure_Sx3",
        "large_scale_present_in_main": False, "model_inference_performed": False,
    }


PANEL_DRAWERS = {"a": draw_panel_a, "b": draw_panel_b, "c": draw_panel_c, "d": draw_panel_d}


def draw_panel(label: str, parent, ctx: PublicationContext):
    return PANEL_DRAWERS[label](parent, ctx)

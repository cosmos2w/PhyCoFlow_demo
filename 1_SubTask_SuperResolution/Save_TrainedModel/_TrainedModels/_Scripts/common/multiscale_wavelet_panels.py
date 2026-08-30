"""Native Matplotlib drawers for sensor efficiency and wavelet panels d--f."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import global_style as manuscript

from .config import RESULTS_DIR
from .figure_style import (
    LW_ERRORBAR,
    LW_LINE_PLOT,
    LW_LINE_SECONDARY,
    NEUTRAL_DARK,
    NEUTRAL_MID,
    compact_colorbar_ticks,
    mark_missing,
    method_line_style,
    model_alphas,
    model_colors,
)
from .io_utils import matching_or_latest
from .multiscale_wavelet import decompose_field
from .publication_panels import _draw_field, _float, _inset, _int
from .representative_snapshots import resolve_panel_snapshot
from .workflow import grid_order


SCALE_ORDER = ("large", "intermediate", "fine")
MODEL_MARKERS = ("o", "s", "^", "D")


# =============================================================================
# MULTISCALE PANEL INTERNAL LAYOUT USER TUNING API
# =============================================================================
# All values are fractions of the exact physical parent panel rectangle. They
# apply identically to standalone and composite output; do not branch on output
# destination. Increase margins/gaps only when the boundary audit requires it.
PANEL_D_LAYOUT = {
    "left": 0.070, "right": 0.995, "column_gap": 0.016,
    "axis_bottom": 0.220, "axis_height": 0.600,
    "legend_y": 0.995,
    "title_pad_pt": 1.5,
}
PANEL_E_LAYOUT = {
    "left": 0.055, "right": 0.800, "bottom": 0.025, "top": 0.910,
    "column_gap": 0.012, "row_gap": 0.012,
    "header_y": 0.995,
    "row_label_x": 0.012,
    "residual_colorbar_left": 0.835,
    "residual_colorbar_width": 0.018,
    "residual_colorbar_height_fraction": 0.75,
    "colorbar_label_pad_pt": 6.0,
}
PANEL_F_LAYOUT = {
    "heat_left": 0.170, "heat_right": 0.810,
    "top_bounds": [0.170, 0.545, 0.640, 0.400],
    "bottom_bounds": [0.170, 0.055, 0.640, 0.400],
    "model_label_x_axis": -0.258,
    "recipe_header_y": 0.995,
    "colorbar_left": 0.845, "colorbar_width": 0.014,
    "colorbar_tick_pad_pt": 2.0,
    "colorbar_label_pad_pt": 8.0,
}
# =============================================================================
# END MULTISCALE PANEL INTERNAL LAYOUT USER TUNING API
# =============================================================================


def _metadata(ctx):
    rid = ctx.source_run_ids.get("MultiscaleWavelet_metadata", ctx.run_id)
    path = matching_or_latest(
        RESULTS_DIR / "MultiscaleWavelet", "MultiscaleWavelet_metadata", rid, "json"
    )
    return path, json.loads(path.read_text(encoding="utf-8"))


def _model_handles(ctx, models):
    colors = model_colors(ctx.cfg)
    alphas = model_alphas(ctx.cfg)
    return [
        Line2D(
            [], [], color=colors[model], alpha=alphas[model], lw=LW_LINE_PLOT,
            ls=method_line_style(index), marker=MODEL_MARKERS[index % len(MODEL_MARKERS)],
            ms=3.6, label=ctx.model_label(model),
        )
        for index, model in enumerate(models)
    ]


def draw_sensor_efficiency(parent, ctx, *, standalone=False, show_legend=True, version=None):
    """Draw the existing all-recipe sensor sweep with its new panel-d role."""
    cfg, v2 = ctx.cfg, ctx.v2
    parent.set_axis_off()
    path, rows = ctx.source("UnifiedPublicationV2", "SensorSweepAllRecipes_summary")
    rows = [row for row in rows if row.get("metric") == "physical_rel_l2"]
    models = ctx.model_order(cfg["sensor_sweep"].get("main_models"))
    recipes = v2["panel_d"]["recipes"]
    counts = [int(value) for value in v2["panel_d"]["sensor_counts"]]
    colors, alphas = model_colors(cfg), model_alphas(cfg)
    valid = [
        row for row in rows
        if row.get("model") in models and row.get("recipe") in recipes
        and _int(row.get("sensor_count")) in counts and np.isfinite(_float(row.get("mean")))
    ]
    if not valid:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "sources": [str(path)]}
    ymin = max(min(_float(row.get("ci95_low")) for row in valid) * .72, 1e-4)
    ymax = max(_float(row.get("ci95_high")) for row in valid) * 1.42
    formal = int(v2["panel_d"].get("formal_sensor_count", 256))
    grid_points = max(_int(row.get("evaluation_grid_points"), 128 * 128) for row in valid)
    density = [100 * count / grid_points for count in counts]
    layout = {**PANEL_D_LAYOUT, **v2["panel_d"].get("layout", {})}
    left = float(layout["left"])
    gap = float(layout["column_gap"])
    width = (float(layout["right"]) - left - 4 * gap) / 5
    missing = []
    axis_bottom = float(layout["axis_bottom"])
    axis_height = float(layout["axis_height"])
    for index, recipe in enumerate(recipes):
        ax = _inset(parent, [left + index * (width + gap), axis_bottom, width, axis_height])
        missing_here = []
        for model_index, model in enumerate(models):
            subset = sorted(
                [row for row in valid if row.get("model") == model and row.get("recipe") == recipe],
                key=lambda row: _int(row["sensor_count"]),
            )
            if not subset:
                missing_here.append(ctx.model_label(model))
                missing.append({"model": model, "recipe": recipe, "status": "missing"})
                continue
            x = np.asarray([_int(row["sensor_count"]) for row in subset])
            mean = np.asarray([_float(row["mean"]) for row in subset])
            low = np.asarray([_float(row["ci95_low"]) for row in subset])
            high = np.asarray([_float(row["ci95_high"]) for row in subset])
            marker = MODEL_MARKERS[model_index % len(MODEL_MARKERS)]
            linestyle = method_line_style(model_index)
            line, = ax.plot(
                x, mean, color=colors[model], alpha=alphas[model], marker=marker,
                ms=3.0, lw=LW_LINE_PLOT, ls=linestyle,
            )
            line.set_gid(f"model-line:{model}")
            ax.errorbar(x, mean, yerr=[mean - low, high - mean], fmt="none",
                        ecolor=colors[model], alpha=alphas[model],
                        elinewidth=LW_ERRORBAR, capsize=1.3)
        ax.set_yscale(v2["panel_d"].get("yscale", "log"))
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(min(counts) - 24, max(counts) + 24)
        ax.set_xticks(counts, [f"{count}\n{pct:.1f}" for count, pct in zip(counts, density)])
        ax.tick_params(axis="x", pad=1)
        ax.set_title(
            v2["recipes"]["short_labels"][recipe],
            pad=float(layout["title_pad_pt"]),
        )
        if index == 2:
            ax.set_xlabel("Sensor count / H-grid density (%)", labelpad=0)
        if index == 0:
            ax.set_ylabel(r"Physical relative $L_2$", labelpad=2)
        else:
            ax.tick_params(labelleft=False)
        if missing_here:
            ax.text(.03, .96, "Missing:\n" + ", ".join(missing_here), transform=ax.transAxes,
                    ha="left", va="top", color=NEUTRAL_MID)
    if show_legend:
        parent.legend(handles=_model_handles(ctx, models), ncol=4, loc="upper center",
                      bbox_to_anchor=(.53, float(layout["legend_y"])), borderaxespad=0,
                      columnspacing=.8, handletextpad=.25)
    return {
        "status": "ok", "sources": [str(path)], "models": models, "recipes": recipes,
        "sensor_counts": counts, "sensor_density_percent": density, "formal_setting": formal,
        "formal_setting_marker_displayed": False,
        "axis_scale": "log", "statistic": "mean with bootstrap 95% CI",
        "model_markers": {
            model: MODEL_MARKERS[index % len(MODEL_MARKERS)]
            for index, model in enumerate(models)
        },
        "model_line_styles": {
            model: str(method_line_style(index)) for index, model in enumerate(models)
        },
        "shared_x_label": "Sensor count / H-grid density (%)",
        "legend_line_style": "model-specific", "shared_y_range": [ymin, ymax], "missing": missing,
    }


def _render_cfg(cfg, contour_levels):
    local = dict(cfg)
    local["rendering"] = dict(cfg["rendering"])
    local["rendering"]["contour_levels"] = int(contour_levels)
    local["rendering"]["contour_color"] = "#F2F0EB"
    local["rendering"]["contour_linewidth"] = .25
    return local


def draw_multiscale_components(parent, ctx, *, standalone=False, show_legend=True, version=None):
    """Draw raw truth components and signed scale-specific residuals."""
    cfg, v2 = ctx.cfg, ctx.v2
    parent.set_axis_off()
    metadata_path, metadata = _metadata(ctx)
    panel_cfg = v2["panel_e"]
    if panel_cfg["display_mode"] != "raw_truth_and_residual":
        raise ValueError("panel_e.display_mode must be 'raw_truth_and_residual'")
    representative = metadata["representative_snapshot"]
    selected_baseline = metadata["representative_baseline"]
    rich = str(metadata["rich_recipe_contract"]["recipe"])
    if str(panel_cfg["recipe"]) != rich:
        raise ValueError(
            f"panel_e.recipe={panel_cfg['recipe']!r} does not match resolved rich recipe {rich!r}"
        )
    model_cfg = panel_cfg["models"]
    proposed = str(model_cfg["proposed"])
    baseline = str(model_cfg["baseline"])
    if baseline != str(selected_baseline["model_key"]):
        raise ValueError(
            f"panel_e.models.baseline={baseline!r} does not match aggregate-selected "
            f"baseline {selected_baseline['model_key']!r}"
        )
    sensor_count = int(metadata["sensor_count"])
    snapshot, snapshot_selection = resolve_panel_snapshot(
        v2, "e", int(representative["snapshot_index"])
    )
    displayed_models = [proposed, baseline]
    cache_payload = {}
    for model in displayed_models:
        payload = ctx.cache(model, rich, snapshot, sensor_count)
        if payload is None:
            raise RuntimeError(f"Missing qualitative cache for {model}/{rich}/s{snapshot}/n{sensor_count}")
        cache_payload[model] = payload
    first_arrays, first_meta, first_row = cache_payload[proposed]
    order, ny, nx = grid_order(first_arrays["coords_phys"], first_meta.get("num_x"), first_meta.get("num_y"))
    coords = np.asarray(first_arrays["coords_phys"])[order]
    truth = np.asarray(first_arrays["truth_phys"], dtype=np.float64).reshape(-1)[order].reshape(ny, nx)
    wave_cfg = v2["multiscale_wavelet"]
    truth_components, _, _ = decompose_field(
        truth, wavelet=metadata["actual_wavelet"], level=int(metadata["wavelet_level"]),
        boundary_mode=metadata["boundary_mode"], groups=metadata["scale_groups"],
        reconstruction_tolerance=float(wave_cfg["reconstruction_tolerance"]),
    )
    predicted_components = {}
    cache_sources = []
    for model in displayed_models:
        arrays, meta, row = cache_payload[model]
        pred = np.asarray(arrays["recon_phys"], dtype=np.float64).reshape(-1)[order].reshape(ny, nx)
        components, _, _ = decompose_field(
            pred, wavelet=metadata["actual_wavelet"], level=int(metadata["wavelet_level"]),
            boundary_mode=metadata["boundary_mode"], groups=metadata["scale_groups"],
            reconstruction_tolerance=float(wave_cfg["reconstruction_tolerance"]),
        )
        predicted_components[model] = components
        cache_sources.append(row["cache_path"])

    epsilon = float(panel_cfg["gt_rms_epsilon"])
    gt_rms = {
        scale: float(np.sqrt(np.mean(np.square(truth_components[scale], dtype=np.float64))))
        for scale in SCALE_ORDER
    }
    displayed_truth = {
        scale: truth_components[scale] for scale in SCALE_ORDER
    }
    displayed_residuals = {
        model: {
            scale: predicted_components[model][scale] - truth_components[scale]
            for scale in SCALE_ORDER
        }
        for model in displayed_models
    }
    component_percentile = float(panel_cfg["component_percentile"])
    residual_percentile = float(panel_cfg["residual_percentile"])
    component_limits = {
        scale: max(
            float(np.percentile(np.abs(displayed_truth[scale]), component_percentile)),
            epsilon,
        )
        for scale in SCALE_ORDER
    }
    residual_limits = {
        scale: max(
            float(np.percentile(np.concatenate([
                np.abs(displayed_residuals[model][scale]).ravel()
                for model in displayed_models
            ]), residual_percentile)),
            epsilon,
        )
        for scale in SCALE_ORDER
    }
    component_norms = {
        scale: TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        for scale, limit in component_limits.items()
    }
    residual_norm_name = str(panel_cfg.get("residual_norm", "linear")).lower()
    if residual_norm_name != "linear":
        raise ValueError("panel_e.residual_norm must be 'linear'")
    residual_norms = {
        scale: TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        for scale, limit in residual_limits.items()
    }
    residual_cmap = str(panel_cfg.get("residual_cmap", "RdBu_r"))

    layout = {**PANEL_E_LAYOUT, **panel_cfg.get("layout", {})}
    left = float(layout["left"]); right = float(layout["right"])
    bottom = float(layout["bottom"]); top = float(layout["top"])
    col_gap = float(layout["column_gap"])
    row_gap = float(layout["row_gap"])
    col_width = (right - left - 2 * col_gap) / 3
    row_height = (top - bottom - 2 * row_gap) / 3
    truth_render_cfg = _render_cfg(cfg, panel_cfg["contour_levels_truth"])
    residual_render_cfg = _render_cfg(cfg, panel_cfg["contour_levels_residual"])
    column_headers = [
        f"{str(model_cfg['truth'])}\ncomponent",
        f"{ctx.model_label(proposed)}\nresidual",
        f"{ctx.model_label(baseline)}\nresidual",
    ]
    extent = [float(coords[:, 0].min()), float(coords[:, 0].max()), float(coords[:, 1].min()), float(coords[:, 1].max())]
    for column_index, label in enumerate(column_headers):
        header = parent.text(
            left + column_index * (col_width + col_gap) + col_width / 2,
            float(layout["header_y"]), label, transform=parent.transAxes,
            ha="center", va="top", color=NEUTRAL_DARK,
        )
        manuscript.tag_font_role(header, "subplot_title")
    residual_artists = {}
    relative_l2_by_model_scale = {model: {} for model in displayed_models}
    for row_index, scale in enumerate(SCALE_ORDER):
        y = top - (row_index + 1) * row_height - row_index * row_gap
        values = [displayed_truth[scale]] + [
            displayed_residuals[model][scale] for model in displayed_models
        ]
        for column_index, values_here in enumerate(values):
            x = left + column_index * (col_width + col_gap)
            ax = _inset(parent, [x, y, col_width, row_height])
            is_truth = column_index == 0
            artist = _draw_field(
                ax, coords, values_here.reshape(-1),
                truth_render_cfg if is_truth else residual_render_cfg,
                cmap=manuscript.CMAP_SIGNED_COMPONENT if is_truth else residual_cmap,
                norm=component_norms[scale] if is_truth else residual_norms[scale],
                mode="native_cells", contours=is_truth and int(panel_cfg["contour_levels_truth"]) > 0,
            )
            ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            if not is_truth:
                residual_artists[scale] = artist
                model = displayed_models[column_index - 1]
                denominator = max(float(np.linalg.norm(truth_components[scale].ravel())), epsilon)
                relative_l2 = float(np.linalg.norm(
                    predicted_components[model][scale].ravel() - truth_components[scale].ravel()
                ) / denominator)
                relative_l2_by_model_scale[model][scale] = relative_l2
                annotation = ax.text(
                    .035, .035, rf"Rel. $L_2$ = {relative_l2:.2f}",
                    transform=ax.transAxes, ha="left", va="bottom", color=NEUTRAL_DARK,
                    bbox=dict(boxstyle="round,pad=.12", fc="#F7F7F5", ec="none", alpha=.78),
                )
                manuscript.tag_font_role(annotation, "annotation")
        row_label = parent.text(
            float(layout["row_label_x"]), y + row_height / 2,
            scale.capitalize(), transform=parent.transAxes,
            ha="center", va="center", rotation=float(panel_cfg["row_label_rotation"]),
            color=NEUTRAL_DARK,
        )
        row_label_size = layout.get("row_label_fontsize")
        manuscript.tag_font_role(
            row_label, "axis_label",
            **({"size_pt": float(row_label_size)} if row_label_size is not None else {}),
        )

    colorbar_tick_labels = {}
    residual_colorbar_bounds = {}
    for row_index, scale in enumerate(SCALE_ORDER):
        row_y = top - (row_index + 1) * row_height - row_index * row_gap
        colorbar_height = row_height * float(layout["residual_colorbar_height_fraction"])
        y = row_y + (row_height - colorbar_height) / 2
        bounds = [
            float(layout["residual_colorbar_left"]), y,
            float(layout["residual_colorbar_width"]), colorbar_height,
        ]
        residual_colorbar_bounds[scale] = bounds
        limit = residual_limits[scale]
        field_name = str(first_meta.get("selected_raw_field_name", "field"))
        label = f"Signed residual ({field_name})"
        cax = _inset(parent, bounds)
        colorbar = parent.figure.colorbar(
            residual_artists[scale], cax=cax, orientation="vertical"
        )
        ticks = [-limit, 0.0, limit]
        colorbar.set_ticks(ticks)
        colorbar.ax.yaxis.set_ticks_position("right")
        colorbar.ax.tick_params(
            length=1.4, pad=.7,
        )
        if row_index == 1:
            colorbar.set_label(
                label, rotation=270, labelpad=float(layout["colorbar_label_pad_pt"]),
            )
            colorbar.ax.yaxis.set_label_position("right")
        colorbar_tick_labels[scale] = compact_colorbar_ticks(
            colorbar, ticks, use_common_exponent=True,
        )

    truth_residual_max_abs = max(
        float(np.max(np.abs((truth_components[scale] - truth_components[scale])
                            / max(gt_rms[scale], epsilon))))
        for scale in SCALE_ORDER
    )
    return {
        "status": "ok",
        "sources": [str(metadata_path)],
        "cache_sources": cache_sources,
        "snapshot": snapshot,
        "snapshot_selection": snapshot_selection,
        "representative_snapshot": representative,
        "representative_baseline": selected_baseline,
        "recipe": rich,
        "recipe_label": metadata["rich_recipe_contract"]["resolved_label"],
        "displayed_models": {
            "truth": str(model_cfg["truth"]), "proposed": proposed, "baseline": baseline,
        },
        "columns": column_headers,
        "scale_groups": list(SCALE_ORDER),
        "scale_group_labels": metadata["scale_group_labels"],
        "gt_rms": gt_rms,
        "gt_rms_epsilon": epsilon,
        "component_definition": "component_truth_s",
        "residual_definition": "component_pred_model_s - component_truth_s",
        "display_units": str(first_meta.get("selected_raw_field_name", "field")),
        "residual_sign": "prediction minus truth",
        "component_color_limits": {
            scale: [-limit, limit] for scale, limit in component_limits.items()
        },
        "residual_color_limits": {
            scale: [-limit, limit] for scale, limit in residual_limits.items()
        },
        "component_percentile": component_percentile,
        "residual_percentile": residual_percentile,
        "color_normalization": "row-specific robust symmetric linear raw-component and raw-residual limits",
        "residual_norm": {
            scale: {
                "name": "TwoSlopeNorm", "vcenter": 0.0,
                "vmin": -limit, "vmax": limit,
            }
            for scale, limit in residual_limits.items()
        },
        "colorbar_count": 3,
        "residual_colorbar_height_fraction": float(layout["residual_colorbar_height_fraction"]),
        "residual_colorbar_bounds_parent": residual_colorbar_bounds,
        "residual_row_height_parent": row_height,
        "colorbar_ticks": colorbar_tick_labels,
        "relative_l2_by_model_scale": relative_l2_by_model_scale,
        "relative_l2_annotation_format": "Rel. L_2 = {value:.2f}",
        "component_cmap": manuscript.CMAP_SIGNED_COMPONENT,
        "residual_cmap": residual_cmap,
        "colorbar_tick_formatter": "compact row-specific linear raw-value ticks with a shared exponent per bar",
        "coordinate_extent": extent,
        "sensor_overlays": False,
        "model_specific_adjustment": False,
        "smoothing": False,
        "contour_levels": {
            "truth": int(panel_cfg["contour_levels_truth"]),
            "residual": int(panel_cfg["contour_levels_residual"]),
        },
        "truth_residual_max_abs": truth_residual_max_abs,
        "width_fraction": float(panel_cfg["width_fraction"]),
        "wavelet": metadata["actual_wavelet"],
        "wavelet_level": metadata["wavelet_level"],
        "boundary_mode": metadata["boundary_mode"],
        "model_inference_performed": False,
        "missing": [],
    }


def _annotate_heatmap(ax, artist, matrix, fmt, *, annotate, fontsize):
    """Annotate a heatmap using the luminance of the rendered cell color."""
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if np.isfinite(value):
                if not annotate:
                    continue
                red, green, blue, _ = artist.cmap(artist.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                label = fmt(value) if callable(fmt) else fmt.format(value)
                color = "white" if luminance < .50 else "black"
            else:
                label, color = "Missing", "#4D4D4D"
            ax.text(column_index, row_index, label, ha="center", va="center",
                    fontsize=fontsize if np.isfinite(value) else max(3.5, fontsize - .7),
                    color=color)


def _style_heatmap_axis(
    ax, *, model_labels, show_models, show_scales, model_label_x_axis=None,
):
    ax.set_yticks(np.arange(len(model_labels)))
    ax.set_yticklabels(model_labels if show_models else [])
    ax.tick_params(axis="y", length=0, pad=2)
    if show_models and model_label_x_axis is not None:
        for label in ax.get_yticklabels():
            label.set_horizontalalignment("left")
            label.set_x(float(model_label_x_axis))
    if show_scales:
        ax.set_xticks(np.arange(9), ["Large", "Interm.", "Fine"] * 3)
        ax.tick_params(axis="x", length=0, pad=2)
    else:
        ax.set_xticks([])
    for boundary in (2.5, 5.5):
        ax.axvline(boundary, color="#D0D0D0", lw=.55, zorder=3)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_multiscale_fidelity(parent, ctx, *, standalone=False, show_legend=True, version=None):
    """Draw median multiscale fidelity as two compact stacked heatmaps."""
    cfg, v2 = ctx.cfg, ctx.v2
    parent.set_axis_off()
    summary_path, rows = ctx.source("MultiscaleWavelet", "MultiscaleWavelet_summary")
    metadata_path, metadata = _metadata(ctx)
    panel_cfg = v2["panel_f"]
    if panel_cfg["plot_type"] != "stacked_heatmaps":
        raise ValueError("panel_f.plot_type must be 'stacked_heatmaps'")
    statistic = str(panel_cfg["statistic"])
    if statistic != "median":
        raise ValueError("panel_f.statistic must be 'median' for the main heatmaps")
    recipes = [str(value) for value in panel_cfg["recipes"]]
    rich = str(metadata["rich_recipe_contract"]["recipe"])
    if rich not in recipes:
        raise ValueError(f"Resolved rich recipe {rich!r} is absent from panel_f.recipes")
    models = [str(value) for value in panel_cfg["models"]]
    unknown_models = [model for model in models if model not in ctx.model_order()]
    if unknown_models:
        raise ValueError(f"Unknown panel_f models: {unknown_models}")
    metrics = ("pattern_correlation", "variance_fraction_bias_pp")
    selected = [
        row for row in rows
        if row.get("model_key") in models and row.get("recipe") in recipes
        and row.get("scale_group") in SCALE_ORDER and row.get("metric") in metrics
    ]
    if not selected:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "sources": [str(summary_path), str(metadata_path)]}
    index = {
        (row["model_key"], row["recipe"], row["scale_group"], row["metric"]): row
        for row in selected
    }
    if len(index) != len(selected):
        raise ValueError("Duplicate multiscale summary cells found for panel f")

    matrices = {metric: np.full((len(models), len(recipes) * len(SCALE_ORDER)), np.nan)
                for metric in metrics}
    heatmap_values = []
    missing = []
    for model_index, model in enumerate(models):
        for recipe_index, recipe in enumerate(recipes):
            for scale_index, scale in enumerate(SCALE_ORDER):
                column_index = recipe_index * len(SCALE_ORDER) + scale_index
                for metric in metrics:
                    row = index.get((model, recipe, scale, metric))
                    value = float(row[statistic]) if row is not None and row.get(statistic) not in {None, ""} else np.nan
                    if row is None or row.get("status") != "ok" or not np.isfinite(value):
                        missing.append({"model": model, "recipe": recipe, "scale_group": scale, "metric": metric})
                        continue
                    matrices[metric][model_index, column_index] = value
                    heatmap_values.append({
                        "model_key": model, "recipe": recipe, "scale_group": scale,
                        "metric": metric, "statistic": statistic, "value": value,
                        "valid_n": int(row["valid_n"]),
                    })

    corr_range = [float(value) for value in panel_cfg["correlation_range"]]
    if corr_range[0] > -0.10 or corr_range[1] != 1.0:
        raise ValueError("panel_f.correlation_range must include -0.10 and end at 1.0")
    finite_bias = np.abs(matrices["variance_fraction_bias_pp"][
        np.isfinite(matrices["variance_fraction_bias_pp"])
    ])
    if finite_bias.size == 0:
        raise ValueError("No finite variance-allocation bias medians are available")
    bias_percentile = float(panel_cfg["bias_robust_percentile"])
    bias_limit = max(float(np.percentile(finite_bias, bias_percentile)), np.finfo(float).eps)
    norms = {
        "pattern_correlation": Normalize(vmin=corr_range[0], vmax=corr_range[1], clip=True),
        "variance_fraction_bias_pp": TwoSlopeNorm(vmin=-bias_limit, vcenter=0.0, vmax=bias_limit),
    }
    cmaps = {
        "pattern_correlation": matplotlib.colormaps[manuscript.CMAP_CORRELATION].copy(),
        "variance_fraction_bias_pp": matplotlib.colormaps[manuscript.CMAP_SIGNED_BIAS].copy(),
    }
    for cmap in cmaps.values():
        cmap.set_bad("#E1E1E1")

    layout = {**PANEL_F_LAYOUT, **panel_cfg.get("layout", {})}
    heat_left = float(layout["heat_left"])
    heat_right = float(layout["heat_right"])
    heat_width = heat_right - heat_left
    top_bounds = list(layout["top_bounds"])
    bottom_bounds = list(layout["bottom_bounds"])
    recipe_labels = [
        metadata["rich_recipe_contract"]["resolved_label"] if recipe == rich
        else v2["recipes"]["short_labels"][recipe]
        for recipe in recipes
    ]
    for recipe_index, label in enumerate(recipe_labels):
        x = heat_left + heat_width * ((recipe_index * 3 + 1.5) / 9)
        header = parent.text(
            x, float(layout["recipe_header_y"]), label,
            transform=parent.transAxes, ha="center", va="top", color=NEUTRAL_DARK,
        )
        manuscript.tag_font_role(header, "subplot_title")

    model_labels = [ctx.model_label(model) for model in models]
    colorbar_ticks = {}
    axes = []
    def format_bias(value):
        rounded = round(float(value), 2)
        return "0.00" if rounded == 0.0 else f"{rounded:+.2f}"

    for metric, bounds, fmt, colorbar_label in (
        ("pattern_correlation", top_bounds, "{:.2f}", "Spatial pattern correlation"),
        ("variance_fraction_bias_pp", bottom_bounds, format_bias, "Variance allocation bias"),
    ):
        ax = _inset(parent, bounds)
        axes.append(ax)
        artist = ax.imshow(
            matrices[metric], cmap=cmaps[metric], norm=norms[metric],
            aspect="auto", interpolation="nearest", rasterized=True,
        )
        _style_heatmap_axis(
            ax, model_labels=model_labels,
            show_models=True,
            show_scales=True,
            model_label_x_axis=float(layout["model_label_x_axis"]),
        )
        _annotate_heatmap(
            ax, artist, matrices[metric], fmt,
            annotate=bool(panel_cfg["annotate_cells"]), fontsize=manuscript.SIZE_ANNOTATION,
        )
        cax = _inset(parent, [
            float(layout["colorbar_left"]), bounds[1],
            float(layout["colorbar_width"]), bounds[3],
        ])
        colorbar = parent.figure.colorbar(artist, cax=cax, orientation="vertical")
        if metric == "pattern_correlation":
            ticks = np.linspace(corr_range[0], corr_range[1], 5)
        else:
            ticks = np.linspace(-bias_limit, bias_limit, 5)
        colorbar.set_ticks(ticks)
        colorbar.ax.tick_params(
            length=1.4, pad=float(layout["colorbar_tick_pad_pt"]),
        )
        colorbar.set_label(
            colorbar_label, rotation=270,
            labelpad=float(layout["colorbar_label_pad_pt"]),
        )
        colorbar.ax.yaxis.set_label_position("right")
        colorbar_label_size = layout.get("colorbar_label_fontsize")
        manuscript.tag_font_role(
            colorbar.ax.yaxis.label, "axis_label",
            **({"size_pt": float(colorbar_label_size)}
               if colorbar_label_size is not None else {}),
        )
        colorbar_tick_size = float(
            layout.get("colorbar_tick_fontsize", manuscript.SIZE_TICK_LABEL)
        )
        for tick_label in colorbar.ax.get_yticklabels():
            manuscript.tag_font_role(tick_label, "tick_label", size_pt=colorbar_tick_size)
        colorbar_ticks[metric] = compact_colorbar_ticks(
            colorbar, ticks, target_top_axis=ax,
            tick_size_pt=colorbar_tick_size,
        )

    return {
        "status": "ok",
        "sources": [str(summary_path), str(metadata_path)],
        "heatmap_source_csv": str(summary_path),
        "models": models,
        "model_labels": model_labels,
        "recipes": recipes,
        "recipe_labels": recipe_labels,
        "scale_groups": list(SCALE_ORDER),
        "scale_labels": ["Large", "Interm.", "Fine"],
        "metrics": list(metrics),
        "plot_type": str(panel_cfg["plot_type"]),
        "statistic": statistic,
        "correlation_color_limits": corr_range,
        "variance_bias_color_limits": [-bias_limit, bias_limit],
        "variance_bias_robust_percentile": bias_percentile,
        "colorbar_ticks": colorbar_ticks,
        "colorbar_tick_formatter": "one-decimal mantissa with shared order-of-magnitude multiplier",
        "annotation_formats": {
            "pattern_correlation": ".2f", "variance_fraction_bias_pp": "signed .2f; zero as 0.00",
        },
        "annotation_text_color": "rendered-cell luminance: white below 0.50, black otherwise",
        "standalone_heatmap_titles": False,
        "colorbar_titles": {
            "pattern_correlation": "Spatial pattern correlation",
            "variance_fraction_bias_pp": "Variance allocation bias",
        },
        "heatmap_values": heatmap_values,
        "rich_recipe_contract": metadata["rich_recipe_contract"],
        "all_summary_cells_valid_n_300": metadata["validation"]["all_summary_cells_valid_n_300"],
        "width_fraction": float(panel_cfg["width_fraction"]),
        "model_inference_performed": False,
        "missing": missing,
    }


def draw_multiscale_fidelity_intervals(parent, ctx, *, standalone=False, show_legend=True, version=None):
    """Preserve the former line-and-interval panel as an SI diagnostic."""
    cfg, v2 = ctx.cfg, ctx.v2
    parent.set_axis_off()
    summary_path, rows = ctx.source("MultiscaleWavelet", "MultiscaleWavelet_summary")
    metadata_path, metadata = _metadata(ctx)
    recipes = list(metadata["quantitative_recipes"])
    models = ctx.model_order()
    colors, alphas = model_colors(cfg), model_alphas(cfg)
    wave_cfg, panel_cfg = v2["multiscale_wavelet"], v2["panel_f"]
    aggregate = str(panel_cfg.get("aggregate", "median"))
    interval = str(panel_cfg.get("interval", "iqr"))
    if aggregate not in {"median", "mean"}:
        raise ValueError("panel_f.aggregate must be median or mean")
    interval_keys = {"iqr": ("q25", "q75"), "bootstrap_95ci": ("ci95_low", "ci95_high")}
    if interval not in interval_keys:
        raise ValueError("panel_f.interval must be iqr or bootstrap_95ci")
    low_key, high_key = interval_keys[interval]
    selected = [
        row for row in rows
        if row.get("model_key") in models and row.get("recipe") in recipes
        and row.get("scale_group") in SCALE_ORDER
        and row.get("metric") in {"pattern_correlation", "variance_fraction_bias_pp"}
    ]
    if not selected:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "sources": [str(summary_path), str(metadata_path)]}
    index = {
        (row["model_key"], row["recipe"], row["scale_group"], row["metric"]): row
        for row in selected
    }
    corr_config = [float(value) for value in wave_cfg["correlation_ylim"]]
    corr_low_data = min(float(row[low_key]) for row in selected if row["metric"] == "pattern_correlation")
    corr_ylim = [min(corr_config[0], corr_low_data - .02), max(corr_config[1], 1.02)]
    bias_values = np.asarray([
        abs(float(row[key])) for row in selected if row["metric"] == "variance_fraction_bias_pp"
        for key in (low_key, aggregate, high_key)
    ])
    percentile = float(wave_cfg["variance_bias_robust_percentile"])
    bias_limit = max(0.5, float(np.percentile(bias_values, percentile)) * 1.08)
    axes = {"correlation": [], "bias": []}
    plotted_values = []
    left, right, gap = .085, .99, .035
    width = (right - left - 2 * gap) / 3
    if standalone:
        top_y, bottom_y, height = .555, .115, .285
        top_title_y, top_note_y = .905, .865
        bottom_title_y, bottom_note_y = .475, .435
    else:
        # Composite panel f is shallower than its standalone export.  Keep the
        # interpretation notes clear of both recipe titles and tick labels.
        top_y, bottom_y, height = .550, .100, .235
        top_title_y, top_note_y = .905, .845
        bottom_title_y, bottom_note_y = .465, .415
    parent.text(.535, top_title_y, "Scale-wise spatial correlation",
                transform=parent.transAxes, ha="center", va="center",
                color=NEUTRAL_DARK, fontsize=cfg["figure_style"]["font_sizes"]["title"] - (0 if standalone else .8))
    parent.text(.535, bottom_title_y, "Scale-wise variance allocation",
                transform=parent.transAxes, ha="center", va="center",
                color=NEUTRAL_DARK, fontsize=cfg["figure_style"]["font_sizes"]["title"] - (0 if standalone else .8))
    x = np.arange(3, dtype=float)
    for recipe_index, recipe in enumerate(recipes):
        for metric_index, (metric, y, key) in enumerate((
            ("pattern_correlation", top_y, "correlation"),
            ("variance_fraction_bias_pp", bottom_y, "bias"),
        )):
            ax = _inset(parent, [left + recipe_index * (width + gap), y, width, height])
            axes[key].append(ax)
            for model in models:
                subset = [index[(model, recipe, scale, metric)] for scale in SCALE_ORDER]
                center = np.asarray([float(row[aggregate]) for row in subset])
                low = np.asarray([float(row[low_key]) for row in subset])
                high = np.asarray([float(row[high_key]) for row in subset])
                plotted_values.extend({
                    "model_key": model,
                    "recipe": recipe,
                    "scale_group": scale,
                    "metric": metric,
                    "center": float(center[scale_index]),
                    "low": float(low[scale_index]),
                    "high": float(high[scale_index]),
                } for scale_index, scale in enumerate(SCALE_ORDER))
                ax.plot(x, center, color=colors[model], alpha=alphas[model], lw=LW_LINE_PLOT,
                        marker="o", ms=3.0, ls="-")
                ax.errorbar(x, center, yerr=np.vstack([center - low, high - center]), fmt="none",
                            ecolor=colors[model], alpha=alphas[model],
                            elinewidth=LW_ERRORBAR, capsize=1.4)
            ax.set_xlim(-.2, 2.2)
            ax.set_xticks(x, ["Large", "Intermediate", "Fine"])
            ax.tick_params(axis="x", labelsize=max(4.6, cfg["figure_style"]["font_sizes"]["tick"] - 1.0))
            if metric == "pattern_correlation":
                ax.axhline(1.0, color=NEUTRAL_MID, lw=LW_LINE_SECONDARY, ls=":")
                ax.set_ylim(*corr_ylim)
                ax.set_title(v2["recipes"]["short_labels"][recipe], pad=2.0)
                if recipe_index == 0:
                    ax.set_ylabel("Pattern correlation", labelpad=2)
                else:
                    ax.tick_params(labelleft=False)
            else:
                ax.axhline(0.0, color=NEUTRAL_MID, lw=LW_LINE_SECONDARY, ls=":")
                ax.set_ylim(-bias_limit, bias_limit)
                if recipe_index == 0:
                    ax.set_ylabel("Variance-fraction bias\n(percentage points)", labelpad=2)
                else:
                    ax.tick_params(labelleft=False)
    parent.text(.535, top_note_y, "correct spatial placement of scale-specific structures",
                transform=parent.transAxes, ha="center", va="center", color=NEUTRAL_MID,
                fontsize=cfg["figure_style"]["font_sizes"]["tick"] - (0 if standalone else .7))
    parent.text(.535, bottom_note_y, "correct distribution of field variability across scales; zero is ideal",
                transform=parent.transAxes, ha="center", va="center", color=NEUTRAL_MID,
                fontsize=cfg["figure_style"]["font_sizes"]["tick"] - (0 if standalone else .7))
    if show_legend:
        parent.legend(handles=_model_handles(ctx, models), ncol=4, loc="upper center",
                      bbox_to_anchor=(.54, .995), borderaxespad=0,
                      columnspacing=.65, handletextpad=.22)
    return {
        "status": "ok",
        "sources": [str(summary_path), str(metadata_path)],
        "models": models,
        "recipes": recipes,
        "scale_groups": list(SCALE_ORDER),
        "metrics": ["pattern_correlation", "variance_fraction_bias_pp"],
        "statistic": f"{aggregate} with {interval}",
        "correlation_ylim": corr_ylim,
        "variance_bias_ylim": [-bias_limit, bias_limit],
        "variance_bias_robust_percentile": percentile,
        "reference_lines": {"pattern_correlation": 1.0, "variance_fraction_bias_pp": 0.0},
        "plotted_values": plotted_values,
        "rich_recipe_contract": metadata["rich_recipe_contract"],
        "all_summary_cells_valid_n_300": metadata["validation"]["all_summary_cells_valid_n_300"],
        "missing": [],
    }

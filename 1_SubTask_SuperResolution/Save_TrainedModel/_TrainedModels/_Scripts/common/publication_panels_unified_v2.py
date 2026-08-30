"""Reusable native Matplotlib panels for ``mixed_resolution_unified_v2``.

Every quantitative curve is read from a finalized or source-only derived CSV.
Field panels read audited cache entries only.  This module never runs inference.

PANEL C USER CONTROLS
---------------------
Edit ``common/panel_c_tuning.py`` for snapshot, sensor-count, manual ROI, grid
position, header-gap, and Panel C/master-height experiments.  Both figure entry
points consume that one API; the drawing implementation below should normally
remain untouched during manual layout tuning.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import global_style as manuscript
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from .figure_style import (
    COLOR_DIVIDER,
    COLOR_GRID,
    LW_DIVIDER,
    LW_ERRORBAR,
    LW_GRID,
    LW_LINE_PLOT,
    LW_LINE_SECONDARY,
    NEUTRAL_DARK,
    NEUTRAL_LIGHT,
    NEUTRAL_MID,
    SIZE_PANEL_LABEL,
    compact_colorbar_ticks,
    condition_colors,
    mark_missing,
    model_alphas,
    model_colors,
)
from .multiscale_v2 import decompose
from .multiscale_wavelet_panels import (
    draw_multiscale_components,
    draw_multiscale_fidelity,
    draw_sensor_efficiency,
)
from .config import RESULTS_DIR
from .io_utils import matching_or_latest
from .panels_de_data import (
    load_canonical_truth_snapshot,
    normalize_recipe_keys,
)
from .panel_c_tuning import resolve_panel_c_roi, resolve_panel_c_selection
from .representative_snapshots import resolve_panel_snapshot
from .publication_panels import (
    PublicationContext, _draw_field, _error_norm, _field_norm, _horizontal_colorbar,
    _inset, _int, _float, _roi_mask, _structured, draw_panel_a as draw_legacy_panel_a,
)
from .rendering import automatic_integrated_gradient_roi
from .statistics import relative_l2


PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_ResolutionProtocol",
    "b": "Panel_b_AllRecipeAccuracy",
    "c": "Panel_c_SelectedQualitative",
    "d": "Panel_d_SensorSweep",
    "e": "Panel_e_MultiscaleComponents",
    "f": "Panel_f_MultiscaleFidelity",
}


ALL_RECIPES = [
    "1_H_only", "2_H_limited", "3_Mixed_HML",
    "4_ZeroH_Balanced", "5_ZeroH_MRich",
]

# =============================================================================
# PANELS A--B INTERNAL LAYOUT USER TUNING API
# =============================================================================
# Panel A's field/bar coordinates are the resolution_protocol.image_layout and
# resolution_protocol.bar_bounds entries in the main project config. Panel B's
# parent-relative legend and missing-data positions are centralized here. Both
# destinations always call the same publication render profile.
PANEL_B_LAYOUT = {
    "missing_text_y": 0.82,
    "legend_location": "upper center",
    "legend_anchor_y": 0.995,
    "legend_column_spacing": 0.70,
    "legend_handle_text_pad": 0.25,
    "legend_label_spacing": 0.25,
}
# Panel C is fully controlled by common/panel_c_tuning.py. Panels D--F are
# controlled at the front of common/multiscale_wavelet_panels.py.
# =============================================================================
# END PANELS A--B INTERNAL LAYOUT USER TUNING API
# =============================================================================


def _rich_recipe_contract(ctx):
    """Resolve the zero-H rich label from audited manifest L:M:H ratios."""
    cached = getattr(ctx, "_rich_recipe_contract", None)
    if cached is not None:
        return cached
    recipe = "5_ZeroH_MRich"
    contracts = {}
    for row in ctx.manifest_rows:
        if row.get("status") != "ok" or row.get("recipe") != recipe or not row.get("manifest_path"):
            continue
        path = Path(row["manifest_path"])
        manifest = json.loads(path.read_text(encoding="utf-8"))
        ratio = str(manifest.get("multires_ratio", ""))
        parts = ratio.split(":")
        if len(parts) != 3:
            continue
        l_count, m_count, h_count = map(float, parts)
        contracts[ratio] = {
            "ratio_l_m_h": ratio,
            "counts_l_m_h": [l_count, m_count, h_count],
            "source_manifest": str(path.resolve()),
        }
    if len(contracts) != 1:
        raise RuntimeError(f"Expected one audited rich-recipe ratio, found {sorted(contracts)}")
    contract = next(iter(contracts.values()))
    l_count, m_count, _ = contract["counts_l_m_h"]
    if m_count > l_count:
        label = "Zero-H-M-rich"
    elif l_count > m_count:
        label = "Zero-H-L-rich"
    else:
        label = "Zero-H-balanced"
    contract["recipe"] = recipe
    contract["resolved_label"] = label
    ctx.v2["recipes"]["short_labels"][recipe] = label
    ctx._rich_recipe_contract = contract
    return contract


def panel_label(ax, label, cfg, x=-0.025, y=1.015, ha="right", va="bottom"):
    manuscript.tag_font_role(
        ax.text(
            x, y, label, transform=ax.transAxes, ha=ha, va=va,
            fontweight="bold", color=NEUTRAL_DARK,
        ),
        "panel_label",
    )


def model_legend_handles(cfg):
    colors = model_colors(cfg)
    alphas = model_alphas(cfg)
    return [
        Line2D([], [], color=colors[item["key"]], alpha=alphas[item["key"]],
               lw=LW_LINE_PLOT, ls="-", marker="o", ms=3.8,
               label=item["label"])
        for item in cfg["models"]
    ]


def _recipe_handles(ctx):
    cfg, v2 = ctx.cfg, ctx.v2
    colors = condition_colors(cfg)
    handles = []
    for recipe in v2["recipes"]["order"]:
        marker = v2["recipes"]["markers"][recipe]
        handles.append(Line2D(
            [], [], marker=marker, ls="none", ms=4.4,
            mfc="white" if recipe == "1_H_only" else colors[recipe],
            mec=colors[recipe], mew=1.0,
            label=v2["recipes"]["short_labels"][recipe],
        ))
    return handles


def _find(rows, model, recipe, metric=None, count=None):
    for row in rows:
        if row.get("model") != model or row.get("recipe") != recipe:
            continue
        if metric is not None and row.get("metric") != metric:
            continue
        if count is not None and _int(row.get("sensor_count")) != int(count):
            continue
        return row
    return None


def _valid_summary(row):
    return (
        row is not None
        and row.get("status", "ok") in {"", "ok"}
        and np.isfinite(_float(row.get("mean")))
    )


def draw_panel_a(parent, ctx, *, standalone=False, show_legend=True, **_):
    return draw_legacy_panel_a(parent, ctx, standalone=standalone)


def draw_panel_b(parent, ctx, *, standalone=False, show_legend=True, **_):
    cfg, v2 = ctx.cfg, ctx.v2
    count = int(v2["panel_b"].get("sensor_count", cfg["sensor_plan"]["default_count"]))
    formal_count = int(cfg["sensor_plan"]["default_count"])
    source_prefix = (
        "AllRecipeAccuracy_summary"
        if count == formal_count else "SensorSweepAllRecipes_summary"
    )
    path, rows = ctx.source("UnifiedPublicationV2", source_prefix)
    recipes = list(v2["recipes"]["order"])
    colors = condition_colors(cfg)
    models = ctx.model_order()
    offsets = np.linspace(-float(v2["panel_b"]["group_offset"]), float(v2["panel_b"]["group_offset"]), len(recipes))
    missing = []
    values = []
    for i, model in enumerate(models):
        points = {}
        for offset, recipe in zip(offsets, recipes):
            row = _find(
                rows, model, recipe, "physical_rel_l2",
                count=count if source_prefix == "SensorSweepAllRecipes_summary" else None,
            )
            if not _valid_summary(row):
                missing.append({"model": model, "recipe": recipe, "status": (row or {}).get("status", "missing")})
                continue
            mean = _float(row["mean"]); lo = _float(row["ci95_low"]); hi = _float(row["ci95_high"])
            marker = v2["recipes"]["markers"][recipe]
            parent.errorbar(
                i + offset, mean, yerr=[[mean - lo], [hi - mean]], marker=marker, ls="none",
                ms=float(v2["panel_b"]["marker_size"]), capsize=1.8,
                elinewidth=LW_ERRORBAR, capthick=LW_ERRORBAR,
                mfc="white" if recipe == "1_H_only" else colors[recipe],
                mec=colors[recipe], mew=1.0, ecolor=colors[recipe], zorder=3,
            )
            points[recipe] = (i + offset, mean)
            values.extend([lo, hi])
        for pair, linestyle in ((["2_H_limited", "3_Mixed_HML"], "-"), (["4_ZeroH_Balanced", "5_ZeroH_MRich"], "--")):
            if all(recipe in points for recipe in pair):
                parent.plot(
                    [points[recipe][0] for recipe in pair], [points[recipe][1] for recipe in pair],
                    color=COLOR_DIVIDER, lw=LW_LINE_SECONDARY, ls=linestyle, zorder=1,
                )
    parent.set_yscale(v2["panel_b"].get("yscale", "log"))
    if values:
        lo = max(min(values) * .72, 1e-5); hi = max(values) * 1.55
        parent.set_ylim(lo, hi)
    parent.set_xlim(-.48, len(models) - .52)
    model_tick_labels = [
        ctx.model_label(model).replace("FFM-Perceiver", "FFM-\nPerceiver")
        for model in models
    ]
    parent.set_xticks(range(len(models)), model_tick_labels, rotation=0, ha="center")
    parent.tick_params(axis="x", pad=1.5)
    parent.set_ylabel("Physical relative $L_2$\n" f"({count} sensors)", labelpad=0)
    parent.grid(axis="y", which="major", color=COLOR_GRID, lw=LW_GRID, zorder=0)
    fully_missing_recipes = [
        recipe for recipe in recipes
        if all(not _valid_summary(_find(
            rows, model, recipe, "physical_rel_l2",
            count=count if source_prefix == "SensorSweepAllRecipes_summary" else None,
        )) for model in models)
    ]
    if fully_missing_recipes:
        labels = ", ".join(v2["recipes"]["short_labels"][recipe] for recipe in fully_missing_recipes)
        parent.text(
            .02, float(PANEL_B_LAYOUT["missing_text_y"]),
            f"Missing at {count} sensors: {labels}",
            transform=parent.transAxes, ha="left", va="top",
            fontsize=max(4.2, cfg["figure_style"]["font_sizes"]["tick"] - 1.3),
            color=NEUTRAL_MID,
        )
    if show_legend:
        parent.legend(
            handles=_recipe_handles(ctx), ncol=int(v2["panel_b"].get("legend_columns", 3)),
            loc=str(PANEL_B_LAYOUT["legend_location"]),
            bbox_to_anchor=(.5, float(PANEL_B_LAYOUT["legend_anchor_y"])), borderaxespad=0,
            columnspacing=float(PANEL_B_LAYOUT["legend_column_spacing"]),
            handletextpad=float(PANEL_B_LAYOUT["legend_handle_text_pad"]),
            labelspacing=float(PANEL_B_LAYOUT["legend_label_spacing"]),
        )
    return {
        "status": "ok", "sources": [str(path)], "models": models, "recipes": recipes,
        "sensor_count": count, "source_table": source_prefix,
        "fully_missing_recipes": fully_missing_recipes,
        "axis_scale": "log", "statistic": "mean with bootstrap 95% CI", "missing": missing,
        "connections": [["2_H_limited", "3_Mixed_HML", "solid"], ["4_ZeroH_Balanced", "5_ZeroH_MRich", "dashed"]],
    }


def _local_render_cfg(cfg, contour_levels):
    out = dict(cfg)
    out["rendering"] = dict(cfg["rendering"])
    out["rendering"]["contour_levels"] = int(contour_levels)
    return out


def _reference_observations(ax, coords, truth, obs_indices, roi, cmap, norm, cfg):
    obs = np.asarray(obs_indices, dtype=int)
    keep = _roi_mask(np.asarray(coords)[obs], roi)
    shown = obs[keep]
    ax.set_facecolor("#FAFAF7")
    if len(shown):
        ax.scatter(np.asarray(coords)[shown, 0], np.asarray(coords)[shown, 1], c=np.asarray(truth)[shown],
                   cmap=cmap, norm=norm, marker="s", s=7.0, linewidths=0, rasterized=True)
        ax.scatter(np.asarray(coords)[shown, 0], np.asarray(coords)[shown, 1], marker="o", s=10.0,
                   facecolors="none", edgecolors=cfg["rendering"]["sensor_color"], linewidths=.45)
    ax.set_xlim(roi[0], roi[1]); ax.set_ylim(roi[2], roi[3])
    ax.set_gid("geometric-field")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color(NEUTRAL_DARK); spine.set_linewidth(LW_GRID)
    return int(len(shown))


def _qualitative_payload(ctx, strips, snapshot, count):
    payload = {}
    for strip in strips:
        for model in strip["models"]:
            for recipe in strip["recipes"]:
                payload[(model, recipe)] = ctx.cache(model, recipe, snapshot, count)
    return payload


def _draw_panel_c_version1(parent, ctx, *, standalone=False, show_legend=True, version=1, **_):
    cfg, v2 = ctx.cfg, ctx.v2
    parent.set_axis_off()
    cache_before = set(ctx.used_cache_paths)
    version = int(version or v2["panel_c"]["default_version"])
    spec = v2["panel_c"][f"version_{version}"]
    strips = spec["strips"]
    default_snapshot = ctx.representatives.get(
        v2["panel_c"].get("snapshot_block", "questionA"),
        _int(cfg["canonical_test"]["representative_snapshot"]),
    )
    shared_snapshot, snapshot_selection = resolve_panel_snapshot(v2, "c", default_snapshot)
    snapshot, count = resolve_panel_c_selection(
        shared_snapshot,
        int(v2["panel_c"].get("sensor_count", cfg["sensor_plan"]["default_count"])),
        version,
    )
    payload = _qualitative_payload(ctx, strips, snapshot, count)
    available = [loaded for loaded in payload.values() if loaded is not None]
    if not available:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "snapshot": snapshot}
    arrays0, meta0, _ = available[0]
    coords = arrays0["coords_phys"]
    truth = arrays0["truth_phys"].reshape(-1)
    predictions = [loaded[0]["recon_phys"].reshape(-1) for loaded in available]
    cmap, norm, field_limits = _field_norm(np.concatenate([truth, *predictions]), cfg)
    errors = [np.abs(pred - truth) for pred in predictions]
    error_cmap, error_norm, error_limits = _error_norm(np.concatenate(errors), cfg)
    roi, roi_metadata = resolve_panel_c_roi(
        coords, truth, automatic_integrated_gradient_roi,
        automatic_fraction=float(v2["panel_c"].get("roi_fraction", .25)),
    )
    render_cfg = _local_render_cfg(cfg, v2["panel_c"].get("contour_levels", 16))
    full_truth_sensors = (
        arrays0["obs_indices"]
        if bool(v2["panel_c"].get("show_sensors_on_full_ground_truth", False))
        else None
    )
    gap_x = .004
    strip_gap = .055 if len(strips) > 1 else 0.0
    top = .965
    bottom = .085
    strip_height = (top - bottom - strip_gap * (len(strips) - 1)) / len(strips)
    field_artist = error_artist = None
    missing = []
    shown_sensors = []
    for strip_index, strip in enumerate(strips):
        models = strip["models"]; recipes = strip["recipes"]
        columns = [(None, None)] + [(model, recipe) for model in models for recipe in recipes]
        left, right = .012, .99
        col_width = (right - left - gap_x * (len(columns) - 1)) / len(columns)
        strip_top = top - strip_index * (strip_height + strip_gap)
        header_model_y = strip_top
        if len(strips) > 1:
            header_recipe_y = strip_top - .027
            image_top = strip_top - .073
        else:
            header_recipe_y = strip_top - .038
            image_top = strip_top - .102
        image_bottom = strip_top - strip_height + .035 * strip_height
        row_gap = .012 * strip_height
        row_height = (image_top - image_bottom - 2 * row_gap) / 3
        parent.text(left + col_width / 2, header_model_y, "Reference", transform=parent.transAxes,
                    ha="center", va="top", fontsize=cfg["figure_style"]["font_sizes"]["title"] - .3, fontweight="bold")
        parent.text(left + col_width / 2, header_recipe_y, "Ground truth", transform=parent.transAxes,
                    ha="center", va="top", fontsize=cfg["figure_style"]["font_sizes"]["tick"] - .2)
        cursor = 1
        for model in models:
            span_left = left + cursor * (col_width + gap_x)
            span_right = left + (cursor + len(recipes) - 1) * (col_width + gap_x) + col_width
            parent.text((span_left + span_right) / 2, header_model_y, ctx.model_label(model), transform=parent.transAxes,
                        ha="center", va="top", fontsize=cfg["figure_style"]["font_sizes"]["title"] - .3, fontweight="bold")
            for recipe_index, recipe in enumerate(recipes):
                x = left + (cursor + recipe_index) * (col_width + gap_x) + col_width / 2
                parent.text(x, header_recipe_y, v2["recipes"]["short_labels"][recipe], transform=parent.transAxes,
                            ha="center", va="top", fontsize=max(4.5, cfg["figure_style"]["font_sizes"]["tick"] - .9))
            cursor += len(recipes)
        for col_index, (model, recipe) in enumerate(columns):
            x = left + col_index * (col_width + gap_x)
            axes = [
                _inset(parent, [x, image_top - (row + 1) * row_height - row * row_gap, col_width, row_height])
                for row in range(3)
            ]
            if model is None:
                field_artist = _draw_field(axes[0], coords, truth, render_cfg, cmap=cmap, norm=norm,
                                           mode="native_cells", contours=True, sensor_indices=full_truth_sensors,
                                           roi=roi, draw_roi=True)
                _draw_field(axes[1], coords, truth, render_cfg, cmap=cmap, norm=norm,
                            mode="native_cells", contours=True, sensor_indices=arrays0["obs_indices"], crop=roi)
                shown_sensors.append(_reference_observations(
                    axes[2], coords, truth, arrays0["obs_indices"], roi, cmap, norm, render_cfg,
                ))
                if strip_index == 0:
                    labels = ["Full H field", "Common ROI", "Observed values"] if len(strips) == 1 else ["Full", "ROI", "|error|"]
                    for ax, label in zip(axes, labels):
                        ax.text(-.075, .5, label, transform=ax.transAxes, rotation=90, ha="center", va="center",
                                fontsize=max(4.3, cfg["figure_style"]["font_sizes"]["tick"] - 1.1), color=NEUTRAL_DARK)
                continue
            loaded = payload[(model, recipe)]
            if loaded is None:
                status = ctx.cache_status(model, recipe, snapshot, count)
                for ax in axes:
                    mark_missing(ax, f"Missing\n{status}", cfg)
                missing.append({"model": model, "recipe": recipe, "status": status})
                continue
            pred = loaded[0]["recon_phys"].reshape(-1)
            _draw_field(axes[0], coords, pred, render_cfg, cmap=cmap, norm=norm,
                        mode="native_cells", contours=True, roi=roi, draw_roi=True)
            _draw_field(axes[1], coords, pred, render_cfg, cmap=cmap, norm=norm,
                        mode="native_cells", contours=True, crop=roi)
            err = np.abs(pred - truth)
            error_artist = _draw_field(axes[2], coords, err, render_cfg, cmap=error_cmap, norm=error_norm,
                                       mode="native_cells", crop=roi)
            axes[2].text(.03, .04, f"L2={relative_l2(truth, pred):.3f}", transform=axes[2].transAxes,
                         ha="left", va="bottom", color="white",
                         fontsize=max(4.4, cfg["figure_style"]["font_sizes"]["tick"] - 1.1),
                         bbox=dict(boxstyle="round,pad=.07", fc="black", ec="none", alpha=.48))
        if strip_index < len(strips) - 1:
            divider_y = strip_top - strip_height - strip_gap / 2
            parent.plot(
                [.01, .99], [divider_y, divider_y], transform=parent.transAxes,
                color=COLOR_DIVIDER, lw=LW_DIVIDER, clip_on=False,
            )
    if field_artist is not None:
        _horizontal_colorbar(parent, field_artist, [.27, .018, .20, .012], meta0.get("selected_raw_field_name", "field"), cfg)
    if error_artist is not None:
        _horizontal_colorbar(parent, error_artist, [.59, .018, .20, .012], "Absolute error", cfg)
    return {
        "status": "ok", "version": version, "cache_sources": sorted(ctx.used_cache_paths - cache_before),
        "models": sorted({model for strip in strips for model in strip["models"]}),
        "recipes": sorted({recipe for strip in strips for recipe in strip["recipes"]}, key=ALL_RECIPES.index),
        "snapshot": snapshot, "snapshot_selection": snapshot_selection,
        "case_id": meta0.get("case_id"), "time_index": meta0.get("time_index"),
        "field": meta0.get("selected_raw_field_name"), "sensor_count": count,
        "roi": {"xmin": roi[0], "xmax": roi[1], "ymin": roi[2], "ymax": roi[3],
                **roi_metadata},
        "field_limits": field_limits, "error_limits": error_limits,
        "field_cmap": cmap, "error_cmap": error_cmap, "contour_levels": render_cfg["rendering"]["contour_levels"],
        "sensors_on_full_ground_truth": full_truth_sensors is not None,
        "missing": missing, "reference_roi_sensor_counts": shown_sensors,
    }


def _box_field_axis(ax, *, bordered: bool):
    """Keep a black frame only on Panel C's zoomed-in middle row."""
    ax.set_gid("panel-c-image-cell")
    for spine in ax.spines.values():
        spine.set_visible(bool(bordered))
        spine.set_color(NEUTRAL_DARK)
        spine.set_linewidth(LW_DIVIDER)


def _compact_inset_grid(parent, n_data_columns, layout_cfg, *, standalone=False):
    """Build the qualitative grid entirely in local parent coordinates."""
    left = float(layout_cfg.get("grid_left", .030))
    right = float(layout_cfg.get("grid_right", .925))
    bottom = float(layout_cfg.get("grid_bottom", .055))
    top_key = "grid_top_standalone" if standalone else "grid_top_composite"
    top = float(layout_cfg.get(top_key, layout_cfg.get("grid_top", .805)))
    colorbar_ratio = float(layout_cfg.get("colorbar_width_ratio", .085))
    field_colorbar_length_ratio = float(layout_cfg.get("field_colorbar_length_ratio", 1.0))
    error_colorbar_length_ratio = float(layout_cfg.get("error_colorbar_length_ratio", 1.0))
    if not 0.0 < field_colorbar_length_ratio <= 1.0:
        raise ValueError("panel_c field colorbar length ratio must be in (0, 1]")
    if not 0.0 < error_colorbar_length_ratio <= 1.0:
        raise ValueError("panel_c error colorbar length ratio must be in (0, 1]")
    column_gap = float(layout_cfg.get("column_gap", .055))
    # User-facing IMAGE_ROW_GAP is injected here by panel_c_tuning.py.
    row_gap = float(layout_cfg.get("row_gap", .105))

    width_ratios = [1.0] * n_data_columns + [colorbar_ratio]
    unit_width = (right - left) / (sum(width_ratios) + column_gap * (len(width_ratios) - 1))
    x_positions, cursor = [], left
    for ratio in width_ratios:
        x_positions.append(cursor)
        cursor += ratio * unit_width + column_gap * unit_width

    unit_height = (top - bottom) / (3.0 + 2.0 * row_gap)
    row_bottoms = [top - (row + 1) * unit_height - row * row_gap * unit_height for row in range(3)]
    image_axes = [
        [parent.inset_axes([x_positions[col], row_bottoms[row], unit_width, unit_height])
         for col in range(n_data_columns)]
        for row in range(3)
    ]
    cbar_x = x_positions[-1]
    cbar_width = colorbar_ratio * unit_width
    field_full_height = 2.0 * unit_height + row_gap * unit_height
    field_height = field_full_height * field_colorbar_length_ratio
    field_bottom = row_bottoms[1] + 0.5 * (field_full_height - field_height)
    error_height = unit_height * error_colorbar_length_ratio
    error_bottom = row_bottoms[2] + 0.5 * (unit_height - error_height)
    field_cax = parent.inset_axes([cbar_x, field_bottom, cbar_width, field_height])
    error_cax = parent.inset_axes([cbar_x, error_bottom, cbar_width, error_height])
    return image_axes, field_cax, error_cax, x_positions, unit_width


def _draw_panel_c_compact(parent, ctx, *, standalone=False, show_legend=True, version=2, **_):
    """Compact Version-2 plate with figure-level grouping and vertical bars."""
    cfg, v2 = ctx.cfg, ctx.v2
    rich_recipe = _rich_recipe_contract(ctx)
    fig = parent.figure
    parent.set_axis_off()
    cache_before = set(ctx.used_cache_paths)
    strip = v2["panel_c"]["version_2"]["strips"][0]
    models, recipes = list(strip["models"]), list(strip["recipes"])
    default_snapshot = ctx.representatives.get(
        v2["panel_c"].get("snapshot_block", "questionA"),
        _int(cfg["canonical_test"]["representative_snapshot"]),
    )
    shared_snapshot, snapshot_selection = resolve_panel_snapshot(v2, "c", default_snapshot)
    snapshot, count = resolve_panel_c_selection(
        shared_snapshot,
        int(v2["panel_c"].get("sensor_count", cfg["sensor_plan"]["default_count"])),
        version,
    )
    payload = _qualitative_payload(ctx, [strip], snapshot, count)
    available = [loaded for loaded in payload.values() if loaded is not None]
    if not available:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "version": 2, "snapshot": snapshot}

    arrays0, meta0, _ = available[0]
    coords = arrays0["coords_phys"]
    truth = arrays0["truth_phys"].reshape(-1)
    predictions = [loaded[0]["recon_phys"].reshape(-1) for loaded in available]
    cmap, norm, field_limits = _field_norm(np.concatenate([truth, *predictions]), cfg)
    errors = [np.abs(pred - truth) for pred in predictions]
    error_cmap, error_norm, error_limits = _error_norm(np.concatenate(errors), cfg)
    roi, roi_metadata = resolve_panel_c_roi(
        coords, truth, automatic_integrated_gradient_roi,
        automatic_fraction=float(v2["panel_c"].get("roi_fraction", .25)),
    )
    render_cfg = _local_render_cfg(cfg, v2["panel_c"].get("contour_levels", 16))
    full_truth_sensors = (
        arrays0["obs_indices"]
        if bool(v2["panel_c"].get("show_sensors_on_full_ground_truth", False))
        else None
    )

    columns = [(None, None)] + [(model, recipe) for model in models for recipe in recipes]
    layout_cfg = v2["panel_c"].get("compact_layout", {})
    image_axes, field_cax, error_cax, x_positions, unit_width = _compact_inset_grid(
        parent, len(columns), layout_cfg, standalone=standalone,
    )

    scheme_header_size = float(layout_cfg.get("scheme_header_fontsize", 5.2))

    row_labels = ["Full H-resolution field", "Zoomed-in region", ""]
    display_row_labels = (
        row_labels if standalone
        else ["Full H-resolution\nfield", "Zoomed-in\nregion", ""]
    )
    # User-facing LEFT_ROW_LABEL_FONTSIZE_PT controls all three left labels.
    row_label_size = float(layout_cfg.get(
        "side_label_fontsize",
        max(6.4, cfg["figure_style"]["font_sizes"]["axis"] - .2) if standalone
        else max(5.8, cfg["figure_style"]["font_sizes"]["axis"] - 1.0),
    ))
    for ax, label in zip([row[0] for row in image_axes], display_row_labels):
        ax.set_ylabel(label, fontsize=row_label_size, labelpad=4.0 if standalone else 2.5, color=NEUTRAL_DARK)
        manuscript.tag_font_role(ax.yaxis.label, "axis_label", size_pt=row_label_size)

    field_artist = error_artist = None
    missing = []
    shown_sensors = 0
    for col_index, (model, recipe) in enumerate(columns):
        axes = [image_axes[row][col_index] for row in range(3)]
        if model is None:
            field_artist = _draw_field(
                axes[0], coords, truth, render_cfg, cmap=cmap, norm=norm,
                mode="native_cells", contours=True, sensor_indices=full_truth_sensors,
                roi=roi, draw_roi=True,
            )
            _draw_field(
                axes[1], coords, truth, render_cfg, cmap=cmap, norm=norm,
                mode="native_cells", contours=True, crop=roi,
            )
            shown_sensors = _reference_observations(
                axes[2], coords, truth, arrays0["obs_indices"], roi, cmap, norm, render_cfg,
            )
        else:
            loaded = payload[(model, recipe)]
            if loaded is None:
                status = ctx.cache_status(model, recipe, snapshot, count)
                for ax in axes:
                    mark_missing(ax, f"Missing\n{status}", cfg)
                missing.append({"model": model, "recipe": recipe, "status": status})
            else:
                pred = loaded[0]["recon_phys"].reshape(-1)
                _draw_field(
                    axes[0], coords, pred, render_cfg, cmap=cmap, norm=norm,
                    mode="native_cells", contours=True, roi=roi, draw_roi=True,
                )
                _draw_field(
                    axes[1], coords, pred, render_cfg, cmap=cmap, norm=norm,
                    mode="native_cells", contours=True, crop=roi,
                )
                err = np.abs(pred - truth)
                error_artist = _draw_field(
                    axes[2], coords, err, render_cfg, cmap=error_cmap,
                    norm=error_norm, mode="native_cells", crop=roi,
                )
                relative_l2_label = axes[2].text(
                    .04, .05, rf"Rel. $L_2$ = {relative_l2(truth, pred):.3f}",
                    transform=axes[2].transAxes, ha="left", va="bottom", color="white",
                    fontsize=float(layout_cfg.get(
                        "relative_l2_fontsize",
                        max(4.3, cfg["figure_style"]["font_sizes"]["tick"] - 1.2),
                    )),
                    bbox=dict(boxstyle="round,pad=.10", fc="black", ec="none", alpha=.58),
                )
                manuscript.tag_font_role(
                    relative_l2_label, "annotation",
                    size_pt=float(layout_cfg.get(
                        "relative_l2_fontsize",
                        max(4.3, cfg["figure_style"]["font_sizes"]["tick"] - 1.2),
                    )),
                )
        for row_index, ax in enumerate(axes):
            _box_field_axis(
                ax,
                bordered=(row_index == 1 or (row_index == 2 and col_index == 0)),
            )

    field_ticks = np.linspace(field_limits[0], field_limits[1], 4)
    error_ticks = np.linspace(error_limits[0], error_limits[1], 4)
    # Both colorbars share the user-facing typography controls.  Their axes are
    # anchored by GRID_RIGHT; smaller text therefore cannot move the right edge.
    colorbar_tick_size = float(layout_cfg.get(
        "colorbar_tick_fontsize", cfg["figure_style"]["font_sizes"]["tick"] - .7,
    ))
    colorbar_label_size = float(layout_cfg.get(
        "colorbar_label_fontsize", max(5.6, cfg["figure_style"]["font_sizes"]["axis"] - 1.0),
    ))
    field_tick_format = error_tick_format = None
    if field_artist is not None:
        field_cb = fig.colorbar(field_artist, cax=field_cax, orientation="vertical")
        field_cb.ax.tick_params(labelsize=colorbar_tick_size, length=2, pad=1.5)
        field_cb.set_ticks(field_ticks)
        field_cb.set_label(
            "Field value", rotation=270,
            labelpad=float(layout_cfg.get("colorbar_labelpad", 7.0)),
            fontsize=colorbar_label_size,
        )
        manuscript.tag_font_role(
            field_cb.ax.yaxis.label, "axis_label", size_pt=colorbar_label_size,
        )
        field_tick_format = compact_colorbar_ticks(
            field_cb, field_ticks, target_top_axis=image_axes[0][0],
            tick_size_pt=colorbar_tick_size,
        )
        for tick_label in field_cb.ax.get_yticklabels():
            manuscript.tag_font_role(tick_label, "tick_label", size_pt=colorbar_tick_size)
        field_cb.ax.yaxis.set_label_position("right")
    else:
        field_cax.set_axis_off()
    if error_artist is not None:
        error_cb = fig.colorbar(error_artist, cax=error_cax, orientation="vertical")
        error_cb.ax.tick_params(labelsize=colorbar_tick_size, length=2, pad=1.5)
        error_cb.set_ticks(error_ticks)
        error_cb.set_label(
            "Absolute error", rotation=270,
            labelpad=float(layout_cfg.get("colorbar_labelpad", 7.0)),
            fontsize=colorbar_label_size,
        )
        manuscript.tag_font_role(
            error_cb.ax.yaxis.label, "axis_label", size_pt=colorbar_label_size,
        )
        error_tick_format = compact_colorbar_ticks(
            error_cb, error_ticks, target_top_axis=image_axes[2][0],
            tick_size_pt=colorbar_tick_size, preserve_bar_bottom=True,
        )
        for tick_label in error_cb.ax.get_yticklabels():
            manuscript.tag_font_role(tick_label, "tick_label", size_pt=colorbar_tick_size)
        error_cb.ax.yaxis.set_label_position("right")
    else:
        error_cax.set_axis_off()

    group_spans = [("Reference", 0, 0)]
    cursor = 1
    for model in models:
        group_spans.append((ctx.model_label(model), cursor, cursor + len(recipes) - 1))
        cursor += len(recipes)
    model_header_key = "model_header_y" if standalone else "model_header_y_composite"
    recipe_header_key = "recipe_header_y" if standalone else "recipe_header_y_composite"
    model_header_y = float(layout_cfg.get(
        model_header_key,
        layout_cfg.get("header_y" if standalone else "header_y_composite", .97),
    ))
    recipe_header_y = float(layout_cfg.get(recipe_header_key, model_header_y - .045))
    model_header_size = float(layout_cfg.get("model_header_fontsize", 5.8))
    header_positions = {}
    for label, first, last in group_spans:
        center_x = .5 * (x_positions[first] + x_positions[last] + unit_width)
        header_positions[label] = [center_x, model_header_y]
        header = fig.text(
            center_x, model_header_y, label, transform=parent.transAxes,
            ha="center", va="center", fontweight="bold",
            gid="panel-c-model-header",
        )
        manuscript.tag_font_role(header, "subplot_title", size_pt=model_header_size)

    recipe_titles = ["Ground truth"] + [
        v2["recipes"]["short_labels"][recipe]
        for model in models for recipe in recipes
    ]
    # At final publication size these labels fit within their columns.  Forced
    # wrapping makes the second line collide with the image grid and pushes the
    # first line into the model-header row, so keep every recipe title on one
    # centered line.
    display_titles = recipe_titles
    recipe_header_positions = {}
    for title, display_title, xpos in zip(recipe_titles, display_titles, x_positions):
        center_x = xpos + .5 * unit_width
        recipe_header_positions.setdefault(title, []).append([center_x, recipe_header_y])
        header = fig.text(
            center_x, recipe_header_y, display_title, transform=parent.transAxes,
            ha="center", va="top", fontweight="normal",
            gid="panel-c-recipe-header",
        )
        manuscript.tag_font_role(header, "subplot_title", size_pt=scheme_header_size)

    sensor_layout_title = image_axes[2][0].set_title("Sensor layout", pad=.6)
    manuscript.tag_font_role(sensor_layout_title, "axis_label", size_pt=scheme_header_size)
    absolute_error_title = image_axes[2][len(columns) // 2].set_title("Absolute error", pad=.6)
    manuscript.tag_font_role(absolute_error_title, "axis_label", size_pt=scheme_header_size)

    divider_x = []
    divider_y_min = float(layout_cfg.get("vertical_divider_y_min", .05))
    divider_y_max = float(layout_cfg.get("vertical_divider_y_max", recipe_header_y - .012))
    for left_col, right_col in ((0, 1), (3, 4), (6, 7)):
        xpos = .5 * (x_positions[left_col] + unit_width + x_positions[right_col])
        divider_x.append(xpos)
        fig.add_artist(Line2D(
            [xpos, xpos],
            [divider_y_min, divider_y_max], transform=parent.transAxes,
            color=COLOR_DIVIDER, linewidth=LW_DIVIDER,
            linestyle="-", solid_capstyle="butt", zorder=20,
        ))

    return {
        "status": "ok", "version": 2,
        "cache_sources": sorted(ctx.used_cache_paths - cache_before),
        "models": models, "recipes": recipes,
        "snapshot": snapshot, "snapshot_selection": snapshot_selection,
        "case_id": meta0.get("case_id"),
        "time_index": meta0.get("time_index"), "field": meta0.get("selected_raw_field_name"),
        "sensor_count": count,
        "roi": {"xmin": roi[0], "xmax": roi[1], "ymin": roi[2], "ymax": roi[3],
                **roi_metadata},
        "field_limits": field_limits, "error_limits": error_limits,
        "field_cmap": cmap, "error_cmap": error_cmap,
        "error_map_definition": "pointwise absolute error = abs(reconstruction - ground truth)",
        "relative_l2_annotation_format": "Rel. L_2 = {value:.3f}",
        "reference_sensor_normalization": "identical field-value cmap and normalization as field maps",
        "rich_recipe_contract": rich_recipe,
        "contour_levels": render_cfg["rendering"]["contour_levels"],
        "sensors_on_full_ground_truth": full_truth_sensors is not None,
        "sensors_on_reference_zoom": False,
        "missing": missing, "reference_roi_sensor_counts": [shown_sensors],
        "layout": {
            "type": "three-row local inset grid with dedicated right colorbar column",
            "grid_bounds_parent_fraction": {
                "left": float(layout_cfg.get("grid_left", .030)),
                "right": float(layout_cfg.get("grid_right", .925)),
                "bottom": float(layout_cfg.get("grid_bottom", .055)),
                "top": float(layout_cfg.get(
                    "grid_top_standalone" if standalone else "grid_top_composite",
                    layout_cfg.get("grid_top", .805),
                )),
            },
            "header_coordinate_system": "two fixed-height rows in parent axes coordinates via fig.text",
            "model_header_y_parent_fraction": model_header_y,
            "recipe_header_y_parent_fraction": recipe_header_y,
            "model_header_fontsize_pt": model_header_size,
            "scheme_header_fontsize_pt": scheme_header_size,
            "colorbar_label_fontsize_pt": colorbar_label_size,
            "colorbar_tick_fontsize_pt": colorbar_tick_size,
            "base_column_gap": float(layout_cfg.get("column_gap_base", layout_cfg.get("column_gap", .055))),
            "effective_column_gap": float(layout_cfg.get("column_gap", .055)),
            "image_row_gap_relative_to_row_height": float(layout_cfg.get("row_gap", .105)),
            "derived_panel_height_mm": float(layout_cfg.get("derived_panel_height_mm", np.nan)),
            "physical_height_reference_mm": float(layout_cfg.get("physical_height_reference_mm", np.nan)),
            "physical_height_reference_row_gap": float(
                layout_cfg.get("physical_height_reference_row_gap", np.nan)
            ),
            "header_positions": header_positions,
            "recipe_header_positions": recipe_header_positions,
            "vertical_divider_x_parent": divider_x,
            "vertical_divider_y_parent_fraction": [divider_y_min, divider_y_max],
            "colorbars": {
                "field_length_ratio": float(layout_cfg.get("field_colorbar_length_ratio", 1.0)),
                "error_length_ratio": float(layout_cfg.get("error_colorbar_length_ratio", 1.0)),
                "field": "vertical, centered on rows 1 and 2; exactly 4 ticks; Field value label",
                "error": "vertical, bottom-aligned on row 3 with a top multiplier cap; exactly 4 ticks; Absolute error label",
            },
            "field_colorbar_ticks": [float(value) for value in field_ticks],
            "error_colorbar_ticks": [float(value) for value in error_ticks],
            "colorbar_tick_format": {
                "field": field_tick_format,
                "error": error_tick_format,
                "mantissa_decimals": 1,
                "common_exponent_at_top": True,
            },
            "colorbar_labels": {"field": "Field value", "error": "Absolute error"},
            "row_labels": ["Full H-resolution field", "Zoomed-in region"],
            "bottom_row_headers": {"reference": "Sensor layout", "models": "Absolute error"},
            "subplot_borders": "solid black on the middle zoomed-in row and row 3 column 1",
        },
    }


def draw_panel_c(parent, ctx, *, standalone=False, show_legend=True, version=None, **kwargs):
    version = int(version or ctx.v2["panel_c"]["default_version"])
    if version == 1:
        return _draw_panel_c_version1(
            parent, ctx, standalone=standalone, show_legend=show_legend, version=1, **kwargs,
        )
    return _draw_panel_c_compact(
        parent, ctx, standalone=standalone, show_legend=show_legend, version=2, **kwargs,
    )


def _component_axis(ax, rows, ctx, metric, xscale, compact=False, xlabel=None):
    cfg, v2 = ctx.cfg, ctx.v2
    models = ctx.model_order(); recipes = v2["recipes"]["order"]
    colors = condition_colors(cfg)
    ybase = np.arange(len(models))[::-1]
    offsets = np.linspace(.22, -.22, len(recipes))
    missing = []
    for i, model in enumerate(models):
        points = {}
        for offset, recipe in zip(offsets, recipes):
            row = _find(rows, model, recipe, metric)
            if not _valid_summary(row):
                missing.append({"model": model, "recipe": recipe, "metric": metric})
                continue
            mean = _float(row["mean"]); lo = _float(row["ci95_low"]); hi = _float(row["ci95_high"])
            marker = v2["recipes"]["markers"][recipe]
            ypos = ybase[i] + offset
            ax.errorbar(mean, ypos, xerr=[[mean - lo], [hi - mean]], marker=marker, ls="none", ms=3.5,
                        capsize=1.4, elinewidth=LW_ERRORBAR,
                        mfc="white" if recipe == "1_H_only" else colors[recipe],
                        mec=colors[recipe], mew=.8, ecolor=colors[recipe], zorder=3)
            points[recipe] = (mean, ypos)
        for pair, linestyle in ((["2_H_limited", "3_Mixed_HML"], "-"), (["4_ZeroH_Balanced", "5_ZeroH_MRich"], "--")):
            if all(recipe in points for recipe in pair):
                ax.plot(
                    [points[r][0] for r in pair], [points[r][1] for r in pair],
                    color=COLOR_DIVIDER, lw=LW_LINE_SECONDARY, ls=linestyle, zorder=1,
                )
    labels = [ctx.model_label(model) for model in models]
    ax.set_yticks(ybase, labels)
    if compact:
        ax.tick_params(axis="y", labelsize=max(4.0, cfg["figure_style"]["font_sizes"]["tick"] - 1.6))
    ax.tick_params(axis="y", pad=1.5)
    ax.set_ylim(-.55, len(models) - .45)
    ax.set_xscale(xscale)
    ax.grid(axis="x", which="major", color=COLOR_GRID, lw=LW_GRID, zorder=0)
    ax.set_xlabel(xlabel or "Relative L2", labelpad=1.5)
    return missing


def _json_source(ctx, prefix):
    rid = ctx.source_run_ids.get(prefix, ctx.run_id)
    path = matching_or_latest(RESULTS_DIR / "UnifiedPublicationV2", prefix, rid, "json")
    return path, json.loads(path.read_text(encoding="utf-8"))


def draw_panel_d(parent, ctx, *, standalone=False, show_legend=True, **_):
    cfg, v2 = ctx.cfg, ctx.v2
    rich_recipe = _rich_recipe_contract(ctx)
    parent.set_axis_off()
    path, rows = ctx.source("UnifiedPublicationV2", "CoarseDetailFidelity_summary")
    metadata_path, metadata = _json_source(ctx, "CoarseDetailFidelity_metadata")
    representative = metadata.get("representative_truth", {})
    if not rows or not representative:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "sources": [str(path), str(metadata_path)]}
    coords_ordered, truth_ordered, source_shape = load_canonical_truth_snapshot(representative)
    target_shape = tuple(int(value) for value in representative["target_shape_yx"])
    coarse, detail, _ = decompose(truth_ordered, source_shape, target_shape)
    field_cmap, field_norm, field_limits = _field_norm(np.concatenate([truth_ordered, coarse]), cfg)
    detail_limit = max(float(np.nanquantile(np.abs(detail), .995)), 1e-12)
    detail_norm = TwoSlopeNorm(vmin=-detail_limit, vcenter=0, vmax=detail_limit)
    render_cfg = _local_render_cfg(cfg, v2["panel_c"].get("contour_levels", 16))
    projector = representative["projector_resolution"]
    labels = [
        (r"Original H field, $u$", truth_ordered),
        (rf"{projector}-resolvable component, $P_{{{projector}}}u$", coarse),
        (rf"H-only detail, $(I-P_{{{projector}}})u$", detail),
    ]
    parent.text(
        .205, .975, rf"$u = P_{{{projector}}}u + (I-P_{{{projector}}})u$",
        transform=parent.transAxes, ha="center", va="top",
        fontsize=cfg["figure_style"]["font_sizes"]["title"] - (0 if standalone else 1.5),
        color=NEUTRAL_DARK,
    )
    map_left = .055
    map_width = .30
    map_height = .190
    map_bottoms = [.720, .455, .150]
    artists = []
    for i, ((label, values), map_bottom) in enumerate(zip(labels, map_bottoms)):
        ax = _inset(parent, [map_left, map_bottom, map_width, map_height])
        if i < 2:
            artist = _draw_field(ax, coords_ordered, values, render_cfg, cmap=field_cmap, norm=field_norm, mode="native_cells", contours=True)
        else:
            artist = _draw_field(ax, coords_ordered, values, render_cfg, cmap=cfg["rendering"]["cmap"], norm=detail_norm, mode="native_cells", contours=True)
        artists.append(artist)
        display_label = label if standalone else label.replace(", $P_", ",\n$P_").replace(", $(I-", ",\n$(I-")
        ax.set_title(
            display_label, fontsize=cfg["figure_style"]["font_sizes"]["tick"] - (.1 if standalone else 1.5),
            pad=1.2,
        )
    field_cb = _horizontal_colorbar(
        parent, artists[0], [.085, .426, .24, .010],
        representative["selected_raw_field_name"], cfg,
    )
    detail_cb = _horizontal_colorbar(
        parent, artists[2], [.085, .105, .24, .010],
        "", cfg,
    )
    if not standalone:
        compact_tick_size = max(3.6, cfg["figure_style"]["font_sizes"]["tick"] - 2.1)
        compact_label_size = max(3.8, cfg["figure_style"]["font_sizes"]["axis"] - 2.0)
        for cb in (field_cb, detail_cb):
            cb.ax.tick_params(labelsize=compact_tick_size, length=1.4, pad=.6)
        field_cb.set_ticks(np.linspace(field_limits[0], field_limits[1], 3))
        detail_cb.set_ticks([-detail_limit, 0.0, detail_limit])
        field_cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        detail_cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
        field_cb.ax.set_title(representative["selected_raw_field_name"], fontsize=compact_label_size, pad=.7)
    ny, nx = source_shape; target_y, target_x = target_shape
    parent.text(
        .205, .055,
        rf"$P_{{{projector}}}$: {nx} × {ny} → {target_x} × {target_y} → {nx} × {ny}",
        transform=parent.transAxes, ha="center", va="center", color=NEUTRAL_DARK,
        fontsize=cfg["figure_style"]["font_sizes"]["tick"] - (.1 if standalone else .7),
    )
    parent.text(
        .205, .034, f"{projector}-resolution cutoff",
        transform=parent.transAxes, ha="center", va="center", color=NEUTRAL_MID,
        fontsize=cfg["figure_style"]["font_sizes"]["tick"] - (.3 if standalone else .8),
    )
    detail_fraction_raw = float(representative["detail_energy_fraction_true"])
    detail_fraction_percent = 100.0 * detail_fraction_raw
    parent.text(
        .205, .013, f"Detail energy fraction = {detail_fraction_percent:.4f}%",
        transform=parent.transAxes, ha="center", va="center", color=NEUTRAL_DARK,
        fontsize=cfg["figure_style"]["font_sizes"]["tick"] - (.2 if standalone else .8),
    )

    coarse_ax = _inset(parent, [.515, .565, .465, .245])
    detail_ax = _inset(parent, [.515, .125, .465, .245])
    coarse_scale = v2["panel_d"].get("coarse_xscale", "linear")
    detail_scale = v2["panel_d"].get("detail_xscale", "linear")
    missing = _component_axis(
        coarse_ax, rows, ctx, "coarse_rel_l2", coarse_scale,
        compact=not standalone, xlabel="Relative L2",
    )
    missing += _component_axis(
        detail_ax, rows, ctx, "detail_correlation", detail_scale,
        compact=not standalone, xlabel="Correlation",
    )
    title_size = cfg["figure_style"]["font_sizes"]["title"] - (0 if standalone else 1.8)
    coarse_ax.set_title("Large-scale relative error", pad=2.0, fontsize=title_size, loc="left")
    detail_ax.set_title("Fine-detail pattern correlation", pad=2.0, fontsize=title_size, loc="left")
    coarse_ax.text(
        .985, .95 if not standalone else 1.025, "lower is better", transform=coarse_ax.transAxes,
        ha="right", va="top" if not standalone else "bottom", color=NEUTRAL_MID,
        fontsize=cfg["figure_style"]["font_sizes"]["tick"] - .6,
    )
    detail_ax.text(
        .985, .74 if not standalone else 1.025, "higher is better", transform=detail_ax.transAxes,
        ha="right", va="top" if not standalone else "bottom", color=NEUTRAL_MID,
        fontsize=cfg["figure_style"]["font_sizes"]["tick"] - .6,
    )
    coarse_ci_high = [
        _float(row["ci95_high"]) for row in rows
        if row.get("metric") == "coarse_rel_l2" and _valid_summary(row)
        and row.get("model") in ctx.model_order() and row.get("recipe") in ALL_RECIPES
    ]
    coarse_upper = 1.08 * max(coarse_ci_high) if coarse_ci_high else 1.0
    coarse_ax.set_xlim(0.0, coarse_upper)
    detail_ax.axvline(1.0, color=NEUTRAL_MID, lw=LW_LINE_SECONDARY, ls=":", zorder=0)
    detail_ax.text(.985, .98, "ideal = 1", transform=detail_ax.transAxes,
                   ha="right", va="top", color=NEUTRAL_MID,
                   fontsize=cfg["figure_style"]["font_sizes"]["tick"] - .6)
    detail_ax.set_xlim(0.0, 1.02)
    if show_legend:
        parent.legend(
            handles=_recipe_handles(ctx), ncol=3 if standalone else 2,
            loc="upper center", bbox_to_anchor=(.748, .985),
            borderaxespad=0, columnspacing=.55 if standalone else .28,
            handletextpad=.18, labelspacing=.18 if standalone else .10,
            fontsize=None if standalone else max(4.0, cfg["figure_style"]["font_sizes"]["legend"] - 1.5),
        )
    return {
        "status": "ok", "sources": [str(path), str(metadata_path)],
        "models": ctx.model_order(), "recipes": ALL_RECIPES,
        "snapshot": representative["snapshot_index"], "case_id": representative["case_id"],
        "time_index": representative["time_index"], "field": representative["selected_raw_field_name"],
        "projector_resolution": projector, "source_shape_yx": list(source_shape), "target_shape_yx": list(target_shape),
        "projector": metadata.get("projector_definition"),
        "downsample_method": metadata.get("downsample_method"),
        "upsample_method": metadata.get("upsample_method"), "align_corners": metadata.get("align_corners"),
        "representative_truth": representative, "field_limits": field_limits,
        "detail_energy_fraction_raw": detail_fraction_raw,
        "detail_energy_fraction_percent": detail_fraction_percent,
        "rich_recipe_contract": rich_recipe,
        "recipe_encoding": {
            recipe: {
                "label": v2["recipes"]["short_labels"][recipe],
                "marker": v2["recipes"]["markers"][recipe],
                "color": condition_colors(cfg)[recipe],
                "fill": "open" if recipe == "1_H_only" else "filled",
            }
            for recipe in ALL_RECIPES
        },
        "detail_limits": [-detail_limit, detail_limit], "axis_scales": {
            "coarse": coarse_scale, "detail": detail_scale,
        }, "axis_limits": {"coarse": [0.0, coarse_upper], "detail": [0.0, 1.02]},
        "missing": missing,
    }


def _draw_band_metric(parent, ctx, rows, metadata, *, metric, ylabel, standalone, show_legend,
                      symmetric=False, zero_reference=False):
    cfg, v2 = ctx.cfg, ctx.v2
    rich_recipe = _rich_recipe_contract(ctx)
    models = ctx.model_order(cfg["frequency_error"].get("main_models"))
    recipes = normalize_recipe_keys(v2["panel_e"]["recipes"])
    colors = model_colors(cfg)
    alphas = model_alphas(cfg)
    selected = [
        row for row in rows if row.get("model") in models and row.get("recipe") in recipes
        and row.get("metric") == metric
    ]
    if not selected:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing"}
    band_order = ["L-resolvable", "M-only", "H-only"]
    left, gap = .085, .030
    width = (.985 - left - 2 * gap) / 3
    axes = []
    for j, recipe in enumerate(recipes):
        ax = _inset(parent, [left + j * (width + gap), .14, width, .66])
        axes.append(ax)
        if bool(v2["panel_e"].get("highlight_h_only_band", True)):
            ax.axvspan(1.5, 2.25, color=NEUTRAL_LIGHT, alpha=.22, lw=0, zorder=-5)
        for model in models:
            by_band = {
                row["band"]: row for row in selected
                if row.get("model") == model and row.get("recipe") == recipe
            }
            if not all(band in by_band for band in band_order):
                continue
            subset = [by_band[band] for band in band_order]
            x = np.arange(3, dtype=float)
            med = np.asarray([_float(row["median"]) for row in subset])
            q25 = np.asarray([_float(row["q25"]) for row in subset])
            q75 = np.asarray([_float(row["q75"]) for row in subset])
            ax.plot(
                x, med, color=colors[model], alpha=alphas[model], lw=LW_LINE_PLOT,
                ls="-", zorder=2,
            )
            ax.errorbar(
                x, med, yerr=np.vstack([med - q25, q75 - med]),
                fmt="o", ms=3.2, color=colors[model], alpha=alphas[model],
                ecolor=colors[model], elinewidth=LW_ERRORBAR, capsize=1.7,
                capthick=LW_ERRORBAR, zorder=3,
            )
        if zero_reference:
            ax.axhline(0.0, color=NEUTRAL_MID, lw=LW_LINE_SECONDARY, ls=":", zorder=0)
        ax.set_xlim(-.25, 2.25)
        ax.set_xscale("linear")
        configured_yscale = v2["panel_e"].get("yscale", "linear")
        ax.set_yscale(configured_yscale)
        ax.set_xticks(np.arange(3), ["L-\nresolvable", "M-only", "H-only"])
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_title(v2["recipes"]["short_labels"][recipe], pad=2.5)
        if j > 0:
            ax.tick_params(labelleft=False)
        if j > 0 and bool(v2["panel_e"].get("annotate_unseen_zero_h", True)):
            ax.text(
                .50, .985, "unseen during training", transform=ax.transAxes,
                ha="center", va="top", color=NEUTRAL_MID,
                fontsize=cfg["figure_style"]["font_sizes"]["tick"] - .9,
            )

    pooled = np.asarray([
        _float(row[key]) for row in selected for key in ("q25", "median", "q75")
        if np.isfinite(_float(row.get(key)))
    ], dtype=float)
    p1, p99 = np.percentile(pooled, [1.0, 99.0])
    span = max(float(p99 - p1), 1.0)
    margin = max(
        float(v2["panel_e"].get("minimum_ylim_margin_db", 1.0)),
        float(v2["panel_e"].get("ylim_margin_fraction", .05)) * span,
    )
    if zero_reference:
        data_lower = min(-5.0, float(p1 - margin))
        data_upper = max(0.0, float(p99 + margin))
    else:
        data_lower = min(0.0, float(p1 - margin))
        data_upper = float(p99 + margin)
    if symmetric:
        limit = max(abs(data_lower), abs(data_upper))
        shared_y_range = [-limit, limit]
    else:
        shared_y_range = [data_lower, data_upper]
    for ax in axes:
        ax.set_ylim(*shared_y_range)

    parent.text(
        .018, .47, ylabel, transform=parent.transAxes, rotation=90,
        ha="center", va="center", fontsize=cfg["figure_style"]["font_sizes"]["axis"],
        color=NEUTRAL_DARK,
    )
    if show_legend:
        solid_handles = [
            Line2D([], [], color=colors[model], alpha=alphas[model],
                   lw=LW_LINE_PLOT, ls="-", marker="o", ms=3.2, label=ctx.model_label(model))
            for model in models
        ]
        parent.legend(handles=solid_handles, ncol=4, loc="upper center", bbox_to_anchor=(.54, .995),
                      borderaxespad=0, columnspacing=.75, handletextpad=.25)
    if zero_reference:
        parent.text(
            .54, .855,
            "0 dB = correct band energy; positive = excess; negative = deficit",
            transform=parent.transAxes, ha="center", va="center", color=NEUTRAL_MID,
            fontsize=cfg["figure_style"]["font_sizes"]["tick"] - (.1 if standalone else .5),
        )
    return {
        "status": "ok", "models": models, "recipes": recipes, "metric": metric,
        "statistic": "median with categorical interquartile dot intervals",
        "legend_line_style": "solid", "shared_y_range": shared_y_range,
        "y_limit_calculation": {
            "pooled_values": "displayed q25, median, and q75 values",
            "percentiles": [1.0, 99.0], "p1": float(p1), "p99": float(p99),
            "margin_db": margin, "lower_rule": "min(-5, p1 - margin)" if zero_reference else "min(0, p1 - margin)",
            "upper_rule": "p99 + margin", "symmetric": bool(symmetric),
            "applied": shared_y_range,
        },
        "axis_scale": configured_yscale,
        "rich_recipe_contract": rich_recipe,
    }


def draw_panel_e(parent, ctx, *, standalone=False, show_legend=True, **_):
    parent.set_axis_off()
    path, rows = ctx.source("UnifiedPublicationV2", "SpectralBands_summary")
    metadata_path, metadata = _json_source(ctx, "SpectralBands_metadata")
    result = _draw_band_metric(
        parent, ctx, rows, metadata,
        metric=ctx.v2["panel_e"].get("main_metric", "band_energy_bias_db"),
        ylabel="Spectral-energy bias (dB)", standalone=standalone,
        show_legend=show_legend, symmetric=bool(ctx.v2["panel_e"].get("symmetric_ylim", True)),
        zero_reference=True,
    )
    result.update({
        "sources": [str(path), str(metadata_path)],
        "native_nyquist_boundaries": metadata.get("native_nyquist_boundaries"),
        "spectral_band_definitions": metadata.get("spectral_band_definitions"),
        "robust_y_limit": metadata.get("robust_y_limit"),
        "axis_scale": ctx.v2["panel_e"].get("yscale", "linear"),
        "missing": metadata.get("missing_model_recipe_entries", []),
    })
    return result


def draw_panel_d_detail_rel_l2_diagnostic(parent, ctx, *, standalone=True, show_legend=True):
    parent.set_axis_off()
    path, rows = ctx.source("UnifiedPublicationV2", "CoarseDetailFidelity_summary")
    ax = _inset(parent, [.16, .18, .75, .64])
    scale = ctx.v2["panel_d"].get("detail_xscale", "linear")
    missing = _component_axis(ax, rows, ctx, "detail_rel_l2", scale, xlabel="Relative L2")
    ax.set_title("Fine-detail relative error (diagnostic)\nLower is better", pad=3)
    if show_legend:
        parent.legend(handles=_recipe_handles(ctx), ncol=3, loc="lower center", bbox_to_anchor=(.56, .94))
    return {"status": "ok", "sources": [str(path)], "axis_scale": scale, "missing": missing}


def draw_panel_e_weighted_band_lsd(parent, ctx, *, standalone=True, show_legend=True):
    parent.set_axis_off()
    path, rows = ctx.source("UnifiedPublicationV2", "SpectralBands_summary")
    metadata_path, metadata = _json_source(ctx, "SpectralBands_metadata")
    result = _draw_band_metric(
        parent, ctx, rows, metadata, metric="weighted_band_lsd_db",
        ylabel="Weighted band LSD (dB)", standalone=standalone,
        show_legend=show_legend, symmetric=False, zero_reference=False,
    )
    result["sources"] = [str(path), str(metadata_path)]
    return result


def draw_panel_e_shellwise_diagnostic(parent, ctx, *, standalone=True, show_legend=True):
    cfg, v2 = ctx.cfg, ctx.v2
    parent.set_axis_off()
    path, rows = ctx.source("FrequencyError", "FrequencyError_summary")
    models = ctx.model_order(cfg["frequency_error"].get("main_models"))
    recipes = normalize_recipe_keys(v2["panel_e"]["recipes"])
    colors = model_colors(cfg); alphas = model_alphas(cfg)
    selected = [row for row in rows if row.get("model") in models and row.get("recipe") in recipes]
    if not selected:
        mark_missing(parent, cfg=cfg); return {"status": "missing", "sources": [str(path)]}
    ymax = float(np.nanquantile([_float(row.get("q75")) for row in selected], .995)) * 1.08
    nyq_l = _float(selected[0]["L_nyquist_normalized"]); nyq_m = _float(selected[0]["M_nyquist_normalized"])
    left, gap = .075, .035; width = (.985 - left - 2 * gap) / 3
    for j, recipe in enumerate(recipes):
        ax = _inset(parent, [left + j * (width + gap), .16, width, .68])
        for model in models:
            subset = sorted(
                [row for row in selected if row.get("model") == model and row.get("recipe") == recipe],
                key=lambda row: _int(row["shell_index"]),
            )
            if not subset: continue
            x = np.asarray([_float(row["k_normalized_H_nyquist"]) for row in subset])
            med = np.asarray([_float(row["median"]) for row in subset])
            q25 = np.asarray([_float(row["q25"]) for row in subset]); q75 = np.asarray([_float(row["q75"]) for row in subset])
            ax.plot(x, med, color=colors[model], alpha=alphas[model], lw=LW_LINE_PLOT, ls="-")
            ax.fill_between(x, q25, q75, color=colors[model], alpha=float(v2["panel_e"].get("iqr_alpha", .10)), lw=0)
        for xpos, label in ((nyq_l, "L Nyq."), (nyq_m, "M Nyq.")):
            ax.axvline(xpos, color=NEUTRAL_MID, lw=LW_LINE_SECONDARY, ls=":")
            ax.text(xpos + .008, .96, label, transform=ax.get_xaxis_transform(), rotation=90,
                    ha="left", va="top", color=NEUTRAL_DARK,
                    fontsize=cfg["figure_style"]["font_sizes"]["tick"] - .4)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, ymax); ax.set_xscale("linear"); ax.set_yscale("linear")
        ax.set_title(v2["recipes"]["short_labels"][recipe], pad=2.5)
        ax.set_xlabel(r"$k/k_{\mathrm{Nyq},H}$", labelpad=1.0)
        if j == 0: ax.set_ylabel("Absolute shell mismatch (dB)", labelpad=2)
        else: ax.tick_params(labelleft=False)
    if show_legend:
        parent.legend(handles=model_legend_handles(cfg), ncol=4, loc="lower center", bbox_to_anchor=(.54, .94))
    return {"status": "ok", "sources": [str(path)], "shared_y_range": [0, ymax]}


def draw_panel_f(parent, ctx, *, standalone=False, show_legend=True, **_):
    cfg, v2 = ctx.cfg, ctx.v2
    parent.set_axis_off()
    path, rows = ctx.source("UnifiedPublicationV2", "SensorSweepAllRecipes_summary")
    rows = [row for row in rows if row.get("metric") == "physical_rel_l2"]
    models = ctx.model_order(cfg["sensor_sweep"].get("main_models"))
    recipes = v2["panel_f"]["recipes"]
    counts = [int(value) for value in v2["panel_f"]["sensor_counts"]]
    colors = model_colors(cfg)
    alphas = model_alphas(cfg)
    valid = [row for row in rows if row.get("model") in models and row.get("recipe") in recipes
             and _int(row.get("sensor_count")) in counts and np.isfinite(_float(row.get("mean")))]
    if not valid:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "sources": [str(path)]}
    ymin = max(min(_float(row.get("ci95_low")) for row in valid) * .72, 1e-4)
    ymax = max(_float(row.get("ci95_high")) for row in valid) * 1.42
    formal = int(v2["panel_f"].get("formal_sensor_count", 256))
    grid_points = max(_int(row.get("evaluation_grid_points"), 128 * 128) for row in valid)
    density = [100 * count / grid_points for count in counts]
    left, gap = .055, .020
    width = (.99 - left - 4 * gap) / 5
    missing = []
    axis_bottom = .20 if standalone else .22
    axis_height = .62 if standalone else .61
    for j, recipe in enumerate(recipes):
        ax = _inset(parent, [left + j * (width + gap), axis_bottom, width, axis_height])
        missing_here = []
        for model in models:
            subset = sorted([row for row in valid if row.get("model") == model and row.get("recipe") == recipe], key=lambda row: _int(row["sensor_count"]))
            if not subset:
                missing_here.append(ctx.model_label(model)); missing.append({"model": model, "recipe": recipe, "status": "missing"})
                continue
            x = np.asarray([_int(row["sensor_count"]) for row in subset])
            mean = np.asarray([_float(row["mean"]) for row in subset])
            lo = np.asarray([_float(row["ci95_low"]) for row in subset]); hi = np.asarray([_float(row["ci95_high"]) for row in subset])
            ax.plot(
                x, mean, color=colors[model], alpha=alphas[model], marker="o",
                ms=3.0, lw=LW_LINE_PLOT, ls="-",
            )
            ax.errorbar(
                x, mean, yerr=[mean - lo, hi - mean], fmt="none",
                ecolor=colors[model], alpha=alphas[model],
                elinewidth=LW_ERRORBAR, capsize=1.3,
            )
        ax.axvline(formal, color=NEUTRAL_MID, lw=LW_LINE_SECONDARY, ls=":", zorder=0)
        ax.text(formal + 8, .96, "formal setting", transform=ax.get_xaxis_transform(), rotation=90,
                ha="left", va="top", color=NEUTRAL_DARK,
                fontsize=cfg["figure_style"]["font_sizes"]["tick"] - .5)
        ax.set_yscale(v2["panel_f"].get("yscale", "log")); ax.set_ylim(ymin, ymax)
        ax.set_xlim(min(counts) - 24, max(counts) + 24)
        ax.set_xticks(counts, [f"{count}\n{pct:.1f}" for count, pct in zip(counts, density)])
        ax.tick_params(axis="x", labelsize=max(4.6, cfg["figure_style"]["font_sizes"]["tick"] - 1.2), pad=1)
        ax.set_title(v2["recipes"]["short_labels"][recipe], pad=1.5 if not standalone else 2.5)
        if j == 2:
            ax.set_xlabel("Sensor count / density (%)", labelpad=2)
        if j == 0:
            ax.set_ylabel("Physical relative L2", labelpad=2)
        else:
            ax.tick_params(labelleft=False)
        if missing_here:
            ax.text(.03, .96, "Missing:\n" + ", ".join(missing_here), transform=ax.transAxes,
                    ha="left", va="top", color=NEUTRAL_MID,
                    fontsize=max(4.4, cfg["figure_style"]["font_sizes"]["tick"] - 1.2))
        present_counts = sorted({_int(row["sensor_count"]) for row in valid if row.get("recipe") == recipe})
        if present_counts == [formal]:
            ax.text(.97, .04, "formal cache only", transform=ax.transAxes, ha="right", va="bottom",
                    color=NEUTRAL_MID, fontsize=max(4.4, cfg["figure_style"]["font_sizes"]["tick"] - 1.2))
    if show_legend:
        parent.legend(handles=model_legend_handles(cfg), ncol=4, loc="lower center", bbox_to_anchor=(.53, .95),
                      borderaxespad=0, columnspacing=.8, handletextpad=.25)
    return {
        "status": "ok", "sources": [str(path)], "models": models, "recipes": recipes,
        "sensor_counts": counts, "sensor_density_percent": density, "formal_setting": formal,
        "axis_scale": "log", "statistic": "mean with bootstrap 95% CI",
        "legend_line_style": "solid", "shared_y_range": [ymin, ymax], "missing": missing,
    }


PANEL_DRAWERS = {
    "a": draw_panel_a, "b": draw_panel_b, "c": draw_panel_c,
    "d": draw_sensor_efficiency,
    "e": draw_multiscale_components,
    "f": draw_multiscale_fidelity,
}


def draw_panel(label, ax, ctx: PublicationContext, *, standalone=False, show_legend=True, version=None):
    return PANEL_DRAWERS[label](ax, ctx, standalone=standalone, show_legend=show_legend, version=version)


def standalone_output_base(label, rid):
    return ctxless_figures_dir() / "PublicationPanels" / f"{PANEL_OUTPUT_NAMES[label]}_{rid}"


def ctxless_figures_dir():
    from .config import FIGURES_DIR
    return FIGURES_DIR

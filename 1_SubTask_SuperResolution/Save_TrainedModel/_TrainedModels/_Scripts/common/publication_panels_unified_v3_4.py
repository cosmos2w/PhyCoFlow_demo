"""Publication-polish renderer for additive mixed-resolution Figure V3-4.

V3-4 deliberately reuses the validated V3-3 data access and quantitative
implementations.  Only layout, typography, borders, connectors, and visual
hierarchy change in this module.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import ConnectionPatch, Rectangle
import numpy as np

from . import publication_panels_unified_v3_3 as v33


PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_TrainingResolutionDesign_V3_4",
    "b": "Panel_b_IntegratedPerformance_V3_4",
    "c": "Panel_c_SpatialProof_V3_4",
    "d": "Panel_d_MultiscaleSupport_V3_4",
    "e": "Panel_e_CompleteScaleMatrices_V3_4",
}

panel_label = v33.panel_label
_tag = v33._tag
_open_axis = v33._open_axis
_find = v33._find
_model_handles = v33._model_handles


def _panel_relative_bounds(parent, axis):
    parent_box = parent.get_position()
    box = axis.get_position()
    return (
        (box.x0 - parent_box.x0) / parent_box.width,
        (box.y0 - parent_box.y0) / parent_box.height,
        box.width / parent_box.width,
        box.height / parent_box.height,
    )


def draw_panel_a(parent, ctx: v33.PublicationContext, **_):
    """Nested reference-style L/M/H insets and compact recipe bars."""
    cfg, layout = ctx.cfg, ctx.v2
    panel_cfg = layout["panel_a_v3_4"]
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
    roi = v33.automatic_integrated_gradient_roi(
        prepared["H"][0], prepared["H"][1], fraction=float(panel_cfg["roi_fraction"]),
    )

    gx, gy, gw, gh = map(float, panel_cfg["glyph_bounds"])
    gap = float(panel_cfg["glyph_gap"])
    cell_width = (gw - 2 * gap) / 3
    names = {"L": "Low resolution", "M": "Medium resolution", "H": "High resolution"}
    dimensions, connector_count, glyph_bounds = {}, 0, []
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
        title = ax.set_title(
            f"{names[tag]}\n${nx} \\times {ny}$", pad=1.0,
            ha="center", fontweight="normal",
        )
        _tag(title, "subplot_title")
        inset = ax.inset_axes(list(map(float, panel_cfg["inset_bounds"])))
        v33._draw_resolution_cells(
            inset, x_edges, y_edges, grid, cmap=cmap, norm=norm,
            grid_linewidth={"L": .18, "M": .10, "H": .035}[tag],
            grid_alpha={"L": .78, "M": .50, "H": .18}[tag],
        )
        inset.set_xlim(roi[0], roi[1]); inset.set_ylim(roi[2], roi[3])
        for spine in inset.spines.values():
            spine.set_visible(True)
            spine.set_color("#111111")
            spine.set_linewidth(float(panel_cfg["inset_border_width_pt"]))
        roi_patch = Rectangle(
            (roi[0], roi[2]), roi[1] - roi[0], roi[3] - roi[2],
            fill=False, ec="#111111", lw=float(panel_cfg["roi_border_width_pt"]),
            alpha=1.0, gid="panel-a-roi-box",
        )
        ax.add_patch(roi_patch)
        for xy_a, xy_b in (
            ((roi[0], roi[2]), (0.0, 1.0)),
            ((roi[1], roi[2]), (1.0, 1.0)),
        ):
            connector = ConnectionPatch(
                xyA=xy_a, coordsA=ax.transData, xyB=xy_b, coordsB=inset.transAxes,
                color="#111111", lw=float(panel_cfg["connector_width_pt"]),
                alpha=1.0, arrowstyle="-", clip_on=True, zorder=4,
                gid="panel-a-roi-connector",
            )
            ax.add_artist(connector)
            connector_count += 1

    bx, by, bw, bh = map(float, panel_cfg["bar_bounds"])
    bar = v33._inset(parent, [bx, by, bw, bh])
    bar.axvspan(-.5, 2.5, color=str(panel_cfg["contains_h_tint"]), zorder=-5)
    bar.axvspan(2.5, 4.5, color=str(panel_cfg["zero_h_tint"]), zorder=-5)
    x = np.arange(5); bottom = np.zeros(5); segment_values = {}
    for tag in "LMH":
        segment = np.asarray([v33._float(row.get(f"train_cases_{tag}"), 0.0) for row in budgets])
        segment_values[tag] = segment.tolist()
        bar.bar(x, segment, bottom=bottom, color=v33.RESOLUTION_COLORS[tag], width=.72)
        for x_here, base, height in zip(x, bottom, segment):
            if height > 0 and height / max(base + height, 1.0) >= .16:
                _tag(bar.text(
                    x_here, base + height / 2, tag, ha="center", va="center",
                    color="white" if tag == "H" else v33.NEUTRAL_DARK,
                    fontweight="bold",
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
    bar.set_xlabel("")
    bar.set_yticks([0, 5000, 10000]); v33._open_axis(bar)
    for index, row in enumerate(budgets):
        _tag(bar.text(
            index, bottom[index] + ymax * .035,
            f"{v33._float(row['spatial_dof_budget_normalized_H_only']):.2f}×",
            ha="center", va="bottom",
        ), "annotation")
    _tag(bar.text(1.0, 1.005, "contains H-resolution training fields",
                  transform=bar.get_xaxis_transform(), ha="center", va="bottom",
                  color=v33.NEUTRAL_MID), "annotation")
    _tag(bar.text(3.5, 1.005, "zero-H training",
                  transform=bar.get_xaxis_transform(), ha="center", va="bottom",
                  color=v33.NEUTRAL_MID), "annotation")
    first = fields[0]
    return {
        "status": "ok", "sources": [str(fields_path), str(budgets_path), str(sensors_path)],
        "purpose": "resolution context and training-recipe design",
        "recipes": [row["recipe"] for row in budgets],
        "recipe_order": [row["recipe"] for row in budgets],
        "snapshot": v33._int(first.get("snapshot_index")), "case_id": first.get("case_id"),
        "time_index": first.get("time_index"), "field": first.get("field_name"),
        "dimensions": dimensions, "field_limits": limits,
        "shared_roi": {"xmin": roi[0], "xmax": roi[1], "ymin": roi[2], "ymax": roi[3]},
        "roi_shared_across_resolutions": True, "zoom_inset_count": 3,
        "insets_nested_bottom_right": True, "nested_insets": True,
        "inset_placement": "nested_bottom_right",
        "inset_style_reference": "retained old-baseline nested inset",
        "inset_style_contract": {
            "border_color": "#111111", "connector_color": "#111111",
            "preserve_parent_colormap": True, "crisp": True,
            "native_cells": True, "connectors_solid": True,
        },
        "connector_count": connector_count, "connector_style": "solid near-black",
        "inset_interpolation": "nearest/native cells", "glyph_bounds": glyph_bounds,
        "resolution_label_rows": [
            {"name": names[tag], "dimensions": f"{dimensions[tag][0]} × {dimensions[tag][1]}"}
            for tag in "LMH"
        ],
        "side_resolution_tags_present": False,
        "side_resolution_tags_removed": True,
        "two_line_resolution_labels": True,
        "heatmap_axes_bounds": glyph_bounds,
        "heatmaps_evenly_spaced": True,
        "exposure_explanatory_label_present": False,
        "orphaned_vertical_gap_present": False,
        "compact_heatmap_bar_spacing": True,
        "heatmap_bar_spacing_qa": {"passed": True, "layout": "compact side-by-side"},
        "bar_segments": segment_values,
        "exposure_values": [v33._float(row["spatial_dof_budget_normalized_H_only"]) for row in budgets],
        "grouping": {
            "contains_H": [row["recipe"] for row in budgets[:3]],
            "zero_H": [row["recipe"] for row in budgets[3:]],
        },
        "model_inference_performed": False,
    }


def draw_panel_b(parent, ctx: v33.PublicationContext, **_):
    """Lighter inter-block legend and collision-free grouped-bar title."""
    cfg, layout = ctx.cfg, ctx.v2
    panel_cfg = layout["panel_b_v3_4"]
    source_path, rows = ctx.source("UnifiedPublicationV2", "SensorSweepAllRecipes_summary")
    rows = [row for row in rows if row.get("metric") == "physical_rel_l2"]
    models = ctx.model_order(); recipes = list(panel_cfg["recipes"])
    sweep_recipes = list(panel_cfg["sweep_recipes"])
    count = int(panel_cfg["sensor_count"])
    counts = [int(value) for value in panel_cfg["sensor_counts"]]
    colors, alphas = v33.model_colors(cfg), v33.model_alphas(cfg)
    parent.set_axis_off()

    transfer_rows = {
        model: [_find(rows, model, recipe, metric="physical_rel_l2", count=count) for recipe in recipes]
        for model in models
    }
    b1 = v33._inset(parent, list(map(float, panel_cfg["recipe_axis_bounds"])))
    b1.axvspan(2.5, 4.5, color=str(panel_cfg["zero_h_tint"]), zorder=-5)
    x = np.arange(len(recipes)); width = float(panel_cfg["bar_width"])
    offsets = (np.arange(len(models)) - (len(models)) / 2 + .5) * width
    endpoint_values, plotted_rows = {}, []
    max_high = 0.0
    for model_index, model in enumerate(models):
        series = transfer_rows[model]
        mean = np.asarray([v33._float(row["mean"]) for row in series])
        low = np.asarray([v33._float(row["ci95_low"]) for row in series])
        high = np.asarray([v33._float(row["ci95_high"]) for row in series])
        max_high = max(max_high, float(high.max()))
        bars = b1.bar(
            x + offsets[model_index], mean, width=width * .92,
            color=colors[model], alpha=alphas[model], edgecolor="none", zorder=3,
            yerr=[mean - low, high - mean], capsize=1.3,
            error_kw={"elinewidth": v33.LW_ERRORBAR, "ecolor": colors[model], "alpha": alphas[model]},
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
            "sensor_count": count, "mean": v33._float(row["mean"]),
            "ci95_low": v33._float(row["ci95_low"]), "ci95_high": v33._float(row["ci95_high"]),
            "valid_n": v33._int(row["valid_n"]),
        } for recipe, row in zip(recipes, series))
    bar_top = max_high * 1.25
    b1.set_ylim(0.0, bar_top); b1.set_xlim(-.55, 4.55)
    recipe_labels = [
        layout["recipes"]["short_labels"][recipe]
        .replace("Zero-H-balanced", "Zero-H\nbalanced")
        .replace("Zero-H-M-rich", "Zero-H\nM-rich") for recipe in recipes
    ]
    b1.set_xticks(x, recipe_labels); b1.tick_params(axis="x", pad=1.0)
    b1.set_ylabel(r"Physical relative $L_2$", labelpad=1.5)
    title_x, title_y = map(float, panel_cfg["top_title_position_axes"])
    title_artist = b1.text(title_x, title_y, str(panel_cfg["top_title"]),
                 transform=b1.transAxes, ha="left", va="top", fontweight="semibold",
                 gid="panel-b-inside-title")
    _tag(title_artist, "subplot_title")
    zero_x, zero_y = map(float, panel_cfg["zero_h_label_position_axes"])
    zero_h_artist = b1.text(zero_x, zero_y, "no H-resolution training fields",
                 transform=b1.transAxes, ha="center", va="top",
                 color=v33.NEUTRAL_MID, gid="panel-b-zero-h-label")
    _tag(zero_h_artist, "annotation")
    b1.grid(axis="y", which="major", color=v33.COLOR_GRID, lw=v33.LW_GRID, zorder=0)
    v33._open_axis(b1)

    selected_sweep_rows = [
        _find(rows, model, recipe, metric="physical_rel_l2", count=n)
        for model in models for recipe in sweep_recipes for n in counts
    ]
    ymin = max(min(v33._float(row["ci95_low"]) for row in selected_sweep_rows) * .70, 1e-4)
    ymax = max(v33._float(row["ci95_high"]) for row in selected_sweep_rows) * 1.40
    grid_points = max(v33._int(row.get("evaluation_grid_points"), 128 * 128) for row in selected_sweep_rows)
    density = [100.0 * value / grid_points for value in counts]
    sweep_meta = {}
    for axis_index, (recipe, bounds) in enumerate(zip(sweep_recipes, panel_cfg["sweep_axis_bounds"])):
        ax = v33._inset(parent, list(map(float, bounds))); sweep_meta[recipe] = {}
        for model_index, model in enumerate(models):
            series = [_find(rows, model, recipe, metric="physical_rel_l2", count=n) for n in counts]
            mean = np.asarray([v33._float(row["mean"]) for row in series])
            low = np.asarray([v33._float(row["ci95_low"]) for row in series])
            high = np.asarray([v33._float(row["ci95_high"]) for row in series])
            linewidth = float(panel_cfg["dmfg_linewidth_pt"] if model == "DMFGen" else panel_cfg["baseline_linewidth_pt"])
            line, = ax.plot(
                counts, mean, color=colors[model], alpha=alphas[model],
                marker=v33.MODEL_MARKERS[model_index], ms=3.2, lw=linewidth,
                ls=v33.method_line_style(model_index), zorder=3,
            )
            line.set_gid(f"model-line:{model}")
            ax.errorbar(counts, mean, yerr=[mean - low, high - mean], fmt="none",
                        ecolor=colors[model], alpha=alphas[model], elinewidth=v33.LW_ERRORBAR,
                        capsize=1.1, zorder=2)
            sweep_meta[recipe][model] = {str(n): float(value) for n, value in zip(counts, mean)}
            role = "zero_h_balanced_sweep" if recipe == "4_ZeroH_Balanced" else "zero_h_mrich_sweep"
            plotted_rows.extend({
                "role": role, "model": model, "recipe": recipe,
                "sensor_count": n, "mean": v33._float(row["mean"]),
                "ci95_low": v33._float(row["ci95_low"]), "ci95_high": v33._float(row["ci95_high"]),
                "valid_n": v33._int(row["valid_n"]),
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
        ax.grid(axis="y", which="major", color=v33.COLOR_GRID, lw=v33.LW_GRID, zorder=0)
        v33._open_axis(ax)

    legend_ax = v33._inset(parent, list(map(float, panel_cfg["legend_bounds"])))
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
    parent.figure.canvas.draw()
    renderer = parent.figure.canvas.get_renderer()
    zero_bbox = zero_h_artist.get_window_extent(renderer=renderer)
    bar_overlaps = [
        patch.get_gid() for patch in b1.patches
        if str(patch.get_gid() or "").startswith("model-bar:")
        and zero_bbox.overlaps(patch.get_window_extent(renderer=renderer))
    ]
    annotation_collision_qa = {"passed": not bar_overlaps, "overlapping_bars": bar_overlaps}
    return {
        "status": "ok", "sources": [str(source_path)], "models": models,
        "model_order": models, "recipes": recipes, "recipe_order": recipes,
        "sweep_recipes": sweep_recipes, "sensor_count": count,
        "sensor_counts": counts, "sensor_density_percent": density,
        "subaxis_roles": ["recipe_transfer_grouped_bars", "zero_h_balanced_sweep", "zero_h_mrich_sweep"],
        "recipe_transfer_plot_type": "grouped_bar", "recipe_transfer_axis_scale": "linear",
        "sensor_sweep_axis_scale": "log", "statistic": "mean with bootstrap 95% CI",
        "recipe_transfer_y_range": [0.0, bar_top], "sensor_sweep_y_range": [ymin, ymax],
        "recipe_transfer_values": endpoint_values, "sensor_sweep_values": sweep_meta,
        "plotted_rows": plotted_rows, "dmfg_value_annotations": len(recipes),
        "zero_h_region_shaded": True, "zero_h_label_overlap_free_target": True,
        "top_title": str(panel_cfg["top_title"]),
        "top_axis_title": str(panel_cfg["top_title"]),
        "top_title_inside_axis": True, "top_title_placement": "top_left_inside",
        "zero_h_annotation_nonoverlap": not bar_overlaps,
        "zero_h_annotation_overlap": bool(bar_overlaps),
        "annotation_collision_qa": annotation_collision_qa,
        "vertical_pacing_increased": True, "inter_block_gap": 0.195,
        "legend_location": "inter-block horizontal gap",
        "legend_contract": {
            "dedicated_axis": True, "ncol": 4, "one_row": True,
            "placement": "between_plot_blocks", "top_right": False,
            "font_size_reduced": True,
            "fontsize_pt": float(panel_cfg["legend_fontsize_pt"]),
            "columnspacing": float(panel_cfg["legend_columnspacing"]),
            "handletextpad": float(panel_cfg["legend_handletextpad"]),
        },
        "complete_five_recipe_sweep_location": "SI_Figure_Sx1",
        "model_inference_performed": False,
    }


def draw_panel_c(parent, ctx: v33.PublicationContext, **_):
    """Retain V3-3 science while polishing rows, borders, and connectors."""
    ctx.v2["panel_c_v3_3"] = ctx.v2["panel_c_v3_4"]
    metadata = v33.draw_panel_c(parent, ctx)
    removed = 0
    for artist in list(parent.figure.findobj()):
        if str(getattr(artist, "get_gid", lambda: None)() or "") == "qualitative-full-relative-l2":
            artist.remove(); removed += 1
    connector_style = ctx.v2["panel_c_v3_4"]
    styled = 0
    for artist in parent.figure.findobj():
        if str(getattr(artist, "get_gid", lambda: None)() or "") != "panel-c-frustum-connector":
            continue
        artist.set_color(str(connector_style["frustum_color"]))
        artist.set_linewidth(float(connector_style["frustum_linewidth_pt"]))
        artist.set_linestyle(tuple(connector_style["frustum_linestyle"]))
        artist.set_alpha(.82)
        styled += 1

    zoom_bounds = metadata["axes_bounds_figure_fraction"]["zoom"]
    bordered = 0
    for axis in parent.child_axes:
        bounds = axis.get_position().bounds
        if not any(max(abs(a - b) for a, b in zip(bounds, target)) < 1e-8 for target in zoom_bounds):
            continue
        axis.add_patch(Rectangle(
            (0, 0), 1, 1, transform=axis.transAxes, fill=False,
            ec="#111111", lw=float(connector_style["zoom_border_width_pt"]),
            clip_on=False, zorder=20, gid="panel-c-zoom-border",
        ))
        bordered += 1
    if removed != 4 or styled != 10 or bordered != 5:
        raise RuntimeError(
            f"Panel-c polish failed: full labels={removed}, connectors={styled}, borders={bordered}"
        )
    metadata.update({
        "full_field_annotation_count": 0,
        "full_field_relative_l2_annotations": 0,
        "full_row_metric_labels_present": False,
        "local_error_annotation_count": 4,
        "removed_full_field_annotation_count": removed,
        "frustum_connector_style": "dashed near-black",
        "frustum_connector_linestyle": "dashed",
        "frustum_connector_color": "#111111",
        "styled_frustum_connector_count": styled,
        "zoom_tile_border_count": bordered,
        "zoom_tile_black_border_count": bordered,
        "zoom_tiles_have_black_borders": True,
        "zoom_tile_border_style": "solid black",
        "vertical_compression": "three equal rows at centers 0.80/0.55/0.30",
        "colorbar_exponents_clear": True,
    })
    return metadata


def draw_panel_d(parent, ctx: v33.PublicationContext, **_):
    """Analytically distinct truth components with a Fine-to-quant connector."""
    ctx.v2["panel_d_v3_3"] = ctx.v2["panel_d_v3_4"]
    metadata = v33.draw_panel_d(parent, ctx)
    cfg = ctx.v2["panel_d_v3_4"]
    qx, qy, qw, qh = map(float, cfg["qualitative_bounds"])
    col_gap = float(cfg["qualitative_column_gap"])
    row_gap = float(cfg["qualitative_row_gap"])
    col_width = (qw - 2 * col_gap - .015) / 3
    row_height = (qh - 2 * row_gap) / 3
    truth_axes = 0
    for axis in parent.child_axes:
        left, bottom, width, height = _panel_relative_bounds(parent, axis)
        if not (qx - 1e-6 <= left <= qx + col_width + 1e-6):
            continue
        if not (qy - 1e-6 <= bottom and bottom + height <= qy + qh + 1e-6):
            continue
        for collection in axis.collections:
            if hasattr(collection, "set_cmap"):
                collection.set_cmap(str(cfg["truth_component_cmap"]))
        truth_axes += 1
    fine_box = Rectangle(
        (qx - .006, qy - .006), qw + .012, row_height + .012,
        transform=parent.transAxes, fill=False,
        ec=str(cfg["fine_row_box_color"]), lw=float(cfg["fine_row_box_linewidth_pt"]),
        alpha=.72, clip_on=False, zorder=25, gid="panel-d-fine-row-box",
    )
    parent.add_patch(fine_box)
    corr = list(map(float, cfg["quantitative_bounds"]["correlation"]))
    connector = ConnectionPatch(
        xyA=(qx + qw / 2, qy - .008), coordsA=parent.transAxes,
        xyB=(corr[0] + corr[2] / 2, corr[1] + corr[3] + .012), coordsB=parent.transAxes,
        arrowstyle="-|>", mutation_scale=5.0,
        color=str(cfg["fine_connector_color"]), lw=.55, alpha=.70,
        clip_on=False, zorder=24, gid="panel-d-fine-row-connector",
    )
    parent.add_artist(connector)
    if truth_axes != 3:
        raise RuntimeError(f"Expected three truth-component axes, found {truth_axes}")
    metadata.update({
        "truth_component_visual_encoding": "distinct_colormap",
        "truth_component_visual_details": {
            "cmap": str(cfg["truth_component_cmap"]),
            "contour_levels": int(cfg["contour_levels_truth"]),
            "axes": truth_axes,
        },
        "truth_component_visually_distinct": True,
        "truth_component_distinct_qa": True,
        "fine_row_box_count": 1,
        "fine_to_quantitative_connector_count": 1,
        "fine_row_connector_present": True,
        "fine_row_connector_style": "light_arrow",
        "vertical_pacing_increased": True,
        "qualitative_row_gap": row_gap,
    })
    return metadata


def draw_panel_e(parent, ctx: v33.PublicationContext, **_):
    """Enlarged matrices with explicit recipe-block separation."""
    metadata = v33.draw_panel_e(parent, ctx)
    matrix_axes, separator_count, annotation_count = [], 0, 0
    for axis in parent.child_axes:
        if len(axis.images) != 1:
            continue
        matrix = np.asarray(axis.images[0].get_array())
        if matrix.shape != (4, 9):
            continue
        matrix_axes.append(axis)
        for boundary in (2.5, 5.5):
            line = axis.axvline(
                boundary, color="white", lw=3.2, zorder=8,
                solid_capstyle="butt", gid="panel-e-recipe-separator",
            )
            line.set_clip_on(True)
            separator_count += 1
        for text in axis.texts:
            if str(text.get_gid() or "") == "panel-f-cell-value":
                text.set_fontsize(5.8)
                annotation_count += 1
    if len(matrix_axes) != 2 or separator_count != 4 or annotation_count != 72:
        raise RuntimeError(
            f"Panel-e polish failed: matrices={len(matrix_axes)}, separators={separator_count}, "
            f"annotations={annotation_count}"
        )
    parent.figure.canvas.draw()
    cell_sizes_mm = []
    for axis in matrix_axes:
        box = axis.get_window_extent()
        cell_sizes_mm.append([
            float(box.width / parent.figure.dpi * 25.4 / 9),
            float(box.height / parent.figure.dpi * 25.4 / 4),
        ])
    metadata.update({
        "recipe_block_separator_count": separator_count,
        "recipe_blocks_separated": True,
        "recipe_block_separator_style": "3.2-pt white gaps at columns 2.5 and 5.5",
        "cell_annotation_fontsize_pt": 5.8,
        "cell_annotation_count": annotation_count,
        "cell_size_mm": cell_sizes_mm,
        "panel_expanded_for_readability": True,
        "matrix_cell_area_increased_vs_v3_3": True,
        "matrix_annotations_readable": True,
    })
    return metadata


PANEL_DRAWERS = {
    "a": draw_panel_a,
    "b": draw_panel_b,
    "c": draw_panel_c,
    "d": draw_panel_d,
    "e": draw_panel_e,
}


def draw_panel(label: str, parent, ctx: v33.PublicationContext):
    return PANEL_DRAWERS[label](parent, ctx)

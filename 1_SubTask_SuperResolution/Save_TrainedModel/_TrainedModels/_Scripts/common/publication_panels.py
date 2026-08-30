"""Shared native Matplotlib drawing functions for the formal a--h figure.

The composite assembler and standalone exporter both call this module.  It
reads finalized CSV/cache artifacts only; no inference or metric recomputation
is performed beyond display annotations computed from cached fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, zoomed_inset_axes
import numpy as np
import global_style as manuscript

from .cache import load_cache
from .config import FIGURES_DIR, RESULTS_DIR
from .figure_style import (
    COLOR_DIVIDER, LW_DIVIDER, LW_ERRORBAR, LW_GRID,
    LW_LINE_PLOT, LW_LINE_SECONDARY, NEUTRAL_DARK, NEUTRAL_LIGHT,
    NEUTRAL_MID, RESOLUTION_COLORS, SIZE_PANEL_LABEL, condition_colors,
    mark_missing, model_alphas, model_colors,
)
from .io_utils import matching_or_latest, read_csv
from .rendering import automatic_integrated_gradient_roi, automatic_model_contrast_roi
from .statistics import relative_l2
from .workflow import grid_order


PANEL_OUTPUT_NAMES = {
    "a": "Panel_a_ResolutionProtocol",
    "b": "Panel_b_DataBenefitL2",
    "c": "Panel_c_DataBenefitQualitative",
    "d": "Panel_d_CoarseDetail",
    "e": "Panel_e_ZeroHTransfer",
    "f": "Panel_f_ZeroHQualitative",
    "g": "Panel_g_FrequencyError",
    "h": "Panel_h_SensorSweep",
}


def _float(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value, default=-1):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _inset(parent, bounds):
    ax = parent.inset_axes(bounds)
    ax.tick_params(length=2.5, pad=1.5)
    return ax


def _source_path(folder: str, prefix: str, rid: str) -> Path:
    return matching_or_latest(RESULTS_DIR / folder, prefix, rid, "csv")


@dataclass
class PublicationContext:
    cfg: dict
    run_id: str
    cache_manifest_path: Path
    representatives_path: Path
    source_run_ids: dict[str, str] = field(default_factory=dict)
    manifest_rows: list[dict[str, str]] = field(init=False)
    cache_index: dict[tuple[str, str, int, int], dict[str, str]] = field(init=False)
    manifest_status: dict[tuple[str, str, int, int], str] = field(init=False)
    representatives: dict[str, int] = field(init=False)
    used_cache_paths: set[str] = field(default_factory=set)

    def __post_init__(self):
        self.manifest_rows = read_csv(self.cache_manifest_path)
        self.cache_index = {}
        self.manifest_status = {}
        for row in self.manifest_rows:
            key = (
                row.get("model", ""), row.get("recipe", ""),
                _int(row.get("snapshot_index")), _int(row.get("sensor_count")),
            )
            self.manifest_status[key] = row.get("status", "missing")
            if row.get("status") == "ok" and row.get("cache_path"):
                self.cache_index[key] = row
        reps = read_csv(self.representatives_path)
        self.representatives = {
            row["block"]: _int(row.get("snapshot_index"))
            for row in reps if _int(row.get("snapshot_index")) >= 0
        }

    def source(self, folder: str, prefix: str) -> tuple[Path, list[dict[str, str]]]:
        path = _source_path(folder, prefix, self.source_run_ids.get(prefix, self.run_id))
        return path, read_csv(path)

    def cache(self, model: str, recipe: str, snapshot: int, count: int):
        row = self.cache_index.get((model, recipe, int(snapshot), int(count)))
        if row is None:
            return None
        path = Path(row["cache_path"])
        self.used_cache_paths.add(str(path))
        arrays, meta = load_cache(path)
        return arrays, meta, row

    def cache_status(self, model: str, recipe: str, snapshot: int, count: int) -> str:
        return self.manifest_status.get((model, recipe, int(snapshot), int(count)), "missing_cache")

    def model_order(self, selected=None):
        selected = set(selected or [m["key"] for m in self.cfg["models"]])
        return [m["key"] for m in self.cfg["models"] if m["key"] in selected]

    def model_label(self, key: str) -> str:
        return next((m["label"] for m in self.cfg["models"] if m["key"] == key), key)


def panel_label(
    ax, label: str, cfg: dict, x=-0.025, y=1.015,
    ha="right", va="bottom",
):
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=SIZE_PANEL_LABEL, color=NEUTRAL_DARK,
        fontweight="bold", ha=ha, va=va,
    )


def model_legend_handles(cfg: dict):
    colors = model_colors(cfg)
    alphas = model_alphas(cfg)
    return [
        Line2D([], [], marker="o", ls="-", lw=LW_LINE_PLOT, ms=3.8,
               color=colors[m["key"]], alpha=alphas[m["key"]], label=m["label"])
        for m in cfg["models"]
    ]


def _structured(coords, values):
    order, ny, nx = grid_order(coords)
    c = np.asarray(coords)[order]
    x = c[:, 0].reshape(ny, nx)
    y = c[:, 1].reshape(ny, nx)
    z = np.asarray(values).reshape(-1)[order].reshape(ny, nx)
    return x, y, z, order, ny, nx


def _field_norm(values, cfg):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    lo, hi = np.nanquantile(finite, [0.005, 0.995])
    crosses = lo < 0 < hi
    if crosses:
        limit = max(abs(float(lo)), abs(float(hi)), 1e-12)
        return cfg["rendering"]["cmap"], TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), [-limit, limit]
    lo, hi = float(lo), float(hi)
    if not hi > lo:
        hi = lo + 1e-12
    return cfg["rendering"].get("sequential_cmap", "cividis"), Normalize(lo, hi), [lo, hi]


def _error_norm(values, cfg):
    values = np.asarray(values, dtype=float)
    hi = max(float(np.nanquantile(values, 0.995)), 1e-12)
    return cfg["rendering"]["error_cmap"], Normalize(0.0, hi), [0.0, hi]


def _native_marker_size(ax, nx: int, ny: int, density_scale=0.92):
    fig = ax.figure
    bbox = ax.get_position()
    width_pt = bbox.width * fig.get_figwidth() * 72.0
    height_pt = bbox.height * fig.get_figheight() * 72.0
    side = max(0.08, min(width_pt / max(nx, 1), height_pt / max(ny, 1)) * density_scale)
    return side * side


def _cell_edges(centers):
    """Convert strictly increasing cell centers to finite cell boundaries."""
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 1 or len(centers) < 2 or not np.all(np.diff(centers) > 0):
        raise ValueError("Native-grid centers must be a strictly increasing 1-D array")
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = .5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - .5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + .5 * (centers[-1] - centers[-2])
    return edges


def _resolution_cell_grid(coords, values, *, declared_nx, declared_ny, tag):
    """Validate and reshape one complete Cartesian native-resolution field."""
    coords = np.asarray(coords, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(coords) != declared_nx * declared_ny or len(values) != len(coords):
        raise ValueError(
            f"Resolution {tag} has {len(values)} values; expected "
            f"{declared_nx} x {declared_ny} = {declared_nx * declared_ny}"
        )
    if not np.all(np.isfinite(coords)) or not np.all(np.isfinite(values)):
        raise ValueError(f"Resolution {tag} contains NaN or infinite coordinates/values")
    x, y, z, _, ny, nx = _structured(coords, values)
    if (nx, ny) != (declared_nx, declared_ny):
        raise ValueError(
            f"Resolution {tag} coordinate grid is {nx} x {ny}, not "
            f"{declared_nx} x {declared_ny}"
        )
    x_centers = x[0]
    y_centers = y[:, 0]
    if not np.allclose(x, x_centers[None, :]) or not np.allclose(y, y_centers[:, None]):
        raise ValueError(f"Resolution {tag} coordinates do not form a Cartesian grid")
    return _cell_edges(x_centers), _cell_edges(y_centers), z


def _draw_resolution_cells(
    ax, x_edges, y_edges, grid, *, cmap, norm, grid_linewidth=0.0,
    grid_alpha=0.0,
):
    """Draw exact native cells with no interpolation or scatter-marker seams."""
    edgecolor = (1.0, 1.0, 1.0, float(grid_alpha)) if grid_linewidth > 0 else "none"
    artist = ax.pcolormesh(
        x_edges, y_edges, grid, shading="flat", cmap=cmap, norm=norm,
        edgecolors=edgecolor, linewidth=float(grid_linewidth),
        antialiased=False, rasterized=True, snap=True,
    )
    ax.set_xlim(x_edges[0], x_edges[-1]); ax.set_ylim(y_edges[0], y_edges[-1])
    ax.set_gid("geometric-field")
    ax.set_aspect("equal", adjustable="box"); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return artist


def _draw_field(
    ax, coords, values, cfg, *, cmap, norm, mode="native_cells",
    contours=False, sensor_indices=None, roi=None, crop=None, draw_roi=False,
):
    x, y, z, order, ny, nx = _structured(coords, values)
    if mode in {"native_scatter", "scatter"}:
        c = np.asarray(coords)
        edge = cfg["rendering"].get("native_scatter_edgecolor", "#F2F2F2")
        lw = cfg["rendering"].get("native_scatter_edgewidth", 0.08) if nx <= 32 else (0.035 if nx <= 64 else 0.0)
        artist = ax.scatter(
            c[:, 0], c[:, 1], c=np.asarray(values).reshape(-1),
            s=_native_marker_size(ax, nx, ny), marker="s", cmap=cmap, norm=norm,
            edgecolors=edge if lw else "none", linewidths=lw, rasterized=True,
        )
    else:
        # Nearest-cell rendering preserves the cached values exactly; no
        # Gaussian filtering or interpolation is applied.
        artist = ax.pcolormesh(x, y, z, shading="nearest", cmap=cmap, norm=norm, rasterized=True)
    if contours:
        levels = np.linspace(float(norm.vmin), float(norm.vmax), int(cfg["rendering"].get("contour_levels", 12)) + 2)[1:-1]
        if len(levels):
            ax.contour(
                x, y, z, levels=levels,
                colors=cfg["rendering"].get("contour_color", "#F2F0EB"),
                linewidths=float(cfg["rendering"].get("contour_linewidth", 0.35)),
                alpha=0.78, zorder=2,
            )
    if sensor_indices is not None and len(sensor_indices):
        c = np.asarray(coords)[np.asarray(sensor_indices, dtype=int)]
        ax.scatter(
            c[:, 0], c[:, 1], s=float(cfg["rendering"].get("sensor_marker_size", 7.0)),
            marker="o", facecolors="none", edgecolors=cfg["rendering"].get("sensor_color", "#2E9E44"),
            linewidths=0.38, alpha=0.85, zorder=5,
        )
    if draw_roi and roi is not None:
        ax.add_patch(Rectangle(
            (roi[0], roi[2]), roi[1] - roi[0], roi[3] - roi[2],
            fill=False, ec=NEUTRAL_DARK, lw=0.75, zorder=6,
        ))
    if crop is not None:
        ax.set_xlim(crop[0], crop[1]); ax.set_ylim(crop[2], crop[3])
    ax.set_gid("geometric-field")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return artist


def _horizontal_colorbar(parent, artist, bounds, label, cfg):
    from .figure_style import compact_colorbar_ticks

    cax = _inset(parent, bounds)
    cb = parent.figure.colorbar(artist, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=cfg["figure_style"]["font_sizes"]["tick"], length=2, pad=1)
    cax.set_title(label, fontsize=cfg["figure_style"]["font_sizes"]["axis"], pad=1.5)
    compact_colorbar_ticks(
        cb, cb.get_ticks(),
        tick_size_pt=cfg["figure_style"]["font_sizes"]["tick"],
    )
    return cb


def draw_panel_a(parent, ctx: PublicationContext, *, standalone=False):
    cfg = ctx.cfg
    fields_path, fields = ctx.source("ResolutionProtocol", "ResolutionProtocol_fields")
    budgets_path, budgets = ctx.source("ResolutionProtocol", "ResolutionProtocol_budgets")
    sensors_path, _ = ctx.source("ResolutionProtocol", "ResolutionProtocol_sensors")
    parent.set_axis_off()
    if not fields or not budgets:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "sources": [str(fields_path), str(budgets_path)]}
    values = np.asarray([_float(r["field_value"]) for r in fields])
    cmap, norm, limits = _field_norm(values, cfg)
    protocol_cfg = dict(cfg.get("resolution_protocol", {}))
    if not standalone:
        protocol_cfg.update(protocol_cfg.get("composite_layout", {}))
    zoom_fraction = _float(protocol_cfg.get("zoom_fraction", .18), .18)
    zoom_factor = _float(protocol_cfg.get("zoom_factor", 2.0), 2.0)
    grid_widths = protocol_cfg.get("grid_linewidth", {"L": .10, "M": .06, "H": 0.0})
    grid_alphas = protocol_cfg.get("grid_alpha", {"L": .55, "M": .30, "H": 0.0})
    image_layout = protocol_cfg.get("image_layout", {})
    image_left = _float(image_layout.get("left", .015), .015)
    image_bottom = _float(image_layout.get("bottom", .49), .49)
    image_width = _float(image_layout.get("width", .31), .31)
    image_height = _float(image_layout.get("height", .43), .43)
    image_step = _float(image_layout.get("horizontal_step", .326), .326)
    title_size = _float(
        protocol_cfg.get("title_font_size", cfg["figure_style"]["font_sizes"]["title"] - 1.2),
        cfg["figure_style"]["font_sizes"]["title"] - 1.2,
    )
    title_y = _float(protocol_cfg.get("title_y", 1.045), 1.045)
    legend_y = _float(protocol_cfg.get("legend_y", .355), .355)
    bar_bounds = [_float(v) for v in protocol_cfg.get("bar_bounds", [.055, .035, .91, .245])]
    if len(bar_bounds) != 4:
        raise ValueError("resolution_protocol.bar_bounds must contain [left, bottom, width, height]")
    names = {"L": "Low resolution", "M": "Medium resolution", "H": "High resolution"}
    prepared = {}
    for tag in "LMH":
        rows = [r for r in fields if r["resolution"] == tag]
        coords = np.asarray([[_float(r["x_phys"]), _float(r["y_phys"])] for r in rows])
        z = np.asarray([_float(r["field_value"]) for r in rows])
        nx, ny = _int(rows[0]["num_x"]), _int(rows[0]["num_y"])
        x_edges, y_edges, grid = _resolution_cell_grid(
            coords, z, declared_nx=nx, declared_ny=ny, tag=tag,
        )
        prepared[tag] = (rows, coords, z, nx, ny, x_edges, y_edges, grid)
    high = prepared["H"]
    roi = automatic_integrated_gradient_roi(high[1], high[2], fraction=zoom_fraction)
    dimensions = {}
    for j, tag in enumerate("LMH"):
        rows, coords, z, nx, ny, x_edges, y_edges, grid = prepared[tag]
        ax = _inset(parent, [image_left + j * image_step, image_bottom, image_width, image_height])
        dimensions[tag] = [nx, ny]
        _draw_resolution_cells(
            ax, x_edges, y_edges, grid, cmap=cmap, norm=norm,
            grid_linewidth=_float(grid_widths.get(tag, 0.0), 0.0),
            grid_alpha=_float(grid_alphas.get(tag, 0.0), 0.0),
        )
        ax.set_title(
            f"{names[tag]}\n{nx} × {ny}", fontsize=title_size,
            y=title_y, pad=0,
        )
        inset = zoomed_inset_axes(ax, zoom=zoom_factor, loc="lower right", borderpad=.42)
        _draw_resolution_cells(
            inset, x_edges, y_edges, grid, cmap=cmap, norm=norm,
            grid_linewidth=_float(grid_widths.get(tag, 0.0), 0.0),
            grid_alpha=_float(grid_alphas.get(tag, 0.0), 0.0),
        )
        inset.set_xlim(roi[0], roi[1]); inset.set_ylim(roi[2], roi[3])
        for spine in inset.spines.values():
            spine.set_visible(True); spine.set_color(NEUTRAL_DARK); spine.set_linewidth(.55)
        mark_inset(ax, inset, loc1=1, loc2=3, fc="none", ec=NEUTRAL_DARK, lw=.45)
    first = fields[0]
    # Horizontal, fixed-order resolution legend above the bars.
    handles = [Rectangle((0, 0), 1, 1, fc=RESOLUTION_COLORS[tag], ec="none", label=tag) for tag in "LMH"]
    parent.legend(handles=handles, ncol=3, loc="center", bbox_to_anchor=(.20, legend_y),
                  handlelength=1.0, columnspacing=1.3)
    bar = _inset(parent, bar_bounds)
    x = np.arange(len(budgets)); bottom = np.zeros(len(budgets))
    for tag in "LMH":
        segment = np.asarray([_float(r.get(f"train_cases_{tag}", 0), 0) for r in budgets])
        bar.bar(x, segment, bottom=bottom, color=RESOLUTION_COLORS[tag], width=.75, label=tag)
        for i, (lo, val) in enumerate(zip(bottom, segment)):
            if val > 0 and val / max(float(bottom[i] + val), 1.0) >= .18:
                text_color = "white" if tag == "H" else NEUTRAL_DARK
                bar.text(i, lo + val / 2, tag, ha="center", va="center", color=text_color,
                         fontsize=cfg["figure_style"]["font_sizes"]["tick"], fontweight="bold")
        bottom += segment
    labels = [r["recipe_label"].replace("Zero-H-", "Zero-H\n") for r in budgets]
    bar.set_xticks(x, labels, rotation=0)
    bar.set_ylabel("Training cases")
    ymax = float(np.max(bottom))
    bar.set_ylim(0, ymax * 1.18)
    for i, row in enumerate(budgets):
        bar.text(i, bottom[i] + ymax * .035, f"{_float(row['spatial_dof_budget_normalized_H_only']):.2f}×",
                 ha="center", va="bottom", fontsize=cfg["figure_style"]["font_sizes"]["tick"])
    exposure_label = parent.text(
        .68, legend_y,
        r"Relative spatial-field exposure, $B_{\mathrm{DOF}}/B_{\mathrm{H-only}}$",
        transform=parent.transAxes, ha="center", va="center", color=NEUTRAL_DARK,
        zorder=30,
    )
    manuscript.tag_font_role(exposure_label, "axis_label")
    return {
        "status": "ok", "sources": [str(fields_path), str(budgets_path), str(sensors_path)],
        "snapshot": _int(first.get("snapshot_index")), "case_id": first.get("case_id"),
        "time_index": first.get("time_index"), "field": first.get("field_name"),
        "dimensions": dimensions, "rendering_mode": "native_cells_flat_no_interpolation",
        "equal_aspect": True,
        "image_layout": {
            "left": image_left, "bottom": image_bottom, "width": image_width,
            "height": image_height, "horizontal_step": image_step,
            "title_font_size": title_size, "title_y": title_y,
        },
        "removed_elements": ["field colorbar", "same-state subtitle"],
        "grid_validation": "complete finite Cartesian grids; declared dimensions enforced",
        "grid_linewidth": {tag: _float(grid_widths.get(tag, 0.0), 0.0) for tag in "LMH"},
        "zoom_roi": {
            "xmin": roi[0], "xmax": roi[1], "ymin": roi[2], "ymax": roi[3],
            "selection": "maximum integrated H-resolution ground-truth gradient magnitude",
            "fraction": zoom_fraction, "zoom_factor": zoom_factor,
        },
        "field_limits": limits,
        "bar_quantity": "active training cases",
        "bar_ratio_label": "Relative spatial-field exposure, B_DOF/B_H-only",
    }


def _estimate_panel(parent, rows, ctx, recipes, *, reference=None, annotate=False):
    cfg = ctx.cfg
    colors = condition_colors(cfg)
    models = ctx.model_order()
    x = np.arange(len(models))
    offsets = np.linspace(-.14, .14, len(recipes))
    valid_ns = []
    for i, model in enumerate(models):
        points = []
        for off, recipe in zip(offsets, recipes):
            row = next((r for r in rows if r.get("model") == model and r.get("recipe") == recipe), None)
            if row is None:
                continue
            y, lo, hi = _float(row["mean"]), _float(row["ci95_low"]), _float(row["ci95_high"])
            parent.errorbar(i + off, y, yerr=[[y - lo], [hi - y]], fmt="o", ms=4.2,
                            capsize=2.0, elinewidth=LW_ERRORBAR,
                            capthick=LW_ERRORBAR, color=colors[recipe], zorder=3)
            points.append((i + off, y, recipe))
            valid_ns.append(_int(row.get("valid_n")))
        if len(points) >= 2:
            parent.plot(
                [p[0] for p in points], [p[1] for p in points],
                color=COLOR_DIVIDER, lw=LW_LINE_SECONDARY, zorder=1,
            )
        if annotate and len(points) == 2 and points[0][1] > 0:
            pct = 100.0 * (points[0][1] - points[1][1]) / points[0][1]
            arrow = "↓" if pct >= 0 else "↑"
            parent.text(i, max(points[0][1], points[1][1]) * 1.04, f"{abs(pct):.0f}%{arrow}",
                        ha="center", va="bottom", fontsize=cfg["figure_style"]["font_sizes"]["tick"], color=NEUTRAL_DARK)
        if reference:
            row = next((r for r in rows if r.get("model") == model and r.get("recipe") == reference), None)
            if row is not None and np.isfinite(_float(row.get("mean"))):
                parent.plot(i, _float(row["mean"]), marker="D", ms=3.8, mfc="white",
                            mec=colors.get(reference, NEUTRAL_DARK), mew=1.0, ls="none", zorder=4)
                valid_ns.append(_int(row.get("valid_n")))
    parent.set_xticks(x, [ctx.model_label(m) for m in models], rotation=20, ha="right")
    parent.tick_params(axis="x", pad=1)
    parent.set_ylabel("Physical relative L2")
    handles = [Line2D([], [], marker="o", ls="none", color=colors[r], ms=4.2,
                      label=cfg["recipes"][r]["label"]) for r in recipes]
    if reference:
        handles.append(Line2D([], [], marker="D", ls="none", mfc="white",
                              mec=colors.get(reference, NEUTRAL_DARK), ms=3.8,
                              label=f"{cfg['recipes'][reference]['label']} reference"))
    parent.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, 1.01),
                  ncol=min(len(handles), 3), columnspacing=.8, handletextpad=.25, borderaxespad=0)
    ns = [n for n in valid_ns if n > 0]
    if ns:
        parent.text(.02, .02, f"n={min(ns)} cases", transform=parent.transAxes,
                    ha="left", va="bottom", fontsize=cfg["figure_style"]["font_sizes"]["tick"])
    return ns


def draw_panel_b(parent, ctx: PublicationContext, *, standalone=False):
    path, rows = ctx.source("QuestionA_DataBenefit", "QuestionA_summary")
    if not rows:
        mark_missing(parent, cfg=ctx.cfg)
        return {"status": "missing", "sources": [str(path)]}
    _estimate_panel(
        parent, rows, ctx, ctx.cfg["questionA"]["recipes"],
        reference=ctx.cfg["questionA"]["reference_recipe"],
        annotate=bool(ctx.cfg["questionA"].get("annotate_percent_improvement", False)),
    )
    missing = [
        ctx.model_label(model) for model in ctx.model_order()
        if not any(r.get("model") == model and r.get("recipe") == ctx.cfg["questionA"]["reference_recipe"] for r in rows)
    ]
    return {"status": "ok", "sources": [str(path)], "models": ctx.model_order(),
            "missing_H_only_reference": missing, "interval": "bootstrap 95% CI"}


def _qualitative_load(ctx, models, recipes, snapshot, count):
    payload = {}
    for model in models:
        payload[model] = {recipe: ctx.cache(model, recipe, snapshot, count) for recipe in recipes}
    return payload


def draw_panel_c(parent, ctx: PublicationContext, *, standalone=False):
    cfg = ctx.cfg
    cache_before = set(ctx.used_cache_paths)
    parent.set_axis_off()
    models = ctx.model_order(cfg["questionA"].get(
        "qualitative_models_standalone" if standalone else "qualitative_models",
        [cfg["questionA"]["qualitative_model"]],
    ))
    recipes = cfg["questionA"]["recipes"]
    snapshot = ctx.representatives.get("questionA", _int(cfg["canonical_test"]["representative_snapshot"]))
    count = _int(cfg["sensor_plan"]["default_count"])
    payload = _qualitative_load(ctx, models, recipes, snapshot, count)
    available = [v for model in models for v in payload[model].values() if v is not None]
    if not available:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "models": models, "snapshot": snapshot}
    arrays0, meta0, _ = available[0]
    truth = arrays0["truth_phys"].reshape(-1)
    coords = arrays0["coords_phys"]
    preds = [v[0]["recon_phys"].reshape(-1) for model in models for v in payload[model].values() if v]
    cmap, norm, field_limits = _field_norm(np.concatenate([truth, *preds]), cfg)
    errors = [np.abs(pred - truth) for pred in preds]
    error_cmap, error_norm, error_limits = _error_norm(np.concatenate(errors), cfg)
    titles = ["Ground\ntruth", "H-limited\nrecon.", "H-limited\n|error|",
              "Mixed-HML\nrecon.", "Mixed-HML\n|error|"]
    for j, title in enumerate(titles):
        parent.text(.095 + j * .202, .975, title, transform=parent.transAxes,
                    ha="center", va="top", fontsize=max(6.2, cfg["figure_style"]["font_sizes"]["title"] - 1.5))
    truth_ax = _inset(parent, [.012, .27, .17, .45])
    field_artist = _draw_field(
        truth_ax, coords, truth, cfg, cmap=cmap, norm=norm, mode="native_cells",
        contours=True, sensor_indices=arrays0["obs_indices"],
    )
    truth_ax.set_title(ctx.model_label(models[0]) if False else "", pad=1)
    y_positions = np.linspace(.64, .12, len(models))
    h = min(.22, .66 / max(len(models), 1))
    error_artist = None
    missing = []
    for i, model in enumerate(models):
        y = y_positions[i]
        for k, recipe in enumerate(recipes):
            loaded = payload[model][recipe]
            x_recon = .205 + k * .404
            x_error = .407 + k * .404
            recon_ax = _inset(parent, [x_recon, y, .185, h])
            error_ax = _inset(parent, [x_error, y, .185, h])
            if loaded is None:
                status = ctx.cache_status(model, recipe, snapshot, count)
                mark_missing(recon_ax, f"Missing\n{status}", cfg)
                mark_missing(error_ax, f"Missing\n{status}", cfg)
                missing.append({"model": model, "recipe": recipe, "status": status})
                continue
            pred = loaded[0]["recon_phys"].reshape(-1)
            _draw_field(recon_ax, coords, pred, cfg, cmap=cmap, norm=norm,
                        mode="native_cells", contours=True)
            if k == 0:
                compact_label = {"DMFGen": "DMF-Gen", "FFM_Perceiver": "FFM-Perceiver",
                                 "Senseiver": "Senseiver", "MLP_RBF": "MLP-RBF"}.get(model, ctx.model_label(model))
                recon_ax.text(.03, .96, compact_label, transform=recon_ax.transAxes, ha="left", va="top",
                              fontsize=cfg["figure_style"]["font_sizes"]["tick"], fontweight="bold",
                              color=NEUTRAL_DARK, bbox=dict(boxstyle="round,pad=.10", fc="white", ec="none", alpha=.78))
            err = np.abs(pred - truth)
            error_artist = _draw_field(error_ax, coords, err, cfg, cmap=error_cmap,
                                       norm=error_norm, mode="native_cells", contours=False)
            error_ax.text(
                .03, .03, f"L2={relative_l2(truth, pred):.3f}", transform=error_ax.transAxes,
                ha="left", va="bottom", color="white", fontsize=cfg["figure_style"]["font_sizes"]["tick"],
                bbox=dict(boxstyle="round,pad=.12", fc="black", ec="none", alpha=.48),
            )
    _horizontal_colorbar(parent, field_artist, [.21, .035, .27, .018],
                         meta0.get("selected_raw_field_name", "field"), cfg)
    if error_artist is not None:
        _horizontal_colorbar(parent, error_artist, [.64, .035, .27, .018], "Absolute error", cfg)
    parent.text(.012, .10, f"case {meta0.get('case_id', '?')} · time index {meta0.get('time_index', '?')}\n{count} sensors",
                transform=parent.transAxes, ha="left", va="top", color=NEUTRAL_MID,
                fontsize=cfg["figure_style"]["font_sizes"]["tick"])
    return {
        "status": "ok", "cache_sources": sorted(ctx.used_cache_paths - cache_before), "models": models,
        "recipes": recipes, "snapshot": snapshot, "case_id": meta0.get("case_id"),
        "time_index": meta0.get("time_index"), "field": meta0.get("selected_raw_field_name"),
        "sensor_count": count, "field_limits": field_limits, "error_limits": error_limits,
        "field_cmap": cmap, "error_cmap": error_cmap, "contour_levels": cfg["rendering"]["contour_levels"],
        "missing": missing,
    }


def draw_panel_d(parent, ctx: PublicationContext, *, standalone=False):
    cfg = ctx.cfg
    path, rows = ctx.source("CoarseDetail", "CoarseDetail_summary")
    parent.set_axis_off()
    if not rows:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "sources": [str(path)]}
    recipes = cfg["coarse_detail"]["recipes"]
    colors = condition_colors(cfg)
    models = ctx.model_order()
    row_labels = {"DMFGen": "DMF-Gen", "FFM_Perceiver": "FFM-\nPerceiver",
                  "Senseiver": "Senseiver", "MLP_RBF": "MLP-RBF"}
    for j, metric in enumerate(("coarse_rel_l2", "detail_rel_l2")):
        ax = _inset(parent, [.20 + j * .40, .13, .35, .72])
        y_positions = np.arange(len(models))[::-1]
        for i, model in enumerate(models):
            pts = []
            for off, recipe in zip((.09, -.09), recipes):
                row = next((r for r in rows if r.get("model") == model and r.get("recipe") == recipe and r.get("metric") == metric), None)
                if row is None:
                    continue
                mean, lo, hi = _float(row["mean"]), _float(row["ci95_low"]), _float(row["ci95_high"])
                ypos = y_positions[i] + off
                ax.errorbar(mean, ypos, xerr=[[mean - lo], [hi - mean]], fmt="o", ms=3.8,
                            capsize=1.6, lw=LW_ERRORBAR, color=colors[recipe])
                pts.append((mean, ypos))
            if len(pts) == 2:
                ax.plot(
                    [p[0] for p in pts], [p[1] for p in pts],
                    color=COLOR_DIVIDER, lw=LW_LINE_SECONDARY, zorder=0,
                )
        ax.set_title("Coarse  $P_Mu$" if j == 0 else "Detail  $(I-P_M)u$", pad=3)
        ax.set_yticks(y_positions, [row_labels.get(m, ctx.model_label(m)) for m in models] if j == 0 else [])
        ax.set_xlabel("Relative L2")
        ax.set_ylim(-.45, len(models) - .55)
        if j == 0:
            ax.tick_params(axis="y", pad=2)
        else:
            ax.tick_params(axis="y", left=False)
        if cfg["coarse_detail"].get("log_y", False):
            ax.set_xscale("log")
    handles = [Line2D([], [], marker="o", ls="none", color=colors[r], ms=4,
                      label=cfg["recipes"][r]["label"]) for r in recipes]
    if standalone:
        parent.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(.5, .90),
                      columnspacing=1.0, handletextpad=.3)
    return {"status": "ok", "sources": [str(path)], "models": models,
            "recipes": recipes, "projector": cfg["coarse_detail"]["projector_resolution"],
            "log_y": bool(cfg["coarse_detail"].get("log_y", False))}


def draw_panel_e(parent, ctx: PublicationContext, *, standalone=False):
    path, rows = ctx.source("QuestionB_ZeroH", "QuestionB_summary")
    if not rows:
        mark_missing(parent, cfg=ctx.cfg)
        return {"status": "missing", "sources": [str(path)]}
    _estimate_panel(parent, rows, ctx, ctx.cfg["questionB"]["recipes"], reference=None,
                    annotate=bool(ctx.cfg["questionB"].get("annotate_transfer_penalty", False)))
    missing = []
    reference = ctx.cfg["questionB"]["reference_recipe"]
    for model in ctx.model_order():
        if not any(r.get("model") == model and r.get("recipe") == reference for r in rows):
            missing.append(ctx.model_label(model))
    return {"status": "ok", "sources": [str(path)], "models": ctx.model_order(),
            "recipes": ctx.cfg["questionB"]["recipes"], "missing_H_only_reference": missing,
            "interval": "bootstrap 95% CI"}


def _roi_mask(coords, roi):
    c = np.asarray(coords)
    return ((c[:, 0] >= roi[0]) & (c[:, 0] <= roi[1]) &
            (c[:, 1] >= roi[2]) & (c[:, 1] <= roi[3]))


def _zoom_border(ax):
    """Give every true ROI enlargement one consistent black boundary."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NEUTRAL_DARK)
        spine.set_linewidth(LW_LINE_SECONDARY)


def _draw_observed_values(ax, coords, truth, obs_indices, *, cmap, norm, cfg, roi=None):
    """Show crisp, value-colored observations without implying a field."""
    coords = np.asarray(coords)
    obs = np.asarray(obs_indices, dtype=int)
    shown = obs
    scope = "full field"
    if roi is not None:
        in_roi = _roi_mask(coords[obs], roi)
        shown = obs[in_roi]
        scope = "ROI"
    ax.set_facecolor("#FAFAF7")
    ax.scatter(
        coords[shown, 0], coords[shown, 1], c=np.asarray(truth).reshape(-1)[shown],
        cmap=cmap, norm=norm, marker="o",
        s=float(cfg["questionB"].get("observed_marker_size", 10.0)),
        edgecolors=NEUTRAL_DARK, linewidths=LW_GRID, rasterized=False, zorder=2,
    )
    bounds = roi or [
        float(coords[:, 0].min()), float(coords[:, 0].max()),
        float(coords[:, 1].min()), float(coords[:, 1].max()),
    ]
    pad_x = .025 * (bounds[1] - bounds[0])
    pad_y = .025 * (bounds[3] - bounds[2])
    ax.set_xlim(bounds[0] - pad_x, bounds[1] + pad_x)
    ax.set_ylim(bounds[2] - pad_y, bounds[3] + pad_y)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(NEUTRAL_DARK)
        spine.set_linewidth(LW_DIVIDER)
    ax.set_title(
        f"Observed values in {scope} (n={len(shown)})",
        fontsize=max(5.0, cfg["figure_style"]["font_sizes"]["tick"] - .8),
        color=NEUTRAL_DARK, pad=1.6,
    )
    return {
        "total_count": int(len(obs)), "displayed_count": int(len(shown)),
        "scope": scope, "vector_markers": True,
    }


def draw_panel_f(parent, ctx: PublicationContext, *, standalone=False):
    cfg = ctx.cfg
    cache_before = set(ctx.used_cache_paths)
    parent.set_axis_off()
    models = ctx.model_order(cfg["questionB"]["qualitative_models"])
    recipe = cfg["questionB"].get("qualitative_recipe", "4_ZeroH_Balanced")
    parent.text(.01, 1.015, f"{cfg['recipes'][recipe]['label']} qualitative comparison",
                transform=parent.transAxes, ha="left", va="bottom",
                fontsize=cfg["figure_style"]["font_sizes"]["title"], fontweight="bold")
    snapshot = ctx.representatives.get("questionB", _int(cfg["canonical_test"]["representative_snapshot"]))
    count = _int(cfg["sensor_plan"]["default_count"])
    payload = {model: ctx.cache(model, recipe, snapshot, count) for model in models}
    available = [v for v in payload.values() if v is not None]
    if not available:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "models": models, "recipe": recipe, "snapshot": snapshot}
    arrays0, meta0, _ = available[0]
    truth = arrays0["truth_phys"].reshape(-1)
    coords = arrays0["coords_phys"]
    preds = [v[0]["recon_phys"].reshape(-1) for v in available]
    cmap, norm, field_limits = _field_norm(np.concatenate([truth, *preds]), cfg)
    errors = [np.abs(pred - truth) for pred in preds]
    error_cmap, error_norm, error_limits = _error_norm(np.concatenate(errors), cfg)
    roi_fraction = _float(cfg["questionB"].get("roi_fraction", .25))
    roi_mode = str(cfg["questionB"].get("qualitative_roi_selection", "truth_gradient")).strip().lower()
    roi_metadata = {}
    if roi_mode == "model_contrast":
        contrast_models = list(cfg["questionB"].get("qualitative_roi_models", ["DMFGen", "Senseiver"]))
        if len(contrast_models) != 2:
            raise ValueError("questionB.qualitative_roi_models must contain exactly two model keys")
        contrast_payload = [payload.get(model) for model in contrast_models]
        if any(loaded is None for loaded in contrast_payload):
            raise ValueError(
                "Model-contrast ROI requires caches for " + ", ".join(contrast_models)
            )
        contrast_predictions = [loaded[0]["recon_phys"].reshape(-1) for loaded in contrast_payload]
        gradient_quantile = _float(
            cfg["questionB"].get("qualitative_roi_min_gradient_quantile", .5), .5
        )
        roi = automatic_model_contrast_roi(
            coords, truth, contrast_predictions[0], contrast_predictions[1],
            fraction=roi_fraction, min_truth_gradient_quantile=gradient_quantile,
        )
        roi_metadata = {
            "selection": "maximum integrated absolute prediction difference among windows meeting the ground-truth gradient threshold",
            "selection_mode": "model_contrast",
            "contrast_models": contrast_models,
            "minimum_ground_truth_gradient_window_quantile": gradient_quantile,
        }
    elif roi_mode == "truth_gradient":
        roi = automatic_integrated_gradient_roi(coords, truth, fraction=roi_fraction)
        roi_metadata = {
            "selection": "maximum integrated ground-truth gradient magnitude",
            "selection_mode": "truth_gradient",
        }
    else:
        raise ValueError(
            "questionB.qualitative_roi_selection must be 'truth_gradient' or "
            f"'model_contrast', got {roi_mode!r}"
        )
    mask = _roi_mask(coords, roi)
    if roi_mode == "model_contrast":
        roi_metadata["mean_absolute_prediction_difference"] = float(
            np.mean(np.abs(contrast_predictions[0][mask] - contrast_predictions[1][mask]))
        )
    l2_scope = str(cfg["questionB"].get("qualitative_l2_scope", "full_field")).strip().lower()
    if l2_scope not in {"full_field", "roi"}:
        raise ValueError(
            "questionB.qualitative_l2_scope must be 'full_field' or 'roi', "
            f"got {l2_scope!r}"
        )
    l2_mask = mask if l2_scope == "roi" else None
    l2_label = "ROI L2" if l2_scope == "roi" else "Full L2"
    columns = [("Ground truth", None), *[(ctx.model_label(model), model) for model in models]]
    x0, gap = .035, .010
    width = (.98 - x0 - gap * (len(columns) - 1)) / len(columns)
    row_specs = [("Full field", .69, .24), ("ROI field", .395, .24), ("ROI |error|", .10, .24)]
    field_artist = error_artist = None
    missing = []
    observation_metadata = {}
    for j, (title, model) in enumerate(columns):
        x = x0 + j * (width + gap)
        parent.text(x + width / 2, .975, title, transform=parent.transAxes,
                    ha="center", va="top", fontsize=cfg["figure_style"]["font_sizes"]["title"])
        full_ax = _inset(parent, [x, row_specs[0][1], width, row_specs[0][2]])
        zoom_ax = _inset(parent, [x, row_specs[1][1], width, row_specs[1][2]])
        err_ax = _inset(parent, [x, row_specs[2][1], width, row_specs[2][2]])
        if model is None:
            field_artist = _draw_field(full_ax, coords, truth, cfg, cmap=cmap, norm=norm,
                                       mode="native_cells", contours=True,
                                       roi=roi, draw_roi=True)
            _draw_field(zoom_ax, coords, truth, cfg, cmap=cmap, norm=norm,
                        mode="native_cells", contours=True, crop=roi)
            _zoom_border(zoom_ax)
            zoom_ax.text(
                -.18, .5, "Zoomed-in region", transform=zoom_ax.transAxes,
                rotation=90, ha="center", va="center",
                fontsize=cfg["figure_style"]["font_sizes"]["tick"], color=NEUTRAL_DARK,
            )
            observation_metadata = _draw_observed_values(
                err_ax, coords, truth, arrays0["obs_indices"], cmap=cmap, norm=norm,
                cfg=cfg, roi=roi,
            )
            coord_mid_x = .5 * (float(coords[:, 0].min()) + float(coords[:, 0].max()))
            coord_mid_y = .5 * (float(coords[:, 1].min()) + float(coords[:, 1].max()))
            roi_mid_x = .5 * (roi[0] + roi[1])
            roi_mid_y = .5 * (roi[2] + roi[3])
            full_x, full_ha = (.03, "left") if roi_mid_x >= coord_mid_x else (.97, "right")
            full_y, full_va = (.97, "top") if roi_mid_y < coord_mid_y else (.03, "bottom")
            row_labels = (("Full field", full_ax, full_x, full_y, full_ha, full_va),
                          ("ROI field", zoom_ax, .03, .97, "left", "top"))
            for label, target, tx, ty, ha, va in row_labels:
                target.text(tx, ty, label, transform=target.transAxes, ha=ha, va=va,
                            fontsize=max(5.2, cfg["figure_style"]["font_sizes"]["tick"] - .8),
                            color=NEUTRAL_DARK, bbox=dict(boxstyle="round,pad=.06", fc="white", ec="none", alpha=.72))
            continue
        loaded = payload[model]
        if loaded is None:
            status = ctx.cache_status(model, recipe, snapshot, count)
            for ax in (full_ax, zoom_ax, err_ax):
                mark_missing(ax, f"Missing\n{status}", cfg)
            missing.append({"model": model, "recipe": recipe, "status": status})
            continue
        pred = loaded[0]["recon_phys"].reshape(-1)
        _draw_field(full_ax, coords, pred, cfg, cmap=cmap, norm=norm,
                    mode="native_cells", contours=True, roi=roi, draw_roi=True)
        _draw_field(zoom_ax, coords, pred, cfg, cmap=cmap, norm=norm,
                    mode="native_cells", contours=True, crop=roi)
        _zoom_border(zoom_ax)
        err = np.abs(pred - truth)
        error_artist = _draw_field(err_ax, coords, err, cfg, cmap=error_cmap,
                                   norm=error_norm, mode="native_cells", crop=roi)
        err_ax.text(
            .03, .03, f"{l2_label}={relative_l2(truth, pred, l2_mask):.3f}", transform=err_ax.transAxes,
            ha="left", va="bottom", color="white", fontsize=cfg["figure_style"]["font_sizes"]["tick"],
            bbox=dict(boxstyle="round,pad=.10", fc="black", ec="none", alpha=.48),
        )
    if field_artist is not None:
        _horizontal_colorbar(parent, field_artist, [.22, .025, .25, .015],
                             meta0.get("selected_raw_field_name", "field"), cfg)
    if error_artist is not None:
        _horizontal_colorbar(parent, error_artist, [.63, .025, .25, .015], "Absolute error", cfg)
    return {
        "status": "ok", "cache_sources": sorted(ctx.used_cache_paths - cache_before), "models": models,
        "recipe": recipe, "snapshot": snapshot, "case_id": meta0.get("case_id"),
        "time_index": meta0.get("time_index"), "field": meta0.get("selected_raw_field_name"),
        "sensor_count": count, "roi": {
            "xmin": roi[0], "xmax": roi[1], "ymin": roi[2], "ymax": roi[3],
            **roi_metadata,
        },
        "field_limits": field_limits, "error_limits": error_limits,
        "field_cmap": cmap, "error_cmap": error_cmap, "missing": missing,
        "l2_annotation_scope": l2_scope,
        "l2_annotation_metric": "physical relative L2",
        "ground_truth_sensor_overlay": False,
        "observation_schematic": {
            **observation_metadata,
            "marker": "solid value-colored circle with black observed-value border",
        },
        "zoom_border": "black, shared ROI across truth and all models",
    }


def draw_panel_g(parent, ctx: PublicationContext, *, standalone=False, show_legend=True):
    cfg = ctx.cfg
    path, rows = ctx.source("FrequencyError", "FrequencyError_summary")
    parent.set_axis_off()
    if not rows:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "sources": [str(path)]}
    models = ctx.model_order(cfg["frequency_error"].get("main_models"))
    recipes = cfg["frequency_error"]["recipes"]
    colors = model_colors(cfg)
    alphas = model_alphas(cfg)
    all_y = np.asarray([_float(r.get("q75")) for r in rows if r.get("model") in models and r.get("recipe") in recipes])
    ymax = float(np.nanquantile(all_y, .995)) * 1.08
    nyq_l, nyq_m = _float(rows[0]["L_nyquist_normalized"]), _float(rows[0]["M_nyquist_normalized"])
    valid_ns = []
    subplot_width = .275
    subplot_gap = .035
    subplot_left = .07
    for j, recipe in enumerate(recipes):
        ax = _inset(parent, [subplot_left + j * (subplot_width + subplot_gap), .17, subplot_width, .68])
        for model in models:
            subset = sorted([r for r in rows if r.get("model") == model and r.get("recipe") == recipe], key=lambda r: _int(r["shell_index"]))
            if not subset:
                continue
            x = np.asarray([_float(r["k_normalized_H_nyquist"]) for r in subset])
            y = np.asarray([_float(r["median"]) for r in subset])
            lo = np.asarray([_float(r["q25"]) for r in subset])
            hi = np.asarray([_float(r["q75"]) for r in subset])
            ax.plot(x, y, color=colors[model], alpha=alphas[model], lw=LW_LINE_PLOT)
            ax.fill_between(x, lo, hi, color=colors[model], alpha=.10, lw=0)
            valid_ns.extend(_int(r.get("valid_n")) for r in subset)
        for xpos, label in ((nyq_l, "L Nyq."), (nyq_m, "M Nyq.")):
            ax.axvline(xpos, color=NEUTRAL_MID, ls=":", lw=LW_LINE_SECONDARY, zorder=0)
            ax.text(xpos + .008, .97, label, transform=ax.get_xaxis_transform(), rotation=90,
                    ha="left", va="top",
                    fontsize=cfg["figure_style"]["font_sizes"]["tick"], color=NEUTRAL_DARK)
        ax.set_xlim(0, 1.02); ax.set_ylim(0, ymax)
        ax.set_title(cfg["recipes"][recipe]["label"], pad=3)
        ax.set_xlabel(r"$k/k_{\mathrm{Nyq},H}$")
        if j == 0:
            ax.set_ylabel("Spectral error (dB)")
        else:
            ax.tick_params(labelleft=False)
    if show_legend:
        parent.legend(handles=model_legend_handles(cfg), ncol=2, loc="lower center",
                      bbox_to_anchor=(.5, .88), columnspacing=1.0, handletextpad=.35)
    ns = [n for n in valid_ns if n > 0]
    if ns and standalone:
        parent.text(.99, .02, f"n={min(ns)}", transform=parent.transAxes, ha="right", va="bottom",
                    fontsize=cfg["figure_style"]["font_sizes"]["tick"])
    return {"status": "ok", "sources": [str(path)], "models": models, "recipes": recipes,
            "L_nyquist_normalized": nyq_l, "M_nyquist_normalized": nyq_m,
            "statistic": "median with interquartile range", "shared_y_range": [0.0, ymax]}


def draw_panel_h(parent, ctx: PublicationContext, *, standalone=False, show_legend=True):
    cfg = ctx.cfg
    path, rows = ctx.source("SensorSweep", "SensorSweep_summary")
    rows = [r for r in rows if r.get("metric") == "physical_rel_l2"]
    parent.set_axis_off()
    if not rows:
        mark_missing(parent, cfg=cfg)
        return {"status": "missing", "sources": [str(path)]}
    models = ctx.model_order(cfg["sensor_sweep"].get("main_models"))
    recipes = cfg["sensor_sweep"]["recipes"]
    colors = model_colors(cfg)
    alphas = model_alphas(cfg)
    formal = _int(cfg["sensor_plan"]["default_count"])
    all_hi = np.asarray([_float(r["ci95_high"]) for r in rows if r.get("model") in models and r.get("recipe") in recipes])
    ymin = max(0.0, float(np.nanmin([_float(r["ci95_low"]) for r in rows])) * .86)
    ymax = float(np.nanmax(all_hi)) * 1.08
    valid_ns = []
    counts = [int(value) for value in cfg["sensor_sweep"]["counts"]]
    summary_grid_points = max((_int(row.get("evaluation_grid_points"), 0) for row in rows), default=0)
    h_row = next((row for row in ctx.manifest_rows if row.get("status") == "ok" and _int(row.get("num_x")) > 0), {})
    h_points = summary_grid_points or max(1, _int(h_row.get("num_x"), 128) * _int(h_row.get("num_y"), 128))
    tick_labels = []
    compact_tick_labels = []
    for count in counts:
        density = 100.0 * count / h_points
        density_text = f"{density:.1f}"
        tick_labels.append(f"{count}\n{density_text}")
        compact_tick_labels.append(f"{count}/{density_text}")
    subplot_width = .275
    subplot_gap = .035
    subplot_left = .07
    x_positions = np.arange(len(counts))
    formal_position = counts.index(formal) if formal in counts else int(np.argmin(np.abs(np.asarray(counts) - formal)))
    subaxes = []
    for j, recipe in enumerate(recipes):
        ax = _inset(parent, [subplot_left + j * (subplot_width + subplot_gap), .25, subplot_width, .60])
        subaxes.append(ax)
        for model in models:
            subset = sorted([r for r in rows if r.get("model") == model and r.get("recipe") == recipe], key=lambda r: _int(r["sensor_count"]))
            if not subset:
                continue
            subset = [r for r in subset if _int(r["sensor_count"]) in counts]
            x = np.asarray([counts.index(_int(r["sensor_count"])) for r in subset])
            y = np.asarray([_float(r["mean"]) for r in subset])
            lo = np.asarray([_float(r["ci95_low"]) for r in subset])
            hi = np.asarray([_float(r["ci95_high"]) for r in subset])
            ax.plot(
                x, y, color=colors[model], alpha=alphas[model],
                marker="o", ms=3.0, lw=LW_LINE_PLOT,
            )
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=colors[model],
                        elinewidth=LW_ERRORBAR, capsize=1.4, alpha=alphas[model])
            valid_ns.extend(_int(r.get("valid_n")) for r in subset)
        ax.axvline(formal_position, color=NEUTRAL_MID, ls=":", lw=LW_LINE_SECONDARY, zorder=0)
        formal_label = "formal setting" if standalone else "formal"
        ax.text(formal_position + .12, .97, formal_label, transform=ax.get_xaxis_transform(), rotation=90,
                ha="left", va="top", fontsize=cfg["figure_style"]["font_sizes"]["tick"], color=NEUTRAL_DARK)
        ax.set_xticks(x_positions, tick_labels if standalone else compact_tick_labels)
        ax.tick_params(
            axis="x", labelsize=max(5.0, cfg["figure_style"]["font_sizes"]["tick"] - 1.2), pad=1.2,
            labelrotation=0 if standalone else 90,
        )
        ax.set_xlim(-.35, len(counts) - .65)
        ax.set_ylim(ymin, ymax)
        ax.set_title(cfg["recipes"][recipe]["label"], pad=3)
        if j == 0:
            ax.set_ylabel("Physical relative L2")
        else:
            ax.tick_params(labelleft=False)
    if show_legend:
        parent.legend(handles=model_legend_handles(cfg), ncol=2, loc="lower center",
                      bbox_to_anchor=(.5, .88), columnspacing=1.0, handletextpad=.35)
    if standalone:
        parent.text(.5, .005, "Sensor count / density (%)", transform=parent.transAxes,
                    ha="center", va="bottom", fontsize=cfg["figure_style"]["font_sizes"]["axis"])
    else:
        parent.text(.10, 1.015, "x ticks: count / density (%)", transform=parent.transAxes,
                    ha="left", va="bottom", fontsize=cfg["figure_style"]["font_sizes"]["tick"] - .5,
                    color=NEUTRAL_MID)
    ns = [n for n in valid_ns if n > 0]
    if ns and standalone:
        subaxes[0].text(.02, .03, f"n={min(ns)}", transform=subaxes[0].transAxes,
                        ha="left", va="bottom", fontsize=cfg["figure_style"]["font_sizes"]["tick"],
                        color=NEUTRAL_DARK)
    return {"status": "ok", "sources": [str(path)], "models": models, "recipes": recipes,
            "sensor_counts": counts, "sensor_density_percent": [100.0 * count / h_points for count in counts],
            "evaluation_grid_points": h_points, "formal_setting": formal,
            "statistic": "mean with bootstrap 95% CI", "shared_y_range": [ymin, ymax]}


PANEL_DRAWERS = {
    "a": draw_panel_a, "b": draw_panel_b, "c": draw_panel_c, "d": draw_panel_d,
    "e": draw_panel_e, "f": draw_panel_f, "g": draw_panel_g, "h": draw_panel_h,
}


def draw_panel(label: str, ax, ctx: PublicationContext, *, standalone=False, show_legend=True):
    kwargs = {"standalone": standalone}
    if label in {"g", "h"}:
        kwargs["show_legend"] = show_legend
    return PANEL_DRAWERS[label](ax, ctx, **kwargs)


def standalone_output_base(label: str, rid: str) -> Path:
    return FIGURES_DIR / "PublicationPanels" / f"{PANEL_OUTPUT_NAMES[label]}_{rid}"

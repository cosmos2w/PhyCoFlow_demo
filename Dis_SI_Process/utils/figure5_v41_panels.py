"""Matplotlib renderers for the Figure 5 V4.1 main and Zero-H backup bundles."""
from __future__ import annotations

import hashlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure5_v41_style import (
    AXIS,
    DMF_HIGHLIGHT,
    MM,
    add_panel_label,
    method_alpha,
    method_colors,
    method_legend_handles,
    method_markers,
    style_grid,
)


def _jitter(count: int, salt: str, scale: float = 0.115) -> np.ndarray:
    seed = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed).uniform(-scale, scale, size=count)


def _method_rows(methods: list[str]) -> np.ndarray:
    return np.arange(len(methods), dtype=float)[::-1]


def _draw_boxscatter(
    ax,
    samples: pd.DataFrame,
    summary: pd.DataFrame | None,
    config: dict[str, Any],
    *,
    methods: list[str],
    value_column: str,
    estimate_column: str | None,
    low_column: str | None,
    high_column: str | None,
    xlabel: str,
    title: str,
    panel_label: str,
    show_methods: bool,
    reference: float | None = None,
    bootstrap_scatter: bool = False,
    log_x: bool = False,
) -> None:
    colors, markers = method_colors(config), method_markers(config)
    positions = _method_rows(methods)
    for index, method in enumerate(methods):
        values = samples.loc[samples["method"].astype(str).eq(method), value_column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError(f"No finite {value_column} samples for {method}")
        position = positions[index]
        if method == "DMF-Gen":
            ax.axhspan(position - 0.43, position + 0.43, color=DMF_HIGHLIGHT, alpha=0.055, zorder=0)
        box = ax.boxplot(
            [values],
            positions=[position],
            vert=False,
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            whis=(2.5, 97.5) if bootstrap_scatter else 1.5,
            medianprops={"color": colors[method], "linewidth": 1.15},
            boxprops={"facecolor": colors[method], "edgecolor": colors[method], "alpha": 0.16, "linewidth": 0.8},
            whiskerprops={"color": colors[method], "alpha": 0.65, "linewidth": 0.7},
            capprops={"color": colors[method], "alpha": 0.65, "linewidth": 0.7},
            zorder=2,
        )
        for patch in box["boxes"]:
            patch.set_gid(f"distribution:{panel_label}:{method}")
        if bootstrap_scatter and len(values) > 140:
            selected = np.linspace(0, len(values) - 1, 140, dtype=int)
            scatter_values = values[selected]
        else:
            scatter_values = values
        ax.scatter(
            scatter_values,
            position + _jitter(len(scatter_values), f"{panel_label}|{method}"),
            s=5.0,
            marker="o",
            color=colors[method],
            alpha=0.18 if method != "DMF-Gen" else 0.23,
            linewidths=0,
            zorder=1.5,
            rasterized=False,
        )
        if summary is not None and estimate_column:
            row = summary.loc[summary["method"].astype(str).eq(method)].iloc[0]
            estimate = float(row[estimate_column])
            if low_column and high_column:
                lo, hi = float(row[low_column]), float(row[high_column])
                ax.hlines(position, lo, hi, color=colors[method], linewidth=1.1, zorder=3.2)
                ax.vlines((lo, hi), position - 0.10, position + 0.10, color=colors[method], linewidth=0.9, zorder=3.2)
            ax.plot(
                estimate,
                position,
                marker=markers[method],
                markersize=5.2 if method == "DMF-Gen" else 4.6,
                markerfacecolor="white",
                markeredgecolor=colors[method],
                markeredgewidth=0.9,
                linestyle="none",
                zorder=4,
            )
    if reference is not None:
        ax.axvline(reference, color="#6C757D", linestyle=(0, (3, 2)), linewidth=0.72, zorder=1)
    ax.set_ylim(-0.58, len(methods) - 0.42)
    ax.set_yticks(positions)
    if show_methods:
        ax.set_yticklabels(methods)
        for tick, method in zip(ax.get_yticklabels(), methods):
            tick.set_color(colors[method])
            if method == "DMF-Gen":
                tick.set_fontweight("bold")
    else:
        ax.tick_params(axis="y", labelleft=False, length=0)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="semibold", pad=4)
    style_grid(ax, axis="x")
    add_panel_label(ax, panel_label)


def draw_crps(ax, data: dict[str, Any], config: dict[str, Any], *, show_methods: bool = True) -> None:
    _draw_boxscatter(
        ax,
        data["uq_crps_samples"],
        data["uq_crps"],
        config,
        methods=list(config["paper_contract"]["generative_method_order"]),
        value_column="normalized_crps",
        estimate_column="mean_normalized_crps",
        low_column="crps_ci_low",
        high_column="crps_ci_high",
        xlabel="Normalized CRPS (lower is better)",
        title="State-wise normalized CRPS",
        panel_label="a",
        show_methods=show_methods,
    )


def draw_spearman(ax, data: dict[str, Any], config: dict[str, Any], *, show_methods: bool = False) -> None:
    _draw_boxscatter(
        ax,
        data["uq_spearman_bootstrap"],
        data["uq_spread"],
        config,
        methods=list(config["paper_contract"]["generative_method_order"]),
        value_column="spearman_rho",
        estimate_column="spearman_rho",
        low_column=None,
        high_column=None,
        xlabel="Spearman ρ",
        title="Bootstrap spread–error association",
        panel_label="b",
        show_methods=show_methods,
        reference=0.0,
        bootstrap_scatter=True,
    )


def _shared_log_error_limits(*tables: pd.DataFrame | None) -> tuple[float, float]:
    values: list[float] = []
    for table in tables:
        if table is None:
            continue
        rows = table.loc[table["status"].astype(str).str.lower().eq("ok")] if "status" in table else table
        for column in ("error_ci_low", "error_ci_high"):
            if column in rows:
                values.extend(float(value) for value in rows[column] if np.isfinite(value) and float(value) > 0)
    if not values:
        return (0.05, 1.0)
    return (min(values) / 1.16, max(values) * 1.16)


def draw_accuracy_cost(
    ax,
    table: pd.DataFrame,
    config: dict[str, Any],
    *,
    title: str,
    xlabel: str,
    panel_label: str,
    ylabel: str | None,
    ylim: tuple[float, float],
) -> None:
    methods = list(config["paper_contract"]["method_order"])
    colors, markers = method_colors(config), method_markers(config)
    ok = table.loc[table["status"].astype(str).str.lower().eq("ok")].copy()
    for method in methods:
        match = ok.loc[ok["method"].astype(str).eq(method)]
        if match.empty:
            continue
        row = match.iloc[0]
        x, y = float(row["cost_value"]), float(row["error"])
        xerr = np.asarray([[x - float(row["cost_low"])], [float(row["cost_high"]) - x]])
        yerr = np.asarray([[y - float(row["error_ci_low"])], [float(row["error_ci_high"]) - y]])
        ax.errorbar(
            x,
            y,
            xerr=np.maximum(xerr, 0.0),
            yerr=np.maximum(yerr, 0.0),
            fmt=markers[method],
            color=colors[method],
            markeredgecolor=colors[method],
            markeredgewidth=0.6,
            markersize=5.4 if method == "DMF-Gen" else 4.7,
            alpha=method_alpha(method),
            ecolor=colors[method],
            elinewidth=0.78,
            capsize=1.55,
            capthick=0.72,
            zorder=3,
        )
        if panel_label == "d" and method == "Geo-FNO" and int(float(row.get("device_count", 1))) == 2:
            peak = float(row.get("peak_allocated_mib_per_device_max", np.nan)) / 1024.0
            metric = str(row.get("cost_metric", ""))
            note = (
                "2-GPU DDP"
                if metric == "training_update_time_ms"
                else "2 GPU"
                if not np.isfinite(peak)
                else f"2 GPU; {peak:.1f} GiB/device"
            )
            ax.annotate(
                note,
                (x, y),
                xytext=(6, 7) if metric == "training_update_time_ms" else (-5, -10),
                textcoords="offset points",
                ha="left" if metric == "training_update_time_ms" else "right",
                fontsize=5.7,
                color=colors[method],
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    positive_x = ok["cost_value"].to_numpy(dtype=float)
    positive_x = positive_x[np.isfinite(positive_x) & (positive_x > 0)]
    ax.set_xlim(positive_x.min() / 1.30, positive_x.max() * 1.35)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_title(title, loc="left", fontweight="semibold", pad=4)
    style_grid(ax)
    ax.text(0.02, 0.035, "lower left preferable", transform=ax.transAxes, fontsize=5.4, color="#6C757D")
    unavailable = table.loc[~table["status"].astype(str).str.lower().eq("ok")]
    if not unavailable.empty:
        ax.text(
            0.98,
            0.96,
            ", ".join(unavailable["method"].astype(str)) + ": unavailable (see SI)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.5,
            color="#6C757D",
        )
    add_panel_label(ax, panel_label)


def _support_map(table: pd.DataFrame, methods: list[str]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for method in methods:
        match = table.loc[table["method"].astype(str).eq(method), "variable_query_supported"]
        value = match.iloc[0] if not match.empty else False
        result[method] = value if isinstance(value, bool) else str(value).lower() in {"true", "1", "yes"}
    return result


def draw_memory_scaling(ax, data: dict[str, Any], config: dict[str, Any], *, show_legend: bool = True) -> None:
    table = data["scale_memory"]
    support = _support_map(data["query_support"], list(config["paper_contract"]["method_order"]))
    methods = list(config["paper_contract"]["method_order"])
    colors, markers = method_colors(config), method_markers(config)
    native = int(config["formal_protocol"]["scale_stress"]["native_limit"])
    xmax = max(int(table["N"].max()), int(config["formal_protocol"]["scale_stress"]["adaptive_query_cap"]))
    for method in methods:
        group = table.loc[table["method"].astype(str).eq(method)].sort_values("N")
        ok = group.loc[group["status"].astype(str).str.lower().eq("ok")]
        if support.get(method, False) and not ok.empty:
            ax.plot(
                ok["N"],
                ok["peak_allocated_mib"],
                color=colors[method],
                marker=markers[method],
                markersize=4.2,
                linewidth=1.1,
                alpha=method_alpha(method),
                zorder=3,
            )
        elif not support.get(method, False):
            row = ok.loc[ok["N"].astype(int).eq(native)]
            if not row.empty:
                ax.plot(
                    native,
                    float(row.iloc[0]["peak_allocated_mib"]),
                    marker=markers[method],
                    markersize=5.2,
                    markerfacecolor="white",
                    markeredgecolor=colors[method],
                    markeredgewidth=1.0,
                    linestyle="none",
                    zorder=4,
                )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(max(1.0, float(table["N"].min()) * 0.82), xmax * 1.12)
    ax.axvspan(native, xmax * 1.12, color="#F1F3F5", alpha=0.82, zorder=-2)
    ax.axvline(native, color="#6C757D", linestyle=(0, (3, 2)), linewidth=0.74, zorder=1)
    ticks = [1024, 4096, 16384, 40300, 100000, 250000, 500000, 1000000, 2000000, 4000000, 8000000]
    ticks = [value for value in ticks if value <= xmax]

    def compact(value: int) -> str:
        if value == 40300:
            return "40.3k"
        if value >= 1_000_000:
            return f"{value / 1_000_000:g}M"
        if value >= 1000:
            return f"{value / 1000:g}k"
        return str(value)

    ax.set_xticks(ticks)
    ax.set_xticklabels([compact(value) for value in ticks])
    ax.set_xlabel("Requested query points, N")
    ax.set_ylabel("Peak allocated memory (MiB)")
    ax.set_title("Peak allocated memory versus requested query count", loc="left", fontweight="semibold", pad=4)
    ax.text(0.73, 0.93, "throughput-only beyond 40.3k; no accuracy claim", transform=ax.transAxes, ha="center", va="top", fontsize=6.0, color="#6C757D")
    style_grid(ax)
    add_panel_label(ax, "e", x=-0.055, y=1.045)
    if show_legend:
        ax.legend(
            handles=method_legend_handles(config),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.16),
            ncol=8,
            fontsize=6.7,
            handlelength=1.15,
            columnspacing=0.70,
            handletextpad=0.30,
            borderaxespad=0.0,
        )


def make_standalone(panel: str, data: dict[str, Any], config: dict[str, Any]):
    if panel in {"a", "b"}:
        fig, ax = plt.subplots(figsize=(89 * MM, 70 * MM))
        fig.subplots_adjust(left=0.21, right=0.98, bottom=0.17, top=0.86)
        (draw_crps if panel == "a" else draw_spearman)(ax, data, config, show_methods=True)
        return fig
    if panel in {"c", "d"}:
        fig, ax = plt.subplots(figsize=(89 * MM, 74 * MM))
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.28, top=0.86)
        ylim = _shared_log_error_limits(data["cost_native"], data["training_cost"])
        if panel == "c":
            draw_accuracy_cost(ax, data["cost_native"], config, title="Native 40,300-point accuracy–latency", xlabel="Warm model-core latency (ms)", panel_label="c", ylabel="Mean unobserved-field relative L2", ylim=ylim)
        else:
            draw_accuracy_cost(ax, data["training_cost"], config, title="Canonical training accuracy–memory", xlabel=data["training_metric_label"], panel_label="d", ylabel="Mean unobserved-field relative L2", ylim=ylim)
        fig.legend(handles=method_legend_handles(config), loc="lower center", bbox_to_anchor=(0.56, 0.012), ncol=4, fontsize=6.2, handlelength=1.1, columnspacing=0.75, handletextpad=0.3)
        return fig
    if panel == "e":
        fig, ax = plt.subplots(figsize=(183 * MM, 72 * MM))
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.78)
        draw_memory_scaling(ax, data, config, show_legend=True)
        return fig
    raise ValueError(panel)


def make_composed(data: dict[str, Any], config: dict[str, Any]):
    layout = config["figure"]["layout"]
    margins = layout["margins"]
    fig = plt.figure(figsize=(float(config["figure"]["width_mm"]) * MM, float(config["figure"]["composed_height_mm"]) * MM))
    outer = fig.add_gridspec(
        3,
        1,
        height_ratios=layout["height_ratios"],
        left=margins["left"],
        right=margins["right"],
        bottom=margins["bottom"],
        top=margins["top"],
        hspace=layout["outer_hspace"],
    )
    ab = outer[0].subgridspec(1, 2, wspace=layout["ab_wspace"])
    ax_a = fig.add_subplot(ab[0, 0])
    ax_b = fig.add_subplot(ab[0, 1], sharey=ax_a)
    draw_spearman(ax_b, data, config, show_methods=False)
    # Draw the labeled shared-y axis last: setting ticks on a shared partner
    # can otherwise replace the method formatter with numeric positions.
    draw_crps(ax_a, data, config, show_methods=True)

    cd = outer[1].subgridspec(1, 2, wspace=layout["cd_wspace"])
    ax_c = fig.add_subplot(cd[0, 0])
    ax_d = fig.add_subplot(cd[0, 1], sharey=ax_c)
    ylim = _shared_log_error_limits(data["cost_native"], data["training_cost"])
    draw_accuracy_cost(ax_c, data["cost_native"], config, title="Native inference accuracy–latency", xlabel="Warm model-core latency (ms)", panel_label="c", ylabel="Mean unobserved-field relative L2", ylim=ylim)
    draw_accuracy_cost(ax_d, data["training_cost"], config, title="Training accuracy–memory", xlabel=data["training_metric_label"], panel_label="d", ylabel=None, ylim=ylim)

    ax_e = fig.add_subplot(outer[2])
    draw_memory_scaling(ax_e, data, config, show_legend=True)
    return fig


BACKUP_METRICS = {
    "a": ("physical_rel_l2", "Physical relative L2", "Zero-H-balanced physical reconstruction error", False),
    "b": ("gradient_rel_l2", "Gradient relative L2", "Zero-H-balanced gradient fidelity", False),
    "c": ("physical_rel_l2_sensor_excluded", "Sensor-excluded relative L2", "Zero-H-balanced sensor-excluded error", True),
    "d": ("normalized_rel_l2", "Normalized relative L2", "Zero-H-balanced normalized error", True),
}


def draw_backup(ax, panel: str, data: dict[str, Any], config: dict[str, Any], *, show_methods: bool) -> None:
    metric, xlabel, title, log_x = BACKUP_METRICS[panel]
    _draw_boxscatter(
        ax,
        data["zeroh"],
        None,
        config,
        methods=list(config["paper_contract"]["zeroh_method_order"]),
        value_column=metric,
        estimate_column=None,
        low_column=None,
        high_column=None,
        xlabel=xlabel + " (lower is better)",
        title=title,
        panel_label=panel,
        show_methods=show_methods,
        log_x=log_x,
    )


def make_backup_standalone(panel: str, data: dict[str, Any], config: dict[str, Any]):
    fig, ax = plt.subplots(figsize=(89 * MM, 70 * MM))
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.85)
    draw_backup(ax, panel, data, config, show_methods=True)
    return fig


def make_backup_composed(data: dict[str, Any], config: dict[str, Any]):
    fig = plt.figure(figsize=(183 * MM, 104 * MM))
    grid = fig.add_gridspec(2, 2, left=0.12, right=0.985, bottom=0.10, top=0.96, wspace=0.15, hspace=0.58)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    for panel, ax in zip("abcd", axes):
        draw_backup(ax, panel, data, config, show_methods=panel in {"a", "c"})
    return fig

"""Publication renderers for the focused four-panel Figure 5 V5."""
from __future__ import annotations

import hashlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .figure5_v41_style import (
    AXIS,
    DMF_HIGHLIGHT,
    MM,
    add_panel_label,
    method_alpha,
    method_colors,
    method_markers,
    style_grid,
)


def _jitter(count: int, salt: str, scale: float = 0.105) -> np.ndarray:
    seed = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed).uniform(-scale, scale, size=count)


def _distribution_panel(
    ax,
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
    *,
    value: str,
    estimate: str,
    low: str,
    high: str,
    xlabel: str,
    title: str,
    panel: str,
    show_methods: bool,
    reference: float | None = None,
) -> None:
    methods = list(config["paper_contract"]["generative_method_order"])
    colors, markers = method_colors(config), method_markers(config)
    positions = np.arange(len(methods), dtype=float)[::-1]
    for index, method in enumerate(methods):
        position = positions[index]
        values = samples.loc[samples["method"].astype(str).eq(method), value].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if method == "DMF-Gen":
            ax.axhspan(position - 0.43, position + 0.43, color=DMF_HIGHLIGHT, alpha=0.055, zorder=0)
        box = ax.boxplot(
            [values],
            positions=[position],
            vert=False,
            widths=0.46,
            patch_artist=True,
            showfliers=False,
            whis=(2.5, 97.5) if panel == "b" else 1.5,
            medianprops={"color": colors[method], "linewidth": 1.0},
            boxprops={"facecolor": colors[method], "edgecolor": colors[method], "alpha": 0.14, "linewidth": 0.7},
            whiskerprops={"color": colors[method], "alpha": 0.55, "linewidth": 0.65},
            capprops={"color": colors[method], "alpha": 0.55, "linewidth": 0.65},
            zorder=2,
        )
        for patch in box["boxes"]:
            patch.set_gid(f"distribution:{panel}:{method}")
        display = values if len(values) <= 300 else values[np.linspace(0, len(values) - 1, 360, dtype=int)]
        ax.scatter(
            display,
            position + _jitter(len(display), f"v5|{panel}|{method}"),
            s=4.2,
            color=colors[method],
            alpha=0.16 if method != "DMF-Gen" else 0.21,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        row = summary.loc[summary["method"].astype(str).eq(method)].iloc[0]
        center, lo, hi = float(row[estimate]), float(row[low]), float(row[high])
        ax.hlines(position, lo, hi, color=colors[method], linewidth=1.05, zorder=3)
        ax.vlines((lo, hi), position - 0.09, position + 0.09, color=colors[method], linewidth=0.8, zorder=3)
        ax.plot(
            center,
            position,
            marker=markers[method],
            markersize=5.1 if method == "DMF-Gen" else 4.5,
            markerfacecolor="white",
            markeredgecolor=colors[method],
            markeredgewidth=0.9,
            linestyle="none",
            zorder=4,
        )
    if reference is not None:
        ax.axvline(reference, color="#6C757D", linestyle=(0, (3, 2)), linewidth=0.72, zorder=0.5)
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
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="semibold", pad=4)
    style_grid(ax, axis="x")
    add_panel_label(ax, panel, x=-0.12 if show_methods else -0.08, y=1.04)


def draw_crps(ax, data: dict[str, Any], config: dict[str, Any], *, show_methods: bool = True) -> None:
    _distribution_panel(
        ax,
        data["uq_crps_samples"],
        data["uq_crps"],
        config,
        value="normalized_crps",
        estimate="mean_normalized_crps",
        low="crps_ci_low",
        high="crps_ci_high",
        xlabel="Normalized CRPS (lower is better)",
        title="Probabilistic reconstruction",
        panel="a",
        show_methods=show_methods,
    )


def draw_spearman(ax, data: dict[str, Any], config: dict[str, Any], *, show_methods: bool = False) -> None:
    _distribution_panel(
        ax,
        data["uq_spearman_bootstrap"],
        data["uq_spread"],
        config,
        value="spearman_rho",
        estimate="spearman_rho",
        low="spearman_ci_low",
        high="spearman_ci_high",
        xlabel="Spearman ρ",
        title="Uncertainty tracks difficult states",
        panel="b",
        show_methods=show_methods,
        reference=0.0,
    )


def _method_handles(config: dict[str, Any], methods: list[str]) -> list[Line2D]:
    colors, markers = method_colors(config), method_markers(config)
    return [
        Line2D(
            [],
            [],
            color=colors[method],
            marker=markers[method],
            linewidth=1.2,
            markersize=4.0,
            label=method,
        )
        for method in methods
    ]


def draw_error_capture(ax, data: dict[str, Any], config: dict[str, Any], *, show_legend: bool = True) -> None:
    methods = list(config["paper_contract"]["generative_method_order"])
    colors, markers = method_colors(config), method_markers(config)
    table = data["localization"]["macro"]
    ax.plot([0, 1], [0, 1], color="#858585", linestyle=(0, (3, 2)), linewidth=0.8, label="Random ranking", zorder=1)
    for method in methods:
        group = table.loc[table["method"].astype(str).eq(method)].sort_values("spatial_fraction")
        x = np.concatenate(([0.0], group["spatial_fraction"].to_numpy(dtype=float)))
        y = np.concatenate(([0.0], group["metric_value"].to_numpy(dtype=float)))
        lo = np.concatenate(([0.0], group["ci_low"].to_numpy(dtype=float)))
        hi = np.concatenate(([0.0], group["ci_high"].to_numpy(dtype=float)))
        ax.fill_between(x, lo, hi, color=colors[method], alpha=0.075, linewidth=0, zorder=1.5)
        ax.plot(
            x,
            y,
            color=colors[method],
            marker=markers[method],
            markevery=range(1, len(x)),
            markersize=3.2,
            linewidth=1.25 if method == "DMF-Gen" else 1.0,
            alpha=method_alpha(method),
            zorder=3,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_xlabel("Most-uncertain spatial fraction")
    ax.set_ylabel("Reconstruction error captured")
    ax.set_title("Uncertainty localizes reconstruction error", loc="left", fontweight="semibold", pad=4)
    style_grid(ax)
    add_panel_label(ax, "c", x=-0.12, y=1.04)
    if show_legend:
        handles = _method_handles(config, methods)
        handles.append(Line2D([], [], color="#858585", linestyle=(0, (3, 2)), linewidth=0.8, label="Random"))
        ax.legend(
            handles=handles,
            loc="lower right",
            ncol=2,
            fontsize=5.4,
            handlelength=1.4,
            columnspacing=0.65,
            handletextpad=0.35,
            borderaxespad=0.35,
        )


def _bubble_size(error: np.ndarray | float, limits: tuple[float, float]) -> np.ndarray | float:
    low, high = limits
    value = np.asarray(error, dtype=float)
    scaled = 48.0 + 155.0 * np.clip((value - low) / max(high - low, 1e-12), 0.0, 1.0)
    return float(scaled) if np.ndim(error) == 0 else scaled


def draw_lifecycle(ax, data: dict[str, Any], config: dict[str, Any]) -> None:
    table = data["lifecycle"]["summary"]
    methods = list(config["paper_contract"]["method_order"])
    colors, markers = method_colors(config), method_markers(config)
    error_limits = (
        float(table["mean_unobserved_relative_l2"].min()),
        float(table["mean_unobserved_relative_l2"].max()),
    )
    offsets = {
        "DMF-Gen": (-5, 12),
        "FFM-FNO": (-7, -10),
        "FFM-Perceiver": (-5, -18),
        "Latent FM": (8, -12),
        "SiT": (-8, -12),
        "MLP-RBF": (6, 8),
        "Geo-FNO": (6, -14),
        "Senseiver": (-5, 13),
    }
    for method in methods:
        row = table.loc[table["method"].astype(str).eq(method)].iloc[0]
        x = float(row["native_latency_ms"])
        y = float(row["replay_equivalent_gpu_hours"])
        error = float(row["mean_unobserved_relative_l2"])
        ax.scatter(
            x,
            y,
            s=_bubble_size(error, error_limits),
            marker=markers[method],
            facecolor=colors[method],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.86,
            zorder=3,
        )
        dx, dy = offsets[method]
        ax.annotate(
            f"{method}\n{error:.3f}",
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=5.2,
            color=colors[method],
            linespacing=0.9,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(float(table["native_latency_ms"].min()) / 1.35, float(table["native_latency_ms"].max()) * 1.55)
    ax.set_ylim(
        float(table["replay_equivalent_gpu_hours"].min()) / 1.55,
        float(table["replay_equivalent_gpu_hours"].max()) * 1.55,
    )
    ax.set_xticks([3, 5, 10, 20])
    ax.set_xticklabels(["3", "5", "10", "20"])
    ax.set_yticks([20, 50, 100, 200, 500])
    ax.set_yticklabels(["20", "50", "100", "200", "500"])
    ax.set_xlabel("Warm native inference latency (ms)")
    ax.set_ylabel("Replay-equivalent training compute (GPU h)")
    ax.set_title("Training–inference lifecycle footprint", loc="left", fontweight="semibold", pad=4)
    style_grid(ax)
    ax.text(0.98, 0.03, "lower-left and smaller preferable", transform=ax.transAxes, ha="right", fontsize=5.2, color="#6C757D")
    refs = np.quantile(table["mean_unobserved_relative_l2"].to_numpy(dtype=float), [0.0, 0.5, 1.0])
    handles = [
        ax.scatter([], [], s=_bubble_size(value, error_limits), facecolor="none", edgecolor="#555555", linewidth=0.7)
        for value in refs
    ]
    legend = ax.legend(
        handles,
        [f"{value:.2f}" for value in refs],
        title="Mean unobserved-field\nrelative L₂",
        loc="upper center",
        bbox_to_anchor=(0.42, 0.985),
        ncol=3,
        fontsize=5.1,
        title_fontsize=5.3,
        labelspacing=0.4,
        columnspacing=0.7,
        handletextpad=0.6,
        borderaxespad=0.2,
    )
    ax.add_artist(legend)
    add_panel_label(ax, "d", x=-0.11, y=1.04)


def make_standalone(panel: str, data: dict[str, Any], config: dict[str, Any]):
    fig, ax = plt.subplots(figsize=(89 * MM, 67 * MM))
    if panel in {"a", "b"}:
        fig.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.86)
        (draw_crps if panel == "a" else draw_spearman)(ax, data, config, show_methods=True)
    elif panel == "c":
        fig.subplots_adjust(left=0.18, right=0.98, bottom=0.19, top=0.86)
        draw_error_capture(ax, data, config, show_legend=True)
    elif panel == "d":
        fig.subplots_adjust(left=0.19, right=0.96, bottom=0.20, top=0.86)
        draw_lifecycle(ax, data, config)
    else:
        raise ValueError(panel)
    return fig


def make_composed(data: dict[str, Any], config: dict[str, Any]):
    layout = config["figure"]["layout"]
    margins = layout["margins"]
    fig = plt.figure(
        figsize=(
            float(config["figure"]["width_mm"]) * MM,
            float(config["figure"]["composed_height_mm"]) * MM,
        )
    )
    grid = fig.add_gridspec(
        2,
        2,
        left=margins["left"],
        right=margins["right"],
        bottom=margins["bottom"],
        top=margins["top"],
        width_ratios=layout["width_ratios"],
        height_ratios=layout["height_ratios"],
        wspace=layout["wspace"],
        hspace=layout["hspace"],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)
    draw_spearman(ax_b, data, config, show_methods=False)
    draw_crps(ax_a, data, config, show_methods=True)
    draw_error_capture(fig.add_subplot(grid[1, 0]), data, config, show_legend=True)
    draw_lifecycle(fig.add_subplot(grid[1, 1]), data, config)
    return fig


def make_si_calibration(data: dict[str, Any], config: dict[str, Any]):
    methods = list(config["paper_contract"]["generative_method_order"])
    colors, markers = method_colors(config), method_markers(config)
    table = data["reliability_si"]
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 62 * MM))
    for method in methods:
        group = table.loc[table["method"].astype(str).eq(method)]
        macro = group.groupby("nominal_level", as_index=False)[["empirical_coverage", "mean_interval_width_normalized"]].mean()
        axes[0].plot(macro["nominal_level"], macro["empirical_coverage"], color=colors[method], marker=markers[method], ms=3.2, label=method)
        axes[1].plot(macro["nominal_level"], macro["mean_interval_width_normalized"], color=colors[method], marker=markers[method], ms=3.2)
    axes[0].plot([0.45, 1.0], [0.45, 1.0], color="#777777", linestyle=(0, (3, 2)), linewidth=0.8)
    axes[0].set(xlabel="Nominal central interval", ylabel="Empirical coverage")
    axes[1].set(xlabel="Nominal central interval", ylabel="Normalized interval width")
    for ax, title in zip(axes, ("Calibration / reliability", "Interval width")):
        style_grid(ax)
        ax.set_title(title, loc="left", fontweight="semibold")
    axes[0].legend(ncol=2, fontsize=5.5, loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.19, top=0.87, wspace=0.25)
    return fig


def make_si_fieldwise_uq(data: dict[str, Any], config: dict[str, Any]):
    methods = list(config["paper_contract"]["generative_method_order"])
    fields = list(config["paper_contract"]["unobserved_fields"])
    states = data["run_metadata"]["uq"]["states"]
    crps = np.empty((len(fields), len(methods)))
    rho = np.empty_like(crps)
    from scipy.stats import spearmanr

    for i, field in enumerate(fields):
        for j, method in enumerate(methods):
            group = states.loc[states["method"].astype(str).eq(method)].sort_values("original_time_index")
            crps[i, j] = float(group[f"crps_{field}"].mean())
            rho[i, j] = float(spearmanr(group[f"spread_{field}"], group[f"error_{field}"]).statistic)
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 62 * MM))
    for ax, matrix, title, cmap, limits in (
        (axes[0], crps, "Fieldwise normalized CRPS", "magma_r", (None, None)),
        (axes[1], rho, "Fieldwise spread–error Spearman ρ", "coolwarm", (-1, 1)),
    ):
        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=limits[0], vmax=limits[1])
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color="white" if abs(matrix[i, j]) > np.nanmedian(abs(matrix)) else "black")
        ax.set_xticks(range(len(methods)), labels=methods, rotation=32, ha="right")
        ax.set_yticks(range(len(fields)), labels=fields)
        ax.set_title(title, loc="left", fontweight="semibold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.25, top=0.88, wspace=0.28)
    return fig


def make_si_fieldwise_capture(data: dict[str, Any], config: dict[str, Any]):
    methods = list(config["paper_contract"]["generative_method_order"])
    fields = list(config["paper_contract"]["unobserved_fields"])
    colors, markers = method_colors(config), method_markers(config)
    table = data["localization"]["summary"]
    fig, axes = plt.subplots(2, 2, figsize=(174 * MM, 112 * MM), sharex=True, sharey=True)
    for ax, field in zip(axes.flat, fields):
        ax.plot([0, 1], [0, 1], color="#888888", linestyle=(0, (3, 2)), linewidth=0.7)
        for method in methods:
            group = table.loc[table["field"].astype(str).eq(field) & table["method"].astype(str).eq(method)].sort_values("spatial_fraction")
            x = np.concatenate(([0.0], group["spatial_fraction"].to_numpy(dtype=float)))
            y = np.concatenate(([0.0], group["metric_value"].to_numpy(dtype=float)))
            ax.plot(x, y, color=colors[method], marker=markers[method], ms=2.6, lw=0.9)
        ax.set_title(field, loc="left", fontweight="semibold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        style_grid(ax)
    axes[1, 0].set_xlabel("Most-uncertain spatial fraction")
    axes[1, 1].set_xlabel("Most-uncertain spatial fraction")
    axes[0, 0].set_ylabel("Error captured")
    axes[1, 0].set_ylabel("Error captured")
    fig.legend(handles=_method_handles(config, methods), loc="upper center", ncol=5, fontsize=5.8, bbox_to_anchor=(0.52, 0.99))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.10, top=0.88, wspace=0.18, hspace=0.28)
    return fig


def make_si_scalability(data: dict[str, Any], config: dict[str, Any]):
    methods = list(config["paper_contract"]["method_order"])
    colors, markers = method_colors(config), method_markers(config)
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 64 * MM))
    for ax, table, metric, ylabel in (
        (axes[0], data["scale_latency"], "median_latency_ms", "Warm latency (ms)"),
        (axes[1], data["scale_memory"], "peak_allocated_mib", "Peak allocated memory (MiB)"),
    ):
        for method in methods:
            group = table.loc[table["method"].astype(str).eq(method) & table["status"].astype(str).str.lower().eq("ok")].sort_values("N")
            if group.empty:
                continue
            ax.plot(group["N"], group[metric], color=colors[method], marker=markers[method], ms=3.0, lw=0.9, label=method)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(40300, color="#777777", linestyle=(0, (3, 2)), linewidth=0.7)
        ax.axvspan(40300, float(table["N"].max()) * 1.08, color="#F1F3F5", alpha=0.75, zorder=-2)
        ax.set_xlabel("Requested query points, N")
        ax.set_ylabel(ylabel)
        style_grid(ax)
    axes[0].set_title("40.3k–8M query latency stress", loc="left", fontweight="semibold")
    axes[1].set_title("40.3k–8M query memory stress", loc="left", fontweight="semibold")
    axes[0].legend(ncol=2, fontsize=5.2, loc="upper left")
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.87, wspace=0.28)
    return fig


def make_si_nfe(data: dict[str, Any], config: dict[str, Any]):
    table = data["nfe_si"].copy()
    fig, ax = plt.subplots(figsize=(86 * MM, 62 * MM))
    nfe_column = "measured_nfe" if "measured_nfe" in table.columns else "nfe"
    error_column = (
        "unobserved_mean_error"
        if "unobserved_mean_error" in table.columns
        else "mean_error"
        if "mean_error" in table.columns
        else "error"
    )
    latency_column = "median_latency_ms" if "median_latency_ms" in table.columns else None
    group = table.groupby(nfe_column, as_index=False).mean(numeric_only=True).sort_values(nfe_column)
    ax.plot(group[nfe_column], group[error_column], color=method_colors(config)["DMF-Gen"], marker="o", label="Relative L₂ error")
    ax.set_xlabel("Measured NFE")
    ax.set_ylabel("Mean unobserved-field relative L₂")
    style_grid(ax)
    if latency_column and latency_column in group:
        twin = ax.twinx()
        twin.plot(group[nfe_column], group[latency_column], color="#555555", marker="s", linestyle="--", label="Latency")
        twin.set_ylabel("Warm latency (ms)")
        twin.spines["top"].set_visible(False)
    ax.set_title("DMF-Gen solver / NFE diagnostics", loc="left", fontweight="semibold")
    fig.subplots_adjust(left=0.17, right=0.83, bottom=0.19, top=0.86)
    return fig

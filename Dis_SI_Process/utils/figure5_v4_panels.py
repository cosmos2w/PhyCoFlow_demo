"""Python/Matplotlib panel renderers for Figure 5 V4.

The renderers operate only on source tables supplied by
``figure5_v4_data.load_figure5_v4_data``.  A missing source is represented by a
visible pending panel in exploratory builds; no placeholder numbers are ever
drawn.
"""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure5_v4_style import (
    DIVIDER,
    DMF_HIGHLIGHT,
    PENDING_EDGE,
    PENDING_FACE,
    add_panel_label,
    method_alpha,
    method_colors,
    method_legend_handles,
    method_markers,
    style_grid,
)


def _pending(ax, label: str, title: str, requirement: str) -> None:
    """Draw an explicit missing-evidence panel, never a synthetic estimate."""

    ax.set_facecolor(PENDING_FACE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(PENDING_EDGE)
        spine.set_linestyle((0, (3, 2)))
        spine.set_linewidth(0.65)
    ax.set_xticks([])
    # Do not clear shared y tick locations: doing so also strips the paired
    # evidence panel of its quantitative scale when this panel is pending.
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_title(title, loc="left", fontweight="semibold", pad=4)
    ax.text(
        0.5,
        0.57,
        "V4 formal evidence pending",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.0,
        fontweight="bold",
        color="#5C6770",
    )
    ax.text(
        0.5,
        0.39,
        requirement,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=5.8,
        color="#5C6770",
        wrap=True,
    )
    add_panel_label(ax, label)


def _row_geometry(methods: list[str]) -> tuple[np.ndarray, tuple[float, float]]:
    y = np.arange(len(methods), dtype=float)[::-1]
    return y, (-0.55, float(len(methods) - 0.45))


def _highlight_rows(ax, methods: list[str], y: np.ndarray) -> None:
    dmf_index = methods.index("DMF-Gen")
    ax.axhspan(y[dmf_index] - 0.43, y[dmf_index] + 0.43, color=DMF_HIGHLIGHT, alpha=0.065, zorder=0)
    for row_y in y:
        ax.axhline(row_y, color=DIVIDER, linewidth=0.38, alpha=0.45, zorder=0)


def _draw_forest(
    ax,
    table: pd.DataFrame | None,
    config: dict[str, Any],
    *,
    estimate: str,
    low: str,
    high: str,
    xlabel: str,
    title: str,
    panel_label: str,
    show_methods: bool,
    ref: float | None = None,
    xlim: tuple[float, float] | None = None,
) -> None:
    methods = list(config["paper_contract"]["generative_method_order"])
    if table is None:
        _pending(ax, panel_label, title, "V3 formal paired source required; no estimate is substituted.")
        return
    colors, markers = method_colors(config), method_markers(config)
    y, ylim = _row_geometry(methods)
    _highlight_rows(ax, methods, y)
    for index, method in enumerate(methods):
        matches = table[table["method"].astype(str).eq(method)]
        if len(matches) != 1:
            raise ValueError(f"Forest table must contain exactly one row for {method}")
        row = matches.iloc[0]
        value, lo, hi = float(row[estimate]), float(row[low]), float(row[high])
        color = colors[method]
        # Draw a thin CI line and short caps explicitly so the forest grammar is
        # identical in panels a and b and remains editable in SVG.
        ax.hlines(y[index], lo, hi, color=color, linewidth=0.95, alpha=method_alpha(method), zorder=2)
        ax.vlines((lo, hi), y[index] - 0.105, y[index] + 0.105, color=color, linewidth=0.85, alpha=method_alpha(method), zorder=2)
        ax.plot(
            value,
            y[index],
            marker=markers[method],
            linestyle="none",
            color=color,
            markeredgecolor=color,
            markeredgewidth=0.55,
            markersize=5.0 if method == "DMF-Gen" else 4.3,
            alpha=method_alpha(method),
            zorder=3,
        )
    if ref is not None:
        ax.axvline(ref, color="#6C757D", linestyle=(0, (3, 2)), linewidth=0.7, zorder=1)
    ax.set_ylim(*ylim)
    ax.set_yticks(y)
    if show_methods:
        ax.set_yticklabels(methods)
        for tick, method in zip(ax.get_yticklabels(), methods):
            tick.set_color(colors[method])
            if method == "DMF-Gen":
                tick.set_fontweight("bold")
    else:
        ax.tick_params(axis="y", length=0, labelleft=False)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="semibold", pad=4)
    style_grid(ax, axis="x")
    add_panel_label(ax, panel_label)


def draw_crps(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "a", show_methods: bool = True) -> None:
    _draw_forest(
        ax,
        data.get("uq_crps"),
        config,
        estimate="mean_normalized_crps",
        low="crps_ci_low",
        high="crps_ci_high",
        xlabel="Normalized CRPS (lower is better)",
        title="Normalized CRPS over unobserved fields",
        panel_label=panel_label,
        show_methods=show_methods,
    )


def draw_spread_association(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "b", show_methods: bool = False) -> None:
    table = data.get("uq_spread")
    xlim = None
    if table is not None:
        xlim = (
            min(-0.08, float(table["spearman_ci_low"].min()) - 0.05),
            max(0.75, float(table["spearman_ci_high"].max()) + 0.05),
        )
    _draw_forest(
        ax,
        table,
        config,
        estimate="spearman_rho",
        low="spearman_ci_low",
        high="spearman_ci_high",
        xlabel="Spearman ρ",
        title="Uncertainty informativeness",
        panel_label=panel_label,
        show_methods=show_methods,
        ref=0.0,
        xlim=xlim,
    )


def _error_limits(native: pd.DataFrame | None, training: pd.DataFrame | None) -> tuple[float, float]:
    values: list[float] = []
    for table in (native, training):
        if table is None or "status" not in table:
            continue
        rows = table[table["status"].astype(str).str.lower().eq("ok")]
        for column in ("error_ci_low", "error_ci_high"):
            if column in rows.columns:
                values.extend(float(value) for value in rows[column].to_numpy(dtype=float) if np.isfinite(value))
    if not values:
        return (0.0, 1.0)
    lo, hi = min(values), max(values)
    span = max(hi - lo, 0.04)
    return (max(0.0, lo - 0.10 * span), hi + 0.10 * span)


def draw_accuracy_cost(
    ax,
    table: pd.DataFrame | None,
    config: dict[str, Any],
    *,
    x_column: str,
    x_low_column: str,
    x_high_column: str,
    xlabel: str,
    title: str,
    panel_label: str,
    ylabel: str | None,
    ylim: tuple[float, float] | None = None,
    show_legend: bool = False,
) -> None:
    methods = list(config["paper_contract"]["method_order"])
    if table is None:
        _pending(ax, panel_label, title, "V4 formal source required; no training/inference-cost value is substituted.")
        return
    colors, markers = method_colors(config), method_markers(config)
    rows = table[table["status"].astype(str).str.lower().eq("ok")].copy() if "status" in table else table.copy()
    if rows.empty:
        _pending(ax, panel_label, title, "No valid rows passed the formal source gate.")
        return
    for method in methods:
        match = rows[rows["method"].astype(str).eq(method)]
        if match.empty:
            continue
        row = match.iloc[0]
        x, y = float(row[x_column]), float(row["error"])
        xerr = np.asarray([[max(0.0, x - float(row[x_low_column]))], [max(0.0, float(row[x_high_column]) - x)]])
        yerr = np.asarray([[max(0.0, y - float(row["error_ci_low"]))], [max(0.0, float(row["error_ci_high"]) - y)]])
        color = colors[method]
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=markers[method],
            color=color,
            markeredgecolor=color,
            markeredgewidth=0.55,
            markersize=5.0 if method == "DMF-Gen" else 4.2,
            alpha=method_alpha(method),
            ecolor=color,
            elinewidth=0.75,
            capsize=1.45,
            capthick=0.7,
            zorder=3,
            label=method,
        )
    ax.set_xscale("log")
    valid_x = rows[x_column].astype(float).to_numpy()
    valid_x = valid_x[np.isfinite(valid_x) & (valid_x > 0)]
    if valid_x.size:
        ax.set_xlim(float(valid_x.min()) * 0.78, float(valid_x.max()) * 1.28)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(*_error_limits(None, table))
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_title(title, loc="left", fontweight="semibold", pad=4)
    style_grid(ax)
    ax.text(
        0.02,
        0.035,
        "lower left preferable",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=4.9,
        color="#6C757D",
    )
    unavailable = table[~table["status"].astype(str).str.lower().eq("ok")] if "status" in table else table.iloc[0:0]
    if not unavailable.empty:
        names = ", ".join(unavailable["method"].astype(str))
        ax.text(
            0.98,
            0.96,
            f"{names}: unavailable (see SI)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.2,
            color="#6C757D",
        )
    add_panel_label(ax, panel_label)
    if show_legend:
        ax.legend(handles=method_legend_handles(config), loc="upper left", ncol=2, handlelength=1.5, columnspacing=0.8, borderaxespad=0.2)


def _support_map(support: pd.DataFrame | None, methods: list[str]) -> dict[str, bool]:
    if support is None or "method" not in support.columns:
        return {method: False for method in methods}
    return {
        method: bool(
            _as_bool(support.loc[support["method"].astype(str).eq(method), "variable_query_supported"].iloc[0])
        )
        if not support.loc[support["method"].astype(str).eq(method)].empty
        else False
        for method in methods
    }


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _plot_scalability_axis(
    ax,
    table: pd.DataFrame,
    support: dict[str, bool],
    config: dict[str, Any],
    *,
    metric: str,
    low_metric: str | None,
    high_metric: str | None,
    native: int,
    xmax: int,
) -> None:
    methods = list(config["paper_contract"]["method_order"])
    colors, markers = method_colors(config), method_markers(config)
    for method in methods:
        group = table[table["method"].astype(str).eq(method)].sort_values("N")
        if group.empty:
            continue
        variable = support.get(method, False)
        ok = group[group["status"].astype(str).str.lower().eq("ok")]
        failures = group[~group["status"].astype(str).str.lower().eq("ok")]
        if variable and not ok.empty:
            x = ok["N"].to_numpy(dtype=float)
            y = ok[metric].to_numpy(dtype=float)
            ax.plot(x, y, color=colors[method], marker=markers[method], markersize=3.8, linewidth=1.0, alpha=method_alpha(method), label=method, zorder=3)
            if low_metric and high_metric:
                lo = ok[low_metric].to_numpy(dtype=float)
                hi = ok[high_metric].to_numpy(dtype=float)
                ax.fill_between(x, lo, hi, color=colors[method], alpha=0.10, linewidth=0, zorder=1)
        elif not variable:
            row = group[group["N"].astype(int).eq(native)]
            if not row.empty and str(row.iloc[0]["status"]).lower() == "ok":
                ax.plot(
                    native,
                    float(row.iloc[0][metric]),
                    marker=markers[method],
                    markersize=5.0,
                    markerfacecolor="white",
                    markeredgecolor=colors[method],
                    markeredgewidth=0.95,
                    linestyle="none",
                    zorder=3,
                )
        # Failure points are evidence, not silently dropped.  Draw a small x at
        # the first failed query count when a numeric metric survived.
        for failure in failures.itertuples():
            failure_value = getattr(failure, metric, np.nan)
            if np.isfinite(failure_value) and float(failure_value) > 0:
                ax.plot(float(failure.N), float(failure_value), marker="x", color=colors[method], markersize=4.5, markeredgewidth=0.8, linestyle="none", zorder=4)
    ax.set_xscale("log")
    positive_n = table.loc[table["N"].astype(float) > 0, "N"].astype(float)
    xmin = float(positive_n.min()) if not positive_n.empty else float(native)
    ax.set_xlim(xmin * 0.82, xmax * 1.12)
    ax.axvspan(native, xmax * 1.12, color="#F1F3F5", alpha=0.82, zorder=-2)
    ax.axvline(native, color="#6C757D", linestyle=(0, (3, 2)), linewidth=0.72, zorder=1)
    scale_cfg = config["formal_protocol"]["scale_stress"]
    ticks = [int(x) for x in scale_cfg["native_query_counts"]]
    ticks.extend(int(x) for x in scale_cfg["throughput_query_counts"])
    adaptive_cap = int(scale_cfg.get("adaptive_query_cap", ticks[-1]))
    if adaptive_cap <= xmax:
        ticks.append(adaptive_cap)
    ticks = list(dict.fromkeys(value for value in ticks if value <= xmax))

    def compact_count(value: int) -> str:
        if value == 40300:
            return "40.3k"
        if value >= 1_000_000:
            millions = value / 1_000_000
            return f"{millions:g}M"
        if value >= 1000:
            return f"{value / 1000:g}k"
        return str(value)

    ax.set_xticks(ticks)
    ax.set_xticklabels([compact_count(value) for value in ticks])
    style_grid(ax, axis="both")


def draw_scalability(
    ax_latency,
    ax_memory,
    data: dict[str, Any],
    config: dict[str, Any],
    *,
    panel_label: str = "e",
    show_legend: bool = False,
) -> None:
    latency, memory, support_table = data.get("scale_latency"), data.get("scale_memory"), data.get("query_support")
    if latency is None or memory is None or support_table is None:
        _pending(ax_latency, panel_label, "High-resolution scalability envelope", "V4 clean-GPU high-N latency and memory source required; V3 native-only scaling is not substituted.")
        _pending(ax_memory, "", "Peak allocated memory versus requested query count", "")
        return
    methods = list(config["paper_contract"]["method_order"])
    native = int(config["formal_protocol"]["scale_stress"]["native_limit"])
    declared_max = max(int(x) for x in config["formal_protocol"]["scale_stress"]["throughput_query_counts"])
    observed_max = max(
        int(max(table["N"].astype(int)))
        for table in (latency, memory)
        if table is not None and not table.empty
    )
    xmax = max(native, declared_max, observed_max)
    support = _support_map(support_table, methods)
    _plot_scalability_axis(ax_latency, latency, support, config, metric="median_latency_ms", low_metric="latency_q25_ms", high_metric="latency_q75_ms", native=native, xmax=xmax)
    _plot_scalability_axis(ax_memory, memory, support, config, metric="peak_allocated_mib", low_metric=None, high_metric=None, native=native, xmax=xmax)
    ax_latency.set_yscale("log")
    ax_memory.set_yscale("log")
    ax_latency.set_ylabel("Latency (ms)")
    ax_memory.set_ylabel("Peak memory (MiB)")
    # Separate the two vertical labels at the internal axis boundary so their
    # glyph boxes cannot collide at final 183-mm print size.
    ax_latency.yaxis.set_label_coords(-0.055, 0.63)
    ax_memory.yaxis.set_label_coords(-0.055, 0.37)
    ax_memory.set_xlabel("Requested query points, N")
    ax_latency.set_title("Warm latency versus requested query count", loc="left", fontweight="semibold", pad=2)
    ax_memory.set_title("Peak allocated memory versus requested query count", loc="left", fontweight="semibold", pad=2)
    ax_latency.tick_params(axis="x", labelbottom=False, length=0)
    ax_latency.text(0.72, 0.90, "throughput-only stress test", transform=ax_latency.transAxes, ha="center", va="top", fontsize=5.7, color="#6C757D")
    ax_memory.text(0.72, 0.90, "no accuracy claim above 40.3k", transform=ax_memory.transAxes, ha="center", va="top", fontsize=5.2, color="#6C757D")
    add_panel_label(ax_latency, panel_label, x=-0.055, y=1.14)
    if show_legend:
        handles = method_legend_handles(config)
        ax_latency.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, handlelength=1.5, columnspacing=0.9, borderaxespad=0.0)


def make_standalone(panel: str, data: dict[str, Any], config: dict[str, Any]):
    """Create one V4 standalone panel using the same renderer as the composite."""

    from .figure5_v4_style import MM

    if panel in {"a", "b", "c", "d"}:
        height_mm = 62 if panel in {"a", "b"} else 68
        fig, ax = plt.subplots(figsize=(89 * MM, height_mm * MM))
        fig.subplots_adjust(
            left=0.20 if panel in {"a", "b"} else 0.16,
            right=0.98,
            bottom=0.20 if panel in {"a", "b"} else 0.30,
            top=0.84,
        )
        if panel == "a":
            draw_crps(ax, data, config, panel_label=panel, show_methods=True)
        elif panel == "b":
            draw_spread_association(ax, data, config, panel_label=panel, show_methods=True)
        elif panel == "c":
            draw_accuracy_cost(
                ax,
                data.get("cost_native"),
                config,
                x_column="cost_value",
                x_low_column="cost_low",
                x_high_column="cost_high",
                xlabel="Warm model-core latency (ms)",
                title="Native 40,300-point accuracy–cost trade-off",
                panel_label=panel,
                ylabel="Mean unobserved-field relative L2",
                ylim=_error_limits(data.get("cost_native"), data.get("training_cost")),
                show_legend=False,
            )
        else:
            draw_accuracy_cost(
                ax,
                data.get("training_cost"),
                config,
                x_column="cost_value",
                x_low_column="cost_low",
                x_high_column="cost_high",
                xlabel=data.get("training_metric_label") or "Training compute",
                title="Training-compute accuracy–cost trade-off",
                panel_label=panel,
                ylabel="Mean unobserved-field relative L2",
                ylim=_error_limits(data.get("cost_native"), data.get("training_cost")),
                show_legend=False,
            )
        if panel in {"c", "d"}:
            fig.legend(
                handles=method_legend_handles(config),
                loc="lower center",
                bbox_to_anchor=(0.56, 0.015),
                ncol=4,
                fontsize=5.6,
                handlelength=1.1,
                columnspacing=0.65,
                handletextpad=0.3,
                borderaxespad=0.0,
            )
        return fig
    if panel != "e":
        raise ValueError(f"Unknown V4 panel {panel!r}")
    fig = plt.figure(figsize=(183 * MM, 82 * MM))
    grid = fig.add_gridspec(2, 1, left=0.09, right=0.985, bottom=0.15, top=0.88, hspace=0.12)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1], sharex=None)]
    draw_scalability(axes[0], axes[1], data, config, panel_label=panel, show_legend=True)
    axes[1].set_xlim(axes[0].get_xlim())
    return fig


def make_composed(data: dict[str, Any], config: dict[str, Any]):
    """Compose a/b, c/d, and full-width e using explicit nested GridSpecs."""

    from .figure5_v4_style import MM

    width = float(config["figure"]["width_mm"]) * MM
    height = float(config["figure"]["composed_height_mm"]) * MM
    layout = config["figure"]["layout"]
    margins = layout["margins"]
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=layout["height_ratios"],
        left=margins["left"],
        right=margins["right"],
        bottom=margins["bottom"],
        top=margins["top"],
        wspace=layout["wspace"],
        hspace=layout["hspace"],
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)
    draw_crps(ax_a, data, config, panel_label="a", show_methods=True)
    draw_spread_association(ax_b, data, config, panel_label="b", show_methods=False)

    shared_ylim = _error_limits(data.get("cost_native"), data.get("training_cost"))
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1], sharey=ax_c)
    draw_accuracy_cost(
        ax_c,
        data.get("cost_native"),
        config,
        x_column="cost_value",
        x_low_column="cost_low",
        x_high_column="cost_high",
        xlabel="Warm model-core latency (ms)",
        title="Native inference accuracy–cost",
        panel_label="c",
        ylabel="Mean unobserved-field relative L2",
        ylim=shared_ylim,
    )
    draw_accuracy_cost(
        ax_d,
        data.get("training_cost"),
        config,
        x_column="cost_value",
        x_low_column="cost_low",
        x_high_column="cost_high",
        xlabel=data.get("training_metric_label") or "Training compute",
        title="Training accuracy–cost",
        panel_label="d",
        ylabel=None,
        ylim=shared_ylim,
    )

    egrid = grid[2, :].subgridspec(2, 1, hspace=layout["e_hspace"])
    ax_e_latency = fig.add_subplot(egrid[0])
    ax_e_memory = fig.add_subplot(egrid[1])
    draw_scalability(ax_e_latency, ax_e_memory, data, config, panel_label="e", show_legend=False)
    # Keep both internal axes exactly aligned in x even when one source has a
    # failure row beyond the other's largest successful point.
    ax_e_memory.set_xlim(ax_e_latency.get_xlim())
    handles = method_legend_handles(config)
    # One shared legend for all computational panels, placed directly above the
    # full-width scalability anchor.  c/d/e do not repeat method legends.
    legend_y = float(ax_e_latency.get_position().y1) + 0.040
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.53, legend_y),
        ncol=8,
        fontsize=5.45,
        handlelength=1.05,
        columnspacing=0.55,
        handletextpad=0.25,
        borderaxespad=0.0,
    )
    return fig


DRAWERS = {"a": draw_crps, "b": draw_spread_association}

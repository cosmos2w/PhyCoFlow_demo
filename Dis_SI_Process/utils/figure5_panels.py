"""Five-panel Figure 5 V3 renderers."""
from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

from .figure5_style import AXIS, GRID, MM, add_panel_label, style_grid


MARKERS = ("o", "s", "D", "^", "v", "P", "X", "h")


def _colors(config: dict[str, Any]) -> dict[str, str]:
    return config["style"]["method_colors"]


def _pending(ax, label: str, title: str, requirement: str) -> None:
    ax.set_facecolor("#F3F5F6")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#ADB5BD")
        spine.set_linestyle((0, (3, 2)))
        spine.set_linewidth(0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    ax.text(0.5, 0.56, "Formal V3 evidence pending", transform=ax.transAxes, ha="center", va="center", fontsize=6.2, fontweight="bold", color="#5C6770")
    ax.text(0.5, 0.39, requirement, transform=ax.transAxes, ha="center", va="center", fontsize=5.1, color="#5C6770", wrap=True)
    add_panel_label(ax, label)


def _forest(ax, table, config, *, estimate: str, low: str, high: str, xlabel: str, ref: float | None, label: str, title: str, show_methods: bool = True) -> None:
    methods = config["paper_contract"]["generative_method_order"]
    colors = _colors(config)
    y = np.arange(len(methods))[::-1]
    for index, method in enumerate(methods):
        row = table[table["method"].eq(method)].iloc[0]
        value, lo, hi = float(row[estimate]), float(row[low]), float(row[high])
        ax.errorbar(value, y[index], xerr=np.asarray([[value - lo], [hi - value]]), fmt=MARKERS[index], color=colors[method], markersize=5.0 if method == "DMF-Gen" else 4.1, capsize=2.0, elinewidth=1.25, zorder=3)
    if ref is not None:
        ax.axvline(ref, color="#8D99AE", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels(methods if show_methods else [""] * len(methods))
    if show_methods:
        for tick, method in zip(ax.get_yticklabels(), methods):
            tick.set_color(colors[method])
            if method == "DMF-Gen":
                tick.set_fontweight("bold")
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    style_grid(ax, axis="x")
    add_panel_label(ax, label)


def draw_crps(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "a", show_methods: bool = True) -> None:
    table = data.get("uq_crps")
    if table is None:
        _pending(ax, panel_label, "Probabilistic reconstruction", "Five methods · 200 states × 64 draws · normalized empirical CRPS")
        return
    _forest(ax, table, config, estimate="mean_normalized_crps", low="crps_ci_low", high="crps_ci_high", xlabel="Normalized CRPS (lower is better)", ref=None, label=panel_label, title="Probabilistic reconstruction", show_methods=show_methods)


def draw_spread_association(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "b", show_methods: bool = True) -> None:
    table = data.get("uq_spread")
    if table is None:
        _pending(ax, panel_label, "Spread–error association", "Five methods · state-level macro spread and macro ensemble-mean error")
        return
    _forest(ax, table, config, estimate="spearman_rho", low="spearman_ci_low", high="spearman_ci_high", xlabel="Spearman ρ", ref=0.0, label=panel_label, title="Spread associated with error", show_methods=show_methods)
    ax.set_xlim(min(-0.08, float(table["spearman_ci_low"].min()) - 0.04), max(0.75, float(table["spearman_ci_high"].max()) + 0.04))


def draw_accuracy_latency(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "c") -> None:
    table = data.get("cost_native")
    if table is None:
        _pending(ax, panel_label, "Native accuracy–latency", "Eight methods · clean GPU · warm model core · N=40,300")
        return
    methods = config["paper_contract"]["method_order"]
    colors = _colors(config)
    offsets = {
        "DMF-Gen": (4, 4, "left"), "FFM-FNO": (4, 3, "left"), "FFM-Perceiver": (-4, 3, "right"),
        "Latent FM": (4, 3, "left"), "SiT": (-4, 3, "right"), "MLP-RBF": (4, -7, "left"),
        "Geo-FNO": (4, 3, "left"), "Senseiver": (4, -7, "left"),
    }
    for index, method in enumerate(methods):
        row = table[table["method"].eq(method)].iloc[0]
        x, y = float(row["median_latency_ms"]), float(row["error"])
        xerr = np.asarray([[x - float(row["latency_q25_ms"])], [float(row["latency_q75_ms"]) - x]])
        yerr = np.asarray([[y - float(row["error_ci_low"])], [float(row["error_ci_high"]) - y]])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=MARKERS[index], color=colors[method], markersize=5.4 if method == "DMF-Gen" else 4.3, capsize=1.8, zorder=3)
        dx, dy, alignment = offsets[method]
        ax.annotate(method, (x, y), xytext=(dx, dy), textcoords="offset points", ha=alignment, fontsize=4.8, color=colors[method], fontweight="bold" if method == "DMF-Gen" else "normal")
    ax.set_xscale("log")
    ax.set_xlim(float(table["median_latency_ms"].min()) * 0.90, float(table["median_latency_ms"].max()) * 1.16)
    ax.set_ylim(float(table["error_ci_low"].min()) - 0.018, float(table["error_ci_high"].max()) + 0.018)
    ax.set_xlabel("Warm model-core latency (ms)")
    ax.set_ylabel("Mean unobserved-field relative L2")
    ax.set_title("Native 40,300-point trade-off", loc="left", fontweight="bold")
    style_grid(ax)
    add_panel_label(ax, panel_label)


def _scaling(ax, table, support, config, *, metric: str, ylabel: str, panel_label: str, title: str, show_legend: bool) -> None:
    if table is None or support is None:
        _pending(ax, panel_label, title, "Audited native query support · N=1,024/4,096/16,384/40,300")
        return
    methods = config["paper_contract"]["method_order"]
    colors = _colors(config)
    for index, method in enumerate(methods):
        group = table[table["method"].eq(method)].sort_values("N")
        variable = bool(support[support["method"].eq(method)]["variable_query_supported"].iloc[0])
        if variable:
            if metric == "median_latency_ms":
                y = group[metric].to_numpy(dtype=float)
                yerr = np.vstack([y - group["latency_q25_ms"].to_numpy(dtype=float), group["latency_q75_ms"].to_numpy(dtype=float) - y])
                ax.errorbar(group["N"], y, yerr=yerr, color=colors[method], marker=MARKERS[index], markersize=3.6, capsize=1.4, label=method)
            else:
                ax.plot(group["N"], group[metric], color=colors[method], marker=MARKERS[index], markersize=3.6, label=method)
        else:
            row = group.iloc[0]
            ax.plot(float(row["N"]), float(row[metric]), marker=MARKERS[index], markersize=5.0, markerfacecolor="white", markeredgecolor=colors[method], markeredgewidth=1.1, linestyle="none")
    ax.axvline(40300, color="#8D99AE", linestyle="--", linewidth=0.8, zorder=0)
    ax.text(40300, 0.98, "native", transform=ax.get_xaxis_transform(), ha="right", va="top", fontsize=4.7, color="#6C757D")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1024, 4096, 16384, 40300], labels=["1k", "4k", "16k", "40.3k"])
    ax.set_xlabel("Requested query points, N")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    style_grid(ax)
    add_panel_label(ax, panel_label)
    if show_legend:
        handles = [mlines.Line2D([], [], color=colors[method], marker=MARKERS[index], markersize=3.6, label=method) for index, method in enumerate(methods) if bool(support[support["method"].eq(method)]["variable_query_supported"].iloc[0])]
        handles.append(mlines.Line2D([], [], color="#606060", marker="o", markerfacecolor="white", linestyle="none", label="fixed grid: native only"))
        ax.legend(handles=handles, loc="upper left", fontsize=4.5, handlelength=1.3, ncol=1)


def draw_query_latency(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "d", show_legend: bool = True) -> None:
    _scaling(ax, data.get("cost_query"), data.get("query_support"), config, metric="median_latency_ms", ylabel="Warm latency (ms)", panel_label=panel_label, title="Query-count latency", show_legend=show_legend)


def draw_query_memory(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "e", show_legend: bool = False) -> None:
    _scaling(ax, data.get("cost_memory"), data.get("query_support"), config, metric="peak_allocated_mib", ylabel="Peak allocated memory (MiB)", panel_label=panel_label, title="Query-count memory", show_legend=show_legend)


DRAWERS = {"a": draw_crps, "b": draw_spread_association, "c": draw_accuracy_latency, "d": draw_query_latency, "e": draw_query_memory}


def make_standalone(panel: str, data: dict[str, Any], config: dict[str, Any]):
    width = 108 if panel == "c" else 88
    height = 66 if panel in "de" else 60
    fig, ax = plt.subplots(figsize=(width * MM, height * MM))
    fig.subplots_adjust(left=0.22 if panel in "ab" else 0.17, right=0.96, top=0.84, bottom=0.20)
    DRAWERS[panel](ax, data, config, panel_label=panel)
    return fig


def make_composed(data: dict[str, Any], config: dict[str, Any]):
    width = float(config["figure"]["width_mm"]) * MM
    height = float(config["figure"]["composed_height_mm"]) * MM
    fig = plt.figure(figsize=(width, height))
    grid = fig.add_gridspec(2, 12, left=0.092, right=0.985, bottom=0.09, top=0.90, wspace=1.45, hspace=0.82)
    ax_a = fig.add_subplot(grid[0, 0:6])
    ax_b = fig.add_subplot(grid[0, 6:12])
    draw_crps(ax_a, data, config, panel_label="a", show_methods=True)
    draw_spread_association(ax_b, data, config, panel_label="b", show_methods=True)
    ax_c = fig.add_subplot(grid[1, 0:6])
    ax_d = fig.add_subplot(grid[1, 6:9])
    ax_e = fig.add_subplot(grid[1, 9:12])
    draw_accuracy_latency(ax_c, data, config, panel_label="c")
    draw_query_latency(ax_d, data, config, panel_label="d", show_legend=False)
    draw_query_memory(ax_e, data, config, panel_label="e", show_legend=False)
    fig.text(0.092, 0.975, config["figure"]["row_headers"]["top"], ha="left", va="top", fontsize=7.0, fontweight="bold", color=AXIS)
    fig.text(0.092, 0.505, config["figure"]["row_headers"]["bottom"], ha="left", va="bottom", fontsize=7.0, fontweight="bold", color=AXIS)
    return fig

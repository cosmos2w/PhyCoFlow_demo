"""Six-panel Figure 5 V2 renderers."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure5_style import AXIS, DMF, MM, NEUTRAL, add_panel_label, add_status_badge, style_grid


MARKERS = ("o", "s", "^", "D", "v", "P", "X", "h")
FIELD_LABELS = {"Y_CH4": r"$Y_{\mathrm{CH_4}}$", "Y_CO": r"$Y_{\mathrm{CO}}$", "U1": r"$U_1$", "p": r"$p$"}


def _field_colors(config: dict[str, Any]) -> dict[str, str]:
    return dict(config["style"]["field_colors"])


def _method_colors(config: dict[str, Any]) -> dict[str, str]:
    return dict(config["style"]["method_colors"])


def _finish(ax, label: str, *, formal: bool = True) -> None:
    add_panel_label(ax, label)
    if not formal:
        add_status_badge(ax, "AWAITING FORMAL RUN", kind="pending", y=0.98, va="top")


def _pending(ax, label: str, title: str, requirement: str) -> None:
    ax.set_facecolor("#F1F3F5")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#ADB5BD")
        spine.set_linestyle((0, (3, 2)))
        spine.set_linewidth(0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontweight="bold", pad=4)
    ax.text(0.5, 0.56, "Formal evidence pending", transform=ax.transAxes, ha="center", va="center", fontsize=6.2, fontweight="bold", color="#5C6770")
    ax.text(0.5, 0.39, requirement, transform=ax.transAxes, ha="center", va="center", fontsize=5.1, color="#5C6770", wrap=True)
    _finish(ax, label, formal=False)


def draw_calibration(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "a") -> None:
    table = data.get("coverage")
    if table is None:
        _pending(ax, panel_label, "Calibration", "U2 · 200 states × 64 draws · M=256")
        return
    colors = _field_colors(config)
    ax.plot([0.48, 0.97], [0.48, 0.97], linestyle="--", color=NEUTRAL, linewidth=0.9, label="Ideal")
    for i, field in enumerate(config["paper_contract"]["unobserved_fields"]):
        group = table[table["field"].eq(field)].sort_values("nominal_level")
        y = group["empirical_coverage"].to_numpy()
        yerr = np.vstack([y - group["coverage_ci_low"].to_numpy(), group["coverage_ci_high"].to_numpy() - y])
        ax.errorbar(group["nominal_level"], y, yerr=yerr, marker=MARKERS[i], color=colors[field], capsize=1.8, label=FIELD_LABELS[field])
    ax.set(xlabel="Nominal central coverage", ylabel="Empirical state-level coverage", xlim=(0.48, 0.97), ylim=(0.0, 1.0))
    ax.set_xticks([0.5, 0.8, 0.9, 0.95])
    ax.set_title("Calibration", loc="left", fontweight="bold")
    style_grid(ax)
    ax.legend(ncol=2, loc="upper left", handlelength=1.3, columnspacing=0.9)
    _finish(ax, panel_label)


def draw_sharpness(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "b") -> None:
    table = data.get("coverage")
    if table is None:
        _pending(ax, panel_label, "Sharpness", "U2 · interval width / frozen training s.d.")
        return
    colors = _field_colors(config)
    for i, field in enumerate(config["paper_contract"]["unobserved_fields"]):
        group = table[table["field"].eq(field)].sort_values("nominal_level")
        y = group["mean_interval_width_normalized"].to_numpy()
        yerr = np.vstack([y - group["width_normalized_ci_low"].to_numpy(), group["width_normalized_ci_high"].to_numpy() - y])
        ax.errorbar(group["nominal_level"], y, yerr=yerr, marker=MARKERS[i], color=colors[field], capsize=1.8, label=FIELD_LABELS[field])
    ax.set(xlabel="Nominal central coverage", ylabel="Interval width / training s.d.")
    ax.set_xticks([0.5, 0.8, 0.9, 0.95])
    ax.set_title("Normalized interval width", loc="left", fontweight="bold")
    style_grid(ax)
    ax.legend(ncol=2, loc="upper left", handlelength=1.3, columnspacing=0.9)
    _finish(ax, panel_label)


def draw_spread_error(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "c") -> None:
    result = data.get("spread_error")
    if result is None:
        _pending(ax, panel_label, "Spread–error association", "U1 · 1,000 states × 16 draws · M=256")
        return
    table, colors = result["table"], _field_colors(config)
    for i, field in enumerate(config["paper_contract"]["unobserved_fields"]):
        group = table[table["field"].eq(field)].sort_values("bin")
        association = result["associations"][field]
        low, high = association.get("spearman_ci_low", np.nan), association.get("spearman_ci_high", np.nan)
        ci = f" [{low:.2f}, {high:.2f}]" if np.isfinite(low) and np.isfinite(high) else ""
        label = f"{FIELD_LABELS[field]} ρ={association['spearman_rho']:.2f}{ci}"
        x, y = group["spread"].to_numpy(), group["error"].to_numpy()
        ax.plot(x, y, marker=MARKERS[i], color=colors[field], label=label)
        ax.fill_between(x, group["error_q25"].to_numpy(), group["error_q75"].to_numpy(), color=colors[field], alpha=0.11, linewidth=0)
    ax.set(xlabel="Spatial RMS ensemble s.d. / training s.d.", ylabel="Ensemble-mean relative L2")
    ax.set_title("Spread associated with error", loc="left", fontweight="bold")
    style_grid(ax)
    ax.legend(loc="best", handlelength=1.2)
    _finish(ax, panel_label)


def draw_accuracy_latency(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "d") -> None:
    table = data.get("cost_native")
    if table is None:
        _pending(ax, panel_label, "Native-mesh accuracy–latency", "Eight canonical Cond_T methods · N=40,300")
        return
    colors = _method_colors(config)
    available = table[table["status"].eq("ok")].copy()
    for i, (_, row) in enumerate(available.iterrows()):
        method = str(row["method"])
        x, y = float(row["median_latency_ms"]), float(row["error"])
        xerr = np.asarray([[x - float(row["latency_q25_ms"])], [float(row["latency_q75_ms"]) - x]])
        yerr = np.asarray([[y - float(row["error_ci_low"])], [float(row["error_ci_high"]) - y]])
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=MARKERS[i], color=colors[method], markersize=5.2 if method == "DMF-Gen" else 4.2, capsize=1.8, zorder=3)
        offsets = (4, 2) if method != "Senseiver" else (4, -7)
        ax.annotate(method, (x, y), xytext=offsets, textcoords="offset points", fontsize=4.8, color=colors[method])
    ax.set_xscale("log")
    ax.set(xlabel="Median warm latency (ms)", ylabel="Mean relative L2")
    ax.set_title("Native 40,300-point trade-off", loc="left", fontweight="bold")
    style_grid(ax)
    _finish(ax, panel_label)


def draw_query_latency(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str | None = "e") -> None:
    table = data.get("cost_query")
    if table is None:
        _pending(ax, panel_label or "e", "DMF query scaling", "Real-coordinate N sweep · M=256")
        return
    x, y = table["N"].to_numpy(), table["median_latency_ms"].to_numpy()
    yerr = np.vstack([y - table["latency_q25_ms"].to_numpy(), table["latency_q75_ms"].to_numpy() - y])
    ax.errorbar(x, y, yerr=yerr, marker="o", color=DMF, capsize=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(40300, color=NEUTRAL, linestyle="--", linewidth=0.8)
    ax.set(ylabel="Warm latency (ms)")
    ax.set_title("DMF query scaling", loc="left", fontweight="bold", pad=2)
    style_grid(ax)
    if panel_label:
        _finish(ax, panel_label)


def draw_query_memory(ax, data: dict[str, Any], config: dict[str, Any]) -> None:
    table = data.get("cost_query")
    if table is None:
        ax.axis("off")
        return
    ax.plot(table["N"], table["peak_allocated_mib"], marker="s", color=config["style"]["field_colors"]["Y_CO"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(40300, color=NEUTRAL, linestyle="--", linewidth=0.8)
    ax.set(xlabel="Query points, N", ylabel="Peak allocated (MiB)")
    style_grid(ax)


def draw_nfe_tradeoff(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "f") -> None:
    table = data.get("cost_nfe")
    if table is None:
        _pending(ax, panel_label, "Few-step accuracy–cost", "Fixed 50-state cohort · measured NFE")
        return
    table = table.sort_values("measured_nfe")
    x, y = table["median_latency_ms"].to_numpy(), table["unobserved_mean_error"].to_numpy()
    xerr = np.vstack([x - table["latency_q25_ms"].to_numpy(), table["latency_q75_ms"].to_numpy() - x])
    yerr = np.vstack([y - table["error_ci_low"].to_numpy(), table["error_ci_high"].to_numpy() - y])
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, color=DMF, marker="o", capsize=1.8)
    for x_i, y_i, nfe in zip(x, y, table["measured_nfe"].astype(int)):
        ax.annotate(f"NFE {nfe}", (x_i, y_i), xytext=(3, 3), textcoords="offset points", fontsize=4.8)
    ax.set(xlabel="Median warm latency (ms)", ylabel="Mean relative L2")
    ax.set_title("Few-step accuracy–cost path", loc="left", fontweight="bold")
    style_grid(ax)
    _finish(ax, panel_label)


DRAWERS = {"a": draw_calibration, "b": draw_sharpness, "c": draw_spread_error, "d": draw_accuracy_latency, "f": draw_nfe_tradeoff}


def make_standalone(panel: str, data: dict[str, Any], config: dict[str, Any]):
    if panel == "e":
        if data.get("cost_query") is None:
            fig, ax = plt.subplots(figsize=(82 * MM, 76 * MM))
            fig.subplots_adjust(left=0.16, right=0.96, top=0.88, bottom=0.14)
            _pending(ax, "e", "DMF query and memory scaling", "Real-coordinate N sweep · M=256")
            return fig
        fig, axes = plt.subplots(2, 1, figsize=(82 * MM, 76 * MM), sharex=True, gridspec_kw={"hspace": 0.16})
        fig.subplots_adjust(left=0.20, right=0.96, top=0.88, bottom=0.16)
        draw_query_latency(axes[0], data, config, panel_label="e")
        axes[0].tick_params(labelbottom=False)
        draw_query_memory(axes[1], data, config)
        return fig
    width = 105 if panel == "d" else 86
    fig, ax = plt.subplots(figsize=(width * MM, 62 * MM))
    fig.subplots_adjust(left=0.18, right=0.96, top=0.84, bottom=0.20)
    DRAWERS[panel](ax, data, config, panel_label=panel)
    return fig


def make_composed(data: dict[str, Any], config: dict[str, Any]):
    width, height = float(config["figure"]["width_mm"]) * MM, float(config["figure"]["composed_height_mm"]) * MM
    fig = plt.figure(figsize=(width, height))
    outer = fig.add_gridspec(2, 12, left=0.075, right=0.985, bottom=0.075, top=0.89, wspace=1.15, hspace=0.68)
    top_axes = [fig.add_subplot(outer[0, 0:4]), fig.add_subplot(outer[0, 4:8]), fig.add_subplot(outer[0, 8:12])]
    draw_calibration(top_axes[0], data, config, panel_label="a")
    draw_sharpness(top_axes[1], data, config, panel_label="b")
    draw_spread_error(top_axes[2], data, config, panel_label="c")
    d_ax = fig.add_subplot(outer[1, 0:6])
    draw_accuracy_latency(d_ax, data, config, panel_label="d")
    if data.get("cost_query") is None:
        e_ax = fig.add_subplot(outer[1, 6:9])
        _pending(e_ax, "e", "DMF query and memory scaling", "Real-coordinate N sweep · M=256")
    else:
        e_grid = outer[1, 6:9].subgridspec(2, 1, hspace=0.16)
        e_axes = [fig.add_subplot(e_grid[i, 0]) for i in range(2)]
        draw_query_latency(e_axes[0], data, config, panel_label="e")
        e_axes[0].tick_params(labelbottom=False)
        draw_query_memory(e_axes[1], data, config)
    f_ax = fig.add_subplot(outer[1, 9:12])
    f_position = f_ax.get_position()
    f_ax.set_position([f_position.x0 + 0.008, f_position.y0, f_position.width - 0.008, f_position.height])
    draw_nfe_tradeoff(f_ax, data, config, panel_label="f")
    headers = config["figure"]["row_headers"]
    fig.text(0.075, 0.97, headers["top"], ha="left", va="top", fontsize=7.0, fontweight="bold", color=AXIS)
    fig.text(0.075, 0.505, headers["bottom"], ha="left", va="bottom", fontsize=7.0, fontweight="bold", color=AXIS)
    return fig

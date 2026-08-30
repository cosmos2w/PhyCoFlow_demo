"""Panel drawers and standalone/composed Figure 5 renderers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from .figure5_style import (
    AMBER,
    AXIS,
    BLUE,
    DMF,
    MM,
    NEUTRAL,
    TEAL,
    VIOLET,
    add_panel_label,
    add_status_badge,
    style_grid,
)

FIELD_FALLBACK = {"CO": BLUE, "T": DMF, "U_0": VIOLET, "U_1": TEAL, "p": AMBER}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]


def _mode_badge(mode: str) -> tuple[str, str]:
    if mode == "formal":
        return "FORMAL VALIDATION", "formal"
    if mode == "pending":
        return "AWAITING FORMAL RUN", "pending"
    return "DRAFT PROXY — NOT MANUSCRIPT EVIDENCE", "proxy"


def _field_colors(config: dict[str, Any]) -> dict[str, str]:
    return {**FIELD_FALLBACK, **config.get("style", {}).get("field_colors", {})}


def _method_colors(config: dict[str, Any]) -> dict[str, str]:
    return config.get("style", {}).get("method_colors", {})


def _draw_pending(ax, label: str, title: str, lines: list[str]) -> None:
    ax.set_facecolor("#F1F3F5")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#ADB5BD")
        spine.set_linestyle((0, (3, 2)))
        spine.set_linewidth(0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, loc="left", fontweight="bold", pad=5)
    add_panel_label(ax, label)
    ax.text(0.5, 0.64, "Formal validation output pending", transform=ax.transAxes, ha="center", va="center", fontsize=6.3, fontweight="bold", color="#5C6770")
    ax.text(0.5, 0.38, "\n".join(lines), transform=ax.transAxes, ha="center", va="center", fontsize=5.2, color="#5C6770", linespacing=1.45)
    add_status_badge(ax, "AWAITING FORMAL RUN", kind="pending")


def draw_uq_map(fig, axes, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "a") -> None:
    values = data["uq_map"]
    mode = data["modes"][panel_label]
    x, y = values["x"], values["y"]
    robust = float(config["statistics"].get("robust_quantile", 0.99))
    physical = np.concatenate([values["truth"][np.isfinite(values["truth"])], values["mean"][np.isfinite(values["mean"])]] )
    vmin, vmax = np.quantile(physical, [1 - robust, robust])
    error_max = max(float(np.nanquantile(values["error"], robust)), 1e-12)
    std_max = max(float(np.nanquantile(values["std"], robust)), 1e-12)
    titles = [
        "Truth",
        "Ensemble mean" if mode == "formal" else "NFE=2 reconstruction",
        "Absolute error",
        "Ensemble s.d." if mode == "formal" else "Solver sensitivity s.d.",
    ]
    arrays = [values["truth"], values["mean"], values["error"], values["std"]]
    cmaps = ["viridis", "viridis", "magma", "magma"]
    limits = [(vmin, vmax), (vmin, vmax), (0, error_max), (0, std_max)]
    for i, (ax, title, array, cmap, (lo, hi)) in enumerate(zip(axes, titles, arrays, cmaps, limits)):
        image = ax.pcolormesh(x, y, array, shading="nearest", cmap=cmap, vmin=lo, vmax=hi, rasterized=True)
        ax.set_title(title, pad=3)
        ax.set_aspect("auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.08, pad=0.06, aspect=20)
        cbar.ax.tick_params(labelsize=4.3, length=1.8, width=0.5)
        cbar.outline.set_linewidth(0.45)
        if i == 0:
            add_panel_label(ax, panel_label, x=-0.08, y=1.05)
            ax.text(0.0, -0.28, f"{values['field']} field", transform=ax.transAxes, ha="left", va="top", fontsize=5.2, fontweight="bold")
        if i == len(axes) - 1:
            badge, kind = _mode_badge(mode)
            add_status_badge(ax, badge, kind=kind, y=0.97, va="top")


def draw_coverage(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "b") -> None:
    table = data.get("coverage")
    if table is None or table.empty:
        _draw_pending(ax, panel_label, "Calibration", ["Requires U2: 200 states × 64 draws", "coverage_by_level.csv", "50/80/90/95% central intervals"])
        return
    colors = _field_colors(config)
    ax.plot([0.45, 1.0], [0.45, 1.0], linestyle="--", color=NEUTRAL, linewidth=0.9, label="Ideal")
    for i, (field, group) in enumerate(table.groupby("field", sort=False)):
        group = group.sort_values("nominal")
        ax.plot(group["nominal"], group["empirical"], marker=MARKERS[i % len(MARKERS)], color=colors.get(field, NEUTRAL), label=field)
    ax.set(xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(0.47, 0.98), ylim=(0.47, 0.98))
    ax.set_title("Calibration", loc="left", fontweight="bold")
    style_grid(ax)
    ax.legend(ncol=2, loc="lower right")
    add_panel_label(ax, panel_label)
    add_status_badge(ax, "FORMAL VALIDATION", kind="formal")


def draw_interval_width(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "c") -> None:
    table = data.get("coverage")
    if table is None or table.empty or not table["width"].notna().any():
        _draw_pending(ax, panel_label, "Sharpness", ["Requires U2 physical-unit widths", "state-level mean interval width", "interpreted jointly with coverage"])
        return
    colors = _field_colors(config)
    for i, (field, group) in enumerate(table.dropna(subset=["width"]).groupby("field", sort=False)):
        group = group.sort_values("nominal")
        ax.plot(group["nominal"], group["width"], marker=MARKERS[i % len(MARKERS)], color=colors.get(field, NEUTRAL), label=field)
    ax.set(xlabel="Nominal coverage", ylabel="Mean interval width (physical units)")
    ax.set_title("Sharpness", loc="left", fontweight="bold")
    style_grid(ax)
    ax.legend(ncol=2)
    add_panel_label(ax, panel_label)
    add_status_badge(ax, "FORMAL VALIDATION", kind="formal")


def draw_spread_error(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "d") -> None:
    result = data["spread_error"]
    table: pd.DataFrame = result["table"]
    colors = _field_colors(config)
    for i, (field, group) in enumerate(table.groupby("field", sort=False)):
        group = group.sort_values("bin")
        color = colors.get(field, NEUTRAL)
        ax.plot(group["spread"], group["error"], marker=MARKERS[i % len(MARKERS)], color=color, label=f"{field} ρ={result['rho'].get(field, np.nan):.2f}")
        ax.fill_between(group["spread"].to_numpy(), group["error_q25"].to_numpy(), group["error_q75"].to_numpy(), color=color, alpha=0.12, linewidth=0)
    finite = table[["spread", "error"]].to_numpy()
    if finite.size and np.all(finite > 0):
        ax.set_xscale("log")
        ax.set_yscale("log")
    mode = data["modes"][panel_label]
    ax.set_xlabel("Spatial RMS ensemble s.d." if mode == "formal" else "Cross-NFE sensitivity (normalized)")
    ax.set_ylabel("Ensemble-mean relative L2" if mode == "formal" else "NFE=2 absolute error (normalized)")
    ax.set_title("Spread–error association", loc="left", fontweight="bold")
    style_grid(ax)
    ax.legend(ncol=2, loc="best")
    add_panel_label(ax, panel_label)
    badge, kind = _mode_badge(mode)
    add_status_badge(ax, badge, kind=kind)


def draw_cost_native(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "e") -> None:
    table: pd.DataFrame = data["cost_native"]
    mode = data["modes"][panel_label]
    colors = _method_colors(config)
    xcol = "latency_ms" if "latency_ms" in table.columns else "latency_s"
    for i, row in table.reset_index(drop=True).iterrows():
        method = str(row["method"])
        color = colors.get(method, DMF if "S7" in method or "DMF" in method else NEUTRAL)
        ax.scatter(row[xcol], row["error"], color=color, marker=MARKERS[i % len(MARKERS)], s=28, edgecolor="white", linewidth=0.45, zorder=3)
        ax.annotate(method.replace("-e1000", ""), (row[xcol], row["error"]), xytext=(3, 2), textcoords="offset points", fontsize=4.8)
    ax.set_xscale("log")
    ax.set_xlabel("Median warm latency (ms)" if mode == "formal" else "1M-query NFE=4 time (s)")
    ax.set_ylabel("Unobserved-field relative L2" if mode == "formal" else "Reconstruction mean relative error")
    ax.set_title("Accuracy–latency trade-off", loc="left", fontweight="bold")
    style_grid(ax)
    add_panel_label(ax, panel_label)
    badge, kind = _mode_badge(mode)
    add_status_badge(ax, badge, kind=kind)


def draw_query_scaling(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "f") -> None:
    table: pd.DataFrame = data["cost_query"].sort_values("N")
    ax.plot(table["N"], table["latency_ms"], marker="o", color=DMF)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(40300, color=NEUTRAL, linestyle="--", linewidth=0.9)
    ax.text(40300, ax.get_ylim()[1], " native", fontsize=4.7, color=NEUTRAL, va="top", ha="left")
    ax.set(xlabel="Query points, N", ylabel="Warm latency (ms)")
    ax.set_title("DMF query scaling", loc="left", fontweight="bold")
    style_grid(ax)
    add_panel_label(ax, panel_label)
    badge, kind = _mode_badge(data["modes"][panel_label])
    add_status_badge(ax, badge, kind=kind)


def draw_memory_scaling(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "g") -> None:
    table: pd.DataFrame = data["cost_query"].sort_values("N")
    ax.plot(table["N"], table["memory_mib"], marker="s", color=BLUE)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(40300, color=NEUTRAL, linestyle="--", linewidth=0.9)
    ax.set(xlabel="Query points, N", ylabel="Peak allocated memory (MiB)")
    ax.set_title("DMF memory scaling", loc="left", fontweight="bold")
    style_grid(ax)
    add_panel_label(ax, panel_label)
    badge, kind = _mode_badge(data["modes"][panel_label])
    add_status_badge(ax, badge, kind=kind)


def draw_nfe_error(ax, data: dict[str, Any], config: dict[str, Any], *, panel_label: str = "h") -> None:
    table: pd.DataFrame = data["cost_nfe"]
    mode = data["modes"][panel_label]
    colors = _method_colors(config)
    for i, (method, group) in enumerate(table.groupby("method", sort=False)):
        group = group.sort_values("nfe")
        color = colors.get(str(method), DMF if "DMF" in str(method) or "S7" in str(method) else NEUTRAL)
        ax.plot(group["nfe"], group["error"], marker=MARKERS[i % len(MARKERS)], color=color, label=str(method).replace("-e1000", ""))
    ax.set(xlabel="Measured NFE", ylabel="Unobserved-field relative L2" if mode == "formal" else "Reconstruction mean relative error")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_title("Numerical effort", loc="left", fontweight="bold")
    style_grid(ax)
    if table["method"].nunique() > 1:
        ax.legend(ncol=min(3, table["method"].nunique()), loc="best")
    add_panel_label(ax, panel_label)
    badge, kind = _mode_badge(mode)
    add_status_badge(ax, badge, kind=kind)


DRAWERS = {
    "b": draw_coverage,
    "c": draw_interval_width,
    "d": draw_spread_error,
    "e": draw_cost_native,
    "f": draw_query_scaling,
    "g": draw_memory_scaling,
    "h": draw_nfe_error,
}


def make_standalone(panel: str, data: dict[str, Any], config: dict[str, Any]):
    width = float(config["figure"]["width_mm"]) * MM
    if panel == "a":
        fig, axes = plt.subplots(1, 4, figsize=(width, float(config["figure"]["qualitative_height_mm"]) * MM), gridspec_kw={"wspace": 0.13})
        fig.subplots_adjust(left=0.025, right=0.995, top=0.83, bottom=0.22)
        draw_uq_map(fig, axes, data, config, panel_label="a")
        return fig
    fig, ax = plt.subplots(figsize=(width * 0.50, float(config["figure"]["standalone_height_mm"]) * MM))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.83, bottom=0.22)
    DRAWERS[panel](ax, data, config, panel_label=panel)
    return fig


def make_composed(data: dict[str, Any], config: dict[str, Any]):
    width = float(config["figure"]["width_mm"]) * MM
    height = float(config["figure"]["composed_height_mm"]) * MM
    fig = plt.figure(figsize=(width, height))
    outer = fig.add_gridspec(4, 6, height_ratios=[1.15, 1.0, 1.0, 1.0], hspace=0.62, wspace=0.72, left=0.075, right=0.985, top=0.94, bottom=0.065)

    a_grid = outer[0, :].subgridspec(1, 4, wspace=0.14)
    a_axes = [fig.add_subplot(a_grid[0, i]) for i in range(4)]
    draw_uq_map(fig, a_axes, data, config, panel_label="a")

    for panel, slot in {
        "b": outer[1, 0:2],
        "c": outer[1, 2:4],
        "d": outer[1, 4:6],
        "e": outer[2, 0:3],
        "f": outer[2, 3:6],
        "g": outer[3, 0:3],
        "h": outer[3, 3:6],
    }.items():
        ax = fig.add_subplot(slot)
        DRAWERS[panel](ax, data, config, panel_label=panel)

    fig.text(0.075, 0.982, "Empirical conditional uncertainty", ha="left", va="top", fontsize=7.2, fontweight="bold", color=AXIS)
    fig.text(0.075, 0.493, "Computational cost and scaling", ha="left", va="bottom", fontsize=7.2, fontweight="bold", color=AXIS)
    return fig

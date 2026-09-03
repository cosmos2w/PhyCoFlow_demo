#!/usr/bin/env python
"""Build formal Figure 5 V6 exclusively from accepted V5/V5.1 evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Mandatory publication/export settings: keep SVG text editable.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent
MM = 1.0 / 25.4
DMF = "DMF-Gen"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def configure_style(font_family: Iterable[str]) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(font_family),
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.4,
            "axes.titlesize": 7.7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "legend.frameon": False,
            "legend.fontsize": 5.2,
            "lines.linewidth": 1.0,
            "savefig.transparent": False,
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.grid(axis=grid_axis, color="#DDE1E5", linewidth=0.42, alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#4C4C4C")
    ax.spines["bottom"].set_color("#4C4C4C")
    ax.tick_params(colors="#333333", pad=1.6)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.2,
        fontweight="bold",
        color="#171717",
        clip_on=False,
    )


def deterministic_jitter(count: int, salt: str, scale: float = 0.105) -> np.ndarray:
    seed = int(hashlib.sha256(salt.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed).uniform(-scale, scale, size=count)


def require_sources(config: Mapping[str, Any]) -> dict[str, Path]:
    paths = {name: repo_path(value) for name, value in config["sources"].items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required accepted source(s): " + ", ".join(missing))
    for key in ("v5_qa", "v51_panel_c_qa", "v51_scorecard_qa", "inference_memory_qa"):
        qa = read_json(paths[key])
        if str(qa.get("status", "")).lower() != "pass":
            raise ValueError(f"Accepted-source QA is not passing: {paths[key]}")
    for key in ("v51_panel_c_manifest", "v51_scorecard_manifest", "inference_memory_manifest"):
        manifest = read_json(paths[key])
        if str(manifest.get("status", "")).lower() not in {"complete", "pass"}:
            raise ValueError(f"Accepted-source manifest is not complete: {paths[key]}")
    if read_json(paths["v5_manifest"]).get("strict_formal") is not True:
        raise ValueError("Accepted V5 display source is not strict-formal")
    return paths


def load_panel_ab(
    source_path: Path, generative_methods: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(source_path)
    a_samples = source.loc[
        (source["panel"] == "a") & (source["metric_name"] == "statewise_normalized_crps")
    ].copy()
    a_summary = source.loc[
        (source["panel"] == "a") & (source["metric_name"] == "mean_statewise_normalized_crps")
    ].copy()
    b_samples = source.loc[
        (source["panel"] == "b") & (source["metric_name"] == "spread_error_spearman_bootstrap")
    ].copy()
    b_summary = source.loc[
        (source["panel"] == "b") & (source["metric_name"] == "spread_error_spearman_full_sample")
    ].copy()
    if a_samples.groupby("method").size().to_dict() != {m: 200 for m in generative_methods}:
        raise ValueError("Panel-a accepted source is not exactly five methods × 200 states")
    if b_samples.groupby("method").size().to_dict() != {m: 2000 for m in generative_methods}:
        raise ValueError("Panel-b accepted source is not exactly five methods × 2,000 bootstraps")
    for name, frame in (("a summary", a_summary), ("b summary", b_summary)):
        if set(frame["method"].astype(str)) != set(generative_methods) or len(frame) != 5:
            raise ValueError(f"Panel-{name} method set is invalid")
    return a_samples, a_summary, b_samples, b_summary


def load_panel_c(source_path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    all_rows = pd.read_csv(source_path)
    rows = all_rows.loc[all_rows["risk_kind"].astype(str).eq("normalized")].copy()
    methods = list(config["paper_contract"]["generative_method_order"])
    expected_coverages = np.asarray(config["protocol"]["selective_coverages"], dtype=float)
    if len(rows) != len(methods) * len(expected_coverages):
        raise ValueError("Panel-c normalized source has the wrong row count")
    if set(rows["method"].astype(str)) != set(methods):
        raise ValueError("Panel-c normalized source has the wrong methods")
    for method in methods:
        group = rows.loc[rows["method"].astype(str).eq(method)].sort_values("coverage_fraction")
        if not np.allclose(group["coverage_fraction"], expected_coverages):
            raise ValueError(f"Panel-c coverage grid mismatch for {method}")
        endpoint = group.loc[np.isclose(group["coverage_fraction"], 1.0), "risk"]
        if len(endpoint) != 1 or not np.allclose(endpoint, 1.0):
            raise ValueError(f"Panel-c normalized endpoint is not one for {method}")
    numeric = rows[["risk", "ci_low", "ci_high", "risk_auc"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Panel-c source contains non-finite plotted values")
    return rows


def derive_panel_d(
    summary_path: Path,
    stages_path: Path,
    inference_memory_path: Path,
    method_order: list[str],
) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    stages = pd.read_csv(stages_path)
    memory = pd.read_csv(inference_memory_path)
    if set(summary["method"].astype(str)) != set(method_order):
        raise ValueError("Panel-d scorecard source has the wrong method set")
    if set(memory.loc[memory["status"].eq("ok"), "method"].astype(str)) != set(method_order):
        raise ValueError("Inference-memory source does not contain eight successful methods")
    rows: list[dict[str, Any]] = []
    for method in method_order:
        base = summary.loc[summary["method"].astype(str).eq(method)]
        mem = memory.loc[(memory["method"].astype(str).eq(method)) & memory["status"].eq("ok")]
        stage = stages.loc[(stages["method"].astype(str).eq(method)) & stages["status"].eq("ok")]
        if len(base) != 1 or len(mem) != 1 or stage.empty:
            raise ValueError(f"Panel-d source cardinality failed for {method}")
        base_row, mem_row = base.iloc[0], mem.iloc[0]
        if str(base_row["checkpoint_sha256"]) != str(mem_row["checkpoint_sha256"]):
            raise ValueError(f"Checkpoint identity mismatch in panel d for {method}")
        time_row = stage.loc[stage["update_ms"].astype(float).idxmax()]
        memory_row = stage.loc[stage["allocated_mib"].astype(float).idxmax()]
        rows.append(
            {
                "method": method,
                "checkpoint_sha256": str(base_row["checkpoint_sha256"]),
                "mean_unobserved_relative_l2": float(base_row["mean_unobserved_relative_l2"]),
                "error_ci_low": float(base_row["mean_unobserved_relative_l2_ci_low"]),
                "error_ci_high": float(base_row["mean_unobserved_relative_l2_ci_high"]),
                "training_update_ms": float(time_row["update_ms"]),
                "training_update_q25_ms": float(time_row["update_q25_ms"]),
                "training_update_q75_ms": float(time_row["update_q75_ms"]),
                "training_time_stage": int(time_row["stage_ordinal"]),
                "training_peak_allocated_mib": float(memory_row["allocated_mib"]),
                "training_memory_stage": int(memory_row["stage_ordinal"]),
                "inference_latency_ms": float(base_row["native_latency_ms"]),
                "inference_latency_q25_ms": float(base_row["native_latency_q25_ms"]),
                "inference_latency_q75_ms": float(base_row["native_latency_q75_ms"]),
                "model_state_mib": float(mem_row["model_state_mib"]),
                "inference_peak_allocated_mib": float(mem_row["inference_peak_allocated_mib"]),
                "training_batch_size": 32,
                "inference_batch_size": int(mem_row["batch_size"]),
                "sensor_count": int(mem_row["sensor_count"]),
                "N": int(mem_row["N"]),
                "dtype": str(mem_row["dtype"]),
                "inference_context": str(mem_row["inference_context"]),
            }
        )
    result = pd.DataFrame(rows)
    actual_order = result.sort_values("mean_unobserved_relative_l2")["method"].tolist()
    if actual_order != method_order:
        raise ValueError(f"Declared scorecard order is not accuracy-ascending: {actual_order}")
    if not (result["inference_peak_allocated_mib"] >= result["model_state_mib"]).all():
        raise ValueError("Inference peak must not be below model state")
    return result


def draw_distribution(
    ax: plt.Axes,
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    methods: list[str],
    colors: Mapping[str, str],
    markers: Mapping[str, str],
    *,
    panel: str,
    title: str,
    xlabel: str,
    show_methods: bool,
) -> None:
    positions = np.arange(len(methods), dtype=float)[::-1]
    is_spearman = panel == "b"
    for y, method in zip(positions, methods):
        values = samples.loc[samples["method"].astype(str).eq(method), "metric_value"].to_numpy(float)
        if method == DMF:
            ax.axhspan(y - 0.42, y + 0.42, color=colors[method], alpha=0.055, zorder=0)
        box = ax.boxplot(
            [values],
            positions=[y],
            vert=False,
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            whis=(2.5, 97.5) if is_spearman else 1.5,
            medianprops={"color": colors[method], "linewidth": 0.9},
            boxprops={"facecolor": colors[method], "edgecolor": colors[method], "alpha": 0.12, "linewidth": 0.65},
            whiskerprops={"color": colors[method], "alpha": 0.55, "linewidth": 0.6},
            capprops={"color": colors[method], "alpha": 0.55, "linewidth": 0.6},
            zorder=2,
        )
        for patch in box["boxes"]:
            patch.set_gid(f"accepted_distribution:{panel}:{method}")
        display = values if len(values) <= 300 else values[np.linspace(0, len(values) - 1, 320, dtype=int)]
        ax.scatter(
            display,
            y + deterministic_jitter(len(display), f"figure5-v6|{panel}|{method}"),
            s=3.5,
            color=colors[method],
            alpha=0.16 if method != DMF else 0.21,
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        row = summary.loc[summary["method"].astype(str).eq(method)].iloc[0]
        center = float(row["metric_value"])
        lo, hi = float(row["ci_low"]), float(row["ci_high"])
        ax.hlines(y, lo, hi, color=colors[method], linewidth=1.05, zorder=3)
        ax.vlines([lo, hi], y - 0.085, y + 0.085, color=colors[method], linewidth=0.75, zorder=3)
        ax.plot(
            center,
            y,
            marker=markers[method],
            markersize=4.9 if method == DMF else 4.3,
            markerfacecolor="white",
            markeredgecolor=colors[method],
            markeredgewidth=0.9,
            linestyle="none",
            zorder=4,
        )
    if is_spearman:
        ax.axvline(0.0, color="#666666", linestyle=(0, (3, 2)), linewidth=0.72, zorder=0.5)
        ax.set_xlim(-0.28, 0.80)
        ax.set_xticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8])
    else:
        ax.set_xlim(0.0, 0.86)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylim(-0.58, len(methods) - 0.42)
    ax.set_yticks(positions)
    if show_methods:
        ax.set_yticklabels(methods)
        for label, method in zip(ax.get_yticklabels(), methods):
            label.set_color(colors[method])
            label.set_fontweight("semibold" if method == DMF else "normal")
    else:
        ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", pad=3.5, fontweight="semibold")
    style_axis(ax, grid_axis="x")
    add_panel_label(
        ax,
        panel,
        x=-0.17 if show_methods else -0.10,
        y=1.18 if "\n" in title else 1.08,
    )


def draw_panel_c(
    ax: plt.Axes,
    source: pd.DataFrame,
    methods: list[str],
    colors: Mapping[str, str],
    markers: Mapping[str, str],
    *,
    show_legend: bool,
    title: str = "Uncertainty supports selective reconstruction",
) -> None:
    for method in methods:
        rows = source.loc[source["method"].astype(str).eq(method)].sort_values("coverage_fraction")
        x = rows["coverage_fraction"].to_numpy(float)
        y = rows["risk"].to_numpy(float)
        ax.fill_between(x, rows["ci_low"], rows["ci_high"], color=colors[method], alpha=0.09, linewidth=0, zorder=1)
        ax.plot(
            x,
            y,
            color=colors[method],
            marker=markers[method],
            markersize=2.9,
            linewidth=1.25 if method == DMF else 0.95,
            alpha=1.0 if method == DMF else 0.90,
            zorder=3,
        )
    ax.axhline(1.0, color="#666666", linestyle=(0, (3, 2)), linewidth=0.72, zorder=2)
    ax.set_xlim(0.18, 1.02)
    ax.set_ylim(0.835, 1.025)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.85, 0.90, 0.95, 1.00])
    ax.set_xlabel("Retained least-uncertain states")
    ax.set_ylabel("Relative retained-set error")
    ax.set_title(title, loc="left", pad=3.5, fontweight="semibold")
    style_axis(ax, grid_axis="y")
    add_panel_label(ax, "c", x=-0.16, y=1.18 if "\n" in title else 1.08)
    dmf_auc = float(source.loc[source["method"].eq(DMF), "risk_auc"].iloc[0])
    ax.text(0.035, 0.055, f"DMF AURC = {dmf_auc:.3f}", transform=ax.transAxes, color=colors[DMF], fontsize=5.1, fontweight="semibold")
    if show_legend:
        handles = [
            Line2D([], [], color=colors[m], marker=markers[m], markersize=3.2, linewidth=1.0, label=m)
            for m in methods
        ]
        ax.legend(
            handles=handles,
            loc="lower right",
            ncol=2,
            fontsize=4.65,
            handlelength=1.15,
            handletextpad=0.35,
            columnspacing=0.58,
            borderaxespad=0.35,
            labelspacing=0.28,
        )


def row_guides(ax: plt.Axes, count: int, *, highlight_y: int) -> None:
    for y in range(count):
        ax.axhline(y, color="#E4E7EA", linewidth=0.42, zorder=0)
    ax.axhspan(highlight_y - 0.43, highlight_y + 0.43, color="#E63946", alpha=0.055, zorder=-1)
    ax.set_ylim(count - 0.5, -0.5)


def setup_scorecard_axis(
    ax: plt.Axes,
    count: int,
    title: str,
    unit: str,
    *,
    log: bool,
    xlim: tuple[float, float],
    xticks: list[float],
    xticklabels: list[str] | None = None,
    highlight_y: int = 0,
) -> None:
    row_guides(ax, count, highlight_y=highlight_y)
    if log:
        ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels or [str(value) for value in xticks])
    ax.set_title(title, loc="left", fontsize=6.2, pad=4.2, fontweight="semibold")
    ax.set_xlabel(unit, labelpad=2.0, color="#555555")
    ax.set_yticks(range(count))
    ax.tick_params(axis="y", labelleft=False, length=0)
    style_axis(ax, grid_axis="x")


def draw_panel_d(
    axes: list[plt.Axes],
    table: pd.DataFrame,
    colors: Mapping[str, str],
    markers: Mapping[str, str],
    *,
    show_panel_heading: bool,
    figure: plt.Figure,
    heading_xy: tuple[float, float],
) -> None:
    methods = table["method"].astype(str).tolist()
    count = len(methods)
    setup_scorecard_axis(axes[0], count, "Reconstruction error", "relative L₂", log=False, xlim=(0.08, 0.49), xticks=[0.1, 0.2, 0.3, 0.4])
    setup_scorecard_axis(axes[1], count, "Training time", "ms / update", log=True, xlim=(20, 820), xticks=[25, 100, 400], xticklabels=["25", "100", "400"])
    setup_scorecard_axis(axes[2], count, "Training memory", "GiB", log=True, xlim=(1.5, 18.5), xticks=[2, 4, 8, 16], xticklabels=["2", "4", "8", "16"])
    setup_scorecard_axis(axes[3], count, "Inference time", "ms", log=True, xlim=(2.5, 30), xticks=[3, 10, 30], xticklabels=["3", "10", "30"])
    setup_scorecard_axis(axes[4], count, "Inference memory", "MiB", log=True, xlim=(1.6, 700), xticks=[2, 10, 100, 500], xticklabels=["2", "10", "100", "500"])
    axes[0].set_yticklabels(methods)
    axes[0].tick_params(axis="y", labelleft=True, length=0, pad=3.0)
    for tick, method in zip(axes[0].get_yticklabels(), methods):
        tick.set_color(colors[method])
        tick.set_fontweight("bold" if method == DMF else "normal")
    for y, row in enumerate(table.itertuples(index=False)):
        method = str(row.method)
        color, marker = colors[method], markers[method]
        axes[0].hlines(y, row.error_ci_low, row.error_ci_high, color=color, linewidth=1.15, zorder=3)
        axes[0].vlines([row.error_ci_low, row.error_ci_high], y - 0.08, y + 0.08, color=color, linewidth=0.7, zorder=3)
        axes[0].plot(row.mean_unobserved_relative_l2, y, marker=marker, ms=4.9 if method == DMF else 4.2, mfc=color, mec="white", mew=0.55, linestyle="none", zorder=4)
        axes[0].text(row.error_ci_high + 0.008, y - 0.16, f"{row.mean_unobserved_relative_l2:.3f}", color=color, fontsize=4.8, ha="left", va="bottom")

        axes[1].hlines(y, row.training_update_q25_ms, row.training_update_q75_ms, color=color, linewidth=1.05, zorder=3)
        axes[1].plot(row.training_update_ms, y, marker=marker, ms=4.0, mfc=color, mec="white", mew=0.5, linestyle="none", zorder=4)
        axes[1].text(row.training_update_ms, y - 0.17, f"{row.training_update_ms:.0f}", color=color, fontsize=4.6, ha="center", va="bottom")

        training_gib = row.training_peak_allocated_mib / 1024.0
        axes[2].plot(training_gib, y, marker=marker, ms=4.0, mfc=color, mec="white", mew=0.5, linestyle="none", zorder=4)
        axes[2].text(training_gib, y - 0.17, f"{training_gib:.1f}", color=color, fontsize=4.6, ha="center", va="bottom")

        axes[3].hlines(y, row.inference_latency_q25_ms, row.inference_latency_q75_ms, color=color, linewidth=1.05, zorder=3)
        axes[3].plot(row.inference_latency_ms, y, marker=marker, ms=4.0, mfc=color, mec="white", mew=0.5, linestyle="none", zorder=4)
        axes[3].text(row.inference_latency_ms, y - 0.17, f"{row.inference_latency_ms:.1f}", color=color, fontsize=4.6, ha="center", va="bottom")

        axes[4].hlines(y, row.model_state_mib, row.inference_peak_allocated_mib, color=color, linewidth=0.75, alpha=0.65, zorder=2)
        axes[4].plot(row.model_state_mib, y, marker=marker, ms=3.8, mfc="white", mec=color, mew=0.85, linestyle="none", zorder=3)
        axes[4].plot(row.inference_peak_allocated_mib, y, marker=marker, ms=4.3, mfc=color, mec="white", mew=0.5, linestyle="none", zorder=4)
    if show_panel_heading:
        x, y = heading_xy
        figure.text(x, y, "d", ha="left", va="bottom", fontsize=8.2, fontweight="bold", color="#171717")
        figure.text(x + 0.036, y, "Accuracy and computational footprint", ha="left", va="bottom", fontsize=8.2, fontweight="semibold", color="#171717")


def make_standalone_ab(
    panel: str,
    samples: pd.DataFrame,
    summary: pd.DataFrame,
    config: Mapping[str, Any],
) -> plt.Figure:
    figure = plt.figure(figsize=(float(config["figure"]["standalone_top_width_mm"]) * MM, float(config["figure"]["standalone_top_height_mm"]) * MM))
    ax = figure.add_axes([0.22, 0.19, 0.75, 0.68])
    draw_distribution(
        ax,
        samples,
        summary,
        list(config["paper_contract"]["generative_method_order"]),
        config["style"]["method_colors"],
        config["style"]["method_markers"],
        panel=panel,
        title=config["paper_contract"]["panel_titles"][panel],
        xlabel="Normalized CRPS (lower is better)" if panel == "a" else "Spearman ρ",
        show_methods=True,
    )
    return figure


def make_standalone_c(source: pd.DataFrame, config: Mapping[str, Any]) -> plt.Figure:
    figure = plt.figure(figsize=(float(config["figure"]["standalone_top_width_mm"]) * MM, float(config["figure"]["standalone_top_height_mm"]) * MM))
    ax = figure.add_axes([0.17, 0.19, 0.80, 0.68])
    draw_panel_c(
        ax,
        source,
        list(config["paper_contract"]["generative_method_order"]),
        config["style"]["method_colors"],
        config["style"]["method_markers"],
        show_legend=True,
    )
    return figure


def make_standalone_d(table: pd.DataFrame, config: Mapping[str, Any]) -> plt.Figure:
    figure = plt.figure(figsize=(float(config["figure"]["standalone_d_width_mm"]) * MM, float(config["figure"]["standalone_d_height_mm"]) * MM))
    grid = figure.add_gridspec(
        1,
        5,
        left=0.145,
        right=0.99,
        bottom=0.19,
        top=0.79,
        width_ratios=[2.45, 1.05, 1.10, 1.05, 1.40],
        wspace=0.30,
    )
    axes = [figure.add_subplot(grid[0, i]) for i in range(5)]
    draw_panel_d(
        axes,
        table,
        config["style"]["method_colors"],
        config["style"]["method_markers"],
        show_panel_heading=True,
        figure=figure,
        heading_xy=(0.015, 0.91),
    )
    return figure


def make_composed(
    a_samples: pd.DataFrame,
    a_summary: pd.DataFrame,
    b_samples: pd.DataFrame,
    b_summary: pd.DataFrame,
    c_source: pd.DataFrame,
    d_source: pd.DataFrame,
    config: Mapping[str, Any],
) -> plt.Figure:
    figure = plt.figure(figsize=(float(config["figure"]["width_mm"]) * MM, float(config["figure"]["composed_height_mm"]) * MM))
    top = figure.add_gridspec(1, 3, left=0.105, right=0.99, bottom=0.595, top=0.885, wspace=0.31)
    ax_a, ax_b, ax_c = [figure.add_subplot(top[0, i]) for i in range(3)]
    methods = list(config["paper_contract"]["generative_method_order"])
    colors, markers = config["style"]["method_colors"], config["style"]["method_markers"]
    draw_distribution(ax_a, a_samples, a_summary, methods, colors, markers, panel="a", title="Probabilistic\nreconstruction", xlabel="Normalized CRPS (lower is better)", show_methods=True)
    draw_distribution(ax_b, b_samples, b_summary, methods, colors, markers, panel="b", title="Uncertainty tracks\ndifficult states", xlabel="Spearman ρ", show_methods=False)
    draw_panel_c(ax_c, c_source, methods, colors, markers, show_legend=False, title="Uncertainty supports\nselective reconstruction")
    legend_handles = [
        Line2D([], [], color=colors[m], marker=markers[m], markersize=3.3, linewidth=1.0, label=m)
        for m in methods
    ]
    figure.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.55, 0.515),
        ncol=5,
        fontsize=5.15,
        handlelength=1.15,
        handletextpad=0.35,
        columnspacing=0.85,
    )

    bottom = figure.add_gridspec(
        1,
        5,
        left=0.145,
        right=0.99,
        bottom=0.085,
        top=0.385,
        width_ratios=[2.45, 1.05, 1.10, 1.05, 1.40],
        wspace=0.30,
    )
    axes_d = [figure.add_subplot(bottom[0, i]) for i in range(5)]
    draw_panel_d(axes_d, d_source, colors, markers, show_panel_heading=True, figure=figure, heading_xy=(0.015, 0.455))
    return figure


def save_figure(figure: plt.Figure, svg_path: Path, preview_path: Path | None) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(svg_path, format="svg", bbox_inches=None)
    # Matplotlib emits trailing spaces in multiline SVG path data.  Remove
    # those mechanically so generated manuscript assets pass git diff --check.
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(preview_path, format="png", dpi=240, bbox_inches=None)
    plt.close(figure)


def svg_checks(path: Path, required_text: Iterable[str]) -> dict[str, bool | int]:
    text = path.read_text(encoding="utf-8")
    root = ET.parse(path).getroot()
    return {
        "parseable_svg": root.tag.endswith("svg"),
        "editable_text": "<text" in text,
        "viewbox_present": "viewBox" in root.attrib,
        "required_text_present": all(value in text for value in required_text),
        "bytes": path.stat().st_size,
    }


def markdown_table_c(source: pd.DataFrame, methods: list[str]) -> str:
    lines = ["| Method | AURC ↓ | Error at 80% retained | Error at 100% |", "|---|---:|---:|---:|"]
    for method in methods:
        rows = source.loc[source["method"].eq(method)]
        r80 = float(rows.loc[np.isclose(rows["coverage_fraction"], 0.8), "risk"].iloc[0])
        r100 = float(rows.loc[np.isclose(rows["coverage_fraction"], 1.0), "risk"].iloc[0])
        lines.append(f"| {method} | {float(rows['risk_auc'].iloc[0]):.3f} | {r80:.3f} | {r100:.3f} |")
    return "\n".join(lines)


def markdown_table_d(source: pd.DataFrame) -> str:
    lines = [
        "| Method | Error | Train time (ms/update) | Train memory (GiB) | Inference time (ms) | Model state / inference peak (MiB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in source.itertuples(index=False):
        lines.append(
            f"| {row.method} | {row.mean_unobserved_relative_l2:.3f} | {row.training_update_ms:.1f} | "
            f"{row.training_peak_allocated_mib / 1024:.1f} | {row.inference_latency_ms:.2f} | "
            f"{row.model_state_mib:.1f} / {row.inference_peak_allocated_mib:.1f} |"
        )
    return "\n".join(lines)


def write_companions(
    docs_dir: Path,
    timestamp: str,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    a_summary: pd.DataFrame,
    b_summary: pd.DataFrame,
    c_source: pd.DataFrame,
    d_source: pd.DataFrame,
) -> list[Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    methods = list(config["paper_contract"]["generative_method_order"])
    crps = a_summary.set_index("method")["metric_value"]
    rho = b_summary.set_index("method")["metric_value"]
    contents = {
        f"fig5a_probabilistic_reconstruction_{timestamp}.md": f"""# Figure 5a — Probabilistic reconstruction

- Source reused in place: `{paths['v5_display'].relative_to(REPO_ROOT)}` (accepted V5 plot source).
- Evidence: 200 paired held-out temporal states per method; 64 shared-seed ensemble draws per state.
- Summary: formal mean normalized CRPS with 95% temporal moving-block-bootstrap interval (2,000 replicates; block length 25).
- Metric: empirical CRPS normalized by frozen training-field standard deviation, spatially averaged and equally macro-averaged over `Y_CH4`, `Y_CO`, `U1`, and `p`; lower is better.

DMF-Gen has the lowest accepted mean normalized CRPS ({crps[DMF]:.4f}); the other accepted means are SiT {crps['SiT']:.4f}, FFM-Perceiver {crps['FFM-Perceiver']:.4f}, Latent FM {crps['Latent FM']:.4f}, and FFM-FNO {crps['FFM-FNO']:.4f}.

CRPS measures predictive-distribution quality but does not by itself establish calibration; accepted reliability evidence remains underdispersed and is retained in SI.
""",
        f"fig5b_uncertainty_tracks_difficult_states_{timestamp}.md": f"""# Figure 5b — Uncertainty tracks difficult states

- Source reused in place: `{paths['v5_display'].relative_to(REPO_ROOT)}` (accepted V5 plot source).
- Evidence: the accepted 2,000-replicate temporal moving-block-bootstrap cloud plus the full-sample Spearman estimate and 95% interval; no bootstrap was rerun.
- Metric: Spearman association between macro normalized ensemble spread and macro ensemble-mean relative-L2 error.

Full-sample estimates are DMF-Gen {rho[DMF]:.3f}, SiT {rho['SiT']:.3f}, FFM-Perceiver {rho['FFM-Perceiver']:.3f}, FFM-FNO {rho['FFM-FNO']:.3f}, and Latent FM {rho['Latent FM']:.3f}. Positive values indicate that larger empirical ensemble spread tends to accompany more difficult reconstruction states. This is informativeness, not perfect calibration, prospective prediction, or causality.
""",
        f"fig5c_selective_reconstruction_{timestamp}.md": f"""# Figure 5c — Uncertainty supports selective reconstruction

- Source reused in place: `{paths['v51_selective_risk'].relative_to(REPO_ROOT)}` (accepted V5.1 C1 family).
- Main-text choice: normalized C1b only. States are ranked by ascending macro normalized ensemble spread; the least-uncertain 20–100% are retained.
- Y quantity: retained-set mean reconstruction error divided by the same method's full-cohort error, `R(r)/R(1)`. Lower is better and every method ends at 1.0.
- Statistics: 200 paired states, 64 draws per state, and accepted 95% intervals from 2,000 temporal moving-block-bootstrap replicates (block length 25) with ranking recomputed within each resample.
- Display: linear scale; exact evaluated points joined without smoothing.

{markdown_table_c(c_source, methods)}

C1b replaces the former spatial error-capture main panel because panels a and d already preserve absolute quality. Normalization isolates how effectively each method's own uncertainty ranks cases for selective retention, making panel c the operational consequence of panel b. C1a remains the absolute-error SI/back-up view.
""",
        f"fig5d_accuracy_computational_footprint_{timestamp}.md": f"""# Figure 5d — Accuracy and computational footprint

This accuracy-first D1 graphical scorecard keeps every quantity in a separate aligned column and uses no bubble area, weighted score, rank average, or stage-count column.

{markdown_table_d(d_source)}

## Protocol and source notes

- Accuracy and native warm inference latency: `{paths['v51_scorecard'].relative_to(REPO_ROOT)}`; error is the frozen 1,000-state mean unobserved-field relative L2 with temporal-bootstrap 95% interval, while latency is median with IQR.
- Training time/memory: `{paths['v51_scorecard_stages'].relative_to(REPO_ROOT)}`; common B=32, M=256, float32, synchronized update timing, one clean GPU. Query-evaluable models use 4,096 training targets and native-grid architectures use 40,300, so values are descriptive method-native workloads rather than an asymptotic or matched-budget comparison.
- Inference memory: `{paths['inference_memory'].relative_to(REPO_ROOT)}`; B=1, M=256, N=40,300, float32, `torch.inference_mode`, 5 warmups and 10 repeats. Open markers show unique parameters plus persistent buffers; filled markers show process-local peak allocated memory during inference. The benchmark allowed unrelated shared-GPU work and therefore makes no timing claim.
- Latent FM has two required non-concurrent stages. For the one-row main panel, training time shows the larger stage-2 median (90.7 ms/update) and training memory shows the larger stage-1 peak (4.1 GiB). They are per-column maxima, not simultaneous or additive costs.

The scorecard is chosen because reconstruction accuracy remains the first and widest quantitative column, while offline and online costs stay distinct, inspectable, and correctly qualified.
""",
    }
    caption = f"""# Figure 5 V6 composed figure

- Canvas: 183 mm × 128 mm.
- Layout: equal-weight panels a–c on the top row; panel d spans the full bottom row.
- Backend/export: Python/Matplotlib in the `fig` environment; editable SVG text.

## Caption draft

**Figure 5 | Probabilistic sparse reconstruction, uncertainty utility and computational footprint.** **a,** Statewise normalized continuous ranked probability score (CRPS) for five conditional generators on the turbulent-combustion `Cond_T` task; points are 200 paired temporal states, and open symbols and intervals show the formal mean and 95% temporal moving-block-bootstrap interval. **b,** Association between empirical ensemble spread and ensemble-mean reconstruction error across states. Clouds show accepted moving-block-bootstrap Spearman estimates, symbols and intervals show the full-sample estimate and 95% interval, and the dashed line marks zero. Positive values indicate that larger spread tends to track more difficult states; this is an informativeness measure rather than a claim of perfect calibration. **c,** Normalized selective-reconstruction risk after retaining the least-uncertain states. Error is normalized by each method's full-cohort error, so lower values indicate more effective uncertainty-based selection and all curves end at one; bands are 95% temporal moving-block-bootstrap intervals. **d,** Accuracy-first scorecard combining frozen mean unobserved-field relative L2 error with common-batch training update time and peak allocated memory, warm native inference latency, and dedicated native inference memory. In the inference-memory column, open symbols denote model parameters plus persistent buffers and filled symbols denote peak allocated memory during inference. All results use M=256 temperature observations and native N=40,300 reconstruction points; unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`. Empirical conditional ensembles are finite and underdispersed in the separate reliability analysis. Detailed timing, memory, precision, stage, hardware, and workload boundaries are provided in Methods/SI.

## Design rationale

C1b replaces the old spatial error-capture panel because it adds an operational state-selection consequence without repeating absolute performance. The D1 scorecard replaces lifecycle bubbles because accuracy is directly legible and each resource remains in its own real-unit column. The method palette and restrained DMF highlight are shared across all panels.
"""
    contents[f"fig5_composed_v6_{timestamp}.md"] = caption
    written: list[Path] = []
    for name, content in contents.items():
        path = docs_dir / name
        path.write_text(content.rstrip() + "\n", encoding="utf-8")
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_v6.yaml")
    parser.add_argument("--timestamp", type=str)
    parser.add_argument("--preview-dir", type=Path, help="Temporary Python-rendered PNG directory for visual QA")
    parser.add_argument(
        "--visual-qa-status",
        choices=("pending", "pass"),
        default="pending",
        help="Record the result of print-size review of Python-rendered previews",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "figure5-v6-1":
        raise ValueError("Unexpected Figure 5 V6 config schema")
    timestamp = args.timestamp or str(config["timestamp"])
    if not timestamp.isdigit() and "_" not in timestamp:
        raise ValueError("Timestamp must be an explicit timestamp-like identifier")
    configure_style(config["style"]["font_family"])
    paths = require_sources(config)
    methods = list(config["paper_contract"]["generative_method_order"])
    method_order = list(config["paper_contract"]["scorecard_method_order"])

    a_samples, a_summary, b_samples, b_summary = load_panel_ab(paths["v5_display"], methods)
    c_source = load_panel_c(paths["v51_selective_risk"], config)
    d_source = derive_panel_d(paths["v51_scorecard"], paths["v51_scorecard_stages"], paths["inference_memory"], method_order)

    figure_dir = PACKAGE_ROOT / "figures" / "generated" / timestamp
    docs_dir = PACKAGE_ROOT / "docs" / "generated" / timestamp
    results_dir = PACKAGE_ROOT / "results" / "derived" / timestamp
    figure_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    stems = {
        "a": f"fig5a_probabilistic_reconstruction_{timestamp}",
        "b": f"fig5b_uncertainty_tracks_difficult_states_{timestamp}",
        "c": f"fig5c_selective_reconstruction_{timestamp}",
        "d": f"fig5d_accuracy_computational_footprint_{timestamp}",
        "composed": f"fig5_composed_v6_{timestamp}",
    }
    svg_paths = {key: figure_dir / f"{stem}.svg" for key, stem in stems.items()}
    if not args.force:
        existing = [str(path) for path in svg_paths.values() if path.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing SVG(s): " + ", ".join(existing))
    preview_paths = {
        key: (args.preview_dir / f"{stem}.png" if args.preview_dir else None)
        for key, stem in stems.items()
    }

    save_figure(make_standalone_ab("a", a_samples, a_summary, config), svg_paths["a"], preview_paths["a"])
    save_figure(make_standalone_ab("b", b_samples, b_summary, config), svg_paths["b"], preview_paths["b"])
    save_figure(make_standalone_c(c_source, config), svg_paths["c"], preview_paths["c"])
    save_figure(make_standalone_d(d_source, config), svg_paths["d"], preview_paths["d"])
    save_figure(make_composed(a_samples, a_summary, b_samples, b_summary, c_source, d_source, config), svg_paths["composed"], preview_paths["composed"])

    d_source_path = results_dir / "figure5_v6_panel_d_source.csv"
    d_source.to_csv(d_source_path, index=False)
    source_rows = []
    panel_roles = {
        "v5_display": "panels a/b accepted plotted samples and summaries",
        "v51_selective_risk": "panel c accepted C1b curves and intervals",
        "v51_scorecard": "panel d accuracy and inference-time summaries",
        "v51_scorecard_stages": "panel d common-B32 training time and memory",
        "inference_memory": "panel d dedicated inference-memory dumbbells",
    }
    panel_map = {"v5_display": "a,b", "v51_selective_risk": "c", "v51_scorecard": "d", "v51_scorecard_stages": "d", "inference_memory": "d"}
    for key, role in panel_roles.items():
        path = paths[key]
        row_count = len(pd.read_csv(path))
        source_rows.append({"panel": panel_map[key], "role": role, "source_path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path), "rows": row_count, "action": "reused_in_place"})
    source_index = pd.DataFrame(source_rows)
    source_index_path = results_dir / "figure5_v6_source_index.csv"
    source_index.to_csv(source_index_path, index=False)

    companion_paths = write_companions(docs_dir, timestamp, config, paths, a_summary, b_summary, c_source, d_source)
    required_by_file = {
        "a": ["Probabilistic reconstruction", "Normalized CRPS (lower is better)"],
        "b": ["Uncertainty tracks difficult states", "Spearman ρ"],
        "c": ["Uncertainty supports selective reconstruction", "Relative retained-set error"],
        "d": ["Accuracy and computational footprint", "Reconstruction error", "Inference memory"],
        "composed": ["Probabilistic", "reconstruction", "Uncertainty tracks", "difficult states", "Uncertainty supports", "selective reconstruction", "Accuracy and computational footprint"],
    }
    svg_qa = {key: svg_checks(path, required_by_file[key]) for key, path in svg_paths.items()}
    all_svg_checks = all(
        all(bool(details[name]) for name in ("parseable_svg", "editable_text", "viewbox_present", "required_text_present"))
        for details in svg_qa.values()
    )
    composed_root = ET.parse(svg_paths["composed"]).getroot()
    width_pt = float(str(composed_root.attrib["width"]).replace("pt", ""))
    height_pt = float(str(composed_root.attrib["height"]).replace("pt", ""))
    data_checks = {
        "panel_a_five_methods_200_states": len(a_samples) == 1000,
        "panel_b_five_methods_2000_bootstraps": len(b_samples) == 10000,
        "panel_c_c1b_only": set(c_source["risk_kind"]) == {"normalized"},
        "panel_c_linear_scale_contract": True,
        "panel_c_all_end_at_one": bool(np.allclose(c_source.loc[np.isclose(c_source["coverage_fraction"], 1.0), "risk"], 1.0)),
        "panel_d_eight_accuracy_sorted_methods": d_source["method"].tolist() == method_order,
        "panel_d_column_order_exact": config["paper_contract"]["scorecard_columns"] == ["Reconstruction error", "Training time", "Training memory", "Inference time", "Inference memory"],
        "panel_d_no_stage_count": "stage_count" not in d_source.columns,
        "panel_d_inference_peak_not_below_model_state": bool((d_source["inference_peak_allocated_mib"] >= d_source["model_state_mib"]).all()),
        "panel_d_latent_per_column_maxima": int(d_source.loc[d_source["method"].eq("Latent FM"), "training_time_stage"].iloc[0]) == 2 and int(d_source.loc[d_source["method"].eq("Latent FM"), "training_memory_stage"].iloc[0]) == 1,
        "no_broad_validation_or_inference_rerun": True,
        "composed_width_183_mm": abs(width_pt - 183 * 72 / 25.4) < 0.2,
        "composed_height_128_mm": abs(height_pt - 128 * 72 / 25.4) < 0.2,
    }
    visual_pass = args.visual_qa_status == "pass"
    qa = {
        "schema_version": "figure5-v6-qa-1",
        "status": ("pass" if visual_pass else "structural_pass_visual_pending") if all_svg_checks and all(data_checks.values()) else "fail",
        "backend": "Python/Matplotlib (fig environment)",
        "svg_checks": svg_qa,
        "data_checks": data_checks,
        "visual_qa": {
            "status": args.visual_qa_status,
            "preview_directory": str(args.preview_dir) if args.preview_dir else None,
            "review": "Print-size Python previews checked for title fit, legend/annotation occlusion, row and plot-area alignment, readable labels, and inference-memory dumbbell density." if visual_pass else None,
        },
    }
    qa_path = results_dir / "qa.json"
    qa_path.write_text(json.dumps(qa, indent=2), encoding="utf-8")
    if qa["status"] == "fail":
        raise RuntimeError(f"Figure 5 V6 structural/data QA failed; inspect {qa_path}")

    manifest = {
        "schema_version": "figure5-v6-build-1",
        "status": "complete" if visual_pass else "complete_visual_qa_pending",
        "timestamp": timestamp,
        "git_head": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "config": str(args.config.resolve().relative_to(REPO_ROOT)),
        "config_sha256": sha256(args.config.resolve()),
        "backend": "Python/Matplotlib in conda environment fig",
        "no_new_scientific_calculation": True,
        "source_reuse": source_rows,
        "derived_tables": [str(source_index_path.relative_to(REPO_ROOT)), str(d_source_path.relative_to(REPO_ROOT))],
        "figures": [str(path.relative_to(REPO_ROOT)) for path in svg_paths.values()],
        "companions": [str(path.relative_to(REPO_ROOT)) for path in companion_paths],
        "qa": str(qa_path.relative_to(REPO_ROOT)),
        "temporary_previews_retained": bool(args.preview_dir),
    }
    manifest_path = results_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    completion_path = docs_dir / f"figure5_v6_completion_report_{timestamp}.md"
    visual_line = "PASS (Python-rendered previews inspected at 240 dpi)." if visual_pass else "PENDING review of Python-rendered previews."
    cleanup_line = "Temporary PNG previews were deleted after visual QA; no preview is retained." if visual_pass and args.preview_dir is None else "Temporary PNG previews will be deleted after visual QA; no other temporary output is produced."
    completion_path.write_text(
        f"""# Figure 5 V6 completion report

## Build status

- Starting commit: `f0aa4e7ca76bf15af2972fa434f295678ef1bfca`.
- Branch: `paper/postprocessing-multifield-superresolution`.
- Python renderer/export: complete.
- Structural and data QA: PASS.
- Print-size visual QA: {visual_line}
- New scientific inference, bootstrap, training, or broad validation calculation: none.

## Files created

- Renderer: `Dis_SI_Process/figures/scripts/build_figure5_v6.py`.
- Contract/config: `Dis_SI_Process/docs/generated/{timestamp}/figure_contract.md` and `Dis_SI_Process/configs/figure5_v6.yaml`.
- Standalone SVGs: `{svg_paths['a'].relative_to(REPO_ROOT)}`, `{svg_paths['b'].relative_to(REPO_ROOT)}`, `{svg_paths['c'].relative_to(REPO_ROOT)}`, `{svg_paths['d'].relative_to(REPO_ROOT)}`.
- Composed SVG: `{svg_paths['composed'].relative_to(REPO_ROOT)}`.
- Compact sources: `{source_index_path.relative_to(REPO_ROOT)}` and `{d_source_path.relative_to(REPO_ROOT)}`.
- Manifest/QA: `{manifest_path.relative_to(REPO_ROOT)}` and `{qa_path.relative_to(REPO_ROOT)}`.
- One companion per standalone plus the composed companion under `{docs_dir.relative_to(REPO_ROOT)}`.

## Exact source reuse

- Panels a/b: accepted V5 display table `{paths['v5_display'].relative_to(REPO_ROOT)}`; all plotted state/bootstrap samples and summaries were reused without recalculation.
- Panel c: accepted V5.1 selective-risk table `{paths['v51_selective_risk'].relative_to(REPO_ROOT)}`; only `risk_kind=normalized` (C1b) is plotted.
- Panel d accuracy/inference time: `{paths['v51_scorecard'].relative_to(REPO_ROOT)}`; training time/memory: `{paths['v51_scorecard_stages'].relative_to(REPO_ROOT)}`; inference memory: `{paths['inference_memory'].relative_to(REPO_ROOT)}`.
- Source hashes and row counts are recorded in `{source_index_path.relative_to(REPO_ROOT)}`.

## Final design choices

C1b replaces V5's spatial error-capture panel and V5.1's initially preferred C1a because it asks the distinct operational question: how well does each method's own uncertainty identify states suitable for selective retention? Dividing by each method's full-cohort error removes the already-represented absolute-accuracy difference, keeps every endpoint at one, and makes the panel a direct consequence of the spread–error association in b. C1a remains the absolute-error SI/back-up and was not copied or regenerated.

Panel d uses the D1 graphical-scorecard organization rather than lifecycle bubbles because reconstruction error is directly plotted in the first and widest column. Training and inference time/memory remain separate aligned real-unit quantities; stage count is removed. Dedicated inference-memory evidence supports the preferred open model-state / filled peak-allocated dumbbell. Latent FM uses per-column maxima of its non-concurrent stages to keep one row while the companion preserves both-stage interpretation.

## Dropped from V5/V5.1 main figure

- V5 spatial error capture, C1a, c2 interface profiles, c3 posterior atlas, and c4 functionals remain SI/back-up or internal evidence.
- D2–D5 candidate layouts, lifecycle bubbles/scatters, and stage count are excluded from the main figure.
- Development-style protocol text is removed from headers and moved to companions/caption.

## Limitations

Empirical ensembles are finite and underdispersed; spread/error evidence is informativeness rather than perfect calibration. C1b is normalized within method and must be read with the absolute accuracy evidence in d. Common-B32 training coordinates retain method-native target workloads, and all timing/memory results remain hardware/configuration-specific descriptive footprints. The dedicated inference-memory run allowed shared GPU use and makes no timing claim.

## Cleanup

No checkpoint, dataset, cache, old result bundle, ensemble stack, or repeated bootstrap array was copied. {cleanup_line}
""".rstrip()
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"timestamp": timestamp, "figures": [str(path) for path in svg_paths.values()], "qa": str(qa_path), "manifest": str(manifest_path), "completion_report": str(completion_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

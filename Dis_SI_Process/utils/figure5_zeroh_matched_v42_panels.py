"""Metric-matched Zero-H-balanced Figure 5 backup renderers."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .figure5_v41_panels import _shared_log_error_limits, draw_accuracy_cost, draw_crps, draw_spearman
from .figure5_v41_style import MM, method_legend_handles


def make_standalone(panel: str, data: dict, config: dict):
    if panel in {"a", "b"}:
        fig, ax = plt.subplots(figsize=(89 * MM, 62 * MM))
        fig.subplots_adjust(left=0.24, right=0.98, bottom=0.19, top=0.84)
        (draw_crps if panel == "a" else draw_spearman)(ax, data, config, show_methods=True)
        return fig
    fig, ax = plt.subplots(figsize=(89 * MM, 72 * MM))
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.28, top=0.85)
    ylim = _shared_log_error_limits(data["cost_native"], data["training_cost"])
    if panel == "c":
        draw_accuracy_cost(ax, data["cost_native"], config, title="Zero-H native accuracy–latency", xlabel="Warm model-core latency (ms)", panel_label="c", ylabel="Mean density relative L2", ylim=ylim)
    elif panel == "d":
        draw_accuracy_cost(ax, data["training_cost"], config, title="Zero-H accuracy–training update time", xlabel=data["training_metric_label"], panel_label="d", ylabel="Mean density relative L2", ylim=ylim)
    else:
        raise ValueError(panel)
    fig.legend(handles=method_legend_handles(config), loc="lower center", bbox_to_anchor=(0.56, 0.012), ncol=2, fontsize=6.5, handlelength=1.1, columnspacing=0.9, handletextpad=0.3)
    return fig


def make_composed(data: dict, config: dict):
    fig = plt.figure(figsize=(float(config["figure"]["width_mm"]) * MM, float(config["figure"]["height_mm"]) * MM))
    layout = config["figure"]["layout"]
    grid = fig.add_gridspec(2, 2, left=layout["left"], right=layout["right"], bottom=layout["bottom"], top=layout["top"], wspace=layout["wspace"], hspace=layout["hspace"])
    ax_a, ax_b, ax_c, ax_d = (fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2))
    draw_crps(ax_a, data, config, show_methods=True)
    draw_spearman(ax_b, data, config, show_methods=False)
    ylim = _shared_log_error_limits(data["cost_native"], data["training_cost"])
    draw_accuracy_cost(ax_c, data["cost_native"], config, title="Zero-H native accuracy–latency", xlabel="Warm model-core latency (ms)", panel_label="c", ylabel="Mean density relative L2", ylim=ylim)
    draw_accuracy_cost(ax_d, data["training_cost"], config, title="Zero-H accuracy–training update time", xlabel=data["training_metric_label"], panel_label="d", ylabel=None, ylim=ylim)
    fig.legend(handles=method_legend_handles(config), loc="lower center", bbox_to_anchor=(0.55, 0.012), ncol=4, fontsize=6.7, handlelength=1.1, columnspacing=0.9, handletextpad=0.3)
    return fig

"""V4.2 render composition with canonical training update time restored in panel d."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .figure5_v41_panels import (
    _shared_log_error_limits,
    draw_accuracy_cost,
    draw_crps,
    draw_memory_scaling,
    draw_spearman,
)
from .figure5_v41_style import MM, method_legend_handles


def make_standalone(panel: str, data: dict, config: dict):
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
            draw_accuracy_cost(ax, data["training_cost"], config, title="Canonical training accuracy–update time", xlabel=data["training_metric_label"], panel_label="d", ylabel="Mean unobserved-field relative L2", ylim=ylim)
        fig.legend(handles=method_legend_handles(config), loc="lower center", bbox_to_anchor=(0.56, 0.012), ncol=4, fontsize=6.2, handlelength=1.1, columnspacing=0.75, handletextpad=0.3)
        return fig
    if panel == "e":
        fig, ax = plt.subplots(figsize=(183 * MM, 72 * MM))
        fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.78)
        draw_memory_scaling(ax, data, config, show_legend=True)
        return fig
    raise ValueError(panel)


def make_composed(data: dict, config: dict):
    layout, margins = config["figure"]["layout"], config["figure"]["layout"]["margins"]
    fig = plt.figure(figsize=(float(config["figure"]["width_mm"]) * MM, float(config["figure"]["composed_height_mm"]) * MM))
    outer = fig.add_gridspec(3, 1, height_ratios=layout["height_ratios"], left=margins["left"], right=margins["right"], bottom=margins["bottom"], top=margins["top"], hspace=layout["outer_hspace"])
    ab = outer[0].subgridspec(1, 2, wspace=layout["ab_wspace"])
    ax_a, ax_b = fig.add_subplot(ab[0, 0]), fig.add_subplot(ab[0, 1])
    draw_spearman(ax_b, data, config, show_methods=False)
    draw_crps(ax_a, data, config, show_methods=True)
    cd = outer[1].subgridspec(1, 2, wspace=layout["cd_wspace"])
    ax_c, ax_d = fig.add_subplot(cd[0, 0]), fig.add_subplot(cd[0, 1])
    ylim = _shared_log_error_limits(data["cost_native"], data["training_cost"])
    draw_accuracy_cost(ax_c, data["cost_native"], config, title="Native inference accuracy–latency", xlabel="Warm model-core latency (ms)", panel_label="c", ylabel="Mean unobserved-field relative L2", ylim=ylim)
    draw_accuracy_cost(ax_d, data["training_cost"], config, title="Training accuracy–update time", xlabel=data["training_metric_label"], panel_label="d", ylabel=None, ylim=ylim)
    draw_memory_scaling(fig.add_subplot(outer[2]), data, config, show_legend=True)
    return fig

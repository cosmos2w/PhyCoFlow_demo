#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "_CheckNotes/Stage7_smart_cq/evaluation_1000"
OUTPUT = ROOT / "figures/generated/stage7_final_pareto"

DISPLAY = {
    "F0-e1000": "F0",
    "CQ-LR-128-e1000": "CQ-LR-128",
    "CQ-LR-256-best-e840-partial": "CQ-LR-256†",
    "S7-B-e1000": "Stage7-All256",
}
COLORS = {
    "F0-e1000": "#4D4D4D",
    "CQ-LR-128-e1000": "#4C78A8",
    "CQ-LR-256-best-e840-partial": "#2A9D8F",
    "S7-B-e1000": "#9C6ADE",
}
CONVERGENCE_MAP = {
    "F0": "F0-e1000",
    "CQ-LR-128": "CQ-LR-128-e1000",
    "CQ-LR-256 (partial)": "CQ-LR-256-best-e840-partial",
    "S7-B": "S7-B-e1000",
}


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=10, weight="bold")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(EVALUATION / "final_comparison.csv").set_index("candidate")
    convergence = pd.read_csv(EVALUATION / "convergence.csv")

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
        }
    )

    fig = plt.figure(figsize=(183 / 25.4, 140 / 25.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    recon_grid = grid[1, 0].subgridspec(1, 2, wspace=0.10)
    ax_c1 = fig.add_subplot(recon_grid[0, 0])
    ax_c4 = fig.add_subplot(recon_grid[0, 1], sharey=ax_c1)
    ax_d = fig.add_subplot(grid[1, 1])

    for raw_label, model_label in CONVERGENCE_MAP.items():
        rows = convergence[convergence["candidate"] == raw_label].sort_values("epoch")
        partial = "partial" in raw_label
        ax_a.plot(
            rows["epoch"], rows["mean_rf_loss"],
            marker="o", markersize=3.0, markerfacecolor="white" if partial else COLORS[model_label],
            markeredgecolor=COLORS[model_label], linestyle="--" if partial else "-",
            color=COLORS[model_label], label=DISPLAY[model_label],
        )
    f0_final = float(table.loc["F0-e1000", "fixed_manifest_rf_mean"])
    ax_a.axhline(f0_final, color="#A5A5A5", linewidth=0.7, linestyle=":")
    ax_a.set_yscale("log")
    ax_a.set_ylim(0.235, 2.2)
    ax_a.set_yticks([0.25, 0.5, 1.0, 2.0], ["0.25", "0.5", "1.0", "2.0"])
    ax_a.set_xlim(-20, 1020)
    ax_a.set_xticks([0, 200, 400, 600, 800, 1000])
    ax_a.tick_params(axis="y", which="minor", labelleft=False)
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Fixed-manifest RF loss (log scale)")
    ax_a.set_title("Stage7 crosses F0 final quality by epoch 400", loc="left", weight="bold")
    ax_a.legend(ncols=2, fontsize=6.1, loc="upper right")

    compared = [
        "CQ-LR-128-e1000", "CQ-LR-256-best-e840-partial", "S7-B-e1000"
    ]
    values = np.array([
        -100 * float(table.loc[label, "paired_difference_vs_f0_e1000_mean"]) / f0_final
        for label in compared
    ])
    lows = np.array([
        -100 * float(table.loc[label, "paired_difference_vs_f0_e1000_ci95_high"]) / f0_final
        for label in compared
    ])
    highs = np.array([
        -100 * float(table.loc[label, "paired_difference_vs_f0_e1000_ci95_low"]) / f0_final
        for label in compared
    ])
    x = np.arange(len(compared))
    bars = ax_b.bar(
        x, values, width=0.66, color=[COLORS[label] for label in compared],
        edgecolor=[COLORS[label] for label in compared], linewidth=0.9,
    )
    bars[1].set_facecolor("white")
    ax_b.errorbar(
        x, values, yerr=np.vstack([values - lows, highs - values]), fmt="none",
        ecolor="#222222", elinewidth=0.8, capsize=2.5,
    )
    ax_b.axhline(0, color="#777777", linewidth=0.7)
    ax_b.set_xticks(x, [DISPLAY[label] for label in compared], rotation=16, ha="right")
    ax_b.set_ylabel("Paired RF improvement vs F0 e1000 (%)")
    ax_b.set_ylim(-13, 24)
    ax_b.set_title("19.7% paired RF improvement at epoch 1000", loc="left", weight="bold")
    for xpos, value in zip(x, values):
        ax_b.text(
            xpos, value + (1.0 if value >= 0 else -1.0), f"{value:+.1f}%",
            ha="center", va="bottom" if value >= 0 else "top", fontsize=6.3,
        )

    recon_order = [
        "F0-e1000", "CQ-LR-128-e1000", "CQ-LR-256-best-e840-partial", "S7-B-e1000"
    ]
    y = np.arange(len(recon_order))
    for axis, nfe in ((ax_c1, 1), (ax_c4, 4)):
        for ypos, label in zip(y, recon_order):
            mean_value = float(table.loc[label, f"recon_nfe{nfe}_mean"])
            worst_value = float(table.loc[label, f"recon_nfe{nfe}_worst"])
            partial = "partial" in label
            axis.plot([mean_value, worst_value], [ypos, ypos], color=COLORS[label], alpha=0.55)
            axis.scatter(
                mean_value, ypos, s=18, facecolor="white" if partial else COLORS[label],
                edgecolor=COLORS[label], linewidth=0.9, zorder=3,
            )
            axis.scatter(
                worst_value, ypos, s=20, facecolor="white", edgecolor=COLORS[label],
                linewidth=1.0, marker="D", zorder=3,
            )
        axis.set_xlabel("Relative L2")
        axis.set_title(f"Euler NFE{nfe}", loc="left", weight="bold")
        axis.set_xlim(0.19, 0.73)
        axis.set_xticks([0.2, 0.4, 0.6])
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    ax_c1.set_yticks(y, [DISPLAY[label] for label in recon_order])
    ax_c1.invert_yaxis()
    ax_c4.tick_params(labelleft=False)
    ax_c1.text(
        0.0, -0.22, "● five-field mean   ◇ worst field (U₁)",
        transform=ax_c1.transAxes, fontsize=6.0,
    )

    pareto = ["F0-e1000", "CQ-LR-128-e1000", "S7-B-e1000"]
    offsets = {
        "F0-e1000": (-62, 8),
        "CQ-LR-128-e1000": (-38, -31),
        "S7-B-e1000": (-78, 8),
    }
    for label in pareto:
        row = table.loc[label]
        memory_gib = float(row["train_peak_allocated_mb"]) / 1024
        size = 38 + 3.0 * memory_gib
        ax_d.scatter(
            row["train_step_ms"], row["fixed_manifest_rf_mean"], s=size,
            color=COLORS[label], edgecolor="white", linewidth=0.7, zorder=3,
        )
        ax_d.annotate(
            f"{DISPLAY[label]}\n{memory_gib:.1f} GiB | 1M {row['persistent_1m_nfe4_speedup_vs_f0']:.2f}×",
            (row["train_step_ms"], row["fixed_manifest_rf_mean"]),
            xytext=offsets[label], textcoords="offset points", fontsize=5.9,
            color=COLORS[label],
        )
    ax_d.set_xlabel("B128/Q4096 training step (ms)")
    ax_d.set_ylabel("Epoch-1000 fixed-manifest RF loss")
    ax_d.set_xlim(365, 575)
    ax_d.set_ylim(0.245, 0.375)
    ax_d.set_title("Stage7 defines the balanced Pareto point", loc="left", weight="bold")
    ax_d.text(
        0.03, 0.04, "Lower left is better\nBubble area encodes peak memory",
        transform=ax_d.transAxes, fontsize=5.9, color="#555555",
    )

    for axis in (ax_a, ax_b, ax_d):
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=0)
        axis.set_axisbelow(True)
    for label, axis in zip("abcd", (ax_a, ax_b, ax_c1, ax_d)):
        panel_label(axis, label)

    fig.suptitle(
        "Stage7-All256 restores quality while retaining compact-query efficiency",
        fontsize=9.5, weight="bold",
    )
    stem = OUTPUT / "stage7_final_quality_throughput_pareto"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()

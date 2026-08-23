#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "_CheckNotes/Stage7_smart_cq/screen_200/evaluation"
OUTPUT = ROOT / "figures/generated/stage7_epoch200_pareto"

ORDER = ["F0-128", "CQ-LR-128", "CQ-LR-256", "S7-A", "S7-B"]
DISPLAY = {
    "F0-128": "F0",
    "CQ-LR-128": "CQ-LR-128",
    "CQ-LR-256": "CQ-LR-256†",
    "S7-A": "S7-A",
    "S7-B": "S7-B",
}
COLORS = {
    "F0-128": "#4D4D4D",
    "CQ-LR-128": "#4C78A8",
    "CQ-LR-256": "#2A9D8F",
    "S7-A": "#E69F00",
    "S7-B": "#9C6ADE",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(EVALUATION / "comparison_table.csv").set_index("candidate")
    convergence = pd.read_csv(EVALUATION / "convergence.csv")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "lines.linewidth": 1.5,
        }
    )

    fig = plt.figure(figsize=(183 / 25.4, 135 / 25.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    recon_grid = grid[1, 0].subgridspec(1, 2, wspace=0.12)
    ax_c1 = fig.add_subplot(recon_grid[0, 0])
    ax_c4 = fig.add_subplot(recon_grid[0, 1], sharey=ax_c1)
    ax_d = fig.add_subplot(grid[1, 1])

    for candidate in ORDER:
        rows = convergence[convergence["candidate"] == candidate].sort_values("epoch")
        marker = "o" if len(rows) > 1 else "D"
        linestyle = "-" if len(rows) > 1 else "none"
        ax_a.plot(
            rows["epoch"],
            rows["mean_rf_loss"],
            marker=marker,
            markersize=3.3,
            linestyle=linestyle,
            color=COLORS[candidate],
            label=DISPLAY[candidate],
        )
    ax_a.set_xlabel("Epoch")
    ax_a.set_ylabel("Fixed-manifest RF loss")
    ax_a.set_xlim(-5, 205)
    ax_a.set_ylim(0.35, 2.05)
    ax_a.set_xticks([0, 50, 100, 150, 200])
    ax_a.set_title("Controlled convergence", loc="left", weight="bold")
    ax_a.legend(ncols=2, fontsize=6.2, loc="upper right")

    compared = ["CQ-LR-128", "CQ-LR-256", "S7-A", "S7-B"]
    f0_loss = float(table.loc["F0-128", "fixed_manifest_rf_mean"])
    improvement = np.array(
        [-100 * table.loc[label, "paired_difference_vs_f0_mean"] / f0_loss for label in compared]
    )
    low = np.array(
        [-100 * table.loc[label, "paired_difference_vs_f0_ci95_high"] / f0_loss for label in compared]
    )
    high = np.array(
        [-100 * table.loc[label, "paired_difference_vs_f0_ci95_low"] / f0_loss for label in compared]
    )
    x = np.arange(len(compared))
    ax_b.bar(x, improvement, color=[COLORS[label] for label in compared], width=0.68)
    ax_b.errorbar(
        x,
        improvement,
        yerr=np.vstack([improvement - low, high - improvement]),
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=2.5,
    )
    ax_b.axhline(0, color="#777777", linewidth=0.7)
    ax_b.set_xticks(x, [DISPLAY[label] for label in compared], rotation=18, ha="right")
    ax_b.set_ylabel("Paired RF improvement vs F0 (%)")
    ax_b.set_title("S7-B: 19.4% paired RF improvement", loc="left", weight="bold")
    for xpos, value in zip(x, improvement):
        ax_b.text(xpos, value + (0.8 if value >= 0 else -1.0), f"{value:+.1f}%", ha="center",
                  va="bottom" if value >= 0 else "top", fontsize=6.2)

    y = np.arange(len(ORDER))
    for axis, nfe in ((ax_c1, 1), (ax_c4, 4)):
        mean_values = np.array([table.loc[label, f"recon_nfe{nfe}_mean"] for label in ORDER])
        worst_values = np.array([table.loc[label, f"recon_nfe{nfe}_worst"] for label in ORDER])
        for ypos, label, mean_value, worst_value in zip(y, ORDER, mean_values, worst_values):
            axis.plot([mean_value, worst_value], [ypos, ypos], color=COLORS[label], alpha=0.55)
            axis.scatter(mean_value, ypos, s=18, color=COLORS[label], marker="o", zorder=3)
            axis.scatter(worst_value, ypos, s=20, facecolor="white", edgecolor=COLORS[label],
                         linewidth=1.0, marker="D", zorder=3)
        axis.set_xlabel("Relative L2")
        axis.set_title(f"Euler NFE{nfe}", loc="left", weight="bold")
        axis.set_xlim(0.20, 1.02)
        axis.set_xticks([0.25, 0.50, 0.75, 1.00])
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    ax_c1.set_yticks(y, [DISPLAY[label] for label in ORDER])
    ax_c1.invert_yaxis()
    ax_c4.tick_params(labelleft=False)
    ax_c1.text(0.02, -0.22, "● mean   ◇ worst field (U₁)", transform=ax_c1.transAxes, fontsize=6.1)

    for label in ORDER:
        row = table.loc[label]
        memory_gib = float(row["train_peak_allocated_mb"]) / 1024
        size = 35 + 3.2 * memory_gib
        filled = label != "CQ-LR-256"
        ax_d.scatter(
            row["train_step_ms"],
            row["fixed_manifest_rf_mean"],
            s=size,
            color=COLORS[label] if filled else "white",
            edgecolor=COLORS[label],
            linewidth=1.1,
            zorder=3,
        )
        inference = row["persistent_1m_nfe4_speedup_vs_f0"]
        inference_text = "n/a" if pd.isna(inference) else f"{inference:.2f}×"
        offset = {
            "F0-128": (-66, 7),
            "CQ-LR-128": (5, 5),
            "CQ-LR-256": (5, -25),
            "S7-A": (5, 7),
            "S7-B": (-54, 8),
        }[label]
        ax_d.annotate(
            f"{DISPLAY[label]}\n{memory_gib:.1f} GiB | 1M {inference_text}",
            (row["train_step_ms"], row["fixed_manifest_rf_mean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=5.8,
            color=COLORS[label],
        )
    ax_d.set_xlabel("B128/Q4096 training step (ms)")
    ax_d.set_ylabel("Epoch-200 fixed-manifest RF loss")
    ax_d.set_xlim(300, 575)
    ax_d.set_ylim(0.385, 0.535)
    ax_d.set_title("Quality–throughput Pareto", loc="left", weight="bold")
    ax_d.text(0.02, 0.03, "Lower left is better\n† unmatched cost diagnostic",
              transform=ax_d.transAxes, fontsize=5.8, color="#555555")

    for axis in (ax_a, ax_b, ax_d):
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=0)
        axis.set_axisbelow(True)
    for label, axis in zip("abcd", (ax_a, ax_b, ax_c1, ax_d)):
        axis.text(-0.16, 1.08, label, transform=axis.transAxes, fontsize=10, weight="bold")

    fig.suptitle(
        "Stage7-All256 earns the sole epoch-1000 continuation",
        fontsize=9.5,
        weight="bold",
    )
    stem = OUTPUT / "stage7_epoch200_selection"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

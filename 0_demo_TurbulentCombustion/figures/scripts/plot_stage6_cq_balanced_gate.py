#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "_CheckNotes/Stage6_CQ_balanced_quality_recovery"
OUTPUT = ROOT / "figures/generated/stage6_cq_balanced_gate"

COLORS = {
    "F0": "#4D4D4D",
    "CQ-LR": "#4C78A8",
    "CQ-Balanced-192-Full": "#E69F00",
    "CQ-Balanced-224-Full": "#9C6ADE",
}


def load(name: str) -> dict:
    return json.loads((PACKAGE / "cost_benchmark" / name).read_text())


def row(data: dict, label: str, n_query: int = 4096) -> dict:
    return next(
        item for item in data["scaling"]
        if item["label"] == label and int(item["N_query"]) == n_query
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    contract = """# Figure contract — CQ-Balanced efficiency gate

- Core conclusion: structured-concat CQ-Balanced (192-D primary and 224-D sole
  fallback) does not retain enough of CQ-LR's training efficiency to justify a
  quality-training screen.
- Evidence chain: a, clean batch-128 step latency against the 1.15x gate; b,
  allocated/reserved memory reductions against the 10% gate; c, batch-1
  scaling at 4k/16k/65k queries; d, unchanged persistent 1M-query NFE-4 path.
- Archetype: quantitative grid with the clean-protocol gate as the hero row.
- Backend: Python/matplotlib exclusively, rendered in the `fig` environment.
- Export: 183 mm x 120 mm; editable SVG text, PDF, and 300-dpi PNG preview.
- Sources: the three JSON benchmark artifacts in the Stage 6 evidence package.
- Review risks: random candidate weights are sufficient for architecture cost
  but do not provide CQ-Balanced quality; the 192 and 224 candidates therefore
  must not be placed on a validation-loss Pareto axis.
"""
    (OUTPUT / "figure_contract.md").write_text(contract)

    scaling = load("cost_benchmark.json")
    primary = load("clean_b128_q4096.json")
    fallback = load("fallback_224_clean_b128_q4096.json")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    })

    fig = plt.figure(figsize=(183 / 25.4, 120 / 25.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.08, 1.0])
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    labels = ["F0", "CQ-LR", "CQ-Balanced-192-Full", "CQ-Balanced-224-Full"]
    display = ["F0", "CQ-LR", "Balanced-192", "Balanced-224"]
    clean_rows = {
        "F0": row(primary, "F0"),
        "CQ-LR": row(primary, "CQ-LR"),
        "CQ-Balanced-192-Full": row(primary, "CQ-Balanced-192-Full"),
        "CQ-Balanced-224-Full": row(fallback, "CQ-Balanced-224-Full"),
    }
    step = [clean_rows[label]["training_step_ms"] for label in labels]
    positions = np.arange(len(labels))
    ax_a.bar(positions, step, color=[COLORS[label] for label in labels], width=0.68)
    threshold = step[0] / 1.15
    ax_a.axhline(threshold, color="#B22222", linestyle="--", linewidth=1.0,
                 label="1.15x speed gate")
    ax_a.set_xticks(positions, display, rotation=18, ha="right")
    ax_a.set_ylabel("Full training step (ms)")
    ax_a.set_title("Clean protocol: batch 128, 4,096 queries", loc="left", weight="bold")
    ax_a.legend(loc="upper left")
    for x, value in zip(positions, step):
        ax_a.text(x, value + 8, f"{value:.0f}", ha="center", va="bottom", fontsize=6.3)

    candidates = labels[1:]
    x = np.arange(len(candidates))
    allocated = [
        100 * (1 - clean_rows[label]["peak_allocated_mb"] / clean_rows["F0"]["peak_allocated_mb"])
        for label in candidates
    ]
    reserved = [
        100 * (1 - clean_rows[label]["peak_reserved_mb"] / clean_rows["F0"]["peak_reserved_mb"])
        for label in candidates
    ]
    width = 0.34
    ax_b.bar(x - width / 2, allocated, width, color="#6BAED6", label="Allocated")
    ax_b.bar(x + width / 2, reserved, width, color="#9ECAE1", label="Reserved")
    ax_b.axhline(10, color="#B22222", linestyle="--", linewidth=1.0, label="10% gate")
    ax_b.axhline(0, color="#777777", linewidth=0.6)
    ax_b.set_xticks(x, ["CQ-LR", "Balanced-192", "Balanced-224"], rotation=18, ha="right")
    ax_b.set_ylabel("Memory reduction vs F0 (%)")
    ax_b.set_title("Only CQ-LR clears the memory gate", loc="left", weight="bold")
    ax_b.legend(ncols=3, fontsize=6, loc="upper right")

    query_sizes = np.array([4096, 16384, 65536])
    for label in ("CQ-LR", "CQ-Balanced-192-Full"):
        speedups = []
        for n_query in query_sizes:
            f0 = row(scaling, "F0", int(n_query))
            candidate = row(scaling, label, int(n_query))
            speedups.append(f0["training_step_ms"] / candidate["training_step_ms"])
        ax_c.plot(
            query_sizes, speedups, marker="o", linewidth=1.6,
            color=COLORS[label], label=("CQ-LR" if label == "CQ-LR" else "Balanced-192"),
        )
    ax_c.axhline(1.15, color="#B22222", linestyle="--", linewidth=1.0)
    ax_c.axhline(1.0, color="#777777", linewidth=0.6)
    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks(query_sizes, ["4k", "16k", "65k"])
    ax_c.set_ylabel("Training-step speedup vs F0")
    ax_c.set_xlabel("Queries per sample (batch 1)")
    ax_c.set_title("Balanced-192 never reaches 1.15x", loc="left", weight="bold")
    ax_c.legend(loc="upper left")

    million = {item["label"]: item for item in scaling["million_query_reconstruction"]}
    persistent_labels = ["F0", "CQ-LR", "CQ-Balanced-192-Full"]
    persistent_display = ["F0", "CQ-LR", "Balanced-192"]
    latency = [million[label]["wall_s"] for label in persistent_labels]
    p = np.arange(len(persistent_labels))
    ax_d.bar(p, latency, color=[COLORS[label] for label in persistent_labels], width=0.68)
    ax_d.set_xticks(p, persistent_display, rotation=18, ha="right")
    ax_d.set_ylabel("Steady latency (s)")
    ax_d.set_title("1M queries, Euler NFE=4, persistent Top-K", loc="left", weight="bold")
    for x_pos, value in zip(p, latency):
        ax_d.text(x_pos, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=6.3)

    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        axis.text(-0.16, 1.08, label, transform=axis.transAxes, fontsize=10, weight="bold")
        axis.grid(axis="y", color="#E6E6E6", linewidth=0.5, zorder=0)
        axis.set_axisbelow(True)

    fig.suptitle(
        "CQ-Balanced restores structure but not the required training efficiency",
        fontsize=9, weight="bold",
    )
    stem = OUTPUT / "cq_balanced_efficiency_gate"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

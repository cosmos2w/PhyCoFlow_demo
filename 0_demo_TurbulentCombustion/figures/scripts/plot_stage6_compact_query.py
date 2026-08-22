#!/usr/bin/env python3
"""Plot the measured Stage-6 compact-query cost/quality decision."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "_CheckNotes/Stage6_compact_query"
OUTPUT = ROOT / "figures/generated/stage6_compact_query"
COLORS = {
    "F0": "#555555",
    "CQ-Full": "#2B6CB0",
    "CQ-LR": "#DD6B20",
    "CQ-160": "#2F855A",
}
PERFORMANCE_LABELS = ("F0", "CQ-Full", "CQ-LR")


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing real Stage-6 source artifact: {path}. "
            "Run _CheckNotes/Stage6_compact_query/run_gates_c_d.sh first."
        )
    return json.loads(path.read_text())


def epoch_losses(path: Path) -> tuple[list[int], list[float]]:
    result = load(path)
    rows = sorted(
        (
            int(value["epoch"]),
            float(value["mean_rf_loss"]),
        )
        for value in result["summary"].values()
    )
    return [row[0] for row in rows], [row[1] for row in rows]


def reconstruction(path: Path) -> tuple[list[int], list[float]]:
    rows = sorted(load(path), key=lambda row: int(row["nfe"]))
    return (
        [int(row["nfe"]) for row in rows],
        [float(row["mean_field_relative_l2"]) for row in rows],
    )


def main() -> None:
    cost = load(EVIDENCE / "benchmarks/cost_benchmark.json")
    selection = load(EVIDENCE / "formal_candidate/selection.json")
    full_epochs, full_losses = epoch_losses(
        EVIDENCE / "screen_cq_full/evaluation/fixed_manifest/milestones.json"
    )
    lr_epochs, lr_losses = epoch_losses(
        EVIDENCE / "screen_cq_lr/evaluation/fixed_manifest/milestones.json"
    )
    rescue_epochs, rescue_losses = epoch_losses(
        EVIDENCE / "screen_cq_rescue160/evaluation/fixed_manifest/milestones.json"
    )
    full_nfe, full_recon = reconstruction(
        EVIDENCE
        / "screen_cq_full/evaluation/matched_reconstruction/epoch_0060/summary.json"
    )
    lr_nfe, lr_recon = reconstruction(
        EVIDENCE
        / "screen_cq_lr/evaluation/matched_reconstruction/epoch_0060/summary.json"
    )
    rescue_nfe, rescue_recon = reconstruction(
        EVIDENCE
        / "screen_cq_rescue160/evaluation/matched_reconstruction/epoch_0060/summary.json"
    )
    f0_nfe, f0_recon = reconstruction(
        ROOT
        / "_CheckNotes/Stage6_formal_baseline/evaluation/"
        "matched_reconstruction/F0_best/summary.json"
    )
    f0_fixed = load(
        ROOT / "_CheckNotes/Stage6_formal_baseline/evaluation/fixed_manifest_best.json"
    )
    f0_rf = next(
        float(value["mean_rf_loss"])
        for label, value in f0_fixed["summary"].items()
        if label.startswith("F0")
    )

    scaling = {
        label: sorted(
            [
                row for row in cost["scaling"]
                if row["label"] == label and row.get("status") == "ok"
            ],
            key=lambda row: int(row["N_query"]),
        )
        for label in PERFORMANCE_LABELS
    }
    million = {
        row["label"]: row
        for row in cost["million_query_reconstruction"]
        if row.get("status") == "ok"
    }

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 7.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7), constrained_layout=True)

    ax = axes[0, 0]
    for label, rows in scaling.items():
        x = [row["N_query"] for row in rows]
        ax.plot(
            x,
            [row["forward_ms"] for row in rows],
            color=COLORS[label],
            marker="o",
            linewidth=1.8,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Queries")
    ax.set_ylabel("Forward time (ms)")
    ax.set_title("a  Query-model scaling", loc="left", fontweight="bold")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    labels = list(PERFORMANCE_LABELS)
    positions = np.arange(len(labels))
    peak = [
        next(
            row["peak_allocated_mb"]
            for row in scaling[label]
            if int(row["N_query"]) == 65536
        )
        for label in labels
    ]
    cache = [million[label]["static_query_cache_mb"] for label in labels]
    width = 0.36
    ax.bar(
        positions - width / 2,
        peak,
        width,
        color=[COLORS[label] for label in labels],
        alpha=0.9,
        label="65,536-query training peak",
    )
    ax.bar(
        positions + width / 2,
        cache,
        width,
        color=[COLORS[label] for label in labels],
        alpha=0.45,
        hatch="//",
        label="1M static cache",
    )
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Memory (MiB)")
    ax.set_title("b  Activation/cache footprint", loc="left", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.axhline(f0_rf, color=COLORS["F0"], linestyle="--", linewidth=1.4, label="F0 best")
    ax.plot(
        full_epochs, full_losses, marker="o", color=COLORS["CQ-Full"],
        linewidth=1.8, label="CQ-Full",
    )
    ax.plot(
        lr_epochs, lr_losses, marker="o", color=COLORS["CQ-LR"],
        linewidth=1.8, label="CQ-LR",
    )
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Fixed-manifest RF loss")
    ax.set_title("c  Controlled 60-epoch screen", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(
        f0_nfe, f0_recon, marker="o", color=COLORS["F0"],
        linestyle="--", linewidth=1.5, label="F0 best (epoch 180)",
    )
    ax.plot(
        full_nfe, full_recon, marker="o", color=COLORS["CQ-Full"],
        linewidth=1.8, label="CQ-Full epoch 60",
    )
    ax.plot(
        lr_nfe, lr_recon, marker="o", color=COLORS["CQ-LR"],
        linewidth=1.8, label="CQ-LR epoch 60",
    )
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel("Euler NFE")
    ax.set_ylabel("Mean five-field relative L2")
    ax.set_title("d  Matched reconstruction", loc="left", fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    selected = selection["selected_candidate"]
    selection_status = selection.get("formal_candidate_status", "primary_ready")
    title = (
        f"Stage-6 compact query decoder — primary pick: {selected}; do not replace F0"
        if selection_status == "prepared_not_recommended_rescue_failed"
        else
        f"Stage-6 compact query decoder — selected for formal run: {selected}"
        if selection_status == "primary_ready"
        else f"Stage-6 compact query decoder — primary pick: {selected}; CQ-160 rescue required"
    )
    fig.suptitle(title, fontsize=11, fontweight="bold")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT / "stage6_compact_query_decision"
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)

    contract = f"""# Figure contract — Stage 6 compact query decoder

## Core scientific claim

The compact query decoder reduces repeated-query execution and memory while
the selected 60-epoch candidate remains within the prespecified CQ quality
screen. Primary selected candidate: {selected}. Status: {selection_status}.

## Source files

- {EVIDENCE / "benchmarks/cost_benchmark.json"}
- {EVIDENCE / "screen_cq_full/evaluation/fixed_manifest/milestones.json"}
- {EVIDENCE / "screen_cq_lr/evaluation/fixed_manifest/milestones.json"}
- {EVIDENCE / "screen_cq_full/evaluation/matched_reconstruction/epoch_0060/summary.json"}
- {EVIDENCE / "screen_cq_lr/evaluation/matched_reconstruction/epoch_0060/summary.json"}
- {ROOT / "_CheckNotes/Stage6_formal_baseline/evaluation/fixed_manifest_best.json"}
- {ROOT / "_CheckNotes/Stage6_formal_baseline/evaluation/matched_reconstruction/F0_best/summary.json"}
- {EVIDENCE / "formal_candidate/selection.json"}

## Panel map

- a: measured model forward scaling at 4,096/16,384/65,536 queries.
- b: measured 65,536-query training peak and 1M static cache.
- c: 64-layout, three-repeat fixed-manifest RF loss at epochs 1/20/40/60.
- d: matched snapshot/sensors/RF seed Euler NFE 1/2/4 reconstruction.

## Metrics/statistics

Timings and memory are same-run CUDA measurements. RF points are means over
the fixed 64 layouts and three RF repeats per layout. Reconstruction is the
five-field mean relative L2 on the established controlled snapshot.

## Caveats

The CQ screen is one seed and only 60 epochs. F0 is its 200-epoch best
checkpoint, so panel c/d do not establish formal CQ-versus-F0 parity. That
requires the prepared selected-candidate 200-epoch run.
"""
    (OUTPUT / "figure_contract.md").write_text(contract)


if __name__ == "__main__":
    main()

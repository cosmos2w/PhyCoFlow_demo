#!/usr/bin/env python3
"""Plot the reproducible CQ persistent Top-K training/reconstruction A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

COLORS = {"no_persistent": "#4C78A8", "persistent_topk": "#E45756"}


def load(path: Path):
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    comparison = load(args.comparison)
    runs = comparison["runs"]
    histories = {
        key: load(Path(value["run_dir"]) / "loss_history.json")
        for key, value in runs.items()
    }
    benchmark_root = args.comparison.parent / "benchmarks"
    benchmark = {
        "no_persistent": load(benchmark_root / "no_persistent_checkpoint.json"),
        "persistent_topk": load(benchmark_root / "persistent_topk_checkpoint.json"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    labels = {
        "no_persistent": "Pre-cache revision",
        "persistent_topk": "Persistent-cache revision",
    }
    for key, history in histories.items():
        color = COLORS[key]
        axes[0, 0].plot(
            [row["epoch"] for row in history],
            [row["train_loss"] for row in history],
            color=color,
            linewidth=1.5,
            alpha=0.9,
            label=labels[key],
        )
        val = [row for row in history if row.get("val_loss") is not None]
        axes[0, 1].plot(
            [row["epoch"] for row in val],
            [row["val_loss"] for row in val],
            color=color,
            marker="o",
            markersize=2.5,
            linewidth=1.3,
            label=labels[key],
        )

    nfes = [1, 2, 4, 8]
    modes = {
        "no_persistent": "static_per_call",
        "persistent_topk": "static_persistent_geometry",
    }
    latency = {}
    for key in ("no_persistent", "persistent_topk"):
        rows = benchmark[key]["rows"]
        latency[key] = [
            next(
                float(row["mean_wall_s"])
                for row in rows
                if int(row["N_query"]) == 1_000_000
                and int(row["NFE"]) == nfe
                and row["mode"] == modes[key]
            )
            for nfe in nfes
        ]
        axes[1, 0].plot(
            nfes,
            latency[key],
            color=COLORS[key],
            marker="o",
            linewidth=1.8,
            label=("Stage-4 per-call static" if key == "no_persistent" else "Persistent geometry + static"),
        )
    speedups = [a / b for a, b in zip(latency["no_persistent"], latency["persistent_topk"])]
    axes[1, 1].bar(nfes, speedups, width=0.65, color="#72B7B2")
    axes[1, 1].axhline(1.15, color="#777777", linestyle="--", linewidth=1.1, label="1.15× acceptance")

    axes[0, 0].set_title("a  Training convergence")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("RF training loss")
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_title("b  Validation convergence")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("RF validation loss")
    axes[1, 0].set_title("c  1M-query steady reconstruction")
    axes[1, 0].set_xlabel("Euler NFE")
    axes[1, 0].set_ylabel("Latency (s)")
    axes[1, 0].set_xticks(nfes)
    axes[1, 1].set_title("d  Persistent-cache speedup")
    axes[1, 1].set_xlabel("Euler NFE")
    axes[1, 1].set_ylabel("Speedup (×)")
    axes[1, 1].set_xticks(nfes)

    for ax in axes.flat:
        ax.grid(True, alpha=0.22, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    fig.suptitle("CQ-LR persistent Top-K: quality neutrality and reconstruction efficiency", y=0.995)
    fig.tight_layout()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "cq_persistent_training_ab"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)

    contract = f"""# Figure contract

## Core scientific claim

Persistent geometry-only Top-K reuse reduces repeated CQ-LR reconstruction
latency while leaving the training trajectory and fixed-manifest RF quality
unchanged within the predeclared tolerance.

## Source files

- `{args.comparison.resolve()}`
- both run directories' `loss_history.json`
- `benchmarks/no_persistent_checkpoint.json`
- `benchmarks/persistent_topk_checkpoint.json`

## Panel map

- a: paired 200-epoch training RF curves;
- b: paired validation RF curves;
- c: one-million-query steady latency at Euler NFE 1/2/4/8;
- d: persistent speedup over the prior per-call Stage-4 static cache.

## Metrics and caveats

Latency is CUDA-synchronized and excludes the separately reported one-time
geometry build. Training timing is a neutrality control because persistent
Top-K is not used in the RF training objective. This is a one-seed, two-GPU
paired implementation check rather than a multi-seed scientific comparison.
"""
    (args.output_dir / "figure_contract.md").write_text(contract)


if __name__ == "__main__":
    main()

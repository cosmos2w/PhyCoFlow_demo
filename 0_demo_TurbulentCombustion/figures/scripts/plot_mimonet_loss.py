"""Replot a MIMONet Cond_T training history from its saved CSV."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["svg.fonttype"] = "none"

DEMO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    history = run / "loss_history.csv"
    rows = list(csv.DictReader(history.open(newline="")))
    if not rows:
        raise ValueError(f"No training rows in {history}")
    train = [(int(row["epoch"]), float(row["train_loss"])) for row in rows]
    validation = [(int(row["epoch"]), float(row["val_loss"])) for row in rows
                  if row["val_loss"]]
    if any(value <= 0 for _, value in train + validation):
        raise ValueError("Loss curve requires positive finite MSE values")
    destination = DEMO / "figures/generated/mimonet_condT_loss" / run.name
    destination.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)
    ax.plot(*zip(*train), label="Training", color="#244C70", linewidth=1.7)
    if validation:
        ax.plot(*zip(*validation), label="Validation", color="#C05C2D", linewidth=1.7,
                marker="o", markersize=2.8)
    ax.set(xlabel="Epoch", ylabel="Normalized five-field MSE",
           title="MIMONet Cond_T training")
    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    for extension in ("svg", "pdf", "png"):
        fig.savefig(destination / f"loss_history.{extension}", dpi=180)
    plt.close(fig)
    shutil.copy2(destination / "loss_history.png", run / "loss_history.png")
    (destination / "figure_contract.md").write_text(
        "# MIMONet Cond_T loss curve\n\n"
        "- Claim: shows optimization and validation behavior over the observed epochs; it does not establish final reconstruction fidelity.\n"
        f"- Source: `{history}`.\n"
        "- Panel: training and holdout-validation normalized five-field MSE by epoch, log y-axis.\n"
        "- Caveat: A0-style validation resamples sparse T sensors, so individual validation values fluctuate.\n"
    )


if __name__ == "__main__":
    main()

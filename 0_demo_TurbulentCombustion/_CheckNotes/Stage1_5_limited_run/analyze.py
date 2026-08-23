#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_rows(path: Path):
    with open(path) as handle:
        return list(csv.DictReader(handle))


def summarize_run(run: Path, fixed_csv: Path, log_path: Path):
    history = load_rows(run / "loss_history.csv")
    diagnostics = load_rows(run / "data_path_diagnostics.csv")
    fixed = load_rows(fixed_csv)
    recon_metrics = {}
    for path in sorted((run / "Evaluation").glob("epoch_*/*metrics*.json")):
        recon_metrics[str(path.relative_to(run))] = json.loads(path.read_text())
    log_text = log_path.read_text(errors="replace")
    train_matches = re.findall(r"\[train\] epoch=\d+ loss=([0-9.eE+-]+)", log_text)
    val_matches = re.findall(r"\[valid\] epoch=\d+ loss=([0-9.eE+-]+)", log_text)
    return {
        "run": str(run),
        "epochs_logged": len(history),
        "epochs_completed": len(train_matches),
        "initial_train_loss": float(history[0]["train_loss"]),
        "final_train_loss": float(train_matches[-1]),
        "final_val_loss": float(val_matches[-1]),
        "mean_epoch_seconds": sum(float(row["epoch_seconds"]) for row in history) / len(history),
        "steady_epoch_seconds": sum(float(row["epoch_seconds"]) for row in history[1:]) / max(len(history) - 1, 1),
        "diagnostic_peak_allocated_mb": max(float(row["gpu_peak_allocated_mb"]) for row in diagnostics),
        "fixed_manifest_mean_loss": sum(float(row["loss"]) for row in fixed) / len(fixed),
        "fixed_manifest_rows": len(fixed),
        "reconstruction_metrics": recon_metrics,
    }


def main():
    control_run, large_run = [Path(line) for line in (ROOT / "run_paths.txt").read_text().splitlines()]
    summary = {
        "control": summarize_run(
            control_run, ROOT / "control_fixed_manifest.csv", ROOT / "logs/control.log"
        ),
        "large_effective_query": summarize_run(
            large_run,
            ROOT / "large_effective_query_fixed_manifest.csv",
            ROOT / "logs/large_effective_query.log",
        ),
        "reconstruction_stress": json.loads(
            (ROOT.parent / "Stage4_reconstruction/reconstruction_scaling.json").read_text()
        )["rows"][-1],
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

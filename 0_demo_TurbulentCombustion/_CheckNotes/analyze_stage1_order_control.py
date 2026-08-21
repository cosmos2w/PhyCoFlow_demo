#!/usr/bin/env python3
"""Analyze the 12-epoch optimized-first Stage-1 timing control."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "_CheckNotes" / "Stage1_order_runs"
RUNTIME = ROOT / "_CheckNotes" / "Stage1_order_runtime"


def one_history(path: Path) -> Path:
    matches = list(path.glob("*/loss_history.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one history under {path}, found {matches}.")
    return matches[0]


def summarize(path: Path) -> dict:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12 or int(rows[-1]["epoch"]) != 12:
        raise RuntimeError(f"Incomplete 12-epoch history: {path}.")
    # Epochs 1–2 include compilation, startup validation, and worker warm-up.
    steady = [float(row["train_seconds"]) for row in rows[2:]]
    return {
        "history": str(path),
        "epochs": len(rows),
        "steady_epochs": "3-12",
        "steady_mean_seconds_per_epoch": statistics.fmean(steady),
        "steady_median_seconds_per_epoch": statistics.median(steady),
        "steady_min_seconds_per_epoch": min(steady),
        "steady_max_seconds_per_epoch": max(steady),
        "final_train_loss": float(rows[-1]["train_loss"]),
    }


def main() -> None:
    status = dict(
        line.split("=", 1) for line in (RUNTIME / "exit_status.txt").read_text().splitlines()
    )
    if status != {"optimized_exit": "0", "legacy_exit": "0"}:
        raise RuntimeError(f"Timing jobs did not both succeed: {status}.")
    result = {
        name: summarize(one_history(RUN_ROOT / name)) for name in ("optimized", "legacy")
    }
    ratio = (
        result["optimized"]["steady_mean_seconds_per_epoch"]
        / result["legacy"]["steady_mean_seconds_per_epoch"]
    )
    result["comparison"] = {
        "optimized_over_legacy_steady_epoch_ratio": ratio,
        "optimized_time_reduction_fraction": 1.0 - ratio,
        "optimized_speedup": 1.0 / ratio,
        "run_order": ["optimized", "legacy"],
    }
    output = RUNTIME / "analysis.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

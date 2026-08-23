#!/usr/bin/env python3
"""Summarize convergence speed and training memory for the clean A/B pair."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def latest(pattern: str) -> Path:
    matches = sorted(ROOT.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No run matches {pattern}")
    return matches[-1]


def load_json(path: Path):
    return json.loads(path.read_text())


def fixed_rf(path: Path) -> dict[int, float] | None:
    if not path.exists():
        return None
    result = load_json(path)
    return {
        int(row["epoch"]): float(row["mean_rf_loss"])
        for row in result["summary"].values()
    }


def summarize(label: str, run: Path, evaluation: Path) -> dict:
    history = load_json(run / "loss_history.json")
    steady = [float(row["epoch_seconds"]) for row in history if int(row["epoch"]) >= 2]
    cumulative = 0.0
    threshold_times = {}
    for row in history:
        cumulative += float(row["epoch_seconds"])
        for threshold in (1.0, 0.8, 0.7, 0.6, 0.55):
            key = f"train_loss_le_{threshold:.1f}"
            if key not in threshold_times and float(row["train_loss"]) <= threshold:
                threshold_times[key] = {
                    "epoch": int(row["epoch"]),
                    "wall_seconds": cumulative,
                }

    validations = {
        int(row["epoch"]): float(row["val_loss"])
        for row in history if row.get("val_loss") is not None
    }
    best_epoch, best_val = min(validations.items(), key=lambda item: item[1])

    diagnostic_rows = list(csv.DictReader((run / "data_path_diagnostics.csv").open()))
    peak_allocated = max(float(row["gpu_peak_allocated_mb"]) for row in diagnostic_rows)
    peak_reserved = max(float(row["gpu_peak_reserved_mb"]) for row in diagnostic_rows)
    mean_step = statistics.mean(
        float(row["total_training_step_ms"])
        for row in diagnostic_rows if float(row["backward_ms"]) > 0.0
    )

    return {
        "label": label,
        "run_dir": str(run.resolve()),
        "epochs_completed": len(history),
        "mean_epoch_seconds_2_to_end": statistics.mean(steady),
        "median_epoch_seconds_2_to_end": statistics.median(steady),
        "total_recorded_seconds": sum(float(row["epoch_seconds"]) for row in history),
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_validation_loss": float(history[-1]["val_loss"]),
        "best_validation_loss": best_val,
        "best_validation_epoch": best_epoch,
        "validation_by_epoch": validations,
        "fixed_manifest_rf_by_epoch": fixed_rf(evaluation),
        "time_to_training_threshold": threshold_times,
        "max_diagnostic_peak_allocated_mb": peak_allocated,
        "max_diagnostic_peak_reserved_mb": peak_reserved,
        "mean_diagnostic_training_step_ms": mean_step,
    }


def ratio(new: float, old: float) -> float:
    return new / old - 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--new-run", type=Path)
    args = parser.parse_args()

    baseline_run = args.baseline_run or latest("runs/F0_ENH_1K_B128_DemoN9510_*")
    new_run = args.new_run or latest("runs/CQ_LR_1K_B128_DemoN9511_*")
    baseline = summarize(
        "F0-ENH", baseline_run, ROOT / "evaluation/F0_ENH/milestones.json"
    )
    new = summarize(
        "CQ-LR", new_run, ROOT / "evaluation/CQ_LR/milestones.json"
    )
    comparison = {
        "protocol": {
            "seed": 42,
            "epochs": 1000,
            "scheduler_t_max": 1000,
            "batch_size": 128,
            "n_query_points": 4096,
            "query_microbatch": None,
            "only_model_difference": "GL_rbf_ENH versus GL_rbf_ENH_CQ/CQ-LR",
        },
        "models": {"F0-ENH": baseline, "CQ-LR": new},
        "cq_lr_relative_to_f0_enh": {
            "mean_epoch_time_fraction": ratio(
                new["mean_epoch_seconds_2_to_end"],
                baseline["mean_epoch_seconds_2_to_end"],
            ),
            "diagnostic_step_time_fraction": ratio(
                new["mean_diagnostic_training_step_ms"],
                baseline["mean_diagnostic_training_step_ms"],
            ),
            "peak_allocated_fraction": ratio(
                new["max_diagnostic_peak_allocated_mb"],
                baseline["max_diagnostic_peak_allocated_mb"],
            ),
            "peak_reserved_fraction": ratio(
                new["max_diagnostic_peak_reserved_mb"],
                baseline["max_diagnostic_peak_reserved_mb"],
            ),
            "final_validation_fraction": ratio(
                new["final_validation_loss"], baseline["final_validation_loss"]
            ),
            "best_validation_fraction": ratio(
                new["best_validation_loss"], baseline["best_validation_loss"]
            ),
        },
    }
    output = ROOT / "comparison.json"
    output.write_text(json.dumps(comparison, indent=2) + "\n")

    delta = comparison["cq_lr_relative_to_f0_enh"]
    results = f"""# Clean F0-ENH versus CQ-LR comparison

Both runs use the same extended batch-128 protocol and differ only in run identity and backbone.

| Metric | F0-ENH | CQ-LR | CQ-LR change |
|---|---:|---:|---:|
| Mean epoch time, epochs 2–1000 (s) | {baseline['mean_epoch_seconds_2_to_end']:.3f} | {new['mean_epoch_seconds_2_to_end']:.3f} | {delta['mean_epoch_time_fraction']:+.2%} |
| Diagnostic train step (ms) | {baseline['mean_diagnostic_training_step_ms']:.3f} | {new['mean_diagnostic_training_step_ms']:.3f} | {delta['diagnostic_step_time_fraction']:+.2%} |
| Peak allocated (MiB) | {baseline['max_diagnostic_peak_allocated_mb']:.1f} | {new['max_diagnostic_peak_allocated_mb']:.1f} | {delta['peak_allocated_fraction']:+.2%} |
| Peak reserved (MiB) | {baseline['max_diagnostic_peak_reserved_mb']:.1f} | {new['max_diagnostic_peak_reserved_mb']:.1f} | {delta['peak_reserved_fraction']:+.2%} |
| Final validation loss | {baseline['final_validation_loss']:.6f} | {new['final_validation_loss']:.6f} | {delta['final_validation_fraction']:+.2%} |
| Best validation loss | {baseline['best_validation_loss']:.6f} | {new['best_validation_loss']:.6f} | {delta['best_validation_fraction']:+.2%} |

Fixed-manifest results, milestone histories, and threshold times are stored in
`comparison.json`.
"""
    (ROOT / "RESULTS.md").write_text(results)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

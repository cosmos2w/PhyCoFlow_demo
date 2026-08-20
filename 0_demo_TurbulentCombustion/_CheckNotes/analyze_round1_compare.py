#!/usr/bin/env python3
"""Analyze the completed Round-1 legacy/optimized 100-epoch runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = ROOT / "_CheckNotes" / "Round1_runtime"
DEFAULT_RUN_ROOT = ROOT / "_CheckNotes" / "Round1_runs"
SAMPLES_PER_EPOCH = 9_000
STEPS_PER_EPOCH = 63


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_RUNTIME / "analysis.json")
    return parser.parse_args()


def latest_run(base: Path) -> Path:
    candidates = sorted(path.parent for path in base.glob("**/loss_history.csv"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one completed run under {base}, found {candidates}.")
    return candidates[0]


def read_history(run_dir: Path) -> list[dict[str, float]]:
    with open(run_dir / "loss_history.csv", newline="") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            row = {"epoch": int(raw["epoch"]), "train_loss": float(raw["train_loss"])}
            for key in ("val_loss", "train_seconds", "validation_seconds", "epoch_seconds"):
                row[key] = float(raw[key]) if raw.get(key) not in (None, "") else math.nan
            rows.append(row)
    if len(rows) != 100 or rows[-1]["epoch"] != 100:
        raise RuntimeError(f"Incomplete history in {run_dir}: {len(rows)} rows, last={rows[-1] if rows else None}.")
    return rows


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def first_loss_hit(rows: list[dict[str, float]], threshold: float) -> dict | None:
    cumulative = 0.0
    for row in rows:
        cumulative += row["train_seconds"]
        if row["train_loss"] <= threshold:
            return {
                "epoch": row["epoch"],
                "cumulative_train_seconds": cumulative,
                "loss": row["train_loss"],
            }
    return None


def summarize_history(rows: list[dict[str, float]]) -> dict:
    train_times = [row["train_seconds"] for row in rows]
    steady_times = train_times[5:]
    losses = [row["train_loss"] for row in rows]
    cumulative = 0.0
    checkpoints = {}
    for row in rows:
        cumulative += row["train_seconds"]
        if row["epoch"] in {1, 5, 10, 25, 50, 75, 100}:
            checkpoints[str(row["epoch"])] = {
                "train_loss": row["train_loss"],
                "cumulative_train_seconds": cumulative,
            }
    best_index = min(range(len(rows)), key=lambda index: losses[index])
    log_losses = [math.log(max(loss, 1e-30)) for loss in losses]
    log_auc = sum((log_losses[i - 1] + log_losses[i]) * 0.5 for i in range(1, len(log_losses)))
    relative_thresholds = {
        name: losses[0] * fraction
        for name, fraction in (("75pct_initial", 0.75), ("50pct_initial", 0.50), ("25pct_initial", 0.25))
    }
    return {
        "initial_train_loss": losses[0],
        "final_train_loss": losses[-1],
        "best_train_loss": losses[best_index],
        "best_train_loss_epoch": rows[best_index]["epoch"],
        "last_10_mean_train_loss": statistics.fmean(losses[-10:]),
        "loss_reduction_fraction": 1.0 - losses[-1] / losses[0],
        "log_loss_auc_100_epochs": log_auc,
        "total_train_seconds": sum(train_times),
        "total_validation_seconds": sum(
            row["validation_seconds"] for row in rows if math.isfinite(row["validation_seconds"])
        ),
        "mean_train_seconds_per_epoch": statistics.fmean(train_times),
        "steady_mean_train_seconds_per_epoch_epochs_6_100": statistics.fmean(steady_times),
        "steady_median_train_seconds_per_epoch_epochs_6_100": statistics.median(steady_times),
        "steady_p95_train_seconds_per_epoch_epochs_6_100": percentile(steady_times, 0.95),
        "mean_seconds_per_step": statistics.fmean(steady_times) / STEPS_PER_EPOCH,
        "mean_samples_per_second": SAMPLES_PER_EPOCH / statistics.fmean(steady_times),
        "checkpoints": checkpoints,
        "relative_loss_milestones": {
            name: first_loss_hit(rows, threshold) for name, threshold in relative_thresholds.items()
        },
        "validation": [
            {"epoch": row["epoch"], "loss": row["val_loss"]}
            for row in rows if math.isfinite(row["val_loss"])
        ],
    }


def parse_elapsed(value: str) -> float:
    pieces = [float(piece) for piece in value.strip().split(":")]
    if len(pieces) == 2:
        return pieces[0] * 60 + pieces[1]
    if len(pieces) == 3:
        return pieces[0] * 3600 + pieces[1] * 60 + pieces[2]
    raise ValueError(f"Unknown elapsed format: {value!r}")


def read_time_file(path: Path) -> dict:
    text = path.read_text()

    def capture(pattern: str) -> str:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is None:
            raise RuntimeError(f"Missing {pattern!r} in {path}.")
        return match.group(1).strip()

    return {
        "elapsed_seconds": parse_elapsed(capture(r"Elapsed \(wall clock\) time.*?:\s*([0-9:.]+)$")),
        "user_cpu_seconds": float(capture(r"User time \(seconds\):\s*([0-9.]+)$")),
        "system_cpu_seconds": float(capture(r"System time \(seconds\):\s*([0-9.]+)$")),
        "cpu_percent": float(capture(r"Percent of CPU this job got:\s*([0-9.]+)%$")),
        "max_host_rss_kb": int(capture(r"Maximum resident set size \(kbytes\):\s*([0-9]+)$")),
        "file_system_inputs": int(capture(r"File system inputs:\s*([0-9]+)$")),
        "file_system_outputs": int(capture(r"File system outputs:\s*([0-9]+)$")),
    }


def read_gpu_samples(path: Path) -> list[dict[str, float]]:
    rows = []
    with open(path, newline="") as handle:
        for raw in csv.reader(handle):
            if len(raw) < 5:
                continue
            try:
                row = {
                    "utilization_pct": float(raw[2].strip()),
                    "memory_mib": float(raw[3].strip()),
                    "power_w": float(raw[4].strip()),
                }
            except ValueError:
                continue
            rows.append(row)
    return rows


def summarize_gpu(samples: list[dict[str, float]], elapsed_seconds: float) -> dict:
    # The monitor interval is two seconds and starts immediately before this job.
    sample_count = min(len(samples), math.ceil(elapsed_seconds / 2.0) + 1)
    rows = samples[:sample_count]
    if not rows:
        raise RuntimeError("No GPU telemetry rows found.")
    util = [row["utilization_pct"] for row in rows]
    memory = [row["memory_mib"] for row in rows]
    power = [row["power_w"] for row in rows]
    return {
        "samples": len(rows),
        "mean_utilization_pct": statistics.fmean(util),
        "median_utilization_pct": statistics.median(util),
        "p95_utilization_pct": percentile(util, 0.95),
        "idle_sample_fraction_util_below_10pct": sum(value < 10 for value in util) / len(util),
        "minimum_memory_mib": min(memory),
        "peak_memory_mib": max(memory),
        "memory_delta_mib": max(memory) - min(memory),
        "mean_power_w": statistics.fmean(power),
        "estimated_energy_wh": sum(power) * 2.0 / 3600.0,
    }


def main() -> None:
    args = parse_args()
    exit_status = dict(
        line.split("=", 1) for line in (args.runtime_dir / "exit_status.txt").read_text().splitlines()
    )
    if exit_status != {"legacy_exit": "0", "optimized_exit": "0"}:
        raise RuntimeError(f"Runs did not both succeed: {exit_status}.")

    run_dirs = {
        "legacy": latest_run(args.run_root / "legacy"),
        "optimized": latest_run(args.run_root / "optimized"),
    }
    time_data = {
        name: read_time_file(args.runtime_dir / f"{name}_time.txt") for name in run_dirs
    }
    gpu_rows = {
        name: read_gpu_samples(args.runtime_dir / f"{name}_gpu_samples.csv")
        for name in run_dirs
    }
    result = {
        "runs": {
            name: {
                "run_dir": str(run_dir),
                "history": summarize_history(read_history(run_dir)),
                "external_time": time_data[name],
                "gpu": summarize_gpu(gpu_rows[name], time_data[name]["elapsed_seconds"]),
            }
            for name, run_dir in run_dirs.items()
        }
    }
    legacy = result["runs"]["legacy"]
    optimized = result["runs"]["optimized"]
    result["optimized_over_legacy"] = {
        "external_wall_time_ratio": (
            optimized["external_time"]["elapsed_seconds"] / legacy["external_time"]["elapsed_seconds"]
        ),
        "steady_epoch_time_ratio": (
            optimized["history"]["steady_mean_train_seconds_per_epoch_epochs_6_100"]
            / legacy["history"]["steady_mean_train_seconds_per_epoch_epochs_6_100"]
        ),
        "final_train_loss_ratio": (
            optimized["history"]["final_train_loss"] / legacy["history"]["final_train_loss"]
        ),
        "last_10_mean_train_loss_ratio": (
            optimized["history"]["last_10_mean_train_loss"]
            / legacy["history"]["last_10_mean_train_loss"]
        ),
        "log_loss_auc_ratio": (
            optimized["history"]["log_loss_auc_100_epochs"]
            / legacy["history"]["log_loss_auc_100_epochs"]
        ),
        "peak_gpu_memory_ratio": (
            optimized["gpu"]["peak_memory_mib"] / legacy["gpu"]["peak_memory_mib"]
        ),
        "estimated_energy_ratio": (
            optimized["gpu"]["estimated_energy_wh"] / legacy["gpu"]["estimated_energy_wh"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

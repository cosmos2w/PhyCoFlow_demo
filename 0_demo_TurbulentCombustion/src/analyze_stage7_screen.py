#!/usr/bin/env python3
"""Joint epoch-200 quality/efficiency decision for Stage-7 Smart CQ."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entry", action="append", nargs=5, required=True,
        metavar=("LABEL", "FIXED_JSON", "RECON_JSON", "RUN_DIR", "BENCHMARK_LABEL"),
        help="Use '-' for an unavailable reconstruction JSON or run directory.",
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--f0-label", default="F0")
    parser.add_argument("--cq-label", default="CQ-LR-128")
    parser.add_argument("--candidate-labels", nargs="+", default=["S7-A", "S7-B"])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def fixed_epoch(path: str | Path, epoch: int) -> dict[str, Any]:
    payload = load_json(path)
    rows = [row for row in payload["summary"].values() if int(row.get("epoch", -1)) == epoch]
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one epoch {epoch} row, found {len(rows)}")
    return rows[0]


def reconstruction(path: str) -> dict[str, Any] | None:
    if path == "-":
        return None
    payload = load_json(path)
    metrics = payload.get("metrics", payload)
    fields = {
        key: float(value) for key, value in metrics.items()
        if not key.startswith("obs_") and isinstance(value, (int, float)) and math.isfinite(float(value))
    }
    if not fields:
        raise ValueError(f"{path}: no finite field metrics found")
    worst_field = max(fields, key=fields.get)
    return {
        "path": str(Path(path).resolve()),
        "fields": fields,
        "mean_field_rel_l2": sum(fields.values()) / len(fields),
        "worst_field": worst_field,
        "worst_field_rel_l2": fields[worst_field],
    }


def training_row(run_dir: str, epoch: int) -> dict[str, Any] | None:
    if run_dir == "-":
        return None
    path = Path(run_dir) / "loss_history.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(row["epoch"]) == epoch]
    if len(matches) != 1:
        raise ValueError(f"{path}: expected one epoch {epoch} row, found {len(matches)}")
    row = matches[0]
    result: dict[str, Any] = {"path": str(path.resolve()), "epoch": epoch}
    for key, value in row.items():
        if key == "epoch" or value in (None, ""):
            continue
        result[key] = float(value)
    return result


def benchmark_maps(payload: dict[str, Any]):
    formal = {row["label"]: row for row in payload.get("formal_training_step", [])}
    persistent = {row["label"]: row for row in payload.get("persistent_inference", [])}
    summaries = {row["label"]: row for row in payload.get("model_summaries", [])}
    return formal, persistent, summaries


def ratio_gap(value: float, reference: float) -> float:
    return value / reference - 1.0


def main() -> None:
    args = parse_args()
    benchmark = load_json(args.benchmark)
    formal, persistent, summaries = benchmark_maps(benchmark)
    rows: dict[str, dict[str, Any]] = {}
    for label, fixed_path, recon_path, run_dir, benchmark_label in args.entry:
        fixed = fixed_epoch(fixed_path, args.epoch)
        rows[label] = {
            "label": label,
            "benchmark_label": benchmark_label,
            "fixed_manifest": {
                "path": str(Path(fixed_path).resolve()),
                "epoch": int(fixed["epoch"]),
                "stored_val_loss": fixed.get("stored_val_loss"),
                "mean_rf_loss": float(fixed["mean_rf_loss"]),
                "std_rf_loss": float(fixed["std_rf_loss"]),
                "evaluations": int(fixed["evaluations"]),
            },
            "reconstruction": reconstruction(recon_path),
            "training": training_row(run_dir, args.epoch),
            "formal_cost": formal.get(benchmark_label),
            "persistent_cost": persistent.get(benchmark_label),
            "model_summary": summaries.get(benchmark_label),
        }

    if args.f0_label not in rows or args.cq_label not in rows:
        raise ValueError("Both F0 and CQ-LR-128 entries are required.")
    f0 = rows[args.f0_label]
    cq = rows[args.cq_label]
    f0_rf = f0["fixed_manifest"]["mean_rf_loss"]
    cq_rf = cq["fixed_manifest"]["mean_rf_loss"]
    quality_gap = cq_rf - f0_rf
    for row in rows.values():
        rf = row["fixed_manifest"]["mean_rf_loss"]
        row["fixed_rf_gap_vs_f0_fraction"] = ratio_gap(rf, f0_rf)
        row["cq_to_f0_fixed_gap_recovery_fraction"] = (
            (cq_rf - rf) / quality_gap if quality_gap > 0 else None
        )
        if row["reconstruction"] and f0["reconstruction"]:
            row["reconstruction_mean_gap_vs_f0_fraction"] = ratio_gap(
                row["reconstruction"]["mean_field_rel_l2"],
                f0["reconstruction"]["mean_field_rel_l2"],
            )

    f0_formal = f0["formal_cost"]
    f0_persistent = f0["persistent_cost"]
    for row in rows.values():
        cost = row["formal_cost"]
        persistent_cost = row["persistent_cost"]
        row["train_speedup_vs_f0"] = (
            f0_formal["full_step_ms"] / cost["full_step_ms"]
            if f0_formal and cost and cost.get("status") == "ok" else None
        )
        row["train_memory_reduction_vs_f0_fraction"] = (
            1.0 - cost["peak_allocated_mb"] / f0_formal["peak_allocated_mb"]
            if f0_formal and cost and cost.get("status") == "ok" else None
        )
        row["persistent_nfe4_speedup_vs_f0"] = (
            f0_persistent["steady_nfe4_s"] / persistent_cost["steady_nfe4_s"]
            if f0_persistent and persistent_cost and persistent_cost.get("status") == "ok" else None
        )

    candidates = [rows[label] for label in args.candidate_labels if label in rows]
    eligible = [
        row for row in candidates
        if row["fixed_manifest"]["mean_rf_loss"] < cq_rf
        and (row["train_speedup_vs_f0"] or 0.0) >= 1.10
        and (row["persistent_nfe4_speedup_vs_f0"] or 0.0) >= 1.15
    ]
    if not eligible:
        decision = {
            "continue_label": None,
            "recommendation": "Stop Stage-7 long training: no candidate improves CQ-LR-128 while passing both efficiency gates.",
        }
    else:
        eligible.sort(key=lambda row: row["fixed_manifest"]["mean_rf_loss"])
        winner = eligible[0]
        if len(eligible) > 1:
            a, b = eligible[:2]
            relative_tie = abs(a["fixed_manifest"]["mean_rf_loss"] - b["fixed_manifest"]["mean_rf_loss"]) / min(
                a["fixed_manifest"]["mean_rf_loss"], b["fixed_manifest"]["mean_rf_loss"]
            )
            if relative_tie <= 0.01:
                winner = max(eligible[:2], key=lambda row: row["train_speedup_vs_f0"] or 0.0)
        decision = {
            "continue_label": winner["label"],
            "recommendation": f"Continue only {winner['label']} to epoch 1000; it has the best eligible quality/efficiency tradeoff.",
        }

    result = {
        "epoch": args.epoch,
        "benchmark": str(args.benchmark.resolve()),
        "models": rows,
        "decision": decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")

    lines = [
        f"# Stage-7 epoch-{args.epoch} joint decision",
        "",
        "| Model | Fixed RF | Gap vs F0 | Gap recovery | Train speedup | Memory reduction | 1M NFE4 speedup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows.values():
        def pct(value):
            return "n/a" if value is None else f"{100.0 * value:.2f}%"
        def mult(value):
            return "n/a" if value is None else f"{value:.3f}x"
        lines.append(
            f"| {row['label']} | {row['fixed_manifest']['mean_rf_loss']:.6f} | "
            f"{pct(row['fixed_rf_gap_vs_f0_fraction'])} | "
            f"{pct(row['cq_to_f0_fixed_gap_recovery_fraction'])} | "
            f"{mult(row['train_speedup_vs_f0'])} | "
            f"{pct(row['train_memory_reduction_vs_f0_fraction'])} | "
            f"{mult(row['persistent_nfe4_speedup_vs_f0'])} |"
        )
    lines.extend(["", f"**Decision:** {decision['recommendation']}", ""])
    args.output.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

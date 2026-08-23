#!/usr/bin/env python3
"""Build the final controlled Stage-7 quality/throughput evidence tables."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[3]
REFERENCE = Path("/home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion")
STAGE7 = PROJECT / "_CheckNotes/Stage7_smart_cq"
OUTPUT = STAGE7 / "evaluation_1000"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summary_epoch(payload: dict[str, Any], epoch: int) -> dict[str, Any]:
    rows = [row for row in payload["summary"].values() if int(row["epoch"]) == epoch]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one epoch-{epoch} summary, got {len(rows)}")
    return rows[0]


def fixed_rows(path: Path, suffix: str | None = None) -> dict[tuple[int, int, int, int], float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if suffix is not None:
        rows = [row for row in rows if row["checkpoint"].endswith(suffix)]
    labels = {row["checkpoint"] for row in rows}
    if not rows or len(labels) != 1:
        raise RuntimeError(f"Expected one checkpoint row set in {path} ({suffix}); labels={labels}")
    return {
        (
            int(row["repeat"]), int(row["batch_index"]),
            int(row["manifest_start_index"]), int(row["rf_seed"]),
        ): float(row["loss"])
        for row in rows
    }


def paired(candidate: dict, reference: dict) -> dict[str, float | int]:
    if candidate.keys() != reference.keys():
        raise RuntimeError("Paired evaluations do not share identical layout/RNG keys")
    values = [candidate[key] - reference[key] for key in sorted(reference)]
    mean = statistics.fmean(values)
    std = statistics.stdev(values)
    sem = std / math.sqrt(len(values))
    return {
        "paired_n": len(values),
        "paired_difference_vs_f0_e1000_mean": mean,
        "paired_difference_vs_f0_e1000_std": std,
        "paired_difference_vs_f0_e1000_ci95_low": mean - 1.96 * sem,
        "paired_difference_vs_f0_e1000_ci95_high": mean + 1.96 * sem,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    f0_milestones_path = REFERENCE / "_CheckNotes/Stage6_clean_ab/evaluation/F0_ENH/milestones.json"
    cq_milestones_path = REFERENCE / "_CheckNotes/Stage6_clean_ab/evaluation/CQ_LR/milestones.json"
    f0_milestones = load_json(f0_milestones_path)
    cq_milestones = load_json(cq_milestones_path)
    s7_milestones = load_json(OUTPUT / "S7_B_milestones_and_best.json")
    f0_best = load_json(OUTPUT / "F0_best_fixed_manifest.json")
    cq_best = load_json(OUTPUT / "CQ_LR_128_best_fixed_manifest.json")
    cq256_best = load_json(OUTPUT / "CQ_LR_256_best_fixed_manifest.json")
    reconstruction = load_json(OUTPUT / "matched_reconstruction/summary.json")
    benchmark = load_json(STAGE7 / "benchmarks/pretraining_cost.json")

    f0_reference_rows = fixed_rows(f0_milestones_path.with_suffix(".csv"), "epoch_1000.pt")
    row_sources = {
        "F0-e1000": (summary_epoch(f0_milestones, 1000), f0_reference_rows),
        "CQ-LR-128-e1000": (
            summary_epoch(cq_milestones, 1000),
            fixed_rows(cq_milestones_path.with_suffix(".csv"), "epoch_1000.pt"),
        ),
        "CQ-LR-256-best-e840-partial": (
            summary_epoch(cq256_best, 840),
            fixed_rows(OUTPUT / "CQ_LR_256_best_fixed_manifest.csv"),
        ),
        "S7-B-e1000": (
            summary_epoch(s7_milestones, 1000),
            fixed_rows(OUTPUT / "S7_B_milestones_and_best.csv", "epoch_1000.pt"),
        ),
        "S7-B-best-e965": (
            summary_epoch(s7_milestones, 965),
            fixed_rows(OUTPUT / "S7_B_milestones_and_best.csv", "best.pt"),
        ),
    }
    recon_label = {
        "F0-e1000": "F0-best",
        "CQ-LR-128-e1000": "CQ-LR-128-best",
        "CQ-LR-256-best-e840-partial": "CQ-LR-256-best-e840",
        "S7-B-e1000": "S7-B-e1000",
        "S7-B-best-e965": "S7-B-best-e965",
    }
    cost_label = {
        "F0-e1000": "F0",
        "CQ-LR-128-e1000": "Frozen-CQ-LR-128",
        "S7-B-e1000": "Stage7-All256",
        "S7-B-best-e965": "Stage7-All256",
    }
    formal_cost = {row["label"]: row for row in benchmark["formal_training_step"]}
    persistent = {row["label"]: row for row in benchmark["persistent_inference"]}
    model_summary = {row["label"]: row for row in benchmark["model_summaries"]}
    f0_cost = formal_cost["F0"]
    f0_persistent = persistent["F0"]
    f0_rf = float(row_sources["F0-e1000"][0]["mean_rf_loss"])

    rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    for label, (quality, samples) in row_sources.items():
        pair = paired(samples, f0_reference_rows)
        paired_rows.append({"candidate": label, **pair})
        n = int(quality["evaluations"])
        rf_sem = float(quality["std_rf_loss"]) / math.sqrt(n)
        recon = reconstruction["candidates"][recon_label[label]]
        row: dict[str, Any] = {
            "candidate": label,
            "epoch": int(quality["epoch"]),
            "checkpoint_scope": "partial_run_best" if "partial" in label else (
                "best" if "best-e965" in label else "exact_milestone"
            ),
            "fixed_manifest_rf_mean": float(quality["mean_rf_loss"]),
            "fixed_manifest_rf_std": float(quality["std_rf_loss"]),
            "fixed_manifest_rf_ci95_low": float(quality["mean_rf_loss"]) - 1.96 * rf_sem,
            "fixed_manifest_rf_ci95_high": float(quality["mean_rf_loss"]) + 1.96 * rf_sem,
            "rf_relative_change_vs_f0_e1000": (
                float(quality["mean_rf_loss"]) / f0_rf - 1.0
            ),
            **pair,
            "reconstruction_checkpoint": recon_label[label],
            "recon_nfe1_mean": recon["nfe"]["1"]["mean_field_relative_l2"],
            "recon_nfe1_worst_field": recon["nfe"]["1"]["worst_field"],
            "recon_nfe1_worst": recon["nfe"]["1"]["worst_field_relative_l2"],
            "recon_nfe4_mean": recon["nfe"]["4"]["mean_field_relative_l2"],
            "recon_nfe4_worst_field": recon["nfe"]["4"]["worst_field"],
            "recon_nfe4_worst": recon["nfe"]["4"]["worst_field_relative_l2"],
        }
        if label in cost_label:
            key = cost_label[label]
            cost = formal_cost[key]
            inference = persistent[key]
            parameters = model_summary[key]
            row.update(
                train_step_ms=cost["full_step_ms"],
                train_peak_allocated_mb=cost["peak_allocated_mb"],
                train_speedup_vs_f0=f0_cost["full_step_ms"] / cost["full_step_ms"],
                train_memory_reduction_vs_f0=(
                    1.0 - cost["peak_allocated_mb"] / f0_cost["peak_allocated_mb"]
                ),
                persistent_1m_nfe4_s=inference["steady_nfe4_s"],
                persistent_1m_nfe4_speedup_vs_f0=(
                    f0_persistent["steady_nfe4_s"] / inference["steady_nfe4_s"]
                ),
                total_parameters=parameters["total_parameters"],
                cost_source="formal_architecture_benchmark_B128_Q4096",
            )
        else:
            row["cost_source"] = "not_formally_matched_available_partial_reference"
        rows.append(row)

    convergence_sources = {
        "F0": [f0_milestones],
        "CQ-LR-128": [cq_milestones],
        "S7-B": [
            load_json(STAGE7 / "screen_200/evaluation/S7_B_fixed_manifest.json"),
            s7_milestones,
        ],
        "CQ-LR-256 (partial)": [
            load_json(STAGE7 / "screen_200/evaluation/CQ_LR_L256_fixed_manifest.json"),
            cq256_best,
        ],
    }
    convergence_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for label, payloads in convergence_sources.items():
        for payload in payloads:
            for entry in payload["summary"].values():
                convergence_by_key[(label, int(entry["epoch"]))] = {
                    "candidate": label,
                    "epoch": int(entry["epoch"]),
                    "mean_rf_loss": float(entry["mean_rf_loss"]),
                    "std_rf_loss": float(entry["std_rf_loss"]),
                }
    convergence = [convergence_by_key[key] for key in sorted(convergence_by_key)]
    s7_crossing = min(
        row["epoch"] for row in convergence
        if row["candidate"] == "S7-B" and row["mean_rf_loss"] <= f0_rf
    )

    write_csv(OUTPUT / "final_comparison.csv", rows)
    write_csv(OUTPUT / "paired_statistics.csv", paired_rows)
    write_csv(OUTPUT / "convergence.csv", convergence)
    result = {
        "recommendation": "Stage7-All256",
        "recommended_checkpoint": "S7-B epoch_1000.pt (EMA trainable weights; exact live frozen state)",
        "decision_basis": (
            "At the exact 1000-epoch milestone S7-B improves controlled RF loss over "
            "F0 and CQ-LR-128 while retaining the formal Stage-7 efficiency gates."
        ),
        "first_measured_epoch_beating_f0_e1000_rf": s7_crossing,
        "fixed_manifest_protocol": {
            key: s7_milestones[key]
            for key in (
                "manifest_checksum_sha256", "materialized_input_checksum_sha256",
                "batch_size", "repeats", "rf_seed",
            )
        },
        "reconstruction_protocol": reconstruction["protocol"],
        "comparison": rows,
    }
    (OUTPUT / "final_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

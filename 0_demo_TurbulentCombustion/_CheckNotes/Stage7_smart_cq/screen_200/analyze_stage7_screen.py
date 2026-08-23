#!/usr/bin/env python3
"""Consolidate the controlled Stage-7 epoch-200 comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/home/wanglz/Desktop/src/PhyCoFlow/0_demo_TurbulentCombustion"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def epoch_entry(payload: dict[str, Any], epoch: int) -> dict[str, Any]:
    matches = [row for row in payload["summary"].values() if int(row["epoch"]) == epoch]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one epoch-{epoch} row, found {len(matches)}")
    return matches[0]


def epoch_rows(path: Path, epoch: int = 200) -> dict[tuple[int, int, int, int], float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = {row["checkpoint"] for row in rows}
    if len(labels) == 1:
        selected = rows
    else:
        suffix = f"epoch_{epoch:04d}.pt"
        selected = [row for row in rows if row["checkpoint"].endswith(suffix)]
    if not selected:
        raise RuntimeError(f"No epoch-{epoch} rows in {path}")
    return {
        (
            int(row["repeat"]),
            int(row["batch_index"]),
            int(row["manifest_start_index"]),
            int(row["rf_seed"]),
        ): float(row["loss"])
        for row in selected
    }


def paired_stats(candidate: dict, reference: dict) -> dict[str, float | int]:
    if candidate.keys() != reference.keys():
        raise RuntimeError("Paired RF rows do not share identical RNG/layout keys")
    differences = [candidate[key] - reference[key] for key in sorted(reference)]
    mean = statistics.fmean(differences)
    std = statistics.stdev(differences)
    sem = std / math.sqrt(len(differences))
    return {
        "paired_n": len(differences),
        "paired_difference_vs_f0_mean": mean,
        "paired_difference_vs_f0_std": std,
        "paired_difference_vs_f0_ci95_low": mean - 1.96 * sem,
        "paired_difference_vs_f0_ci95_high": mean + 1.96 * sem,
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    reference_root = args.reference_root.resolve()
    screen = root / "_CheckNotes/Stage7_smart_cq/screen_200"
    evaluation = screen / "evaluation"

    fixed_paths = {
        "F0-128": reference_root / "_CheckNotes/Stage6_clean_ab/evaluation/F0_ENH/milestones.json",
        "CQ-LR-128": reference_root / "_CheckNotes/Stage6_clean_ab/evaluation/CQ_LR/milestones.json",
        "CQ-LR-256": evaluation / "CQ_LR_L256_fixed_manifest.json",
        "S7-A": evaluation / "S7_A_fixed_manifest.json",
        "S7-B": evaluation / "S7_B_fixed_manifest.json",
    }
    fixed_csv = {label: path.with_suffix(".csv") for label, path in fixed_paths.items()}
    fixed = {label: load_json(path) for label, path in fixed_paths.items()}
    reference_rows = epoch_rows(fixed_csv["F0-128"])
    paired = {
        label: paired_stats(epoch_rows(path), reference_rows)
        for label, path in fixed_csv.items()
    }

    reconstruction = load_json(evaluation / "matched_reconstruction/summary.json")
    benchmark = load_json(root / "_CheckNotes/Stage7_smart_cq/benchmarks/pretraining_cost.json")
    benchmark_label = {
        "F0-128": "F0",
        "CQ-LR-128": "Frozen-CQ-LR-128",
        "S7-A": "Stage7-Cond128",
        "S7-B": "Stage7-All256",
    }
    formal_step = {row["label"]: row for row in benchmark["formal_training_step"]}
    persistent = {row["label"]: row for row in benchmark["persistent_inference"]}
    f0_formal = formal_step["F0"]

    diagnostic_paths = {
        "F0-128": reference_root / "_CheckNotes/Stage6_clean_ab/runs/F0_ENH_1K_B128_DemoN9510_20260821_235104/data_path_diagnostics_summary.json",
        "CQ-LR-256": reference_root / "_CheckNotes/Stage6_clean_ab/runs/CQ_LR_L256_1K_B128_DemoN9561_20260822_144624/data_path_diagnostics_summary.json",
    }
    diagnostics = {label: load_json(path)["cumulative"]["mean"] for label, path in diagnostic_paths.items()}

    table = []
    for label in fixed_paths:
        rf = epoch_entry(fixed[label], 200)
        recon = reconstruction["candidates"][label]
        row: dict[str, Any] = {
            "candidate": label,
            "epoch": 200,
            "fixed_manifest_rf_mean": rf["mean_rf_loss"],
            "fixed_manifest_rf_std": rf["std_rf_loss"],
            **paired[label],
            "paired_relative_rf_vs_f0": paired[label]["paired_difference_vs_f0_mean"]
            / epoch_entry(fixed["F0-128"], 200)["mean_rf_loss"],
            "recon_nfe1_mean": recon["nfe"]["1"]["mean_field_relative_l2"],
            "recon_nfe1_worst_field": recon["nfe"]["1"]["worst_field"],
            "recon_nfe1_worst": recon["nfe"]["1"]["worst_field_relative_l2"],
            "recon_nfe4_mean": recon["nfe"]["4"]["mean_field_relative_l2"],
            "recon_nfe4_worst_field": recon["nfe"]["4"]["worst_field"],
            "recon_nfe4_worst": recon["nfe"]["4"]["worst_field_relative_l2"],
        }
        if label in benchmark_label:
            key = benchmark_label[label]
            cost = formal_step[key]
            inference = persistent[key]
            row.update(
                train_step_ms=cost["full_step_ms"],
                train_peak_allocated_mb=cost["peak_allocated_mb"],
                train_speedup_vs_f0=f0_formal["full_step_ms"] / cost["full_step_ms"],
                train_memory_reduction_vs_f0=1.0
                - cost["peak_allocated_mb"] / f0_formal["peak_allocated_mb"],
                persistent_1m_nfe4_s=inference["steady_nfe4_s"],
                persistent_1m_nfe4_speedup_vs_f0=persistent["F0"]["steady_nfe4_s"]
                / inference["steady_nfe4_s"],
                cost_source="formal_stage7_benchmark",
            )
        else:
            cost = diagnostics[label]
            f0_diag = diagnostics["F0-128"]
            row.update(
                train_step_ms=cost["total_training_step_ms"],
                train_peak_allocated_mb=cost["gpu_peak_allocated_mb"],
                train_speedup_vs_f0=f0_diag["total_training_step_ms"]
                / cost["total_training_step_ms"],
                train_memory_reduction_vs_f0=1.0
                - cost["gpu_peak_allocated_mb"] / f0_diag["gpu_peak_allocated_mb"],
                persistent_1m_nfe4_s=None,
                persistent_1m_nfe4_speedup_vs_f0=None,
                cost_source="cumulative_training_diagnostics_unmatched",
            )
        table.append(row)

    with (evaluation / "comparison_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    convergence = []
    for label, payload in fixed.items():
        for entry in payload["summary"].values():
            convergence.append(
                {
                    "candidate": label,
                    "epoch": int(entry["epoch"]),
                    "mean_rf_loss": float(entry["mean_rf_loss"]),
                    "std_rf_loss": float(entry["std_rf_loss"]),
                }
            )
    with (evaluation / "convergence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(convergence[0]))
        writer.writeheader()
        writer.writerows(convergence)

    result = {
        "decision": "continue_S7-B_only",
        "decision_basis": (
            "S7-B has the lowest epoch-200 fixed-manifest RF loss, the strongest "
            "Stage-7 matched reconstruction, and passes all formal efficiency gates."
        ),
        "fixed_manifest_protocol": {
            key: fixed["F0-128"][key]
            for key in (
                "manifest_checksum_sha256",
                "materialized_input_checksum_sha256",
                "batch_size",
                "repeats",
                "rf_seed",
            )
        },
        "reconstruction_protocol": reconstruction["protocol"],
        "rows": table,
    }
    (evaluation / "comparison_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

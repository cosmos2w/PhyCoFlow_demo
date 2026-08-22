#!/usr/bin/env python3
"""Produce the final quality/efficiency analysis for the pinned CQ cache A/B."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def ratio(new: float, old: float) -> float:
    return new / old - 1.0


def fixed_by_epoch(path: Path) -> dict[int, float]:
    data = load_json(path)
    return {
        int(row["epoch"]): float(row["mean_rf_loss"])
        for row in data["summary"].values()
    }


def summarize_run(label: str, run: Path, evaluation: Path) -> dict[str, Any]:
    history = load_json(run / "loss_history.json")
    steady = [float(row["epoch_seconds"]) for row in history if int(row["epoch"]) >= 2]
    validations = {
        int(row["epoch"]): float(row["val_loss"])
        for row in history if row.get("val_loss") is not None
    }
    best_epoch, best_val = min(validations.items(), key=lambda item: item[1])
    diagnostic_path = run / "data_path_diagnostics.csv"
    diagnostic_rows = list(csv.DictReader(diagnostic_path.open()))
    measured_steps = [
        row for row in diagnostic_rows if float(row.get("backward_ms", 0.0)) > 0.0
    ]
    return {
        "label": label,
        "run_dir": str(run.resolve()),
        "epochs_completed": len(history),
        "mean_epoch_seconds_2_to_200": statistics.fmean(steady),
        "median_epoch_seconds_2_to_200": statistics.median(steady),
        "total_recorded_seconds": sum(float(row["epoch_seconds"]) for row in history),
        "final_train_loss": float(history[-1]["train_loss"]),
        "final_validation_loss": float(history[-1]["val_loss"]),
        "best_validation_loss": float(best_val),
        "best_validation_epoch": int(best_epoch),
        "validation_by_epoch": validations,
        "fixed_manifest_rf_by_epoch": fixed_by_epoch(evaluation),
        "max_diagnostic_peak_allocated_mb": max(
            float(row["gpu_peak_allocated_mb"]) for row in diagnostic_rows
        ),
        "max_diagnostic_peak_reserved_mb": max(
            float(row["gpu_peak_reserved_mb"]) for row in diagnostic_rows
        ),
        "mean_diagnostic_training_step_ms": statistics.fmean(
            float(row["total_training_step_ms"]) for row in measured_steps
        ),
    }


def model_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    return {key: value for key, value in state.items() if torch.is_tensor(value)}


def state_comparison(old_path: Path, new_path: Path) -> dict[str, Any]:
    old = model_state(old_path)
    new = model_state(new_path)
    if old.keys() != new.keys():
        raise RuntimeError("The paired checkpoints have different model-state keys.")

    max_abs = 0.0
    mean_abs_numerator = 0.0
    elements = 0
    old_digest = hashlib.sha256()
    new_digest = hashlib.sha256()
    for key in old:
        a = old[key].detach().cpu().contiguous()
        b = new[key].detach().cpu().contiguous()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise RuntimeError(f"Checkpoint tensor mismatch: {key}")
        for digest, tensor in ((old_digest, a), (new_digest, b)):
            digest.update(key.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        if a.is_floating_point() or a.is_complex():
            delta = (a - b).abs().double()
            max_abs = max(max_abs, float(delta.max()) if delta.numel() else 0.0)
            mean_abs_numerator += float(delta.sum())
            elements += delta.numel()
        elif not torch.equal(a, b):
            max_abs = math.inf
    return {
        "old_sha256": old_digest.hexdigest(),
        "new_sha256": new_digest.hexdigest(),
        "exact_match": old_digest.digest() == new_digest.digest(),
        "max_abs_parameter_difference": max_abs,
        "mean_abs_parameter_difference": mean_abs_numerator / max(elements, 1),
        "floating_elements_compared": elements,
    }


def indexed_rows(path: Path) -> dict[tuple[int, int, str], dict[str, Any]]:
    data = load_json(path)
    return {
        (int(row["N_query"]), int(row["NFE"]), str(row["mode"])): row
        for row in data["rows"]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--persistent-run", type=Path, required=True)
    args = parser.parse_args()

    old = summarize_run(
        "CQ-LR without persistent Top-K",
        args.baseline_run,
        ROOT / "evaluation/no_persistent/milestones.json",
    )
    new = summarize_run(
        "CQ-LR with persistent Top-K",
        args.persistent_run,
        ROOT / "evaluation/persistent_topk/milestones.json",
    )
    weights = state_comparison(
        args.baseline_run / "epoch_0200.pt",
        args.persistent_run / "epoch_0200.pt",
    )

    old_bench = indexed_rows(ROOT / "benchmarks/no_persistent_checkpoint.json")
    new_bench = indexed_rows(ROOT / "benchmarks/persistent_topk_checkpoint.json")
    old_focus = old_bench[(1_000_000, 4, "static_per_call")]
    new_focus = new_bench[(1_000_000, 4, "static_persistent_geometry")]
    new_none = new_bench[(1_000_000, 4, "none")]
    steady_speedup = float(old_focus["mean_wall_s"]) / float(new_focus["mean_wall_s"])
    speedup_vs_none = float(new_none["mean_wall_s"]) / float(new_focus["mean_wall_s"])

    old_rf = old["fixed_manifest_rf_by_epoch"][200]
    new_rf = new["fixed_manifest_rf_by_epoch"][200]
    max_output_diff = max(
        float(row["max_abs_diff_vs_geometry_per_call"] or 0.0)
        for row in new_bench.values()
        if row["mode"] == "static_persistent_geometry"
    )
    checks = {
        "completed_200_epochs": old["epochs_completed"] == new["epochs_completed"] == 200,
        "fixed_manifest_rf_within_0_5_percent": abs(ratio(new_rf, old_rf)) <= 0.005,
        "final_validation_within_0_5_percent": abs(
            ratio(new["final_validation_loss"], old["final_validation_loss"])
        ) <= 0.005,
        "persistent_output_max_abs_le_1e_5": max_output_diff <= 1.0e-5,
        "persistent_zero_knn_calls": all(
            float(row["mean_topk_calls"]) == 0.0
            for row in new_bench.values()
            if "persistent" in row["mode"]
        ),
        "persistent_at_least_15_percent_faster_than_per_call_static": steady_speedup >= 1.15,
    }
    checks["all_acceptance_checks_pass"] = all(checks.values())

    comparison = {
        "protocol": {
            "old_commit": "01d284767af9cbbf6b2e185b2ea52c50545ca607",
            "persistent_commit": "3f3eefbe5ddeb2d530318bf7686d03b61c051ff4",
            "training_driver_and_data_helpers": "byte-identical across commits",
            "seed": 42,
            "epochs": 200,
            "scheduler_t_max": 200,
            "batch_size": 128,
            "n_query_points": 4096,
            "query_microbatch": None,
            "cache_scope": "post-training cached-streamed reconstruction only",
        },
        "runs": {"no_persistent": old, "persistent_topk": new},
        "checkpoint_state_comparison": weights,
        "training_relative_change": {
            "mean_epoch_time": ratio(
                new["mean_epoch_seconds_2_to_200"], old["mean_epoch_seconds_2_to_200"]
            ),
            "diagnostic_step_time": ratio(
                new["mean_diagnostic_training_step_ms"], old["mean_diagnostic_training_step_ms"]
            ),
            "peak_allocated": ratio(
                new["max_diagnostic_peak_allocated_mb"], old["max_diagnostic_peak_allocated_mb"]
            ),
            "final_validation": ratio(new["final_validation_loss"], old["final_validation_loss"]),
            "fixed_manifest_rf_epoch_200": ratio(new_rf, old_rf),
        },
        "reconstruction_focus_1m_nfe4": {
            "baseline_mode": "static_features rebuilt per sample() call",
            "persistent_mode": "persistent geometry + per-call static_features",
            "baseline_steady_seconds": float(old_focus["mean_wall_s"]),
            "persistent_steady_seconds": float(new_focus["mean_wall_s"]),
            "persistent_speedup_vs_prior_stage4_static": steady_speedup,
            "persistent_speedup_vs_no_cache": speedup_vs_none,
            "baseline_peak_allocated_mb": float(old_focus["mean_peak_allocated_mb"]),
            "persistent_peak_allocated_mb": float(new_focus["mean_peak_allocated_mb"]),
            "geometry_build_seconds": float(new_focus["geometry_build_s"]),
            "geometry_cache_mb": float(new_focus["geometry_cache_mb"]),
            "persistent_mean_topk_calls": float(new_focus["mean_topk_calls"]),
            "max_output_abs_difference_all_sizes_nfes": max_output_diff,
        },
        "acceptance_checks": checks,
    }
    (ROOT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")

    tr = comparison["training_relative_change"]
    rec = comparison["reconstruction_focus_1m_nfe4"]
    verdict = (
        "PASS: persistent Top-K improves repeated reconstruction without a measurable quality change."
        if checks["all_acceptance_checks_pass"]
        else "REVIEW: at least one predeclared acceptance check did not pass."
    )
    results = f"""# CQ-LR persistent Top-K — 200-epoch paired result

**{verdict}**

The paired validator confirms byte-identical training-driver, data-helper, and
data-path source across revisions. Persistent Top-K is inference-only, so training
speed is a neutrality/control measurement; the expected efficiency gain is in
repeated cached-streamed reconstruction on fixed geometry.

| Training/quality metric | No persistent Top-K | Persistent implementation | Change |
|---|---:|---:|---:|
| Mean epoch time, epochs 2–200 (s) | {old['mean_epoch_seconds_2_to_200']:.3f} | {new['mean_epoch_seconds_2_to_200']:.3f} | {tr['mean_epoch_time']:+.2%} |
| Diagnostic step time (ms) | {old['mean_diagnostic_training_step_ms']:.3f} | {new['mean_diagnostic_training_step_ms']:.3f} | {tr['diagnostic_step_time']:+.2%} |
| Peak allocated (MiB) | {old['max_diagnostic_peak_allocated_mb']:.1f} | {new['max_diagnostic_peak_allocated_mb']:.1f} | {tr['peak_allocated']:+.2%} |
| Final validation RF loss | {old['final_validation_loss']:.6f} | {new['final_validation_loss']:.6f} | {tr['final_validation']:+.2%} |
| Epoch-200 fixed-manifest RF loss | {old_rf:.6f} | {new_rf:.6f} | {tr['fixed_manifest_rf_epoch_200']:+.2%} |

| 1M-query Euler NFE-4 reconstruction | Prior Stage-4 static cache | Persistent geometry + static cache |
|---|---:|---:|
| Steady latency (s) | {rec['baseline_steady_seconds']:.4f} | {rec['persistent_steady_seconds']:.4f} |
| Speedup | 1.00x | **{rec['persistent_speedup_vs_prior_stage4_static']:.2f}x** |
| Peak allocated (MiB) | {rec['baseline_peak_allocated_mb']:.1f} | {rec['persistent_peak_allocated_mb']:.1f} |
| Top-K searches after cache construction | n/a | {rec['persistent_mean_topk_calls']:.1f} |

One-time geometry construction costs {rec['geometry_build_seconds']:.4f} s and
stores {rec['geometry_cache_mb']:.1f} MiB at one million queries. The maximum
output difference across persistent benchmark cases is
{rec['max_output_abs_difference_all_sizes_nfes']:.3e}.

Final checkpoint tensors are {'bitwise identical' if weights['exact_match'] else 'not bitwise identical'}
(maximum parameter difference {weights['max_abs_parameter_difference']:.3e}).
Full raw metrics and acceptance checks are in `comparison.json`.
"""
    (ROOT / "RESULTS.md").write_text(results)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()

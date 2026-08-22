#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
COST = PACKAGE / "cost_benchmark"


def load(name: str) -> dict:
    return json.loads((COST / name).read_text())


def indexed(rows: list[dict]) -> dict[tuple[str, int], dict]:
    return {(row["label"], int(row["N_query"])): row for row in rows}


def comparison(reference: dict, candidate: dict) -> dict:
    speedup = reference["training_step_ms"] / candidate["training_step_ms"]
    allocated_reduction = 1.0 - candidate["peak_allocated_mb"] / reference["peak_allocated_mb"]
    reserved_reduction = 1.0 - candidate["peak_reserved_mb"] / reference["peak_reserved_mb"]
    return {
        "training_step_speedup": speedup,
        "allocated_memory_reduction_fraction": allocated_reduction,
        "reserved_memory_reduction_fraction": reserved_reduction,
        "speed_gate_pass": speedup >= 1.15,
        "memory_gate_pass": max(allocated_reduction, reserved_reduction) >= 0.10,
        "efficiency_gate_pass": (
            speedup >= 1.15
            and max(allocated_reduction, reserved_reduction) >= 0.10
        ),
    }


def main() -> None:
    scaling = load("cost_benchmark.json")
    primary = load("clean_b128_q4096.json")
    fallback = load("fallback_224_clean_b128_q4096.json")

    primary_rows = indexed(primary["scaling"])
    fallback_rows = indexed(fallback["scaling"])
    primary_gate = comparison(
        primary_rows[("F0", 4096)],
        primary_rows[("CQ-Balanced-192-Full", 4096)],
    )
    fallback_gate = comparison(
        fallback_rows[("F0", 4096)],
        fallback_rows[("CQ-Balanced-224-Full", 4096)],
    )

    scaling_rows = indexed(scaling["scaling"])
    scaling_comparisons = {}
    for n_query in (4096, 16384, 65536):
        f0 = scaling_rows[("F0", n_query)]
        scaling_comparisons[str(n_query)] = {
            label: comparison(f0, scaling_rows[(label, n_query)])
            for label in ("CQ-LR", "CQ-Balanced-192-Full")
        }

    million = {row["label"]: row for row in scaling["million_query_reconstruction"]}
    persistent = {
        label: {
            "steady_nfe4_s": row["wall_s"],
            "geometry_build_s": row["geometry_build_s"],
            "geometry_cache_mb": row["geometry_cache_mb"],
            "steady_speedup_vs_f0": million["F0"]["wall_s"] / row["wall_s"],
        }
        for label, row in million.items()
    }

    result = {
        "gate_thresholds": {
            "training_step_speedup": 1.15,
            "memory_reduction_fraction": 0.10,
        },
        "primary_192_clean_b128_q4096": primary_gate,
        "sole_fallback_224_clean_b128_q4096": fallback_gate,
        "scaling_batch1": scaling_comparisons,
        "persistent_1m_euler_nfe4": persistent,
        "model_summaries": scaling["model_summaries"],
        "decision": {
            "launch_192_screen_200": False,
            "launch_224_screen_200": False,
            "reason": (
                "Both structured-concat candidates fail the mandatory pre-training "
                "speed and memory gate at the clean batch-128, 4096-query protocol."
            ),
            "kernel_benchmark": "not_run_no_scientific_candidate_selected",
        },
    }
    (COST / "gate_decision.json").write_text(json.dumps(result, indent=2) + "\n")

    rows = []
    for n_query in (4096, 16384, 65536):
        f0 = scaling_rows[("F0", n_query)]
        for label in ("F0", "CQ-LR", "CQ-Balanced-192-Full"):
            row = scaling_rows[(label, n_query)]
            cmp = comparison(f0, row)
            rows.append(
                f"| {n_query:,} | {label} | {row['training_step_ms']:.3f} | "
                f"{cmp['training_step_speedup']:.3f}x | "
                f"{100 * cmp['allocated_memory_reduction_fraction']:.2f}% | "
                f"{100 * cmp['reserved_memory_reduction_fraction']:.2f}% |"
            )

    primary_row = primary_rows[("CQ-Balanced-192-Full", 4096)]
    primary_f0 = primary_rows[("F0", 4096)]
    cq_lr_row = primary_rows[("CQ-LR", 4096)]
    fallback_row = fallback_rows[("CQ-Balanced-224-Full", 4096)]
    fallback_f0 = fallback_rows[("F0", 4096)]
    markdown = f"""# CQ-Balanced quality-recovery result

**Decision: stop before training.** Both the 192-D primary and the sole 224-D
fallback fail the mandatory efficiency gate under the clean batch-128,
4,096-query protocol. No 200-epoch or kernel run was launched.

## Decisive clean-protocol gate

| Candidate | Step (ms) | Speedup vs F0 | Allocated reduction | Reserved reduction | Gate |
|---|---:|---:|---:|---:|---|
| F0 (primary run) | {primary_f0['training_step_ms']:.3f} | 1.000x | 0.00% | 0.00% | reference |
| CQ-Balanced-192-Full | {primary_row['training_step_ms']:.3f} | {primary_gate['training_step_speedup']:.3f}x | {100 * primary_gate['allocated_memory_reduction_fraction']:.2f}% | {100 * primary_gate['reserved_memory_reduction_fraction']:.2f}% | **FAIL** |
| F0 (fallback run) | {fallback_f0['training_step_ms']:.3f} | 1.000x | 0.00% | 0.00% | reference |
| CQ-Balanced-224-Full | {fallback_row['training_step_ms']:.3f} | {fallback_gate['training_step_speedup']:.3f}x | {100 * fallback_gate['allocated_memory_reduction_fraction']:.2f}% | {100 * fallback_gate['reserved_memory_reduction_fraction']:.2f}% | **FAIL** |

Required: at least 1.15x step speedup and at least 10% reduction in allocated
or reserved memory.

## Batch-1 scaling diagnostic

| Queries | Model | Step (ms) | Speedup vs F0 | Allocated reduction | Reserved reduction |
|---:|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Persistent 1M-query inference

Euler NFE=4, persistent geometry plus `static_features`, three repeats:

| Model | Steady latency (s) | Speedup vs F0 | Geometry build (s) | Geometry storage (MiB) |
|---|---:|---:|---:|---:|
"""
    for label in ("F0", "CQ-LR", "CQ-Balanced-192-Full"):
        row = persistent[label]
        markdown += (
            f"| {label} | {row['steady_nfe4_s']:.4f} | "
            f"{row['steady_speedup_vs_f0']:.3f}x | {row['geometry_build_s']:.4f} | "
            f"{row['geometry_cache_mb']:.1f} |\n"
        )
    markdown += f"""

The persistent geometry-only Top-K path remains functional and unchanged. The
192-D candidate is faster than F0 for persistent inference, but that does not
override the failed training efficiency gate.

## Quality/throughput Pareto recommendation

| Model | Clean best validation | Gap vs F0 | B128/Q4096 step | Speedup vs F0 | Peak allocated | Status |
|---|---:|---:|---:|---:|---:|---|
| F0 | 0.353095 | reference | {primary_f0['training_step_ms']:.1f} ms | 1.000x | {primary_f0['peak_allocated_mb']:.1f} MiB | **quality model** |
| CQ-LR | 0.388921 | +10.15% | {cq_lr_row['training_step_ms']:.1f} ms | {primary_f0['training_step_ms'] / cq_lr_row['training_step_ms']:.3f}x | {cq_lr_row['peak_allocated_mb']:.1f} MiB | **throughput model** |
| CQ-Balanced-192 | not measured | n/a | {primary_row['training_step_ms']:.1f} ms | {primary_gate['training_step_speedup']:.3f}x | {primary_row['peak_allocated_mb']:.1f} MiB | rejected before training |

- Quality model: F0.
- Throughput model: CQ-LR, with its known approximately 10% validation penalty.
- Formal 3-D scientific model: F0; use CQ-LR only for throughput-limited or
  exploratory 3-D work where the validated quality loss is acceptable.
- CQ-Balanced: do not promote; it restores much of F0's cost without clearing
  the pre-training efficiency gate, so spending training time would violate the
  staged protocol.

## Scientific outcome

The structured-concat hypothesis was not quality-screened because its intended
information separation restores too much of F0's query-side cost. CQ-Balanced
therefore has no validation/reconstruction point and cannot enter the quality
Pareto frontier.
"""
    (PACKAGE / "RESULTS.md").write_text(markdown)


if __name__ == "__main__":
    main()

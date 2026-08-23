#!/usr/bin/env python3
"""Validate and summarize the measured Stage-6 cost gate before training."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "benchmarks/cost_benchmark.json"
OUTPUT = ROOT / "benchmarks/gate_c_assessment.json"
LABELS = ("F0", "CQ-Full", "CQ-LR")
QUERY_SIZES = (4096, 16384, 65536)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing benchmark evidence: {INPUT}")
    source = json.loads(INPUT.read_text())
    scaling = {
        (row["label"], int(row["N_query"])): row
        for row in source["scaling"]
        if row.get("status") == "ok"
    }
    missing = [
        (label, size) for label in LABELS for size in QUERY_SIZES
        if (label, size) not in scaling
    ]
    million = {
        row["label"]: row
        for row in source["million_query_reconstruction"]
        if row.get("status") == "ok"
    }
    missing_million = [label for label in LABELS if label not in million]
    if missing or missing_million:
        raise RuntimeError(
            f"Incomplete Gate-C measurements: scaling={missing}, million={missing_million}"
        )

    f0 = scaling[("F0", 65536)]
    full = scaling[("CQ-Full", 65536)]
    lowrank = scaling[("CQ-LR", 65536)]
    full_forward_speedup = float(f0["forward_ms"]) / float(full["forward_ms"])
    lr_forward_speedup = float(f0["forward_ms"]) / float(lowrank["forward_ms"])
    lr_additional = float(full["forward_ms"]) / float(lowrank["forward_ms"]) - 1.0
    full_step_speedup = float(f0["training_step_ms"]) / float(full["training_step_ms"])
    lr_step_speedup = float(f0["training_step_ms"]) / float(lowrank["training_step_ms"])
    full_peak_reduction = 1.0 - float(full["peak_allocated_mb"]) / float(f0["peak_allocated_mb"])
    lr_peak_reduction = 1.0 - float(lowrank["peak_allocated_mb"]) / float(f0["peak_allocated_mb"])

    target_flags = {
        "cq_full_step_speedup_at_least_1p4": full_step_speedup >= 1.4,
        "cq_lr_step_speedup_at_least_1p7": lr_step_speedup >= 1.7,
        "cq_full_peak_reduction_at_least_25pct": full_peak_reduction >= 0.25,
        "cq_lr_peak_reduction_at_least_35pct": lr_peak_reduction >= 0.35,
    }
    safe_to_train = lr_additional >= 0.05
    result = {
        "status": "pass" if safe_to_train else "inspect_before_training",
        "meaningful_lr_additional_speedup_threshold": 0.05,
        "measurements_at_65536": {
            "cq_full_forward_speedup_vs_f0": full_forward_speedup,
            "cq_lr_forward_speedup_vs_f0": lr_forward_speedup,
            "cq_lr_additional_forward_speedup_vs_cq_full": lr_additional,
            "cq_full_training_step_speedup_vs_f0": full_step_speedup,
            "cq_lr_training_step_speedup_vs_f0": lr_step_speedup,
            "cq_full_peak_allocation_reduction_vs_f0": full_peak_reduction,
            "cq_lr_peak_allocation_reduction_vs_f0": lr_peak_reduction,
        },
        "suggested_target_flags": target_flags,
        "component_timings_ms_at_65536": {
            label: {
                key: value for key, value in scaling[(label, 65536)].items()
                if key.endswith("_ms") and key not in {
                    "forward_ms", "backward_ms", "optimizer_ms", "training_step_ms"
                }
            }
            for label in LABELS
        },
        "million_query": {
            label: {
                key: million[label][key]
                for key in (
                    "wall_s", "seconds_per_million_queries_per_nfe",
                    "peak_allocated_mb", "peak_reserved_mb",
                    "condition_context_mb", "static_query_cache_mb",
                    "dynamic_peak_above_static_cache_mb",
                )
            }
            for label in LABELS
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not safe_to_train:
        raise SystemExit(
            "Gate C requires implementation inspection before training: "

            f"CQ-LR additional={lr_additional:.1%}."
        )


if __name__ == "__main__":
    main()

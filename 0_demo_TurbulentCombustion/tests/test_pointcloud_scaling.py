from __future__ import annotations

import sys
from pathlib import Path

import torch

BENCHMARKS = Path(__file__).resolve().parents[1] / "research" / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from benchmark_pointcloud_scaling import SCALING_COLUMNS, add_derived_metrics, tensor_megabytes


def test_scaling_schema_contains_required_separated_phases():
    required = {
        "N_full", "N_query", "N_obs", "batch_size", "read_ms", "normalize_ms",
        "index_sampling_ms", "h2d_ms", "pre_model_ms", "forward_ms", "backward_ms",
        "optimizer_ms", "step_ms", "gpu_peak_allocated_mb", "gpu_peak_reserved_mb",
        "host_rss_mb", "queries_per_sec", "samples_per_sec", "ms_per_1k_queries",
        "gpu_memory_mb_per_1k_queries",
    }
    assert required.issubset(SCALING_COLUMNS)


def test_scaling_derived_metrics_and_storage_deduplication():
    row = add_derived_metrics({
        "batch_size": 2, "N_query": 5000, "step_ms": 20.0,
        "gpu_peak_allocated_mb": 100.0,
    })
    assert row["queries_per_sec"] == 500_000.0
    assert row["samples_per_sec"] == 100.0
    assert row["ms_per_1k_queries"] == 2.0
    assert row["gpu_memory_mb_per_1k_queries"] == 10.0

    tensor = torch.zeros(10, dtype=torch.float32)
    assert tensor_megabytes({"a": tensor, "alias": tensor.view(2, 5)}) == 40 / (1024 ** 2)

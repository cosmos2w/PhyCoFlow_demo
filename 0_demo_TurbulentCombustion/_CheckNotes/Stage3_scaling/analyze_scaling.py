#!/usr/bin/env python3
"""Create the compact Stage-3 scaling summary from benchmark CSVs."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows)


def main() -> None:
    data = list(csv.DictReader(open(ROOT / "data_scaling.csv")))
    model = list(csv.DictReader(open(ROOT / "model_scaling.csv")))
    data_summary = []
    for n_full in sorted({int(row["N_full"]) for row in data}):
        for n_query in sorted({int(row["N_query"]) for row in data if int(row["N_full"]) == n_full}):
            rows = [
                row for row in data
                if int(row["N_full"]) == n_full and int(row["N_query"]) == n_query
            ]
            data_summary.append({
                "N_full": n_full, "N_query": n_query,
                "pre_model_ms": mean(rows, "pre_model_ms"),
                "read_ms": mean(rows, "read_ms"),
                "normalize_ms": mean(rows, "normalize_ms"),
                "host_rss_mb": mean(rows, "host_rss_mb"),
                "selected_gpu_input_mb": mean(rows, "selected_gpu_input_mb"),
            })
    model_summary = [{
        "N_query": int(row["N_query"]), "N_obs": int(row["N_obs"]),
        "forward_ms": float(row["forward_ms"]), "backward_ms": float(row["backward_ms"]),
        "step_ms": float(row["step_ms"]),
        "gpu_peak_allocated_mb": float(row["gpu_peak_allocated_mb"]),
        "queries_per_sec": float(row["queries_per_sec"]),
    } for row in model]
    summary = {"data": data_summary, "model": model_summary}
    with open(ROOT / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    lines = [
        "# Stage 3 PointCloud FFM scaling report", "",
        "The real dataset contains 40,300 points. Rows at 250k and 1M use a tiled real",
        "snapshot in host memory; their `read_ms` is an in-memory clone and is not labeled",
        "as HDF5 throughput. Model rows use the active GL_rbf_ENH + topk_rbf_glres +",
        "KeOps configuration with synthetic 3-D tensors and batch size 1.", "",
        "GPU 0 had a pre-existing ~10.6 GiB, high-utilization process. The goal permits",
        "co-location when memory fits, so wall times are diagnostic rather than clean",
        "exclusive-GPU absolutes. CUDA peak numbers are process-local.", "",
        "## Data-path scaling (mean over M=256/512/1024)", "",
        "| N_full | N_query | pre-model ms | read ms | normalize ms | host RSS MB | selected GPU input MB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data_summary:
        lines.append(
            f"| {row['N_full']:,} | {row['N_query']:,} | {row['pre_model_ms']:.2f} | "
            f"{row['read_ms']:.2f} | {row['normalize_ms']:.2f} | {row['host_rss_mb']:.1f} | "
            f"{row['selected_gpu_input_mb']:.3f} |"
        )
    lines += [
        "", "## Model execution scaling", "",
        "| N_query | M | forward ms | backward ms | step ms | peak allocated MB | queries/s |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary:
        lines.append(
            f"| {row['N_query']:,} | {row['N_obs']:,} | {row['forward_ms']:.2f} | "
            f"{row['backward_ms']:.2f} | {row['step_ms']:.2f} | "
            f"{row['gpu_peak_allocated_mb']:.1f} | {row['queries_per_sec']:,.0f} |"
        )
    lines += [
        "", "## Interpretation", "",
        "- At fixed 4,096 queries, expanding the full mesh from 40.3k to 1M changes",
        "  mean pre-model time from 17.99 ms to 24.44 ms and host RSS from 642.0 MB",
        "  to 694.3 MB. Selected GPU inputs remain 0.582 MB.",
        "- The model step is already 61.15–64.53 ms at 4,096 queries, and reaches",
        "  709.39–801.20 ms at 65,536 queries. At 1M/65,536, the data path averages",
        "  45.73 ms while the model requires 0.71–0.80 s: query-model work dominates.",
        "- Increasing observations from 256 to 1,024 raises model step time by 5.5%",
        "  at 4,096 queries and 12.4% at 65,536 queries. KeOps prevents pairwise-memory",
        "  explosion, but exact top-k search still adds work.",
        "- Model peak allocation rises from 255–271 MB at 4,096 queries to about",
        "  3.02–3.04 GB at 65,536 queries. This near-linear activation scaling motivates",
        "  Stage 4 end-to-end inference streaming and Stage 5 training microbatching.",
        "", "## Exact commands", "",
        "See the command blocks below; both used the project `phycoflow_env` and GPU 0.", "",
        "```bash",
        "python src/benchmark_pointcloud_scaling.py --config Save_config/config_pointcloud_ffm.yaml \\",
        "  --stats-path Save_TrainedModel/ffm_tc_pointcloud_DemoN51_20260718_083538/dataset_stats.pt \\",
        "  --classes data --n-full 40300 250000 1000000 --n-query 4096 16384 65536 \\",
        "  --n-obs 256 512 1024 --data-batch-size 4 --iterations 3 --warmup 1 \\",
        "  --device cuda:0 --output-csv _CheckNotes/Stage3_scaling/data_scaling.csv \\",
        "  --output-json _CheckNotes/Stage3_scaling/data_scaling.json",
        "", "python src/benchmark_pointcloud_scaling.py --config Save_config/config_pointcloud_ffm.yaml \\",
        "  --stats-path Save_TrainedModel/ffm_tc_pointcloud_DemoN51_20260718_083538/dataset_stats.pt \\",
        "  --classes model --n-query 4096 16384 65536 --n-obs 256 512 1024 \\",
        "  --model-batch-size 1 --iterations 3 --warmup 1 --device cuda:0 \\",
        "  --output-csv _CheckNotes/Stage3_scaling/model_scaling.csv \\",
        "  --output-json _CheckNotes/Stage3_scaling/model_scaling.json",
        "```", "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

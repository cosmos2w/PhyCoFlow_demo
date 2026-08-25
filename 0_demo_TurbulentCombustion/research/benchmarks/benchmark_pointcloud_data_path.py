#!/usr/bin/env python3
"""Short real-dataset A/B benchmark for PointCloud FFM input pipelines."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
import yaml

import _bootstrap  # noqa: F401  # adds the runtime src/ directory

from helpers import TurbulentCombustionH5Dataset, build_sparse_condition_legacy
from pointcloud_data_path import PointCloudBatchCollator, resolve_data_path_config


def collate_snapshots_legacy(items):
    start = time.perf_counter()
    batch = {
        "coords": torch.stack([item["coords"] for item in items]),
        "fields": torch.stack([item["fields"] for item in items]),
        "time_index": torch.stack([item["time_index"] for item in items]),
    }
    batch["data_path_timings"] = {
        "index_sampling_ms": 0.0,
        "hdf5_read_ms": sum(
            item.get("data_path_item_timings", {}).get("hdf5_read_ms", 0.0) for item in items
        ),
        "cpu_normalization_ms": sum(
            item.get("data_path_item_timings", {}).get("cpu_normalization_ms", 0.0)
            for item in items
        ),
        "cpu_materialization_ms": (time.perf_counter() - start) * 1000.0,
    }
    return batch


def prepare_for_device(batch, cfg, device, cond_fields, n_obs, n_query):
    """Materialize the pre-model tensors without importing any model code."""
    non_blocking = cfg.non_blocking_transfer
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    if batch.get("materialized_selected", False):
        tensors = {
            key: batch[key].to(device, non_blocking=non_blocking)
            for key in (
                "coords_q", "fields_q", "obs_coords", "obs_values",
                "obs_mask", "obs_indices", "obs_field_ids",
            )
        }
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return tensors, {
            "h2d_ms": (time.perf_counter() - start) * 1000.0,
            "sparse_condition_materialization_ms": 0.0,
            "query_materialization_ms": 0.0,
        }

    coords_full = batch["coords"].to(device, non_blocking=non_blocking)
    fields_full = batch["fields"].to(device, non_blocking=non_blocking)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    h2d_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition_legacy(
        coords_full, fields_full, cond_fields, [n_obs], [n_obs]
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    sparse_ms = (time.perf_counter() - start) * 1000.0
    start = time.perf_counter()
    count = min(int(n_query), coords_full.shape[1])
    indices = torch.stack(
        [torch.randperm(coords_full.shape[1], device=device)[:count].sort().values
         for _ in range(coords_full.shape[0])]
    )
    coords_q = torch.gather(
        coords_full, 1, indices.unsqueeze(-1).expand(-1, -1, coords_full.shape[-1])
    )
    fields_q = torch.gather(
        fields_full, 1, indices.unsqueeze(-1).expand(-1, -1, fields_full.shape[-1])
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    query_ms = (time.perf_counter() - start) * 1000.0
    return {
        "coords_q": coords_q,
        "fields_q": fields_q,
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_indices": obs_indices,
        "obs_field_ids": obs_field_ids,
    }, {
        "h2d_ms": h2d_ms,
        "sparse_condition_materialization_ms": sparse_ms,
        "query_materialization_ms": query_ms,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Save_config/config_pointcloud_ffm.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--stats-path", default=None)
    parser.add_argument("--output", default="data_path_benchmark.csv")
    parser.add_argument(
        "--profiles", nargs="+",
        default=["legacy", "optimized_fullnorm", "optimized", "optimized_indexed"],
        choices=["legacy", "optimized_fullnorm", "optimized", "optimized_indexed"],
    )
    parser.add_argument("--batch-size", dest="batch_sizes", type=int, nargs="+", default=[4])
    parser.add_argument("--n-query-points", nargs="+", default=["4096", "16384", "65536"])
    parser.add_argument("--n-obs", dest="n_obs_counts", type=int, nargs="+", default=[256])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--benchmark-indexed-full",
        action="store_true",
        help="Also force fancy-indexed HDF5 reads when the query is the complete mesh.",
    )
    return parser.parse_args()


def profile_config(name: str):
    overrides = {"data_path_mode": "legacy" if name == "legacy" else "optimized"}
    if name == "optimized_fullnorm":
        overrides["field_normalization_mode"] = "legacy_full_after_read"
    if name == "optimized_indexed":
        overrides["field_read_mode"] = "indexed_union"
    return resolve_data_path_config(overrides)


def make_loader(dataset, cfg, yaml_cfg, cli, n_query, batch_size, n_obs):
    if cfg.data_path_mode == "legacy":
        collate_fn = collate_snapshots_legacy
    else:
        collate_fn = PointCloudBatchCollator(
            dataset=dataset,
            config=cfg,
            cond_fields=yaml_cfg.get("cond_fields", [yaml_cfg.get("cond_field", 2)]),
            n_obs_min=[n_obs],
            n_obs_max=[n_obs],
            n_query_points=n_query,
            query_sampling="uniform",
        )
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": cli.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_fn,
    }
    if cli.num_workers > 0:
        kwargs["persistent_workers"] = cfg.dataloader_persistent_workers
        if cfg.dataloader_prefetch_factor is not None:
            kwargs["prefetch_factor"] = cfg.dataloader_prefetch_factor
    return torch.utils.data.DataLoader(**kwargs)


def benchmark_one(name, n_query, batch_size, n_obs, yaml_cfg, cli, device):
    cfg = profile_config(name)
    dataset = TurbulentCombustionH5Dataset(
        cli.data,
        split="train",
        train_ratio=float(yaml_cfg.get("train_ratio", 0.9)),
        seed=cli.seed,
        time_stride=int(yaml_cfg.get("time_stride", 1)),
        field_names=yaml_cfg.get("FIELD_NAMES", yaml_cfg.get("field_names")),
        stats_path=cli.stats_path,
        coord_batch_mode=cfg.coord_batch_mode,
        defer_field_read=(cfg.sampling_device == "cpu"),
        instrument_data_path=True,
    )
    loader = make_loader(dataset, cfg, yaml_cfg, cli, n_query, batch_size, n_obs)
    iterator = iter(loader)
    phase_rows = []
    cond_fields = yaml_cfg.get("cond_fields", [yaml_cfg.get("cond_field", 2)])
    total_iterations = cli.warmup + cli.iterations

    for iteration in range(total_iterations):
        start_total = time.perf_counter()
        batch = next(iterator)
        loader_wait_ms = (time.perf_counter() - start_total) * 1000.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tensors, prep = prepare_for_device(
            batch, cfg, device, cond_fields, n_obs, n_query
        )
        total_ms = (time.perf_counter() - start_total) * 1000.0
        if iteration < cli.warmup:
            del tensors, batch
            continue
        collate = batch.get("data_path_timings", {})
        phase_rows.append({
            "loader_wait_ms": loader_wait_ms,
            "index_sampling_ms": float(collate.get("index_sampling_ms", 0.0)),
            "hdf5_read_ms": float(collate.get("hdf5_read_ms", 0.0)),
            "cpu_normalization_ms": float(collate.get("cpu_normalization_ms", 0.0)),
            "cpu_materialization_ms": float(collate.get("cpu_materialization_ms", 0.0)),
            "h2d_ms": prep["h2d_ms"],
            "sparse_condition_materialization_ms": prep["sparse_condition_materialization_ms"],
            "query_materialization_ms": prep["query_materialization_ms"],
            "total_ms": total_ms,
            "n_query": int(tensors["coords_q"].shape[1]),
            "n_obs": int(tensors["obs_mask"].sum()),
            "peak_allocated_mb": (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
            ),
            "peak_reserved_mb": (
                torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else 0.0
            ),
        })
        del tensors, batch

    def avg(key):
        return sum(row[key] for row in phase_rows) / len(phase_rows)

    pre_model_keys = (
        "index_sampling_ms", "hdf5_read_ms", "cpu_normalization_ms",
        "cpu_materialization_ms", "h2d_ms", "sparse_condition_materialization_ms",
        "query_materialization_ms",
    )
    pre_model_ms = sum(avg(key) for key in pre_model_keys)
    total_ms = avg("total_ms")
    effective_q = int(phase_rows[0]["n_query"])
    result = {
        "profile": name,
        "data_path_mode": cfg.data_path_mode,
        "coord_batch_mode": cfg.coord_batch_mode,
        "index_sampling_mode": cfg.index_sampling_mode,
        "sampling_device": cfg.sampling_device,
        "field_read_mode": cfg.field_read_mode,
        "field_normalization_mode": cfg.field_normalization_mode,
        "gpu_transfer_mode": cfg.gpu_transfer_mode,
        "batch_size": batch_size,
        "N_full": dataset.num_points,
        "N_query": effective_q,
        "N_obs_total": int(phase_rows[0]["n_obs"]),
        "iterations": cli.iterations,
        "samples_per_sec": batch_size / (total_ms / 1000.0),
        "selected_query_points_per_sec": batch_size * effective_q / (total_ms / 1000.0),
        "pre_model_latency_ms": pre_model_ms,
        "loader_wait_ms": avg("loader_wait_ms"),
        "index_sampling_ms": avg("index_sampling_ms"),
        "hdf5_read_ms": avg("hdf5_read_ms"),
        "cpu_normalization_ms": avg("cpu_normalization_ms"),
        "cpu_materialization_ms": avg("cpu_materialization_ms"),
        "h2d_ms": avg("h2d_ms"),
        "sparse_condition_materialization_ms": avg("sparse_condition_materialization_ms"),
        "query_materialization_ms": avg("query_materialization_ms"),
        "total_step_ms": total_ms,
        "peak_allocated_mb": avg("peak_allocated_mb"),
        "peak_reserved_mb": avg("peak_reserved_mb"),
        "optimized_over_legacy_step_time": "",
        "optimized_over_legacy_pre_model_time": "",
        "optimized_over_legacy_gpu_memory": "",
    }
    dataset.close()
    return result


def main() -> None:
    cli = parse_args()
    torch.manual_seed(cli.seed)
    with open(cli.config) as handle:
        yaml_cfg = yaml.safe_load(handle) or {}
    cli.data = cli.data or yaml_cfg["data"]
    if cli.stats_path is None:
        candidate = Path(cli.data).with_suffix(".stats.pt")
        if not candidate.exists():
            raise FileNotFoundError(
                "A precomputed stats file is required for a short benchmark. "
                "Pass --stats-path to avoid a full-dataset statistics scan."
            )
        cli.stats_path = str(candidate)
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    probe = TurbulentCombustionH5Dataset(
        cli.data,
        stats_path=cli.stats_path,
        field_names=yaml_cfg.get("FIELD_NAMES", yaml_cfg.get("field_names")),
        coord_batch_mode="shared_mesh",
        defer_field_read=True,
    )
    print(f"HDF5 layout: {probe.hdf5_layout()}")
    n_full = probe.num_points
    probe.close()

    query_counts = [n_full if str(value).lower() == "full" else min(int(value), n_full)
                    for value in cli.n_query_points]
    rows = []
    for batch_size in cli.batch_sizes:
        for n_obs in cli.n_obs_counts:
            for n_query in query_counts:
                for profile in cli.profiles:
                    if profile == "optimized_indexed" and n_query >= n_full and not cli.benchmark_indexed_full:
                        print(
                            "optimized_indexed  Nq=full skipped (use --benchmark-indexed-full to force "
                            "the known-pathological contiguous-HDF5 fancy read)."
                        )
                        continue
                    row = benchmark_one(
                        profile, n_query, batch_size, n_obs, yaml_cfg, cli, device
                    )
                    rows.append(row)
                    print(
                        f"{profile:18s} B={batch_size:3d} M={n_obs:5d} Nq={n_query:7d} "
                        f"total={row['total_step_ms']:8.2f}ms "
                        f"HDF5={row['hdf5_read_ms']:8.2f}ms H2D={row['h2d_ms']:7.2f}ms "
                        f"peak={row['peak_allocated_mb']:8.1f}MB"
                    )

    legacy_by_query = {
        (row["batch_size"], row["N_obs_total"], row["N_query"]): row
        for row in rows if row["profile"] == "legacy"
    }
    for row in rows:
        legacy = legacy_by_query.get((row["batch_size"], row["N_obs_total"], row["N_query"]))
        if legacy is None or row["profile"] == "legacy":
            continue
        row["optimized_over_legacy_step_time"] = row["total_step_ms"] / legacy["total_step_ms"]
        row["optimized_over_legacy_pre_model_time"] = (
            row["pre_model_latency_ms"] / legacy["pre_model_latency_ms"]
        )
        row["optimized_over_legacy_gpu_memory"] = (
            row["peak_allocated_mb"] / legacy["peak_allocated_mb"]
            if legacy["peak_allocated_mb"] else ""
        )

    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved benchmark CSV: {output}")


if __name__ == "__main__":
    main()

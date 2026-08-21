#!/usr/bin/env python3
"""Separate full-mesh data cost from GL-RBF selected-query model cost."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from helpers import TurbulentCombustionH5Dataset
from Model import ConditionalPointHybridLocalGlobalRBF
from pointcloud_data_path import (
    materialize_selected_batch,
    sample_query_indices,
    sample_sparse_observation_indices,
)


SCALING_COLUMNS = (
    "benchmark_class", "status", "data_source", "N_full", "N_query", "N_obs",
    "batch_size", "read_ms", "normalize_ms", "index_sampling_ms", "materialize_ms",
    "h2d_ms", "pre_model_ms", "forward_ms", "backward_ms", "optimizer_ms", "step_ms",
    "gpu_peak_allocated_mb", "gpu_peak_reserved_mb", "selected_gpu_input_mb",
    "host_rss_mb", "host_max_rss_mb", "queries_per_sec", "samples_per_sec",
    "ms_per_1k_queries", "gpu_memory_mb_per_1k_queries", "iterations", "notes",
)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def current_rss_mb() -> float:
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def max_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if os.name != "posix" or value > 1024 else value / (1024.0 ** 2)


def tensor_megabytes(tensors: Mapping[str, torch.Tensor]) -> float:
    seen: set[tuple[int, int]] = set()
    total = 0
    for tensor in tensors.values():
        if not torch.is_tensor(tensor):
            continue
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total / (1024.0 ** 2)


def add_derived_metrics(row: dict[str, Any]) -> dict[str, Any]:
    step_ms = float(row.get("step_ms", 0.0) or 0.0)
    batch_size = int(row.get("batch_size", 0) or 0)
    n_query = int(row.get("N_query", 0) or 0)
    if step_ms > 0 and batch_size > 0 and n_query > 0:
        row["queries_per_sec"] = batch_size * n_query / (step_ms / 1000.0)
        row["samples_per_sec"] = batch_size / (step_ms / 1000.0)
        row["ms_per_1k_queries"] = step_ms / (batch_size * n_query / 1000.0)
        row["gpu_memory_mb_per_1k_queries"] = float(
            row.get("gpu_peak_allocated_mb", 0.0) or 0.0
        ) / (batch_size * n_query / 1000.0)
    else:
        for key in (
            "queries_per_sec", "samples_per_sec", "ms_per_1k_queries",
            "gpu_memory_mb_per_1k_queries",
        ):
            row[key] = 0.0
    return row


class ExpandedSnapshotDataset:
    """In-memory expansion of one real snapshot for non-I/O scaling diagnostics."""

    def __init__(self, base: TurbulentCombustionH5Dataset, raw: torch.Tensor, n_full: int):
        repeats = math.ceil(int(n_full) / int(base.num_points))
        self.coords = base.coords.repeat((repeats, 1))[:n_full].contiguous()
        self._raw = raw.repeat((repeats, 1))[:n_full].contiguous()
        self.mean = base.mean
        self.std = base.std
        self.num_points = int(n_full)
        self.num_fields = int(base.num_fields)
        self.fixed_mesh = True

    def read_fields(self, time_index: int, point_indices: torch.Tensor | None = None) -> torch.Tensor:
        del time_index
        if point_indices is None:
            # A clone models one sequential host-memory delivery without claiming
            # that this is an HDF5 timing at synthetic-expanded sizes.
            return self._raw.clone()
        return self._raw.index_select(0, point_indices)


def _mean(rows: Sequence[Mapping[str, float]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def benchmark_data_case(
    *, dataset: Any, data_source: str, n_query: int, n_obs: int, batch_size: int,
    device: torch.device, iterations: int, warmup: int, seed: int,
) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    items = [
        {"time_index": torch.tensor(i), "physical_time": torch.tensor(float(i))}
        for i in range(batch_size)
    ]
    for iteration in range(warmup + iterations):
        generator = torch.Generator().manual_seed(seed + iteration)
        step_start = time.perf_counter()
        index_start = time.perf_counter()
        obs_layout = sample_sparse_observation_indices(
            batch_size=batch_size,
            n_full=dataset.num_points,
            cond_fields=[0],
            n_obs_min=[n_obs],
            n_obs_max=[n_obs],
            index_sampling_mode="scalable",
            generator=generator,
        )
        query_indices = sample_query_indices(
            batch_size=batch_size,
            n_full=dataset.num_points,
            n_query=n_query,
            query_sampling="uniform",
            index_sampling_mode="scalable",
            generator=generator,
        )
        index_ms = (time.perf_counter() - index_start) * 1000.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        batch = materialize_selected_batch(
            dataset=dataset,
            items=items,
            query_indices=query_indices,
            obs_layout=obs_layout,
            field_read_mode="legacy_full_snapshot",
            field_normalization_mode="selected_after_full_read",
        )
        h2d_start = time.perf_counter()
        selected = {
            key: batch[key].to(device, non_blocking=False)
            for key in (
                "coords_q", "fields_q", "obs_coords", "obs_values", "obs_mask",
                "obs_indices", "obs_field_ids",
            )
        }
        _sync(device)
        h2d_ms = (time.perf_counter() - h2d_start) * 1000.0
        step_ms = (time.perf_counter() - step_start) * 1000.0
        if iteration >= warmup:
            timings = batch["data_path_timings"]
            rows.append({
                "read_ms": timings["hdf5_read_ms"],
                "normalize_ms": timings["cpu_normalization_ms"],
                "index_sampling_ms": index_ms,
                "materialize_ms": timings["cpu_materialization_ms"],
                "h2d_ms": h2d_ms,
                "step_ms": step_ms,
                "gpu_peak_allocated_mb": (
                    torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                    if device.type == "cuda" else 0.0
                ),
                "gpu_peak_reserved_mb": (
                    torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
                    if device.type == "cuda" else 0.0
                ),
                "selected_gpu_input_mb": tensor_megabytes(selected),
                "host_rss_mb": current_rss_mb(),
                "host_max_rss_mb": max_rss_mb(),
            })
        del selected, batch

    pre_model = sum(_mean(rows, key) for key in (
        "read_ms", "normalize_ms", "index_sampling_ms", "materialize_ms", "h2d_ms"
    ))
    row = {
        "benchmark_class": "data", "status": "ok", "data_source": data_source,
        "N_full": dataset.num_points, "N_query": min(n_query, dataset.num_points),
        "N_obs": n_obs, "batch_size": batch_size,
        "read_ms": _mean(rows, "read_ms"), "normalize_ms": _mean(rows, "normalize_ms"),
        "index_sampling_ms": _mean(rows, "index_sampling_ms"),
        "materialize_ms": _mean(rows, "materialize_ms"), "h2d_ms": _mean(rows, "h2d_ms"),
        "pre_model_ms": pre_model, "forward_ms": 0.0, "backward_ms": 0.0,
        "optimizer_ms": 0.0, "step_ms": _mean(rows, "step_ms"),
        "gpu_peak_allocated_mb": _mean(rows, "gpu_peak_allocated_mb"),
        "gpu_peak_reserved_mb": _mean(rows, "gpu_peak_reserved_mb"),
        "selected_gpu_input_mb": _mean(rows, "selected_gpu_input_mb"),
        "host_rss_mb": _mean(rows, "host_rss_mb"), "host_max_rss_mb": _mean(rows, "host_max_rss_mb"),
        "iterations": iterations,
        "notes": "real HDF5" if data_source == "real_hdf5" else "real snapshot expanded in memory; read_ms is host clone, not HDF5",
    }
    return add_derived_metrics(row)


def build_model(config: Mapping[str, Any], n_fields: int, device: torch.device):
    enhanced = str(config.get("backbone", "GL_rbf_ENH")) == "GL_rbf_ENH"
    model = ConditionalPointHybridLocalGlobalRBF(
        n_fields=n_fields, coord_dim=3,
        hidden_dim=int(config.get("hidden_dim", 256)),
        cond_dim=int(config.get("cond_dim", 128)),
        field_embed_dim=int(config.get("field_embed_dim", 128)),
        latent_dim=int(config.get("latent_dim", 128)),
        num_latents=int(config.get("num_latents", 128)),
        num_heads=int(config.get("num_heads", 8)),
        num_latent_blocks=int(config.get("num_latent_blocks", 4)),
        ff_mult=int(config.get("ff_mult", 4)),
        attn_dropout=float(config.get("attn_dropout", 0.0)),
        mlp_dropout=float(config.get("mlp_dropout", 0.0)),
        rbf_sigma=float(config.get("rbf_sigma", 0.05)),
        summary_type=str(config.get("summary_type", "mean")),
        gather_mode=str(config.get("gather_mode", "topk_rbf_glres")),
        gather_topk=int(config.get("gather_topk", 32)),
        gather_query_chunk_size=config.get("gather_query_chunk_size", 2048),
        learnable_rbf_sigma=bool(config.get("learnable_rbf_sigma", True)),
        neighbor_backend=str(config.get("neighbor_backend", "keops")),
        sensor_local_topk=int(config.get("sensor_local_topk", 16)),
        sensor_local_dropout=float(config.get("sensor_local_dropout", 0.0)),
        use_fourier_pe=bool(config.get("USE_FOURIER_PE", True)),
        fourier_pe_num_bands=int(config.get("fourier_pe_num_bands", 32)),
        fourier_pe_max_freq=float(config.get("fourier_pe_max_freq", 64.0)),
        enhanced_backbone=enhanced,
        sensor_coord_encoding=str(config.get("sensor_coord_encoding", "fourier" if enhanced else "raw")),
        latent_sensor_reinject=bool(config.get("latent_sensor_reinject", enhanced)),
        latent_reinject_every=int(config.get("latent_reinject_every", 1)),
        query_latent_readout=bool(config.get("query_latent_readout", enhanced)),
        query_readout_type=str(config.get("query_readout_type", "coord" if enhanced else "point")),
        query_readout_scale_init=float(config.get("query_readout_scale_init", 1e-2 if enhanced else 0.0)),
        enhanced_head_norm=bool(config.get("enhanced_head_norm", enhanced)),
        glres_scale_init=float(config.get("glres_scale_init", 1e-2 if enhanced else 0.0)),
    )
    return model.to(device)


def benchmark_model_case(
    *, model: torch.nn.Module, n_query: int, n_obs: int, batch_size: int,
    n_fields: int, device: torch.device, iterations: int, warmup: int, seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    coords = torch.rand(batch_size, n_query, 3, device=device) * 2.0 - 1.0
    x_t = torch.randn(batch_size, n_query, n_fields, device=device)
    obs_coords = torch.rand(batch_size, n_obs, 3, device=device) * 2.0 - 1.0
    obs_values = torch.randn(batch_size, n_obs, 1, device=device)
    obs_mask = torch.ones(batch_size, n_obs, device=device)
    obs_field_ids = torch.zeros(batch_size, n_obs, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rows: list[dict[str, float]] = []
    try:
        for iteration in range(warmup + iterations):
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            _sync(device)
            step_start = time.perf_counter()
            start = time.perf_counter()
            pred = model(
                torch.full((batch_size,), 0.5, device=device), x_t, coords,
                obs_coords, obs_values, obs_mask, obs_field_ids,
            )
            loss = pred.square().mean()
            _sync(device)
            forward_ms = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            loss.backward()
            _sync(device)
            backward_ms = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            optimizer.step()
            _sync(device)
            optimizer_ms = (time.perf_counter() - start) * 1000.0
            step_ms = (time.perf_counter() - step_start) * 1000.0
            if iteration >= warmup:
                rows.append({
                    "forward_ms": forward_ms, "backward_ms": backward_ms,
                    "optimizer_ms": optimizer_ms, "step_ms": step_ms,
                    "gpu_peak_allocated_mb": (
                        torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                        if device.type == "cuda" else 0.0
                    ),
                    "gpu_peak_reserved_mb": (
                        torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
                        if device.type == "cuda" else 0.0
                    ),
                    "host_rss_mb": current_rss_mb(), "host_max_rss_mb": max_rss_mb(),
                })
            del pred, loss
        row = {
            "benchmark_class": "model", "status": "ok", "data_source": "synthetic_3d_gpu",
            "N_full": 0, "N_query": n_query, "N_obs": n_obs, "batch_size": batch_size,
            "read_ms": 0.0, "normalize_ms": 0.0, "index_sampling_ms": 0.0,
            "materialize_ms": 0.0, "h2d_ms": 0.0, "pre_model_ms": 0.0,
            "forward_ms": _mean(rows, "forward_ms"), "backward_ms": _mean(rows, "backward_ms"),
            "optimizer_ms": _mean(rows, "optimizer_ms"), "step_ms": _mean(rows, "step_ms"),
            "gpu_peak_allocated_mb": _mean(rows, "gpu_peak_allocated_mb"),
            "gpu_peak_reserved_mb": _mean(rows, "gpu_peak_reserved_mb"),
            "selected_gpu_input_mb": tensor_megabytes({
                "coords": coords, "x_t": x_t, "obs_coords": obs_coords,
                "obs_values": obs_values, "obs_mask": obs_mask, "obs_field_ids": obs_field_ids,
            }),
            "host_rss_mb": _mean(rows, "host_rss_mb"), "host_max_rss_mb": _mean(rows, "host_max_rss_mb"),
            "iterations": iterations, "notes": "current configured GL-RBF execution; synthetic values",
        }
        return add_derived_metrics(row)
    except torch.cuda.OutOfMemoryError as exc:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        row = {key: 0.0 for key in SCALING_COLUMNS}
        row.update({
            "benchmark_class": "model", "status": "oom", "data_source": "synthetic_3d_gpu",
            "N_full": 0, "N_query": n_query, "N_obs": n_obs, "batch_size": batch_size,
            "iterations": iterations, "notes": str(exc).replace("\n", " ")[:500],
        })
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Save_config/config_pointcloud_ffm.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--output-csv", default="_CheckNotes/Stage3_scaling/scaling.csv")
    parser.add_argument("--output-json", default="_CheckNotes/Stage3_scaling/scaling.json")
    parser.add_argument("--classes", nargs="+", choices=["data", "model"], default=["data", "model"])
    parser.add_argument("--n-full", nargs="+", type=int, default=[40300, 250000, 1000000])
    parser.add_argument("--n-query", nargs="+", type=int, default=[4096, 16384, 65536])
    parser.add_argument("--n-obs", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--data-batch-size", type=int, default=4)
    parser.add_argument("--model-batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    with open(cli.config) as handle:
        config = yaml.safe_load(handle) or {}
    data_path = cli.data or config["data"]
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    base = TurbulentCombustionH5Dataset(
        data_path, split="train", train_ratio=float(config.get("train_ratio", 0.9)),
        seed=cli.seed, time_stride=int(config.get("time_stride", 1)),
        field_names=config.get("FIELD_NAMES", config.get("field_names")),
        stats_path=cli.stats_path, coord_batch_mode="shared_mesh", defer_field_read=True,
    )
    rows: list[dict[str, Any]] = []
    if "data" in cli.classes:
        raw0 = base.read_fields(0)
        for n_full in cli.n_full:
            if int(n_full) == int(base.num_points):
                dataset, source = base, "real_hdf5"
            else:
                dataset, source = ExpandedSnapshotDataset(base, raw0, n_full), "synthetic_expanded_memory"
            for n_query in cli.n_query:
                for n_obs in cli.n_obs:
                    row = benchmark_data_case(
                        dataset=dataset, data_source=source, n_query=min(n_query, n_full), n_obs=n_obs,
                        batch_size=cli.data_batch_size, device=device, iterations=cli.iterations,
                        warmup=cli.warmup, seed=cli.seed,
                    )
                    rows.append(row)
                    print(
                        f"data  Nfull={n_full:>8} Nq={row['N_query']:>6} M={n_obs:>4} "
                        f"pre={row['pre_model_ms']:8.2f}ms rss={row['host_rss_mb']:7.1f}MB"
                    )
    if "model" in cli.classes:
        model = build_model(config, base.num_fields, device)
        model.train()
        for n_query in cli.n_query:
            for n_obs in cli.n_obs:
                row = benchmark_model_case(
                    model=model, n_query=n_query, n_obs=n_obs,
                    batch_size=cli.model_batch_size, n_fields=base.num_fields, device=device,
                    iterations=cli.iterations, warmup=cli.warmup, seed=cli.seed,
                )
                rows.append(row)
                print(
                    f"model Nq={n_query:>6} M={n_obs:>4} status={row['status']} "
                    f"step={float(row['step_ms']):8.2f}ms peak={float(row['gpu_peak_allocated_mb']):8.1f}MB"
                )
    base.close()
    csv_path = Path(cli.output_csv)
    json_path = Path(cli.output_json)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCALING_COLUMNS)
        writer.writeheader()
        writer.writerows({key: row.get(key, "") for key in SCALING_COLUMNS} for row in rows)
    with open(json_path, "w") as handle:
        json.dump({"config": vars(cli), "rows": rows}, handle, indent=2)
    print(f"Saved {len(rows)} rows to {csv_path} and {json_path}")


if __name__ == "__main__":
    main()

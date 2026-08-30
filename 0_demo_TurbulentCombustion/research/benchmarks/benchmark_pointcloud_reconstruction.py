#!/usr/bin/env python3
"""Benchmark legacy-full versus cached-streamed GL-RBF reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import yaml

import _bootstrap  # noqa: F401  # adds the runtime src/ directory

from evaluate_pointcloud_fixed_manifest import build_gl_rbf_ffm
from helpers import TurbulentCombustionH5Dataset, build_sparse_condition
from train_pointcloud_ffm import checkpoint_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-points", type=int, nargs="+", default=[40300, 250000, 1000000])
    parser.add_argument("--legacy-max-points", type=int, default=250000)
    parser.add_argument("--n-obs", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--cache-level", choices=["none", "geometry", "static_features"],
                        default="static_features")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_condition(coords: torch.Tensor, n_fields: int, n_obs: int, seed: int):
    generator = torch.Generator(device=coords.device).manual_seed(seed)
    indices = torch.randperm(coords.shape[1], generator=generator, device=coords.device)[:n_obs]
    indices = indices.sort().values.unsqueeze(0)
    obs_coords = coords[:, indices[0]]
    obs_values = torch.randn(
        coords.shape[0], n_obs, 1, generator=generator,
        device=coords.device, dtype=coords.dtype,
    )
    obs_mask = torch.ones(coords.shape[0], n_obs, device=coords.device, dtype=coords.dtype)
    obs_field_ids = torch.arange(n_obs, device=coords.device).remainder(n_fields).unsqueeze(0)
    return obs_coords, obs_values, obs_mask, indices, obs_field_ids


def benchmark_case(model, coords, condition, mode, cli, device):
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = condition
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(cli.seed + 1000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cli.seed + 1000)
    sync(device)
    start = time.perf_counter()
    output = model.sample(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_steps=cli.n_steps,
        clamp_indices=obs_indices,
        ode_solver="euler",
        obs_consistency_mode="none",
        reconstruction_execution_mode=mode,
        reconstruction_query_chunk_size=cli.chunk_size,
        reconstruction_cache_level=cli.cache_level,
    )
    sync(device)
    wall_s = time.perf_counter() - start
    return output, {
        "execution_mode": mode,
        "status": "ok",
        "N_query": int(coords.shape[1]),
        "N_obs": int(obs_mask.sum()),
        "batch_size": int(coords.shape[0]),
        "n_steps": cli.n_steps,
        "solver": "euler",
        "chunk_size": cli.chunk_size if mode == "cached_streamed" else int(coords.shape[1]),
        "cache_level": cli.cache_level if mode == "cached_streamed" else "none",
        "wall_s": wall_s,
        "seconds_per_million_points_per_nfe": wall_s / (coords.shape[1] / 1e6) / cli.n_steps,
        "gpu_peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
            if device.type == "cuda" else 0.0
        ),
        "gpu_peak_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
            if device.type == "cuda" else 0.0
        ),
        "condition_context_mb": getattr(model, "_last_reconstruction_condition_bytes", 0)
        / (1024.0 ** 2),
        "query_cache_mb": getattr(model, "_last_reconstruction_cache_bytes", 0)
        / (1024.0 ** 2),
    }


def real_snapshot_equivalence(model, dataset, cli, device):
    sample = dataset.get_full_snapshot(0)
    coords = sample["coords"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    torch.manual_seed(cli.seed)
    condition = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=[1],
        n_obs_min=[cli.n_obs],
        n_obs_max=[cli.n_obs],
    )
    kwargs = dict(
        coords=coords,
        obs_coords=condition[0],
        obs_values=condition[1],
        obs_mask=condition[2],
        obs_field_ids=condition[4],
        n_steps=cli.n_steps,
        clamp_indices=condition[3],
        ode_solver="heun",
        obs_consistency_mode="endpoint_smooth",
    )
    torch.manual_seed(cli.seed + 2000)
    legacy = model.sample(**kwargs, reconstruction_execution_mode="legacy_full")
    torch.manual_seed(cli.seed + 2000)
    streamed = model.sample(
        **kwargs,
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=cli.chunk_size,
        reconstruction_cache_level=cli.cache_level,
    )
    difference = streamed - legacy
    legacy_rel_l2 = torch.linalg.vector_norm(legacy - truth) / torch.linalg.vector_norm(truth)
    streamed_rel_l2 = torch.linalg.vector_norm(streamed - truth) / torch.linalg.vector_norm(truth)
    return {
        "N_query": int(coords.shape[1]),
        "solver": "heun",
        "n_steps": cli.n_steps,
        "obs_consistency_mode": "endpoint_smooth",
        "max_abs_difference": float(difference.abs().max()),
        "mean_abs_difference": float(difference.abs().mean()),
        "legacy_relative_l2": float(legacy_rel_l2),
        "streamed_relative_l2": float(streamed_rel_l2),
        "relative_l2_difference": float(streamed_rel_l2 - legacy_rel_l2),
    }


def main() -> None:
    cli = parse_args()
    config = yaml.safe_load(cli.config.read_text()) or {}
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = TurbulentCombustionH5Dataset(
        config["data"], split="val", train_ratio=float(config.get("train_ratio", 0.9)),
        seed=int(config.get("seed", 42)), time_stride=int(config.get("time_stride", 1)),
        field_names=config.get("FIELD_NAMES", config.get("field_names")),
        stats_path=str(cli.stats_path), coord_batch_mode="shared_mesh", defer_field_read=True,
    )
    model = build_gl_rbf_ffm(config, dataset.num_fields, device)
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_model_state(checkpoint), strict=True)
    model.eval()
    rows = []
    for n_points in cli.n_points:
        torch.manual_seed(cli.seed + n_points)
        coords = torch.rand(1, n_points, 3, device=device) * 2.0 - 1.0
        condition = make_condition(coords, dataset.num_fields, cli.n_obs, cli.seed)
        modes = ["cached_streamed"]
        if n_points <= cli.legacy_max_points:
            modes.insert(0, "legacy_full")
        for mode in modes:
            try:
                output, row = benchmark_case(model, coords, condition, mode, cli, device)
                rows.append(row)
                print(
                    f"{mode:15s} N={n_points:>8} wall={row['wall_s']:7.3f}s "
                    f"peak={row['gpu_peak_allocated_mb']:8.1f}MB cache={row['query_cache_mb']:8.1f}MB"
                )
                del output
            except torch.cuda.OutOfMemoryError as exc:
                rows.append({
                    "execution_mode": mode, "status": "oom", "N_query": n_points,
                    "N_obs": cli.n_obs, "batch_size": 1, "n_steps": cli.n_steps,
                    "solver": "euler", "chunk_size": cli.chunk_size,
                    "cache_level": cli.cache_level, "notes": str(exc).replace("\n", " ")[:500],
                })
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        del coords, condition
    equivalence = real_snapshot_equivalence(model, dataset, cli, device)
    dataset.close()
    cli.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cli.output_json.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with open(cli.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with open(cli.output_json, "w") as handle:
        json.dump({"config": vars(cli), "rows": rows, "real_equivalence": equivalence},
                  handle, indent=2, default=str)
    print(f"Real snapshot equivalence: {equivalence}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark persistent Top-K geometry reuse for GL_rbf_ENH_CQ / GL-RBF.

This helper intentionally accepts a model factory by "module:function" so it can
use the exact current local CQ construction path without hard-coding a stale
pre-CQ class.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from helpers import TurbulentCombustionH5Dataset
from persistent_topk_geometry_cache import (
    build_persistent_topk_geometry_cache,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--stats-path", type=Path, required=True)
    p.add_argument(
        "--factory",
        default="evaluate_pointcloud_fixed_manifest:build_gl_rbf_ffm",
        help="Model factory as module:function; must accept (config, n_fields, device).",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--n-points",
        type=int,
        nargs="+",
        default=[250_000, 1_000_000, 1_953_125],
    )
    p.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n-obs", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=8192)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup-repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=1729)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    return p.parse_args()


def resolve_factory(spec: str):
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def checkpoint_state(checkpoint: Any):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    if isinstance(state, dict) and "_metadata" in state:
        state = dict(state)
        state.pop("_metadata", None)
    return state


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_rng(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def make_condition(
    coords: torch.Tensor,
    n_fields: int,
    n_obs: int,
    seed: int,
):
    generator = torch.Generator(device=coords.device).manual_seed(seed)
    indices = torch.randperm(
        coords.shape[1],
        generator=generator,
        device=coords.device,
    )[:n_obs].sort().values.unsqueeze(0)
    obs_coords = coords[:, indices[0]]
    obs_values = torch.randn(
        coords.shape[0],
        n_obs,
        1,
        generator=generator,
        device=coords.device,
        dtype=coords.dtype,
    )
    obs_mask = torch.ones(
        coords.shape[0],
        n_obs,
        device=coords.device,
        dtype=coords.dtype,
    )
    obs_field_ids = (
        torch.arange(n_obs, device=coords.device)
        .remainder(n_fields)
        .unsqueeze(0)
    )
    return {
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_indices": indices,
        "obs_field_ids": obs_field_ids,
    }


class TopKCounter:
    def __init__(self, backbone):
        self.backbone = backbone
        self.original = backbone._get_topk_neighbors
        self.calls = 0

    def install(self):
        def wrapped(*args, **kwargs):
            self.calls += 1
            return self.original(*args, **kwargs)

        self.backbone._get_topk_neighbors = wrapped

    def restore(self):
        self.backbone._get_topk_neighbors = self.original


def timed_sample(
    model,
    *,
    coords,
    condition,
    nfe,
    cache_level,
    geometry_cache,
    chunk_size,
    seed,
    device,
):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    reset_rng(seed, device)
    sync(device)
    start = time.perf_counter()
    output = model.sample(
        coords=coords,
        obs_coords=condition["obs_coords"],
        obs_values=condition["obs_values"],
        obs_mask=condition["obs_mask"],
        obs_field_ids=condition["obs_field_ids"],
        n_steps=nfe,
        clamp_indices=condition["obs_indices"],
        ode_solver="euler",
        obs_consistency_mode="none",
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=chunk_size,
        reconstruction_cache_level=cache_level,
        reconstruction_geometry_cache=geometry_cache,
    )
    sync(device)
    wall_s = time.perf_counter() - start

    return output, {
        "wall_s": wall_s,
        "peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "peak_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / 1024**2
            if device.type == "cuda"
            else 0.0
        ),
        "condition_context_mb": (
            getattr(model, "_last_reconstruction_condition_bytes", 0) / 1024**2
        ),
        "query_cache_mb": (
            getattr(model, "_last_reconstruction_cache_bytes", 0) / 1024**2
        ),
        "condition_context_s": getattr(model, "_last_reconstruction_condition_seconds", 0.0),
        "query_context_s": getattr(model, "_last_reconstruction_query_seconds", 0.0),
        "ode_loop_s": getattr(model, "_last_reconstruction_ode_seconds", 0.0),
    }


def mean(values):
    return sum(values) / max(len(values), 1)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text()) or {}
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset = TurbulentCombustionH5Dataset(
        config["data"],
        split="val",
        train_ratio=float(config.get("train_ratio", 0.9)),
        seed=int(config.get("seed", 42)),
        time_stride=int(config.get("time_stride", 1)),
        field_names=config.get("FIELD_NAMES", config.get("field_names")),
        stats_path=str(args.stats_path),
        coord_batch_mode="shared_mesh",
        defer_field_read=True,
    )

    factory = resolve_factory(args.factory)
    model = factory(config, dataset.num_fields, device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    model.eval()
    model._reconstruction_profile_enabled = True

    backbone = model.model
    if not hasattr(backbone, "_get_topk_neighbors"):
        raise RuntimeError(
            "Current CQ backbone does not expose _get_topk_neighbors. "
            "Adapt the helper to the actual neighbor-search primitive before benchmarking."
        )

    counter = TopKCounter(backbone)
    counter.install()

    warm_n = max(args.n_obs, min(args.chunk_size, min(args.n_points)))
    reset_rng(args.seed + 9, device)
    warm_coords = torch.rand(1, warm_n, 3, device=device) * 2.0 - 1.0
    warm_condition = make_condition(
        warm_coords, dataset.num_fields, args.n_obs, args.seed + 10,
    )
    warm_geometry = model.prepare_reconstruction_geometry_cache(
        coords=warm_coords, obs_coords=warm_condition["obs_coords"],
        obs_mask=warm_condition["obs_mask"], chunk_size=args.chunk_size,
    )
    warm_output, _ = timed_sample(
        model, coords=warm_coords, condition=warm_condition, nfe=1,
        cache_level="static_features", geometry_cache=warm_geometry,
        chunk_size=args.chunk_size, seed=args.seed + 11, device=device,
    )
    del warm_output, warm_geometry, warm_condition, warm_coords
    if device.type == "cuda":
        torch.cuda.empty_cache()
    counter.calls = 0

    rows = []
    summary = []
    try:
        for n_points in args.n_points:
            reset_rng(args.seed + n_points, device)
            coords = torch.rand(
                1,
                n_points,
                3,
                device=device,
                dtype=torch.float32,
            ) * 2.0 - 1.0
            condition = make_condition(
                coords,
                dataset.num_fields,
                args.n_obs,
                args.seed + 17 + n_points,
            )

            calls0 = counter.calls
            sync(device)
            t0 = time.perf_counter()

            # Prefer public wrapper API once integrated. Fall back to supplied
            # helper only for local development before the wrapper is wired.
            if hasattr(model, "prepare_reconstruction_geometry_cache"):
                geometry_cache = model.prepare_reconstruction_geometry_cache(
                    coords=coords,
                    obs_coords=condition["obs_coords"],
                    obs_mask=condition["obs_mask"],
                    chunk_size=args.chunk_size,
                )
            else:
                geometry_cache = build_persistent_topk_geometry_cache(
                    backbone,
                    coords=coords,
                    obs_coords=condition["obs_coords"],
                    obs_mask=condition["obs_mask"],
                    chunk_size=args.chunk_size,
                )

            sync(device)
            geometry_build_s = time.perf_counter() - t0
            geometry_build_topk_calls = counter.calls - calls0
            geometry_cache_mb = (
                geometry_cache.nbytes() / 1024**2
                if hasattr(geometry_cache, "nbytes")
                else backbone.context_nbytes(geometry_cache) / 1024**2
            )

            for nfe in args.nfe:
                modes = [
                    ("geometry_per_call", "geometry", None),
                    ("none", "none", None),
                    ("static_per_call", "static_features", None),
                    ("geometry_persistent", "geometry", geometry_cache),
                    (
                        "static_persistent_geometry",
                        "static_features",
                        geometry_cache,
                    ),
                ]

                mode_rows = []
                reference_outputs = {}

                for label, cache_level, persistent in modes:
                    wall = []
                    alloc = []
                    reserve = []
                    topk_calls = []
                    qcache = []
                    condition_times = []
                    query_times = []
                    ode_times = []
                    diffs = []

                    for warmup in range(max(0, args.warmup_repeats)):
                        warm_output, _ = timed_sample(
                            model, coords=coords, condition=condition, nfe=nfe,
                            cache_level=cache_level, geometry_cache=persistent,
                            chunk_size=args.chunk_size,
                            seed=args.seed + 90_000 + warmup, device=device,
                        )
                        del warm_output

                    for repeat in range(args.repeats):
                        calls_before = counter.calls
                        out, metrics = timed_sample(
                            model,
                            coords=coords,
                            condition=condition,
                            nfe=nfe,
                            cache_level=cache_level,
                            geometry_cache=persistent,
                            chunk_size=args.chunk_size,
                            seed=args.seed + 100_000 + repeat,
                            device=device,
                        )
                        topk_calls.append(counter.calls - calls_before)

                        ref_key = repeat
                        if label == "geometry_per_call":
                            reference_outputs[ref_key] = out.detach().clone()
                            diff = 0.0
                        elif ref_key in reference_outputs:
                            diff = float(
                                (out - reference_outputs[ref_key]).abs().max()
                            )
                        else:
                            diff = float("nan")

                        wall.append(metrics["wall_s"])
                        alloc.append(metrics["peak_allocated_mb"])
                        reserve.append(metrics["peak_reserved_mb"])
                        qcache.append(metrics["query_cache_mb"])
                        condition_times.append(metrics["condition_context_s"])
                        query_times.append(metrics["query_context_s"])
                        ode_times.append(metrics["ode_loop_s"])
                        diffs.append(diff)
                        del out

                    row = {
                        "N_query": n_points,
                        "N_obs": args.n_obs,
                        "NFE": nfe,
                        "warmup_repeats": args.warmup_repeats,
                        "mode": label,
                        "cache_level": cache_level,
                        "mean_wall_s": mean(wall),
                        "min_wall_s": min(wall),
                        "mean_topk_calls": mean(topk_calls),
                        "mean_peak_allocated_mb": mean(alloc),
                        "mean_peak_reserved_mb": mean(reserve),
                        "mean_query_cache_mb": mean(qcache),
                        "mean_condition_context_s": mean(condition_times),
                        "mean_query_context_s": mean(query_times),
                        "mean_ode_loop_s": mean(ode_times),
                        "max_abs_diff_vs_geometry_per_call": (
                            max(v for v in diffs if v == v)
                            if any(v == v for v in diffs)
                            else None
                        ),
                        "geometry_build_s": geometry_build_s,
                        "geometry_build_topk_calls": geometry_build_topk_calls,
                        "geometry_cache_mb": geometry_cache_mb,
                        "amortized_wall_s": mean(wall) + (
                            geometry_build_s / args.repeats
                            if persistent is not None else 0.0
                        ),
                    }
                    rows.append(row)
                    mode_rows.append(row)
                    print(
                        f"N={n_points:,} NFE={nfe} {label:30s} "
                        f"wall={row['mean_wall_s']:.3f}s "
                        f"KNN={row['mean_topk_calls']:.1f}"
                    )

                by = {row["mode"]: row for row in mode_rows}
                base = by["none"]["mean_wall_s"]
                per_geometry = by["geometry_per_call"]["mean_wall_s"]
                persistent_geometry = by["geometry_persistent"]["mean_wall_s"]
                persistent_static = by["static_persistent_geometry"]["mean_wall_s"]

                for row in mode_rows:
                    row["speedup_vs_none"] = base / row["mean_wall_s"]
                    row["speedup_vs_per_call_geometry"] = (
                        per_geometry / row["mean_wall_s"]
                    )

                summary.append(
                    {
                        "N_query": n_points,
                        "NFE": nfe,
                        "speedup_geometry_per_call_vs_none":
                            base / per_geometry,
                        "speedup_persistent_geometry_vs_none":
                            base / persistent_geometry,
                        "speedup_persistent_vs_per_call_geometry":
                            per_geometry / persistent_geometry,
                        "speedup_static_persistent_vs_none":
                            base / persistent_static,
                        "persistent_geometry_amortized_s":
                            persistent_geometry
                            + geometry_build_s / args.repeats,
                        "marginal_context_note":
                            "steady persistent times exclude one-time geometry build",
                    }
                )

            del geometry_cache, condition, coords
            if device.type == "cuda":
                torch.cuda.empty_cache()

    finally:
        counter.restore()
        dataset.close()

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    columns = sorted({key for row in rows for key in row})
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    args.output_json.write_text(
        json.dumps(
            {
                "args": vars(args),
                "rows": rows,
                "summary": summary,
            },
            indent=2,
            default=str,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

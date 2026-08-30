#!/usr/bin/env python3
"""Stage-6 cost benchmark for F0, CQ-LR, and CQ-Balanced."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import yaml

from _bootstrap import ROOT

from evaluate_pointcloud_fixed_manifest import build_gl_rbf_ffm
from train_pointcloud_ffm import checkpoint_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--f0-config",
        type=Path,
        default=ROOT / "_CheckNotes/Stage6_formal_baseline/F0_frozen_current.yaml",
    )
    parser.add_argument(
        "--f0-checkpoint",
        type=Path,
        default=ROOT / "_CheckNotes/Stage6_formal_baseline/runs/"
        "F0_frozen_current_DemoN9300_20260821_075633/best.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-sizes", type=int, nargs="+", default=[4096, 16384, 65536])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--balanced-query-dim", type=int, default=192)
    parser.add_argument("--n-obs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--component-iterations", type=int, default=5)
    parser.add_argument("--million-query-count", type=int, default=1_000_000)
    parser.add_argument("--million-chunk-size", type=int, default=8192)
    parser.add_argument("--million-iterations", type=int, default=3)
    parser.add_argument("--skip-million", action="store_true")
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "_CheckNotes/Stage6_compact_query/benchmarks/cost_benchmark.json",
    )
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(device: torch.device, fn: Callable[[], Any]) -> tuple[Any, float]:
    sync(device)
    start = time.perf_counter()
    value = fn()
    sync(device)
    return value, (time.perf_counter() - start) * 1000.0


def make_inputs(
    n_query: int,
    n_obs: int,
    n_fields: int,
    device: torch.device,
    seed: int,
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    coords = torch.rand(batch_size, n_query, 3, generator=generator, device=device)
    return {
        "coords": coords,
        "x_t": torch.randn(batch_size, n_query, n_fields, generator=generator, device=device),
        "t": torch.rand(batch_size, generator=generator, device=device),
        "obs_coords": torch.rand(batch_size, n_obs, 3, generator=generator, device=device),
        "obs_values": torch.randn(batch_size, n_obs, 1, generator=generator, device=device),
        "obs_mask": torch.ones(batch_size, n_obs, device=device),
        "obs_field_ids": torch.arange(n_obs, device=device).remainder(n_fields).unsqueeze(0).expand(batch_size, -1),
    }


def query_parameter_count(backbone: torch.nn.Module) -> int:
    if hasattr(backbone, "cq_query_dim"):
        prefixes = ("cq_", "query_to_cond", "gather_gate")
    else:
        prefixes = (
            "point_encoder", "query_decoder_token", "query_readout_in",
            "query_latent_readout", "query_readout_out", "query_readout_scale",
            "head", "head_in_norm", "coarse_film", "coarse_head", "coarse_scale",
        )
    return sum(
        parameter.numel()
        for name, parameter in backbone.named_parameters()
        if name.startswith(prefixes)
    )


def model_summary(label: str, model: torch.nn.Module) -> dict[str, Any]:
    backbone = model.model
    total = sum(parameter.numel() for parameter in backbone.parameters())
    query = query_parameter_count(backbone)
    if hasattr(backbone, "model_summary"):
        result = dict(backbone.model_summary())
    else:
        hidden = int(backbone.summary_proj[-1].out_features)
        cond = int(backbone.sensor_out_proj[-1].out_features)
        result = {
            "backbone": "GL_rbf_ENH",
            "query_dim": hidden,
            "latent_dim": int(backbone.latent_dim),
            "cond_dim": cond,
            "readout_mode": "full",
            "readout_rank": None,
            "readout_heads": int(backbone.query_latent_readout.attn.num_heads),
            "point_state_width": hidden,
            "global_width": hidden,
            "local_width": cond,
            "legacy_concat_width": 2 * hidden + cond,
            "cq_fused_width": None,
        }
    result.update(
        label=label,
        total_parameters=total,
        query_decoder_parameters=query,
        condition_core_parameters=total - query,
    )
    return result


def build_models(
    config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    balanced_query_dim: int = 192,
) -> dict[str, torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    models = {}
    torch.manual_seed(42)
    f0 = build_gl_rbf_ffm(config, n_fields=5, device=device)
    f0.load_state_dict(checkpoint_model_state(checkpoint), strict=True)
    models["F0"] = f0
    candidates = (
        ("CQ-LR", 128, "lowrank", "additive"),
        (f"CQ-Balanced-{balanced_query_dim}-Full", balanced_query_dim, "full", "structured_concat"),
    )
    for label, query_dim, readout_mode, fusion_mode in candidates:
        candidate = copy.deepcopy(config)
        candidate.update(
            backbone="GL_rbf_ENH_CQ",
            cq_query_dim=query_dim,
            cq_readout_mode=readout_mode,
            cq_fusion_mode=fusion_mode,
            cq_readout_rank=64,
            cq_readout_heads=4,
            cq_global_scale_init=1.0,
            cq_local_scale_init=1.0,
            cq_readout_scale_init=1.0e-2,
        )
        torch.manual_seed(42)
        models[label] = build_gl_rbf_ffm(candidate, n_fields=5, device=device)
    return models


def benchmark_step(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    *,
    iterations: int,
    warmup: int,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-6)

    def one_step(measure: bool) -> dict[str, float]:
        optimizer.zero_grad(set_to_none=True)
        if measure and device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        prediction, forward_ms = timed(device, lambda: model.model(
            values["t"], values["x_t"], values["coords"],
            values["obs_coords"], values["obs_values"],
            values["obs_mask"], values["obs_field_ids"],
        ))
        loss = prediction.square().mean()
        _, backward_ms = timed(device, loss.backward)
        _, optimizer_ms = timed(device, optimizer.step)
        return {
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "optimizer_ms": optimizer_ms,
            "training_step_ms": forward_ms + backward_ms + optimizer_ms,
            "peak_allocated_mb": (
                torch.cuda.max_memory_allocated(device) / 1024.0**2
                if device.type == "cuda" else 0.0
            ),
            "peak_reserved_mb": (
                torch.cuda.max_memory_reserved(device) / 1024.0**2
                if device.type == "cuda" else 0.0
            ),
        }

    for _ in range(warmup):
        one_step(False)
    rows = [one_step(True) for _ in range(iterations)]
    result = {
        key: statistics.fmean(row[key] for row in rows)
        for key in rows[0]
    }
    n_query = int(values["coords"].shape[1])
    result["queries_per_second_forward"] = n_query / (result["forward_ms"] / 1000.0)
    result["ms_per_1k_queries_forward"] = result["forward_ms"] / (n_query / 1000.0)
    del optimizer
    return result


def component_profile(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    *,
    iterations: int,
    device: torch.device,
) -> dict[str, float]:
    backbone = model.model
    model.eval()

    def average(fn: Callable[[], Any]) -> tuple[Any, float]:
        value, _ = timed(device, fn)
        samples = []
        for _ in range(iterations):
            value, elapsed = timed(device, fn)
            samples.append(elapsed)
        return value, statistics.fmean(samples)

    with torch.no_grad():
        condition, condition_ms = average(lambda: backbone.prepare_condition_context(
            values["obs_coords"], values["obs_values"],
            values["obs_mask"], values["obs_field_ids"],
        ))
        coord_feat, coord_ms = average(lambda: (
            backbone.pos_enc(values["coords"])
            if backbone.pos_enc is not None else values["coords"]
        ))
        batch_size = int(values["coords"].shape[0])
        t_feat = values["t"].view(batch_size, 1, 1).expand(batch_size, values["coords"].shape[1], 1)
        if hasattr(backbone, "cq_query_dim"):
            point, point_ms = average(lambda: backbone.cq_point_encoder(torch.cat(
                [coord_feat, values["x_t"], t_feat], dim=-1,
            )))
            readout, readout_ms = average(
                lambda: backbone._cq_readout_chunked(coord_feat, condition)
            )
        else:
            point, point_ms = average(lambda: backbone.point_encoder(torch.cat(
                [coord_feat, values["x_t"], t_feat], dim=-1,
            )))
            readout, readout_ms = average(
                lambda: backbone._readout_query_global_chunked(
                    point, values["coords"], condition["latents"],
                )
            )
        local, local_ms = average(lambda: backbone.aggregate_sparse_obs(
            values["coords"], point, condition["obs_coords"],
            condition["refined_sensor_feat"], condition["obs_mask"],
            condition.get("sensor_importance_bias"),
        ))

        if hasattr(backbone, "cq_query_dim"):
            global_q = condition["global_q"]

            def fusion_head() -> torch.Tensor:
                if backbone.cq_fusion_mode == "structured_concat":
                    global_for_head = (
                        global_q.unsqueeze(1)
                        + backbone.cq_readout_scale * readout
                    )
                    head_input = torch.cat(
                        [point, global_for_head, local], dim=-1,
                    )
                else:
                    head_input = (
                        point
                        + backbone.cq_global_scale * global_q.unsqueeze(1)
                        + backbone.cq_local_scale * backbone.cq_local_proj(local)
                        + backbone.cq_readout_scale * readout
                    )
                return backbone.cq_head(backbone.cq_fusion_norm(head_input))

            def coarse() -> torch.Tensor:
                return backbone.cq_coarse_scale * backbone._predict_cq_coarse(
                    point, global_q,
                )
        else:
            global_feat = condition["global_feat"]
            global_for_head = (
                global_feat.unsqueeze(1)
                + backbone.query_readout_scale * readout
            )

            def fusion_head() -> torch.Tensor:
                head_input = torch.cat([point, global_for_head, local], dim=-1)
                return backbone.head(backbone.head_in_norm(head_input))

            def coarse() -> torch.Tensor:
                return backbone.coarse_scale * backbone._predict_global_coarse(
                    point, global_feat,
                )

        _, fusion_head_ms = average(fusion_head)
        _, coarse_ms = average(coarse)
    return {
        "condition_encoding_ms": condition_ms,
        "coordinate_encoding_ms": coord_ms,
        "point_encoder_ms": point_ms,
        "latent_readout_ms": readout_ms,
        "local_gather_ms": local_ms,
        "fusion_head_ms": fusion_head_ms,
        "glres_coarse_ms": coarse_ms,
    }


def benchmark_million(
    label: str,
    model: torch.nn.Module,
    *,
    n_query: int,
    n_obs: int,
    chunk_size: int,
    device: torch.device,
) -> dict[str, Any]:
    values = make_inputs(n_query, n_obs, 5, device, seed=1_000_003)
    model.eval()
    sync(device)
    geometry_start = time.perf_counter()
    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"],
        chunk_size=chunk_size,
    )
    sync(device)
    geometry_build_s = time.perf_counter() - geometry_start
    geometry_mb = geometry.nbytes() / 1024.0**2
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(1042)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1042)
    sync(device)
    start = time.perf_counter()
    with torch.no_grad():
        output = model.sample(
            coords=values["coords"],
            obs_coords=values["obs_coords"],
            obs_values=values["obs_values"],
            obs_mask=values["obs_mask"],
            obs_field_ids=values["obs_field_ids"],
            n_steps=4,
            ode_solver="euler",
            obs_consistency_mode="none",
            reconstruction_execution_mode="cached_streamed",
            reconstruction_query_chunk_size=chunk_size,
            reconstruction_cache_level="static_features",
            reconstruction_geometry_cache=geometry,
        )
    sync(device)
    wall_s = time.perf_counter() - start
    cache_mb = getattr(model, "_last_reconstruction_cache_bytes", 0) / 1024.0**2
    condition_mb = getattr(model, "_last_reconstruction_condition_bytes", 0) / 1024.0**2
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) / 1024.0**2
        if device.type == "cuda" else 0.0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) / 1024.0**2
        if device.type == "cuda" else 0.0
    )
    del output, values
    return {
        "label": label,
        "status": "ok",
        "N_query": n_query,
        "N_obs": n_obs,
        "nfe": 4,
        "solver": "euler",
        "chunk_size": chunk_size,
        "cache_level": "static_persistent_geometry",
        "geometry_build_s": geometry_build_s,
        "geometry_cache_mb": geometry_mb,
        "wall_s": wall_s,
        "seconds_per_million_queries_per_nfe": wall_s / (n_query / 1e6) / 4.0,
        "peak_allocated_mb": peak_allocated,
        "peak_reserved_mb": peak_reserved,
        "condition_context_mb": condition_mb,
        "static_query_cache_mb": cache_mb,
        "dynamic_peak_above_static_cache_mb": max(0.0, peak_allocated - cache_mb),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Stage-6 cost gates require a CUDA device.")
    torch.cuda.set_device(device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if not args.allow_busy_gpu and free_bytes / 1024.0**2 < 30_000:
        raise RuntimeError(
            f"Refusing interference-sensitive benchmark: only {free_bytes / 1024.0**2:.0f} MiB free."
        )

    config = yaml.safe_load(args.f0_config.read_text())
    models = build_models(
        config, args.f0_checkpoint, device, args.balanced_query_dim,
    )
    summaries = [model_summary(label, model) for label, model in models.items()]
    scaling_rows = []
    for label, model in models.items():
        for n_query in args.query_sizes:
            values = make_inputs(
                n_query, args.n_obs, 5, device, seed=10_000 + n_query,
                batch_size=args.batch_size,
            )
            try:
                row = {
                    "label": label,
                    "status": "ok",
                    "N_query": n_query,
                    "N_obs": args.n_obs,
                    "batch_size": args.batch_size,
                    **benchmark_step(
                        model, values, iterations=args.iterations,
                        warmup=args.warmup, device=device,
                    ),
                    **component_profile(
                        model, values, iterations=args.component_iterations, device=device,
                    ),
                }
            except torch.cuda.OutOfMemoryError as exc:
                row = {
                    "label": label,
                    "status": "oom",
                    "N_query": n_query,
                    "N_obs": args.n_obs,
                    "error": str(exc).replace("\n", " ")[:500],
                }
                torch.cuda.empty_cache()
            scaling_rows.append(row)
            print(json.dumps(row, sort_keys=True))
            del values

    million_rows = []
    if not args.skip_million:
        for label, model in models.items():
            try:
                samples = [
                    benchmark_million(
                        label,
                        model,
                        n_query=args.million_query_count,
                        n_obs=args.n_obs,
                        chunk_size=args.million_chunk_size,
                        device=device,
                    )
                    for _ in range(args.million_iterations)
                ]
                row = dict(samples[0])
                averaged = (
                    "wall_s", "seconds_per_million_queries_per_nfe",
                    "condition_context_mb", "static_query_cache_mb",
                    "geometry_build_s", "geometry_cache_mb",
                )
                peaked = (
                    "peak_allocated_mb", "peak_reserved_mb",
                    "dynamic_peak_above_static_cache_mb",
                )
                for key in averaged:
                    values = [float(sample[key]) for sample in samples]
                    row[key] = statistics.fmean(values)
                    row[f"{key}_std"] = (
                        statistics.stdev(values) if len(values) > 1 else 0.0
                    )
                    row[f"{key}_samples"] = values
                for key in peaked:
                    values = [float(sample[key]) for sample in samples]
                    row[key] = max(values)
                    row[f"{key}_samples"] = values
                row["iterations"] = int(args.million_iterations)
            except torch.cuda.OutOfMemoryError as exc:
                row = {
                    "label": label,
                    "status": "oom",
                    "N_query": args.million_query_count,
                    "error": str(exc).replace("\n", " ")[:500],
                }
                torch.cuda.empty_cache()
            million_rows.append(row)
            print(json.dumps(row, sort_keys=True))

    result = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "batch_size": args.batch_size,
        "f0_config": str(args.f0_config.resolve()),
        "f0_checkpoint": str(args.f0_checkpoint.resolve()),
        "model_summaries": summaries,
        "scaling": scaling_rows,
        "million_query_reconstruction": million_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    rows = scaling_rows + [
        {"benchmark": "million_reconstruction", **row} for row in million_rows
    ]
    columns = sorted({key for row in rows for key in row})
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

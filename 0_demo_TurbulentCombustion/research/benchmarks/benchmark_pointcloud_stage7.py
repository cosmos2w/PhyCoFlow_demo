#!/usr/bin/env python3
"""Stage-7 Smart-CQ training/scaling and persistent-inference benchmark."""

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

from benchmark_pointcloud_cq import make_inputs, model_summary, sync, timed
from evaluate_pointcloud_fixed_manifest import build_gl_rbf_ffm
from model_ema import ModelEMA
from train_pointcloud_ffm import checkpoint_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f0-config", type=Path, required=True)
    parser.add_argument("--f0-checkpoint", type=Path, required=True)
    parser.add_argument("--cq-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n-obs", type=int, default=256)
    parser.add_argument("--query-sizes", type=int, nargs="+", default=[4096, 16384, 65536])
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--formal-batch-size", type=int, default=128)
    parser.add_argument("--formal-query-size", type=int, default=4096)
    parser.add_argument("--million-query-count", type=int, default=1_000_000)
    parser.add_argument("--million-chunk-size", type=int, default=32768)
    parser.add_argument("--skip-million", action="store_true")
    parser.add_argument("--minimum-free-mb", type=float, default=30_000.0)
    parser.add_argument("--allow-busy-gpu", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "_CheckNotes/Stage7_smart_cq/benchmarks/pretraining_cost.json",
    )
    return parser.parse_args()


def _candidate_config(base: dict[str, Any], *, latent_dim: int, smart: bool) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config.update(
        backbone="GL_rbf_ENH_CQ",
        latent_dim=latent_dim,
        num_latents=128,
        num_latent_blocks=4,
        cq_query_dim=128,
        cq_readout_mode="lowrank",
        cq_readout_rank=64,
        cq_readout_heads=4,
        cq_fusion_mode="additive",
        cq_time_conditioning="sinusoidal_film" if smart else "scalar_concat",
        cq_time_embed_dim=128,
        cq_time_max_period=10000.0,
        cq_time_film_zero_init=True,
        cq_measurement_support_mode="rbf_value_support" if smart else "none",
        cq_measurement_support_normalize=True,
        model_ema_enabled=smart,
        model_ema_decay=0.999,
        model_ema_eval=True,
        train_query_microbatch_size=2048 if smart else config.get("train_query_microbatch_size"),
    )
    return config


def build_candidates(args: argparse.Namespace, device: torch.device):
    f0_config = yaml.safe_load(args.f0_config.read_text())
    cq_config = yaml.safe_load(args.cq_config.read_text())
    checkpoint = torch.load(args.f0_checkpoint, map_location="cpu", weights_only=False)
    torch.manual_seed(42)
    f0 = build_gl_rbf_ffm(f0_config, n_fields=5, device=device)
    f0.load_state_dict(checkpoint_model_state(checkpoint), strict=True)
    definitions = {
        "F0": (f0, False, f0_config),
        "Frozen-CQ-LR-128": (None, False, _candidate_config(cq_config, latent_dim=128, smart=False)),
        "Stage7-Cond128": (None, True, _candidate_config(cq_config, latent_dim=128, smart=True)),
        "Stage7-All256": (None, True, _candidate_config(cq_config, latent_dim=256, smart=True)),
    }
    result = {}
    for label, (model, ema_enabled, config) in definitions.items():
        if model is None:
            torch.manual_seed(42)
            model = build_gl_rbf_ffm(config, n_fields=5, device=device)
        result[label] = (model, ema_enabled, config)
    return result


def benchmark_training_step(
    model: torch.nn.Module,
    values: dict[str, torch.Tensor],
    *,
    ema_enabled: bool,
    iterations: int,
    warmup: int,
    device: torch.device,
    query_microbatch_size: int | None = None,
) -> dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-6)
    ema = ModelEMA(model, decay=0.999) if ema_enabled else None
    microbatch_active = (
        query_microbatch_size is not None
        and 0 < int(query_microbatch_size) < int(values["coords"].shape[1])
    )

    def one_step(measure: bool) -> dict[str, float]:
        optimizer.zero_grad(set_to_none=True)
        if measure and device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        full_start = time.perf_counter()
        if microbatch_active:
            (_, metrics), _ = timed(device, lambda: model.training_loss_microbatched(
                x1=values["x_t"], coords=values["coords"],
                obs_coords=values["obs_coords"], obs_values=values["obs_values"],
                obs_mask=values["obs_mask"], obs_field_ids=values["obs_field_ids"],
                obs_indices=None, query_microbatch_size=int(query_microbatch_size),
                backward=True, reuse_condition_context=True,
                synchronize_timing=True,
            ))
            forward_ms = (
                metrics["rf_bridge_ms"] + metrics["condition_context_ms"]
                + metrics["query_chunk_forward_ms"]
            )
            backward_ms = metrics["query_chunk_backward_ms"]
        else:
            (loss, _), forward_ms = timed(device, lambda: model.training_loss(
                x1=values["x_t"], coords=values["coords"],
                obs_coords=values["obs_coords"], obs_values=values["obs_values"],
                obs_mask=values["obs_mask"], obs_field_ids=values["obs_field_ids"],
                obs_indices=None,
            ))
            _, backward_ms = timed(device, loss.backward)
        _, grad_clip_ms = timed(
            device, lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0),
        )
        _, optimizer_ms = timed(device, optimizer.step)
        ema_ms = 0.0
        if ema is not None:
            _, ema_ms = timed(device, lambda: ema.update(model))
        sync(device)
        full_step_ms = (time.perf_counter() - full_start) * 1000.0
        return {
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "grad_clip_ms": grad_clip_ms,
            "optimizer_ms": optimizer_ms,
            "ema_update_ms": ema_ms,
            "full_step_ms": full_step_ms,
            "query_microbatch_size": int(query_microbatch_size or 0),
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
    result = {key: statistics.fmean(row[key] for row in rows) for key in rows[0]}
    result["full_step_samples_ms"] = [row["full_step_ms"] for row in rows]
    del optimizer, ema
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

    def average(fn: Callable[[], Any]):
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
        _, query_ms = average(lambda: backbone.forward_query_chunk(
            values["t"], values["x_t"], values["coords"], condition,
        ))
    return {
        "condition_context_ms": condition_ms,
        "query_forward_ms": query_ms,
    }


def persistent_inference(
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
    start = time.perf_counter()
    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"], obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"], chunk_size=chunk_size,
    )
    sync(device)
    row: dict[str, Any] = {
        "label": label,
        "status": "ok",
        "N_query": n_query,
        "N_obs": n_obs,
        "solver": "euler",
        "cache_level": "static_features",
        "geometry_build_s": time.perf_counter() - start,
        "geometry_cache_mb": geometry.nbytes() / 1024.0**2,
    }
    for nfe in (2, 4):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(1042)
        torch.cuda.manual_seed_all(1042)
        sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            output = model.sample(
                coords=values["coords"], obs_coords=values["obs_coords"],
                obs_values=values["obs_values"], obs_mask=values["obs_mask"],
                obs_field_ids=values["obs_field_ids"], n_steps=nfe,
                ode_solver="euler", obs_consistency_mode="none",
                reconstruction_execution_mode="cached_streamed",
                reconstruction_query_chunk_size=chunk_size,
                reconstruction_cache_level="static_features",
                reconstruction_geometry_cache=geometry,
            )
        sync(device)
        row[f"steady_nfe{nfe}_s"] = time.perf_counter() - start
        row[f"steady_nfe{nfe}_peak_allocated_mb"] = (
            torch.cuda.max_memory_allocated(device) / 1024.0**2
        )
        row[f"steady_nfe{nfe}_peak_reserved_mb"] = (
            torch.cuda.max_memory_reserved(device) / 1024.0**2
        )
        row[f"steady_nfe{nfe}_condition_context_mb"] = (
            getattr(model, "_last_reconstruction_condition_bytes", 0) / 1024.0**2
        )
        row[f"steady_nfe{nfe}_static_query_cache_mb"] = (
            getattr(model, "_last_reconstruction_cache_bytes", 0) / 1024.0**2
        )
        del output
    del values, geometry
    return row


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Stage-7 cost gates require CUDA.")
    torch.cuda.set_device(device)
    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_mb = free_bytes / 1024.0**2
    if not args.allow_busy_gpu and free_mb < args.minimum_free_mb:
        raise RuntimeError(
            f"Refusing interference-sensitive benchmark: only {free_mb:.0f} MiB free "
            f"(< {args.minimum_free_mb:.0f} MiB)."
        )

    candidates = build_candidates(args, device)
    summaries = [model_summary(label, model) for label, (model, _, _) in candidates.items()]
    formal_rows = []
    scaling_rows = []
    for label, (model, ema_enabled, config) in candidates.items():
        values = make_inputs(
            args.formal_query_size, args.n_obs, 5, device, seed=42,
            batch_size=args.formal_batch_size,
        )
        try:
            row = {
                "label": label, "status": "ok",
                "batch_size": args.formal_batch_size,
                "N_query": args.formal_query_size,
                **benchmark_training_step(
                    model, values, ema_enabled=ema_enabled,
                    iterations=args.iterations, warmup=args.warmup, device=device,
                    query_microbatch_size=config.get("train_query_microbatch_size"),
                ),
            }
        except torch.cuda.OutOfMemoryError as exc:
            row = {"label": label, "status": "oom", "error": str(exc)[:500]}
            torch.cuda.empty_cache()
        formal_rows.append(row)
        print(json.dumps({"benchmark": "formal_step", **row}, sort_keys=True))
        del values

        for n_query in args.query_sizes:
            values = make_inputs(n_query, args.n_obs, 5, device, seed=10_000 + n_query)
            try:
                row = {
                    "label": label, "status": "ok", "batch_size": 1,
                    "N_query": n_query,
                    **benchmark_training_step(
                        model, values, ema_enabled=ema_enabled,
                        iterations=args.iterations, warmup=args.warmup, device=device,
                    ),
                    **component_profile(model, values, iterations=args.iterations, device=device),
                }
            except torch.cuda.OutOfMemoryError as exc:
                row = {"label": label, "status": "oom", "N_query": n_query, "error": str(exc)[:500]}
                torch.cuda.empty_cache()
            scaling_rows.append(row)
            print(json.dumps({"benchmark": "scaling", **row}, sort_keys=True))
            del values

    million_rows = []
    if not args.skip_million:
        for label, (model, _, _) in candidates.items():
            try:
                row = persistent_inference(
                    label, model, n_query=args.million_query_count,
                    n_obs=args.n_obs, chunk_size=args.million_chunk_size, device=device,
                )
            except torch.cuda.OutOfMemoryError as exc:
                row = {"label": label, "status": "oom", "error": str(exc)[:500]}
                torch.cuda.empty_cache()
            million_rows.append(row)
            print(json.dumps({"benchmark": "persistent", **row}, sort_keys=True))

    by_label = {row["label"]: row for row in formal_rows if row.get("status") == "ok"}
    persistent_by_label = {
        row["label"]: row for row in million_rows if row.get("status") == "ok"
    }
    f0 = by_label.get("F0")
    f0_persistent = persistent_by_label.get("F0")
    gates = {}
    if f0:
        for label, row in by_label.items():
            if label == "F0":
                continue
            speedup = f0["full_step_ms"] / row["full_step_ms"]
            memory_reduction = 1.0 - row["peak_allocated_mb"] / f0["peak_allocated_mb"]
            gates[label] = {
                "train_speedup_vs_f0": speedup,
                "peak_memory_reduction_vs_f0": memory_reduction,
                "passes_train_speed_gate_1p10x": speedup >= 1.10,
                "passes_memory_reduction_gate_0p10": memory_reduction >= 0.10,
            }
            if f0_persistent and label in persistent_by_label:
                persistent_speedup = (
                    f0_persistent["steady_nfe4_s"]
                    / persistent_by_label[label]["steady_nfe4_s"]
                )
                gates[label]["persistent_nfe4_speedup_vs_f0"] = persistent_speedup
                gates[label]["passes_persistent_nfe4_gate_1p15x"] = persistent_speedup >= 1.15
                gates[label]["scientific_launch_eligible"] = (
                    speedup >= 1.10 and memory_reduction >= 0.10 and persistent_speedup >= 1.15
                )
    result = {
        "device": str(device), "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__, "free_mb_at_start": free_mb,
        "f0_config": str(args.f0_config.resolve()),
        "f0_checkpoint": str(args.f0_checkpoint.resolve()),
        "cq_config": str(args.cq_config.resolve()),
        "model_summaries": summaries,
        "formal_training_step": formal_rows,
        "model_scaling": scaling_rows,
        "persistent_inference": million_rows,
        "efficiency_gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    rows = (
        [{"benchmark": "formal_step", **row} for row in formal_rows]
        + [{"benchmark": "scaling", **row} for row in scaling_rows]
        + [{"benchmark": "persistent", **row} for row in million_rows]
    )
    columns = sorted({key for row in rows for key in row})
    with open(args.output.with_suffix(".csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

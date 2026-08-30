#!/usr/bin/env python3
"""Measure effective-query training time and memory versus query microbatch size."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch
import yaml

import _bootstrap  # noqa: F401  # adds the runtime src/ directory

from evaluate_pointcloud_fixed_manifest import build_gl_rbf_ffm
from train_pointcloud_ffm import checkpoint_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n-query", type=int, nargs="+", default=[4096, 16384, 65536])
    parser.add_argument("--microbatch", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--n-obs", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows)


def make_inputs(n_query, n_obs, batch_size, n_fields, device, seed):
    torch.manual_seed(seed + n_query)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed + n_query)
    return {
        "coords": torch.rand(batch_size, n_query, 3, device=device) * 2.0 - 1.0,
        "x1": torch.randn(batch_size, n_query, n_fields, device=device),
        "obs_coords": torch.rand(batch_size, n_obs, 3, device=device) * 2.0 - 1.0,
        "obs_values": torch.randn(batch_size, n_obs, 1, device=device),
        "obs_mask": torch.ones(batch_size, n_obs, device=device),
        "obs_field_ids": torch.zeros(batch_size, n_obs, dtype=torch.long, device=device),
        "obs_indices": torch.zeros(batch_size, n_obs, dtype=torch.long, device=device),
    }


def benchmark_case(model, initial_state, values, microbatch_size, cli, device):
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    rows = []
    n_query = values["coords"].shape[1]
    micro_active = microbatch_size is not None and microbatch_size < n_query
    for iteration in range(cli.warmup + cli.iterations):
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(cli.seed + iteration)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cli.seed + iteration)
        sync(device)
        start = time.perf_counter()
        if micro_active:
            loss, metrics = model.training_loss_microbatched(
                **values,
                query_microbatch_size=microbatch_size,
                backward=True,
                reuse_condition_context=True,
                synchronize_timing=True,
            )
        else:
            loss, _ = model.training_loss(**values)
            sync(device)
            loss.backward()
            metrics = {}
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        sync(device)
        step_ms = (time.perf_counter() - start) * 1000.0
        if iteration >= cli.warmup:
            rows.append({
                "step_ms": step_ms,
                "gpu_peak_allocated_mb": (
                    torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
                    if device.type == "cuda" else 0.0
                ),
                "gpu_peak_reserved_mb": (
                    torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
                    if device.type == "cuda" else 0.0
                ),
                "rf_bridge_ms": metrics.get("rf_bridge_ms", 0.0),
                "condition_context_ms": metrics.get("condition_context_ms", 0.0),
                "query_chunk_forward_ms": metrics.get("query_chunk_forward_ms", 0.0),
                "query_chunk_backward_ms": metrics.get("query_chunk_backward_ms", 0.0),
                "loss": float(loss),
            })
    step_ms = mean(rows, "step_ms")
    return {
        "N_query_effective": n_query,
        "query_microbatch_size": 0 if not micro_active else microbatch_size,
        "execution": "monolithic" if not micro_active else "microbatched",
        "batch_size": cli.batch_size,
        "N_obs": cli.n_obs,
        "iterations": cli.iterations,
        "step_ms": step_ms,
        "queries_per_sec": cli.batch_size * n_query / (step_ms / 1000.0),
        "gpu_peak_allocated_mb": mean(rows, "gpu_peak_allocated_mb"),
        "gpu_peak_reserved_mb": mean(rows, "gpu_peak_reserved_mb"),
        "rf_bridge_ms": mean(rows, "rf_bridge_ms"),
        "condition_context_ms": mean(rows, "condition_context_ms"),
        "query_chunk_forward_ms": mean(rows, "query_chunk_forward_ms"),
        "query_chunk_backward_ms": mean(rows, "query_chunk_backward_ms"),
        "loss_mean": mean(rows, "loss"),
    }


def main() -> None:
    cli = parse_args()
    config = yaml.safe_load(cli.config.read_text()) or {}
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    initial_state = checkpoint_model_state(checkpoint)
    n_fields = int(checkpoint.get("mean", torch.empty(5)).numel())
    model = build_gl_rbf_ffm(config, n_fields, device)
    model.train()
    rows = []
    for n_query in cli.n_query:
        values = make_inputs(
            n_query, cli.n_obs, cli.batch_size, n_fields, device, cli.seed,
        )
        cases = [None] + [size for size in cli.microbatch if size < n_query]
        for microbatch_size in cases:
            row = benchmark_case(
                model, initial_state, values, microbatch_size, cli, device,
            )
            rows.append(row)
            print(
                f"Nq={n_query:>6} micro={row['query_microbatch_size']:>5} "
                f"step={row['step_ms']:8.2f}ms peak={row['gpu_peak_allocated_mb']:8.1f}MB "
                f"q/s={row['queries_per_sec']:9.0f}"
            )
        del values
    cli.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cli.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(cli.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(cli.output_json, "w") as handle:
        json.dump({"config": vars(cli), "rows": rows}, handle, indent=2, default=str)


if __name__ == "__main__":
    main()

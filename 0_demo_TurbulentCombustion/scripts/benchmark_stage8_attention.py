#!/usr/bin/env python3
"""Benchmark Stage-8 condition-attention execution at the validated real shape."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phycoflow_pointcloud.models.factory import build_pointcloud_model

MODES = (
    ("A_legacy_full", "legacy_mha", "full", (256, 320, 384)),
    ("B_cached_full", "cached_kv", "full", (256, 320, 384)),
    ("C_cached_buckets_3", "cached_kv", "static_buckets", (256, 320, 384)),
    (
        "D_dynamic_trim_diagnostic",
        "cached_kv",
        "static_buckets",
        tuple(range(192, 385)),
    ),
    ("bucket_2", "cached_kv", "static_buckets", (288, 384)),
    ("bucket_4", "cached_kv", "static_buckets", (224, 288, 336, 384)),
)


def _inputs(
    device: torch.device,
    *,
    batch_size: int,
    query_count: int,
    max_sensors: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(8088)
    counts = torch.linspace(192, max_sensors, batch_size).round().long()
    mask = torch.arange(max_sensors).unsqueeze(0) < counts.unsqueeze(1)
    obs_coords = torch.rand(batch_size, max_sensors, 3, generator=generator)
    obs_values = torch.randn(batch_size, max_sensors, 1, generator=generator)
    obs_coords = obs_coords * mask.unsqueeze(-1)
    obs_values = obs_values * mask.unsqueeze(-1)
    return {
        "x1": torch.randn(batch_size, query_count, 5, generator=generator).to(device),
        "coords": torch.rand(batch_size, query_count, 3, generator=generator).to(device),
        "obs_coords": obs_coords.to(device),
        "obs_values": obs_values.to(device),
        "obs_mask": mask.float().to(device),
        "obs_field_ids": (
            torch.arange(max_sensors).remainder(5).expand(batch_size, -1).to(device)
        ),
        "obs_indices": (
            torch.arange(max_sensors).expand(batch_size, -1).to(device)
        ),
        "valid_sensor_counts": counts.tolist(),
    }


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def _time_cuda(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    value = fn()
    end.record()
    end.synchronize()
    del value
    return float(start.elapsed_time(end))


def _condition_breakdown(model, values: dict[str, Any], repeats: int) -> dict[str, float]:
    backbone = model.model
    sensor_times: list[float] = []
    encode_times: list[float] = []
    latent_times: list[float] = []
    back_times: list[float] = []
    condition_times: list[float] = []
    with torch.no_grad():
        sensor_tokens = backbone._build_sensor_tokens(
            values["obs_coords"], values["obs_values"],
            values["obs_mask"], values["obs_field_ids"],
        )
        latents = backbone._encode_latents(sensor_tokens, values["obs_mask"])
        for _ in range(repeats):
            sensor_times.append(_time_cuda(lambda: backbone._build_sensor_tokens(
                values["obs_coords"], values["obs_values"],
                values["obs_mask"], values["obs_field_ids"],
            )))
            encode_times.append(_time_cuda(
                lambda: backbone._encode_latents(sensor_tokens, values["obs_mask"])
            ))
            latent_times.append(_time_cuda(lambda: _run_latent_blocks(backbone, latents)))
            back_times.append(_time_cuda(
                lambda: backbone._refine_sensor_tokens(
                    sensor_tokens, latents, values["obs_mask"]
                )
            ))
            condition_times.append(_time_cuda(lambda: backbone.prepare_condition_context(
                values["obs_coords"], values["obs_values"],
                values["obs_mask"], values["obs_field_ids"],
            )))
    latent_ms = _median(latent_times)
    encode_ms = _median(encode_times)
    return {
        "sensor_tokenization_ms": _median(sensor_times),
        "sensor_to_latent_attention_ms": max(0.0, encode_ms - latent_ms),
        "latent_self_attention_ms": latent_ms,
        "sensor_back_attention_ms": _median(back_times),
        "condition_context_total_ms": _median(condition_times),
    }


def _run_latent_blocks(backbone, latents: torch.Tensor) -> torch.Tensor:
    output = latents
    for block in backbone.latent_blocks:
        output = block(output)
    return output


def _whole_step(
    model,
    optimizer,
    values: dict[str, Any],
    *,
    query_microbatch: int,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(values["x1"].device)
    start = torch.cuda.Event(enable_timing=True)
    before_optimizer = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    _, metrics = model.training_loss_microbatched(
        **{key: value for key, value in values.items() if key != "valid_sensor_counts"},
        query_microbatch_size=query_microbatch,
        backward=True,
        reuse_condition_context=True,
        synchronize_timing=True,
    )
    before_optimizer.record()
    optimizer.step()
    end.record()
    end.synchronize()
    return {
        "condition_context_ms": float(metrics["condition_context_ms"]),
        "query_forward_ms": float(metrics["query_chunk_forward_ms"]),
        "backward_ms": float(metrics["query_chunk_backward_ms"]),
        "optimizer_ms": float(before_optimizer.elapsed_time(end)),
        "whole_step_ms": float(start.elapsed_time(end)),
        "peak_allocated_mib": float(
            torch.cuda.max_memory_allocated(values["x1"].device) / 2**20
        ),
        "peak_reserved_mib": float(
            torch.cuda.max_memory_reserved(values["x1"].device) / 2**20
        ),
    }


def _profile_once(model, optimizer, values, query_microbatch: int, path: Path) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    optimizer.zero_grad(set_to_none=True)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profile:
        model.training_loss_microbatched(
            **{key: value for key, value in values.items() if key != "valid_sensor_counts"},
            query_microbatch_size=query_microbatch,
            backward=True,
            reuse_condition_context=True,
        )
        optimizer.step()
    path.parent.mkdir(parents=True, exist_ok=True)
    profile.export_chrome_trace(str(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gl_rbf_cq.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "_CheckNotes/Stage8_attention_optimization",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-count", type=int, default=4096)
    parser.add_argument("--query-microbatch", type=int, default=2048)
    parser.add_argument("--max-sensors", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[mode[0] for mode in MODES],
        help="Optional subset of benchmark modes (default: all).",
    )
    parser.add_argument("--profile-best", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("Stage-8 real-shape benchmark requires CUDA.")
    torch.cuda.set_device(device)
    config = yaml.safe_load(args.config.read_text())
    config["neighbor_backend"] = "keops"
    torch.manual_seed(int(config.get("seed", 42)))
    base = build_pointcloud_model(config, n_fields=5, device="cpu")
    base_state = base.state_dict()
    values = _inputs(
        device,
        batch_size=args.batch_size,
        query_count=args.query_count,
        max_sensors=args.max_sensors,
    )
    rows: list[dict[str, Any]] = []
    selected_modes = [
        mode for mode in MODES if args.modes is None or mode[0] in args.modes
    ]
    for name, execution, padding, buckets in selected_modes:
        mode_config = dict(config)
        mode_config.update(
            condition_attention_execution=execution,
            sensor_attention_padding_mode=padding,
            sensor_attention_buckets=list(buckets),
        )
        model = build_pointcloud_model(mode_config, n_fields=5, device=device).train()
        model.load_state_dict(base_state, strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config.get("lr", 1.0e-4)),
            weight_decay=float(config.get("weight_decay", 1.0e-6)),
        )
        for _ in range(args.warmup):
            _whole_step(
                model, optimizer, values, query_microbatch=args.query_microbatch
            )
        samples = [
            _whole_step(model, optimizer, values, query_microbatch=args.query_microbatch)
            for _ in range(args.steps)
        ]
        model.model.input_cross_attn.reset_execution_counters()
        with torch.no_grad():
            model.model.prepare_condition_context(
                values["obs_coords"], values["obs_values"],
                values["obs_mask"], values["obs_field_ids"],
            )
        row: dict[str, Any] = {
            "mode": name,
            "condition_attention_execution": execution,
            "sensor_attention_padding_mode": padding,
            "sensor_attention_buckets": list(buckets),
            "kv_projection_calls": model.model.input_cross_attn.kv_projection_calls,
        }
        for key in samples[0]:
            row[key] = _median([sample[key] for sample in samples])
        row.update(_condition_breakdown(model, values, max(3, args.steps // 2)))
        rows.append(row)
        del optimizer, model
        torch.cuda.empty_cache()

    oracle_ms = rows[0]["whole_step_ms"]
    oracle_memory = rows[0]["peak_allocated_mib"]
    for row in rows:
        row["whole_step_speedup_percent_vs_legacy"] = (
            100.0 * (oracle_ms - row["whole_step_ms"]) / oracle_ms
        )
        row["peak_allocated_change_percent_vs_legacy"] = (
            100.0 * (row["peak_allocated_mib"] - oracle_memory) / oracle_memory
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "shape": {
            "batch_size": args.batch_size,
            "query_count": args.query_count,
            "query_microbatch": args.query_microbatch,
            "max_sensors": args.max_sensors,
            "valid_sensors_min": min(values["valid_sensor_counts"]),
            "valid_sensors_max": max(values["valid_sensor_counts"]),
            "num_latents": int(config["num_latents"]),
            "latent_dim": int(config["latent_dim"]),
            "num_heads": int(config["num_heads"]),
            "num_latent_blocks": int(config["num_latent_blocks"]),
        },
        "warmup_steps": args.warmup,
        "measured_steps": args.steps,
        "rows": rows,
    }
    (args.output_dir / "benchmark.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output_dir / "benchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    bucket_rows = [row for row in rows if row["sensor_attention_padding_mode"] == "static_buckets"]
    (args.output_dir / "bucket_comparison.json").write_text(
        json.dumps(bucket_rows, indent=2) + "\n"
    )
    if bucket_rows:
        with (args.output_dir / "bucket_comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(bucket_rows[0]))
            writer.writeheader()
            writer.writerows(bucket_rows)

    if args.profile_best:
        eligible = [row for row in rows if not row["mode"].startswith("D_")]
        best = min(eligible, key=lambda row: row["whole_step_ms"])
        mode_config = dict(config)
        mode_config.update(
            condition_attention_execution=best["condition_attention_execution"],
            sensor_attention_padding_mode=best["sensor_attention_padding_mode"],
            sensor_attention_buckets=best["sensor_attention_buckets"],
        )
        model = build_pointcloud_model(mode_config, n_fields=5, device=device).train()
        model.load_state_dict(base_state, strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("lr", 1e-4)))
        _profile_once(
            model, optimizer, values, args.query_microbatch,
            args.output_dir / "profiler" / f"{best['mode']}.json",
        )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

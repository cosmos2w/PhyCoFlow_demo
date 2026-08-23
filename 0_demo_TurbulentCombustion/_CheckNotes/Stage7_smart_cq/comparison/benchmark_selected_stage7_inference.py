#!/usr/bin/env python3
"""Time selected Stage7-All256 full-grid reconstruction without plot I/O."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

DEMO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = DEMO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluate_pointcloud_fixed_manifest import build_gl_rbf_ffm, reset_rf_rng
from helpers import TurbulentCombustionH5Dataset, build_sparse_condition
from train_pointcloud_ffm import checkpoint_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.demo_root.resolve()
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config
    checkpoint_path = (
        (root / args.checkpoint).resolve()
        if not args.checkpoint.is_absolute()
        else args.checkpoint
    )
    output_path = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    device = torch.device(args.device)

    dataset = TurbulentCombustionH5Dataset(
        config["data"],
        split="val",
        train_ratio=float(config.get("train_ratio", 0.9)),
        seed=int(config.get("seed", 42)),
        time_stride=int(config.get("time_stride", 1)),
        field_names=config.get("FIELD_NAMES", config.get("field_names")),
        stats_path=config.get("dataset_stats_path"),
        coord_batch_mode="shared_mesh",
        defer_field_read=True,
    )
    sample = dataset.get_full_snapshot(0)
    coords = sample["coords"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    reset_rf_rng(42, device)
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=[1],
        n_obs_min=[256],
        n_obs_max=[256],
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_gl_rbf_ffm(config, dataset.num_fields, device)
    state = checkpoint_model_state(checkpoint, model=model)
    model.load_state_dict(state, strict=True)
    model.eval()

    result = {
        "protocol": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "weights": "ema",
            "condition": "T",
            "sensors": 256,
            "grid_points": int(coords.shape[1]),
            "solver": "euler",
            "execution_mode": "cached_streamed",
            "cache_level": "static_features",
            "query_chunk_size": 8192,
            "timing_scope": "full-grid reconstruction after model load; includes source sampling, condition/static-cache/geometry construction, NFE, and hard clamps; excludes checkpoint/dataset load and CPU result copy",
        },
        "parameters": {
            "executable_total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        },
        "inference": {},
    }

    for nfe in (1, 4):
        def infer():
            reset_rf_rng(1729, device)
            return model.sample(
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                n_steps=nfe,
                clamp_indices=obs_indices,
                ode_solver="euler",
                obs_consistency_mode="default_hard",
                reconstruction_execution_mode="cached_streamed",
                reconstruction_query_chunk_size=8192,
                reconstruction_cache_level="static_features",
            )

        for _ in range(args.warmup):
            infer()
        samples = []
        peaks = []
        for _ in range(args.repeats):
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            out = infer()
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
            peaks.append(torch.cuda.max_memory_allocated(device) / (1024.0**2))
            del out
        result["inference"][f"nfe{nfe}"] = {
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
            "peak_allocated_mib": max(peaks),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

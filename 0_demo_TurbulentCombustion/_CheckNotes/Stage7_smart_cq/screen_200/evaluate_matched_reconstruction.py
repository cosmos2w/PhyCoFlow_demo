#!/usr/bin/env python3
"""Evaluate epoch-matched PointCloud FFM reconstructions with shared randomness.

Each candidate receives the same validation snapshot, sparse sensor layout, and
RF seed at every requested NFE. Stage-7 checkpoints automatically use EMA
weights when their checkpoint metadata enables EMA evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from evaluate_pointcloud_fixed_manifest import build_gl_rbf_ffm, reset_rf_rng
from helpers import (
    TurbulentCombustionH5Dataset,
    build_sparse_condition,
    visualize_reconstruction,
)
from train_pointcloud_ffm import checkpoint_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-root", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        nargs=3,
        metavar=("LABEL", "CONFIG", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--snapshot-index", type=int, default=0)
    parser.add_argument("--cond-fields", type=int, nargs="+", default=[1])
    parser.add_argument("--n-obs", type=int, nargs="+", default=[256])
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--obs-seed", type=int, default=42)
    parser.add_argument("--rf-seed", type=int, default=1729)
    return parser.parse_args()


def resolve(root: Path, path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else root / path


def load_config(path: Path, root: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text()) or {}
    for key in ("data", "dataset_stats_path"):
        value = config.get(key)
        if value:
            config[key] = str(resolve(root, value).resolve())
    return config


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(tensors):
        value = tensors[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prior_digest(state: dict[str, torch.Tensor]) -> str:
    tensors = {
        key: value
        for key, value in state.items()
        if key.startswith("prior.") and torch.is_tensor(value)
    }
    return tensor_digest(tensors)


def slugify(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def main() -> None:
    args = parse_args()
    root = args.demo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    candidates = []
    for label, config_value, checkpoint_value in args.candidate:
        config_path = resolve(root, config_value).resolve()
        checkpoint_path = resolve(root, checkpoint_value).resolve()
        candidates.append(
            (label, config_path, checkpoint_path, load_config(config_path, root))
        )

    first_config = candidates[0][3]
    dataset = TurbulentCombustionH5Dataset(
        first_config["data"],
        split="val",
        train_ratio=float(first_config.get("train_ratio", 0.9)),
        seed=int(first_config.get("seed", 42)),
        time_stride=int(first_config.get("time_stride", 1)),
        field_names=first_config.get("FIELD_NAMES", first_config.get("field_names")),
        stats_path=first_config.get("dataset_stats_path"),
        coord_batch_mode="shared_mesh",
        defer_field_read=True,
    )
    sample = dataset.get_full_snapshot(args.snapshot_index)
    coords = sample["coords"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    reset_rf_rng(args.obs_seed, device)
    condition_tuple = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=args.cond_fields,
        n_obs_min=args.n_obs,
        n_obs_max=args.n_obs,
    )
    condition_keys = (
        "obs_coords",
        "obs_values",
        "obs_mask",
        "obs_indices",
        "obs_field_ids",
    )
    sparse_condition = dict(zip(condition_keys, condition_tuple))
    condition_checksum = tensor_digest(sparse_condition)
    torch.save(
        {
            "snapshot_index": args.snapshot_index,
            "cond_fields": args.cond_fields,
            "n_obs": args.n_obs,
            "obs_seed": args.obs_seed,
            "checksum_sha256": condition_checksum,
            **{key: value.detach().cpu() for key, value in sparse_condition.items()},
        },
        output_dir / "shared_sparse_condition.pt",
    )

    summary: dict[str, Any] = {
        "protocol": {
            "snapshot_index": args.snapshot_index,
            "cond_fields": args.cond_fields,
            "n_obs": args.n_obs,
            "nfe": args.nfe,
            "ode_solver": "euler",
            "obs_consistency_mode": "default_hard",
            "obs_seed": args.obs_seed,
            "rf_seed": args.rf_seed,
            "shared_sparse_condition_checksum_sha256": condition_checksum,
        },
        "candidates": {},
    }
    field_names = list(dataset.field_names)

    for label, config_path, checkpoint_path, config in candidates:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_gl_rbf_ffm(config, dataset.num_fields, device)
        state = checkpoint_model_state(checkpoint, model=model)
        model.load_state_dict(state, strict=True)
        model.eval()
        label_dir = output_dir / slugify(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        candidate_result = {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "epoch": int(checkpoint.get("epoch", 0)),
            "weights": (
                "ema"
                if checkpoint.get("model_ema_eval", checkpoint.get("model_ema_enabled", False))
                and "model_ema" in checkpoint
                else "live"
            ),
            "prior_checksum_sha256": prior_digest(state),
            "nfe": {},
        }
        for nfe in args.nfe:
            reset_rf_rng(args.rf_seed, device)
            metrics = visualize_reconstruction(
                model=model,
                dataset=dataset,
                epoch=int(checkpoint.get("epoch", 0)),
                device=device,
                save_dir=str(label_dir),
                cond_fields=args.cond_fields,
                n_obs=args.n_obs,
                n_steps=int(nfe),
                ode_solver="euler",
                snapshot_index=args.snapshot_index,
                file_tag=f"euler_nfe{nfe}",
                save_metrics_json=True,
                sparse_condition=sparse_condition,
                reconstruction_execution_mode=str(
                    config.get("reconstruction_execution_mode", "cached_streamed")
                ),
                reconstruction_query_chunk_size=int(
                    config.get("reconstruction_query_chunk_size", 8192)
                ),
                reconstruction_cache_level=str(
                    config.get("reconstruction_cache_level", "static_features")
                ),
            )
            field_metrics = {name: float(metrics[name]) for name in field_names}
            worst_field = max(field_metrics, key=field_metrics.get)
            candidate_result["nfe"][str(nfe)] = {
                "field_relative_l2": field_metrics,
                "mean_field_relative_l2": sum(field_metrics.values()) / len(field_metrics),
                "worst_field": worst_field,
                "worst_field_relative_l2": field_metrics[worst_field],
            }
        summary["candidates"][label] = candidate_result
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

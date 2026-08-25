#!/usr/bin/env python3
"""Evaluate PointCloud FFM checkpoints on identical layouts and RF RNG draws."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from helpers import TurbulentCombustionH5Dataset
from Model import PointCloudFFM
from phycoflow_pointcloud.models.factory import build_pointcloud_model
from pointcloud_data_path import materialize_selected_batch
from pointcloud_eval_manifest import load_validation_manifest, slice_manifest_layout
from train_pointcloud_ffm import checkpoint_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Use 1 for true paired per-manifest loss rows; larger values report paired batch means.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rf-seed", type=int, default=1729)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _get(config: Mapping[str, Any], key: str, default: Any) -> Any:
    value = config.get(key, default)
    return default if value is None else value


def build_gl_rbf_ffm(config: Mapping[str, Any], n_fields: int, device: torch.device) -> PointCloudFFM:
    """Historical evaluator shim around the public frozen builder."""
    return build_pointcloud_model(config, n_fields=n_fields, device=device)


def reset_rf_rng(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))


def _tensor_digest(digest: Any, tensor: torch.Tensor) -> None:
    cpu = tensor.detach().cpu().contiguous()
    digest.update(str(cpu.dtype).encode())
    digest.update(str(tuple(cpu.shape)).encode())
    digest.update(cpu.numpy().tobytes())


def materialize_batches(
    dataset: TurbulentCombustionH5Dataset,
    manifest: Mapping[str, Any],
    batch_size: int,
) -> tuple[list[dict[str, torch.Tensor]], str]:
    batches = []
    digest = hashlib.sha256()
    count = int(manifest["sample_indices"].numel())
    for start in range(0, count, int(batch_size)):
        end = min(start + int(batch_size), count)
        local_indices = manifest["sample_indices"][start:end]
        items = [dataset[int(index)] for index in local_indices]
        actual_times = torch.stack([item["time_index"] for item in items])
        expected_times = manifest["time_indices"][start:end]
        if not torch.equal(actual_times, expected_times):
            raise RuntimeError("Manifest sample/time indices do not match the current validation split.")
        batch = materialize_selected_batch(
            dataset=dataset,
            items=items,
            query_indices=manifest["query_indices"][start:end],
            obs_layout=slice_manifest_layout(manifest, start, end),
            field_read_mode="legacy_full_snapshot",
        )
        tensors = {
            key: batch[key]
            for key in (
                "coords_q",
                "fields_q",
                "obs_coords",
                "obs_values",
                "obs_mask",
                "obs_indices",
                "obs_field_ids",
            )
        }
        for key in tensors:
            digest.update(key.encode())
            _tensor_digest(digest, tensors[key])
        batches.append(tensors)
    return batches, digest.hexdigest()


@torch.no_grad()
def evaluate_model(
    model: PointCloudFFM,
    batches: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    repeats: int,
    rf_seed: int,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    for repeat in range(int(repeats)):
        for batch_index, cpu_batch in enumerate(batches):
            seed = int(rf_seed) + repeat * 1_000_003 + batch_index
            reset_rf_rng(seed, device)
            batch = {
                key: value.to(device, non_blocking=False)
                for key, value in cpu_batch.items()
            }
            loss, metrics = model.training_loss(
                x1=batch["fields_q"],
                coords=batch["coords_q"],
                obs_coords=batch["obs_coords"],
                obs_values=batch["obs_values"],
                obs_mask=batch["obs_mask"],
                obs_field_ids=batch["obs_field_ids"],
                obs_indices=batch["obs_indices"],
            )
            rows.append(
                {
                    "repeat": repeat,
                    "batch_index": batch_index,
                    "manifest_start_index": sum(
                        int(previous["coords_q"].shape[0]) for previous in batches[:batch_index]
                    ),
                    "rf_seed": seed,
                    "loss": float(loss),
                    "target_rms": float(metrics["target_rms"]),
                    "batch_samples": int(batch["coords_q"].shape[0]),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text()) or {}
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = TurbulentCombustionH5Dataset(
        str(_get(config, "data", "Dataset/Merged_COTU0U1P.h5")),
        split="val",
        train_ratio=float(_get(config, "train_ratio", 0.9)),
        seed=int(_get(config, "seed", 42)),
        time_stride=int(_get(config, "time_stride", 1)),
        field_names=_get(config, "FIELD_NAMES", _get(config, "field_names", None)),
        stats_path=str(_get(config, "dataset_stats_path", "")) or None,
        coord_batch_mode="shared_mesh",
        defer_field_read=True,
    )
    manifest = load_validation_manifest(args.manifest, dataset=dataset)
    batches, input_checksum = materialize_batches(dataset, manifest, args.batch_size)

    all_rows: dict[str, list[dict[str, Any]]] = {}
    checkpoint_meta = {}
    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_gl_rbf_ffm(config, dataset.num_fields, device)
        model.load_state_dict(checkpoint_model_state(checkpoint, model=model), strict=True)
        label = checkpoint_path.parent.name
        if label in all_rows:
            label = f"{label}/{checkpoint_path.name}"
        rows = evaluate_model(
            model,
            batches,
            device=device,
            repeats=args.repeats,
            rf_seed=args.rf_seed,
        )
        all_rows[label] = rows
        checkpoint_meta[label] = {
            "path": str(checkpoint_path.resolve()),
            "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
            "stored_val_loss": checkpoint.get("val_loss") if isinstance(checkpoint, dict) else None,
        }
        del model

    labels = list(all_rows)
    reference = labels[0]
    summary = {}
    for label, rows in all_rows.items():
        losses = [row["loss"] for row in rows]
        paired = [loss - ref["loss"] for loss, ref in zip(losses, all_rows[reference])]
        summary[label] = {
            **checkpoint_meta[label],
            "mean_rf_loss": statistics.fmean(losses),
            "std_rf_loss": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            "paired_difference_vs_reference_mean": statistics.fmean(paired),
            "paired_difference_vs_reference_std": (
                statistics.stdev(paired) if len(paired) > 1 else 0.0
            ),
            "paired_difference_vs_reference_max_abs": max(abs(value) for value in paired),
            "evaluations": len(losses),
        }

    result = {
        "config": str(args.config.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_checksum_sha256": manifest["checksum_sha256"],
        "materialized_input_checksum_sha256": input_checksum,
        "device": str(device),
        "batch_size": int(args.batch_size),
        "repeats": int(args.repeats),
        "rf_seed": int(args.rf_seed),
        "controlled_rng": True,
        "reference_checkpoint": reference,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    csv_path = args.output.with_suffix(".csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "checkpoint",
                "repeat",
                "batch_index",
                "manifest_start_index",
                "rf_seed",
                "loss",
                "target_rms",
                "batch_samples",
            ],
        )
        writer.writeheader()
        for label, rows in all_rows.items():
            for row in rows:
                writer.writerow({"checkpoint": label, **row})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

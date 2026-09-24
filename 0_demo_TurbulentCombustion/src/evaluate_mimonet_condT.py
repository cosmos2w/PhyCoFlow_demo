"""Evaluate a trained MIMONet checkpoint on the archived A0 Cond_T sensor plan.

This is a post-training evaluator. Do not use held-out results to select the
architecture or hyperparameters of the controlled MIMONet baseline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluate_ablation_condT import PLAN, read_plan
from helpers import TurbulentCombustionH5Dataset
from train_mimonet_condT import (FIELDS, UNOBSERVED, branch_inputs, make_model,
                                 physical_relative_l2, resolve, sha256)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    cfg = yaml.safe_load((run / "run_config.yaml").read_text())
    metadata = json.loads((run / "run_metadata.json").read_text())
    if args.device != "cuda:0":
        raise ValueError("Use the authorized GPU 0 for this evaluation")
    device = torch.device(args.device)
    dataset = TurbulentCombustionH5Dataset(
        str(resolve(cfg["data"])), split="test", train_ratio=cfg["train_ratio"],
        seed=cfg["seed"], time_stride=cfg["time_stride"],
        stats_path=str(resolve(cfg["dataset_stats_path"])))
    if len(dataset) != 1000 or hashlib.sha256(dataset.indices.tobytes()).hexdigest() != metadata["val_test_indices_sha256"]:
        raise ValueError("Held-out split differs from the A0-matched training run")
    groups = read_plan(PLAN)
    if sorted(groups) != list(range(len(dataset))):
        raise ValueError("Archived Cond_T sensor plan does not cover the held-out split")
    checkpoint_path = run / f"{args.checkpoint}.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_model(cfg).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    if not torch.equal(checkpoint["mean"].float(), dataset.mean) or not torch.equal(checkpoint["std"].float(), dataset.std):
        raise ValueError("Checkpoint normalization differs from A0")
    coords = dataset.coords.to(device)
    mean, std = dataset.mean.to(device), dataset.std.to(device)
    rows = []
    with torch.inference_mode():
        for snapshot in range(len(dataset)):
            sample = dataset[snapshot]
            truth = sample["fields"].unsqueeze(0).to(device)
            sensors = groups[snapshot]
            if int(sample["time_index"]) != int(dataset.indices[snapshot]):
                raise ValueError("Snapshot ordering mismatch")
            indices = torch.tensor([int(row["point_index"]) for row in sensors],
                                   device=device, dtype=torch.long)
            values = truth[0, indices, 2]
            recorded = np.asarray([float(row["normalized_value"]) for row in sensors])
            np.testing.assert_allclose(values.cpu().numpy(), recorded, atol=1e-6, rtol=1e-6)
            if any(int(row["field_index"]) != 2 for row in sensors):
                raise ValueError("Non-temperature sensor in archived plan")
            mask = torch.zeros((1, 384), device=device)
            mask[:, :256] = 1
            obs_values = torch.zeros((1, 384, 1), device=device)
            obs_values[0, :256, 0] = values
            obs_coords = torch.zeros((1, 384, 3), device=device)
            obs_coords[0, :256] = coords[indices]
            branches = branch_inputs(obs_coords, obs_values, mask)
            prediction = torch.cat([model(branches, chunk.unsqueeze(0))
                                    for chunk in coords.split(2048)], dim=1)
            # A0's evaluator hard-clamps the observed temperature values.
            prediction[0, indices, 2] = values
            if not torch.isfinite(prediction).all():
                raise FloatingPointError(f"Nonfinite output on snapshot {snapshot}")
            physical = physical_relative_l2(truth, prediction, mean, std)[0].cpu().tolist()
            normalized = (torch.linalg.vector_norm((prediction - truth).double(), dim=1) /
                          (torch.linalg.vector_norm(truth.double(), dim=1) + 1e-12))[0].cpu().tolist()
            row = {"snapshot": snapshot, "time_index": int(dataset.indices[snapshot]),
                   "unobserved_mean_physical_relative_l2": float(np.mean(np.asarray(physical)[list(UNOBSERVED)]))}
            for c, field in enumerate(FIELDS):
                row[f"{field}_physical_relative_l2"] = physical[c]
                row[f"{field}_normalized_relative_l2"] = normalized[c]
            rows.append(row)
            if (snapshot + 1) % 100 == 0:
                print(f"{snapshot + 1}/{len(dataset)}", flush=True)
    output = run / "Evaluation" / f"{args.checkpoint}_paper_sensor_plan"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_snapshot.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
               "checkpoint_epoch": checkpoint["epoch"], "sensor_plan": str(PLAN),
               "sensor_plan_sha256": sha256(PLAN), "split": "test (same holdout as validation)",
               "snapshot_count": len(rows), "field_order": list(FIELDS),
               "metric": "per-snapshot physical relative L2, then mean across snapshots",
               "mean": {key: float(np.mean([row[key] for row in rows])) for key in rows[0]
                        if key.endswith("relative_l2")}}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

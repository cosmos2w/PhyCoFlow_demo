#!/usr/bin/env python3
"""Recover/run the final cached reconstruction for an existing limited run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from evaluate_pointcloud_fixed_manifest import build_gl_rbf_ffm
from helpers import TurbulentCombustionH5Dataset, build_sparse_condition, visualize_reconstruction
from train_pointcloud_ffm import checkpoint_model_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = yaml.safe_load((args.run / "run_config.yaml").read_text()) or {}
    device = torch.device(args.device)
    dataset = TurbulentCombustionH5Dataset(
        config["data"], split="val", train_ratio=float(config.get("train_ratio", 0.9)),
        seed=int(config.get("seed", 42)), time_stride=int(config.get("time_stride", 1)),
        stats_path=config.get("dataset_stats_path"), coord_batch_mode="shared_mesh",
        defer_field_read=True,
    )
    model = build_gl_rbf_ffm(config, dataset.num_fields, device)
    checkpoint = torch.load(args.run / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_model_state(checkpoint), strict=True)
    model.eval()
    full = dataset.get_full_snapshot(0)
    coords = full["coords"].unsqueeze(0).to(device)
    truth = full["fields"].unsqueeze(0).to(device)
    torch.manual_seed(int(config.get("seed", 42)))
    condition = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=config.get("vis_cond_fields", [1]),
        n_obs_min=config.get("vis_n_obs_list", [256]),
        n_obs_max=config.get("vis_n_obs_list", [256]),
    )
    sparse = dict(zip(
        ("obs_coords", "obs_values", "obs_mask", "obs_indices", "obs_field_ids"),
        condition,
    ))
    output = args.run / "Evaluation" / f"epoch_{args.epoch:04d}"
    output.mkdir(parents=True, exist_ok=True)
    for nfe in config.get("benchmark_n_steps", [1, 2]):
        torch.manual_seed(int(config.get("seed", 42)) + 1000)
        metrics = visualize_reconstruction(
            model=model, dataset=dataset, epoch=args.epoch, device=device,
            save_dir=str(output), cond_fields=config.get("vis_cond_fields", [1]),
            n_obs=config.get("vis_n_obs_list", [256]), n_steps=int(nfe),
            ode_solver=config.get("ode_solver", "euler"), snapshot_index=0,
            file_tag=f"{config.get('ode_solver', 'euler')}_nfe{nfe}",
            save_metrics_json=True, sparse_condition=sparse,
            reconstruction_execution_mode="cached_streamed",
            reconstruction_query_chunk_size=int(config.get("reconstruction_query_chunk_size", 8192)),
            reconstruction_cache_level=config.get("reconstruction_cache_level", "static_features"),
        )
        print(f"nfe={nfe}: {metrics}")
    dataset.close()


if __name__ == "__main__":
    main()

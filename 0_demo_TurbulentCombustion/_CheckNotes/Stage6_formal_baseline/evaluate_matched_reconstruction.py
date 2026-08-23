#!/usr/bin/env python3
"""Generate non-visual, matched best-checkpoint reconstruction source data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from evaluate_pointcloud_fixed_manifest import build_gl_rbf_ffm
from helpers import TurbulentCombustionH5Dataset, build_sparse_condition
from train_pointcloud_ffm import checkpoint_model_state


def reset_rng(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def digest_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--condition-seed", type=int, default=42)
    parser.add_argument("--sample-seed", type=int, default=1042)
    parser.add_argument("--nfe", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()

    config = yaml.safe_load((args.run / "run_config.yaml").read_text()) or {}
    device = torch.device(args.device)
    dataset = TurbulentCombustionH5Dataset(
        config["data"],
        split="val",
        train_ratio=float(config.get("train_ratio", 0.9)),
        seed=int(config.get("seed", 42)),
        time_stride=int(config.get("time_stride", 1)),
        stats_path=config.get("dataset_stats_path"),
        coord_batch_mode="shared_mesh",
        defer_field_read=True,
    )
    model = build_gl_rbf_ffm(config, dataset.num_fields, device)
    checkpoint_path = args.run / args.checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint_model_state(checkpoint), strict=True)
    model.eval()

    sample = dataset.get_full_snapshot(0)
    coords = sample["coords"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"].cpu().numpy()

    reset_rng(args.condition_seed)
    condition = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=config.get("vis_cond_fields", [1]),
        n_obs_min=config.get("vis_n_obs_list", [256]),
        n_obs_max=config.get("vis_n_obs_list", [256]),
    )
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = condition

    args.output.mkdir(parents=True, exist_ok=True)
    valid = obs_mask[0].bool()
    obs_indices_np = obs_indices[0, valid].detach().cpu().numpy()
    obs_field_ids_np = obs_field_ids[0, valid].detach().cpu().numpy()
    np.savez_compressed(
        args.output / "condition.npz",
        obs_indices=obs_indices_np,
        obs_field_ids=obs_field_ids_np,
        obs_values=obs_values[0, valid].detach().cpu().numpy(),
    )

    mean = dataset.mean.to(device).view(1, 1, -1)
    std = dataset.std.to(device).view(1, 1, -1)
    truth_phys = (truth * std + mean)[0].detach().cpu().numpy()
    field_names = list(dataset.field_names)
    condition_checksum = digest_arrays(obs_indices_np, obs_field_ids_np)
    summaries = []

    for nfe in args.nfe:
        reset_rng(args.sample_seed)
        with torch.no_grad():
            recon = model.sample(
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                n_steps=int(nfe),
                ode_solver=config.get("ode_solver", "euler"),
                clamp_indices=obs_indices,
                reconstruction_execution_mode="cached_streamed",
                reconstruction_query_chunk_size=int(
                    config.get("reconstruction_query_chunk_size", 8192)
                ),
                reconstruction_cache_level=config.get(
                    "reconstruction_cache_level", "static_features"
                ),
            )
        recon_phys = (recon * std + mean)[0].detach().cpu().numpy()
        relative_l2 = {
            name: float(
                np.linalg.norm(truth_phys[:, index] - recon_phys[:, index])
                / (np.linalg.norm(truth_phys[:, index]) + 1.0e-8)
            )
            for index, name in enumerate(field_names)
        }
        summary = {
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
            "stored_val_loss": float(checkpoint.get("val_loss", float("nan"))),
            "snapshot_index": 0,
            "condition_seed": int(args.condition_seed),
            "sample_seed": int(args.sample_seed),
            "condition_checksum_sha256": condition_checksum,
            "n_obs": int(valid.sum()),
            "nfe": int(nfe),
            "ode_solver": config.get("ode_solver", "euler"),
            "field_names": field_names,
            "relative_l2": relative_l2,
            "mean_field_relative_l2": float(np.mean(list(relative_l2.values()))),
        }
        np.savez_compressed(
            args.output / f"nfe{nfe}.npz",
            coords_raw=coords_raw,
            truth_phys=truth_phys,
            recon_phys=recon_phys,
            field_names=np.asarray(field_names),
        )
        (args.output / f"nfe{nfe}_metrics.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        summaries.append(summary)

    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    dataset.close()


if __name__ == "__main__":
    main()

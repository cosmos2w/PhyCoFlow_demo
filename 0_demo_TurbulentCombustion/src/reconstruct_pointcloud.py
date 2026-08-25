"""Reconstruct one real snapshot with explicit seeds and protocol metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from helpers import TurbulentCombustionH5Dataset, build_sparse_condition

from phycoflow_pointcloud.checkpointing import resolve_checkpoint_state
from phycoflow_pointcloud.config import load_public_config, project_root
from phycoflow_pointcloud.models.factory import build_pointcloud_model


def _digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--cond-fields", type=int, nargs="+")
    parser.add_argument("--n-obs", type=int, nargs="+")
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument(
        "--cache-level", choices=["none", "geometry", "static_features"]
    )
    parser.add_argument(
        "--output", type=Path, default=project_root() / "runs/reconstruction/sample.pt"
    )
    cli = parser.parse_args(argv)
    config = load_public_config(cli.config)
    device = torch.device(cli.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = TurbulentCombustionH5Dataset(
        config["data"],
        split="val",
        train_ratio=float(config.get("train_ratio", 0.9)),
        seed=int(config.get("seed", 42)),
        time_stride=int(config.get("time_stride", 1)),
        field_names=config.get("FIELD_NAMES", config.get("field_names")),
        stats_path=str(config["dataset_stats_path"]),
        coord_batch_mode="shared_mesh",
        defer_field_read=True,
    )
    try:
        sample = dataset.get_full_snapshot(cli.sample_index)
        coords = sample["coords"].unsqueeze(0).to(device)
        truth = sample["fields"].unsqueeze(0).to(device)
        cond_fields = cli.cond_fields or list(
            config.get("vis_cond_fields", config["cond_fields"])
        )
        n_obs = cli.n_obs or list(
            config.get("vis_n_obs_list", config["n_obs_max_list"])
        )
        torch.manual_seed(cli.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cli.seed)
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = (
            build_sparse_condition(
                coords_full=coords,
                fields_full=truth,
                cond_fields=cond_fields,
                n_obs_min=n_obs,
                n_obs_max=n_obs,
            )
        )
        model = build_pointcloud_model(
            config, n_fields=dataset.num_fields, device=device
        )
        checkpoint = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
        resolved = resolve_checkpoint_state(checkpoint, model=model)
        model.load_state_dict(resolved.state_dict, strict=True)
        model.eval()
        torch.manual_seed(cli.seed + 1)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(cli.seed + 1)
        with torch.no_grad():
            reconstruction = model.sample(
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                clamp_indices=obs_indices,
                n_steps=cli.n_steps,
                ode_solver=cli.solver,
                obs_consistency_mode="endpoint_smooth",
                reconstruction_execution_mode=str(
                    config.get("reconstruction_execution_mode", "cached_streamed")
                ),
                reconstruction_query_chunk_size=int(
                    cli.chunk_size
                    or config.get("reconstruction_query_chunk_size", 8192)
                ),
                reconstruction_cache_level=str(
                    cli.cache_level
                    or config.get("reconstruction_cache_level", "static_features")
                ),
            )
        metadata = {
            "format_version": 1,
            "model_name": config["model_name"],
            "checkpoint": str(cli.checkpoint.resolve()),
            "checkpoint_selection": resolved.selection,
            "sample_index": cli.sample_index,
            "condition_seed": cli.seed,
            "rf_seed": cli.seed + 1,
            "condition_sha256": _digest(torch.cat([obs_coords, obs_values], dim=-1)),
            "reconstruction_sha256": _digest(reconstruction),
            "field_names": list(dataset.field_names),
            "n_steps": cli.n_steps,
            "solver": cli.solver,
        }
        cli.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "reconstruction": reconstruction.cpu(),
                "truth": truth.cpu(),
                "coords": coords.cpu(),
                "obs_indices": obs_indices.cpu(),
                "metadata": metadata,
            },
            cli.output,
        )
        cli.output.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(json.dumps(metadata, indent=2))
    finally:
        dataset.close()

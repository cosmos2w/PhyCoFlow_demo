"""Train one released MIMONet baseline for a saved multi-field combustion condition.

This adapter changes input dimensions and sensor packing only. The upstream
branch/trunk FCNs, multiplicative merge and 256-dimensional basis are retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from helpers import TurbulentCombustionH5Dataset, build_sparse_condition, visualize_reconstruction
from mimonet_upstream import MIMONet
from train_pointcloud_ffm import sample_query_subset_legacy
from train_mimonet_condT import (DEMO, FIELDS, assert_finite, branch_inputs,
                                 collate_fields, physical_relative_l2,
                                 plot_loss_curve, resolve, save_checkpoint, sha256)


CONDITIONS = {"Cond_TU1": (2, 3), "Cond_COTU1P": (1, 2, 3, 4)}


def unobserved_fields(cfg: dict) -> tuple[int, ...]:
    return tuple(i for i in range(5) if i not in cfg["cond_fields"])


def max_sensors(cfg: dict) -> int:
    return sum(int(value) for value in cfg["n_obs_max_list"])


def make_model(cfg: dict) -> MIMONet:
    count = max_sensors(cfg)
    basis = int(cfg["latent_dim"])
    branch_width = int(cfg["branch_hidden_dim"])
    trunk_width = int(cfg["trunk_hidden_dim"])
    return MIMONet(
        branch_arch_list=[
            [2 * count, branch_width, branch_width, branch_width, basis],
            [4 * count, branch_width, branch_width, branch_width, basis],
        ],
        trunk_arch=[3, trunk_width, trunk_width, trunk_width, basis],
        num_outputs=5, activation_fn=torch.nn.ReLU, merge_type=cfg["merge_type"],
    )


def pack_sensors(obs_coords: torch.Tensor, obs_values: torch.Tensor,
                 obs_mask: torch.Tensor, field_ids: torch.Tensor,
                 cfg: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Give each field its own fixed slots without changing the A0 sensor draw."""
    fields = [int(field) for field in cfg["cond_fields"]]
    capacities = [int(n) for n in cfg["n_obs_max_list"]]
    if len(fields) != len(capacities):
        raise ValueError("Each conditioned field needs a sensor capacity")
    batch, packed_count = obs_mask.shape
    total = sum(capacities)
    if packed_count > total:
        raise ValueError("Sampler returned more observations than branch capacity")
    counts = torch.stack([((field_ids == field) & obs_mask.bool()).sum(dim=1)
                          for field in fields], dim=1)
    source_starts = counts.cumsum(dim=1) - counts
    coords = obs_coords.new_zeros((batch, total, 3))
    values = obs_values.new_zeros((batch, total, 1))
    mask = obs_mask.new_zeros((batch, total))
    target_start = 0
    for j, capacity in enumerate(capacities):
        positions = torch.arange(capacity, device=obs_mask.device)[None, :]
        source = (source_starts[:, j:j + 1] + positions).clamp_max(packed_count - 1)
        valid = positions < counts[:, j:j + 1]
        coords[:, target_start:target_start + capacity] = (
            obs_coords.gather(1, source[..., None].expand(-1, -1, 3)) * valid[..., None])
        values[:, target_start:target_start + capacity] = (
            obs_values.gather(1, source[..., None]) * valid[..., None])
        mask[:, target_start:target_start + capacity] = valid.to(obs_mask.dtype)
        target_start += capacity
    if not torch.equal(mask.sum(dim=1).long(), obs_mask.sum(dim=1).long()):
        raise ValueError("Sensor packing lost observations")
    return coords, values, mask


def sample_batch(fields: torch.Tensor, coords: torch.Tensor, cfg: dict,
                 *, fixed_n_obs: int | None = None, n_query: int | None = None,
                 training: bool = True):
    batch, _, channels = fields.shape
    if channels != 5:
        raise ValueError("Expected the five A0 combustion targets")
    mesh = coords.unsqueeze(0).expand(batch, -1, -1)
    counts_min = ([fixed_n_obs] * len(cfg["cond_fields"]) if fixed_n_obs is not None
                  else cfg["n_obs_min_list"])
    counts_max = ([fixed_n_obs] * len(cfg["cond_fields"]) if fixed_n_obs is not None
                  else cfg["n_obs_max_list"])
    obs_coords, obs_values, obs_mask, _, field_ids = build_sparse_condition(
        mesh, fields, cond_fields=cfg["cond_fields"],
        n_obs_min=counts_min, n_obs_max=counts_max)
    packed_coords, packed_values, packed_mask = pack_sensors(
        obs_coords, obs_values, obs_mask, field_ids, cfg)
    sampling = cfg["query_sampling"] if training else "uniform"
    query_coords, query_fields, _ = sample_query_subset_legacy(
        coords=mesh, fields=fields, n_query=int(n_query or cfg["n_query_points"]),
        mode=sampling, obs_coords=obs_coords, obs_mask=obs_mask,
        near_ratio=float(cfg["query_sample_near_ratio"]),
        far_ratio=float(cfg["query_sample_far_ratio"]),
        sigma_ratio=float(cfg["query_sample_sigma_ratio"]),
    )
    return (branch_inputs(packed_coords, packed_values, packed_mask),
            query_coords, query_fields, packed_mask)


@torch.no_grad()
def diagnostic_full_validation(model, dataset, coords, cfg, device, *, count: int) -> dict:
    model.eval()
    mean, std = dataset.mean.to(device), dataset.std.to(device)
    ratios = []
    prediction_ranges = [[] for _ in range(5)]
    prediction_stds = [[] for _ in range(5)]
    for i in range(min(count, len(dataset))):
        truth = dataset[i]["fields"].unsqueeze(0).to(device)
        torch.manual_seed(81000 + i)
        if device.type == "cuda":
            torch.cuda.manual_seed(81000 + i)
        branches, _, _, mask = sample_batch(
            truth, coords, cfg, fixed_n_obs=256, n_query=8, training=False)
        if int(mask.sum()) != 256 * len(cfg["cond_fields"]):
            raise ValueError("Validation sensor counts do not match the condition")
        prediction = torch.cat([model(branches, chunk[None]) for chunk in coords.split(2048)], dim=1)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite full-mesh validation prediction")
        ratios.append(physical_relative_l2(truth, prediction, mean, std)[0].cpu())
        physical = (prediction * std + mean)[0]
        for channel in range(5):
            prediction_ranges[channel].append((float(physical[:, channel].min()),
                                               float(physical[:, channel].max())))
            prediction_stds[channel].append(float(physical[:, channel].std()))
    values = torch.stack(ratios).mean(0)
    missing = list(unobserved_fields(cfg))
    return {"sample_count": len(ratios), "fixed_sensors_per_conditioned_field": 256,
            "conditioned_fields": [FIELDS[i] for i in cfg["cond_fields"]],
            "physical_relative_l2": {FIELDS[i]: float(values[i]) for i in range(5)},
            "unobserved_mean": float(values[missing].mean()),
            "prediction_range": {FIELDS[i]: [min(x[0] for x in prediction_ranges[i]),
                                            max(x[1] for x in prediction_ranges[i])]
                                 for i in range(5)},
            "prediction_spatial_std": {FIELDS[i]: float(np.mean(prediction_stds[i]))
                                       for i in range(5)}}


class MultiConditionSampleAdapter:
    def __init__(self, model: MIMONet, cfg: dict):
        self.model = model
        self.cfg = cfg

    def eval(self):
        self.model.eval()
        return self

    @torch.no_grad()
    def sample(self, *, coords, obs_coords, obs_values, obs_mask,
               obs_field_ids, clamp_indices, n_steps=1, **_unused):
        valid_ids = obs_field_ids[obs_mask.bool()]
        if not torch.isin(valid_ids, torch.as_tensor(self.cfg["cond_fields"], device=valid_ids.device)).all():
            raise ValueError("An unconfigured field entered the MIMONet branches")
        packed = pack_sensors(obs_coords, obs_values, obs_mask, obs_field_ids, self.cfg)
        branches = branch_inputs(*packed)
        prediction = torch.cat(
            [self.model(branches, chunk) for chunk in coords.split(2048, dim=1)], dim=1)
        for batch in range(coords.shape[0]):
            valid = obs_mask[batch].bool()
            prediction[batch, clamp_indices[batch, valid], obs_field_ids[batch, valid]] = (
                obs_values[batch, valid, 0])
        return prediction


def sensor_plan_rows(cfg: dict, snapshot: int) -> list[dict]:
    plan = resolve(cfg["reconstruction_sensor_plan"])
    with plan.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["condition"] == cfg["condition"] and int(row["snapshot"]) == snapshot]
    rows.sort(key=lambda row: int(row["sensor_order"]))
    fields = cfg["cond_fields"]
    if len(rows) != 256 * len(fields) or [int(row["sensor_order"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Archived sensor plan count/order mismatch")
    if any(sum(int(row["field_index"]) == field for row in rows) != 256 for field in fields):
        raise ValueError("Archived plan must have 256 sensors per conditioned field")
    return rows


def reconstruction_diagnosis(model, dataset, cfg: dict, run_dir: Path,
                             epoch: int, device: torch.device) -> dict:
    snapshot = int(cfg["reconstruction_snapshot_index"])
    rows = sensor_plan_rows(cfg, snapshot)
    sample = dataset[snapshot]
    truth = sample["fields"].unsqueeze(0).to(device)
    coords = sample["coords"].unsqueeze(0).to(device)
    indices = torch.tensor([[int(row["point_index"]) for row in rows]],
                           dtype=torch.long, device=device)
    field_ids = torch.tensor([[int(row["field_index"]) for row in rows]],
                             dtype=torch.long, device=device)
    values = truth[0, indices[0], field_ids[0]].view(1, len(rows), 1)
    np.testing.assert_allclose(values[0, :, 0].cpu().numpy(),
        np.asarray([float(row["normalized_value"]) for row in rows]), atol=1e-6, rtol=1e-6)
    sparse = {"obs_coords": coords[:, indices[0]], "obs_values": values,
              "obs_mask": torch.ones((1, len(rows)), device=device),
              "obs_indices": indices, "obs_field_ids": field_ids}
    output = run_dir / "Evaluation" / f"epoch_{epoch:04d}"
    output.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        metrics, payload = visualize_reconstruction(
            model=MultiConditionSampleAdapter(model, cfg), dataset=dataset,
            epoch=epoch, device=device, save_dir=str(output),
            cond_fields=cfg["cond_fields"], n_obs=[256] * len(cfg["cond_fields"]),
            n_steps=1, snapshot_index=snapshot, file_tag="mimonet",
            return_payload=True, obs_consistency_mode="default_hard",
            sparse_condition=sparse)
    truth_phys = np.asarray(payload["truth_phys"], dtype=np.float64)
    prediction_phys = np.asarray(payload["recon_phys"], dtype=np.float64)
    physical_l2 = np.sqrt(np.sum((prediction_phys - truth_phys) ** 2, axis=0)) / (
        np.sqrt(np.sum(truth_phys ** 2, axis=0)) + 1e-12)
    missing = list(unobserved_fields(cfg))
    diagnosis = {
        "epoch": epoch, "condition": cfg["condition"], "snapshot_index": snapshot,
        "time_index": int(sample["time_index"]),
        "sensor_plan": str(resolve(cfg["reconstruction_sensor_plan"])),
        "sensor_plan_sha256": sha256(resolve(cfg["reconstruction_sensor_plan"])),
        "sensors_per_conditioned_field": 256,
        "model": "deterministic MIMONet; no ODE/NFE variants",
        "normalized_relative_l2": {str(dataset.field_names[i]): float(metrics[dataset.field_names[i]])
                                   for i in range(5)},
        "physical_relative_l2": {str(dataset.field_names[i]): float(physical_l2[i])
                                 for i in range(5)},
        "unobserved_mean_physical_relative_l2": float(np.mean(physical_l2[missing])),
        "sensor_relative_l2": float(metrics["obs_rel_l2_SenConsis"]),
        "fields": {str(dataset.field_names[i]): {
            "truth_range": [float(truth_phys[:, i].min()), float(truth_phys[:, i].max())],
            "prediction_range": [float(prediction_phys[:, i].min()),
                                 float(prediction_phys[:, i].max())],
            "truth_spatial_std": float(truth_phys[:, i].std()),
            "prediction_spatial_std": float(prediction_phys[:, i].std()),
            "prediction_nonphysical_fraction": (float(np.mean(
                prediction_phys[:, i] <= 0 if i in (2, 4) else prediction_phys[:, i] < 0))
                if i != 3 else None),
        } for i in range(5)},
    }
    (output / "quality_diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n")
    (output / "figure_contract.md").write_text(
        f"# MIMONet {cfg['condition']} reconstruction diagnosis\n\n"
        f"- Claim: inspect five-field reconstruction for held-out snapshot {snapshot} at epoch {epoch}.\n"
        f"- Source: live epoch model, reference HDF5/statistics, and `{cfg['reconstruction_sensor_plan']}`.\n"
        "- Panels: physical truth, reconstruction, and absolute error for each field using A0's plotting helper.\n"
        "- Metrics: normalized and physical relative L2, sensor consistency, ranges, and spatial variation.\n"
        "- Caveat: one holdout snapshot does not establish the full 1,000-state test score.\n"
    )
    return diagnosis


def validate_reference(cfg: dict, dataset: TurbulentCombustionH5Dataset) -> dict:
    reference = resolve(cfg["reference_run"])
    source = yaml.safe_load((reference / "run_config.yaml").read_text())
    keys = ("seed", "train_ratio", "time_stride", "cond_fields", "n_obs_min_list",
            "n_obs_max_list", "n_query_points", "query_sampling")
    for key in keys:
        if cfg[key] != source[key]:
            raise ValueError(f"{cfg['condition']} reference mismatch in {key}: {cfg[key]} != {source[key]}")
    for key in ("query_sample_near_ratio", "query_sample_far_ratio",
                "query_sample_sigma_ratio"):
        if float(cfg[key]) != float(source[key]):
            raise ValueError(f"{cfg['condition']} reference mismatch in {key}")
    if resolve(cfg["data"]).resolve() != resolve(source["data"]).resolve():
        raise ValueError("Reference HDF5 path mismatch")
    stats = reference / "dataset_stats.pt"
    if resolve(cfg["dataset_stats_path"]).resolve() != stats.resolve():
        raise ValueError("Reference normalization path mismatch")
    if (len(dataset), dataset.num_points, dataset.num_fields) != (9000, 40300, 5):
        raise ValueError("Unexpected reference training population or mesh")
    if tuple(name.replace("_", "") for name in dataset.field_names) != FIELDS:
        raise ValueError("Unexpected field ordering")
    checkpoint = torch.load(reference / "best.pt", map_location="cpu", weights_only=False)
    for key in ("mean", "std"):
        if key in checkpoint and not torch.equal(checkpoint[key].float(), getattr(dataset, key)):
            raise ValueError(f"Reference checkpoint {key} differs from dataset statistics")
    return {"reference_config_sha256": sha256(reference / "run_config.yaml"),
            "reference_best_sha256": sha256(reference / "best.pt"),
            "stats_sha256": sha256(stats),
            "dataset_size_bytes": resolve(cfg["data"]).stat().st_size}


def run_epoch(model, loader, coords, cfg, device, optimizer=None) -> tuple[float, float, int]:
    training = optimizer is not None
    model.train(training)
    weighted_loss = 0.0
    examples = 0
    started = time.perf_counter()
    for batch in loader:
        fields = batch["fields"].to(device)
        branches, query_coords, target, mask = sample_batch(
            fields, coords, cfg, training=training)
        if mask.shape[1] != max_sensors(cfg):
            raise ValueError("Unexpected branch sensor capacity")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(branches, query_coords)
            loss = F.mse_loss(prediction, target)
            if training:
                loss.backward()
            assert_finite(model, prediction, loss, gradients=training)
            if training:
                optimizer.step()
        weighted_loss += float(loss.detach()) * fields.shape[0]
        examples += fields.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return weighted_loss / examples, time.perf_counter() - started, examples


def smoke(cfg: dict, train_set, val_set, model, device) -> None:
    coords = train_set.coords.to(device)
    raw = train_set.read_fields(int(train_set.indices[0]))
    normalized = train_set[0]["fields"]
    if not torch.allclose(normalized * train_set.std + train_set.mean,
                          raw, atol=1e-5, rtol=1e-5):
        raise ValueError("Reference inverse normalization mismatch")
    fields = torch.stack([train_set[0]["fields"], train_set[1]["fields"]]).to(device)
    branches, query, target, mask = sample_batch(fields, coords, cfg, n_query=128)
    # A field omitted from the condition must never change either branch.
    changed = fields.clone()
    changed[..., list(unobserved_fields(cfg))] += 100.0
    torch.manual_seed(91042)
    torch.cuda.manual_seed(91042)
    original_branches, _, _, _ = sample_batch(fields, coords, cfg, n_query=128)
    torch.manual_seed(91042)
    torch.cuda.manual_seed(91042)
    changed_branches, _, _, _ = sample_batch(changed, coords, cfg, n_query=128)
    if not all(torch.equal(a, b) for a, b in zip(original_branches, changed_branches)):
        raise ValueError("Unobserved fields leaked into the sensor branches")
    expected = max_sensors(cfg)
    assert branches[0].shape == (2, 2 * expected)
    assert branches[1].shape == (2, 4 * expected)
    assert query.shape == (2, 128, 3) and target.shape == (2, 128, 5)
    minimum = sum(cfg["n_obs_min_list"])
    assert bool(((mask.sum(dim=1) >= minimum) & (mask.sum(dim=1) <= expected)).all())
    prediction = model(branches, query)
    assert prediction.shape == (2, 128, 5)
    loss = F.mse_loss(prediction, target)
    loss.backward()
    assert_finite(model, prediction, loss, gradients=True)
    model.zero_grad(set_to_none=True)
    diagnostic = diagnostic_full_validation(model, val_set, coords, cfg, device, count=1)
    assert all(math.isfinite(value) for value in diagnostic["physical_relative_l2"].values())
    print(json.dumps({"smoke": "passed", "condition": cfg["condition"],
        "branch_shapes": [list(branch.shape) for branch in branches],
        "query_shape": list(query.shape), "prediction_shape": list(prediction.shape),
        "sensor_counts": mask.sum(dim=1).tolist(), "normalized_loss": float(loss),
        "validation_physical_relative_l2": diagnostic["physical_relative_l2"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    if cfg.get("condition") not in CONDITIONS or tuple(cfg["cond_fields"]) != CONDITIONS[cfg["condition"]]:
        raise ValueError("This trainer supports the saved Cond_TU1 and Cond_COTU1P profiles only")
    if int(cfg["epochs"]) != 5000 or int(cfg["num_outputs"]) != 5:
        raise ValueError("Controlled comparison requires 5000 epochs and five outputs")
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device(cfg["device"])
    if device.type != "cuda" or device.index != 0:
        raise ValueError("These runs are authorized for GPU 0")
    train_set = TurbulentCombustionH5Dataset(
        str(resolve(cfg["data"])), split="train", train_ratio=cfg["train_ratio"],
        seed=cfg["seed"], time_stride=cfg["time_stride"],
        stats_path=str(resolve(cfg["dataset_stats_path"])))
    val_set = TurbulentCombustionH5Dataset(
        str(resolve(cfg["data"])), split="val", train_ratio=cfg["train_ratio"],
        seed=cfg["seed"], time_stride=cfg["time_stride"],
        stats_path=str(resolve(cfg["dataset_stats_path"])))
    provenance = validate_reference(cfg, train_set)
    assert len(val_set) == 1000 and not np.intersect1d(train_set.indices, val_set.indices).size
    assert torch.equal(train_set.mean, val_set.mean) and torch.equal(train_set.std, val_set.std)
    model = make_model(cfg).to(device)
    parameter_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if args.smoke:
        smoke(cfg, train_set, val_set, model, device)
        return
    run_dir = args.run_dir or (resolve(cfg["save_root"]) /
        f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=DEMO, text=True).strip()
    source_files = ("src/train_mimonet_multicond.py", "src/train_mimonet_condT.py",
        "src/mimonet_upstream/mimonet.py", "src/mimonet_upstream/fcn.py",
        "src/helpers.py", "src/train_pointcloud_ffm.py",
        "figures/scripts/plot_mimonet_loss.py", str(cfg_path.relative_to(DEMO)))
    metadata = {**provenance, "git_commit": git_hash,
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"],
                                               cwd=DEMO, text=True).strip(),
        "upstream": "Zenodo v2 https://zenodo.org/records/21986357",
        "source_sha256": {name: sha256(DEMO / name) for name in source_files},
        "parameter_count": parameter_count, "latent_basis_dimension": cfg["latent_dim"],
        "condition": cfg["condition"], "conditioned_fields": cfg["cond_fields"],
        "unobserved_fields": list(unobserved_fields(cfg)),
        "train_indices_sha256": hashlib.sha256(train_set.indices.tobytes()).hexdigest(),
        "val_test_indices_sha256": hashlib.sha256(val_set.indices.tobytes()).hexdigest(),
        "train_samples": len(train_set), "val_test_samples": len(val_set),
        "field_names": train_set.field_names,
        "launch_command": str(Path(sys.executable).resolve()) +
            " -u src/train_mimonet_multicond.py --config " + args.config +
            " --run-dir " + str(run_dir)}
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    capacity = max_sensors(cfg)
    (run_dir / "model_summary.json").write_text(json.dumps({
        "model": "upstream MIMONet", "branch_architectures": [
            [2 * capacity, 512, 512, 512, 256], [4 * capacity, 512, 512, 512, 256]],
        "trunk_architecture": [3, 256, 256, 256, 1280],
        "merge_type": "mul", "latent_basis_dimension": 256, "outputs": list(FIELDS),
        "trainable_parameters": parameter_count}, indent=2) + "\n")
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], collate_fn=collate_fields)
    val_loader = DataLoader(val_set, batch_size=cfg["val_batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], collate_fn=collate_fields)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["scheduler_t_max"])
    coords = train_set.coords.to(device)
    best_val = float("inf")
    rows = []
    print(json.dumps({"run_dir": str(run_dir), "condition": cfg["condition"],
        "parameters": parameter_count, "latent_basis": cfg["latent_dim"],
        "train_samples": len(train_set), "val_test_samples": len(val_set),
        "device": str(device)}), flush=True)
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_seconds, train_count = run_epoch(
            model, train_loader, coords, cfg, device, optimizer)
        scheduler.step()
        val_loss, val_seconds = None, 0.0
        if epoch == 1 or epoch % cfg["eval_every"] == 0:
            val_loss, val_seconds, val_count = run_epoch(model, val_loader, coords, cfg, device)
            assert val_count == len(val_set)
        diagnostic = None
        if epoch in (1, 10, 25, 50):
            diagnostic = diagnostic_full_validation(
                model, val_set, coords, cfg, device,
                count=int(cfg["diagnostic_val_samples"]))
            (run_dir / f"diagnostic_epoch_{epoch:04d}.json").write_text(
                json.dumps(diagnostic, indent=2) + "\n")
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch,
                            train_loss, val_loss, best_val, train_set, cfg)
        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch,
                        train_loss, val_loss, best_val, train_set, cfg)
        if epoch % int(cfg["reconstruction_eval_every"]) == 0:
            save_checkpoint(run_dir / f"epoch_{epoch:04d}.pt", model, optimizer,
                            scheduler, epoch, train_loss, val_loss, best_val, train_set, cfg)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"], "train_seconds": train_seconds,
            "val_seconds": val_seconds, "train_samples": train_count,
            "train_samples_per_second": train_count / train_seconds,
            "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            "diagnostic_unobserved_mean_physical_relative_l2": (
                diagnostic["unobserved_mean"] if diagnostic else None)}
        rows.append(row)
        with (run_dir / "loss_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        (run_dir / "loss_history.json").write_text(json.dumps(rows, indent=2) + "\n")
        if epoch == 1 or epoch % int(cfg["loss_curve_plot_every"]) == 0:
            plot_loss_curve(run_dir)
        print(json.dumps({"epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "lr": row["lr"], "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "train_samples_per_second": row["train_samples_per_second"],
            "gpu_peak_allocated_gib": row["gpu_peak_allocated_gib"],
            "diagnostic": diagnostic}), flush=True)
        if epoch % int(cfg["reconstruction_eval_every"]) == 0:
            quality = reconstruction_diagnosis(model, val_set, cfg, run_dir, epoch, device)
            print(json.dumps({"reconstruction_epoch": epoch,
                              "quality_diagnosis": quality}), flush=True)


if __name__ == "__main__":
    main()

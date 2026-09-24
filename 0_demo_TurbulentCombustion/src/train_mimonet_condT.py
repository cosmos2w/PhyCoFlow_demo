"""Train the released MIMONet operator on the A0 Cond_T combustion task.

The upstream branch/trunk network is unchanged apart from its package import.
Only temperature readings, their locations, and a padding mask enter the branches.
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

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from helpers import TurbulentCombustionH5Dataset, build_sparse_condition
from mimonet_upstream import MIMONet


DEMO = Path(__file__).resolve().parents[1]
FIELDS = ("CH4", "CO", "T", "U1", "p")
UNOBSERVED = (0, 1, 3, 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else DEMO / path


def collate_fields(batch: list[dict]) -> dict:
    return {"fields": torch.stack([row["fields"] for row in batch]),
            "time_index": torch.stack([row["time_index"] for row in batch])}


def make_model(cfg: dict) -> MIMONet:
    n = int(cfg["n_obs_max_list"][0])
    d = int(cfg["latent_dim"])
    bh = int(cfg["branch_hidden_dim"])
    th = int(cfg["trunk_hidden_dim"])
    # Branch 1: paired (normalized T, validity); branch 2: paired
    # (normalized x,y,z, validity). Neither branch sees other fields or time.
    return MIMONet(
        branch_arch_list=[[2 * n, bh, bh, bh, d], [4 * n, bh, bh, bh, d]],
        trunk_arch=[3, th, th, th, d],
        num_outputs=5,
        activation_fn=torch.nn.ReLU,
        merge_type=cfg["merge_type"],
    )


def branch_inputs(obs_coords: torch.Tensor, obs_values: torch.Tensor,
                  obs_mask: torch.Tensor) -> list[torch.Tensor]:
    mask = obs_mask.unsqueeze(-1)
    values = torch.cat((obs_values * mask, mask), dim=-1).flatten(1)
    geometry = torch.cat((obs_coords * mask, mask), dim=-1).flatten(1)
    return [values, geometry]


def sample_batch(fields: torch.Tensor, coords: torch.Tensor, cfg: dict,
                 *, fixed_n_obs: int | None = None, n_query: int | None = None):
    batch, n_points, n_fields = fields.shape
    assert n_fields == 5 and tuple(cfg["cond_fields"]) == (2,)
    mesh = coords.unsqueeze(0).expand(batch, -1, -1)
    minimum = [fixed_n_obs] if fixed_n_obs is not None else cfg["n_obs_min_list"]
    maximum = [fixed_n_obs] if fixed_n_obs is not None else cfg["n_obs_max_list"]
    obs_coords, obs_values, mask, indices, field_ids = build_sparse_condition(
        mesh, fields, cond_fields=[2], n_obs_min=minimum, n_obs_max=maximum)
    if fixed_n_obs is not None:
        # The branch FCNs have fixed input size, even in fixed-sensor diagnostics.
        pad = int(cfg["n_obs_max_list"][0]) - fixed_n_obs
        obs_coords = F.pad(obs_coords, (0, 0, 0, pad))
        obs_values = F.pad(obs_values, (0, 0, 0, pad))
        mask = F.pad(mask, (0, pad))
        indices = F.pad(indices, (0, pad))
        field_ids = F.pad(field_ids, (0, pad), value=-1)
    assert torch.all(field_ids[mask.bool()] == 2)
    count = int(n_query or cfg["n_query_points"])
    query_indices = torch.stack([
        torch.randperm(n_points, device=fields.device)[:count].sort().values
        for _ in range(batch)])
    query_coords = mesh.gather(1, query_indices[..., None].expand(-1, -1, 3))
    query_fields = fields.gather(1, query_indices[..., None].expand(-1, -1, 5))
    return branch_inputs(obs_coords, obs_values, mask), query_coords, query_fields, mask


def physical_relative_l2(truth: torch.Tensor, prediction: torch.Tensor,
                         mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Per-snapshot, per-field metric used by the Cond_T physical audit."""
    truth_physical = (truth * std + mean).double()
    pred_physical = (prediction * std + mean).double()
    return torch.linalg.vector_norm(pred_physical - truth_physical, dim=1) / (
        torch.linalg.vector_norm(truth_physical, dim=1) + 1e-12)


def assert_finite(model: torch.nn.Module, prediction: torch.Tensor,
                  loss: torch.Tensor, *, gradients: bool) -> None:
    if not torch.isfinite(loss) or not torch.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite MIMONet loss or prediction")
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise FloatingPointError(f"Nonfinite parameter: {name}")
        if gradients and (param.grad is None or not torch.isfinite(param.grad).all()):
            raise FloatingPointError(f"Missing/nonfinite gradient: {name}")


def validate_reference(cfg: dict, dataset: TurbulentCombustionH5Dataset) -> dict:
    ref_dir = resolve(cfg["reference_run"])
    ref_cfg = yaml.safe_load((ref_dir / "run_config.yaml").read_text())
    required = ("seed", "train_ratio", "time_stride", "cond_fields",
                "n_obs_min_list", "n_obs_max_list", "n_query_points", "query_sampling")
    for key in required:
        if cfg[key] != ref_cfg[key]:
            raise ValueError(f"A0 mismatch in {key}: {cfg[key]!r} != {ref_cfg[key]!r}")
    if resolve(cfg["data"]).resolve() != Path(ref_cfg["data"]).resolve():
        raise ValueError("A0 dataset path mismatch")
    if resolve(cfg["dataset_stats_path"]).resolve() != Path(ref_cfg["dataset_stats_path"]).resolve():
        raise ValueError("A0 statistics path mismatch")
    if tuple(name.replace("_", "") for name in dataset.field_names) != FIELDS:
        raise ValueError(f"Unexpected target channels: {dataset.field_names}")
    if (len(dataset), dataset.num_points, dataset.num_fields) != (9000, 40300, 5):
        raise ValueError("Unexpected A0 training population or mesh")
    checkpoint = torch.load(ref_dir / "best.pt", map_location="cpu", weights_only=False)
    for key in ("mean", "std"):
        if key in checkpoint and not torch.equal(checkpoint[key].float(), getattr(dataset, key)):
            raise ValueError(f"A0 checkpoint {key} differs from dataset statistics")
    return {"reference_config_sha256": sha256(ref_dir / "run_config.yaml"),
            "reference_best_sha256": sha256(ref_dir / "best.pt"),
            "stats_sha256": sha256(resolve(cfg["dataset_stats_path"])),
            "dataset_size_bytes": resolve(cfg["data"]).stat().st_size}


@torch.no_grad()
def diagnostic_full_validation(model, dataset, coords, cfg, device, *, count: int) -> dict:
    model.eval()
    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    ratios = []
    predicted = [[] for _ in range(5)]
    true = [[] for _ in range(5)]
    for i in range(min(count, len(dataset))):
        fields = dataset[i]["fields"].unsqueeze(0).to(device)
        # A fixed 256-sensor Cond_T diagnostic, matching the A0 paper sensor count.
        torch.manual_seed(81000 + i)
        if device.type == "cuda":
            torch.cuda.manual_seed(81000 + i)
        inp, _, _, mask = sample_batch(fields, coords, cfg, fixed_n_obs=256, n_query=8)
        assert int(mask.sum()) == 256
        pieces = []
        for chunk in coords.split(2048):
            pieces.append(model(inp, chunk[None]))
        pred = torch.cat(pieces, dim=1)
        ratios.append(physical_relative_l2(fields, pred, mean, std)[0].cpu())
        truth_phys = fields * std + mean
        pred_phys = pred * std + mean
        for c in range(5):
            predicted[c].append((float(pred_phys[..., c].min()), float(pred_phys[..., c].max()),
                                 float(pred_phys[..., c].std())))
            true[c].append((float(truth_phys[..., c].min()), float(truth_phys[..., c].max())))
    values = torch.stack(ratios).mean(0)
    return {"sample_count": len(ratios), "fixed_temperature_sensors": 256,
            "physical_relative_l2": {FIELDS[c]: float(values[c]) for c in range(5)},
            "unobserved_mean": float(values[list(UNOBSERVED)].mean()),
            "prediction_range": {FIELDS[c]: [min(x[0] for x in predicted[c]),
                                           max(x[1] for x in predicted[c])] for c in range(5)},
            "prediction_spatial_std": {FIELDS[c]: float(np.mean([x[2] for x in predicted[c]]))
                                       for c in range(5)},
            "truth_range": {FIELDS[c]: [min(x[0] for x in true[c]), max(x[1] for x in true[c])]
                            for c in range(5)}}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int,
                    train_loss: float, val_loss: float | None, best_val: float,
                    dataset, config: dict) -> None:
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "epoch": epoch,
             "train_loss": train_loss, "val_loss": val_loss, "best_val_loss": best_val,
             "mean": dataset.mean, "std": dataset.std,
             "field_names": dataset.field_names, "config": config}
    temporary = path.with_suffix(".tmp.pt")
    torch.save(state, temporary)
    os.replace(temporary, path)


def run_epoch(model, loader, coords, cfg, device, optimizer=None) -> tuple[float, float, int]:
    training = optimizer is not None
    model.train(training)
    weighted_loss = 0.0
    examples = 0
    started = time.perf_counter()
    for batch in loader:
        fields = batch["fields"].to(device)
        inp, query_coords, target, mask = sample_batch(fields, coords, cfg)
        if mask.shape[1] != int(cfg["n_obs_max_list"][0]):
            raise ValueError("Unexpected sensor padding")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction = model(inp, query_coords)
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
    assert torch.allclose(normalized * train_set.std + train_set.mean, raw, atol=1e-5, rtol=1e-5)
    assert torch.all((coords >= 0) & (coords <= 1))
    fields = torch.stack([train_set[0]["fields"], train_set[1]["fields"]]).to(device)
    inp, q, target, mask = sample_batch(fields, coords, cfg, n_query=128)
    assert inp[0].shape == (2, 768) and inp[1].shape == (2, 1536)
    assert q.shape == (2, 128, 3) and target.shape == (2, 128, 5)
    assert bool(((mask.sum(1) >= 192) & (mask.sum(1) <= 384)).all())
    prediction = model(inp, q)
    assert prediction.shape == (2, 128, 5)
    loss = F.mse_loss(prediction, target)
    loss.backward()
    assert_finite(model, prediction, loss, gradients=True)
    model.zero_grad(set_to_none=True)
    metric = diagnostic_full_validation(model, val_set, coords, cfg, device, count=1)
    assert all(math.isfinite(x) for x in metric["physical_relative_l2"].values())
    print(json.dumps({"smoke": "passed", "input_shapes": [list(x.shape) for x in inp],
                      "query_shape": list(q.shape), "prediction_shape": list(prediction.shape),
                      "sensor_counts": mask.sum(1).tolist(), "normalized_loss": float(loss),
                      "validation_physical_relative_l2": metric["physical_relative_l2"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Save_config/mimonet/config_MIMONet_condT.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    cfg_path = resolve(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())
    if int(cfg["epochs"]) != 5000 or int(cfg["num_outputs"]) != 5:
        raise ValueError("This controlled comparison requires 5000 epochs and five channels")
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    device = torch.device(cfg["device"])
    if device.type != "cuda" or device.index != 0:
        raise ValueError("This experiment is authorized for GPU 0")
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
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.smoke:
        smoke(cfg, train_set, val_set, model, device)
        return
    run_dir = args.run_dir or (resolve(cfg["save_root"]) /
        f"MIMONet_condT_DemoN600_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=DEMO, text=True).strip()
    metadata = {**provenance, "git_commit": git_hash, "git_branch":
                subprocess.check_output(["git", "branch", "--show-current"], cwd=DEMO, text=True).strip(),
                "upstream": "Zenodo v2 https://zenodo.org/records/21986357",
                "upstream_files": {name: sha256(DEMO / "src" / "mimonet_upstream" / name)
                                   for name in ("mimonet.py", "fcn.py")},
                "parameter_count": count, "latent_basis_dimension": cfg["latent_dim"],
                "train_indices_sha256": hashlib.sha256(train_set.indices.tobytes()).hexdigest(),
                "val_test_indices_sha256": hashlib.sha256(val_set.indices.tobytes()).hexdigest(),
                "train_samples": len(train_set), "val_test_samples": len(val_set),
                "field_names": train_set.field_names,
                "launch_command": str(Path(sys.executable).resolve()) +
                    " -u src/train_mimonet_condT.py --config " + args.config +
                    " --run-dir " + str(run_dir)}
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "model_summary.json").write_text(json.dumps({
        "model": "upstream MIMONet", "branch_architectures": [[768, 512, 512, 512, 256],
            [1536, 512, 512, 512, 256]], "trunk_architecture": [3, 256, 256, 256, 1280],
        "merge_type": "mul", "latent_basis_dimension": 256, "outputs": list(FIELDS),
        "trainable_parameters": count}, indent=2) + "\n")
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], collate_fn=collate_fields)
    val_loader = DataLoader(val_set, batch_size=cfg["val_batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], collate_fn=collate_fields)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler_t_max"])
    coords = train_set.coords.to(device)
    best_val = float("inf")
    rows = []
    print(json.dumps({"run_dir": str(run_dir), "parameters": count, "latent_basis": 256,
                      "train_samples": len(train_set), "val_test_samples": len(val_set),
                      "device": str(device)}), flush=True)
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_seconds, train_count = run_epoch(
            model, train_loader, coords, cfg, device, optimizer)
        scheduler.step()
        val_loss = None
        val_seconds = 0.0
        if epoch == 1 or epoch % cfg["eval_every"] == 0:
            val_loss, val_seconds, val_count = run_epoch(model, val_loader, coords, cfg, device)
            assert val_count == len(val_set)
        diag = None
        if epoch in (1, 10, 25, 50):
            diag = diagnostic_full_validation(model, val_set, coords, cfg, device,
                count=cfg["diagnostic_val_samples"])
            (run_dir / f"diagnostic_epoch_{epoch:04d}.json").write_text(
                json.dumps(diag, indent=2) + "\n")
        if val_loss is not None and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(run_dir / "best.pt", model, optimizer, scheduler, epoch,
                            train_loss, val_loss, best_val, train_set, cfg)
        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch,
                        train_loss, val_loss, best_val, train_set, cfg)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "lr": optimizer.param_groups[0]["lr"], "train_seconds": train_seconds,
               "val_seconds": val_seconds, "train_samples": train_count,
               "train_samples_per_second": train_count / train_seconds,
               "gpu_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
               "gpu_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
               "diagnostic_unobserved_mean_physical_relative_l2":
                   diag["unobserved_mean"] if diag else None}
        rows.append(row)
        with (run_dir / "loss_history.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        (run_dir / "loss_history.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                          "lr": row["lr"], "train_seconds": train_seconds,
                          "val_seconds": val_seconds, "train_samples_per_second":
                          row["train_samples_per_second"], "gpu_peak_allocated_gib":
                          row["gpu_peak_allocated_gib"], "diagnostic": diag}), flush=True)


if __name__ == "__main__":
    main()

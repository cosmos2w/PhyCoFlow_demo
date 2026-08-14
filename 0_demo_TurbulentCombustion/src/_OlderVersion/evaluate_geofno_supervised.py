"""
Standalone evaluator for trained GeoFNO supervised regression models.

Mirrors the interface of evaluate_ffm.py but uses a single forward pass
(supervised regression) rather than ODE integration.

Usage:
    python evaluate_geofno_supervised.py --Demo-Num 0 --demo-root ..
"""

import argparse
import json
import os
import re
import sys
import pickle
from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import numpy as np

from helpers import (
    TurbulentCombustionH5Dataset,
    NonlinearPoissonDataset,
    build_sparse_condition,
    _build_structured_triangulation,
    _save_single_field_plot as _shared_save_single_field_plot,
)

# ── neuraloperator imports ──────────────────────────────────────────────────
_neuralop_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "FNO", "neuraloperator"
)
if os.path.isdir(_neuralop_path) and _neuralop_path not in sys.path:
    sys.path.insert(0, _neuralop_path)

from neuralop.models import FNO as NeuralOpFNO



# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser("Standalone evaluator for trained GeoFNO supervised models.")
    p.add_argument("--Demo-Num", dest="Demo_Num", type=int, required=True,
                   help="Demo ID to recover.")
    p.add_argument("--demo-root", type=str, default=".",
                   help="Project/demo root directory.")
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--snapshot-index", type=int, default=0,
                   help="Index within the selected split.")

    p.add_argument("--vis-cond-fields", type=int, nargs="+", default=None,
                   help="Override visualization cond_fields.")
    p.add_argument("--vis-n-obs-list", type=int, nargs="+", default=None,
                   help="Override visualization n_obs list.")
    p.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"],
                   help="Which checkpoint to load.")
    p.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    p.add_argument(
        "--dataset", type=str, default="turbulent_combustion",
        choices=["turbulent_combustion", "poisson", "elasticity",
                 "airfoil", "car_cfd"],
        help="Which dataset was used for training. Used to disambiguate "
             "Demo_Num collisions across datasets when picking the YAML config.")
    p.add_argument(
        "--physics-metrics", action="store_true",
        help="Also compute domain-specific integrated metrics "
             "(Cl/Cd/Cp for airfoil, F_drag/Cd_p/Cp for car-cfd, "
             "max von-Mises + K_t for elasticity).")

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def grid_to_pointcloud(x_grid: torch.Tensor, Num_y: int, Num_x: int) -> torch.Tensor:
    B, C, _, _ = x_grid.shape
    return x_grid.permute(0, 2, 3, 1).reshape(B, Num_y * Num_x, C).contiguous()


def build_obs_grid_mask(
    obs_values, obs_mask, obs_field_ids, obs_indices,
    n_fields, n_pts, Num_y, Num_x,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = obs_values.shape[0]
    device = obs_values.device
    dtype = obs_values.dtype

    value_flat = torch.zeros(B, n_fields, n_pts, device=device, dtype=dtype)
    mask_flat = torch.zeros(B, n_fields, n_pts, device=device, dtype=dtype)

    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        idx = obs_indices[b, valid].long()
        fld = obs_field_ids[b, valid].long()
        val = obs_values[b, valid, 0]
        value_flat[b, fld, idx] = val
        mask_flat[b, fld, idx] = 1.0

    obs_value_grid = value_flat.reshape(B, n_fields, Num_y, Num_x)
    obs_mask_grid = mask_flat.reshape(B, n_fields, Num_y, Num_x)
    return obs_value_grid, obs_mask_grid


# ═════════════════════════════════════════════════════════════════════════════
# Model (must match training script)
# ═════════════════════════════════════════════════════════════════════════════

class FNOSupervisedGrid(nn.Module):
    def __init__(
        self,
        n_fields: int,
        Num_x: int,
        Num_y: int,
        n_modes_x: int = 32,
        n_modes_y: int = 8,
        hidden_channels: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_fields = n_fields
        self.Num_x = Num_x
        self.Num_y = Num_y
        self.fno = NeuralOpFNO(
            n_modes=(n_modes_y, n_modes_x),
            in_channels=2 * n_fields,
            out_channels=n_fields,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            positional_embedding="grid",
        )

    def forward(self, obs_value_grid, obs_mask_grid):
        x = torch.cat([obs_value_grid, obs_mask_grid], dim=1)
        return self.fno(x)


class FNOSupervisedIrregular(nn.Module):
    """FNO for irregular meshes using splat → FNO → gather via grid_sample."""

    def __init__(
        self,
        n_fields: int,
        latent_Nx: int = 64,
        latent_Ny: int = 64,
        n_modes_x: int = 24,
        n_modes_y: int = 24,
        hidden_channels: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_fields = n_fields
        self.latent_Nx = latent_Nx
        self.latent_Ny = latent_Ny
        self.fno = NeuralOpFNO(
            n_modes=(n_modes_y, n_modes_x),
            in_channels=2 * n_fields,
            out_channels=n_fields,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            positional_embedding="grid",
        )

    def _gather_from_grid(self, grid, coords_2d):
        sample_pts = coords_2d * 2.0 - 1.0
        sample_pts = sample_pts.unsqueeze(2)
        gathered = F.grid_sample(
            grid, sample_pts, mode="bilinear",
            padding_mode="border", align_corners=True,
        )
        return gathered.squeeze(-1).permute(0, 2, 1)

    def forward(self, coords_2d, obs_values, obs_mask, obs_field_ids, obs_indices):
        B = coords_2d.shape[0]
        C = self.n_fields
        Ny, Nx = self.latent_Ny, self.latent_Nx
        device = coords_2d.device
        dtype = coords_2d.dtype

        val_grid = torch.zeros(B, C, Ny * Nx, device=device, dtype=dtype)
        msk_grid = torch.zeros(B, C, Ny * Nx, device=device, dtype=dtype)

        for b in range(B):
            valid = obs_mask[b].bool()
            if not valid.any():
                continue
            idx = obs_indices[b, valid].long()
            fld = obs_field_ids[b, valid].long()
            val = obs_values[b, valid, 0]
            obs_xy = coords_2d[b, idx]
            ix = (obs_xy[:, 0] * (Nx - 1)).long().clamp(0, Nx - 1)
            iy = (obs_xy[:, 1] * (Ny - 1)).long().clamp(0, Ny - 1)
            flat = iy * Nx + ix
            val_grid[b, fld, flat] = val
            msk_grid[b, fld, flat] = 1.0

        val_grid = val_grid.reshape(B, C, Ny, Nx)
        msk_grid = msk_grid.reshape(B, C, Ny, Nx)

        fno_in = torch.cat([val_grid, msk_grid], dim=1)
        pred_grid = self.fno(fno_in)
        return self._gather_from_grid(pred_grid, coords_2d)


# ═════════════════════════════════════════════════════════════════════════════
# Config discovery
# ═════════════════════════════════════════════════════════════════════════════

def _extract_timestamp(path: Path) -> Optional[str]:
    m = re.search(r"DemoN(\d+)_(\d{8}_\d{6})", path.name)
    if m is None:
        m = re.search(r"demo_N(\d+)_(\d{8}_\d{6})", path.name)
    return m.group(2) if m else None


def _find_latest_yaml(cfg_dir: Path, demo_num: int,
                      dataset_filter: Optional[str] = None) -> Path:
    pattern = f"config_geofno_supervised_DemoN{demo_num}_*.yaml"
    if (cfg_dir / "geofno_supervised").exists() or cfg_dir.name == "Save_config":
        candidates = sorted(cfg_dir.rglob(pattern))
    else:
        candidates = sorted(cfg_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No config backup found for Demo_Num={demo_num} in {cfg_dir}"
        )

    if dataset_filter is not None:
        filtered = []
        for p in candidates:
            try:
                with open(p, "r") as fh:
                    c = yaml.safe_load(fh) or {}
                if c.get("dataset", "turbulent_combustion") == dataset_filter:
                    filtered.append(p)
            except Exception:
                continue
        if not filtered:
            raise FileNotFoundError(
                f"No config backup found for Demo_Num={demo_num} with "
                f"dataset='{dataset_filter}' in {cfg_dir}"
            )
        candidates = filtered

    def _sort_key(p: Path):
        ts = _extract_timestamp(p)
        return ts if ts is not None else p.stat().st_mtime

    candidates = sorted(candidates, key=_sort_key)
    return candidates[-1]


def _normalize_eval_config(cfg: dict) -> dict:
    cfg = dict(cfg)

    if cfg.get("cond_fields") is None:
        cfg["cond_fields"] = [cfg.get("cond_field", 2)]
    if cfg.get("n_obs_min_list") is None:
        cfg["n_obs_min_list"] = [cfg.get("n_obs_min", 64)]
    if cfg.get("n_obs_max_list") is None:
        cfg["n_obs_max_list"] = [cfg.get("n_obs_max", 256)]

    if cfg.get("vis_cond_fields") in (None, ""):
        cfg["vis_cond_fields"] = list(cfg["cond_fields"])
    if cfg.get("vis_n_obs_list") in (None, ""):
        cfg["vis_n_obs_list"] = list(cfg["n_obs_max_list"])

    cfg.setdefault("geofno_variant", "fno")
    cfg.setdefault("Num_x", 403)
    cfg.setdefault("Num_y", 100)
    cfg.setdefault("fno_modes_x", 32)
    cfg.setdefault("fno_modes_y", 8)
    cfg.setdefault("fno_hidden_channels", 64)
    cfg.setdefault("fno_n_layers", 4)
    cfg.setdefault("latent_Nx", 64)
    cfg.setdefault("latent_Ny", 64)

    return cfg


def _build_model(cfg: dict, n_fields: int) -> nn.Module:
    variant = cfg.get("geofno_variant", "fno")

    if variant == "fno":
        return FNOSupervisedGrid(
            n_fields=n_fields,
            Num_x=cfg["Num_x"],
            Num_y=cfg["Num_y"],
            n_modes_x=cfg["fno_modes_x"],
            n_modes_y=cfg["fno_modes_y"],
            hidden_channels=cfg["fno_hidden_channels"],
            n_layers=cfg["fno_n_layers"],
        )

    if variant == "irregular":
        return FNOSupervisedIrregular(
            n_fields=n_fields,
            latent_Nx=cfg["latent_Nx"],
            latent_Ny=cfg["latent_Ny"],
            n_modes_x=cfg["fno_modes_x"],
            n_modes_y=cfg["fno_modes_y"],
            hidden_channels=cfg["fno_hidden_channels"],
            n_layers=cfg["fno_n_layers"],
        )

    raise ValueError(f"Unknown geofno_variant: {variant}")


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation visualization
# ═════════════════════════════════════════════════════════════════════════════

def _normalized_l2(u_true, u_pred):
    return float(np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8))


def _save_single_field_plot(true_f, pred_f, coords_xy, sensor_coords, field_name,
                            epoch, save_dir, file_prefix=None,
                            dpi=300, cmap_field="coolwarm", cmap_err="inferno",
                            contour_levels=20, contour_linewidth=0.5,
                            contour_alpha=0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    import matplotlib.gridspec as gridspec

    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    x_plot = coords_xy[:, 0]
    y_plot = coords_xy[:, 1]

    err = np.abs(true_f - pred_f)
    l2_error = _normalized_l2(true_f, pred_f)

    field_min = float(np.nanmin([true_f.min(), pred_f.min()]))
    field_max = float(np.nanmax([true_f.max(), pred_f.max()]))

    positive_err = err[err > 0]
    err_min = float(positive_err.min()) if positive_err.size > 0 else 0.0
    err_max = float(err.max()) if err.size > 0 else 1.0

    # Interpolate onto a regular grid; mask grid cells outside the domain
    # using a nearest-neighbour distance threshold.
    n_grid = 500
    gx = np.linspace(x_plot.min(), x_plot.max(), n_grid)
    gy = np.linspace(y_plot.min(), y_plot.max(), n_grid)
    GX, GY = np.meshgrid(gx, gy)

    tree = cKDTree(coords_xy)
    nn_dist = tree.query(coords_xy, k=2)[0][:, 1]
    dist_thresh = np.percentile(nn_dist, 95) * 1.5
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])
    grid_dist, _ = tree.query(grid_pts)
    outside = grid_dist > dist_thresh

    Z_true = griddata(coords_xy, true_f, (GX, GY), method="cubic")
    Z_pred = griddata(coords_xy, pred_f, (GX, GY), method="cubic")
    Z_err  = griddata(coords_xy, err,    (GX, GY), method="cubic")
    Z_true[outside.reshape(GX.shape)] = np.nan
    Z_pred[outside.reshape(GX.shape)] = np.nan
    Z_err[outside.reshape(GX.shape)]  = np.nan

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 1, wspace=0.0, hspace=0.20)

    ax_true = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[1, 0])
    ax_err  = fig.add_subplot(gs[2, 0])

    im_true = ax_true.pcolormesh(
        GX, GY, Z_true, cmap=cmap_field,
        vmin=field_min, vmax=field_max, shading="auto",
    )
    if contour_levels is not None:
        ax_true.contour(
            GX, GY, Z_true, levels=contour_levels, colors="white",
            linewidths=contour_linewidth, alpha=contour_alpha,
        )

    im_pred = ax_pred.pcolormesh(
        GX, GY, Z_pred, cmap=cmap_field,
        vmin=field_min, vmax=field_max, shading="auto",
    )
    if contour_levels is not None:
        ax_pred.contour(
            GX, GY, Z_pred, levels=contour_levels, colors="white",
            linewidths=contour_linewidth, alpha=contour_alpha,
        )

    im_err = ax_err.pcolormesh(
        GX, GY, Z_err, cmap=cmap_err,
        vmin=err_min, vmax=err_max, shading="auto",
    )

    if sensor_coords is not None and len(sensor_coords) > 0:
        ax_true.scatter(
            sensor_coords[:, 0], sensor_coords[:, 1],
            s=12.5, c="none", edgecolors="tab:green", linewidths=2.0,
            marker="o", zorder=4,
        )

    ax_true.set_title("Ground truth", fontsize=13)
    ax_pred.set_title("Reconstruction", fontsize=13)
    ax_err.set_title("|Error|", fontsize=13)

    for ax in (ax_true, ax_pred, ax_err):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_field = fig.colorbar(
        im_true, ax=[ax_true, ax_pred], shrink=0.6, pad=0.02,
    )
    cbar_field.set_label(field_name)

    cbar_err = fig.colorbar(im_err, ax=ax_err, shrink=0.6, pad=0.02)
    cbar_err.set_label(f"|{field_name} - û|")

    fig.suptitle(
        f"{field_name}    |    Normalized L2 = {l2_error:.3e}",
        y=0.96, fontsize=14,
    )

    prefix = file_prefix if file_prefix is not None else f"epoch_{epoch:04d}"
    filename = os.path.join(save_dir, f"{prefix}_field_{field_name}.png")
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return l2_error


@torch.no_grad()
def evaluate_snapshot(
    model, dataset,
    Num_x, Num_y,
    epoch, device, save_dir,
    cond_fields=(2,), n_obs=256,
    snapshot_index=0, file_tag=None, save_metrics_json=True,
    irregular=False,
    compute_physics: bool = False,
):
    if isinstance(cond_fields, int):
        cond_fields = [cond_fields]
    if isinstance(n_obs, int):
        n_obs = [n_obs] * len(cond_fields)

    n_fields = dataset.num_fields

    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    valid_mask = sample.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0).to(device)

    _, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords, fields_full=truth,
        cond_fields=cond_fields, n_obs_min=n_obs, n_obs_max=n_obs,
        valid_mask=valid_mask,
    )

    if irregular:
        coords_2d = coords[:, :, :2]
        recon = model(coords_2d, obs_values, obs_mask, obs_field_ids, obs_indices)
    else:
        n_pts = dataset.num_points
        obs_value_grid, obs_mask_grid = build_obs_grid_mask(
            obs_values, obs_mask, obs_field_ids, obs_indices,
            n_fields, n_pts, Num_y, Num_x,
        )
        pred_grid = model(obs_value_grid, obs_mask_grid)
        recon = grid_to_pointcloud(pred_grid, Num_y, Num_x)

    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = truth * std.view(1, 1, -1) + mean.view(1, 1, -1)

    recon_phys = recon_phys[0].cpu().numpy()
    truth_phys = truth_phys[0].cpu().numpy()

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()
    coords_np = coords_raw[0].cpu().numpy()
    coords_xy = coords_np[:, :2]

    triang = None
    body_polygon = None
    if hasattr(dataset, "grid_shape") and dataset.grid_shape is not None:
        triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)
    if hasattr(dataset, "airfoil_body_indices") and dataset.airfoil_body_indices is not None:
        body_polygon = coords_xy[dataset.airfoil_body_indices]

    metrics = {}
    n_fields_out = truth_phys.shape[1]
    field_names = (
        dataset.field_names
        if len(dataset.field_names) == n_fields_out
        else tuple(f"field_{i}" for i in range(n_fields_out))
    )
    for c, name in enumerate(field_names):
        true_f = truth_phys[:, c]
        pred_f = recon_phys[:, c]

        sensor_coords = None
        field_sensor_mask = (obs_field_ids_cpu == c)
        if np.any(field_sensor_mask):
            sensor_coords = coords_xy[obs_indices_cpu[field_sensor_mask]]

        l2_error = _shared_save_single_field_plot(
            true_f=true_f, pred_f=pred_f, coords_xy=coords_xy,
            sensor_coords=sensor_coords, field_name=name,
            epoch=epoch, save_dir=save_dir, file_prefix=file_tag,
            triang=triang, body_polygon=body_polygon,
        )
        metrics[name] = l2_error

    if compute_physics:
        from physics_metrics import compute_physics_metrics
        phys = compute_physics_metrics(dataset, truth_phys, recon_phys, snapshot_index)
        if phys:
            metrics["physics"] = phys

    if save_metrics_json:
        prefix = file_tag or f"epoch_{epoch:04d}"
        metrics_path = os.path.join(save_dir, f"{prefix}_metrics.json")
        payload = {
            "epoch": int(epoch),
            "snapshot_index": int(snapshot_index),
            "cond_fields": [int(v) for v in cond_fields],
            "n_obs": [int(v) for v in n_obs],
            "method": "GeoFNO_supervised",
            "metrics": metrics,
        }
        with open(metrics_path, "w") as f:
            json.dump(payload, f, indent=2)

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    demo_root = Path(args.demo_root).resolve()
    if args.dataset is not None:
        cfg_dir = demo_root / "Save_config" / args.dataset / "geofno_supervised"
    else:
        cfg_dir = demo_root / "Save_config"

    try:
        yaml_path = _find_latest_yaml(cfg_dir, args.Demo_Num,
                                      dataset_filter=args.dataset)
    except FileNotFoundError as e:
        print(f"[Warning: !] {e}")
        raise SystemExit(1)

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _normalize_eval_config(cfg)

    train_timestamp = _extract_timestamp(yaml_path)
    if train_timestamp is None:
        print(f"[Warning: !] Could not parse timestamp from config filename: {yaml_path.name}")
        raise SystemExit(1)

    dataset_tag = args.dataset

    # Auto-select irregular variant for poisson (mirrors train script logic).
    if dataset_tag == "poisson" and cfg.get("geofno_variant", "fno") == "fno":
        print("[*] Poisson dataset detected — switching to 'irregular' variant.\n")
        cfg["geofno_variant"] = "irregular"

    save_dir_cfg = Path(cfg.get("save_dir", "Save_TrainedModel/geofno_supervised"))
    model_root = demo_root / "Save_TrainedModel" / dataset_tag / f"{save_dir_cfg.name}_DemoN{args.Demo_Num}_{train_timestamp}"

    if not model_root.exists():
        print(f"[Warning: !] Matching model directory not found: {model_root}")
        raise SystemExit(1)

    ckpt_path = model_root / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        print(f"[Warning: !] Checkpoint not found: {ckpt_path}")
        raise SystemExit(1)

    device = torch.device(args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    # ── Dataset ─────────────────────────────────────────────────────────
    _default_data_paths = {
        "turbulent_combustion": "Dataset/Merged_CH4COTU1P.h5",
        "poisson": "Dataset/nonlinear_poisson.obj",
    }
    data_path = cfg.get("data", _default_data_paths.get(dataset_tag))

    if dataset_tag == "turbulent_combustion":
        dataset = TurbulentCombustionH5Dataset(
            data_path,
            split=args.split,
            train_ratio=cfg.get("train_ratio", 0.9),
            seed=cfg.get("seed", 42),
            time_stride=cfg.get("time_stride", 1),
            stats_path=str(model_root / "dataset_stats.pt"),
        )
    elif dataset_tag == "poisson":
        dataset = NonlinearPoissonDataset(
            data_path,
            split=args.split,
            train_ratio=0.7,
            seed=cfg.get("seed", 42),
            n_points=cfg.get("poisson_n_points", 6000),
            n_bound=cfg.get("poisson_n_bound", 1024),
            stats_path=str(model_root / "dataset_stats.pt"),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_tag}")

    # ── Build model ─────────────────────────────────────────────────────
    try:
        model = _build_model(cfg, dataset.num_fields).to(device)
    except Exception as e:
        print(f"[Warning: !] Model construction failed: {e}")
        raise SystemExit(1)

    variant = cfg.get("geofno_variant", "fno")
    print(f"[*] GeoFNO supervised variant: {variant}")

    # ── Load checkpoint ─────────────────────────────────────────────────
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except pickle.UnpicklingError:
        print("[Warning: !] Restricted torch.load failed; retrying with weights_only=False.")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = dict(state_dict)
        state_dict.pop("_metadata", None)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[Warning: !] Checkpoint is incompatible with the reconstructed model: {e}")
        raise SystemExit(1)

    model.eval()

    # ── Resolve evaluation parameters ───────────────────────────────────
    vis_cond_fields = args.vis_cond_fields if args.vis_cond_fields is not None else cfg["vis_cond_fields"]
    vis_n_obs_list = args.vis_n_obs_list if args.vis_n_obs_list is not None else cfg["vis_n_obs_list"]

    Num_x = cfg["Num_x"]
    Num_y = cfg["Num_y"]

    print(f"\nSupervised FNO evaluation (single forward pass)\n")

    # ── Output directory ────────────────────────────────────────────────
    from datetime import datetime
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = demo_root / "Save_reconstruction_files" / dataset_tag / "ForOfflineEvaluation" / f"eval_geofno_sup_N{args.Demo_Num}_{eval_timestamp}_from_{train_timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Run evaluation ──────────────────────────────────────────────────
    is_irregular = (variant == "irregular")
    metrics = evaluate_snapshot(
        model=model, dataset=dataset,
        Num_x=Num_x, Num_y=Num_y,
        epoch=int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0,
        device=device, save_dir=str(out_dir),
        cond_fields=vis_cond_fields, n_obs=vis_n_obs_list,
        snapshot_index=args.snapshot_index,
        file_tag=f"snapshot_{args.snapshot_index:04d}",
        save_metrics_json=True, irregular=is_irregular,
        compute_physics=args.physics_metrics,
    )

    # ── Save evaluation summary ─────────────────────────────────────────
    summary = {
        "demo_num": int(args.Demo_Num),
        "yaml_path": str(yaml_path),
        "model_root": str(model_root),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "snapshot_index": int(args.snapshot_index),
        "vis_cond_fields": [int(v) for v in vis_cond_fields],
        "vis_n_obs_list": [int(v) for v in vis_n_obs_list],
        "geofno_variant": variant,
        "method": "GeoFNO_supervised",
        "metrics": metrics,
    }

    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[*] Evaluation finished.")
    print(f"[*] YAML      : {yaml_path}")
    print(f"[*] Checkpoint: {ckpt_path}")
    print(f"[*] Output dir : {out_dir}")
    print(f"[*] Metrics    : {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()

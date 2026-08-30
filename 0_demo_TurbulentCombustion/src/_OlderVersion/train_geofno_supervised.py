"""
GeoFNO / FNOGNO supervised regression baseline for sparse field reconstruction.

Uses the neuraloperator library's FNO (for regular grids) or FNOGNO (for
irregular meshes) with direct supervised MSE loss.  The model learns a mapping
    sparse observations  -->  full-field reconstruction
in a single forward pass, without any generative / flow-matching objective.

Key differences from the flow-matching GeoFNO baseline:
  - Training:  Supervised MSE regression (not 1-Rectified Flow)
  - Inference: Single forward pass (not ODE integration)
  - Architecture: neuraloperator FNO / FNOGNO (not custom deformation backbone)

Usage:
    python train_geofno_supervised.py --config Save_config/config_geofno_supervised_turbulent_combustion.yaml
"""

import argparse
import yaml
import shutil
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── PhyCoFlow helpers (same data pipeline) ──────────────────────────────────
from helpers import (
    MetricsLogger,
    TurbulentCombustionH5Dataset,
    NonlinearPoissonDataset,
    ElasticityDataset,
    AirfoilCGridDataset,
    CarCFDDataset,
    validate_regular_grid_compatibility,
    create_recon_dir,
    build_sparse_condition,
    collate_snapshots,
    _build_structured_triangulation,
    _save_single_field_plot as _shared_save_single_field_plot,
    find_latest_run_dir,
    extract_run_timestamp,
    backup_existing_artifact,
)

# ── neuraloperator imports ──────────────────────────────────────────────────
_neuralop_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "..", "FNO", "neuraloperator"
)
if os.path.isdir(_neuralop_path) and _neuralop_path not in sys.path:
    sys.path.insert(0, _neuralop_path)

from neuralop.models import FNO as NeuralOpFNO



# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pointcloud_to_grid(x: torch.Tensor, Num_y: int, Num_x: int) -> torch.Tensor:
    """[B, N, C] -> [B, C, Num_y, Num_x]."""
    B = x.shape[0]
    return x.reshape(B, Num_y, Num_x, -1).permute(0, 3, 1, 2).contiguous()


def grid_to_pointcloud(x_grid: torch.Tensor, Num_y: int, Num_x: int) -> torch.Tensor:
    """[B, C, Num_y, Num_x] -> [B, N, C]."""
    B, C, _, _ = x_grid.shape
    return x_grid.permute(0, 2, 3, 1).reshape(B, Num_y * Num_x, C).contiguous()


def build_obs_grid_mask(
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: torch.Tensor,
    n_fields: int,
    n_pts: int,
    Num_y: int,
    Num_x: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert PhyCoFlow sparse obs tensors to dense grid maps.

    Returns:
        obs_value_grid:  [B, C, Num_y, Num_x]
        obs_mask_grid:   [B, C, Num_y, Num_x]
    """
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
# Model wrappers
# ═════════════════════════════════════════════════════════════════════════════

class FNOSupervisedGrid(nn.Module):
    """
    Wrapper around neuraloperator FNO for supervised sparse→full reconstruction
    on a regular grid.

    Input channels  = 2 * n_fields  (observed-value maps + mask maps)
    Output channels = n_fields
    """

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

    def forward(
        self,
        obs_value_grid: torch.Tensor,
        obs_mask_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            obs_value_grid: [B, C, Ny, Nx] observed values (0 where unobserved)
            obs_mask_grid:  [B, C, Ny, Nx] binary mask (1 where observed)
        Returns:
            pred_grid: [B, C, Ny, Nx] predicted full field
        """
        x = torch.cat([obs_value_grid, obs_mask_grid], dim=1)  # [B, 2C, Ny, Nx]
        return self.fno(x)  # [B, C, Ny, Nx]


class FNOSupervisedIrregular(nn.Module):
    """
    Supervised FNO for irregular meshes.

    Architecture follows the GeoFNO idea (Li et al. 2023):
      1. Sparse observations are scattered onto a regular latent grid
         via bilinear splatting.
      2. neuraloperator FNO processes the latent grid.
      3. Predictions are gathered at arbitrary query points via
         F.grid_sample (bilinear interpolation).

    No GNO dependencies (open3d / torch_scatter) required.
    """

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

    # ── Splat: scatter point values onto latent grid ────────────────────

    def _splat_to_grid(
        self,
        coords_2d: torch.Tensor,
        values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Bilinear splatting of per-point values onto a regular latent grid.

        Args:
            coords_2d: [B, K, 2] normalised coords in [0, 1]
            values:    [B, K, D] feature values at those points
        Returns:
            grid:   [B, D, Ny, Nx] accumulated values
            weight: [B, 1, Ny, Nx] accumulated weights (for normalisation)
        """
        B, K, D = values.shape
        Ny, Nx = self.latent_Ny, self.latent_Nx
        device = values.device
        dtype = values.dtype

        # Continuous pixel coordinates
        px = coords_2d[..., 0] * (Nx - 1)  # [B, K]
        py = coords_2d[..., 1] * (Ny - 1)

        ix0 = px.long().clamp(0, Nx - 2)
        iy0 = py.long().clamp(0, Ny - 2)
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        wx = (px - ix0.float()).unsqueeze(-1)  # [B, K, 1]
        wy = (py - iy0.float()).unsqueeze(-1)

        grid = torch.zeros(B, D, Ny * Nx, device=device, dtype=dtype)
        weight = torch.zeros(B, 1, Ny * Nx, device=device, dtype=dtype)

        for dy, dwy in [(iy0, 1 - wy), (iy1, wy)]:
            for dx, dwx in [(ix0, 1 - wx), (ix1, wx)]:
                w = dwy * dwx                      # [B, K, 1]
                flat = (dy * Nx + dx).unsqueeze(1)  # [B, 1, K]

                wf = (values * w).permute(0, 2, 1)  # [B, D, K]
                grid.scatter_add_(2, flat.expand_as(wf), wf)

                ww = w.permute(0, 2, 1)              # [B, 1, K]
                weight.scatter_add_(2, flat, ww)

        weight = weight.clamp(min=1e-6)
        grid = grid / weight
        return grid.reshape(B, D, Ny, Nx), weight.reshape(B, 1, Ny, Nx)

    # ── Gather: read from grid at arbitrary query points ────────────────

    def _gather_from_grid(
        self,
        grid: torch.Tensor,
        coords_2d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Bilinear interpolation from grid to arbitrary query points.

        Args:
            grid:      [B, C, Ny, Nx]
            coords_2d: [B, M, 2] normalised coords in [0, 1]
        Returns:
            values: [B, M, C]
        """
        # F.grid_sample expects coords in [-1, 1]
        sample_pts = coords_2d * 2.0 - 1.0             # [B, M, 2]
        sample_pts = sample_pts.unsqueeze(2)            # [B, M, 1, 2]
        gathered = F.grid_sample(
            grid, sample_pts, mode="bilinear",
            padding_mode="border", align_corners=True,
        )                                                # [B, C, M, 1]
        return gathered.squeeze(-1).permute(0, 2, 1)    # [B, M, C]

    # ── Forward ─────────────────────────────────────────────────────────

    def forward(
        self,
        coords_2d: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            coords_2d:     [B, M, 2] normalised 2-D coords of ALL points
            obs_values:    [B, max_obs, 1]
            obs_mask:      [B, max_obs]
            obs_field_ids: [B, max_obs]
            obs_indices:   [B, max_obs]
        Returns:
            pred: [B, M, C] predicted field at all query points
        """
        B = coords_2d.shape[0]
        M = coords_2d.shape[1]
        C = self.n_fields
        Ny, Nx = self.latent_Ny, self.latent_Nx
        device = coords_2d.device
        dtype = coords_2d.dtype

        # 1. Scatter sparse observations onto latent grid
        val_grid = torch.zeros(B, C, Ny * Nx, device=device, dtype=dtype)
        msk_grid = torch.zeros(B, C, Ny * Nx, device=device, dtype=dtype)

        for b in range(B):
            valid = obs_mask[b].bool()
            if not valid.any():
                continue
            # Get observed point coordinates
            idx = obs_indices[b, valid].long()
            fld = obs_field_ids[b, valid].long()
            val = obs_values[b, valid, 0]
            obs_xy = coords_2d[b, idx]  # [K_valid, 2]

            # Nearest-cell scatter onto latent grid
            ix = (obs_xy[:, 0] * (Nx - 1)).long().clamp(0, Nx - 1)
            iy = (obs_xy[:, 1] * (Ny - 1)).long().clamp(0, Ny - 1)
            flat = iy * Nx + ix
            val_grid[b, fld, flat] = val
            msk_grid[b, fld, flat] = 1.0

        val_grid = val_grid.reshape(B, C, Ny, Nx)
        msk_grid = msk_grid.reshape(B, C, Ny, Nx)

        # 2. FNO on latent grid
        fno_in = torch.cat([val_grid, msk_grid], dim=1)  # [B, 2C, Ny, Nx]
        pred_grid = self.fno(fno_in)                       # [B, C, Ny, Nx]

        # 3. Gather at all query points via bilinear interpolation
        return self._gather_from_grid(pred_grid, coords_2d)  # [B, M, C]


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation (same normalized L2 metrics as all other baselines)
# ═════════════════════════════════════════════════════════════════════════════

def _normalized_l2(u_true, u_pred):
    return float(np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8))


def _save_single_field_plot(true_f, pred_f, coords_xy, sensor_coords, field_name,
                            epoch, save_dir, file_prefix=None,
                            dpi=300, cmap_field="coolwarm", cmap_err="inferno",
                            contour_levels=20, contour_linewidth=0.5,
                            contour_alpha=0.5, grid_shape=None):
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

    # If the mesh has a structured (Ny, Nx) topology (e.g. airfoil C-grid),
    # plot directly on the curvilinear mesh so geometry features like the
    # airfoil cavity are rendered correctly instead of being filled in by
    # interpolation onto a Cartesian grid.
    use_structured = (
        grid_shape is not None
        and len(grid_shape) == 2
        and grid_shape[0] * grid_shape[1] == coords_xy.shape[0]
    )

    if use_structured:
        Ny, Nx = grid_shape
        GX = coords_xy[:, 0].reshape(Ny, Nx)
        GY = coords_xy[:, 1].reshape(Ny, Nx)
        Z_true = true_f.reshape(Ny, Nx)
        Z_pred = pred_f.reshape(Ny, Nx)
        Z_err  = err.reshape(Ny, Nx)
    else:
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
def visualize_reconstruction_supervised(
    model, dataset,
    Num_x, Num_y,
    epoch, device, save_dir,
    cond_fields=(2,), n_obs=256,
    snapshot_index=0, file_tag=None, save_metrics_json=True,
    irregular=False,
):
    """
    Single-forward-pass reconstruction and evaluation.

    Args:
        irregular: If True, use the irregular-mesh forward path.
    """
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

    # Build sparse observations using the SAME function as other baselines.
    _, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords, fields_full=truth,
        cond_fields=cond_fields, n_obs_min=n_obs, n_obs_max=n_obs,
        valid_mask=valid_mask,
    )

    if irregular:
        # Irregular path: model handles splat → FNO → gather.
        coords_2d = coords[:, :, :2]  # [1, M, 2]
        recon = model(coords_2d, obs_values, obs_mask, obs_field_ids, obs_indices)
    else:
        # Grid path: rasterize to grid, FNO, back to pointcloud.
        n_pts = dataset.num_points
        obs_value_grid, obs_mask_grid = build_obs_grid_mask(
            obs_values, obs_mask, obs_field_ids, obs_indices,
            n_fields, n_pts, Num_y, Num_x,
        )
        pred_grid = model(obs_value_grid, obs_mask_grid)   # [1, C, Ny, Nx]
        recon = grid_to_pointcloud(pred_grid, Num_y, Num_x)  # [1, N, C]

    # Denormalize.
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

    # Structured triangulation / body polygon for airfoil C-grid meshes.
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
# Training loop
# ═════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cond_fields: Sequence[int],
    n_obs_min_list: Sequence[int],
    n_obs_max_list: Sequence[int],
    Num_y: int,
    Num_x: int,
    epoch: int = 0,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    count = 0
    mode_str = "Train" if training else "Eval"
    pbar = tqdm(loader, desc=f"Epoch {epoch:04d} [{mode_str}]", leave=False)

    for batch in pbar:
        coords_full = batch["coords"].to(device)
        fields_full = batch["fields"].to(device)
        B = coords_full.shape[0]
        n_fields = fields_full.shape[2]
        n_pts = fields_full.shape[1]
        valid_mask = batch.get("valid_sensor_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)

        # Build sparse observations.
        _, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords_full,
            fields_full=fields_full,
            cond_fields=cond_fields,
            n_obs_min=n_obs_min_list,
            n_obs_max=n_obs_max_list,
            valid_mask=valid_mask,
        )

        # Rasterize to grid.
        obs_value_grid, obs_mask_grid = build_obs_grid_mask(
            obs_values, obs_mask, obs_field_ids, obs_indices,
            n_fields, n_pts, Num_y, Num_x,
        )

        # Target: full field on the grid.
        target_grid = pointcloud_to_grid(fields_full, Num_y, Num_x)  # [B, C, Ny, Nx]

        # Forward.
        pred_grid = model(obs_value_grid, obs_mask_grid)  # [B, C, Ny, Nx]
        loss = F.mse_loss(pred_grid, target_grid)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    return total_loss / max(count, 1)


def run_epoch_irregular(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cond_fields: Sequence[int],
    n_obs_min_list: Sequence[int],
    n_obs_max_list: Sequence[int],
    epoch: int = 0,
) -> float:
    """Training / validation epoch for irregular-mesh data."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    count = 0
    mode_str = "Train" if training else "Eval"
    pbar = tqdm(loader, desc=f"Epoch {epoch:04d} [{mode_str}]", leave=False)

    for batch in pbar:
        coords_full = batch["coords"].to(device)       # [B, M, 3]
        fields_full = batch["fields"].to(device)        # [B, M, C]

        # 2-D coords for the FNO latent grid (drop z=0 padding)
        coords_2d = coords_full[:, :, :2]              # [B, M, 2]
        valid_mask = batch.get("valid_sensor_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)

        # Build sparse observations.
        _, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords_full,
            fields_full=fields_full,
            cond_fields=cond_fields,
            n_obs_min=n_obs_min_list,
            n_obs_max=n_obs_max_list,
            valid_mask=valid_mask,
        )

        # Forward: model handles splat → FNO → gather internally.
        pred = model(coords_2d, obs_values, obs_mask, obs_field_ids, obs_indices)
        loss = F.mse_loss(pred, fields_full)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    return total_loss / max(count, 1)


# ═════════════════════════════════════════════════════════════════════════════
# CLI & main
# ═════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        "Train GeoFNO *supervised regression* baseline for sparse reconstruction."
    )

    p.add_argument("--config", type=str,
                   default="Save_config/turbulent_combustion/config_geofno_supervised.yaml")
    p.add_argument("--Demo-Num", type=int, default=0)
    p.add_argument("--device-ids", type=int, nargs="+", default=[0])

    # Dataset
    p.add_argument(
        "--dataset", type=str, default="turbulent_combustion",
        choices=["turbulent_combustion", "poisson", "elasticity", "airfoil", "car_cfd"],
        help="Which dataset to use.")
    # Airfoil-specific (overridden via YAML when applicable; harmless otherwise).
    p.add_argument("--sensor-surface-offset-min", dest="sensor_surface_offset_min",
                   type=int, default=1,
                   help="AirfoilCGridDataset: min wall-normal offset (cells) for sensors near the body.")
    p.add_argument("--sensor-surface-offset-max", dest="sensor_surface_offset_max",
                   type=int, default=3,
                   help="AirfoilCGridDataset: max wall-normal offset (cells) for sensors near the body.")
    p.add_argument("--select-fields", dest="select_fields", type=int, nargs="+",
                   default=None,
                   help="Optional field-channel subset for AirfoilCGridDataset.")
    # Car_cfd-specific.
    p.add_argument("--car-n-points", dest="car_n_points", type=int, default=8192,
                   help="Fixed surface-point count per car for CarCFDDataset.")
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--save-dir", type=str,
                   default="Save_TrainedModel/geofno_supervised")
    p.add_argument("--RELOAD", action="store_true",
                   help="If set, try to reload the latest matching checkpoint and continue training.")

    # Training
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)

    # Grid (for regular-grid variant)
    p.add_argument("--Num-x", dest="Num_x", type=int, default=403)
    p.add_argument("--Num-y", dest="Num_y", type=int, default=100)

    # FNO architecture
    p.add_argument("--fno-modes-x", type=int, default=32)
    p.add_argument("--fno-modes-y", type=int, default=8)
    p.add_argument("--fno-hidden-channels", type=int, default=64)
    p.add_argument("--fno-n-layers", type=int, default=4)

    # Variant
    p.add_argument(
        "--geofno-variant", type=str, default="fno",
        choices=["fno", "irregular"],
        help="'fno' uses neuralop FNO on regular grid (fast). "
             "'irregular' uses FNO + splat/gather for irregular meshes.")

    # FNOGNO-specific (only used when variant=fnogno)
    p.add_argument("--latent-Nx", type=int, default=64)
    p.add_argument("--latent-Ny", type=int, default=64)
    p.add_argument("--gno-radius", type=float, default=0.1)
    p.add_argument("--gno-transform-type", type=str, default="linear")

    # Poisson-specific
    p.add_argument("--poisson-n-points", type=int, default=6000)
    p.add_argument("--poisson-n-bound", type=int, default=1024)

    # Sparse conditioning
    p.add_argument("--cond-field", type=int, default=2)
    p.add_argument("--n-obs-min", type=int, default=128)
    p.add_argument("--n-obs-max", type=int, default=512)
    p.add_argument("--cond-fields", type=int, nargs="+", default=None)
    p.add_argument("--n-obs-min-list", type=int, nargs="+", default=None)
    p.add_argument("--n-obs-max-list", type=int, nargs="+", default=None)

    # Visualization
    p.add_argument("--vis-cond-fields", type=int, nargs="+", default=None)
    p.add_argument("--vis-n-obs-list", type=int, nargs="+", default=None)

    # Evaluation schedule
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=200)

    return p.parse_args()


def normalize_conditioning_args(args):
    if args.cond_fields is None:
        args.cond_fields = [args.cond_field]
    if args.n_obs_min_list is None:
        args.n_obs_min_list = [args.n_obs_min]
    if args.n_obs_max_list is None:
        args.n_obs_max_list = [args.n_obs_max]

    n_cf = len(args.cond_fields)
    for attr in ("n_obs_min_list", "n_obs_max_list"):
        lst = getattr(args, attr)
        if len(lst) == 1:
            setattr(args, attr, lst * n_cf)
        elif len(lst) != n_cf:
            setattr(args, attr, lst[:n_cf] if len(lst) > n_cf
                    else lst + [lst[-1]] * (n_cf - len(lst)))

    if args.vis_cond_fields is None:
        args.vis_cond_fields = list(args.cond_fields)
    if args.vis_n_obs_list is None:
        args.vis_n_obs_list = list(args.n_obs_max_list)

    n_vcf = len(args.vis_cond_fields)
    if len(args.vis_n_obs_list) != n_vcf:
        if len(args.vis_n_obs_list) == 1:
            args.vis_n_obs_list = args.vis_n_obs_list * n_vcf
        else:
            args.vis_n_obs_list = (
                args.vis_n_obs_list[:n_vcf]
                if len(args.vis_n_obs_list) > n_vcf
                else args.vis_n_obs_list + [args.vis_n_obs_list[-1]] * (n_vcf - len(args.vis_n_obs_list))
            )
    return args


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    demo_dir = os.path.dirname(script_dir)

    # ── YAML config loading ──────────────────────────────────────────────
    config_path = os.path.join(demo_dir, args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    _cli_snapshot = dict(vars(args))
    _cli_flags = {
        a.lstrip('-').replace('-', '_')
        for a in sys.argv[1:] if a.startswith('--')
    }

    if os.path.exists(config_path):
        print(f"\n[*] Loading config from: {config_path}\n")
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config is not None:
            for key, value in yaml_config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"Warning: YAML key '{key}' not recognized. Ignoring.")

        for flag in _cli_flags:
            if flag in _cli_snapshot:
                setattr(args, flag, _cli_snapshot[flag])
    else:
        print(f"\n[Warning] Config not found at {config_path}. Using defaults.\n")

    args = normalize_conditioning_args(args)
    set_seed(args.seed)

    # ── Directories ──────────────────────────────────────────────────────
    dataset_tag = args.dataset

    start_epoch = 1
    best_val = float("inf")
    reload_ckpt = None
    run_timestamp = timestamp
    save_root = Path(demo_dir) / "Save_TrainedModel" / dataset_tag
    run_prefix = f"{os.path.basename(args.save_dir)}_DemoN{args.Demo_Num}_"
    save_dir = save_root / f"{run_prefix}{timestamp}"

    if args.RELOAD:
        latest_run_dir = find_latest_run_dir(save_root, run_prefix)
        if latest_run_dir is not None and (latest_run_dir / "best.pt").exists():
            save_dir = latest_run_dir
            run_timestamp = extract_run_timestamp(latest_run_dir, run_prefix)
            reload_ckpt = torch.load(latest_run_dir / "best.pt", map_location="cpu")
            start_epoch = int(reload_ckpt.get("epoch", 0)) + 1
            best_val = float(reload_ckpt.get("val_loss", float("inf")))

            backup_existing_artifact(latest_run_dir / "best.pt")
            print(f"[*] RELOAD=True, resuming from: {latest_run_dir / 'best.pt'}")
            print(f"[*] Resume will start from epoch {start_epoch}\n")
        else:
            print("[*] RELOAD=True, but no matching best.pt was found. Training will start from scratch.\n")

    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Backup config — dataset-nested layout (Save_config/<dataset>/geofno_supervised/).
    backup_dir = os.path.join(demo_dir, "Save_config", args.dataset, "geofno_supervised")
    os.makedirs(backup_dir, exist_ok=True)
    backup_filename = f"config_geofno_supervised_DemoN{args.Demo_Num}_{timestamp}.yaml"
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(backup_dir, backup_filename))

    # Loss + recon artifacts live INSIDE the run dir (matching the unified
    # Gen-baseline layout). Save_loss_csv/ is reserved for our model
    # (pointcloud_ffm); Save_reconstruction_files/ likewise stays clean for
    # the FFM offline-eval flow.
    csv_base_dir = str(save_dir)
    recon_base_dir = str(save_dir / "Evaluation")
    os.makedirs(recon_base_dir, exist_ok=True)

    if args.RELOAD and reload_ckpt is not None:
        loss_dir = Path(csv_base_dir) / f"Loss_DemoN{args.Demo_Num}_{run_timestamp}"
        recon_dir_existing = Path(recon_base_dir) / "geofno_supervised" / f"demo_N{args.Demo_Num}_{run_timestamp}"
        backup_existing_artifact(loss_dir)
        backup_existing_artifact(recon_dir_existing)

    logger = MetricsLogger(
        base_dir=csv_base_dir, Demo_Num=args.Demo_Num,
        timestamp=run_timestamp, method_name="GeoFNO-Sup",
    )
    recon_dir = create_recon_dir(
        base_dir=recon_base_dir, Demo_Num=args.Demo_Num,
        timestamp=run_timestamp, method_name="geofno_supervised",
    )

    print(f"[*] Model checkpoints: {save_dir}")
    print(f"[*] Loss logs:         {logger.save_dir}")
    print(f"[*] Recon plots:       {recon_dir}\n")

    device_ids = args.device_ids
    device = torch.device(
        f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}\n")

    # ── Data ─────────────────────────────────────────────────────────────
    _default_data_paths = {
        "turbulent_combustion": "Dataset/Merged_CH4COTU1P.h5",
        "poisson": "Dataset/nonlinear_poisson.obj",
        "elasticity": "Dataset/elasticity",
        "airfoil": "Dataset/airfoil",
        "car_cfd": "Dataset/car_cfd/shapenet_car",
    }
    if args.data is None:
        args.data = _default_data_paths.get(args.dataset)
    data_path = (
        os.path.join(demo_dir, args.data)
        if not os.path.isabs(args.data) else args.data
    )

    # Auto-select irregular variant for poisson if still set to grid default.
    if args.dataset == "poisson" and args.geofno_variant == "fno":
        print("[*] Poisson dataset detected — switching to 'irregular' variant.\n")
        args.geofno_variant = "irregular"

    if args.dataset == "turbulent_combustion":
        train_set = TurbulentCombustionH5Dataset(
            data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, time_stride=args.time_stride,
            stats_path=str(save_dir / "dataset_stats.pt"),
        )
        val_set = TurbulentCombustionH5Dataset(
            data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, time_stride=args.time_stride,
            stats_path=str(save_dir / "dataset_stats.pt"),
        )
        if args.geofno_variant == "fno":
            validate_regular_grid_compatibility(train_set, args.Num_x, args.Num_y)
            validate_regular_grid_compatibility(val_set, args.Num_x, args.Num_y)
    elif args.dataset == "poisson":
        train_set = NonlinearPoissonDataset(
            data_path, split="train", train_ratio=0.7,
            seed=args.seed,
            n_points=args.poisson_n_points,
            n_bound=args.poisson_n_bound,
            stats_path=str(save_dir / "dataset_stats.pt"),
        )
        val_set = NonlinearPoissonDataset(
            data_path, split="val", train_ratio=0.7,
            seed=args.seed,
            n_points=args.poisson_n_points,
            n_bound=args.poisson_n_bound,
            stats_path=str(save_dir / "dataset_stats.pt"),
        )
    elif args.dataset == "elasticity":
        train_set = ElasticityDataset(
            data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed,
            stats_path=str(save_dir / "dataset_stats.pt"),
        )
        val_set = ElasticityDataset(
            data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed,
            stats_path=str(save_dir / "dataset_stats.pt"),
        )
        # 41x41 elasticity is a regular grid → "fno" variant works directly.
        if args.geofno_variant == "fno":
            validate_regular_grid_compatibility(train_set, args.Num_x, args.Num_y)
            validate_regular_grid_compatibility(val_set, args.Num_x, args.Num_y)
    elif args.dataset == "airfoil":
        # NACA C-grid is a deformed (irregular) mesh — must use the
        # 'irregular' variant (FNOSupervisedIrregular: splat → 2D FNO → gather).
        if args.geofno_variant != "irregular":
            print(f"[Warning] airfoil + geofno_variant='{args.geofno_variant}': "
                  f"forcing 'irregular' (the only supported variant for the C-grid).")
            args.geofno_variant = "irregular"
        train_set = AirfoilCGridDataset(
            data_dir=data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=str(save_dir / "dataset_stats.pt"),
            select_fields=args.select_fields,
            sensor_surface_offset_min=args.sensor_surface_offset_min,
            sensor_surface_offset_max=args.sensor_surface_offset_max,
        )
        val_set = AirfoilCGridDataset(
            data_dir=data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=str(save_dir / "dataset_stats.pt"),
            select_fields=args.select_fields,
            sensor_surface_offset_min=args.sensor_surface_offset_min,
            sensor_surface_offset_max=args.sensor_surface_offset_max,
        )
    elif args.dataset == "car_cfd":
        # 3D ShapeNet surface — irregular variant only. NOTE: the current
        # FNOSupervisedIrregular is 2D; running car_cfd here will fail until
        # the splat/FNO/gather pipeline is extended to 3D.
        if args.geofno_variant != "irregular":
            print(f"[Warning] car_cfd + geofno_variant='{args.geofno_variant}': "
                  f"forcing 'irregular' (3D surface mesh).")
            args.geofno_variant = "irregular"
        train_set = CarCFDDataset(
            data_dir=data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, n_points=args.car_n_points,
            stats_path=str(save_dir / "dataset_stats.pt"),
        )
        val_set = CarCFDDataset(
            data_dir=data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, n_points=args.car_n_points,
            stats_path=str(save_dir / "dataset_stats.pt"),
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"[*] Dataset: {args.dataset} ({data_path})")
    print(f"[*] Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    print(f"[*] Fields: {train_set.num_fields}\n")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
    )

    # ── Model ────────────────────────────────────────────────────────────
    n_fields = train_set.num_fields
    variant = args.geofno_variant

    is_irregular = (variant == "irregular")

    if variant == "fno":
        model = FNOSupervisedGrid(
            n_fields=n_fields,
            Num_x=args.Num_x,
            Num_y=args.Num_y,
            n_modes_x=args.fno_modes_x,
            n_modes_y=args.fno_modes_y,
            hidden_channels=args.fno_hidden_channels,
            n_layers=args.fno_n_layers,
        ).to(device)
        print(f"[*] neuraloperator FNO (supervised, grid)")
        print(f"[*] Grid: {args.Num_y}x{args.Num_x}, "
              f"modes=({args.fno_modes_y}, {args.fno_modes_x}), "
              f"hidden={args.fno_hidden_channels}, layers={args.fno_n_layers}")
    elif variant == "irregular":
        model = FNOSupervisedIrregular(
            n_fields=n_fields,
            latent_Nx=args.latent_Nx,
            latent_Ny=args.latent_Ny,
            n_modes_x=args.fno_modes_x,
            n_modes_y=args.fno_modes_y,
            hidden_channels=args.fno_hidden_channels,
            n_layers=args.fno_n_layers,
        ).to(device)
        print(f"[*] neuraloperator FNO (supervised, irregular via splat/gather)")
        print(f"[*] Latent grid: {args.latent_Ny}x{args.latent_Nx}, "
              f"modes=({args.fno_modes_y}, {args.fno_modes_x}), "
              f"hidden={args.fno_hidden_channels}, layers={args.fno_n_layers}")
    else:
        raise ValueError(f"Unknown geofno_variant: {variant}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Trainable parameters: {n_params:,}\n")

    # ── Optimizer & scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    if reload_ckpt is not None:
        model.load_state_dict(reload_ckpt["model"])
        if "optimizer" in reload_ckpt:
            optimizer.load_state_dict(reload_ckpt["optimizer"])
        if reload_ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(reload_ckpt["scheduler"])
        print("[*] Reloaded model state from best.pt")

    # ── Training loop ────────────────────────────────────────────────────
    # Select training function based on variant.
    _run_epoch_fn = run_epoch_irregular if is_irregular else run_epoch

    def _run(loader, opt, ep):
        if is_irregular:
            return _run_epoch_fn(
                model=model, loader=loader, optimizer=opt,
                device=device, cond_fields=args.cond_fields,
                n_obs_min_list=args.n_obs_min_list,
                n_obs_max_list=args.n_obs_max_list,
                epoch=ep,
            )
        return _run_epoch_fn(
            model=model, loader=loader, optimizer=opt,
            device=device, cond_fields=args.cond_fields,
            n_obs_min_list=args.n_obs_min_list,
            n_obs_max_list=args.n_obs_max_list,
            Num_y=args.Num_y, Num_x=args.Num_x, epoch=ep,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss = _run(train_loader, optimizer, epoch)
        scheduler.step()
        print(f"[train] epoch={epoch:04d} loss={tr_loss:.6e}")

        val_loss = None
        if epoch % args.eval_every == 0 or epoch == 1:
            with torch.no_grad():
                val_loss = _run(val_loader, None, epoch)
            print(f"[valid] epoch={epoch:04d} loss={val_loss:.6e}")

            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "mean": train_set.mean,
                "std": train_set.std,
                "field_names": train_set.field_names,
                "method": "supervised_regression",
                "geofno_variant": variant,
                "Num_x": args.Num_x,
                "Num_y": args.Num_y,
            }
            torch.save(ckpt, save_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, save_dir / "best.pt")
                print("Saving the best model...")

        if epoch % args.save_every == 0:
            recon_dir_epoch = os.path.join(recon_dir, f"Epoch_{epoch}")
            os.makedirs(recon_dir_epoch, exist_ok=True)
            model.eval()

            recon_metrics = visualize_reconstruction_supervised(
                model=model, dataset=val_set,
                Num_x=args.Num_x, Num_y=args.Num_y,
                epoch=epoch, device=device, save_dir=recon_dir_epoch,
                cond_fields=args.vis_cond_fields, n_obs=args.vis_n_obs_list,
                snapshot_index=0, file_tag="geofno_supervised",
                save_metrics_json=True, irregular=is_irregular,
            )
            metric_str = ", ".join(
                [f"{k}:{v:.4e}" for k, v in recon_metrics.items()]
            )
            print(f"[recon] epoch={epoch:04d} | {metric_str}")

        logger.log_and_plot(epoch=epoch, train_loss=tr_loss, val_loss=val_loss)

    print("Training complete.")
    print(f"Best validation loss: {best_val:.6e}")


if __name__ == "__main__":
    main()

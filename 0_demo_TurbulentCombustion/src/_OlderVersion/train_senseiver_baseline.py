#!/usr/bin/env python3
"""
Senseiver baseline for PhyCoFlow sparse-sensor reconstruction.

Implements the Senseiver architecture (Janny et al., 2023) — a Perceiver-IO
style encoder-decoder — adapted for multi-field point-cloud datasets.
Trained with MSE loss for deterministic sparse → full-field reconstruction.

Usage:
    python train_senseiver_baseline.py --config ../Save_config/config_senseiver_turbulent_combustion.yaml
"""

import argparse
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm

from helpers import (
    TurbulentCombustionH5Dataset,
    NonlinearPoissonDataset,
    ElasticityDataset,
    AirfoilCGridDataset,
    CarCFDDataset,
    MetricsLogger,
    build_sparse_condition,
    collate_snapshots,
    _save_single_field_plot,
    _save_car_surface_field_plot,
    _normalized_l2,
    _build_structured_triangulation,
    find_latest_run_dir,
    extract_run_timestamp,
    backup_existing_artifact,
)


# ═════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════════════════════════════════════
# Model: Senseiver (Perceiver-IO for sparse-sensor reconstruction)
# ═════════════════════════════════════════════════════════════════════════════

class FourierPositionalEncoding(nn.Module):
    """Sine-cosine frequency encoding for spatial coordinates (Senseiver-style)."""

    def __init__(self, coord_dim: int = 2, num_bands: int = 32, max_freq: float = 64.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_bands = num_bands
        self.out_dim = coord_dim * num_bands * 2  # sin + cos per dim per band
        freqs = torch.linspace(1.0, max_freq / 2.0, num_bands)
        self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: [..., coord_dim] in [0, 1]
        Returns:
            encodings: [..., out_dim]
        """
        coords = coords[..., : self.coord_dim] * 2.0 - 1.0  # scale to [-1, 1]
        # [..., D, 1] * [F] -> [..., D, F]
        x = coords.unsqueeze(-1) * self.freqs * math.pi
        enc = torch.cat([x.sin(), x.cos()], dim=-1)  # [..., D, 2F]
        return enc.reshape(*coords.shape[:-1], self.out_dim)


class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + MLP with residual connections."""

    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class EncoderBlock(nn.Module):
    """One cross-attention (sensors → latent) + K self-attention layers."""

    def __init__(
        self,
        latent_dim: int,
        kv_dim: int,
        num_cross_heads: int,
        num_self_heads: int,
        num_self_attn_layers: int = 3,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.cross_attn = nn.MultiheadAttention(
            latent_dim, num_cross_heads,
            kdim=kv_dim, vdim=kv_dim,
            dropout=dropout, batch_first=True,
        )
        self.self_attn_layers = nn.ModuleList([
            SelfAttentionBlock(latent_dim, num_self_heads, ff_mult, dropout)
            for _ in range(num_self_attn_layers)
        ])

    def forward(
        self, latent: torch.Tensor, kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self.norm_q(latent)
        k = self.norm_kv(kv)
        h, _ = self.cross_attn(q, k, k, key_padding_mask=key_padding_mask)
        latent = latent + h
        for sa in self.self_attn_layers:
            latent = sa(latent)
        return latent


class Senseiver(nn.Module):
    """
    Senseiver: Perceiver-IO for sparse-sensor field reconstruction.

    Encoder compresses sparse, multi-field sensor observations into a fixed-size
    latent array via cross-attention.  Decoder expands the latent to arbitrary
    query locations via a second cross-attention.

    Adapted to support multi-field conditioning (each sensor carries a field ID).
    """

    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 2,
        num_latents: int = 128,
        latent_dim: int = 256,
        num_encoder_layers: int = 3,
        num_self_attn_per_block: int = 3,
        num_cross_attn_heads: int = 4,
        num_self_attn_heads: int = 4,
        dec_num_cross_attn_heads: int = 4,
        field_embed_dim: int = 32,
        space_bands: int = 32,
        max_freq: float = 64.0,
        ff_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_fields = n_fields
        self.coord_dim = coord_dim

        # Positional encoding
        self.pos_enc = FourierPositionalEncoding(coord_dim, space_bands, max_freq)
        pos_dim = self.pos_enc.out_dim

        # Field embedding for multi-field sensor inputs
        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        # Encoder input projection:  value (1) + field_embed + pos_encoding → latent_dim
        enc_input_dim = 1 + field_embed_dim + pos_dim
        self.encoder_preproc = nn.Linear(enc_input_dim, latent_dim)

        # Learnable latent array
        self.latent = nn.Parameter(torch.randn(num_latents, latent_dim) * 0.02)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                latent_dim, latent_dim, num_cross_attn_heads,
                num_self_attn_heads, num_self_attn_per_block, ff_mult, dropout,
            )
            for _ in range(num_encoder_layers)
        ])

        # Decoder
        self.decoder_query_token = nn.Parameter(torch.randn(1, latent_dim) * 0.02)
        dec_input_dim = pos_dim + latent_dim
        self.decoder_preproc = nn.Linear(dec_input_dim, latent_dim)
        self.decoder_norm_q = nn.LayerNorm(latent_dim)
        self.decoder_norm_kv = nn.LayerNorm(latent_dim)
        self.decoder_cross_attn = nn.MultiheadAttention(
            latent_dim, dec_num_cross_attn_heads,
            dropout=dropout, batch_first=True,
        )
        self.decoder_postproc = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, n_fields),
        )

    def forward(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query_coords:  [B, M, D]  query point coordinates (normalized [0,1])
            obs_coords:    [B, S, D]  sensor coordinates
            obs_values:    [B, S, 1]  sensor values
            obs_mask:      [B, S]     binary mask (1 = valid sensor)
            obs_field_ids: [B, S]     field index per sensor
        Returns:
            pred: [B, M, n_fields]  predicted fields at query points
        """
        B, M, _ = query_coords.shape

        # ── Encoder ──────────────────────────────────────────────────────
        sensor_pos = self.pos_enc(obs_coords)                       # [B, S, pos_dim]
        # build_sparse_condition fills unused sensor slots with field_id=-1
        # (non-uniform n_obs across batch). Clamp the index so the embedding
        # lookup is in-range; the corresponding rows are masked out by
        # `obs_mask` / `key_pad_mask` later, so the bogus embedding never
        # contributes to the output.
        sensor_field = self.field_embed(obs_field_ids.long().clamp(min=0))  # [B, S, field_embed_dim]
        sensor_input = torch.cat([obs_values, sensor_field, sensor_pos], dim=-1)
        sensor_input = self.encoder_preproc(sensor_input)           # [B, S, latent_dim]

        key_pad_mask = ~obs_mask.bool()  # True = ignore
        latent = self.latent.unsqueeze(0).expand(B, -1, -1)        # [B, L, latent_dim]
        for block in self.encoder_blocks:
            latent = block(latent, sensor_input, key_padding_mask=key_pad_mask)

        # ── Decoder ──────────────────────────────────────────────────────
        query_pos = self.pos_enc(query_coords)                      # [B, M, pos_dim]
        dq = self.decoder_query_token.expand(B, M, -1)             # [B, M, latent_dim]
        query_input = torch.cat([query_pos, dq], dim=-1)           # [B, M, pos_dim+latent_dim]
        query_input = self.decoder_preproc(query_input)             # [B, M, latent_dim]

        q = self.decoder_norm_q(query_input)
        kv = self.decoder_norm_kv(latent)
        h, _ = self.decoder_cross_attn(q, kv, kv)
        output = query_input + h                                    # residual

        return self.decoder_postproc(output)                        # [B, M, n_fields]


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation / Visualization
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def visualize_reconstruction_senseiver(
    model: nn.Module,
    dataset,
    epoch: int,
    device: torch.device,
    save_dir: str,
    cond_fields: Sequence[int] = (2,),
    n_obs: Union[int, Sequence[int]] = 256,
    snapshot_index: int = 0,
    file_tag: Optional[str] = None,
    save_metrics_json: bool = True,
) -> Dict[str, float]:
    """Reconstruct full fields and save plots (same format as FFM evaluator)."""
    model.eval()

    if isinstance(cond_fields, int):
        cond_fields = [cond_fields]
    if isinstance(n_obs, int):
        n_obs = [n_obs] * len(cond_fields)

    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    valid_mask = sample.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0).to(device)

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords, fields_full=truth,
        cond_fields=cond_fields, n_obs_min=n_obs, n_obs_max=n_obs,
        valid_mask=valid_mask,
    )

    # For 3D surface datasets (e.g. CarCFDDataset) the dataset can return a
    # dense un-subsampled mesh for plotting; the FFM visualizer follows the
    # same pattern. Sample model predictions on the dense mesh so the field
    # renders as a continuous color map rather than a sparse scatter.
    use_full_mesh = hasattr(dataset, 'load_full_mesh_sample')
    mesh_vertices = None
    mesh_triangles = None
    if use_full_mesh:
        full = dataset.load_full_mesh_sample(snapshot_index)
        plot_coords = full['coords'].unsqueeze(0).to(device)
        plot_coords_raw = full['coords_raw'].unsqueeze(0)
        plot_truth = full['fields'].unsqueeze(0).to(device)
        mesh_vertices = full.get('vertices')
        mesh_triangles = full.get('triangles')
    else:
        plot_coords = coords
        plot_coords_raw = coords_raw
        plot_truth = truth

    recon = model(plot_coords, obs_coords, obs_values, obs_mask, obs_field_ids)

    # Denormalize
    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = plot_truth * std.view(1, 1, -1) + mean.view(1, 1, -1)

    recon_np = recon_phys[0].cpu().numpy()
    truth_np = truth_phys[0].cpu().numpy()

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()
    # Sensor coords are picked from the training subsample, not the full mesh,
    # so carry them in physical units (3D-safe) to overlay on either coord set.
    obs_coords_raw = coords_raw[0, obs_indices[0].cpu()].cpu().numpy()[valid.cpu().numpy()]

    coords_np = plot_coords_raw[0].cpu().numpy()
    coords_xy = coords_np[:, :2]
    # Detect 3D surface (CarCFDDataset): >=3 coord dims, no structured grid,
    # and non-trivial variation along z. Matches FFM's detection.
    is_3d_surface = (
        coords_np.shape[-1] >= 3
        and getattr(dataset, 'grid_shape', None) is None
        and bool(np.ptp(coords_np[:, 2]) > 1e-6)
    )

    # Structured triangulation / body polygon for 2D grid datasets only.
    triang = None
    body_polygon = None
    if not is_3d_surface:
        if hasattr(dataset, "grid_shape") and dataset.grid_shape is not None:
            triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)
        if hasattr(dataset, "airfoil_body_indices") and dataset.airfoil_body_indices is not None:
            body_polygon = coords_xy[dataset.airfoil_body_indices]

    metrics = {}
    n_fields = truth_np.shape[1]
    field_names = (
        dataset.field_names if len(dataset.field_names) == n_fields
        else tuple(f"field_{i}" for i in range(n_fields))
    )

    for c, name in enumerate(field_names):
        sensor_coords = None
        field_mask = obs_field_ids_cpu == c
        if np.any(field_mask):
            if use_full_mesh:
                # Sensors are from the training subsample; their physical
                # coords already live in the same frame as the dense mesh.
                sensor_coords = obs_coords_raw[field_mask]
                if not is_3d_surface:
                    sensor_coords = sensor_coords[:, :2]
            elif is_3d_surface:
                sensor_coords = coords_np[obs_indices_cpu[field_mask]]
            else:
                sensor_coords = coords_xy[obs_indices_cpu[field_mask]]

        if is_3d_surface:
            l2 = _save_car_surface_field_plot(
                true_f=truth_np[:, c],
                pred_f=recon_np[:, c],
                coords_xyz=coords_np,
                sensor_coords=sensor_coords,
                vertices=mesh_vertices,
                triangles=mesh_triangles,
                field_name=name,
                epoch=epoch,
                save_dir=save_dir,
                file_prefix=file_tag,
            )
        else:
            l2 = _save_single_field_plot(
                true_f=truth_np[:, c],
                pred_f=recon_np[:, c],
                coords_xy=coords_xy,
                sensor_coords=sensor_coords,
                field_name=name,
                epoch=epoch,
                save_dir=save_dir,
                file_prefix=file_tag,
                triang=triang,
                body_polygon=body_polygon,
            )
        metrics[name] = l2

    if save_metrics_json:
        prefix = file_tag or f"epoch_{epoch:04d}"
        payload = {
            "epoch": int(epoch),
            "snapshot_index": int(snapshot_index),
            "cond_fields": [int(v) for v in cond_fields],
            "n_obs": [int(v) for v in n_obs],
            "method": "Senseiver",
            "metrics": metrics,
        }
        with open(os.path.join(save_dir, f"{prefix}_metrics.json"), "w") as f:
            json.dump(payload, f, indent=2)

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cond_fields: Sequence[int],
    n_obs_min_list: Sequence[int],
    n_obs_max_list: Sequence[int],
    n_query_points: int,
    epoch: int = 0,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    count = 0
    desc = f"Epoch {epoch:04d} [{'Train' if training else 'Valid'}]"
    pbar = tqdm(loader, desc=desc, leave=False)

    for batch in pbar:
        coords = batch["coords"].to(device)   # [B, N, D]
        fields = batch["fields"].to(device)    # [B, N, C]
        B, N, _ = coords.shape
        valid_mask = batch.get("valid_sensor_mask")
        if valid_mask is not None:
            valid_mask = valid_mask.to(device)

        # Build sparse observations
        obs_coords, obs_values, obs_mask, _, obs_field_ids = build_sparse_condition(
            coords_full=coords, fields_full=fields,
            cond_fields=cond_fields,
            n_obs_min=n_obs_min_list,
            n_obs_max=n_obs_max_list,
            valid_mask=valid_mask,
        )

        # Sub-sample query points for memory efficiency
        if n_query_points > 0 and n_query_points < N:
            idx = torch.randperm(N, device=device)[:n_query_points]
            query_coords = coords[:, idx]
            query_fields = fields[:, idx]
        else:
            query_coords = coords
            query_fields = fields

        pred = model(query_coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        loss = F.mse_loss(pred, query_fields)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * B
        count += B
        pbar.set_postfix(loss=f"{loss.item():.4e}")

    return total_loss / max(count, 1)


# ═════════════════════════════════════════════════════════════════════════════
# Config / arg handling
# ═════════════════════════════════════════════════════════════════════════════

def normalize_conditioning_args(args):
    if args.cond_fields is None:
        args.cond_fields = [getattr(args, "cond_field", 2)]
    if args.n_obs_min_list is None:
        args.n_obs_min_list = [getattr(args, "n_obs_min", 64)]
    if args.n_obs_max_list is None:
        args.n_obs_max_list = [getattr(args, "n_obs_max", 256)]

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
                args.vis_n_obs_list[:n_vcf] if len(args.vis_n_obs_list) > n_vcf
                else args.vis_n_obs_list + [args.vis_n_obs_list[-1]] * (n_vcf - len(args.vis_n_obs_list))
            )
    return args


def parse_args():
    p = argparse.ArgumentParser("Senseiver baseline training")
    p.add_argument("--config", type=str, default="Save_config/turbulent_combustion/config_senseiver.yaml")
    p.add_argument("--demo-root", type=str, default=None,
                   help="Project root. Defaults to parent of src/.")

    # Dataset
    p.add_argument("--dataset", type=str, default="turbulent_combustion",
                   choices=["turbulent_combustion", "poisson", "elasticity", "airfoil", "car_cfd"])
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="Save_TrainedModel/senseiver_baseline")
    p.add_argument("--RELOAD", action="store_true",
                   help="If set, try to reload the latest matching checkpoint and continue training.")

    # Training
    p.add_argument("--Demo-Num", dest="Demo_Num", type=int, default=0)
    p.add_argument("--device-ids", type=int, nargs="+", default=[0])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", dest="weight_decay", type=float, default=1e-6)
    p.add_argument("--train-ratio", dest="train_ratio", type=float, default=0.9)
    p.add_argument("--time-stride", dest="time_stride", type=int, default=1)
    p.add_argument("--num-workers", dest="num_workers", type=int, default=4)
    p.add_argument("--n-query-points", dest="n_query_points", type=int, default=4096)
    p.add_argument("--sensor-surface-offset-min", dest="sensor_surface_offset_min", type=int, default=1)
    p.add_argument("--sensor-surface-offset-max", dest="sensor_surface_offset_max", type=int, default=3)

    # Architecture
    p.add_argument("--coord-dim", dest="coord_dim", type=int, default=2)
    p.add_argument("--num-latents", dest="num_latents", type=int, default=128)
    p.add_argument("--latent-dim", dest="latent_dim", type=int, default=256)
    p.add_argument("--num-encoder-layers", dest="num_encoder_layers", type=int, default=3)
    p.add_argument("--num-self-attn-per-block", dest="num_self_attn_per_block", type=int, default=3)
    p.add_argument("--num-cross-attn-heads", dest="num_cross_attn_heads", type=int, default=4)
    p.add_argument("--num-self-attn-heads", dest="num_self_attn_heads", type=int, default=4)
    p.add_argument("--dec-num-cross-attn-heads", dest="dec_num_cross_attn_heads", type=int, default=4)
    p.add_argument("--field-embed-dim", dest="field_embed_dim", type=int, default=32)
    p.add_argument("--space-bands", dest="space_bands", type=int, default=32)
    p.add_argument("--max-freq", dest="max_freq", type=float, default=64.0)
    p.add_argument("--ff-mult", dest="ff_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)

    # Conditioning
    p.add_argument("--cond-fields", dest="cond_fields", type=int, nargs="+", default=None)
    p.add_argument("--cond-field", dest="cond_field", type=int, default=2)
    p.add_argument("--n-obs-min", dest="n_obs_min", type=int, default=64)
    p.add_argument("--n-obs-max", dest="n_obs_max", type=int, default=256)
    p.add_argument("--n-obs-min-list", dest="n_obs_min_list", type=int, nargs="+", default=None)
    p.add_argument("--n-obs-max-list", dest="n_obs_max_list", type=int, nargs="+", default=None)
    p.add_argument("--vis-cond-fields", dest="vis_cond_fields", type=int, nargs="+", default=None)
    p.add_argument("--vis-n-obs-list", dest="vis_n_obs_list", type=int, nargs="+", default=None)

    # Schedule
    p.add_argument("--eval-every", dest="eval_every", type=int, default=5)
    p.add_argument("--save-every", dest="save_every", type=int, default=200)

    # Poisson-specific
    p.add_argument("--poisson-n-points", dest="poisson_n_points", type=int, default=6000)
    p.add_argument("--poisson-n-bound", dest="poisson_n_bound", type=int, default=1024)

    # car_cfd-specific
    p.add_argument("--car-n-points", dest="car_n_points", type=int, default=8192,
                   help="Fixed surface-point count per car for CarCFDDataset.")
    p.add_argument("--car-sensor-min-height-norm", dest="car_sensor_min_height_norm",
                   type=float, default=0.0,
                   help="CarCFDDataset: exclude sensors below this fraction of the "
                        "global up-axis bounding box (0.15 drops wheels / underbody).")

    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Resolve demo root
    script_dir = os.path.dirname(os.path.realpath(__file__))
    demo_dir = args.demo_root or os.path.dirname(script_dir)
    demo_dir = os.path.abspath(demo_dir)

    # Load YAML config — YAML values are defaults, CLI flags take priority
    config_path = os.path.join(demo_dir, args.config)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            yaml_cfg = yaml.safe_load(f) or {}
        # Apply YAML values, then re-parse CLI to override
        parser = parse_args.__self__ if hasattr(parse_args, "__self__") else None
        for key, val in yaml_cfg.items():
            if hasattr(args, key):
                setattr(args, key, val)
    else:
        print(f"[Warning] Config not found: {config_path}. Using CLI defaults.")

    args = normalize_conditioning_args(args)

    # ── Timestamp & directories ──────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    start_epoch = 1
    best_val = float("inf")
    reload_ckpt = None
    run_timestamp = timestamp
    _sd = Path(demo_dir) / args.save_dir
    save_root = _sd.parent
    run_prefix = f"{_sd.name}_DemoN{args.Demo_Num}_"
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

    # Loss + recon artifacts now live INSIDE the run dir (matching the
    # unified Gen-baseline layout). Save_loss_csv/ is reserved for our model
    # (pointcloud_ffm); Save_reconstruction_files/ likewise stays clean for
    # the FFM offline-eval flow.
    csv_base = str(save_dir)
    recon_base = os.path.join(save_dir, "Evaluation")
    os.makedirs(recon_base, exist_ok=True)

    if args.RELOAD and reload_ckpt is not None:
        loss_dir = Path(csv_base) / f"Loss_DemoN{args.Demo_Num}_{run_timestamp}"
        backup_existing_artifact(loss_dir)
        backup_existing_artifact(Path(recon_base))

    logger = MetricsLogger(csv_base, args.Demo_Num, run_timestamp, method_name="Senseiver")

    # Backup config — dataset-nested layout (Save_config/<dataset>/senseiver/).
    backup_dir = os.path.join(demo_dir, "Save_config", args.dataset, "senseiver")
    os.makedirs(backup_dir, exist_ok=True)
    backup_name = f"config_senseiver_DemoN{args.Demo_Num}_{timestamp}.yaml"
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(backup_dir, backup_name))

    # Save args
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────
    device_ids = args.device_ids
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")

    # ── Dataset ──────────────────────────────────────────────────────────
    _default_data = {
        "turbulent_combustion": "Dataset/Merged_CH4COTU1P.h5",
        "poisson": "Dataset/nonlinear_poisson.obj",
        "elasticity": "Dataset/elasticity/Interp",
        "airfoil": "Dataset/airfoil",
        "car_cfd": "Dataset/car_cfd/shapenet_car",
    }
    data_path = args.data or _default_data.get(args.dataset)
    if not os.path.isabs(data_path):
        data_path = os.path.join(demo_dir, data_path)

    stats_path = str(save_dir / "dataset_stats.pt")

    if args.dataset == "turbulent_combustion":
        train_set = TurbulentCombustionH5Dataset(
            data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, time_stride=args.time_stride, stats_path=stats_path,
        )
        val_set = TurbulentCombustionH5Dataset(
            data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, time_stride=args.time_stride, stats_path=stats_path,
        )
    elif args.dataset == "poisson":
        train_set = NonlinearPoissonDataset(
            data_path, split="train", train_ratio=0.7, seed=args.seed,
            n_points=args.poisson_n_points, n_bound=args.poisson_n_bound,
            stats_path=stats_path,
        )
        val_set = NonlinearPoissonDataset(
            data_path, split="val", train_ratio=0.7, seed=args.seed,
            n_points=args.poisson_n_points, n_bound=args.poisson_n_bound,
            stats_path=stats_path,
        )
    elif args.dataset == "elasticity":
        train_set = ElasticityDataset(
            data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=stats_path,
        )
        val_set = ElasticityDataset(
            data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=stats_path,
        )
    elif args.dataset == "airfoil":
        train_set = AirfoilCGridDataset(
            data_dir=data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=stats_path,
            sensor_surface_offset_min=args.sensor_surface_offset_min,
            sensor_surface_offset_max=args.sensor_surface_offset_max,
        )
        val_set = AirfoilCGridDataset(
            data_dir=data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, stats_path=stats_path,
            sensor_surface_offset_min=args.sensor_surface_offset_min,
            sensor_surface_offset_max=args.sensor_surface_offset_max,
        )
    elif args.dataset == "car_cfd":
        train_set = CarCFDDataset(
            data_dir=data_path, split="train", train_ratio=args.train_ratio,
            seed=args.seed, n_points=args.car_n_points, stats_path=stats_path,
            sensor_min_height_norm=args.car_sensor_min_height_norm,
        )
        val_set = CarCFDDataset(
            data_dir=data_path, split="val", train_ratio=args.train_ratio,
            seed=args.seed, n_points=args.car_n_points, stats_path=stats_path,
            sensor_min_height_norm=args.car_sensor_min_height_norm,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

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
    model = Senseiver(
        n_fields=train_set.num_fields,
        coord_dim=args.coord_dim,
        num_latents=args.num_latents,
        latent_dim=args.latent_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_self_attn_per_block=args.num_self_attn_per_block,
        num_cross_attn_heads=args.num_cross_attn_heads,
        num_self_attn_heads=args.num_self_attn_heads,
        dec_num_cross_attn_heads=args.dec_num_cross_attn_heads,
        field_embed_dim=args.field_embed_dim,
        space_bands=args.space_bands,
        max_freq=args.max_freq,
        ff_mult=args.ff_mult,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Senseiver  |  {n_params:,} trainable parameters")
    print(f"[*] Dataset    |  {args.dataset}  train={len(train_set)}  val={len(val_set)}")
    print(f"[*] Fields     |  {train_set.num_fields}  ({', '.join(train_set.field_names)})")
    print(f"[*] Condition  |  fields={args.cond_fields}  "
          f"n_obs=[{','.join(map(str,args.n_obs_min_list))}]-"
          f"[{','.join(map(str,args.n_obs_max_list))}]")
    print(f"[*] Output     |  {save_dir}")

    # ── Optimizer / scheduler ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if reload_ckpt is not None:
        model.load_state_dict(reload_ckpt["model"])
        if "optimizer" in reload_ckpt:
            optimizer.load_state_dict(reload_ckpt["optimizer"])
        if reload_ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(reload_ckpt["scheduler"])
        print("[*] Reloaded Senseiver state from best.pt")

    # ── Training loop ────────────────────────────────────────────────────

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss = run_epoch(
            model, train_loader, optimizer, device,
            args.cond_fields, args.n_obs_min_list, args.n_obs_max_list,
            args.n_query_points, epoch,
        )
        scheduler.step()
        print(f"[train] epoch={epoch:04d}  loss={tr_loss:.6e}")

        # Validation
        val_loss = None
        if epoch % args.eval_every == 0 or epoch == 1:
            val_loss = run_epoch(
                model, val_loader, None, device,
                args.cond_fields, args.n_obs_min_list, args.n_obs_max_list,
                args.n_query_points, epoch,
            )
            print(f"[valid] epoch={epoch:04d}  loss={val_loss:.6e}")

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
                "method": "senseiver",
            }
            torch.save(ckpt, save_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, save_dir / "best.pt")
                print(f"  ↳ best model saved (val_loss={val_loss:.6e})")

        # Visualization
        if epoch % args.save_every == 0:
            recon_dir = os.path.join(recon_base, f"Epoch_{epoch}")
            os.makedirs(recon_dir, exist_ok=True)

            metrics = visualize_reconstruction_senseiver(
                model, val_set, epoch, device, recon_dir,
                cond_fields=args.vis_cond_fields,
                n_obs=args.vis_n_obs_list,
                snapshot_index=0,
            )
            metric_str = ", ".join(f"{k}:{v:.4e}" for k, v in metrics.items())
            print(f"[recon] epoch={epoch:04d}  |  {metric_str}")

        logger.log_and_plot(epoch=epoch, train_loss=tr_loss, val_loss=val_loss)

    print("[*] Training complete.")
    print(f"[*] Best val loss: {best_val:.6e}")
    print(f"[*] Checkpoints:   {save_dir}")


if __name__ == "__main__":
    main()

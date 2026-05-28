
'''
With this patch:

- training can use any configured field combination like [0], [2], [0, 2], [0, 2, 4]

- each conditioned field can have its own n_obs_min / n_obs_max

- visualization can use its own cond_fields and exact n_obs list, independent of training

- Model backbone can be ConditionalPointMLPRBF, ConditionalPointPerceiver
'''

import argparse
import csv
import yaml
import shutil
import json
import math
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime

from helpers import (
    TurbulentCombustionH5Dataset,
    validate_regular_grid_compatibility,
    visualize_reconstruction,
    build_sparse_condition,
)
from Model import (
    ConditionalPointFFM, 
    ConditionalPointMLPRBF, 
    ConditionalPointPerceiver,
    ConditionalPointHybridLocalGlobalRBF,
    PointCloudFFM,
    FNO,
    FNOFFM,
    )

def parse_args():

    p = argparse.ArgumentParser("Train a starter conditional point-cloud FFM on turbulent combustion HDF5 data.")

    p.add_argument("--config", type=str, 
                   default="Save_config/config_pointcloud_ffm.yaml", help="Path to YAML config")
    p.add_argument("--Demo-Num", type=int, 
                   default=0, help="Demo ID tag for saving directories")
    p.add_argument("--device-ids", type=int, nargs="+", default=[0])

    p.add_argument("--data", type=str, 
                   default="Dataset/Merged_CH4COTU1P.h5")
    p.add_argument("--save-dir", type=str, 
                   default=f"Save_TrainedModel/ffm_tc_pointcloud")
    p.add_argument("--RELOAD", action="store_true",
                   help="If set, try to reload the latest matching checkpoint and continue training.")
    
    # ------------------------------
    # Backbone selection
    # ------------------------------
    p.add_argument(
        "--backbone", type=str, default="mlp_rbf", choices = ["mlp_rbf", "perceiver", "fno", "GL_rbf"], 
        help="Backbone type. point-cloud MLP+RBF, point-cloud Perceiver, or grid-based FNO baseline.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)

    # ------------------------------
    # These are hyperparameters for mlp_rbf backbone or part of GL_rbf
    # ------------------------------
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--field-embed-dim", type=int, default=64)
    p.add_argument("--rbf-sigma", type=float, default=0.05)
    p.add_argument("--USE-FOURIER-PE", "--USE_FOURIER_PE", dest="USE_FOURIER_PE", action="store_true",
                   help="If set, feed Fourier positional coordinate features to point_encoder.")
    p.add_argument("--fourier-pe-num-bands", type=int, default=32,
                   help="Number of frequency bands for Fourier positional coordinate encoding.")
    p.add_argument("--fourier-pe-max-freq", type=float, default=64.0,
                   help="Maximum frequency scale for Fourier positional coordinate encoding.")

    # ------------------------------
    # These are hyperparameters for Perceiver backbone or part of GL_rbf
    # ------------------------------
    p.add_argument("--latent-dim", type=int, default=256, 
                   help="Token / latent width for the Perceiver backbone.",)
    p.add_argument("--num-latents", type=int, default=128, 
                   help="Number of learned latent slots in the Perceiver.",)
    p.add_argument("--num-heads", type=int, default=8, 
                   help="Number of attention heads for Perceiver attention blocks.",)
    p.add_argument("--num-latent-blocks", type=int, default=4, 
                   help="Number of latent self-attention blocks.",)
    p.add_argument("--ff-mult", type=int, default=4, 
                   help="Expansion factor for Transformer feed-forward layers.",)
    p.add_argument("--attn-dropout", type=float, default=0.0, 
                   help="Dropout used inside attention layers.",)
    p.add_argument("--mlp-dropout", type=float, default=0.0, 
                   help="Dropout used inside token projection / FFN layers.",)
    p.add_argument("--decode-chunk-size", type=int, default=4096,
                   help="Chunk size for Perceiver output decoding. Useful for full-resolution reconstruction.",)
    p.add_argument("--share-query-proj", action="store_true",
        help="If set, use the same projection for Perceiver encoder query tokens and decoder query tokens.",)

    p.add_argument("--summary-type", type=str, default='cls',
        help="Only for GL_rbf; select either cls or mean",)

    # ----------------------------------------------------------
    # Hybrid local-global gather options
    # ----------------------------------------------------------
    p.add_argument(
        "--gather-mode", type=str, default="rbf", choices=["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"],
        help="Gather mode used by ConditionalPointHybridLocalGlobalRBF. 'rbf' preserves the current full gather as default.",
    )
    p.add_argument(
        "--gather-topk", type=int, default=32, 
        help="Number of nearest refined sensor tokens used in top-k gather modes.",
    )
    p.add_argument(
        "--gather-query-chunk-size", type=int, default=None,
        help="Optional query chunk size for memory-friendly gathering. Applies to all gather modes.",
    )
    p.add_argument(
        "--learnable-rbf-sigma", action="store_true",
        help="If set, make the RBF sigma in the hybrid gather learnable.",
    )
    p.add_argument(
        "--neighbor-backend", type=str, default="torch", choices=["auto", "torch", "keops"],
        help="Neighbor / kernel backend for the hybrid gather. "
            "'auto' uses KeOps if available, otherwise falls back to pure PyTorch.",)
    p.add_argument(
        "--sensor-local-topk", type=int, default=8,
        help="Number of local sensor neighbors used by the sensor-side Point-Transformer refinement in gather_mode='topk_rbf_ptlocal'.",)
    p.add_argument(
        "--sensor-local-dropout", type=float, default=0.0,
        help="Dropout used inside the sensor-side local refinement block for gather_mode='topk_rbf_ptlocal'.",
    )

    # ----------------------------------------------------------
    # These are hyperparameters for fno backbone
    # Num_x / Num_y must be supplied for the FNO baseline.
    # ----------------------------------------------------------
    p.add_argument( "--Num-x", dest="Num_x", type=int, default=None,
        help="Number of grid points along x for the FNO baseline. Required when backbone='fno'.",)
    p.add_argument("--Num-y", dest="Num_y", type=int, default=None,
        help="Number of grid points along y for the FNO baseline. Required when backbone='fno'.",)
    p.add_argument( "--fno-modes-x", type=int, default=32,
        help="Number of retained Fourier modes along x for the FNO baseline.",)
    p.add_argument( "--fno-modes-y", type=int, default=8,
        help="Number of retained Fourier modes along y for the FNO baseline.",)
    p.add_argument( "--fno-hidden-channels", type=int, default=64,
        help="Hidden channel width of the neuraloperator FNO baseline.",)
    p.add_argument( "--fno-n-layers", type=int, default=4,
        help="Number of Fourier layers in the FNO baseline.",)
    p.add_argument(
        "--condition-blur",
        action="store_true",
        help="If set, Gaussian-splat sparse FNO conditioning maps before concatenation.",
    )
    p.add_argument(
        "--condition-blur-kernel",
        type=int,
        default=5,
        help="Odd Gaussian kernel size used to splat sparse FNO conditioning maps.",
    )
    p.add_argument(
        "--condition-blur-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used to splat sparse FNO conditioning maps.",
    )

    # ------------------------------
    # These are hyperparameters for training process
    # ------------------------------
    p.add_argument("--n-query-points", type=int, default=4096)
    p.add_argument("--query-sampling", type=str, default="uniform", choices=["uniform", "obs_mix"])
    p.add_argument("--query-sample-near-ratio", type=float, default=0.25)
    p.add_argument("--query-sample-far-ratio", type=float, default=0.25)
    p.add_argument("--query-sample-sigma-ratio", type=float, default=0.05)
    p.add_argument("--prior", type=str, default="rff", choices=["iid", "rff"])
    p.add_argument("--rff-features", type=int, default=256)
    p.add_argument("--rff-lengthscale", type=float, default=0.15)
    p.add_argument("--sigma-min", type=float, default=1e-4) # backward-compatible old args

    p.add_argument("--cond-field", type=int, default=2, help="Legacy single conditioned field.")
    p.add_argument("--n-obs-min", type=int, default=64, help="Legacy single-field minimum sensors.")
    p.add_argument("--n-obs-max", type=int, default=256, help="Legacy single-field maximum sensors.")

    # generalized args
    p.add_argument("--cond-fields", type=int, nargs="+", default=None,
                   help="Conditioned field ids, e.g. --cond-fields 0 2")
    p.add_argument("--n-obs-min-list", type=int, nargs="+", default=None,
                   help="Per-field minimum sensors. Length 1 broadcasts to all cond_fields.")
    p.add_argument("--n-obs-max-list", type=int, nargs="+", default=None,
                   help="Per-field maximum sensors. Length 1 broadcasts to all cond_fields.")

    p.add_argument("--vis-cond-fields", type=int, nargs="+", default=None,
                   help="Visualization conditioned fields. Defaults to cond_fields.")
    p.add_argument("--vis-n-obs-list", type=int, nargs="+", default=None,
                   help="Visualization exact sensors per field. Defaults to n_obs_max_list.")
    
    # ODE solver used at generation time. For 1-RF, Euler is the main benchmark because the method is designed for coarse-step sampling.
    p.add_argument(
        "--ode-solver", type=str, default="euler",
            choices=["euler", "heun"], help="ODE solver for generation. Use Euler for the main 1-RF benchmark; Heun is optional.")
    # Reconstruction benchmark step counts. These are the NFEs to compare after moving to 1-RF.
    p.add_argument(
        "--benchmark-n-steps", type=int, nargs="+", default=[2, 4, 8, 16],
            help="Sampling step counts used for reconstruction benchmarking.")

    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--n-steps-generation", type=int, default=32)

    return p.parse_args()

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def normalize_conditioning_args(args):
    # training
    if args.cond_fields is None:
        args.cond_fields = [args.cond_field]
    if args.n_obs_min_list is None:
        args.n_obs_min_list = [args.n_obs_min]
    if args.n_obs_max_list is None:
        args.n_obs_max_list = [args.n_obs_max]

    # visualization
    if args.vis_cond_fields is None:
        args.vis_cond_fields = list(args.cond_fields)
    if args.vis_n_obs_list is None:
        args.vis_n_obs_list = list(args.n_obs_max_list)

    return args

class IIDGaussianPrior(nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.randn(bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype)


class RFFGaussianPrior(nn.Module):
    """Scalable smooth Gaussian-field approximation via random Fourier features."""

    def __init__(self, coord_dim: int = 3, n_features: int = 256, lengthscale: float = 0.15):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.register_buffer("omega", torch.randn(coord_dim, n_features) / max(lengthscale, 1e-6))
        self.register_buffer("phase", 2 * math.pi * torch.rand(n_features))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords @ self.omega + self.phase
        return math.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        phi = self._features(coords)
        bsz, _, n_feat = phi.shape
        weights = torch.randn(bsz, n_channels, n_feat, device=coords.device, dtype=coords.dtype)
        return torch.einsum("bnf,bcf->bnc", phi, weights)


def collate_snapshots(batch):
    return {
        "coords": torch.stack([b["coords"] for b in batch], dim=0),
        "fields": torch.stack([b["fields"] for b in batch], dim=0),
        "time_index": torch.stack([b["time_index"] for b in batch], dim=0),
        "physical_time": torch.stack([b["physical_time"] for b in batch], dim=0),
    }


def sample_query_subset(
    coords: torch.Tensor,
    fields: torch.Tensor,
    n_query: Optional[int],
    mode: str = "uniform",
    obs_coords: Optional[torch.Tensor] = None,
    obs_mask: Optional[torch.Tensor] = None,
    near_ratio: float = 0.25,
    far_ratio: float = 0.25,
    sigma_ratio: float = 0.05,
):
    if n_query is None or n_query >= coords.shape[1]:
        return coords, fields, None

    bsz, n_pts, coord_dim = coords.shape
    n_query = int(n_query)

    def take_weighted(weights: torch.Tensor, count: int, selected: torch.Tensor) -> torch.Tensor:
        count = min(int(count), int((~selected).sum().item()))
        if count <= 0:
            return torch.empty(0, device=coords.device, dtype=torch.long)

        weights = weights.to(dtype=coords.dtype).clamp_min(0.0)
        weights = weights.masked_fill(selected, 0.0)
        pieces = []

        positive = weights > 0
        if positive.any():
            n_weighted = min(count, int(positive.sum().item()))
            sampled = torch.multinomial(weights, num_samples=n_weighted, replacement=False)
            pieces.append(sampled)
            selected[sampled] = True
            count -= n_weighted

        if count > 0:
            available = (~selected).nonzero(as_tuple=False).squeeze(-1)
            fill = available[torch.randperm(available.numel(), device=coords.device)[:count]]
            pieces.append(fill)
            selected[fill] = True

        return torch.cat(pieces, dim=0) if pieces else torch.empty(0, device=coords.device, dtype=torch.long)

    all_idx = []
    for b in range(bsz):
        if mode == "obs_mix" and obs_coords is not None and obs_mask is not None:
            valid = obs_mask[b].bool()
        else:
            valid = None

        if mode != "obs_mix" or valid is None or not valid.any():
            idx = torch.randperm(n_pts, device=coords.device)[:n_query].sort().values
            all_idx.append(idx)
            continue

        d_min = torch.cdist(coords[b:b + 1], obs_coords[b, valid].unsqueeze(0), p=2.0).squeeze(0).amin(dim=-1)
        bbox_diag = (coords[b].amax(dim=0) - coords[b].amin(dim=0)).norm().clamp_min(1e-6)
        sigma = (sigma_ratio * bbox_diag).clamp_min(1e-6)

        near_count = min(n_query, max(0, int(round(n_query * near_ratio))))
        far_count = min(n_query - near_count, max(0, int(round(n_query * far_ratio))))
        uniform_count = n_query - near_count - far_count

        selected = torch.zeros(n_pts, device=coords.device, dtype=torch.bool)
        near_weights = torch.exp(-(d_min ** 2) / (2 * sigma ** 2 + 1e-12))
        far_weights = d_min.clamp_min(0.0)

        pieces = [
            take_weighted(near_weights, near_count, selected),
            take_weighted(far_weights, far_count, selected),
            take_weighted(torch.ones(n_pts, device=coords.device, dtype=coords.dtype), uniform_count, selected),
        ]
        if int(selected.sum().item()) < n_query:
            pieces.append(
                take_weighted(
                    torch.ones(n_pts, device=coords.device, dtype=coords.dtype),
                    n_query - int(selected.sum().item()),
                    selected,
                )
            )

        idx = torch.cat([p for p in pieces if p.numel() > 0], dim=0).sort().values
        all_idx.append(idx)

    idx = torch.stack(all_idx, dim=0)
    coord_idx = idx.unsqueeze(-1).expand(-1, -1, coord_dim)
    field_idx = idx.unsqueeze(-1).expand(-1, -1, fields.shape[-1])
    return torch.gather(coords, dim=1, index=coord_idx), torch.gather(fields, dim=1, index=field_idx), idx

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cond_fields: Sequence[int],
    n_obs_min_list: Sequence[int],
    n_obs_max_list: Sequence[int],
    n_query_points: Optional[int],
    query_sampling: str = "uniform",
    query_sample_near_ratio: float = 0.25,
    query_sample_far_ratio: float = 0.25,
    query_sample_sigma_ratio: float = 0.05,
    epoch: int = 0,
) -> float:
    training = optimizer is not None
    model.train(training)

    total = 0.0
    count = 0

    mode_str = "Train" if training else "Eval"
    pbar = tqdm(loader, desc=f"Epoch {epoch:04d} [{mode_str}]", leave=False)

    for batch in pbar:
        coords_full = batch["coords"].to(device)
        fields_full = batch["fields"].to(device)

        # Build generalized sparse observations.
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords_full,
            fields_full=fields_full,
            cond_fields=cond_fields,
            n_obs_min=n_obs_min_list,
            n_obs_max=n_obs_max_list,
        )

        # for models that must operate on the full regular grid like FNO, 
        # point subsampling will be disabled.
        effective_n_query = None if getattr(model, "requires_full_grid", False) else n_query_points
        sampling_mode = query_sampling if training else "uniform"
        coords_q, fields_q, _ = sample_query_subset(
            coords=coords_full,
            fields=fields_full,
            n_query=effective_n_query,
            mode=sampling_mode,
            obs_coords=obs_coords,
            obs_mask=obs_mask,
            near_ratio=query_sample_near_ratio,
            far_ratio=query_sample_far_ratio,
            sigma_ratio=query_sample_sigma_ratio,
        )

        loss, _ = model.training_loss(
            x1=fields_q,
            coords=coords_q,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            obs_indices=obs_indices,
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        current_loss = float(loss.detach().cpu())
        total += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    return total / max(count, 1)


def find_latest_run_dir(demo_dir: str, save_dir: str, demo_num: int) -> Optional[Path]:
    save_root = Path(demo_dir) / Path(save_dir).parent
    run_prefix = f"{Path(save_dir).name}_DemoN{demo_num}_"
    if not save_root.exists():
        return None

    candidates = [
        path for path in save_root.glob(f"{run_prefix}*")
        if path.is_dir()
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def extract_run_timestamp(run_dir: Path, save_dir: str, demo_num: int) -> str:
    run_prefix = f"{Path(save_dir).name}_DemoN{demo_num}_"
    run_name = run_dir.name
    if run_name.startswith(run_prefix):
        return run_name[len(run_prefix):]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_path(path: Path, suffix: str = "_bk") -> Path:
    candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not candidate.exists():
        return candidate

    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}{suffix}{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def backup_existing_artifact(path: Path) -> None:
    if not path.exists():
        return

    target = backup_path(path)
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)


class TrainingHistoryLogger:
    def __init__(self, run_dir: Path) -> None:
        self.csv_path = run_dir / "loss_history.csv"
        self.json_path = run_dir / "loss_history.json"
        self.plot_path = run_dir / "loss_history.png"
        self.rows = []
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    def log_and_plot(self, epoch: int, train_loss: float, val_loss: Optional[float] = None) -> None:
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
        }
        self.rows.append(row)

        with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                row["epoch"],
                row["train_loss"],
                "" if row["val_loss"] is None else row["val_loss"],
            ])
        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(self.rows, handle, indent=2)

        train_points = [
            (item["epoch"], item["train_loss"])
            for item in self.rows
            if item["train_loss"] is not None and item["train_loss"] > 0.0
        ]
        val_points = [
            (item["epoch"], item["val_loss"])
            for item in self.rows
            if item["val_loss"] is not None and item["val_loss"] > 0.0
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        if train_points:
            ax.plot(
                [item[0] for item in train_points],
                [item[1] for item in train_points],
                label="Train Loss",
                marker="o",
                color="blue",
                markersize=4,
            )
        if val_points:
            ax.plot(
                [item[0] for item in val_points],
                [item[1] for item in val_points],
                label="Validation Loss",
                marker="s",
                color="orange",
                markersize=5,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Conditional Point-Cloud FFM Training Progress")
        if train_points or val_points:
            ax.set_yscale("log")
            ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)


def main():

    args = parse_args()
    script_dir = os.path.dirname(os.path.realpath(__file__))
    demo_dir = os.path.dirname(script_dir) # Go up one level to \demo
    
    # YAML Loading and Backup
    config_path = os.path.join(demo_dir, args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if os.path.exists(config_path):
        print(f"\n[*] Starting:... I found config file at: {config_path}\n")
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        
        # Overwrite default args with YAML values
        if yaml_config is not None:
            for key, value in yaml_config.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                else:
                    print(f"Warning: YAML key '{key}' is not a recognized argument. Ignoring.")
        args = normalize_conditioning_args(args)
                    
        # Backup the YAML file
        backup_dir = os.path.join(demo_dir, "Save_config", "pointcloud_ffm")
        os.makedirs(backup_dir, exist_ok=True)
        backup_filename = f"config_pointcloud_ffm_DemoN{args.Demo_Num}_{timestamp}.yaml"
        shutil.copy(config_path, os.path.join(backup_dir, backup_filename))
        print(f"[*] Config backed up to: {os.path.join(backup_dir, backup_filename)}\n")
    else:
        print(f"\n[Warning: !] Config file not found at {config_path}. Using default parameters.\n")
        args.Demo_Num = 0  # Force Demo_Num to 0 as default
    
    # Setup the Dynamic Directories with Demo_Num
    set_seed(args.seed)

    start_epoch = 1
    best_val = float("inf")
    reload_ckpt = None
    run_timestamp = timestamp
    save_dir = Path(os.path.join(demo_dir, args.save_dir + f"_DemoN{args.Demo_Num}" + f"_{timestamp}"))

    if args.RELOAD:
        latest_run_dir = find_latest_run_dir(demo_dir=demo_dir, save_dir=args.save_dir, demo_num=args.Demo_Num)
        if latest_run_dir is not None and (latest_run_dir / "best.pt").exists():
            save_dir = latest_run_dir
            run_timestamp = extract_run_timestamp(latest_run_dir, args.save_dir, args.Demo_Num)
            reload_ckpt = torch.load(latest_run_dir / "best.pt", map_location="cpu")
            start_epoch = int(reload_ckpt.get("epoch", 0)) + 1
            best_val = float(reload_ckpt.get("val_loss", float("inf")))

            backup_existing_artifact(latest_run_dir / "best.pt")
            print(f"[*] RELOAD=True, resuming from: {latest_run_dir / 'best.pt'}")
            print(f"[*] Resume will start from epoch {start_epoch}\n")
        else:
            print("[*] RELOAD=True, but no matching best.pt was found. Training will start from scratch.\n")

    save_dir.mkdir(parents=True, exist_ok=True)

    # Save the final parsed args to a JSON in the model folder just to be safe
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    if os.path.exists(config_path):
        shutil.copy(config_path, save_dir / "run_config.yaml")

    # Keep all run artifacts under the model directory, matching the unified
    # baseline trainers. The old Save_loss_csv/ and Save_reconstruction_files/
    # roots are no longer used by this trainer.
    recon_dir = save_dir / "Evaluation"

    if args.RELOAD and reload_ckpt is not None:
        for loss_artifact in ("loss_history.csv", "loss_history.json", "loss_history.png"):
            backup_existing_artifact(save_dir / loss_artifact)
        backup_existing_artifact(recon_dir)

    # Initialize helpers
    logger = TrainingHistoryLogger(save_dir)
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[*] Model checkpoints will save to: {save_dir}")
    print(f"[*] Logging losses to: {save_dir}")
    print(f"[*] Saving recon plots to: {recon_dir}\n")

    device_ids = args.device_ids
    device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    train_set = TurbulentCombustionH5Dataset(
        args.data,
        split="train",
        train_ratio=args.train_ratio,
        seed=args.seed,
        time_stride=args.time_stride,
        stats_path=str(save_dir / "dataset_stats.pt"),
    )
    val_set = TurbulentCombustionH5Dataset(
        args.data,
        split="val",
        train_ratio=args.train_ratio,
        seed=args.seed,
        time_stride=args.time_stride,
        stats_path=str(save_dir / "dataset_stats.pt"),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
    )

    prior = IIDGaussianPrior() if args.prior == "iid" else RFFGaussianPrior(
        coord_dim=3, n_features=args.rff_features, lengthscale=args.rff_lengthscale
    )

    if args.backbone == "mlp_rbf":
        backbone = ConditionalPointMLPRBF(
            n_fields=train_set.num_fields,
            coord_dim=3,
            hidden_dim=args.hidden_dim,
            cond_dim=args.cond_dim,
            field_embed_dim=args.field_embed_dim,
            rbf_sigma=args.rbf_sigma,
            use_fourier_pe=args.USE_FOURIER_PE,
            fourier_pe_num_bands=args.fourier_pe_num_bands,
            fourier_pe_max_freq=args.fourier_pe_max_freq,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "perceiver":
        backbone = ConditionalPointPerceiver(
            n_fields=train_set.num_fields,
            coord_dim=3,
            latent_dim=args.latent_dim,
            num_latents=args.num_latents,
            num_heads=args.num_heads,
            num_latent_blocks=args.num_latent_blocks,
            field_embed_dim=args.field_embed_dim,
            ff_mult=args.ff_mult,
            attn_dropout=args.attn_dropout,
            mlp_dropout=args.mlp_dropout,
            decode_chunk_size=args.decode_chunk_size,
            share_query_proj=args.share_query_proj,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "GL_rbf":
        backbone = ConditionalPointHybridLocalGlobalRBF(
            n_fields=train_set.num_fields,
            coord_dim=3,
            hidden_dim=args.hidden_dim,
            cond_dim=args.cond_dim,
            field_embed_dim=args.field_embed_dim,
            latent_dim=args.latent_dim,
            num_latents=args.num_latents,
            num_heads=args.num_heads,
            num_latent_blocks=args.num_latent_blocks,
            ff_mult=args.ff_mult,
            attn_dropout=args.attn_dropout,
            mlp_dropout=args.mlp_dropout,
            rbf_sigma=args.rbf_sigma,
            summary_type=args.summary_type,

            gather_mode=args.gather_mode,
            gather_topk=args.gather_topk,
            gather_query_chunk_size=args.gather_query_chunk_size,
            learnable_rbf_sigma=args.learnable_rbf_sigma,
            neighbor_backend=args.neighbor_backend,

            sensor_local_topk=args.sensor_local_topk,
            sensor_local_dropout=args.sensor_local_dropout,
            use_fourier_pe=args.USE_FOURIER_PE,
            fourier_pe_num_bands=args.fourier_pe_num_bands,
            fourier_pe_max_freq=args.fourier_pe_max_freq,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "fno":
        # FNO requires an explicit regular-grid interpretation of the dataset.
        try:
            grid_info = validate_regular_grid_compatibility(train_set, args.Num_x, args.Num_y)
            validate_regular_grid_compatibility(val_set, args.Num_x, args.Num_y)
        except ValueError as e:
            print(f"\n[Warning: !] {e}")
            print("[Warning: !] FNO baseline cannot start because the provided Num_x / Num_y "
                  "are missing or incompatible with the dataset.\n")
            raise SystemExit(1)

        print(
            "[*] FNO grid detected: "
            f"{grid_info['unique_x']} unique x values x {grid_info['unique_y']} unique y values "
            f"= {grid_info['num_points']} points."
        )
        print(
            "[*] FNO grid spacing in normalized coords: "
            f"x min/med/max={grid_info['x_spacing_min']:.3e}/"
            f"{grid_info['x_spacing_median']:.3e}/{grid_info['x_spacing_max']:.3e}, "
            f"y min/med/max={grid_info['y_spacing_min']:.3e}/"
            f"{grid_info['y_spacing_median']:.3e}/{grid_info['y_spacing_max']:.3e}."
        )
        if grid_info["requires_permutation"]:
            print(
                "[*] FNO grid order: dataset is not row-major; the FNO backbone will "
                "internally permute point order -> row-major grid and invert the "
                "permutation on output."
            )
            print(
                "[*] FNO grid permutation sample: first row-major cells come from "
                f"original indices {grid_info['first_row_original_indices']}; "
                "first original points map to grid cells "
                f"{grid_info['first_original_to_grid_indices']}."
            )
        if not grid_info["spacing_regular"]:
            print(
                "[*] FNO grid note: physical x/y spacing is nonuniform. FNO will run "
                "on the topological index grid; point-cloud baselines still use "
                "the physical coordinates directly."
            )

        backbone = FNO(
            n_fields=train_set.num_fields,
            Num_x=args.Num_x,
            Num_y=args.Num_y,
            n_modes_x=args.fno_modes_x,
            n_modes_y=args.fno_modes_y,
            hidden_channels=args.fno_hidden_channels,
            n_layers=args.fno_n_layers,
            condition_blur=args.condition_blur,
            condition_blur_kernel=args.condition_blur_kernel,
            condition_blur_sigma=args.condition_blur_sigma,
        )
        model = FNOFFM(backbone, prior, sigma_min=args.sigma_min).to(device)

        print(f"[*] Using grid-based FNO baseline with Num_x={args.Num_x}, Num_y={args.Num_y}")
        print("[*] Note: n_query_points is ignored for FNO because it requires the full grid.\n")
    else:
        raise ValueError(
            f'Error!!! Your backbone is not supported: {args.backbone}.'
            'Please select in ["mlp_rbf", "perceiver", "fno"]'
            )
    print(f'\nSelected Backbone: {args.backbone}\n')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if reload_ckpt is not None:
        model.load_state_dict(reload_ckpt["model"])
        if "optimizer" in reload_ckpt:
            optimizer.load_state_dict(reload_ckpt["optimizer"])
        print("[*] Reloaded model state from best.pt")

    for epoch in range(start_epoch, args.epochs + 1):
        tr_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cond_fields=args.cond_fields,
            n_obs_min_list=args.n_obs_min_list,
            n_obs_max_list=args.n_obs_max_list,
            n_query_points=args.n_query_points,
            query_sampling=args.query_sampling,
            query_sample_near_ratio=args.query_sample_near_ratio,
            query_sample_far_ratio=args.query_sample_far_ratio,
            query_sample_sigma_ratio=args.query_sample_sigma_ratio,
            epoch=epoch,
        )
        scheduler.step()

        print(f"[train] epoch={epoch:04d} loss={tr_loss:.6e}")
        val_loss = None
        if epoch % args.eval_every == 0 or epoch == 1:
            with torch.no_grad():
                val_loss = run_epoch(
                    model=model,
                    loader=val_loader,
                    optimizer=None,
                    device=device,
                    cond_fields=args.cond_fields,
                    n_obs_min_list=args.n_obs_min_list,
                    n_obs_max_list=args.n_obs_max_list,
                    n_query_points=args.n_query_points,
                    query_sampling=args.query_sampling,
                    query_sample_near_ratio=args.query_sample_near_ratio,
                    query_sample_far_ratio=args.query_sample_far_ratio,
                    query_sample_sigma_ratio=args.query_sample_sigma_ratio,
                    epoch=epoch,
                )
            print(f"[valid] epoch={epoch:04d} loss={val_loss:.6e}")

            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "mean": train_set.mean,
                "std": train_set.std,
                "field_names": train_set.field_names,
                "method": "1_rectified_flow",
                "backbone": args.backbone,
                "summary_type": args.summary_type,
                "ode_solver": args.ode_solver,
                "Num_x": args.Num_x,
                "Num_y": args.Num_y,
            }
            torch.save(ckpt, save_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, save_dir / "best.pt")
                print('Saving the best model...')
        
        if epoch % args.save_every == 0:
            # Benchmark the same validation snapshot at several NFEs.

            recon_dir_epoch = recon_dir / f"epoch_{epoch:04d}"
            recon_dir_epoch.mkdir(parents=True, exist_ok=True)
            
            step_list = args.benchmark_n_steps if args.benchmark_n_steps else [args.n_steps_generation]
            for nfe in step_list:
                # recon_metrics = visualize_reconstruction(
                #     model=model,
                #     dataset=val_set,
                #     epoch=epoch,
                #     device=device,
                #     save_dir=recon_dir_epoch,

                #     cond_fields=args.vis_cond_fields,
                #     n_obs=args.vis_n_obs_list,

                #     n_steps=nfe,
                #     ode_solver=args.ode_solver,
                #     snapshot_index=0,
                #     file_tag=f"{args.ode_solver}_nfe{nfe}",
                # )

                recon_metrics = visualize_reconstruction(
                    model=model,
                    dataset=val_set,
                    epoch=epoch,
                    device=device,
                    save_dir=str(recon_dir_epoch),

                    cond_fields=args.vis_cond_fields,
                    n_obs=args.vis_n_obs_list,
                    n_steps=nfe,
                    ode_solver=args.ode_solver,
                    snapshot_index=0,
                    file_tag=f"{args.ode_solver}_nfe{nfe}",
                    save_metrics_json = True,
                )

                metric_str = ", ".join([f"{k}:{v:.4e}" for k, v in recon_metrics.items()])
                print(f"[recon] epoch={epoch:04d} solver={args.ode_solver} n_steps={nfe} | {metric_str}")

        logger.log_and_plot(epoch=epoch, train_loss=tr_loss, val_loss=val_loss)

    print("Training complete.")
    print(f"Best validation loss: {best_val:.6e}")


if __name__ == "__main__":
    main()

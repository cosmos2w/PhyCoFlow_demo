
'''
With this patch:

- training can use any configured field combination like [0], [2], [0, 2], [0, 2, 4]

- each conditioned field can have its own n_obs_min / n_obs_max

- visualization can use its own cond_fields and exact n_obs list, independent of training

- Model backbone can be ConditionalPointMLPRBF, ConditionalPointPerceiver
'''

import argparse
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime

from helpers import (
    MetricsLogger,
    TurbulentCombustionH5Dataset,
    validate_regular_grid_compatibility,
    create_recon_dir,
    visualize_reconstruction,
    build_sparse_condition,
)
from Model import (
    ConditionalPointFFM, 
    ConditionalPointMLPRBF, 
    ConditionalPointPerceiver,
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
    
    # ------------------------------
    # Backbone selection
    # ------------------------------
    p.add_argument(
        "--backbone", type=str, default="mlp_rbf", choices = ["mlp_rbf", "perceiver", "fno"], 
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
    # These are hyperparameters for mlp_rbf backbone
    # ------------------------------
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--field-embed-dim", type=int, default=64)
    p.add_argument("--rbf-sigma", type=float, default=0.05)

    # ------------------------------
    # These are hyperparameters for Perceiver backbone
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

    # ------------------------------
    # These are hyperparameters for training process
    # ------------------------------
    p.add_argument("--n-query-points", type=int, default=4096)
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


def random_query_subset(coords: torch.Tensor, fields: torch.Tensor, n_query: Optional[int]):
    if n_query is None or n_query >= coords.shape[1]:
        return coords, fields, None
    idx = torch.randperm(coords.shape[1], device=coords.device)[:n_query].sort().values
    return coords[:, idx], fields[:, idx], idx

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    cond_fields: Sequence[int],
    n_obs_min_list: Sequence[int],
    n_obs_max_list: Sequence[int],
    n_query_points: Optional[int],
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
        coords_q, fields_q, _ = random_query_subset(coords_full, fields_full, effective_n_query)

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
    
    # Update Model Save Dir
    save_dir = Path(os.path.join(demo_dir, args.save_dir + f"_DemoN{args.Demo_Num}" + f"_{timestamp}"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the final parsed args to a JSON in the model folder just to be safe
    with open(save_dir / "args.json", "w") as f:
        import json
        json.dump(vars(args), f, indent=2)
    
    # Setup CSV and Recon Dirs
    csv_base_dir = os.path.join(demo_dir, f"Save_loss_csv")
    recon_base_dir = os.path.join(demo_dir, f"Save_reconstruction_files")
    
    # Initialize helpers
    logger = MetricsLogger(base_dir=csv_base_dir, Demo_Num=args.Demo_Num, timestamp=timestamp)
    recon_dir = create_recon_dir(base_dir=recon_base_dir, Demo_Num=args.Demo_Num, timestamp=timestamp)
    
    print(f"[*] Model checkpoints will save to: {save_dir}")
    print(f"[*] Logging losses to: {logger.save_dir}")
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
    elif args.backbone == "fno":
        # FNO requires an explicit regular-grid interpretation of the dataset.
        try:
            validate_regular_grid_compatibility(train_set, args.Num_x, args.Num_y)
            validate_regular_grid_compatibility(val_set, args.Num_x, args.Num_y)
        except ValueError as e:
            print(f"\n[Warning: !] {e}")
            print("[Warning: !] FNO baseline cannot start because the provided Num_x / Num_y "
                  "are missing or incompatible with the dataset.\n")
            raise SystemExit(1)

        backbone = FNO(
            n_fields=train_set.num_fields,
            Num_x=args.Num_x,
            Num_y=args.Num_y,
            n_modes_x=args.fno_modes_x,
            n_modes_y=args.fno_modes_y,
            hidden_channels=args.fno_hidden_channels,
            n_layers=args.fno_n_layers,
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

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            cond_fields=args.cond_fields,
            n_obs_min_list=args.n_obs_min_list,
            n_obs_max_list=args.n_obs_max_list,
            n_query_points=args.n_query_points,
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

            recon_dir_epoch = os.path.join(recon_dir, f"Epoch_{epoch}")
            os.makedirs(recon_dir_epoch, exist_ok=True)
            
            step_list = args.benchmark_n_steps if args.benchmark_n_steps else [args.n_steps_generation]
            for nfe in step_list:
                recon_metrics = visualize_reconstruction(
                    model=model,
                    dataset=val_set,
                    epoch=epoch,
                    device=device,
                    save_dir=recon_dir_epoch,

                    cond_fields=args.vis_cond_fields,
                    n_obs=args.vis_n_obs_list,

                    n_steps=nfe,
                    ode_solver=args.ode_solver,
                    snapshot_index=0,
                    file_tag=f"{args.ode_solver}_nfe{nfe}",
                )

                metric_str = ", ".join([f"{k}:{v:.4e}" for k, v in recon_metrics.items()])
                print(f"[recon] epoch={epoch:04d} solver={args.ode_solver} n_steps={nfe} | {metric_str}")

        logger.log_and_plot(epoch=epoch, train_loss=tr_loss, val_loss=val_loss)

    print("Training complete.")
    print(f"Best validation loss: {best_val:.6e}")


if __name__ == "__main__":
    main()

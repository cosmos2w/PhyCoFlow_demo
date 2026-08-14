
'''
With this patch:

- training can use any configured field combination like [0], [2], [0, 2], [0, 2, 4]

- each conditioned field can have its own n_obs_min / n_obs_max

- visualization can use its own cond_fields and exact n_obs list, independent of training
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

from helpers_V1 import MetricsLogger, TurbulentCombustionH5Dataset, create_recon_dir, visualize_reconstruction, build_sparse_condition

def parse_args():

    # Demo_Num = 0
    p = argparse.ArgumentParser("Train a starter conditional point-cloud FFM on turbulent combustion HDF5 data.")

    p.add_argument("--config", type=str, 
                   default="Save_config/config_pointcloud_ffm.yaml", help="Path to YAML config")
    p.add_argument("--Demo-Num", type=int, 
                   default=0, help="Demo ID tag for saving directories")
    p.add_argument("--data", type=str, 
                   default="Dataset/Merged_CH4COTU1P.h5")
    p.add_argument("--save-dir", type=str, 
                   default=f"Save_TrainedModel/ffm_tc_pointcloud")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--field-embed-dim", type=int, default=64)
    p.add_argument("--sigma-min", type=float, default=1e-4)
    p.add_argument("--rbf-sigma", type=float, default=0.05)

    p.add_argument("--n-query-points", type=int, default=4096)
    p.add_argument("--prior", type=str, default="rff", choices=["iid", "rff"])
    p.add_argument("--rff-features", type=int, default=256)
    p.add_argument("--rff-lengthscale", type=float, default=0.15)

    # backward-compatible old args
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

    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--n-steps-generation", type=int, default=100)
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

def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3, act=nn.GELU) -> nn.Sequential:
    layers = []
    dim = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(dim, hidden_dim), act()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


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


class ConditionalPointFFM(nn.Module):
    """
    Instead of one global cond_field_idx, each observation now carries its own field id
    by giving each sensor a learnable field_embed_dim, allowing the model to know 
    what physical property the sensor is measuring, not just where it is.
    """
    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        field_embed_dim: int = 32,
        rbf_sigma: float = 0.05,
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma

        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        self.point_encoder = make_mlp(coord_dim + n_fields + 1, hidden_dim, hidden_dim, depth=3)
        self.obs_encoder = make_mlp(coord_dim + 1 + field_embed_dim, cond_dim, cond_dim, depth=3)
        self.global_encoder = make_mlp(hidden_dim, hidden_dim, hidden_dim, depth=2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_fields),
        )

    # For any given query point, the model calculates the physical squared distance to every available sensor. 
    # Using a Radial Basis Function (RBF) kernel, it applies an attention weight: 
    # sensors that are physically closer exert massive influence on the query point, while distant sensors are ignored.
    def aggregate_sparse_obs(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        safe_field_ids = obs_field_ids.clamp_min(0)
        obs_field_feat = self.field_embed(safe_field_ids)                 # [B, M, E]
        obs_field_feat = obs_field_feat * obs_mask.unsqueeze(-1)          # zero padded rows

        obs_in = torch.cat([obs_coords, obs_values, obs_field_feat], dim=-1)
        obs_feat = self.obs_encoder(obs_in)
        obs_feat = obs_feat * obs_mask.unsqueeze(-1)

        d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2
        large = torch.full_like(d2, 1e6)
        d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

        weights = torch.softmax(-d2 / (2 * self.rbf_sigma ** 2 + 1e-12), dim=-1)
        return torch.einsum("bnm,bmd->bnd", weights, obs_feat)

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_pts, _ = x_t.shape
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)

        point_feat = self.point_encoder(torch.cat([coords, x_t, t_feat], dim=-1))
        local_cond = self.aggregate_sparse_obs(coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        global_feat = self.global_encoder(point_feat.mean(dim=1)).unsqueeze(1).expand(bsz, n_pts, -1)

        return self.head(torch.cat([point_feat, global_feat, local_cond], dim=-1))


class PointCloudFFM(nn.Module):
    def __init__(self, model: nn.Module, prior: nn.Module, sigma_min: float = 1e-4):
        super().__init__()
        self.model = model
        self.prior = prior
        self.sigma_min = sigma_min

    def simulate(self, t: torch.Tensor, x1: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        noise = self.prior(coords, x1.shape[-1])
        sigma = 1.0 - (1.0 - self.sigma_min) * t.view(-1, 1, 1)
        mu = t.view(-1, 1, 1) * x1
        return mu + sigma * noise

    def target_vector_field(self, t: torch.Tensor, x1: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        c = 1.0 - (1.0 - self.sigma_min) * t.view(-1, 1, 1)
        return (x1 - (1.0 - self.sigma_min) * x_t) / c

    def training_loss(
        self,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)
        
        x_t = self.simulate(t, x1, coords)
        target = self.target_vector_field(t, x1, x_t)
        pred = self.model(t, x_t, coords, obs_coords, obs_values, obs_mask, obs_field_ids)

        loss = F.mse_loss(pred, target)
        return loss, {"loss": float(loss.detach().cpu())}

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        n_steps: int = 100,
        clamp_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = coords.shape[0]
        x = self.prior(coords, self.model.n_fields)
        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=coords.device, dtype=coords.dtype)

        for i in range(n_steps):
            t0 = ts[i].expand(bsz)
            t1 = ts[i + 1].expand(bsz)

            v0 = self.model(t0, x, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
            x_euler = x + dt * v0
            v1 = self.model(t1, x_euler, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
            x = x + 0.5 * dt * (v0 + v1)

            if clamp_indices is not None:
                for b in range(bsz):
                    valid = obs_mask[b].bool()
                    idx = clamp_indices[b, valid].long()
                    fld = obs_field_ids[b, valid].long()
                    val = obs_values[b, valid, 0]
                    x[b, idx, fld] = val

        return x


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
    model: PointCloudFFM,
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

        obs_coords, obs_values, obs_mask, _, obs_field_ids = build_sparse_condition(
            coords_full=coords_full,
            fields_full=fields_full,
            cond_fields=cond_fields,
            n_obs_min=n_obs_min_list,
            n_obs_max=n_obs_max_list,
        )

        coords_q, fields_q, _ = random_query_subset(coords_full, fields_full, n_query_points)

        loss, _ = model.training_loss(
            x1=fields_q,
            coords=coords_q,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
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
        print(f"[*] Found config file at: {config_path}\n")
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

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
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
    backbone = ConditionalPointFFM(
        n_fields=train_set.num_fields,
        coord_dim=3,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        field_embed_dim=args.field_embed_dim,
        rbf_sigma=args.rbf_sigma,
    )
    model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
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
            }
            torch.save(ckpt, save_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                torch.save(ckpt, save_dir / "best.pt")
                print('Saving the best model...')
            
        if epoch % args.save_every == 0:

            # torch.save({"model": model.state_dict(), "epoch": epoch}, save_dir / f"CheckPoint_DemoN{args.Demo_Num}.pt")
            visualize_reconstruction(
                model=model,
                dataset=val_set,
                epoch=epoch,
                device=device,
                save_dir=recon_dir,
                cond_fields=args.vis_cond_fields,
                n_obs=args.vis_n_obs_list,
                n_steps=args.n_steps_generation,       # Euler steps for generation
                snapshot_index=0,   # Keeps the same fluid snapshot (from validation set) across epochs, pick from 0 to 999
            )
        
        logger.log_and_plot(epoch=epoch, train_loss=tr_loss, val_loss=val_loss)

    print("Training complete.")
    print(f"Best validation loss: {best_val:.6e}")


if __name__ == "__main__":
    main()

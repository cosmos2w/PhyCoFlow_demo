import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_pointcloud_ffm import (
    TurbulentCombustionH5Dataset,
    ConditionalPointFFM,
    PointCloudFFM,
    IIDGaussianPrior,
    RFFGaussianPrior,
)

FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")

def parse_args():
    p = argparse.ArgumentParser("Demo sparse-conditioned reconstruction with trained point-cloud FFM")
    p.add_argument("--data", type=str, default="Dataset/Merged_CH4COTU1P.h5")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="Save_reconstruction_files/ffm_tc_pointcloud/demo")
    p.add_argument("--snapshot-index", type=int, default=0, help="Index within validation split")

    p.add_argument("--cond-field", type=int, default=2)
    p.add_argument("--n-obs", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--prior", type=str, default="rff", choices=["iid", "rff"])
    p.add_argument("--rff-features", type=int, default=256)
    p.add_argument("--rff-lengthscale", type=float, default=0.15)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--field-embed-dim", type=int, default=32)
    p.add_argument("--rbf-sigma", type=float, default=0.05)
    p.add_argument("--sigma-min", type=float, default=1e-4)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot-grid-nx", type=int, default=403)
    p.add_argument("--plot-grid-ny", type=int, default=100)
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_set = TurbulentCombustionH5Dataset(
        args.data,
        split="val",
        train_ratio=args.train_ratio,
        seed=args.seed,
        stats_path=str(Path(args.ckpt).parent / "dataset_stats.pt"),
    )
    sample = val_set[args.snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)

    prior = IIDGaussianPrior() if args.prior == "iid" else RFFGaussianPrior(
        coord_dim=3, n_features=args.rff_features, lengthscale=args.rff_lengthscale
    )
    backbone = ConditionalPointFFM(
        n_fields=truth.shape[-1],
        coord_dim=3,
        hidden_dim=args.hidden_dim,
        cond_dim=args.cond_dim,
        field_embed_dim=args.field_embed_dim,
        rbf_sigma=args.rbf_sigma,
    )
    model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()

    n_pts = coords.shape[1]
    idx = torch.randperm(n_pts, device=device)[: args.n_obs].sort().values
    fld = args.cond_field
    obs_coords = coords[:, idx]
    obs_values = truth[:, idx, fld : fld + 1]
    obs_mask = torch.ones(1, args.n_obs, device=device, dtype=coords.dtype)
    obs_indices = idx.unsqueeze(0)
    cond_field_idx = torch.tensor([fld], device=device, dtype=torch.long)

    recon = model.sample(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        cond_field_idx=cond_field_idx,
        n_steps=args.n_steps,
        clamp_indices=obs_indices,
    )

    mean = val_set.mean.to(device)
    std = val_set.std.to(device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = truth * std.view(1, 1, -1) + mean.view(1, 1, -1)

    np.savez(
        out_dir / "reconstruction_demo.npz",
        coords=coords[0].cpu().numpy(),
        truth=truth_phys[0].cpu().numpy(),
        recon=recon_phys[0].cpu().numpy(),
        obs_idx=idx.cpu().numpy(),
        obs_field=np.array([fld], dtype=np.int64),
    )

    nx, ny = args.plot_grid_nx, args.plot_grid_ny
    assert nx * ny == n_pts, f"plot-grid-nx * plot-grid-ny must equal N={n_pts}"

    for c, name in enumerate(FIELD_NAMES):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        im0 = axs[0].imshow(truth_phys[0, :, c].reshape(nx, ny).T, origin="lower", aspect="auto")
        axs[0].set_title(f"Truth: {name}")
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

        im1 = axs[1].imshow(recon_phys[0, :, c].reshape(nx, ny).T, origin="lower", aspect="auto")
        axs[1].set_title(f"Recon: {name}")
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        if c == fld:
            iy = (idx.cpu().numpy() % ny)
            ix = (idx.cpu().numpy() // ny)
            axs[1].scatter(ix, iy, s=5, c="white")

        fig.savefig(out_dir / f"field_{c}_{name}.png", dpi=200)
        plt.close(fig)

    print(f"Saved demo outputs to {out_dir}")


if __name__ == "__main__":
    main()

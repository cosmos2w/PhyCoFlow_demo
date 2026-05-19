"""
Sparse-observation consistency usage:

- default_hard is the default and preserves current pointwise hard replacement behavior.
- endpoint applies rectified-flow clean-endpoint pointwise observation masking.
- endpoint_smooth applies rectified-flow clean-endpoint Gaussian/RBF smooth observation masking.
- All added SenConsis outputs are saved under SenConsis/, 
                 activate this using --obs-consistency-mode & visualize by --save-obs-consistency-plots.
- SenConsis metrics are relative L2 sensor-consistency errors.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import torch
import yaml
import pickle
import numpy as np

import matplotlib.pyplot as plt
import torch.nn.functional as F

from helpers import (
    TurbulentCombustionH5Dataset,
    save_obs_consistency_comparison,
    validate_regular_grid_compatibility,
    visualize_reconstruction,
)

from Model import (
    ConditionalPointMLPRBF,
    ConditionalPointPerceiver,
    ConditionalPointHybridLocalGlobalRBF,
    PointCloudFFM,
)
try:
    from Model import FNO, FNOFFM
except ImportError:
    FNO = None
    FNOFFM = None

def parse_args():
    p = argparse.ArgumentParser(
        "Standalone evaluator for trained FFM models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--Demo-Num", dest="Demo_Num", type=int, required=True, 
                   help="Demo ID to recover.")
    p.add_argument("--demo-root", type=str, default=".", 
                   help="Project/demo root directory.")
    p.add_argument("--split", type=str, default="test", 
                   choices=["train", "val", "test"])
    p.add_argument("--snapshot-index", type=int, default=0, 
                   help="Index within the selected split.")
    
    p.add_argument("--vis-cond-fields", type=int, nargs="+", default=None,
                   help="Override visualization cond_fields. Defaults to YAML vis_cond_fields or cond_fields.")
    p.add_argument("--vis-n-obs-list", type=int, nargs="+", default=None,
                   help="Override visualization n_obs list. Defaults to YAML vis_n_obs_list or n_obs_max_list.")
    
    p.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"],
                   help="Which checkpoint to load from the recovered run directory.")
    p.add_argument("--n-steps-generation", type=int, default = 4,
                   help="Override generation steps. Defaults to YAML n_steps_generation if present.")
    p.add_argument(
        "--ode-solver",
        choices=["euler", "heun"],
        default=None,
        help="ODE solver for generation. Defaults to YAML ode_solver, then euler.",
    )
    p.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    
    # Added extra metrics for evaluation
    # Run like: python src/evaluate_ffm.py --Demo-Num 0 --split test --snapshot-index 0  --extra-metrics ssim grad spectrum --save-analysis-npz
    p.add_argument("--extra-metrics", type=str, nargs="*", default=[], choices=["ssim", "grad", "spectrum"], 
                   help="Optional extra metrics to compute on structured 2D grids.",)
    # SSIM: higher is better; SSIM = 1.0 → perfect structural match
    # grad_rel_l2: smaller is better, below ~0.3 are very good, above ~0.7 means the local derivative is not being captured well
    p.add_argument("--save-analysis-npz", action="store_true",
                   help="If set, save per-field intermediate arrays (grids, gradients, spectra) to .npz files.",
    )
    p.add_argument(
        "--obs-consistency-mode",
        choices=["none", "default_hard", "endpoint", "endpoint_smooth"],
        default="default_hard",
        help="Sparse-observation consistency mode used during sampling.",
    )
    p.add_argument("--obs-consistency-strength", type=float, default=1.0)
    p.add_argument("--obs-consistency-sigma", type=float, default=0.05)
    p.add_argument("--obs-consistency-schedule-power", type=float, default=2.0)
    p.add_argument(
        "--no-obs-consistency-final-clamp",
        action="store_true",
        help="Disable the final exact sensor clamp for observation-consistency modes.",
    )
    p.add_argument(
        "--save-obs-consistency-plots",
        action="store_true",
        help="Save SenConsis metrics and sensor-consistency figures.",
    )
    p.add_argument(
        "--obs-consistency-compare-modes",
        nargs="+",
        default=None,
        choices=["none", "default_hard", "endpoint", "endpoint_smooth"],
        help="Evaluate multiple sparse-observation consistency modes using the same sensor set.",
    )

    return p.parse_args()

class IIDGaussianPrior(torch.nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.randn(bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype)


class RFFGaussianPrior(torch.nn.Module):
    def __init__(self, coord_dim: int = 3, n_features: int = 256, lengthscale: float = 0.15):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.register_buffer("omega", torch.randn(coord_dim, n_features) / max(lengthscale, 1e-6))
        self.register_buffer("phase", 2 * np.pi * torch.rand(n_features))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords @ self.omega + self.phase
        return np.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        phi = self._features(coords)
        bsz, _, n_feat = phi.shape
        weights = torch.randn(bsz, n_channels, n_feat, device=coords.device, dtype=coords.dtype)
        return torch.einsum("bnf,bcf->bnc", phi, weights)

def _extract_timestamp(path: Path) -> Optional[str]:
    m = re.search(r"DemoN(\d+)_(\d{8}_\d{6})", path.name)
    if m is None:
        m = re.search(r"demo_N(\d+)_(\d{8}_\d{6})", path.name)
    return m.group(2) if m else None


def _find_latest_yaml(cfg_dir: Path, demo_num: int) -> Path:
    pattern = f"config_pointcloud_ffm_DemoN{demo_num}_*.yaml"
    candidates = sorted(cfg_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No config backup found for Demo_Num={demo_num} in {cfg_dir}"
        )

    def _sort_key(p: Path):
        ts = _extract_timestamp(p)
        return ts if ts is not None else p.stat().st_mtime

    candidates = sorted(candidates, key=_sort_key)
    return candidates[-1]


def _normalize_eval_config(cfg: dict) -> dict:
    cfg = dict(cfg)

    # Backward-compatible defaults
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

    if cfg.get("backbone") is None:
        cfg["backbone"] = "mlp_rbf"

    return cfg


def _build_prior(cfg: dict):
    if cfg.get("prior", "rff") == "iid":
        return IIDGaussianPrior()
    return RFFGaussianPrior(
        coord_dim=3,
        n_features=cfg.get("rff_features", 256),
        lengthscale=cfg.get("rff_lengthscale", 0.15),
    )


def _build_model(cfg: dict, dataset) -> torch.nn.Module:
    prior = _build_prior(cfg)
    backbone_name = cfg.get("backbone", "mlp_rbf")

    if backbone_name == "perceiver":
        backbone = ConditionalPointPerceiver(
            n_fields=dataset.num_fields,
            coord_dim=3,
            latent_dim=cfg.get("latent_dim", 256),
            num_latents=cfg.get("num_latents", 128),
            num_heads=cfg.get("num_heads", 8),
            num_latent_blocks=cfg.get("num_latent_blocks", 4),
            field_embed_dim=cfg.get("field_embed_dim", 128),
            ff_mult=cfg.get("ff_mult", 4),
            attn_dropout=cfg.get("attn_dropout", 0.0),
            mlp_dropout=cfg.get("mlp_dropout", 0.0),
            decode_chunk_size=cfg.get("decode_chunk_size", 4096),
            share_query_proj=cfg.get("share_query_proj", False),
        )
        model = PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
        return model

    if backbone_name == "fno":
        if FNO is None or FNOFFM is None:
            raise RuntimeError("YAML says backbone='fno' but FNO/FNOFFM are not available in Model.py")
        Num_x = cfg.get("Num_x", None)
        Num_y = cfg.get("Num_y", None)
        if Num_x is None or Num_y is None:
            raise ValueError("FNO evaluation requires Num_x and Num_y in YAML.")
        backbone = FNO(
            n_fields=dataset.num_fields,
            Num_x=Num_x,
            Num_y=Num_y,
            n_modes_x=cfg.get("fno_modes_x", 32),
            n_modes_y=cfg.get("fno_modes_y", 8),
            hidden_channels=cfg.get("fno_hidden_channels", 64),
            n_layers=cfg.get("fno_n_layers", 4),
            condition_blur=cfg.get("condition_blur", False),
            condition_blur_kernel=cfg.get("condition_blur_kernel", 5),
            condition_blur_sigma=cfg.get("condition_blur_sigma", 1.0),
        )
        model = FNOFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
        return model

    if backbone_name == "GL_rbf":
        backbone = ConditionalPointHybridLocalGlobalRBF(
            n_fields=dataset.num_fields,
            coord_dim=3,
            hidden_dim=cfg.get("hidden_dim", 256),
            cond_dim=cfg.get("cond_dim", 128),
            field_embed_dim=cfg.get("field_embed_dim", 128),
            latent_dim=cfg.get("latent_dim", 256),
            num_latents=cfg.get("num_latents", 128),
            num_heads=cfg.get("num_heads", 8),
            num_latent_blocks=cfg.get("num_latent_blocks", 4),
            ff_mult=cfg.get("ff_mult", 4),
            attn_dropout=cfg.get("attn_dropout", 0),
            mlp_dropout=cfg.get("mlp_dropout", 0),
            rbf_sigma=cfg.get("rbf_sigma", 0.05),
            summary_type=cfg.get("summary_type", "cls"),

            gather_mode=cfg.get("gather_mode", "rbf"),
            gather_topk=cfg.get("gather_topk", 32),
            gather_query_chunk_size=cfg.get("gather_query_chunk_size", None),
            learnable_rbf_sigma=cfg.get("learnable_rbf_sigma", False),
            neighbor_backend=cfg.get("neighbor_backend", "torch"),

            sensor_local_topk=cfg.get("sensor_local_topk", 32),
            sensor_local_dropout=cfg.get("sensor_local_dropout", 0.0),
            use_fourier_pe=cfg.get("USE_FOURIER_PE", False),
            fourier_pe_num_bands=cfg.get("fourier_pe_num_bands", 32),
            fourier_pe_max_freq=cfg.get("fourier_pe_max_freq", 64.0),
        )
        model = PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
        return model

    backbone = ConditionalPointMLPRBF(
        n_fields=dataset.num_fields,
        coord_dim=3,
        hidden_dim=cfg.get("hidden_dim", 256),
        cond_dim=cfg.get("cond_dim", 128),
        field_embed_dim=cfg.get("field_embed_dim", 128),
        rbf_sigma=cfg.get("rbf_sigma", 0.05),
        use_fourier_pe=cfg.get("USE_FOURIER_PE", False),
        fourier_pe_num_bands=cfg.get("fourier_pe_num_bands", 32),
        fourier_pe_max_freq=cfg.get("fourier_pe_max_freq", 64.0),
    )
    model = PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))

    return model


def _infer_structured_grid_from_coords(
    coords_xy: np.ndarray,
    decimals: int = 8,
    num_x: Optional[int] = None,
    num_y: Optional[int] = None,
):
    """
    Recover a structured 2D grid description from point coordinates. 
    Priority:
      1) If YAML/grid metadata provides num_x and num_y and num_x*num_y == N,
         use them directly and build a stable lexicographic ordering.
      2) Otherwise, infer (ny, nx) from unique rounded x/y coordinates.
      3) If neither works, raise ValueError.
    Returns:
        {
            "nx": nx,
            "ny": ny,
            "sort_idx": sort_idx,
            "x_unique": unique_x,
            "y_unique": unique_y,
            "dx": dx,
            "dy": dy,
        }
    """
    x = np.round(coords_xy[:, 0], decimals=decimals)
    y = np.round(coords_xy[:, 1], decimals=decimals)
    n_pts = coords_xy.shape[0]

    # ----------------------------------------------------------
    # Option 1: use explicit grid shape from YAML/config if valid
    # ----------------------------------------------------------
    if num_x is not None and num_y is not None:
        nx = int(num_x)
        ny = int(num_y)

        if nx > 0 and ny > 0 and nx * ny == n_pts:
            # Stable row-major style ordering: sort by y, then x
            sort_idx = np.lexsort((x, y))

            unique_x = np.unique(x)
            unique_y = np.unique(y)

            # Even if the unique counts do not exactly match because of coordinate
            # noise or duplicated values, the explicit YAML shape is the primary source.
            dx = float(np.mean(np.diff(unique_x))) if len(unique_x) > 1 else 1.0
            dy = float(np.mean(np.diff(unique_y))) if len(unique_y) > 1 else 1.0

            return {
                "nx": nx,
                "ny": ny,
                "sort_idx": sort_idx,
                "x_unique": unique_x,
                "y_unique": unique_y,
                "dx": dx,
                "dy": dy,
            }
        elif nx > 0 and ny > 0:
            print(
                f"[Warning: !] Provided Num_x={nx}, Num_y={ny} are inconsistent with "
                f"N={n_pts}; falling back to coordinate inference."
            )

    # ----------------------------------------------------------
    # Option 2: infer from coordinates
    # ----------------------------------------------------------
    unique_x = np.unique(x)
    unique_y = np.unique(y)

    nx = len(unique_x)
    ny = len(unique_y)

    if nx * ny != n_pts:
        raise ValueError(
            f"Coordinates do not form a complete structured 2D grid and no valid "
            f"(Num_x, Num_y) was provided. "
            f"Inferred nx={nx}, ny={ny}, nx*ny={nx*ny}, N={n_pts}"
        )

    sort_idx = np.lexsort((x, y))
    dx = float(np.mean(np.diff(unique_x))) if nx > 1 else 1.0
    dy = float(np.mean(np.diff(unique_y))) if ny > 1 else 1.0

    return {
        "nx": nx,
        "ny": ny,
        "sort_idx": sort_idx,
        "x_unique": unique_x,
        "y_unique": unique_y,
        "dx": dx,
        "dy": dy,
    }


def _reshape_flat_field_to_grid(field_flat: np.ndarray, grid_info: dict) -> np.ndarray:
    vals = field_flat[grid_info["sort_idx"]]
    return vals.reshape(grid_info["ny"], grid_info["nx"])


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device: str = "cpu"):
    ax = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, window_size, window_size)


def _ssim2d(u: np.ndarray, v: np.ndarray, data_range: Optional[float] = None,
            window_size: int = 11, sigma: float = 1.5) -> float:
    """
    Single-scale SSIM for one scalar 2D field.
    """
    device = "cpu"
    x = torch.from_numpy(u).float().unsqueeze(0).unsqueeze(0).to(device)
    y = torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0).to(device)

    if data_range is None:
        data_range = float(max(u.max(), v.max()) - min(u.min(), v.min()))
    data_range = max(float(data_range), 1e-8)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel(window_size=window_size, sigma=sigma, device=device)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad)
    mu_y = F.conv2d(y, kernel, padding=pad)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, kernel, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12
    )
    return float(ssim_map.mean().item())


def _gradient_metrics(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Gradient-based metrics using finite differences with physical spacing.
    """
    uy, ux = np.gradient(u, dy, dx, edge_order=2)
    vy, vx = np.gradient(v, dy, dx, edge_order=2)

    diff_x = vx - ux
    diff_y = vy - uy

    grad_mse = float(np.mean(diff_x ** 2 + diff_y ** 2))
    grad_rel_l2 = float(
        np.sqrt(np.sum(diff_x ** 2 + diff_y ** 2)) /
        (np.sqrt(np.sum(ux ** 2 + uy ** 2)) + 1e-12)
    )

    val_diff = v - u
    h1_num = np.sum(val_diff ** 2) + np.sum(diff_x ** 2 + diff_y ** 2)
    h1_den = np.sum(u ** 2) + np.sum(ux ** 2 + uy ** 2)
    h1_rel = float(np.sqrt(h1_num) / (np.sqrt(h1_den) + 1e-12))

    metrics = {
        "grad_mse": grad_mse,
        "grad_rel_l2": grad_rel_l2,
        "h1_rel": h1_rel,
    }
    payload = {
        "grad_true_x": ux,
        "grad_true_y": uy,
        "grad_pred_x": vx,
        "grad_pred_y": vy,
        "grad_abs_err": np.sqrt(diff_x ** 2 + diff_y ** 2),
    }
    return metrics, payload


def _radial_spectrum(u: np.ndarray, dx: float, dy: float):
    """
    Sharper shell-averaged 2D power spectrum of a zero-mean field.

    Compared with the previous coarse linear binning, this version builds
    spectral shells using the native FFT grid spacing, which preserves much
    more detail in the radial spectrum and produces sharper curves.
    """
    ny, nx = u.shape
    uu = u - np.mean(u)

    # Shifted FFT so low wavenumbers sit near the center in the 2D spectrum.
    fft = np.fft.fftshift(np.fft.fft2(uu))
    psd2 = (np.abs(fft) ** 2) / (nx * ny)

    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    KX, KY = np.meshgrid(kx, ky)
    kmag = np.sqrt(KX ** 2 + KY ** 2)

    # Native shell spacing from the Fourier grid
    dkx = np.min(np.diff(np.unique(kx))) if nx > 1 else 1.0
    dky = np.min(np.diff(np.unique(ky))) if ny > 1 else 1.0
    dk = float(min(abs(dkx), abs(dky))) if (nx > 1 and ny > 1) else 1.0
    dk = max(dk, 1e-12)

    shell_id = np.rint(kmag / dk).astype(np.int64)
    n_shells = int(shell_id.max()) + 1

    shell_sum = np.bincount(shell_id.ravel(), weights=psd2.ravel(), minlength=n_shells)
    shell_count = np.bincount(shell_id.ravel(), minlength=n_shells)
    shell_k_sum = np.bincount(shell_id.ravel(), weights=kmag.ravel(), minlength=n_shells)

    radial = shell_sum / np.maximum(shell_count, 1)
    k = shell_k_sum / np.maximum(shell_count, 1)

    valid = shell_count > 0
    k = k[valid]
    radial = radial[valid]

    # Drop the zero-frequency term from the radial line plot / metrics
    if len(k) > 1:
        k = k[1:]
        radial = radial[1:]

    return {
        "k": k,
        "psd2": psd2,
        "radial_spectrum": radial,
    }


def _band_energy_breakdown(k: np.ndarray, spectrum: np.ndarray):
    """
    Split radial spectrum into low / mid / high wavenumber bands and compute
    band energies by trapezoidal integration.
    """
    if len(k) == 0:
        return {
            "band_names": ["large", "medium", "small"],
            "band_edges": [0.0, 0.0, 0.0, 0.0],
            "band_energy": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "band_fraction": np.array([0.0, 0.0, 0.0], dtype=np.float64),
        }

    kmax = float(np.max(k))
    e1 = kmax / 3.0
    e2 = 2.0 * kmax / 3.0

    masks = [
        (k <= e1),
        ((k > e1) & (k <= e2)),
        (k > e2),
    ]

    energies = []
    for mask in masks:
        if np.count_nonzero(mask) >= 2:
            energies.append(float(np.trapezoid(spectrum[mask], k[mask])))
        elif np.count_nonzero(mask) == 1:
            energies.append(float(spectrum[mask][0]))
        else:
            energies.append(0.0)

    energies = np.asarray(energies, dtype=np.float64)
    total = float(np.sum(energies)) + 1e-12
    fractions = energies / total

    return {
        "band_names": ["large", "medium", "small"],
        "band_edges": [0.0, e1, e2, kmax],
        "band_energy": energies,
        "band_fraction": fractions,
    }


def _spectral_metrics(u: np.ndarray, v: np.ndarray, dx: float, dy: float):
    """
    Spectral comparison using sharper shell-averaged radial spectra plus
    low/mid/high band energies.
    """
    su = _radial_spectrum(u, dx=dx, dy=dy)
    sv = _radial_spectrum(v, dx=dx, dy=dy)

    eps = 1e-12
    k = su["k"]
    ru = su["radial_spectrum"]
    rv = sv["radial_spectrum"]

    # Align lengths conservatively
    n = min(len(ru), len(rv))
    k = k[:n]
    ru = ru[:n]
    rv = rv[:n]

    spectral_lsd = float(np.sqrt(np.mean((np.log(rv + eps) - np.log(ru + eps)) ** 2)))

    gt_band = _band_energy_breakdown(k, ru)
    pr_band = _band_energy_breakdown(k, rv)

    band_ratio = pr_band["band_energy"] / (gt_band["band_energy"] + eps)
    band_rel_err = np.abs(pr_band["band_energy"] - gt_band["band_energy"]) / (gt_band["band_energy"] + eps)

    metrics = {
        "spectral_lsd": spectral_lsd,
        "spectral_ratio_large": float(band_ratio[0]),
        "spectral_ratio_medium": float(band_ratio[1]),
        "spectral_ratio_small": float(band_ratio[2]),
        "spectral_relerr_large": float(band_rel_err[0]),
        "spectral_relerr_medium": float(band_rel_err[1]),
        "spectral_relerr_small": float(band_rel_err[2]),
    }

    payload = {
        "k": k,
        "spectrum_true": ru,
        "spectrum_pred": rv,
        "psd2_true": su["psd2"],
        "psd2_pred": sv["psd2"],
        "band_names": np.array(gt_band["band_names"]),
        "band_edges": np.array(gt_band["band_edges"], dtype=np.float64),
        "band_energy_true": gt_band["band_energy"],
        "band_energy_pred": pr_band["band_energy"],
        "band_fraction_true": gt_band["band_fraction"],
        "band_fraction_pred": pr_band["band_fraction"],
        "band_ratio_pred_over_true": band_ratio,
    }
    return metrics, payload


def _save_spectrum_plot(
    k: np.ndarray,
    s_true: np.ndarray,
    s_pred: np.ndarray,
    band_edges: np.ndarray,
    save_path: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Background bands
    ax.axvspan(band_edges[0], band_edges[1], color="#c9c9f5", alpha=0.25)
    ax.axvspan(band_edges[1], band_edges[2], color="#cfe8cf", alpha=0.25)
    ax.axvspan(band_edges[2], band_edges[3], color="#f3d6d6", alpha=0.25)

    ax.semilogy(k + 1e-12, s_true + 1e-12, color="black", linewidth=2.5, label="Ground Truth")
    ax.semilogy(k + 1e-12, s_pred + 1e-12, color="red", linewidth=2.2, label="Reconstruction")

    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$E(k)$")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.25)

    ymax = max(np.max(s_true), np.max(s_pred)) + 1e-12
    ymin = max(min(np.min(s_true[s_true > 0]) if np.any(s_true > 0) else 1e-12,
                   np.min(s_pred[s_pred > 0]) if np.any(s_pred > 0) else 1e-12), 1e-12)
    ax.set_ylim(bottom=ymin * 0.7, top=ymax * 1.25)

    ax.text(0.5 * (band_edges[0] + band_edges[1]), ymin * 1.2, "large scales",
            color="#303080", fontsize=12, ha="center", va="bottom", fontstyle="italic")
    ax.text(0.5 * (band_edges[1] + band_edges[2]), ymin * 1.2, "medium scales",
            color="#2f6f2f", fontsize=12, ha="center", va="bottom", fontstyle="italic")
    ax.text(0.5 * (band_edges[2] + band_edges[3]), ymin * 1.2, "small scales",
            color="#7a3030", fontsize=12, ha="center", va="bottom", fontstyle="italic")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _save_band_energy_plot(
    band_names: np.ndarray,
    band_ratio: np.ndarray,
    save_path: Path,
    title: str,
):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(band_names))

    ax.bar(x, band_ratio, width=0.6)
    ax.axhline(1.0, linestyle=":", linewidth=1.8, color="black")

    ax.set_xticks(x)
    ax.set_xticklabels([str(v).capitalize() for v in band_names])
    ax.set_ylabel(r"$E_{\mathrm{pred}} / E_{\mathrm{GT}}$")
    ax.set_title(title)
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _mean_full_field_relative_l2(metrics: dict) -> float:
    values = []
    for key, value in metrics.items():
        if key.startswith("obs_"):
            continue
        if isinstance(value, (int, float)) and np.isfinite(value):
            values.append(float(value))
    return float(np.mean(values)) if values else float("nan")

def main():
    args = parse_args()

    demo_root = Path(args.demo_root).resolve()
    cfg_dir = demo_root / "Save_config" / "pointcloud_ffm"

    try:
        yaml_path = _find_latest_yaml(cfg_dir, args.Demo_Num)
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

    save_dir_cfg = Path(cfg.get("save_dir", "Save_TrainedModel/ffm_tc_pointcloud"))
    model_root = demo_root / save_dir_cfg.parent / f"{save_dir_cfg.name}_DemoN{args.Demo_Num}_{train_timestamp}"

    if not model_root.exists():
        print(f"[Warning: !] Matching model directory not found: {model_root}")
        raise SystemExit(1)

    ckpt_path = model_root / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        print(f"[Warning: !] Checkpoint not found: {ckpt_path}")
        raise SystemExit(1)

    device = torch.device(args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    dataset = TurbulentCombustionH5Dataset(
        cfg.get("data", "Dataset/Merged_CH4COTU1P.h5"),
        split=args.split,
        train_ratio=cfg.get("train_ratio", 0.9),
        seed=cfg.get("seed", 42),
        time_stride=cfg.get("time_stride", 1),
        stats_path=str(model_root / "dataset_stats.pt"),
    )

    if cfg.get("backbone") == "fno":
        grid_info = validate_regular_grid_compatibility(dataset, cfg.get("Num_x", None), cfg.get("Num_y", None))
        print(
            "[*] FNO grid detected: "
            f"{grid_info['unique_x']} unique x values x {grid_info['unique_y']} unique y values "
            f"= {grid_info['num_points']} points."
        )
        if grid_info["requires_permutation"]:
            print(
                "[*] FNO grid order: dataset is not row-major; the FNO backbone will "
                "apply its coordinate-derived permutation during evaluation."
            )

    try:
        # Build and restore on CPU first to avoid temporarily holding both the
        # checkpoint tensors and the live model weights on the target device.
        model = _build_model(cfg, dataset)
    except Exception as e:
        print(f"[Warning: !] Model construction failed: {e}")
        raise SystemExit(1)

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    except pickle.UnpicklingError:
        print("[Warning: !] Restricted torch.load failed; retrying with weights_only=False for a trusted local checkpoint.")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Some checkpoints may carry "_metadata" as a literal key after serialization.
    # It is not a model parameter and must be removed before load_state_dict(...).
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = dict(state_dict)   # make a plain mutable copy
        state_dict.pop("_metadata", None)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[Warning: !] Checkpoint is incompatible with the reconstructed model: {e}")
        if cfg.get("backbone") == "fno":
            print(
                "[Warning: !] FNO conditioning now uses normalized, support-weighted, "
                "and soft-support channels (4 * n_fields + 1 inputs). Older FNO "
                "checkpoints trained with the previous 3 * n_fields + 1 input layout "
                "must be retrained."
            )
        raise SystemExit(1)

    epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    del state_dict
    del ckpt

    model = model.to(device)
    model.eval()

    vis_cond_fields = args.vis_cond_fields if args.vis_cond_fields is not None else cfg["vis_cond_fields"]
    vis_n_obs_list = args.vis_n_obs_list if args.vis_n_obs_list is not None else cfg["vis_n_obs_list"]

    print(f'\nvis_n_obs_list is {vis_n_obs_list}\n')
    
    n_steps_generation = (
        args.n_steps_generation if args.n_steps_generation is not None
        else cfg.get("n_steps_generation", 100)
    )
    ode_solver = args.ode_solver if args.ode_solver is not None else cfg.get("ode_solver", "euler")
    if ode_solver not in ("euler", "heun"):
        raise ValueError(f"Unsupported ode_solver={ode_solver!r}; expected 'euler' or 'heun'.")
    print(f'\nResults are generated from solver={ode_solver}, n_steps={n_steps_generation}\n')

    eval_timestamp = torch.tensor([])  # dummy to avoid importing datetime twice
    from datetime import datetime
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = demo_root / "Save_reconstruction_files" / "ForOfflineEvaluation" / f"eval_N{args.Demo_Num}_{eval_timestamp}_from_{train_timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    final_clamp = not args.no_obs_consistency_final_clamp
    need_payload = (
        (len(args.extra_metrics) > 0)
        or args.save_analysis_npz
        or args.save_obs_consistency_plots
        or args.obs_consistency_compare_modes is not None
    )
    metrics_by_mode = None

    if args.obs_consistency_compare_modes is None:
        result = visualize_reconstruction(
            model=model,
            dataset=dataset,
            epoch=epoch,
            device=device,
            save_dir=str(out_dir),
            cond_fields=vis_cond_fields,
            n_obs=vis_n_obs_list,
            n_steps=n_steps_generation,
            ode_solver=ode_solver,
            snapshot_index=args.snapshot_index,
            file_tag=f"snapshot_{args.snapshot_index:04d}_{ode_solver}",
            save_metrics_json=True,
            return_payload=need_payload,
            obs_consistency_mode=args.obs_consistency_mode,
            obs_consistency_strength=args.obs_consistency_strength,
            obs_consistency_sigma=args.obs_consistency_sigma,
            obs_consistency_schedule_power=args.obs_consistency_schedule_power,
            obs_consistency_final_clamp=final_clamp,
            save_obs_consistency_plots=args.save_obs_consistency_plots,
        )

        if need_payload:
            metrics, payload = result
        else:
            metrics = result
            payload = None
    else:
        metrics_by_mode = {}
        comparison_rows = []
        sparse_condition = None
        payload = None
        metrics = {}
        for mode in args.obs_consistency_compare_modes:
            mode_result = visualize_reconstruction(
                model=model,
                dataset=dataset,
                epoch=epoch,
                device=device,
                save_dir=str(out_dir),
                cond_fields=vis_cond_fields,
                n_obs=vis_n_obs_list,
                n_steps=n_steps_generation,
                ode_solver=ode_solver,
                snapshot_index=args.snapshot_index,
                file_tag=f"snapshot_{args.snapshot_index:04d}_{ode_solver}_{mode}",
                save_metrics_json=True,
                return_payload=True,
                obs_consistency_mode=mode,
                obs_consistency_strength=args.obs_consistency_strength,
                obs_consistency_sigma=args.obs_consistency_sigma,
                obs_consistency_schedule_power=args.obs_consistency_schedule_power,
                obs_consistency_final_clamp=final_clamp,
                save_obs_consistency_plots=False,
                sparse_condition=sparse_condition,
            )
            mode_metrics, mode_payload = mode_result
            if sparse_condition is None:
                sparse_condition = {
                    "obs_coords": mode_payload["obs_coords"],
                    "obs_values": mode_payload["obs_values"],
                    "obs_mask": mode_payload["obs_mask"],
                    "obs_indices": mode_payload["obs_indices"],
                    "obs_field_ids": mode_payload["obs_field_ids"],
                }
            metrics_by_mode[mode] = mode_metrics
            payload = mode_payload
            metrics = mode_metrics
            row = {
                "mode": mode,
                "relative_l2": _mean_full_field_relative_l2(mode_metrics),
                "obs_rel_l2_SenConsis": mode_metrics.get("obs_rel_l2_SenConsis", float("nan")),
                "obs_count_SenConsis_total": mode_metrics.get("obs_count_SenConsis_total", 0),
            }
            comparison_rows.append(row)

        senconsis_dir = out_dir / "SenConsis"
        save_obs_consistency_comparison(comparison_rows, str(senconsis_dir))

    extra_metrics = {}

    if payload is not None and len(args.extra_metrics) > 0:
        try:
            grid_info = _infer_structured_grid_from_coords(
                payload["coords_xy"],
                num_x=cfg.get("Num_x", None),
                num_y=cfg.get("Num_y", None),
            )
        except ValueError as e:
            print(f"[Warning: !] Extra structured-grid metrics skipped: {e}")
            grid_info = None

        if grid_info is not None:
            field_names = payload["field_names"]
            truth_phys = payload["truth_phys"]
            recon_phys = payload["recon_phys"]

            prefix = f"snapshot_{args.snapshot_index:04d}"

            for c, name in enumerate(field_names):
                u = _reshape_flat_field_to_grid(truth_phys[:, c], grid_info)
                v = _reshape_flat_field_to_grid(recon_phys[:, c], grid_info)

                field_metrics = {}
                analysis_payload = {
                    "true_grid": u,
                    "pred_grid": v,
                    "abs_err_grid": np.abs(v - u),
                    "x_unique": grid_info["x_unique"],
                    "y_unique": grid_info["y_unique"],
                }

                if "ssim" in args.extra_metrics:
                    field_metrics["ssim"] = _ssim2d(
                        u, v,
                        data_range=float(u.max() - u.min())
                    )

                if "grad" in args.extra_metrics:
                    grad_metrics, grad_payload = _gradient_metrics(
                        u, v,
                        dx=grid_info["dx"],
                        dy=grid_info["dy"],
                    )
                    field_metrics.update(grad_metrics)
                    analysis_payload.update(grad_payload)

                if "spectrum" in args.extra_metrics:
                    spec_metrics, spec_payload = _spectral_metrics(
                        u, v,
                        dx=grid_info["dx"],
                        dy=grid_info["dy"],
                    )
                    field_metrics.update(spec_metrics)
                    analysis_payload.update(spec_payload)

                    spec_plot_path = out_dir / f"{prefix}_field_{name}_spectrum.png"
                    _save_spectrum_plot(
                        spec_payload["k"],
                        spec_payload["spectrum_true"],
                        spec_payload["spectrum_pred"],
                        band_edges=spec_payload["band_edges"],
                        save_path=spec_plot_path,
                        title=f"{name} spectrum",
                    )
                    band_plot_path = out_dir / f"{prefix}_field_{name}_band_energy_ratio.png"
                    _save_band_energy_plot(
                        spec_payload["band_names"],
                        spec_payload["band_ratio_pred_over_true"],
                        save_path=band_plot_path,
                        title=f"{name} band energy ratio",
                    )

                extra_metrics[name] = field_metrics

                if args.save_analysis_npz:
                    npz_path = out_dir / f"{prefix}_field_{name}_analysis.npz"
                    np.savez_compressed(npz_path, **analysis_payload)

    summary = {
        "demo_num": int(args.Demo_Num),
        "yaml_path": str(yaml_path),
        "model_root": str(model_root),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "snapshot_index": int(args.snapshot_index),
        "vis_cond_fields": [int(v) for v in vis_cond_fields],
        "vis_n_obs_list": [int(v) for v in vis_n_obs_list],
        "n_steps_generation": int(n_steps_generation),
        "ode_solver": ode_solver,
        "obs_consistency_mode": args.obs_consistency_mode,
        "obs_consistency_strength": float(args.obs_consistency_strength),
        "obs_consistency_sigma": float(args.obs_consistency_sigma),
        "obs_consistency_schedule_power": float(args.obs_consistency_schedule_power),
        "obs_consistency_final_clamp": bool(final_clamp),
        "obs_consistency_compare_modes": args.obs_consistency_compare_modes,
        "metrics": metrics,
        "metrics_by_mode": metrics_by_mode,
        "extra_metric_names": list(args.extra_metrics),
        "extra_metrics": extra_metrics,
    }

    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[*] Evaluation finished.")
    print(f"[*] YAML      : {yaml_path}")
    print(f"[*] Checkpoint: {ckpt_path}")
    print(f"[*] Output dir : {out_dir}")
    print(f"[*] Metrics         : {json.dumps(metrics, indent=2)}")
    if len(extra_metrics) > 0:
        print(f"[*] Extra metrics   : {json.dumps(extra_metrics, indent=2)}")


if __name__ == "__main__":
    main()

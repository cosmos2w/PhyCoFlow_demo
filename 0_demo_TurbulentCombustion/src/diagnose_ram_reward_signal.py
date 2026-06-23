"""
Offline RAM reward-strength diagnostic for turbulent-combustion PointCloudFFM.

This script checks whether the RAM coherence reward creates a useful ranking
signal within fixed sparse conditions.  It loads the pretrained source/base
model, samples G endpoint candidates per condition, computes coherence rewards,
and summarizes reward spread plus correlations with reconstruction metrics.

Run this script like:
python src/diagnose_ram_reward_signal.py \
  --source-Demo-Num 15 \
  --source-checkpoint last \
  --num-conditions 100 \
  --num-samples-per-condition 24 \
  --split test \
  --n-steps 2

"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
import yaml
from tqdm import tqdm

from coherence_dist import compute_ram_coherence_cost
from helpers import TurbulentCombustionH5Dataset, build_sparse_condition
from model_finetune import find_source_run_dir, load_pretrained_ffm, load_source_config
from train_finetune import (
    _repeat_batch,
    _sample_endpoints_microbatched,
    align_per_field_values,
    build_ram_coherence_config,
    inherit_conditioning_from_source,
    load_ram_config,
    resolve_demo_path,
    set_seed,
    subsample_reward_points,
)


DEFAULT_CONFIG = {
    "source_Demo_Num": 15,
    "source_checkpoint": "last",
    "source_run_dir": None,
    "data": "Dataset/Merged_CH4COTU1P.h5",
    "split": "test",
    "seed": 42,
    "num_conditions": 50,
    "condition_indices": None,
    "num_samples_per_condition": 12,
    "n_obs_list": None,
    "n_steps": 2,
    "ode_solver": "euler",
    "ram_endpoint_microbatch_size": 72,
    "ram_reward_n_points": 4096,
    "ram_reward_sampling": "uniform",
    "reward_subsample_seed": 1234,
    "reward_mode": "global_dist",
    "reward_spread_tiny_threshold": 1.0e-4,
    "coherence_use_denorm": False,
    "global_lambda_marg": 1.0,
    "global_lambda_joint": 1.0,
    "global_num_directions": 32,
    "global_joint_top_frac": 0.10,
    "global_include_pairwise": True,
    "global_lambda_pairwise": 0.25,
    "global_include_pairwise_in_score": False,
    "obs_consistency_mode": "endpoint_smooth",
    "obs_consistency_strength": 1.0,
    "obs_consistency_sigma": 0.05,
    "obs_consistency_schedule_power": 2.0,
    "obs_consistency_final_clamp": True,
    "save_candidate_fields": False,
    "num_visual_examples": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Offline RAM reward ranking-signal diagnostic.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="Save_config/config_pointcloud_ffm_ram.yaml")
    parser.add_argument("--source-Demo-Num", dest="source_Demo_Num", type=int, default=None)
    parser.add_argument("--source-run-dir", type=str, default=None)
    parser.add_argument("--source-checkpoint", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-conditions", type=int, default=None)
    parser.add_argument("--condition-indices", type=int, nargs="+", default=None)
    parser.add_argument("--num-samples-per-condition", type=int, default=None)
    parser.add_argument("--n-obs-list", type=int, nargs="+", default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-candidate-fields", action="store_true")
    parser.add_argument("--num-visual-examples", type=int, default=None)
    return parser.parse_args()


def load_diagnosis_config(args: argparse.Namespace, demo_dir: Path) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    config_path = resolve_demo_path(demo_dir, args.config)
    if config_path.exists():
        ram_cfg = load_ram_config(config_path)
        for key in [
            "source_Demo_Num",
            "source_checkpoint",
            "source_run_dir",
            "data",
            "train_ratio",
            "time_stride",
            "cond_fields",
            "n_obs_min_list",
            "n_obs_max_list",
            "ram_endpoint_microbatch_size",
            "ram_reward_n_points",
            "ram_reward_sampling",
            "reward_subsample_seed",
            "reward_mode",
            "coherence_use_denorm",
            "global_lambda_marg",
            "global_lambda_joint",
            "global_num_directions",
            "global_joint_top_frac",
            "global_include_pairwise",
            "global_lambda_pairwise",
            "global_include_pairwise_in_score",
            "obs_consistency_strength",
            "obs_consistency_sigma",
            "obs_consistency_schedule_power",
            "obs_consistency_final_clamp",
        ]:
            if key in ram_cfg:
                cfg[key] = ram_cfg[key]
        cfg["num_samples_per_condition"] = ram_cfg.get(
            "num_samples_per_condition",
            cfg["num_samples_per_condition"],
        )
        cfg["n_steps"] = ram_cfg.get("ram_endpoint_steps", cfg["n_steps"])
        cfg["ode_solver"] = ram_cfg.get("ode_solver", cfg["ode_solver"])
        cfg["obs_consistency_mode"] = ram_cfg.get(
            "ram_obs_consistency_mode",
            cfg["obs_consistency_mode"],
        )
    cfg["config_path"] = str(config_path)

    for attr in [
        "source_Demo_Num",
        "source_run_dir",
        "source_checkpoint",
        "data",
        "split",
        "seed",
        "num_conditions",
        "condition_indices",
        "num_samples_per_condition",
        "n_steps",
        "num_visual_examples",
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            cfg[attr] = value
    if args.n_obs_list is not None:
        cfg["n_obs_list"] = list(args.n_obs_list)
    if args.save_candidate_fields:
        cfg["save_candidate_fields"] = True
    return cfg


def select_condition_indices(dataset_len: int, cfg: dict) -> list[int]:
    explicit = cfg.get("condition_indices")
    if explicit:
        indices = [int(v) for v in explicit]
    else:
        n = min(int(cfg.get("num_conditions", 50)), int(dataset_len))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(cfg.get("seed", 42)))
        indices = torch.randperm(dataset_len, generator=generator)[:n].tolist()
    invalid = [idx for idx in indices if idx < 0 or idx >= dataset_len]
    if invalid:
        raise ValueError(f"Condition indices outside split length {dataset_len}: {invalid}")
    if not indices:
        raise ValueError("No condition indices selected for diagnosis.")
    return indices


def corrcoef_safe(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) < 2:
        return float("nan")
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def rel_l2(pred: torch.Tensor, ref: torch.Tensor, dims: tuple[int, ...]) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - ref, dim=dims) / (
        torch.linalg.vector_norm(ref, dim=dims) + 1e-12
    )


def structured_grid_info(coords_xy: np.ndarray) -> Optional[dict]:
    x_vals = np.unique(coords_xy[:, 0])
    y_vals = np.unique(coords_xy[:, 1])
    if len(x_vals) * len(y_vals) != coords_xy.shape[0]:
        return None
    order = np.lexsort((coords_xy[:, 0], coords_xy[:, 1]))
    sorted_coords = coords_xy[order]
    expected_x, expected_y = np.meshgrid(x_vals, y_vals)
    expected = np.stack([expected_x.ravel(), expected_y.ravel()], axis=1)
    if not np.allclose(sorted_coords[:, :2], expected, rtol=1e-6, atol=1e-8):
        return None
    dx = float(np.median(np.diff(x_vals))) if len(x_vals) > 1 else 1.0
    dy = float(np.median(np.diff(y_vals))) if len(y_vals) > 1 else 1.0
    return {"nx": len(x_vals), "ny": len(y_vals), "order": order, "dx": dx, "dy": dy}


def radial_spectrum(field_2d: np.ndarray, dx: float, dy: float) -> np.ndarray:
    centered = field_2d - np.mean(field_2d)
    fft = np.fft.fftshift(np.fft.fft2(centered))
    psd2 = (np.abs(fft) ** 2) / max(field_2d.size, 1)
    ny, nx = field_2d.shape
    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    kmag = np.sqrt(kx_grid**2 + ky_grid**2)
    dk = max(float(min(np.min(np.diff(np.unique(kx))) if nx > 1 else 1.0, np.min(np.diff(np.unique(ky))) if ny > 1 else 1.0)), 1e-12)
    shell_id = np.rint(kmag / abs(dk)).astype(np.int64)
    n_shells = int(shell_id.max()) + 1
    shell_sum = np.bincount(shell_id.ravel(), weights=psd2.ravel(), minlength=n_shells)
    shell_count = np.bincount(shell_id.ravel(), minlength=n_shells)
    radial = shell_sum / np.maximum(shell_count, 1)
    return radial[1:] if radial.shape[0] > 1 else radial


def spectral_rel_error(
    pred: np.ndarray,
    ref: np.ndarray,
    grid: Optional[dict],
) -> float:
    if grid is None:
        return float("nan")
    order = grid["order"]
    ny = int(grid["ny"])
    nx = int(grid["nx"])
    values = []
    for channel in range(ref.shape[1]):
        ref_grid = ref[order, channel].reshape(ny, nx)
        pred_grid = pred[order, channel].reshape(ny, nx)
        ref_spec = radial_spectrum(ref_grid, dx=float(grid["dx"]), dy=float(grid["dy"]))
        pred_spec = radial_spectrum(pred_grid, dx=float(grid["dx"]), dy=float(grid["dy"]))
        n = min(len(ref_spec), len(pred_spec))
        if n == 0:
            continue
        values.append(float(np.linalg.norm(pred_spec[:n] - ref_spec[:n]) / (np.linalg.norm(ref_spec[:n]) + 1e-12)))
    return float(np.mean(values)) if values else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_hist(path: Path, values: Sequence[float], title: str, xlabel: str) -> None:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.hist(arr, bins=min(30, max(8, int(math.sqrt(arr.size)) + 1)), color="#4C78A8", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_scatter(path: Path, x: Sequence[float], y: Sequence[float], title: str, xlabel: str, ylabel: str) -> None:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if int(mask.sum()) == 0:
        return
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.scatter(x_arr[mask], y_arr[mask], s=20, alpha=0.75, color="#F58518")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_candidate_comparison(
    path: Path,
    coords_xy: np.ndarray,
    ref_phys: np.ndarray,
    best_phys: np.ndarray,
    worst_phys: np.ndarray,
    field_names: Sequence[str],
    title: str,
) -> None:
    n_fields = len(field_names)
    triang = mtri.Triangulation(coords_xy[:, 0], coords_xy[:, 1])
    fig, axes = plt.subplots(n_fields, 3, figsize=(12, max(2.2 * n_fields, 5.0)), squeeze=False, constrained_layout=True)
    for c, name in enumerate(field_names):
        panels = [(ref_phys[:, c], "reference"), (best_phys[:, c], "top reward"), (worst_phys[:, c], "bottom reward")]
        lo = float(min(values.min() for values, _ in panels))
        hi = float(max(values.max() for values, _ in panels))
        for j, (values, label) in enumerate(panels):
            ax = axes[c, j]
            im = ax.tricontourf(triang, values, levels=64, cmap="coolwarm", vmin=lo, vmax=hi)
            ax.set_title(f"{name} {label}", fontsize=9)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summarize(values: Sequence[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    demo_dir = script_dir.parent
    cfg = load_diagnosis_config(args, demo_dir)
    set_seed(int(cfg.get("seed", 42)))

    device = torch.device(
        args.device
        if args.device is not None
        else (f"cuda:{int(cfg.get('device_ids', [0])[0])}" if torch.cuda.is_available() else "cpu")
    )

    source_run_dir = find_source_run_dir(
        demo_dir=demo_dir,
        source_run_dir=cfg.get("source_run_dir", None),
        source_Demo_Num=cfg.get("source_Demo_Num", None),
    )
    source_cfg = load_source_config(source_run_dir)
    cfg = inherit_conditioning_from_source(cfg, source_cfg)

    if cfg.get("n_obs_list") is not None:
        source_cond_fields = (
            source_cfg.get("cond_fields")
            if source_cfg.get("cond_fields") is not None
            else [source_cfg.get("cond_field", 2)]
        )
        n_obs_list = align_per_field_values(
            cfg["n_obs_list"],
            cfg["cond_fields"],
            "n_obs_list",
            source_fields=source_cond_fields,
        )
        cfg["n_obs_min_list"] = list(n_obs_list)
        cfg["n_obs_max_list"] = list(n_obs_list)
    cfg["ram_endpoint_steps"] = int(cfg.get("n_steps", 2))
    cfg["ode_solver"] = str(cfg.get("ode_solver", "euler"))
    cfg["ram_obs_consistency_mode"] = str(cfg.get("obs_consistency_mode", "endpoint_smooth"))

    data_path = resolve_demo_path(demo_dir, cfg.get("data", DEFAULT_CONFIG["data"]))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = source_run_dir / f"Coherence_Diagnosis_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(exist_ok=True)

    dataset = TurbulentCombustionH5Dataset(
        str(data_path),
        split=str(cfg.get("split", "test")),
        train_ratio=float(cfg.get("train_ratio", 0.9)),
        seed=int(cfg.get("seed", 42)),
        time_stride=int(cfg.get("time_stride", 1)),
        stats_path=str(source_run_dir / "dataset_stats.pt"),
    )
    model, source_cfg_loaded, _ = load_pretrained_ffm(
        source_run_dir=source_run_dir,
        checkpoint=str(cfg.get("source_checkpoint", "last")),
        dataset=dataset,
        device=device,
    )
    model.eval()
    ram_coh_cfg = build_ram_coherence_config(cfg)
    field_names = list(getattr(dataset, "field_names", [f"field_{i}" for i in range(dataset.num_fields)]))
    condition_indices = select_condition_indices(len(dataset), cfg)

    diagnosis_config = dict(cfg)
    diagnosis_config.update(
        {
            "source_run_dir": str(source_run_dir),
            "source_checkpoint": str(cfg.get("source_checkpoint", "last")),
            "source_backbone": source_cfg_loaded.get("backbone"),
            "output_dir": str(out_dir),
            "device": str(device),
            "condition_indices": condition_indices,
            "spectral_metrics_note": "available only when coords form a complete structured grid",
        }
    )
    with open(out_dir / "diagnosis_config.json", "w", encoding="utf-8") as handle:
        json.dump(diagnosis_config, handle, indent=2, default=str)

    candidate_rows: list[dict] = []
    condition_rows: list[dict] = []
    saved_examples = 0
    mean = dataset.mean.to(device).view(1, 1, -1)
    std = dataset.std.to(device).view(1, 1, -1)
    G = int(cfg.get("num_samples_per_condition", 12))

    for condition_rank, dataset_index in enumerate(tqdm(condition_indices, desc="diagnosing reward signal")):
        sample = dataset[int(dataset_index)]
        coords = sample["coords"].unsqueeze(0).to(device)
        fields = sample["fields"].unsqueeze(0).to(device)
        coords_raw = sample.get("coords_raw", sample["coords"]).detach().cpu().numpy()
        coords_xy = coords_raw[:, :2]
        grid = structured_grid_info(coords_xy)

        torch.manual_seed(int(cfg.get("seed", 42)) + condition_rank * 1009)
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords,
            fields_full=fields,
            cond_fields=cfg["cond_fields"],
            n_obs_min=cfg["n_obs_min_list"],
            n_obs_max=cfg["n_obs_max_list"],
        )
        coords_g = _repeat_batch(coords, G)
        fields_g = _repeat_batch(fields, G)
        obs_coords_g = _repeat_batch(obs_coords, G)
        obs_values_g = _repeat_batch(obs_values, G)
        obs_mask_g = _repeat_batch(obs_mask, G)
        obs_indices_g = _repeat_batch(obs_indices, G)
        obs_field_ids_g = _repeat_batch(obs_field_ids, G)

        with torch.no_grad():
            x_end = _sample_endpoints_microbatched(
                model=model,
                coords=coords_g,
                obs_coords=obs_coords_g,
                obs_values=obs_values_g,
                obs_mask=obs_mask_g,
                obs_field_ids=obs_field_ids_g,
                obs_indices=obs_indices_g,
                cfg=cfg,
            )
            x_end_reward, fields_reward, reward_point_count = subsample_reward_points(
                x_gen=x_end,
                x_ref=fields_g,
                n_points=cfg.get("ram_reward_n_points", None),
                mode=str(cfg.get("ram_reward_sampling", "uniform")),
            )
            candidate_costs = []
            candidate_coh_metrics = []
            for candidate_idx in range(G):
                cost_item, coh_metrics_item = compute_ram_coherence_cost(
                    x_gen=x_end_reward[candidate_idx : candidate_idx + 1],
                    x_ref=fields_reward[candidate_idx : candidate_idx + 1],
                    cfg=ram_coh_cfg,
                    mean=dataset.mean.to(device),
                    std=dataset.std.to(device),
                )
                candidate_costs.append(cost_item.reshape(-1))
                candidate_coh_metrics.append(dict(coh_metrics_item))
            cost = torch.cat(candidate_costs, dim=0)

        rewards = (-cost).detach().cpu().numpy()
        costs = cost.detach().cpu().numpy()
        norm_l2 = rel_l2(x_end, fields_g, dims=(1, 2)).detach().cpu().numpy()
        x_end_phys = x_end * std + mean
        fields_phys = fields_g * std + mean
        phys_l2 = rel_l2(x_end_phys, fields_phys, dims=(1, 2)).detach().cpu().numpy()
        per_field_l2 = rel_l2(x_end_phys, fields_phys, dims=(1,)).detach().cpu().numpy()
        x_end_phys_np = x_end_phys.detach().cpu().numpy()
        fields_phys_np = fields_phys.detach().cpu().numpy()
        spectral_errors = [
            spectral_rel_error(x_end_phys_np[g], fields_phys_np[g], grid)
            for g in range(G)
        ]

        for g in range(G):
            row: Dict[str, Any] = {
                "condition_rank": condition_rank,
                "dataset_index": int(dataset_index),
                "candidate": g,
                "reward": float(rewards[g]),
                "coherence_cost": float(costs[g]),
                "normalized_rel_l2": float(norm_l2[g]),
                "physical_rel_l2": float(phys_l2[g]),
                "spectral_rel_error": float(spectral_errors[g]),
                "reward_point_count": int(reward_point_count),
            }
            for key, value in candidate_coh_metrics[g].items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    row[f"coherence/{key}"] = float(value)
            for c, name in enumerate(field_names):
                row[f"physical_rel_l2/{name}"] = float(per_field_l2[g, c])
            candidate_rows.append(row)

        best_idx = int(np.argmax(rewards))
        worst_idx = int(np.argmin(rewards))
        condition_row = {
            "condition_rank": condition_rank,
            "dataset_index": int(dataset_index),
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "reward_gap_top_bottom": float(np.max(rewards) - np.min(rewards)),
            "cost_gap": float(np.max(costs) - np.min(costs)),
            "cost_min": float(np.min(costs)),
            "cost_max": float(np.max(costs)),
            "best_candidate": best_idx,
            "worst_candidate": worst_idx,
            "best_reward": float(rewards[best_idx]),
            "worst_reward": float(rewards[worst_idx]),
            "best_physical_rel_l2": float(phys_l2[best_idx]),
            "worst_physical_rel_l2": float(phys_l2[worst_idx]),
            "reward_norm_l2_corr": corrcoef_safe(rewards, norm_l2),
            "reward_phys_l2_corr": corrcoef_safe(rewards, phys_l2),
            "reward_spectral_corr": corrcoef_safe(rewards, spectral_errors),
            "spectral_metrics_available": bool(grid is not None),
        }
        condition_rows.append(condition_row)

        if bool(cfg.get("save_candidate_fields", False)):
            np.savez_compressed(
                out_dir / f"condition_{condition_rank:04d}_candidates.npz",
                dataset_index=int(dataset_index),
                coords=coords.detach().cpu().numpy(),
                reference=fields.detach().cpu().numpy(),
                candidates=x_end.detach().cpu().numpy(),
                rewards=rewards,
                costs=costs,
            )
        if saved_examples < int(cfg.get("num_visual_examples", 3)):
            save_candidate_comparison(
                examples_dir / f"condition_{condition_rank:04d}_top_bottom.png",
                coords_xy=coords_xy,
                ref_phys=fields_phys_np[0],
                best_phys=x_end_phys_np[best_idx],
                worst_phys=x_end_phys_np[worst_idx],
                field_names=field_names,
                title=(
                    f"condition {condition_rank} dataset_index={int(dataset_index)} | "
                    f"reward gap={condition_row['reward_gap_top_bottom']:.3e}"
                ),
            )
            saved_examples += 1

    write_csv(out_dir / "candidate_metrics.csv", candidate_rows)
    write_csv(out_dir / "condition_summary.csv", condition_rows)

    reward_stds = [row["reward_std"] for row in condition_rows]
    reward_gaps = [row["reward_gap_top_bottom"] for row in condition_rows]
    cost_gaps = [row["cost_gap"] for row in condition_rows]
    reward_phys_corrs = [row["reward_phys_l2_corr"] for row in condition_rows]
    reward_spectral_corrs = [row["reward_spectral_corr"] for row in condition_rows]
    spectral_available = any(bool(row["spectral_metrics_available"]) for row in condition_rows)
    threshold = float(cfg.get("reward_spread_tiny_threshold", 1.0e-4))
    summary = {
        "num_conditions": len(condition_rows),
        "num_candidates": len(candidate_rows),
        "num_samples_per_condition": G,
        "reward_std": summarize(reward_stds),
        "reward_gap_top_bottom": summarize(reward_gaps),
        "cost_gap": summarize(cost_gaps),
        "reward_physical_l2_corr": summarize(reward_phys_corrs),
        "reward_spectral_corr": summarize(reward_spectral_corrs),
        "spectral_metrics_available": spectral_available,
        "reward_spread_tiny_threshold": threshold,
        "tiny_reward_spread_fraction": float(np.mean(np.asarray(reward_stds) < threshold)) if reward_stds else None,
        "warning_tiny_reward_spread": bool(np.nanmedian(np.asarray(reward_stds, dtype=np.float64)) < threshold) if reward_stds else False,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    save_hist(plots_dir / "hist_reward_std.png", reward_stds, "Within-condition reward std", "std(reward)")
    save_hist(plots_dir / "hist_reward_gap_top_bottom.png", reward_gaps, "Top-bottom reward gap", "max(reward) - min(reward)")
    save_hist(plots_dir / "hist_reward_l2_corr.png", reward_phys_corrs, "Reward vs physical L2 correlation", "corr(reward, physical L2)")
    save_scatter(
        plots_dir / "scatter_reward_vs_normalized_l2.png",
        [row["reward"] for row in candidate_rows],
        [row["normalized_rel_l2"] for row in candidate_rows],
        "Reward vs normalized relative L2",
        "reward",
        "normalized relative L2",
    )
    save_scatter(
        plots_dir / "scatter_reward_vs_physical_l2.png",
        [row["reward"] for row in candidate_rows],
        [row["physical_rel_l2"] for row in candidate_rows],
        "Reward vs physical relative L2",
        "reward",
        "physical relative L2",
    )
    if spectral_available:
        save_scatter(
            plots_dir / "scatter_reward_vs_spectral_error.png",
            [row["reward"] for row in candidate_rows],
            [row["spectral_rel_error"] for row in candidate_rows],
            "Reward vs spectral relative error",
            "reward",
            "spectral relative error",
        )

    print(f"[*] Reward diagnosis complete: {out_dir}")
    print(
        "[*] Reward std median="
        f"{summary['reward_std']['median']} | "
        f"top-bottom gap median={summary['reward_gap_top_bottom']['median']} | "
        f"tiny_spread_warning={summary['warning_tiny_reward_spread']}"
    )


if __name__ == "__main__":
    main()

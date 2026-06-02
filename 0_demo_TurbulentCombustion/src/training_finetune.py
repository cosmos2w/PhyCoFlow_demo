"""
RAM fine-tuning entrypoint for turbulent-combustion PointCloudFFM models.

Quick usage from ``0_demo_TurbulentCombustion/``:
    python src/training_finetune.py \
      --config Save_config/config_pointcloud_ffm_ram.yaml \
      --Demo-Num 20

General structure:
    1) Load a pretrained PointCloudFFM source run.
    2) Create four model roles:
       ref_model    - frozen pretrained base velocity
       policy_model - trainable RAM policy
       old_model    - lagged EMA policy used for endpoint sampling/targets
       eval_model   - smoother EMA policy saved in checkpoints
    3) Sample multiple endpoints per sparse-condition group with old_model.
    4) Convert coherence costs into group-relative advantages.
    5) Re-noise endpoints analytically under the PhyCoFlow convention
       x_t = (1 - t) * z + t * x, target velocity = x - z.
    6) Fit policy velocity to the detached RAM target.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from coherence_dist import RAMCoherenceConfig, compute_ram_coherence_cost
from helpers import (
    TurbulentCombustionH5Dataset,
    build_sparse_condition,
    validate_regular_grid_compatibility,
    visualize_reconstruction,
)
from model_finetune import (
    clone_model,
    find_source_run_dir,
    load_pretrained_ffm,
    load_source_config,
    set_trainable_scope,
    sync_params,
    update_ema_params,
)


DEFAULTS = {
    "Demo_Num": 20,
    "seed": 42,
    "device_ids": [0],
    "source_run_dir": None,
    "source_Demo_Num": 15,
    "source_checkpoint": "best",
    "data": "Dataset/Merged_CH4COTU1P.h5",
    "train_ratio": 0.9,
    "time_stride": 1,
    "num_workers": 4,
    "save_dir": "Save_TrainedModel/ram_tc_pointcloud",
    "cond_fields": None,
    "n_obs_min_list": None,
    "n_obs_max_list": None,
    "ram_endpoint_steps": 4,
    "ode_solver": "euler",
    "ram_obs_consistency_mode": "endpoint_smooth",
    "obs_consistency_strength": 1.0,
    "obs_consistency_sigma": 0.05,
    "obs_consistency_schedule_power": 2.0,
    "obs_consistency_final_clamp": True,
    "ram_epochs": 200,
    "batch_size": 2,
    "num_samples_per_condition": 4,
    "num_loss_targets_per_endpoint": 4,
    "timestep_sampling": "mirrored_weighted",
    "t_eps": 1.0e-3,
    "reward_mode": "global_dist",
    "reward_multiplier": 1.0,
    "reward_scaling": "running_epoch_std",
    "reward_eps": 1.0e-4,
    "coherence_use_denorm": False,
    "global_lambda_marg": 1.0,
    "global_lambda_joint": 1.0,
    "global_num_directions": 64,
    "global_joint_top_frac": 0.10,
    "global_include_pairwise": True,
    "global_lambda_pairwise": 0.25,
    "global_include_pairwise_in_score": True,
    "lr": 2.0e-5,
    "weight_decay": 1.0e-4,
    "beta1": 0.9,
    "beta2": 0.99,
    "grad_clip": 1.0,
    "old_ema_decay": 0.9,
    "eval_ema_decay": 0.99,
    "finetune_mode": "head_glres",
    "eval_every": 5,
    "save_every": 20,
    "n_steps_generation_eval": 4,
    "eval_num_batches": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(
        "RAM fine-tuning for pretrained PointCloudFFM models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="Save_config/config_pointcloud_ffm_ram.yaml")
    parser.add_argument("--Demo-Num", dest="Demo_Num", type=int, default=None)
    parser.add_argument("--device-ids", type=int, nargs="+", default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_snapshots(batch):
    return {
        "coords": torch.stack([item["coords"] for item in batch], dim=0),
        "fields": torch.stack([item["fields"] for item in batch], dim=0),
        "time_index": torch.stack([item["time_index"] for item in batch], dim=0),
        "physical_time": torch.stack([item["physical_time"] for item in batch], dim=0),
    }


def resolve_demo_path(demo_dir: Path, path_like) -> Path:
    path = Path(str(path_like))
    return path if path.is_absolute() else demo_dir / path


def load_ram_config(config_path: Path) -> dict:
    cfg = dict(DEFAULTS)
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        cfg.update(payload)
    else:
        raise FileNotFoundError(f"RAM config not found: {config_path}")
    return cfg


def _as_int_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (int, np.integer)):
        return [int(value)]
    return [int(v) for v in value]


def inherit_conditioning_from_source(cfg: dict, source_cfg: dict) -> dict:
    """
    Fill sparse-condition RAM settings from the source pretraining config.

    Fine-tuning can override these, but the default behavior mirrors the
    pretrained model's sparse-sensor distribution.
    """
    out = dict(cfg)

    if out.get("cond_fields") is None:
        if source_cfg.get("cond_fields") is not None:
            out["cond_fields"] = source_cfg.get("cond_fields")
        else:
            out["cond_fields"] = [source_cfg.get("cond_field", 2)]

    if out.get("n_obs_min_list") is None:
        if source_cfg.get("n_obs_min_list") is not None:
            out["n_obs_min_list"] = source_cfg.get("n_obs_min_list")
        else:
            out["n_obs_min_list"] = [source_cfg.get("n_obs_min", 64)]

    if out.get("n_obs_max_list") is None:
        if source_cfg.get("n_obs_max_list") is not None:
            out["n_obs_max_list"] = source_cfg.get("n_obs_max_list")
        else:
            out["n_obs_max_list"] = [source_cfg.get("n_obs_max", 256)]

    out["cond_fields"] = _as_int_list(out["cond_fields"])
    out["n_obs_min_list"] = _as_int_list(out["n_obs_min_list"])
    out["n_obs_max_list"] = _as_int_list(out["n_obs_max_list"])
    return out


def build_ram_coherence_config(cfg: dict) -> RAMCoherenceConfig:
    return RAMCoherenceConfig(
        mode=cfg.get("reward_mode", "global_dist"),
        use_denorm=bool(cfg.get("coherence_use_denorm", False)),
        lambda_global=float(cfg.get("lambda_global", 1.0)),
        lambda_marg=float(cfg.get("global_lambda_marg", 1.0)),
        lambda_joint=float(cfg.get("global_lambda_joint", 1.0)),
        num_directions=int(cfg.get("global_num_directions", 64)),
        n_iter_theta=int(cfg.get("global_n_iter_theta", 5)),
        lr_theta=float(cfg.get("global_lr_theta", 0.1)),
        ortho_reg=float(cfg.get("global_ortho_reg", 1e-2)),
        n_proj_pairwise=int(cfg.get("global_n_proj_pairwise", 32)),
        include_pairwise=bool(cfg.get("global_include_pairwise", True)),
        joint_method=str(cfg.get("global_joint_method", "topk_swd")),
        joint_top_frac=float(cfg.get("global_joint_top_frac", 0.10)),
        joint_qmc=bool(cfg.get("global_joint_qmc", True)),
        include_axes=bool(cfg.get("global_include_axes", True)),
        lambda_pairwise=float(cfg.get("global_lambda_pairwise", 0.25)),
        include_pairwise_in_score=bool(cfg.get("global_include_pairwise_in_score", True)),
        seed=cfg.get("global_seed", cfg.get("seed", None)),
    )


def sample_ram_times(n: int, device, dtype, eps: float, mode: str) -> torch.Tensor:
    """
    Sample RF times under PhyCoFlow's convention: t=0 source/noise, t=1 clean.

    ``mirrored_weighted`` uses p(t)=2(1-t), emphasizing source-side states.
    """
    mode = str(mode)
    if mode == "mirrored_weighted":
        u = torch.rand(n, device=device, dtype=dtype)
        t = 1.0 - torch.sqrt(u)
    elif mode == "uniform":
        t = torch.rand(n, device=device, dtype=dtype)
    else:
        raise ValueError("timestep_sampling must be 'mirrored_weighted' or 'uniform'.")
    eps = float(eps)
    return eps + (1.0 - 2.0 * eps) * t


class RunningRewardStd:
    """Online pooled reward standard deviation for group-relative advantages."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, values: torch.Tensor) -> None:
        flat = values.detach().float().reshape(-1).cpu()
        for value in flat:
            self.count += 1
            delta = float(value.item()) - self.mean
            self.mean += delta / self.count
            delta2 = float(value.item()) - self.mean
            self.m2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count <= 1:
            return 0.0
        return math.sqrt(max(self.m2 / self.count, 0.0))


class EpochMetrics:
    """Small averaging helper for scalar RAM metrics."""

    def __init__(self):
        self.totals: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value):
                continue
            self.totals[key] = self.totals.get(key, 0.0) + value
            self.counts[key] = self.counts.get(key, 0) + 1

    def mean(self) -> Dict[str, float]:
        return {
            key: self.totals[key] / max(self.counts.get(key, 1), 1)
            for key in self.totals
        }


class RAMHistoryLogger:
    """
    Write standard loss history plus detailed RAM metrics to CSV/JSON/PNG.
    """

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.loss_csv = run_dir / "loss_history.csv"
        self.loss_json = run_dir / "loss_history.json"
        self.loss_plot = run_dir / "loss_history.png"
        self.metrics_csv = run_dir / "ram_metrics.csv"
        self.metrics_json = run_dir / "ram_metrics.json"
        self.loss_rows = []
        self.metric_rows = []
        self.metric_header = [
            "epoch",
            "ram_loss",
            "reward_mean",
            "reward_std",
            "adv_abs_mean",
            "coherence/global_dist_score",
            "coherence/marginal_score",
            "coherence/joint_score",
            "coherence/pairwise_mean",
            "output_delta_norm",
            "target_velocity_norm",
            "endpoint_steps",
            "lr",
            "val_rel_l2",
            "val_coherence_cost",
            "val_loss",
            "train_batches",
        ]

        with open(self.loss_csv, "w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(["epoch", "train_loss", "val_loss"])
        with open(self.metrics_csv, "w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(self.metric_header)

    def log(self, epoch: int, train_loss: float, val_loss: Optional[float], metrics: Dict[str, float]) -> None:
        loss_row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
        }
        self.loss_rows.append(loss_row)
        with open(self.loss_csv, "a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([
                loss_row["epoch"],
                loss_row["train_loss"],
                "" if loss_row["val_loss"] is None else loss_row["val_loss"],
            ])
        with open(self.loss_json, "w", encoding="utf-8") as handle:
            json.dump(self.loss_rows, handle, indent=2)

        metric_row = {"epoch": int(epoch), **metrics}
        self.metric_rows.append(metric_row)
        with open(self.metrics_csv, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([metric_row.get(key, "") for key in self.metric_header])
        with open(self.metrics_json, "w", encoding="utf-8") as handle:
            json.dump(self.metric_rows, handle, indent=2)

        self._plot_loss()

    def _plot_loss(self) -> None:
        train_points = [(row["epoch"], row["train_loss"]) for row in self.loss_rows if row["train_loss"] > 0]
        val_points = [
            (row["epoch"], row["val_loss"])
            for row in self.loss_rows
            if row["val_loss"] is not None and row["val_loss"] > 0
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        if train_points:
            ax.plot([p[0] for p in train_points], [p[1] for p in train_points], marker="o", label="RAM train")
        if val_points:
            ax.plot([p[0] for p in val_points], [p[1] for p in val_points], marker="s", label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("RAM Fine-Tuning Progress")
        ax.grid(True, which="both", linestyle="--", alpha=0.45)
        if train_points or val_points:
            ax.set_yscale("log")
            ax.legend()
        fig.tight_layout()
        fig.savefig(self.loss_plot, dpi=150)
        plt.close(fig)


def _repeat_batch(x: torch.Tensor, repeats: int) -> torch.Tensor:
    return x.repeat_interleave(int(repeats), dim=0)


def _freeze_model(model: nn.Module) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)


def train_ram_epoch(
    *,
    epoch: int,
    policy_model: nn.Module,
    ref_model: nn.Module,
    old_model: nn.Module,
    eval_model: nn.Module,
    trainable_params: Sequence[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    ram_coh_cfg: RAMCoherenceConfig,
    train_set: TurbulentCombustionH5Dataset,
) -> tuple[float, Dict[str, float]]:
    """
    One RAM epoch: sample endpoints, score them, build advantages, then fit the
    policy velocity to the advantage-scaled detached target.
    """
    policy_model.train()
    ref_model.eval()
    old_model.eval()
    eval_model.eval()

    G = int(cfg["num_samples_per_condition"])
    K = int(cfg["num_loss_targets_per_endpoint"])
    reward_scaling = str(cfg.get("reward_scaling", "running_epoch_std"))
    reward_eps = float(cfg.get("reward_eps", 1.0e-4))
    running_reward_std = RunningRewardStd()
    epoch_metrics = EpochMetrics()

    pbar = tqdm(loader, desc=f"RAM epoch {epoch:04d}", leave=False)
    for batch in pbar:
        coords_full = batch["coords"].to(device)
        fields_full = batch["fields"].to(device)
        bsz = coords_full.shape[0]

        # Build sparse observations once per physical condition, then repeat
        # the condition G times so rewards are comparable within each group.
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords_full,
            fields_full=fields_full,
            cond_fields=cfg["cond_fields"],
            n_obs_min=cfg["n_obs_min_list"],
            n_obs_max=cfg["n_obs_max_list"],
        )

        coords_g = _repeat_batch(coords_full, G)
        x_ref_g = _repeat_batch(fields_full, G)
        obs_coords_g = _repeat_batch(obs_coords, G)
        obs_values_g = _repeat_batch(obs_values, G)
        obs_mask_g = _repeat_batch(obs_mask, G)
        obs_indices_g = _repeat_batch(obs_indices, G)
        obs_field_ids_g = _repeat_batch(obs_field_ids, G)

        with torch.no_grad():
            x_end = old_model.sample(
                coords=coords_g,
                obs_coords=obs_coords_g,
                obs_values=obs_values_g,
                obs_mask=obs_mask_g,
                obs_field_ids=obs_field_ids_g,
                n_steps=int(cfg["ram_endpoint_steps"]),
                clamp_indices=obs_indices_g,
                ode_solver=str(cfg.get("ode_solver", "euler")),
                obs_consistency_mode=str(cfg.get("ram_obs_consistency_mode", "endpoint_smooth")),
                obs_consistency_strength=float(cfg.get("obs_consistency_strength", 1.0)),
                obs_consistency_sigma=float(cfg.get("obs_consistency_sigma", 0.05)),
                obs_consistency_schedule_power=float(cfg.get("obs_consistency_schedule_power", 2.0)),
                obs_consistency_final_clamp=bool(cfg.get("obs_consistency_final_clamp", True)),
            )

            cost, coh_metrics = compute_ram_coherence_cost(
                x_gen=x_end,
                x_ref=x_ref_g,
                cfg=ram_coh_cfg,
                mean=train_set.mean.to(device),
                std=train_set.std.to(device),
            )
            rewards = -cost
            running_reward_std.update(rewards)

            rewards_grouped = rewards.view(bsz, G)
            adv = rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)
            if reward_scaling == "running_epoch_std":
                scale = running_reward_std.std + reward_eps
                adv = adv / scale
            elif reward_scaling == "group":
                adv = adv / (rewards_grouped.std(dim=1, keepdim=True, unbiased=False) + reward_eps)
            elif reward_scaling in ("none", "false", "False"):
                pass
            else:
                raise ValueError("reward_scaling must be 'running_epoch_std', 'group', or 'none'.")
            adv_flat = adv.reshape(-1)

        # Analytically re-noise each sampled endpoint K times.  This is RAM's
        # supervised loss batch and is independent of endpoint sampling steps.
        x_end_l = _repeat_batch(x_end.detach(), K)
        coords_l = _repeat_batch(coords_g, K)
        obs_coords_l = _repeat_batch(obs_coords_g, K)
        obs_values_l = _repeat_batch(obs_values_g, K)
        obs_mask_l = _repeat_batch(obs_mask_g, K)
        obs_field_ids_l = _repeat_batch(obs_field_ids_g, K)
        adv_l = _repeat_batch(adv_flat.detach(), K)

        z = policy_model.sample_source(coords_l)
        t = sample_ram_times(
            n=x_end_l.shape[0],
            device=device,
            dtype=x_end_l.dtype,
            eps=float(cfg.get("t_eps", 1.0e-3)),
            mode=str(cfg.get("timestep_sampling", "mirrored_weighted")),
        )
        t_view = t.view(-1, 1, 1)
        x_t = (1.0 - t_view) * z + t_view * x_end_l
        bridge_direction = x_end_l - z

        v_policy = policy_model.model(
            t, x_t, coords_l, obs_coords_l, obs_values_l, obs_mask_l, obs_field_ids_l
        )
        with torch.no_grad():
            v_ref = ref_model.model(
                t, x_t, coords_l, obs_coords_l, obs_values_l, obs_mask_l, obs_field_ids_l
            )
            v_old = old_model.model(
                t, x_t, coords_l, obs_coords_l, obs_values_l, obs_mask_l, obs_field_ids_l
            )

        scaled_adv = float(cfg.get("reward_multiplier", 1.0)) * adv_l.view(-1, 1, 1)
        target_v = v_ref + scaled_adv * (bridge_direction - v_old)
        loss = ((v_policy - target_v.detach()) ** 2).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.get("grad_clip", 1.0)))
        optimizer.step()

        # Step-level EMA keeps the old/eval policies close enough for short RAM
        # runs while still providing lagged targets.
        update_ema_params(policy_model, old_model, float(cfg.get("old_ema_decay", 0.9)))
        update_ema_params(policy_model, eval_model, float(cfg.get("eval_ema_decay", 0.99)))

        current_lr = float(optimizer.param_groups[0]["lr"])
        metrics = {
            "ram_loss": float(loss.detach().cpu()),
            "reward_mean": float(rewards.mean().detach().cpu()),
            "reward_std": float(rewards.std(unbiased=False).detach().cpu()),
            "adv_abs_mean": float(adv.abs().mean().detach().cpu()),
            "coherence/global_dist_score": float(coh_metrics.get("global_dist_score", np.nan)),
            "coherence/marginal_score": float(coh_metrics.get("marginal_score", np.nan)),
            "coherence/joint_score": float(coh_metrics.get("joint_score", np.nan)),
            "coherence/pairwise_mean": float(coh_metrics.get("pairwise_mean", np.nan)),
            "output_delta_norm": float(((v_policy - v_ref) ** 2).mean().detach().cpu()),
            "target_velocity_norm": float((target_v ** 2).mean().detach().cpu()),
            "endpoint_steps": float(cfg.get("ram_endpoint_steps", 4)),
            "lr": current_lr,
            "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
            "train_batches": 1.0,
        }
        epoch_metrics.update(metrics)
        pbar.set_postfix_str(
            f"loss={metrics['ram_loss']:.3e} reward={metrics['reward_mean']:.3e} "
            f"adv={metrics['adv_abs_mean']:.3e}"
        )

    avg = epoch_metrics.mean()
    avg["train_batches"] = float(epoch_metrics.counts.get("ram_loss", 0))
    return avg.get("ram_loss", float("nan")), avg


@torch.no_grad()
def validate_ram(
    *,
    eval_model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    ram_coh_cfg: RAMCoherenceConfig,
    val_set: TurbulentCombustionH5Dataset,
    max_batches: int,
) -> Dict[str, float]:
    """Validate with eval EMA sampling, reconstruction L2, and RAM coherence cost."""
    eval_model.eval()
    metrics = EpochMetrics()

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(max_batches):
            break
        coords_full = batch["coords"].to(device)
        fields_full = batch["fields"].to(device)

        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords_full,
            fields_full=fields_full,
            cond_fields=cfg["cond_fields"],
            n_obs_min=cfg["n_obs_min_list"],
            n_obs_max=cfg["n_obs_max_list"],
        )

        recon = eval_model.sample(
            coords=coords_full,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            n_steps=int(cfg.get("n_steps_generation_eval", 4)),
            clamp_indices=obs_indices,
            ode_solver=str(cfg.get("ode_solver", "euler")),
            obs_consistency_mode=str(cfg.get("ram_obs_consistency_mode", "endpoint_smooth")),
            obs_consistency_strength=float(cfg.get("obs_consistency_strength", 1.0)),
            obs_consistency_sigma=float(cfg.get("obs_consistency_sigma", 0.05)),
            obs_consistency_schedule_power=float(cfg.get("obs_consistency_schedule_power", 2.0)),
            obs_consistency_final_clamp=bool(cfg.get("obs_consistency_final_clamp", True)),
        )

        rel_l2 = (
            torch.linalg.vector_norm(recon - fields_full, dim=(1, 2))
            / (torch.linalg.vector_norm(fields_full, dim=(1, 2)) + 1e-12)
        )
        cost, coh_metrics = compute_ram_coherence_cost(
            x_gen=recon,
            x_ref=fields_full,
            cfg=ram_coh_cfg,
            mean=val_set.mean.to(device),
            std=val_set.std.to(device),
        )
        metrics.update({
            "val_rel_l2": float(rel_l2.mean().cpu()),
            "val_coherence_cost": float(cost.mean().cpu()),
            "val_coherence/global_dist_score": float(coh_metrics.get("global_dist_score", np.nan)),
        })

    out = metrics.mean()
    out["val_loss"] = float(out.get("val_rel_l2", 0.0) + out.get("val_coherence_cost", 0.0))
    return out


def save_checkpoint(
    *,
    path: Path,
    eval_model: nn.Module,
    policy_model: nn.Module,
    old_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    train_set: TurbulentCombustionH5Dataset,
    source_run_dir: Path,
    source_cfg: dict,
    cfg: dict,
) -> None:
    """Save RAM checkpoint with eval EMA under the standard ``model`` key."""
    ckpt = {
        "model": eval_model.state_dict(),
        "policy_model": policy_model.state_dict(),
        "old_model": old_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "mean": train_set.mean,
        "std": train_set.std,
        "field_names": train_set.field_names,
        "method": "ram_pointcloud_ffm",
        "finetune_method": "RAM",
        "source_run_dir": str(source_run_dir),
        "source_checkpoint": cfg.get("source_checkpoint", "best"),
        "source_config": source_cfg,
        "finetune_config": cfg,
        "backbone": source_cfg.get("backbone"),
        "summary_type": source_cfg.get("summary_type"),
        "ode_solver": source_cfg.get("ode_solver", "euler"),
    }
    torch.save(ckpt, path)


def maybe_save_preview(
    *,
    eval_model: nn.Module,
    val_set: TurbulentCombustionH5Dataset,
    epoch: int,
    device: torch.device,
    run_dir: Path,
    cfg: dict,
) -> None:
    """Save a reconstruction preview when the helper can run on the dataset."""
    preview_dir = run_dir / "Evaluation" / f"epoch_{epoch:04d}"
    preview_dir.mkdir(parents=True, exist_ok=True)
    try:
        visualize_reconstruction(
            model=eval_model,
            dataset=val_set,
            epoch=epoch,
            device=device,
            save_dir=str(preview_dir),
            cond_fields=cfg["cond_fields"],
            n_obs=cfg["n_obs_max_list"],
            n_steps=int(cfg.get("n_steps_generation_eval", 4)),
            ode_solver=str(cfg.get("ode_solver", "euler")),
            snapshot_index=0,
            file_tag=f"ram_eval_nfe{int(cfg.get('n_steps_generation_eval', 4))}",
            save_metrics_json=True,
            obs_consistency_mode=str(cfg.get("ram_obs_consistency_mode", "endpoint_smooth")),
            obs_consistency_strength=float(cfg.get("obs_consistency_strength", 1.0)),
            obs_consistency_sigma=float(cfg.get("obs_consistency_sigma", 0.05)),
            obs_consistency_schedule_power=float(cfg.get("obs_consistency_schedule_power", 2.0)),
            obs_consistency_final_clamp=bool(cfg.get("obs_consistency_final_clamp", True)),
        )
    except Exception as exc:
        print(f"[Warning: !] RAM preview skipped at epoch {epoch}: {exc}")


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    demo_dir = script_dir.parent
    config_path = resolve_demo_path(demo_dir, args.config)

    cfg = load_ram_config(config_path)
    if args.Demo_Num is not None:
        cfg["Demo_Num"] = int(args.Demo_Num)
    if args.device_ids is not None:
        cfg["device_ids"] = [int(v) for v in args.device_ids]

    set_seed(int(cfg.get("seed", 42)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    source_run_dir = find_source_run_dir(
        demo_dir=demo_dir,
        source_run_dir=cfg.get("source_run_dir", None),
        source_Demo_Num=cfg.get("source_Demo_Num", None),
    )
    source_cfg = load_source_config(source_run_dir)
    cfg = inherit_conditioning_from_source(cfg, source_cfg)
    cfg["source_run_dir"] = str(source_run_dir)

    data_path = resolve_demo_path(demo_dir, cfg.get("data", source_cfg.get("data", DEFAULTS["data"])))
    run_dir = resolve_demo_path(
        demo_dir,
        f"{cfg.get('save_dir', DEFAULTS['save_dir'])}_DemoN{int(cfg['Demo_Num'])}_{timestamp}",
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "Evaluation").mkdir(parents=True, exist_ok=True)

    # Save the resolved fine-tune config and source architecture config before
    # training so interrupted runs are still inspectable.
    with open(run_dir / "args.json", "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)
    with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    with open(run_dir / "source_run_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(source_cfg, handle, sort_keys=False)

    backup_dir = demo_dir / "Save_config" / "pointcloud_ffm_ram"
    backup_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists():
        shutil.copy(config_path, backup_dir / f"config_pointcloud_ffm_ram_DemoN{cfg['Demo_Num']}_{timestamp}.yaml")

    device_ids = cfg.get("device_ids", [0])
    device = torch.device(f"cuda:{int(device_ids[0])}" if torch.cuda.is_available() else "cpu")
    print(f"[*] RAM source run : {source_run_dir}")
    print(f"[*] RAM output dir : {run_dir}")
    print(f"[*] Using device   : {device}")

    train_set = TurbulentCombustionH5Dataset(
        str(data_path),
        split="train",
        train_ratio=float(cfg.get("train_ratio", 0.9)),
        seed=int(cfg.get("seed", 42)),
        time_stride=int(cfg.get("time_stride", 1)),
        stats_path=str(run_dir / "dataset_stats.pt"),
    )
    val_set = TurbulentCombustionH5Dataset(
        str(data_path),
        split="val",
        train_ratio=float(cfg.get("train_ratio", 0.9)),
        seed=int(cfg.get("seed", 42)),
        time_stride=int(cfg.get("time_stride", 1)),
        stats_path=str(run_dir / "dataset_stats.pt"),
    )
    torch.save({"mean": train_set.mean, "std": train_set.std}, run_dir / "dataset_stats.pt")

    if source_cfg.get("backbone") == "fno":
        validate_regular_grid_compatibility(train_set, source_cfg.get("Num_x", None), source_cfg.get("Num_y", None))
        validate_regular_grid_compatibility(val_set, source_cfg.get("Num_x", None), source_cfg.get("Num_y", None))

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
    )

    ref_model, source_cfg_loaded, _ = load_pretrained_ffm(
        source_run_dir=source_run_dir,
        checkpoint=str(cfg.get("source_checkpoint", "best")),
        dataset=train_set,
        device=device,
    )
    source_cfg = source_cfg_loaded
    policy_model = clone_model(ref_model, device)
    old_model = clone_model(ref_model, device)
    eval_model = clone_model(ref_model, device)
    sync_params(ref_model, policy_model)
    sync_params(ref_model, old_model)
    sync_params(ref_model, eval_model)

    _freeze_model(ref_model)
    _freeze_model(old_model)
    _freeze_model(eval_model)
    trainable_params = set_trainable_scope(policy_model, str(cfg.get("finetune_mode", "head_glres")))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.get("lr", 2.0e-5)),
        betas=(float(cfg.get("beta1", 0.9)), float(cfg.get("beta2", 0.99))),
        weight_decay=float(cfg.get("weight_decay", 1.0e-4)),
    )
    ram_coh_cfg = build_ram_coherence_config(cfg)
    logger = RAMHistoryLogger(run_dir)

    best_val = float("inf")
    last_val_loss: Optional[float] = None
    for epoch in range(1, int(cfg.get("ram_epochs", 200)) + 1):
        train_loss, train_metrics = train_ram_epoch(
            epoch=epoch,
            policy_model=policy_model,
            ref_model=ref_model,
            old_model=old_model,
            eval_model=eval_model,
            trainable_params=trainable_params,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            cfg=cfg,
            ram_coh_cfg=ram_coh_cfg,
            train_set=train_set,
        )

        val_metrics: Dict[str, float] = {}
        if epoch == 1 or epoch % int(cfg.get("eval_every", 5)) == 0:
            val_metrics = validate_ram(
                eval_model=eval_model,
                loader=val_loader,
                device=device,
                cfg=cfg,
                ram_coh_cfg=ram_coh_cfg,
                val_set=val_set,
                max_batches=int(cfg.get("eval_num_batches", 2)),
            )
            last_val_loss = val_metrics.get("val_loss", None)
            print(
                f"[valid] epoch={epoch:04d} val_loss={last_val_loss:.6e} "
                f"rel_l2={val_metrics.get('val_rel_l2', float('nan')):.6e} "
                f"coh={val_metrics.get('val_coherence_cost', float('nan')):.6e}"
            )

        merged_metrics = dict(train_metrics)
        merged_metrics.update(val_metrics)
        merged_metrics["val_loss"] = last_val_loss
        logger.log(epoch=epoch, train_loss=train_loss, val_loss=last_val_loss, metrics=merged_metrics)

        save_checkpoint(
            path=run_dir / "last.pt",
            eval_model=eval_model,
            policy_model=policy_model,
            old_model=old_model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=last_val_loss,
            train_set=train_set,
            source_run_dir=source_run_dir,
            source_cfg=source_cfg,
            cfg=cfg,
        )
        if last_val_loss is not None and last_val_loss < best_val:
            best_val = float(last_val_loss)
            save_checkpoint(
                path=run_dir / "best.pt",
                eval_model=eval_model,
                policy_model=policy_model,
                old_model=old_model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=last_val_loss,
                train_set=train_set,
                source_run_dir=source_run_dir,
                source_cfg=source_cfg,
                cfg=cfg,
            )
            print(f"[*] Saved new RAM best.pt at epoch {epoch}")

        if epoch % int(cfg.get("save_every", 20)) == 0:
            maybe_save_preview(
                eval_model=eval_model,
                val_set=val_set,
                epoch=epoch,
                device=device,
                run_dir=run_dir,
                cfg=cfg,
            )

        print(f"[train] epoch={epoch:04d} ram_loss={train_loss:.6e}")

    if not (run_dir / "best.pt").exists():
        shutil.copy(run_dir / "last.pt", run_dir / "best.pt")
    print("[*] RAM fine-tuning complete.")
    print(f"[*] Best validation loss: {best_val:.6e}")


if __name__ == "__main__":
    main()

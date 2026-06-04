"""
RAM fine-tuning entrypoint for turbulent-combustion PointCloudFFM models.

Quick usage from ``0_demo_TurbulentCombustion/``:
    python src/train_finetune.py \
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
    6) Fit policy velocity to the detached RAM target on a configurable query
       subset, with optional endpoint/loss microbatching for memory control.
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
import matplotlib.tri as mtri
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset
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
from train_pointcloud_ffm import sample_query_subset


DEFAULTS = {
    "Demo_Num": 20,
    "seed": 42,
    "device_ids": [0],
    "source_run_dir": None,
    "source_Demo_Num": 15,
    "source_checkpoint": "last",
    "data": "Dataset/Merged_CH4COTU1P.h5",
    "train_ratio": 0.9,
    "train_ratio_downsample": 0.50,
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
    "max_train_batches": None,
    "batch_size": 2,
    "num_samples_per_condition": 4,
    "num_loss_targets_per_endpoint": 4,
    "ram_n_query_points": 4096,
    "ram_query_sampling": "obs_mix",
    "ram_query_sample_near_ratio": 0.25,
    "ram_query_sample_far_ratio": 0.25,
    "ram_query_sample_sigma_ratio": 0.05,
    "ram_endpoint_microbatch_size": 32,
    "ram_loss_microbatch_size": 64,
    "timestep_sampling": "mirrored_weighted",
    "t_eps": 1.0e-3,
    "reward_mode": "global_dist",
    "reward_multiplier": 1.0,
    "reward_scaling": "running_epoch_std",
    "reward_eps": 1.0e-4,
    "adv_clip_abs": None,
    "coherence_use_denorm": False,
    "global_lambda_marg": 1.0,
    "global_lambda_joint": 1.0,
    "global_num_directions": 64,
    "global_joint_top_frac": 0.10,
    "global_include_pairwise": False,
    "global_lambda_pairwise": 0.25,
    "global_include_pairwise_in_score": False,
    "lr": 2.0e-5,
    "weight_decay": 1.0e-4,
    "beta1": 0.9,
    "beta2": 0.99,
    "grad_clip": 1.0,
    "old_ema_decay": 0.9,
    "eval_ema_decay": 0.99,
    "finetune_mode": "head_glres",
    "val_loss_rel_l2_weight": 1.0,
    "val_loss_coherence_weight": 0.1,
    "eval_every": 5,
    "save_every": 20,
    "n_steps_generation_eval": 4,
    "eval_num_batches": 2,
    "rollout_eval_enabled": True,
    "rollout_eval_every": 20,
    "rollout_eval_split": "test",
    "rollout_eval_snapshot_index": 0,
    "rollout_eval_n_steps": 2,
    "rollout_eval_obs_consistency_mode": "endpoint_smooth",
    "rollout_eval_num_sensors": None,
    "rollout_eval_save_fields": True,
    "save_history_json": False,
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


def build_epoch_train_loader(
    train_set: TurbulentCombustionH5Dataset,
    cfg: dict,
    epoch: int,
) -> DataLoader:
    """
    Build a fresh train loader for one RAM epoch.

    `train_ratio` still defines the train/val split.  `train_ratio_downsample`
    only selects a random portion of the already-built training split each
    epoch, which shortens fine-tuning epochs without changing validation/test.
    """
    ratio = float(cfg.get("train_ratio_downsample", 1.0))
    ratio = min(max(ratio, 0.0), 1.0)
    n_total = len(train_set)
    n_epoch = n_total if ratio >= 1.0 else max(1, int(math.ceil(n_total * ratio)))

    generator = torch.Generator()
    generator.manual_seed(int(cfg.get("seed", 42)) + int(epoch) * 1009)
    if n_epoch < n_total:
        indices = torch.randperm(n_total, generator=generator)[:n_epoch].tolist()
        epoch_dataset = Subset(train_set, indices)
    else:
        epoch_dataset = train_set

    return DataLoader(
        epoch_dataset,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=True,
        generator=generator,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
    )


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


def validate_source_model_support(source_cfg: dict, cfg: dict) -> None:
    """
    Guard the first RAM implementation to the source family it was designed for.

    The RAM loss is generic, but this script's default safe adaptation scope is
    specifically the GL_rbf/topk_rbf_glres residual head path.
    """
    backbone = source_cfg.get("backbone")
    if backbone != "GL_rbf":
        raise NotImplementedError(
            "RAM fine-tuning currently supports GL_rbf PointCloudFFM sources only. "
            f"Got source backbone={backbone!r}. Use a GL_rbf source run such as DemoN15, "
            "or extend train_finetune.py before using other backbones."
        )

    finetune_mode = str(cfg.get("finetune_mode", "head_glres")).strip().lower()
    gather_mode = source_cfg.get("gather_mode")
    if finetune_mode == "head_glres" and gather_mode != "topk_rbf_glres":
        raise ValueError(
            "finetune_mode='head_glres' requires a GL_rbf source with "
            f"gather_mode='topk_rbf_glres', got gather_mode={gather_mode!r}."
        )


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
        include_pairwise=bool(cfg.get("global_include_pairwise", False)),
        joint_method=str(cfg.get("global_joint_method", "topk_swd")),
        joint_top_frac=float(cfg.get("global_joint_top_frac", 0.10)),
        joint_qmc=bool(cfg.get("global_joint_qmc", True)),
        include_axes=bool(cfg.get("global_include_axes", True)),
        lambda_pairwise=float(cfg.get("global_lambda_pairwise", 0.25)),
        include_pairwise_in_score=bool(cfg.get("global_include_pairwise_in_score", False)),
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
    Write standard history plus detailed RAM metrics to CSV/PNG.

    The compact loss_history file intentionally keeps the conventional
    train_loss/val_loss columns for compatibility with other trainers.  In RAM,
    however, these are not the same physical quantity: train_loss is the RAM
    velocity-matching MSE, while val_loss is a rollout reconstruction/coherence
    score used for model selection.
    """

    def __init__(self, run_dir: Path, save_json: bool = False):
        self.run_dir = run_dir
        self.save_json = bool(save_json)
        self.loss_csv = run_dir / "loss_history.csv"
        self.loss_json = run_dir / "loss_history.json"
        self.loss_plot = run_dir / "loss_history.png"
        self.validation_plot = run_dir / "validation_history.png"
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
            "adv_max_abs",
            "adv_clip_abs",
            "coherence/global_dist_score",
            "coherence/marginal_score",
            "coherence/joint_score",
            "coherence/pairwise_mean",
            "output_delta_norm",
            "target_velocity_norm",
            "endpoint_steps",
            "ram_n_query_points",
            "ram_endpoint_microbatch_size",
            "ram_loss_microbatch_size",
            "train_ratio_downsample",
            "train_epoch_cases",
            "lr",
            "val_rel_l2",
            "val_coherence_cost",
            "val_loss_rel_l2_weight",
            "val_loss_coherence_weight",
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
        if self.save_json:
            with open(self.loss_json, "w", encoding="utf-8") as handle:
                json.dump(self.loss_rows, handle, indent=2)

        metric_row = {"epoch": int(epoch), **metrics}
        self.metric_rows.append(metric_row)
        with open(self.metrics_csv, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([metric_row.get(key, "") for key in self.metric_header])
        if self.save_json:
            with open(self.metrics_json, "w", encoding="utf-8") as handle:
                json.dump(self.metric_rows, handle, indent=2)

        self._plot_train_objective()
        self._plot_validation_score()

    def _plot_train_objective(self) -> None:
        """Plot only the RAM objective so it is not compared to rollout scores."""
        train_points = [(row["epoch"], row["train_loss"]) for row in self.loss_rows if row["train_loss"] > 0]
        if not train_points:
            return

        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.plot(
            [p[0] for p in train_points],
            [p[1] for p in train_points],
            marker="o",
            label="RAM velocity MSE",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Velocity MSE")
        ax.set_title("RAM Training Objective")
        ax.grid(True, which="both", linestyle="--", alpha=0.45)
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.loss_plot, dpi=150)
        plt.close(fig)

    def _plot_validation_score(self) -> None:
        """Plot validation rollout score separately because it has different units."""
        val_points = [
            (row["epoch"], row["val_loss"])
            for row in self.loss_rows
            if row["val_loss"] is not None and row["val_loss"] > 0
        ]
        if not val_points:
            return

        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.plot(
            [p[0] for p in val_points],
            [p[1] for p in val_points],
            marker="s",
            color="#B23A48",
            label="Validation rollout score",
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weighted rollout score")
        ax.set_title("RAM Validation Rollout Score")
        ax.grid(True, which="both", linestyle="--", alpha=0.45)
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.validation_plot, dpi=150)
        plt.close(fig)


class RolloutHistoryLogger:
    """
    Track a fixed clean rollout through training without changing Evaluation/.
    """

    def __init__(self, run_dir: Path, field_names: Sequence[str], save_json: bool = False):
        self.run_dir = run_dir
        self.field_names = [str(name) for name in field_names]
        self.save_json = bool(save_json)
        self.csv_path = run_dir / "rollout_metrics.csv"
        self.json_path = run_dir / "rollout_metrics.json"
        self.plot_path = run_dir / "rollout_metrics.png"
        self.rows = []
        self.header = [
            "epoch",
            "split",
            "snapshot_index",
            "n_steps",
            "mean_rel_l2_phys",
            "mean_rel_l2_norm",
            "coherence_cost",
            "coherence_reward",
            *[f"rel_l2_phys/{name}" for name in self.field_names],
            *[f"rel_l2_norm/{name}" for name in self.field_names],
        ]
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(self.header)

    def log(self, row: Dict[str, float]) -> None:
        self.rows.append(row)
        with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([row.get(key, "") for key in self.header])
        if self.save_json:
            with open(self.json_path, "w", encoding="utf-8") as handle:
                json.dump(self.rows, handle, indent=2)
        self._plot()

    def _plot(self) -> None:
        if not self.rows:
            return

        epochs = [int(row["epoch"]) for row in self.rows]
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

        colors = plt.cm.tab10(np.linspace(0, 1, max(len(self.field_names), 1)))
        fidelity_values_for_range = []
        for idx, name in enumerate(self.field_names):
            key = f"rel_l2_phys/{name}"
            values = [float(row.get(key, np.nan)) for row in self.rows]
            axes[0].plot(epochs, values, marker="o", linewidth=1.8, color=colors[idx], label=name)
            fidelity_values_for_range.extend(values)
        mean_values = [float(row.get("mean_rel_l2_phys", np.nan)) for row in self.rows]
        fidelity_values_for_range.extend(mean_values)
        axes[0].plot(
            epochs,
            mean_values,
            marker="s",
            linewidth=2.4,
            color="black",
            label="mean",
        )
        # Relative L2 values can span fields with very different scales, so the
        # rollout summary uses a positive-only automatic log range for readability.
        positive_fidelity = [
            value
            for value in fidelity_values_for_range
            if np.isfinite(value) and value > 0.0
        ]
        if positive_fidelity:
            y_min = min(positive_fidelity)
            y_max = max(positive_fidelity)
            axes[0].set_yscale("log")
            axes[0].set_ylim(max(y_min * 0.75, 1e-12), max(y_max * 1.35, y_min * 1.5))
        axes[0].set_title("Rollout Field Fidelity")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Relative L2 (physical)")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend(fontsize=8, ncol=2)

        # The physical-coherence panel tracks the minimized RAM cost only; the
        # reward is simply its negative and would duplicate the same information.
        axes[1].plot(
            epochs,
            [float(row.get("coherence_cost", np.nan)) for row in self.rows],
            marker="o",
            linewidth=2.2,
            color="#B23A48",
            label="coherence cost",
        )
        axes[1].set_title("Rollout Physical Coherence")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Coherence cost")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=9)

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=180)
        plt.close(fig)


def _resolve_rollout_n_obs(cfg: dict) -> list[int]:
    value = cfg.get("rollout_eval_num_sensors", None)
    if value is None:
        value = cfg.get("vis_n_obs_list", None)
    if value is None:
        value = cfg.get("n_obs_max_list", None)
    values = _as_int_list(value)
    if not values:
        raise ValueError("Rollout evaluation needs rollout_eval_num_sensors or n_obs_max_list.")
    return values


def _rel_l2_per_field(x_pred: torch.Tensor, x_ref: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    diff = torch.linalg.vector_norm(x_pred - x_ref, dim=0)
    denom = torch.linalg.vector_norm(x_ref, dim=0).clamp_min(eps)
    return diff / denom


def _save_rollout_field_figure(
    *,
    coords_xy: np.ndarray,
    truth_phys: np.ndarray,
    recon_phys: np.ndarray,
    rel_l2_phys: np.ndarray,
    field_names: Sequence[str],
    metrics: Dict[str, float],
    save_path: Path,
) -> None:
    n_fields = len(field_names)
    triang = mtri.Triangulation(coords_xy[:, 0], coords_xy[:, 1])
    fig, axes = plt.subplots(
        n_fields,
        3,
        figsize=(13.5, max(2.4 * n_fields, 6.0)),
        squeeze=False,
        constrained_layout=True,
    )
    fig.suptitle(
        "RAM rollout "
        f"epoch {int(metrics['epoch'])} | split={metrics['split']} "
        f"snapshot={int(metrics['snapshot_index'])} | NFE={int(metrics['n_steps'])} | "
        f"mean L2={metrics['mean_rel_l2_phys']:.3e} | "
        f"cost={metrics['coherence_cost']:.3e}",
        fontsize=12,
    )

    for c, name in enumerate(field_names):
        true_f = truth_phys[:, c]
        pred_f = recon_phys[:, c]
        err_f = np.abs(pred_f - true_f)
        vmin = float(np.nanmin([true_f.min(), pred_f.min()]))
        vmax = float(np.nanmax([true_f.max(), pred_f.max()]))

        panels = [
            (true_f, "truth", "coolwarm", vmin, vmax),
            (pred_f, "rollout", "coolwarm", vmin, vmax),
            (err_f, f"|error|  L2={rel_l2_phys[c]:.2e}", "inferno", 0.0, float(err_f.max() + 1e-12)),
        ]
        for j, (values, title, cmap, lo, hi) in enumerate(panels):
            ax = axes[c, j]
            image = ax.tricontourf(triang, values, levels=64, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{name} {title}", fontsize=9)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)

    fig.savefig(save_path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def run_rollout_evaluation(
    *,
    model: nn.Module,
    dataset: TurbulentCombustionH5Dataset,
    epoch: int,
    device: torch.device,
    run_dir: Path,
    cfg: dict,
    ram_coh_cfg: RAMCoherenceConfig,
    logger: RolloutHistoryLogger,
) -> Dict[str, float]:
    """
    Roll out one fixed clean test sample and log fidelity/coherence over epochs.
    """
    model.eval()
    snapshot_index = int(cfg.get("rollout_eval_snapshot_index", 0))
    n_steps = int(cfg.get("rollout_eval_n_steps", 2))
    sample = dataset[snapshot_index]

    coords = sample["coords"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"].cpu().numpy()
    n_obs = _resolve_rollout_n_obs(cfg)

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cfg["cond_fields"],
        n_obs_min=n_obs,
        n_obs_max=n_obs,
    )

    recon = model.sample(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_steps=n_steps,
        clamp_indices=obs_indices,
        ode_solver=str(cfg.get("ode_solver", "euler")),
        obs_consistency_mode=str(cfg.get("rollout_eval_obs_consistency_mode", "endpoint_smooth")),
        obs_consistency_strength=float(cfg.get("obs_consistency_strength", 1.0)),
        obs_consistency_sigma=float(cfg.get("obs_consistency_sigma", 0.05)),
        obs_consistency_schedule_power=float(cfg.get("obs_consistency_schedule_power", 2.0)),
        obs_consistency_final_clamp=bool(cfg.get("obs_consistency_final_clamp", True)),
    )

    rel_l2_norm = _rel_l2_per_field(recon[0], truth[0])
    mean = dataset.mean.to(device).view(1, 1, -1)
    std = dataset.std.to(device).view(1, 1, -1)
    recon_phys_t = recon * std + mean
    truth_phys_t = truth * std + mean
    rel_l2_phys = _rel_l2_per_field(recon_phys_t[0], truth_phys_t[0])

    cost, _ = compute_ram_coherence_cost(
        x_gen=recon,
        x_ref=truth,
        cfg=ram_coh_cfg,
        mean=dataset.mean.to(device),
        std=dataset.std.to(device),
    )
    cost_value = float(cost.mean().cpu())
    field_names = list(getattr(dataset, "field_names", [f"field_{i}" for i in range(truth.shape[-1])]))

    row: Dict[str, float] = {
        "epoch": int(epoch),
        "split": str(cfg.get("rollout_eval_split", "test")),
        "snapshot_index": snapshot_index,
        "n_steps": n_steps,
        "mean_rel_l2_phys": float(rel_l2_phys.mean().cpu()),
        "mean_rel_l2_norm": float(rel_l2_norm.mean().cpu()),
        "coherence_cost": cost_value,
        "coherence_reward": -cost_value,
    }
    for c, name in enumerate(field_names):
        row[f"rel_l2_phys/{name}"] = float(rel_l2_phys[c].cpu())
        row[f"rel_l2_norm/{name}"] = float(rel_l2_norm[c].cpu())

    rollout_dir = run_dir / "Rollout" / f"epoch_{int(epoch):04d}"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.get("save_history_json", False)):
        with open(rollout_dir / "rollout_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2)

    if bool(cfg.get("rollout_eval_save_fields", True)):
        _save_rollout_field_figure(
            coords_xy=coords_raw[:, :2],
            truth_phys=truth_phys_t[0].cpu().numpy(),
            recon_phys=recon_phys_t[0].cpu().numpy(),
            rel_l2_phys=rel_l2_phys.cpu().numpy(),
            field_names=field_names,
            metrics=row,
            save_path=rollout_dir / "rollout_fields.png",
        )

    logger.log(row)
    print(
        f"[rollout] epoch={epoch:04d} mean_l2={row['mean_rel_l2_phys']:.6e} "
        f"coh_cost={row['coherence_cost']:.6e}"
    )
    return row


def _repeat_batch(x: torch.Tensor, repeats: int) -> torch.Tensor:
    return x.repeat_interleave(int(repeats), dim=0)


def _microbatch_slices(n_items: int, microbatch_size: Optional[int]) -> list[slice]:
    """
    Build batch-axis slices for RAM memory throttling.

    A value of None or <=0 keeps the original all-at-once behavior, which is
    useful for tiny smoke tests but not for full-grid endpoint sampling.
    """
    if microbatch_size is None or int(microbatch_size) <= 0 or int(microbatch_size) >= n_items:
        return [slice(0, n_items)]
    step = int(microbatch_size)
    return [slice(start, min(start + step, n_items)) for start in range(0, n_items, step)]


@torch.no_grad()
def _sample_endpoints_microbatched(
    *,
    model: nn.Module,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: torch.Tensor,
    cfg: dict,
) -> torch.Tensor:
    """
    Sample full-grid endpoints in smaller condition batches.

    Rewards are still computed on complete physical fields, but the ODE rollout
    no longer has to materialize every repeated condition inside one model call.
    """
    chunks = []
    microbatch_size = cfg.get("ram_endpoint_microbatch_size", None)
    for sl in _microbatch_slices(coords.shape[0], microbatch_size):
        chunks.append(
            model.sample(
                coords=coords[sl],
                obs_coords=obs_coords[sl],
                obs_values=obs_values[sl],
                obs_mask=obs_mask[sl],
                obs_field_ids=obs_field_ids[sl],
                n_steps=int(cfg["ram_endpoint_steps"]),
                clamp_indices=obs_indices[sl],
                ode_solver=str(cfg.get("ode_solver", "euler")),
                obs_consistency_mode=str(cfg.get("ram_obs_consistency_mode", "endpoint_smooth")),
                obs_consistency_strength=float(cfg.get("obs_consistency_strength", 1.0)),
                obs_consistency_sigma=float(cfg.get("obs_consistency_sigma", 0.05)),
                obs_consistency_schedule_power=float(cfg.get("obs_consistency_schedule_power", 2.0)),
                obs_consistency_final_clamp=bool(cfg.get("obs_consistency_final_clamp", True)),
            )
        )
    return torch.cat(chunks, dim=0)


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
    adv_clip_abs = cfg.get("adv_clip_abs", None)
    adv_clip_abs_value = None if adv_clip_abs is None else float(adv_clip_abs)
    max_train_batches = cfg.get("max_train_batches", None)
    max_train_batches = None if max_train_batches is None else int(max_train_batches)
    ram_n_query_points = cfg.get("ram_n_query_points", 4096)
    ram_n_query_points = None if ram_n_query_points is None else int(ram_n_query_points)
    ram_loss_microbatch_size = cfg.get("ram_loss_microbatch_size", None)
    ram_loss_microbatch_size = None if ram_loss_microbatch_size is None else int(ram_loss_microbatch_size)
    running_reward_std = RunningRewardStd()
    epoch_metrics = EpochMetrics()

    pbar = tqdm(loader, desc=f"RAM epoch {epoch:04d}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_train_batches is not None and batch_idx >= max_train_batches:
            break

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
            x_end = _sample_endpoints_microbatched(
                model=old_model,
                coords=coords_g,
                obs_coords=obs_coords_g,
                obs_values=obs_values_g,
                obs_mask=obs_mask_g,
                obs_field_ids=obs_field_ids_g,
                obs_indices=obs_indices_g,
                cfg=cfg,
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
            if adv_clip_abs_value is not None:
                adv = adv.clamp(-adv_clip_abs_value, adv_clip_abs_value)
            adv_flat = adv.reshape(-1)

        # RAM scores full-grid endpoints above, then performs velocity matching
        # on a query subset, mirroring base FFM's n_query_points memory control.
        coords_q, x_end_q, _ = sample_query_subset(
            coords=coords_g,
            fields=x_end.detach(),
            n_query=ram_n_query_points,
            mode=str(cfg.get("ram_query_sampling", "obs_mix")),
            obs_coords=obs_coords_g,
            obs_mask=obs_mask_g,
            near_ratio=float(cfg.get("ram_query_sample_near_ratio", 0.25)),
            far_ratio=float(cfg.get("ram_query_sample_far_ratio", 0.25)),
            sigma_ratio=float(cfg.get("ram_query_sample_sigma_ratio", 0.05)),
        )

        # Analytically re-noise each sampled endpoint K times.  This supervised
        # RAM loss batch is independent of endpoint sampling steps and can be
        # split into microbatches for gradient accumulation.
        x_end_l = _repeat_batch(x_end_q, K)
        coords_l = _repeat_batch(coords_q, K)
        obs_coords_l = _repeat_batch(obs_coords_g, K)
        obs_values_l = _repeat_batch(obs_values_g, K)
        obs_mask_l = _repeat_batch(obs_mask_g, K)
        obs_field_ids_l = _repeat_batch(obs_field_ids_g, K)
        adv_l = _repeat_batch(adv_flat.detach(), K)

        optimizer.zero_grad(set_to_none=True)

        loss_total = 0.0
        output_delta_total = 0.0
        target_velocity_total = 0.0
        loss_weight_total = 0
        n_loss_items = x_end_l.shape[0]
        for sl in _microbatch_slices(n_loss_items, ram_loss_microbatch_size):
            coords_mb = coords_l[sl]
            x_end_mb = x_end_l[sl]
            obs_coords_mb = obs_coords_l[sl]
            obs_values_mb = obs_values_l[sl]
            obs_mask_mb = obs_mask_l[sl]
            obs_field_ids_mb = obs_field_ids_l[sl]
            adv_mb = adv_l[sl]

            z = policy_model.sample_source(coords_mb)
            t = sample_ram_times(
                n=x_end_mb.shape[0],
                device=device,
                dtype=x_end_mb.dtype,
                eps=float(cfg.get("t_eps", 1.0e-3)),
                mode=str(cfg.get("timestep_sampling", "mirrored_weighted")),
            )
            t_view = t.view(-1, 1, 1)
            x_t = (1.0 - t_view) * z + t_view * x_end_mb
            bridge_direction = x_end_mb - z

            v_policy = policy_model.model(
                t, x_t, coords_mb, obs_coords_mb, obs_values_mb, obs_mask_mb, obs_field_ids_mb
            )
            with torch.no_grad():
                v_ref = ref_model.model(
                    t, x_t, coords_mb, obs_coords_mb, obs_values_mb, obs_mask_mb, obs_field_ids_mb
                )
                v_old = old_model.model(
                    t, x_t, coords_mb, obs_coords_mb, obs_values_mb, obs_mask_mb, obs_field_ids_mb
                )

            scaled_adv = float(cfg.get("reward_multiplier", 1.0)) * adv_mb.view(-1, 1, 1)
            target_v = v_ref + scaled_adv * (bridge_direction - v_old)
            loss_mb = ((v_policy - target_v.detach()) ** 2).mean()

            weight = x_end_mb.shape[0]
            weighted_loss = loss_mb * (float(weight) / float(n_loss_items))
            weighted_loss.backward()

            loss_total += float(loss_mb.detach().cpu()) * weight
            output_delta_total += float(((v_policy.detach() - v_ref) ** 2).mean().cpu()) * weight
            target_velocity_total += float((target_v.detach() ** 2).mean().cpu()) * weight
            loss_weight_total += weight

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.get("grad_clip", 1.0)))
        optimizer.step()

        # Step-level EMA keeps the old/eval policies close enough for short RAM
        # runs while still providing lagged targets.
        update_ema_params(policy_model, old_model, float(cfg.get("old_ema_decay", 0.9)))
        update_ema_params(policy_model, eval_model, float(cfg.get("eval_ema_decay", 0.99)))

        current_lr = float(optimizer.param_groups[0]["lr"])
        loss_value = loss_total / max(loss_weight_total, 1)
        output_delta_value = output_delta_total / max(loss_weight_total, 1)
        target_velocity_value = target_velocity_total / max(loss_weight_total, 1)
        metrics = {
            "ram_loss": loss_value,
            "reward_mean": float(rewards.mean().detach().cpu()),
            "reward_std": float(rewards.std(unbiased=False).detach().cpu()),
            "adv_abs_mean": float(adv.abs().mean().detach().cpu()),
            "adv_max_abs": float(adv.abs().max().detach().cpu()),
            "adv_clip_abs": np.nan if adv_clip_abs_value is None else adv_clip_abs_value,
            "coherence/global_dist_score": float(coh_metrics.get("global_dist_score", np.nan)),
            "coherence/marginal_score": float(coh_metrics.get("marginal_score", np.nan)),
            "coherence/joint_score": float(coh_metrics.get("joint_score", np.nan)),
            "coherence/pairwise_mean": float(coh_metrics.get("pairwise_mean", np.nan)),
            "output_delta_norm": output_delta_value,
            "target_velocity_norm": target_velocity_value,
            "endpoint_steps": float(cfg.get("ram_endpoint_steps", 4)),
            "ram_n_query_points": float(coords_q.shape[1]),
            "ram_endpoint_microbatch_size": float(
                cfg.get("ram_endpoint_microbatch_size") or coords_g.shape[0]
            ),
            "ram_loss_microbatch_size": float(ram_loss_microbatch_size or n_loss_items),
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
    avg["train_ratio_downsample"] = float(cfg.get("train_ratio_downsample", 1.0))
    avg["train_epoch_cases"] = float(len(loader.dataset))
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
    rel_l2_weight = float(cfg.get("val_loss_rel_l2_weight", 1.0))
    coherence_weight = float(cfg.get("val_loss_coherence_weight", 0.1))
    out["val_loss_rel_l2_weight"] = rel_l2_weight
    out["val_loss_coherence_weight"] = coherence_weight
    out["val_loss"] = float(
        rel_l2_weight * out.get("val_rel_l2", 0.0)
        + coherence_weight * out.get("val_coherence_cost", 0.0)
    )
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
    validate_source_model_support(source_cfg, cfg)
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
    print(
        "[*] RAM memory profile: "
        f"B={int(cfg.get('batch_size', 1))}, "
        f"G={int(cfg.get('num_samples_per_condition', 1))}, "
        f"K={int(cfg.get('num_loss_targets_per_endpoint', 1))}, "
        f"train_ratio_downsample={float(cfg.get('train_ratio_downsample', 1.0)):.3f}, "
        f"n_query={cfg.get('ram_n_query_points')}, "
        f"endpoint_mb={cfg.get('ram_endpoint_microbatch_size')}, "
        f"loss_mb={cfg.get('ram_loss_microbatch_size')}"
    )

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

    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg.get("batch_size", 2)),
        shuffle=False,
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
    )
    rollout_set = None
    if bool(cfg.get("rollout_eval_enabled", True)):
        rollout_set = TurbulentCombustionH5Dataset(
            str(data_path),
            split=str(cfg.get("rollout_eval_split", "test")),
            train_ratio=float(cfg.get("train_ratio", 0.9)),
            seed=int(cfg.get("seed", 42)),
            time_stride=int(cfg.get("time_stride", 1)),
            stats_path=str(run_dir / "dataset_stats.pt"),
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
    save_history_json = bool(cfg.get("save_history_json", False))
    logger = RAMHistoryLogger(run_dir, save_json=save_history_json)
    rollout_logger = None
    if rollout_set is not None:
        rollout_logger = RolloutHistoryLogger(
            run_dir,
            getattr(rollout_set, "field_names", []),
            save_json=save_history_json,
        )
        run_rollout_evaluation(
            model=eval_model,
            dataset=rollout_set,
            epoch=0,
            device=device,
            run_dir=run_dir,
            cfg=cfg,
            ram_coh_cfg=ram_coh_cfg,
            logger=rollout_logger,
        )

    best_val = float("inf")
    last_val_loss: Optional[float] = None
    for epoch in range(1, int(cfg.get("ram_epochs", 200)) + 1):
        train_loader = build_epoch_train_loader(train_set, cfg, epoch)
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
                f"[valid] epoch={epoch:04d} val_score={last_val_loss:.6e} "
                f"rel_l2={val_metrics.get('val_rel_l2', float('nan')):.6e} "
                f"coh={val_metrics.get('val_coherence_cost', float('nan')):.6e} "
                f"(score={float(cfg.get('val_loss_rel_l2_weight', 1.0)):.3g}*rel_l2+"
                f"{float(cfg.get('val_loss_coherence_weight', 0.1)):.3g}*coh)"
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
        if (
            rollout_set is not None
            and rollout_logger is not None
            and int(cfg.get("rollout_eval_every", 20)) > 0
            and epoch % int(cfg.get("rollout_eval_every", 20)) == 0
        ):
            run_rollout_evaluation(
                model=eval_model,
                dataset=rollout_set,
                epoch=epoch,
                device=device,
                run_dir=run_dir,
                cfg=cfg,
                ram_coh_cfg=ram_coh_cfg,
                logger=rollout_logger,
            )

        print(f"[train] epoch={epoch:04d} ram_loss={train_loss:.6e}")

    if not (run_dir / "best.pt").exists():
        shutil.copy(run_dir / "last.pt", run_dir / "best.pt")
    print("[*] RAM fine-tuning complete.")
    print(f"[*] Best validation rollout score: {best_val:.6e}")


if __name__ == "__main__":
    main()

"""
RAM fine-tuning entrypoint for turbulent-combustion PointCloudFFM models.

Quick usage from ``0_demo_TurbulentCombustion/``:
    python src/train_finetune.py \
      --config Save_config/config_pointcloud_ffm_ram.yaml \
      --Demo-Num 20

General structure:
    1) Load a pretrained PointCloudFFM source run.
    2) Create RAM roles. Full-copy mode uses separate models:
       ref_model    - frozen pretrained base velocity
       policy_model - trainable RAM policy
       old_model    - lagged EMA policy used for endpoint sampling/targets
       eval_model   - smoother EMA policy saved in checkpoints
       LoRA mode keeps one model with default/old/evaluation adapters.
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
from lora_finetune import (
    collect_lora_state,
    ema_lora_adapter,
    export_merged_lora_state_dict,
    inject_lora_adapters,
    lora_trainable_params,
    sync_lora_adapter,
    use_adapter,
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
    "reward_std_ema_decay": 0.95,
    "reward_eps": 1.0e-4,
    "adv_clip_abs": 3.0,
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
    "beta2": 0.95,
    "grad_clip": 1.0,
    "old_ema_decay": 0.9,
    "eval_ema_decay": 0.99,
    "finetune_mode": "head_glres",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_target_scope": None,
    "ram_reward_n_points": None,
    "ram_reward_sampling": "uniform",
    "reward_subsample_seed": 1234,
    "fixed_reward_points_for_eval": True,
    "fixed_reward_points_for_rollout": True,
    "rollout_reward_point_path": None,
    "use_eval_ema": True,
    "eval_ema_device": "gpu",
    "val_loss_rel_l2_weight": 0.1,
    "val_loss_coherence_weight": 1.0,
    "eval_every": 5,
    "save_every": 20,
    "n_steps_generation_eval": 2,
    "eval_num_batches": 2,
    "rollout_eval_enabled": True,
    "rollout_eval_every": 20,
    "rollout_eval_split": "test",
    "rollout_eval_snapshot_index": [0, 10, 100],
    "rollout_eval_n_steps": 2,
    "rollout_eval_obs_consistency_mode": "endpoint_smooth",
    "align_ram_and_rollout_obs_consistency": True,
    "rollout_eval_num_sensors": None,
    "rollout_eval_save_fields": True,
    "rollout_eval_fixed_condition": True,
    "rollout_eval_condition_path": None,
    "rollout_eval_policy_model": True,
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
    if bool(cfg.get("align_ram_and_rollout_obs_consistency", True)):
        cfg["ram_obs_consistency_mode"] = cfg.get(
            "rollout_eval_obs_consistency_mode",
            cfg.get("ram_obs_consistency_mode", "endpoint_smooth"),
        )
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
    supported_modes = {"head_glres", "all", "lora_head_glres", "lora_all_linear_glrbf"}
    if finetune_mode not in supported_modes:
        raise ValueError(f"finetune_mode must be one of {sorted(supported_modes)}, got {finetune_mode!r}.")

    if finetune_mode.startswith("lora_"):
        _resolve_lora_scope(cfg, finetune_mode)

    if finetune_mode in ("head_glres", "lora_head_glres") and gather_mode != "topk_rbf_glres":
        raise ValueError(
            f"finetune_mode={finetune_mode!r} requires a GL_rbf source with "
            f"gather_mode='topk_rbf_glres', got gather_mode={gather_mode!r}."
        )


def _resolve_lora_scope(cfg: dict, finetune_mode: str) -> str:
    """
    Derive the LoRA injection scope from ``finetune_mode``.

    ``lora_target_scope`` is kept only as a backward-compatible sanity check.
    A mismatch is almost certainly a configuration error because it changes
    which layers are trainable while the mode name says something different.
    """
    implied_by_mode = {
        "lora_head_glres": "head_glres",
        "lora_all_linear_glrbf": "all_linear_glrbf",
    }
    if finetune_mode not in implied_by_mode:
        raise ValueError(f"finetune_mode={finetune_mode!r} is not a LoRA mode.")

    implied_scope = implied_by_mode[finetune_mode]
    raw_scope = cfg.get("lora_target_scope", None)
    if raw_scope is None:
        return implied_scope

    configured_scope = str(raw_scope).strip().lower()
    if configured_scope in ("", "auto"):
        return implied_scope

    valid_scopes = set(implied_by_mode.values())
    if configured_scope not in valid_scopes:
        raise ValueError(
            f"lora_target_scope must be one of {sorted(valid_scopes)} or null/auto, "
            f"got {configured_scope!r}."
        )
    if configured_scope != implied_scope:
        raise ValueError(
            f"Conflicting LoRA config: finetune_mode={finetune_mode!r} implies "
            f"lora_target_scope={implied_scope!r}, but lora_target_scope is "
            f"{configured_scope!r}. Set lora_target_scope to null/auto, or choose "
            "the matching finetune_mode."
        )
    return implied_scope


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


class EMARewardStd:
    """Persistent reward-scale tracker shared across RAM epochs."""

    def __init__(self, decay: float = 0.95):
        self.decay = float(decay)
        self.value: Optional[float] = None

    def update(self, batch_std: float) -> float:
        batch_std = float(max(batch_std, 0.0))
        if self.value is None:
            self.value = batch_std
        else:
            self.value = self.decay * self.value + (1.0 - self.decay) * batch_std
        return float(self.value)


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
            "scaled_adv_abs_mean",
            "scaled_adv_abs_max",
            "reward_scale_std",
            "reward_mode",
            "coherence/ram_cost",
            "coherence/global_dist_score",
            "coherence/marginal_score",
            "coherence/joint_score",
            "coherence/pairwise_mean",
            "coherence/field_l2_rel",
            "coherence/field_l2_mse",
            "output_delta_norm",
            "policy_ref_delta_norm",
            "target_delta_norm",
            "bridge_residual_norm",
            "target_velocity_norm",
            "endpoint_steps",
            "ram_n_query_points",
            "reward_point_count",
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

    def __init__(
        self,
        run_dir: Path,
        field_names: Sequence[str],
        save_json: bool = False,
        model_role: str = "eval",
    ):
        self.run_dir = run_dir
        self.field_names = [str(name) for name in field_names]
        self.save_json = bool(save_json)
        self.model_role = str(model_role)
        suffix = "" if self.model_role == "eval" else f"_{self.model_role}"
        self.csv_path = run_dir / f"rollout_metrics{suffix}.csv"
        self.json_path = run_dir / f"rollout_metrics{suffix}.json"
        self.plot_path = run_dir / f"rollout_metrics{suffix}.png"
        self.rows = []
        self.header = [
            "epoch",
            "model_role",
            "split",
            "snapshot_index",
            "n_steps",
            "mean_rel_l2_phys",
            "mean_rel_l2_norm",
            "coherence_cost",
            "coherence_reward",
            "reward_point_count",
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


def _resolve_rollout_snapshot_indices(cfg: dict) -> list[int]:
    value = cfg.get("rollout_eval_snapshot_index", 0)
    if isinstance(value, (int, np.integer)):
        indices = [int(value)]
    elif isinstance(value, (list, tuple)):
        indices = [int(v) for v in value]
    else:
        raise ValueError(
            "rollout_eval_snapshot_index must be an integer or a list of integers, "
            f"got {value!r}."
        )
    if not indices:
        raise ValueError("rollout_eval_snapshot_index must not be an empty list.")
    return indices


def _rollout_condition_path(
    run_dir: Path,
    cfg: dict,
    snapshot_index: Optional[int] = None,
    multi_snapshot: bool = False,
) -> Path:
    """Resolve the fixed rollout condition cache path."""
    configured = cfg.get("rollout_eval_condition_path", None)
    if configured:
        if multi_snapshot:
            raise ValueError(
                "rollout_eval_condition_path cannot be used with multiple "
                "rollout_eval_snapshot_index values because it would reuse one "
                "fixed sparse condition cache across different snapshots."
            )
        path = Path(str(configured))
        return path if path.is_absolute() else run_dir / path
    if multi_snapshot and snapshot_index is not None:
        return run_dir / "Rollout" / f"fixed_rollout_condition_snapshot_{int(snapshot_index):06d}.pt"
    return run_dir / "Rollout" / "fixed_rollout_condition.pt"


def _rollout_reward_point_path(run_dir: Path, cfg: dict) -> Path:
    """Resolve the fixed rollout reward-point cache path."""
    configured = cfg.get("rollout_reward_point_path", None)
    if configured:
        path = Path(str(configured))
        return path if path.is_absolute() else run_dir / path
    return run_dir / "Rollout" / "fixed_reward_point_indices.pt"


def _load_rollout_condition_file(path: Path) -> dict:
    """
    Load fixed rollout-condition caches with PyTorch's safe tensor-only mode.

    The cache is produced by this script and contains only tensors plus simple
    scalar/list metadata, so weights_only=True avoids pickle warnings without
    losing any needed information.  The fallback keeps older torch versions
    usable if they do not yet expose the keyword.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_or_build_rollout_condition(
    *,
    coords: torch.Tensor,
    truth: torch.Tensor,
    run_dir: Path,
    cfg: dict,
    device: torch.device,
    snapshot_index: int,
    multi_snapshot: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, list[int]]:
    """
    Reuse the exact same sparse rollout condition across epochs.

    This removes sensor-sampling randomness from rollout monitoring so trends in
    L2/coherence reflect model changes, not changing sparse observations.
    """
    n_obs = _resolve_rollout_n_obs(cfg)
    snapshot_index = int(snapshot_index)
    cache_path = _rollout_condition_path(
        run_dir,
        cfg,
        snapshot_index=snapshot_index,
        multi_snapshot=multi_snapshot,
    )
    use_fixed = bool(cfg.get("rollout_eval_fixed_condition", True))

    if use_fixed and cache_path.exists():
        payload = _load_rollout_condition_file(cache_path)
        obs_coords = payload["obs_coords"].to(device)
        obs_values = payload["obs_values"].to(device)
        obs_mask = payload["obs_mask"].to(device)
        obs_indices = payload["obs_indices"].to(device)
        obs_field_ids = payload["obs_field_ids"].to(device)
        snapshot_index = int(payload.get("snapshot_index", snapshot_index))
        stored_n_obs = [int(v) for v in payload.get("n_obs", n_obs)]
        return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids, snapshot_index, stored_n_obs

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cfg["cond_fields"],
        n_obs_min=n_obs,
        n_obs_max=n_obs,
    )

    if use_fixed:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "obs_coords": obs_coords.detach().cpu(),
                "obs_values": obs_values.detach().cpu(),
                "obs_mask": obs_mask.detach().cpu(),
                "obs_indices": obs_indices.detach().cpu(),
                "obs_field_ids": obs_field_ids.detach().cpu(),
                "snapshot_index": snapshot_index,
                "split": str(cfg.get("rollout_eval_split", "test")),
                "cond_fields": list(cfg["cond_fields"]),
                "n_obs": list(n_obs),
            },
            cache_path,
        )

    return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids, snapshot_index, n_obs


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
        f"epoch {int(metrics['epoch'])} | role={metrics.get('model_role', 'eval')} | split={metrics['split']} "
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
    model_role: str = "eval",
    snapshot_index: Optional[int] = None,
    multi_snapshot: bool = False,
    log_row: bool = True,
    print_row: bool = True,
) -> Dict[str, float]:
    """
    Roll out one fixed clean test sample and log fidelity/coherence over epochs.
    """
    model.eval()
    if snapshot_index is None:
        snapshot_index = _resolve_rollout_snapshot_indices(cfg)[0]
    snapshot_index = int(snapshot_index)
    condition_path = _rollout_condition_path(
        run_dir,
        cfg,
        snapshot_index=snapshot_index,
        multi_snapshot=multi_snapshot,
    )
    if bool(cfg.get("rollout_eval_fixed_condition", True)) and condition_path.exists():
        condition_meta = _load_rollout_condition_file(condition_path)
        snapshot_index = int(condition_meta.get("snapshot_index", snapshot_index))
    n_steps = int(cfg.get("rollout_eval_n_steps", 2))
    sample = dataset[snapshot_index]

    coords = sample["coords"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"].cpu().numpy()
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids, snapshot_index, n_obs = _load_or_build_rollout_condition(
        coords=coords,
        truth=truth,
        run_dir=run_dir,
        cfg=cfg,
        device=device,
        snapshot_index=snapshot_index,
        multi_snapshot=multi_snapshot,
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

    recon_reward, truth_reward, reward_point_count = subsample_reward_points(
        x_gen=recon,
        x_ref=truth,
        n_points=cfg.get("ram_reward_n_points", None),
        mode=str(cfg.get("ram_reward_sampling", "uniform")),
        seed=int(cfg.get("reward_subsample_seed", 1234)),
        cache_path=(
            _rollout_reward_point_path(run_dir, cfg)
            if bool(cfg.get("fixed_reward_points_for_rollout", True))
            else None
        ),
        fixed=bool(cfg.get("fixed_reward_points_for_rollout", True)),
    )
    cost, _ = compute_ram_coherence_cost(
        x_gen=recon_reward,
        x_ref=truth_reward,
        cfg=ram_coh_cfg,
        mean=dataset.mean.to(device),
        std=dataset.std.to(device),
    )
    cost_value = float(cost.mean().cpu())
    field_names = list(getattr(dataset, "field_names", [f"field_{i}" for i in range(truth.shape[-1])]))

    row: Dict[str, float] = {
        "epoch": int(epoch),
        "model_role": str(model_role),
        "split": str(cfg.get("rollout_eval_split", "test")),
        "snapshot_index": snapshot_index,
        "n_steps": n_steps,
        "mean_rel_l2_phys": float(rel_l2_phys.mean().cpu()),
        "mean_rel_l2_norm": float(rel_l2_norm.mean().cpu()),
        "coherence_cost": cost_value,
        "coherence_reward": -cost_value,
        "reward_point_count": float(reward_point_count),
    }
    for c, name in enumerate(field_names):
        row[f"rel_l2_phys/{name}"] = float(rel_l2_phys[c].cpu())
        row[f"rel_l2_norm/{name}"] = float(rel_l2_norm[c].cpu())

    if str(model_role) == "eval":
        rollout_dir = run_dir / "Rollout" / f"epoch_{int(epoch):04d}"
    else:
        rollout_dir = run_dir / "Rollout" / str(model_role) / f"epoch_{int(epoch):04d}"
    if multi_snapshot:
        rollout_dir = rollout_dir / f"snapshot_{int(snapshot_index):06d}"
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

    if log_row:
        logger.log(row)
    if print_row:
        print(
            f"[rollout] epoch={epoch:04d} mean_l2={row['mean_rel_l2_phys']:.6e} "
            f"coh_cost={row['coherence_cost']:.6e} role={model_role}"
        )
    return row


@torch.no_grad()
def run_rollout_evaluations(
    *,
    model: nn.Module,
    dataset: TurbulentCombustionH5Dataset,
    epoch: int,
    device: torch.device,
    run_dir: Path,
    cfg: dict,
    ram_coh_cfg: RAMCoherenceConfig,
    logger: RolloutHistoryLogger,
    model_role: str = "eval",
) -> Dict[str, float]:
    snapshot_indices = _resolve_rollout_snapshot_indices(cfg)
    invalid_indices = [idx for idx in snapshot_indices if idx < 0 or idx >= len(dataset)]
    if invalid_indices:
        raise ValueError(
            f"rollout_eval_snapshot_index contains indices outside the "
            f"rollout {cfg.get('rollout_eval_split', 'test')!r} split of size "
            f"{len(dataset)}: {invalid_indices}"
        )
    if len(snapshot_indices) == 1:
        return run_rollout_evaluation(
            model=model,
            dataset=dataset,
            epoch=epoch,
            device=device,
            run_dir=run_dir,
            cfg=cfg,
            ram_coh_cfg=ram_coh_cfg,
            logger=logger,
            model_role=model_role,
            snapshot_index=snapshot_indices[0],
            multi_snapshot=False,
        )

    if cfg.get("rollout_eval_condition_path", None):
        _rollout_condition_path(
            run_dir,
            cfg,
            snapshot_index=snapshot_indices[0],
            multi_snapshot=True,
        )

    rows = [
        run_rollout_evaluation(
            model=model,
            dataset=dataset,
            epoch=epoch,
            device=device,
            run_dir=run_dir,
            cfg=cfg,
            ram_coh_cfg=ram_coh_cfg,
            logger=logger,
            model_role=model_role,
            snapshot_index=snapshot_index,
            multi_snapshot=True,
            log_row=False,
            print_row=False,
        )
        for snapshot_index in snapshot_indices
    ]

    avg_row: Dict[str, float] = {
        "epoch": int(epoch),
        "model_role": str(model_role),
        "split": str(cfg.get("rollout_eval_split", "test")),
        "snapshot_index": ",".join(str(v) for v in snapshot_indices),
        "n_steps": int(cfg.get("rollout_eval_n_steps", 2)),
    }
    metadata_keys = {"epoch", "model_role", "split", "snapshot_index", "n_steps"}
    metric_keys = sorted(set().union(*(row.keys() for row in rows)) - metadata_keys)
    for key in metric_keys:
        values = []
        for row in rows:
            value = row.get(key, None)
            if isinstance(value, (int, float, np.integer, np.floating)):
                values.append(float(value))
        if values:
            avg_row[key] = float(np.mean(values))

    logger.log(avg_row)
    compact_indices = ",".join(str(v) for v in snapshot_indices)
    print(
        f"[rollout] epoch={epoch:04d} snapshots=[{compact_indices}] "
        f"mean_l2={avg_row.get('mean_rel_l2_phys', float('nan')):.6e} "
        f"coh_cost={avg_row.get('coherence_cost', float('nan')):.6e} "
        f"role={model_role}"
    )
    return avg_row


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


def _load_reward_point_indices(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def get_or_build_reward_point_indices(
    n_total: int,
    n_points: Optional[int],
    device,
    seed: int,
    cache_path: Optional[Path] = None,
    fixed: bool = False,
) -> Optional[torch.Tensor]:
    """
    Build reward/coherence point indices, optionally persisted for fixed evals.
    """
    n_total = int(n_total)
    if n_points is None:
        return None
    n_select = int(n_points)
    if n_select <= 0 or n_select >= n_total:
        return None

    if fixed and cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            idx = _load_reward_point_indices(cache_path).to(dtype=torch.long)
            if idx.ndim != 1 or int(idx.numel()) != n_select:
                raise ValueError(
                    f"Cached reward indices at {cache_path} have shape {tuple(idx.shape)}, "
                    f"expected ({n_select},)."
                )
            if int(idx.min().item()) < 0 or int(idx.max().item()) >= n_total:
                raise ValueError(
                    f"Cached reward indices at {cache_path} are incompatible with n_total={n_total}."
                )
            return idx.to(device)

    if fixed:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        idx = torch.randperm(n_total, generator=generator, device="cpu")[:n_select].to(dtype=torch.long)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(idx, cache_path)
        return idx.to(device)

    return torch.randperm(n_total, device=device)[:n_select]


def subsample_reward_points(
    x_gen: torch.Tensor,
    x_ref: torch.Tensor,
    n_points: Optional[int],
    mode: str = "uniform",
    generator: Optional[torch.Generator] = None,
    indices: Optional[torch.Tensor] = None,
    seed: int = 1234,
    cache_path: Optional[Path] = None,
    fixed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Subsample endpoint points for scalar reward/coherence evaluation only.

    The same point indices are applied to generated and reference fields.
    """
    n_total = int(x_gen.shape[1])
    if indices is None and (n_points is None or int(n_points) <= 0 or int(n_points) >= n_total):
        return x_gen, x_ref, n_total
    if str(mode).lower() != "uniform":
        raise ValueError("ram_reward_sampling currently supports only 'uniform'.")
    if indices is None:
        if generator is not None and not fixed and cache_path is None:
            n_select = int(n_points)
            idx = torch.randperm(n_total, device=x_gen.device, generator=generator)[:n_select]
        else:
            idx = get_or_build_reward_point_indices(
                n_total=n_total,
                n_points=n_points,
                device=x_gen.device,
                seed=seed,
                cache_path=cache_path,
                fixed=fixed,
            )
    else:
        idx = indices.to(device=x_gen.device, dtype=torch.long)
    if idx is None:
        return x_gen, x_ref, n_total
    n_select = int(idx.numel())
    return x_gen[:, idx], x_ref[:, idx], n_select


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


@torch.no_grad()
def update_ema_params_cross_device(source: nn.Module, target: nn.Module, decay: float) -> None:
    """EMA update when the target eval model may live on CPU."""
    decay = float(decay)
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.mul_(decay).add_(src_param.detach().data.to(tgt_param.device), alpha=1.0 - decay)

    for src_buffer, tgt_buffer in zip(source.buffers(), target.buffers()):
        tgt_buffer.data.copy_(src_buffer.detach().data.to(tgt_buffer.device))


def train_ram_epoch(
    *,
    epoch: int,
    policy_model: nn.Module,
    ref_model: nn.Module,
    old_model: nn.Module,
    eval_model: Optional[nn.Module],
    trainable_params: Sequence[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    ram_coh_cfg: RAMCoherenceConfig,
    train_set: TurbulentCombustionH5Dataset,
    reward_std_ema: Optional[EMARewardStd] = None,
) -> tuple[float, Dict[str, float]]:
    """
    One RAM epoch: sample endpoints, score them, build advantages, then fit the
    policy velocity to the advantage-scaled detached target.
    """
    policy_model.train()
    ref_model.eval()
    old_model.eval()
    if eval_model is not None:
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

            x_end_reward, x_ref_reward, reward_point_count = subsample_reward_points(
                x_gen=x_end,
                x_ref=x_ref_g,
                n_points=cfg.get("ram_reward_n_points", None),
                mode=str(cfg.get("ram_reward_sampling", "uniform")),
            )
            cost, coh_metrics = compute_ram_coherence_cost(
                x_gen=x_end_reward,
                x_ref=x_ref_reward,
                cfg=ram_coh_cfg,
                mean=train_set.mean.to(device),
                std=train_set.std.to(device),
            )
            rewards = -cost
            running_reward_std.update(rewards)

            rewards_grouped = rewards.view(bsz, G)
            adv = rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)
            batch_adv_std = float(adv.reshape(-1).std(unbiased=False).detach().cpu())
            reward_scale_std = float("nan")
            if reward_scaling == "running_epoch_std":
                scale = running_reward_std.std + reward_eps
                adv = adv / scale
                reward_scale_std = float(scale)
            elif reward_scaling == "batch_std":
                scale = batch_adv_std + reward_eps
                adv = adv / scale
                reward_scale_std = float(scale)
            elif reward_scaling == "ema_std":
                if reward_std_ema is None:
                    raise ValueError("reward_scaling='ema_std' requires a persistent EMARewardStd tracker.")
                scale = reward_std_ema.update(batch_adv_std) + reward_eps
                adv = adv / scale
                reward_scale_std = float(scale)
            elif reward_scaling == "group":
                adv = adv / (rewards_grouped.std(dim=1, keepdim=True, unbiased=False) + reward_eps)
                reward_scale_std = float("nan")
            elif reward_scaling in ("none", "false", "False"):
                pass
            else:
                raise ValueError(
                    "reward_scaling must be 'batch_std', 'running_epoch_std', "
                    "'ema_std', 'group', or 'none'."
                )
            if adv_clip_abs_value is not None:
                adv = adv.clamp(-adv_clip_abs_value, adv_clip_abs_value)
            adv_flat = adv.reshape(-1)
            scaled_adv_flat = float(cfg.get("reward_multiplier", 1.0)) * adv_flat.detach()

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
        target_delta_total = 0.0
        bridge_residual_total = 0.0
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
            target_delta_total += float(((target_v.detach() - v_ref) ** 2).mean().cpu()) * weight
            bridge_residual_total += float(((bridge_direction.detach() - v_old) ** 2).mean().cpu()) * weight
            target_velocity_total += float((target_v.detach() ** 2).mean().cpu()) * weight
            loss_weight_total += weight

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.get("grad_clip", 1.0)))
        optimizer.step()

        # Step-level EMA keeps the old/eval policies close enough for short RAM
        # runs while still providing lagged targets.
        update_ema_params(policy_model, old_model, float(cfg.get("old_ema_decay", 0.9)))
        if eval_model is not None:
            update_ema_params_cross_device(policy_model, eval_model, float(cfg.get("eval_ema_decay", 0.99)))

        current_lr = float(optimizer.param_groups[0]["lr"])
        loss_value = loss_total / max(loss_weight_total, 1)
        output_delta_value = output_delta_total / max(loss_weight_total, 1)
        target_delta_value = target_delta_total / max(loss_weight_total, 1)
        bridge_residual_value = bridge_residual_total / max(loss_weight_total, 1)
        target_velocity_value = target_velocity_total / max(loss_weight_total, 1)
        metrics = {
            "ram_loss": loss_value,
            "reward_mean": float(rewards.mean().detach().cpu()),
            "reward_std": float(rewards.std(unbiased=False).detach().cpu()),
            "adv_abs_mean": float(adv.abs().mean().detach().cpu()),
            "adv_max_abs": float(adv.abs().max().detach().cpu()),
            "adv_clip_abs": np.nan if adv_clip_abs_value is None else adv_clip_abs_value,
            "scaled_adv_abs_mean": float(scaled_adv_flat.abs().mean().detach().cpu()),
            "scaled_adv_abs_max": float(scaled_adv_flat.abs().max().detach().cpu()),
            "reward_scale_std": reward_scale_std,
            "coherence/ram_cost": float(coh_metrics.get("ram_cost", np.nan)),
            "coherence/global_dist_score": float(coh_metrics.get("global_dist_score", np.nan)),
            "coherence/marginal_score": float(coh_metrics.get("marginal_score", np.nan)),
            "coherence/joint_score": float(coh_metrics.get("joint_score", np.nan)),
            "coherence/pairwise_mean": float(coh_metrics.get("pairwise_mean", np.nan)),
            "coherence/field_l2_rel": float(coh_metrics.get("field_l2_rel", np.nan)),
            "coherence/field_l2_mse": float(coh_metrics.get("field_l2_mse", np.nan)),
            "output_delta_norm": output_delta_value,
            "policy_ref_delta_norm": output_delta_value,
            "target_delta_norm": target_delta_value,
            "bridge_residual_norm": bridge_residual_value,
            "target_velocity_norm": target_velocity_value,
            "endpoint_steps": float(cfg.get("ram_endpoint_steps", 4)),
            "ram_n_query_points": float(coords_q.shape[1]),
            "reward_point_count": float(reward_point_count),
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
    avg["reward_mode"] = str(cfg.get("reward_mode", "global_dist"))
    return avg.get("ram_loss", float("nan")), avg


def train_ram_lora_epoch(
    *,
    epoch: int,
    lora_model: nn.Module,
    trainable_params: Sequence[nn.Parameter],
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    device: torch.device,
    cfg: dict,
    ram_coh_cfg: RAMCoherenceConfig,
    train_set: TurbulentCombustionH5Dataset,
    reward_std_ema: Optional[EMARewardStd] = None,
) -> tuple[float, Dict[str, float]]:
    """
    One RAM epoch using one base model plus default/old/evaluation LoRA adapters.
    """
    lora_model.train()

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

    pbar = tqdm(loader, desc=f"RAM LoRA epoch {epoch:04d}", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if max_train_batches is not None and batch_idx >= max_train_batches:
            break

        coords_full = batch["coords"].to(device)
        fields_full = batch["fields"].to(device)
        bsz = coords_full.shape[0]

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

        lora_model.eval()
        with torch.no_grad(), use_adapter(lora_model, "old"):
            x_end = _sample_endpoints_microbatched(
                model=lora_model,
                coords=coords_g,
                obs_coords=obs_coords_g,
                obs_values=obs_values_g,
                obs_mask=obs_mask_g,
                obs_field_ids=obs_field_ids_g,
                obs_indices=obs_indices_g,
                cfg=cfg,
            )

            x_end_reward, x_ref_reward, reward_point_count = subsample_reward_points(
                x_gen=x_end,
                x_ref=x_ref_g,
                n_points=cfg.get("ram_reward_n_points", None),
                mode=str(cfg.get("ram_reward_sampling", "uniform")),
            )
            cost, coh_metrics = compute_ram_coherence_cost(
                x_gen=x_end_reward,
                x_ref=x_ref_reward,
                cfg=ram_coh_cfg,
                mean=train_set.mean.to(device),
                std=train_set.std.to(device),
            )
            rewards = -cost
            running_reward_std.update(rewards)

            rewards_grouped = rewards.view(bsz, G)
            adv = rewards_grouped - rewards_grouped.mean(dim=1, keepdim=True)
            batch_adv_std = float(adv.reshape(-1).std(unbiased=False).detach().cpu())
            reward_scale_std = float("nan")
            if reward_scaling == "running_epoch_std":
                scale = running_reward_std.std + reward_eps
                adv = adv / scale
                reward_scale_std = float(scale)
            elif reward_scaling == "batch_std":
                scale = batch_adv_std + reward_eps
                adv = adv / scale
                reward_scale_std = float(scale)
            elif reward_scaling == "ema_std":
                if reward_std_ema is None:
                    raise ValueError("reward_scaling='ema_std' requires a persistent EMARewardStd tracker.")
                scale = reward_std_ema.update(batch_adv_std) + reward_eps
                adv = adv / scale
                reward_scale_std = float(scale)
            elif reward_scaling == "group":
                adv = adv / (rewards_grouped.std(dim=1, keepdim=True, unbiased=False) + reward_eps)
            elif reward_scaling in ("none", "false", "False"):
                pass
            else:
                raise ValueError(
                    "reward_scaling must be 'batch_std', 'running_epoch_std', "
                    "'ema_std', 'group', or 'none'."
                )
            if adv_clip_abs_value is not None:
                adv = adv.clamp(-adv_clip_abs_value, adv_clip_abs_value)
            adv_flat = adv.reshape(-1)
            scaled_adv_flat = float(cfg.get("reward_multiplier", 1.0)) * adv_flat.detach()

        lora_model.train()
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
        target_delta_total = 0.0
        bridge_residual_total = 0.0
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

            z = lora_model.sample_source(coords_mb)
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

            lora_model.train()
            with use_adapter(lora_model, "default"):
                v_policy = lora_model.model(
                    t, x_t, coords_mb, obs_coords_mb, obs_values_mb, obs_mask_mb, obs_field_ids_mb
                )
            lora_model.eval()
            with torch.no_grad(), use_adapter(lora_model, None):
                v_ref = lora_model.model(
                    t, x_t, coords_mb, obs_coords_mb, obs_values_mb, obs_mask_mb, obs_field_ids_mb
                )
            with torch.no_grad(), use_adapter(lora_model, "old"):
                v_old = lora_model.model(
                    t, x_t, coords_mb, obs_coords_mb, obs_values_mb, obs_mask_mb, obs_field_ids_mb
                )
            lora_model.train()

            scaled_adv = float(cfg.get("reward_multiplier", 1.0)) * adv_mb.view(-1, 1, 1)
            target_v = v_ref + scaled_adv * (bridge_direction - v_old)
            loss_mb = ((v_policy - target_v.detach()) ** 2).mean()

            weight = x_end_mb.shape[0]
            weighted_loss = loss_mb * (float(weight) / float(n_loss_items))
            weighted_loss.backward()

            loss_total += float(loss_mb.detach().cpu()) * weight
            output_delta_total += float(((v_policy.detach() - v_ref) ** 2).mean().cpu()) * weight
            target_delta_total += float(((target_v.detach() - v_ref) ** 2).mean().cpu()) * weight
            bridge_residual_total += float(((bridge_direction.detach() - v_old) ** 2).mean().cpu()) * weight
            target_velocity_total += float((target_v.detach() ** 2).mean().cpu()) * weight
            loss_weight_total += weight

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, float(cfg.get("grad_clip", 1.0)))
        optimizer.step()

        ema_lora_adapter(lora_model, "default", "old", float(cfg.get("old_ema_decay", 0.9)))
        ema_lora_adapter(lora_model, "default", "evaluation", float(cfg.get("eval_ema_decay", 0.99)))

        current_lr = float(optimizer.param_groups[0]["lr"])
        loss_value = loss_total / max(loss_weight_total, 1)
        output_delta_value = output_delta_total / max(loss_weight_total, 1)
        target_delta_value = target_delta_total / max(loss_weight_total, 1)
        bridge_residual_value = bridge_residual_total / max(loss_weight_total, 1)
        target_velocity_value = target_velocity_total / max(loss_weight_total, 1)
        metrics = {
            "ram_loss": loss_value,
            "reward_mean": float(rewards.mean().detach().cpu()),
            "reward_std": float(rewards.std(unbiased=False).detach().cpu()),
            "adv_abs_mean": float(adv.abs().mean().detach().cpu()),
            "adv_max_abs": float(adv.abs().max().detach().cpu()),
            "adv_clip_abs": np.nan if adv_clip_abs_value is None else adv_clip_abs_value,
            "scaled_adv_abs_mean": float(scaled_adv_flat.abs().mean().detach().cpu()),
            "scaled_adv_abs_max": float(scaled_adv_flat.abs().max().detach().cpu()),
            "reward_scale_std": reward_scale_std,
            "coherence/ram_cost": float(coh_metrics.get("ram_cost", np.nan)),
            "coherence/global_dist_score": float(coh_metrics.get("global_dist_score", np.nan)),
            "coherence/marginal_score": float(coh_metrics.get("marginal_score", np.nan)),
            "coherence/joint_score": float(coh_metrics.get("joint_score", np.nan)),
            "coherence/pairwise_mean": float(coh_metrics.get("pairwise_mean", np.nan)),
            "coherence/field_l2_rel": float(coh_metrics.get("field_l2_rel", np.nan)),
            "coherence/field_l2_mse": float(coh_metrics.get("field_l2_mse", np.nan)),
            "output_delta_norm": output_delta_value,
            "policy_ref_delta_norm": output_delta_value,
            "target_delta_norm": target_delta_value,
            "bridge_residual_norm": bridge_residual_value,
            "target_velocity_norm": target_velocity_value,
            "endpoint_steps": float(cfg.get("ram_endpoint_steps", 4)),
            "ram_n_query_points": float(coords_q.shape[1]),
            "reward_point_count": float(reward_point_count),
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
    avg["reward_mode"] = str(cfg.get("reward_mode", "global_dist"))
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
            n_steps=int(cfg.get("n_steps_generation_eval", 2)),
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
        recon_reward, fields_reward, reward_point_count = subsample_reward_points(
            x_gen=recon,
            x_ref=fields_full,
            n_points=cfg.get("ram_reward_n_points", None),
            mode=str(cfg.get("ram_reward_sampling", "uniform")),
            seed=int(cfg.get("reward_subsample_seed", 1234)),
            fixed=bool(cfg.get("fixed_reward_points_for_eval", True)),
        )
        cost, coh_metrics = compute_ram_coherence_cost(
            x_gen=recon_reward,
            x_ref=fields_reward,
            cfg=ram_coh_cfg,
            mean=val_set.mean.to(device),
            std=val_set.std.to(device),
        )
        metrics.update({
            "val_rel_l2": float(rel_l2.mean().cpu()),
            "val_coherence_cost": float(cost.mean().cpu()),
            "val_coherence/global_dist_score": float(coh_metrics.get("global_dist_score", np.nan)),
            "reward_point_count": float(reward_point_count),
        })

    out = metrics.mean()
    rel_l2_weight = float(cfg.get("val_loss_rel_l2_weight", 0.1))
    coherence_weight = float(cfg.get("val_loss_coherence_weight", 1.0))
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


def save_lora_checkpoint(
    *,
    path: Path,
    lora_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float],
    train_set: TurbulentCombustionH5Dataset,
    source_run_dir: Path,
    source_cfg: dict,
    cfg: dict,
) -> None:
    """Save RAM-LoRA with merged evaluation adapter weights under ``model``."""
    merged_state = export_merged_lora_state_dict(
        lora_model=lora_model,
        source_cfg=source_cfg,
        dataset=train_set,
        adapter="evaluation",
        device="cpu",
    )
    ckpt = {
        "model": merged_state,
        "lora_state": collect_lora_state(lora_model),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "val_loss": None if val_loss is None else float(val_loss),
        "mean": train_set.mean,
        "std": train_set.std,
        "field_names": train_set.field_names,
        "method": "ram_pointcloud_ffm",
        "finetune_method": "RAM_LORA",
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
            n_steps=int(cfg.get("n_steps_generation_eval", 2)),
            ode_solver=str(cfg.get("ode_solver", "euler")),
            snapshot_index=0,
            file_tag=f"ram_eval_nfe{int(cfg.get('n_steps_generation_eval', 2))}",
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

    finetune_mode = str(cfg.get("finetune_mode", "head_glres")).strip().lower()
    is_lora_mode = finetune_mode.startswith("lora_")
    use_eval_ema = bool(cfg.get("use_eval_ema", True))
    eval_ema_device_name = str(cfg.get("eval_ema_device", "gpu")).strip().lower()
    if eval_ema_device_name not in ("gpu", "cpu"):
        raise ValueError("eval_ema_device must be 'gpu' or 'cpu'.")

    ref_model, source_cfg_loaded, _ = load_pretrained_ffm(
        source_run_dir=source_run_dir,
        checkpoint=str(cfg.get("source_checkpoint", "best")),
        dataset=train_set,
        device=device,
    )
    source_cfg = source_cfg_loaded

    lora_model: Optional[nn.Module] = None
    policy_model: Optional[nn.Module] = None
    old_model: Optional[nn.Module] = None
    eval_model: Optional[nn.Module] = None
    eval_model_storage_device = device

    if is_lora_mode:
        lora_model = ref_model
        _freeze_model(lora_model)
        lora_scope = _resolve_lora_scope(cfg, finetune_mode)
        wrapped_paths = inject_lora_adapters(
            lora_model,
            scope=lora_scope,
            rank=int(cfg.get("lora_rank", 8)),
            alpha=float(cfg.get("lora_alpha", 16)),
        )
        cfg["lora_wrapped_paths"] = wrapped_paths
        sync_lora_adapter(lora_model, "default", "old")
        sync_lora_adapter(lora_model, "default", "evaluation")
        trainable_params = lora_trainable_params(lora_model)
    else:
        policy_model = clone_model(ref_model, device)
        old_model = clone_model(ref_model, device)
        sync_params(ref_model, policy_model)
        sync_params(ref_model, old_model)
        if use_eval_ema:
            eval_model_storage_device = torch.device("cpu") if eval_ema_device_name == "cpu" else device
            eval_model = clone_model(ref_model, eval_model_storage_device)
            sync_params(ref_model, eval_model)
        else:
            eval_model = None

        _freeze_model(ref_model)
        _freeze_model(old_model)
        if eval_model is not None:
            _freeze_model(eval_model)
        trainable_params = set_trainable_scope(policy_model, finetune_mode)

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
    rollout_policy_logger = None
    if rollout_set is not None:
        rollout_logger = RolloutHistoryLogger(
            run_dir,
            getattr(rollout_set, "field_names", []),
            save_json=save_history_json,
            model_role="eval",
        )
        if is_lora_mode:
            with use_adapter(lora_model, "evaluation"):
                run_rollout_evaluations(
                    model=lora_model,
                    dataset=rollout_set,
                    epoch=0,
                    device=device,
                    run_dir=run_dir,
                    cfg=cfg,
                    ram_coh_cfg=ram_coh_cfg,
                    logger=rollout_logger,
                    model_role="eval",
                )
        else:
            eval_runtime_model = eval_model if eval_model is not None else policy_model
            if eval_model is not None and eval_ema_device_name == "cpu":
                eval_runtime_model = eval_model.to(device)
            run_rollout_evaluations(
                model=eval_runtime_model,
                dataset=rollout_set,
                epoch=0,
                device=device,
                run_dir=run_dir,
                cfg=cfg,
                ram_coh_cfg=ram_coh_cfg,
                logger=rollout_logger,
                model_role="eval",
            )
            if eval_model is not None and eval_ema_device_name == "cpu":
                eval_model.to(eval_model_storage_device)
        if bool(cfg.get("rollout_eval_policy_model", True)):
            rollout_policy_logger = RolloutHistoryLogger(
                run_dir,
                getattr(rollout_set, "field_names", []),
                save_json=save_history_json,
                model_role="policy",
            )
            if is_lora_mode:
                with use_adapter(lora_model, "default"):
                    run_rollout_evaluations(
                        model=lora_model,
                        dataset=rollout_set,
                        epoch=0,
                        device=device,
                        run_dir=run_dir,
                        cfg=cfg,
                        ram_coh_cfg=ram_coh_cfg,
                        logger=rollout_policy_logger,
                        model_role="policy",
                    )
            else:
                run_rollout_evaluations(
                    model=policy_model,
                    dataset=rollout_set,
                    epoch=0,
                    device=device,
                    run_dir=run_dir,
                    cfg=cfg,
                    ram_coh_cfg=ram_coh_cfg,
                    logger=rollout_policy_logger,
                    model_role="policy",
                )

    best_val = float("inf")
    last_val_loss: Optional[float] = None
    reward_std_ema = EMARewardStd(decay=float(cfg.get("reward_std_ema_decay", 0.95)))
    for epoch in range(1, int(cfg.get("ram_epochs", 200)) + 1):
        train_loader = build_epoch_train_loader(train_set, cfg, epoch)
        if is_lora_mode:
            train_loss, train_metrics = train_ram_lora_epoch(
                epoch=epoch,
                lora_model=lora_model,
                trainable_params=trainable_params,
                optimizer=optimizer,
                loader=train_loader,
                device=device,
                cfg=cfg,
                ram_coh_cfg=ram_coh_cfg,
                train_set=train_set,
                reward_std_ema=reward_std_ema,
            )
        else:
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
                reward_std_ema=reward_std_ema,
            )

        val_metrics: Dict[str, float] = {}
        if epoch == 1 or epoch % int(cfg.get("eval_every", 5)) == 0:
            if is_lora_mode:
                with use_adapter(lora_model, "evaluation"):
                    val_metrics = validate_ram(
                        eval_model=lora_model,
                        loader=val_loader,
                        device=device,
                        cfg=cfg,
                        ram_coh_cfg=ram_coh_cfg,
                        val_set=val_set,
                        max_batches=int(cfg.get("eval_num_batches", 2)),
                    )
            else:
                eval_runtime_model = eval_model if eval_model is not None else policy_model
                if eval_model is not None and eval_ema_device_name == "cpu":
                    eval_runtime_model = eval_model.to(device)
                val_metrics = validate_ram(
                    eval_model=eval_runtime_model,
                    loader=val_loader,
                    device=device,
                    cfg=cfg,
                    ram_coh_cfg=ram_coh_cfg,
                    val_set=val_set,
                    max_batches=int(cfg.get("eval_num_batches", 2)),
                )
                if eval_model is not None and eval_ema_device_name == "cpu":
                    eval_model.to(eval_model_storage_device)
            last_val_loss = val_metrics.get("val_loss", None)
            print(
                f"[valid] epoch={epoch:04d} val_score={last_val_loss:.6e} "
                f"rel_l2={val_metrics.get('val_rel_l2', float('nan')):.6e} "
                f"coh={val_metrics.get('val_coherence_cost', float('nan')):.6e} "
                f"(score={float(cfg.get('val_loss_rel_l2_weight', 0.1)):.3g}*rel_l2+"
                f"{float(cfg.get('val_loss_coherence_weight', 1.0)):.3g}*coh)"
            )

        merged_metrics = dict(train_metrics)
        merged_metrics.update(val_metrics)
        merged_metrics["val_loss"] = last_val_loss
        logger.log(epoch=epoch, train_loss=train_loss, val_loss=last_val_loss, metrics=merged_metrics)

        if is_lora_mode:
            save_lora_checkpoint(
                path=run_dir / "last.pt",
                lora_model=lora_model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=last_val_loss,
                train_set=train_set,
                source_run_dir=source_run_dir,
                source_cfg=source_cfg,
                cfg=cfg,
            )
        else:
            checkpoint_eval_model = eval_model if eval_model is not None else policy_model
            save_checkpoint(
                path=run_dir / "last.pt",
                eval_model=checkpoint_eval_model,
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
            if is_lora_mode:
                save_lora_checkpoint(
                    path=run_dir / "best.pt",
                    lora_model=lora_model,
                    optimizer=optimizer,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=last_val_loss,
                    train_set=train_set,
                    source_run_dir=source_run_dir,
                    source_cfg=source_cfg,
                    cfg=cfg,
                )
            else:
                checkpoint_eval_model = eval_model if eval_model is not None else policy_model
                save_checkpoint(
                    path=run_dir / "best.pt",
                    eval_model=checkpoint_eval_model,
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
            if is_lora_mode:
                with use_adapter(lora_model, "evaluation"):
                    maybe_save_preview(
                        eval_model=lora_model,
                        val_set=val_set,
                        epoch=epoch,
                        device=device,
                        run_dir=run_dir,
                        cfg=cfg,
                    )
            else:
                preview_model = eval_model if eval_model is not None else policy_model
                if eval_model is not None and eval_ema_device_name == "cpu":
                    preview_model = eval_model.to(device)
                maybe_save_preview(
                    eval_model=preview_model,
                    val_set=val_set,
                    epoch=epoch,
                    device=device,
                    run_dir=run_dir,
                    cfg=cfg,
                )
                if eval_model is not None and eval_ema_device_name == "cpu":
                    eval_model.to(eval_model_storage_device)
        if (
            rollout_set is not None
            and rollout_logger is not None
            and int(cfg.get("rollout_eval_every", 20)) > 0
            and epoch % int(cfg.get("rollout_eval_every", 20)) == 0
        ):
            if is_lora_mode:
                with use_adapter(lora_model, "evaluation"):
                    run_rollout_evaluations(
                        model=lora_model,
                        dataset=rollout_set,
                        epoch=epoch,
                        device=device,
                        run_dir=run_dir,
                        cfg=cfg,
                        ram_coh_cfg=ram_coh_cfg,
                        logger=rollout_logger,
                        model_role="eval",
                    )
            else:
                eval_runtime_model = eval_model if eval_model is not None else policy_model
                if eval_model is not None and eval_ema_device_name == "cpu":
                    eval_runtime_model = eval_model.to(device)
                run_rollout_evaluations(
                    model=eval_runtime_model,
                    dataset=rollout_set,
                    epoch=epoch,
                    device=device,
                    run_dir=run_dir,
                    cfg=cfg,
                    ram_coh_cfg=ram_coh_cfg,
                    logger=rollout_logger,
                    model_role="eval",
                )
                if eval_model is not None and eval_ema_device_name == "cpu":
                    eval_model.to(eval_model_storage_device)
            if rollout_policy_logger is not None:
                if is_lora_mode:
                    with use_adapter(lora_model, "default"):
                        run_rollout_evaluations(
                            model=lora_model,
                            dataset=rollout_set,
                            epoch=epoch,
                            device=device,
                            run_dir=run_dir,
                            cfg=cfg,
                            ram_coh_cfg=ram_coh_cfg,
                            logger=rollout_policy_logger,
                            model_role="policy",
                        )
                else:
                    run_rollout_evaluations(
                        model=policy_model,
                        dataset=rollout_set,
                        epoch=epoch,
                        device=device,
                        run_dir=run_dir,
                        cfg=cfg,
                        ram_coh_cfg=ram_coh_cfg,
                        logger=rollout_policy_logger,
                        model_role="policy",
                    )

        print(f"[train] epoch={epoch:04d} ram_loss={train_loss:.6e}")

    if not (run_dir / "best.pt").exists():
        shutil.copy(run_dir / "last.pt", run_dir / "best.pt")
    print("[*] RAM fine-tuning complete.")
    print(f"[*] Best validation rollout score: {best_val:.6e}")


if __name__ == "__main__":
    main()

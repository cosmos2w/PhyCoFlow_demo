
'''
With this patch:

- training can use any configured field combination like [0], [2], [0, 2], [0, 2, 4]

- each conditioned field can have its own n_obs_min / n_obs_max

- visualization can use its own cond_fields and exact n_obs list, independent of training

- Model backbone can be ConditionalPointMLPRBF, ConditionalPointPerceiver
'''

import argparse
import csv
import hashlib
import yaml
import shutil
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from datetime import datetime

from helpers import (
    TurbulentCombustionH5Dataset,
    validate_regular_grid_compatibility,
    visualize_reconstruction,
    build_sparse_condition,
)
from pointcloud_data_path import (
    DataPathDiagnostics,
    PointCloudBatchCollator,
    ResolvedDataPathConfig,
    apply_resolved_data_path_config,
    materialize_queries_from_full,
    materialize_sparse_condition_from_layout,
    print_resolved_data_path_config,
    resolve_data_path_config,
)
from Model import (
    ConditionalPointFFM, 
    ConditionalPointMLPRBF, 
    ConditionalPointPerceiver,
    ConditionalPointHybridLocalGlobalRBF,
    ConditionalPointHybridLocalGlobalRBFCQ,
    PointCloudFFM,
    FNO,
    FNOFFM,
    )
from direct_coherence_loss import (
    DirectCoherenceConfig,
    DirectGlobalCoherenceLoss,
    apply_two_objective_update,
    differentiable_rf_rollout,
    sample_coherence_points,
)
from model_finetune import find_source_run_dir, load_source_config
from model_ema import ModelEMA
from phycoflow_pointcloud.checkpointing import checkpoint_model_state
from phycoflow_pointcloud.config import load_public_config, resolve_model_identity
from phycoflow_pointcloud.models.factory import build_pointcloud_model
from phycoflow_pointcloud.priors import IIDGaussianPrior, RFFGaussianPrior


def _yaml_value(value: str):
    return yaml.safe_load(value)


def _set_overrides(entries: Sequence[str]) -> dict:
    overrides = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected KEY=YAML_VALUE for --set, got {entry!r}.")
        key, value = entry.split("=", 1)
        overrides[key] = _yaml_value(value)
    return overrides


def _option_was_supplied(argv: Sequence[str], *names: str) -> bool:
    return any(
        token == name or token.startswith(f"{name}=")
        for token in argv
        for name in names
    )


def _model_schema_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
    return digest.hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None):

    p = argparse.ArgumentParser("Train a starter conditional point-cloud FFM on turbulent combustion HDF5 data.")

    p.add_argument("--config", type=str, 
                   default="Save_config/config_pointcloud_ffm.yaml", help="Path to YAML config")
    p.add_argument("--model-name", type=str, default=None,
                   help="Public model name; historical backbone remains authoritative for compatibility.")
    p.add_argument("--coord-dim", type=int, default=3,
                   help="Coordinate-channel count (the turbulent-combustion profile uses 3).")
    p.add_argument("--Demo-Num", "--demo-num", dest="Demo_Num", type=int,
                   default=0, help="Demo ID tag for saving directories")
    p.add_argument("--device-ids", type=int, nargs="+", default=[0])

    p.add_argument("--data", type=str, 
                   default="Dataset/Merged_CH4COTU1P.h5")
    p.add_argument(
        "--dataset-stats-path",
        type=str,
        default=None,
        help="Optional precomputed normalization statistics. Defaults to dataset_stats.pt in the run directory.",
    )
    p.add_argument("--FIELD-NAMES", "--FIELD_NAMES", dest="FIELD_NAMES", nargs="+", default=None)
    p.add_argument("--field-names", dest="field_names", nargs="+", default=None)
    p.add_argument("--save-dir", type=str, 
                   default=f"Save_TrainedModel/ffm_tc_pointcloud")
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=YAML_VALUE",
        help="Override any recognized YAML value after loading the config.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the resolved config and model schema without reading data.",
    )
    # Populated by complete YAML files.  There is deliberately no fragmented
    # CLI shorthand for scientific id/variant/reference provenance.
    p.set_defaults(ablation=None)
    p.add_argument("--RELOAD", action="store_true",
                   help="If set, try to reload the latest matching checkpoint and continue training.")
    p.add_argument("--training-mode", dest="training_mode", type=str, default="standard",
                   choices=["standard", "direct_coherence"])
    p.add_argument("--initialization", type=str, default="scratch", choices=["scratch", "pretrained"])
    p.add_argument("--pretrained-run-dir", dest="pretrained_run_dir", type=str, default=None)
    p.add_argument("--pretrained-source-Demo-Num", dest="pretrained_source_Demo_Num", type=int, default=None)
    p.add_argument("--pretrained-checkpoint", dest="pretrained_checkpoint", type=str, default="best")
    p.add_argument("--pretrained-load-optimizer", dest="pretrained_load_optimizer", action="store_true")
    p.add_argument("--no-pretrained-load-optimizer", dest="pretrained_load_optimizer", action="store_false")
    p.set_defaults(pretrained_load_optimizer=False)
    p.add_argument("--pretrained-strict", dest="pretrained_strict", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pretrained-use-source-base-config", dest="pretrained_use_source_base_config",
                   action=argparse.BooleanOptionalAction, default=True)
    
    # ------------------------------
    # Backbone selection
    # ------------------------------
    p.add_argument(
        "--backbone", type=str, default="mlp_rbf", choices = ["mlp_rbf", "perceiver", "fno", "GL_rbf", "GL_rbf_ENH", "GL_rbf_ENH_CQ"],
        help="Backbone type. point-cloud MLP+RBF, point-cloud Perceiver, or grid-based FNO baseline.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--scheduler-t-max", type=int, default=None,
                   help="Optional cosine schedule horizon; defaults to epochs for legacy behavior.")
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--train-ratio-downsample", dest="train_ratio_downsample", type=float, default=1.0)
    p.add_argument("--time-stride", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)

    # ----------------------------------------------------------
    # Data-path performance / compatibility experiment
    # ----------------------------------------------------------
    p.add_argument("--data-path-mode", type=str, default="legacy", choices=["legacy", "optimized"])
    p.add_argument("--coord-batch-mode", type=str, default=None,
                   choices=["legacy_clone", "shared_mesh"])
    p.add_argument("--index-sampling-mode", type=str, default=None,
                   choices=["legacy_randperm", "scalable"])
    p.add_argument("--sampling-device", type=str, default=None,
                   choices=["legacy_gpu", "cpu"])
    p.add_argument("--field-read-mode", type=str, default=None,
                   choices=["legacy_full_snapshot", "indexed_union"])
    p.add_argument("--field-normalization-mode", type=str, default=None,
                   choices=["legacy_full_after_read", "selected_after_full_read"])
    p.add_argument("--gpu-transfer-mode", type=str, default=None,
                   choices=["legacy_full", "selected_only"])
    p.add_argument("--data-path-diag-storage-mode", type=str, default=None,
                   choices=["legacy_rewrite", "append"])
    p.add_argument("--dataloader-persistent-workers", default=None,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--dataloader-prefetch-factor", type=int, default=None)
    p.add_argument("--non-blocking-transfer", default=None,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--data-path-diagnostics", default=False,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--data-path-diag-every-n-steps", type=int, default=50)
    p.add_argument("--data-path-diag-warmup-steps", type=int, default=5)
    p.add_argument("--data-path-diag-max-steps-per-epoch", type=int, default=10)
    p.add_argument("--training-log-every-n-steps", type=int, default=None)
    p.add_argument(
        "--training-history-plot-every-n-epochs",
        type=int,
        default=50,
        help="Regenerate the loss-history PNG every N epochs; use 0 to disable plotting.",
    )

    # ------------------------------
    # These are hyperparameters for mlp_rbf backbone or part of GL_rbf
    # ------------------------------
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--cond-dim", type=int, default=128)
    p.add_argument("--field-embed-dim", type=int, default=64)
    p.add_argument("--rbf-sigma", type=float, default=0.05)
    p.add_argument("--USE-FOURIER-PE", "--USE_FOURIER_PE", dest="USE_FOURIER_PE", action="store_true",
                   help="If set, feed Fourier positional coordinate features to point/sensor token encoders.")
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
    p.add_argument("--sensor-coord-encoding", type=str, default=None,
                   choices=["raw", "fourier"],
                   help="Sensor coordinate encoding for GL_rbf/GL_rbf_ENH. "
                        "Use 'fourier' to give sensors the same coordinate features as queries.")
    p.add_argument("--latent-sensor-reinject", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="If enabled, latents periodically re-attend to sparse sensor tokens.")
    p.add_argument("--latent-reinject-every", type=int, default=1,
                   help="Re-inject sensor information every N latent blocks when latent_sensor_reinject is enabled.")
    p.add_argument(
        "--condition-attention-execution",
        choices=["legacy_mha", "cached_kv"],
        default="legacy_mha",
        help="Execution-only sensor-to-latent attention path.",
    )
    p.add_argument(
        "--sensor-attention-padding-mode",
        choices=["full", "static_buckets"],
        default="full",
        help="Use full padded sensor tensors or fixed sensor-length buckets.",
    )
    p.add_argument(
        "--sensor-attention-buckets",
        type=int,
        nargs="+",
        default=[256, 320, 384],
        help="Static sensor lengths used when sensor_attention_padding_mode=static_buckets.",
    )
    p.add_argument("--query-latent-readout", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="If enabled, each query reads global context from latent memory before the final head.")
    p.add_argument("--query-readout-type", type=str, default=None,
                   choices=["point", "coord"],
                   help="'coord' uses Senseiver-style coordinate decoder tokens; "
                        "'point' uses the current flow-state point features.")
    p.add_argument("--query-readout-scale-init", type=float, default=None,
                   help="Initial scale for query-to-latent readout. "
                        "Use small positive values such as 1e-2 for GL_rbf_ENH.")
    p.add_argument("--enhanced-head-norm", default=None,
                   action=argparse.BooleanOptionalAction,
                   help="If enabled, apply LayerNorm to the fused [query, global, local] head input.")
    p.add_argument("--glres-scale-init", type=float, default=None,
                   help="Initial scale for topk_rbf_glres residual terms: sensor importance and coarse scaffold.")

    # ----------------------------------------------------------
    # Compact Query Decoder — GL_rbf_ENH_CQ
    # ----------------------------------------------------------
    p.add_argument("--cq-query-dim", type=int, default=128)
    p.add_argument("--cq-readout-mode", choices=["full", "lowrank"], default="lowrank")
    p.add_argument(
        "--cq-fusion-mode",
        choices=["additive", "structured_concat"],
        default="additive",
    )
    p.add_argument("--cq-readout-rank", type=int, default=64)
    p.add_argument("--cq-readout-heads", type=int, default=4)
    p.add_argument("--cq-global-scale-init", type=float, default=1.0)
    p.add_argument("--cq-local-scale-init", type=float, default=1.0)
    p.add_argument("--cq-readout-scale-init", type=float, default=1.0e-2)
    p.add_argument("--cq-time-conditioning", choices=["scalar_concat", "sinusoidal_film"],
                   default="scalar_concat")
    p.add_argument("--cq-time-embed-dim", type=int, default=128)
    p.add_argument("--cq-time-max-period", type=float, default=10000.0)
    p.add_argument("--cq-time-film-zero-init", default=True,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--cq-measurement-support-mode", choices=["none", "rbf_value_support"],
                   default="none")
    p.add_argument("--cq-measurement-support-normalize", default=True,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--model-ema-enabled", default=False,
                   action=argparse.BooleanOptionalAction)
    p.add_argument("--model-ema-decay", type=float, default=0.999)
    p.add_argument("--model-ema-eval", default=True,
                   action=argparse.BooleanOptionalAction)

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
    p.add_argument("--train-query-microbatch-size", type=int, default=None)
    p.add_argument("--reuse-condition-context-across-query-microbatches",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reconstruction-execution-mode", type=str, default="legacy_full",
                   choices=["legacy_full", "cached_streamed"])
    p.add_argument("--reconstruction-query-chunk-size", type=int, default=8192)
    p.add_argument("--reconstruction-cache-level", type=str, default="static_features",
                   choices=["none", "geometry", "static_features"])
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
    p.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="*",
        default=[],
        help="Optional epochs whose checkpoints are retained as epoch_XXXX.pt.",
    )
    p.add_argument("--n-steps-generation", type=int, default=32)

    # ----------------------------------------------------------
    # Direct coherence training / post-training
    # ----------------------------------------------------------
    p.add_argument("--direct-coherence-enabled", dest="direct_coherence_enabled",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--data-loss-weight", dest="data_loss_weight", type=float, default=1.0)
    p.add_argument("--coherence-loss-weight", dest="coherence_loss_weight", type=float, default=0.1)
    p.add_argument("--coherence-start-epoch", dest="coherence_start_epoch", type=int, default=1)
    p.add_argument("--coherence-every-n-steps", dest="coherence_every_n_steps", type=int, default=1)
    p.add_argument("--coherence-interval-rescale", dest="coherence_interval_rescale",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--coherence-batch-size", dest="coherence_batch_size", type=int, default=4)
    p.add_argument("--coherence-downsample", dest="coherence_downsample",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coherence-n-points", dest="coherence_n_points", type=int, default=4096)
    p.add_argument("--coherence-rollout-steps", dest="coherence_rollout_steps", type=int, default=2)
    p.add_argument("--coherence-rollout-solver", dest="coherence_rollout_solver", type=str, default="euler",
                   choices=["euler", "heun"])
    p.add_argument("--coherence-obs-consistency-mode", dest="coherence_obs_consistency_mode",
                   type=str, default="endpoint_smooth",
                   choices=["none", "default_hard", "endpoint", "endpoint_smooth"])
    p.add_argument("--coherence-obs-consistency-strength", dest="coherence_obs_consistency_strength",
                   type=float, default=1.0)
    p.add_argument("--coherence-obs-consistency-sigma", dest="coherence_obs_consistency_sigma",
                   type=float, default=0.05)
    p.add_argument("--coherence-obs-consistency-schedule-power",
                   dest="coherence_obs_consistency_schedule_power", type=float, default=2.0)
    p.add_argument("--coherence-obs-consistency-final-clamp",
                   dest="coherence_obs_consistency_final_clamp",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coherence-self-weight", dest="coherence_self_weight", type=float, default=1.0)
    p.add_argument("--coherence-mutual-weight", dest="coherence_mutual_weight", type=float, default=0.10)
    p.add_argument("--coherence-cross-weight", dest="coherence_cross_weight", type=float, default=1.0)
    p.add_argument("--coherence-channel-weights", dest="coherence_channel_weights",
                   type=float, nargs="+", default=None)
    p.add_argument("--coherence-cross-num-directions", dest="coherence_cross_num_directions",
                   type=int, default=32)
    p.add_argument("--coherence-cross-top-frac", dest="coherence_cross_top_frac", type=float, default=0.10)
    p.add_argument("--coherence-cross-seed", dest="coherence_cross_seed", type=int, default=1234)
    p.add_argument("--coherence-cross-include-axes", dest="coherence_cross_include_axes",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coherence-cross-qmc", dest="coherence_cross_qmc",
                   action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--coherence-mutual-num-directions", dest="coherence_mutual_num_directions",
                   type=int, default=16)
    p.add_argument("--coherence-mutual-seed", dest="coherence_mutual_seed", type=int, default=1234)
    p.add_argument("--coherence-use-denorm", dest="coherence_use_denorm",
                   action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gradient-balance-mode", dest="gradient_balance_mode", type=str,
                   default="weighted_sum", choices=["weighted_sum", "config"])
    p.add_argument("--config-missing-behavior", dest="config_missing_behavior", type=str,
                   default="error", choices=["error", "weighted_sum"])
    p.add_argument("--config-data-grad-scale", dest="config_data_grad_scale", type=float, default=1.0)
    p.add_argument("--config-coherence-grad-scale", dest="config_coherence_grad_scale",
                   type=float, default=1.0)
    p.add_argument("--coherence-weight-warmup-epochs", dest="coherence_weight_warmup_epochs",
                   type=int, default=0)

    return p.parse_args(argv)

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

# =====================================================================
# LEGACY DATA PATH — BEGIN
# Temporary A/B reference implementation.
# Remove after optimized path is validated.
# =====================================================================
def collate_snapshots_legacy(batch):
    start = time.perf_counter()
    result = {
        "coords": torch.stack([b["coords"] for b in batch], dim=0),
        "fields": torch.stack([b["fields"] for b in batch], dim=0),
        "time_index": torch.stack([b["time_index"] for b in batch], dim=0),
        "physical_time": torch.stack([b["physical_time"] for b in batch], dim=0),
    }
    if any("data_path_item_timings" in item for item in batch):
        hdf5_ms = sum(item.get("data_path_item_timings", {}).get("hdf5_read_ms", 0.0) for item in batch)
        normalization_ms = sum(
            item.get("data_path_item_timings", {}).get("cpu_normalization_ms", 0.0)
            for item in batch
        )
        result["data_path_timings"] = {
            "index_sampling_ms": 0.0,
            "hdf5_read_ms": hdf5_ms,
            "cpu_normalization_ms": normalization_ms,
            "cpu_materialization_ms": max(
                0.0, (time.perf_counter() - start) * 1000.0
            ),
        }
    return result


def sample_query_subset_legacy(
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


# Backward-compatible names for reconstruction helpers and old imports.
collate_snapshots = collate_snapshots_legacy
sample_query_subset = sample_query_subset_legacy
# =====================================================================
# LEGACY DATA PATH — END
# =====================================================================

def build_pointcloud_loader(
    dataset: Dataset,
    args,
    data_path_config: ResolvedDataPathConfig,
    *,
    training: bool,
    shuffle: bool,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    """Build a loader with one resolved data-path configuration."""
    collator_dataset = dataset
    while isinstance(collator_dataset, Subset):
        collator_dataset = collator_dataset.dataset
    exact_legacy_collate = (
        data_path_config.coord_batch_mode == "legacy_clone"
        and data_path_config.sampling_device == "legacy_gpu"
        and data_path_config.gpu_transfer_mode == "legacy_full"
    )
    collate_fn = collate_snapshots_legacy if exact_legacy_collate else PointCloudBatchCollator(
        dataset=collator_dataset,
        config=data_path_config,
        cond_fields=args.cond_fields,
        n_obs_min=args.n_obs_min_list,
        n_obs_max=args.n_obs_max_list,
        n_query_points=None if args.backbone == "fno" else args.n_query_points,
        query_sampling=args.query_sampling if training else "uniform",
        query_sample_near_ratio=args.query_sample_near_ratio,
        query_sample_far_ratio=args.query_sample_far_ratio,
        query_sample_sigma_ratio=args.query_sample_sigma_ratio,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "generator": generator,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "collate_fn": collate_fn,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = data_path_config.dataloader_persistent_workers
        if data_path_config.dataloader_prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = data_path_config.dataloader_prefetch_factor
    return DataLoader(**loader_kwargs)


def _sync_cuda_for_diagnostic(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def _prepare_training_tensors(
    *,
    batch: Dict[str, object],
    device: torch.device,
    data_path_config: ResolvedDataPathConfig,
    diagnostic_step: bool,
    cond_fields: Sequence[int],
    n_obs_min_list: Sequence[int],
    n_obs_max_list: Sequence[int],
    n_query_points: Optional[int],
    query_sampling: str,
    query_sample_near_ratio: float,
    query_sample_far_ratio: float,
    query_sample_sigma_ratio: float,
) -> tuple[Dict[str, torch.Tensor], Dict[str, float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Transfer/materialize a batch while retaining an exact legacy branch."""
    timings = {
        "h2d_ms": 0.0,
        "sparse_condition_materialization_ms": 0.0,
        "query_materialization_ms": 0.0,
    }
    non_blocking = bool(data_path_config.non_blocking_transfer)
    _sync_cuda_for_diagnostic(device, diagnostic_step)
    start = time.perf_counter()

    if bool(batch.get("materialized_selected", False)):
        tensors = {
            key: batch[key].to(device, non_blocking=non_blocking)
            for key in (
                "coords_q", "fields_q", "obs_coords", "obs_values",
                "obs_mask", "obs_indices", "obs_field_ids",
            )
        }
        _sync_cuda_for_diagnostic(device, diagnostic_step)
        timings["h2d_ms"] = (time.perf_counter() - start) * 1000.0
        return tensors, timings, None, None

    fields_full = batch["fields"].to(device, non_blocking=non_blocking)
    if "coords_shared" in batch:
        coords_shared = batch["coords_shared"].to(device, non_blocking=non_blocking)
        coords_full = coords_shared.unsqueeze(0).expand(fields_full.shape[0], -1, -1)
    else:
        coords_full = batch["coords"].to(device, non_blocking=non_blocking)
    _sync_cuda_for_diagnostic(device, diagnostic_step)
    timings["h2d_ms"] = (time.perf_counter() - start) * 1000.0

    if "obs_layout" in batch:
        start = time.perf_counter()
        sparse = materialize_sparse_condition_from_layout(
            coords_full=coords_full,
            fields_full=fields_full,
            obs_layout=batch["obs_layout"],
        )
        _sync_cuda_for_diagnostic(device, diagnostic_step)
        timings["sparse_condition_materialization_ms"] = (time.perf_counter() - start) * 1000.0
        start = time.perf_counter()
        coords_q, fields_q = materialize_queries_from_full(
            coords_full, fields_full, batch["query_indices"]
        )
        _sync_cuda_for_diagnostic(device, diagnostic_step)
        timings["query_materialization_ms"] = (time.perf_counter() - start) * 1000.0
    else:
        # =====================================================================
        # LEGACY DATA PATH — BEGIN
        # Historical GPU sampling/materialization preserved for exact A/B runs.
        # =====================================================================
        start = time.perf_counter()
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords_full,
            fields_full=fields_full,
            cond_fields=cond_fields,
            n_obs_min=n_obs_min_list,
            n_obs_max=n_obs_max_list,
        )
        _sync_cuda_for_diagnostic(device, diagnostic_step)
        timings["sparse_condition_materialization_ms"] = (time.perf_counter() - start) * 1000.0
        sparse = {
            "obs_coords": obs_coords,
            "obs_values": obs_values,
            "obs_mask": obs_mask,
            "obs_indices": obs_indices,
            "obs_field_ids": obs_field_ids,
        }
        start = time.perf_counter()
        coords_q, fields_q, _ = sample_query_subset_legacy(
            coords=coords_full,
            fields=fields_full,
            n_query=n_query_points,
            mode=query_sampling,
            obs_coords=obs_coords,
            obs_mask=obs_mask,
            near_ratio=query_sample_near_ratio,
            far_ratio=query_sample_far_ratio,
            sigma_ratio=query_sample_sigma_ratio,
        )
        _sync_cuda_for_diagnostic(device, diagnostic_step)
        timings["query_materialization_ms"] = (time.perf_counter() - start) * 1000.0
        # =====================================================================
        # LEGACY DATA PATH — END
        # =====================================================================

    tensors = {"coords_q": coords_q, "fields_q": fields_q, **sparse}
    return tensors, timings, coords_full, fields_full


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
    data_path_config: Optional[ResolvedDataPathConfig] = None,
    diagnostics: Optional[DataPathDiagnostics] = None,
    train_query_microbatch_size: Optional[int] = None,
    reuse_condition_context_across_query_microbatches: bool = True,
    model_ema: Optional[ModelEMA] = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    if data_path_config is None:
        data_path_config = resolve_data_path_config({"data_path_mode": "legacy"})

    total = None
    count = 0
    mode_str = "Train" if training else "Eval"
    pbar = tqdm(loader, desc=f"Epoch {epoch:04d} [{mode_str}]", leave=False)
    iterator = iter(pbar)
    step = 0
    while True:
        total_step_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        loader_wait_ms = (time.perf_counter() - total_step_start) * 1000.0
        diagnostic_step = diagnostics is not None and diagnostics.should_sample(epoch, step)
        if diagnostic_step and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        effective_n_query = None if getattr(model, "requires_full_grid", False) else n_query_points
        sampling_mode = query_sampling if training else "uniform"
        tensors, prep_timings, _, _ = _prepare_training_tensors(
            batch=batch,
            device=device,
            data_path_config=data_path_config,
            diagnostic_step=diagnostic_step,
            cond_fields=cond_fields,
            n_obs_min_list=n_obs_min_list,
            n_obs_max_list=n_obs_max_list,
            n_query_points=effective_n_query,
            query_sampling=sampling_mode,
            query_sample_near_ratio=query_sample_near_ratio,
            query_sample_far_ratio=query_sample_far_ratio,
            query_sample_sigma_ratio=query_sample_sigma_ratio,
        )
        allocated_after_materialization_mb = (
            torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        )

        if training:
            optimizer.zero_grad(set_to_none=True)
        allocated_before_model_mb = (
            torch.cuda.memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
        )
        microbatch_active = (
            train_query_microbatch_size is not None
            and int(train_query_microbatch_size) > 0
            and int(train_query_microbatch_size) < int(tensors["coords_q"].shape[1])
            and hasattr(model, "training_loss_microbatched")
        )
        _sync_cuda_for_diagnostic(device, diagnostic_step)
        start = time.perf_counter()
        if microbatch_active:
            grad_context = torch.enable_grad() if training else torch.no_grad()
            with grad_context:
                loss, microbatch_metrics = model.training_loss_microbatched(
                    x1=tensors["fields_q"],
                    coords=tensors["coords_q"],
                    obs_coords=tensors["obs_coords"],
                    obs_values=tensors["obs_values"],
                    obs_mask=tensors["obs_mask"],
                    obs_field_ids=tensors["obs_field_ids"],
                    obs_indices=tensors["obs_indices"],
                    query_microbatch_size=int(train_query_microbatch_size),
                    backward=training,
                    reuse_condition_context=reuse_condition_context_across_query_microbatches,
                    synchronize_timing=diagnostic_step,
                )
        else:
            microbatch_metrics = {}
            loss, _ = model.training_loss(
                x1=tensors["fields_q"],
                coords=tensors["coords_q"],
                obs_coords=tensors["obs_coords"],
                obs_values=tensors["obs_values"],
                obs_mask=tensors["obs_mask"],
                obs_field_ids=tensors["obs_field_ids"],
                obs_indices=tensors["obs_indices"],
            )
        _sync_cuda_for_diagnostic(device, diagnostic_step)
        forward_ms = (time.perf_counter() - start) * 1000.0

        backward_ms = 0.0
        optimizer_ms = 0.0
        if training:
            start = time.perf_counter()
            if not microbatch_active:
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            _sync_cuda_for_diagnostic(device, diagnostic_step)
            backward_ms = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)
            _sync_cuda_for_diagnostic(device, diagnostic_step)
            optimizer_ms = (time.perf_counter() - start) * 1000.0

        detached = loss.detach()
        total = detached if total is None else total + detached
        count += 1
        if step % data_path_config.training_log_every_n_steps == 0:
            pbar.set_postfix_str(f"loss={float(detached):.6e}")

        if diagnostic_step:
            _sync_cuda_for_diagnostic(device, True)
            collate_timings = batch.get("data_path_timings", {})
            pre_model_ms = sum(float(collate_timings.get(key, 0.0)) for key in (
                "index_sampling_ms", "hdf5_read_ms", "cpu_normalization_ms", "cpu_materialization_ms"
            )) + sum(prep_timings.values())
            diagnostics.record({
                "epoch": int(epoch),
                "step": int(step),
                "data_path_mode": data_path_config.data_path_mode,
                "coord_batch_mode": data_path_config.coord_batch_mode,
                "index_sampling_mode": data_path_config.index_sampling_mode,
                "sampling_device": data_path_config.sampling_device,
                "field_read_mode": data_path_config.field_read_mode,
                "field_normalization_mode": data_path_config.field_normalization_mode,
                "gpu_transfer_mode": data_path_config.gpu_transfer_mode,
                "data_path_diag_storage_mode": data_path_config.data_path_diag_storage_mode,
                "batch_size": int(tensors["coords_q"].shape[0]),
                "N_full": int(batch.get("n_full", batch.get("fields", tensors["fields_q"]).shape[1])),
                "N_query": int(tensors["coords_q"].shape[1]),
                "N_obs_total": int(tensors["obs_mask"].sum()),
                "loader_wait_ms": loader_wait_ms,
                "index_sampling_ms": float(collate_timings.get("index_sampling_ms", 0.0)),
                "hdf5_read_ms": float(collate_timings.get("hdf5_read_ms", 0.0)),
                "cpu_normalization_ms": float(collate_timings.get("cpu_normalization_ms", 0.0)),
                "cpu_materialization_ms": float(collate_timings.get("cpu_materialization_ms", 0.0)),
                "h2d_ms": prep_timings["h2d_ms"],
                "sparse_condition_materialization_ms": prep_timings["sparse_condition_materialization_ms"],
                "query_materialization_ms": prep_timings["query_materialization_ms"],
                "pre_model_total_ms": pre_model_ms,
                "model_forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "query_microbatch_active": int(microbatch_active),
                "train_query_microbatch_size": (
                    int(train_query_microbatch_size) if microbatch_active else 0
                ),
                "rf_bridge_ms": float(microbatch_metrics.get("rf_bridge_ms", 0.0)),
                "condition_context_ms": float(microbatch_metrics.get("condition_context_ms", 0.0)),
                "query_chunk_forward_ms": float(microbatch_metrics.get("query_chunk_forward_ms", 0.0)),
                "query_chunk_backward_ms": float(microbatch_metrics.get("query_chunk_backward_ms", 0.0)),
                "query_microbatches": int(microbatch_metrics.get("query_microbatches", 1)),
                "optimizer_ms": optimizer_ms,
                "total_training_step_ms": (time.perf_counter() - total_step_start) * 1000.0,
                "allocated_after_materialization_mb": allocated_after_materialization_mb,
                "allocated_before_model_mb": allocated_before_model_mb,
                "gpu_peak_allocated_mb": (
                    torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0
                ),
                "gpu_peak_reserved_mb": (
                    torch.cuda.max_memory_reserved(device) / (1024 ** 2) if device.type == "cuda" else 0.0
                ),
            })
        # Release selected/full batch views before the next diagnostic resets
        # peak memory, otherwise the next step would measure overlap with stale
        # references from this iteration.
        del tensors, loss, detached
        step += 1

    if diagnostics is not None:
        diagnostics.flush()
        diagnostics.print_epoch_summary(epoch)
    return float(total.cpu()) / max(count, 1) if total is not None else float("nan")


def build_direct_coherence_config(args) -> DirectCoherenceConfig:
    return DirectCoherenceConfig(
        enabled=bool(args.direct_coherence_enabled),
        self_weight=float(args.coherence_self_weight),
        mutual_weight=float(args.coherence_mutual_weight),
        cross_weight=float(args.coherence_cross_weight),
        channel_weights=args.coherence_channel_weights,
        cross_num_directions=int(args.coherence_cross_num_directions),
        cross_top_frac=float(args.coherence_cross_top_frac),
        cross_seed=int(args.coherence_cross_seed),
        cross_include_axes=bool(args.coherence_cross_include_axes),
        cross_qmc=bool(args.coherence_cross_qmc),
        mutual_num_directions=int(args.coherence_mutual_num_directions),
        mutual_seed=int(args.coherence_mutual_seed),
        use_denorm=bool(args.coherence_use_denorm),
    )


def current_coherence_loss_weight(args, epoch: int) -> float:
    base = float(args.coherence_loss_weight)
    warmup = int(getattr(args, "coherence_weight_warmup_epochs", 0) or 0)
    if warmup <= 0 or epoch < int(args.coherence_start_epoch):
        return base
    progress = (epoch - int(args.coherence_start_epoch) + 1) / float(warmup)
    return base * min(max(progress, 0.0), 1.0)


def build_epoch_train_loader(
    train_set,
    args,
    epoch: int,
    data_path_config: Optional[ResolvedDataPathConfig] = None,
) -> DataLoader:
    """
    Build a fresh random subset loader for one epoch.

    `train_ratio` still defines the train/validation split.  This helper only
    down-samples the already-built training split for the current epoch.
    """
    ratio = min(max(float(getattr(args, "train_ratio_downsample", 1.0)), 0.0), 1.0)
    n_total = len(train_set)
    n_epoch = n_total if ratio >= 1.0 else max(1, int(math.ceil(n_total * ratio)))
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + int(epoch) * 1009)
    if n_epoch < n_total:
        indices = torch.randperm(n_total, generator=generator)[:n_epoch].tolist()
        epoch_set = Subset(train_set, indices)
    else:
        epoch_set = train_set
    if data_path_config is None:
        data_path_config = resolve_data_path_config({"data_path_mode": "legacy"})
    # The collator must retain the base dataset (for shared coordinates/HDF5),
    # while DataLoader may iterate a Subset of it.
    return build_pointcloud_loader(
        epoch_set,
        args,
        data_path_config,
        training=True,
        shuffle=True,
        generator=generator,
    )


def _grad_norm(model: nn.Module) -> float:
    grads = [
        p.grad.detach().reshape(-1)
        for p in model.parameters()
        if p.requires_grad and p.grad is not None
    ]
    if not grads:
        return float("nan")
    vec = torch.cat(grads)
    return float(torch.linalg.vector_norm(vec).detach().cpu())


def _mean_metric(rows: list[dict], key: str) -> float:
    vals = [row[key] for row in rows if key in row and row[key] is not None and math.isfinite(float(row[key]))]
    if not vals:
        return float("nan")
    return float(sum(float(v) for v in vals) / len(vals))


def run_epoch_direct_coherence(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cond_fields: Sequence[int],
    n_obs_min_list: Sequence[int],
    n_obs_max_list: Sequence[int],
    n_query_points: Optional[int],
    query_sampling: str,
    query_sample_near_ratio: float,
    query_sample_far_ratio: float,
    query_sample_sigma_ratio: float,
    direct_cfg: DirectCoherenceConfig,
    args,
    global_step: int,
    epoch: int,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    data_path_config: Optional[ResolvedDataPathConfig] = None,
    model_ema: Optional[ModelEMA] = None,
) -> tuple[dict, int]:
    model.train(True)
    if data_path_config is None:
        data_path_config = resolve_data_path_config({"data_path_mode": "legacy"})
    if data_path_config.gpu_transfer_mode == "selected_only":
        raise ValueError(
            "direct_coherence currently requires gpu_transfer_mode='legacy_full' because its "
            "coherence rollout consumes additional full/reference points."
        )
    loss_module = DirectGlobalCoherenceLoss(direct_cfg)
    rows = []
    applied = 0
    conflict_count = 0
    mode_str = "DirectCoherence"
    pbar = tqdm(loader, desc=f"Epoch {epoch:04d} [{mode_str}]", leave=False)

    for batch in pbar:
        effective_n_query = None if getattr(model, "requires_full_grid", False) else n_query_points
        tensors, _, coords_full, fields_full = _prepare_training_tensors(
            batch=batch,
            device=device,
            data_path_config=data_path_config,
            diagnostic_step=False,
            cond_fields=cond_fields,
            n_obs_min_list=n_obs_min_list,
            n_obs_max_list=n_obs_max_list,
            n_query_points=effective_n_query,
            query_sampling=query_sampling,
            query_sample_near_ratio=query_sample_near_ratio,
            query_sample_far_ratio=query_sample_far_ratio,
            query_sample_sigma_ratio=query_sample_sigma_ratio,
        )
        coords_q = tensors["coords_q"]
        fields_q = tensors["fields_q"]
        obs_coords = tensors["obs_coords"]
        obs_values = tensors["obs_values"]
        obs_mask = tensors["obs_mask"]
        obs_indices = tensors["obs_indices"]
        obs_field_ids = tensors["obs_field_ids"]
        data_loss, _ = model.training_loss(
            x1=fields_q,
            coords=coords_q,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            obs_indices=obs_indices,
        )

        global_step += 1
        every = max(1, int(args.coherence_every_n_steps))
        coherence_weight = current_coherence_loss_weight(args, epoch)
        coherence_active = (
            bool(direct_cfg.enabled)
            and int(epoch) >= int(args.coherence_start_epoch)
            and global_step % every == 0
            and coherence_weight > 0.0
        )

        row = {
            "data_loss": float(data_loss.detach().cpu()),
            "coherence_loss": float("nan"),
            "coherence_self": float("nan"),
            "coherence_mutual": float("nan"),
            "coherence_cross": float("nan"),
            "coherence_applied": 0.0,
            "data_grad_norm": float("nan"),
            "coherence_grad_norm": float("nan"),
            "gradient_cosine": float("nan"),
            "gradient_conflict": 0.0,
            "combined_grad_norm": float("nan"),
            "config_data_weight": float("nan"),
            "config_coherence_weight": float("nan"),
            "config_fallback_used": 0.0,
            "config_actual_used": 0.0,
            "config_aligned_used": 0.0,
            "global_step": float(global_step),
        }

        if not coherence_active:
            optimizer.zero_grad(set_to_none=True)
            total_loss = float(args.data_loss_weight) * data_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            row["data_grad_norm"] = _grad_norm(model)
            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)
            row["total_loss"] = float(total_loss.detach().cpu())
        else:
            bsz = coords_full.shape[0]
            cbsz = min(max(1, int(args.coherence_batch_size)), bsz)
            sel = torch.randperm(bsz, device=device)[:cbsz].sort().values

            coords_c = coords_full.index_select(0, sel)
            fields_c = fields_full.index_select(0, sel)
            obs_coords_c = obs_coords.index_select(0, sel)
            obs_values_c = obs_values.index_select(0, sel)
            obs_mask_c = obs_mask.index_select(0, sel)
            obs_field_ids_c = obs_field_ids.index_select(0, sel)
            obs_indices_c = obs_indices.index_select(0, sel) if obs_indices is not None else None

            if getattr(model, "requires_full_grid", False) or not bool(args.coherence_downsample):
                coords_coh, fields_ref, _ = coords_c, fields_c, None
            else:
                coords_coh, fields_ref, _ = sample_coherence_points(
                    coords=coords_c,
                    fields=fields_c,
                    n_points=args.coherence_n_points,
                )

            x_gen = differentiable_rf_rollout(
                ffm_model=model,
                coords=coords_coh,
                obs_coords=obs_coords_c,
                obs_values=obs_values_c,
                obs_mask=obs_mask_c,
                obs_field_ids=obs_field_ids_c,
                obs_indices=obs_indices_c,
                n_steps=args.coherence_rollout_steps,
                ode_solver=args.coherence_rollout_solver,
                obs_consistency_mode=args.coherence_obs_consistency_mode,
                obs_consistency_strength=args.coherence_obs_consistency_strength,
                obs_consistency_sigma=args.coherence_obs_consistency_sigma,
                obs_consistency_schedule_power=args.coherence_obs_consistency_schedule_power,
                obs_consistency_final_clamp=args.coherence_obs_consistency_final_clamp,
            )
            coherence_loss_raw, components = loss_module(
                x_gen=x_gen,
                x_ref=fields_ref,
                mean=mean,
                std=std,
            )
            coherence_loss_for_update = coherence_loss_raw
            if bool(args.coherence_interval_rescale):
                coherence_loss_for_update = coherence_loss_for_update * every

            scheduled_data_weight = float(args.data_loss_weight)
            scheduled_coherence_weight = float(coherence_weight)
            if args.gradient_balance_mode == "config":
                scheduled_data_weight *= float(args.config_data_grad_scale)
                scheduled_coherence_weight *= float(args.config_coherence_grad_scale)

            grad_info = apply_two_objective_update(
                model=model,
                optimizer=optimizer,
                data_loss=data_loss,
                coherence_loss=coherence_loss_for_update,
                mode=args.gradient_balance_mode,
                data_weight=scheduled_data_weight,
                coherence_weight=scheduled_coherence_weight,
                grad_clip_norm=1.0,
                config_missing_behavior=args.config_missing_behavior,
            )
            if model_ema is not None:
                model_ema.update(model)
            applied += 1
            if bool(grad_info.get("gradient_conflict", False)):
                conflict_count += 1
            total_for_log = float(args.data_loss_weight) * data_loss.detach() + coherence_weight * coherence_loss_for_update.detach()
            row.update({
                "total_loss": float(total_for_log.cpu()),
                "coherence_loss": float(coherence_loss_raw.detach().cpu()),
                "coherence_self": float(components["self_loss"].detach().cpu()),
                "coherence_mutual": float(components["mutual_loss"].detach().cpu()),
                "coherence_cross": float(components["cross_loss"].detach().cpu()),
                "coherence_applied": 1.0,
                "data_grad_norm": float(grad_info.get("data_grad_norm", float("nan"))),
                "coherence_grad_norm": float(grad_info.get("coherence_grad_norm", float("nan"))),
                "gradient_cosine": float(grad_info.get("gradient_cosine", float("nan"))),
                "gradient_conflict": 1.0 if bool(grad_info.get("gradient_conflict", False)) else 0.0,
                "combined_grad_norm": float(grad_info.get("combined_grad_norm", float("nan"))),
                "config_data_weight": float(grad_info.get("config_data_weight", float("nan"))),
                "config_coherence_weight": float(grad_info.get("config_coherence_weight", float("nan"))),
                "config_fallback_used": 1.0 if bool(grad_info.get("config_fallback_used", False)) else 0.0,
                "config_actual_used": 1.0 if grad_info.get("config_update_mode") == "config" else 0.0,
                "config_aligned_used": 1.0 if grad_info.get("config_update_mode") == "weighted_sum_aligned" else 0.0,
            })

        rows.append(row)
        pbar.set_postfix_str(
            f"data={row['data_loss']:.3e} coh={row['coherence_loss']:.3e} applied={int(row['coherence_applied'])}"
        )

    count = max(len(rows), 1)
    metrics = {
        "total_loss": _mean_metric(rows, "total_loss"),
        "data_loss": _mean_metric(rows, "data_loss"),
        "coherence_loss": _mean_metric(rows, "coherence_loss"),
        "coherence_self": _mean_metric(rows, "coherence_self"),
        "coherence_mutual": _mean_metric(rows, "coherence_mutual"),
        "coherence_cross": _mean_metric(rows, "coherence_cross"),
        "coherence_application_fraction": float(applied / count),
        "data_grad_norm": _mean_metric(rows, "data_grad_norm"),
        "coherence_grad_norm": _mean_metric(rows, "coherence_grad_norm"),
        "gradient_cosine": _mean_metric(rows, "gradient_cosine"),
        "gradient_conflict_fraction": float(conflict_count / max(applied, 1)),
        "combined_grad_norm": _mean_metric(rows, "combined_grad_norm"),
        "config_data_weight": _mean_metric(rows, "config_data_weight"),
        "config_coherence_weight": _mean_metric(rows, "config_coherence_weight"),
        "config_fallback_fraction": _mean_metric(rows, "config_fallback_used"),
        "config_actual_fraction": _mean_metric(rows, "config_actual_used"),
        "config_aligned_fraction": _mean_metric(rows, "config_aligned_used"),
        "global_step": float(global_step),
    }
    return metrics, global_step


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


def backup_artifact_to_dir(path: Path, backup_dir: Path) -> Optional[Path]:
    if not path.exists():
        return None

    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / path.name
    if target.exists():
        target = backup_path(target)
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)
    return target


def copy_file_if_different(src, dst) -> bool:
    src_path = Path(src).resolve()
    dst_path = Path(dst).resolve()
    if src_path == dst_path:
        return False
    shutil.copy2(src_path, dst_path)
    return True


def _coerce_history_value(value):
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return int(numeric) if numeric.is_integer() else numeric


def load_history_rows(csv_path: Path, max_epoch: Optional[int] = None) -> list[dict]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []

    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw_row in reader:
            if not raw_row:
                continue
            row = {key: _coerce_history_value(value) for key, value in raw_row.items()}
            epoch = row.get("epoch")
            if epoch is None:
                continue
            epoch = int(epoch)
            if max_epoch is not None and epoch > max_epoch:
                continue
            row["epoch"] = epoch
            rows.append(row)
    return rows


class TrainingHistoryLogger:
    def __init__(
        self,
        run_dir: Path,
        initial_rows: Optional[list[dict]] = None,
        plot_every_n_epochs: int = 1,
    ) -> None:
        self.csv_path = run_dir / "loss_history.csv"
        self.json_path = run_dir / "loss_history.json"
        self.plot_path = run_dir / "loss_history.png"
        self.plot_every_n_epochs = max(0, int(plot_every_n_epochs))
        self.rows = []
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "epoch", "train_loss", "val_loss",
                "train_seconds", "validation_seconds", "epoch_seconds",
            ])
            for raw_row in initial_rows or []:
                row = {
                    "epoch": int(raw_row["epoch"]),
                    "train_loss": None if raw_row.get("train_loss") is None else float(raw_row["train_loss"]),
                    "val_loss": None if raw_row.get("val_loss") is None else float(raw_row["val_loss"]),
                    "train_seconds": None if raw_row.get("train_seconds") is None else float(raw_row["train_seconds"]),
                    "validation_seconds": None if raw_row.get("validation_seconds") is None else float(raw_row["validation_seconds"]),
                    "epoch_seconds": None if raw_row.get("epoch_seconds") is None else float(raw_row["epoch_seconds"]),
                }
                self.rows.append(row)
                writer.writerow([
                    row["epoch"],
                    "" if row["train_loss"] is None else row["train_loss"],
                    "" if row["val_loss"] is None else row["val_loss"],
                    "" if row["train_seconds"] is None else row["train_seconds"],
                    "" if row["validation_seconds"] is None else row["validation_seconds"],
                    "" if row["epoch_seconds"] is None else row["epoch_seconds"],
                ])
        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(self.rows, handle, indent=2)
        if self.plot_every_n_epochs > 0:
            self._plot()

    def log_and_plot(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        train_seconds: Optional[float] = None,
        validation_seconds: Optional[float] = None,
        epoch_seconds: Optional[float] = None,
    ) -> None:
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
            "train_seconds": None if train_seconds is None else float(train_seconds),
            "validation_seconds": None if validation_seconds is None else float(validation_seconds),
            "epoch_seconds": None if epoch_seconds is None else float(epoch_seconds),
        }
        self.rows.append(row)

        with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                row["epoch"],
                row["train_loss"],
                "" if row["val_loss"] is None else row["val_loss"],
                "" if row["train_seconds"] is None else row["train_seconds"],
                "" if row["validation_seconds"] is None else row["validation_seconds"],
                "" if row["epoch_seconds"] is None else row["epoch_seconds"],
            ])
        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(self.rows, handle, indent=2)
        if self.plot_every_n_epochs > 0 and epoch % self.plot_every_n_epochs == 0:
            self._plot()

    def _plot(self) -> None:
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


class DirectCoherenceHistoryLogger:
    def __init__(self, run_dir: Path, initial_rows: Optional[list[dict]] = None) -> None:
        self.csv_path = run_dir / "direct_coherence_history.csv"
        self.json_path = run_dir / "direct_coherence_history.json"
        self.plot_path = run_dir / "direct_coherence_history.png"
        self.rows = []
        self.fieldnames = [
            "epoch",
            "train_total_loss",
            "train_data_loss",
            "train_coherence_loss",
            "coherence_self",
            "coherence_mutual",
            "coherence_cross",
            "coherence_application_fraction",
            "data_grad_norm",
            "coherence_grad_norm",
            "gradient_cosine",
            "gradient_conflict_fraction",
            "combined_grad_norm",
            "config_data_weight",
            "config_coherence_weight",
            "config_fallback_fraction",
            "config_actual_fraction",
            "config_aligned_fraction",
            "lr",
            "global_step",
        ]
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            for raw_row in initial_rows or []:
                row = {key: raw_row.get(key) for key in self.fieldnames}
                row["epoch"] = int(row["epoch"])
                self.rows.append(row)
                writer.writerow(row)
        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(self.rows, handle, indent=2)
        self._plot()

    def log(self, epoch: int, metrics: dict, lr: float, global_step: int) -> None:
        row = {
            "epoch": int(epoch),
            "train_total_loss": metrics.get("total_loss", float("nan")),
            "train_data_loss": metrics.get("data_loss", float("nan")),
            "train_coherence_loss": metrics.get("coherence_loss", float("nan")),
            "coherence_self": metrics.get("coherence_self", float("nan")),
            "coherence_mutual": metrics.get("coherence_mutual", float("nan")),
            "coherence_cross": metrics.get("coherence_cross", float("nan")),
            "coherence_application_fraction": metrics.get("coherence_application_fraction", float("nan")),
            "data_grad_norm": metrics.get("data_grad_norm", float("nan")),
            "coherence_grad_norm": metrics.get("coherence_grad_norm", float("nan")),
            "gradient_cosine": metrics.get("gradient_cosine", float("nan")),
            "gradient_conflict_fraction": metrics.get("gradient_conflict_fraction", float("nan")),
            "combined_grad_norm": metrics.get("combined_grad_norm", float("nan")),
            "config_data_weight": metrics.get("config_data_weight", float("nan")),
            "config_coherence_weight": metrics.get("config_coherence_weight", float("nan")),
            "config_fallback_fraction": metrics.get("config_fallback_fraction", float("nan")),
            "config_actual_fraction": metrics.get("config_actual_fraction", float("nan")),
            "config_aligned_fraction": metrics.get("config_aligned_fraction", float("nan")),
            "lr": float(lr),
            "global_step": int(global_step),
        }
        self.rows.append(row)
        with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writerow(row)
        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(self.rows, handle, indent=2)
        self._plot()

    @staticmethod
    def _series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
        xs = []
        ys = []
        for row in rows:
            value = row.get(key)
            if value is None:
                continue
            value = float(value)
            if not math.isfinite(value) or value <= 0.0:
                continue
            xs.append(int(row["epoch"]))
            ys.append(value)
        return xs, ys

    def _plot(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        left_specs = [
            ("train_total_loss", "Total", "#1f77b4"),
            ("train_data_loss", "Data", "#2ca02c"),
            ("train_coherence_loss", "Physical coherence", "#d62728"),
        ]
        for key, label, color in left_specs:
            xs, ys = self._series(self.rows, key)
            if xs:
                axes[0].plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=label, color=color)
        axes[0].set_title("Direct Post-Training Losses")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True, which="both", ls="--", alpha=0.4)
        if axes[0].lines:
            axes[0].set_yscale("log")
            axes[0].legend()

        right_specs = [
            ("coherence_self", "Self", "#9467bd"),
            ("coherence_mutual", "Mutual", "#ff7f0e"),
            ("coherence_cross", "Cross", "#17becf"),
        ]
        for key, label, color in right_specs:
            xs, ys = self._series(self.rows, key)
            if xs:
                axes[1].plot(xs, ys, marker="o", markersize=3, linewidth=1.6, label=label, color=color)
        axes[1].set_title("Physical Coherence Components")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].grid(True, which="both", ls="--", alpha=0.4)
        if axes[1].lines:
            axes[1].set_yscale("log")
            axes[1].legend()

        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)


def resolve_checkpoint_path(source_run: Path, checkpoint: str) -> Path:
    ckpt_path = Path(str(checkpoint))
    if ckpt_path.suffix != ".pt":
        ckpt_path = ckpt_path.with_suffix(".pt")
    if not ckpt_path.is_absolute():
        ckpt_path = source_run / ckpt_path
    return ckpt_path


def resolve_pretrained_checkpoint(demo_dir: str, args, reload_ckpt=None) -> tuple[Optional[Path], Optional[Path]]:
    if bool(args.RELOAD) and isinstance(reload_ckpt, dict) and reload_ckpt.get("source_run_dir"):
        source_run = Path(reload_ckpt["source_run_dir"])
        if not source_run.is_absolute():
            source_run = Path(demo_dir) / source_run
        if not source_run.exists():
            raise FileNotFoundError(f"Reload checkpoint source_run_dir does not exist: {source_run}")
        source_checkpoint = reload_ckpt.get("source_checkpoint")
        if source_checkpoint in (None, ""):
            return source_run, None
        ckpt_path = Path(source_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = source_run / ckpt_path
        return source_run, ckpt_path

    if str(args.initialization) != "pretrained":
        return None, None

    source_run = find_source_run_dir(
        demo_dir=demo_dir,
        source_run_dir=args.pretrained_run_dir,
        source_Demo_Num=args.pretrained_source_Demo_Num,
        save_dir_hint=args.save_dir,
    )
    ckpt_path = resolve_checkpoint_path(source_run, args.pretrained_checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    return source_run, ckpt_path


SOURCE_BASE_CONFIG_KEYS = (
    # Dataset/split identity. Keep these aligned with the pretrained model's
    # normalization statistics and split semantics.
    "data",
    "train_ratio",
    "time_stride",

    # Backbone and architecture shape.
    "backbone",
    "hidden_dim",
    "cond_dim",
    "field_embed_dim",
    "rbf_sigma",
    "sigma_min",
    "USE_FOURIER_PE",
    "fourier_pe_num_bands",
    "fourier_pe_max_freq",
    "latent_dim",
    "num_latents",
    "num_heads",
    "num_latent_blocks",
    "ff_mult",
    "attn_dropout",
    "mlp_dropout",
    "decode_chunk_size",
    "share_query_proj",
    "summary_type",
    "gather_mode",
    "gather_topk",
    "gather_query_chunk_size",
    "learnable_rbf_sigma",
    "neighbor_backend",
    "sensor_local_topk",
    "sensor_local_dropout",
    "sensor_coord_encoding",
    "latent_sensor_reinject",
    "latent_reinject_every",
    "condition_attention_execution",
    "sensor_attention_padding_mode",
    "sensor_attention_buckets",
    "query_latent_readout",
    "query_readout_type",
    "query_readout_scale_init",
    "enhanced_head_norm",
    "glres_scale_init",

    # FNO architecture keys.
    "Num_x",
    "Num_y",
    "fno_modes_x",
    "fno_modes_y",
    "fno_hidden_channels",
    "fno_n_layers",
    "condition_blur",
    "condition_blur_kernel",
    "condition_blur_sigma",

    # Base RF query/prior choices.
    "n_query_points",
    "query_sampling",
    "query_sample_near_ratio",
    "query_sample_far_ratio",
    "query_sample_sigma_ratio",
    "prior",
    "rff_features",
    "rff_lengthscale",

    # Sparse conditioning and generation/evaluation defaults.
    "cond_field",
    "n_obs_min",
    "n_obs_max",
    "cond_fields",
    "n_obs_min_list",
    "n_obs_max_list",
    "vis_cond_fields",
    "vis_n_obs_list",
    "ode_solver",
    "benchmark_n_steps",
    "n_steps_generation",
)


def apply_pretrained_source_base_config(args, source_run_dir: Optional[Path]) -> list[str]:
    """
    In pretrained post-training, prefer the source run's base model/data/
    conditioning config for checkpoint compatibility.

    Post-training controls such as Demo_Num, device, optimizer, epochs,
    coherence schedules, and loss weights remain from the direct-posttrain
    config/CLI.
    """
    if source_run_dir is None or not bool(getattr(args, "pretrained_use_source_base_config", True)):
        return []

    source_cfg = load_source_config(source_run_dir)
    inherited = []
    for key in SOURCE_BASE_CONFIG_KEYS:
        if key not in source_cfg or not hasattr(args, key):
            continue
        setattr(args, key, source_cfg[key])
        inherited.append(key)
    return inherited


def checkpoint_metadata(args, direct_cfg: Optional[DirectCoherenceConfig], source_run_dir, source_checkpoint) -> dict:
    ablation = getattr(args, "ablation", None)
    if isinstance(ablation, dict) and ablation.get("enabled") and ablation.get("id") == "A1":
        method = "deterministic_direct_field_mse"
    elif str(args.training_mode) == "direct_coherence":
        method = "direct_coherence_rectified_flow"
    else:
        method = "1_rectified_flow"
    metadata = {
        "method": method,
        "training_mode": args.training_mode,
        "initialization": args.initialization,
        "source_run_dir": None if source_run_dir is None else str(source_run_dir),
        "source_checkpoint": None if source_checkpoint is None else str(source_checkpoint),
        "data_loss_weight": float(args.data_loss_weight),
        "coherence_loss_weight": float(args.coherence_loss_weight),
        "gradient_balance_mode": args.gradient_balance_mode,
        "coherence_config": None if direct_cfg is None else direct_cfg.to_dict(),
        "pretrained_use_source_base_config": bool(getattr(args, "pretrained_use_source_base_config", True)),
        "pretrained_inherited_base_config_keys": list(getattr(args, "pretrained_inherited_base_config_keys", [])),
    }
    if isinstance(ablation, dict) and ablation.get("enabled"):
        from model_ablation import ablation_metadata

        metadata["ablation"] = ablation_metadata(vars(args))
    return metadata


def architecture_compatibility_hint(args, source_run_dir: Optional[Path]) -> str:
    if source_run_dir is None:
        return ""
    try:
        source_cfg = load_source_config(source_run_dir)
    except Exception as exc:
        return f"\nCould not read source config from {source_run_dir}: {exc}"

    keys = [
        "backbone",
        "hidden_dim",
        "cond_dim",
        "field_embed_dim",
        "USE_FOURIER_PE",
        "fourier_pe_num_bands",
        "fourier_pe_max_freq",
        "latent_dim",
        "num_latents",
        "num_heads",
        "num_latent_blocks",
        "summary_type",
        "gather_mode",
        "gather_topk",
        "gather_query_chunk_size",
        "learnable_rbf_sigma",
        "sensor_coord_encoding",
        "latent_sensor_reinject",
        "condition_attention_execution",
        "sensor_attention_padding_mode",
        "sensor_attention_buckets",
        "query_latent_readout",
        "query_readout_type",
        "query_readout_scale_init",
        "enhanced_head_norm",
        "glres_scale_init",
        "cq_query_dim",
        "cq_readout_mode",
        "cq_fusion_mode",
        "cq_readout_rank",
        "cq_readout_heads",
        "cq_global_scale_init",
        "cq_local_scale_init",
        "cq_readout_scale_init",
        "cq_time_conditioning",
        "cq_time_embed_dim",
        "cq_time_max_period",
        "cq_time_film_zero_init",
        "cq_measurement_support_mode",
        "cq_measurement_support_normalize",
        "model_ema_enabled",
        "model_ema_decay",
        "model_ema_eval",
    ]
    current = vars(args)
    lines = ["\nArchitecture keys that commonly affect checkpoint compatibility:"]
    for key in keys:
        src = source_cfg.get(key, "<missing>")
        cur = current.get(key, "<missing>")
        if src != cur:
            lines.append(f"  {key}: source={src!r}, current={cur!r}")
    if len(lines) == 1:
        lines.append("  No obvious architecture-key differences found in saved config.")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None):

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(raw_argv)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    demo_dir = os.path.dirname(script_dir) # Go up one level to \demo

    # Public command-line overrides are deliberately applied after YAML. This
    # preserves the historical YAML-first engine while making the default
    # entry point convenient and unambiguous for collaborators.
    overrides = _set_overrides(args.set)
    dedicated_overrides = {
        "data": ("--data",),
        "dataset_stats_path": ("--dataset-stats-path",),
        "save_dir": ("--save-dir",),
        "device_ids": ("--device-ids",),
        "Demo_Num": ("--Demo-Num", "--demo-num"),
    }
    for key, names in dedicated_overrides.items():
        if _option_was_supplied(raw_argv, *names):
            overrides[key] = getattr(args, key)
    
    # YAML Loading and Backup
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = Path(demo_dir) / config_path
    config_path = config_path.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_snapshot_path = None
    config_snapshot_payload = None
    
    if config_path.exists():
        print(f"\n[*] Starting:... I found config file at: {config_path}\n")
        with config_path.open("r", encoding="utf-8") as f:
            source_yaml = yaml.safe_load(f) or {}
        if not isinstance(source_yaml, dict):
            raise TypeError(f"Expected a YAML mapping in {config_path}.")

        candidate = dict(source_yaml)
        candidate.update(overrides)
        is_gl_profile = candidate.get("model_name") is not None or candidate.get(
            "backbone"
        ) in {"GL_rbf", "GL_rbf_ENH", "GL_rbf_ENH_CQ"}
        if is_gl_profile:
            yaml_config = load_public_config(
                config_path,
                overrides=overrides,
                root=demo_dir,
            )
        else:
            yaml_config = candidate
        
        # Overwrite default args with YAML values
        for key, value in yaml_config.items():
            if hasattr(args, key):
                setattr(args, key, value)
            elif key != "config_source":
                print(f"Warning: YAML key '{key}' is not a recognized argument. Ignoring.")
        args = normalize_conditioning_args(args)
        if args.model_name is not None or args.backbone in {
            "GL_rbf", "GL_rbf_ENH", "GL_rbf_ENH_CQ"
        }:
            identity = resolve_model_identity(vars(args))
            args.model_name = identity.public_name
            args.backbone = identity.internal_backbone
                    
        # Backup the YAML file
        backup_dir = os.path.join(demo_dir, "Save_config", "pointcloud_ffm")
        os.makedirs(backup_dir, exist_ok=True)
        backup_filename = f"config_pointcloud_ffm_DemoN{args.Demo_Num}_{timestamp}.yaml"
        config_snapshot_path = Path(backup_dir) / backup_filename
        config_snapshot_payload = yaml_config
    else:
        print(f"\n[Warning: !] Config file not found at {config_path}. Using default parameters.\n")
        args.Demo_Num = 0  # Force Demo_Num to 0 as default

    if args.dry_run:
        build_config = dict(vars(args))
        build_config["neighbor_backend"] = "torch"
        torch.manual_seed(int(args.seed))
        model = build_pointcloud_model(build_config, n_fields=5, device="cpu")
        print(
            json.dumps(
                {
                    "status": "valid",
                    "model_name": args.model_name,
                    "backbone": args.backbone,
                    "coord_dim": args.coord_dim,
                    "Demo_Num": args.Demo_Num,
                    "device_ids": args.device_ids,
                    "epochs": args.epochs,
                    "scheduler_t_max": args.scheduler_t_max,
                    "condition_attention_execution": args.condition_attention_execution,
                    "sensor_attention_padding_mode": args.sensor_attention_padding_mode,
                    "state_key_count": len(model.state_dict()),
                    "model_schema_sha256": _model_schema_digest(model),
                    "data": args.data,
                    "dataset_stats_path": args.dataset_stats_path,
                    "save_dir": args.save_dir,
                    "ablation": getattr(model, "ablation_metadata", None),
                    "trainable_parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                },
                indent=2,
            )
        )
        return

    if config_snapshot_path is not None:
        config_snapshot_path.write_text(
            yaml.safe_dump(config_snapshot_payload, sort_keys=False), encoding="utf-8"
        )
        print(f"[*] Resolved config backed up to: {config_snapshot_path}\n")
    
    # Setup the Dynamic Directories with Demo_Num
    set_seed(args.seed)

    start_epoch = 1
    best_val = float("inf")
    reload_ckpt = None
    reload_checkpoint_path = None
    resume_backup_dir = None
    global_step = 0
    run_timestamp = timestamp
    save_dir = Path(os.path.join(demo_dir, args.save_dir + f"_DemoN{args.Demo_Num}" + f"_{timestamp}"))

    if args.RELOAD:
        latest_run_dir = find_latest_run_dir(demo_dir=demo_dir, save_dir=args.save_dir, demo_num=args.Demo_Num)
        if latest_run_dir is not None:
            reload_checkpoint_path = latest_run_dir / "last.pt"
            if not reload_checkpoint_path.exists():
                reload_checkpoint_path = latest_run_dir / "best.pt"
            if not reload_checkpoint_path.exists():
                reload_checkpoint_path = None

        if reload_checkpoint_path is not None:
            save_dir = latest_run_dir
            run_timestamp = extract_run_timestamp(latest_run_dir, args.save_dir, args.Demo_Num)
            reload_ckpt = torch.load(reload_checkpoint_path, map_location="cpu", weights_only=False)
            start_epoch = int(reload_ckpt.get("epoch", 0)) + 1
            best_val = float(reload_ckpt.get("val_loss", float("inf")))
            best_checkpoint_path = latest_run_dir / "best.pt"
            if best_checkpoint_path.exists() and best_checkpoint_path != reload_checkpoint_path:
                best_ckpt = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
                best_val = min(best_val, float(best_ckpt.get("val_loss", float("inf"))))
            global_step = int(reload_ckpt.get("global_step", 0))

            print(f"[*] RELOAD=True, resuming from: {reload_checkpoint_path}")
            print(f"[*] Resume will start from epoch {start_epoch}\n")
        else:
            print("[*] RELOAD=True, but no matching last.pt or best.pt was found. Training will start from scratch.\n")

    source_run_dir, source_checkpoint = resolve_pretrained_checkpoint(demo_dir, args, reload_ckpt=reload_ckpt)
    if source_run_dir is not None:
        args.source_run_dir = str(source_run_dir)
        args.source_checkpoint = None if source_checkpoint is None else str(source_checkpoint)
        inherited_keys = apply_pretrained_source_base_config(args, source_run_dir)
        args.pretrained_inherited_base_config_keys = inherited_keys
        args = normalize_conditioning_args(args)
        if reload_ckpt is not None:
            print(f"[*] RELOAD source base config run: {source_run_dir}")
            if source_checkpoint is not None:
                print(f"[*] RELOAD original source checkpoint: {source_checkpoint}")
        else:
            print(f"[*] initialization=pretrained, source run: {source_run_dir}")
            print(f"[*] initialization=pretrained, checkpoint: {source_checkpoint}")
        if inherited_keys:
            print(
                "[*] Inherited base model/data/conditioning keys from source run: "
                + ", ".join(inherited_keys)
            )
        else:
            print("[*] No base config keys inherited from source run.")
    else:
        args.pretrained_inherited_base_config_keys = []

    data_path_config = resolve_data_path_config(args)
    args = apply_resolved_data_path_config(args, data_path_config)
    if args.training_mode == "direct_coherence" and data_path_config.gpu_transfer_mode == "selected_only":
        raise ValueError(
            "direct_coherence needs full reference fields. Set gpu_transfer_mode='legacy_full'; "
            "shared coordinates and CPU/scalable index sampling remain available as hybrid ablations."
        )
    print_resolved_data_path_config(data_path_config)

    save_dir.mkdir(parents=True, exist_ok=True)
    recon_dir = save_dir / "Evaluation"
    resume_checkpoint_epoch = int(reload_ckpt.get("epoch", 0)) if reload_ckpt is not None else None
    initial_history_rows = []
    initial_direct_history_rows = []

    if args.RELOAD and reload_ckpt is not None:
        resume_backup_dir = save_dir / "bk" / f"resume_{timestamp}"
        for artifact in (
            "last.pt",
            "best.pt",
            "args.json",
            "run_config.yaml",
            "loss_history.csv",
            "loss_history.json",
            "loss_history.png",
            "direct_coherence_history.csv",
            "direct_coherence_history.json",
            "direct_coherence_history.png",
        ):
            backup_artifact_to_dir(save_dir / artifact, resume_backup_dir)
        backup_artifact_to_dir(recon_dir, resume_backup_dir)
        print(f"[*] Pre-resume artifacts backed up to: {resume_backup_dir}")

        initial_history_rows = load_history_rows(save_dir / "loss_history.csv", max_epoch=resume_checkpoint_epoch)
        initial_direct_history_rows = load_history_rows(
            save_dir / "direct_coherence_history.csv",
            max_epoch=resume_checkpoint_epoch,
        )
        print(f"[*] Active loss history restored through epoch {resume_checkpoint_epoch}.")

    # Save the final parsed args to a JSON in the model folder just to be safe
    with open(save_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    if config_snapshot_path is not None:
        config_copied = copy_file_if_different(
            config_snapshot_path, save_dir / "run_config.yaml"
        )
        if not config_copied:
            print("[*] Config already lives in run folder; skipping run_config.yaml copy.")

    # Keep all run artifacts under the model directory, matching the unified
    # baseline trainers. The old Save_loss_csv/ and Save_reconstruction_files/
    # roots are no longer used by this trainer.

    # Initialize helpers
    logger = TrainingHistoryLogger(
        save_dir,
        initial_rows=initial_history_rows,
        plot_every_n_epochs=args.training_history_plot_every_n_epochs,
    )
    direct_cfg = build_direct_coherence_config(args)
    direct_logger = (
        DirectCoherenceHistoryLogger(save_dir, initial_rows=initial_direct_history_rows)
        if args.training_mode == "direct_coherence"
        else None
    )
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
        field_names=args.FIELD_NAMES if args.FIELD_NAMES is not None else args.field_names,
        stats_path=args.dataset_stats_path or str(save_dir / "dataset_stats.pt"),
        coord_batch_mode=data_path_config.coord_batch_mode,
        defer_field_read=(data_path_config.sampling_device == "cpu"),
        instrument_data_path=data_path_config.data_path_diagnostics,
    )
    val_set = TurbulentCombustionH5Dataset(
        args.data,
        split="val",
        train_ratio=args.train_ratio,
        seed=args.seed,
        time_stride=args.time_stride,
        field_names=args.FIELD_NAMES if args.FIELD_NAMES is not None else args.field_names,
        stats_path=args.dataset_stats_path or str(save_dir / "dataset_stats.pt"),
        coord_batch_mode=data_path_config.coord_batch_mode,
        defer_field_read=(data_path_config.sampling_device == "cpu"),
        instrument_data_path=data_path_config.data_path_diagnostics,
    )
    hdf5_layout = train_set.hdf5_layout()
    print(
        "[*] HDF5 fields layout: "
        f"shape={hdf5_layout['shape']} chunks={hdf5_layout['chunks']} "
        f"dtype={hdf5_layout['dtype']} compression={hdf5_layout['compression']}"
    )
    train_loader = build_pointcloud_loader(
        train_set,
        args,
        data_path_config,
        training=True,
        shuffle=True,
    )
    val_loader = build_pointcloud_loader(
        val_set,
        args,
        data_path_config,
        training=False,
        shuffle=False,
    )
    data_path_diagnostics = DataPathDiagnostics(save_dir, data_path_config)

    prior = IIDGaussianPrior() if args.prior == "iid" else RFFGaussianPrior(
        coord_dim=args.coord_dim, n_features=args.rff_features, lengthscale=args.rff_lengthscale
    )

    if args.backbone == "mlp_rbf":
        backbone = ConditionalPointMLPRBF(
            n_fields=train_set.num_fields,
            coord_dim=args.coord_dim,
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
            coord_dim=args.coord_dim,
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
            use_fourier_pe=args.USE_FOURIER_PE,
            fourier_pe_num_bands=args.fourier_pe_num_bands,
            fourier_pe_max_freq=args.fourier_pe_max_freq,
        )
        model = PointCloudFFM(backbone, prior, sigma_min=args.sigma_min).to(device)
    elif args.backbone == "GL_rbf_ENH_CQ":
        sensor_coord_encoding = args.sensor_coord_encoding or "fourier"
        latent_sensor_reinject = (
            True if args.latent_sensor_reinject is None else args.latent_sensor_reinject
        )
        glres_scale_init = (
            1.0e-2 if args.glres_scale_init is None else args.glres_scale_init
        )
        print(
            "[*] GL_rbf_ENH_CQ settings: "
            f"query_dim={args.cq_query_dim}, "
            f"readout_mode={args.cq_readout_mode}, "
            f"fusion_mode={args.cq_fusion_mode}, "
            f"readout_rank={args.cq_readout_rank}, "
            f"readout_heads={args.cq_readout_heads}, "
            f"sensor_coord_encoding={sensor_coord_encoding}, "
            f"latent_sensor_reinject={latent_sensor_reinject}, "
            f"glres_scale_init={glres_scale_init}, "
            f"time_conditioning={args.cq_time_conditioning}, "
            f"measurement_support={args.cq_measurement_support_mode}, "
            f"condition_attention={args.condition_attention_execution}, "
            f"sensor_padding={args.sensor_attention_padding_mode}, "
            f"ema={args.model_ema_enabled}, "
            f"ema_decay={args.model_ema_decay}"
        )
        model = build_pointcloud_model(
            vars(args),
            n_fields=train_set.num_fields,
            device=device,
            prior_override=prior,
        )
    elif args.backbone in ["GL_rbf", "GL_rbf_ENH"]:
        enhanced = args.backbone == "GL_rbf_ENH"

        # Enhanced defaults are resolved here so legacy GL_rbf configs/checkpoints
        # keep their original point-readout and zero-scale behavior.
        sensor_coord_encoding = args.sensor_coord_encoding
        if sensor_coord_encoding is None:
            sensor_coord_encoding = "fourier" if enhanced else "raw"

        latent_sensor_reinject = args.latent_sensor_reinject
        if latent_sensor_reinject is None:
            latent_sensor_reinject = enhanced

        query_latent_readout = args.query_latent_readout
        if query_latent_readout is None:
            query_latent_readout = enhanced

        enhanced_head_norm = args.enhanced_head_norm
        if enhanced_head_norm is None:
            enhanced_head_norm = enhanced

        query_readout_type = args.query_readout_type
        if query_readout_type is None:
            query_readout_type = "coord" if enhanced else "point"

        query_readout_scale_init = args.query_readout_scale_init
        if query_readout_scale_init is None:
            query_readout_scale_init = 1.0e-2 if enhanced else 0.0

        glres_scale_init = args.glres_scale_init
        if glres_scale_init is None:
            glres_scale_init = 1.0e-2 if enhanced else 0.0

        print(
            "[*] GL_rbf settings: "
            f"enhanced={enhanced}, "
            f"sensor_coord_encoding={sensor_coord_encoding}, "
            f"latent_sensor_reinject={latent_sensor_reinject}, "
            f"latent_reinject_every={args.latent_reinject_every}, "
            f"query_latent_readout={query_latent_readout}, "
            f"query_readout_type={query_readout_type}, "
            f"query_readout_scale_init={query_readout_scale_init}, "
            f"enhanced_head_norm={enhanced_head_norm}, "
            f"glres_scale_init={glres_scale_init}"
        )

        model = build_pointcloud_model(
            vars(args),
            n_fields=train_set.num_fields,
            device=device,
            prior_override=prior,
        )
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.scheduler_t_max or args.epochs))
    model_ema = ModelEMA(model, decay=args.model_ema_decay) if args.model_ema_enabled else None

    if reload_ckpt is not None:
        model.load_state_dict(checkpoint_model_state(reload_ckpt, prefer_ema=False))
        if "optimizer" in reload_ckpt:
            optimizer.load_state_dict(reload_ckpt["optimizer"])
        if "scheduler" in reload_ckpt:
            scheduler.load_state_dict(reload_ckpt["scheduler"])
        if model_ema is not None:
            if "model_ema" in reload_ckpt:
                model_ema.load_state_dict(reload_ckpt["model_ema"])
            else:
                model_ema = ModelEMA(model, decay=args.model_ema_decay)
        print(f"[*] Reloaded model state from {reload_checkpoint_path.name}")
    elif source_checkpoint is not None:
        pretrained_ckpt = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
        pretrained_state = checkpoint_model_state(pretrained_ckpt)
        try:
            load_result = model.load_state_dict(pretrained_state, strict=bool(args.pretrained_strict))
        except RuntimeError as exc:
            raise RuntimeError(
                "Pretrained checkpoint is architecture-incompatible with the current model. "
                "Check backbone/hidden dimensions/conditioning architecture or set pretrained_strict=false "
                "only if the mismatch is intentional."
                f"{architecture_compatibility_hint(args, source_run_dir)}"
            ) from exc
        if not bool(args.pretrained_strict):
            print(f"[*] Pretrained load missing keys: {load_result.missing_keys}")
            print(f"[*] Pretrained load unexpected keys: {load_result.unexpected_keys}")
        if bool(args.pretrained_load_optimizer) and isinstance(pretrained_ckpt, dict) and "optimizer" in pretrained_ckpt:
            optimizer.load_state_dict(pretrained_ckpt["optimizer"])
            print("[*] Loaded optimizer state from pretrained checkpoint")
        print("[*] Loaded pretrained model state; new run starts at epoch 1")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_wall_start = time.perf_counter()
        train_wall_start = time.perf_counter()
        if args.training_mode == "direct_coherence":
            epoch_train_loader = build_epoch_train_loader(
                train_set, args, epoch, data_path_config=data_path_config
            )
            direct_metrics, global_step = run_epoch_direct_coherence(
                model=model,
                loader=epoch_train_loader,
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
                direct_cfg=direct_cfg,
                args=args,
                global_step=global_step,
                epoch=epoch,
                mean=train_set.mean.to(device),
                std=train_set.std.to(device),
                data_path_config=data_path_config,
                model_ema=model_ema,
            )
            tr_loss = float(direct_metrics["total_loss"])
        else:
            direct_metrics = None
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
                data_path_config=data_path_config,
                diagnostics=data_path_diagnostics,
                train_query_microbatch_size=args.train_query_microbatch_size,
                reuse_condition_context_across_query_microbatches=(
                    args.reuse_condition_context_across_query_microbatches
                ),
                model_ema=model_ema,
            )
            global_step += len(train_loader)
        train_seconds = time.perf_counter() - train_wall_start
        scheduler.step()

        if direct_metrics is not None:
            print(
                f"[train] epoch={epoch:04d} total={direct_metrics['total_loss']:.6e} "
                f"data={direct_metrics['data_loss']:.6e} coherence={direct_metrics['coherence_loss']:.6e} "
                f"self={direct_metrics['coherence_self']:.3e} mutual={direct_metrics['coherence_mutual']:.3e} "
                f"cross={direct_metrics['coherence_cross']:.3e} "
                f"applied={direct_metrics['coherence_application_fraction']:.2%}"
            )
            if direct_logger is not None:
                direct_logger.log(
                    epoch=epoch,
                    metrics=direct_metrics,
                    lr=optimizer.param_groups[0]["lr"],
                    global_step=global_step,
                )
        else:
            print(f"[train] epoch={epoch:04d} loss={tr_loss:.6e}")
        val_loss = None
        validation_seconds = 0.0
        if epoch % args.eval_every == 0 or epoch == 1:
            validation_wall_start = time.perf_counter()
            eval_weights = (
                model_ema.average_parameters(model)
                if model_ema is not None and args.model_ema_eval else nullcontext()
            )
            with eval_weights, torch.no_grad():
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
                    data_path_config=data_path_config,
                    diagnostics=data_path_diagnostics,
                    train_query_microbatch_size=args.train_query_microbatch_size,
                    reuse_condition_context_across_query_microbatches=(
                        args.reuse_condition_context_across_query_microbatches
                    ),
                )
            validation_seconds = time.perf_counter() - validation_wall_start
            print(f"[valid] epoch={epoch:04d} loss={val_loss:.6e}")

            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "mean": train_set.mean,
                "std": train_set.std,
                "field_names": train_set.field_names,
                "backbone": args.backbone,
                "summary_type": args.summary_type,
                "ode_solver": args.ode_solver,
                "Num_x": args.Num_x,
                "Num_y": args.Num_y,
            }
            if model_ema is not None:
                ckpt["model_ema"] = model_ema.state_dict()
            ckpt["model_ema_enabled"] = bool(args.model_ema_enabled)
            ckpt["model_ema_decay"] = float(args.model_ema_decay)
            ckpt["model_ema_eval"] = bool(args.model_ema_eval)
            ckpt["best_selection_weights"] = (
                "ema" if model_ema is not None and args.model_ema_eval else "live"
            )
            if args.backbone == "GL_rbf_ENH_CQ":
                ckpt.update({
                    "cq_query_dim": args.cq_query_dim,
                    "cq_readout_mode": args.cq_readout_mode,
                    "cq_fusion_mode": args.cq_fusion_mode,
                    "cq_readout_rank": args.cq_readout_rank,
                    "cq_readout_heads": args.cq_readout_heads,
                    "cq_global_scale_init": args.cq_global_scale_init,
                    "cq_local_scale_init": args.cq_local_scale_init,
                    "cq_readout_scale_init": args.cq_readout_scale_init,
                    "cq_time_conditioning": args.cq_time_conditioning,
                    "cq_time_embed_dim": args.cq_time_embed_dim,
                    "cq_time_max_period": args.cq_time_max_period,
                    "cq_time_film_zero_init": args.cq_time_film_zero_init,
                    "cq_measurement_support_mode": args.cq_measurement_support_mode,
                    "cq_measurement_support_normalize": args.cq_measurement_support_normalize,
                })
            ckpt.update(checkpoint_metadata(args, direct_cfg, source_run_dir, source_checkpoint))
            torch.save(ckpt, save_dir / "last.pt")
            if epoch in set(args.checkpoint_epochs or []):
                torch.save(ckpt, save_dir / f"epoch_{epoch:04d}.pt")
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

                recon_weights = (
                    model_ema.average_parameters(model)
                    if model_ema is not None and args.model_ema_eval else nullcontext()
                )
                with recon_weights:
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
                        reconstruction_execution_mode=args.reconstruction_execution_mode,
                        reconstruction_query_chunk_size=args.reconstruction_query_chunk_size,
                        reconstruction_cache_level=args.reconstruction_cache_level,
                    )

                metric_str = ", ".join([f"{k}:{v:.4e}" for k, v in recon_metrics.items()])
                print(f"[recon] epoch={epoch:04d} solver={args.ode_solver} n_steps={nfe} | {metric_str}")

        logger.log_and_plot(
            epoch=epoch,
            train_loss=tr_loss,
            val_loss=val_loss,
            train_seconds=train_seconds,
            validation_seconds=validation_seconds,
            epoch_seconds=time.perf_counter() - epoch_wall_start,
        )

    print("Training complete.")
    print(f"Best validation loss: {best_val:.6e}")


if __name__ == "__main__":
    main()

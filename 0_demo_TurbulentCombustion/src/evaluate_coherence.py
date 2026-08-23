"""
Evaluate data-driven physical coherence terms on trained reconstruction models.

Current supported mode:
    - global_dist : global state-distribution discrepancy

Design goal:
    This script is written to be extensible.
    In the future, other coherence terms (e.g. spectral or topological) can be added to coherence_dist.py
    and exposed here through the same command-line interface and save-directory pattern:

        Save_PhyCoEval/{coherence_mode}/{timestamp}/

Quick usage
-----------
Run from `0_demo_TurbulentCombustion/`.

The most convenient path is to provide only a checkpoint. The script will infer
whether it is a main FFM checkpoint, a unified generative baseline, or a unified
deterministic baseline from `run_config.yaml`, the embedded checkpoint config,
or the run directory name:

python src/evaluate_coherence.py \
  --checkpoint-path ./Save_TrainedModel/Baseline_geofno_Stage1_DemoN0_20260528_100704/best.pt \
  --coherence-mode global_dist \
  --snapshot-indices 0

Deterministic baselines trained by `train_Det_Baseline.py` are supported:
    mlp_rbf, senseiver, geofno

Generative baselines trained by `train_Gen_Baseline.py` are supported:
    latent_fm, s3gm, sit

Main FFM checkpoints trained by `train_pointcloud_ffm.py` are also supported.

Usage details
-------------
1) Single-snapshot diagnostic mode

python src/evaluate_coherence.py \
  --checkpoint-path <checkpoint_or_run_dir_or_pt_file> \
  --coherence-mode global_dist \
  --snapshot-indices 0 \
  --n-obs-list 256 256 \
  --n-steps 2

This reconstructs the requested snapshot(s) and saves detailed per-snapshot coherence plots and metrics.
By default, global_dist now uses a deterministic fixed-bank top-k sliced
Wasserstein joint score:

  --joint-method topk_swd \
  --num-directions 64 \
  --joint-top-frac 0.10 \
  --include-pairwise-in-score

The older adaptive projection-pursuit score is still available:

  --joint-method adaptive_maxswd

Pairwise 2D SWD remains available as a diagnostic and can be included in the
overall mode_score with --include-pairwise-in-score or excluded with
--no-include-pairwise-in-score. All global_dist scores are discrepancies, so
lower is better.

2) Statistical-analysis mode

python src/evaluate_coherence.py \
  --checkpoint ./Save_TrainedModel/ffm_tc_pointcloud_DemoN15_20260501_124557/last.pt \
  --coherence-mode global_dist \
  --Statistical-Analysis CoL2_Scatter Aggregate_Violin \
  --statistical-split test \
  --stat-max-snapshots 100 \
  --n-obs-list 256 256 \
  --n-steps 2

This scans many frames from the selected split, suppresses individual snapshot plots, and saves:
    coherence_l2_scatter.png/pdf
    aggregate_violin.png/pdf
    statistical_coherence_table.csv
    statistical_analysis_summary.json

The script is intentionally evaluation-only.

Checkpoint families
-------------------
By default `--baseline-model auto` infers the model family. You can still force
one of:
    pointcloud_ffm, latent_fm, s3gm, sit, mlp_rbf, senseiver, geofno

Baseline loading intentionally reuses model_baseline.py so this evaluator does
not duplicate architecture/checkpoint construction logic from the unified
generative and deterministic baseline trainers/evaluators.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    import model_baseline as baseline_lib
except Exception:
    baseline_lib = None

# -----------------------------------------------------------------------------
# Local imports
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = SCRIPT_DIR.parent
REPO_DIR = DEMO_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from coherence_dist import (
    GlobalDistConfig,
    coherence_result_to_scalars,
    compute_coherence,
    project_channels,
)

# Prefer the generalized helpers if available.
from helpers import (
    FIELD_NAMES,
    TurbulentCombustionH5Dataset,
    build_sparse_condition,
)

try:
    from helpers import reconstruct_snapshot as helpers_reconstruct_snapshot
except Exception:
    helpers_reconstruct_snapshot = None

# Model imports. The current repo layout keeps model code in src/Model.py.
from Model import (
    ConditionalPointMLPRBF,
    ConditionalPointPerceiver,
    ConditionalPointHybridLocalGlobalRBF,
    ConditionalPointHybridLocalGlobalRBFCQ,
    PointCloudFFM,
)
try:
    from Model import FNO, FNOFFM
except ImportError:
    FNO = None
    FNOFFM = None

# Older or intermediate repos may not export priors from Model.py. We provide
# local fallback definitions so this script stays runnable.
try: 
    from Model import IIDGaussianPrior, RFFGaussianPrior 
except Exception:
    import math

    class IIDGaussianPrior(nn.Module):
        def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
            bsz, n_pts, _ = coords.shape
            return torch.randn(bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype)

    class RFFGaussianPrior(nn.Module):
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

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate physical coherence diagnostics on trained checkpoints.")

    p.add_argument("--coherence-mode", type=str, default="global_dist",
                   help="Which coherence evaluator to use. Currently supports 'global_dist' as Global Distribution Coherence.")
    p.add_argument("--coherence-space", type=str, default="normalized", choices=["normalized", "physical"],
                   help="Whether to evaluate coherence in normalized model space or physical-unit space.")

    p.add_argument("--checkpoint", type=str, required=False,
                   help="Path to a checkpoint file (.pt) or to a run directory containing best.pt/last.pt.")
    p.add_argument("--checkpoint-path", type=str, default=None,
                   help="Alias used by baseline evaluators; overrides --checkpoint when provided.")
    p.add_argument("--baseline-model", type=str, default="auto",
                   choices=["auto", "pointcloud_ffm", "latent_fm", "s3gm", "sit", "mlp_rbf", "senseiver", "geofno"],
                   help=(
                       "Checkpoint family. Default 'auto' infers from run_config.yaml, checkpoint config, or run name. "
                       "Use pointcloud_ffm for train_pointcloud_ffm.py checkpoints; latent_fm/s3gm/sit for "
                       "train_Gen_Baseline.py checkpoints; mlp_rbf/senseiver/geofno for train_Det_Baseline.py checkpoints."
                   ))
    p.add_argument("--data", type=str, default=None,
                   help="Dataset path. Defaults to the training run's args.json value.")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                   help="Dataset split used for evaluation.")
    p.add_argument("--snapshot-indices", type=int, nargs="+", default=[0],
                   help="Snapshot indices within the chosen split to reconstruct and evaluate.")
    p.add_argument("--Statistical-Analysis", "--statistical-analysis",
                   dest="statistical_analysis",
                   type=str,
                   nargs="+",
                   default=None,
                   choices=["CoL2_Scatter", "Aggregate_Violin", "both"],
                   help=(
                       "Enable dataset-level statistical analysis. Values: "
                       "CoL2_Scatter, Aggregate_Violin, or both. In this mode "
                       "--snapshot-indices is ignored and per-snapshot plotting is disabled."
                   ))
    p.add_argument("--statistical-split", type=str, default="test", choices=["train", "test", "all"],
                   help="Split to scan in statistical-analysis mode.")
    p.add_argument("--stat-max-snapshots", type=int, default=None,
                   help="Optional maximum number of frames per split to process in statistical mode.")
    p.add_argument("--stat-stride", type=int, default=1,
                   help="Process every stat_stride-th frame in statistical mode.")
    p.add_argument("--stat-seed", type=int, default=None,
                   help="Random seed for statistical subsampling. Defaults to --seed.")
    p.add_argument("--stat-save-csv", action="store_true", default=True,
                   help="Save the raw statistical coherence table as CSV when statistical mode is active.")
    p.add_argument("--stat-plot-style", type=str, default="paper", choices=["paper", "default"],
                   help="Plot style for statistical aggregate figures.")
    p.add_argument("--stat-plot-formats", type=str, nargs="+", default=["png", "pdf", "svg"],
                   choices=["png", "pdf", "svg"],
                   help="File formats for statistical aggregate figures.")
    p.add_argument("--stat-dpi", type=int, default=400,
                   help="Raster DPI for statistical PNG figures.")
    p.add_argument("--stat-log-y", action=argparse.BooleanOptionalAction, default=True,
                   help="Use a log y-axis for aggregate violin plots when all plotted values are positive.")
    p.add_argument("--stat-panel-labels", action=argparse.BooleanOptionalAction, default=True,
                   help="Add panel labels such as '(a)' and '(b)' to statistical figures.")
    p.add_argument("--stat-show-regression", action=argparse.BooleanOptionalAction, default=True,
                   help="Add a least-squares trend line to the Coherence-L2 scatter when enough samples exist.")
    p.add_argument("--stat-color-by", type=str, default="obs_consistency_error",
                   choices=["obs_consistency_error", "relative_l2", "none"],
                   help="Metric used to color the Coherence-L2 scatter points.")
    p.add_argument("--stat-jitter-alpha", type=float, default=0.55,
                   help="Point alpha for aggregate violin/raincloud jittered samples.")
    p.add_argument("--stat-point-size", type=float, default=26.0,
                   help="Point size for the Coherence-L2 scatter.")

    p.add_argument("--cond-fields", type=int, nargs="+", default=None,
                   help="Override conditioned field ids used at evaluation time if wanted.")
    p.add_argument("--n-obs-list", type=int, nargs="+", default=None,
                   help="Exact number of sensors per conditioned field at evaluation time.")
    p.add_argument("--n-steps", type=int, default=None,
                   help="Generation steps. Defaults to n_steps_generation from args.json if available.")
    p.add_argument("--ode-solver", type=str, default=None,
                   help="Optional ODE solver name for RF checkpoints that expose it.")
    
    p.add_argument("--save-root", type=str, default=None,
                   help="Root folder for coherence evaluations. Defaults to <demo_dir>/Save_PhyCoEval.")
    p.add_argument("--device", type=str, default=None,
                   help="Torch device string. Defaults to cuda:0 if available, else cpu.")
    p.add_argument("--seed", type=int, default=42)

    # Global distribution coherence hyperparameters
    p.add_argument("--lambda-marg", type=float, default=1.0)
    p.add_argument("--lambda-joint", type=float, default=1.0)
    p.add_argument("--joint-method", type=str, default="topk_swd", choices=["topk_swd", "adaptive_maxswd"],
                   help="Joint channel-space discrepancy: deterministic fixed-bank top-k SWD or adaptive Max-SWD.")
    p.add_argument("--joint-top-frac", type=float, default=0.10,
                   help="Fraction of largest fixed-bank projected W2 values averaged by --joint-method topk_swd.")
    p.add_argument("--joint-qmc", action=argparse.BooleanOptionalAction, default=True,
                   help="Use Sobol quasi-random directions for the fixed projection bank when available.")
    p.add_argument("--include-axes", action=argparse.BooleanOptionalAction, default=True,
                   help="Include canonical channel axes at the start of the fixed projection bank.")
    p.add_argument("--lambda-pairwise", type=float, default=0.25,
                   help="Weight for pairwise_mean when pairwise diagnostics are included in mode_score.")
    p.add_argument("--include-pairwise-in-score", action=argparse.BooleanOptionalAction, default=True,
                   help="Add lambda_pairwise * pairwise_mean into mode_score when pairwise diagnostics are enabled.")
    p.add_argument("--exclude-axis-projections-when-marginal-included", action=argparse.BooleanOptionalAction, default=True,
                   help="When canonical axes are in the top-k projection bank, exclude them from joint scoring if lambda_marg is nonzero.")
    p.add_argument("--num-directions", type=int, default=None,
                   help="Number of joint channel directions. Defaults depend on --joint-method.")
    p.add_argument("--n-iter-theta", type=int, default=None,
                   help="Inner optimization steps for adaptive_maxswd directions. Defaults to an auto choice.")
    p.add_argument("--lr-theta", type=float, default=None,
                   help="Inner optimization step size for adaptive_maxswd directions. Defaults to an auto choice.")
    p.add_argument("--ortho-reg", type=float, default=None,
                   help="Orthogonality regularization for adaptive_maxswd directions. Defaults to an auto choice.")
    p.add_argument("--n-proj-pairwise", type=int, default=None,
                   help="Random projection count for pairwise 2D SWD. Defaults to an auto choice.")
    p.add_argument("--disable-pairwise", action="store_true")
    return p.parse_args()

# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_list(x: Optional[Sequence[int] | int]) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]


def broadcast_per_field(
    values: Sequence[int] | int | None,
    fields: Sequence[int],
    name: str,
    source_fields: Sequence[int] | int | None = None,
) -> List[int]:
    vals = ensure_list(values)
    fields = ensure_list(fields)
    if len(vals) == 0:
        raise ValueError(f"{name} cannot be empty once fields are specified.")
    if len(vals) == 1:
        vals = vals * len(fields)
    if len(vals) != len(fields):
        source = ensure_list(source_fields)
        if source and len(vals) == len(source):
            value_by_field = {field: value for field, value in zip(source, vals)}
            if all(field in value_by_field for field in fields):
                return [value_by_field[field] for field in fields]
        if len(fields) == 1 and all(value == vals[0] for value in vals):
            return [vals[0]]
        raise ValueError(f"{name} must have length 1 or match len(cond_fields). Got {len(vals)} vs {len(fields)}")
    return vals


def normalize_conditioning_args_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mirror the logic used in the training script for generalized conditioning.
    """
    d = dict(d)
    if d.get("cond_fields") is None:
        d["cond_fields"] = [int(d.get("cond_field", 2))]
    if d.get("n_obs_min_list") is None:
        d["n_obs_min_list"] = [int(d.get("n_obs_min", 64))]
    if d.get("n_obs_max_list") is None:
        d["n_obs_max_list"] = [int(d.get("n_obs_max", 256))]
    if d.get("vis_cond_fields") is None:
        d["vis_cond_fields"] = list(d["cond_fields"])
    if d.get("vis_n_obs_list") is None:
        d["vis_n_obs_list"] = list(d["n_obs_max_list"])
    return d


def cfg_get_not_none(cfg: Dict[str, Any], key: str, default: Any) -> Any:
    value = cfg.get(key, default)
    return default if value is None else value


def _candidate_base_dirs(*extra_dirs: Optional[Path]) -> List[Path]:
    seen: set[Path] = set()
    bases: List[Path] = []
    for raw_base in [Path.cwd(), SCRIPT_DIR, DEMO_DIR, REPO_DIR, *extra_dirs]:
        if raw_base is None:
            continue
        base = Path(raw_base).expanduser().resolve()
        if base in seen:
            continue
        seen.add(base)
        bases.append(base)
    return bases


def resolve_input_path(
    path_str: str,
    *,
    label: str,
    extra_base_dirs: Optional[Sequence[Path]] = None,
) -> Path:
    """
    Resolve a path provided on the CLI or loaded from args.json.

    For relative paths, try the current working directory first, then the script,
    demo, and repo roots, plus any supplied extra bases. This makes the script
    robust to being launched from either `src/` or the demo root.
    """
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(f"{label} path not found: {resolved}")

    attempted: List[Path] = []
    for base in _candidate_base_dirs(*(extra_base_dirs or [])):
        candidate = (base / raw_path).resolve()
        attempted.append(candidate)
        if candidate.exists():
            return candidate

    attempted_msg = "\n  - ".join(str(p) for p in attempted)
    raise FileNotFoundError(
        f"{label} path not found: {path_str}\n"
        f"Tried:\n  - {attempted_msg}"
    )


def resolve_output_path(
    path_str: str,
    *,
    extra_base_dirs: Optional[Sequence[Path]] = None,
) -> Path:
    """
    Resolve an output path without requiring it to exist yet.
    """
    raw_path = Path(path_str).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    base_dirs = _candidate_base_dirs(*(extra_base_dirs or []))
    return (base_dirs[0] / raw_path).resolve()


def infer_demo_dir(run_dir: Path) -> Path:
    """
    Infer the demo root from a run directory like:
      <demo_dir>/Save_TrainedModel/<run_name>
    """
    if run_dir.parent.name == "Save_TrainedModel":
        return run_dir.parent.parent
    return DEMO_DIR


def choose_checkpoint(path_str: str) -> Tuple[Path, Path]:
    """
    Resolve a user-supplied checkpoint argument.

    Returns:
        checkpoint_path, run_dir
    """
    path = resolve_input_path(path_str, label="Checkpoint")
    if path.is_file():
        return path, path.parent

    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    # Prefer best.pt, then last.pt, then any .pt file.
    candidates = [path / "best.pt", path / "last.pt"]
    for ckpt in candidates:
        if ckpt.exists():
            return ckpt, path

    pts = sorted(path.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt checkpoint files found under: {path}")
    return pts[0], path


def resolve_latentfm_stage1_checkpoint(
    *,
    cfg: Dict[str, Any],
    checkpoint: Optional[Dict[str, Any]],
    run_dir: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    """Resolve a Latent-FM VAE checkpoint, including a portable archive copy.

    Stage-2 checkpoints record the stage-1 path used during training.  That
    absolute path is informative but is often invalid after a model archive is
    relocated.  A complete archived Latent-FM family can instead provide its
    shared VAE in a sibling ``Stage0`` (or legacy ``Stage1``) directory.  The
    returned pair is ``(resolved_path, recorded_path)`` so callers can report
    when the portable fallback was required.  No architecture is inferred from
    the directory: the normal baseline adapter still validates and loads the
    checkpoint itself.
    """
    if str(cfg.get("baseline_model", "")).lower() != "latent_fm" or int(cfg.get("training_stage", 1)) != 2:
        return None, None

    stage2_cfg = cfg.get("latent_fm_params", {}).get("stage2", {})
    embedded = checkpoint.get("ae_checkpoint") if isinstance(checkpoint, dict) else None
    configured = stage2_cfg.get("stage1_checkpoint")
    recorded_value = embedded or configured
    recorded_path = Path(recorded_value).expanduser() if recorded_value else None

    candidates: list[Path] = []
    if recorded_path is not None:
        candidates.extend((recorded_path, run_dir / recorded_path, DEMO_DIR / recorded_path))

    # ``Stage0`` is the portable archive convention for the shared VAE.  Keep
    # ``Stage1`` as a compatibility alias for older archives.
    for stage_dir in ("Stage0", "Stage1"):
        candidates.extend((run_dir.parent / stage_dir / "last.pt", run_dir.parent / stage_dir / "best.pt"))

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            canonical = candidate.resolve()
        except OSError:
            canonical = candidate
        if canonical in seen:
            continue
        seen.add(canonical)
        if canonical.is_file():
            return canonical, recorded_path
    return None, recorded_path


def infer_checkpoint_family(
    checkpoint_path: Path,
    run_dir: Path,
    checkpoint: Optional[Dict[str, Any]] = None,
) -> str:
    run_cfg_path = run_dir / "run_config.yaml"
    if run_cfg_path.exists():
        try:
            with open(run_cfg_path, "r", encoding="utf-8") as handle:
                run_cfg = yaml.safe_load(handle) or {}
            baseline_name = str(run_cfg.get("baseline_model", "")).strip().lower()
            if baseline_name:
                return baseline_name
        except Exception:
            pass

    if isinstance(checkpoint, dict):
        baseline_name = str(checkpoint.get("baseline_model", "")).strip().lower()
        if baseline_name:
            return baseline_name
        embedded_cfg = checkpoint.get("config")
        if isinstance(embedded_cfg, dict):
            baseline_name = str(embedded_cfg.get("baseline_model", "")).strip().lower()
            if baseline_name:
                return baseline_name
        if checkpoint.get("backbone") is not None or checkpoint.get("method") == "1_rectified_flow":
            return "pointcloud_ffm"

    run_name = run_dir.name.lower()
    match = re.match(r"baseline_([^_]+(?:_[^_]+)*)_stage\d+_", run_name)
    if match:
        candidate = match.group(1)
        for known in ("latent_fm", "s3gm", "sit", "mlp_rbf", "senseiver", "geofno"):
            if candidate == known:
                return known
    if run_name.startswith("ffm_tc_pointcloud") or "pointcloud_ffm" in run_name:
        return "pointcloud_ffm"

    raise ValueError(
        "Could not infer checkpoint family automatically. Pass --baseline-model explicitly "
        f"for checkpoint: {checkpoint_path}"
    )


def checkpoint_arg_from_args(args: argparse.Namespace) -> str:
    checkpoint_arg = args.checkpoint_path or args.checkpoint
    if checkpoint_arg is None:
        raise ValueError("Pass --checkpoint for pointcloud FFM or --checkpoint-path/--checkpoint for baseline checkpoints.")
    return str(checkpoint_arg)


def load_run_args(run_dir: Path) -> Dict[str, Any]:
    args_json = run_dir / "args.json"
    if args_json.exists():
        with open(args_json, "r") as f:
            d = json.load(f)
        return normalize_conditioning_args_dict(d)
    raise FileNotFoundError(f"args.json not found in run directory: {run_dir}")


def _extract_timestamp(path: Path) -> Optional[str]:
    m = re.search(r"DemoN(\d+)_(\d{8}_\d{6})", path.name)
    if m is None:
        m = re.search(r"demo_N(\d+)_(\d{8}_\d{6})", path.name)
    return m.group(2) if m else None


def _extract_demo_num(path: Path) -> Optional[int]:
    m = re.search(r"DemoN(\d+)", path.name)
    if m is None:
        m = re.search(r"demo_N(\d+)", path.name)
    return int(m.group(1)) if m else None


def load_run_config(run_dir: Path) -> Dict[str, Any]:
    """
    Prefer the backed-up YAML config used by evaluate_ffm.py so model
    reconstruction stays identical across evaluators. Fall back to args.json
    when the YAML cannot be located.
    """
    run_cfg_path = run_dir / "run_config.yaml"
    if run_cfg_path.exists():
        with open(run_cfg_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        if "baseline_model" not in cfg:
            return normalize_conditioning_args_dict(cfg)

    demo_dir = infer_demo_dir(run_dir)
    cfg_dir = demo_dir / "Save_config" / "pointcloud_ffm"
    train_timestamp = _extract_timestamp(run_dir)
    demo_num = _extract_demo_num(run_dir)

    if train_timestamp is not None and demo_num is not None:
        yaml_path = cfg_dir / f"config_pointcloud_ffm_DemoN{demo_num}_{train_timestamp}.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            return normalize_conditioning_args_dict(cfg)

    return load_run_args(run_dir)


def resolve_effective_pointcloud_config(
    run_dir: Path,
    checkpoint: Any,
    yaml_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Recover the effective configuration used to build a PointCloudFFM checkpoint.

    Standard runs can be rebuilt directly from run_config.yaml.  Pretrained
    direct-coherence runs are different: their YAML is the launch template and
    may retain fallback base-model/data values, while args.json is written after
    source-run inheritance and therefore records the resolved configuration that
    actually produced the checkpoint.

    If a legacy inherited checkpoint has no args.json, reconstruct only the
    explicitly recorded inherited keys from the source run.  Never silently use
    the launch-template fallbacks for an inherited checkpoint.
    """
    cfg = normalize_conditioning_args_dict(dict(yaml_cfg))
    if not isinstance(checkpoint, dict):
        return cfg

    def reconcile_checkpoint_architecture(candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Repair stale Perceiver positional-encoding YAML from state metadata.

        Older archived PointCloudFFM runs can carry a launch-template YAML that
        disagrees with the serialized Perceiver module.  The positional encoder
        owns a registered ``model.pos_enc.freqs`` buffer, so its presence is an
        unambiguous checkpoint-level architecture signal.  Only reconcile this
        one state-defined option; all other user configuration remains intact.
        """
        state = checkpoint.get("model")
        if str(candidate.get("backbone", "")).lower() != "perceiver" or not isinstance(state, dict):
            return normalize_conditioning_args_dict(candidate)
        freq_key = "model.pos_enc.freqs"
        checkpoint_uses_fourier = freq_key in state
        configured_uses_fourier = bool(candidate.get("USE_FOURIER_PE", False))
        if configured_uses_fourier != checkpoint_uses_fourier:
            candidate = dict(candidate)
            candidate["USE_FOURIER_PE"] = checkpoint_uses_fourier
            if checkpoint_uses_fourier:
                candidate["fourier_pe_num_bands"] = int(state[freq_key].numel())
            print(
                "[*] Perceiver checkpoint/YAML positional-encoding mismatch; "
                f"using checkpoint metadata (USE_FOURIER_PE={checkpoint_uses_fourier})."
            )
        return normalize_conditioning_args_dict(candidate)

    inherited_keys = [str(key) for key in checkpoint.get("pretrained_inherited_base_config_keys", [])]
    training_mode = str(checkpoint.get("training_mode", cfg.get("training_mode", "standard"))).lower()
    initialization = str(checkpoint.get("initialization", cfg.get("initialization", "scratch"))).lower()
    uses_source_base = bool(
        checkpoint.get(
            "pretrained_use_source_base_config",
            cfg.get("pretrained_use_source_base_config", False),
        )
    )
    inherited_run = bool(inherited_keys) or (
        training_mode == "direct_coherence"
        and initialization == "pretrained"
        and uses_source_base
    )
    if not inherited_run:
        return reconcile_checkpoint_architecture(cfg)

    args_path = run_dir / "args.json"
    if args_path.exists():
        effective_cfg = load_run_args(run_dir)
        print(
            "[*] Inherited PointCloudFFM checkpoint detected; rebuilding from "
            "the resolved effective configuration in args.json."
        )
        return reconcile_checkpoint_architecture(effective_cfg)

    if not inherited_keys:
        raise FileNotFoundError(
            f"Inherited checkpoint under {run_dir} has no args.json and does not record "
            "pretrained_inherited_base_config_keys, so its effective architecture cannot be recovered safely."
        )

    source_run_value = checkpoint.get("source_run_dir") or cfg.get("pretrained_run_dir")
    if not source_run_value:
        raise FileNotFoundError(
            f"Inherited checkpoint under {run_dir} has no args.json and no source_run_dir metadata."
        )
    source_run = resolve_input_path(
        str(source_run_value),
        label="Pretrained source run",
        extra_base_dirs=[run_dir, infer_demo_dir(run_dir)],
    )
    if not source_run.is_dir():
        raise FileNotFoundError(f"Pretrained source run is not a directory: {source_run}")
    source_cfg = load_run_args(source_run) if (source_run / "args.json").exists() else load_run_config(source_run)
    missing = [key for key in inherited_keys if key not in source_cfg]
    if missing:
        raise KeyError(
            "Cannot reconstruct inherited PointCloudFFM configuration; source run is missing keys: "
            + ", ".join(missing)
        )
    for key in inherited_keys:
        cfg[key] = source_cfg[key]
    print(
        "[*] Inherited PointCloudFFM checkpoint detected; args.json is unavailable, "
        "so recorded inherited keys were recovered from the source run."
    )
    return reconcile_checkpoint_architecture(cfg)


def build_prior(cfg: Dict[str, Any]) -> nn.Module:
    prior_name = str(cfg.get("prior", "rff"))
    if prior_name == "iid":
        return IIDGaussianPrior()
    return RFFGaussianPrior(
        coord_dim=int(cfg.get("coord_dim", 3)),
        n_features=int(cfg.get("rff_features", 256)),
        lengthscale=float(cfg.get("rff_lengthscale", 0.15)),
    )

def build_model(cfg: Dict[str, Any], dataset: TurbulentCombustionH5Dataset) -> nn.Module:
    """
    Mirror evaluate_ffm.py model reconstruction so checkpoints are loaded
    against the same architecture and hyperparameters used offline there.
    """
    prior = build_prior(cfg)
    backbone_name = str(cfg.get("backbone", "mlp_rbf"))

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
            use_fourier_pe=cfg.get("USE_FOURIER_PE", False),
            fourier_pe_num_bands=cfg.get("fourier_pe_num_bands", 32),
            fourier_pe_max_freq=cfg.get("fourier_pe_max_freq", 64.0),
        )
        return PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))

    if backbone_name == "fno":
        if FNO is None or FNOFFM is None:
            raise RuntimeError("Config says backbone='fno' but FNO/FNOFFM are not available in Model.py")
        backbone = FNO(
            n_fields=dataset.num_fields,
            Num_x=cfg.get("Num_x"),
            Num_y=cfg.get("Num_y"),
            n_modes_x=cfg.get("fno_modes_x", 32),
            n_modes_y=cfg.get("fno_modes_y", 8),
            hidden_channels=cfg.get("fno_hidden_channels", 64),
            n_layers=cfg.get("fno_n_layers", 4),
            condition_blur=cfg.get("condition_blur", False),
            condition_blur_kernel=cfg.get("condition_blur_kernel", 5),
            condition_blur_sigma=cfg.get("condition_blur_sigma", 1.0),
        )
        return FNOFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))

    if backbone_name == "GL_rbf_ENH_CQ":
        backbone = ConditionalPointHybridLocalGlobalRBFCQ(
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
            gather_mode=cfg.get("gather_mode", "topk_rbf_glres"),
            gather_topk=cfg.get("gather_topk", 32),
            gather_query_chunk_size=cfg.get("gather_query_chunk_size", None),
            learnable_rbf_sigma=cfg.get("learnable_rbf_sigma", False),
            neighbor_backend=cfg.get("neighbor_backend", "torch"),
            sensor_local_topk=cfg.get("sensor_local_topk", 32),
            sensor_local_dropout=cfg.get("sensor_local_dropout", 0.0),
            use_fourier_pe=cfg.get("USE_FOURIER_PE", False),
            fourier_pe_num_bands=cfg.get("fourier_pe_num_bands", 32),
            fourier_pe_max_freq=cfg.get("fourier_pe_max_freq", 64.0),
            sensor_coord_encoding=cfg_get_not_none(
                cfg, "sensor_coord_encoding", "fourier"
            ),
            latent_sensor_reinject=cfg_get_not_none(
                cfg, "latent_sensor_reinject", True
            ),
            latent_reinject_every=cfg.get("latent_reinject_every", 1),
            glres_scale_init=cfg_get_not_none(cfg, "glres_scale_init", 1e-2),
            cq_query_dim=cfg.get("cq_query_dim", 128),
            cq_readout_mode=cfg.get("cq_readout_mode", "lowrank"),
            cq_fusion_mode=cfg.get("cq_fusion_mode", "additive"),
            cq_readout_rank=cfg.get("cq_readout_rank", 64),
            cq_readout_heads=cfg.get("cq_readout_heads", 4),
            cq_global_scale_init=cfg.get("cq_global_scale_init", 1.0),
            cq_local_scale_init=cfg.get("cq_local_scale_init", 1.0),
            cq_readout_scale_init=cfg.get("cq_readout_scale_init", 1e-2),
        )
        return PointCloudFFM(
            backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4)
        )

    if backbone_name in {"GL_rbf", "GL_rbf_ENH", "hybrid_localglobal_rbf"}:
        enhanced = backbone_name == "GL_rbf_ENH"
        sensor_coord_encoding = cfg_get_not_none(cfg, "sensor_coord_encoding", "fourier" if enhanced else "raw")
        latent_sensor_reinject = cfg_get_not_none(cfg, "latent_sensor_reinject", enhanced)
        query_latent_readout = cfg_get_not_none(cfg, "query_latent_readout", enhanced)
        enhanced_head_norm = cfg_get_not_none(cfg, "enhanced_head_norm", enhanced)
        query_readout_scale_init = cfg_get_not_none(cfg, "query_readout_scale_init", 1e-2 if enhanced else 0.0)
        glres_scale_init = cfg_get_not_none(cfg, "glres_scale_init", 1e-2 if enhanced else 0.0)
        query_readout_type = cfg_get_not_none(
            cfg,
            "query_readout_type",
            "coord" if enhanced or query_latent_readout else "point",
        )

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
            enhanced_backbone=enhanced,
            sensor_coord_encoding=sensor_coord_encoding,
            latent_sensor_reinject=latent_sensor_reinject,
            latent_reinject_every=cfg.get("latent_reinject_every", 1),
            query_latent_readout=query_latent_readout,
            query_readout_type=query_readout_type,
            query_readout_scale_init=query_readout_scale_init,
            enhanced_head_norm=enhanced_head_norm,
            glres_scale_init=glres_scale_init,
        )
        return PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))

    # Default / backward-compatible point MLP+RBF.
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
    return PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))

def denormalize_fields(x: torch.Tensor, dataset: TurbulentCombustionH5Dataset) -> torch.Tensor:
    return x * dataset.std.to(x.device).view(1, 1, -1) + dataset.mean.to(x.device).view(1, 1, -1)


def _call_model_sample(
    model: nn.Module,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: torch.Tensor,
    n_steps: int,
    ode_solver: Optional[str] = None,
) -> torch.Tensor:
    """
    Compatibility wrapper around model.sample for old and new checkpoint APIs.
    """
    sig = inspect.signature(model.sample)
    kwargs = {
        "coords": coords,
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "n_steps": n_steps,
        "clamp_indices": obs_indices,
    }

    if "obs_field_ids" in sig.parameters:
        kwargs["obs_field_ids"] = obs_field_ids
    elif "cond_field_idx" in sig.parameters:
        # Fallback for older single-field API. All obs_field_ids should be the same.
        unique = torch.unique(obs_field_ids[obs_mask.bool()])
        if unique.numel() != 1:
            raise ValueError(
                "Loaded checkpoint expects single-field conditioning (cond_field_idx), "
                "but the requested evaluation uses multiple conditioned fields."
            )
        kwargs["cond_field_idx"] = unique.view(1).to(obs_field_ids.device)

    if "ode_solver" in sig.parameters and ode_solver is not None:
        kwargs["ode_solver"] = ode_solver

    return model.sample(**kwargs)


@torch.no_grad()
def reconstruct_snapshot_local(
    model: nn.Module,
    dataset: TurbulentCombustionH5Dataset,
    device: torch.device,
    snapshot_index: int,
    cond_fields: Sequence[int],
    n_obs_list: Sequence[int],
    n_steps: int,
    ode_solver: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct one snapshot under sparse conditioning.

    Returns normalized truth/reconstruction tensors so coherence can be computed
    in the model's normalized field space.
    """
    model.eval()

    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)   # [1, N, D]
    truth = sample["fields"].unsqueeze(0).to(device)    # [1, N, C]

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cond_fields,
        n_obs_min=n_obs_list,
        n_obs_max=n_obs_list,
    )

    recon = _call_model_sample(
        model=model,
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        obs_indices=obs_indices,
        n_steps=n_steps,
        ode_solver=ode_solver,
    )

    return {
        "coords": coords,
        "truth": truth,
        "recon": recon,
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_indices": obs_indices,
        "obs_field_ids": obs_field_ids,
    }


def _baseline_num_xy(cfg: Dict[str, Any]) -> Tuple[int, int]:
    data_cfg = cfg["shared"]["data"]
    return int(data_cfg["num_x"]), int(data_cfg["num_y"])


def _baseline_sampling_defaults(cfg: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
    stage_cfg = baseline_lib.resolve_stage_config(cfg)
    sampling_cfg = stage_cfg.get("sampling", {})
    if cfg["baseline_model"] == "latent_fm" and int(cfg["training_stage"]) == 2:
        steps = sampling_cfg.get("benchmark_n_steps", [None])
        if isinstance(steps, (list, tuple)):
            steps = steps[0] if steps else None
        return (None if steps is None else int(steps), str(sampling_cfg.get("ode_solver", "euler")))
    if cfg["baseline_model"] in {"s3gm", "sit"}:
        return (int(sampling_cfg.get("sampling_N", 50)), str(sampling_cfg.get("ode_solver", "euler")))
    if cfg["baseline_model"] in {"mlp_rbf", "senseiver", "geofno"}:
        return None, None
    return None, None


def _baseline_build_sparse(
    *,
    dataset: Any,
    coords: torch.Tensor,
    truth: torch.Tensor,
    cond_fields: Sequence[int],
    n_obs_list: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_valid_mask = None
    return baseline_lib.build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cond_fields,
        n_obs_min=n_obs_list,
        n_obs_max=n_obs_list,
        valid_mask=sample_valid_mask,
    )


def _baseline_reconstruct_latentfm(
    bundle: Any,
    dataset: Any,
    coords: torch.Tensor,
    truth: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_indices: torch.Tensor,
    obs_field_ids: torch.Tensor,
    n_steps: int,
    ode_solver: Optional[str],
) -> torch.Tensor:
    num_x, num_y = _baseline_num_xy(bundle.config)
    n_fields = dataset.num_fields
    n_pts = dataset.num_points

    if int(bundle.training_stage) == 1:
        x_grid = baseline_lib.pointcloud_to_grid(truth, num_y, num_x)
        x_hat, _ = bundle.model(x_grid)
        x_hat = x_hat[:, :, :num_y, :num_x]
        return baseline_lib.grid_to_pointcloud(x_hat, num_y, num_x)

    obs_value_grid, obs_mask_grid = baseline_lib.build_obs_grid_mask(
        obs_values,
        obs_mask,
        obs_field_ids,
        obs_indices,
        n_fields,
        n_pts,
        num_y,
        num_x,
        num_y,
        num_x,
    )
    ix = (obs_indices % num_x).float() / max(num_x - 1, 1)
    iy = (obs_indices.div(num_x, rounding_mode="floor")).float() / max(num_y - 1, 1)
    obs_coords_2d = torch.stack([ix, iy], dim=-1)
    cond_inputs = {
        "obs_value_grid": obs_value_grid,
        "obs_mask_grid": obs_mask_grid,
        "obs_coords_2d": obs_coords_2d,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_field_ids": obs_field_ids,
    }
    recon_grid = bundle.model.sample(cond_inputs, n_steps=n_steps, ode_solver=ode_solver or "euler")
    return baseline_lib.grid_to_pointcloud(recon_grid, num_y, num_x)


def _baseline_reconstruct_sit(
    bundle: Any,
    dataset: Any,
    coords: torch.Tensor,
    truth: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_indices: torch.Tensor,
    obs_field_ids: torch.Tensor,
    n_steps: int,
    ode_solver: Optional[str],
) -> torch.Tensor:
    num_x, num_y = _baseline_num_xy(bundle.config)
    h_pad = int(bundle.components["H_pad"])
    w_pad = int(bundle.components["W_pad"])
    n_fields = int(bundle.components["n_fields"])
    n_pts = dataset.num_points
    tokenizer = str(bundle.components["tokenizer"])
    cond_mode = str(bundle.components["cond_mode"])

    if tokenizer == "pointnet":
        obs_value_nodes, obs_mask_nodes = baseline_lib.scatter_sensors_to_nodes(
            obs_values,
            obs_mask,
            obs_field_ids,
            obs_indices,
            1,
            n_pts,
            n_fields,
            bundle.device,
            truth.dtype,
        )
        return baseline_lib.sit_conditional_sample(
            net=bundle.model,
            transport=bundle.components["transport"],
            shape=(1, n_pts, n_fields),
            coords=coords,
            obs_value_nodes=obs_value_nodes,
            obs_mask_nodes=obs_mask_nodes,
            device=bundle.device,
            n_steps=n_steps,
            sampler_type=ode_solver or "euler",
        )

    obs_value_grid, obs_mask_grid = baseline_lib.build_obs_grid_mask(
        obs_values,
        obs_mask,
        obs_field_ids,
        obs_indices,
        n_fields,
        n_pts,
        num_y,
        num_x,
        h_pad,
        w_pad,
        point_to_grid=bundle.components.get("point_to_grid"),
    )
    if cond_mode == "interp":
        obs_value_grid = baseline_lib.nearest_fill_grid(obs_value_grid, obs_mask_grid)
    recon_grid = baseline_lib.sit_conditional_sample(
        net=bundle.model,
        transport=bundle.components["transport"],
        shape=(1, n_fields, h_pad, w_pad),
        obs_value_grid=obs_value_grid,
        obs_mask_grid=obs_mask_grid,
        device=bundle.device,
        n_steps=n_steps,
        sampler_type=ode_solver or "euler",
    )
    return baseline_lib.grid_to_pointcloud(
        recon_grid,
        num_y,
        num_x,
        point_to_grid=bundle.components.get("point_to_grid"),
    )


def _baseline_reconstruct_s3gm(
    bundle: Any,
    dataset: Any,
    truth: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_indices: torch.Tensor,
    obs_field_ids: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    stage_cfg = baseline_lib.resolve_stage_config(bundle.config)
    sampling_cfg = stage_cfg["sampling"]
    num_x, num_y = _baseline_num_xy(bundle.config)
    h_pad = int(bundle.components["H_pad"])
    w_pad = int(bundle.components["W_pad"])
    n_fields = dataset.num_fields
    n_pts = dataset.num_points
    obs_value_grid, obs_mask_grid = baseline_lib.build_obs_grid_mask(
        obs_values,
        obs_mask,
        obs_field_ids,
        obs_indices,
        n_fields,
        n_pts,
        num_y,
        num_x,
        h_pad,
        w_pad,
    )
    recon_grid = baseline_lib.dps_sample(
        net=bundle.model,
        sde=bundle.components["sde"],
        shape_5d=(1, 1, n_fields, h_pad, w_pad),
        obs_value_grid=obs_value_grid,
        obs_mask_grid=obs_mask_grid,
        device=bundle.device,
        N_steps=n_steps,
        snr=float(sampling_cfg["snr"]),
        n_corrector_steps=int(sampling_cfg["n_corrector_steps"]),
        alpha_obs=float(sampling_cfg["alpha_obs"]),
    )
    return baseline_lib.grid_to_pointcloud(recon_grid, num_y, num_x)


def _baseline_reconstruct_deterministic(
    bundle: Any,
    dataset: Any,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_indices: torch.Tensor,
    obs_field_ids: torch.Tensor,
) -> torch.Tensor:
    if bundle.baseline_model in {"mlp_rbf", "senseiver"}:
        return bundle.model(coords, obs_coords, obs_values, obs_mask, obs_field_ids)

    if bundle.baseline_model == "geofno":
        variant = str(bundle.components.get("variant", "fno"))
        if variant == "irregular":
            return bundle.model(coords[..., :2], obs_values, obs_mask, obs_field_ids, obs_indices)

        num_x, num_y = _baseline_num_xy(bundle.config)
        n_fields = dataset.num_fields
        n_pts = dataset.num_points
        obs_value_grid, obs_mask_grid = baseline_lib.build_obs_grid_mask(
            obs_values,
            obs_mask,
            obs_field_ids,
            obs_indices,
            n_fields,
            n_pts,
            num_y,
            num_x,
            num_y,
            num_x,
            point_to_grid=bundle.components.get("point_to_grid"),
        )
        pred_grid = bundle.model(obs_value_grid, obs_mask_grid)
        return baseline_lib.grid_to_pointcloud(
            pred_grid,
            num_y,
            num_x,
            point_to_grid=bundle.components.get("point_to_grid"),
        )

    raise ValueError(f"Unsupported deterministic baseline_model: {bundle.baseline_model}")


def reconstruct_baseline_snapshot(
    model: Any,
    dataset: Any,
    device: torch.device,
    snapshot_index: int,
    cond_fields: Sequence[int],
    n_obs_list: Sequence[int],
    n_steps: int,
    ode_solver: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    """
    Reconstruct one snapshot for train_Gen_Baseline.py checkpoints.

    model is a BaselineBundle from model_baseline.py. The return contract
    mirrors reconstruct_snapshot_local so the coherence code can stay shared.
    """
    bundle = model
    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = _baseline_build_sparse(
        dataset=dataset,
        coords=coords,
        truth=truth,
        cond_fields=cond_fields,
        n_obs_list=n_obs_list,
    )

    with bundle.adapter.evaluation_weights(bundle):
        bundle.model.eval()
        if bundle.baseline_model == "latent_fm":
            with torch.no_grad():
                recon = _baseline_reconstruct_latentfm(
                    bundle,
                    dataset,
                    coords,
                    truth,
                    obs_values,
                    obs_mask,
                    obs_indices,
                    obs_field_ids,
                    n_steps,
                    ode_solver,
                )
        elif bundle.baseline_model == "sit":
            with torch.no_grad():
                recon = _baseline_reconstruct_sit(
                    bundle,
                    dataset,
                    coords,
                    truth,
                    obs_values,
                    obs_mask,
                    obs_indices,
                    obs_field_ids,
                    n_steps,
                    ode_solver,
                )
        elif bundle.baseline_model == "s3gm":
            # S3GM DPS sampling differentiates through an observation residual,
            # so this branch intentionally cannot be wrapped in torch.no_grad().
            recon = _baseline_reconstruct_s3gm(
                bundle,
                dataset,
                truth,
                obs_values,
                obs_mask,
                obs_indices,
                obs_field_ids,
                n_steps,
            )
        elif bundle.baseline_model in {"mlp_rbf", "senseiver", "geofno"}:
            with torch.no_grad():
                recon = _baseline_reconstruct_deterministic(
                    bundle,
                    dataset,
                    coords,
                    obs_coords,
                    obs_values,
                    obs_mask,
                    obs_indices,
                    obs_field_ids,
                )
        else:
            raise ValueError(f"Unsupported baseline_model for coherence reconstruction: {bundle.baseline_model}")

    return {
        "coords": coords,
        "truth": truth,
        "recon": recon.detach(),
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_indices": obs_indices,
        "obs_field_ids": obs_field_ids,
    }


def get_reconstruction_fn(model_family: str = "pointcloud_ffm") -> Any:
    """
    Use the helper-provided reconstruct_snapshot if the repo has been patched,
    otherwise fall back to the local compatible implementation.
    """
    if model_family != "pointcloud_ffm":
        return reconstruct_baseline_snapshot
    return helpers_reconstruct_snapshot or reconstruct_snapshot_local


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_text(path: Path, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


def normalize_statistical_modes(raw_modes: Optional[Sequence[str]]) -> List[str]:
    if raw_modes is None:
        return []
    modes = list(raw_modes)
    if "both" in modes:
        return ["CoL2_Scatter", "Aggregate_Violin"]
    ordered = []
    for mode in modes:
        if mode not in ordered:
            ordered.append(mode)
    return ordered


def compute_relative_l2(pred: torch.Tensor, ref: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Relative L2 reconstruction error over all query points and channels.
    """
    if pred.shape != ref.shape:
        raise ValueError(f"compute_relative_l2 shape mismatch: {tuple(pred.shape)} vs {tuple(ref.shape)}")
    diff = pred.reshape(-1) - ref.reshape(-1)
    denom = ref.reshape(-1).norm() + eps
    return float(diff.norm().detach().cpu() / denom.detach().cpu())


def compute_observation_consistency_error(
    pred_full: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    coords_full: torch.Tensor,
    obs_indices: Optional[torch.Tensor] = None,
    obs_field_ids: Optional[torch.Tensor] = None,
    field_mean: Optional[torch.Tensor] = None,
    field_std: Optional[torch.Tensor] = None,
    eps: float = 1e-12,
) -> float:
    """
    Relative L2 error on valid conditioned sensor values only.

    The preferred path uses obs_indices from build_sparse_condition. A nearest
    neighbor coordinate lookup is kept as a fallback for older helper versions.
    field_mean/field_std can be supplied when pred_full is in physical units
    but obs_values are still normalized dataset values.
    """
    if pred_full.ndim == 2:
        pred_full = pred_full.unsqueeze(0)
    if coords_full.ndim == 2:
        coords_full = coords_full.unsqueeze(0)
    if obs_coords.ndim == 2:
        obs_coords = obs_coords.unsqueeze(0)
    if obs_values.ndim == 2:
        obs_values = obs_values.unsqueeze(0)
    if obs_mask.ndim == 1:
        obs_mask = obs_mask.unsqueeze(0)

    if obs_field_ids is None:
        raise ValueError("obs_field_ids is required to compute observation consistency across conditioned fields.")
    if obs_field_ids.ndim == 1:
        obs_field_ids = obs_field_ids.unsqueeze(0)

    valid = obs_mask.bool()
    if valid.sum().item() == 0:
        return float("nan")

    if obs_indices is not None:
        if obs_indices.ndim == 1:
            obs_indices = obs_indices.unsqueeze(0)
        gather_indices = obs_indices.long()
    else:
        # Fallback for older observation builders: map each observation coordinate
        # back onto the full point cloud by nearest neighbor.
        gather_indices = torch.zeros_like(obs_field_ids, dtype=torch.long)
        for b in range(obs_coords.shape[0]):
            vb = valid[b]
            if vb.any():
                dist = torch.cdist(obs_coords[b, vb].unsqueeze(0), coords_full[b].unsqueeze(0))[0]
                gather_indices[b, vb] = torch.argmin(dist, dim=1)

    batch_ids = torch.arange(pred_full.shape[0], device=pred_full.device).view(-1, 1).expand_as(gather_indices)
    pred_obs = pred_full[batch_ids[valid], gather_indices[valid], obs_field_ids.long()[valid]]
    target_obs = obs_values[..., 0][valid].to(pred_obs.device, dtype=pred_obs.dtype)

    if field_mean is not None and field_std is not None:
        fields = obs_field_ids.long()[valid].to(pred_obs.device)
        mean = field_mean.to(pred_obs.device, dtype=pred_obs.dtype)[fields]
        std = field_std.to(pred_obs.device, dtype=pred_obs.dtype)[fields]
        target_obs = target_obs * std + mean

    return float((pred_obs - target_obs).norm().detach().cpu() / (target_obs.norm().detach().cpu() + eps))


def _select_stat_indices(n_items: int, stride: int, max_snapshots: Optional[int], seed: int) -> List[int]:
    if stride <= 0:
        raise ValueError(f"--stat-stride must be positive, got {stride}")
    if max_snapshots is not None and max_snapshots <= 0:
        raise ValueError(f"--stat-max-snapshots must be positive when supplied, got {max_snapshots}")

    indices = list(range(0, n_items, stride))
    if max_snapshots is not None and len(indices) > max_snapshots:
        rng = np.random.default_rng(seed)
        indices = sorted(int(v) for v in rng.choice(indices, size=max_snapshots, replace=False))
    return indices


def _stat_progress_iter(indices: Sequence[int], split_name: str) -> Any:
    """
    Progress helper for statistical mode.

    tqdm is optional so the evaluator does not gain a required runtime
    dependency. When unavailable, callers still emit periodic text progress.
    """
    if tqdm is None:
        return enumerate(indices, start=1)
    return enumerate(
        tqdm(
            indices,
            total=len(indices),
            desc=f"[stat] {split_name}",
            unit="snapshot",
            dynamic_ncols=True,
            leave=True,
        ),
        start=1,
    )


def apply_paper_style() -> None:
    """
    Matplotlib defaults for paper-level aggregate diagnostic plots.

    These settings keep the statistical figures compact, vector-friendly, and
    readable in manuscript layouts without changing single-snapshot diagnostics.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def save_figure_all_formats(fig: Any, stem_path: Path, formats: Sequence[str], dpi: int = 400) -> None:
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        suffix = str(fmt).lower().lstrip(".")
        if suffix not in {"png", "pdf", "svg"}:
            raise ValueError(f"Unsupported statistical plot format: {fmt}")
        fig.savefig(stem_path.with_suffix(f".{suffix}"), dpi=dpi)


def _finite_np(x: Any) -> np.ndarray:
    vals = np.asarray(x, dtype=float).reshape(-1)
    return vals[np.isfinite(vals)]


def _finite_array(rows: Sequence[Dict[str, Any]], key: str) -> np.ndarray:
    return _finite_np([float(row.get(key, np.nan)) for row in rows])


def _stat_triplet(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, Optional[float]]:
    vals = _finite_array(rows, key)
    if vals.size == 0:
        return {"finite_count": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None, "iqr": None}
    q25, q75 = np.percentile(vals, [25, 75])
    return {
        "finite_count": int(vals.size),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def _compute_l2_metric_correlations(rows: Sequence[Dict[str, Any]], metric_key: str) -> Dict[str, Any]:
    x = np.asarray([float(row.get("relative_l2", np.nan)) for row in rows], dtype=float)
    y = np.asarray([float(row.get(metric_key, np.nan)) for row in rows], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    out: Dict[str, Any] = {
        "pearson_r": None,
        "spearman_rho": None,
        "note": None,
    }
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        out["note"] = "Not enough non-constant finite samples to compute correlations."
        return out

    try:
        from scipy import stats as scipy_stats  # type: ignore

        pearson = scipy_stats.pearsonr(x, y)
        spearman = scipy_stats.spearmanr(x, y)
        out["pearson_r"] = float(pearson.statistic if hasattr(pearson, "statistic") else pearson[0])
        out["spearman_rho"] = float(spearman.statistic if hasattr(spearman, "statistic") else spearman[0])
    except Exception:
        out["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
        out["spearman_rho"] = None
        out["note"] = "scipy is unavailable; Pearson was computed with numpy and Spearman was set to null."
    return out


def _compute_l2_mode_correlations(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return _compute_l2_metric_correlations(rows, "mode_score")


def _robust_limits(values: np.ndarray, pad_frac: float = 0.04) -> Optional[Tuple[float, float]]:
    vals = _finite_np(values)
    if vals.size == 0:
        return None
    if vals.size >= 8:
        lo, hi = np.percentile(vals, [1, 99])
    else:
        lo, hi = float(np.min(vals)), float(np.max(vals))
    lo = float(lo)
    hi = float(hi)
    if lo == hi:
        pad = max(abs(lo) * pad_frac, 1e-12)
        return lo - pad, hi + pad
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def _safe_percentile_color_limits(values: np.ndarray, lo: float = 2, hi: float = 98, eps: float = 1e-12) -> Optional[Tuple[float, float]]:
    vals = _finite_np(values)
    if vals.size == 0:
        return None
    vmin, vmax = np.nanpercentile(vals, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or float(vmax - vmin) < eps:
        return None
    return float(vmin), float(vmax)


def _add_panel_label(ax: Any, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )


def _despine(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _format_metric_label(name: str) -> str:
    return {
        "marginal_score": "Marginal",
        "joint_score": "Joint SWD",
        "pairwise_mean": "Pairwise",
        "mode_score": "Global dist.",
        "global_dist_score": "Global dist.",
        "base_score": "Base global dist.",
        "relative_l2": r"Relative $L_2$ error",
        "obs_consistency_error": "Observation consistency error",
    }.get(name, name)


def save_statistical_table_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "snapshot_index",
        "relative_l2",
        "obs_consistency_error",
        "global_dist_score",
        "base_score",
        "marginal_score",
        "joint_score",
        "pairwise_mean",
        "mode_score",
        "n_steps",
        "coherence_space",
        "n_obs_total",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_statistical_summary(
    rows: Sequence[Dict[str, Any]],
    *,
    split_selection: str,
    requested_modes: Sequence[str],
) -> Dict[str, Any]:
    metrics = [
        "relative_l2",
        "obs_consistency_error",
        "global_dist_score",
        "base_score",
        "marginal_score",
        "joint_score",
        "pairwise_mean",
        "mode_score",
    ]
    summary: Dict[str, Any] = {
        "number_of_processed_snapshots": len(rows),
        "split_selection": split_selection,
        "requested_figures": list(requested_modes),
        "metrics": {key: _stat_triplet(rows, key) for key in metrics},
        "relative_l2_mode_score_correlation": _compute_l2_mode_correlations(rows),
        "relative_l2_correlations": {
            key: _compute_l2_metric_correlations(rows, key)
            for key in ["global_dist_score", "base_score", "marginal_score", "joint_score", "pairwise_mean", "mode_score"]
        },
    }
    if _finite_array(rows, "pairwise_mean").size == 0:
        summary["metrics"]["pairwise_mean"]["note"] = "pairwise_mean unavailable; pairwise diagnostics may be disabled."
    return summary


def save_coherence_l2_scatter(
    rows: Sequence[Dict[str, Any]],
    stem_path: Path,
    args: argparse.Namespace,
    correlations: Optional[Dict[str, Any]] = None,
) -> None:
    x = np.asarray([float(row.get("relative_l2", np.nan)) for row in rows], dtype=float)
    y = np.asarray([float(row.get("mode_score", np.nan)) for row in rows], dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    color_key = str(args.stat_color_by)
    color_values = None
    if color_key != "none":
        color_values = np.asarray([float(row.get(color_key, np.nan)) for row in rows], dtype=float)
        mask = mask & np.isfinite(color_values)

    has_colorbar = color_key != "none" and color_values is not None and _safe_percentile_color_limits(color_values[mask]) is not None
    fig, ax = plt.subplots(figsize=(4.2, 3.2) if has_colorbar else (3.6, 3.0))
    note_lines: List[str] = []
    if mask.any():
        x_plot = x[mask]
        y_plot = y[mask]
        if color_key == "none" or color_values is None:
            ax.scatter(
                x_plot,
                y_plot,
                marker="o",
                s=float(args.stat_point_size),
                alpha=0.72,
                linewidths=0.25,
                edgecolors="white",
                color="#4C78A8",
            )
        else:
            c_plot = color_values[mask]
            clim = _safe_percentile_color_limits(c_plot)
            if clim is None:
                ax.scatter(
                    x_plot,
                    y_plot,
                    marker="o",
                    s=float(args.stat_point_size),
                    alpha=0.72,
                    linewidths=0.25,
                    edgecolors="white",
                    color="#4C78A8",
                )
                if color_key == "obs_consistency_error":
                    note_lines.append("Obs. consistency nearly constant")
            else:
                cmap = "viridis" if color_key == "relative_l2" else "magma"
                sc = ax.scatter(
                    x_plot,
                    y_plot,
                    c=c_plot,
                    vmin=clim[0],
                    vmax=clim[1],
                    cmap=cmap,
                    marker="o",
                    s=float(args.stat_point_size),
                    alpha=0.72,
                    linewidths=0.25,
                    edgecolors="white",
                )
                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
                cbar.set_label(_format_metric_label(color_key))

        if bool(args.stat_show_regression) and x_plot.size >= 3 and np.std(x_plot) > 0.0:
            coeff = np.polyfit(x_plot, y_plot, deg=1)
            x_line = np.linspace(float(np.min(x_plot)), float(np.max(x_plot)), 100)
            y_line = coeff[0] * x_line + coeff[1]
            ax.plot(x_line, y_line, color="0.15", linewidth=1.2, alpha=0.8)

        xlim = _robust_limits(x_plot, pad_frac=0.03)
        ylim = _robust_limits(y_plot, pad_frac=0.03)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            y_finite = _finite_np(y_plot)
            if y_finite.size > 0 and np.all(y_finite >= 0.0):
                y_low_pct = float(np.percentile(y_finite, 5))
                y_high_pct = float(np.percentile(y_finite, 95))
                if y_low_pct <= 0.08 * max(y_high_pct, 1e-12):
                    ylim = (0.0, ylim[1])
            ax.set_ylim(*ylim)

    correlations = correlations or _compute_l2_mode_correlations(rows)
    pearson = correlations.get("pearson_r")
    spearman = correlations.get("spearman_rho")
    stat_lines = [f"n = {int(mask.sum())}"]
    if int(mask.sum()) >= 3:
        if pearson is not None:
            stat_lines.append(f"Pearson r = {pearson:.2f}, R^2 = {pearson * pearson:.2f}")
        else:
            stat_lines.append("Pearson r = n/a, R^2 = n/a")
        stat_lines.append(f"Spearman \u03c1 = {spearman:.2f}" if spearman is not None else "Spearman \u03c1 = n/a")
    ax.text(
        0.04,
        0.96,
        "\n".join(stat_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.9),
    )
    if note_lines:
        ax.text(
            0.04,
            0.73,
            "\n".join(note_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.5,
            color="0.3",
        )

    ax.set_xlabel(r"Relative $L_2$ error")
    ax.set_ylabel("Global state-distribution discrepancy")
    if bool(args.stat_panel_labels):
        _add_panel_label(ax, "(a)")
    else:
        ax.set_title("Coherence vs. reconstruction error")
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    if args.stat_plot_style == "paper":
        _despine(ax)
    fig.text(0.52, -0.02, "Each point denotes one reconstructed snapshot.", ha="center", va="top", fontsize=7)
    fig.tight_layout()
    save_figure_all_formats(fig, stem_path, args.stat_plot_formats, dpi=int(args.stat_dpi))
    plt.close(fig)


def save_aggregate_violin(
    rows: Sequence[Dict[str, Any]],
    stem_path: Path,
    args: argparse.Namespace,
    *,
    seed: int,
) -> None:
    metric_specs = [
        ("marginal_score", "Marginal"),
        ("joint_score", "Joint SWD"),
        ("pairwise_mean", "Pairwise"),
        ("mode_score", "Global dist."),
    ]
    palette = {
        "marginal_score": "#4C78A8",
        "joint_score": "#F58518",
        "pairwise_mean": "#54A24B",
        "mode_score": "#B279A2",
    }

    keys: List[str] = []
    labels: List[str] = []
    data: List[np.ndarray] = []
    for key, label in metric_specs:
        vals = _finite_array(rows, key)
        if vals.size == 0:
            continue
        keys.append(key)
        labels.append(label)
        data.append(vals)

    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    if data:
        positions = np.arange(1, len(data) + 1)
        parts = ax.violinplot(data, positions=positions, showmeans=False, showmedians=False, showextrema=False)
        for body, key in zip(parts["bodies"], keys):
            body.set_facecolor(palette[key])
            body.set_edgecolor("0.3")
            body.set_linewidth(0.8)
            body.set_alpha(0.28)

        rng = np.random.default_rng(seed)
        for pos, key, vals in zip(positions, keys, data):
            jitter = rng.uniform(-0.075, 0.075, size=vals.size)
            ax.scatter(
                np.full(vals.size, pos) + jitter,
                vals,
                s=12,
                alpha=float(args.stat_jitter_alpha),
                edgecolor="none",
                color=palette[key],
                rasterized=vals.size > 250,
            )
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            mean = float(np.mean(vals))
            ax.plot([pos, pos], [q25, q75], color="black", lw=1.2, zorder=4)
            ax.scatter([pos], [float(med)], s=26, marker="D", color="black", zorder=5)
            ax.scatter(
                [pos],
                [mean],
                s=42,
                marker="o",
                facecolor="white",
                edgecolor=palette[key],
                linewidth=1.4,
                zorder=6,
            )
            ax.annotate(
                f"mean = {mean:.2e}",
                xy=(pos, mean),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color=palette[key],
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.82),
                zorder=7,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=20, ha="right")

        all_vals = np.concatenate(data)
        if bool(args.stat_log_y) and np.all(all_vals > 0.0):
            ax.set_yscale("log")
            ax.set_ylabel("Discrepancy (log scale)")
        else:
            if bool(args.stat_log_y) and not np.all(all_vals > 0.0):
                print("[stat] Aggregate violin uses linear y-scale because nonpositive scores are present.")
            ax.set_ylabel("Discrepancy")
    else:
        ax.text(0.5, 0.5, "No finite coherence scores available", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Discrepancy")

    ax.set_xlabel("")
    if bool(args.stat_panel_labels):
        _add_panel_label(ax, "(b)")
    else:
        ax.set_title("Distribution of coherence components")
    ax.text(
        0.98,
        0.96,
        f"n = {len(rows)} snapshots",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
    )
    ax.grid(axis="y", color="0.9", linewidth=0.5)
    if args.stat_plot_style == "paper":
        _despine(ax)
    fig.tight_layout()
    save_figure_all_formats(fig, stem_path, args.stat_plot_formats, dpi=int(args.stat_dpi))
    plt.close(fig)


def _build_eval_dataset(
    *,
    data_path: Path,
    split: str,
    cfg: Dict[str, Any],
    stats_path: Path,
) -> TurbulentCombustionH5Dataset:
    return TurbulentCombustionH5Dataset(
        h5_path=str(data_path),
        split=split,
        train_ratio=float(cfg.get("train_ratio", 0.9)),
        seed=int(cfg.get("seed", 42)),
        time_stride=int(cfg.get("time_stride", 1)),
        field_names=cfg.get("FIELD_NAMES", cfg.get("field_names", None)),
        stats_path=str(stats_path) if stats_path.exists() else None,
    )


def load_baseline_context(
    *,
    args: argparse.Namespace,
    checkpoint_arg: str,
    device: torch.device,
) -> Tuple[Path, Path, Dict[str, Any], Any, Any, Callable[[str], Any], int, Optional[str]]:
    """
    Load a unified generative or deterministic baseline checkpoint using
    model_baseline.py adapters.
    """
    if baseline_lib is None:
        raise ImportError("model_baseline.py could not be imported; baseline checkpoint evaluation is unavailable.")

    checkpoint_path, run_dir = choose_checkpoint(checkpoint_arg)
    checkpoint = baseline_lib.safe_torch_load(checkpoint_path, map_location="cpu")

    run_cfg_path = run_dir / "run_config.yaml"
    if run_cfg_path.exists():
        cfg = baseline_lib.validate_and_normalize_config(baseline_lib.load_yaml(run_cfg_path))
    elif isinstance(checkpoint, dict) and checkpoint.get("config") is not None:
        cfg = baseline_lib.validate_and_normalize_config(checkpoint["config"])
    else:
        raise FileNotFoundError(
            f"Baseline checkpoint needs {run_cfg_path} or an embedded 'config' entry."
        )

    requested_baseline = str(args.baseline_model).strip().lower()
    if requested_baseline != "auto" and requested_baseline != cfg["baseline_model"]:
        print(
            f"[warning] --baseline-model={requested_baseline} does not match run_config baseline_model={cfg['baseline_model']}; "
            "using the run_config value."
        )
    args.baseline_model = cfg["baseline_model"]

    if cfg["baseline_model"] == "latent_fm" and int(cfg["training_stage"]) == 2:
        ae_checkpoint, recorded_ae_checkpoint = resolve_latentfm_stage1_checkpoint(
            cfg=cfg, checkpoint=checkpoint if isinstance(checkpoint, dict) else None, run_dir=run_dir
        )
        if ae_checkpoint is None:
            recorded = recorded_ae_checkpoint or "<none>"
            raise FileNotFoundError(
                "Latent FM stage 2 requires its shared stage-0/stage-1 VAE checkpoint. "
                f"Recorded dependency is unavailable: {recorded}. Expected a valid archive sibling "
                f"at {run_dir.parent / 'Stage0'} or {run_dir.parent / 'Stage1'}."
            )
        cfg["latent_fm_params"]["stage2"]["stage1_checkpoint"] = str(ae_checkpoint)
        if recorded_ae_checkpoint is not None and ae_checkpoint != recorded_ae_checkpoint:
            print(
                f"[DEPENDENCY] Latent FM / {run_dir.name} | using archived shared VAE: {ae_checkpoint}",
                flush=True,
            )

    stats_path = run_dir / "dataset_stats.pt"

    def dataset_builder(split: str) -> Any:
        return baseline_lib.build_dataset(cfg, split=split, stats_path=stats_path)

    dataset = dataset_builder(args.split)
    adapter = baseline_lib.get_baseline_adapter(cfg["baseline_model"])
    bundle = adapter.build_for_training(
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        train_set=dataset,
        val_set=dataset,
    )
    adapter.load_checkpoint(bundle, checkpoint)
    bundle.adapter = adapter
    bundle.model.eval()

    default_steps, default_solver = _baseline_sampling_defaults(cfg)
    n_steps = int(args.n_steps if args.n_steps is not None else (default_steps if default_steps is not None else 1))
    ode_solver = args.ode_solver or default_solver

    return checkpoint_path, run_dir, cfg, dataset, bundle, dataset_builder, n_steps, ode_solver


def run_statistical_analysis(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    data_path: Path,
    cfg: Dict[str, Any],
    stats_path: Path,
    eval_dir: Path,
    device: torch.device,
    cond_fields: Sequence[int],
    n_obs_list: Sequence[int],
    n_steps: int,
    coherence_cfg: GlobalDistConfig,
    reco_fn: Optional[Any] = None,
    dataset_builder: Optional[Callable[[str], Any]] = None,
    ode_solver: Optional[str] = None,
) -> None:
    """
    Run dataset-level statistical coherence analysis.

    Statistical mode is intended for paper-level evidence that coherence metrics
    are not redundant with L2 and are stable across many frames. It intentionally
    suppresses snapshot-level diagnostic plots and saves only aggregate artifacts.
    """
    if args.coherence_mode != "global_dist":
        raise ValueError("Statistical analysis currently supports --coherence-mode global_dist only.")

    if args.stat_plot_style == "paper":
        apply_paper_style()

    requested_modes = normalize_statistical_modes(args.statistical_analysis)
    stat_dir = eval_dir / "statistical_analysis"
    stat_dir.mkdir(parents=True, exist_ok=True)

    split_names = ["train", "test"] if args.statistical_split == "all" else [args.statistical_split]
    reco_fn = reco_fn or get_reconstruction_fn(args.baseline_model)
    rows: List[Dict[str, Any]] = []

    print("[stat] Statistical-analysis mode enabled.")
    print("[stat] --snapshot-indices will be ignored; snapshot-level plotting is disabled.")

    for split_name in split_names:
        dataset = (
            dataset_builder(split_name)
            if dataset_builder is not None
            else _build_eval_dataset(data_path=data_path, split=split_name, cfg=cfg, stats_path=stats_path)
        )
        selected = _select_stat_indices(
            n_items=len(dataset),
            stride=int(args.stat_stride),
            max_snapshots=args.stat_max_snapshots,
            seed=int(args.stat_seed),
        )
        print(f"[stat] split={split_name}: processing {len(selected)} of {len(dataset)} snapshots")

        for pos, snapshot_index in _stat_progress_iter(selected, split_name):
            rec = reco_fn(
                model=model,
                dataset=dataset,
                device=device,
                snapshot_index=int(snapshot_index),
                    cond_fields=cond_fields,
                    n_obs_list=n_obs_list,
                    n_steps=n_steps,
                    ode_solver=ode_solver if ode_solver is not None else args.ode_solver,
                )

            truth = rec["truth"]
            recon = rec["recon"]
            if args.coherence_space == "physical":
                truth_eval = denormalize_fields(truth, dataset)
                recon_eval = denormalize_fields(recon, dataset)
                obs_field_mean = dataset.mean
                obs_field_std = dataset.std
            else:
                truth_eval = truth
                recon_eval = recon
                obs_field_mean = None
                obs_field_std = None

            x_ref = truth_eval[0]
            x_gen = recon_eval[0]
            result = compute_coherence(args.coherence_mode, x_gen=x_gen, x_ref=x_ref, cfg=coherence_cfg)
            scalars = coherence_result_to_scalars(result)

            n_obs_total = int(rec["obs_mask"].bool().sum().detach().cpu())
            row = {
                "split": split_name,
                "snapshot_index": int(snapshot_index),
                "relative_l2": compute_relative_l2(x_gen, x_ref),
                "obs_consistency_error": compute_observation_consistency_error(
                    pred_full=recon_eval,
                    obs_coords=rec["obs_coords"],
                    obs_values=rec["obs_values"],
                    obs_mask=rec["obs_mask"],
                    coords_full=rec["coords"],
                    obs_indices=rec.get("obs_indices"),
                    obs_field_ids=rec.get("obs_field_ids"),
                    field_mean=obs_field_mean,
                    field_std=obs_field_std,
                ),
                "global_dist_score": scalars["global_dist_score"],
                "base_score": scalars["base_score"],
                "marginal_score": scalars["marginal_score"],
                "joint_score": scalars["joint_score"],
                "pairwise_mean": scalars["pairwise_mean"],
                "mode_score": scalars["mode_score"],
                "n_steps": int(n_steps),
                "coherence_space": args.coherence_space,
                "n_obs_total": n_obs_total,
            }
            rows.append(row)

            if tqdm is None and (pos == 1 or pos == len(selected) or pos % 5 == 0):
                print(f"[stat] processed {pos}/{len(selected)} for split={split_name}")

            del rec, truth, recon, truth_eval, recon_eval, x_ref, x_gen, result

    save_statistical_table_csv(stat_dir / "statistical_coherence_table.csv", rows)
    summary = build_statistical_summary(
        rows,
        split_selection=args.statistical_split,
        requested_modes=requested_modes,
    )
    save_json(stat_dir / "statistical_analysis_summary.json", summary)

    if "CoL2_Scatter" in requested_modes:
        save_coherence_l2_scatter(
            rows,
            stat_dir / "coherence_l2_scatter",
            args,
            correlations=summary["relative_l2_mode_score_correlation"],
        )
        # save_coherence_l2_scatter(
        #     rows,
        #     stat_dir / "fig_stat_coherence_l2_scatter",
        #     args,
        #     correlations=summary["relative_l2_mode_score_correlation"],
        # )
    if "Aggregate_Violin" in requested_modes:
        save_aggregate_violin(
            rows,
            stat_dir / "aggregate_violin",
            args,
            seed=int(args.stat_seed),
        )
        # save_aggregate_violin(
        #     rows,
        #     stat_dir / "fig_stat_coherence_components",
        #     args,
        #     seed=int(args.stat_seed),
        # )

    print(f"\n[*] Statistical coherence analysis saved to: {stat_dir}\n")


def summarize_direction(
    vec: np.ndarray,
    field_names: Sequence[str],
    top_k: int = 3,
    decimals: int = 2,
) -> str:
    """
    Return a short readable description of the dominant field combination.
    """
    vec = np.asarray(vec).copy()
    if vec.size == 0:
        return ""

    j = int(np.argmax(np.abs(vec)))
    if vec[j] < 0:
        vec = -vec

    order = np.argsort(-np.abs(vec))[: min(top_k, vec.size)]
    terms = [f"{vec[idx]:+.{decimals}f}*{field_names[idx]}" for idx in order]
    return " ".join(terms)


def resolve_global_dist_config(
    args: argparse.Namespace,
    num_fields: int,
) -> Tuple[GlobalDistConfig, Dict[str, str], List[str]]:
    """
    Resolve coherence hyperparameters, using more stable defaults for the
    current low-channel, high-sample turbulent-combustion setting.
    """
    sources: Dict[str, str] = {}
    notes: List[str] = []

    def pick(name: str, explicit: Any, recommended: Any) -> Any:
        if explicit is not None:
            sources[name] = "cli"
            return explicit
        sources[name] = "auto"
        return recommended

    joint_method = str(args.joint_method)
    recommended_num_directions = 64 if joint_method == "topk_swd" else min(max(num_fields, 1), 5)
    recommended_n_iter_theta = 20 if num_fields <= 8 else 12
    recommended_lr_theta = 0.05
    recommended_ortho_reg = 1e-2
    recommended_n_proj_pairwise = 32

    num_directions = int(pick("num_directions", args.num_directions, recommended_num_directions))
    n_iter_theta = int(pick("n_iter_theta", args.n_iter_theta, recommended_n_iter_theta))
    lr_theta = float(pick("lr_theta", args.lr_theta, recommended_lr_theta))
    ortho_reg = float(pick("ortho_reg", args.ortho_reg, recommended_ortho_reg))
    n_proj_pairwise = int(pick("n_proj_pairwise", args.n_proj_pairwise, recommended_n_proj_pairwise))

    if sources["num_directions"] == "auto":
        if joint_method == "topk_swd":
            notes.append(
                f"Auto-selected num_directions={num_directions}: fixed-bank top-k SWD benefits from a larger "
                "deterministic projection bank while remaining cheap because only channel-space projections are sorted."
            )
        else:
            notes.append(
                f"Auto-selected num_directions={num_directions}: adaptive Max-SWD uses at most a few learned directions "
                "in the low-dimensional channel space."
            )
    if sources["n_iter_theta"] == "auto":
        notes.append(
            f"Auto-selected n_iter_theta={n_iter_theta}: the previous very short inner optimization can under-fit "
            "the maximizing directions; a longer search is still cheap because channel dimension is small."
        )
    if sources["lr_theta"] == "auto":
        notes.append(
            f"Auto-selected lr_theta={lr_theta:.3f}: paired with more theta steps, a slightly smaller step size is "
            "more stable than a large step with only a few iterations."
        )
    if sources["ortho_reg"] == "auto":
        notes.append(
            f"Auto-selected ortho_reg={ortho_reg:.2e}: this is a reasonable mild diversity penalty and was kept unchanged."
        )
    if sources["n_proj_pairwise"] == "auto":
        notes.append(
            f"Auto-selected n_proj_pairwise={n_proj_pairwise}: this keeps pairwise 2D SWD stable enough for scoring or "
            "diagnostics without making the global distribution evaluator unnecessarily expensive."
        )

    notes.append(
        "Kept lambda_marg = lambda_joint = 1.0 by default because both terms are squared Wasserstein discrepancies in "
        "the same units, so equal weighting is the most interpretable baseline."
    )
    if joint_method == "topk_swd":
        notes.append(
            "topk_swd uses a deterministic projection bank and averages the top fraction of projected discrepancies, "
            "providing a stable approximation of worst projection mismatches."
        )
    else:
        notes.append(
            "adaptive_maxswd preserves the older adaptive projection-pursuit SWD behavior; n_iter_theta, lr_theta, "
            "and ortho_reg only affect this mode."
        )
    if args.disable_pairwise:
        notes.append(
            "Pairwise diagnostics were disabled, so pairwise 2D dependence mismatches will not be visualized."
        )
    elif args.include_pairwise_in_score:
        notes.append(
            f"pairwise_mean is included in mode_score with lambda_pairwise={float(args.lambda_pairwise):.3g}."
        )
    else:
        notes.append(
            "pairwise_mean is computed as a diagnostic but excluded from mode_score because include_pairwise_in_score=False."
        )

    cfg = GlobalDistConfig(
        lambda_marg=float(args.lambda_marg),
        lambda_joint=float(args.lambda_joint),
        num_directions=num_directions,
        n_iter_theta=n_iter_theta,
        lr_theta=lr_theta,
        ortho_reg=ortho_reg,
        n_proj_pairwise=n_proj_pairwise,
        include_pairwise=not args.disable_pairwise,
        seed=args.seed,
        joint_method=joint_method,
        joint_top_frac=float(args.joint_top_frac),
        joint_qmc=bool(args.joint_qmc),
        include_axes=bool(args.include_axes),
        lambda_pairwise=float(args.lambda_pairwise),
        include_pairwise_in_score=bool(args.include_pairwise_in_score),
        exclude_axis_projections_when_marginal_included=bool(args.exclude_axis_projections_when_marginal_included),
    )
    return cfg, sources, notes


def build_interpretation_guide_text(
    *,
    checkpoint_path: Path,
    run_dir: Path,
    data_path: Path,
    split: str,
    snapshot_indices: Sequence[int],
    cond_fields: Sequence[int],
    n_obs_list: Sequence[int],
    n_steps: int,
    coherence_space: str,
    field_names: Sequence[str],
    coherence_cfg: GlobalDistConfig,
    hparam_sources: Dict[str, str],
    hparam_notes: Sequence[str],
) -> str:
    cond_field_names = [field_names[idx] if 0 <= idx < len(field_names) else str(idx) for idx in cond_fields]
    joint_label = (
        "adaptive projection-pursuit SWD"
        if coherence_cfg.joint_method == "adaptive_maxswd"
        else "fixed-bank top-k sliced Wasserstein"
    )
    pairwise_score_text = (
        "included in mode_score"
        if coherence_cfg.include_pairwise and coherence_cfg.include_pairwise_in_score
        else "not included in mode_score"
    )
    lines = [
        "PhyCoFlow coherence evaluation guide",
        "===================================",
        "",
        "Run context",
        "-----------",
        f"checkpoint      : {checkpoint_path}",
        f"run_dir         : {run_dir}",
        f"data            : {data_path}",
        f"split           : {split}",
        f"snapshot_indices: {list(snapshot_indices)}",
        f"conditioning    : fields {list(cond_fields)} -> {cond_field_names}, n_obs_list={list(n_obs_list)}, n_steps={n_steps}",
        f"coherence_space : {coherence_space}",
        "",
        "What the metrics mean",
        "---------------------",
        "All main discrepancy values are squared Wasserstein distances, so smaller is better and zero is ideal.",
        "The metrics compare the reconstructed snapshot distribution against the reference snapshot distribution over spatial points.",
        "This is a global state-distribution discrepancy, not a full physical-law residual.",
        "These metrics do not directly measure pointwise spatial alignment; they measure whether the distribution of field states is realistic.",
        "Spatial alignment is not directly measured by the scalar score.",
        "",
        "mode_score",
        "  base_score plus an optional pairwise contribution.",
        "  This is the main scalar summary used by this evaluator.",
        f"  Here pairwise_mean is {pairwise_score_text}.",
        "",
        "base_score",
        "  lambda_marg * marginal_score + lambda_joint * joint_score.",
        "  When canonical coordinate axes are included in the fixed projection bank and marginal_score is active,",
        "  those axis projections are kept as diagnostics but excluded from joint_score to avoid double-counting",
        "  the same single-field distribution mismatch.",
        "  This excludes the optional pairwise contribution.",
        "",
        "marginal_score",
        "  Mean of the per-channel 1D Wasserstein distances.",
        "  It answers: does each field individually have the right value distribution?",
        "",
        "joint_score",
        f"  Joint channel-space discrepancy from {joint_label}.",
        "  It answers: do cross-field combinations such as T/U_1 or CH4/CO behave correctly when fields are mixed together?",
        "",
        "pairwise_mean",
        "  Mean pairwise 2D sliced Wasserstein discrepancy over all channel pairs.",
        f"  Depending on config, it may be diagnostic only or added to mode_score; for this run it is {pairwise_score_text}.",
        "",
        "How to interpret each figure",
        "----------------------------",
        "1_per_channel_w2.png",
        "  Bar chart of per-field 1D W2^2.",
        "  Higher bars indicate which physical variables have the largest marginal distribution mismatch.",
        "  The number above each bar shows the absolute error and its percentage contribution to the sum across channels.",
        "",
        "2_maxswd_theta.png",
        f"  Channel-combination directions from {joint_label}, ranked by per-direction W2^2.",
        "  A large positive or negative coefficient means that field contributes strongly to the discrepancy direction.",
        "  Use this plot to identify which cross-field combinations dominate the joint mismatch.",
        "",
        "2b_joint_projection_spectrum.png",
        "  Ranked projected W2^2 values across the joint projection directions.",
        "  Highlighted points mark directions included in the top-k joint score when available.",
        "",
        "3_maxswd_projected_diagnostics.png",
        "  For the worst joint projection directions, the top panel compares sorted projected values and the bottom panel shows quantile-wise residuals.",
        "  If the two sorted curves separate systematically, the model is misrepresenting that projected distribution.",
        "  The title reports W2^2 for each direction; larger shaded gaps mean larger transport discrepancy.",
        "",
        "4_worst_direction_spatial.png",
        "  Spatial view of the single worst joint projection direction.",
        "  Left: projected reference field. Middle: projected reconstruction. Right: projected discrepancy (recon - ref).",
        "  This figure localizes where the worst joint channel discrepancy lives in physical space.",
        "",
        "5_pairwise_2d_swd.png",
        "  Upper-triangular annotated heatmap of pairwise 2D SWD values.",
        "  Bright/high entries indicate field pairs whose joint marginal distribution is poorly matched.",
        "",
        "6_worst_pair_hexbin.png",
        "  Side-by-side density maps for the worst channel pair according to 2D SWD.",
        "  The panels now use a shared axis range and shared count color scale, so shape differences are visually comparable.",
        "",
        "aggregate_per_channel_w2.png",
        "  Mean per-channel W2^2 across all evaluated snapshots.",
        "",
        "aggregate_pairwise_2d_swd.png",
        "  Mean pairwise 2D SWD across all evaluated snapshots.",
        "",
        "Important interpretation cautions",
        "---------------------------------",
        "A low marginal score with a high joint score means each field individually looks plausible, but cross-field coupling is still wrong.",
        "A high marginal score usually means at least one field has the wrong histogram or dynamic range even before considering coupling.",
        "Normalized-space evaluation focuses on model-space statistical consistency. Physical-space evaluation is easier to interpret in original units but can be dominated by high-variance fields.",
        "Because these are distributional metrics over spatial samples, a figure can look visually smooth while still scoring poorly if the wrong states occur too often or too rarely.",
        "",
        "Resolved global distribution hyperparameters",
        "--------------------------------------------",
        f"lambda_marg     : {coherence_cfg.lambda_marg}",
        f"lambda_joint    : {coherence_cfg.lambda_joint}",
        f"joint_method    : {coherence_cfg.joint_method}",
        f"joint_top_frac  : {coherence_cfg.joint_top_frac}",
        f"joint_qmc       : {coherence_cfg.joint_qmc}",
        f"include_axes    : {coherence_cfg.include_axes}",
        f"exclude_axis_projections_when_marginal_included: {coherence_cfg.exclude_axis_projections_when_marginal_included}",
        f"num_directions  : {coherence_cfg.num_directions} ({hparam_sources.get('num_directions', 'n/a')})",
        f"n_iter_theta    : {coherence_cfg.n_iter_theta} ({hparam_sources.get('n_iter_theta', 'n/a')})",
        f"lr_theta        : {coherence_cfg.lr_theta} ({hparam_sources.get('lr_theta', 'n/a')})",
        f"ortho_reg       : {coherence_cfg.ortho_reg} ({hparam_sources.get('ortho_reg', 'n/a')})",
        f"n_proj_pairwise : {coherence_cfg.n_proj_pairwise} ({hparam_sources.get('n_proj_pairwise', 'n/a')})",
        f"include_pairwise: {coherence_cfg.include_pairwise}",
        f"lambda_pairwise : {coherence_cfg.lambda_pairwise}",
        f"include_pairwise_in_score: {coherence_cfg.include_pairwise_in_score}",
        "",
        "Hyperparameter assessment",
        "-------------------------",
    ]

    lines.extend(f"- {note}" for note in hparam_notes)
    lines.extend([
        "",
        "Practical reading guide",
        "-----------------------",
        "Start with aggregate_metrics.json for the global numbers.",
        "Then inspect 1_per_channel_w2.png to find which fields are most problematic.",
        "Next inspect 2b_joint_projection_spectrum.png, 5_pairwise_2d_swd.png, and 2_maxswd_theta.png to understand whether the problem is mainly pairwise coupling or higher-order channel mixing.",
        "Finally use 3_maxswd_projected_diagnostics.png and 4_worst_direction_spatial.png to see how the worst discrepancy appears in quantile space and physical space.",
        "",
    ])
    return "\n".join(lines)

def save_worst_direction_spatial_map(
    path: Path,
    coords: torch.Tensor,
    x_gen: torch.Tensor,
    x_ref: torch.Tensor,
    theta: torch.Tensor,
    field_names: Sequence[str],
    per_direction_w2: Optional[torch.Tensor] = None,
    direction_mask: Optional[torch.Tensor] = None,
    title: str = "",
) -> None:
    """
    Visualize the worst joint projection back in physical space.

    Panels:
      1) projected reference field
      2) projected reconstruction field
      3) projected discrepancy (recon - ref)
    """
    import matplotlib.tri as mtri

    theta_np = theta.detach().cpu().numpy()
    if per_direction_w2 is not None:
        vals = per_direction_w2.detach().clone()
        if direction_mask is not None:
            mask = direction_mask.detach().to(device=vals.device, dtype=torch.bool)
            if mask.shape == vals.shape and torch.any(mask):
                vals = vals.masked_fill(~mask, -torch.inf)
        dir_idx = int(torch.argmax(vals).item())
        dir_w2 = float(per_direction_w2[dir_idx].detach().cpu())
    else:
        dir_idx = 0
        dir_w2 = None

    vec = theta_np[dir_idx].copy()
    dominant_formula = summarize_direction(vec, field_names, top_k=3)
    x_ref_proj = (x_ref.detach().cpu().numpy() @ vec)
    x_gen_proj = (x_gen.detach().cpu().numpy() @ vec)
    diff = x_gen_proj - x_ref_proj
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    linf = float(np.max(np.abs(diff)))

    coords_np = coords.detach().cpu().numpy()
    tri = mtri.Triangulation(coords_np[:, 0], coords_np[:, 1])

    fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    vmin = min(x_ref_proj.min(), x_gen_proj.min())
    vmax = max(x_ref_proj.max(), x_gen_proj.max())

    im0 = axs[0].tricontourf(tri, x_ref_proj, levels=200, vmin=vmin, vmax=vmax, cmap="viridis")
    axs[0].set_title(f"Reference projection | dir {dir_idx}")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].tricontourf(tri, x_gen_proj, levels=200, vmin=vmin, vmax=vmax, cmap="viridis")
    axs[1].set_title(f"Reconstruction projection | dir {dir_idx}")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    lim = max(abs(diff.min()), abs(diff.max()))
    im2 = axs[2].tricontourf(tri, diff, levels=200, vmin=-lim, vmax=lim, cmap="coolwarm")
    err_title = f"Projected discrepancy | dir {dir_idx}"
    if dir_w2 is not None:
        err_title += f" | W2^2={dir_w2:.3e}"
    axs[2].set_title(err_title)
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("auto")

    fig.text(
        0.5,
        0.01,
        f"dominant direction: {dominant_formula} | projected RMSE={rmse:.3e} | max|recon-ref|={linf:.3e}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)

def save_worst_pair_hexbin(
    path: Path,
    x_gen: torch.Tensor,
    x_ref: torch.Tensor,
    pairwise_mat: torch.Tensor,
    field_names: Sequence[str],
    title: str = "",
    gridsize: int = 70,
) -> None:
    """
    Visualize the worst pairwise 2D marginal discrepancy as side-by-side hexbin plots.
    """
    pair_np = pairwise_mat.detach().cpu().numpy()
    n = pair_np.shape[0]

    best = None
    best_val = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if pair_np[i, j] > best_val:
                best_val = pair_np[i, j]
                best = (i, j)

    if best is None:
        return

    i, j = best
    ref_np = x_ref.detach().cpu().numpy()
    gen_np = x_gen.detach().cpu().numpy()

    xmin = float(min(ref_np[:, i].min(), gen_np[:, i].min()))
    xmax = float(max(ref_np[:, i].max(), gen_np[:, i].max()))
    ymin = float(min(ref_np[:, j].min(), gen_np[:, j].min()))
    ymax = float(max(ref_np[:, j].max(), gen_np[:, j].max()))
    extent = (xmin, xmax, ymin, ymax)

    fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)

    hb0 = axs[0].hexbin(
        ref_np[:, i], ref_np[:, j], gridsize=gridsize, mincnt=1, extent=extent, bins="log", cmap="viridis"
    )
    hb1 = axs[1].hexbin(
        gen_np[:, i], gen_np[:, j], gridsize=gridsize, mincnt=1, extent=extent, bins="log", cmap="viridis"
    )
    vmax = max(float(np.max(hb0.get_array())), float(np.max(hb1.get_array())))
    hb0.set_clim(1, vmax)
    hb1.set_clim(1, vmax)

    axs[0].set_title(f"Reference density")
    axs[0].set_xlabel(field_names[i]); axs[0].set_ylabel(field_names[j])
    axs[1].set_title(f"Reconstruction density")
    axs[1].set_xlabel(field_names[i]); axs[1].set_ylabel(field_names[j])
    plt.colorbar(hb0, ax=axs[0], fraction=0.046, pad=0.04, label="log10(count)")
    plt.colorbar(hb1, ax=axs[1], fraction=0.046, pad=0.04, label="log10(count)")

    fig.suptitle(
        f"{title} | worst pair=({field_names[i]}, {field_names[j]}) | 2D SWD={best_val:.3e} | shared axis/color scale"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)

def save_per_channel_bar(path: Path, values: np.ndarray, field_names: Sequence[str], title: str) -> None:
    """
    Save per-channel marginal discrepancy with numeric annotations.
    """
    values = np.asarray(values)
    total = float(values.sum()) + 1e-12

    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(values))
    bars = plt.bar(x, values)

    plt.xticks(x, field_names, rotation=30)
    plt.ylabel("1D W2^2")
    plt.title(title)

    ymax = max(values.max() * 1.18, 1e-12)
    plt.ylim(0, ymax)

    for i, (b, v) in enumerate(zip(bars, values)):
        frac = 100.0 * float(v) / total
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{v:.2e}\n({frac:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def save_pairwise_heatmap(path: Path, mat: np.ndarray, field_names: Sequence[str], title: str) -> None:
    """
    Save an annotated upper-triangular pairwise discrepancy heatmap.
    """
    mat = np.array(mat, copy=True)
    n = mat.shape[0]

    # Mask diagonal and lower triangle for cleaner visualization.
    mask = np.tril(np.ones_like(mat, dtype=bool))
    disp = mat.copy()
    disp[mask] = np.nan

    plt.figure(figsize=(6.8, 5.8))
    im = plt.imshow(disp, interpolation="nearest", cmap="magma")
    plt.colorbar(im, label="2D SWD")

    plt.xticks(np.arange(n), field_names, rotation=45, ha="right")
    plt.yticks(np.arange(n), field_names)
    plt.title(title)

    finite_vals = mat[np.triu_indices(n, k=1)]
    thresh = float(np.nanmedian(finite_vals)) if finite_vals.size > 0 else 0.0

    # Annotate upper-triangle cells.
    for i in range(n):
        for j in range(n):
            if i < j:
                color = "white" if mat[i, j] <= thresh else "black"
                plt.text(j, i, f"{mat[i, j]:.2e}", ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def save_theta_bar(
    path: Path,
    theta: np.ndarray,
    field_names: Sequence[str],
    title: str,
    per_direction_w2: Optional[np.ndarray] = None,
    top_indices: Optional[Any] = None,
    direction_mask: Optional[Any] = None,
    top_dirs: int = 4,
) -> None:
    """
    Save the joint projection directions as bar charts.

    Directions are optionally sorted by descending per-direction discrepancy.
    """
    k_total = theta.shape[0]
    if top_indices is not None:
        top_np = np.asarray(top_indices.detach().cpu() if torch.is_tensor(top_indices) else top_indices, dtype=int).reshape(-1)
        order = top_np[(top_np >= 0) & (top_np < k_total)]
    elif per_direction_w2 is not None:
        order = np.argsort(-per_direction_w2)
    else:
        order = np.arange(k_total)

    if direction_mask is not None and top_indices is None:
        mask_np = np.asarray(direction_mask.detach().cpu() if torch.is_tensor(direction_mask) else direction_mask, dtype=bool).reshape(-1)
        if mask_np.size == k_total and np.any(mask_np):
            order = np.asarray([idx for idx in order if mask_np[idx]], dtype=int)

    order = order[: min(top_dirs, k_total)]
    if order.size == 0:
        return

    fig, axes = plt.subplots(len(order), 1, figsize=(8, max(3, 2.2 * len(order))), constrained_layout=True)
    if len(order) == 1:
        axes = [axes]

    x = np.arange(theta.shape[1])

    for row_idx, dir_idx in enumerate(order):
        vec = theta[dir_idx].copy()

        # Optional sign convention for readability:
        # flip so the largest-magnitude coefficient is positive.
        j = np.argmax(np.abs(vec))
        if vec[j] < 0:
            vec = -vec

        axes[row_idx].bar(x, vec)
        axes[row_idx].axhline(0.0, color="black", linewidth=0.8)
        axes[row_idx].set_xticks(x)
        axes[row_idx].set_xticklabels(field_names, rotation=30)
        axes[row_idx].set_ylabel("weight")

        if per_direction_w2 is not None:
            axes[row_idx].set_title(
                f"dir {dir_idx} | W2^2 = {per_direction_w2[dir_idx]:.3e} | {summarize_direction(vec, field_names)}"
            )
        else:
            axes[row_idx].set_title(f"dir {dir_idx}")

    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_joint_projection_spectrum(
    path: Path,
    per_direction_w2: Any,
    top_indices: Optional[Any] = None,
    axis_direction_mask: Optional[Any] = None,
    score_mask: Optional[Any] = None,
    title: str = "",
) -> None:
    """
    Plot ranked projected W2^2 values for the joint projection directions.
    """
    vals = np.asarray(per_direction_w2.detach().cpu() if torch.is_tensor(per_direction_w2) else per_direction_w2, dtype=float).reshape(-1)
    if vals.size == 0:
        return

    order = np.argsort(-vals)
    sorted_vals = vals[order]
    ranks = np.arange(1, sorted_vals.size + 1)

    top_mask = np.zeros_like(sorted_vals, dtype=bool)
    if top_indices is not None:
        top_np = np.asarray(top_indices.detach().cpu() if torch.is_tensor(top_indices) else top_indices, dtype=int).reshape(-1)
        if top_np.size > 0:
            top_mask = np.isin(order, top_np)

    axis_rank_mask = np.zeros_like(sorted_vals, dtype=bool)
    if axis_direction_mask is not None:
        axis_np = np.asarray(axis_direction_mask.detach().cpu() if torch.is_tensor(axis_direction_mask) else axis_direction_mask, dtype=bool).reshape(-1)
        if axis_np.size == vals.size:
            axis_rank_mask = axis_np[order]

    excluded_rank_mask = np.zeros_like(sorted_vals, dtype=bool)
    if score_mask is not None:
        score_np = np.asarray(score_mask.detach().cpu() if torch.is_tensor(score_mask) else score_mask, dtype=bool).reshape(-1)
        if score_np.size == vals.size:
            excluded_rank_mask = ~score_np[order]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(ranks, sorted_vals, color="#4C78A8", linewidth=1.4, marker="o", markersize=3.2, label="projection bank")
    if np.any(axis_rank_mask):
        ax.scatter(ranks[axis_rank_mask], sorted_vals[axis_rank_mask], color="#BAB0AC", s=38, zorder=3, label="canonical axes")
    if np.any(excluded_rank_mask):
        ax.scatter(
            ranks[excluded_rank_mask],
            sorted_vals[excluded_rank_mask],
            facecolors="none",
            edgecolors="#666666",
            s=58,
            linewidths=1.1,
            zorder=4,
            label="excluded from joint score",
        )
    if np.any(top_mask):
        ax.scatter(ranks[top_mask], sorted_vals[top_mask], color="#F58518", s=34, zorder=3, label="included in top-k")
        lo = int(np.min(ranks[top_mask]))
        hi = int(np.max(ranks[top_mask]))
        ax.axvspan(lo - 0.5, hi + 0.5, color="#F58518", alpha=0.10, linewidth=0)

    ax.set_xlabel("Projection rank")
    ax.set_ylabel("Projected W2^2")
    ax.set_title(title)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    if np.any(top_mask) or np.any(axis_rank_mask) or np.any(excluded_rank_mask):
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_projected_sorted_curves(
    path: Path,
    x_gen: torch.Tensor,
    x_ref: torch.Tensor,
    theta: torch.Tensor,
    per_direction_w2: Optional[torch.Tensor] = None,
    top_indices: Optional[torch.Tensor] = None,
    direction_mask: Optional[torch.Tensor] = None,
    title: str = "",
    top_dirs: int = 3,
    add_tail_zoom: bool = True,
) -> None:
    """
    Save a more diagnostic version of the joint projected curve plot.

    For the worst directions (ranked by per_direction_w2 if provided), show:
      1) sorted projected curves with shaded absolute gap
      2) residual curve (recon - ref) over quantile
    """
    proj_gen = project_channels(x_gen, theta).detach().cpu().numpy()   # [N, K]
    proj_ref = project_channels(x_ref, theta).detach().cpu().numpy()   # [N, K]

    k_total = proj_gen.shape[1]

    if top_indices is not None:
        order = top_indices.detach().cpu().numpy().astype(int).reshape(-1)
        order = order[(order >= 0) & (order < k_total)]
    elif per_direction_w2 is not None:
        order = np.argsort(-per_direction_w2.detach().cpu().numpy())
    else:
        order = np.arange(k_total)

    if direction_mask is not None and top_indices is None:
        mask_np = direction_mask.detach().cpu().numpy().astype(bool).reshape(-1)
        if mask_np.size == k_total and np.any(mask_np):
            order = np.asarray([idx for idx in order if mask_np[idx]], dtype=int)

    order = order[: min(top_dirs, k_total)]
    if order.size == 0:
        return

    fig, axes = plt.subplots(
        2 * len(order),
        1,
        figsize=(9, max(4, 3.2 * len(order))),
        constrained_layout=True,
    )
    if len(order) == 1:
        axes = [axes[0], axes[1]]

    for panel_idx, dir_idx in enumerate(order):
        ref_sorted = np.sort(proj_ref[:, dir_idx])
        gen_sorted = np.sort(proj_gen[:, dir_idx])

        q = np.linspace(0.0, 1.0, ref_sorted.shape[0])
        diff = gen_sorted - ref_sorted

        ax_top = axes[2 * panel_idx]
        ax_bot = axes[2 * panel_idx + 1]

        # Top panel: projected sorted curves
        ax_top.plot(q, ref_sorted, label="reference", linewidth=1.8)
        ax_top.plot(q, gen_sorted, label="reconstruction", linewidth=1.4)
        ax_top.fill_between(q, ref_sorted, gen_sorted, alpha=0.20)
        ax_top.set_ylabel(f"dir {dir_idx}")

        if per_direction_w2 is not None:
            w2_val = float(per_direction_w2[dir_idx].detach().cpu())
            ax_top.set_title(f"dir {dir_idx} | projected sorted curves | W2^2 = {w2_val:.3e}")
        else:
            ax_top.set_title(f"dir {dir_idx} | projected sorted curves")

        if panel_idx == 0:
            ax_top.legend()

        # Bottom panel: residual curve
        ax_bot.plot(q, diff, linewidth=1.2)
        ax_bot.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax_bot.fill_between(q, 0.0, diff, alpha=0.20)
        ax_bot.set_ylabel("recon - ref")
        ax_bot.set_xlabel("quantile")

        # Optional tail emphasis
        if add_tail_zoom:
            tail_mask = q >= 0.95
            if tail_mask.any():
                tail_max = np.max(np.abs(diff[tail_mask])) + 1e-12
                ax_bot.set_ylim(
                    min(np.min(diff), -1.2 * tail_max),
                    max(np.max(diff),  1.2 * tail_max),
                )

    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)

def main() -> None:
    args = parse_args()
    if args.stat_seed is None:
        args.stat_seed = int(args.seed if args.seed is not None else 0)
    set_seed(args.seed)

    checkpoint_arg = checkpoint_arg_from_args(args)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

    dataset_builder: Optional[Callable[[str], Any]] = None
    checkpoint_path, run_dir = choose_checkpoint(checkpoint_arg)
    checkpoint_for_infer: Optional[Dict[str, Any]] = None
    if args.baseline_model == "auto":
        try:
            checkpoint_for_infer = baseline_lib.safe_torch_load(checkpoint_path, map_location="cpu") if baseline_lib is not None else torch.load(checkpoint_path, map_location="cpu")
        except Exception:
            checkpoint_for_infer = None
        args.baseline_model = infer_checkpoint_family(checkpoint_path, run_dir, checkpoint_for_infer)
        print(f"[*] Auto-detected checkpoint family: {args.baseline_model}")

    if args.baseline_model == "pointcloud_ffm":
        cfg = load_run_config(run_dir)
        demo_dir = infer_demo_dir(run_dir)

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except pickle.UnpicklingError:
            print("[warning] Restricted torch.load failed; retrying with weights_only=False for a trusted local checkpoint.")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        is_ram_checkpoint = isinstance(checkpoint, dict) and str(checkpoint.get("finetune_method", "")).startswith("RAM")
        if not is_ram_checkpoint:
            cfg = resolve_effective_pointcloud_config(run_dir, checkpoint, cfg)

        build_cfg = cfg
        if is_ram_checkpoint:
            source_cfg = checkpoint.get("source_config")
            if source_cfg is None:
                source_cfg_path = run_dir / "source_run_config.yaml"
                if source_cfg_path.exists():
                    with open(source_cfg_path, "r", encoding="utf-8") as handle:
                        source_cfg = yaml.safe_load(handle) or {}
            if source_cfg is None:
                raise ValueError(
                    "RAM-finetuned checkpoint detected, but neither checkpoint['source_config'] "
                    "nor source_run_config.yaml is available for architecture reconstruction."
                )
            source_cfg = normalize_conditioning_args_dict(dict(source_cfg))
            finetune_cfg = dict(checkpoint.get("finetune_config") or cfg)
            eval_cfg = dict(source_cfg)
            eval_cfg.update({key: value for key, value in finetune_cfg.items() if value is not None})
            cfg = normalize_conditioning_args_dict(eval_cfg)
            build_cfg = source_cfg
            print("[*] RAM-finetuned PointCloudFFM checkpoint detected; rebuilding architecture from source_config.")

        # Dataset path and stats path default to the training run settings.
        data_path = args.data or cfg.get("data")
        if data_path is None:
            raise ValueError("Dataset path is missing. Pass --data or ensure args.json contains 'data'.")

        # If the path stored in args.json is relative, resolve it against the demo dir.
        data_path = resolve_input_path(
            str(data_path),
            label="Dataset",
            extra_base_dirs=[run_dir, demo_dir],
        )

        stats_path = run_dir / "dataset_stats.pt"
        dataset = TurbulentCombustionH5Dataset(
            h5_path=str(data_path),
            split=args.split,
            train_ratio=float(cfg.get("train_ratio", 0.9)),
            seed=int(cfg.get("seed", 42)),
            time_stride=int(cfg.get("time_stride", 1)),
            field_names=cfg.get("FIELD_NAMES", cfg.get("field_names", None)),
            stats_path=str(stats_path) if stats_path.exists() else None,
        )

        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        if isinstance(state_dict, dict) and "_metadata" in state_dict:
            state_dict = dict(state_dict)
            state_dict.pop("_metadata", None)

        model = build_model(build_cfg, dataset).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        cond_fields = ensure_list(args.cond_fields) if args.cond_fields is not None else ensure_list(cfg.get("vis_cond_fields"))
        if len(cond_fields) == 0:
            cond_fields = ensure_list(cfg.get("cond_fields", [cfg.get("cond_field", 2)]))

        if args.n_obs_list is not None:
            n_obs_list = broadcast_per_field(
                args.n_obs_list,
                cond_fields,
                "n_obs_list",
                source_fields=cfg.get("cond_fields", [cfg.get("cond_field", 2)]),
            )
        else:
            default_obs = cfg.get("vis_n_obs_list", cfg.get("n_obs_max_list", [cfg.get("n_obs_max", 256)]))
            default_source_fields = (
                cfg.get("vis_cond_fields")
                if cfg.get("vis_n_obs_list") is not None
                else cfg.get("cond_fields", [cfg.get("cond_field", 2)])
            )
            n_obs_list = broadcast_per_field(
                default_obs,
                cond_fields,
                "n_obs_list",
                source_fields=default_source_fields,
            )

        n_steps = int(args.n_steps if args.n_steps is not None else cfg.get("n_steps_generation", 100))
        active_ode_solver = args.ode_solver
        reco_fn = get_reconstruction_fn(args.baseline_model)
    else:
        checkpoint_path, run_dir, cfg, dataset, model, dataset_builder, n_steps, active_ode_solver = load_baseline_context(
            args=args,
            checkpoint_arg=checkpoint_arg,
            device=device,
        )
        demo_dir = infer_demo_dir(run_dir)
        data_path = baseline_lib.ensure_absolute(cfg["shared"]["paths"]["data_path"]) if baseline_lib is not None else Path(cfg["shared"]["paths"]["data_path"])
        stats_path = run_dir / "dataset_stats.pt"
        shared_cond = cfg["shared"]["conditioning"]
        cond_fields = ensure_list(args.cond_fields) if args.cond_fields is not None else ensure_list(shared_cond.get("vis_cond_fields"))
        if len(cond_fields) == 0:
            cond_fields = ensure_list(shared_cond.get("cond_fields"))
        if args.n_obs_list is not None:
            n_obs_list = broadcast_per_field(
                args.n_obs_list,
                cond_fields,
                "n_obs_list",
                source_fields=shared_cond.get("cond_fields"),
            )
        else:
            n_obs_list = broadcast_per_field(
                shared_cond.get("vis_n_obs_list"),
                cond_fields,
                "n_obs_list",
                source_fields=shared_cond.get("vis_cond_fields", shared_cond.get("cond_fields")),
            )
        reco_fn = get_reconstruction_fn(args.baseline_model)

    save_root = (
        resolve_output_path(args.save_root, extra_base_dirs=[demo_dir])
        if args.save_root is not None
        else (demo_dir / "Save_PhyCoEval").resolve()
    )
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = save_root / args.coherence_mode / timestamp
    eval_dir.mkdir(parents=True, exist_ok=True)

    coherence_cfg, hparam_sources, hparam_notes = resolve_global_dist_config(args, num_fields=dataset.num_fields)
    field_names = list(getattr(dataset, "field_names", FIELD_NAMES))

    save_json(eval_dir / "meta.json", {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "data": str(data_path),
        "split": args.split,
        "snapshot_indices": args.snapshot_indices,
        "statistical_analysis": normalize_statistical_modes(args.statistical_analysis),
        "statistical_split": args.statistical_split,
        "stat_max_snapshots": args.stat_max_snapshots,
        "stat_stride": args.stat_stride,
        "stat_seed": args.stat_seed,
        "stat_plot_style": args.stat_plot_style,
        "stat_plot_formats": args.stat_plot_formats,
        "stat_dpi": args.stat_dpi,
        "stat_log_y": args.stat_log_y,
        "stat_panel_labels": args.stat_panel_labels,
        "stat_show_regression": args.stat_show_regression,
        "stat_color_by": args.stat_color_by,
        "stat_jitter_alpha": args.stat_jitter_alpha,
        "stat_point_size": args.stat_point_size,
        "baseline_model": args.baseline_model,
        "coherence_mode": args.coherence_mode,
        "cond_fields": cond_fields,
        "n_obs_list": n_obs_list,
        "n_steps": n_steps,
        "device": str(device),
        "coherence_space": args.coherence_space,
        "global_dist_hparams": {
            "lambda_marg": coherence_cfg.lambda_marg,
            "lambda_joint": coherence_cfg.lambda_joint,
            "joint_method": coherence_cfg.joint_method,
            "joint_top_frac": coherence_cfg.joint_top_frac,
            "joint_qmc": coherence_cfg.joint_qmc,
            "include_axes": coherence_cfg.include_axes,
            "exclude_axis_projections_when_marginal_included": coherence_cfg.exclude_axis_projections_when_marginal_included,
            "num_directions": coherence_cfg.num_directions,
            "n_iter_theta": coherence_cfg.n_iter_theta,
            "lr_theta": coherence_cfg.lr_theta,
            "ortho_reg": coherence_cfg.ortho_reg,
            "n_proj_pairwise": coherence_cfg.n_proj_pairwise,
            "include_pairwise": coherence_cfg.include_pairwise,
            "lambda_pairwise": coherence_cfg.lambda_pairwise,
            "include_pairwise_in_score": coherence_cfg.include_pairwise_in_score,
            "seed": coherence_cfg.seed,
        },
        "global_dist_hparam_sources": hparam_sources,
        "global_dist_hparam_notes": hparam_notes,
    })

    if args.statistical_analysis is not None:
        run_statistical_analysis(
            args=args,
            model=model,
            data_path=data_path,
            cfg=cfg,
            stats_path=stats_path,
            eval_dir=eval_dir,
            device=device,
            cond_fields=cond_fields,
            n_obs_list=n_obs_list,
            n_steps=n_steps,
            coherence_cfg=coherence_cfg,
            reco_fn=reco_fn,
            dataset_builder=dataset_builder,
            ode_solver=active_ode_solver,
        )
        return

    guide_text = build_interpretation_guide_text(
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
        data_path=data_path,
        split=args.split,
        snapshot_indices=args.snapshot_indices,
        cond_fields=cond_fields,
        n_obs_list=n_obs_list,
        n_steps=n_steps,
        coherence_space=args.coherence_space,
        field_names=field_names,
        coherence_cfg=coherence_cfg,
        hparam_sources=hparam_sources,
        hparam_notes=hparam_notes,
    )
    save_text(eval_dir / "coherence_interpretation_guide.txt", guide_text)

    aggregate_per_channel: List[np.ndarray] = []
    aggregate_pairwise: List[np.ndarray] = []
    aggregate_pairwise_means: List[float] = []
    aggregate_mode_scores: List[float] = []
    aggregate_base_scores: List[float] = []
    aggregate_joint_scores: List[float] = []
    aggregate_marg_scores: List[float] = []

    for snapshot_index in args.snapshot_indices:
        snap_dir = eval_dir / f"snapshot_{snapshot_index:04d}"
        snap_dir.mkdir(parents=True, exist_ok=True)

        rec = reco_fn(
            model=model,
            dataset=dataset,
            device=device,
            snapshot_index=int(snapshot_index),
            cond_fields=cond_fields,
            n_obs_list=n_obs_list,
            n_steps=n_steps,
            ode_solver=active_ode_solver,
        )

        truth = rec["truth"]
        recon = rec["recon"]

        if args.coherence_space == "physical":
            truth_eval = denormalize_fields(truth, dataset)
            recon_eval = denormalize_fields(recon, dataset)
        else:
            truth_eval = truth
            recon_eval = recon

        x_ref = truth_eval[0]
        x_gen = recon_eval[0]

        result = compute_coherence(args.coherence_mode, x_gen=x_gen, x_ref=x_ref, cfg=coherence_cfg)

        # Save raw tensors that are useful for later analysis / replotting.
        np.savez(
            snap_dir / "coherence_tensors.npz",
            truth=x_ref.detach().cpu().numpy(),
            recon=x_gen.detach().cpu().numpy(),
            per_channel_w2=result["per_channel_w2"].detach().cpu().numpy(),
            theta=result["theta"].detach().cpu().numpy(),
            per_direction_w2=result["per_direction_w2"].detach().cpu().numpy(),
            **({"top_indices": result["top_indices"].detach().cpu().numpy()} if "top_indices" in result else {}),
            **({"top_values": result["top_values"].detach().cpu().numpy()} if "top_values" in result else {}),
            **({"axis_direction_mask": result["axis_direction_mask"].detach().cpu().numpy()} if "axis_direction_mask" in result else {}),
            **({"joint_score_mask": result["joint_score_mask"].detach().cpu().numpy()} if "joint_score_mask" in result else {}),
            **({"pairwise_2d_swd": result["pairwise_2d_swd"].detach().cpu().numpy()} if "pairwise_2d_swd" in result else {}),
        )

        snap_metrics = {
            "snapshot_index": int(snapshot_index),
            "mode_score": float(result["mode_score"].detach().cpu()),
            "global_dist_score": float(result["global_dist_score"].detach().cpu()),
            "base_score": float(result["base_score"].detach().cpu()),
            "marginal_score": float(result["marginal_score"].detach().cpu()),
            "joint_score": float(result["joint_score"].detach().cpu()),
            "pairwise_score_included": bool(result.get("pairwise_score_included", False)),
            "lambda_pairwise": float(result.get("lambda_pairwise", coherence_cfg.lambda_pairwise)),
            "joint_method": str(result.get("joint_method", coherence_cfg.joint_method)),
            "per_channel_w2": result["per_channel_w2"].detach().cpu().tolist(),
            "per_direction_w2": result["per_direction_w2"].detach().cpu().tolist(),
            "theta": result["theta"].detach().cpu().tolist(),
        }
        if "top_indices" in result:
            snap_metrics["top_indices"] = result["top_indices"].detach().cpu().tolist()
        if "top_values" in result:
            snap_metrics["top_values"] = result["top_values"].detach().cpu().tolist()
        if "axis_direction_mask" in result:
            snap_metrics["axis_direction_mask"] = result["axis_direction_mask"].detach().cpu().tolist()
        if "joint_score_mask" in result:
            snap_metrics["joint_score_mask"] = result["joint_score_mask"].detach().cpu().tolist()
        if "exclude_axes_from_score" in result:
            snap_metrics["exclude_axes_from_score"] = bool(result["exclude_axes_from_score"].detach().cpu().item())
        if "pairwise_mean" in result:
            snap_metrics["pairwise_mean"] = float(result["pairwise_mean"].detach().cpu())
        save_json(snap_dir / "metrics.json", snap_metrics)

        # Plots
        per_channel_np = result["per_channel_w2"].detach().cpu().numpy()
        theta_np = result["theta"].detach().cpu().numpy()
        per_dir_np = result["per_direction_w2"].detach().cpu().numpy()
        theta_title = (
            "Top joint projection directions (adaptive Max-SW)"
            if str(result.get("joint_method", coherence_cfg.joint_method)) == "adaptive_maxswd"
            else "Top projection-bank directions"
        )
        save_per_channel_bar(
            snap_dir / "1_per_channel_w2.png",
            per_channel_np,
            field_names,
            title=f"Per-channel W2^2 | snapshot {snapshot_index} | mean={float(result['marginal_score'].detach().cpu()):.3e}",
        )
        save_theta_bar(
            snap_dir / "2_maxswd_theta.png",
            theta_np,
            field_names,
            title=f"{theta_title} | snapshot {snapshot_index} | joint={float(result['joint_score'].detach().cpu()):.3e}",
            per_direction_w2=per_dir_np,
            top_indices=result.get("top_indices"),
            direction_mask=result.get("joint_score_mask"),
            top_dirs=3,
        )
        save_joint_projection_spectrum(
            snap_dir / "2b_joint_projection_spectrum.png",
            per_direction_w2=result["per_direction_w2"].detach(),
            top_indices=result.get("top_indices"),
            axis_direction_mask=result.get("axis_direction_mask"),
            score_mask=result.get("joint_score_mask"),
            title=f"Joint projection spectrum | snapshot {snapshot_index}",
        )
        save_projected_sorted_curves(
            snap_dir / "3_maxswd_projected_diagnostics.png",
            x_gen=x_gen.detach(),
            x_ref=x_ref.detach(),
            theta=result["theta"].detach(),
            per_direction_w2=result["per_direction_w2"].detach(),
            top_indices=result.get("top_indices"),
            direction_mask=result.get("joint_score_mask"),
            title=f"Projected diagnostics | snapshot {snapshot_index} | joint={float(result['joint_score'].detach().cpu()):.3e}",
            top_dirs=3,
            add_tail_zoom=True,
        )
        save_worst_direction_spatial_map(
            snap_dir / "4_worst_direction_spatial.png",
            coords=rec["coords"][0],
            x_gen=x_gen.detach(),
            x_ref=x_ref.detach(),
            theta=result["theta"].detach(),
            field_names=field_names,
            per_direction_w2=result["per_direction_w2"].detach(),
            direction_mask=result.get("joint_score_mask"),
            title=f"Worst projection spatial map | snapshot {snapshot_index}",
        )
        if "pairwise_2d_swd" in result:
            save_pairwise_heatmap(
                snap_dir / "5_pairwise_2d_swd.png",
                result["pairwise_2d_swd"].detach().cpu().numpy(),
                field_names,
                title=f"Pairwise 2D SWD | snapshot {snapshot_index} | mean={float(result['pairwise_mean'].detach().cpu()):.3e}",
            )
            save_worst_pair_hexbin(
                snap_dir / "6_worst_pair_hexbin.png",
                x_gen=x_gen.detach(),
                x_ref=x_ref.detach(),
                pairwise_mat=result["pairwise_2d_swd"].detach(),
                field_names=field_names,
                title=f"Worst pair diagnostic | snapshot {snapshot_index}",
            )

        aggregate_per_channel.append(result["per_channel_w2"].detach().cpu().numpy())
        aggregate_mode_scores.append(float(result["mode_score"].detach().cpu()))
        aggregate_base_scores.append(float(result["base_score"].detach().cpu()))
        aggregate_joint_scores.append(float(result["joint_score"].detach().cpu()))
        aggregate_marg_scores.append(float(result["marginal_score"].detach().cpu()))
        if "pairwise_2d_swd" in result:
            aggregate_pairwise.append(result["pairwise_2d_swd"].detach().cpu().numpy())
            aggregate_pairwise_means.append(float(result["pairwise_mean"].detach().cpu()))

        print(
            f"[snapshot {snapshot_index}] mode={aggregate_mode_scores[-1]:.6e} | "
            f"marg={aggregate_marg_scores[-1]:.6e} | joint={aggregate_joint_scores[-1]:.6e}"
        )

    # Aggregate reports across snapshots
    per_channel_mean = np.mean(np.stack(aggregate_per_channel, axis=0), axis=0)
    aggregate_report = {
        "num_snapshots": len(args.snapshot_indices),
        "mode_score_mean": float(np.mean(aggregate_mode_scores)),
        "mode_score_std": float(np.std(aggregate_mode_scores)),
        "global_dist_score_mean": float(np.mean(aggregate_mode_scores)),
        "global_dist_score_std": float(np.std(aggregate_mode_scores)),
        "base_score_mean": float(np.mean(aggregate_base_scores)),
        "base_score_std": float(np.std(aggregate_base_scores)),
        "marginal_score_mean": float(np.mean(aggregate_marg_scores)),
        "marginal_score_std": float(np.std(aggregate_marg_scores)),
        "joint_score_mean": float(np.mean(aggregate_joint_scores)),
        "joint_score_std": float(np.std(aggregate_joint_scores)),
        "per_channel_w2_mean": per_channel_mean.tolist(),
    }

    save_json(eval_dir / "aggregate_metrics.json", aggregate_report)
    save_per_channel_bar(
        eval_dir / "aggregate_per_channel_w2.png",
        per_channel_mean,
        field_names,
        title=f"Mean per-channel W2^2 across evaluated snapshots | mean={float(np.mean(aggregate_marg_scores)):.3e}",
    )

    if aggregate_pairwise:
        pairwise_mean = np.mean(np.stack(aggregate_pairwise, axis=0), axis=0)
        aggregate_report["pairwise_mean_mean"] = float(np.mean(aggregate_pairwise_means))
        aggregate_report["pairwise_mean_std"] = float(np.std(aggregate_pairwise_means))
        save_pairwise_heatmap(
            eval_dir / "aggregate_pairwise_2d_swd.png",
            pairwise_mean,
            field_names,
            title=f"Mean pairwise 2D SWD across evaluated snapshots | mean={float(np.mean(aggregate_pairwise_means)):.3e}",
        )
        aggregate_report["pairwise_2d_swd_mean"] = pairwise_mean.tolist()
        save_json(eval_dir / "aggregate_metrics.json", aggregate_report)

    print(f"\n[*] Coherence evaluation saved to: {eval_dir}\n")


if __name__ == "__main__":
    main()

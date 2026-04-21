"""
Evaluate data-driven physical coherence terms on trained reconstruction models.

Current supported mode:
    - global_dist : global distributional coherence

Design goal:
    This script is written to be extensible. 
    In the future, other coherence terms (e.g. spectral or topological) can be added to coherence_dist.py 
    and exposed here through the same command-line interface and save-directory pattern:

        Save_PhyCoEval/{coherence_mode}/{timestamp}/

The script is intentionally evaluation-only. 
It loads a trained checkpoint, reconstructs one or more validation snapshots under sparse conditioning, 
then computes / visualizes coherence discrepancies between the generated and reference fields.

A typical run:
python src/evaluate_coherence.py \
  --checkpoint ../Save_TrainedModel/ffm_tc_pointcloud_DemoN10_20260406_215158 \
  --coherence-mode global_dist \
  --snapshot-indices 0 \
  --n-obs-list 256 256 \
  --n-steps 2

"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml

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

    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to a checkpoint file (.pt) or to a run directory containing best.pt/last.pt.")
    p.add_argument("--data", type=str, default=None,
                   help="Dataset path. Defaults to the training run's args.json value.")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"],
                   help="Dataset split used for evaluation.")
    p.add_argument("--snapshot-indices", type=int, nargs="+", default=[0],
                   help="Snapshot indices within the chosen split to reconstruct and evaluate.")

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
    p.add_argument("--num-directions", type=int, default=None,
                   help="Number of Max-SW channel directions. Defaults to an auto choice based on field count.")
    p.add_argument("--n-iter-theta", type=int, default=None,
                   help="Inner optimization steps for Max-SW directions. Defaults to an auto choice.")
    p.add_argument("--lr-theta", type=float, default=None,
                   help="Inner optimization step size for Max-SW directions. Defaults to an auto choice.")
    p.add_argument("--ortho-reg", type=float, default=None,
                   help="Orthogonality regularization for Max-SW directions. Defaults to an auto choice.")
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


def broadcast_per_field(values: Sequence[int] | int | None, fields: Sequence[int], name: str) -> List[int]:
    vals = ensure_list(values)
    if len(vals) == 0:
        raise ValueError(f"{name} cannot be empty once fields are specified.")
    if len(vals) == 1:
        vals = vals * len(fields)
    if len(vals) != len(fields):
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

    if backbone_name in {"GL_rbf", "hybrid_localglobal_rbf"}:
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


def get_reconstruction_fn() -> Any:
    """
    Use the helper-provided reconstruct_snapshot if the repo has been patched,
    otherwise fall back to the local compatible implementation.
    """
    return helpers_reconstruct_snapshot or reconstruct_snapshot_local


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_text(path: Path, text: str) -> None:
    with open(path, "w") as f:
        f.write(text)


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

    recommended_num_directions = min(max(num_fields, 1), 5)
    recommended_n_iter_theta = 20 if num_fields <= 8 else 12
    recommended_lr_theta = 0.05
    recommended_ortho_reg = 1e-2
    recommended_n_proj_pairwise = 64 if num_fields <= 6 else 48

    num_directions = int(pick("num_directions", args.num_directions, recommended_num_directions))
    n_iter_theta = int(pick("n_iter_theta", args.n_iter_theta, recommended_n_iter_theta))
    lr_theta = float(pick("lr_theta", args.lr_theta, recommended_lr_theta))
    ortho_reg = float(pick("ortho_reg", args.ortho_reg, recommended_ortho_reg))
    n_proj_pairwise = int(pick("n_proj_pairwise", args.n_proj_pairwise, recommended_n_proj_pairwise))

    if sources["num_directions"] == "auto":
        notes.append(
            f"Auto-selected num_directions={num_directions}: this problem has {num_fields} fields, "
            "so using one direction per channel dimension avoids leaving a field-space axis unexplored."
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
            f"Auto-selected n_proj_pairwise={n_proj_pairwise}: with only {num_fields} fields there are few channel pairs, "
            "so using more slice directions reduces Monte Carlo noise in the pairwise diagnostics."
        )

    notes.append(
        "Kept lambda_marg = lambda_joint = 1.0 by default because both terms are squared Wasserstein discrepancies in "
        "the same units, so equal weighting is the most interpretable baseline."
    )
    if args.disable_pairwise:
        notes.append(
            "Pairwise diagnostics were disabled, so pairwise 2D dependence mismatches will not be visualized."
        )
    else:
        notes.append(
            "Pairwise 2D SWD is treated as a diagnostic term only; it is not added into mode_score."
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
        "These metrics do not directly measure pointwise spatial alignment; they measure whether the distribution of field states is realistic.",
        "",
        "mode_score",
        "  lambda_marg * marginal_score + lambda_joint * joint_score.",
        "  This is the main scalar summary used by this evaluator.",
        "",
        "marginal_score",
        "  Mean of the per-channel 1D Wasserstein distances.",
        "  It answers: does each field individually have the right value distribution?",
        "",
        "joint_score",
        "  Mean Max-Sliced Wasserstein discrepancy over learned channel-combination directions.",
        "  It answers: do cross-field combinations such as T/U_1 or CH4/CO behave correctly when fields are mixed together?",
        "",
        "pairwise_mean",
        "  Mean pairwise 2D sliced Wasserstein discrepancy over all channel pairs.",
        "  It is diagnostic only in the current implementation and is not included in mode_score.",
        "",
        "How to interpret each figure",
        "----------------------------",
        "1_per_channel_w2.png",
        "  Bar chart of per-field 1D W2^2.",
        "  Higher bars indicate which physical variables have the largest marginal distribution mismatch.",
        "  The number above each bar shows the absolute error and its percentage contribution to the sum across channels.",
        "",
        "2_maxswd_theta.png",
        "  Learned channel-combination directions ranked by per-direction W2^2.",
        "  A large positive or negative coefficient means that field contributes strongly to the discrepancy direction.",
        "  Use this plot to identify which cross-field combinations dominate the joint mismatch.",
        "",
        "3_maxswd_projected_diagnostics.png",
        "  For the worst Max-SW directions, the top panel compares sorted projected values and the bottom panel shows quantile-wise residuals.",
        "  If the two sorted curves separate systematically, the model is misrepresenting that projected distribution.",
        "  The title reports W2^2 for each direction; larger shaded gaps mean larger transport discrepancy.",
        "",
        "4_worst_direction_spatial.png",
        "  Spatial view of the single worst Max-SW direction.",
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
        f"num_directions  : {coherence_cfg.num_directions} ({hparam_sources.get('num_directions', 'n/a')})",
        f"n_iter_theta    : {coherence_cfg.n_iter_theta} ({hparam_sources.get('n_iter_theta', 'n/a')})",
        f"lr_theta        : {coherence_cfg.lr_theta} ({hparam_sources.get('lr_theta', 'n/a')})",
        f"ortho_reg       : {coherence_cfg.ortho_reg} ({hparam_sources.get('ortho_reg', 'n/a')})",
        f"n_proj_pairwise : {coherence_cfg.n_proj_pairwise} ({hparam_sources.get('n_proj_pairwise', 'n/a')})",
        f"include_pairwise: {coherence_cfg.include_pairwise}",
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
        "Next inspect 5_pairwise_2d_swd.png and 2_maxswd_theta.png to understand whether the problem is mainly pairwise coupling or higher-order channel mixing.",
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
    title: str = "",
) -> None:
    """
    Visualize the worst Max-SW projection back in physical space.

    Panels:
      1) projected reference field
      2) projected reconstruction field
      3) projected discrepancy (recon - ref)
    """
    import matplotlib.tri as mtri

    theta_np = theta.detach().cpu().numpy()
    if per_direction_w2 is not None:
        dir_idx = int(torch.argmax(per_direction_w2).item())
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
    top_dirs: int = 4,
) -> None:
    """
    Save the learned Max-SW directions as bar charts.

    Directions are optionally sorted by descending per-direction discrepancy.
    """
    k_total = theta.shape[0]
    if per_direction_w2 is not None:
        order = np.argsort(-per_direction_w2)
    else:
        order = np.arange(k_total)

    order = order[: min(top_dirs, k_total)]

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

def save_projected_sorted_curves(
    path: Path,
    x_gen: torch.Tensor,
    x_ref: torch.Tensor,
    theta: torch.Tensor,
    per_direction_w2: Optional[torch.Tensor] = None,
    title: str = "",
    top_dirs: int = 3,
    add_tail_zoom: bool = True,
) -> None:
    """
    Save a more diagnostic version of the Max-SW projected curve plot.

    For the worst directions (ranked by per_direction_w2 if provided), show:
      1) sorted projected curves with shaded absolute gap
      2) residual curve (recon - ref) over quantile
    """
    proj_gen = project_channels(x_gen, theta).detach().cpu().numpy()   # [N, K]
    proj_ref = project_channels(x_ref, theta).detach().cpu().numpy()   # [N, K]

    k_total = proj_gen.shape[1]

    if per_direction_w2 is not None:
        order = np.argsort(-per_direction_w2.detach().cpu().numpy())
    else:
        order = np.arange(k_total)

    order = order[: min(top_dirs, k_total)]

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
    set_seed(args.seed)

    checkpoint_path, run_dir = choose_checkpoint(args.checkpoint)
    cfg = load_run_config(run_dir)
    demo_dir = infer_demo_dir(run_dir)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))

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
        stats_path=str(stats_path) if stats_path.exists() else None,
    )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except pickle.UnpicklingError:
        print("[warning] Restricted torch.load failed; retrying with weights_only=False for a trusted local checkpoint.")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = dict(state_dict)
        state_dict.pop("_metadata", None)

    model = build_model(cfg, dataset).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    cond_fields = ensure_list(args.cond_fields) if args.cond_fields is not None else ensure_list(cfg.get("vis_cond_fields"))
    if len(cond_fields) == 0:
        cond_fields = ensure_list(cfg.get("cond_fields", [cfg.get("cond_field", 2)]))

    if args.n_obs_list is not None:
        n_obs_list = broadcast_per_field(args.n_obs_list, cond_fields, "n_obs_list")
    else:
        default_obs = cfg.get("vis_n_obs_list", cfg.get("n_obs_max_list", [cfg.get("n_obs_max", 256)]))
        n_obs_list = broadcast_per_field(default_obs, cond_fields, "n_obs_list")

    n_steps = int(args.n_steps if args.n_steps is not None else cfg.get("n_steps_generation", 100))

    save_root = (
        resolve_output_path(args.save_root, extra_base_dirs=[demo_dir])
        if args.save_root is not None
        else (demo_dir / "Save_PhyCoEval").resolve()
    )
    timestamp = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = save_root / args.coherence_mode / timestamp
    eval_dir.mkdir(parents=True, exist_ok=True)

    coherence_cfg, hparam_sources, hparam_notes = resolve_global_dist_config(args, num_fields=dataset.num_fields)

    save_json(eval_dir / "meta.json", {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "data": str(data_path),
        "split": args.split,
        "snapshot_indices": args.snapshot_indices,
        "coherence_mode": args.coherence_mode,
        "cond_fields": cond_fields,
        "n_obs_list": n_obs_list,
        "n_steps": n_steps,
        "device": str(device),
        "coherence_space": args.coherence_space,
        "global_dist_hparams": {
            "lambda_marg": coherence_cfg.lambda_marg,
            "lambda_joint": coherence_cfg.lambda_joint,
            "num_directions": coherence_cfg.num_directions,
            "n_iter_theta": coherence_cfg.n_iter_theta,
            "lr_theta": coherence_cfg.lr_theta,
            "ortho_reg": coherence_cfg.ortho_reg,
            "n_proj_pairwise": coherence_cfg.n_proj_pairwise,
            "include_pairwise": coherence_cfg.include_pairwise,
            "seed": coherence_cfg.seed,
        },
        "global_dist_hparam_sources": hparam_sources,
        "global_dist_hparam_notes": hparam_notes,
    })

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
        field_names=FIELD_NAMES,
        coherence_cfg=coherence_cfg,
        hparam_sources=hparam_sources,
        hparam_notes=hparam_notes,
    )
    save_text(eval_dir / "coherence_interpretation_guide.txt", guide_text)

    reco_fn = get_reconstruction_fn()

    aggregate_per_channel: List[np.ndarray] = []
    aggregate_pairwise: List[np.ndarray] = []
    aggregate_pairwise_means: List[float] = []
    aggregate_mode_scores: List[float] = []
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
            ode_solver=args.ode_solver,
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
            **({"pairwise_2d_swd": result["pairwise_2d_swd"].detach().cpu().numpy()} if "pairwise_2d_swd" in result else {}),
        )

        snap_metrics = {
            "snapshot_index": int(snapshot_index),
            "mode_score": float(result["mode_score"].detach().cpu()),
            "marginal_score": float(result["marginal_score"].detach().cpu()),
            "joint_score": float(result["joint_score"].detach().cpu()),
            "per_channel_w2": result["per_channel_w2"].detach().cpu().tolist(),
            "per_direction_w2": result["per_direction_w2"].detach().cpu().tolist(),
            "theta": result["theta"].detach().cpu().tolist(),
        }
        if "pairwise_mean" in result:
            snap_metrics["pairwise_mean"] = float(result["pairwise_mean"].detach().cpu())
        save_json(snap_dir / "metrics.json", snap_metrics)

        # Plots
        per_channel_np = result["per_channel_w2"].detach().cpu().numpy()
        theta_np = result["theta"].detach().cpu().numpy()
        per_dir_np = result["per_direction_w2"].detach().cpu().numpy()
        save_per_channel_bar(
            snap_dir / "1_per_channel_w2.png",
            per_channel_np,
            FIELD_NAMES,
            title=f"Per-channel W2^2 | snapshot {snapshot_index} | mean={float(result['marginal_score'].detach().cpu()):.3e}",
        )
        save_theta_bar(
            snap_dir / "2_maxswd_theta.png",
            theta_np,
            FIELD_NAMES,
            title=f"Max-SW directions | snapshot {snapshot_index} | joint={float(result['joint_score'].detach().cpu()):.3e}",
            per_direction_w2=per_dir_np,
            top_dirs=3,
        )
        save_projected_sorted_curves(
            snap_dir / "3_maxswd_projected_diagnostics.png",
            x_gen=x_gen.detach(),
            x_ref=x_ref.detach(),
            theta=result["theta"].detach(),
            per_direction_w2=result["per_direction_w2"].detach(),
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
            field_names=FIELD_NAMES,
            per_direction_w2=result["per_direction_w2"].detach(),
            title=f"Worst projection spatial map | snapshot {snapshot_index}",
        )
        if "pairwise_2d_swd" in result:
            save_pairwise_heatmap(
                snap_dir / "5_pairwise_2d_swd.png",
                result["pairwise_2d_swd"].detach().cpu().numpy(),
                FIELD_NAMES,
                title=f"Pairwise 2D SWD | snapshot {snapshot_index} | mean={float(result['pairwise_mean'].detach().cpu()):.3e}",
            )
            save_worst_pair_hexbin(
                snap_dir / "6_worst_pair_hexbin.png",
                x_gen=x_gen.detach(),
                x_ref=x_ref.detach(),
                pairwise_mat=result["pairwise_2d_swd"].detach(),
                field_names=FIELD_NAMES,
                title=f"Worst pair diagnostic | snapshot {snapshot_index}",
            )

        aggregate_per_channel.append(result["per_channel_w2"].detach().cpu().numpy())
        aggregate_mode_scores.append(float(result["mode_score"].detach().cpu()))
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
        FIELD_NAMES,
        title=f"Mean per-channel W2^2 across evaluated snapshots | mean={float(np.mean(aggregate_marg_scores)):.3e}",
    )

    if aggregate_pairwise:
        pairwise_mean = np.mean(np.stack(aggregate_pairwise, axis=0), axis=0)
        aggregate_report["pairwise_mean_mean"] = float(np.mean(aggregate_pairwise_means))
        aggregate_report["pairwise_mean_std"] = float(np.std(aggregate_pairwise_means))
        save_pairwise_heatmap(
            eval_dir / "aggregate_pairwise_2d_swd.png",
            pairwise_mean,
            FIELD_NAMES,
            title=f"Mean pairwise 2D SWD across evaluated snapshots | mean={float(np.mean(aggregate_pairwise_means)):.3e}",
        )
        aggregate_report["pairwise_2d_swd_mean"] = pairwise_mean.tolist()
        save_json(eval_dir / "aggregate_metrics.json", aggregate_report)

    print(f"\n[*] Coherence evaluation saved to: {eval_dir}\n")


if __name__ == "__main__":
    main()

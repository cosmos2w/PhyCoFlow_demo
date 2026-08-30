"""
Evaluate reconstruction, learned coherence, and flow residuals for COTU0U1P.

This evaluator is for the COTU0U1P turbulent-combustion dataset.  The HDF5
metadata is authoritative (the current file stores [CO, T, U_0, U_1, p]); the
diagnostic/plotting order is [CO, U_0, U_1, T, P].  It evaluates:

  * reconstruction error;
  * data-driven global-distribution coherence;
  * explicit incompressible-flow residuals.

Continuity and pressure-Poisson are the main PDE diagnostics.  Momentum is
optional and is labelled a steady proxy because no physical time derivative is
used.  Cross-spectral diagnostics are intentionally omitted in this revision.

Run from ``0_demo_TurbulentCombustion`` and provide exactly one checkpoint file
or run directory with ``--checkpoint`` or ``--checkpoint-path``.

Example
-------
python src/evaluate_phy_coherence.py \
  --checkpoint-path Save_TrainedModel/<run>/best.pt \
  --data Dataset/Merged_COTU0U1P.h5 \
  --field-names CO T U_0 U_1 P \
  --n-obs-list 256 \
  --split test \
  --max-snapshots 100 \
  --coherence-space normalized \
  --rho-mode unit \
  --rho 1.0 \
  --save-snapshot-plots \

Conditioning is taken from the effective checkpoint configuration by default.
If it is unavailable, the field named ``T`` is used (legacy index 3 is recorded
as the fallback convention, but names and HDF5 metadata take precedence).

All PDE norms use RMS over the complete physical grid.  Derivatives use raw
physical x/y coordinates and denormalized fields.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DEMO_DIR = SCRIPT_DIR.parent
REPO_DIR = DEMO_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import evaluate_coherence as legacy_eval
from coherence_dist import GlobalDistConfig, coherence_result_to_scalars, compute_coherence
from helpers import TurbulentCombustionH5Dataset


EXPECTED_DATA_BASENAME = "Merged_COTU0U1P.h5"
DEFAULT_DATA = "Dataset/Merged_COTU0U1P.h5"
DEFAULT_SAVE_ROOT = "Save_PhyCoEval/phy_coherence"
CHECKPOINT_FAMILIES = (
    "auto",
    "pointcloud_ffm",
    "latent_fm",
    "s3gm",
    "sit",
    "mlp_rbf",
    "senseiver",
    "geofno",
)
EPS = 1e-12


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate reconstruction, distribution coherence, and physical flow residuals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    checkpoint_help = "One checkpoint file or run directory to evaluate."
    checkpoint_group = p.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", default=None, help=checkpoint_help)
    checkpoint_group.add_argument("--checkpoint-path", default=None, help="Alias for --checkpoint.")
    p.add_argument("--baseline-model", choices=CHECKPOINT_FAMILIES, default="auto")

    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--snapshot-indices", type=int, nargs="+", default=None)
    p.add_argument("--max-snapshots", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--save-root", default=DEFAULT_SAVE_ROOT)

    p.add_argument("--field-names", nargs="+", default=["CO", "T", "U_0", "U_1", "P"])
    p.add_argument("--co-field", default="CO")
    p.add_argument("--u0-field", default="U_0")
    p.add_argument("--u1-field", default="U_1")
    p.add_argument("--temperature-field", default="T")
    p.add_argument("--pressure-field", default="P")
    p.add_argument("--cond-fields", type=int, nargs="+", default=None)
    p.add_argument("--n-obs-list", type=int, nargs="+", default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--ode-solver", default=None)

    p.add_argument("--coherence-space", choices=("normalized", "physical"), default="normalized")
    p.add_argument("--lambda-marg", type=float, default=1.0)
    p.add_argument("--lambda-joint", type=float, default=1.0)
    p.add_argument("--lambda-pairwise", type=float, default=0.25)
    p.add_argument("--joint-method", choices=("topk_swd", "adaptive_maxswd"), default="topk_swd")
    p.add_argument("--num-directions", type=int, default=64)
    p.add_argument("--joint-top-frac", type=float, default=0.10)
    p.add_argument("--n-proj-pairwise", type=int, default=32)
    p.add_argument("--disable-pairwise", action="store_true")
    p.add_argument("--include-pairwise-in-score", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--rho", type=float, default=1.0)
    p.add_argument("--rho-mode", choices=("unit", "fit_global_gt"), default="unit")
    p.add_argument("--nu", type=float, default=0.0)
    p.add_argument("--compute-momentum-proxy", action=argparse.BooleanOptionalAction, default=False)

    p.add_argument("--require-grid", action="store_true")
    p.add_argument("--grid-tolerance", type=float, default=1e-7)
    p.add_argument("--save-snapshot-plots", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-npz", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--plot-formats", nargs="+", default=["png", "pdf"], choices=("png", "pdf", "svg"))
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    print(f"[warning] {message}")


def canonical_field_name(name: Any) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def resolve_field_index(field_names: Sequence[str], requested: str) -> int:
    key = canonical_field_name(requested)
    matches = [i for i, name in enumerate(field_names) if canonical_field_name(name) == key]
    if len(matches) != 1:
        raise ValueError(
            f"Could not uniquely resolve field {requested!r} from HDF5-resolved names {tuple(field_names)!r}."
        )
    return matches[0]


def resolve_required_fields(dataset: TurbulentCombustionH5Dataset, args: argparse.Namespace) -> Dict[str, int]:
    names = tuple(dataset.field_names)
    if dataset.num_fields != len(names):
        raise ValueError(f"Dataset reports {dataset.num_fields} fields but resolved {len(names)} names: {names}")
    if dataset.num_fields != len(args.field_names):
        raise ValueError(
            f"Dataset has {dataset.num_fields} fields, but --field-names has {len(args.field_names)} entries."
        )
    return {
        "CO": resolve_field_index(names, args.co_field),
        "U_0": resolve_field_index(names, args.u0_field),
        "U_1": resolve_field_index(names, args.u1_field),
        "T": resolve_field_index(names, args.temperature_field),
        "P": resolve_field_index(names, args.pressure_field),
    }


def _resolve_relative(path_like: str | Path, bases: Sequence[Path]) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (bases[0] / path).resolve()


def resolve_and_validate_data(path_like: str | Path) -> Path:
    path = _resolve_relative(path_like, (Path.cwd(), DEMO_DIR, REPO_DIR))
    expected = (DEMO_DIR / DEFAULT_DATA).resolve()
    if path.name != EXPECTED_DATA_BASENAME or path != expected:
        raise ValueError(
            "evaluate_phy_coherence.py is locked to the new turbulent-combustion dataset: "
            f"{expected}. Refusing requested dataset: {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    return path


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if torch.is_tensor(value):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True)


def save_figure(fig: Any, stem: Path, formats: Sequence[str], dpi: int) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def load_effective_pointcloud_config(run_dir: Path, checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    """Use the shared standard/post-training PointCloudFFM config resolver."""
    yaml_cfg = legacy_eval.load_run_config(run_dir)
    return legacy_eval.resolve_effective_pointcloud_config(run_dir, checkpoint, yaml_cfg)


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _checkpoint_value(args: argparse.Namespace) -> str:
    return str(args.checkpoint_path if args.checkpoint_path is not None else args.checkpoint)


def load_model_context(
    checkpoint_arg: str,
    args: argparse.Namespace,
    device: torch.device,
    data_path: Path,
    label: str,
) -> Dict[str, Any]:
    checkpoint_path, run_dir = legacy_eval.choose_checkpoint(checkpoint_arg)
    checkpoint_cpu = _torch_load(checkpoint_path, torch.device("cpu"))
    requested_family = args.baseline_model
    family = (
        legacy_eval.infer_checkpoint_family(checkpoint_path, run_dir, checkpoint_cpu)
        if requested_family == "auto"
        else requested_family
    )

    if family == "pointcloud_ffm":
        # Keep optimizer/checkpoint bookkeeping on CPU; load_state_dict copies
        # only model weights into the requested CPU/CUDA model below.
        checkpoint = checkpoint_cpu
        cfg = load_effective_pointcloud_config(run_dir, checkpoint if isinstance(checkpoint, Mapping) else {})
        stats_path = run_dir / "dataset_stats.pt"
        dataset = TurbulentCombustionH5Dataset(
            h5_path=str(data_path),
            split=args.split,
            train_ratio=float(cfg.get("train_ratio", 0.9)),
            seed=int(cfg.get("seed", args.seed)),
            time_stride=int(cfg.get("time_stride", 1)),
            field_names=args.field_names,
            stats_path=str(stats_path) if stats_path.exists() else None,
        )
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
        if isinstance(state_dict, Mapping) and "_metadata" in state_dict:
            state_dict = dict(state_dict)
            state_dict.pop("_metadata", None)
        model = legacy_eval.build_model(cfg, dataset).to(device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        reco_fn = legacy_eval.get_reconstruction_fn("pointcloud_ffm")
        checkpoint_field_names = checkpoint.get("field_names") if isinstance(checkpoint, Mapping) else None
    else:
        baseline_args = argparse.Namespace(**vars(args))
        baseline_args.baseline_model = family
        (
            checkpoint_path,
            run_dir,
            cfg,
            dataset,
            model,
            _dataset_builder,
            baseline_steps,
            baseline_solver,
        ) = legacy_eval.load_baseline_context(
            args=baseline_args,
            checkpoint_arg=checkpoint_arg,
            device=device,
        )
        dataset_path = Path(getattr(dataset, "h5_path", data_path)).resolve()
        if dataset_path != data_path:
            raise ValueError(
                f"Baseline configuration resolved dataset {dataset_path}; this evaluator only permits {data_path}."
            )
        family = baseline_args.baseline_model
        reco_fn = legacy_eval.get_reconstruction_fn(family)
        checkpoint_field_names = checkpoint_cpu.get("field_names") if isinstance(checkpoint_cpu, Mapping) else None

    fields = resolve_required_fields(dataset, args)
    if checkpoint_field_names is not None:
        ckpt_names = tuple(str(x) for x in checkpoint_field_names)
        data_names = tuple(str(x) for x in dataset.field_names)
        if tuple(map(canonical_field_name, ckpt_names)) != tuple(map(canonical_field_name, data_names)):
            warn(
                f"Checkpoint {checkpoint_path.name} has stale/different field-name metadata {ckpt_names}; "
                f"using authoritative HDF5 names {data_names}."
            )

    if family == "pointcloud_ffm":
        cfg_cond = cfg
        configured_cond = cfg_cond.get("vis_cond_fields") or cfg_cond.get("cond_fields")
        cond_fields = list(args.cond_fields) if args.cond_fields is not None else legacy_eval.ensure_list(configured_cond)
        if not cond_fields:
            cond_fields = [fields["T"]]
        if args.n_obs_list is not None:
            n_obs_list = legacy_eval.broadcast_per_field(
                args.n_obs_list, cond_fields, "n_obs_list", source_fields=cfg_cond.get("cond_fields")
            )
        else:
            default_obs = cfg_cond.get("vis_n_obs_list") or cfg_cond.get("n_obs_max_list") or [256]
            default_source = cfg_cond.get("vis_cond_fields") or cfg_cond.get("cond_fields")
            n_obs_list = legacy_eval.broadcast_per_field(
                default_obs, cond_fields, "n_obs_list", source_fields=default_source
            )
        n_steps = int(args.n_steps if args.n_steps is not None else cfg_cond.get("n_steps_generation", 32))
        ode_solver = args.ode_solver or cfg_cond.get("ode_solver")
    else:
        shared_cond = cfg["shared"]["conditioning"]
        configured_cond = shared_cond.get("vis_cond_fields") or shared_cond.get("cond_fields")
        cond_fields = list(args.cond_fields) if args.cond_fields is not None else legacy_eval.ensure_list(configured_cond)
        if not cond_fields:
            cond_fields = [fields["T"]]
        selected_obs = args.n_obs_list if args.n_obs_list is not None else (
            shared_cond.get("vis_n_obs_list") or shared_cond.get("n_obs_max_list") or [256]
        )
        n_obs_list = legacy_eval.broadcast_per_field(
            selected_obs,
            cond_fields,
            "n_obs_list",
            source_fields=shared_cond.get("vis_cond_fields") or shared_cond.get("cond_fields"),
        )
        n_steps = int(args.n_steps if args.n_steps is not None else baseline_steps)
        ode_solver = args.ode_solver or baseline_solver

    if any(index < 0 or index >= dataset.num_fields for index in cond_fields):
        raise ValueError(f"Conditioned field indices {cond_fields} are invalid for {dataset.num_fields} fields.")

    return {
        "label": label,
        "family": family,
        "checkpoint_path": checkpoint_path,
        "run_dir": run_dir,
        "cfg": cfg,
        "dataset": dataset,
        "model": model,
        "reco_fn": reco_fn,
        "fields": fields,
        "cond_fields": cond_fields,
        "n_obs_list": n_obs_list,
        "n_steps": n_steps,
        "ode_solver": ode_solver,
    }


def select_snapshot_indices(
    n_items: int,
    explicit: Optional[Sequence[int]],
    stride: int,
    max_snapshots: Optional[int],
    seed: int,
) -> List[int]:
    if stride <= 0:
        raise ValueError(f"--stride must be positive, got {stride}")
    if max_snapshots is not None and max_snapshots <= 0:
        raise ValueError(f"--max-snapshots must be positive, got {max_snapshots}")
    if explicit is not None:
        indices = [int(i) for i in explicit]
    else:
        indices = list(range(0, n_items, stride))
        if max_snapshots is not None and len(indices) > max_snapshots:
            rng = np.random.default_rng(seed)
            indices = sorted(int(i) for i in rng.choice(indices, size=max_snapshots, replace=False))
    if any(i < 0 or i >= n_items for i in indices):
        raise IndexError(f"Snapshot indices must lie in [0, {n_items - 1}], got {indices}")
    if not indices:
        raise ValueError("No snapshots selected.")
    return indices


def infer_regular_grid(coords_raw: np.ndarray, tolerance: float = 1e-7) -> Dict[str, Any]:
    """Infer a complete tensor-product grid and map [ny,nx] cells to point indices."""
    coords = np.asarray(coords_raw, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Expected raw coordinates [N, >=2], got {coords.shape}")
    if tolerance <= 0:
        raise ValueError(f"grid tolerance must be positive, got {tolerance}")
    decimals = max(0, int(math.ceil(-math.log10(tolerance))))
    xr = np.round(coords[:, 0], decimals)
    yr = np.round(coords[:, 1], decimals)
    x_unique = np.unique(xr)
    y_unique = np.unique(yr)
    nx, ny = x_unique.size, y_unique.size
    n = coords.shape[0]
    if nx < 3 or ny < 3:
        raise ValueError(f"Need at least 3x3 points for edge_order=2 derivatives; inferred nx={nx}, ny={ny}.")
    if nx * ny != n:
        raise ValueError(f"Incomplete tensor-product grid: nx={nx}, ny={ny}, nx*ny={nx * ny}, N={n}.")
    xi = np.searchsorted(x_unique, xr)
    yi = np.searchsorted(y_unique, yr)
    linear = yi * nx + xi
    if np.unique(linear).size != n:
        raise ValueError("Rounded coordinates contain duplicate tensor-product cells.")
    grid_to_point = np.empty(n, dtype=np.int64)
    grid_to_point[linear] = np.arange(n, dtype=np.int64)
    # Use means of the original physical coordinates in each rounded bucket.
    x_physical = np.asarray([coords[xi == i, 0].mean() for i in range(nx)], dtype=np.float64)
    y_physical = np.asarray([coords[yi == i, 1].mean() for i in range(ny)], dtype=np.float64)
    if not (np.all(np.diff(x_physical) > 0) and np.all(np.diff(y_physical) > 0)):
        raise ValueError("Physical x/y coordinates are not strictly increasing after grid inference.")
    return {
        "nx": int(nx),
        "ny": int(ny),
        "x_unique": x_physical,
        "y_unique": y_physical,
        "grid_to_point": grid_to_point,
        "tolerance": float(tolerance),
    }


def flat_to_grid(values: np.ndarray, grid: Mapping[str, Any]) -> np.ndarray:
    flat = np.asarray(values)
    return flat[np.asarray(grid["grid_to_point"], dtype=np.int64)].reshape(int(grid["ny"]), int(grid["nx"]))


def gradient_x_y(
    f_grid: np.ndarray, x_unique: np.ndarray, y_unique: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (df/dx, df/dy) using physical coordinate arrays."""
    df_dy, df_dx = np.gradient(
        np.asarray(f_grid, dtype=np.float64),
        np.asarray(y_unique, dtype=np.float64),
        np.asarray(x_unique, dtype=np.float64),
        edge_order=2,
    )
    return df_dx, df_dy


def laplacian(f_grid: np.ndarray, x_unique: np.ndarray, y_unique: np.ndarray) -> np.ndarray:
    fx, fy = gradient_x_y(f_grid, x_unique, y_unique)
    fxx, _ = gradient_x_y(fx, x_unique, y_unique)
    _, fyy = gradient_x_y(fy, x_unique, y_unique)
    return fxx + fyy


def rms(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(value))))


def flow_residual_arrays(
    fields_physical: np.ndarray,
    grid: Mapping[str, Any],
    field_indices: Mapping[str, int],
    rho: float,
    nu: float,
    momentum: bool,
) -> Dict[str, np.ndarray]:
    x = np.asarray(grid["x_unique"])
    y = np.asarray(grid["y_unique"])
    u = flat_to_grid(fields_physical[:, field_indices["U_0"]], grid)
    v = flat_to_grid(fields_physical[:, field_indices["U_1"]], grid)
    p = flat_to_grid(fields_physical[:, field_indices["P"]], grid)
    ux, uy = gradient_x_y(u, x, y)
    vx, vy = gradient_x_y(v, x, y)
    div = ux + vy
    strain_a = ux**2 + 2.0 * uy * vx + vy**2
    lap_p = laplacian(p, x, y)
    ppe = lap_p + float(rho) * strain_a
    omega = vx - uy
    out = {
        "u": u,
        "v": v,
        "p": p,
        "ux": ux,
        "uy": uy,
        "vx": vx,
        "vy": vy,
        "div": div,
        "A": strain_a,
        "lap_p": lap_p,
        "ppe": ppe,
        "omega": omega,
    }
    if momentum:
        px, py = gradient_x_y(p, x, y)
        lap_u = laplacian(u, x, y)
        lap_v = laplacian(v, x, y)
        out["mom_x"] = u * ux + v * uy + px / float(rho) - float(nu) * lap_u
        out["mom_y"] = u * vx + v * vy + py / float(rho) - float(nu) * lap_v
    return out


def compute_pde_metrics(
    gt: Mapping[str, np.ndarray],
    rec: Mapping[str, np.ndarray],
    rho: float,
    momentum: bool,
) -> Dict[str, float]:
    grad_gt = math.sqrt(rms(gt["ux"]) ** 2 + rms(gt["uy"]) ** 2 + rms(gt["vx"]) ** 2 + rms(gt["vy"]) ** 2)
    grad_rec = math.sqrt(rms(rec["ux"]) ** 2 + rms(rec["uy"]) ** 2 + rms(rec["vx"]) ** 2 + rms(rec["vy"]) ** 2)
    ppe_scale_gt = rms(gt["lap_p"]) + rms(float(rho) * gt["A"])
    ppe_scale_rec = rms(rec["lap_p"]) + rms(float(rho) * rec["A"])
    metrics = {
        "div_rmse_gt": rms(gt["div"]),
        "div_rmse_rec": rms(rec["div"]),
        "div_rel_gt": rms(gt["div"]) / (grad_gt + EPS),
        "div_rel_rec": rms(rec["div"]) / (grad_rec + EPS),
        "div_excess_rel": rms(rec["div"] - gt["div"]) / (grad_gt + EPS),
        "ppe_rel_gt": rms(gt["ppe"]) / (ppe_scale_gt + EPS),
        "ppe_rel_rec": rms(rec["ppe"]) / (ppe_scale_rec + EPS),
        "ppe_excess_rel": rms(rec["ppe"] - gt["ppe"]) / (ppe_scale_gt + EPS),
        "omega_rel_l2": rms(rec["omega"] - gt["omega"]) / (rms(gt["omega"]) + EPS),
    }
    if momentum:
        mom_gt = math.sqrt(rms(gt["mom_x"]) ** 2 + rms(gt["mom_y"]) ** 2)
        mom_rec = math.sqrt(rms(rec["mom_x"]) ** 2 + rms(rec["mom_y"]) ** 2)
        mom_delta = math.sqrt(rms(rec["mom_x"] - gt["mom_x"]) ** 2 + rms(rec["mom_y"] - gt["mom_y"]) ** 2)
        metrics.update(
            {
                "steady_momentum_proxy_rmse_gt": mom_gt,
                "steady_momentum_proxy_rmse_rec": mom_rec,
                "steady_momentum_proxy_excess_rel": mom_delta / (mom_gt + EPS),
            }
        )
    return metrics


PDE_METRIC_KEYS = (
    "div_rmse_gt",
    "div_rmse_rec",
    "div_rel_gt",
    "div_rel_rec",
    "div_excess_rel",
    "ppe_rel_gt",
    "ppe_rel_rec",
    "ppe_excess_rel",
    "omega_rel_l2",
    "steady_momentum_proxy_rmse_gt",
    "steady_momentum_proxy_rmse_rec",
    "steady_momentum_proxy_excess_rel",
)


def nan_pde_metrics() -> Dict[str, float]:
    return {key: float("nan") for key in PDE_METRIC_KEYS}


def denormalize(x: torch.Tensor, dataset: TurbulentCombustionH5Dataset) -> torch.Tensor:
    return legacy_eval.denormalize_fields(x, dataset)


def reconstruction_metrics(
    truth_norm: torch.Tensor,
    recon_norm: torch.Tensor,
    truth_phys: torch.Tensor,
    recon_phys: torch.Tensor,
    field_names: Sequence[str],
) -> Dict[str, float]:
    gt = truth_phys[0].detach().cpu().numpy().astype(np.float64)
    pr = recon_phys[0].detach().cpu().numpy().astype(np.float64)
    out = {
        "rel_l2_all": float(np.linalg.norm(pr - gt) / (np.linalg.norm(gt) + EPS)),
        "rel_l2_all_normalized": legacy_eval.compute_relative_l2(recon_norm, truth_norm),
    }
    for i, name in enumerate(field_names):
        diff = pr[:, i] - gt[:, i]
        key = str(name)
        out[f"rel_l2_{key}"] = float(np.linalg.norm(diff) / (np.linalg.norm(gt[:, i]) + EPS))
        out[f"rmse_{key}"] = rms(diff)
        out[f"mae_{key}"] = float(np.mean(np.abs(diff)))
    return out


def _rename_required_field_metric_keys(metrics: Dict[str, float], names: Sequence[str], fields: Mapping[str, int]) -> None:
    for canonical, index in fields.items():
        actual = str(names[index])
        for prefix in ("rel_l2", "rmse", "mae"):
            source = f"{prefix}_{actual}"
            metrics[f"{prefix}_{canonical}"] = metrics[source]


def temperature_observation_error(rec: Mapping[str, torch.Tensor], recon_norm: torch.Tensor, idx_t: int) -> float:
    ids = rec.get("obs_field_ids")
    mask = rec.get("obs_mask")
    if ids is None or mask is None:
        return float("nan")
    t_mask = mask.bool() & (ids.long() == int(idx_t))
    if not bool(t_mask.any()):
        return float("nan")
    return legacy_eval.compute_observation_consistency_error(
        pred_full=recon_norm,
        obs_coords=rec["obs_coords"],
        obs_values=rec["obs_values"],
        obs_mask=t_mask,
        coords_full=rec["coords"],
        obs_indices=rec.get("obs_indices"),
        obs_field_ids=ids,
    )


def make_coherence_config(args: argparse.Namespace) -> GlobalDistConfig:
    return GlobalDistConfig(
        lambda_marg=float(args.lambda_marg),
        lambda_joint=float(args.lambda_joint),
        num_directions=int(args.num_directions),
        n_iter_theta=20,
        lr_theta=0.1,
        ortho_reg=1e-2,
        n_proj_pairwise=int(args.n_proj_pairwise),
        include_pairwise=not args.disable_pairwise,
        seed=int(args.seed),
        joint_method=args.joint_method,
        joint_top_frac=float(args.joint_top_frac),
        joint_qmc=True,
        include_axes=True,
        lambda_pairwise=float(args.lambda_pairwise),
        include_pairwise_in_score=bool(args.include_pairwise_in_score),
        exclude_axis_projections_when_marginal_included=True,
    )


def compute_distribution_metrics(
    truth_norm: torch.Tensor,
    recon_norm: torch.Tensor,
    truth_phys: torch.Tensor,
    recon_phys: torch.Tensor,
    args: argparse.Namespace,
    cfg: GlobalDistConfig,
) -> Tuple[Dict[str, float], Dict[str, Any], torch.Tensor, torch.Tensor]:
    if args.coherence_space == "physical":
        x_ref, x_gen = truth_phys[0], recon_phys[0]
    else:
        x_ref, x_gen = truth_norm[0], recon_norm[0]
    result = compute_coherence("global_dist", x_gen=x_gen, x_ref=x_ref, cfg=cfg)
    scalars = coherence_result_to_scalars(result)
    pairwise = scalars["pairwise_mean"]
    dist_mutual = scalars["joint_score"]
    if args.include_pairwise_in_score and math.isfinite(pairwise):
        dist_mutual += float(args.lambda_pairwise) * pairwise
    scalars["dist_self"] = scalars["marginal_score"]
    scalars["dist_mutual"] = dist_mutual
    return scalars, result, x_ref, x_gen


def _sensor_xy(rec: Mapping[str, torch.Tensor], coords_raw: np.ndarray, field_idx: int) -> np.ndarray:
    if rec.get("obs_indices") is None or rec.get("obs_field_ids") is None or rec.get("obs_mask") is None:
        return np.empty((0, 2))
    valid = rec["obs_mask"].bool() & (rec["obs_field_ids"].long() == int(field_idx))
    indices = rec["obs_indices"][valid].detach().cpu().numpy().astype(int)
    return coords_raw[indices, :2]


def _field_color_limits(gt: np.ndarray, rec: np.ndarray) -> Tuple[float, float]:
    values = np.concatenate([np.ravel(gt), np.ravel(rec)])
    lo, hi = np.nanpercentile(values, [1, 99])
    if not np.isfinite(lo + hi) or lo == hi:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values) + EPS)
    return float(lo), float(hi)


def _add_error_metric_box(ax: Any, primary_label: str, primary_value: float, rmse_value: float) -> None:
    """Place quantitative error metrics directly on an error-map panel."""
    primary_text = f"{primary_label} = {primary_value:.2e}" if math.isfinite(primary_value) else f"{primary_label} = N/A"
    rmse_text = f"RMSE = {rmse_value:.2e}" if math.isfinite(rmse_value) else "RMSE = N/A"
    ax.text(
        0.985,
        0.94,
        f"{primary_text}\n{rmse_text}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="black",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "0.25",
            "linewidth": 0.4,
            "alpha": 0.88,
        },
        zorder=6,
    )


def plot_fields(
    truth_phys: np.ndarray,
    recon_phys: np.ndarray,
    grid: Mapping[str, Any],
    fields: Mapping[str, int],
    sensors_t: np.ndarray,
    stem: Path,
    args: argparse.Namespace,
) -> None:
    order = ("CO", "U_0", "U_1", "T", "P")
    x, y = np.asarray(grid["x_unique"]), np.asarray(grid["y_unique"])
    fig, axes = plt.subplots(len(order), 3, figsize=(14.5, 8.2), constrained_layout=True, sharex=True, sharey=True)
    for row, name in enumerate(order):
        gt = flat_to_grid(truth_phys[:, fields[name]], grid)
        pr = flat_to_grid(recon_phys[:, fields[name]], grid)
        err = np.abs(pr - gt)
        vmin = float(min(np.nanmin(gt), np.nanmin(pr)))
        vmax = float(max(np.nanmax(gt), np.nanmax(pr)))
        emax = max(float(np.nanmax(err)), EPS)
        ims = (
            axes[row, 0].pcolormesh(x, y, gt, shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax),
            axes[row, 1].pcolormesh(x, y, pr, shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax),
            axes[row, 2].pcolormesh(x, y, err, shading="auto", cmap="inferno", vmin=0.0, vmax=emax),
        )
        rel_l2 = float(np.linalg.norm(err.reshape(-1)) / (np.linalg.norm(gt.reshape(-1)) + EPS))
        _add_error_metric_box(axes[row, 2], "rel. L2", rel_l2, rms(err))
        axes[row, 0].set_ylabel(name)
        for col, im in enumerate(ims):
            fig.colorbar(im, ax=axes[row, col], fraction=0.025, pad=0.02)
        if name == "T" and sensors_t.size:
            for ax in axes[row, :2]:
                ax.scatter(
                    sensors_t[:, 0], sensors_t[:, 1], s=5, facecolors="none",
                    edgecolors="tab:green", linewidths=0.45, zorder=4,
                )
        for ax in axes[row]:
            ax.set_xlim(float(x.min()), float(x.max()))
            ax.set_ylim(float(y.min()), float(y.max()))
            ax.set_aspect("equal", adjustable="box")
    for col, title in enumerate(("Ground truth", "Reconstruction", "Absolute error")):
        axes[0, col].set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("x (physical)")
    save_figure(fig, stem, args.plot_formats, args.dpi)


def plot_fields_unstructured(
    truth_phys: np.ndarray,
    recon_phys: np.ndarray,
    coords_raw: np.ndarray,
    fields: Mapping[str, int],
    sensors_t: np.ndarray,
    stem: Path,
    args: argparse.Namespace,
) -> None:
    """Fallback field plate for a point cloud that is not a complete grid."""
    order = ("CO", "U_0", "U_1", "T", "P")
    xy = np.asarray(coords_raw)[:, :2]
    fig, axes = plt.subplots(len(order), 3, figsize=(14.5, 8.2), constrained_layout=True, sharex=True, sharey=True)
    for row, name in enumerate(order):
        gt = truth_phys[:, fields[name]]
        pr = recon_phys[:, fields[name]]
        err = np.abs(pr - gt)
        vmin = float(min(np.nanmin(gt), np.nanmin(pr)))
        vmax = float(max(np.nanmax(gt), np.nanmax(pr)))
        emax = max(float(np.nanmax(err)), EPS)
        ims = (
            axes[row, 0].scatter(xy[:, 0], xy[:, 1], c=gt, s=2, cmap="coolwarm", vmin=vmin, vmax=vmax),
            axes[row, 1].scatter(xy[:, 0], xy[:, 1], c=pr, s=2, cmap="coolwarm", vmin=vmin, vmax=vmax),
            axes[row, 2].scatter(xy[:, 0], xy[:, 1], c=err, s=2, cmap="inferno", vmin=0.0, vmax=emax),
        )
        rel_l2 = float(np.linalg.norm(err.reshape(-1)) / (np.linalg.norm(gt.reshape(-1)) + EPS))
        _add_error_metric_box(axes[row, 2], "rel. L2", rel_l2, rms(err))
        axes[row, 0].set_ylabel(name)
        for col, im in enumerate(ims):
            fig.colorbar(im, ax=axes[row, col], fraction=0.025, pad=0.02)
        if name == "T" and sensors_t.size:
            for ax in axes[row, :2]:
                ax.scatter(
                    sensors_t[:, 0], sensors_t[:, 1], s=5, facecolors="none",
                    edgecolors="tab:green", linewidths=0.45, zorder=4,
                )
        for ax in axes[row]:
            ax.set_xlim(float(xy[:, 0].min()), float(xy[:, 0].max()))
            ax.set_ylim(float(xy[:, 1].min()), float(xy[:, 1].max()))
            ax.set_aspect("equal", adjustable="box")
    for col, title in enumerate(("Ground truth", "Reconstruction", "Absolute error")):
        axes[0, col].set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("x (physical)")
    save_figure(fig, stem, args.plot_formats, args.dpi)


def plot_pde(
    gt: Mapping[str, np.ndarray],
    rec: Mapping[str, np.ndarray],
    grid: Mapping[str, Any],
    stem: Path,
    args: argparse.Namespace,
    metrics: Optional[Mapping[str, float]] = None,
) -> None:
    rows = (
        ("u", "U_0", "rel_l2_U_0", "rel. L2"),
        ("v", "U_1", "rel_l2_U_1", "rel. L2"),
        ("div", "Divergence", "div_excess_rel", "div. excess rel."),
        ("ppe", "Pressure-Poisson residual", "ppe_excess_rel", "PPE excess rel."),
        ("omega", "Vorticity", "omega_rel_l2", "rel. L2"),
    )
    x, y = np.asarray(grid["x_unique"]), np.asarray(grid["y_unique"])
    fig, axes = plt.subplots(len(rows), 3, figsize=(14.5, 8.2), constrained_layout=True, sharex=True, sharey=True)
    for row, (key, label, metric_key, metric_label) in enumerate(rows):
        a, b = gt[key], rec[key]
        delta = b - a
        vmin, vmax = _field_color_limits(a, b)
        dmax = max(float(np.nanpercentile(np.abs(delta), 99)), EPS)
        ims = (
            axes[row, 0].pcolormesh(x, y, a, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax),
            axes[row, 1].pcolormesh(x, y, b, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax),
            axes[row, 2].pcolormesh(x, y, delta, shading="auto", cmap="coolwarm", vmin=-dmax, vmax=dmax),
        )
        fallback_rel = float(np.linalg.norm(delta.reshape(-1)) / (np.linalg.norm(a.reshape(-1)) + EPS))
        metric_value = float(metrics.get(metric_key, fallback_rel)) if metrics is not None else fallback_rel
        _add_error_metric_box(axes[row, 2], metric_label, metric_value, rms(delta))
        axes[row, 0].set_ylabel(label)
        for col, im in enumerate(ims):
            fig.colorbar(im, ax=axes[row, col], fraction=0.025, pad=0.02)
        for ax in axes[row]:
            ax.set_xlim(float(x.min()), float(x.max()))
            ax.set_ylim(float(y.min()), float(y.max()))
            ax.set_aspect("equal", adjustable="box")
    for col, title in enumerate(("Ground truth", "Reconstruction", "Reconstruction − GT")):
        axes[0, col].set_title(title)
    for ax in axes[-1]:
        ax.set_xlabel("x (physical)")
    save_figure(fig, stem, args.plot_formats, args.dpi)


def _worst_pair(matrix: np.ndarray) -> Optional[Tuple[int, int]]:
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return None
    upper = np.array(matrix, dtype=float, copy=True)
    upper[np.tril_indices_from(upper)] = np.nan
    if not np.isfinite(upper).any():
        return None
    index = int(np.nanargmax(upper))
    return tuple(int(x) for x in np.unravel_index(index, matrix.shape))


def plot_distribution_summary(
    result: Mapping[str, Any],
    x_ref: torch.Tensor,
    x_gen: torch.Tensor,
    field_names: Sequence[str],
    stem: Path,
    args: argparse.Namespace,
) -> None:
    per_channel = result["per_channel_w2"].detach().cpu().numpy()
    fig = plt.figure(figsize=(10.8, 7.6), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 2, height_ratios=(0.82, 1.18), width_ratios=(1.0, 1.0))
    ax_marginal = fig.add_subplot(grid_spec[0, 0])
    ax_pairwise = fig.add_subplot(grid_spec[0, 1])
    ax_ref = fig.add_subplot(grid_spec[1, 0])
    ax_gen = fig.add_subplot(grid_spec[1, 1], sharex=ax_ref, sharey=ax_ref)

    bars = ax_marginal.bar(np.arange(len(per_channel)), per_channel, color="#4C78A8", width=0.72)
    ax_marginal.set_xticks(np.arange(len(per_channel)), field_names, rotation=25)
    ax_marginal.set_ylabel(r"1D $W_2^2$")
    ax_marginal.set_title("Marginal discrepancy")
    for bar, value in zip(bars, per_channel):
        ax_marginal.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1e}",
            ha="center",
            va="bottom",
            fontsize=6.5,
        )

    pair = result.get("pairwise_2d_swd")
    if pair is None:
        for ax in (ax_pairwise, ax_ref, ax_gen):
            ax.axis("off")
        ax_pairwise.text(0.5, 0.5, "Pairwise coherence disabled", ha="center", va="center")
    else:
        matrix = pair.detach().cpu().numpy()
        shown = matrix.copy()
        shown[np.tril_indices_from(shown)] = np.nan
        im = ax_pairwise.imshow(shown, cmap="magma", interpolation="nearest")
        ax_pairwise.set_xticks(np.arange(len(field_names)), field_names, rotation=35, ha="right")
        ax_pairwise.set_yticks(np.arange(len(field_names)), field_names)
        ax_pairwise.set_title("Pairwise sliced-Wasserstein")
        finite_pair = matrix[np.triu_indices_from(matrix, k=1)]
        threshold = float(np.nanmedian(finite_pair)) if finite_pair.size else 0.0
        for i in range(len(field_names)):
            for j in range(i + 1, len(field_names)):
                ax_pairwise.text(
                    j, i, f"{matrix[i, j]:.1e}", ha="center", va="center", fontsize=6.2,
                    color="white" if matrix[i, j] <= threshold else "black",
                )
        fig.colorbar(im, ax=ax_pairwise, fraction=0.046, pad=0.04, label="2D SWD")
        worst = _worst_pair(matrix)
        if worst is not None:
            i, j = worst
            ref = x_ref.detach().cpu().numpy()
            gen = x_gen.detach().cpu().numpy()
            x_min = float(min(ref[:, i].min(), gen[:, i].min()))
            x_max = float(max(ref[:, i].max(), gen[:, i].max()))
            y_min = float(min(ref[:, j].min(), gen[:, j].min()))
            y_max = float(max(ref[:, j].max(), gen[:, j].max()))
            if x_min == x_max:
                x_max = x_min + EPS
            if y_min == y_max:
                y_max = y_min + EPS
            x_edges = np.linspace(x_min, x_max, 91)
            y_edges = np.linspace(y_min, y_max, 91)
            hist_ref, _, _ = np.histogram2d(ref[:, i], ref[:, j], bins=(x_edges, y_edges))
            hist_gen, _, _ = np.histogram2d(gen[:, i], gen[:, j], bins=(x_edges, y_edges))
            max_count = max(float(hist_ref.max()), float(hist_gen.max()), 1.0)
            density_cmap = plt.get_cmap("viridis").copy()
            density_cmap.set_bad("white")
            norm = LogNorm(vmin=1.0, vmax=max_count)
            mesh_ref = ax_ref.pcolormesh(
                x_edges, y_edges, np.ma.masked_less(hist_ref.T, 1.0),
                shading="auto", cmap=density_cmap, norm=norm,
            )
            ax_gen.pcolormesh(
                x_edges, y_edges, np.ma.masked_less(hist_gen.T, 1.0),
                shading="auto", cmap=density_cmap, norm=norm,
            )
            for ax, title in ((ax_ref, "Ground truth"), (ax_gen, "Reconstruction")):
                ax.set_xlabel(field_names[i])
                ax.set_ylabel(field_names[j])
                ax.set_title(f"{title}: {field_names[i]}–{field_names[j]}")
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.grid(False)
            fig.colorbar(
                mesh_ref, ax=(ax_ref, ax_gen), fraction=0.035, pad=0.025,
                label="Full-grid count per bin (log scale)",
            )
        else:
            ax_ref.axis("off")
            ax_gen.axis("off")
    save_figure(fig, stem, args.plot_formats, args.dpi)


def plot_snapshot_bars(metrics: Mapping[str, float], stem: Path, args: argparse.Namespace) -> None:
    keys = ("rel_l2_all", "dist_self", "dist_mutual", "spec_cross_coherence_mae", "div_excess_rel", "ppe_excess_rel")
    labels = ("Rel. L2", "Dist. self", "Dist. mutual", "Cross-spectrum\n(omitted)", "Div. excess", "PPE excess")
    values = np.asarray([float(metrics.get(key, np.nan)) for key in keys])
    finite = np.isfinite(values)
    display = np.where(finite, values, 0.0)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bars = ax.bar(np.arange(len(keys)), display, color=np.where(finite, "#4C78A8", "#BAB0AC"))
    ax.set_xticks(np.arange(len(keys)), labels, rotation=20, ha="right")
    ax.set_ylabel("Metric value (lower is better)")
    ax.set_title("Snapshot diagnostic summary")
    for bar, ok in zip(bars, finite):
        if not ok:
            bar.set_hatch("//")
            ax.text(bar.get_x() + bar.get_width() / 2, 0, "N/A", ha="center", va="bottom", fontsize=7)
    save_figure(fig, stem, args.plot_formats, args.dpi)


def fit_rho_global_gt(
    dataset: TurbulentCombustionH5Dataset,
    indices: Sequence[int],
    grid: Mapping[str, Any],
    fields: Mapping[str, int],
) -> float:
    numerator = 0.0
    denominator = 0.0
    for index in indices:
        sample = dataset[int(index)]
        truth_norm = sample["fields"].unsqueeze(0)
        truth_phys = denormalize(truth_norm, dataset)[0].cpu().numpy()
        arrays = flow_residual_arrays(truth_phys, grid, fields, rho=1.0, nu=0.0, momentum=False)
        numerator += float(np.sum(arrays["lap_p"] * arrays["A"], dtype=np.float64))
        denominator += float(np.sum(arrays["A"] ** 2, dtype=np.float64))
    return -numerator / (denominator + EPS)


def tensor_result_np(result: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    keys = (
        "per_channel_w2",
        "theta",
        "per_direction_w2",
        "top_indices",
        "top_values",
        "axis_direction_mask",
        "joint_score_mask",
        "pairwise_2d_swd",
    )
    return {
        key: value.detach().cpu().numpy()
        for key in keys
        if (value := result.get(key)) is not None and torch.is_tensor(value)
    }


def save_snapshot_npz(
    path: Path,
    truth_norm: torch.Tensor,
    recon_norm: torch.Tensor,
    truth_phys: torch.Tensor,
    recon_phys: torch.Tensor,
    result: Mapping[str, Any],
    pde_gt: Optional[Mapping[str, np.ndarray]],
    pde_rec: Optional[Mapping[str, np.ndarray]],
) -> None:
    payload: Dict[str, Any] = {
        "truth_normalized": truth_norm[0].detach().cpu().numpy(),
        "reconstruction_normalized": recon_norm[0].detach().cpu().numpy(),
        "truth_physical": truth_phys[0].detach().cpu().numpy(),
        "reconstruction_physical": recon_phys[0].detach().cpu().numpy(),
        **tensor_result_np(result),
    }
    if pde_gt is not None and pde_rec is not None:
        for key in ("div", "ppe", "omega", "mom_x", "mom_y"):
            if key in pde_gt:
                payload[f"{key}_gt"] = pde_gt[key]
                payload[f"{key}_rec"] = pde_rec[key]
    np.savez_compressed(path, **payload)


def scalar_metric_keys(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    excluded = {"model", "checkpoint", "split", "snapshot_index", "time_index", "physical_time", "joint_method"}
    keys: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if key in excluded or isinstance(value, (str, bool, list, tuple, dict)):
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            keys.add(key)
    return sorted(keys)


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key in scalar_metric_keys(rows):
        values = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary[key] = {"count": 0, "mean": None, "std": None, "median": None, "q25": None, "q75": None}
            continue
        q25, q75 = np.percentile(values, (25, 75))
        summary[key] = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "q25": float(q25),
            "q75": float(q75),
        }
    return summary


def write_metrics_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    preferred = ["model", "checkpoint", "split", "snapshot_index", "time_index", "physical_time"]
    all_keys = set().union(*(row.keys() for row in rows))
    keys = preferred + sorted(all_keys.difference(preferred))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


CORRELATION_PAIRS = (
    ("rel_l2_all", "dist_mutual"),
    ("rel_l2_all", "div_excess_rel"),
    ("rel_l2_all", "ppe_excess_rel"),
    ("dist_mutual", "div_excess_rel"),
    ("dist_mutual", "ppe_excess_rel"),
    ("spec_cross_coherence_mae", "div_excess_rel"),
    ("spec_cross_coherence_mae", "ppe_excess_rel"),
)


def compute_correlation(rows: Sequence[Mapping[str, Any]], x_key: str, y_key: str) -> Dict[str, Any]:
    x = np.asarray([float(row.get(x_key, np.nan)) for row in rows], dtype=float)
    y = np.asarray([float(row.get(y_key, np.nan)) for row in rows], dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    out: Dict[str, Any] = {
        "x": x_key,
        "y": y_key,
        "n": int(x.size),
        "pearson_r": None,
        "pearson_p": None,
        "spearman_rho": None,
        "spearman_p": None,
    }
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        out["note"] = "Insufficient non-constant finite samples."
        return out
    try:
        from scipy import stats

        pearson = stats.pearsonr(x, y)
        spearman = stats.spearmanr(x, y)
        out.update(
            {
                "pearson_r": float(getattr(pearson, "statistic", pearson[0])),
                "pearson_p": float(getattr(pearson, "pvalue", pearson[1])),
                "spearman_rho": float(getattr(spearman, "statistic", spearman[0])),
                "spearman_p": float(getattr(spearman, "pvalue", spearman[1])),
            }
        )
    except Exception:
        out["pearson_r"] = float(np.corrcoef(x, y)[0, 1])
        out["note"] = "scipy unavailable; Pearson computed with NumPy and Spearman omitted."
    return out


def correlation_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    labels = sorted({str(row["model"]) for row in rows})
    return {
        "all_models": {
            f"{x}_vs_{y}": compute_correlation(rows, x, y) for x, y in CORRELATION_PAIRS
        },
        "by_model": {
            label: {
                f"{x}_vs_{y}": compute_correlation([r for r in rows if r["model"] == label], x, y)
                for x, y in CORRELATION_PAIRS
            }
            for label in labels
        },
        "cross_spectral_note": "Cross-spectral metrics were explicitly omitted in this revision; related correlations are null.",
    }


MODEL_COLORS = ("#4C78A8", "#E45756", "#72B7B2", "#F2CF5B", "#B279A2")


def plot_scatter(
    rows: Sequence[Mapping[str, Any]],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    stem: Path,
    args: argparse.Namespace,
) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    plotted = False
    for color, label in zip(MODEL_COLORS, sorted({str(r["model"]) for r in rows})):
        selected = [r for r in rows if r["model"] == label]
        x = np.asarray([float(r.get(x_key, np.nan)) for r in selected])
        y = np.asarray([float(r.get(y_key, np.nan)) for r in selected])
        keep = np.isfinite(x) & np.isfinite(y)
        if keep.any():
            ax.scatter(x[keep], y[keep], s=24, alpha=0.7, color=color, label=label)
            plotted = True
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No finite paired values", transform=ax.transAxes, ha="center", va="center")
    save_figure(fig, stem, args.plot_formats, args.dpi)


def plot_matched_l2_bins(rows: Sequence[Mapping[str, Any]], stem: Path, args: argparse.Namespace) -> None:
    l2 = np.asarray([float(r.get("rel_l2_all", np.nan)) for r in rows])
    finite = l2[np.isfinite(l2)]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    if finite.size < 4:
        ax.text(0.5, 0.5, "Too few finite samples for matched L2 bins", transform=ax.transAxes, ha="center")
    else:
        edges = np.unique(np.quantile(finite, np.linspace(0, 1, 5)))
        centers = 0.5 * (edges[:-1] + edges[1:])
        for color, label in zip(MODEL_COLORS, sorted({str(r["model"]) for r in rows})):
            selected = [r for r in rows if r["model"] == label]
            medians = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                vals = [
                    float(r.get("ppe_excess_rel", np.nan))
                    for r in selected
                    if lo <= float(r.get("rel_l2_all", np.nan)) <= hi
                ]
                vals = [v for v in vals if math.isfinite(v)]
                medians.append(float(np.median(vals)) if vals else np.nan)
            ax.plot(centers, medians, marker="o", color=color, label=label)
        ax.set_xlabel("Matched relative-L2 bin center")
        ax.set_ylabel("Median PPE excess")
        ax.legend()
    save_figure(fig, stem, args.plot_formats, args.dpi)


def plot_violin_summary(rows: Sequence[Mapping[str, Any]], stem: Path, args: argparse.Namespace) -> None:
    metrics = ("rel_l2_all", "dist_mutual", "div_excess_rel", "ppe_excess_rel")
    labels = ("Rel. L2", "Dist. mutual", "Div. excess", "PPE excess")
    models = sorted({str(r["model"]) for r in rows})
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    width = 0.72 / max(len(models), 1)
    legend_handles = []
    for model_i, (color, model) in enumerate(zip(MODEL_COLORS, models)):
        for metric_i, key in enumerate(metrics):
            values = np.asarray(
                [float(r.get(key, np.nan)) for r in rows if r["model"] == model], dtype=float
            )
            values = values[np.isfinite(values) & (values > 0)]
            if not values.size:
                continue
            position = metric_i - 0.36 + width / 2 + model_i * width
            violin = ax.violinplot(np.log10(values), positions=[position], widths=width * 0.9, showmedians=True)
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_alpha(0.65)
            for part in ("cmedians", "cbars", "cmins", "cmaxes"):
                violin[part].set_color(color)
        legend_handles.append(Line2D([0], [0], color=color, lw=6, alpha=0.65, label=model))
    ax.set_xticks(np.arange(len(metrics)), labels)
    ax.set_ylabel(r"$\log_{10}$(metric)")
    if legend_handles:
        ax.legend(handles=legend_handles)
    save_figure(fig, stem, args.plot_formats, args.dpi)


def save_aggregate_plots(rows: Sequence[Mapping[str, Any]], root: Path, args: argparse.Namespace) -> None:
    specifications = (
        ("rel_l2_all", "div_excess_rel", "Relative L2", "Divergence excess", "scatter_l2_vs_divergence"),
        ("rel_l2_all", "ppe_excess_rel", "Relative L2", "PPE excess", "scatter_l2_vs_ppe"),
        ("dist_mutual", "div_excess_rel", "Distribution mutual", "Divergence excess", "scatter_dist_mutual_vs_divergence"),
        ("dist_mutual", "ppe_excess_rel", "Distribution mutual", "PPE excess", "scatter_dist_mutual_vs_ppe"),
        ("spec_cross_coherence_mae", "ppe_excess_rel", "Cross-spectral coherence MAE", "PPE excess", "scatter_spec_cross_vs_ppe"),
    )
    for x, y, xlabel, ylabel, name in specifications:
        plot_scatter(rows, x, y, xlabel, ylabel, root / name, args)
    plot_matched_l2_bins(rows, root / "matched_l2_bins", args)
    plot_violin_summary(rows, root / "metric_violin_summary", args)


def _grid_context(dataset: TurbulentCombustionH5Dataset, args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    try:
        return infer_regular_grid(dataset.coords_raw.cpu().numpy(), tolerance=args.grid_tolerance)
    except Exception as exc:
        message = f"Physical grid inference failed; PDE diagnostics will be NaN: {exc}"
        if args.require_grid:
            raise ValueError(message) from exc
        warn(message)
        return None


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.compute_momentum_proxy and args.rho_mode == "unit" and abs(float(args.rho)) <= EPS:
        raise ValueError("--compute-momentum-proxy requires nonzero --rho because pressure gradients use 1/rho.")
    apply_plot_style()
    set_seed(args.seed)
    data_path = resolve_and_validate_data(args.data)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint_arg = _checkpoint_value(args)
    save_root = _resolve_relative(args.save_root, (DEMO_DIR, Path.cwd()))
    eval_dir = save_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir.mkdir(parents=True, exist_ok=True)
    coherence_cfg = make_coherence_config(args)

    all_rows: List[Dict[str, Any]] = []
    run_records: List[Dict[str, Any]] = []
    selected_indices: Optional[List[int]] = None
    reference_grid: Optional[Dict[str, Any]] = None
    rho_used: Optional[float] = None

    print(f"[*] Device: {device}")
    print(f"[*] Dataset: {data_path}")
    print(f"[*] Checkpoint: {checkpoint_arg}")

    for checkpoint_i, checkpoint_arg in enumerate((checkpoint_arg,)):
        provisional_path, provisional_run = legacy_eval.choose_checkpoint(checkpoint_arg)
        label = provisional_run.name
        print(f"\n[*] Loading {label}: {provisional_path}")
        context = load_model_context(checkpoint_arg, args, device, data_path, label)
        dataset: TurbulentCombustionH5Dataset = context["dataset"]
        fields: Dict[str, int] = context["fields"]
        if selected_indices is None:
            selected_indices = select_snapshot_indices(
                len(dataset), args.snapshot_indices, args.stride, args.max_snapshots, args.seed
            )
        elif any(index >= len(dataset) for index in selected_indices):
            raise ValueError(f"Dataset length differs across model contexts; {label} has {len(dataset)} samples.")

        grid = _grid_context(dataset, args)
        if checkpoint_i == 0:
            reference_grid = grid
            if args.rho_mode == "fit_global_gt" and grid is not None:
                rho_used = fit_rho_global_gt(dataset, selected_indices, grid, fields)
                print(f"[*] Frozen globally fitted rho: {rho_used:.8g}")
                if args.compute_momentum_proxy and abs(float(rho_used)) <= EPS:
                    raise ValueError(
                        "Globally fitted rho is numerically zero; the optional steady momentum proxy requires 1/rho."
                    )
            elif args.rho_mode == "fit_global_gt":
                rho_used = float("nan")
            else:
                rho_used = float(args.rho)
        elif (reference_grid is None) != (grid is None):
            warn(f"Grid availability differs for model {label}; continuing with per-model grid status.")

        run_record = {
            "model": label,
            "checkpoint": context["checkpoint_path"],
            "run_dir": context["run_dir"],
            "family": context["family"],
            "field_names": tuple(dataset.field_names),
            "field_indices": fields,
            "cond_fields": context["cond_fields"],
            "conditioned_field_names": [dataset.field_names[i] for i in context["cond_fields"]],
            "n_obs_list": context["n_obs_list"],
            "n_steps": context["n_steps"],
            "ode_solver": context["ode_solver"],
            "grid_available": grid is not None,
            "grid_shape": [grid["ny"], grid["nx"]] if grid is not None else None,
        }
        run_records.append(run_record)
        model_dir = eval_dir / label
        model_dir.mkdir(parents=True, exist_ok=True)

        for ordinal, snapshot_index in enumerate(selected_indices, start=1):
            print(f"[{label}] snapshot {snapshot_index} ({ordinal}/{len(selected_indices)})")
            set_seed(args.seed + int(snapshot_index))
            reco_kwargs = {
                "model": context["model"],
                "dataset": dataset,
                "device": device,
                "snapshot_index": int(snapshot_index),
                "cond_fields": context["cond_fields"],
                "n_obs_list": context["n_obs_list"],
                "n_steps": context["n_steps"],
                "ode_solver": context["ode_solver"],
            }
            if context["family"] == "s3gm":
                rec = context["reco_fn"](**reco_kwargs)
            else:
                with torch.no_grad():
                    rec = context["reco_fn"](**reco_kwargs)

            truth_norm = rec["truth"]
            recon_norm = rec["recon"]
            truth_phys = denormalize(truth_norm, dataset)
            recon_phys = denormalize(recon_norm, dataset)
            metrics = reconstruction_metrics(
                truth_norm, recon_norm, truth_phys, recon_phys, dataset.field_names
            )
            _rename_required_field_metric_keys(metrics, dataset.field_names, fields)
            obs_error = temperature_observation_error(rec, recon_norm, fields["T"])
            metrics["observation_consistency_error"] = obs_error
            metrics["obs_consistency_T"] = obs_error

            dist_scalars, dist_result, x_ref, x_gen = compute_distribution_metrics(
                truth_norm, recon_norm, truth_phys, recon_phys, args, coherence_cfg
            )
            metrics.update(dist_scalars)
            metrics["spec_cross_coherence_mae"] = float("nan")

            gt_pde: Optional[Dict[str, np.ndarray]] = None
            rec_pde: Optional[Dict[str, np.ndarray]] = None
            if grid is not None and rho_used is not None and math.isfinite(rho_used):
                gt_np = truth_phys[0].detach().cpu().numpy()
                rec_np = recon_phys[0].detach().cpu().numpy()
                gt_pde = flow_residual_arrays(
                    gt_np, grid, fields, rho_used, args.nu, args.compute_momentum_proxy
                )
                rec_pde = flow_residual_arrays(
                    rec_np, grid, fields, rho_used, args.nu, args.compute_momentum_proxy
                )
                metrics.update(compute_pde_metrics(gt_pde, rec_pde, rho_used, args.compute_momentum_proxy))
            else:
                metrics.update(nan_pde_metrics())

            sample = dataset[int(snapshot_index)]
            row: Dict[str, Any] = {
                "model": label,
                "checkpoint": str(context["checkpoint_path"]),
                "split": args.split,
                "snapshot_index": int(snapshot_index),
                "time_index": int(sample["time_index"].item()),
                "physical_time": float(sample["physical_time"].item()),
                "rho": float(rho_used) if rho_used is not None else float("nan"),
                **metrics,
            }
            all_rows.append(row)
            snap_dir = model_dir / f"snapshot_{snapshot_index:05d}"
            snap_dir.mkdir(parents=True, exist_ok=True)
            save_json(
                snap_dir / "metrics.json",
                {
                    **row,
                    "per_channel_w2": dist_result["per_channel_w2"],
                    "per_direction_w2": dist_result["per_direction_w2"],
                    "theta": dist_result["theta"],
                    "top_indices": dist_result.get("top_indices"),
                    "pairwise_2d_swd": dist_result.get("pairwise_2d_swd"),
                },
            )
            if args.save_npz:
                save_snapshot_npz(
                    snap_dir / "snapshot_diagnostics.npz",
                    truth_norm,
                    recon_norm,
                    truth_phys,
                    recon_phys,
                    dist_result,
                    gt_pde,
                    rec_pde,
                )
            if args.save_snapshot_plots:
                raw_coords = dataset.coords_raw.cpu().numpy()
                sensors_t = _sensor_xy(rec, raw_coords, fields["T"])
                if grid is not None:
                    plot_fields(
                        truth_phys[0].detach().cpu().numpy(),
                        recon_phys[0].detach().cpu().numpy(),
                        grid,
                        fields,
                        sensors_t,
                        snap_dir / "fields_truth_recon_error",
                        args,
                    )
                    if gt_pde is not None and rec_pde is not None:
                        plot_pde(
                            gt_pde,
                            rec_pde,
                            grid,
                            snap_dir / "velocity_pressure_pde_residuals",
                            args,
                            metrics=metrics,
                        )
                else:
                    plot_fields_unstructured(
                        truth_phys[0].detach().cpu().numpy(),
                        recon_phys[0].detach().cpu().numpy(),
                        raw_coords,
                        fields,
                        sensors_t,
                        snap_dir / "fields_truth_recon_error",
                        args,
                    )
                plot_distribution_summary(
                    dist_result,
                    x_ref,
                    x_gen,
                    dataset.field_names,
                    snap_dir / "global_distribution_summary",
                    args,
                )
                plot_snapshot_bars(metrics, snap_dir / "snapshot_metric_bars", args)

            del rec, truth_norm, recon_norm, truth_phys, recon_phys, dist_result
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del context["model"]
        del context
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if selected_indices is None:
        raise RuntimeError("No model contexts were evaluated.")
    write_metrics_csv(eval_dir / f"metrics_{args.split}.csv", all_rows)
    per_model = {
        label: summarize_rows([row for row in all_rows if row["model"] == label])
        for label in sorted({str(row["model"]) for row in all_rows})
    }
    summary = {
        "split": args.split,
        "num_rows": len(all_rows),
        "num_snapshots_per_model": len(selected_indices),
        "overall": summarize_rows(all_rows),
        "by_model": per_model,
        "cross_spectral_status": "omitted_in_this_revision",
        "pde_norm_convention": "RMS over the complete inferred physical grid",
    }
    save_json(eval_dir / f"summary_{args.split}.json", summary)
    save_json(eval_dir / "correlation_report.json", correlation_report(all_rows))
    save_json(
        eval_dir / "run_context.json",
        {
            "created_at": datetime.now().isoformat(),
            "dataset": data_path,
            "split": args.split,
            "selected_snapshot_indices": selected_indices,
            "device": str(device),
            "rho_mode": args.rho_mode,
            "rho_requested": args.rho,
            "rho_hat": rho_used if args.rho_mode == "fit_global_gt" else None,
            "rho_used": rho_used,
            "nu": args.nu,
            "compute_momentum_proxy": args.compute_momentum_proxy,
            "coherence_space": args.coherence_space,
            "coherence_config": asdict(coherence_cfg),
            "grid_available": reference_grid is not None,
            "grid_shape": [reference_grid["ny"], reference_grid["nx"]] if reference_grid is not None else None,
            "field_name_priority": "HDF5 fields.selected_fields, HDF5 selected_fields, CLI/config names, helper defaults",
            "legacy_default_conditioning_index": 3,
            "runs": run_records,
            "cross_spectral_status": "omitted_in_this_revision",
        },
    )
    save_aggregate_plots(all_rows, eval_dir, args)
    print(f"\n[*] Physical-coherence evaluation saved to: {eval_dir}")


if __name__ == "__main__":
    main()


"""
Example to run it:
python src/evaluate_full_dataset.py \
  --Demo-Num 15 \
  --split test \
  --checkpoint last \
  --n-steps-generation 2

If we want to evaluate a baseline:
python src/evaluate_full_dataset.py \
  --Demo-Num 0 \
  --baseline-model latent_fm \
  --split test \
  --checkpoint last \
  --n-steps-generation 2

"""

import argparse
import contextlib
import csv
import json
import os
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phycoflow_mplconfig")
os.environ.setdefault("KEOPS_CACHE_FOLDER", "/tmp/phycoflow_keops")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["KEOPS_CACHE_FOLDER"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
import yaml

from helpers import TurbulentCombustionH5Dataset, reconstruct_snapshot
from evaluate_ffm import (
    _align_per_field_values,
    _as_int_list,
    _build_model,
    _extract_timestamp,
    _find_latest_yaml,
    _infer_structured_grid_from_coords,
    _normalize_eval_config,
    _reshape_flat_field_to_grid,
    _spectral_metrics,
    _ssim2d,
)


FIELD_COLORS = {
    "CH4": "#0072B2",
    "CO": "#D55E00",
    "T": "#009E73",
    "U_1": "#CC79A7",
    "p": "#E69F00",
}
FALLBACK_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

METRIC_SPECS = {
    "l2": {
        "label": "Relative $L_2$ error",
        "summary_label": "Rel. L2",
        "better": "lower",
        "scale": "log",
    },
    "ssim": {
        "label": "SSIM",
        "summary_label": "SSIM",
        "better": "higher",
        "scale": "linear",
    },
    "spectral_lsd": {
        "label": "Spectral LSD",
        "summary_label": "Spectral LSD",
        "better": "lower",
        "scale": "log",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Full-dataset evaluator for point-cloud FFM models and generative baselines."
    )
    parser.add_argument(
        "--Demo-Num",
        dest="Demo_Num",
        type=int,
        required=True,
        help="Demo ID used to resolve either FFM or baseline run directories.",
    )
    parser.add_argument("--demo-root", type=str, default=".")
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=None,
        choices=["s3gm", "latent_fm", "sit"],
        help=(
            "Evaluate a generative baseline trained by train_Gen_Baseline.py. "
            "If omitted, evaluate the point-cloud FFM model."
        ),
    )
    parser.add_argument(
        "--baseline-config",
        type=str,
        default="Save_config/config_baseline_Gen.yaml",
        help="Baseline YAML config used to resolve the run when --baseline-model is set.",
    )
    parser.add_argument(
        "--baseline-run-dir",
        type=str,
        default=None,
        help="Specific baseline run directory. Optional; otherwise the latest matching run is used.",
    )
    parser.add_argument(
        "--baseline-checkpoint-path",
        type=str,
        default=None,
        help="Specific baseline checkpoint path. Optional; overrides --baseline-run-dir.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
        help=(
            "Dataset split to evaluate. 'all' evaluates train and test once, "
            "then reports their combined statistics."
        ),
    )
    parser.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"])
    parser.add_argument(
        "--n-steps-generation",
        type=int,
        default=None,
        help="Override generation steps. Defaults to YAML n_steps_generation.",
    )
    parser.add_argument(
        "--cond-fields",
        type=int,
        nargs="+",
        default=None,
        help="Conditioned fields. Defaults to YAML vis_cond_fields or cond_fields.",
    )
    parser.add_argument(
        "--n-obs-list",
        type=int,
        nargs="+",
        default=None,
        help="Exact sensor counts per conditioned field. Defaults to YAML vis_n_obs_list.",
    )
    parser.add_argument(
        "--max-snapshots",
        type=int,
        default=None,
        help="Optional cap per split for quick checks; by default evaluates the full selected split.",
    )
    parser.add_argument(
        "--snapshot-stride",
        type=int,
        default=1,
        help="Evaluate every k-th snapshot within each selected split.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index within each selected split.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=1234,
        help="Seed controlling the one sparse-observation/sample draw per snapshot.",
    )
    parser.add_argument(
        "--ode-solver",
        type=str,
        default=None,
        choices=["euler", "heun"],
        help="ODE solver for models that expose this option. Defaults to YAML ode_solver.",
    )
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every this many reconstructed snapshots.",
    )
    return parser.parse_args()


def _resolve_demo_path(demo_root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    if path.exists():
        return str(path.resolve())
    return str(demo_root / path)


def _resolve_demo_path_obj(demo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return demo_root / path


def _import_model_baseline():
    try:
        import model_baseline as mb
    except ImportError:
        from . import model_baseline as mb
    return mb


def _load_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except (pickle.UnpicklingError, RuntimeError):
        print(
            "[Warning: !] Restricted torch.load failed; retrying with "
            "weights_only=False for a trusted local checkpoint."
        )
        return torch.load(path, map_location="cpu", weights_only=False)


def _load_model_and_config(args: argparse.Namespace):
    demo_root = Path(args.demo_root).resolve()
    cfg_dir = demo_root / "Save_config" / "pointcloud_ffm"
    yaml_path = _find_latest_yaml(cfg_dir, args.Demo_Num)

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _normalize_eval_config(cfg)

    train_timestamp = _extract_timestamp(yaml_path)
    if train_timestamp is None:
        raise RuntimeError(f"Could not parse timestamp from config filename: {yaml_path.name}")

    save_dir_cfg = Path(cfg.get("save_dir", "Save_TrainedModel/ffm_tc_pointcloud"))
    save_root = save_dir_cfg.parent if save_dir_cfg.is_absolute() else demo_root / save_dir_cfg.parent
    model_root = save_root / (
        f"{save_dir_cfg.name}_DemoN{args.Demo_Num}_{train_timestamp}"
    )

    if not model_root.exists():
        # RELOAD training creates a new timestamped config backup, but resumes in
        # the original run directory.  In that case the newest config timestamp
        # does not have a same-timestamp model directory.  Resolve the newest
        # actual run containing the requested checkpoint instead.
        run_prefix = f"{save_dir_cfg.name}_DemoN{args.Demo_Num}_"
        candidates = sorted(
            path
            for path in save_root.glob(f"{run_prefix}*")
            if path.is_dir() and (path / f"{args.checkpoint}.pt").is_file()
        )
        if not candidates:
            raise FileNotFoundError(
                f"Matching model directory not found: {model_root}. "
                f"No fallback run matching {save_root / (run_prefix + '*')} "
                f"contains {args.checkpoint}.pt."
            )

        model_root = candidates[-1]
        print(
            "[Warning: !] The latest config backup has no same-timestamp model "
            f"directory; using resumed run: {model_root}"
        )

    # args.json records the effective training arguments after inherited
    # pretrained settings are applied.  run_config.yaml is only the input
    # config and can therefore disagree with the checkpoint architecture.
    args_path = model_root / "args.json"
    run_config_path = model_root / "run_config.yaml"
    if args_path.is_file():
        yaml_path = args_path
        with open(args_path, "r") as f:
            cfg = _normalize_eval_config(json.load(f) or {})
    elif run_config_path.is_file():
        yaml_path = run_config_path
        with open(run_config_path, "r") as f:
            cfg = _normalize_eval_config(yaml.safe_load(f) or {})

    run_timestamp = _extract_timestamp(model_root)
    if run_timestamp is not None:
        train_timestamp = run_timestamp

    ckpt_path = model_root / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    stats_path = str(model_root / "dataset_stats.pt")
    data_path = _resolve_demo_path(demo_root, cfg.get("data", "Dataset/Merged_CH4COTU1P.h5"))

    build_dataset = lambda split: TurbulentCombustionH5Dataset(
        data_path,
        split=split,
        train_ratio=cfg.get("train_ratio", 0.9),
        seed=cfg.get("seed", 42),
        time_stride=cfg.get("time_stride", 1),
        field_names=cfg.get("FIELD_NAMES", cfg.get("field_names", None)),
        stats_path=stats_path,
    )

    reference_dataset = build_dataset("train")
    model = _build_model(cfg, reference_dataset)

    ckpt = _load_checkpoint(ckpt_path)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = dict(state_dict)
        state_dict.pop("_metadata", None)

    model.load_state_dict(state_dict, strict=True)
    epoch = int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0
    del state_dict
    del ckpt

    return {
        "demo_root": demo_root,
        "cfg": cfg,
        "yaml_path": yaml_path,
        "model_root": model_root,
        "ckpt_path": ckpt_path,
        "train_timestamp": train_timestamp,
        "model": model,
        "epoch": epoch,
        "build_dataset": build_dataset,
        "reference_dataset": reference_dataset,
    }


def _make_baseline_paths_absolute(cfg: dict, demo_root: Path) -> dict:
    cfg = dict(cfg)
    shared = dict(cfg.get("shared", {}))
    paths = dict(shared.get("paths", {}))
    for key in ("data_path", "save_root", "config_backup_root"):
        if key in paths and paths[key] is not None:
            paths[key] = str(_resolve_demo_path_obj(demo_root, paths[key]).resolve())
    shared["paths"] = paths
    cfg["shared"] = shared
    return cfg


def _baseline_stage_for_model(baseline_model: str) -> int:
    if baseline_model == "latent_fm":
        return 2
    return 1


def _validate_baseline_run_config(cfg: dict, *, args: argparse.Namespace, expected_stage: int) -> None:
    if cfg["baseline_model"] != args.baseline_model:
        raise ValueError(
            f"Requested --baseline-model={args.baseline_model}, but run_config.yaml "
            f"contains baseline_model={cfg['baseline_model']}."
        )
    if int(cfg["training_stage"]) != int(expected_stage):
        raise ValueError(
            f"Requested baseline {args.baseline_model} expects stage {expected_stage}, "
            f"but run_config.yaml contains training_stage={cfg['training_stage']}."
        )
    if int(cfg["shared"]["demo_num"]) != int(args.Demo_Num):
        raise ValueError(
            f"Requested --Demo-Num={args.Demo_Num}, but run_config.yaml contains "
            f"shared.demo_num={cfg['shared']['demo_num']}."
        )
    if cfg["baseline_model"] == "latent_fm" and int(cfg["training_stage"]) != 2:
        raise ValueError("Full-dataset evaluation only accepts latent_fm stage-2 runs.")


def _load_baseline_model_and_config(args: argparse.Namespace, device: torch.device):
    mb = _import_model_baseline()
    demo_root = Path(args.demo_root).resolve()
    config_path = _resolve_demo_path_obj(demo_root, args.baseline_config).resolve()
    expected_stage = _baseline_stage_for_model(args.baseline_model)

    # The baseline YAML is used only to locate the save root and establish the
    # desired selector. The selected run's run_config.yaml is authoritative for
    # model construction, mirroring how FFM evaluation restores its saved config.
    cfg = mb.load_yaml(config_path)
    cfg["baseline_model"] = args.baseline_model
    cfg["training_stage"] = expected_stage
    cfg.setdefault("shared", {}).setdefault("paths", {})
    cfg["shared"]["demo_num"] = int(args.Demo_Num)
    cfg = mb.validate_and_normalize_config(_make_baseline_paths_absolute(cfg, demo_root))

    if cfg["baseline_model"] == "latent_fm" and int(cfg["training_stage"]) != 2:
        raise ValueError("Full-dataset evaluation only accepts latent_fm stage-2 checkpoints.")

    if args.baseline_checkpoint_path is not None:
        ckpt_path = _resolve_demo_path_obj(demo_root, args.baseline_checkpoint_path).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found: {ckpt_path}")
        run_dir = ckpt_path.parent
    elif args.baseline_run_dir is not None:
        run_dir = _resolve_demo_path_obj(demo_root, args.baseline_run_dir).resolve()
        if not run_dir.exists():
            raise FileNotFoundError(f"Baseline run directory not found: {run_dir}")
        ckpt_path = run_dir / f"{args.checkpoint}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found: {ckpt_path}")
    else:
        save_root = mb.ensure_absolute(cfg["shared"]["paths"]["save_root"])
        run_dir = mb.find_latest_run_dir(save_root, cfg)
        if run_dir is None:
            raise FileNotFoundError(
                "No matching baseline run directory was found. "
                "Use --baseline-run-dir or --baseline-checkpoint-path if needed."
            )
        ckpt_path = run_dir / f"{args.checkpoint}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Baseline checkpoint not found: {ckpt_path}")

    run_cfg_path = run_dir / "run_config.yaml"
    if not run_cfg_path.exists():
        raise FileNotFoundError(
            f"Selected baseline run does not contain run_config.yaml: {run_cfg_path}"
        )
    cfg = mb.validate_and_normalize_config(
        _make_baseline_paths_absolute(mb.load_yaml(run_cfg_path), demo_root)
    )
    _validate_baseline_run_config(cfg, args=args, expected_stage=expected_stage)

    checkpoint = mb.safe_torch_load(ckpt_path, map_location="cpu")
    if cfg["baseline_model"] == "latent_fm" and int(cfg["training_stage"]) == 2:
        ae_checkpoint = checkpoint.get("ae_checkpoint")
        if ae_checkpoint:
            cfg["latent_fm_params"]["stage2"]["stage1_checkpoint"] = ae_checkpoint

    stats_path = run_dir / "dataset_stats.pt"
    build_dataset = lambda split: mb.build_dataset(cfg, split=split, stats_path=stats_path)
    reference_dataset = build_dataset("train")

    adapter = mb.get_baseline_adapter(cfg["baseline_model"])
    baseline_bundle = adapter.build_for_training(
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        train_set=reference_dataset,
        val_set=reference_dataset,
    )
    adapter.load_checkpoint(baseline_bundle, checkpoint)

    return {
        "model_family": "baseline",
        "baseline_model": cfg["baseline_model"],
        "training_stage": int(cfg["training_stage"]),
        "demo_root": demo_root,
        "cfg": cfg,
        "config_path": config_path,
        "model_root": run_dir,
        "ckpt_path": ckpt_path,
        "train_timestamp": run_dir.name,
        "adapter": adapter,
        "baseline_bundle": baseline_bundle,
        "epoch": int(checkpoint.get("epoch", 0)),
        "build_dataset": build_dataset,
        "reference_dataset": reference_dataset,
        "model_baseline_module": mb,
    }


def _field_color(field_name: str, index: int) -> str:
    return FIELD_COLORS.get(field_name, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def _relative_l2(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    return float(np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8))


def _iter_snapshot_indices(
    dataset: TurbulentCombustionH5Dataset,
    start_index: int,
    stride: int,
    max_snapshots: Optional[int],
) -> List[int]:
    if stride < 1:
        raise ValueError(f"snapshot_stride must be >= 1, got {stride}")
    indices = list(range(max(0, start_index), len(dataset), stride))
    if max_snapshots is not None:
        indices = indices[: max(0, max_snapshots)]
    return indices


def _ffm_reconstruct_phys(
    *,
    model: torch.nn.Module,
    dataset: TurbulentCombustionH5Dataset,
    device: torch.device,
    snapshot_index: int,
    cond_fields: Sequence[int],
    n_obs_list: Sequence[int],
    n_steps_generation: int,
    ode_solver: Optional[str],
) -> Dict[str, np.ndarray]:
    result = reconstruct_snapshot(
        model=model,
        dataset=dataset,
        device=device,
        snapshot_index=snapshot_index,
        cond_fields=cond_fields,
        n_obs_list=n_obs_list,
        n_steps=n_steps_generation,
        ode_solver=ode_solver,
    )
    mean = dataset.mean.to(device).view(1, 1, -1)
    std = dataset.std.to(device).view(1, 1, -1)
    truth_phys = (result["truth"] * std + mean)[0].detach().cpu().numpy()
    recon_phys = (result["recon"] * std + mean)[0].detach().cpu().numpy()
    return {"truth_phys": truth_phys, "recon_phys": recon_phys}


def _baseline_observations(
    mb,
    *,
    dataset,
    device: torch.device,
    snapshot_index: int,
    cond_fields: Sequence[int],
    n_obs_list: Sequence[int],
):
    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    valid_mask = sample.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0).to(device)
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = mb.build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=list(cond_fields),
        n_obs_min=list(n_obs_list),
        n_obs_max=list(n_obs_list),
        valid_mask=valid_mask,
    )
    return {
        "sample": sample,
        "coords": coords,
        "truth": truth,
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_indices": obs_indices,
        "obs_field_ids": obs_field_ids,
    }


def _baseline_to_phys(dataset, device: torch.device, truth: torch.Tensor, recon: torch.Tensor):
    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    truth_phys = truth * std.view(1, 1, -1) + mean.view(1, 1, -1)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    return {
        "truth_phys": truth_phys[0].detach().cpu().numpy(),
        "recon_phys": recon_phys[0].detach().cpu().numpy(),
    }


def _baseline_reconstruct_phys(
    *,
    runtime: dict,
    dataset,
    device: torch.device,
    snapshot_index: int,
    cond_fields: Sequence[int],
    n_obs_list: Sequence[int],
    n_steps_generation: int,
    ode_solver: Optional[str],
) -> Dict[str, np.ndarray]:
    mb = runtime["model_baseline_module"]
    bundle = runtime["baseline_bundle"]
    cfg = runtime["cfg"]
    stage_cfg = mb.resolve_stage_config(cfg)
    baseline_model = cfg["baseline_model"]
    num_x = int(cfg["shared"]["data"]["num_x"])
    num_y = int(cfg["shared"]["data"]["num_y"])

    obs = _baseline_observations(
        mb,
        dataset=dataset,
        device=device,
        snapshot_index=snapshot_index,
        cond_fields=cond_fields,
        n_obs_list=n_obs_list,
    )
    truth = obs["truth"]
    n_fields = int(dataset.num_fields)
    n_pts = int(dataset.num_points)

    if baseline_model == "s3gm":
        sampling_cfg = stage_cfg["sampling"]
        obs_value_grid, obs_mask_grid = mb.build_obs_grid_mask(
            obs["obs_values"],
            obs["obs_mask"],
            obs["obs_field_ids"],
            obs["obs_indices"],
            n_fields,
            n_pts,
            num_y,
            num_x,
            int(bundle.components["H_pad"]),
            int(bundle.components["W_pad"]),
        )
        recon_grid = mb.dps_sample(
            net=bundle.model,
            sde=bundle.components["sde"],
            shape_5d=(1, 1, n_fields, int(bundle.components["H_pad"]), int(bundle.components["W_pad"])),
            obs_value_grid=obs_value_grid,
            obs_mask_grid=obs_mask_grid,
            device=device,
            N_steps=int(n_steps_generation),
            snr=float(sampling_cfg["snr"]),
            n_corrector_steps=int(sampling_cfg["n_corrector_steps"]),
            alpha_obs=float(sampling_cfg["alpha_obs"]),
        )
        recon = mb.grid_to_pointcloud(recon_grid, num_y, num_x)
        return _baseline_to_phys(dataset, device, truth, recon)

    if baseline_model == "latent_fm":
        sampling_cfg = stage_cfg["sampling"]
        obs_value_grid, obs_mask_grid = mb.build_obs_grid_mask(
            obs["obs_values"],
            obs["obs_mask"],
            obs["obs_field_ids"],
            obs["obs_indices"],
            n_fields,
            n_pts,
            num_y,
            num_x,
            num_y,
            num_x,
        )
        ix = (obs["obs_indices"] % num_x).float() / max(num_x - 1, 1)
        iy = (obs["obs_indices"].div(num_x, rounding_mode="floor")).float() / max(num_y - 1, 1)
        cond_inputs = {
            "obs_value_grid": obs_value_grid,
            "obs_mask_grid": obs_mask_grid,
            "obs_coords_2d": torch.stack([ix, iy], dim=-1),
            "obs_values": obs["obs_values"],
            "obs_mask": obs["obs_mask"],
            "obs_field_ids": obs["obs_field_ids"],
        }
        recon_grid = bundle.model.sample(
            cond_inputs,
            n_steps=int(n_steps_generation),
            ode_solver=str(ode_solver or sampling_cfg["ode_solver"]),
        )
        recon = mb.grid_to_pointcloud(recon_grid, num_y, num_x)
        return _baseline_to_phys(dataset, device, truth, recon)

    if baseline_model == "sit":
        sampling_cfg = stage_cfg["sampling"]
        tokenizer = str(bundle.components["tokenizer"])
        cond_mode = str(bundle.components["cond_mode"])
        if tokenizer == "pointnet":
            obs_value_nodes, obs_mask_nodes = mb.scatter_sensors_to_nodes(
                obs["obs_values"],
                obs["obs_mask"],
                obs["obs_field_ids"],
                obs["obs_indices"],
                1,
                n_pts,
                n_fields,
                device,
                truth.dtype,
            )
            recon = mb.sit_conditional_sample(
                net=bundle.model,
                transport=bundle.components["transport"],
                shape=(1, n_pts, n_fields),
                coords=obs["coords"],
                obs_value_nodes=obs_value_nodes,
                obs_mask_nodes=obs_mask_nodes,
                device=device,
                n_steps=int(n_steps_generation),
                sampler_type=str(ode_solver or sampling_cfg["ode_solver"]),
            )
        else:
            obs_value_grid, obs_mask_grid = mb.build_obs_grid_mask(
                obs["obs_values"],
                obs["obs_mask"],
                obs["obs_field_ids"],
                obs["obs_indices"],
                n_fields,
                n_pts,
                num_y,
                num_x,
                int(bundle.components["H_pad"]),
                int(bundle.components["W_pad"]),
            )
            if cond_mode == "interp":
                obs_value_grid = mb.nearest_fill_grid(obs_value_grid, obs_mask_grid)
            recon_grid = mb.sit_conditional_sample(
                net=bundle.model,
                transport=bundle.components["transport"],
                shape=(1, n_fields, int(bundle.components["H_pad"]), int(bundle.components["W_pad"])),
                obs_value_grid=obs_value_grid,
                obs_mask_grid=obs_mask_grid,
                device=device,
                n_steps=int(n_steps_generation),
                sampler_type=str(ode_solver or sampling_cfg["ode_solver"]),
            )
            recon = mb.grid_to_pointcloud(recon_grid, num_y, num_x)
        return _baseline_to_phys(dataset, device, truth, recon)

    raise ValueError(f"Unsupported baseline_model={baseline_model!r}")


def _evaluate_one_split(
    *,
    dataset: TurbulentCombustionH5Dataset,
    split_name: str,
    device: torch.device,
    reconstruct_phys: Callable[[object, int], Dict[str, np.ndarray]],
    num_x: Optional[int],
    num_y: Optional[int],
    eval_seed: int,
    start_index: int,
    snapshot_stride: int,
    max_snapshots: Optional[int],
    progress_every: int,
) -> List[Dict[str, object]]:
    field_names = list(getattr(dataset, "field_names", []))
    snapshot_indices = _iter_snapshot_indices(
        dataset,
        start_index=start_index,
        stride=snapshot_stride,
        max_snapshots=max_snapshots,
    )
    if not snapshot_indices:
        return []

    first_sample = dataset[snapshot_indices[0]]
    grid_info = _infer_structured_grid_from_coords(
        first_sample["coords_raw"].cpu().numpy()[:, :2],
        num_x=num_x,
        num_y=num_y,
    )

    records: List[Dict[str, object]] = []
    total = len(snapshot_indices)

    for count, snapshot_index in enumerate(snapshot_indices, start=1):
        torch.manual_seed(int(eval_seed) + int(dataset.indices[snapshot_index]))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(eval_seed) + int(dataset.indices[snapshot_index]))

        result = reconstruct_phys(dataset, snapshot_index)
        truth_phys = result["truth_phys"]
        recon_phys = result["recon_phys"]
        sample = dataset[snapshot_index]

        time_index = int(sample["time_index"].item())
        physical_time = float(sample["physical_time"].item())

        for c, field_name in enumerate(field_names):
            true_flat = truth_phys[:, c]
            pred_flat = recon_phys[:, c]
            true_grid = _reshape_flat_field_to_grid(true_flat, grid_info)
            pred_grid = _reshape_flat_field_to_grid(pred_flat, grid_info)

            spec_metrics, _ = _spectral_metrics(
                true_grid,
                pred_grid,
                dx=grid_info["dx"],
                dy=grid_info["dy"],
            )

            records.append(
                {
                    "split": split_name,
                    "split_sample_index": int(snapshot_index),
                    "time_index": time_index,
                    "physical_time": physical_time,
                    "field": field_name,
                    "l2": _relative_l2(true_flat, pred_flat),
                    "ssim": _ssim2d(
                        true_grid,
                        pred_grid,
                        data_range=float(true_grid.max() - true_grid.min()),
                    ),
                    "spectral_lsd": float(spec_metrics["spectral_lsd"]),
                }
            )

        del result
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if progress_every > 0 and (count == 1 or count % progress_every == 0 or count == total):
            print(f"[*] {split_name}: reconstructed {count}/{total} snapshots")

    return records


def _metric_values(
    records: Sequence[Dict[str, object]],
    field: str,
    metric: str,
) -> np.ndarray:
    values = [float(r[metric]) for r in records if r["field"] == field]
    return np.asarray(values, dtype=np.float64)


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "q05": float("nan"),
            "q25": float("nan"),
            "median": float("nan"),
            "q75": float("nan"),
            "q95": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "min": float(np.min(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _build_summary(records: Sequence[Dict[str, object]], field_names: Sequence[str]):
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for field in field_names:
        summary[field] = {}
        for metric in METRIC_SPECS:
            summary[field][metric] = _summary_stats(_metric_values(records, field, metric))
    return summary


def _write_records_csv(records: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "split",
        "split_sample_index",
        "time_index",
        "physical_time",
        "field",
        "l2",
        "ssim",
        "spectral_lsd",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row[key] for key in columns})


def _save_figure(fig: plt.Figure, path_without_suffix: Path) -> None:
    path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_without_suffix.with_suffix(".png"), dpi=320, bbox_inches="tight")
    fig.savefig(path_without_suffix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.5,
            "figure.titlesize": 13.0,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _format_mean_label(metric: str, value: float) -> str:
    if not np.isfinite(value):
        return r"$\mu$=n/a"
    if metric == "ssim":
        return rf"$\mu$={value:.3f}"
    return rf"$\mu$={value:.2e}"


def _plot_metric_distributions(
    records: Sequence[Dict[str, object]],
    field_names: Sequence[str],
    split_name: str,
    out_dir: Path,
) -> None:
    _set_paper_style()
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.2), sharex=True)
    x_positions = np.arange(1, len(field_names) + 1)

    for ax, (metric, spec) in zip(axes, METRIC_SPECS.items()):
        data = [_metric_values(records, field, metric) for field in field_names]
        data = [v[np.isfinite(v)] for v in data]
        violin_data = [v if v.size else np.array([np.nan]) for v in data]

        parts = ax.violinplot(
            violin_data,
            positions=x_positions,
            widths=0.74,
            showmeans=False,
            showextrema=False,
            showmedians=False,
        )
        for idx, body in enumerate(parts["bodies"]):
            body.set_facecolor(_field_color(field_names[idx], idx))
            body.set_edgecolor("black")
            body.set_alpha(0.30)
            body.set_linewidth(0.8)

        box = ax.boxplot(
            data,
            positions=x_positions,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#1A1A1A", "linewidth": 1.6},
            boxprops={"linewidth": 1.0},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for idx, patch in enumerate(box["boxes"]):
            patch.set_facecolor(_field_color(field_names[idx], idx))
            patch.set_alpha(0.70)

        rng = np.random.default_rng(2026)
        for idx, values in enumerate(data):
            if values.size == 0:
                continue
            mean_value = float(np.mean(values))
            jitter = rng.normal(0.0, 0.035, size=values.size)
            ax.scatter(
                x_positions[idx] + jitter,
                values,
                s=8,
                color="#2B2B2B",
                alpha=0.22,
                linewidths=0,
                rasterized=True,
            )
            if np.isfinite(mean_value):
                ax.scatter(
                    x_positions[idx],
                    mean_value,
                    marker="D",
                    s=34,
                    color="white",
                    edgecolor="#1A1A1A",
                    linewidth=0.9,
                    zorder=5,
                )

        ax.set_ylabel(spec["label"])
        if spec["scale"] == "log":
            positive_chunks = [v[v > 0] for v in data if v.size and np.any(v > 0)]
            if positive_chunks:
                positive = np.concatenate(positive_chunks)
                ax.set_yscale("log")
                ax.set_ylim(
                    bottom=max(np.min(positive) * 0.65, 1e-12),
                    top=max(np.max(positive) * 3.0, np.min(positive) * 1.2),
                )
        elif metric == "ssim":
            value_chunks = [v for v in data if v.size]
            if value_chunks:
                all_values = np.concatenate(value_chunks)
                ymin = min(0.0, float(np.nanmin(all_values)) - 0.04)
                ymax = min(1.12, max(1.05, float(np.nanmax(all_values)) + 0.10))
                ax.set_ylim(ymin, ymax)

        ax.grid(True, axis="y", color="#D8D8D8", linewidth=0.75, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)

        for idx, values in enumerate(data):
            mean_value = float(np.mean(values)) if values.size else float("nan")
            ax.text(
                x_positions[idx],
                0.985,
                _format_mean_label(metric, mean_value),
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8.0,
                color="#1A1A1A",
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.82,
                },
            )

    axes[-1].set_xticks(x_positions)
    axes[-1].set_xticklabels(field_names)
    fig.suptitle(f"Reconstruction quality spread: {split_name}")
    fig.align_ylabels(axes)
    _save_figure(fig, out_dir / f"{split_name}_metric_distributions")


def _density_curve(values: np.ndarray, *, log_scale: bool):
    values = values[np.isfinite(values)]
    if log_scale:
        values = values[values > 0.0]
        if values.size == 0:
            return np.asarray([]), np.asarray([])
        z = np.log10(values)
    else:
        z = values

    if z.size == 0:
        return np.asarray([]), np.asarray([])

    if z.size == 1:
        center = float(z[0])
        half_width = max(abs(center) * 0.08, 0.08 if log_scale else 1e-6)
        grid = np.linspace(center - half_width, center + half_width, 160)
        bandwidth = max(half_width / 2.5, 1e-12)
    else:
        z_std = float(np.std(z, ddof=1))
        iqr = float(np.subtract(*np.percentile(z, [75, 25])))
        robust_std = min(z_std, iqr / 1.349) if iqr > 0.0 else z_std
        bandwidth = 1.06 * max(robust_std, 1e-12) * (z.size ** (-1.0 / 5.0))
        if not np.isfinite(bandwidth) or bandwidth <= 1e-12:
            bandwidth = max((float(np.max(z)) - float(np.min(z))) * 0.08, 1e-6)
        pad = 3.0 * bandwidth
        grid = np.linspace(float(np.min(z)) - pad, float(np.max(z)) + pad, 240)

    diff = (grid[:, None] - z[None, :]) / bandwidth
    density = np.exp(-0.5 * diff * diff).sum(axis=1)
    density /= max(z.size * bandwidth * np.sqrt(2.0 * np.pi), 1e-12)

    if log_scale:
        return grid, density
    return grid, density


def _density_axis_label(metric: str, spec: dict) -> str:
    if spec["scale"] == "log":
        return rf"$\log_{{10}}$({spec['label']})"
    return spec["label"]


def _plot_metric_pdf_grid(
    records: Sequence[Dict[str, object]],
    field_names: Sequence[str],
    split_name: str,
    out_dir: Path,
) -> None:
    _set_paper_style()
    metric_items = list(METRIC_SPECS.items())
    n_rows = len(field_names)
    n_cols = len(metric_items)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11.4, max(2.15 * n_rows, 3.2)),
        squeeze=False,
        sharey=False,
    )

    for row, field in enumerate(field_names):
        field_color = _field_color(field, row)
        for col, (metric, spec) in enumerate(metric_items):
            ax = axes[row, col]
            values = _metric_values(records, field, metric)
            xs, density = _density_curve(values, log_scale=(spec["scale"] == "log"))
            if xs.size:
                ax.fill_between(
                    xs,
                    density,
                    color=field_color,
                    alpha=0.34,
                    linewidth=0.0,
                    rasterized=True,
                )
                ax.plot(xs, density, color=field_color, linewidth=1.9)
                finite_values = values[np.isfinite(values)]
                if spec["scale"] == "log":
                    finite_values = finite_values[finite_values > 0.0]
                if finite_values.size:
                    mean_value = float(np.mean(finite_values))
                    mean_x = np.log10(mean_value) if spec["scale"] == "log" else mean_value
                    ax.axvline(
                        mean_x,
                        color="#1A1A1A",
                        linestyle="--",
                        linewidth=1.05,
                        alpha=0.85,
                    )
                    ax.text(
                        0.985,
                        0.88,
                        _format_mean_label(metric, mean_value),
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8.0,
                        color="#1A1A1A",
                    )

            if row == n_rows - 1:
                ax.set_xlabel(_density_axis_label(metric, spec))
            if col == 0:
                ax.set_ylabel(f"{field}\nDensity")
            else:
                ax.set_ylabel("Density")

            if row == 0:
                ax.set_title(spec["summary_label"])

            ax.grid(True, color="#D8D8D8", linewidth=0.7, alpha=0.70)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="both", which="major", labelsize=8.8)

    legend_handles = [
        Patch(facecolor=_field_color(field, idx), edgecolor="none", alpha=0.45, label=field)
        for idx, field in enumerate(field_names)
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper center",
        ncol=min(len(field_names), 5),
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.suptitle(
        f"Metric probability density by field: {split_name}",
        y=1.035,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    _save_figure(fig, out_dir / f"{split_name}_metric_pdf")


def _plot_summary_heatmap(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    field_names: Sequence[str],
    split_name: str,
    out_dir: Path,
) -> None:
    _set_paper_style()
    metric_names = list(METRIC_SPECS.keys())
    medians = np.asarray(
        [
            [summary[field][metric]["median"] for metric in metric_names]
            for field in field_names
        ],
        dtype=np.float64,
    )

    display = medians.copy()
    for j, metric in enumerate(metric_names):
        values = display[:, j]
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        if METRIC_SPECS[metric]["better"] == "lower":
            lo = np.nanmin(values[finite])
            hi = np.nanmax(values[finite])
            display[finite, j] = 1.0 - (values[finite] - lo) / (hi - lo + 1e-12)
        else:
            lo = np.nanmin(values[finite])
            hi = np.nanmax(values[finite])
            display[finite, j] = (values[finite] - lo) / (hi - lo + 1e-12)

    fig, ax = plt.subplots(figsize=(5.9, 3.85))
    im = ax.imshow(display, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels([METRIC_SPECS[m]["summary_label"] for m in metric_names])
    ax.set_yticks(np.arange(len(field_names)))
    ax.set_yticklabels(field_names)
    ax.set_title(f"Median quality summary: {split_name}")

    for i, field in enumerate(field_names):
        for j, metric in enumerate(metric_names):
            value = medians[i, j]
            if np.isfinite(value):
                text = f"{value:.3g}" if metric != "ssim" else f"{value:.3f}"
            else:
                text = "n/a"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=10.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized rank within metric")
    fig.tight_layout()
    _save_figure(fig, out_dir / f"{split_name}_median_summary_heatmap")


def _plot_metric_correlation(
    records: Sequence[Dict[str, object]],
    field_names: Sequence[str],
    split_name: str,
    out_dir: Path,
) -> None:
    _set_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.65))
    pairs = [("l2", "ssim"), ("l2", "spectral_lsd")]

    for ax, (x_metric, y_metric) in zip(axes, pairs):
        for idx, field in enumerate(field_names):
            xs = _metric_values(records, field, x_metric)
            ys = _metric_values(records, field, y_metric)
            n = min(xs.size, ys.size)
            if n == 0:
                continue
            ax.scatter(
                xs[:n],
                ys[:n],
                s=18,
                color=_field_color(field, idx),
                alpha=0.55,
                edgecolors="none",
                label=field,
                rasterized=True,
            )
        if METRIC_SPECS[x_metric]["scale"] == "log":
            ax.set_xscale("log")
        if METRIC_SPECS[y_metric]["scale"] == "log":
            ax.set_yscale("log")
        ax.set_xlabel(METRIC_SPECS[x_metric]["label"])
        ax.set_ylabel(METRIC_SPECS[y_metric]["label"])
        ax.grid(True, color="#D8D8D8", linewidth=0.75, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles = [
        Patch(facecolor=_field_color(field, idx), edgecolor="none", label=field)
        for idx, field in enumerate(field_names)
    ]
    axes[-1].legend(handles=handles, frameon=False, loc="best")
    fig.suptitle(f"Metric relationships: {split_name}", y=1.03)
    fig.tight_layout()
    _save_figure(fig, out_dir / f"{split_name}_metric_relationships")


def _make_figures(
    records: Sequence[Dict[str, object]],
    field_names: Sequence[str],
    split_name: str,
    out_dir: Path,
) -> None:
    if not records:
        return
    _plot_metric_distributions(records, field_names, split_name, out_dir)
    _plot_metric_pdf_grid(records, field_names, split_name, out_dir)
    _plot_summary_heatmap(_build_summary(records, field_names), field_names, split_name, out_dir)
    _plot_metric_correlation(records, field_names, split_name, out_dir)


def _records_by_split(records: Iterable[Dict[str, object]]):
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record["split"])].append(record)
    return grouped


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    if args.baseline_model is None:
        runtime = _load_model_and_config(args)
        cfg = runtime["cfg"]
        model = runtime["model"].to(device)
        model.eval()

        cond_fields = args.cond_fields if args.cond_fields is not None else cfg["vis_cond_fields"]
        cond_fields = _as_int_list(cond_fields)
        n_obs_source = args.n_obs_list if args.n_obs_list is not None else cfg["vis_n_obs_list"]
        n_obs_list = _align_per_field_values(
            n_obs_source,
            cond_fields,
            "n_obs_list",
            source_fields=cfg.get("vis_cond_fields", cfg.get("cond_fields")),
        )
        n_steps_generation = (
            int(args.n_steps_generation)
            if args.n_steps_generation is not None
            else int(cfg.get("n_steps_generation", 100))
        )
        ode_solver = args.ode_solver if args.ode_solver is not None else cfg.get("ode_solver", None)
        num_x = cfg.get("Num_x", None)
        num_y = cfg.get("Num_y", None)
        model_family = "pointcloud_ffm"
        method_label = "PointCloud FFM"
        config_path = runtime["yaml_path"]
        demo_num = int(args.Demo_Num)
        eval_context = contextlib.nullcontext()

        def reconstruct_phys(dataset, snapshot_index):
            return _ffm_reconstruct_phys(
                model=model,
                dataset=dataset,
                device=device,
                snapshot_index=snapshot_index,
                cond_fields=cond_fields,
                n_obs_list=n_obs_list,
                n_steps_generation=n_steps_generation,
                ode_solver=ode_solver,
            )

    else:
        runtime = _load_baseline_model_and_config(args, device=device)
        cfg = runtime["cfg"]
        baseline_bundle = runtime["baseline_bundle"]
        baseline_bundle.model.eval()

        shared_cond = cfg["shared"]["conditioning"]
        cond_fields = args.cond_fields if args.cond_fields is not None else shared_cond["vis_cond_fields"]
        cond_fields = _as_int_list(cond_fields)
        n_obs_source = args.n_obs_list if args.n_obs_list is not None else shared_cond["vis_n_obs_list"]
        n_obs_list = _align_per_field_values(
            n_obs_source,
            cond_fields,
            "n_obs_list",
            source_fields=shared_cond.get("vis_cond_fields", shared_cond.get("cond_fields")),
        )
        stage_cfg = runtime["model_baseline_module"].resolve_stage_config(cfg)
        sampling_cfg = stage_cfg.get("sampling", {})
        if args.n_steps_generation is not None:
            n_steps_generation = int(args.n_steps_generation)
        elif cfg["baseline_model"] == "latent_fm":
            n_steps_generation = int(sampling_cfg.get("benchmark_n_steps", [8])[0])
        else:
            n_steps_generation = int(sampling_cfg.get("sampling_N", 50))
        if cfg["baseline_model"] == "sit" and n_steps_generation < 2:
            raise ValueError("SiT baseline sampling requires --n-steps-generation >= 2.")
        ode_solver = args.ode_solver if args.ode_solver is not None else sampling_cfg.get("ode_solver", None)
        num_x = int(cfg["shared"]["data"]["num_x"])
        num_y = int(cfg["shared"]["data"]["num_y"])
        model_family = "baseline"
        method_label = f"{cfg['baseline_model']} stage-{int(cfg['training_stage'])}"
        config_path = runtime["config_path"]
        demo_num = int(cfg["shared"]["demo_num"])
        eval_context = runtime["adapter"].evaluation_weights(baseline_bundle)

        def reconstruct_phys(dataset, snapshot_index):
            return _baseline_reconstruct_phys(
                runtime=runtime,
                dataset=dataset,
                device=device,
                snapshot_index=snapshot_index,
                cond_fields=cond_fields,
                n_obs_list=n_obs_list,
                n_steps_generation=n_steps_generation,
                ode_solver=ode_solver,
            )

    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_family == "baseline":
        out_name = (
            f"full_eval_{cfg['baseline_model']}_Stage{int(cfg['training_stage'])}"
            f"_N{demo_num}_{eval_timestamp}_from_{runtime['model_root'].name}"
        )
    else:
        out_name = f"full_eval_N{demo_num}_{eval_timestamp}_from_{runtime['train_timestamp']}"
    out_dir = (
        runtime["demo_root"]
        / "Save_reconstruction_files"
        / "ForOfflineEvaluation"
        / out_name
    )
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    selected_splits = ["train", "test"] if args.split == "all" else [args.split]
    datasets = {split: runtime["build_dataset"](split) for split in selected_splits}
    field_names = list(getattr(next(iter(datasets.values())), "field_names", []))

    print("[*] Full-dataset reconstruction evaluation")
    print(f"[*] Method     : {method_label}")
    print(f"[*] Config     : {config_path}")
    print(f"[*] Checkpoint : {runtime['ckpt_path']}")
    print(f"[*] Output dir : {out_dir}")
    print(f"[*] Split(s)   : {', '.join(selected_splits)}")
    print(f"[*] cond_fields={list(map(int, cond_fields))}, n_obs={list(map(int, n_obs_list))}")
    print(f"[*] n_steps_generation={n_steps_generation}")

    all_records: List[Dict[str, object]] = []
    with eval_context:
        if model_family == "baseline":
            runtime["baseline_bundle"].model.eval()
        for split in selected_splits:
            split_records = _evaluate_one_split(
                dataset=datasets[split],
                split_name=split,
                device=device,
                reconstruct_phys=reconstruct_phys,
                num_x=num_x,
                num_y=num_y,
                eval_seed=args.eval_seed,
                start_index=args.start_index,
                snapshot_stride=args.snapshot_stride,
                max_snapshots=args.max_snapshots,
                progress_every=args.progress_every,
            )
            all_records.extend(split_records)

    _write_records_csv(all_records, out_dir / "metrics_by_snapshot.csv")

    summaries = {}
    per_split_csvs = {}
    grouped = _records_by_split(all_records)
    for split, records in grouped.items():
        summaries[split] = _build_summary(records, field_names)
        if len(selected_splits) > 1:
            split_csv_path = out_dir / f"{split}_metrics_by_snapshot.csv"
            _write_records_csv(records, split_csv_path)
            per_split_csvs[split] = str(split_csv_path)
        _make_figures(records, field_names, split, figures_dir)

    if args.split == "all":
        combined_records = list(all_records)
        summaries["all"] = _build_summary(combined_records, field_names)
        _make_figures(combined_records, field_names, "all", figures_dir)

    summary_payload = {
        "demo_num": int(demo_num),
        "model_family": model_family,
        "baseline_model": cfg.get("baseline_model"),
        "training_stage": int(cfg.get("training_stage", 0)) if model_family == "baseline" else None,
        "config_path": str(config_path),
        "model_root": str(runtime["model_root"]),
        "checkpoint": str(runtime["ckpt_path"]),
        "epoch": int(runtime["epoch"]),
        "requested_split": args.split,
        "evaluated_splits": selected_splits,
        "cond_fields": [int(v) for v in cond_fields],
        "n_obs_list": [int(v) for v in n_obs_list],
        "n_steps_generation": int(n_steps_generation),
        "ode_solver": ode_solver,
        "eval_seed": int(args.eval_seed),
        "start_index": int(args.start_index),
        "snapshot_stride": int(args.snapshot_stride),
        "max_snapshots": args.max_snapshots,
        "num_metric_rows": len(all_records),
        "num_snapshots_by_split": {
            split: int(len(records) / max(1, len(field_names)))
            for split, records in grouped.items()
        },
        "field_names": field_names,
        "metric_notes": {
            "l2": "Relative L2 error in physical units, computed per field over all grid points.",
            "ssim": "Single-scale 2D SSIM on the recovered structured grid; higher is better.",
            "spectral_lsd": "Log-spectral distance between shell-averaged 2D power spectra; lower is better.",
        },
        "summaries": summaries,
        "outputs": {
            "metrics_csv": str(out_dir / "metrics_by_snapshot.csv"),
            "per_split_csvs": per_split_csvs,
            "figures_dir": str(figures_dir),
        },
    }
    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary_payload, f, indent=2)

    print("[*] Full-dataset evaluation finished.")
    print(f"[*] Metrics CSV: {out_dir / 'metrics_by_snapshot.csv'}")
    print(f"[*] Summary    : {out_dir / 'evaluation_summary.json'}")
    print(f"[*] Figures    : {figures_dir}")


if __name__ == "__main__":
    main()

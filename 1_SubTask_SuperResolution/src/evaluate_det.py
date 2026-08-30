from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

try:
    from model_baseline import (
        build_dataset,
        get_baseline_adapter,
        load_yaml,
        safe_torch_load,
        validate_and_normalize_config,
    )
except ImportError:
    from .model_baseline import (
        build_dataset,
        get_baseline_adapter,
        load_yaml,
        safe_torch_load,
        validate_and_normalize_config,
    )


RUN_NAME_RE = re.compile(
    r"^Baseline_(?P<baseline>.+)_Stage(?P<stage>\d+)_DemoN(?P<demo>\d+)_"
    r"(?P<timestamp>\d{8}_\d{6})$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Standalone evaluator for trained deterministic baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--Demo-Num",
        dest="Demo_Num",
        type=int,
        default=None,
        help="Demo ID to recover. Required unless --run-dir or --checkpoint-path is used.",
    )
    parser.add_argument("--demo-root", type=str, default=".", help="Project/demo root directory.")
    parser.add_argument(
        "--config",
        type=str,
        default="Save_config/config_baseline_Det.yaml",
        help="Fallback config used only when a run-local or checkpoint config is unavailable.",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Explicit checkpoint path.")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific run directory to evaluate.")
    parser.add_argument(
        "--checkpoint",
        "--checkpoint-name",
        dest="checkpoint",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Checkpoint file to load from the recovered run directory.",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=None,
        choices=["senseiver", "mlp_rbf"],
        help="Optional model-family filter when recovering a run by demo number.",
    )
    parser.add_argument(
        "--training-stage",
        type=int,
        default=None,
        choices=[1],
        help="Optional training-stage filter when recovering a run by demo number.",
    )
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--snapshot-index", type=int, default=0, help="Index within the selected split.")
    parser.add_argument(
        "--vis-cond-fields",
        type=int,
        nargs="+",
        default=None,
        help="Override visualization conditioning fields.",
    )
    parser.add_argument(
        "--vis-n-obs-list",
        type=int,
        nargs="+",
        default=None,
        help="Override visualization sensor counts.",
    )
    parser.add_argument(
        "--dataset-mode",
        type=str,
        default=None,
        choices=["default", "pdebench_multires"],
        help="Override the dataset mode stored in the selected run config.",
    )
    parser.add_argument(
        "--eval-resolution",
        type=str,
        default=None,
        choices=["L", "M", "H"],
        help="Override the PDEBench evaluation resolution stored in the run config.",
    )
    parser.add_argument(
        "--Case-Truncate-Ratio",
        dest="Case_Truncate_Ratio",
        type=float,
        default=None,
        help="Override the PDEBench per-case leading-frame truncation ratio.",
    )
    parser.add_argument(
        "--n-steps-generation",
        type=int,
        default=None,
        help="Override sampling steps for generative deterministic-baseline adapters, when applicable.",
    )
    parser.add_argument(
        "--save-obs-consistency-plots",
        action="store_true",
        help="Save SenConsis relative L2 sensor-consistency metrics and figures.",
    )
    return parser.parse_args()


def _resolve_from_demo_root(path_value: str | Path, demo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (demo_root / path).resolve()


def _parse_run_name(run_dir: Path) -> Optional[dict[str, str]]:
    match = RUN_NAME_RE.fullmatch(run_dir.name)
    return match.groupdict() if match is not None else None


def _find_latest_run_dir(
    demo_root: Path,
    demo_num: int,
    baseline_model: Optional[str] = None,
    training_stage: Optional[int] = None,
) -> Path:
    save_root = demo_root / "Save_TrainedModel" / "det_baseline"
    candidates: list[tuple[str, Path]] = []
    if save_root.exists():
        for path in save_root.glob(f"Baseline_*_Stage*_DemoN{demo_num}_*"):
            parsed = _parse_run_name(path)
            if not path.is_dir() or parsed is None:
                continue
            if int(parsed["demo"]) != int(demo_num):
                continue
            if baseline_model is not None and parsed["baseline"] != baseline_model:
                continue
            if training_stage is not None and int(parsed["stage"]) != int(training_stage):
                continue
            candidates.append((parsed["timestamp"], path))

    if not candidates:
        filters = []
        if baseline_model is not None:
            filters.append(f"baseline_model={baseline_model}")
        if training_stage is not None:
            filters.append(f"training_stage={training_stage}")
        suffix = f" with {', '.join(filters)}" if filters else ""
        raise FileNotFoundError(
            f"No deterministic run found for Demo_Num={demo_num}{suffix} in {save_root}"
        )
    return max(candidates, key=lambda item: item[0])[1]


def _load_fallback_config(args: argparse.Namespace, demo_root: Path) -> Optional[dict]:
    config_path = _resolve_from_demo_root(args.config, demo_root)
    return load_yaml(config_path) if config_path.is_file() else None


def _resolve_run_and_checkpoint(
    args: argparse.Namespace,
    demo_root: Path,
    fallback_cfg: Optional[dict],
) -> tuple[Path, Path]:
    if args.checkpoint_path is not None and args.run_dir is not None:
        raise ValueError("Use only one of --checkpoint-path and --run-dir.")

    if args.checkpoint_path is not None:
        checkpoint_path = _resolve_from_demo_root(args.checkpoint_path, demo_root)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path.parent, checkpoint_path

    if args.run_dir is not None:
        run_dir = _resolve_from_demo_root(args.run_dir, demo_root)
    elif args.Demo_Num is not None:
        run_dir = _find_latest_run_dir(
            demo_root=demo_root,
            demo_num=args.Demo_Num,
            baseline_model=args.baseline_model,
            training_stage=args.training_stage,
        )
    else:
        if fallback_cfg is None:
            raise ValueError(
                "Provide --Demo-Num, --run-dir, or --checkpoint-path; the fallback config was not found."
            )
        normalized = validate_and_normalize_config(fallback_cfg)
        run_dir = _find_latest_run_dir(
            demo_root=demo_root,
            demo_num=int(normalized["shared"]["demo_num"]),
            baseline_model=args.baseline_model or normalized["baseline_model"],
            training_stage=args.training_stage or int(normalized["training_stage"]),
        )

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    checkpoint_path = run_dir / f"{args.checkpoint}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return run_dir, checkpoint_path


def _recover_run_config(
    run_dir: Path,
    checkpoint: dict,
    fallback_cfg: Optional[dict],
) -> tuple[dict, str]:
    run_cfg_path = run_dir / "run_config.yaml"
    if run_cfg_path.is_file():
        return load_yaml(run_cfg_path), str(run_cfg_path)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("config"), dict):
        return checkpoint["config"], "checkpoint['config']"
    if fallback_cfg is not None:
        return fallback_cfg, "--config fallback"
    raise FileNotFoundError(
        f"No run_config.yaml or embedded checkpoint config was found for {run_dir}"
    )


def _make_config_paths_absolute(cfg: dict, demo_root: Path) -> None:
    shared = cfg["shared"]
    paths = shared["paths"]
    if paths.get("data_path"):
        paths["data_path"] = str(_resolve_from_demo_root(paths["data_path"], demo_root))

    data_cfg = shared["data"]
    processed_root = data_cfg.get("pdebench_processed_root")
    if processed_root:
        data_cfg["pdebench_processed_root"] = str(
            _resolve_from_demo_root(processed_root, demo_root)
        )
    manifest_path = data_cfg.get("multires_manifest_path")
    if manifest_path:
        data_cfg["multires_manifest_path"] = str(
            _resolve_from_demo_root(manifest_path, demo_root)
        )


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace, demo_root: Path) -> dict:
    cfg = validate_and_normalize_config(cfg)
    if args.baseline_model is not None and cfg["baseline_model"] != args.baseline_model:
        raise ValueError(
            f"Selected run is {cfg['baseline_model']!r}, not requested {args.baseline_model!r}."
        )
    if args.training_stage is not None and int(cfg["training_stage"]) != args.training_stage:
        raise ValueError(
            f"Selected run is stage {cfg['training_stage']}, not requested stage {args.training_stage}."
        )

    data_cfg = cfg["shared"]["data"]
    if args.dataset_mode is not None:
        data_cfg["dataset_mode"] = args.dataset_mode
    if args.eval_resolution is not None:
        data_cfg["eval_resolution"] = args.eval_resolution
    if args.Case_Truncate_Ratio is not None:
        if not 0.0 <= args.Case_Truncate_Ratio < 1.0:
            raise ValueError("--Case-Truncate-Ratio must satisfy 0 <= ratio < 1.")
        data_cfg["Case_Truncate_Ratio"] = float(args.Case_Truncate_Ratio)

    cfg = validate_and_normalize_config(cfg)
    conditioning = cfg["shared"]["conditioning"]
    if args.vis_cond_fields is not None:
        conditioning["vis_cond_fields"] = [int(value) for value in args.vis_cond_fields]
    if args.vis_n_obs_list is not None:
        conditioning["vis_n_obs_list"] = [int(value) for value in args.vis_n_obs_list]
    if len(conditioning["vis_cond_fields"]) != len(conditioning["vis_n_obs_list"]):
        raise ValueError("--vis-cond-fields and --vis-n-obs-list must have matching lengths.")

    _make_config_paths_absolute(cfg, demo_root)
    return cfg


def _validate_checkpoint_identity(cfg: dict, checkpoint: dict, run_dir: Path) -> None:
    parsed = _parse_run_name(run_dir)
    expected_model = cfg["baseline_model"]
    expected_stage = int(cfg["training_stage"])
    if parsed is not None:
        if parsed["baseline"] != expected_model or int(parsed["stage"]) != expected_stage:
            raise ValueError(
                f"Run directory identity does not match its config: {run_dir.name}"
            )
    if isinstance(checkpoint, dict):
        checkpoint_model = checkpoint.get("baseline_model")
        checkpoint_stage = checkpoint.get("training_stage")
        if checkpoint_model is not None and str(checkpoint_model).lower() != expected_model:
            raise ValueError(
                f"Checkpoint model {checkpoint_model!r} does not match config model {expected_model!r}."
            )
        if checkpoint_stage is not None and int(checkpoint_stage) != expected_stage:
            raise ValueError(
                f"Checkpoint stage {checkpoint_stage} does not match config stage {expected_stage}."
            )


def main() -> None:
    args = parse_args()
    demo_root = Path(args.demo_root).expanduser().resolve()
    fallback_cfg = _load_fallback_config(args, demo_root)

    run_dir, checkpoint_path = _resolve_run_and_checkpoint(args, demo_root, fallback_cfg)
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    raw_cfg, config_source = _recover_run_config(run_dir, checkpoint, fallback_cfg)
    cfg = _apply_cli_overrides(raw_cfg, args, demo_root)
    _validate_checkpoint_identity(cfg, checkpoint, run_dir)

    parsed_run = _parse_run_name(run_dir)
    demo_num = int(parsed_run["demo"]) if parsed_run is not None else int(cfg["shared"]["demo_num"])
    train_timestamp = parsed_run["timestamp"] if parsed_run is not None else "unknown"
    if args.Demo_Num is not None and demo_num != args.Demo_Num:
        raise ValueError(f"Selected run has Demo_Num={demo_num}, not requested {args.Demo_Num}.")

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    stats_path = run_dir / "dataset_stats.pt"
    if not stats_path.is_file():
        raise FileNotFoundError(f"Dataset statistics not found: {stats_path}")
    dataset = build_dataset(cfg, split=args.split, stats_path=stats_path)
    if not 0 <= args.snapshot_index < len(dataset):
        raise IndexError(
            f"--snapshot-index {args.snapshot_index} is outside the {args.split} split "
            f"with {len(dataset)} samples."
        )

    adapter = get_baseline_adapter(cfg["baseline_model"])
    try:
        bundle = adapter.build_for_training(
            cfg=cfg,
            device=device,
            run_dir=run_dir,
            train_set=dataset,
            val_set=dataset,
        )
        adapter.load_checkpoint(bundle, checkpoint)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to rebuild {cfg['baseline_model']} from {checkpoint_path}: {exc}"
        ) from exc

    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        demo_root
        / "Save_reconstruction_files"
        / "ForOfflineEvaluation"
        / f"eval_det_{cfg['baseline_model']}_N{demo_num}_{eval_timestamp}_from_{train_timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch = int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0
    bundle.model.eval()
    with torch.no_grad(), adapter.evaluation_weights(bundle):
        metrics = adapter.visualize(
            bundle=bundle,
            dataset=dataset,
            save_dir=output_dir,
            epoch=epoch,
            snapshot_index=int(args.snapshot_index),
            n_steps=args.n_steps_generation,
            save_obs_consistency_plots=args.save_obs_consistency_plots,
        )

    conditioning = cfg["shared"]["conditioning"]
    summary = {
        "demo_num": demo_num,
        "baseline_model": cfg["baseline_model"],
        "training_stage": int(cfg["training_stage"]),
        "config_source": config_source,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": epoch,
        "device": str(device),
        "split": args.split,
        "snapshot_index": int(args.snapshot_index),
        "dataset_mode": cfg["shared"]["data"]["dataset_mode"],
        "eval_resolution": cfg["shared"]["data"].get("eval_resolution"),
        "Case_Truncate_Ratio": cfg["shared"]["data"].get("Case_Truncate_Ratio"),
        "vis_cond_fields": [int(value) for value in conditioning["vis_cond_fields"]],
        "vis_n_obs_list": [int(value) for value in conditioning["vis_n_obs_list"]],
        "n_steps_generation": args.n_steps_generation,
        "save_obs_consistency_plots": bool(args.save_obs_consistency_plots),
        "metrics": metrics,
    }
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("[*] Deterministic baseline evaluation finished.")
    print(f"[*] Run       : {run_dir}")
    print(f"[*] Config    : {config_source}")
    print(f"[*] Checkpoint: {checkpoint_path}")
    print(f"[*] Output dir: {output_dir}")
    print(f"[*] Metrics   : {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()

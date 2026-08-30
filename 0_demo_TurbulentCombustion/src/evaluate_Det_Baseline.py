from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch

try:
    from model_baseline import (
        build_dataset,
        ensure_absolute,
        find_latest_run_dir,
        get_baseline_adapter,
        infer_device,
        load_yaml,
        safe_torch_load,
        validate_and_normalize_config,
    )
except ImportError:
    from .model_baseline import (
        build_dataset,
        ensure_absolute,
        find_latest_run_dir,
        get_baseline_adapter,
        infer_device,
        load_yaml,
        safe_torch_load,
        validate_and_normalize_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified evaluator for deterministic baselines.")
    parser.add_argument(
        "--config",
        type=str,
        default="Save_config/config_baseline_Det.yaml",
        help="Unified deterministic config path. Used when --run-dir / --checkpoint-path are omitted.",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Explicit checkpoint path.")
    parser.add_argument("--baseline-model", type=str, default=None, help="Override baseline_model from YAML.")
    parser.add_argument("--training-stage", type=int, default=None, help="Override training_stage from YAML.")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific unified run directory to evaluate.")
    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best",
        choices=["best", "last"],
        help="Checkpoint file to use when only a run directory is provided.",
    )
    parser.add_argument("--device", type=str, default=None, help="Explicit device, e.g. cuda:0 or cpu.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--snapshot-index", type=int, default=0)
    parser.add_argument(
        "--vis-cond-fields",
        type=int,
        nargs="+",
        default=None,
        help="Optional visualization conditioning-field override, e.g. --vis-cond-fields 2 3.",
    )
    parser.add_argument(
        "--vis-n-obs-list",
        type=int,
        nargs="+",
        default=None,
        help="Optional visualization sensor-count override, e.g. --vis-n-obs-list 256 256.",
    )
    parser.add_argument(
        "--save-obs-consistency-plots",
        action="store_true",
        help="Save SenConsis relative L2 sensor-consistency metrics and figures.",
    )
    return parser.parse_args()


def _resolve_run_and_checkpoint(args: argparse.Namespace, cfg: dict) -> tuple[Path, Path, dict]:
    if args.checkpoint_path is not None:
        checkpoint_path = ensure_absolute(args.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        run_dir = checkpoint_path.parent
        run_cfg_path = run_dir / "run_config.yaml"
        if run_cfg_path.exists():
            cfg = validate_and_normalize_config(load_yaml(run_cfg_path))
        return run_dir, checkpoint_path, cfg

    if args.run_dir is not None:
        run_dir = ensure_absolute(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        checkpoint_path = run_dir / f"{args.checkpoint_name}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        run_cfg_path = run_dir / "run_config.yaml"
        if run_cfg_path.exists():
            cfg = validate_and_normalize_config(load_yaml(run_cfg_path))
        return run_dir, checkpoint_path, cfg

    save_root = ensure_absolute(cfg["shared"]["paths"]["save_root"])
    latest_run_dir = find_latest_run_dir(save_root, cfg)
    if latest_run_dir is None:
        raise FileNotFoundError(
            "No matching deterministic run directory was found. "
            "Provide --run-dir or --checkpoint-path explicitly if needed."
        )
    checkpoint_path = latest_run_dir / f"{args.checkpoint_name}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return latest_run_dir, checkpoint_path, cfg


def main() -> None:
    args = parse_args()

    cfg = load_yaml(ensure_absolute(args.config))
    if args.baseline_model is not None:
        cfg["baseline_model"] = args.baseline_model
    if args.training_stage is not None:
        cfg["training_stage"] = int(args.training_stage)
    cfg = validate_and_normalize_config(cfg)

    run_dir, checkpoint_path, cfg = _resolve_run_and_checkpoint(args, cfg)
    shared_cond = cfg["shared"]["conditioning"]
    if args.vis_cond_fields is not None:
        shared_cond["vis_cond_fields"] = [int(v) for v in args.vis_cond_fields]
    if args.vis_n_obs_list is not None:
        shared_cond["vis_n_obs_list"] = [int(v) for v in args.vis_n_obs_list]

    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    device = infer_device(args.device, cfg["shared"]["device_ids"])

    stats_path = run_dir / "dataset_stats.pt"
    dataset = build_dataset(cfg, split=args.split, stats_path=stats_path)

    adapter = get_baseline_adapter(cfg["baseline_model"])
    bundle = adapter.build_for_training(
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        train_set=dataset,
        val_set=dataset,
    )
    adapter.load_checkpoint(bundle, checkpoint)

    evaluation_root = run_dir / "Evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = evaluation_root / f"offline_eval_{args.split}_{args.snapshot_index:04d}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        bundle.model.eval()
        metrics = adapter.visualize(
            bundle=bundle,
            dataset=dataset,
            save_dir=output_dir,
            epoch=int(checkpoint.get("epoch", 0)),
            snapshot_index=int(args.snapshot_index),
            save_obs_consistency_plots=args.save_obs_consistency_plots,
        )

    summary = {
        "baseline_model": cfg["baseline_model"],
        "training_stage": int(cfg["training_stage"]),
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "split": args.split,
        "snapshot_index": int(args.snapshot_index),
        "effective_vis_cond_fields": [int(v) for v in shared_cond.get("vis_cond_fields", [])],
        "effective_vis_n_obs_list": [int(v) for v in shared_cond.get("vis_n_obs_list", [])],
        "metrics": metrics,
    }
    with open(output_dir / "evaluation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Run directory: {run_dir}")
    print(f"Checkpoint:    {checkpoint_path}")
    print(f"Output dir:    {output_dir}")
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6e}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from datetime import datetime

import torch

try:
    from model_baseline import (
        TrainingHistoryLogger,
        build_dataloader,
        build_dataset,
        build_run_name,
        copy_config_backup,
        ensure_absolute,
        find_latest_run_dir,
        get_baseline_adapter,
        infer_device,
        load_yaml,
        resolve_stage_config,
        safe_torch_load,
        save_yaml,
        set_seed,
        validate_and_normalize_config,
    )
except ImportError:
    from .model_baseline import (
        TrainingHistoryLogger,
        build_dataloader,
        build_dataset,
        build_run_name,
        copy_config_backup,
        ensure_absolute,
        find_latest_run_dir,
        get_baseline_adapter,
        infer_device,
        load_yaml,
        resolve_stage_config,
        safe_torch_load,
        save_yaml,
        set_seed,
        validate_and_normalize_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Unified trainer for deterministic baselines.")
    parser.add_argument(
        "--config",
        type=str,
        default="Save_config/config_baseline_Det.yaml",
        help="Path to the unified deterministic YAML config.",
    )
    parser.add_argument("--baseline-model", type=str, default=None, help="Override baseline_model from YAML.")
    parser.add_argument("--training-stage", type=int, default=None, help="Override training_stage from YAML.")
    parser.add_argument("--device", type=str, default=None, help="Explicit device, e.g. cuda:0 or cpu.")
    parser.add_argument("--reload", action="store_true", help="Resume from the latest unified run for this model/stage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = ensure_absolute(args.config)
    cfg = load_yaml(config_path)

    if args.baseline_model is not None:
        cfg["baseline_model"] = args.baseline_model
    if args.training_stage is not None:
        cfg["training_stage"] = int(args.training_stage)
    if args.reload:
        cfg.setdefault("shared", {}).update({"reload": True})

    cfg = validate_and_normalize_config(cfg)
    stage_cfg = resolve_stage_config(cfg)

    set_seed(int(cfg["shared"]["seed"]))
    device = infer_device(args.device, cfg["shared"]["device_ids"])
    save_root = ensure_absolute(cfg["shared"]["paths"]["save_root"])
    config_backup_root = ensure_absolute(cfg["shared"]["paths"]["config_backup_root"])
    adapter = get_baseline_adapter(cfg["baseline_model"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = build_run_name(cfg, timestamp)
    run_dir = save_root / run_name

    start_epoch = 1
    best_val = float("inf")
    reload_ckpt = None

    if bool(cfg["shared"]["reload"]):
        latest_run_dir = find_latest_run_dir(save_root, cfg)
        if latest_run_dir is not None:
            preferred_checkpoint = latest_run_dir / "last.pt"
            fallback_checkpoint = latest_run_dir / "best.pt"
            if preferred_checkpoint.exists():
                reload_ckpt = safe_torch_load(preferred_checkpoint, map_location="cpu")
                run_dir = latest_run_dir
            elif fallback_checkpoint.exists():
                reload_ckpt = safe_torch_load(fallback_checkpoint, map_location="cpu")
                run_dir = latest_run_dir

        if reload_ckpt is not None:
            start_epoch = int(reload_ckpt.get("epoch", 0)) + 1
            best_val = float(reload_ckpt.get("val_loss", float("inf")))

    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_root = run_dir / "Evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)

    copy_config_backup(config_path, config_backup_root, run_name)
    save_yaml(run_dir / "run_config.yaml", cfg)
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_path": str(config_path),
                "baseline_model": cfg["baseline_model"],
                "training_stage": int(cfg["training_stage"]),
                "device": str(device),
                "started_at": timestamp,
            },
            handle,
            indent=2,
        )

    stats_path = run_dir / "dataset_stats.pt"
    train_set = build_dataset(cfg, split="train", stats_path=stats_path)
    val_set = build_dataset(cfg, split="val", stats_path=stats_path)
    batch_size = int(stage_cfg["training"]["batch_size"])
    num_workers = int(cfg["shared"]["data"]["num_workers"])

    train_loader = build_dataloader(train_set, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    val_loader = build_dataloader(val_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    bundle = adapter.build_for_training(
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        train_set=train_set,
        val_set=val_set,
    )
    if reload_ckpt is not None:
        adapter.load_checkpoint(bundle, reload_ckpt)

    logger = TrainingHistoryLogger(run_dir)

    print(f"Config:         {config_path}")
    print(f"Run directory:  {run_dir}")
    print(f"Baseline:       {cfg['baseline_model']}")
    print(f"Stage:          {cfg['training_stage']}")
    print(f"Device:         {device}")
    print(f"Train samples:  {len(train_set)}")
    print(f"Val samples:    {len(val_set)}")
    if reload_ckpt is not None:
        print(f"Resuming from epoch {start_epoch}")

    total_epochs = int(stage_cfg["training"]["epochs"])
    eval_every = int(stage_cfg["training"]["eval_every"])
    save_every = int(stage_cfg["training"]["save_every"])

    for epoch in range(start_epoch, total_epochs + 1):
        train_loss = adapter.run_epoch(bundle, train_loader, training=True, epoch=epoch)
        if bundle.scheduler is not None:
            bundle.scheduler.step()

        val_loss = None
        if epoch % eval_every == 0 or epoch == 1:
            with torch.no_grad():
                val_loss = adapter.run_epoch(bundle, val_loader, training=False, epoch=epoch)

            checkpoint = adapter.build_checkpoint(
                bundle=bundle,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
            )
            checkpoint["run_dir"] = str(run_dir)
            checkpoint["run_name"] = run_dir.name
            checkpoint["config"] = cfg

            torch.save(checkpoint, run_dir / "last.pt")
            if float(val_loss) < best_val:
                best_val = float(val_loss)
                torch.save(checkpoint, run_dir / "best.pt")

        if epoch % save_every == 0:
            epoch_eval_dir = evaluation_root / f"epoch_{epoch:04d}"
            epoch_eval_dir.mkdir(parents=True, exist_ok=True)
            metrics = adapter.visualize(
                bundle=bundle,
                dataset=val_set,
                save_dir=epoch_eval_dir,
                epoch=epoch,
                snapshot_index=0,
            )
            metric_text = ", ".join(f"{name}:{value:.4e}" for name, value in metrics.items())
            print(f"[eval] epoch={epoch:04d} {metric_text}")

        logger.append(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        if val_loss is None:
            print(f"[train] epoch={epoch:04d} loss={train_loss:.6e}")
        else:
            print(f"[train] epoch={epoch:04d} loss={train_loss:.6e} val={val_loss:.6e}")

    print("Training complete.")
    print(f"Best val loss: {best_val:.6e}")
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()

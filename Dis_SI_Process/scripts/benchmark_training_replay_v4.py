#!/usr/bin/env python
"""Standardized clean-GPU training-update replay for Figure 5 V4.

Historical wall-clock evidence for the adopted Cond_T checkpoints is
incomplete. This runner therefore measures the canonical forward, backward,
gradient-clipping, and optimizer-step path at each stage's adopted batch and
query configuration. The result is direct *training-update time*, not
historical runtime or a total-training estimate. A canonical batch that cannot
execute on the replay GPU remains explicitly unavailable.

The replay is deliberately non-destructive: checkpoint and dataset files are
read only, all parameter changes live in ephemeral model objects, and the only
writes are compact result tables plus manifest/QA under a new ValidationV4 run.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
DEMO_ROOT = REPO_ROOT / "0_demo_TurbulentCombustion"
SCRIPT_ROOT = DEMO_ROOT / "Save_TrainedModel" / "_TrainedModels" / "_Scripts"
SRC_ROOT = DEMO_ROOT / "src"
TOOLS_ROOT = DEMO_ROOT / "tools"
for path in (SCRIPT_ROOT, SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_pointcloud_ffm as pointcloud_train
from benchmark_validation_v3 import (
    assert_clean_gpu,
    environment,
    gpu_index,
    gpu_state,
)
from common.config import load_config
from common.model_loader import baseline_lib, canonical, load_model

SCHEMA_VERSION = "figure5-validation-v4-training-cost-1"
METHOD_ORDER = [
    "DMF-Gen",
    "FFM-FNO",
    "FFM-Perceiver",
    "Latent FM",
    "SiT",
    "MLP-RBF",
    "Geo-FNO",
    "Senseiver",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty formal table: {path.name}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _slug(value: str) -> str:
    return "_".join(part for part in "".join(character.lower() if character.isalnum() else " " for character in value).split())


def _stage_sort_key(stage: Mapping[str, Any]) -> tuple[int, str]:
    role = str(stage.get("role", ""))
    return (0 if role == "required_stage_1" else 1, str(stage.get("name", "")))


def build_replay_plan(config: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[str, Any]:
    """Build the nine-stage replay plan without importing or mutating models."""

    records = {str(record.get("method")): record for record in audit.get("records", [])}
    rows: list[dict[str, Any]] = []
    for method in config.get("method_order", METHOD_ORDER):
        record = records.get(str(method), {})
        stages = sorted(
            [stage for stage in record.get("stages", []) if stage.get("include_in_total", True)],
            key=_stage_sort_key,
        )
        for ordinal, stage in enumerate(stages, start=1):
            metadata = stage.get("metadata") or {}
            role = str(stage.get("role", "adopted_checkpoint_training_stage"))
            rows.append(
                {
                    "method": str(method),
                    "stage_ordinal": ordinal,
                    "stage_id": f"{_slug(str(method))}__{_slug(role)}",
                    "stage_name": str(stage.get("name", role)),
                    "stage_role": role,
                    "checkpoint_path": str(stage.get("path", "")),
                    "config_path": str(stage.get("config_path", "")),
                    "update_count": metadata.get("update_count"),
                    "update_count_status": metadata.get("update_count_status"),
                    "update_count_sources": metadata.get("update_count_sources", []),
                    "training_config": stage.get("training_config", {}),
                    "write_checkpoint": False,
                    "mutate_archive": False,
                }
            )
    requirements = config["promotion"]["replay_requires"]
    return {
        "status": "planned",
        "rows": rows,
        "protocol": {
            "warmup_updates": int(requirements["warmup_updates"]),
            "measured_updates": int(requirements["measured_updates"]),
            "measured_blocks": int(requirements.get("measured_blocks", 10)),
            "updates_per_block": int(requirements.get("updates_per_block", 10)),
            "timing_boundary": str(requirements.get("timing_boundary", "synchronized_update_core_preloaded_batch")),
            "stability_gate": str(requirements.get("stability_gate", "first_five_vs_last_five_block_medians")),
        },
        "safety": {
            "execution_enabled": False,
            "checkpoint_write": False,
            "archive_mutation": False,
            "filesystem_mtime_used": False,
        },
    }


def _move_optimizer_state(optimizer: torch.optim.Optimizer | None, device: torch.device) -> None:
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _device_batch(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=False)
    if isinstance(value, dict):
        return {key: _device_batch(child, device) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_device_batch(child, device) for child in value)
    if isinstance(value, list):
        return [_device_batch(child, device) for child in value]
    return value


def _pointcloud_batch(loaded: Any, batch_size: int) -> dict[str, Any]:
    loader = DataLoader(
        loaded.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=pointcloud_train.collate_snapshots_legacy,
    )
    return _device_batch(next(iter(loader)), loaded.device)


def _baseline_batch(loaded: Any, batch_size: int) -> dict[str, Any]:
    loader = baseline_lib.build_dataloader(
        loaded.dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )
    return _device_batch(next(iter(loader)), loaded.device)


def _quiet_epoch(module: Any, call: Callable[[], float]) -> float:
    class QuietProgress:
        def __init__(self, iterable: Any):
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
            return None

    original = module.tqdm
    module.tqdm = lambda iterable, *args, **kwargs: QuietProgress(iterable)
    try:
        return float(call())
    finally:
        module.tqdm = original


def _load_latent_stage1(stage: Mapping[str, Any], device: torch.device) -> Any:
    checkpoint_path = _resolve(str(stage["checkpoint_path"]))
    args = argparse.Namespace(baseline_model="auto", split="train", n_steps=2, ode_solver="euler")
    _, _, cfg, dataset, bundle, _, _, _ = canonical.load_baseline_context(
        args=args,
        checkpoint_arg=str(checkpoint_path),
        device=device,
    )
    return SimpleNamespace(
        method="Latent FM",
        family="latent_fm_stage1",
        checkpoint_path=checkpoint_path,
        config=cfg,
        dataset=dataset,
        model=bundle,
        device=device,
    )


def _baseline_batch_size(bundle: Any) -> int:
    stage_cfg = baseline_lib.resolve_stage_config(bundle.config)
    return int(stage_cfg["training"]["batch_size"])


def _pointcloud_optimizer(loaded: Any) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(
        loaded.model.parameters(),
        lr=float(loaded.config.get("lr", 1.0e-4)),
        weight_decay=float(loaded.config.get("weight_decay", 0.0)),
    )
    checkpoint = torch.load(loaded.checkpoint_path, map_location=loaded.device, weights_only=False)
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("optimizer"), Mapping):
        optimizer.load_state_dict(checkpoint["optimizer"])
    _move_optimizer_state(optimizer, loaded.device)
    return optimizer


def _prepare_stage_runtime(
    stage: Mapping[str, Any],
    method_config: Mapping[str, Any],
    device: torch.device,
) -> tuple[Any, Callable[[], float], int, dict[str, Any]]:
    method = str(stage["method"])
    if method == "Latent FM" and str(stage["stage_role"]) == "required_stage_1":
        loaded = _load_latent_stage1(stage, device)
    else:
        loaded = load_model(
            dict(method_config),
            "Cond_T",
            checkpoint="last",
            split="train",
            device=str(device),
            n_steps=2,
            ode_solver="euler",
        )

    if loaded.family == "pointcloud_ffm":
        adopted_batch_size = int(loaded.config["batch_size"])
        batch_size = adopted_batch_size
        batch = _pointcloud_batch(loaded, batch_size)
        optimizer = _pointcloud_optimizer(loaded)
        model = loaded.model
        cfg = loaded.config
        data_path_config = pointcloud_train.resolve_data_path_config({"data_path_mode": "legacy"})

        def update() -> float:
            return _quiet_epoch(
                pointcloud_train,
                lambda: pointcloud_train.run_epoch(
                    model=model,
                    loader=[batch],
                    optimizer=optimizer,
                    device=device,
                    cond_fields=cfg["cond_fields"],
                    n_obs_min_list=cfg["n_obs_min_list"],
                    n_obs_max_list=cfg["n_obs_max_list"],
                    n_query_points=int(cfg.get("n_query_points", 4096)),
                    query_sampling=str(cfg.get("query_sampling", "uniform")),
                    query_sample_near_ratio=float(cfg.get("query_sample_near_ratio", 0.25)),
                    query_sample_far_ratio=float(cfg.get("query_sample_far_ratio", 0.25)),
                    query_sample_sigma_ratio=float(cfg.get("query_sample_sigma_ratio", 0.05)),
                    epoch=0,
                    data_path_config=data_path_config,
                    train_query_microbatch_size=cfg.get("train_query_microbatch_size"),
                    reuse_condition_context_across_query_microbatches=bool(
                        cfg.get("reuse_condition_context_across_query_microbatches", True)
                    ),
                ),
            )

        details = {
            "family": loaded.family,
            "backbone": loaded.backbone,
            "batch_size": batch_size,
            "adopted_batch_size": adopted_batch_size,
            "n_query_points": None if getattr(model, "requires_full_grid", False) else int(cfg.get("n_query_points", 4096)),
            "optimizer": type(optimizer).__name__,
        }
        return loaded, update, batch_size, details

    bundle = loaded.model
    if bundle.optimizer is None:
        raise RuntimeError(f"{method} replay bundle has no optimizer")
    _move_optimizer_state(bundle.optimizer, device)
    adopted_batch_size = _baseline_batch_size(bundle)
    batch_size = adopted_batch_size
    batch = _baseline_batch(loaded, batch_size)

    def update() -> float:
        return _quiet_epoch(
            baseline_lib,
            lambda: bundle.adapter.run_epoch(bundle, [batch], training=True, epoch=0),
        )

    details = {
        "family": loaded.family,
        "backbone": getattr(loaded, "backbone", bundle.baseline_model),
        "batch_size": batch_size,
        "adopted_batch_size": adopted_batch_size,
        "n_query_points": baseline_lib.resolve_stage_config(bundle.config)["training"].get("n_query_points"),
        "optimizer": type(bundle.optimizer).__name__,
        "training_stage": int(bundle.training_stage),
    }
    return loaded, update, batch_size, details


def _timed_update(update: Callable[[], float], device: torch.device) -> tuple[float, float, float]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    loss = float(update())
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    peak_mib = torch.cuda.max_memory_allocated(device) / 2**20
    if not math.isfinite(loss) or not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        raise RuntimeError(f"non-finite replay update: loss={loss}, elapsed_ms={elapsed_ms}")
    return loss, elapsed_ms, peak_mib


def _quantiles(values: Sequence[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    return tuple(float(value) for value in np.quantile(array, [0.25, 0.50, 0.75]))


def _benchmark_stage(
    stage: Mapping[str, Any],
    method_config: Mapping[str, Any],
    device: torch.device,
    *,
    warmups: int,
    blocks: int,
    updates_per_block: int,
    tolerance: float,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    checkpoint = _resolve(str(stage["checkpoint_path"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    config_path = _resolve(str(stage["config_path"]))
    config_sha = sha256_file(config_path) if config_path.is_file() else ""
    loaded, update, _, details = _prepare_stage_runtime(stage, method_config, device)
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.cuda.manual_seed_all(seed)
    for _ in range(warmups):
        _timed_update(update, device)
    rows: list[dict[str, Any]] = []
    block_medians: list[float] = []
    for block in range(blocks):
        values: list[float] = []
        for offset in range(updates_per_block):
            loss, elapsed, peak = _timed_update(update, device)
            values.append(elapsed)
            rows.append(
                {
                    "method": stage["method"],
                    "stage_id": stage["stage_id"],
                    "block": block,
                    "update_in_block": offset,
                    "update_index": block * updates_per_block + offset,
                    "elapsed_ms": elapsed,
                    "loss": loss,
                    "peak_allocated_mib": peak,
                }
            )
        block_medians.append(float(statistics.median(values)))
    elapsed_values = [float(row["elapsed_ms"]) for row in rows]
    q25, median, q75 = _quantiles(elapsed_values)
    half = blocks // 2
    if half == 0:
        early = late = float(block_medians[0])
    else:
        early = float(statistics.median(block_medians[:half]))
        late = float(statistics.median(block_medians[half:]))
    stability_delta = abs(early - late) / max(median, 1.0e-12)
    update_count = int(stage["update_count"])
    stage_summary = {
        "method": stage["method"],
        "stage_ordinal": stage["stage_ordinal"],
        "stage_id": stage["stage_id"],
        "stage_name": stage["stage_name"],
        "stage_role": stage["stage_role"],
        "status": "ok" if stability_delta <= tolerance else "unstable",
        "checkpoint_path": stage["checkpoint_path"],
        "checkpoint_sha256": checkpoint_sha,
        "config_path": stage["config_path"],
        "config_sha256": config_sha,
        "update_count": update_count,
        "warmup_updates": warmups,
        "measured_updates": len(rows),
        "measured_blocks": blocks,
        "updates_per_block": updates_per_block,
        "update_time_q25_ms": q25,
        "update_time_median_ms": median,
        "update_time_q75_ms": q75,
        "first_half_block_median_ms": early,
        "last_half_block_median_ms": late,
        "stability_delta_fraction": stability_delta,
        "stability_tolerance_fraction": tolerance,
        "peak_allocated_mib": max(float(row["peak_allocated_mib"]) for row in rows),
        "samples_per_second": float(details["batch_size"]) / (median / 1000.0),
        **details,
    }
    del loaded, update
    gc.collect()
    torch.cuda.empty_cache()
    return stage_summary, rows


def _native_errors() -> dict[str, dict[str, str]]:
    path = PACKAGE_ROOT / "results" / "ValidationV3" / "CostClean" / "formal_cost_clean_v3_20260830_v3" / "native_summary.csv"
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["method"]): row for row in rows}


def _method_summary(stage_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    native = _native_errors()
    output: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        stages = [row for row in stage_rows if row["method"] == method]
        single_stage = len(stages) == 1
        status = "ok" if single_stage and stages[0]["status"] == "ok" else "unavailable"
        source = native[method]
        stage = stages[0] if single_stage else None
        output.append(
            {
                "method": method,
                "status": status,
                "cost_value": float(stage["update_time_median_ms"]) if status == "ok" else "",
                "cost_low": float(stage["update_time_q25_ms"]) if status == "ok" else "",
                "cost_high": float(stage["update_time_q75_ms"]) if status == "ok" else "",
                "training_update_time_ms": float(stage["update_time_median_ms"]) if status == "ok" else "",
                "error": source["error"],
                "error_ci_low": source["error_ci_low"],
                "error_ci_high": source["error_ci_high"],
                "training_cost_basis": "median synchronized canonical training update at the adopted stage batch/query configuration",
                "unavailable_reason": ""
                if status == "ok"
                else (
                    "method has multiple required training stages; no single update-time scalar is defensible"
                    if len(stages) > 1
                    else str(stages[0].get("failure_reason", "canonical training update unavailable"))
                    if stages
                    else "required training stage was not resolved"
                ),
                "stage_count": len(stages),
                "total_update_count": sum(int(row["update_count"]) for row in stages),
                "checkpoint_path": source["checkpoint_path"],
                "checkpoint_sha256": source["checkpoint_sha256"],
            }
        )
    return output


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "training_cost_audit_v4.yaml")
    parser.add_argument("--audit-run-id", default="training_cost_formal_v4")
    parser.add_argument("--run-id", default="training_replay_formal_v4")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--audit-root", type=Path, help="Optional existing audit root when pilot outputs are written elsewhere.")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-in-memory-replay", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--measured-blocks", type=int)
    parser.add_argument("--updates-per-block", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output_root = args.output_root or _resolve(config["output"]["root"])
    audit_root = args.audit_root or output_root
    audit_path = audit_root / args.audit_run_id / "audit.json"
    if not audit_path.is_file():
        print(f"training audit is missing: {audit_path}", file=sys.stderr)
        return 2
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    plan = build_replay_plan(config, audit)
    if not args.execute:
        print(json.dumps(plan, indent=2, default=str))
        return 0
    if not args.confirm_in_memory_replay:
        print("--execute requires --confirm-in-memory-replay", file=sys.stderr)
        return 2

    protocol = plan["protocol"]
    warmups = int(args.warmup_updates if args.warmup_updates is not None else protocol["warmup_updates"])
    blocks = int(args.measured_blocks if args.measured_blocks is not None else protocol["measured_blocks"])
    per_block = int(args.updates_per_block if args.updates_per_block is not None else protocol["updates_per_block"])
    measured = blocks * per_block
    formal = not args.pilot
    if formal and (warmups != 20 or blocks != 10 or per_block != 10 or measured != 100):
        print("formal replay requires exactly 20 warmups and 10×10 measured updates", file=sys.stderr)
        return 2
    if len(plan["rows"]) != 9:
        print(f"expected nine adopted stages, found {len(plan['rows'])}", file=sys.stderr)
        return 2
    invalid_updates = [row["stage_id"] for row in plan["rows"] if row.get("update_count_status") != "explicit_consistent" or not row.get("update_count")]
    if invalid_updates:
        print(f"stages lack explicit consistent update counts: {invalid_updates}", file=sys.stderr)
        return 2

    index = gpu_index(args.device)
    before = assert_clean_gpu(index, allow_current=False)
    run_dir = output_root / args.run_id
    if run_dir.exists():
        print(f"refusing to overwrite existing replay run: {run_dir}", file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True)
    (run_dir / "gpu_state_before.txt").write_text(gpu_state(index), encoding="utf-8")
    tolerance = float(config["promotion"]["replay_validation_tolerance_fraction"])
    post_config = load_config()
    method_configs = {str(row["name"]): row for row in post_config["methods"]}
    stage_summaries: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    replay_environment = environment(args.device)
    replay_environment.pop("batch_size", None)
    replay_environment["batch_policy"] = "adopted canonical batch size recorded per stage"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "formal": formal,
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "metric": {"name": "training_update_time_ms", "unit": "ms/update"},
        "metric_name": "training_update_time_ms",
        "metric_label": "Canonical training update time (ms/update)",
        "basis": "median synchronized canonical training update at each adopted stage batch/query configuration",
        "historical_wall_clock_used": False,
        "filesystem_mtime_used": False,
        "audit_source": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "environment": replay_environment,
        "gpu_clean_before": not before,
        "protocol": {
            "warmup_updates": warmups,
            "measured_blocks": blocks,
            "updates_per_block": per_block,
            "measured_updates": measured,
            "batch_policy": "adopted canonical batch size for every stage",
            "data_policy": "one preloaded canonical training batch; no HDF5 load or host transfer inside timed boundary",
            "timing_boundary": "synchronized_update_core_preloaded_batch",
            "included": ["device_side_batch_preparation", "forward", "loss", "backward", "gradient_clipping", "optimizer_step", "ema_update_if_native"],
            "excluded": ["dataset_IO", "dataloader_workers", "host_to_device_transfer", "validation", "scheduler_epoch_step", "logging", "checkpointing"],
        },
        "safety": {
            "execution_mode": "explicit_confirmed_in_memory_replay",
            "checkpoint_write": False,
            "archive_mutation": False,
            "filesystem_mtime_used": False,
        },
    }
    atomic_json(run_dir / "manifest.json", manifest)
    try:
        for ordinal, stage in enumerate(plan["rows"]):
            print(f"[replay] {stage['method']} / {stage['stage_name']}", flush=True)
            try:
                summary, repeats = _benchmark_stage(
                    stage,
                    method_configs[str(stage["method"])],
                    torch.device(args.device),
                    warmups=warmups,
                    blocks=blocks,
                    updates_per_block=per_block,
                    tolerance=tolerance,
                    seed=20260831 + ordinal * 1000,
                )
            except torch.cuda.OutOfMemoryError as exc:
                checkpoint = _resolve(str(stage["checkpoint_path"]))
                config_path = _resolve(str(stage["config_path"]))
                training_cfg = stage.get("training_config") or {}
                summary = {
                    "method": stage["method"],
                    "stage_ordinal": stage["stage_ordinal"],
                    "stage_id": stage["stage_id"],
                    "stage_name": stage["stage_name"],
                    "stage_role": stage["stage_role"],
                    "status": "unavailable",
                    "failure_type": "cuda_oom",
                    "failure_reason": f"canonical adopted-batch replay exceeded GPU memory: {exc}",
                    "checkpoint_path": stage["checkpoint_path"],
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "config_path": stage["config_path"],
                    "config_sha256": sha256_file(config_path) if config_path.is_file() else "",
                    "update_count": int(stage["update_count"]),
                    "warmup_updates": warmups,
                    "measured_updates": 0,
                    "measured_blocks": 0,
                    "updates_per_block": per_block,
                    "batch_size": training_cfg.get("batch_size"),
                    "adopted_batch_size": training_cfg.get("batch_size"),
                }
                repeats = []
                gc.collect()
                torch.cuda.empty_cache()
            stage_summaries.append(summary)
            update_rows.extend(repeats)
            write_csv(run_dir / "training_stage_summary.csv", stage_summaries)
            if update_rows:
                write_csv(run_dir / "training_update_repeats.csv", update_rows)
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        atomic_json(run_dir / "manifest.json", manifest)
        raise

    after = assert_clean_gpu(index, allow_current=True)
    (run_dir / "gpu_state_after.txt").write_text(gpu_state(index), encoding="utf-8")
    method_rows = _method_summary(stage_summaries)
    write_csv(run_dir / "training_cost_summary.csv", method_rows)
    unchanged = all(
        sha256_file(_resolve(str(row["checkpoint_path"]))) == str(row["checkpoint_sha256"])
        for row in stage_summaries
    )
    successful_stages = [row for row in stage_summaries if row["status"] == "ok"]
    all_attempts_resolved = all(row["status"] in {"ok", "unavailable"} for row in stage_summaries)
    stage_stable = bool(successful_stages) and all(
        float(row["stability_delta_fraction"]) <= tolerance for row in successful_stages
    )
    ok_methods = [row["method"] for row in method_rows if row["status"] == "ok"]
    latent_unavailable = next(row for row in method_rows if row["method"] == "Latent FM")["status"] == "unavailable"
    unavailable_have_reasons = all(
        bool(str(row.get("unavailable_reason", "")).strip())
        for row in method_rows
        if row["status"] != "ok"
    )
    method_gate = bool(ok_methods) and latent_unavailable and unavailable_have_reasons
    validated = formal and stage_stable and all_attempts_resolved and method_gate and unchanged and measured == 100 and not before and len(after) <= 1
    qa = {
        "status": "pass" if validated else "fail",
        "formal_protocol_exact": formal and warmups == 20 and blocks == 10 and per_block == 10,
        "nine_required_stages": len(stage_summaries) == 9,
        "eight_method_rows": len(method_rows) == 8,
        "all_successful_stage_updates_finite": all(math.isfinite(float(row["update_time_median_ms"])) for row in successful_stages),
        "all_successful_stage_stability_gates_pass": stage_stable,
        "all_stage_attempts_resolved": all_attempts_resolved,
        "promoted_methods": ok_methods,
        "unavailable_methods_have_reasons": unavailable_have_reasons,
        "latent_multistage_method_explicitly_unavailable": latent_unavailable,
        "checkpoint_hashes_unchanged": unchanged,
        "historical_wall_clock_unused": True,
        "filesystem_mtime_unused": True,
        "checkpoint_write_forbidden": True,
        "archive_mutation_forbidden": True,
        "gpu_clean_before": not before,
        "gpu_clean_after": len(after) <= 1,
    }
    manifest.update(
        status="complete" if validated else "failed",
        promotion_gate={
            "validated": validated,
            "name": "first_five_vs_last_five_block_medians",
            "tolerance_fraction": tolerance,
            "all_successful_stage_stability_gates_pass": stage_stable,
            "direct_metric_method_gate": method_gate,
        },
        stage_count=len(stage_summaries),
        method_count=len(method_rows),
        gpu_clean_after=len(after) <= 1,
        checkpoint_identities=[
            {
                "method": row["method"],
                "stage_id": row["stage_id"],
                "path": row["checkpoint_path"],
                "sha256": row["checkpoint_sha256"],
            }
            for row in stage_summaries
        ],
    )
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(run_dir / "qa.json", qa)
    print(json.dumps({"output_dir": str(run_dir), "qa": qa}, indent=2))
    return 0 if validated else 2


if __name__ == "__main__":
    raise SystemExit(main())

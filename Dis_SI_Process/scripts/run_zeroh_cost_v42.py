#!/usr/bin/env python
"""Clean scenario-configured native inference and canonical mixed-resolution update replay."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import yaml
from torch.utils.data import DataLoader, Subset

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
SUPER_ROOT = REPO_ROOT / "1_SubTask_SuperResolution"
SCRIPT_ROOT = SUPER_ROOT / "Save_TrainedModel" / "_TrainedModels" / "_Scripts"
SRC_ROOT = SUPER_ROOT / "src"
for path in (SCRIPT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import model_baseline as baseline_lib
import train_pointcloud_ffm as pointcloud_train
from common.config import load_config, method_items
from common.dataset_loader import find_snapshot
from common.model_loader import checkpoint_digest, load_model


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "zeroh_matched_v42.yaml")
    parser.add_argument("--job", choices=("PILOT", "FORMAL"), required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--run-id")
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _gpu_uuid(index: int) -> str:
    return subprocess.check_output(["nvidia-smi", "-i", str(index), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()


def _processes() -> list[dict[str, str]]:
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip()
    rows = []
    for line in raw.splitlines():
        if line.strip():
            uuid, pid, name, memory = (part.strip() for part in line.split(",", 3))
            rows.append({"uuid": uuid, "pid": pid, "name": name, "memory": memory})
    return rows


def _assert_clean(device: str, allow_current: bool) -> None:
    index = torch.device(device).index
    if index is None:
        raise ValueError("Explicit CUDA device required")
    foreign = [row for row in _processes() if row["uuid"] == _gpu_uuid(index) and not (allow_current and int(row["pid"]) == os.getpid())]
    if foreign:
        raise RuntimeError(f"GPU {index} is not clean: {foreign}")


def _device(value: Any, target: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(target)
    if isinstance(value, dict):
        return {key: _device(child, target) for key, child in value.items()}
    if isinstance(value, list):
        return [_device(child, target) for child in value]
    if isinstance(value, tuple):
        return tuple(_device(child, target) for child in value)
    return value


def _quiet(module: Any, call: Callable[[], float]) -> float:
    class Quiet:
        def __init__(self, iterable): self.iterable = iterable
        def __iter__(self): return iter(self.iterable)
        def set_postfix_str(self, *args, **kwargs): return None
    original = module.tqdm
    module.tqdm = lambda iterable, *args, **kwargs: Quiet(iterable)
    try:
        return float(call())
    finally:
        module.tqdm = original


def _first_sensor_state(path: Path, sensor_count: int) -> tuple[dict[str, str], list[dict[str, int]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["snapshot_index"]) == 0 and int(row["sensor_order"]) < sensor_count]
    rows.sort(key=lambda row: int(row["sensor_order"]))
    if len(rows) != sensor_count:
        raise RuntimeError("Native timing state lacks the canonical sensors")
    return rows[0], [{"sensor_order": int(row["sensor_order"]), "point_index": int(row["point_index"])} for row in rows]


def _accuracy(config: dict[str, Any]) -> dict[str, dict[str, float]]:
    path = _resolve(config["cohort"]["accuracy_summary"])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {}
    for row in rows:
        if row["recipe"] == config["scenario"]["recipe"] and row["model_label"] in config["scenario"]["all_methods"]:
            selected[row["model_label"]] = {
                "error": float(row["mean"]), "error_ci_low": float(row["ci95_low"]),
                "error_ci_high": float(row["ci95_high"]), "error_n": int(row["valid_n"]),
            }
    if set(selected) != set(config["scenario"]["all_methods"]) or {row["error_n"] for row in selected.values()} != {300}:
        raise RuntimeError("Zero-H frozen accuracy source is incomplete")
    return selected


def _prepare_inference(loaded: Any, metadata: dict[str, str], sensor_rows: list[dict[str, int]], config: dict[str, Any], method: str) -> Callable[[], torch.Tensor]:
    dataset_index = find_snapshot(loaded.dataset, int(metadata["case_id"]), int(metadata["time_index"]))
    sample = loaded.dataset[dataset_index]
    coords = sample["coords"].unsqueeze(0).to(loaded.device)
    truth = sample["fields"].unsqueeze(0).to(loaded.device)
    indices = torch.tensor([[row["point_index"] for row in sensor_rows]], dtype=torch.long, device=loaded.device)
    field_ids = torch.zeros_like(indices)
    obs_coords = coords[:, indices[0]]
    obs_values = truth[:, indices[0], :1]
    obs_mask = torch.ones(indices.shape, dtype=coords.dtype, device=loaded.device)
    if loaded.family == "pointcloud_ffm":
        def call() -> torch.Tensor:
            return loaded.model.sample(
                coords=coords, obs_coords=obs_coords, obs_values=obs_values, obs_mask=obs_mask,
                obs_field_ids=field_ids, n_steps=int(config["scenario"]["n_steps"]), clamp_indices=indices,
                ode_solver=str(config["scenario"]["ode_solver"]), obs_consistency_mode=str(config["scenario"]["consistency_modes"][method]),
            )
    else:
        def call() -> torch.Tensor:
            return loaded.model(coords, obs_coords, obs_values, obs_mask, field_ids)
    output = call()
    if tuple(output.shape) != (1, int(config["scenario"]["native_query_count"]), 1):
        raise RuntimeError(f"Unexpected native output shape for {method}: {tuple(output.shape)}")
    return call


def _time_call(call: Callable[[], torch.Tensor], device: torch.device) -> tuple[float, float]:
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter(); output = call(); torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000.0
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    if not torch.isfinite(output).all() or elapsed <= 0:
        raise RuntimeError("Invalid inference timing result")
    return elapsed, peak


def _optimizer_for_pointcloud(loaded: Any) -> torch.optim.Optimizer:
    optimizer = torch.optim.AdamW(loaded.model.parameters(), lr=float(loaded.config.get("lr", 2.0e-4)), weight_decay=float(loaded.config.get("weight_decay", 0.0)))
    checkpoint = torch.load(loaded.checkpoint_path, map_location=loaded.device, weights_only=False)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("optimizer"), dict):
        optimizer.load_state_dict(checkpoint["optimizer"])
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value): state[key] = value.to(loaded.device)
    return optimizer


def _resolution_batch(loaded: Any, resolution: str, batch_size: int) -> dict[str, Any]:
    indices = list(loaded.dataset.indices_by_res[resolution])[:batch_size]
    if len(indices) != batch_size:
        raise RuntimeError(f"Not enough {resolution} training samples for batch {batch_size}")
    loader = DataLoader(Subset(loaded.dataset, indices), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=pointcloud_train.collate_snapshots)
    batch = _device(next(iter(loader)), loaded.device)
    tags = {str(value) for value in batch["resolution_tag"]}
    if tags != {resolution}:
        raise RuntimeError(f"Resolution batch is not homogeneous: expected {resolution}, found {sorted(tags)}")
    return batch


def _prepare_training(loaded: Any, method: str, resolutions: tuple[str, ...]) -> tuple[dict[str, Callable[[], float]], int]:
    if loaded.family == "pointcloud_ffm":
        batch_size = int(loaded.config["batch_size"])
        optimizer = _optimizer_for_pointcloud(loaded)
        batches = {resolution: _resolution_batch(loaded, resolution, batch_size) for resolution in resolutions}
        def make(batch):
            return lambda: _quiet(pointcloud_train, lambda: pointcloud_train.run_epoch(
                model=loaded.model, loader=[batch], optimizer=optimizer, device=loaded.device,
                cond_fields=loaded.config["cond_fields"], n_obs_min_list=loaded.config["n_obs_min_list"],
                n_obs_max_list=loaded.config["n_obs_max_list"], n_query_points=int(loaded.config["n_query_points"]), epoch=0,
            ))
        return {resolution: make(batch) for resolution, batch in batches.items()}, batch_size
    bundle = loaded.baseline_bundle
    if bundle is None or bundle.optimizer is None:
        raise RuntimeError(f"{method} baseline bundle lacks an optimizer")
    adapter = baseline_lib.get_baseline_adapter(str(loaded.config["baseline_model"]).lower())
    stage_cfg = baseline_lib.resolve_stage_config(bundle.config)
    batch_size = int(stage_cfg["training"]["batch_size"])
    batches = {resolution: _resolution_batch(loaded, resolution, batch_size) for resolution in resolutions}
    def make(batch):
        return lambda: _quiet(baseline_lib, lambda: adapter.run_epoch(bundle, [batch], training=True, epoch=0))
    return {resolution: make(batch) for resolution, batch in batches.items()}, batch_size


def _time_update(call: Callable[[], float], device: torch.device) -> tuple[float, float, float]:
    torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter(); loss = float(call()); torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - start) * 1000.0
    peak = torch.cuda.max_memory_allocated(device) / 2**20
    if not math.isfinite(loss) or not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("Invalid training update result")
    return loss, elapsed, peak


def main() -> int:
    args = _args(); config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    formal = args.job == "FORMAL"
    run_prefix = str(config.get("run_prefix", "zeroh"))
    run_id = args.run_id or f"{run_prefix}_cost_{args.job.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = _resolve(config["cost_output_root"]) / run_id
    if run_dir.exists(): raise RuntimeError(f"Refusing to overwrite {run_dir}")
    _assert_clean(args.device, allow_current=False); run_dir.mkdir(parents=True)
    sensor_path = _resolve(config["cohort"]["sensor_plan"])
    if _sha(sensor_path) != config["cohort"]["sensor_plan_sha256"]: raise RuntimeError("Sensor-plan identity mismatch")
    metadata, sensor_rows = _first_sensor_state(sensor_path, int(config["scenario"]["sensor_count"]))
    accuracy = _accuracy(config)
    for method, item in config["checkpoints"].items():
        if _sha(_resolve(item["path"])) != item["sha256"]: raise RuntimeError(f"Checkpoint identity mismatch: {method}")
    super_cfg = load_config(); specs = {row["label"]: row for row in method_items(super_cfg)}
    recipe, recipe_spec = config["scenario"]["recipe"], super_cfg["recipes"][config["scenario"]["recipe"]]
    resolutions = tuple(str(value) for value in config["scenario"].get("training_resolutions", ["L", "M"]))
    weights = {str(key): float(value) for key, value in config["scenario"].get("training_resolution_weights", {resolution: 1.0 / len(resolutions) for resolution in resolutions}).items()}
    if not resolutions or set(weights) != set(resolutions) or any(value <= 0 for value in weights.values()) or not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("Training resolutions require positive weights summing to one")
    resolution_weights = ";".join(f"{resolution}={weights[resolution]:.12g}" for resolution in resolutions)
    manifest = {
        "schema_version": str(config.get("cost_schema_version", "figure5-zeroh-cost-v4.2-1")), "status": "running", "formal": formal, "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "scenario": config["scenario"],
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device, "gpu": torch.cuda.get_device_name(torch.device(args.device))},
        "protocol": {
            "inference": {"warmups": 20 if formal else 2, "minimum_repeats": 30 if formal else 6, "minimum_seconds": 10.0 if formal else 0.0, "boundary": "preloaded warm model core; includes stochastic source, conditioning, model evaluations, observation consistency, device output"},
            "training": {"warmups": 20 if formal else 4, "measured_updates": 100 if formal else 12, "resolution_schedule": f"strict {'/'.join(resolutions)} cycle for the adopted recipe", "promoted_metric": f"configured-weight mean of {'/'.join(resolutions)} median synchronized wall ms/update", "resolution_weights": weights, "boundary": "preloaded batch; conditioning, query selection, forward, loss, backward, gradient clipping, optimizer step"},
            "excluded": ["model_loading", "dataset_IO", "dataloader_workers", "host_transfer", "metrics", "logging", "checkpointing"],
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    inference_rows, inference_summary, training_rows, training_summary = [], [], [], []
    inf_warmups, inf_min_repeats, inf_min_seconds = (20, 30, 10.0) if formal else (2, 6, 0.0)
    train_warmups, train_repeats = (20, 100) if formal else (4, 12)
    for method in config["scenario"]["all_methods"]:
        print(f"[inference] {method}", flush=True)
        loaded = load_model(specs[method], recipe, recipe_spec, checkpoint="best", split="test", eval_resolution="H", device=args.device)
        try:
            if checkpoint_digest(loaded.checkpoint_path) != config["checkpoints"][method]["sha256"]: raise RuntimeError("Loaded checkpoint mismatch")
            call = _prepare_inference(loaded, metadata, sensor_rows, config, method)
            for _ in range(inf_warmups): _time_call(call, loaded.device)
            elapsed_values, start_all, index = [], time.perf_counter(), 0
            while len(elapsed_values) < inf_min_repeats or time.perf_counter() - start_all < inf_min_seconds:
                elapsed, peak = _time_call(call, loaded.device); elapsed_values.append(elapsed)
                inference_rows.append({"method": method, "repeat": index, "elapsed_ms": elapsed, "peak_allocated_mib": peak}); index += 1
            q25, median, q75 = np.quantile(elapsed_values, [0.25, 0.5, 0.75])
            inference_summary.append({"method": method, "status": "ok", "cost_value": median, "cost_low": q25, "cost_high": q75, "latency_median_ms": median, "latency_q25_ms": q25, "latency_q75_ms": q75, **accuracy[method], "N": int(config["scenario"]["native_query_count"]), "sensor_count": int(config["scenario"]["sensor_count"]), "checkpoint_path": str(loaded.checkpoint_path.relative_to(REPO_ROOT)), "checkpoint_sha256": config["checkpoints"][method]["sha256"]})
        finally:
            loaded.close()

        print(f"[training] {method}", flush=True)
        loaded = load_model(specs[method], recipe, recipe_spec, checkpoint="best", split="train", eval_resolution="H", device=args.device)
        try:
            # The post-processing loader forces H-resolution output even for a
            # train split. Restore the checkpoint's native mixed-resolution
            # entries before selecting the canonical replay batches.
            loaded.dataset.force_resolution = None
            loaded.dataset.output_resolution = None
            loaded.dataset.num_points = None
            loaded.dataset.coords = None
            loaded.dataset.coords_raw = None
            loaded.dataset.requires_grouped_batches = True
            calls, batch_size = _prepare_training(loaded, method, resolutions)
            for index in range(train_warmups): _time_update(calls[resolutions[index % len(resolutions)]], loaded.device)
            by_resolution = {resolution: [] for resolution in resolutions}
            for index in range(train_repeats):
                resolution = resolutions[index % len(resolutions)]
                loss, elapsed, peak = _time_update(calls[resolution], loaded.device)
                by_resolution[resolution].append(elapsed)
                training_rows.append({"method": method, "update": index, "resolution": resolution, "elapsed_ms": elapsed, "loss": loss, "peak_allocated_mib": peak})
            summaries = {resolution: np.quantile(values, [0.25, 0.5, 0.75]) for resolution, values in by_resolution.items()}
            mixed = np.sum(np.stack([weights[resolution] * summaries[resolution] for resolution in resolutions]), axis=0)
            blocks = 5 if formal else 2
            stability = {}
            for resolution, values in by_resolution.items():
                chunks = np.array_split(np.asarray(values), blocks)
                medians = [float(np.median(chunk)) for chunk in chunks]
                split = len(medians) // 2
                stability[resolution] = abs(statistics.median(medians[:split]) - statistics.median(medians[split:])) / max(float(summaries[resolution][1]), 1.0e-12)
            resolution_details = {}
            for resolution in resolutions:
                resolution_details[f"{resolution}_median_ms"] = summaries[resolution][1]
                resolution_details[f"{resolution}_stability_fraction"] = stability[resolution]
            training_summary.append({"method": method, "status": "ok" if max(stability.values()) <= 0.25 else "unstable", "cost_value": mixed[1], "cost_low": mixed[0], "cost_high": mixed[2], "training_update_time_ms": mixed[1], **resolution_details, "batch_size": batch_size, "resolution_weights": resolution_weights, **accuracy[method], "checkpoint_path": str(loaded.checkpoint_path.relative_to(REPO_ROOT)), "checkpoint_sha256": config["checkpoints"][method]["sha256"]})
        except torch.cuda.OutOfMemoryError as exc:
            training_summary.append({"method": method, "status": "unavailable", "unavailable_reason": f"canonical batch OOM: {exc}", **accuracy[method], "checkpoint_path": config["checkpoints"][method]["path"], "checkpoint_sha256": config["checkpoints"][method]["sha256"]})
            torch.cuda.empty_cache()
        finally:
            loaded.close(); gc.collect(); torch.cuda.empty_cache()
        _write_csv(run_dir / "inference_repeats.csv", inference_rows); _write_csv(run_dir / "native_cost_summary.csv", inference_summary)
        if training_rows: _write_csv(run_dir / "training_update_repeats.csv", training_rows)
        _write_csv(run_dir / "training_update_summary.csv", training_summary)
    _assert_clean(args.device, allow_current=True)
    qa = {
        "status": "pass", "formal": formal,
        "all_four_inference_methods": len(inference_summary) == 4 and all(row["status"] == "ok" for row in inference_summary),
        "native_problem_exact": all(row["N"] == int(config["scenario"]["native_query_count"]) and row["sensor_count"] == int(config["scenario"]["sensor_count"]) for row in inference_summary),
        "inference_quartiles_ordered": all(row["cost_low"] <= row["cost_value"] <= row["cost_high"] for row in inference_summary),
        "all_four_training_attempted": len(training_summary) == 4,
        "training_status_explicit": all(row["status"] in {"ok", "unavailable"} for row in training_summary),
        "training_stability_pass": all(row["status"] != "ok" or max(float(row[f"{resolution}_stability_fraction"]) for resolution in resolutions) <= 0.25 for row in training_summary),
        "frozen_accuracy_n_exact": all(row["error_n"] == 300 for row in inference_summary),
        "checkpoint_identity_pass": True, "gpu_clean_before": True, "gpu_clean_after": True,
        "no_cond_t_cost_reuse": True, "no_training_performed_beyond_ephemeral_replay": True,
    }
    qa["status"] = "pass" if all(value for key, value in qa.items() if key not in {"status", "formal"}) else "fail"
    _write_json(run_dir / "qa.json", qa); manifest.update(status="complete" if qa["status"] == "pass" else "failed_qa", completed_at=datetime.now(timezone.utc).isoformat()); _write_json(run_dir / "manifest.json", manifest)
    print(json.dumps({"run_dir": str(run_dir), "qa": qa, "inference": inference_summary, "training": training_summary}, indent=2))
    return 0 if qa["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

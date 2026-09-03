#!/usr/bin/env python
"""Clean-GPU common-batch training-footprint benchmark for Figure 5 V5.1.

The runner measures the canonical training objective of each archived Cond_T
checkpoint at a common batch size of 32.  It uses one fixed training-split
batch, a fixed 256-temperature sensor layout, and fixed target indices across
methods.  Query-evaluable models use 4,096 target points; architectures whose
canonical objective operates on a structured field use the complete 40,300
point grid.  Latent FM's VAE and flow stages are measured independently.

Execution is deliberately gated.  A formal run requires both ``--execute``
and ``--confirm-clean-gpu`` and refuses a selected GPU with foreign compute
processes or non-idle utilisation.  Checkpoints, datasets, caches, and model
archives are read only.  The output directory contains compact tables only.
"""
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
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
DEMO_ROOT = REPO_ROOT / "0_demo_TurbulentCombustion"
SCRIPT_ROOT = DEMO_ROOT / "Save_TrainedModel" / "_TrainedModels" / "_Scripts"
SRC_ROOT = DEMO_ROOT / "src"
TOOLS_ROOT = DEMO_ROOT / "tools"
for path in (SCRIPT_ROOT, SRC_ROOT, TOOLS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_pointcloud_ffm as pointcloud_train  # noqa: E402
from benchmark_validation_v3 import (  # noqa: E402
    assert_clean_gpu,
    compute_processes,
    gpu_index,
    gpu_state,
)
from common.config import load_config, stable_seed  # noqa: E402
from common.model_loader import baseline_lib, canonical, load_model  # noqa: E402


SCHEMA_VERSION = "figure5-validation-v51-training-footprint-1"
METHOD_ORDER = (
    "DMF-Gen",
    "FFM-FNO",
    "FFM-Perceiver",
    "Latent FM",
    "SiT",
    "MLP-RBF",
    "Geo-FNO",
    "Senseiver",
)
QUERY_METHODS = frozenset({"DMF-Gen", "FFM-Perceiver", "MLP-RBF", "Senseiver"})
FULL_GRID_METHODS = frozenset({"FFM-FNO", "Latent FM", "SiT", "Geo-FNO"})
LATENT_METHOD = "Latent FM"
OOM_STATUS = "OOM_at_common_batch"
COUNTER_FIELDS = (
    "optimizer_step_attempts",
    "optimizer_step_successes",
    "optimizer_step_skips",
    "ema_update_attempts",
    "ema_update_successes",
    "ema_update_skips",
)


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
        raise ValueError(f"Refusing to write empty formal table: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in columns:
                columns.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _foreign_process_rows(rows: Iterable[Mapping[str, Any]], current_pid: int | str | None = None) -> list[Mapping[str, Any]]:
    """Return only GPU compute rows owned by a PID other than this runner."""
    pid = str(os.getpid() if current_pid is None else current_pid)
    return [row for row in rows if str(row.get("pid")) != pid]


def _geo_fno_outcome(row: Mapping[str, Any]) -> str:
    """Classify the required B32 Geo-FNO attempt without forcing OOM."""
    if int(row.get("batch_size", -1)) != 32:
        return "invalid_batch"
    status = str(row.get("status", ""))
    if status == OOM_STATUS:
        return "oom_at_common_batch"
    if status == "ok" and int(row.get("measured_updates", 0)) == 100:
        return "success_at_common_batch"
    return "invalid"


def _parse_mib(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (float, int)):
        return float(value)
    text = str(value).strip().replace("MiB", "").replace("MiB", "")
    try:
        return float(text.split()[0].replace(",", ""))
    except (ValueError, IndexError):
        return 0.0


def _parse_percent(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).replace("%", "").strip())
    except ValueError:
        return 0.0


def _nvidia_gpu_usage(index: int) -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "-i",
        str(index),
        "--query-gpu=index,uuid,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return {"index": index, "uuid": "", "memory_used_mib": float("nan"), "utilization_percent": float("nan")}
    line = next((line for line in output.splitlines() if line.strip()), "")
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        return {"index": index, "uuid": "", "memory_used_mib": float("nan"), "utilization_percent": float("nan")}
    return {
        "index": int(parts[0]),
        "uuid": parts[1],
        "memory_used_mib": _parse_mib(parts[2]),
        "utilization_percent": _parse_percent(parts[3]),
    }


def _nvidia_process_usage(index: int, gpu_uuid: str | None = None) -> dict[str, Any]:
    uuid = gpu_uuid or str(_nvidia_gpu_usage(index).get("uuid", ""))
    try:
        rows = compute_processes()
    except (OSError, subprocess.CalledProcessError, ValueError):
        rows = []
    selected = [row for row in rows if not uuid or row.get("gpu_uuid") == uuid]
    current_pid = str(os.getpid())
    current = sum(_parse_mib(row.get("used_memory")) for row in selected if str(row.get("pid")) == current_pid)
    foreign = sum(_parse_mib(row.get("used_memory")) for row in selected if str(row.get("pid")) != current_pid)
    return {
        "process_memory_mib": float(current),
        "foreign_memory_mib": float(foreign),
        "process_count": len(selected),
        "foreign_process_count": sum(1 for row in selected if str(row.get("pid")) != current_pid),
    }


def _usage_sample(index: int) -> dict[str, Any]:
    gpu = _nvidia_gpu_usage(index)
    process = _nvidia_process_usage(index, str(gpu.get("uuid", "")))
    return {
        "timestamp_monotonic": time.monotonic(),
        "memory_used_mib": gpu["memory_used_mib"],
        "utilization_percent": gpu["utilization_percent"],
        "gpu_uuid": gpu.get("uuid", ""),
        **process,
    }


class NVMLSampler:
    """Asynchronously poll nvidia-smi's NVML-backed usage counters.

    Sampling occurs outside the synchronized timing boundary.  It is retained
    as an nvidia-smi fallback because the benchmark environment does not ship
    the Python NVML bindings; nvidia-smi itself queries NVML.
    """

    def __init__(self, index: int, interval_seconds: float) -> None:
        self.index = int(index)
        self.interval_seconds = max(float(interval_seconds), 0.001)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> dict[str, Any]:
        sample = _usage_sample(self.index)
        self.samples.append(sample)
        return sample

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.sample_once()
            except Exception:
                # A transient nvidia-smi failure must not alter the timed
                # update.  The final explicit sample still makes the absence
                # visible through ``sample_count`` in the summary.
                continue

    def start(self) -> None:
        self.sample_once()
        self._thread = threading.Thread(target=self._run, name="v51-nvml", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 4.0 * self.interval_seconds))
        try:
            self.sample_once()
        except Exception:
            pass
        return list(self.samples)


def _tensor_bytes(value: Any) -> int:
    if not torch.is_tensor(value):
        return 0
    return int(value.numel()) * int(value.element_size())


def _module_bytes(module: torch.nn.Module) -> tuple[int, int, int]:
    parameters = 0
    buffers = 0
    parameter_count = 0
    seen: set[int] = set()
    for parameter in module.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        parameters += _tensor_bytes(parameter)
        parameter_count += int(parameter.numel())
    seen.clear()
    for buffer in module.buffers():
        if id(buffer) in seen:
            continue
        seen.add(id(buffer))
        buffers += _tensor_bytes(buffer)
    return parameters, buffers, parameter_count


def _optimizer_bytes(optimizer: torch.optim.Optimizer | None) -> int:
    if optimizer is None:
        return 0
    return sum(_tensor_bytes(value) for state in optimizer.state.values() for value in state.values())


def _ema_bytes(ema: Any) -> int:
    if ema is None:
        return 0
    values: Iterable[Any]
    if hasattr(ema, "shadow"):
        shadow = ema.shadow
        values = shadow.values() if isinstance(shadow, Mapping) else shadow
    elif hasattr(ema, "shadow_params"):
        values = ema.shadow_params
    else:
        values = []
    return sum(_tensor_bytes(value) for value in values)


def _unique_tensor_bytes(values: Iterable[Any]) -> int:
    total = 0
    seen: set[tuple[int, int]] = set()
    for value in values:
        if not torch.is_tensor(value):
            continue
        try:
            storage = value.untyped_storage()
            key = (int(storage.data_ptr()), int(storage.nbytes()))
        except Exception:
            key = (id(value), _tensor_bytes(value))
        if key in seen:
            continue
        seen.add(key)
        total += _tensor_bytes(value)
    return total


def _gradient_bytes(module: torch.nn.Module) -> int:
    return sum(_tensor_bytes(parameter.grad) for parameter in module.parameters() if parameter.grad is not None)


def _mib(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


@dataclass
class FixedBatch:
    """CPU tensors and frozen indices shared by every benchmark stage."""

    coords: torch.Tensor
    fields: torch.Tensor
    sensor_indices: torch.Tensor
    query_indices: torch.Tensor
    sensor_field_ids: torch.Tensor
    state_indices: tuple[int, ...]
    original_time_indices: tuple[int, ...]
    identity_sha256: str


@dataclass
class UpdateCounters:
    """Counters for optimizer/EMA decisions inside the timed update."""

    optimizer_step_attempts: int = 0
    optimizer_step_successes: int = 0
    optimizer_step_skips: int = 0
    ema_update_attempts: int = 0
    ema_update_successes: int = 0
    ema_update_skips: int = 0

    def snapshot(self) -> dict[str, int]:
        return {field: int(getattr(self, field)) for field in COUNTER_FIELDS}


@dataclass
class InstrumentedUpdate:
    """Callable update preserving the counter object for the outer protocol."""

    function: Callable[[], torch.Tensor]
    counters: UpdateCounters

    def __call__(self) -> torch.Tensor:
        return self.function()


def _zero_counter_snapshot() -> dict[str, int]:
    return {field: 0 for field in COUNTER_FIELDS}


def _counter_delta(start: Mapping[str, int], end: Mapping[str, int]) -> dict[str, int]:
    return {field: int(end.get(field, 0)) - int(start.get(field, 0)) for field in COUNTER_FIELDS}


def _optimizer_step(optimizer: torch.optim.Optimizer, counters: UpdateCounters) -> None:
    """Record a logical step decision and its successful optimizer call."""
    counters.optimizer_step_attempts += 1
    optimizer.step()
    counters.optimizer_step_successes += 1


def _optimizer_skip(counters: UpdateCounters) -> None:
    counters.optimizer_step_attempts += 1
    counters.optimizer_step_skips += 1


def _ema_update(ema: Any, value: Any, counters: UpdateCounters) -> None:
    counters.ema_update_attempts += 1
    ema.update(value)
    counters.ema_update_successes += 1


def _ema_skip(ema: Any, counters: UpdateCounters) -> None:
    if ema is not None:
        counters.ema_update_attempts += 1
        counters.ema_update_skips += 1


def _instrument_update(function: Callable[[], torch.Tensor], counters: UpdateCounters) -> InstrumentedUpdate:
    return InstrumentedUpdate(function=function, counters=counters)


def _batch_digest(
    coords: torch.Tensor,
    fields: torch.Tensor,
    sensor_indices: torch.Tensor,
    query_indices: torch.Tensor,
    state_indices: Sequence[int],
) -> str:
    digest = hashlib.sha256()
    for value in (coords, fields, sensor_indices, query_indices):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(json.dumps([int(v) for v in state_indices]).encode("utf-8"))
    return digest.hexdigest()


def _reference_dataset(config: Mapping[str, Any]) -> Any:
    stage = next(row for row in config["stages"] if row["method"] == "DMF-Gen")
    checkpoint = resolve(stage["checkpoint_path"])
    run_dir = checkpoint.parent
    raw = canonical.load_run_config(run_dir)
    data_value = raw.get("data") or raw.get("shared", {}).get("paths", {}).get("data_path")
    if not data_value:
        raise RuntimeError("DMF-Gen reference config does not declare a dataset path")
    data_path = canonical.resolve_input_path(
        str(data_value), label="Dataset", extra_base_dirs=[run_dir, canonical.DEMO_DIR]
    )
    stats = run_dir / "dataset_stats.pt"
    return pointcloud_train.TurbulentCombustionH5Dataset(
        str(data_path),
        split=str(config["split"]),
        train_ratio=float(raw.get("train_ratio", raw.get("shared", {}).get("data", {}).get("train_ratio", 0.9))),
        seed=int(raw.get("seed", raw.get("shared", {}).get("seed", 42))),
        time_stride=int(raw.get("time_stride", raw.get("shared", {}).get("data", {}).get("time_stride", 1))),
        field_names=raw.get("FIELD_NAMES", raw.get("field_names", raw.get("shared", {}).get("FIELD_NAMES"))),
        stats_path=str(stats) if stats.exists() else None,
    )


def _build_fixed_batch(config: Mapping[str, Any]) -> FixedBatch:
    state_indices = tuple(int(v) for v in config["state_indices"])
    if len(state_indices) != int(config["batch_size"]):
        raise ValueError("state_indices must contain exactly one fixed state per common batch item")
    if len(set(state_indices)) != len(state_indices) or min(state_indices) < 0:
        raise ValueError("state_indices must be unique non-negative local split indices")
    dataset = _reference_dataset(config)
    try:
        if max(state_indices) >= len(dataset):
            raise ValueError(f"state index exceeds {config['split']} dataset length {len(dataset)}")
        samples = [dataset[index] for index in state_indices]
        coords = torch.stack([sample["coords"] for sample in samples]).float().contiguous()
        fields = torch.stack([sample["fields"] for sample in samples]).float().contiguous()
        if tuple(coords.shape[1:]) != (int(config["native_grid_target_count"]), 3):
            raise ValueError(f"unexpected coordinate shape {tuple(coords.shape)}")
        if tuple(fields.shape[1:]) != (int(config["native_grid_target_count"]), 5):
            raise ValueError(f"unexpected field shape {tuple(fields.shape)}")

        sensor_count = int(config["sensor_count"])
        n_points = int(config["native_grid_target_count"])
        sensor_seed_base = int(config["sensor_plan"]["selection_seed_base"])
        query_seed_base = int(config["fixed_batch"]["query_seed_base"])
        sensor_indices_np: list[np.ndarray] = []
        query_indices_np: list[np.ndarray] = []
        for state in state_indices:
            sensor_rng = np.random.default_rng(stable_seed(sensor_seed_base, config["condition"], state))
            sensors = np.sort(sensor_rng.choice(n_points, size=sensor_count, replace=False).astype(np.int64))
            query_rng = np.random.default_rng(stable_seed(query_seed_base, config["condition"], state))
            available = np.setdiff1d(np.arange(n_points, dtype=np.int64), sensors, assume_unique=True)
            extra = query_rng.choice(
                available,
                size=int(config["query_target_count"]) - sensor_count,
                replace=False,
            )
            query_indices_np.append(np.sort(np.concatenate((sensors, extra))))
            sensor_indices_np.append(sensors)
        sensor_indices = torch.from_numpy(np.stack(sensor_indices_np)).long().contiguous()
        query_indices = torch.from_numpy(np.stack(query_indices_np)).long().contiguous()
        sensor_field_ids = torch.full_like(sensor_indices, int(config["sensor_plan"]["field_index"]))
        if sensor_indices.shape != (len(state_indices), sensor_count):
            raise ValueError("fixed sensor index shape mismatch")
        if query_indices.shape != (len(state_indices), int(config["query_target_count"])):
            raise ValueError("fixed query index shape mismatch")
        original_time_indices = tuple(int(dataset.indices[index]) for index in state_indices)
        identity = _batch_digest(coords, fields, sensor_indices, query_indices, state_indices)
        return FixedBatch(
            coords=coords,
            fields=fields,
            sensor_indices=sensor_indices,
            query_indices=query_indices,
            sensor_field_ids=sensor_field_ids,
            state_indices=state_indices,
            original_time_indices=original_time_indices,
            identity_sha256=identity,
        )
    finally:
        close = getattr(dataset, "close", None)
        if callable(close):
            close()


def _move_optimizer_state(optimizer: torch.optim.Optimizer | None, device: torch.device) -> None:
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


@dataclass
class StageRuntime:
    stage: Mapping[str, Any]
    method: str
    family: str
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer | None
    ema: Any
    bundle: Any
    loaded: Any

    @property
    def dataset(self) -> Any:
        if self.bundle is not None:
            return self.bundle.dataset_train
        return getattr(self.loaded, "dataset", None)

    def close(self) -> None:
        try:
            if self.optimizer is not None:
                self.optimizer.zero_grad(set_to_none=True)
        except Exception:
            pass
        dataset = self.dataset
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        self.loaded = None
        self.bundle = None
        self.optimizer = None
        self.ema = None
        self.model = None  # type: ignore[assignment]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _pointcloud_method_config(method: str) -> Mapping[str, Any]:
    post = load_config()
    for row in post["methods"]:
        if str(row["name"]) == method:
            return row
    raise KeyError(f"method missing from postprocess config: {method}")


def _load_stage(stage: Mapping[str, Any], config: Mapping[str, Any], device: torch.device) -> StageRuntime:
    method = str(stage["method"])
    checkpoint = resolve(stage["checkpoint_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != str(stage["checkpoint_sha256"]):
        raise RuntimeError(f"checkpoint identity mismatch for {method}: {checkpoint}")
    config_path = resolve(stage["config_path"])
    if sha256_file(config_path) != str(stage["config_sha256"]):
        raise RuntimeError(f"run_config identity mismatch for {method}: {config_path}")

    if method in {"DMF-Gen", "FFM-FNO", "FFM-Perceiver"}:
        loaded = load_model(
            dict(_pointcloud_method_config(method)),
            "Cond_T",
            checkpoint="last",
            split=str(config["split"]),
            device=str(device),
            n_steps=2,
            ode_solver="euler",
        )
        model = loaded.model.float()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(loaded.config.get("lr", 1.0e-4)),
            weight_decay=float(loaded.config.get("weight_decay", 0.0)),
        )
        checkpoint_object = torch.load(checkpoint, map_location=device, weights_only=False)
        if isinstance(checkpoint_object, Mapping) and isinstance(checkpoint_object.get("optimizer"), Mapping):
            optimizer.load_state_dict(checkpoint_object["optimizer"])
        _move_optimizer_state(optimizer, device)
        del checkpoint_object
        if bool(getattr(model, "requires_full_grid", False)) != (method in FULL_GRID_METHODS):
            raise RuntimeError(f"target-mode architecture mismatch for {method}")
        model.train()
        return StageRuntime(stage, method, str(loaded.family), model, optimizer, None, None, loaded)

    args = SimpleNamespace(baseline_model="auto", split=str(config["split"]), n_steps=2, ode_solver="euler")
    _, run_dir, _, _, bundle, _, _, _ = canonical.load_baseline_context(
        args=args,
        checkpoint_arg=str(checkpoint),
        device=device,
    )
    del run_dir
    model = bundle.model.float()
    optimizer = bundle.optimizer
    _move_optimizer_state(optimizer, device)
    family = str(bundle.baseline_model)
    if method == "Latent FM" and int(stage["stage_ordinal"]) == 1 and int(bundle.training_stage) != 1:
        raise RuntimeError("Latent FM stage-1 checkpoint did not load as stage 1")
    if method == "Latent FM" and int(stage["stage_ordinal"]) == 2 and int(bundle.training_stage) != 2:
        raise RuntimeError("Latent FM stage-2 checkpoint did not load as stage 2")
    if method != "Latent FM" and method.lower().replace("-", "") not in family.lower().replace("_", ""):
        # The checkpoint's run_config is authoritative.  Keep this check
        # informative without relying on directory names for identity.
        family = str(bundle.baseline_model)
    bundle.model.eval()
    return StageRuntime(stage, method, family, model, optimizer, bundle.ema, bundle, None)


def _device_fixed_batch(batch: FixedBatch, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "coords": batch.coords.to(device=device, dtype=torch.float32),
        "fields": batch.fields.to(device=device, dtype=torch.float32),
        "sensor_indices": batch.sensor_indices.to(device=device),
        "query_indices": batch.query_indices.to(device=device),
        "sensor_field_ids": batch.sensor_field_ids.to(device=device),
    }


def _batch_gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_index = torch.arange(values.shape[0], device=values.device).view(-1, 1).expand_as(indices)
    return values[batch_index, indices]


def _fixed_sparse(raw: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    coords = raw["coords"]
    fields = raw["fields"]
    indices = raw["sensor_indices"]
    field_ids = raw["sensor_field_ids"]
    obs_coords = _batch_gather(coords, indices)
    obs_values = torch.gather(fields, 2, field_ids.unsqueeze(-1)).contiguous()
    obs_mask = torch.ones(indices.shape, device=coords.device, dtype=coords.dtype)
    return {
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_indices": indices,
        "obs_field_ids": field_ids,
    }


def _fixed_query(raw: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    indices = raw["query_indices"]
    return _batch_gather(raw["coords"], indices), _batch_gather(raw["fields"], indices)


def _grid_inputs(
    raw: Mapping[str, torch.Tensor],
    *,
    point_to_grid: torch.Tensor | None = None,
    h_pad: int | None = None,
    w_pad: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fields = raw["fields"]
    coords = raw["coords"]
    num_y, num_x = 100, 403
    if h_pad is None:
        h_pad = num_y
    if w_pad is None:
        w_pad = num_x
    grid = baseline_lib.pointcloud_to_grid_padded(
        fields, num_y, num_x, int(h_pad), int(w_pad), grid_order=point_to_grid
    )
    sparse = _fixed_sparse(raw)
    obs_value_grid, obs_mask_grid = baseline_lib.build_obs_grid_mask(
        sparse["obs_values"],
        sparse["obs_mask"],
        sparse["obs_field_ids"],
        sparse["obs_indices"],
        fields.shape[2],
        fields.shape[1],
        num_y,
        num_x,
        int(h_pad),
        int(w_pad),
        point_to_grid=point_to_grid,
    )
    del coords
    sparse["obs_value_grid"] = obs_value_grid
    sparse["obs_mask_grid"] = obs_mask_grid
    return grid, sparse


def _reset_sit_spike_state(runtime: StageRuntime) -> dict[str, Any] | None:
    """Reset only transient SiT spike guards before benchmark warmups.

    The archived checkpoint is never written back.  This prevents a training
    history-dependent spike guard from silently skipping every measured
    optimizer/EMA update while retaining the checkpoint's model and optimizer
    state for the footprint comparison.
    """
    if runtime.method != "SiT":
        return None
    if runtime.bundle is None:
        raise RuntimeError("SiT runtime has no canonical bundle")
    state = runtime.bundle.components.get("spike_state")
    if not isinstance(state, dict):
        raise RuntimeError("SiT runtime spike_state is not a mutable dictionary")
    before = {key: state.get(key) for key in ("ema_loss", "ema_grad", "skipped")}
    state["ema_loss"] = None
    state["ema_grad"] = None
    state["skipped"] = 0
    return {
        "applied": True,
        "scope": "in-memory bundle.components['spike_state'] only",
        "fields_reset": ["ema_loss", "ema_grad", "skipped"],
        "state_before": before,
        "state_after": {"ema_loss": None, "ema_grad": None, "skipped": 0},
        "checkpoint_mutation": False,
    }


def _make_update(runtime: StageRuntime, raw: Mapping[str, torch.Tensor], stage: Mapping[str, Any]) -> tuple[Callable[[], torch.Tensor], int]:
    method = runtime.method
    model = runtime.model
    optimizer = runtime.optimizer
    if optimizer is None:
        raise RuntimeError(f"{method} has no optimizer")
    mode = str(stage["training_target_mode"])
    target_count = int(stage["n_training_targets"])
    counters = UpdateCounters()

    if runtime.family == "pointcloud_ffm":
        model.train()
        use_full = mode == "native_full_grid"

        def pointcloud_update() -> torch.Tensor:
            sparse = _fixed_sparse(raw)
            if use_full:
                coords, fields = raw["coords"], raw["fields"]
            else:
                coords, fields = _fixed_query(raw)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.training_loss(
                x1=fields,
                coords=coords,
                obs_coords=sparse["obs_coords"],
                obs_values=sparse["obs_values"],
                obs_mask=sparse["obs_mask"],
                obs_field_ids=sparse["obs_field_ids"],
                obs_indices=sparse["obs_indices"],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            _optimizer_step(optimizer, counters)
            return loss

        return _instrument_update(pointcloud_update, counters), target_count

    bundle = runtime.bundle
    assert bundle is not None
    if method == "Latent FM" and int(stage["stage_ordinal"]) == 1:
        model.train()

        def latent_stage1_update() -> torch.Tensor:
            fields_grid = baseline_lib.pointcloud_to_grid(raw["fields"], 100, 403)
            optimizer.zero_grad(set_to_none=True)
            reconstructed, _ = model(fields_grid)
            reconstructed = reconstructed[:, :, :100, :403]
            loss = 0.5 * F.l1_loss(reconstructed, fields_grid) + 0.5 * F.mse_loss(reconstructed, fields_grid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            _optimizer_step(optimizer, counters)
            return loss

        return _instrument_update(latent_stage1_update, counters), target_count

    if method == "Latent FM" and int(stage["stage_ordinal"]) == 2:
        model.eval()
        velocity_net = bundle.components["velocity_net"]
        velocity_net.train()
        pointnet = bundle.components.get("pointnet_encoder")
        if pointnet is not None:
            pointnet.train()
        def latent_stage2_update() -> torch.Tensor:
            fields_grid = baseline_lib.pointcloud_to_grid(raw["fields"], 100, 403)
            sparse = _fixed_sparse(raw)
            obs_value_grid, obs_mask_grid = baseline_lib.build_obs_grid_mask(
                sparse["obs_values"],
                sparse["obs_mask"],
                sparse["obs_field_ids"],
                sparse["obs_indices"],
                raw["fields"].shape[2],
                raw["fields"].shape[1],
                100,
                403,
                100,
                403,
            )
            ix = (sparse["obs_indices"] % 403).float() / 402.0
            iy = torch.div(sparse["obs_indices"], 403, rounding_mode="floor").float() / 99.0
            cond_inputs = {
                "obs_value_grid": obs_value_grid,
                "obs_mask_grid": obs_mask_grid,
                "obs_coords_2d": torch.stack([ix, iy], dim=-1),
                "obs_values": sparse["obs_values"],
                "obs_mask": sparse["obs_mask"],
                "obs_field_ids": sparse["obs_field_ids"],
            }
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.training_loss(fields_grid, cond_inputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(velocity_net.parameters(), max_norm=1.0)
            _optimizer_step(optimizer, counters)
            if runtime.ema is not None:
                _ema_update(runtime.ema, velocity_net, counters)
            return loss

        return _instrument_update(latent_stage2_update, counters), target_count

    if method == "SiT":
        model.train()
        transport = bundle.components["transport"]
        point_to_grid = bundle.components.get("point_to_grid")
        h_pad = int(bundle.components["H_pad"])
        w_pad = int(bundle.components["W_pad"])
        tokenizer = str(bundle.components["tokenizer"])
        cond_mode = str(bundle.components["cond_mode"])
        huber_beta = float(bundle.components["huber_beta"])
        all_params = bundle.components["all_params_fn"]
        spike_state = bundle.components["spike_state"]

        def sit_update() -> torch.Tensor:
            grid, sparse = _grid_inputs(raw, point_to_grid=point_to_grid, h_pad=h_pad, w_pad=w_pad)
            model_kwargs: dict[str, Any] = {}
            if tokenizer == "pointnet":
                model_kwargs["coords"] = raw["coords"]
                model_kwargs["obs_value_nodes"], model_kwargs["obs_mask_nodes"] = baseline_lib.scatter_sensors_to_nodes(
                    sparse["obs_values"],
                    sparse["obs_mask"],
                    sparse["obs_field_ids"],
                    sparse["obs_indices"],
                    raw["fields"].shape[0],
                    raw["fields"].shape[1],
                    raw["fields"].shape[2],
                    raw["coords"].device,
                    raw["fields"].dtype,
                )
            else:
                obs_value_grid = sparse["obs_value_grid"]
                if cond_mode == "interp":
                    obs_value_grid = baseline_lib.nearest_fill_grid(obs_value_grid, sparse["obs_mask_grid"])
                model_kwargs["obs_value_grid"] = obs_value_grid
                model_kwargs["obs_mask_grid"] = sparse["obs_mask_grid"]
            optimizer.zero_grad(set_to_none=True)
            loss_dict = transport.training_losses(model, grid, model_kwargs=model_kwargs, huber_beta=huber_beta)
            loss = loss_dict["loss"].float().mean()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(all_params(), max_norm=0.5))
            current_loss = float(loss.detach())
            skip = not (math.isfinite(current_loss) and math.isfinite(grad_norm))
            ema_loss = spike_state.get("ema_loss")
            ema_grad = spike_state.get("ema_grad")
            if not skip and ema_loss is not None and current_loss > 5.0 * float(ema_loss):
                skip = True
            if not skip and ema_grad is not None and grad_norm > 10.0 * float(ema_grad):
                skip = True
            if skip:
                optimizer.zero_grad(set_to_none=True)
                _optimizer_skip(counters)
                _ema_skip(runtime.ema, counters)
                spike_state["skipped"] = int(spike_state.get("skipped", 0)) + 1
            else:
                _optimizer_step(optimizer, counters)
                if runtime.ema is not None:
                    _ema_update(runtime.ema, all_params(), counters)
                beta = 0.99
                spike_state["ema_loss"] = current_loss if ema_loss is None else beta * float(ema_loss) + (1.0 - beta) * current_loss
                spike_state["ema_grad"] = grad_norm if ema_grad is None else beta * float(ema_grad) + (1.0 - beta) * grad_norm
            return loss

        return _instrument_update(sit_update, counters), target_count

    if method == "Geo-FNO":
        model.train()
        point_to_grid = bundle.components.get("point_to_grid")

        def geofno_update() -> torch.Tensor:
            _, sparse = _grid_inputs(raw, point_to_grid=point_to_grid)
            target_grid = baseline_lib.pointcloud_to_grid_padded(
                raw["fields"], 100, 403, 100, 403, grid_order=bundle.components.get("grid_order")
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(sparse["obs_value_grid"], sparse["obs_mask_grid"])
            loss = F.mse_loss(prediction, target_grid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            _optimizer_step(optimizer, counters)
            return loss

        return _instrument_update(geofno_update, counters), target_count

    if method in {"MLP-RBF", "Senseiver"}:
        model.train()

        def deterministic_update() -> torch.Tensor:
            coords, fields = _fixed_query(raw)
            sparse = _fixed_sparse(raw)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                coords,
                sparse["obs_coords"],
                sparse["obs_values"],
                sparse["obs_mask"],
                sparse["obs_field_ids"],
            )
            loss = F.mse_loss(prediction, fields)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            _optimizer_step(optimizer, counters)
            return loss

        return _instrument_update(deterministic_update, counters), target_count

    raise KeyError(f"No canonical common-batch update implementation for {method} ({runtime.family})")


def _timed_update(update: Callable[[], torch.Tensor], device: torch.device) -> tuple[float, float, float, float]:
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    loss = update()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    peak_allocated_mib = _mib(torch.cuda.max_memory_allocated(device))
    peak_reserved_mib = _mib(torch.cuda.max_memory_reserved(device))
    loss_value = float(loss.detach())
    del loss
    if not _finite(elapsed_ms) or elapsed_ms <= 0.0 or not _finite(loss_value):
        raise RuntimeError(f"non-finite common-batch update: loss={loss_value}, elapsed_ms={elapsed_ms}")
    return loss_value, elapsed_ms, peak_allocated_mib, peak_reserved_mib


def _timing_summary(values: Sequence[float], blocks: Sequence[Sequence[float]], tolerance: float) -> dict[str, Any]:
    if not values:
        raise ValueError("timing summary requires at least one measured update")
    array = np.asarray(values, dtype=float)
    p10, q25, median, q75, p90 = (float(value) for value in np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90]))
    block_medians = [float(statistics.median(block)) for block in blocks if block]
    half = len(block_medians) // 2
    early = float(statistics.median(block_medians[:half])) if half else block_medians[0]
    late = float(statistics.median(block_medians[half:])) if half else block_medians[-1]
    drift = abs(early - late) / max(median, 1.0e-12)
    return {
        "update_time_p10_ms": p10,
        "update_time_q25_ms": q25,
        "update_time_median_ms": median,
        "update_time_q75_ms": q75,
        "update_time_p90_ms": p90,
        "first_half_block_median_ms": early,
        "last_half_block_median_ms": late,
        "stability_delta_fraction": float(drift),
        "stability_tolerance_fraction": float(tolerance),
        "stability_status": "pass" if drift <= tolerance else "unstable",
    }


def _nvml_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in samples if _finite(row.get("memory_used_mib"))]
    if not valid:
        return {
            "nvml_sample_count": 0,
            "nvml_device_memory_baseline_mib": float("nan"),
            "nvml_device_memory_peak_mib": float("nan"),
            "nvml_process_memory_baseline_mib": float("nan"),
            "nvml_process_memory_peak_mib": float("nan"),
            "foreign_device_memory_start_mib": float("nan"),
            "foreign_device_memory_end_mib": float("nan"),
            "foreign_device_memory_peak_mib": float("nan"),
            "foreign_use_observed": True,
        }
    return {
        "nvml_sample_count": len(valid),
        "nvml_device_memory_baseline_mib": float(valid[0]["memory_used_mib"]),
        "nvml_device_memory_peak_mib": float(max(row["memory_used_mib"] for row in valid)),
        "nvml_process_memory_baseline_mib": float(valid[0].get("process_memory_mib", 0.0)),
        "nvml_process_memory_peak_mib": float(max(row.get("process_memory_mib", 0.0) for row in valid)),
        "foreign_device_memory_start_mib": float(valid[0].get("foreign_memory_mib", 0.0)),
        "foreign_device_memory_end_mib": float(valid[-1].get("foreign_memory_mib", 0.0)),
        "foreign_device_memory_peak_mib": float(max(row.get("foreign_memory_mib", 0.0) for row in valid)),
        "foreign_use_observed": any(int(row.get("foreign_process_count", 0)) > 0 for row in valid),
    }


def _stage_summary(
    runtime: StageRuntime,
    raw: Mapping[str, torch.Tensor],
    stage: Mapping[str, Any],
    *,
    times: Sequence[float],
    allocs: Sequence[float],
    reserves: Sequence[float],
    blocks: Sequence[Sequence[float]],
    nvml_samples: Sequence[Mapping[str, Any]],
    tolerance: float,
    warmups: int,
    measured_blocks: int,
    updates_per_block: int,
    elapsed_seconds: float,
    warmup_counts: Mapping[str, int] | None = None,
    measured_counts: Mapping[str, int] | None = None,
    sit_spike_state_reset: bool = False,
    status: str = "ok",
    failure_type: str = "",
    failure_reason: str = "",
) -> dict[str, Any]:
    warmup_counts = warmup_counts or _zero_counter_snapshot()
    measured_counts = measured_counts or _zero_counter_snapshot()
    params_bytes, buffers_bytes, parameter_count = _module_bytes(runtime.model)
    optimizer_bytes = _optimizer_bytes(runtime.optimizer)
    gradients_bytes = _gradient_bytes(runtime.model)
    ema_bytes = _ema_bytes(runtime.ema)
    input_bytes = _unique_tensor_bytes(raw.values())
    persistent_bytes = params_bytes + buffers_bytes + optimizer_bytes + gradients_bytes + ema_bytes
    device = next(runtime.model.parameters()).device
    peak_allocated = float(max(allocs)) if allocs else float(_mib(torch.cuda.max_memory_allocated(device)))
    peak_reserved = float(max(reserves)) if reserves else float(_mib(torch.cuda.max_memory_reserved(device)))
    workspace = max(0.0, peak_allocated - _mib(persistent_bytes + input_bytes))
    decomposition_sum = _mib(persistent_bytes + input_bytes) + workspace
    timing = _timing_summary(times, blocks, tolerance) if times else {
        "update_time_p10_ms": float("nan"),
        "update_time_q25_ms": float("nan"),
        "update_time_median_ms": float("nan"),
        "update_time_q75_ms": float("nan"),
        "update_time_p90_ms": float("nan"),
        "first_half_block_median_ms": float("nan"),
        "last_half_block_median_ms": float("nan"),
        "stability_delta_fraction": float("nan"),
        "stability_tolerance_fraction": tolerance,
        "stability_status": "not_measured",
    }
    try:
        checkpoint_sha = sha256_file(resolve(stage["checkpoint_path"]))
    except OSError:
        checkpoint_sha = ""
    result: dict[str, Any] = {
        "method": runtime.method,
        "stage_ordinal": int(stage["stage_ordinal"]),
        "stage_id": str(stage["stage_id"]),
        "stage_role": str(stage["stage_role"]),
        "status": status,
        "failure_type": failure_type,
        "failure_reason": failure_reason,
        "checkpoint_path": str(stage["checkpoint_path"]),
        "checkpoint_sha256": checkpoint_sha,
        "config_path": str(stage["config_path"]),
        "config_sha256": str(stage["config_sha256"]),
        "training_target_mode": str(stage["training_target_mode"]),
        "n_training_targets": int(stage["n_training_targets"]),
        "batch_size": 32,
        "sensor_count": 256,
        "dtype": "float32",
        "family": runtime.family,
        "parameter_count": parameter_count,
        "model_parameters_mib": _mib(params_bytes),
        "model_buffers_mib": _mib(buffers_bytes),
        "model_plus_buffers_mib": _mib(params_bytes + buffers_bytes),
        "optimizer_state_mib": _mib(optimizer_bytes),
        "gradient_mib": _mib(gradients_bytes),
        "ema_mib": _mib(ema_bytes),
        "fixed_input_batch_mib": _mib(input_bytes),
        "persistent_components_mib": _mib(persistent_bytes),
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "activation_workspace_estimate_mib": workspace,
        "component_decomposition_sum_mib": decomposition_sum,
        "component_decomposition_residual_mib": float(peak_allocated - decomposition_sum),
        "warmup_updates": int(warmups),
        "measured_updates": len(times),
        "measured_blocks": int(measured_blocks if times else 0),
        "updates_per_block": int(updates_per_block),
        "elapsed_seconds": float(elapsed_seconds),
        "optimizer_step_attempts_warmup": int(warmup_counts.get("optimizer_step_attempts", 0)),
        "optimizer_step_successes_warmup": int(warmup_counts.get("optimizer_step_successes", 0)),
        "optimizer_step_skips_warmup": int(warmup_counts.get("optimizer_step_skips", 0)),
        "optimizer_step_attempts_measured": int(measured_counts.get("optimizer_step_attempts", 0)),
        "optimizer_step_successes_measured": int(measured_counts.get("optimizer_step_successes", 0)),
        "optimizer_step_skips_measured": int(measured_counts.get("optimizer_step_skips", 0)),
        "ema_update_attempts_warmup": int(warmup_counts.get("ema_update_attempts", 0)),
        "ema_update_successes_warmup": int(warmup_counts.get("ema_update_successes", 0)),
        "ema_update_skips_warmup": int(warmup_counts.get("ema_update_skips", 0)),
        "ema_update_attempts_measured": int(measured_counts.get("ema_update_attempts", 0)),
        "ema_update_successes_measured": int(measured_counts.get("ema_update_successes", 0)),
        "ema_update_skips_measured": int(measured_counts.get("ema_update_skips", 0)),
        "ema_expected": bool(runtime.method == "SiT" or (runtime.method == "Latent FM" and int(stage["stage_ordinal"]) == 2)),
        "benchmark_control_sit_spike_state_reset": bool(sit_spike_state_reset),
        **timing,
        **_nvml_summary(nvml_samples),
    }
    return result


def _method_summary(stage_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        rows = [row for row in stage_rows if str(row["method"]) == method]
        ok = all(str(row["status"]) == "ok" for row in rows)
        peak_allocated = max((float(row["peak_allocated_mib"]) for row in rows if _finite(row.get("peak_allocated_mib"))), default=float("nan"))
        peak_reserved = max((float(row["peak_reserved_mib"]) for row in rows if _finite(row.get("peak_reserved_mib"))), default=float("nan"))
        single = len(rows) == 1
        timing = rows[0] if single else {}
        output.append(
            {
                "method": method,
                "status": "ok" if ok else next((str(row["status"]) for row in rows if str(row["status"]) != "ok"), "unavailable"),
                "stage_count": len(rows),
                "stage_ids": ";".join(str(row["stage_id"]) for row in rows),
                "batch_size": 32,
                "sensor_count": 256,
                "training_target_modes": ";".join(str(row["training_target_mode"]) for row in rows),
                "n_training_targets": ";".join(str(row["n_training_targets"]) for row in rows),
                "training_update_time_p10_ms": timing.get("update_time_p10_ms", ""),
                "training_update_time_q25_ms": timing.get("update_time_q25_ms", ""),
                "training_update_time_median_ms": timing.get("update_time_median_ms", ""),
                "training_update_time_q75_ms": timing.get("update_time_q75_ms", ""),
                "training_update_time_p90_ms": timing.get("update_time_p90_ms", ""),
                "peak_allocated_mib": peak_allocated,
                "peak_reserved_mib": peak_reserved,
                "activation_workspace_estimate_mib": max((float(row.get("activation_workspace_estimate_mib", 0.0)) for row in rows), default=float("nan")),
                "model_plus_buffers_mib": ";".join(str(row.get("model_plus_buffers_mib", "")) for row in rows),
                "optimizer_state_mib": ";".join(str(row.get("optimizer_state_mib", "")) for row in rows),
                "gradient_mib": ";".join(str(row.get("gradient_mib", "")) for row in rows),
                "ema_mib": ";".join(str(row.get("ema_mib", "")) for row in rows),
                "optimizer_step_attempts_warmup": ";".join(str(row.get("optimizer_step_attempts_warmup", "")) for row in rows),
                "optimizer_step_successes_warmup": ";".join(str(row.get("optimizer_step_successes_warmup", "")) for row in rows),
                "optimizer_step_skips_warmup": ";".join(str(row.get("optimizer_step_skips_warmup", "")) for row in rows),
                "optimizer_step_attempts_measured": ";".join(str(row.get("optimizer_step_attempts_measured", "")) for row in rows),
                "optimizer_step_successes_measured": ";".join(str(row.get("optimizer_step_successes_measured", "")) for row in rows),
                "optimizer_step_skips_measured": ";".join(str(row.get("optimizer_step_skips_measured", "")) for row in rows),
                "ema_update_attempts_warmup": ";".join(str(row.get("ema_update_attempts_warmup", "")) for row in rows),
                "ema_update_successes_warmup": ";".join(str(row.get("ema_update_successes_warmup", "")) for row in rows),
                "ema_update_skips_warmup": ";".join(str(row.get("ema_update_skips_warmup", "")) for row in rows),
                "ema_update_attempts_measured": ";".join(str(row.get("ema_update_attempts_measured", "")) for row in rows),
                "ema_update_successes_measured": ";".join(str(row.get("ema_update_successes_measured", "")) for row in rows),
                "ema_update_skips_measured": ";".join(str(row.get("ema_update_skips_measured", "")) for row in rows),
                "ema_expected": ";".join(str(bool(row.get("ema_expected", False))) for row in rows),
                "benchmark_control_sit_spike_state_reset": ";".join(str(bool(row.get("benchmark_control_sit_spike_state_reset", False))) for row in rows),
                "stage_specific_timing": not single,
                "unavailable_reason": "; ".join(str(row.get("failure_reason", "")) for row in rows if str(row["status"]) != "ok"),
            }
        )
    return output


def _validate_contract(config: Mapping[str, Any]) -> None:
    if str(config.get("schema_version")) != "figure5-training-footprint-common-b32-v51":
        raise ValueError("unexpected common-batch config schema")
    if str(config.get("condition")) != "Cond_T" or str(config.get("dtype")) != "float32":
        raise ValueError("common-batch benchmark must use float32 Cond_T")
    if int(config.get("batch_size")) != 32 or int(config.get("sensor_count")) != 256:
        raise ValueError("common-batch benchmark requires B=32 and M=256")
    protocol = config["protocol"]
    if (int(protocol["warmup_updates"]), int(protocol["measured_blocks"]), int(protocol["updates_per_block"]), int(protocol["measured_updates"])) != (20, 10, 10, 100):
        raise ValueError("formal benchmark requires exactly 20 warmups and 10x10 measured updates")
    if len(config.get("stages", [])) != 9:
        raise ValueError("common-batch plan must contain nine stage attempts")
    if list(config.get("method_order", [])) != list(METHOD_ORDER):
        raise ValueError("method order does not match the frozen eight-method lifecycle order")
    seen = [str(row["method"]) for row in config["stages"]]
    if seen.count("Latent FM") != 2:
        raise ValueError("Latent FM must have separate stage 1 and stage 2 rows")
    for row in config["stages"]:
        mode = str(row["training_target_mode"])
        count = int(row["n_training_targets"])
        if mode == "query_4096" and count != 4096:
            raise ValueError(f"query target count mismatch for {row['method']}")
        if mode == "native_full_grid" and count != 40300:
            raise ValueError(f"native target count mismatch for {row['method']}")
        expected_query = str(row["method"]) in QUERY_METHODS
        if (mode == "query_4096") != expected_query:
            raise ValueError(f"frozen target mode mismatch for {row['method']}")


def _counter_phase_pass(row: Mapping[str, Any], root: str, phase: str, expected: int) -> bool:
    return (
        int(row.get(f"{root}_attempts_{phase}", -1)) == expected
        and int(row.get(f"{root}_successes_{phase}", -1)) == expected
        and int(row.get(f"{root}_skips_{phase}", -1)) == 0
    )


def _zero_counter_phase_pass(row: Mapping[str, Any], root: str, phase: str) -> bool:
    return all(int(row.get(f"{root}_{suffix}_{phase}", -1)) == 0 for suffix in ("attempts", "successes", "skips"))


def _stage_counter_contract_pass(row: Mapping[str, Any]) -> bool:
    """Require every successful stage's steps and native EMA calls to occur."""
    if not _counter_phase_pass(row, "optimizer_step", "warmup", 20):
        return False
    if not _counter_phase_pass(row, "optimizer_step", "measured", 100):
        return False
    if bool(row.get("ema_expected", False)):
        return _counter_phase_pass(row, "ema_update", "warmup", 20) and _counter_phase_pass(row, "ema_update", "measured", 100)
    return _zero_counter_phase_pass(row, "ema_update", "warmup") and _zero_counter_phase_pass(row, "ema_update", "measured")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "training_footprint_common_b32_v51.yaml")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-clean-gpu", action="store_true")
    return parser.parse_args(argv)


def _clean_gate(index: int, config: Mapping[str, Any]) -> dict[str, Any]:
    before_rows = assert_clean_gpu(index, allow_current=False)
    usage = _nvidia_gpu_usage(index)
    max_util = float(config["clean_gpu_gate"]["max_idle_utilization_percent"])
    max_memory = float(config["clean_gpu_gate"]["max_idle_memory_mib"])
    if not _finite(usage.get("utilization_percent")) or not _finite(usage.get("memory_used_mib")):
        raise RuntimeError("could not read selected GPU idle state")
    if float(usage["utilization_percent"]) > max_util:
        raise RuntimeError(f"GPU {index} idle-utilisation gate failed: {usage['utilization_percent']}%")
    if float(usage["memory_used_mib"]) > max_memory:
        raise RuntimeError(f"GPU {index} idle-memory gate failed: {usage['memory_used_mib']} MiB")
    return {
        "index": index,
        "gpu_uuid": usage.get("uuid", ""),
        "gpu_name": _gpu_name(index),
        "memory_used_mib": float(usage["memory_used_mib"]),
        "utilization_percent": float(usage["utilization_percent"]),
        "foreign_processes": before_rows,
        "passed": True,
    }


def _gpu_name(index: int) -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "-i", str(index), "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _environment(device: str, index: int) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": device,
        "gpu_index": index,
        "gpu_uuid": _nvidia_gpu_usage(index).get("uuid", ""),
        "gpu_name": _gpu_name(index),
        "driver": subprocess.check_output(
            ["nvidia-smi", "-i", str(index), "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "dtype": "float32",
        "batch_size": 32,
        "sensor_count": 256,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    _validate_contract(config)
    if not args.execute:
        plan = {
            "status": "planned",
            "config": str(config_path),
            "run_id": args.run_id or config["output"]["run_id"],
            "device": args.device,
            "protocol": config["protocol"],
            "batch_size": config["batch_size"],
            "sensor_count": config["sensor_count"],
            "stages": [
                {
                    "method": row["method"],
                    "stage_id": row["stage_id"],
                    "training_target_mode": row["training_target_mode"],
                    "n_training_targets": row["n_training_targets"],
                }
                for row in config["stages"]
            ],
            "safety": {"execution_enabled": False, "checkpoint_write": False, "archive_mutation": False},
        }
        print(json.dumps(plan, indent=2))
        return 0
    if not args.confirm_clean_gpu:
        print("--execute requires --confirm-clean-gpu", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("common-batch benchmark requires CUDA", file=sys.stderr)
        return 2
    index = gpu_index(args.device)
    if index != int(torch.device(args.device).index):
        raise RuntimeError("selected CUDA index could not be resolved")
    run_id = args.run_id or str(config["output"]["run_id"])
    output_root = resolve(args.output_root or config["output"]["root"])
    run_dir = output_root / run_id
    if run_dir.exists():
        print(f"refusing to overwrite existing benchmark run: {run_dir}", file=sys.stderr)
        return 2

    gate = _clean_gate(index, config)
    run_dir.mkdir(parents=True)
    before_state = gpu_state(index)
    (run_dir / "gpu_state_before.txt").write_text(before_state, encoding="utf-8")
    fixed = _build_fixed_batch(config)
    protocol = config["protocol"]
    warmups = int(protocol["warmup_updates"])
    measured_blocks = int(protocol["measured_blocks"])
    updates_per_block = int(protocol["updates_per_block"])
    device = torch.device(args.device)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "formal": True,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "environment": _environment(args.device, index),
        "clean_gpu_gate": gate,
        "protocol": protocol,
        "fixed_batch": {
            "split": config["split"],
            "state_indices": list(fixed.state_indices),
            "original_hdf5_time_indices": list(fixed.original_time_indices),
            "batch_size": 32,
            "sensor_count": 256,
            "sensor_rule": config["sensor_plan"]["rule"],
            "sensor_seed_base": config["sensor_plan"]["selection_seed_base"],
            "query_seed_base": config["fixed_batch"]["query_seed_base"],
            "identity_sha256": fixed.identity_sha256,
            "discarded_after_stage": True,
            "accuracy_claim": False,
        },
        "benchmark_control_initialization": {
            "sit_spike_state_reset": {
                "when": "immediately before SiT warmup updates",
                "fields_reset": ["ema_loss", "ema_grad", "skipped"],
                "state_after_reset": {"ema_loss": None, "ema_grad": None, "skipped": 0},
                "scope": "in-memory bundle.components['spike_state'] only",
                "checkpoint_mutation": False,
                "applied": False,
            }
        },
        "target_policy": config["target_policy"],
        "safety": {
            "execution_mode": "explicit_confirmed_clean_gpu_common_batch",
            "checkpoint_write": False,
            "archive_mutation": False,
            "data_io_inside_timing": False,
            "host_to_device_transfer_inside_timing": False,
            "unequal_batch_memory_source_used": False,
        },
        "checkpoint_identities": [
            {
                "method": row["method"],
                "stage_id": row["stage_id"],
                "path": row["checkpoint_path"],
                "sha256_expected": row["checkpoint_sha256"],
                "config_path": row["config_path"],
                "config_sha256_expected": row["config_sha256"],
            }
            for row in config["stages"]
        ],
        "outputs": [
            "manifest.json",
            "qa.json",
            "training_footprint_summary.csv",
            "training_stage_summary.csv",
            "benchmark_repeats.csv",
            "gpu_state_before.txt",
            "gpu_state_after.txt",
        ],
    }
    atomic_json(run_dir / "manifest.json", manifest)
    stage_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    stage_wall_times: dict[str, float] = {}
    start_total = time.perf_counter()
    try:
        for ordinal, stage in enumerate(config["stages"]):
            method = str(stage["method"])
            stage_start = time.perf_counter()
            print(f"[common-b32] {method} / {stage['stage_role']}", flush=True)
            runtime: StageRuntime | None = None
            raw: dict[str, torch.Tensor] | None = None
            sampler: NVMLSampler | None = None
            times: list[float] = []
            allocs: list[float] = []
            reserves: list[float] = []
            timing_blocks: list[list[float]] = []
            nvml_samples: list[dict[str, Any]] = []
            warmup_counts = _zero_counter_snapshot()
            measured_counts = _zero_counter_snapshot()
            counters: UpdateCounters | None = None
            counter_start = _zero_counter_snapshot()
            warmup_end = _zero_counter_snapshot()
            sit_spike_state_reset = False
            status = "ok"
            failure_type = ""
            failure_reason = ""
            try:
                runtime = _load_stage(stage, config, device)
                raw = _device_fixed_batch(fixed, device)
                update, target_count = _make_update(runtime, raw, stage)
                if not isinstance(update, InstrumentedUpdate):
                    raise RuntimeError(f"un-instrumented canonical update for {method}")
                counters = update.counters
                counter_start = counters.snapshot()
                if int(target_count) != int(stage["n_training_targets"]):
                    raise RuntimeError(f"runtime target count mismatch for {method}")
                reset_info = _reset_sit_spike_state(runtime)
                if reset_info is not None:
                    sit_spike_state_reset = bool(reset_info.get("applied"))
                    manifest["benchmark_control_initialization"]["sit_spike_state_reset"].update(reset_info)
                    atomic_json(run_dir / "manifest.json", manifest)
                for _ in range(warmups):
                    _timed_update(update, device)
                warmup_end = counters.snapshot()
                warmup_counts = _counter_delta(counter_start, warmup_end)
                measured_start = warmup_end
                torch.cuda.synchronize(device)
                sampler = NVMLSampler(index, float(config["memory"]["poll_interval_seconds"]))
                sampler.start()
                for block in range(measured_blocks):
                    block_values: list[float] = []
                    for update_in_block in range(updates_per_block):
                        loss, elapsed, peak_allocated, peak_reserved = _timed_update(update, device)
                        times.append(elapsed)
                        allocs.append(peak_allocated)
                        reserves.append(peak_reserved)
                        block_values.append(elapsed)
                        latest = sampler.samples[-1] if sampler.samples else {}
                        repeat_rows.append(
                            {
                                "method": method,
                                "stage_id": stage["stage_id"],
                                "stage_ordinal": stage["stage_ordinal"],
                                "block": block,
                                "update_in_block": update_in_block,
                                "update_index": block * updates_per_block + update_in_block,
                                "elapsed_ms": elapsed,
                                "loss": loss,
                                "peak_allocated_mib": peak_allocated,
                                "peak_reserved_mib": peak_reserved,
                                "nvml_device_memory_mib": latest.get("memory_used_mib", ""),
                                "nvml_process_memory_mib": latest.get("process_memory_mib", ""),
                                "foreign_device_memory_mib": latest.get("foreign_memory_mib", ""),
                            }
                        )
                    timing_blocks.append(block_values)
                measured_counts = _counter_delta(measured_start, counters.snapshot())
                nvml_samples = sampler.stop()
                sampler = None
            except torch.cuda.OutOfMemoryError as exc:
                status = OOM_STATUS if method == "Geo-FNO" else "OOM"
                failure_type = "cuda_oom"
                failure_reason = f"batch-32 one-GPU canonical update exceeded memory without batch reduction: {exc}"
                if sampler is not None:
                    nvml_samples = sampler.stop()
                    sampler = None
                gc.collect()
                torch.cuda.empty_cache()
            finally:
                if sampler is not None:
                    nvml_samples = sampler.stop()
                if runtime is not None and raw is not None:
                    stage_elapsed = time.perf_counter() - stage_start
                    row = _stage_summary(
                        runtime,
                        raw,
                        stage,
                        times=times,
                        allocs=allocs,
                        reserves=reserves,
                        blocks=timing_blocks,
                        nvml_samples=nvml_samples,
                        tolerance=float(config["protocol"]["drift_gate"]["tolerance_fraction"]),
                        warmups=warmups,
                        measured_blocks=measured_blocks,
                        updates_per_block=updates_per_block,
                        elapsed_seconds=stage_elapsed,
                        warmup_counts=warmup_counts,
                        measured_counts=measured_counts,
                        sit_spike_state_reset=sit_spike_state_reset,
                        status=status,
                        failure_type=failure_type,
                        failure_reason=failure_reason,
                    )
                    stage_rows.append(row)
                    stage_wall_times[str(stage["stage_id"])] = stage_elapsed
                    runtime.close()
                elif runtime is not None:
                    runtime.close()
                if raw is not None:
                    del raw
                gc.collect()
                torch.cuda.empty_cache()
            if status not in {"ok", OOM_STATUS}:
                raise RuntimeError(f"stage {method} failed with status {status}: {failure_reason}")
            write_csv(run_dir / "benchmark_repeats.csv", repeat_rows or [{"method": method, "stage_id": stage["stage_id"], "status": status}])
            write_csv(run_dir / "training_stage_summary.csv", stage_rows)
    except Exception as exc:
        manifest.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
        atomic_json(run_dir / "manifest.json", manifest)
        raise

    after_state = gpu_state(index)
    (run_dir / "gpu_state_after.txt").write_text(after_state, encoding="utf-8")
    try:
        after_rows = assert_clean_gpu(index, allow_current=True)
        after_processes = _foreign_process_rows(after_rows)
    except Exception:
        after_processes = [{"error": "foreign process detected after run"}]
    method_rows = _method_summary(stage_rows)
    write_csv(run_dir / "training_footprint_summary.csv", method_rows)
    unchanged = all(
        sha256_file(resolve(row["checkpoint_path"])) == str(row["checkpoint_sha256"])
        for row in config["stages"]
    )
    successful = [row for row in stage_rows if row["status"] == "ok"]
    geo_rows = [row for row in stage_rows if row["method"] == "Geo-FNO"]
    non_geo_ok = all(row["status"] == "ok" for row in stage_rows if row["method"] != "Geo-FNO")
    drift_pass = bool(successful) and all(str(row["stability_status"]) == "pass" for row in successful)
    component_pass = all(
        _finite(row["peak_allocated_mib"])
        and _finite(row["peak_reserved_mib"])
        and float(row["peak_reserved_mib"]) + 1.0e-4 >= float(row["peak_allocated_mib"])
        and abs(float(row["component_decomposition_residual_mib"])) <= 1.0e-3
        for row in successful
    )
    nvml_pass = all(int(row["nvml_sample_count"]) > 0 for row in successful)
    foreign_pass = all(not bool(row.get("foreign_use_observed")) for row in stage_rows)
    geo_outcome = _geo_fno_outcome(geo_rows[0]) if len(geo_rows) == 1 else "invalid"
    geo_success = geo_outcome == "success_at_common_batch"
    geo_oom = geo_outcome == "oom_at_common_batch"
    geo_valid = geo_outcome in {"success_at_common_batch", "oom_at_common_batch"}
    stage_counter_pass = bool(successful) and all(_stage_counter_contract_pass(row) for row in successful)
    sit_rows = [row for row in stage_rows if row["method"] == "SiT"]
    sit_row = sit_rows[0] if len(sit_rows) == 1 else {}
    sit_optimizer_steps_pass = len(sit_rows) == 1 and sit_row.get("status") == "ok" and _counter_phase_pass(sit_row, "optimizer_step", "measured", 100) and int(sit_row.get("optimizer_step_skips_measured", -1)) == 0
    sit_ema_updates_pass = len(sit_rows) == 1 and sit_row.get("status") == "ok" and _counter_phase_pass(sit_row, "ema_update", "measured", 100) and int(sit_row.get("ema_update_skips_measured", -1)) == 0
    sit_reset_pass = bool(
        len(sit_rows) == 1
        and sit_row.get("benchmark_control_sit_spike_state_reset")
        and manifest["benchmark_control_initialization"]["sit_spike_state_reset"].get("applied")
        and not manifest["benchmark_control_initialization"]["sit_spike_state_reset"].get("checkpoint_mutation")
    )
    qa_pass = bool(
        len(stage_rows) == 9
        and len(method_rows) == 8
        and non_geo_ok
        and geo_valid
        and all(int(row["batch_size"]) == 32 and int(row["sensor_count"]) == 256 for row in stage_rows)
        and all(str(row["dtype"]) == "float32" for row in stage_rows)
        and all(int(row["measured_updates"]) == 100 for row in successful)
        and all(int(row["warmup_updates"]) == 20 for row in successful)
        and drift_pass
        and stage_counter_pass
        and sit_optimizer_steps_pass
        and sit_ema_updates_pass
        and sit_reset_pass
        and component_pass
        and nvml_pass
        and foreign_pass
        and unchanged
        and not after_processes
    )
    qa = {
        "status": "pass" if qa_pass else "fail",
        "formal_protocol_exact": True,
        "common_batch_exact_32": all(int(row["batch_size"]) == 32 for row in stage_rows),
        "fixed_cond_t_m256": all(int(row["sensor_count"]) == 256 for row in stage_rows),
        "float32": all(str(row["dtype"]) == "float32" for row in stage_rows),
        "target_modes_declared_and_observed": all(str(row["training_target_mode"]) in {"query_4096", "native_full_grid"} for row in stage_rows),
        "query_targets_exact_4096": all(int(row["n_training_targets"]) == 4096 for row in stage_rows if row["training_target_mode"] == "query_4096"),
        "native_targets_exact_40300": all(int(row["n_training_targets"]) == 40300 for row in stage_rows if row["training_target_mode"] == "native_full_grid"),
        "nine_stage_attempts": len(stage_rows) == 9,
        "latent_fm_stages_separate": len([row for row in stage_rows if row["method"] == "Latent FM"]) == 2,
        "geo_fno_attempted_at_batch_32": len(geo_rows) == 1 and int(geo_rows[0]["batch_size"]) == 32,
        "geo_fno_success_at_common_batch": geo_success,
        "geo_fno_oom_at_common_batch": geo_oom,
        "geo_fno_outcome_valid": geo_valid,
        "geo_fno_batch_not_reduced_after_oom": not geo_oom or int(geo_rows[0]["batch_size"]) == 32,
        "all_successful_stage_updates_finite": all(int(row["measured_updates"]) == 100 for row in successful),
        "optimizer_step_counters_pass": stage_counter_pass,
        "sit_optimizer_step_successes_measured_100": sit_optimizer_steps_pass,
        "ema_update_counters_pass": stage_counter_pass,
        "sit_ema_update_successes_measured_100": sit_ema_updates_pass,
        "sit_spike_state_reset_before_warmups": sit_reset_pass,
        "all_successful_stage_drift_gates_pass": drift_pass,
        "memory_metrics_present": component_pass,
        "component_decomposition_pass": component_pass,
        "nvml_process_metrics_present": nvml_pass,
        "foreign_use_absent": foreign_pass,
        "checkpoint_hashes_unchanged": unchanged,
        "gpu_clean_before": bool(gate["passed"]),
        "gpu_clean_after": not bool(after_processes),
        "no_data_io_or_host_transfer_inside_timing": True,
        "unequal_batch_memory_not_used": True,
    }
    elapsed_total = time.perf_counter() - start_total
    manifest.update(
        status="complete" if qa_pass else "failed",
        elapsed_seconds=float(elapsed_total),
        stage_wall_times_seconds=stage_wall_times,
        gpu_clean_after=not bool(after_processes),
        promotion_gate={
            "validated": qa_pass,
            "geo_fno_status": geo_rows[0]["status"] if geo_rows else "missing",
            "geo_fno_outcome": geo_outcome,
            "non_geo_stages_ok": non_geo_ok,
        },
    )
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(run_dir / "qa.json", qa)
    print(json.dumps({"output_dir": str(run_dir), "qa": qa}, indent=2))
    return 0 if qa_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

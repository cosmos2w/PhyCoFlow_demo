#!/usr/bin/env python
"""Standardized native inference-memory benchmark for Figure 5 V5.1.

Every model is evaluated with gradients globally disabled.  The benchmark
separates model tensor storage, the inference-ready process allocation, and
the additional workspace reached during one native reconstruction.  It never
loads or writes a new checkpoint and never retains model output fields.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
DEMO_ROOT = REPO_ROOT / "0_demo_TurbulentCombustion"
SCRIPT_ROOT = DEMO_ROOT / "Save_TrainedModel" / "_TrainedModels" / "_Scripts"
SRC_ROOT = DEMO_ROOT / "src"
TOOLS_ROOT = DEMO_ROOT / "tools"
for source_root in (SCRIPT_ROOT, SRC_ROOT, TOOLS_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from benchmark_validation_v3 import (  # noqa: E402
    METHODS,
    core_call,
    load_sensor_rows,
    method_settings,
    prepare_state,
    verify_identities,
)
from common.config import load_config  # noqa: E402
from common.model_loader import load_model  # noqa: E402


SCHEMA_VERSION = "figure5-v51-native-inference-memory-1"


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def mib(value: int | float) -> float:
    return float(value) / 2**20


def nvidia_smi() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    process_command = [
        "nvidia-smi",
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    parts = []
    for current in (command, process_command):
        try:
            parts.append(subprocess.check_output(current, text=True, stderr=subprocess.STDOUT).strip())
        except (OSError, subprocess.CalledProcessError) as exc:
            parts.append(f"unavailable: {type(exc).__name__}: {exc}")
    return "\n".join(parts) + "\n"


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def unique_tensor_bytes(values: Iterable[torch.Tensor]) -> int:
    seen: set[tuple[str, int, int]] = set()
    total = 0
    for tensor in values:
        if not torch.is_tensor(tensor):
            continue
        storage = tensor.untyped_storage()
        key = (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes()))
        if key in seen:
            continue
        seen.add(key)
        total += int(storage.nbytes())
    return total


def inference_modules(loaded: Any) -> list[torch.nn.Module]:
    if loaded.family == "pointcloud_ffm":
        return [loaded.model]
    bundle = loaded.model
    modules = [bundle.model]
    for value in bundle.components.values():
        if isinstance(value, torch.nn.Module):
            modules.append(value)
    return modules


def model_state_bytes(loaded: Any) -> int:
    tensors: list[torch.Tensor] = []
    for module in inference_modules(loaded):
        tensors.extend(module.parameters())
        tensors.extend(module.buffers())
    return unique_tensor_bytes(tensors)


def materialize_evaluation_weights_and_drop_training_state(loaded: Any, method: str) -> str:
    """Keep adopted evaluation weights while releasing optimizer/EMA copies."""

    if loaded.family == "pointcloud_ffm":
        loaded.model.eval()
        return "checkpoint_model_weights"

    bundle = loaded.model
    policy = "checkpoint_model_weights"
    if method == "Latent FM" and int(bundle.training_stage) == 2 and bundle.ema is not None:
        bundle.ema.apply(bundle.components["velocity_net"])
        policy = "materialized_latent_fm_ema"
    elif method == "SiT" and bundle.ema is not None:
        bundle.ema.copy_to(bundle.components["all_params_fn"]())
        policy = "materialized_sit_ema"
    elif bundle.ema is not None:
        all_params = bundle.components.get("all_params_fn")
        if callable(all_params) and hasattr(bundle.ema, "copy_to"):
            bundle.ema.copy_to(all_params())
            policy = "materialized_generic_ema"
        else:
            raise RuntimeError(f"Cannot materialize evaluation EMA for {method}")

    bundle.optimizer = None
    bundle.scheduler = None
    bundle.ema = None
    bundle.dataset_train = None
    bundle.dataset_val = None
    bundle.model.eval()
    return policy


def output_tensor_bytes(output: Any) -> int:
    if torch.is_tensor(output):
        return unique_tensor_bytes([output])
    if isinstance(output, Mapping):
        return unique_tensor_bytes([value for value in output.values() if torch.is_tensor(value)])
    return 0


def run_call(loaded: Any, prepared: dict[str, Any], settings: dict[str, Any], method: str) -> torch.Tensor:
    if torch.is_grad_enabled():
        raise RuntimeError("Inference-memory call entered with gradients enabled")
    output = core_call(loaded, prepared, settings, method)
    if not torch.is_tensor(output):
        raise TypeError(f"Expected tensor output for {method}, got {type(output)!r}")
    if output.requires_grad or output.grad_fn is not None:
        raise RuntimeError(f"Inference output unexpectedly retains autograd state for {method}")
    return output


def benchmark_method(
    method: str,
    method_cfg: Mapping[str, Any],
    sensors: list[dict[str, int]],
    config: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    post_cfg = load_config()
    settings = method_settings(post_cfg, method)
    declared_execution = dict(config["execution"][method])
    settings.update(
        execution_mode=str(declared_execution["mode"]),
        cache_level=str(declared_execution["cache_level"]),
        query_chunk_size=int(declared_execution["query_chunk_size"]),
    )
    # Baseline adapters do not consume these point-cloud execution strings.
    if loaded_mode := str(declared_execution["mode"]):
        if loaded_mode.startswith("native_baseline_adapter"):
            settings["execution_mode"] = "legacy_full"

    loaded = load_model(
        dict(method_cfg),
        str(config["condition"]),
        checkpoint="last",
        split=str(config["split"]),
        device=str(device),
        n_steps=int(settings["n_steps"]),
        ode_solver="euler",
    )
    checkpoint_sha256 = sha256_file(loaded.checkpoint_path)
    eval_weight_policy = materialize_evaluation_weights_and_drop_training_state(loaded, method)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    resident_allocated = torch.cuda.memory_allocated(device)
    resident_reserved = torch.cuda.memory_reserved(device)
    tensor_bytes = model_state_bytes(loaded)

    prepared = prepare_state(
        loaded,
        int(config["state_index"]),
        sensors,
        int(config["query_count"]),
        method,
        retain_truth=False,
    )
    torch.cuda.synchronize(device)
    ready_allocated = torch.cuda.memory_allocated(device)
    ready_reserved = torch.cuda.memory_reserved(device)

    warmups = int(config["protocol"]["warmups"])
    repeats = int(config["protocol"]["measured_repeats"])
    with torch.inference_mode():
        for _ in range(warmups):
            output = run_call(loaded, prepared, settings, method)
            del output
            torch.cuda.synchronize(device)

        repeat_rows: list[dict[str, Any]] = []
        for repeat in range(repeats):
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            output = run_call(loaded, prepared, settings, method)
            torch.cuda.synchronize(device)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            output_bytes = output_tensor_bytes(output)
            repeat_rows.append(
                {
                    "method": method,
                    "repeat": repeat,
                    "peak_allocated_mib": mib(peak_allocated),
                    "peak_reserved_mib": mib(peak_reserved),
                    "ready_allocated_mib": mib(ready_allocated),
                    "incremental_workspace_mib": mib(max(0, peak_allocated - ready_allocated)),
                    "output_mib": mib(output_bytes),
                    "output_requires_grad": bool(output.requires_grad),
                }
            )
            del output
            torch.cuda.synchronize(device)

    peaks = np.asarray([row["peak_allocated_mib"] for row in repeat_rows], dtype=float)
    workspaces = np.asarray([row["incremental_workspace_mib"] for row in repeat_rows], dtype=float)
    summary = {
        "method": method,
        "status": "ok",
        "checkpoint_path": str(loaded.checkpoint_path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": checkpoint_sha256,
        "family": loaded.family,
        "backbone": loaded.backbone,
        "batch_size": int(config["batch_size"]),
        "sensor_count": int(config["sensor_count"]),
        "N": int(config["query_count"]),
        "dtype": str(config["dtype"]),
        "inference_context": "torch.inference_mode",
        "grad_enabled_inside_call": False,
        "output_requires_grad": False,
        "evaluation_weight_policy": eval_weight_policy,
        "execution_mode": str(declared_execution["mode"]),
        "cache_level": str(declared_execution["cache_level"]),
        "query_chunk_size": int(declared_execution["query_chunk_size"]),
        "model_state_mib": mib(tensor_bytes),
        "resident_process_allocated_mib": mib(resident_allocated),
        "resident_process_reserved_mib": mib(resident_reserved),
        "ready_allocated_mib": mib(ready_allocated),
        "ready_reserved_mib": mib(ready_reserved),
        "prepared_input_and_cache_mib": mib(max(0, ready_allocated - resident_allocated)),
        "inference_peak_allocated_mib": float(np.median(peaks)),
        "inference_peak_allocated_min_mib": float(np.min(peaks)),
        "inference_peak_allocated_max_mib": float(np.max(peaks)),
        "inference_workspace_mib": float(np.median(workspaces)),
        "warmups": warmups,
        "measured_repeats": repeats,
    }
    del prepared
    loaded.close()
    del loaded
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    return summary, repeat_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--run-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not args.execute:
        print(json.dumps(config, indent=2))
        return 0
    if bool(config["protocol"].get("allow_shared_gpu", False)) and not args.allow_shared_gpu:
        raise RuntimeError("This shared-GPU protocol requires the explicit --allow-shared-gpu acknowledgement")

    device = torch.device(args.device or str(config["device"]))
    if device.type != "cuda" or device.index is None:
        raise ValueError("A concrete CUDA device such as cuda:2 is required")
    torch.cuda.set_device(device.index)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    required_free = float(config["protocol"]["minimum_free_memory_gib"]) * 2**30
    if free_bytes < required_free:
        raise RuntimeError(
            f"Only {mib(free_bytes) / 1024:.2f} GiB is free on {device}; "
            f"the protocol requires at least {mib(required_free) / 1024:.2f} GiB"
        )

    plan_path = resolve(config["validation_plan"])
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    identity_checks = verify_identities(plan)
    state = int(config["state_index"])
    sensors = load_sensor_rows(plan, [state])[state]
    post_cfg = load_config()
    method_cfg = {str(row["name"]): row for row in post_cfg["methods"]}
    methods = [str(method) for method in config["methods"]]
    if methods != list(METHODS):
        raise RuntimeError("Inference-memory method order differs from the canonical eight-method order")

    run_id = args.run_id or f"inference_memory_native_v51_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = resolve(config["output"]["root"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    before = nvidia_smi()
    (output_dir / "gpu_state_before.txt").write_text(before, encoding="utf-8")

    summaries: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    for method in methods:
        free_now, _ = torch.cuda.mem_get_info(device)
        if free_now < required_free:
            summaries.append(
                {
                    "method": method,
                    "status": "insufficient_free_memory",
                    "failure_reason": f"free memory fell to {mib(free_now) / 1024:.2f} GiB",
                }
            )
            continue
        try:
            summary, current_repeats = benchmark_method(
                method, method_cfg[method], sensors, config, device
            )
            summaries.append(summary)
            repeats.extend(current_repeats)
            print(
                f"[MEMORY] {method}: model={summary['model_state_mib']:.1f} MiB, "
                f"ready={summary['ready_allocated_mib']:.1f} MiB, "
                f"peak={summary['inference_peak_allocated_mib']:.1f} MiB"
            )
        except torch.cuda.OutOfMemoryError as exc:
            summaries.append({"method": method, "status": "cuda_oom", "failure_reason": str(exc)})
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:  # preserve a compact, inspectable failure row
            summaries.append(
                {"method": method, "status": "failed", "failure_reason": f"{type(exc).__name__}: {exc}"}
            )
            gc.collect()
            torch.cuda.empty_cache()

    after = nvidia_smi()
    (output_dir / "gpu_state_after.txt").write_text(after, encoding="utf-8")
    write_csv(output_dir / "inference_memory_summary.csv", summaries)
    if repeats:
        write_csv(output_dir / "inference_memory_repeats.csv", repeats)

    successful = [row for row in summaries if row.get("status") == "ok"]
    qa = {
        "status": "pass" if len(successful) == len(methods) else "partial",
        "all_methods_successful": len(successful) == len(methods),
        "uniform_inference_context": all(row.get("inference_context") == "torch.inference_mode" for row in successful),
        "gradients_disabled_inside_call": all(row.get("grad_enabled_inside_call") is False for row in successful),
        "outputs_have_no_autograd_state": all(row.get("output_requires_grad") is False for row in successful),
        "common_native_workload": all(
            row.get("batch_size") == 1 and row.get("sensor_count") == 256 and row.get("N") == 40300
            for row in successful
        ),
        "peak_repeat_stable": all(
            abs(float(row["inference_peak_allocated_max_mib"]) - float(row["inference_peak_allocated_min_mib"])) < 1.0e-6
            for row in successful
        ),
        "checkpoint_identity_checks_pass": all(bool(row["pass"]) for row in identity_checks),
        "shared_gpu_explicitly_allowed": True,
        "timing_claim": False,
    }
    write_json(output_dir / "qa.json", qa)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if qa["status"] == "pass" else "partial",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": git_head(),
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": sha256_file(config_path),
        "validation_plan": str(plan_path.relative_to(REPO_ROOT)),
        "validation_plan_sha256": sha256_file(plan_path),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "free_memory_at_start_gib": mib(free_bytes) / 1024,
        "total_memory_gib": mib(total_bytes) / 1024,
        "foreign_gpu_work_allowed": True,
        "timing_claim": False,
        "measurement_boundary": dict(config["protocol"]),
        "identity_checks": identity_checks,
        "files": [
            "inference_memory_summary.csv",
            "inference_memory_repeats.csv",
            "qa.json",
            "gpu_state_before.txt",
            "gpu_state_after.txt",
        ],
    }
    write_json(output_dir / "manifest.json", manifest)
    print(output_dir)
    return 0 if qa["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())


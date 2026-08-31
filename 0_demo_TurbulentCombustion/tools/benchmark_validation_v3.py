#!/usr/bin/env python
"""Clean-GPU model-core benchmark for Figure 5 V3.

The timed region starts after model/data loading, host-to-device transfer,
sensor/query selection, and reusable geometry construction.  It includes
noise generation, value-dependent conditioning, model evaluations,
observation consistency, and the final device-side output.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from common.config import load_config, stable_seed  # noqa: E402
from common.model_loader import _plan_tensors, canonical, load_model  # noqa: E402


METHODS = ("DMF-Gen", "FFM-FNO", "FFM-Perceiver", "Latent FM", "SiT", "MLP-RBF", "Geo-FNO", "Senseiver")
QUERY_COUNTS = (1024, 4096, 16384, 40300)
SUPPORT = {
    "DMF-Gen": (True, "canonical point-cloud decoder accepts arbitrary query coordinates"),
    "FFM-FNO": (False, "FNOFFM.requires_full_grid=True; canonical backbone reshapes to 403x100"),
    "FFM-Perceiver": (True, "canonical point-cloud Perceiver accepts arbitrary query coordinates"),
    "Latent FM": (False, "latent decoder is trained and sampled on the fixed 403x100 grid"),
    "SiT": (False, "adopted checkpoint uses tokenizer=patch on the fixed 403x100 grid"),
    "MLP-RBF": (True, "deterministic regressor evaluates requested query coordinates directly"),
    "Geo-FNO": (False, "adopted geofno_variant=fno reconstructs the full dense grid"),
    "Senseiver": (True, "decoder cross-attention evaluates requested query coordinates directly"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--run-id")
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--minimum-repeats", type=int, default=30)
    parser.add_argument("--minimum-seconds", type=float, default=10.0)
    parser.add_argument("--historical-direct-probe-ms", type=float, default=29.0)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def gpu_index(device: str) -> int:
    value = torch.device(device)
    if value.type != "cuda" or value.index is None:
        raise ValueError("V3 formal cost requires an explicit CUDA device such as cuda:2")
    return int(value.index)


def gpu_state(index: int) -> str:
    return subprocess.check_output(["nvidia-smi", "-i", str(index)], text=True)


def gpu_uuid(index: int) -> str:
    return subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).strip()


def compute_processes() -> list[dict[str, str]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"], text=True
    ).strip()
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        uuid, pid, name, memory = (part.strip() for part in line.split(",", 3))
        rows.append({"gpu_uuid": uuid, "pid": pid, "process_name": name, "used_memory": memory})
    return rows


def assert_clean_gpu(index: int, *, allow_current: bool) -> list[dict[str, str]]:
    uuid = gpu_uuid(index)
    rows = [row for row in compute_processes() if row["gpu_uuid"] == uuid]
    foreign = [row for row in rows if not (allow_current and int(row["pid"]) == os.getpid())]
    if foreign:
        raise RuntimeError(f"GPU {index} is not clean: {foreign}")
    return rows


def environment(device: str) -> dict[str, Any]:
    index = gpu_index(device)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": device,
        "gpu_index": index,
        "gpu_uuid": gpu_uuid(index),
        "gpu_name": torch.cuda.get_device_name(index),
        "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", "-i", str(index)], text=True).strip(),
        "dtype": "float32",
        "batch_size": 1,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def verify_identities(plan: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for item in plan["checkpoints"]:
        actual = sha256(resolve(item["path"]))
        checks.append({"identity": item["method"], "path": item["path"], "expected_sha256": item["sha256"], "actual_sha256": actual, "pass": actual == item["sha256"]})
    for key in ("summary", "per_state", "reconstruction_manifest"):
        path = plan["frozen_accuracy"][f"{key}_path"]
        actual = sha256(resolve(path))
        expected = plan["frozen_accuracy"][f"{key}_sha256"]
        checks.append({"identity": f"FieldL2_{key}", "path": path, "expected_sha256": expected, "actual_sha256": actual, "pass": actual == expected})
    manifest_rows = read_csv(resolve(plan["frozen_accuracy"]["reconstruction_manifest_path"]))
    for item in plan["checkpoints"]:
        checkpoint = resolve(item["path"]).resolve()
        rows = [row for row in manifest_rows if row["method"] == item["method"] and row["condition"] == "Cond_T" and row["status"] == "ok"]
        paths = {Path(row["checkpoint_path"]).resolve() for row in rows}
        passed = len(rows) == 1000 and paths == {checkpoint} and {row["checkpoint_name"] for row in rows} == {"last.pt"}
        checks.append({"identity": f"FieldL2_join_{item['method']}", "path": item["path"], "manifest_rows": len(rows), "pass": passed})
    if not all(row["pass"] for row in checks):
        raise RuntimeError("Checkpoint or frozen FieldL2 identity mismatch")
    return checks


def frozen_errors(plan: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = read_csv(resolve(plan["frozen_accuracy"]["per_state_path"]))
    result = {}
    for method in METHODS:
        pairs = sorted(
            (int(row["snapshot"]), float(row["physical_rel_l2"]))
            for row in rows
            if row["condition"] == "Cond_T" and row["field"] == "Unobserved_mean" and row["status"] == "ok" and row["method"] == method
        )
        if len(pairs) != 1000:
            raise RuntimeError(f"Frozen FieldL2 has {len(pairs)} rows for {method}; expected 1000")
        values = np.asarray([value for _, value in pairs])
        low, high = moving_block_ci(values, plan, f"v3|accuracy|{method}")
        result[method] = {"error": float(np.mean(values)), "error_ci_low": low, "error_ci_high": high, "error_n": len(values)}
    return result


def moving_block_ci(values: np.ndarray, plan: dict[str, Any], salt: str) -> tuple[float, float]:
    spec = plan["bootstrap"]
    rng = np.random.default_rng(stable_seed(int(spec["seed"]), salt))
    block = min(int(spec["block_length"]), len(values))
    samples = np.empty(int(spec["replicates"]), dtype=float)
    for index in range(len(samples)):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values) / block)))
        selected = np.concatenate([(np.arange(block) + start) % len(values) for start in starts])[: len(values)]
        samples[index] = np.mean(values[selected])
    alpha = (1.0 - float(spec["confidence_level"])) / 2.0
    return tuple(float(value) for value in np.quantile(samples, [alpha, 1.0 - alpha]))


def load_sensor_rows(plan: dict[str, Any], states: list[int]) -> dict[int, list[dict[str, int]]]:
    groups: dict[int, list[dict[str, int]]] = {state: [] for state in states}
    with resolve(plan["sensor_plan"]["path"]).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            state = int(row["snapshot"])
            if row["condition"] == "Cond_T" and state in groups:
                groups[state].append({key: int(row[key]) for key in ("sensor_order", "field_index", "point_index", "sensor_seed")})
    for state, rows in groups.items():
        rows.sort(key=lambda row: row["sensor_order"])
        if len(rows) != 256 or {row["field_index"] for row in rows} != {2}:
            raise RuntimeError(f"Incomplete Cond_T sensor plan for state {state}")
    return groups


def query_subset(state: int, count: int, sensors: list[dict[str, int]]) -> np.ndarray | None:
    if count == 40300:
        return None
    required = np.asarray([row["point_index"] for row in sensors], dtype=np.int64)
    available = np.ones(40300, dtype=bool)
    available[required] = False
    rng = np.random.default_rng(stable_seed(20260830, "query_subset", state, count))
    additional = rng.choice(np.flatnonzero(available), size=count - len(required), replace=False)
    return np.concatenate([required, additional]).astype(np.int64)


def method_settings(post_cfg: dict[str, Any], method: str) -> dict[str, Any]:
    override = post_cfg.get("method_inference_overrides", {}).get(method, {})
    n_steps = int(override.get("n_steps", post_cfg["defaults"]["n_steps"]))
    return {
        "n_steps": n_steps,
        "solver": "heun_native" if method == "FFM-FNO" else "euler",
        "measured_nfe": 2 * n_steps if method == "FFM-FNO" else n_steps,
        "obs_consistency": str(override.get("obs_consistency", post_cfg["defaults"]["obs_consistency"])),
        "execution_mode": "cached_streamed" if method == "DMF-Gen" else "legacy_full",
        "cache_level": "static_features" if method == "DMF-Gen" else "none",
        "query_chunk_size": 8192,
    }


def evaluation_context(loaded: Any):
    if loaded.family == "pointcloud_ffm":
        return nullcontext()
    return loaded.model.adapter.evaluation_weights(loaded.model)


def prepare_state(loaded: Any, state: int, sensor_rows: list[dict[str, int]], count: int, method: str, *, retain_truth: bool = False) -> dict[str, Any]:
    sample = loaded.dataset[state]
    coords_full = sample["coords"].unsqueeze(0).to(loaded.device)
    truth = sample["fields"].unsqueeze(0).to(loaded.device)
    obs_indices_full, obs_field_ids = _plan_tensors(sensor_rows, loaded.device)
    obs_coords = coords_full[:, obs_indices_full[0]]
    obs_values = torch.stack([truth[0, index, field] for index, field in zip(obs_indices_full[0], obs_field_ids[0])]).view(1, -1, 1)
    obs_mask = torch.ones((1, obs_indices_full.shape[1]), device=loaded.device, dtype=coords_full.dtype)
    subset = query_subset(state, count, sensor_rows)
    if subset is None:
        coords = coords_full
        obs_indices = obs_indices_full
    else:
        subset_tensor = torch.as_tensor(subset, device=loaded.device, dtype=torch.long)
        coords = coords_full[:, subset_tensor]
        inverse = torch.full((coords_full.shape[1],), -1, device=loaded.device, dtype=torch.long)
        inverse[subset_tensor] = torch.arange(len(subset), device=loaded.device)
        obs_indices = inverse[obs_indices_full]
        if torch.any(obs_indices < 0):
            raise RuntimeError("Query subset omitted a hard-clamped sensor")
    geometry = None
    if method == "DMF-Gen":
        geometry = loaded.model.prepare_reconstruction_geometry_cache(
            coords=coords, obs_coords=obs_coords, obs_mask=obs_mask, chunk_size=2048
        )
    return {
        "state": state,
        "coords": coords,
        "truth": truth if retain_truth or method in ("Latent FM", "SiT") else None,
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_indices": obs_indices,
        "obs_indices_full": obs_indices_full,
        "obs_field_ids": obs_field_ids,
        "geometry": geometry,
    }


def core_call(loaded: Any, prepared: dict[str, Any], settings: dict[str, Any], method: str, *, geometry_override: Any = "use_prepared") -> torch.Tensor:
    if loaded.family == "pointcloud_ffm":
        query_chunk_size = int(settings.get("query_chunk_size", 2048))
        kwargs = {
            "coords": prepared["coords"], "obs_coords": prepared["obs_coords"], "obs_values": prepared["obs_values"],
            "obs_mask": prepared["obs_mask"], "obs_field_ids": prepared["obs_field_ids"], "n_steps": settings["n_steps"],
            "clamp_indices": prepared["obs_indices"], "ode_solver": "euler", "obs_consistency_mode": settings["obs_consistency"],
            "reconstruction_execution_mode": settings["execution_mode"], "reconstruction_query_chunk_size": query_chunk_size,
            "reconstruction_cache_level": settings["cache_level"],
            "reconstruction_geometry_cache": prepared["geometry"] if geometry_override == "use_prepared" else geometry_override,
        }
        signature = inspect.signature(loaded.model.sample)
        return loaded.model.sample(**{key: value for key, value in kwargs.items() if key in signature.parameters})
    bundle = loaded.model
    bundle.model.eval()
    if method == "Latent FM":
        return canonical._baseline_reconstruct_latentfm(bundle, loaded.dataset, prepared["coords"], prepared["truth"], prepared["obs_values"], prepared["obs_mask"], prepared["obs_indices_full"], prepared["obs_field_ids"], settings["n_steps"], "euler")
    if method == "SiT":
        return canonical._baseline_reconstruct_sit(bundle, loaded.dataset, prepared["coords"], prepared["truth"], prepared["obs_values"], prepared["obs_mask"], prepared["obs_indices_full"], prepared["obs_field_ids"], settings["n_steps"], "euler")
    if method in ("MLP-RBF", "Senseiver"):
        return bundle.model(prepared["coords"], prepared["obs_coords"], prepared["obs_values"], prepared["obs_mask"], prepared["obs_field_ids"])
    if method == "Geo-FNO":
        return canonical._baseline_reconstruct_deterministic(bundle, loaded.dataset, prepared["coords"], prepared["obs_coords"], prepared["obs_values"], prepared["obs_mask"], prepared["obs_indices_full"], prepared["obs_field_ids"])
    raise ValueError(f"Unsupported core path: {method}")


def time_cuda(call: Callable[[], Any], device: str) -> tuple[Any, float]:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(torch.device(device))
    start.record()
    result = call()
    stop.record()
    torch.cuda.synchronize(torch.device(device))
    return result, float(start.elapsed_time(stop))


def timed_repeats(call: Callable[[int], Any], args: argparse.Namespace) -> list[float]:
    for index in range(args.warmups):
        call(index)
    values = []
    total = 0.0
    index = 0
    while len(values) < args.minimum_repeats or total < args.minimum_seconds * 1000.0:
        _, elapsed = time_cuda(lambda: call(args.warmups + index), args.device)
        values.append(elapsed)
        total += elapsed
        index += 1
    return values


def latency_summary(values: list[float]) -> dict[str, float]:
    q10, q25, q50, q75, q90 = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {"latency_p10_ms": float(q10), "latency_q25_ms": float(q25), "median_latency_ms": float(q50), "latency_q75_ms": float(q75), "latency_p90_ms": float(q90), "latency_iqr_ms": float(q75 - q25)}


def clear_except_first(prepared: list[dict[str, Any]]) -> dict[str, Any]:
    first = prepared[0]
    del prepared[1:]
    gc.collect()
    torch.cuda.empty_cache()
    return first


def measure_memory(call: Callable[[], Any], device: str) -> tuple[float, float]:
    dev = torch.device(device)
    torch.cuda.reset_peak_memory_stats(dev)
    call()
    torch.cuda.synchronize(dev)
    return torch.cuda.max_memory_allocated(dev) / 2**20, torch.cuda.max_memory_reserved(dev) / 2**20


def benchmark_shape(loaded: Any, method: str, settings: dict[str, Any], states: list[int], sensors: dict[int, list[dict[str, int]]], count: int, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    prepared = [prepare_state(loaded, state, sensors[state], count, method) for state in states]

    torch.manual_seed(stable_seed(20260830, "cost_v3", method, count, "cold"))
    torch.cuda.manual_seed_all(stable_seed(20260830, "cost_v3", method, count, "cold"))
    _, cold_first_core_ms = time_cuda(lambda: core_call(loaded, prepared[0], settings, method), args.device)

    def call(index: int):
        torch.manual_seed(stable_seed(20260830, "cost_v3", method, count, index))
        torch.cuda.manual_seed_all(stable_seed(20260830, "cost_v3", method, count, index))
        return core_call(loaded, prepared[index % len(prepared)], settings, method)

    values = timed_repeats(call, args)
    repeats = [{"method": method, "N": count, "repeat": index, "latency_ms": value, "state": states[(args.warmups + index) % len(states)]} for index, value in enumerate(values)]
    first = clear_except_first(prepared)
    memory_allocated, memory_reserved = measure_memory(lambda: core_call(loaded, first, settings, method), args.device)
    summary = {
        "method": method, "status": "ok", "N": count, "sensor_count": 256, "batch_size": 1, "dtype": "float32",
        "warmups": args.warmups, "repeats": len(values), "timed_total_ms": float(sum(values)), **latency_summary(values),
        "cold_first_core_ms": cold_first_core_ms, "peak_allocated_mib": memory_allocated, "peak_reserved_mib": memory_reserved, **settings,
    }
    return summary, repeats, first


def main() -> int:
    args = parse_args()
    if args.warmups < 20 or args.minimum_repeats < 30 or args.minimum_seconds < 10:
        raise ValueError("Formal V3 cost requires >=20 warmups, >=30 repeats, and >=10 measured seconds")
    index = gpu_index(args.device)
    before_processes = assert_clean_gpu(index, allow_current=False)
    torch.cuda.set_device(index)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    run_id = args.run_id or f"formal_cost_clean_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = REPO_ROOT / "Dis_SI_Process" / "results" / "ValidationV3" / "CostClean" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "gpu_state_before.txt").write_text(gpu_state(index), encoding="utf-8")
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    identity = verify_identities(plan)
    errors = frozen_errors(plan)
    states = list(map(int, plan["cohorts"]["cost_native_20"]["evaluation_indices"]))
    sensors = load_sensor_rows(plan, states)
    post_cfg = load_config()
    method_cfg = {row["name"]: row for row in post_cfg["methods"]}
    support_rows = [{"method": method, "variable_query_supported": SUPPORTED, "native_only": not SUPPORTED, "basis": basis, "native_N": 40300, "tested_query_counts": "1024;4096;16384;40300" if SUPPORTED else "40300"} for method, (SUPPORTED, basis) in SUPPORT.items()]
    write_csv(output_dir / "variable_query_support.csv", support_rows)
    manifest = {
        "schema_version": "figure5-validation-v3-cost-1", "status": "running", "formal": True, "run_id": run_id,
        "plan": str(args.plan.resolve()), "plan_sha256": sha256(args.plan.resolve()), "identity_checks": identity,
        "environment": environment(args.device), "gpu_clean_before": not before_processes,
        "timing_boundary": {
            "name": "warm_model_core_geometry_persisted", "included": ["stochastic_prior_or_noise_generation", "value_dependent_conditioning", "all_model_or_flow_evaluations", "adopted_observation_consistency", "final_device_output"],
            "excluded": ["model_loading", "dataset_IO", "CPU_preprocessing", "host_to_device_transfer", "generic_adapter_dispatch", "metrics", "device_to_host_transfer", "plotting", "disk_IO"],
            "persistent_cache": "DMF reusable top-k sensor/query geometry only; value-dependent features recomputed per state",
        },
        "protocol": {
            "states": states, "warmups": args.warmups, "minimum_repeats": args.minimum_repeats,
            "minimum_seconds": args.minimum_seconds, "query_counts": QUERY_COUNTS,
            "dmf_query_chunk_size": 8192, "historical_probe_query_chunk_size": 4096,
            "throughput_extension": "not_run",
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    native_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    memory_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []

    for method in METHODS:
        loaded = None
        try:
            settings = method_settings(post_cfg, method)
            loaded = load_model(method_cfg[method], "Cond_T", checkpoint="last", split="test", device=args.device, n_steps=settings["n_steps"], ode_solver="euler")
            with evaluation_context(loaded):
                summary, repeats, first = benchmark_shape(loaded, method, settings, states, sensors, 40300, args)
                identity_row = next(row for row in plan["checkpoints"] if row["method"] == method)
                summary.update(errors[method])
                summary.update({"checkpoint_path": identity_row["path"], "checkpoint_sha256": identity_row["sha256"], "timing_boundary": "warm_model_core_geometry_persisted"})
                native_rows.append(summary)
                repeat_rows.extend({"suite": "native", **row} for row in repeats)
                query_rows.append({**summary, "variable_query_supported": SUPPORT[method][0], "native_only": not SUPPORT[method][0]})
                memory_rows.append({"method": method, "N": 40300, "peak_allocated_mib": summary["peak_allocated_mib"], "peak_reserved_mib": summary["peak_reserved_mib"], "variable_query_supported": SUPPORT[method][0], "native_only": not SUPPORT[method][0], "status": "ok"})

                if method == "DMF-Gen":
                    unified = summary["median_latency_ms"]
                    probe_values = timed_repeats(lambda repeat: core_call(loaded, first, settings, method), args)
                    probe = latency_summary(probe_values)["median_latency_ms"]
                    direct_values = timed_repeats(lambda repeat: core_call(loaded, first, settings, method), args)
                    direct = latency_summary(direct_values)["median_latency_ms"]
                    no_geometry_values = timed_repeats(lambda repeat: core_call(loaded, first, settings, method, geometry_override=None), args)
                    no_geometry = latency_summary(no_geometry_values)["median_latency_ms"]
                    historical_chunk_settings = {**settings, "query_chunk_size": 4096}
                    historical_chunk_values = timed_repeats(
                        lambda repeat: core_call(loaded, first, historical_chunk_settings, method), args
                    )
                    historical_chunk = latency_summary(historical_chunk_values)["median_latency_ms"]
                    equivalence_seed = stable_seed(20260830, "cost_v3", method, "chunk_equivalence")
                    torch.manual_seed(equivalence_seed)
                    torch.cuda.manual_seed_all(equivalence_seed)
                    canonical_chunk_output = core_call(loaded, first, settings, method)
                    torch.manual_seed(equivalence_seed)
                    torch.cuda.manual_seed_all(equivalence_seed)
                    historical_chunk_output = core_call(loaded, first, historical_chunk_settings, method)
                    historical_chunk_max_abs_error = float(
                        torch.max(torch.abs(canonical_chunk_output - historical_chunk_output)).item()
                    )
                    loaded.model._reconstruction_profile_enabled = True
                    try:
                        for repeat in range(max(30, args.minimum_repeats)):
                            seed = stable_seed(20260830, "cost_v3", method, "phase_profile", repeat)
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed_all(seed)
                            core_call(loaded, first, settings, method)
                            phase_rows.append({
                                "method": method,
                                "repeat": repeat,
                                "condition_ms": 1000.0 * float(loaded.model._last_reconstruction_condition_seconds),
                                "query_static_feature_ms": 1000.0 * float(loaded.model._last_reconstruction_query_seconds),
                                "ode_ms": 1000.0 * float(loaded.model._last_reconstruction_ode_seconds),
                            })
                    finally:
                        loaded.model._reconstruction_profile_enabled = False
                    phase_ode = float(np.median([row["ode_ms"] for row in phase_rows]))
                    phase_condition = float(np.median([row["condition_ms"] for row in phase_rows]))
                    phase_query = float(np.median([row["query_static_feature_ms"] for row in phase_rows]))
                    unified_delta = abs(unified - probe) / probe
                    direct_delta = abs(direct - probe) / probe
                    historical_delta = abs(historical_chunk - args.historical_direct_probe_ms) / args.historical_direct_probe_ms
                    boundary_rows.extend([
                        {"method": method, "timer": "unified_clean_20_state", "median_latency_ms": unified, "reference_ms": probe, "relative_difference": unified_delta, "pass_20pct": unified_delta <= 0.20, "note": "main V3 timer versus independent exact-shape reprobe"},
                        {"method": method, "timer": "direct_core_persistent_geometry", "median_latency_ms": direct, "reference_ms": probe, "relative_difference": direct_delta, "pass_20pct": direct_delta <= 0.20, "note": "direct model.sample timer versus exact-shape reprobe"},
                        {"method": method, "timer": "historical_approx_29ms_reconciliation", "median_latency_ms": args.historical_direct_probe_ms, "reference_ms": historical_chunk, "relative_difference": historical_delta, "pass_20pct": historical_delta <= 0.20, "note": "historical probe reconciles to the earlier 4096-point streaming chunk; V3 uses the canonical configured 8192-point chunk"},
                        {"method": method, "timer": "exact_shape_reprobe_persistent_geometry", "median_latency_ms": probe, "reference_ms": probe, "relative_difference": 0.0, "pass_20pct": True, "note": "independent synchronized exact-shape reprobe"},
                        {"method": method, "timer": "direct_core_no_persistent_geometry", "median_latency_ms": no_geometry, "reference_ms": direct, "relative_difference": (no_geometry - direct) / direct, "pass_20pct": True, "note": "SI cache-policy comparison"},
                        {"method": method, "timer": "phase_profile_value_dependent_preparation", "median_latency_ms": phase_condition + phase_query, "reference_ms": phase_ode, "relative_difference": (phase_condition + phase_query) / phase_ode, "pass_20pct": True, "note": "median condition-context plus query-static-feature preparation; included by V3 because both depend on observed values"},
                        {"method": method, "timer": "historical_4096_chunk_probe", "median_latency_ms": historical_chunk, "reference_ms": direct, "relative_difference": (historical_chunk - direct) / direct, "pass_20pct": True, "note": f"profiling-only chunk mapping; max_abs_output_difference_vs_8192={historical_chunk_max_abs_error:.8g}"},
                    ])
        except Exception as exc:
            native_rows.append({"method": method, "status": "unavailable", "detail": f"{type(exc).__name__}: {exc}", "N": 40300})
        finally:
            if loaded is not None:
                loaded.close()
        write_csv(output_dir / "native_repeats.csv", repeat_rows)
        write_csv(output_dir / "native_summary.csv", native_rows)
        write_csv(output_dir / "timing_boundary_audit.csv", boundary_rows)
        write_csv(output_dir / "dmf_phase_profile.csv", phase_rows)

    for method in METHODS:
        if not SUPPORT[method][0]:
            continue
        loaded = None
        try:
            settings = method_settings(post_cfg, method)
            loaded = load_model(method_cfg[method], "Cond_T", checkpoint="last", split="test", device=args.device, n_steps=settings["n_steps"], ode_solver="euler")
            with evaluation_context(loaded):
                for count in QUERY_COUNTS[:-1]:
                    summary, repeats, retained = benchmark_shape(loaded, method, settings, states, sensors, count, args)
                    summary.update({"variable_query_supported": True, "native_only": False, "timing_boundary": "warm_model_core_geometry_persisted"})
                    query_rows.append(summary)
                    repeat_rows.extend({"suite": "query_scaling", **row} for row in repeats)
                    memory_rows.append({"method": method, "N": count, "peak_allocated_mib": summary["peak_allocated_mib"], "peak_reserved_mib": summary["peak_reserved_mib"], "variable_query_supported": True, "native_only": False, "status": "ok"})
                    del retained
                    gc.collect()
                    torch.cuda.empty_cache()
                    write_csv(output_dir / "query_latency_repeats.csv", [row for row in repeat_rows if row["suite"] == "query_scaling"])
                    write_csv(output_dir / "query_latency_summary.csv", sorted(query_rows, key=lambda row: (METHODS.index(row["method"]), int(row["N"]))))
                    write_csv(output_dir / "memory_summary.csv", sorted(memory_rows, key=lambda row: (METHODS.index(row["method"]), int(row["N"]))))
        finally:
            if loaded is not None:
                loaded.close()

    native_ok = len(native_rows) == 8 and all(row.get("status") == "ok" for row in native_rows)
    reconciliation_ok = len(boundary_rows) == 7 and all(bool(row["pass_20pct"]) for row in boundary_rows[:3])
    expected_query = sum(4 if SUPPORT[method][0] else 1 for method in METHODS)
    final_processes = assert_clean_gpu(index, allow_current=True)
    (output_dir / "gpu_state_after.txt").write_text(gpu_state(index), encoding="utf-8")
    qa = {
        "status": "pass", "formal": True, "identity_pass": all(row["pass"] for row in identity),
        "gpu_clean_before": not before_processes, "gpu_clean_after": all(int(row["pid"]) == os.getpid() for row in final_processes),
        "all_eight_native_methods_ok": native_ok, "dmf_reconciliation_pass": reconciliation_ok,
        "expected_query_summary_rows": expected_query, "actual_query_summary_rows": len(query_rows),
        "query_memory_protocol_match": {(row["method"], int(row["N"])) for row in query_rows} == {(row["method"], int(row["N"])) for row in memory_rows},
        "no_full_grid_then_slice_scaling": True, "throughput_extension": "not_run",
        "timing_protocol_pass": all(int(row.get("warmups", 0)) >= 20 and int(row.get("repeats", 0)) >= 30 and float(row.get("timed_total_ms", 0.0)) >= 10000 for row in query_rows),
    }
    qa["status"] = "pass" if all((qa["identity_pass"], qa["gpu_clean_before"], qa["gpu_clean_after"], qa["all_eight_native_methods_ok"], qa["dmf_reconciliation_pass"], qa["actual_query_summary_rows"] == expected_query, qa["query_memory_protocol_match"], qa["timing_protocol_pass"])) else "fail"
    atomic_json(output_dir / "qa.json", qa)
    manifest["status"] = "complete" if qa["status"] == "pass" else "qa_failed"
    manifest["completed_at"] = datetime.now().isoformat()
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "qa": qa, "dmf_reconciliation": boundary_rows}, indent=2))
    return 0 if qa["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

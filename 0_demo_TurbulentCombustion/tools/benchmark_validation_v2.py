#!/usr/bin/env python
"""Run the frozen Figure 5 validation cost suites on canonical adapters."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
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
from common.model_loader import load_model  # noqa: E402


UNOBSERVED = (0, 1, 3, 4)
METHOD_ORDER = ("DMF-Gen", "FFM-FNO", "FFM-Perceiver", "Latent FM", "SiT", "MLP-RBF", "Geo-FNO", "Senseiver")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--suite", choices=("native_methods", "dmf_query_memory", "dmf_nfe_error", "dmf_m_sweep", "all"), required=True)
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--warmups", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--minimum-seconds", type=float)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def capture_environment(device: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": device}
    if torch.cuda.is_available():
        payload["gpu_name"] = torch.cuda.get_device_name(torch.device(device).index or 0)
    try:
        payload["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader"], text=True
        ).strip()
    except Exception as exc:  # pragma: no cover
        payload["nvidia_smi_error"] = str(exc)
    return payload


def load_sensor_rows(plan: dict[str, Any], states: list[int]) -> dict[int, list[dict[str, Any]]]:
    groups = {state: [] for state in states}
    wanted = set(states)
    with resolve(plan["sensor_plan"]["path"]).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != "Cond_T" or int(row["snapshot"]) not in wanted:
                continue
            state = int(row["snapshot"])
            groups[state].append({"sensor_order": int(row["sensor_order"]), "field_index": 2, "point_index": int(row["point_index"]), "sensor_seed": int(row["sensor_seed"])})
    for state, rows in groups.items():
        rows.sort(key=lambda row: row["sensor_order"])
        if len(rows) != 256:
            raise RuntimeError(f"Incomplete adopted sensor plan for state {state}")
    return groups


def nested_sensor_rows(plan: dict[str, Any], state: int, adopted: list[dict[str, Any]], m: int) -> list[dict[str, Any]]:
    if m <= 256:
        return [dict(row) for row in adopted[:m]]
    rule = plan["sensor_plan"]["nested"]["M384"]
    seed = stable_seed(int(rule["extension_seed_base"]), "M384_extension", state)
    rng = np.random.default_rng(seed)
    base = [int(row["point_index"]) for row in adopted]
    mask = np.ones(40300, dtype=bool)
    mask[base] = False
    extension = rng.choice(np.flatnonzero(mask), 128, replace=False)
    return [dict(row) for row in adopted] + [
        {"sensor_order": 256 + i, "field_index": 2, "point_index": int(point), "sensor_seed": adopted[0]["sensor_seed"]}
        for i, point in enumerate(extension)
    ]


def query_subset(state: int, n: int, sensors: list[dict[str, Any]]) -> np.ndarray | None:
    if n == 40300:
        return None
    required = np.asarray([int(row["point_index"]) for row in sensors], dtype=np.int64)
    mask = np.ones(40300, dtype=bool)
    mask[required] = False
    candidates = np.flatnonzero(mask)
    rng = np.random.default_rng(stable_seed(20260830, "query_subset", state, n))
    additional = rng.choice(candidates, size=n - len(required), replace=False)
    return np.concatenate([required, additional]).astype(np.int64)


def summarize_latency(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    q10, q25, q50, q75, q90 = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {"latency_p10_ms": float(q10), "latency_q25_ms": float(q25), "median_latency_ms": float(q50), "latency_q75_ms": float(q75), "latency_p90_ms": float(q90), "latency_iqr_ms": float(q75 - q25)}


def moving_block_ci(values: np.ndarray, plan: dict[str, Any], salt: str) -> tuple[float, float]:
    bootstrap = plan["bootstrap"]
    values = np.asarray(values, dtype=float)
    n = len(values)
    block = min(int(bootstrap["block_length"]), n)
    rng = np.random.default_rng(stable_seed(int(bootstrap["seed"]), salt))
    samples = []
    for _ in range(int(bootstrap["replicates"])):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = np.concatenate([(np.arange(block) + start) % n for start in starts])[:n]
        samples.append(float(np.mean(values[idx])))
    alpha = (1.0 - float(bootstrap["confidence_level"])) / 2
    return tuple(float(x) for x in np.quantile(samples, [alpha, 1 - alpha]))


def time_cuda_call(call: Callable[[], Any], device: str) -> tuple[Any, float]:
    if not torch.cuda.is_available():
        start = time.perf_counter()
        out = call()
        return out, (time.perf_counter() - start) * 1000.0
    dev = torch.device(device)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(dev)
    start.record()
    out = call()
    stop.record()
    torch.cuda.synchronize(dev)
    return out, float(start.elapsed_time(stop))


def timed_repeats(call_for_repeat: Callable[[int], Any], *, warmups: int, minimum_repeats: int, minimum_seconds: float, device: str) -> tuple[list[float], Any]:
    last = None
    for index in range(warmups):
        last = call_for_repeat(index)
    values: list[float] = []
    elapsed = 0.0
    index = 0
    while len(values) < minimum_repeats or elapsed < minimum_seconds * 1000.0:
        last, milliseconds = time_cuda_call(lambda: call_for_repeat(warmups + index), device)
        values.append(milliseconds)
        elapsed += milliseconds
        index += 1
    return values, last


def frozen_errors(plan: dict[str, Any]) -> dict[str, dict[str, float]]:
    per_state = read_csv(resolve(plan["frozen_accuracy"]["per_state_path"]))
    grouped: dict[str, list[tuple[int, float]]] = {method: [] for method in METHOD_ORDER}
    for row in per_state:
        if row["condition"] == "Cond_T" and row["field"] == "Unobserved_mean" and row["status"] == "ok":
            grouped[row["method"]].append((int(row["snapshot"]), float(row["physical_rel_l2"])))
    output: dict[str, dict[str, float]] = {}
    for method, pairs in grouped.items():
        pairs.sort()
        values = np.asarray([value for _, value in pairs])
        if len(values) != 1000:
            raise RuntimeError(f"Frozen FieldL2 has {len(values)} Cond_T Unobserved_mean rows for {method}, expected 1000")
        low, high = moving_block_ci(values, plan, f"frozen_error|{method}")
        output[method] = {"error": float(np.mean(values)), "error_ci_low": low, "error_ci_high": high, "error_n": len(values)}
    return output


def verify_identities(plan: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    for item in plan["checkpoints"]:
        actual = sha256(resolve(item["path"]))
        checks.append({"method": item["method"], "path": item["path"], "expected_sha256": item["sha256"], "actual_sha256": actual, "pass": actual == item["sha256"]})
    for key in ("summary", "per_state", "reconstruction_manifest"):
        path = plan["frozen_accuracy"][f"{key}_path"]
        expected = plan["frozen_accuracy"][f"{key}_sha256"]
        actual = sha256(resolve(path))
        checks.append({"method": f"frozen_{key}", "path": path, "expected_sha256": expected, "actual_sha256": actual, "pass": actual == expected})
    manifest_rows = read_csv(resolve(plan["frozen_accuracy"]["reconstruction_manifest_path"]))
    checkpoint_by_method = {item["method"]: resolve(item["path"]).resolve() for item in plan["checkpoints"]}
    for method, checkpoint in checkpoint_by_method.items():
        rows = [row for row in manifest_rows if row["method"] == method and row["condition"] == "Cond_T" and row["status"] == "ok"]
        paths = {Path(row["checkpoint_path"]).resolve() for row in rows}
        passed = len(rows) == 1000 and paths == {checkpoint} and {row["checkpoint_name"] for row in rows} == {"last.pt"}
        checks.append({"method": f"FieldL2_join_{method}", "path": str(checkpoint), "expected_sha256": next(item["sha256"] for item in plan["checkpoints"] if item["method"] == method), "actual_sha256": next(item["sha256"] for item in plan["checkpoints"] if item["method"] == method), "manifest_rows": len(rows), "pass": passed})
    if not all(row["pass"] for row in checks):
        raise RuntimeError("One or more frozen benchmark identities do not match validation_v1.yaml")
    return checks


def canonical_methods(post_cfg: dict[str, Any], selected: list[str]) -> list[dict[str, Any]]:
    wanted = set(METHOD_ORDER if "all" in selected else selected)
    result = []
    for name in METHOD_ORDER:
        if name in wanted:
            result.append(next(item for item in post_cfg["methods"] if item["name"] == name))
    return result


def run_native_methods(ctx: dict[str, Any], selected: list[str]) -> None:
    plan, args, post_cfg = ctx["plan"], ctx["args"], ctx["post_cfg"]
    states = [int(x) for x in plan["cohorts"]["cost_native_20"]["evaluation_indices"]]
    if args.pilot:
        states = states[:2]
    sensors = load_sensor_rows(plan, states)
    errors = frozen_errors(plan)
    for method in canonical_methods(post_cfg, selected):
        name = method["name"]
        status = "ok"
        detail = ""
        loaded = None
        try:
            n_steps = int(post_cfg.get("method_inference_overrides", {}).get(name, {}).get("n_steps", post_cfg["defaults"]["n_steps"]))
            consistency = str(post_cfg.get("method_inference_overrides", {}).get(name, {}).get("obs_consistency", post_cfg["defaults"]["obs_consistency"]))
            execution_mode = "cached_streamed" if name == "DMF-Gen" else "legacy_full"
            loaded = load_model(method, "Cond_T", checkpoint="last", split="test", device=args.device, n_steps=n_steps, ode_solver="euler")
            def call(index: int):
                state = states[index % len(states)]
                seed = stable_seed(int(plan["generation_seeds"]["base"]), "benchmark", "native_methods", name, index)
                return loaded.reconstruct(state, {"cond_fields": [2], "n_obs": [256]}, sensors[state], n_steps=n_steps, ode_solver="euler", obs_consistency=consistency, generation_seed=seed, reconstruction_execution_mode=execution_mode, reconstruction_query_chunk_size=int(plan["inference"]["query_chunk_size"]))
            call(0)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            latency, _ = timed_repeats(call, warmups=ctx["warmups"], minimum_repeats=ctx["repeats"], minimum_seconds=ctx["minimum_seconds"], device=args.device)
            memory_allocated = torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**20 if torch.cuda.is_available() else np.nan
            memory_reserved = torch.cuda.max_memory_reserved(torch.device(args.device)) / 2**20 if torch.cuda.is_available() else np.nan
            identity = next(item for item in plan["checkpoints"] if item["method"] == name)
            row = {"suite": "native_methods", "method": name, "status": status, "detail": detail, "condition": "Cond_T", "sensor_count": 256, "N": 40300, "batch_size": 1, "dtype": "float32", "warmups": ctx["warmups"], "repeats": len(latency), "timed_total_ms": float(sum(latency)), **summarize_latency(latency), **errors[name], "peak_allocated_mib": memory_allocated, "peak_reserved_mib": memory_reserved, "checkpoint_path": identity["path"], "checkpoint_sha256": identity["sha256"], "n_steps": n_steps, "measured_nfe": n_steps, "execution_mode": execution_mode}
            ctx["summary"].append(row)
            ctx["repeats_rows"].extend({"suite": "native_methods", "method": name, "repeat": i, "latency_ms": value, "status": "ok"} for i, value in enumerate(latency))
        except Exception as exc:
            status, detail = "unavailable", f"{type(exc).__name__}: {exc}"
            ctx["summary"].append({"suite": "native_methods", "method": name, "status": status, "detail": detail, "condition": "Cond_T", "sensor_count": 256, "N": 40300})
            print(f"[UNAVAILABLE] {name}: {detail}", flush=True)
        finally:
            if loaded is not None:
                loaded.close()
        write_csv(ctx["out"] / "benchmark_repeats.csv", ctx["repeats_rows"])
        write_csv(ctx["out"] / "benchmark_summary.csv", ctx["summary"])


def load_dmf(ctx: dict[str, Any]):
    method = next(item for item in ctx["post_cfg"]["methods"] if item["name"] == "DMF-Gen")
    return load_model(method, "Cond_T", checkpoint="last", split="test", device=ctx["args"].device, n_steps=2, ode_solver="euler")


def run_query_memory(ctx: dict[str, Any]) -> None:
    plan, args = ctx["plan"], ctx["args"]
    states = [int(x) for x in plan["cohorts"]["cost_native_20"]["evaluation_indices"]]
    if args.pilot:
        states = states[:2]
    sensors = load_sensor_rows(plan, states)
    loaded = load_dmf(ctx)
    try:
        first = states[0]
        seed = stable_seed(int(plan["generation_seeds"]["base"]), "benchmark", "cache_equivalence", first)
        legacy = loaded.reconstruct(first, {"cond_fields": [2], "n_obs": [256]}, sensors[first], n_steps=2, ode_solver="euler", obs_consistency="default_hard", generation_seed=seed, reconstruction_execution_mode="legacy_full")
        cached = loaded.reconstruct(first, {"cond_fields": [2], "n_obs": [256]}, sensors[first], n_steps=2, ode_solver="euler", obs_consistency="default_hard", generation_seed=seed, reconstruction_execution_mode="cached_streamed", reconstruction_query_chunk_size=int(plan["inference"]["query_chunk_size"]))
        difference = float(torch.max(torch.abs(legacy["recon"] - cached["recon"])).item())
        ctx["equivalence"].append({"suite": "dmf_query_memory", "state": first, "max_abs_error": difference, "atol": plan["qa_tolerances"]["cache_equivalence_atol"], "pass": difference <= float(plan["qa_tolerances"]["cache_equivalence_atol"])})
        if not ctx["equivalence"][-1]["pass"]:
            raise RuntimeError(f"cached_streamed equivalence failed: max abs {difference}")
        counts = [int(x) for x in plan["cost_protocol"]["query_counts"]]
        for n in counts:
            def call(index: int):
                state = states[index % len(states)]
                seed_i = stable_seed(int(plan["generation_seeds"]["base"]), "benchmark", "dmf_query_memory", n, index)
                subset = query_subset(state, n, sensors[state])
                return loaded.reconstruct(state, {"cond_fields": [2], "n_obs": [256]}, sensors[state], n_steps=2, ode_solver="euler", obs_consistency="default_hard", generation_seed=seed_i, reconstruction_execution_mode="cached_streamed", reconstruction_query_chunk_size=int(plan["inference"]["query_chunk_size"]), query_indices=subset)
            call(0)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(torch.device(args.device))
            latency, _ = timed_repeats(call, warmups=ctx["warmups"], minimum_repeats=ctx["repeats"], minimum_seconds=ctx["minimum_seconds"], device=args.device)
            ctx["summary"].append({"suite": "dmf_query_memory", "method": "DMF-Gen", "status": "ok", "condition": "Cond_T", "sensor_count": 256, "N": n, "batch_size": 1, "dtype": "float32", "warmups": ctx["warmups"], "repeats": len(latency), "timed_total_ms": float(sum(latency)), **summarize_latency(latency), "peak_allocated_mib": torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**20, "peak_reserved_mib": torch.cuda.max_memory_reserved(torch.device(args.device)) / 2**20, "execution_mode": "cached_streamed", "query_chunk_size": int(plan["inference"]["query_chunk_size"]), "measured_nfe": 2})
            ctx["repeats_rows"].extend({"suite": "dmf_query_memory", "method": "DMF-Gen", "N": n, "repeat": i, "latency_ms": value, "status": "ok"} for i, value in enumerate(latency))
            write_csv(ctx["out"] / "benchmark_repeats.csv", ctx["repeats_rows"])
            write_csv(ctx["out"] / "benchmark_summary.csv", ctx["summary"])
    finally:
        loaded.close()


def unobserved_error(out: dict[str, Any], mean: np.ndarray, std: np.ndarray) -> float:
    truth = out["truth"][0].detach().cpu().numpy() * std + mean
    recon = out["recon"][0].detach().cpu().numpy() * std + mean
    return float(np.mean([np.linalg.norm(recon[:, i] - truth[:, i]) / max(np.linalg.norm(truth[:, i]), 1e-12) for i in UNOBSERVED]))


def run_nfe_error(ctx: dict[str, Any]) -> None:
    plan, args = ctx["plan"], ctx["args"]
    states = [int(x) for x in plan["cohorts"]["cost_nfe_50"]["evaluation_indices"]]
    if args.pilot:
        states = states[:3]
    sensors = load_sensor_rows(plan, states)
    mean = np.asarray(plan["dataset_statistics"]["mean"], dtype=np.float32)
    std = np.asarray(plan["dataset_statistics"]["std"], dtype=np.float32)
    loaded = load_dmf(ctx)
    try:
        for nfe in (1, 2, 4, 8):
            errors = []
            latency = []
            for position, state in enumerate(states):
                seed = stable_seed(int(plan["generation_seeds"]["base"]), "benchmark", "dmf_nfe_error", state)
                call = lambda: loaded.reconstruct(state, {"cond_fields": [2], "n_obs": [256]}, sensors[state], n_steps=nfe, ode_solver="euler", obs_consistency="default_hard", generation_seed=seed, reconstruction_execution_mode="cached_streamed", reconstruction_query_chunk_size=int(plan["inference"]["query_chunk_size"]))
                if position == 0:
                    for _ in range(ctx["warmups"]):
                        call()
                out, milliseconds = time_cuda_call(call, args.device)
                errors.append(unobserved_error(out, mean, std))
                latency.append(milliseconds)
            extra = 0
            while len(latency) < ctx["repeats"] or sum(latency) < ctx["minimum_seconds"] * 1000.0:
                state = states[extra % len(states)]
                seed = stable_seed(int(plan["generation_seeds"]["base"]), "benchmark", "dmf_nfe_error", state)
                call = lambda: loaded.reconstruct(state, {"cond_fields": [2], "n_obs": [256]}, sensors[state], n_steps=nfe, ode_solver="euler", obs_consistency="default_hard", generation_seed=seed, reconstruction_execution_mode="cached_streamed", reconstruction_query_chunk_size=int(plan["inference"]["query_chunk_size"]))
                _, milliseconds = time_cuda_call(call, args.device)
                latency.append(milliseconds)
                extra += 1
            error_low, error_high = moving_block_ci(np.asarray(errors), plan, f"nfe_error|{nfe}")
            row = {"suite": "dmf_nfe_error", "method": "DMF-Gen", "status": "ok", "solver": "euler", "n_steps": nfe, "measured_nfe": nfe, "state_count": len(states), "unobserved_mean_error": float(np.mean(errors)), "error_ci_low": error_low, "error_ci_high": error_high, "timed_total_ms": float(sum(latency)), **summarize_latency(latency)}
            ctx["nfe_rows"].append(row)
            ctx["summary"].append({**row, "condition": "Cond_T", "sensor_count": 256, "N": 40300, "batch_size": 1, "dtype": "float32", "warmups": ctx["warmups"], "repeats": len(latency)})
            ctx["repeats_rows"].extend({"suite": "dmf_nfe_error", "method": "DMF-Gen", "measured_nfe": nfe, "state": states[i % len(states)], "repeat": i, "latency_ms": value, "status": "ok"} for i, value in enumerate(latency))
            write_csv(ctx["out"] / "nfe_error.csv", ctx["nfe_rows"])
            write_csv(ctx["out"] / "benchmark_repeats.csv", ctx["repeats_rows"])
            write_csv(ctx["out"] / "benchmark_summary.csv", ctx["summary"])
    finally:
        loaded.close()


def main() -> int:
    args = parse_args()
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "figure5-validation-v1":
        raise ValueError("Unsupported validation plan schema")
    protocol = plan["cost_protocol"]
    warmups = args.warmups if args.warmups is not None else int(protocol["warmups"])
    repeats = args.repeats if args.repeats is not None else int(protocol["minimum_repeats"])
    minimum_seconds = args.minimum_seconds if args.minimum_seconds is not None else float(protocol["minimum_timed_seconds"])
    if not args.pilot and (warmups < 10 or repeats < 30 or minimum_seconds < 10.0):
        raise ValueError("Formal protocol requires >=10 warmups, >=30 repeats, and >=10 timed seconds")
    run_id = args.run_id or f"cost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out = resolve(plan["output_root"]) / "Cost" / run_id
    out.mkdir(parents=True, exist_ok=False)
    identity = verify_identities(plan)
    manifest = {"schema_version": "validation-v2-cost-1", "status": "running", "formal": not args.pilot, "run_id": run_id, "suite": args.suite, "plan": str(args.plan.resolve()), "plan_sha256": sha256(args.plan.resolve()), "identity_checks": identity, "environment": capture_environment(args.device), "protocol": {"warmups": warmups, "minimum_repeats": repeats, "minimum_timed_seconds": minimum_seconds, "synchronize_cuda": True}}
    atomic_json(out / "manifest.json", manifest)
    ctx = {"args": args, "plan": plan, "post_cfg": load_config(), "out": out, "warmups": warmups, "repeats": repeats, "minimum_seconds": minimum_seconds, "summary": [], "repeats_rows": [], "nfe_rows": [], "equivalence": []}
    suites = ("native_methods", "dmf_query_memory", "dmf_nfe_error") if args.suite == "all" else (args.suite,)
    for suite in suites:
        if suite == "native_methods":
            run_native_methods(ctx, args.methods)
        elif suite == "dmf_query_memory":
            run_query_memory(ctx)
        elif suite == "dmf_nfe_error":
            run_nfe_error(ctx)
        else:
            raise NotImplementedError("dmf_m_sweep is an SI-only optional suite and is not implemented in the minimal runner")
    write_csv(out / "benchmark_repeats.csv", ctx["repeats_rows"])
    write_csv(out / "benchmark_summary.csv", ctx["summary"])
    write_csv(out / "nfe_error.csv", ctx["nfe_rows"])
    write_csv(out / "cache_equivalence.csv", ctx["equivalence"])
    resolved_methods = {row["method"]: row["status"] for row in ctx["summary"] if row.get("suite") == "native_methods"}
    protocol_rows = [row for row in ctx["summary"] if row.get("status") == "ok"]
    qa = {"status": "pass", "formal": not args.pilot, "identity_pass": all(row["pass"] for row in identity), "native_method_status": resolved_methods, "all_requested_native_methods_resolved": args.suite not in ("native_methods", "all") or len(resolved_methods) == len(canonical_methods(ctx["post_cfg"], args.methods)), "cache_equivalence": ctx["equivalence"], "finite_summary": all(np.isfinite(float(value)) for row in ctx["summary"] for key, value in row.items() if key not in {"suite", "method", "status", "detail", "condition", "dtype", "checkpoint_path", "checkpoint_sha256", "execution_mode", "solver"} and value not in ("", None)), "timing_protocol_pass": args.pilot or all(int(row.get("repeats", 0)) >= repeats and float(row.get("timed_total_ms", 0.0)) >= minimum_seconds * 1000.0 for row in protocol_rows)}
    qa["status"] = "pass" if qa["identity_pass"] and qa["all_requested_native_methods_resolved"] and qa["finite_summary"] and qa["timing_protocol_pass"] and all(row["pass"] for row in ctx["equivalence"]) else "fail"
    atomic_json(out / "qa.json", qa)
    manifest["status"] = "complete" if qa["status"] == "pass" else "qa_failed"
    manifest["completed_at"] = datetime.now().isoformat()
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(out), "qa": qa}, indent=2))
    return 0 if qa["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Scenario-configured CRPS and spread/error validation for matched Figure 5 panels a/b."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import yaml
from scipy.stats import spearmanr

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
SUPER_ROOT = REPO_ROOT / "1_SubTask_SuperResolution"
SCRIPT_ROOT = SUPER_ROOT / "Save_TrainedModel" / "_TrainedModels" / "_Scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from common.config import load_config, method_items, stable_seed
from common.dataset_loader import find_snapshot
from common.model_loader import checkpoint_digest, load_model


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "zeroh_matched_v42.yaml")
    parser.add_argument("--job", choices=("PILOT", "FORMAL"), required=True)
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--device", default="cuda:2")
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
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _gpu_uuid(index: int) -> str:
    return subprocess.check_output(["nvidia-smi", "-i", str(index), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()


def _compute_processes() -> list[dict[str, str]]:
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip()
    rows = []
    for line in raw.splitlines():
        if line.strip():
            uuid, pid, name, memory = (value.strip() for value in line.split(",", 3))
            rows.append({"uuid": uuid, "pid": pid, "name": name, "memory": memory})
    return rows


def _assert_clean(device: str, allow_current: bool) -> list[dict[str, str]]:
    index = torch.device(device).index
    if index is None:
        raise ValueError("An explicit CUDA device is required")
    rows = [row for row in _compute_processes() if row["uuid"] == _gpu_uuid(index)]
    foreign = [row for row in rows if not (allow_current and int(row["pid"]) == os.getpid())]
    if foreign:
        raise RuntimeError(f"GPU {index} is not clean: {foreign}")
    return rows


def _sensor_groups(path: Path, state_count: int, sensor_count: int) -> list[tuple[dict[str, Any], list[dict[str, int]]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            state = int(row["snapshot_index"])
            if state < state_count and int(row["sensor_order"]) < sensor_count:
                grouped.setdefault(state, []).append(row)
    output = []
    for state in range(state_count):
        rows = sorted(grouped.get(state, []), key=lambda row: int(row["sensor_order"]))
        if len(rows) != sensor_count:
            raise RuntimeError(f"Snapshot {state} has {len(rows)} sensors, expected {sensor_count}")
        metadata = {key: rows[0][key] for key in ("snapshot_index", "dataset_index", "case_id", "time_index", "physical_time", "sensor_plan_id")}
        output.append((metadata, [{"sensor_order": int(row["sensor_order"]), "point_index": int(row["point_index"])} for row in rows]))
    if len({row[0]["case_id"] for row in output}) != state_count:
        raise RuntimeError("UQ cohort must contain unique physical cases")
    return output


def _crps(draws: np.ndarray, truth: np.ndarray) -> float:
    draws64, truth64 = np.asarray(draws, dtype=np.float64), np.asarray(truth, dtype=np.float64)
    count = draws64.shape[0]
    absolute = np.mean(np.abs(draws64 - truth64[None, :]), axis=0)
    ordered = np.sort(draws64, axis=0)
    coefficients = 2.0 * np.arange(1, count + 1, dtype=np.float64) - count - 1.0
    dispersion = np.sum(coefficients[:, None] * ordered, axis=0) / (count * count)
    return float(np.mean(absolute - dispersion))


def _bootstrap(values: np.ndarray, spec: dict[str, Any], salt: str, statistic) -> tuple[float, float]:
    rng = np.random.default_rng(stable_seed(int(spec["seed"]), salt))
    samples = np.empty(int(spec["replicates"]), dtype=float)
    for index in range(len(samples)):
        selected = rng.integers(0, len(values), size=len(values))
        samples[index] = statistic(selected)
    alpha = (1.0 - float(spec["confidence_level"])) / 2.0
    return tuple(float(value) for value in np.nanquantile(samples, [alpha, 1.0 - alpha]))


def main() -> int:
    args = _args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    methods = list(config["scenario"]["generative_methods"])
    if args.methods != ["all"]:
        methods = [method for method in methods if method in set(args.methods)]
    if not methods:
        raise ValueError("No configured generative method selected")
    namespace = str(config.get("run_namespace", "zeroh"))
    run_prefix = str(config.get("run_prefix", "zeroh"))
    formal = args.job == "FORMAL" and set(methods) == set(config["scenario"]["generative_methods"])
    states = int(config["cohort"]["uq_states"] if formal else config["cohort"]["pilot_states"])
    draws = int(config["cohort"]["formal_draws"] if formal else config["cohort"]["pilot_draws"])
    run_id = args.run_id or f"{run_prefix}_uq_{args.job.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = _resolve(config["uq_output_root"]) / run_id
    if run_dir.exists():
        raise RuntimeError(f"Refusing to overwrite {run_dir}")
    _assert_clean(args.device, allow_current=False)
    run_dir.mkdir(parents=True)

    sensor_plan = _resolve(config["cohort"]["sensor_plan"])
    if _sha(sensor_plan) != config["cohort"]["sensor_plan_sha256"]:
        raise RuntimeError("Scenario sensor-plan identity mismatch")
    cohort = _sensor_groups(sensor_plan, states, int(config["scenario"]["sensor_count"]))
    checkpoint_checks = []
    for method in methods:
        path = _resolve(config["checkpoints"][method]["path"])
        actual = _sha(path)
        checkpoint_checks.append({"method": method, "path": str(path), "expected": config["checkpoints"][method]["sha256"], "actual": actual, "pass": actual == config["checkpoints"][method]["sha256"]})
    if not all(row["pass"] for row in checkpoint_checks):
        raise RuntimeError("Scenario checkpoint identity mismatch")

    super_config = load_config()
    specs = {row["label"]: row for row in method_items(super_config)}
    recipe_key = config["scenario"]["recipe"]
    recipe_spec = super_config["recipes"][recipe_key]
    manifest = {
        "schema_version": str(config.get("uq_schema_version", "figure5-zeroh-uq-v4.2-1")),
        "status": "running",
        "formal": formal,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenario": config["scenario"],
        "methods": methods,
        "state_count": states,
        "draws_per_state": draws,
        "seed_schedule": f"stable_seed({int(config['bootstrap']['seed'])},'{namespace}',case_id,time_index,draw_id)",
        "bootstrap": config["bootstrap"],
        "checkpoint_checks": checkpoint_checks,
        "environment": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": args.device, "gpu": torch.cuda.get_device_name(torch.device(args.device))},
        "no_raw_ensemble_cache": True,
    }
    _write_json(run_dir / "manifest.json", manifest)
    state_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    for method in methods:
        loaded = load_model(specs[method], recipe_key, recipe_spec, checkpoint="best", split="test", eval_resolution="H", device=args.device)
        try:
            if checkpoint_digest(loaded.checkpoint_path) != config["checkpoints"][method]["sha256"]:
                raise RuntimeError(f"Loaded checkpoint differs for {method}")
            mean = np.asarray(loaded.dataset.mean.detach().cpu(), dtype=np.float32).reshape(-1)
            std = np.asarray(loaded.dataset.std.detach().cpu(), dtype=np.float32).reshape(-1)
            if len(mean) != 1 or len(std) != 1 or not np.isfinite(mean).all() or not np.isfinite(std).all() or float(std[0]) <= 0:
                raise RuntimeError(f"Invalid density normalization for {method}")
            for position, (metadata, sensor_rows) in enumerate(cohort):
                dataset_index = find_snapshot(loaded.dataset, int(metadata["case_id"]), int(metadata["time_index"]))
                sample = loaded.dataset[dataset_index]
                coords = sample["coords"].unsqueeze(0).to(loaded.device)
                truth = sample["fields"].unsqueeze(0).to(loaded.device)
                indices = torch.tensor([[row["point_index"] for row in sensor_rows]], dtype=torch.long, device=loaded.device)
                field_ids = torch.zeros_like(indices)
                obs_coords = coords[:, indices[0]]
                obs_values = truth[:, indices[0], :1]
                obs_mask = torch.ones(indices.shape, dtype=coords.dtype, device=loaded.device)
                outputs = []
                seeds = []
                with torch.inference_mode():
                    for draw in range(draws):
                        seed = stable_seed(int(config["bootstrap"]["seed"]), namespace, int(metadata["case_id"]), int(metadata["time_index"]), draw)
                        seeds.append(seed)
                        torch.manual_seed(seed)
                        np.random.seed(seed & 0xFFFFFFFF)
                        torch.cuda.manual_seed_all(seed)
                        recon = loaded.model.sample(
                            coords=coords, obs_coords=obs_coords, obs_values=obs_values, obs_mask=obs_mask,
                            obs_field_ids=field_ids, n_steps=int(config["scenario"]["n_steps"]), clamp_indices=indices,
                            ode_solver=str(config["scenario"]["ode_solver"]),
                            obs_consistency_mode=str(config["scenario"]["consistency_modes"][method]),
                        )
                        outputs.append(recon[0, :, 0].detach().cpu())
                    stack = torch.stack(outputs).numpy().astype(np.float32, copy=False)
                    repeat_error = np.nan
                    if not formal:
                        torch.manual_seed(seeds[0]); np.random.seed(seeds[0] & 0xFFFFFFFF); torch.cuda.manual_seed_all(seeds[0])
                        repeated = loaded.model.sample(
                            coords=coords, obs_coords=obs_coords, obs_values=obs_values, obs_mask=obs_mask,
                            obs_field_ids=field_ids, n_steps=int(config["scenario"]["n_steps"]), clamp_indices=indices,
                            ode_solver=str(config["scenario"]["ode_solver"]), obs_consistency_mode=str(config["scenario"]["consistency_modes"][method]),
                        )[0, :, 0].detach().cpu().numpy()
                        repeat_error = float(np.max(np.abs(repeated - stack[0])))
                truth_norm = truth[0, :, 0].detach().cpu().numpy().astype(np.float32, copy=False)
                truth_phys = truth_norm * float(std[0]) + float(mean[0])
                mean_phys = np.mean(stack, axis=0) * float(std[0]) + float(mean[0])
                normalized_crps = _crps(stack, truth_norm)
                normalized_spread = float(np.sqrt(np.mean(np.std(stack, axis=0, ddof=1) ** 2)))
                ensemble_error = float(np.linalg.norm(mean_phys - truth_phys) / max(np.linalg.norm(truth_phys), 1.0e-12))
                clamp_error = float(np.max(np.abs(stack[:, indices[0].detach().cpu().numpy()] - obs_values[0, :, 0].detach().cpu().numpy()[None, :])))
                variance = float(np.mean(np.var(stack, axis=0, ddof=1)))
                state_rows.append({
                    "method": method, "state": int(metadata["snapshot_index"]), "case_id": int(metadata["case_id"]),
                    "time_index": int(metadata["time_index"]), "physical_time": float(metadata["physical_time"]),
                    "sensor_count": int(config["scenario"]["sensor_count"]), "native_query_count": int(coords.shape[1]),
                    "draw_count": draws, "normalized_crps": normalized_crps, "normalized_spread": normalized_spread,
                    "ensemble_mean_relative_l2": ensemble_error,
                })
                audit_rows.append({
                    "method": method, "state": int(metadata["snapshot_index"]), "draw_count": draws,
                    "finite": bool(np.isfinite(stack).all()), "variance_normalized": variance,
                    "genuine_stochasticity": variance > 1.0e-12, "same_seed_max_abs_error": repeat_error,
                    "same_seed_reproducible": formal or repeat_error <= 1.0e-6,
                    "sensor_clamp_max_abs_error_normalized": clamp_error,
                    "sensor_consistency_pass": clamp_error <= 1.0e-5,
                })
                _write_csv(run_dir / "per_state_method.csv", state_rows)
                _write_csv(run_dir / "method_draw_audit.csv", audit_rows)
                if position % 10 == 0 or position + 1 == states:
                    print(f"[{method}] {position + 1}/{states}", flush=True)
        finally:
            loaded.close()

    crps_summary: list[dict[str, Any]] = []
    spread_summary: list[dict[str, Any]] = []
    for method in methods:
        rows = sorted((row for row in state_rows if row["method"] == method), key=lambda row: row["state"])
        crps = np.asarray([row["normalized_crps"] for row in rows], dtype=float)
        spread = np.asarray([row["normalized_spread"] for row in rows], dtype=float)
        error = np.asarray([row["ensemble_mean_relative_l2"] for row in rows], dtype=float)
        crps_low, crps_high = _bootstrap(crps, config["bootstrap"], f"{namespace}|crps|{method}", lambda idx: float(np.mean(crps[idx])))
        rho = float(spearmanr(spread, error).statistic)
        rho_low, rho_high = _bootstrap(spread, config["bootstrap"], f"{namespace}|rho|{method}", lambda idx: float(spearmanr(spread[idx], error[idx]).statistic))
        crps_summary.append({"method": method, "state_count": states, "mean_normalized_crps": float(np.mean(crps)), "crps_ci_low": crps_low, "crps_ci_high": crps_high, "field": "density", "field_macro_weight": 1.0})
        spread_summary.append({"method": method, "state_count": states, "spearman_rho": rho, "spearman_ci_low": rho_low, "spearman_ci_high": rho_high, "spread_definition": "normalized_spatial_rms_ensemble_std", "error_definition": "ensemble_mean_physical_relative_l2", "field_macro_weight": 1.0})
    _write_csv(run_dir / "crps_summary.csv", crps_summary)
    _write_csv(run_dir / "spread_error_summary.csv", spread_summary)
    expected = len(methods) * states
    qa = {
        "status": "pass",
        "formal": formal,
        "state_method_rows_exact": len(state_rows) == expected,
        "paired_unique_case_cohort": all({row["case_id"] for row in state_rows if row["method"] == method} == {int(meta["case_id"]) for meta, _ in cohort} for method in methods),
        "draw_count_exact": {row["draw_count"] for row in state_rows} == {draws},
        "all_finite": all(row["finite"] for row in audit_rows),
        "all_genuinely_stochastic": all(row["genuine_stochasticity"] for row in audit_rows),
        "same_seed_reproducible": all(row["same_seed_reproducible"] for row in audit_rows),
        "sensor_consistency_pass": all(row["sensor_consistency_pass"] for row in audit_rows),
        "only_generative_methods": set(methods) == set(config["scenario"]["generative_methods"]),
        "deterministic_methods_excluded_from_spread": True,
    }
    qa["status"] = "pass" if all(value for key, value in qa.items() if key not in {"status", "formal"}) else "fail"
    _write_json(run_dir / "qa.json", qa)
    manifest.update(status="complete" if qa["status"] == "pass" else "failed_qa", completed_at=datetime.now(timezone.utc).isoformat())
    _write_json(run_dir / "manifest.json", manifest)
    _assert_clean(args.device, allow_current=True)
    print(json.dumps({"run_dir": str(run_dir), "qa": qa, "crps": crps_summary, "spread": spread_summary}, indent=2))
    return 0 if qa["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

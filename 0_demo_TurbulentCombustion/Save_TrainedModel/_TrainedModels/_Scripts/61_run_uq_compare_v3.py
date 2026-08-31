#!/usr/bin/env python
"""Run the Figure 5 V3 cross-model conditional-ensemble comparison.

This intentionally small runner evaluates only the five adopted generative
Cond_T checkpoints on the frozen 12-state pilot or 200-state U2 cohort.  It
streams one state at a time and never stores full ensemble stacks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import yaml
from scipy.stats import spearmanr

from common.config import load_config, stable_seed
from common.model_loader import load_model


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
TOOLS_DIR = REPO_ROOT / "0_demo_TurbulentCombustion" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from benchmark_validation_v3 import core_call, evaluation_context, prepare_state

METHODS = ("DMF-Gen", "FFM-FNO", "FFM-Perceiver", "Latent FM", "SiT")
FIELDS = ("Y_CH4", "Y_CO", "T", "U1", "p")
UNOBSERVED = (0, 1, 3, 4)
LEVELS = (0.50, 0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--job", choices=("PILOT", "FORMAL"), required=True)
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--run-id")
    parser.add_argument("--skip-large-hash", action="store_true")
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


def capture_environment(device: str) -> dict[str, Any]:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": device,
    }
    if torch.cuda.is_available():
        index = torch.device(device).index or 0
        result["gpu_name"] = torch.cuda.get_device_name(index)
        result["gpu_properties"] = str(torch.cuda.get_device_properties(index))
    try:
        result["nvidia_smi"] = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception as exc:  # pragma: no cover
        result["nvidia_smi_error"] = str(exc)
    return result


def verify_identities(plan: dict[str, Any], methods: tuple[str, ...], skip_large_hash: bool) -> list[dict[str, Any]]:
    items = [
        ("dataset", plan["dataset"]["path"], plan["dataset"]["sha256"], skip_large_hash),
        ("dataset_statistics", plan["dataset_statistics"]["path"], plan["dataset_statistics"]["sha256"], False),
        ("sensor_plan", plan["sensor_plan"]["path"], plan["sensor_plan"]["sha256"], False),
    ]
    checkpoint_map = {row["method"]: row for row in plan["checkpoints"]}
    items.extend((method, checkpoint_map[method]["path"], checkpoint_map[method]["sha256"], False) for method in methods)
    checks = []
    for label, raw_path, expected, skip in items:
        path = resolve(raw_path)
        actual = "skipped" if skip else sha256(path)
        checks.append({"label": label, "path": str(path), "expected_sha256": expected, "actual_sha256": actual, "pass": skip or actual == expected})
    if not all(row["pass"] for row in checks):
        raise RuntimeError("Frozen V3 UQ identity check failed")
    return checks


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
            raise RuntimeError(f"State {state} lacks the exact 256-row Cond_T sensor plan")
    return groups


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([(np.arange(block) + start) % n for start in starts])[:n]


def bootstrap_mean(values: np.ndarray, spec: dict[str, Any], salt: str) -> tuple[float, float]:
    rng = np.random.default_rng(stable_seed(int(spec["seed"]), salt))
    block = min(int(spec["block_length"]), len(values))
    samples = np.empty(int(spec["replicates"]), dtype=float)
    for index in range(len(samples)):
        samples[index] = np.mean(values[moving_block_indices(len(values), block, rng)])
    alpha = (1.0 - float(spec["confidence_level"])) / 2.0
    return tuple(float(value) for value in np.quantile(samples, [alpha, 1.0 - alpha]))


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, spec: dict[str, Any], salt: str) -> tuple[float, float]:
    rng = np.random.default_rng(stable_seed(int(spec["seed"]), salt))
    block = min(int(spec["block_length"]), len(x))
    samples = np.empty(int(spec["replicates"]), dtype=float)
    for index in range(len(samples)):
        selected = moving_block_indices(len(x), block, rng)
        samples[index] = float(spearmanr(x[selected], y[selected]).statistic)
    alpha = (1.0 - float(spec["confidence_level"])) / 2.0
    return tuple(float(value) for value in np.nanquantile(samples, [alpha, 1.0 - alpha]))


def empirical_crps_normalized(draws: np.ndarray, truth: np.ndarray) -> float:
    """Mean pointwise empirical CRPS using the O(S log S) sorted identity."""
    crps, _ = empirical_crps_and_quantiles(draws, truth, ())
    return crps


def empirical_crps_and_quantiles(draws: np.ndarray, truth: np.ndarray, quantiles: tuple[float, ...] | list[float]) -> tuple[float, np.ndarray]:
    """Reduce CRPS and linear empirical quantiles from one shared sort."""
    draws64 = np.asarray(draws, dtype=np.float64)
    truth64 = np.asarray(truth, dtype=np.float64)
    count = draws64.shape[0]
    absolute_term = np.mean(np.abs(draws64 - truth64[None, :]), axis=0)
    ordered = np.sort(draws64, axis=0)
    coefficients = 2.0 * np.arange(1, count + 1, dtype=np.float64) - count - 1.0
    dispersion_term = np.sum(coefficients[:, None] * ordered, axis=0) / (count * count)
    probabilities = np.asarray(quantiles, dtype=np.float64)
    if probabilities.size:
        positions = probabilities * (count - 1)
        lower_indices = np.floor(positions).astype(int)
        upper_indices = np.ceil(positions).astype(int)
        fractions = positions - lower_indices
        bounds = ordered[lower_indices] * (1.0 - fractions[:, None]) + ordered[upper_indices] * fractions[:, None]
    else:
        bounds = np.empty((0, draws64.shape[1]), dtype=np.float64)
    return float(np.mean(absolute_term - dispersion_term)), bounds


def method_settings(post_cfg: dict[str, Any], method: str) -> dict[str, Any]:
    override = post_cfg.get("method_inference_overrides", {}).get(method, {})
    n_steps = int(override.get("n_steps", post_cfg["defaults"]["n_steps"]))
    consistency = str(override.get("obs_consistency", post_cfg["defaults"]["obs_consistency"]))
    return {
        "n_steps": n_steps,
        "solver": "heun_native" if method == "FFM-FNO" else "euler",
        "measured_nfe": 2 * n_steps if method == "FFM-FNO" else n_steps,
        "obs_consistency": consistency,
        "execution_mode": "cached_streamed" if method == "DMF-Gen" else "legacy_full",
        "cache_level": "static_features" if method == "DMF-Gen" else "none",
    }


def load_v2_dmf_rows(plan: dict[str, Any]) -> dict[tuple[int, str], dict[str, str]]:
    path = resolve(plan["output_root"]) / "Uncertainty" / "u2_formal_20260830_v1" / "per_state_field.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Required reusable DMF U2 source missing: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = {(int(row["state"]), row["field"]): row for row in rows if int(row["sensor_count"]) == 256}
    if len(selected) != 1000 or {int(row["draw_count"]) for row in selected.values()} != {64}:
        raise RuntimeError("Reusable DMF U2 source does not match 200 states x 5 fields x 64 draws")
    return selected


def summarize(rows: list[dict[str, Any]], plan: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    crps_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    for method in METHODS:
        group = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["original_time_index"]))
        if not group:
            continue
        crps = np.asarray([float(row["macro_normalized_crps"]) for row in group])
        low, high = bootstrap_mean(crps, plan["bootstrap"], f"v3|crps|{method}")
        crps_rows.append({"method": method, "state_count": len(group), "mean_normalized_crps": float(np.mean(crps)), "crps_ci_low": low, "crps_ci_high": high, "field_macro_weight": 0.25})
        spread = np.asarray([float(row["macro_normalized_spread"]) for row in group])
        error = np.asarray([float(row["macro_ensemble_mean_relative_l2"]) for row in group])
        rho = float(spearmanr(spread, error).statistic)
        low, high = bootstrap_spearman(spread, error, plan["bootstrap"], f"v3|spearman|{method}")
        association_rows.append({"method": method, "state_count": len(group), "spearman_rho": rho, "spearman_ci_low": low, "spearman_ci_high": high, "spread_definition": "mean_field_normalized_spatial_rms_ensemble_std", "error_definition": "mean_field_ensemble_mean_physical_relative_l2", "field_macro_weight": 0.25})
    return crps_rows, association_rows


def summarize_reliability(field_rows: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for method in METHODS:
        for field_index in UNOBSERVED:
            field = FIELDS[field_index]
            group = sorted((row for row in field_rows if row["method"] == method and row["field"] == field), key=lambda row: int(row["original_time_index"]))
            if not group:
                continue
            for level in LEVELS:
                tag = int(level * 100)
                coverage = np.asarray([float(row[f"coverage_{tag}"]) for row in group])
                width = np.asarray([float(row[f"width_normalized_{tag}"]) for row in group])
                coverage_low, coverage_high = bootstrap_mean(coverage, plan["bootstrap"], f"v3|coverage|{method}|{field}|{tag}")
                width_low, width_high = bootstrap_mean(width, plan["bootstrap"], f"v3|width|{method}|{field}|{tag}")
                output.append({"method": method, "field": field, "nominal_level": level, "state_count": len(group), "empirical_coverage": float(np.mean(coverage)), "coverage_ci_low": coverage_low, "coverage_ci_high": coverage_high, "mean_interval_width_normalized": float(np.mean(width)), "width_ci_low": width_low, "width_ci_high": width_high})
    return output


def main() -> int:
    args = parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "figure5-validation-v1":
        raise ValueError("Unsupported frozen validation plan")
    methods = METHODS if "all" in args.methods else tuple(method for method in METHODS if method in set(args.methods))
    if not methods:
        raise ValueError("No canonical generative methods selected")
    states = list(map(int, plan["cohorts"]["pilot" if args.job == "PILOT" else "calibration_200"]["evaluation_indices"]))
    draws_per_state = 8 if args.job == "PILOT" else 64
    run_id = args.run_id or f"uq_compare_{args.job.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = REPO_ROOT / "Dis_SI_Process" / "results" / "ValidationV3" / "UQCompare" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    identity = verify_identities(plan, methods, args.skip_large_hash)
    sensors = load_sensor_rows(plan, states)
    post_cfg = load_config()
    method_cfg = {row["name"]: row for row in post_cfg["methods"]}
    original_times = list(map(int, plan["test_states"]["original_hdf5_time_indices"]))
    mean = np.asarray(plan["dataset_statistics"]["mean"], dtype=np.float32)
    std = np.asarray(plan["dataset_statistics"]["std"], dtype=np.float32)
    reusable_dmf = load_v2_dmf_rows(plan) if args.job == "FORMAL" and "DMF-Gen" in methods else {}
    manifest = {
        "schema_version": "figure5-validation-v3-uq-1",
        "status": "running",
        "formal": args.job == "FORMAL" and not args.skip_large_hash and methods == METHODS,
        "run_id": run_id,
        "job": args.job,
        "methods": methods,
        "states": states,
        "draws_per_state": draws_per_state,
        "sensor_count": 256,
        "seed_schedule": "stable_seed(20260830,'generation','U2',state,draw); shared across methods",
        "identity_checks": identity,
        "environment": capture_environment(args.device),
        "bootstrap": plan["bootstrap"],
        "reused_sources": [str(resolve(plan["output_root"]) / "Uncertainty" / "u2_formal_20260830_v1" / "per_state_field.csv")] if reusable_dmf else [],
        "reuse_policy": "DMF fieldwise spread/error/reliability are referenced from V2 U2; only missing normalized CRPS is newly reduced from matching draws.",
    }
    atomic_json(output_dir / "manifest.json", manifest)
    state_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    prepared_path_equivalence: list[dict[str, Any]] = []

    for method in methods:
        settings = method_settings(post_cfg, method)
        loaded = load_model(method_cfg[method], "Cond_T", checkpoint="last", split="test", device=args.device, n_steps=settings["n_steps"], ode_solver="euler")
        dataset_mean = np.asarray(loaded.dataset.mean.detach().cpu() if torch.is_tensor(loaded.dataset.mean) else loaded.dataset.mean, dtype=np.float32)
        dataset_std = np.asarray(loaded.dataset.std.detach().cpu() if torch.is_tensor(loaded.dataset.std) else loaded.dataset.std, dtype=np.float32)
        normalization_mean_error = float(np.max(np.abs(dataset_mean - mean)))
        normalization_std_error = float(np.max(np.abs(dataset_std - std)))
        normalization_mean_relative_error = float(np.max(np.abs(dataset_mean - mean) / np.maximum(np.abs(mean), 1e-12)))
        normalization_std_relative_error = float(np.max(np.abs(dataset_std - std) / np.maximum(np.abs(std), 1e-12)))
        normalization_match = bool(np.allclose(dataset_mean, mean, rtol=1e-6, atol=1e-6) and np.allclose(dataset_std, std, rtol=1e-6, atol=1e-6))
        if not normalization_match:
            raise RuntimeError(f"{method} normalization differs from the frozen common scale")
        try:
            legacy_reference = None
            legacy_consistency = None
            if args.job == "PILOT":
                reference_state = states[0]
                reference_seed = stable_seed(int(plan["generation_seeds"]["base"]), "generation", "U2", reference_state, 0)
                reference = loaded.reconstruct(
                    reference_state, {"cond_fields": [2], "n_obs": [256]}, sensors[reference_state],
                    n_steps=settings["n_steps"], ode_solver="euler", obs_consistency=settings["obs_consistency"],
                    generation_seed=reference_seed, reconstruction_execution_mode=settings["execution_mode"],
                    reconstruction_query_chunk_size=int(plan["inference"]["query_chunk_size"]),
                )
                legacy_reference = reference["recon"][0].detach().cpu().numpy().astype(np.float32, copy=False)
                legacy_consistency = reference["obs_consistency_applied"]

            with evaluation_context(loaded), torch.no_grad():
                for position, state in enumerate(states):
                    seeds = [stable_seed(int(plan["generation_seeds"]["base"]), "generation", "U2", state, draw) for draw in range(draws_per_state)]
                    prepared = prepare_state(loaded, state, sensors[state], 40300, method, retain_truth=True)
                    draws = []
                    for seed in seeds:
                        torch.manual_seed(seed)
                        np.random.seed(seed % (2**32))
                        torch.cuda.manual_seed_all(seed)
                        recon = core_call(loaded, prepared, settings, method)
                        draws.append(recon[0].detach())
                    stack_device = torch.stack(draws)
                    stack = stack_device.cpu().numpy().astype(np.float32, copy=False)
                    truth = prepared["truth"][0].detach().cpu().numpy().astype(np.float32, copy=False)
                    sensor_indices = prepared["obs_indices"][0].detach().cpu().numpy()
                    sensor_values = prepared["obs_values"][0, :, 0].detach().cpu().numpy()
                    sample_parameters = set(inspect.signature(loaded.model.sample).parameters) if loaded.family == "pointcloud_ffm" else set()
                    consistency_applied = settings["obs_consistency"] if "obs_consistency_mode" in sample_parameters else "native_not_applied"
                    repeat_error = np.nan
                    if args.job == "PILOT":
                        torch.manual_seed(seeds[0])
                        np.random.seed(seeds[0] % (2**32))
                        torch.cuda.manual_seed_all(seeds[0])
                        repeated = core_call(loaded, prepared, settings, method)[0].detach()
                        repeat_error = float(torch.max(torch.abs(repeated - stack_device[0])).item())
                    prepared_error = np.nan
                    if args.job == "PILOT" and position == 0:
                        prepared_error = float(np.max(np.abs(legacy_reference - stack[0])))
                        prepared_path_equivalence.append({
                            "method": method, "state": state, "seed": seeds[0], "max_abs_error": prepared_error,
                            "atol": float(plan["qa_tolerances"]["cache_equivalence_atol"]),
                            "pass": prepared_error <= float(plan["qa_tolerances"]["cache_equivalence_atol"]),
                        })
                        consistency_applied = legacy_consistency
                    physical = stack * dataset_std[None, None, :] + dataset_mean[None, None, :]
                    truth_physical = truth * dataset_std[None, :] + dataset_mean[None, :]
                    common_normalized_stack = (physical - mean[None, None, :]) / std[None, None, :]
                    common_normalized_truth = (truth_physical - mean[None, :]) / std[None, :]
                    field_crps = []
                    field_spread = []
                    field_error = []
                    for field_index in UNOBSERVED:
                        field = FIELDS[field_index]
                        quantile_levels = [bound for level in LEVELS for bound in ((1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0)]
                        crps, quantile_bounds = empirical_crps_and_quantiles(
                            common_normalized_stack[:, :, field_index], common_normalized_truth[:, field_index], quantile_levels
                        )
                        spread = float(np.sqrt(np.mean(np.std(common_normalized_stack[:, :, field_index], axis=0, ddof=1) ** 2)))
                        ensemble_mean = np.mean(physical[:, :, field_index], axis=0)
                        target = truth_physical[:, field_index]
                        error = float(np.linalg.norm(ensemble_mean - target) / max(np.linalg.norm(target), 1e-12))
                        source = "v3_draw_reducer"
                        field_row: dict[str, Any] = {"method": method, "state": state, "original_time_index": original_times[state], "field": field, "normalized_crps": crps, "normalized_spread": spread, "ensemble_mean_relative_l2": error}
                        for level_position, level in enumerate(LEVELS):
                            tag = int(level * 100)
                            lower, upper = quantile_bounds[2 * level_position : 2 * level_position + 2]
                            field_row[f"coverage_{tag}"] = float(np.mean((common_normalized_truth[:, field_index] >= lower) & (common_normalized_truth[:, field_index] <= upper)))
                            field_row[f"width_normalized_{tag}"] = float(np.mean(upper - lower))
                        if method == "DMF-Gen" and reusable_dmf:
                            reused = reusable_dmf[(state, field)]
                            spread = float(reused["spread_rms_normalized"])
                            error = float(reused["ensemble_mean_relative_l2"])
                            field_row["normalized_spread"] = spread
                            field_row["ensemble_mean_relative_l2"] = error
                            for level in LEVELS:
                                tag = int(level * 100)
                                field_row[f"coverage_{tag}"] = float(reused[f"coverage_{tag}"])
                                field_row[f"width_normalized_{tag}"] = float(reused[f"interval_width_normalized_{tag}"])
                            source = "reused_validation_v2_u2"
                        field_row["non_crps_source"] = source
                        field_rows.append(field_row)
                        field_crps.append(crps)
                        field_spread.append(spread)
                        field_error.append(error)
                    clamp_error = float(np.max(np.abs(stack[:, sensor_indices, 2] - sensor_values[None, :])))
                    variance = float(np.min([np.mean(np.var(common_normalized_stack[:, :, index], axis=0, ddof=1)) for index in UNOBSERVED]))
                    state_rows.append({
                        "method": method, "state": state, "original_time_index": original_times[state], "sensor_count": 256,
                        "draw_count": draws_per_state, "generation_seed_first": seeds[0], "generation_seed_last": seeds[-1],
                        "macro_normalized_crps": float(np.mean(field_crps)), "macro_normalized_spread": float(np.mean(field_spread)),
                        "macro_ensemble_mean_relative_l2": float(np.mean(field_error)),
                        **{f"crps_{FIELDS[index]}": value for index, value in zip(UNOBSERVED, field_crps)},
                        **{f"spread_{FIELDS[index]}": value for index, value in zip(UNOBSERVED, field_spread)},
                        **{f"error_{FIELDS[index]}": value for index, value in zip(UNOBSERVED, field_error)},
                    })
                    audit_rows.append({
                        "method": method, "state": state, "original_time_index": original_times[state], "draw_count": draws_per_state,
                        "minimum_unobserved_variance_normalized": variance, "same_seed_max_abs_error": repeat_error,
                        "sensor_clamp_max_abs_error_normalized": clamp_error, "finite": bool(np.isfinite(stack).all()),
                        "genuine_stochasticity": variance > float(plan["qa_tolerances"]["minimum_unobserved_variance"]),
                        "same_seed_reproducible": args.job != "PILOT" or repeat_error <= float(plan["qa_tolerances"]["same_seed_atol"]),
                        "normalization_match": normalization_match, "normalization_mean_max_abs_error": normalization_mean_error,
                        "normalization_std_max_abs_error": normalization_std_error,
                        "normalization_mean_max_relative_error": normalization_mean_relative_error,
                        "normalization_std_max_relative_error": normalization_std_relative_error,
                        "prepared_loader_max_abs_error": prepared_error,
                        "obs_consistency_requested": settings["obs_consistency"], "obs_consistency_applied": consistency_applied,
                        **settings,
                    })
                    write_csv(output_dir / "per_state_method.csv", state_rows)
                    write_csv(output_dir / "method_draw_audit.csv", audit_rows)
                    if position % 10 == 0 or position + 1 == len(states):
                        print(f"[{method}] {position + 1}/{len(states)} states", flush=True)
        finally:
            loaded.close()

    crps_summary, spread_summary = summarize(state_rows, plan)
    reliability = summarize_reliability(field_rows, plan)
    write_csv(output_dir / "crps_summary.csv", crps_summary)
    write_csv(output_dir / "spread_error_summary.csv", spread_summary)
    write_csv(output_dir / "reliability_si.csv", reliability)
    write_csv(output_dir / "prepared_path_equivalence.csv", prepared_path_equivalence)
    expected_rows = len(methods) * len(states)
    qa = {
        "status": "pass",
        "formal": manifest["formal"],
        "expected_state_method_rows": expected_rows,
        "actual_state_method_rows": len(state_rows),
        "all_finite": all(bool(row["finite"]) for row in audit_rows),
        "all_genuinely_stochastic": all(bool(row["genuine_stochasticity"]) for row in audit_rows),
        "all_same_seed_reproducible": all(bool(row["same_seed_reproducible"]) for row in audit_rows),
        "all_normalization_match": all(bool(row["normalization_match"]) for row in audit_rows),
        "paired_state_cohort": all({int(row["state"]) for row in state_rows if row["method"] == method} == set(states) for method in methods),
        "equal_draw_count": {int(row["draw_count"]) for row in state_rows} == {draws_per_state},
        "macro_fields": [FIELDS[index] for index in UNOBSERVED],
        "macro_weights": [0.25] * 4,
        "dmf_v2_field_metrics_reused": bool(reusable_dmf),
        "prepared_path_equivalence": prepared_path_equivalence,
        "prepared_path_equivalence_pass": args.job != "PILOT" or (
            len(prepared_path_equivalence) == len(methods) and all(bool(row["pass"]) for row in prepared_path_equivalence)
        ),
    }
    qa["status"] = "pass" if all((qa["actual_state_method_rows"] == expected_rows, qa["all_finite"], qa["all_genuinely_stochastic"], qa["all_same_seed_reproducible"], qa["all_normalization_match"], qa["paired_state_cohort"], qa["equal_draw_count"], qa["prepared_path_equivalence_pass"])) else "fail"
    atomic_json(output_dir / "qa.json", qa)
    manifest["status"] = "complete" if qa["status"] == "pass" else "qa_failed"
    manifest["completed_at"] = datetime.now().isoformat()
    atomic_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "qa": qa}, indent=2))
    return 0 if qa["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

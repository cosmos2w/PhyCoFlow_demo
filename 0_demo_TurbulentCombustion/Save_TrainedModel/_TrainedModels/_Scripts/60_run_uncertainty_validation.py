#!/usr/bin/env python
"""Run the frozen Figure 5 repeated-generation uncertainty campaign.

The runner streams one state at a time and retains full draws only for the
predeclared visual cases.  It deliberately supports only PILOT/U1/U2/U3 from
the frozen validation plan; it is not a general experiment framework.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, rankdata, spearmanr

from common.config import load_config, stable_seed
from common.model_loader import load_model


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
FIELD_NAMES = ("Y_CH4", "Y_CO", "T", "U1", "p")
UNOBSERVED = (0, 1, 3, 4)
LEVELS = (0.50, 0.80, 0.90, 0.95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--job", choices=("PILOT", "U1", "U2", "U3"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--execution-mode", choices=("legacy_full", "cached_streamed"), default="cached_streamed")
    parser.add_argument("--skip-large-hash", action="store_true", help="Skip the 8-GB dataset hash only; all smaller identities remain mandatory.")
    parser.add_argument("--max-states", type=int, help="Developer smoke limit; marks output non-formal.")
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


def capture_environment(device: str) -> dict[str, Any]:
    result: dict[str, Any] = {
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
        result["nvidia_smi"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:  # pragma: no cover - environment capture is best effort
        result["nvidia_smi_error"] = str(exc)
    return result


def verify_plan(plan: dict[str, Any], *, skip_large_hash: bool) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(label: str, raw_path: str, expected: str, skip: bool = False) -> None:
        path = resolve(raw_path)
        actual = "skipped" if skip else sha256(path)
        ok = skip or actual == expected
        checks.append({"label": label, "path": str(path), "expected_sha256": expected, "actual_sha256": actual, "pass": ok})
        if not ok:
            raise RuntimeError(f"Frozen identity mismatch for {label}: {actual} != {expected}")

    dataset = plan["dataset"]
    check("dataset", dataset["path"], dataset["sha256"], skip_large_hash)
    stats = plan["dataset_statistics"]
    check("dataset_statistics", stats["path"], stats["sha256"])
    sensors = plan["sensor_plan"]
    check("sensor_plan", sensors["path"], sensors["sha256"])
    dmf = next(item for item in plan["checkpoints"] if item["method"] == "DMF-Gen")
    check("DMF-Gen checkpoint", dmf["path"], dmf["sha256"])
    return {"checks": checks, "pass": all(row["pass"] for row in checks)}


def load_adopted_sensor_rows(plan: dict[str, Any], states: list[int]) -> dict[int, list[dict[str, Any]]]:
    wanted = set(states)
    groups = {state: [] for state in states}
    with resolve(plan["sensor_plan"]["path"]).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != "Cond_T":
                continue
            state = int(row["snapshot"])
            if state not in wanted:
                continue
            groups[state].append(
                {
                    "sensor_order": int(row["sensor_order"]),
                    "field_index": int(row["field_index"]),
                    "point_index": int(row["point_index"]),
                    "sensor_seed": int(row["sensor_seed"]),
                }
            )
    for state, rows in groups.items():
        rows.sort(key=lambda value: value["sensor_order"])
        if len(rows) != 256 or any(row["field_index"] != 2 for row in rows):
            raise RuntimeError(f"State {state} does not have exactly 256 adopted Cond_T rows")
    return groups


def sensor_rows_for_m(plan: dict[str, Any], state: int, adopted: list[dict[str, Any]], m: int) -> list[dict[str, Any]]:
    if m <= 256:
        return [dict(row) for row in adopted[:m]]
    if m != 384:
        raise ValueError(f"Unsupported sensor count: {m}")
    base = [int(row["point_index"]) for row in adopted]
    rule = plan["sensor_plan"]["nested"]["M384"]
    seed = stable_seed(int(rule["extension_seed_base"]), "M384_extension", state)
    rng = np.random.default_rng(seed)
    mask = np.ones(int(plan["dataset"]["shape"][2]), dtype=bool)
    mask[base] = False
    extension = rng.choice(np.flatnonzero(mask), size=128, replace=False)
    rows = [dict(row) for row in adopted]
    sensor_seed = int(adopted[0]["sensor_seed"])
    rows.extend(
        {"sensor_order": 256 + order, "field_index": 2, "point_index": int(point), "sensor_seed": sensor_seed}
        for order, point in enumerate(extension)
    )
    return rows


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([(np.arange(block) + start) % n for start in starts])[:n]


def bootstrap_mean(values: np.ndarray, bootstrap: dict[str, Any], salt: str) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(stable_seed(int(bootstrap["seed"]), salt))
    reps = int(bootstrap["replicates"])
    block = min(int(bootstrap["block_length"]), len(values))
    samples = np.empty(reps, dtype=float)
    for i in range(reps):
        samples[i] = np.mean(values[moving_block_indices(len(values), block, rng)])
    alpha = (1.0 - float(bootstrap["confidence_level"])) / 2.0
    return tuple(float(x) for x in np.quantile(samples, [alpha, 1.0 - alpha]))


def bootstrap_spearman(spread: np.ndarray, error: np.ndarray, bootstrap: dict[str, Any], salt: str) -> tuple[float, float]:
    rng = np.random.default_rng(stable_seed(int(bootstrap["seed"]), salt))
    reps = int(bootstrap["replicates"])
    block = min(int(bootstrap["block_length"]), len(spread))
    values = np.empty(reps, dtype=float)
    for i in range(reps):
        idx = moving_block_indices(len(spread), block, rng)
        values[i] = np.corrcoef(rankdata(spread[idx]), rankdata(error[idx]))[0, 1]
    alpha = (1.0 - float(bootstrap["confidence_level"])) / 2.0
    return tuple(float(x) for x in np.nanquantile(values, [alpha, 1.0 - alpha]))


def state_metrics(
    draws_norm: np.ndarray,
    truth_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    sensor_indices: np.ndarray,
    *,
    job: str,
    state: int,
    original_time_index: int,
    sensor_count: int,
    generation_seeds: list[int],
    measured_nfe: int,
) -> list[dict[str, Any]]:
    truth_phys = truth_norm * std + mean
    draws_phys = draws_norm * std[None, None, :] + mean[None, None, :]
    ensemble_mean = draws_phys.mean(axis=0)
    ensemble_std = draws_phys.std(axis=0, ddof=1)
    rows: list[dict[str, Any]] = []
    for field_index, field in enumerate(FIELD_NAMES):
        valid = np.ones(truth_phys.shape[0], dtype=bool)
        if field_index == 2:
            valid[sensor_indices] = False
        target = truth_phys[valid, field_index]
        center = ensemble_mean[valid, field_index]
        scale = float(std[field_index])
        denom = max(float(np.linalg.norm(target)), 1e-12)
        row: dict[str, Any] = {
            "job": job,
            "state": state,
            "original_time_index": original_time_index,
            "field_index": field_index,
            "field": field,
            "observed": field_index == 2,
            "sensor_count": sensor_count,
            "draw_count": draws_norm.shape[0],
            "generation_seed_first": generation_seeds[0],
            "generation_seed_last": generation_seeds[-1],
            "measured_nfe": measured_nfe,
            "spread_rms_physical": float(np.sqrt(np.mean(ensemble_std[valid, field_index] ** 2))),
            "spread_rms_normalized": float(np.sqrt(np.mean((ensemble_std[valid, field_index] / scale) ** 2))),
            "ensemble_mean_relative_l2": float(np.linalg.norm(center - target) / denom),
            "single_draw_relative_l2": float(np.linalg.norm(draws_phys[0, valid, field_index] - target) / denom),
            "mean_adjacent_pair_relative_l2": float(np.mean([
                np.linalg.norm(draws_phys[i + 1, valid, field_index] - draws_phys[i, valid, field_index]) /
                max(np.linalg.norm(draws_phys[i, valid, field_index]), 1e-12)
                for i in range(draws_phys.shape[0] - 1)
            ])),
        }
        for level in LEVELS:
            alpha = (1.0 - level) / 2.0
            lower, upper = np.quantile(draws_phys[:, valid, field_index], [alpha, 1.0 - alpha], axis=0)
            covered = (target >= lower) & (target <= upper)
            tag = int(round(level * 100))
            row[f"coverage_{tag}"] = float(np.mean(covered))
            row[f"interval_width_physical_{tag}"] = float(np.mean(upper - lower))
            row[f"interval_width_normalized_{tag}"] = float(np.mean((upper - lower) / scale))
        rows.append(row)
    return rows


def aggregate_coverage(rows: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    bootstrap = plan["bootstrap"]
    for sensor_count in sorted({int(row["sensor_count"]) for row in rows}):
        for field in FIELD_NAMES:
            group = [row for row in rows if int(row["sensor_count"]) == sensor_count and row["field"] == field]
            for level in LEVELS:
                tag = int(round(level * 100))
                coverage = np.asarray([float(row[f"coverage_{tag}"]) for row in group])
                width_p = np.asarray([float(row[f"interval_width_physical_{tag}"]) for row in group])
                width_n = np.asarray([float(row[f"interval_width_normalized_{tag}"]) for row in group])
                c_lo, c_hi = bootstrap_mean(coverage, bootstrap, f"coverage|{sensor_count}|{field}|{tag}")
                p_lo, p_hi = bootstrap_mean(width_p, bootstrap, f"widthp|{sensor_count}|{field}|{tag}")
                n_lo, n_hi = bootstrap_mean(width_n, bootstrap, f"widthn|{sensor_count}|{field}|{tag}")
                output.append({
                    "sensor_count": sensor_count,
                    "field": field,
                    "nominal_level": level,
                    "state_count": len(group),
                    "empirical_coverage": float(np.mean(coverage)),
                    "coverage_ci_low": c_lo,
                    "coverage_ci_high": c_hi,
                    "calibration_error": float(np.mean(coverage) - level),
                    "mean_interval_width_physical": float(np.mean(width_p)),
                    "width_physical_ci_low": p_lo,
                    "width_physical_ci_high": p_hi,
                    "mean_interval_width_normalized": float(np.mean(width_n)),
                    "width_normalized_ci_low": n_lo,
                    "width_normalized_ci_high": n_hi,
                })
    return output


def association_summary(rows: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for field in (FIELD_NAMES[i] for i in UNOBSERVED):
        group = sorted((row for row in rows if row["field"] == field and int(row["sensor_count"]) == 256), key=lambda row: int(row["original_time_index"]))
        spread = np.asarray([float(row["spread_rms_normalized"]) for row in group])
        error = np.asarray([float(row["ensemble_mean_relative_l2"]) for row in group])
        rho = float(spearmanr(spread, error).statistic)
        pearson = float(pearsonr(spread, error).statistic)
        low, high = bootstrap_spearman(spread, error, plan["bootstrap"], f"spearman|{field}")
        output.append({"field": field, "state_count": len(group), "spearman_rho": rho, "spearman_ci_low": low, "spearman_ci_high": high, "pearson_r": pearson})
    return output


def main() -> int:
    args = parse_args()
    plan = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "figure5-validation-v1":
        raise ValueError("Unsupported validation plan schema")
    spec = plan["uncertainty_jobs"][args.job]
    if args.job in ("PILOT", "U1"):
        cohort_name = "pilot" if args.job == "PILOT" else None
        states = plan["cohorts"][cohort_name]["evaluation_indices"] if cohort_name else plan["test_states"]["evaluation_indices"]
    else:
        states = plan["cohorts"]["calibration_200"]["evaluation_indices"]
    if args.max_states is not None:
        states = states[: args.max_states]
    states = [int(value) for value in states]
    sensor_counts = [int(value) for value in spec["sensor_counts"]]
    draws_per_state = int(spec["draws"])
    run_id = args.run_id or f"{args.job.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = resolve(plan["output_root"]) / "Uncertainty" / run_id
    visual_dir = output_dir / "visual_maps"
    output_dir.mkdir(parents=True, exist_ok=False)
    visual_dir.mkdir()

    identity = verify_plan(plan, skip_large_hash=args.skip_large_hash)
    environment = capture_environment(args.device)
    manifest = {
        "schema_version": "validation-v2-uncertainty-1",
        "status": "running",
        "run_id": run_id,
        "job": args.job,
        "formal": args.max_states is None and not args.skip_large_hash,
        "plan": str(args.config.resolve()),
        "plan_sha256": sha256(args.config.resolve()),
        "identity": identity,
        "states": states,
        "sensor_counts": sensor_counts,
        "draws_per_state": draws_per_state,
        "execution_mode": args.execution_mode,
        "environment": environment,
        "bootstrap": plan["bootstrap"],
    }
    atomic_json(output_dir / "manifest.json", manifest)

    post_cfg = load_config()
    method = next(item for item in post_cfg["methods"] if item["name"] == "DMF-Gen")
    loaded = load_model(method, "Cond_T", checkpoint="last", split="test", device=args.device, n_steps=2, ode_solver="euler")
    adopted = load_adopted_sensor_rows(plan, states)
    original_indices = [int(value) for value in plan["test_states"]["original_hdf5_time_indices"]]
    stats_mean = np.asarray(plan["dataset_statistics"]["mean"], dtype=np.float32)
    stats_std = np.asarray(plan["dataset_statistics"]["std"], dtype=np.float32)
    tolerance = plan["qa_tolerances"]
    result_rows: list[dict[str, Any]] = []
    visual_rows: list[dict[str, Any]] = []
    clamp_errors: list[float] = []
    variance_values: list[float] = []
    roundtrip_errors: list[float] = []
    same_seed_error = np.nan
    cache_equivalence_error = np.nan
    visual_states = {states[0], states[len(states) // 2], states[-1]}

    try:
        for state_position, state in enumerate(states):
            for sensor_count in sensor_counts:
                rows = sensor_rows_for_m(plan, state, adopted[state], sensor_count)
                generation_seeds = [stable_seed(int(plan["generation_seeds"]["base"]), "generation", args.job, state, draw) for draw in range(draws_per_state)]
                draws: list[np.ndarray] = []
                truth_norm = None
                sensor_indices = None
                sensor_values = None
                for draw_index, seed in enumerate(generation_seeds):
                    out = loaded.reconstruct(
                        state,
                        {"cond_fields": [2], "n_obs": [sensor_count]},
                        rows,
                        n_steps=int(plan["inference"]["nominal_steps"]),
                        ode_solver=plan["inference"]["solver"],
                        obs_consistency=plan["inference"]["observation_consistency"],
                        generation_seed=seed,
                        reconstruction_execution_mode=args.execution_mode,
                        reconstruction_query_chunk_size=int(plan["inference"]["query_chunk_size"]),
                    )
                    draw = out["recon"][0].detach().cpu().numpy().astype(np.float32, copy=False)
                    if truth_norm is None:
                        truth_norm = out["truth"][0].detach().cpu().numpy().astype(np.float32, copy=False)
                        valid = out["obs_mask"][0].bool()
                        sensor_indices = out["obs_indices"][0, valid].detach().cpu().numpy()
                        sensor_values = out["obs_values"][0, valid, 0].detach().cpu().numpy()
                    clamp_errors.append(float(np.max(np.abs(draw[sensor_indices, 2] - sensor_values))))
                    draws.append(draw)
                stack = np.stack(draws)
                physical = stack * stats_std[None, None, :] + stats_mean[None, None, :]
                roundtrip = (physical - stats_mean[None, None, :]) / stats_std[None, None, :]
                roundtrip_errors.append(float(np.max(np.abs(roundtrip - stack))))
                variance_values.extend(float(np.mean(np.var(stack[:, :, index], axis=0, ddof=1))) for index in UNOBSERVED)
                result_rows.extend(state_metrics(
                    stack, truth_norm, stats_mean, stats_std, sensor_indices,
                    job=args.job, state=state, original_time_index=original_indices[state], sensor_count=sensor_count,
                    generation_seeds=generation_seeds, measured_nfe=int(plan["inference"]["measured_nfe"]),
                ))
                if state in visual_states and sensor_count == 256:
                    visual_path = visual_dir / f"{args.job.lower()}_s{state:04d}_m{sensor_count}.npz"
                    np.savez_compressed(
                        visual_path,
                        coords=out["coords"][0].detach().cpu().numpy(), truth_norm=truth_norm,
                        ensemble_mean_norm=stack.mean(axis=0), ensemble_std_norm=stack.std(axis=0, ddof=1),
                        sensor_indices=sensor_indices, field_names=np.asarray(FIELD_NAMES),
                        state=state, original_time_index=original_indices[state], generation_seeds=np.asarray(generation_seeds),
                    )
                    visual_rows.append({"job": args.job, "state": state, "original_time_index": original_indices[state], "sensor_count": sensor_count, "draw_count": draws_per_state, "path": str(visual_path)})
                write_csv(output_dir / "per_state_field.csv", result_rows)
                write_csv(output_dir / "visual_cases.csv", visual_rows)
                print(f"[STATE] {state_position + 1}/{len(states)} | state={state} | M={sensor_count} | draws={draws_per_state}", flush=True)

        if args.job == "PILOT":
            state = states[0]
            rows = sensor_rows_for_m(plan, state, adopted[state], 256)
            seed = stable_seed(int(plan["generation_seeds"]["base"]), "generation", args.job, state, 0)
            legacy = loaded.reconstruct(state, {"cond_fields": [2], "n_obs": [256]}, rows, n_steps=2, ode_solver="euler", obs_consistency="default_hard", generation_seed=seed, reconstruction_execution_mode="legacy_full")
            repeat = loaded.reconstruct(state, {"cond_fields": [2], "n_obs": [256]}, rows, n_steps=2, ode_solver="euler", obs_consistency="default_hard", generation_seed=seed, reconstruction_execution_mode=args.execution_mode, reconstruction_query_chunk_size=int(plan["inference"]["query_chunk_size"]))
            repeat_again = loaded.reconstruct(state, {"cond_fields": [2], "n_obs": [256]}, rows, n_steps=2, ode_solver="euler", obs_consistency="default_hard", generation_seed=seed, reconstruction_execution_mode=args.execution_mode, reconstruction_query_chunk_size=int(plan["inference"]["query_chunk_size"]))
            same_seed_error = float(torch.max(torch.abs(repeat["recon"] - repeat_again["recon"])).item())
            cache_equivalence_error = float(torch.max(torch.abs(legacy["recon"] - repeat["recon"])).item())

        coverage_rows = aggregate_coverage(result_rows, plan)
        write_csv(output_dir / "coverage_by_level.csv", coverage_rows)
        associations = association_summary(result_rows, plan) if len(states) >= 3 else []
        qa = {
            "status": "pass",
            "expected_per_state_rows": len(states) * len(sensor_counts) * len(FIELD_NAMES),
            "actual_per_state_rows": len(result_rows),
            "finite_metrics": bool(all(np.isfinite(float(value)) for row in result_rows for key, value in row.items() if key not in {"job", "field"})),
            "max_sensor_clamp_error_normalized": max(clamp_errors),
            "sensor_clamp_pass": bool(max(clamp_errors) <= float(tolerance["sensor_clamp_atol_normalized"])),
            "max_roundtrip_error": max(roundtrip_errors),
            "roundtrip_within_pilot_tolerance": bool(max(roundtrip_errors) <= float(tolerance["float32_roundtrip_atol"])),
            "roundtrip_gate_required": args.job == "PILOT",
            "minimum_unobserved_variance": min(variance_values),
            "noncollapsed_pass": bool(min(variance_values) > float(tolerance["minimum_unobserved_variance"])),
            "same_seed_max_abs_error": same_seed_error,
            "same_seed_pass": bool(np.isnan(same_seed_error) or same_seed_error <= float(tolerance["same_seed_atol"])),
            "cache_equivalence_max_abs_error": cache_equivalence_error,
            "cache_equivalence_pass": bool(np.isnan(cache_equivalence_error) or cache_equivalence_error <= float(tolerance["cache_equivalence_atol"])),
            "associations": associations,
        }
        required = [
            qa["actual_per_state_rows"] == qa["expected_per_state_rows"],
            qa["finite_metrics"], qa["sensor_clamp_pass"], qa["noncollapsed_pass"],
        ]
        if args.job == "PILOT":
            required.extend([qa["roundtrip_within_pilot_tolerance"], qa["same_seed_pass"], qa["cache_equivalence_pass"]])
        qa["status"] = "pass" if all(required) else "fail"
        atomic_json(output_dir / "qa.json", qa)
        manifest["status"] = "complete" if qa["status"] == "pass" else "qa_failed"
        manifest["completed_at"] = datetime.now().isoformat()
        manifest["associations"] = associations
        atomic_json(output_dir / "manifest.json", manifest)
        print(json.dumps({"output_dir": str(output_dir), "qa": qa}, indent=2, default=str))
        return 0 if qa["status"] == "pass" else 2
    finally:
        loaded.close()


if __name__ == "__main__":
    raise SystemExit(main())

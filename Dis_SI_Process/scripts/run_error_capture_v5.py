#!/usr/bin/env python
"""Stream Figure 5 V5 spatial uncertainty error-capture summaries.

The runner deliberately mirrors the formal V3 repeated-draw protocol while
retaining only state-level error-capture curves.  Full ensemble fields never
leave memory and are released before the next state is evaluated.
"""
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


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
MODEL_SCRIPT_DIR = (
    REPO_ROOT
    / "0_demo_TurbulentCombustion"
    / "Save_TrainedModel"
    / "_TrainedModels"
    / "_Scripts"
)
TOOLS_DIR = REPO_ROOT / "0_demo_TurbulentCombustion" / "tools"
for path in (REPO_ROOT, MODEL_SCRIPT_DIR, TOOLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.config import load_config, stable_seed  # noqa: E402
from common.model_loader import load_model  # noqa: E402
from benchmark_validation_v3 import core_call, evaluation_context, prepare_state  # noqa: E402
from Dis_SI_Process.utils.figure5_v5_data import error_capture_curve  # noqa: E402


METHODS = ("DMF-Gen", "FFM-FNO", "FFM-Perceiver", "Latent FM", "SiT")
FIELDS = ("Y_CH4", "Y_CO", "T", "U1", "p")
UNOBSERVED = (0, 1, 3, 4)
FRACTIONS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)
DATASET = "turbulent_combustion"
TASK = "missing_channel_reconstruction"
CONDITION = "Cond_T"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPO_ROOT
        / "0_demo_TurbulentCombustion"
        / "Save_TrainedModel"
        / "_TrainedModels"
        / "_ValidationPlans"
        / "validation_v1.yaml",
    )
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--state-limit", type=int)
    parser.add_argument("--draw-limit", type=int)
    parser.add_argument("--run-id", default="uq_localization_formal_v5")
    parser.add_argument("--overwrite-failed", action="store_true")
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
        query = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader"],
            text=True,
        )
        result["gpu_inventory"] = query.strip().splitlines()
    except Exception as exc:  # pragma: no cover - environment dependent
        result["gpu_inventory_error"] = str(exc)
    return result


def verify_identities(plan: dict[str, Any], methods: tuple[str, ...]) -> list[dict[str, Any]]:
    checkpoint_map = {row["method"]: row for row in plan["checkpoints"]}
    items = [
        ("dataset", plan["dataset"]["path"], plan["dataset"]["sha256"]),
        (
            "dataset_statistics",
            plan["dataset_statistics"]["path"],
            plan["dataset_statistics"]["sha256"],
        ),
        ("sensor_plan", plan["sensor_plan"]["path"], plan["sensor_plan"]["sha256"]),
    ]
    items.extend(
        (method, checkpoint_map[method]["path"], checkpoint_map[method]["sha256"])
        for method in methods
    )
    checks = []
    for label, raw_path, expected in items:
        path = resolve(raw_path)
        actual = sha256(path)
        checks.append(
            {
                "label": label,
                "path": str(path),
                "expected_sha256": expected,
                "actual_sha256": actual,
                "pass": actual == expected,
            }
        )
    if not all(row["pass"] for row in checks):
        raise RuntimeError("Frozen V5 identity check failed")
    return checks


def verify_v3_protocol(plan: dict[str, Any], methods: tuple[str, ...], states: list[int]) -> dict[str, Any]:
    directory = (
        PACKAGE_ROOT
        / "results"
        / "ValidationV3"
        / "UQCompare"
        / "uq_compare_formal_20260830_v3r6"
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    qa = json.loads((directory / "qa.json").read_text(encoding="utf-8"))
    checks = {
        "schema": manifest.get("schema_version") == "figure5-validation-v3-uq-1",
        "formal_complete": manifest.get("formal") is True and manifest.get("status") == "complete",
        "qa_pass": qa.get("status") == "pass",
        "methods": tuple(manifest.get("methods", [])) == methods,
        "states": list(map(int, manifest.get("states", []))) == states,
        "draws": int(manifest.get("draws_per_state", -1)) == 64,
        "sensor_count": int(manifest.get("sensor_count", -1)) == 256,
        "seed_schedule": manifest.get("seed_schedule")
        == "stable_seed(20260830,'generation','U2',state,draw); shared across methods",
    }
    if not all(checks.values()):
        raise RuntimeError(f"V3 formal protocol is incompatible with V5: {checks}")
    return {"directory": str(directory), "checks": checks, "manifest_sha256": sha256(directory / "manifest.json")}


def load_sensor_rows(plan: dict[str, Any], states: list[int]) -> dict[int, list[dict[str, int]]]:
    groups: dict[int, list[dict[str, int]]] = {state: [] for state in states}
    with resolve(plan["sensor_plan"]["path"]).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            state = int(row["snapshot"])
            if row["condition"] == CONDITION and state in groups:
                groups[state].append(
                    {key: int(row[key]) for key in ("sensor_order", "field_index", "point_index", "sensor_seed")}
                )
    for state, rows in groups.items():
        rows.sort(key=lambda row: row["sensor_order"])
        if len(rows) != 256 or {row["field_index"] for row in rows} != {2}:
            raise RuntimeError(f"State {state} lacks the exact 256-row Cond_T sensor plan")
    return groups


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


def moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([(np.arange(block) + start) % n for start in starts])[:n]


def bootstrap_mean(values: np.ndarray, spec: dict[str, Any], salt: str) -> tuple[float, float]:
    rng = np.random.default_rng(stable_seed(int(spec["seed"]), salt))
    block = min(int(spec["block_length"]), len(values))
    samples = np.empty(int(spec["replicates"]), dtype=float)
    for index in range(len(samples)):
        samples[index] = float(np.mean(values[moving_block_indices(len(values), block, rng)]))
    alpha = (1.0 - float(spec["confidence_level"])) / 2.0
    return tuple(float(value) for value in np.quantile(samples, [alpha, 1.0 - alpha]))


def summarize(curves: list[dict[str, Any]], plan: dict[str, Any], methods: tuple[str, ...]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    bootstrap = plan["bootstrap"]
    for method in methods:
        group = sorted(
            (row for row in curves if row["method"] == method),
            key=lambda row: int(row["original_time_index"]),
        )
        auc = np.asarray([float(row["ec_auc"]) for row in group], dtype=float)
        auc_low, auc_high = bootstrap_mean(auc, bootstrap, f"v5|ec_auc|macro|{method}")
        for fraction in FRACTIONS:
            key = f"capture_{fraction:.2f}"
            values = np.asarray([float(row[key]) for row in group], dtype=float)
            low, high = bootstrap_mean(values, bootstrap, f"v5|capture|macro|{method}|{fraction:.2f}")
            summary.append(
                {
                    "dataset": DATASET,
                    "task": TASK,
                    "condition": CONDITION,
                    "method": method,
                    "field": "macro_unobserved",
                    "cohort_id": "calibration_200",
                    "metric_name": "error_capture_fraction",
                    "spatial_fraction": fraction,
                    "metric_value": float(np.mean(values)),
                    "ci_low": low,
                    "ci_high": high,
                    "state_count": len(group),
                    "ec_auc": float(np.mean(auc)),
                    "ec_auc_ci_low": auc_low,
                    "ec_auc_ci_high": auc_high,
                }
            )
        for field in (FIELDS[index] for index in UNOBSERVED):
            field_auc = np.asarray([float(row[f"ec_auc_{field}"]) for row in group], dtype=float)
            field_auc_low, field_auc_high = bootstrap_mean(
                field_auc, bootstrap, f"v5|ec_auc|{field}|{method}"
            )
            for fraction in FRACTIONS:
                key = f"capture_{field}_{fraction:.2f}"
                values = np.asarray([float(row[key]) for row in group], dtype=float)
                low, high = bootstrap_mean(
                    values, bootstrap, f"v5|capture|{field}|{method}|{fraction:.2f}"
                )
                summary.append(
                    {
                        "dataset": DATASET,
                        "task": TASK,
                        "condition": CONDITION,
                        "method": method,
                        "field": field,
                        "cohort_id": "calibration_200",
                        "metric_name": "error_capture_fraction",
                        "spatial_fraction": fraction,
                        "metric_value": float(np.mean(values)),
                        "ci_low": low,
                        "ci_high": high,
                        "state_count": len(group),
                        "ec_auc": float(np.mean(field_auc)),
                        "ec_auc_ci_low": field_auc_low,
                        "ec_auc_ci_high": field_auc_high,
                    }
                )
    return summary


def main() -> int:
    args = parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    if plan.get("schema_version") != "figure5-validation-v1":
        raise ValueError("Unsupported frozen validation plan")
    methods = METHODS if "all" in args.methods else tuple(method for method in METHODS if method in set(args.methods))
    states = list(map(int, plan["cohorts"]["calibration_200"]["evaluation_indices"]))
    if args.state_limit:
        states = states[: args.state_limit]
    draws_per_state = min(64, args.draw_limit) if args.draw_limit else 64
    formal = methods == METHODS and len(states) == 200 and draws_per_state == 64
    run_dir = PACKAGE_ROOT / "results" / "ValidationV5" / "UQLocalization" / args.run_id
    if run_dir.exists():
        existing = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8")) if (run_dir / "manifest.json").is_file() else {}
        if existing.get("status") == "complete":
            raise FileExistsError(f"Completed run is immutable: {run_dir}")
        if not args.overwrite_failed:
            raise FileExistsError(f"Existing incomplete run requires --overwrite-failed: {run_dir}")
        for name in ("error_capture_curves.csv", "error_capture_summary.csv", "qa.json", "manifest.json"):
            path = run_dir / name
            if path.is_file():
                path.unlink()
    run_dir.mkdir(parents=True, exist_ok=True)

    identity = verify_identities(plan, methods)
    v3_protocol = verify_v3_protocol(plan, METHODS, list(map(int, plan["cohorts"]["calibration_200"]["evaluation_indices"])))
    sensors = load_sensor_rows(plan, states)
    original_times = list(map(int, plan["test_states"]["original_hdf5_time_indices"]))
    post_cfg = load_config()
    method_cfg = {row["name"]: row for row in post_cfg["methods"]}
    checkpoint_map = {row["method"]: row for row in plan["checkpoints"]}
    manifest: dict[str, Any] = {
        "schema_version": "figure5-validation-v5-uq-localization-1",
        "status": "running",
        "formal": formal,
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET,
        "task": TASK,
        "condition": CONDITION,
        "methods": list(methods),
        "states": states,
        "draws_per_state": draws_per_state,
        "sensor_count": 256,
        "native_query_count": 40300,
        "fields": [FIELDS[index] for index in UNOBSERVED],
        "field_weights": [0.25] * 4,
        "fractions": list(FRACTIONS),
        "seed_schedule": "stable_seed(20260830,'generation','U2',state,draw); shared across methods",
        "identity_checks": identity,
        "v3_protocol_reuse": v3_protocol,
        "bootstrap": plan["bootstrap"],
        "environment": capture_environment(args.device),
        "storage_policy": {
            "full_ensemble_fields_saved": False,
            "per_draw_files_saved": False,
            "scratch_directory_created": False,
            "retained": ["error_capture_curves.csv", "error_capture_summary.csv", "manifest.json", "qa.json"],
        },
    }
    atomic_json(run_dir / "manifest.json", manifest)
    curves: list[dict[str, Any]] = []
    try:
        for method in methods:
            settings = method_settings(post_cfg, method)
            loaded = load_model(
                method_cfg[method],
                CONDITION,
                checkpoint="last",
                split="test",
                device=args.device,
                n_steps=settings["n_steps"],
                ode_solver="euler",
            )
            dataset_mean = np.asarray(
                loaded.dataset.mean.detach().cpu() if torch.is_tensor(loaded.dataset.mean) else loaded.dataset.mean,
                dtype=np.float32,
            )
            dataset_std = np.asarray(
                loaded.dataset.std.detach().cpu() if torch.is_tensor(loaded.dataset.std) else loaded.dataset.std,
                dtype=np.float32,
            )
            frozen_mean = np.asarray(plan["dataset_statistics"]["mean"], dtype=np.float32)
            frozen_std = np.asarray(plan["dataset_statistics"]["std"], dtype=np.float32)
            if not (
                np.allclose(dataset_mean, frozen_mean, rtol=1e-6, atol=1e-6)
                and np.allclose(dataset_std, frozen_std, rtol=1e-6, atol=1e-6)
            ):
                raise RuntimeError(f"{method} normalization differs from the frozen common scale")
            try:
                with evaluation_context(loaded), torch.no_grad():
                    for position, state in enumerate(states):
                        seeds = [
                            stable_seed(int(plan["generation_seeds"]["base"]), "generation", "U2", state, draw)
                            for draw in range(draws_per_state)
                        ]
                        prepared = prepare_state(loaded, state, sensors[state], 40300, method, retain_truth=True)
                        draws = []
                        for seed in seeds:
                            torch.manual_seed(seed)
                            np.random.seed(seed % (2**32))
                            torch.cuda.manual_seed_all(seed)
                            draws.append(core_call(loaded, prepared, settings, method)[0].detach())
                        stack_device = torch.stack(draws)
                        stack = stack_device.cpu().numpy().astype(np.float32, copy=False)
                        truth = prepared["truth"][0].detach().cpu().numpy().astype(np.float32, copy=False)
                        physical = stack * dataset_std[None, None, :] + dataset_mean[None, None, :]
                        truth_physical = truth * dataset_std[None, :] + dataset_mean[None, :]
                        field_curves: list[np.ndarray] = []
                        field_auc: dict[str, float] = {}
                        row: dict[str, Any] = {
                            "dataset": DATASET,
                            "task": TASK,
                            "condition": CONDITION,
                            "method": method,
                            "checkpoint_sha256": checkpoint_map[method]["sha256"],
                            "state_id": state,
                            "original_time_index": original_times[state],
                            "cohort_id": "calibration_200",
                            "draw_count": draws_per_state,
                            "metric_name": "spatial_error_capture_curve",
                        }
                        for field_index in UNOBSERVED:
                            field = FIELDS[field_index]
                            mean_field = np.mean(physical[:, :, field_index], axis=0)
                            std_field = np.std(physical[:, :, field_index], axis=0, ddof=1)
                            absolute_error = np.abs(mean_field - truth_physical[:, field_index])
                            capture = error_capture_curve(std_field, absolute_error, FRACTIONS)
                            auc = float(
                                np.trapezoid(
                                    np.concatenate(([0.0], capture)),
                                    np.concatenate(([0.0], np.asarray(FRACTIONS))),
                                )
                                - 0.5
                            )
                            field_curves.append(capture)
                            field_auc[field] = auc
                            row[f"ec_auc_{field}"] = auc
                            for fraction, value in zip(FRACTIONS, capture):
                                row[f"capture_{field}_{fraction:.2f}"] = float(value)
                        macro = np.mean(np.stack(field_curves), axis=0)
                        row["ec_auc"] = float(
                            np.trapezoid(
                                np.concatenate(([0.0], macro)),
                                np.concatenate(([0.0], np.asarray(FRACTIONS))),
                            )
                            - 0.5
                        )
                        for fraction, value in zip(FRACTIONS, macro):
                            row[f"capture_{fraction:.2f}"] = float(value)
                        curves.append(row)
                        write_csv(run_dir / "error_capture_curves.csv", curves)

                        del field_curves, physical, truth_physical, stack, stack_device, draws, prepared
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if position % 10 == 0 or position + 1 == len(states):
                            print(f"[{method}] {position + 1}/{len(states)} states", flush=True)
            finally:
                loaded.close()

        summary = summarize(curves, plan, methods)
        write_csv(run_dir / "error_capture_summary.csv", summary)
        expected_rows = len(methods) * len(states)
        captures = np.asarray(
            [[float(row[f"capture_{fraction:.2f}"]) for fraction in FRACTIONS] for row in curves],
            dtype=float,
        )
        qa_checks = {
            "formal_protocol": formal,
            "expected_state_method_rows": len(curves) == expected_rows,
            "all_finite": bool(np.isfinite(captures).all()),
            "within_unit_interval": bool(((captures >= -1e-12) & (captures <= 1.0 + 1e-12)).all()),
            "curves_monotone": bool((np.diff(captures, axis=1) >= -1e-12).all()),
            "curves_end_at_one": bool(np.allclose(captures[:, -1], 1.0, rtol=0.0, atol=1e-12)),
            "paired_state_cohort": all(
                {int(row["state_id"]) for row in curves if row["method"] == method} == set(states)
                for method in methods
            ),
            "field_macro_equal_weight": True,
            "no_ensemble_stack_files": not any(run_dir.glob("*.npz")) and not any(run_dir.glob("*.npy")),
            "summary_complete": len(summary) == len(methods) * (1 + len(UNOBSERVED)) * len(FRACTIONS),
        }
        required_checks = {key: value for key, value in qa_checks.items() if key != "formal_protocol"}
        qa = {"status": "pass" if all(required_checks.values()) else "fail", "checks": qa_checks}
        atomic_json(run_dir / "qa.json", qa)
        manifest["status"] = "complete" if qa["status"] == "pass" else "qa_failed"
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["retained_bytes"] = sum(path.stat().st_size for path in run_dir.iterdir() if path.is_file())
        atomic_json(run_dir / "manifest.json", manifest)
        print(json.dumps({"run_dir": str(run_dir), "qa": qa["status"], "rows": len(curves)}, indent=2))
        return 0 if qa["status"] == "pass" else 2
    except Exception as exc:
        for name in ("error_capture_curves.csv", "error_capture_summary.csv"):
            path = run_dir / name
            if path.is_file():
                path.unlink()
        failure = {
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "completed_state_method_rows_before_cleanup": len(curves),
            "partial_curve_table_removed": True,
        }
        atomic_json(run_dir / "qa.json", failure)
        manifest["status"] = "failed"
        manifest["failure"] = failure
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(run_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())

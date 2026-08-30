"""Recipe aliases, expected protocol metadata, and manifest validation."""
from __future__ import annotations
from pathlib import Path
from typing import Any


def resolve_recipe_dir(model_dir: Path, recipe_key: str, spec: dict[str, Any]) -> Path:
    candidates = [recipe_key, *spec.get("directory_aliases", [])]
    for name in candidates:
        path = model_dir / name
        if path.is_dir():
            return path
    return model_dir / recipe_key


def flatten_run_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Return one uniform view for FFM and deterministic run configs."""
    if "shared" not in raw:
        flat = dict(raw)
        # Early point-cloud runs predate this explicit option.  Training used
        # the full case set in that era, which is also the current default.
        flat.setdefault("multires_train_case_fraction", 1.0)
        return flat
    shared = raw.get("shared", {})
    data = shared.get("data", {})
    cond = shared.get("conditioning", {})
    return {
        **raw,
        "seed": shared.get("seed", 42),
        "dataset_mode": data.get("dataset_mode"),
        "pdebench_dataset_name": data.get("dataset_name"),
        "pdebench_processed_root": data.get("pdebench_processed_root"),
        "selected_field_idx_raw": data.get("selected_field_idx_raw"),
        "multires_ratio": data.get("multires_ratio"),
        "multires_train_case_fraction": data.get("multires_train_case_fraction"),
        "multires_manifest_path": data.get("multires_manifest_path", ""),
        "eval_resolution": data.get("eval_resolution", "H"),
        "train_ratio": data.get("train_ratio", .9),
        "Case_Truncate_Ratio": data.get("Case_Truncate_Ratio", 0.0),
        "cond_fields": cond.get("cond_fields", [0]),
        "n_obs_max_list": cond.get("n_obs_max_list", [512]),
        "backbone": raw.get("baseline_model"),
    }


def validate_recipe(recipe_key: str, spec: dict[str, Any], cfg: dict[str, Any], manifest: dict | None) -> dict[str, Any]:
    actual_ratio = str(cfg.get("multires_ratio", ""))
    actual_fraction = float(cfg.get("multires_train_case_fraction", float("nan")))
    expected_fraction = float(spec["expected_case_fraction"])
    tol = float(spec.get("fraction_tolerance", 1e-6))
    mismatch = actual_ratio != str(spec["expected_ratio"]) or abs(actual_fraction - expected_fraction) > tol
    counts = {"L": 0, "M": 0, "H": 0}
    val_count = 0
    n_time = 0
    if manifest:
        split = manifest.get("split", {})
        by_res = split.get("train_cases_by_res", {})
        counts = {tag: len(by_res.get(tag, [])) for tag in "LMH"}
        val_count = len(split.get("val_cases", []))
        n_time = int(manifest.get("n_time", 0))
        manifest_ratio = str(manifest.get("multires_ratio", actual_ratio))
        manifest_fraction = float(manifest.get("multires_train_case_fraction", actual_fraction))
        mismatch |= manifest_ratio != actual_ratio or abs(manifest_fraction - actual_fraction) > 1e-8
    return {
        "recipe": recipe_key, "recipe_label": spec["label"],
        "expected_ratio": spec["expected_ratio"], "actual_ratio": actual_ratio,
        "expected_case_fraction": expected_fraction, "actual_case_fraction": actual_fraction,
        "train_cases_L": counts["L"], "train_cases_M": counts["M"], "train_cases_H": counts["H"],
        "active_train_cases": sum(counts.values()), "active_train_trajectories": sum(counts.values()),
        "train_snapshots": sum(counts.values()) * n_time, "test_cases": val_count,
        "status": "config_recipe_mismatch" if mismatch else "ok",
    }

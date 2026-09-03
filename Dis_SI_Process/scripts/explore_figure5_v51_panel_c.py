#!/usr/bin/env python
"""Explore the Figure 5 V5.1 panel-C candidates.

The panel-C exploration is deliberately additive.  ``C1`` is calculated from
the accepted V3 paired state table and therefore does not require new model
inference.  ``C2``--``C4`` are reducers/renderers for compact pointwise output
that may be produced later by the explicitly gated ``--execute-pointwise``
path.  The default command only audits the frozen contract, writes the C1
candidate files, and records that pointwise inference is waiting for the
GPU2-free confirmation.

The pointwise path is intentionally conservative: one state/method ensemble is
held in memory, reduced immediately, and no full draw file is written.  Model
imports are lazy so the plotting/audit command remains runnable in the ``fig``
environment, where the project inference stack is not installed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

# Must be present before the lazy torch import in the gated inference path;
# otherwise torch's deterministic CuBLAS guard aborts the first GEMM.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.figure5_v41_style import (  # noqa: E402
    MM,
    add_panel_label,
    apply_style,
    method_colors,
    method_markers,
    save_svg,
    style_grid,
)


DATASET = "turbulent_combustion"
TASK = "missing_channel_reconstruction"
CONDITION = "Cond_T"
SCHEMA_VERSION = "figure5-validation-v5.1-panel-c-exploration-1"
QA_SCHEMA_VERSION = "figure5-validation-v5.1-panel-c-exploration-qa-1"
METHODS = ["DMF-Gen", "FFM-FNO", "FFM-Perceiver", "Latent FM", "SiT"]
EXPECTED_V3_AUDIT = {
    "DMF-Gen": {
        "obs_consistency_requested": "default_hard",
        "obs_consistency_applied": "default_hard",
        "n_steps": 2,
        "solver": "euler",
        "measured_nfe": 2,
        "obs_consistency": "default_hard",
        "execution_mode": "cached_streamed",
        "cache_level": "static_features",
    },
    "FFM-FNO": {
        "obs_consistency_requested": "endpoint_smooth",
        "obs_consistency_applied": "endpoint_smooth",
        "n_steps": 2,
        "solver": "heun_native",
        "measured_nfe": 4,
        "obs_consistency": "endpoint_smooth",
        "execution_mode": "legacy_full",
        "cache_level": "none",
    },
    "FFM-Perceiver": {
        "obs_consistency_requested": "endpoint_smooth",
        "obs_consistency_applied": "endpoint_smooth",
        "n_steps": 2,
        "solver": "euler",
        "measured_nfe": 2,
        "obs_consistency": "endpoint_smooth",
        "execution_mode": "legacy_full",
        "cache_level": "none",
    },
    "Latent FM": {
        "obs_consistency_requested": "endpoint_smooth",
        "obs_consistency_applied": "native_not_applied",
        "n_steps": 2,
        "solver": "euler",
        "measured_nfe": 2,
        "obs_consistency": "endpoint_smooth",
        "execution_mode": "legacy_full",
        "cache_level": "none",
    },
    "SiT": {
        "obs_consistency_requested": "endpoint_smooth",
        "obs_consistency_applied": "native_not_applied",
        "n_steps": 4,
        "solver": "euler",
        "measured_nfe": 4,
        "obs_consistency": "endpoint_smooth",
        "execution_mode": "legacy_full",
        "cache_level": "none",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_value(repo_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def stable_seed(base: int, *parts: Any) -> int:
    """Match ``common.config.stable_seed`` used by V3/U2 generation."""

    payload = "|".join(map(str, (base, *parts))).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) & 0x7FFFFFFF


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "figure5-v51-panel-c-exploration-1":
        raise ValueError(f"Unexpected panel-C config schema: {path}")
    if str(config.get("timestamp")) != "20260902_1129":
        raise ValueError("Panel-C config timestamp is not the frozen 20260902_1129 contract")
    if list(config.get("methods", [])) != METHODS:
        raise ValueError("Panel-C method order differs from the frozen V3/V5 order")
    return config


def _required_formal(directory: Path, schema: str, files: Iterable[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    required = ["manifest.json", "qa.json", *files]
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required formal files under {directory}: {missing}")
    manifest = _read_json(directory / "manifest.json")
    qa = _read_json(directory / "qa.json")
    if manifest.get("schema_version") != schema or manifest.get("formal") is not True or manifest.get("status") != "complete":
        raise ValueError(f"Source is not formal/complete {schema}: {directory}")
    if qa.get("status") != "pass":
        raise ValueError(f"Source QA did not pass: {directory}")
    return manifest, qa


def _as_int(value: Any) -> int:
    return int(float(value))


def _audit_row_value(row: pd.Series, key: str) -> Any:
    value = row[key]
    if key in {"n_steps", "measured_nfe"}:
        return _as_int(value)
    return str(value)


def _audit_expected_from_config(config: Mapping[str, Any], method: str) -> dict[str, Any]:
    overrides = config["panel_c"]["pointwise_protocol"]["method_inference_overrides"]
    raw = dict(overrides[method])
    expected = {
        "obs_consistency_requested": str(raw.get("obs_consistency_requested", raw.get("obs_consistency"))),
        "obs_consistency_applied": str(raw.get("obs_consistency_applied", raw.get("obs_consistency_requested", raw.get("obs_consistency")))),
        "n_steps": _as_int(raw["n_steps"]),
        "measured_nfe": _as_int(raw["measured_nfe"]),
        "execution_mode": str(raw["execution_mode"]),
        "cache_level": str(raw["cache_level"]),
    }
    # The canonical audit's solver is the recorded inference solver.  FFM-FNO
    # is the only method whose metadata is heun_native; all other methods use
    # Euler.  Keep both values explicit so a future runner cannot silently
    # collapse a native setting into a global default.
    expected["solver"] = "heun_native" if method == "FFM-FNO" else "euler"
    expected["obs_consistency"] = expected["obs_consistency_requested"]
    return expected


def _compare_method_audit(config: Mapping[str, Any], audit: pd.DataFrame) -> dict[str, Any]:
    required = {
        "method",
        "obs_consistency_requested",
        "obs_consistency_applied",
        "n_steps",
        "solver",
        "measured_nfe",
        "obs_consistency",
        "execution_mode",
        "cache_level",
    }
    missing = sorted(required.difference(audit.columns))
    if missing:
        raise ValueError(f"V3 method_draw_audit is missing contract columns: {missing}")
    observed: dict[str, Any] = {}
    for method in METHODS:
        rows = audit.loc[audit["method"].astype(str) == method]
        if rows.empty:
            raise ValueError(f"V3 method_draw_audit has no rows for {method}")
        values = []
        for key in sorted(required.difference({"method"})):
            values.append((key, tuple(sorted({_audit_row_value(row, key) for _, row in rows.iterrows()}))))
        unique = {key: value for key, value in values}
        expected = _audit_expected_from_config(config, method)
        mismatches: dict[str, Any] = {}
        for key, expected_value in expected.items():
            actual_values = unique[key]
            if actual_values != (expected_value,):
                mismatches[key] = {"expected": expected_value, "observed": list(actual_values)}
        if mismatches:
            raise ValueError(f"V3 method-draw contract mismatch for {method}: {mismatches}")
        observed[method] = {key: list(value) for key, value in unique.items()}
    return observed


def _load_v3_sources(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    root = _repo_path(config["v3_uq_root"], repo_root)
    manifest, qa = _required_formal(
        root,
        "figure5-validation-v3-uq-1",
        ("per_state_method.csv", "method_draw_audit.csv"),
    )
    methods = [str(value) for value in manifest.get("methods", [])]
    if methods != METHODS:
        raise ValueError(f"V3 methods differ from frozen panel-C order: {methods}")
    states = [int(value) for value in manifest.get("states", [])]
    if len(states) != 200 or states != sorted(states) or len(set(states)) != 200:
        raise ValueError("V3 manifest does not declare the paired 200-state cohort")
    if int(manifest.get("draws_per_state", -1)) != 64 or int(manifest.get("sensor_count", -1)) != 256:
        raise ValueError("V3 draw/sensor contract does not match panel C")
    expected_seed = "stable_seed(20260830,'generation','U2',state,draw); shared across methods"
    if str(manifest.get("seed_schedule")) != expected_seed:
        raise ValueError("V3 seed schedule differs from the frozen U2 schedule")
    table_path = root / "per_state_method.csv"
    table = pd.read_csv(table_path)
    required = {
        "method",
        "state",
        "original_time_index",
        "draw_count",
        "macro_normalized_spread",
        "macro_ensemble_mean_relative_l2",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"V3 paired state table is missing columns: {missing}")
    table["method"] = table["method"].astype(str)
    table["state"] = pd.to_numeric(table["state"], errors="raise").astype(int)
    table["original_time_index"] = pd.to_numeric(table["original_time_index"], errors="raise").astype(int)
    table["draw_count"] = pd.to_numeric(table["draw_count"], errors="raise").astype(int)
    if len(table) != 1000 or table.duplicated(["method", "state"]).any():
        raise ValueError("V3 paired table must contain exactly one row per method/state")
    counts = table.groupby("method", sort=False).size().to_dict()
    if counts != {method: 200 for method in METHODS}:
        raise ValueError(f"V3 paired table method counts mismatch: {counts}")
    state_sets = {method: set(table.loc[table["method"] == method, "state"]) for method in METHODS}
    if any(values != set(states) for values in state_sets.values()):
        raise ValueError("V3 methods do not share the same 200 state identifiers")
    mapping_counts = table.groupby("state")["original_time_index"].nunique()
    if bool((mapping_counts != 1).any()):
        raise ValueError("V3 state-to-original-time mapping is not one-to-one")
    if not (table["draw_count"] == 64).all():
        raise ValueError("V3 paired table has a draw count other than 64")
    numeric = table[["macro_normalized_spread", "macro_ensemble_mean_relative_l2"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("V3 selective-risk inputs are not finite and non-negative")
    audit = pd.read_csv(root / "method_draw_audit.csv")
    audit_settings = _compare_method_audit(config, audit)
    fixed = table.loc[(table["method"] == "DMF-Gen") & (table["state"] == int(config["panel_c"]["fixed_state"]["test_index"]))]
    if len(fixed) != 1 or int(fixed.iloc[0]["original_time_index"]) != int(config["panel_c"]["fixed_state"]["original_hdf5_time_index"]):
        raise ValueError("V3 fixed state does not exactly map Figure 4 test index 0 -> original index 5")
    return {
        "root": root,
        "manifest": manifest,
        "qa": qa,
        "table": table,
        "audit": audit,
        "audit_settings": audit_settings,
        "states": states,
        "source_paths": {
            "v3_manifest": root / "manifest.json",
            "v3_qa": root / "qa.json",
            "v3_state_table": table_path,
            "v3_method_draw_audit": root / "method_draw_audit.csv",
        },
    }


def _validate_fig4_state(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    path = _repo_path(config["fig4_state_source"], repo_root)
    if not path.is_file():
        raise FileNotFoundError(f"Missing Figure 4 state source: {path}")
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    found: list[int] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() in {"snapshot_index", "snapshot", "qualitative_snapshot_index"}:
                    try:
                        found.append(int(item))
                    except (TypeError, ValueError):
                        pass
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(source)
    expected = int(config["fig4_state_test_index"])
    if expected not in found:
        raise ValueError(f"Figure 4 source does not declare exact test snapshot_index={expected}: {path}")
    return {"path": path, "test_index": expected, "original_time_index": int(config["fig4_state_original_time_index"]), "declared_snapshot_indices": sorted(set(found))}


def _validate_c0(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    svg = _repo_path(config["outputs"]["c0_svg"], repo_root)
    source = _repo_path(config["outputs"]["c0_source"], repo_root)
    if not svg.is_file() or not source.is_file():
        raise FileNotFoundError(f"Existing C0 source/SVG is required: {svg}, {source}")
    summary = pd.read_csv(source)
    return {"svg": svg, "source": source, "source_rows": int(len(summary)), "source_sha256": _sha256(source), "svg_sha256": _sha256(svg)}


def _new_manifest(config: Mapping[str, Any], config_path: Path, repo_root: Path, run_id: str, *, pointwise_requested: bool, confirm_gpu2_free: bool) -> dict[str, Any]:
    panel = config["panel_c"]
    fixed = panel["fixed_state"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_before_model_output",
        "formal": False,
        "run_id": run_id,
        "timestamp": str(config["timestamp"]),
        "starting_head": _git_value(repo_root, "rev-parse", "HEAD"),
        "dataset": DATASET,
        "task": TASK,
        "condition": CONDITION,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "methods": METHODS,
        "selection_protocol": {
            "c0": "reuse existing formal C0; no rerun",
            "c1": panel["c1"],
            "fixed_state": fixed,
            "c2": panel["ch4_interface"],
            "c3": panel["c3_locations"],
            "c4": panel["c4_functionals"],
            "c2b_comparator": panel["c2b_overlay_comparator"],
        },
        "pointwise_protocol": panel["pointwise_protocol"],
        "pointwise_metric_scales": {
            "interface_profile_crps": "validation_plan.dataset_statistics.std[0] (physical Y_CH4 training normalization)",
            "derived_functional_residuals": "training-split functional mean and std, one functional value per training time",
        },
        "pointwise_execution": {
            "requested": bool(pointwise_requested),
            "confirm_gpu2_free": bool(confirm_gpu2_free),
            "status": "not_started",
            "device": "cuda:2",
            "full_draws_saved": False,
        },
        "source_paths": {},
    }


def _safe_completed_run(run_dir: Path, *, overwrite_failed: bool, allow_pointwise_extension: bool = False, allow_pointwise_render: bool = False) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = _read_json(manifest_path)
    status = str(manifest.get("status", ""))
    if status == "complete":
        if allow_pointwise_extension and manifest.get("pointwise_execution", {}).get("status") == "pending_gpu2_inference" and not (run_dir / "interface_profile.csv").exists():
            return
        if allow_pointwise_render and (run_dir / "interface_profile.csv").is_file():
            return
        raise FileExistsError(f"Refusing to overwrite complete panel-C run: {run_dir}")
    if allow_pointwise_extension and (run_dir / "interface_profile.csv").is_file():
        return
    if not overwrite_failed:
        raise FileExistsError(f"Existing incomplete panel-C run; pass --overwrite-failed to resume: {run_dir}")


def _moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """Return a circular moving-block bootstrap index vector of length ``n``."""

    if n <= 0:
        raise ValueError("Cannot bootstrap an empty cohort")
    if block_length <= 0:
        raise ValueError("Moving-block length must be positive")
    starts = rng.integers(0, n, size=int(math.ceil(n / block_length)))
    indices = np.concatenate([(start + np.arange(block_length)) % n for start in starts])
    return indices[:n]


def _risk_curve(spread: np.ndarray, error: np.ndarray, states: np.ndarray, fractions: Sequence[float]) -> tuple[np.ndarray, float]:
    spread = np.asarray(spread, dtype=float)
    error = np.asarray(error, dtype=float)
    states = np.asarray(states, dtype=int)
    if spread.ndim != 1 or error.shape != spread.shape or states.shape != spread.shape:
        raise ValueError("Selective-risk vectors must have identical one-dimensional shapes")
    if not np.isfinite(spread).all() or not np.isfinite(error).all() or (error < 0).any():
        raise ValueError("Selective-risk vectors must be finite and non-negative")
    # ``states`` is the deterministic tie break.  It keeps the selected
    # fraction reproducible when two uncertainty values are identical.
    order = np.lexsort((states, spread))
    full = float(np.mean(error))
    values = []
    for fraction in fractions:
        count = max(1, int(math.ceil(float(fraction) * len(error))))
        values.append(float(np.mean(error[order[:count]])))
    return np.asarray(values, dtype=float), full


def _selective_risk(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    bootstrap: bool = True,
) -> pd.DataFrame:
    """Calculate absolute and same-cohort-normalized C1 risk curves.

    Bootstrap samples are drawn in original-HDF5 time order and ranking is
    recomputed in every replicate.  The table remains in memory only; no
    duplicate V3 source table is written.
    """

    c1 = config["panel_c"]["c1"]
    fractions = np.asarray(c1["coverage_fractions"], dtype=float)
    if len(fractions) != 9 or not np.allclose(fractions, np.linspace(0.2, 1.0, 9)):
        raise ValueError("C1 coverage grid must be the frozen 0.20--1.00 grid")
    block_length = int(config["bootstrap"]["block_length"])
    replicates = int(config["bootstrap"]["replicates"]) if bootstrap else 0
    confidence = float(config["bootstrap"]["confidence_level"])
    alpha = (1.0 - confidence) / 2.0
    records: list[dict[str, Any]] = []
    for method in METHODS:
        subset = table.loc[table["method"] == method].sort_values(["original_time_index", "state"], kind="mergesort")
        spread = subset["macro_normalized_spread"].to_numpy(dtype=float)
        error = subset["macro_ensemble_mean_relative_l2"].to_numpy(dtype=float)
        states = subset["state"].to_numpy(dtype=int)
        original_time = subset["original_time_index"].to_numpy(dtype=int)
        values, full = _risk_curve(spread, error, states, fractions)
        bootstrap_values = {"absolute": [], "normalized": []}
        if bootstrap and replicates:
            rng = np.random.default_rng(stable_seed(int(config["bootstrap"]["seed"]), "panel_c", "selective_risk", method))
            for _ in range(replicates):
                sample = _moving_block_indices(len(subset), block_length, rng)
                sampled_values, sampled_full = _risk_curve(spread[sample], error[sample], states[sample], fractions)
                bootstrap_values["absolute"].append(sampled_values)
                if sampled_full > 0:
                    bootstrap_values["normalized"].append(sampled_values / sampled_full)
                else:
                    bootstrap_values["normalized"].append(np.ones_like(sampled_values))
        else:
            bootstrap_values = {"absolute": [values], "normalized": [values / full if full > 0 else np.ones_like(values)]}
        normalized = values / full if full > 0 else np.ones_like(values)
        for risk_kind, point_values in (("absolute", values), ("normalized", normalized)):
            samples = np.asarray(bootstrap_values[risk_kind], dtype=float)
            ci_low = np.quantile(samples, alpha, axis=0)
            ci_high = np.quantile(samples, 1.0 - alpha, axis=0)
            auc = float(np.trapezoid(point_values, fractions)) if hasattr(np, "trapezoid") else float(np.trapz(point_values, fractions))
            at_80 = float(point_values[np.isclose(fractions, 0.8)][0])
            reduction = float(full - at_80)
            relative_reduction = float(reduction / full) if full > 0 else float("nan")
            for index, fraction in enumerate(fractions):
                records.append(
                    {
                        "dataset": DATASET,
                        "task": TASK,
                        "condition": CONDITION,
                        "cohort_id": "V3_calibration_200",
                        "method": method,
                        "risk_kind": risk_kind,
                        "coverage_fraction": float(fraction),
                        "risk": float(point_values[index]),
                        "ci_low": float(ci_low[index]),
                        "ci_high": float(ci_high[index]),
                        "state_count": int(len(subset)),
                        "draw_count": 64,
                        "ranking_field": "macro_normalized_spread",
                        "response_field": "macro_ensemble_mean_relative_l2",
                        "risk_at_full": float(full if risk_kind == "absolute" else 1.0),
                        "risk_at_80": float(at_80 if risk_kind == "absolute" else normalized[np.isclose(fractions, 0.8)][0]),
                        "absolute_reduction_100_to_80": reduction if risk_kind == "absolute" else float(1.0 - normalized[np.isclose(fractions, 0.8)][0]),
                        "relative_reduction_100_to_80": relative_reduction if risk_kind == "absolute" else float(1.0 - normalized[np.isclose(fractions, 0.8)][0]),
                        "risk_auc": auc,
                        "auc_definition": "trapezoid over evaluated grid 0.20 to 1.00",
                        "bootstrap_method": "temporal_moving_block_bootstrap" if bootstrap else "none_smoke",
                        "bootstrap_replicates": int(replicates),
                        "bootstrap_block_length": block_length,
                        "bootstrap_confidence_level": confidence,
                        "bootstrap_seed": stable_seed(int(config["bootstrap"]["seed"]), "panel_c", "selective_risk", method),
                        "time_order_field": "original_time_index",
                        "selected_state_count": int(max(1, math.ceil(float(fraction) * len(subset)))),
                        "full_cohort_error": float(full),
                    }
                )
    result = pd.DataFrame.from_records(records)
    if len(result) != len(METHODS) * 9 * 2:
        raise AssertionError("Unexpected C1 row count")
    return result


def _draw_risk_curve(ax: plt.Axes, source: pd.DataFrame, method: str, *, normalized: bool, colors: Mapping[str, str], markers: Mapping[str, str]) -> None:
    risk_kind = "normalized" if normalized else "absolute"
    rows = source.loc[(source["method"] == method) & (source["risk_kind"] == risk_kind)].sort_values("coverage_fraction")
    color = colors[method]
    x = rows["coverage_fraction"].to_numpy(float)
    y = rows["risk"].to_numpy(float)
    lo = rows["ci_low"].to_numpy(float)
    hi = rows["ci_high"].to_numpy(float)
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0, zorder=1)
    ax.plot(x, y, color=color, marker=markers[method], label=method, zorder=3)


def _normalized_c1_ylim(source: pd.DataFrame, config: Mapping[str, Any]) -> tuple[float, float]:
    """Return the predeclared data-driven display range for C1b.

    The normalized selective-risk curves are close to one, so a zero-based
    axis hides the triage differences.  The range is still determined only by
    the finite bootstrap intervals, with a declared floor at 1.02 for the
    upper bound; no model-specific range is selected.
    """

    contract = config["panel_c"]["c1"].get("normalized_display_ylim", {})
    rows = source.loc[source["risk_kind"] == "normalized", ["ci_low", "risk", "ci_high"]]
    values = rows.to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite normalized C1 intervals")
    lower_padding = float(contract.get("lower_padding_fraction", 0.05))
    upper_padding = float(contract.get("upper_padding_fraction", 0.02))
    upper_floor = float(contract.get("upper_floor", 1.02))
    data_low = float(np.min(values))
    data_high = float(np.max(values))
    span = max(data_high - data_low, 1e-8)
    return max(0.0, data_low - lower_padding * span), max(upper_floor, data_high + upper_padding * span)


def _make_c1(source: pd.DataFrame, config: Mapping[str, Any], *, normalized: bool) -> plt.Figure:
    apply_style(config["style"].get("font_family"))
    colors, markers = method_colors(config), method_markers(config)
    fig, ax = plt.subplots(figsize=(89 * MM, 67 * MM))
    for method in METHODS:
        _draw_risk_curve(ax, source, method, normalized=normalized, colors=colors, markers=markers)
    if normalized:
        ax.axhline(1.0, color="#666666", linestyle=(0, (2, 2)), linewidth=0.75, zorder=2)
        ax.set_ylabel("Risk / full-cohort risk")
        ax.set_ylim(*_normalized_c1_ylim(source, config))
        title = "Uncertainty-guided triage"
    else:
        full = source.loc[source["risk_kind"] == "absolute"].groupby("method")["risk_at_full"].first()
        for method in METHODS:
            ax.axhline(float(full.loc[method]), color=colors[method], linestyle=(0, (2, 2)), linewidth=0.45, alpha=0.42, zorder=1)
        ax.set_ylabel("Mean relative $L_2$ risk")
        title = "Uncertainty supports selective reconstruction"
    ax.set_title(title, loc="left", pad=5.0, fontweight="semibold")
    ax.set_xlabel("Retained low-uncertainty fraction")
    ax.set_xlim(0.17, 1.03)
    ax.set_xticks(np.linspace(0.2, 1.0, 5))
    style_grid(ax, axis="y")
    ax.legend(
        handles=[Line2D([], [], color=colors[m], marker=markers[m], label=m, linewidth=1.05) for m in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=3,
        handlelength=1.4,
        columnspacing=0.9,
        borderaxespad=0.1,
    )
    add_panel_label(ax, "C1b" if normalized else "C1a")
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.30, top=0.84)
    return fig


def _c1_companion(stem: str, source: pd.DataFrame, *, normalized: bool, config: Mapping[str, Any], source_paths: Mapping[str, str]) -> str:
    kind = "normalized" if normalized else "absolute"
    lines = [
        f"# Figure 5 V5.1 panel-C candidate: {stem}",
        "",
        f"- Candidate: `{'C1b' if normalized else 'C1a'}` ({kind} selective reconstruction risk).",
        f"- SVG: `{stem}.svg`",
        "- Task: turbulent-combustion `Cond_T`, 256 observed temperature sensors, native 40,300-point query.",
        "",
        "## Scientific question and definition",
        "",
        "This candidate asks whether low ensemble spread can identify a retained subset with lower reconstruction risk. States are ranked by the macro normalized ensemble spread and the lowest-uncertainty fractions in the frozen paired 200-state V3 cohort are retained. This is uncertainty-guided triage/selective reconstruction, not failure prediction.",
        "",
        f"Coverage grid: `{', '.join(f'{value:.2f}' for value in config['panel_c']['c1']['coverage_fractions'])}`. Exact points are connected; no fitted smoothing is used.",
        "",
        "Confidence bands use the predeclared temporal moving-block bootstrap (block length 25, 2,000 replicates, 95% interval) ordered by original HDF5 time index. Ranking is recomputed within every bootstrap sample.",
        "",
        "## Summary metrics",
        "",
        "| Method | AUC | Risk at 80% | Risk at 100% | Relative reduction, 100% → 80% |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        rows = source.loc[(source["method"] == method) & (source["risk_kind"] == kind)].sort_values("coverage_fraction")
        row80 = rows.loc[np.isclose(rows["coverage_fraction"], 0.8)].iloc[0]
        row100 = rows.loc[np.isclose(rows["coverage_fraction"], 1.0)].iloc[0]
        lines.append(f"| {method} | {float(rows['risk_auc'].iloc[0]):.5f} | {float(row80['risk']):.5f} | {float(row100['risk']):.5f} | {float(row100['risk'] - row80['risk']):.5f} |")
    lines.extend(
        [
            "",
            "## Sources and limits",
            "",
            f"- V3 state table: `{source_paths['v3_state_table']}`",
            f"- V3 audit: `{source_paths['v3_method_draw_audit']}`",
            f"- Compact plot source: `{source_paths['selective_risk']}`",
            f"- Existing C0 (preserved for SI): `{source_paths['c0_svg']}` and `{source_paths['c0_source']}`",
            "- The curve is descriptive evidence for the adopted checkpoints and paired cohort. It does not establish causal performance under a changed measurement budget.",
        ]
    )
    return "\n".join(lines) + "\n"


def _add_figure_panel_label(fig: plt.Figure, label: str) -> None:
    """Place C2--C4 tags in the figure margin, away from headings."""

    artist = fig.text(0.015, 0.997, label, ha="left", va="top", fontsize=9.5, fontweight="bold", color="#202020")
    artist.set_gid(f"font-role:panel_label:{label}")


def empirical_crps(draws: np.ndarray, truth: np.ndarray) -> float:
    """Empirical CRPS reduced from one in-memory ensemble.

    The sorted identity avoids materialising an ``S x S`` pairwise distance
    matrix.  ``draws`` is expected to have draw axis zero; all other axes are
    reduced to one mean CRPS value.
    """

    values = np.asarray(draws, dtype=np.float64)
    target = np.asarray(truth, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] < 2 or values.shape[1:] != target.shape:
        raise ValueError("CRPS expects at least two draws and matching truth dimensions")
    if not np.isfinite(values).all() or not np.isfinite(target).all():
        raise ValueError("CRPS input contains non-finite values")
    count = values.shape[0]
    ordered = np.sort(values, axis=0)
    coefficients = (2.0 * np.arange(1, count + 1, dtype=np.float64) - count - 1.0).reshape((count,) + (1,) * (values.ndim - 1))
    dispersion = np.sum(coefficients * ordered, axis=0) / float(count * count)
    return max(float(np.mean(np.mean(np.abs(values - target[None, ...]), axis=0) - dispersion)), 0.0)


def _empirical_quantiles(values: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1 or array.shape[0] < 1:
        raise ValueError("Quantiles require a non-empty draw axis")
    if not np.isfinite(array).all():
        raise ValueError("Quantile input contains non-finite values")
    return np.quantile(array, np.asarray(levels, dtype=float), axis=0, method="linear")


def _grid_geometry(coords: np.ndarray, field: np.ndarray | None = None) -> dict[str, Any]:
    """Return deterministic rectangular-grid indexing and spacings.

    Coordinates are not assumed to be in a particular flattening order.  The
    returned ``grid_index`` maps each original flat point to ``(x, y)`` grid
    indices, while all reductions continue to use original flat indices for
    deterministic tie breaking.
    """

    values = np.asarray(coords, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values[:, :2]).all():
        raise ValueError("Expected finite coordinates with at least x and y columns")
    x_values = np.unique(values[:, 0])
    y_values = np.unique(values[:, 1])
    if len(x_values) * len(y_values) != len(values):
        raise ValueError("Point coordinates are not a complete rectangular grid")
    ix = np.searchsorted(x_values, values[:, 0])
    iy = np.searchsorted(y_values, values[:, 1])
    grid_index = np.column_stack((ix, iy)).astype(np.int64)
    flat_grid = ix * len(y_values) + iy
    if len(np.unique(flat_grid)) != len(values):
        raise ValueError("Point coordinates contain duplicate grid locations")
    dx = float(np.median(np.diff(x_values))) if len(x_values) > 1 else 1.0
    dy = float(np.median(np.diff(y_values))) if len(y_values) > 1 else 1.0
    if dx <= 0 or dy <= 0:
        raise ValueError("Grid coordinates must be strictly increasing")
    result: dict[str, Any] = {
        "x": x_values,
        "y": y_values,
        "grid_index": grid_index,
        "flat_grid": flat_grid,
        "shape": (len(x_values), len(y_values)),
        "spacing": (dx, dy),
    }
    if field is not None:
        array = np.asarray(field, dtype=float)
        if array.shape != (len(values),):
            raise ValueError("Field length does not match coordinates")
        grid = np.empty((len(x_values), len(y_values)), dtype=float)
        grid[ix, iy] = array
        result["field_grid"] = grid
    return result


def signed_distance_field(coords: np.ndarray, field: np.ndarray, threshold: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Calculate truth-only Euclidean distance to a linearly interpolated contour.

    The mesh has nonuniform coordinates, so a pixel-distance transform would
    be an incorrect metric.  Threshold crossings are first interpolated on
    every actual grid edge, then a nearest-neighbour query is performed in the
    normalized coordinate domain.  The sign is positive on the
    ``field >= threshold`` truth side.
    """

    geometry = _grid_geometry(coords, field)
    mask = np.asarray(field, dtype=float) >= float(threshold)
    if not mask.any() or mask.all():
        raise ValueError("Methane threshold does not create a two-sided interface")
    grid_mask = np.empty(geometry["shape"], dtype=bool)
    grid_index = geometry["grid_index"]
    grid_mask[grid_index[:, 0], grid_index[:, 1]] = mask
    field_grid = geometry["field_grid"]
    contour_points: list[tuple[float, float]] = []

    def add_crossing(first_value: float, second_value: float, first_xy: tuple[float, float], second_xy: tuple[float, float]) -> None:
        first_equal = bool(np.isclose(first_value, threshold, rtol=0.0, atol=1e-12))
        second_equal = bool(np.isclose(second_value, threshold, rtol=0.0, atol=1e-12))
        if first_equal:
            contour_points.append(first_xy)
        if second_equal:
            contour_points.append(second_xy)
        if first_equal or second_equal or (first_value - threshold) * (second_value - threshold) > 0:
            return
        weight = (float(threshold) - first_value) / (second_value - first_value)
        weight = min(max(weight, 0.0), 1.0)
        contour_points.append((first_xy[0] + weight * (second_xy[0] - first_xy[0]), first_xy[1] + weight * (second_xy[1] - first_xy[1])))

    x_values, y_values = geometry["x"], geometry["y"]
    for ix in range(field_grid.shape[0]):
        for iy in range(field_grid.shape[1] - 1):
            add_crossing(
                float(field_grid[ix, iy]),
                float(field_grid[ix, iy + 1]),
                (float(x_values[ix]), float(y_values[iy])),
                (float(x_values[ix]), float(y_values[iy + 1])),
            )
    for ix in range(field_grid.shape[0] - 1):
        for iy in range(field_grid.shape[1]):
            add_crossing(
                float(field_grid[ix, iy]),
                float(field_grid[ix + 1, iy]),
                (float(x_values[ix]), float(y_values[iy])),
                (float(x_values[ix + 1]), float(y_values[iy])),
            )
    if not contour_points:
        raise ValueError("No threshold contour crossing was found on the truth grid")
    contour = np.unique(np.asarray(contour_points, dtype=float), axis=0)
    query = np.asarray(coords, dtype=float)[:, :2]
    try:
        from scipy.spatial import cKDTree
        distances, _ = cKDTree(contour).query(query, k=1)
        distance_method = "linear_grid_edge_contour_cKDTree"
    except ImportError:  # pragma: no cover - scipy is present in the fig env
        # The fallback remains an actual Euclidean distance in coordinates;
        # it only changes the search implementation, and is disclosed.
        distances = np.empty(len(query), dtype=float)
        chunk = 8192
        for start in range(0, len(query), chunk):
            delta = query[start : start + chunk, None, :] - contour[None, :, :]
            distances[start : start + chunk] = np.sqrt(np.min(np.sum(delta * delta, axis=2), axis=1))
        distance_method = "linear_grid_edge_contour_bruteforce_fallback"
    signed = np.where(mask, distances, -distances)
    signed_grid = np.empty(geometry["shape"], dtype=float)
    signed_grid[grid_index[:, 0], grid_index[:, 1]] = signed
    geometry.update({"threshold_mask": mask, "threshold_mask_grid": grid_mask, "signed_distance_grid": signed_grid, "contour_points": contour, "distance_method": distance_method})
    return signed.astype(float, copy=False), geometry


def _truth_interface_index(coords: np.ndarray, truth_ch4: np.ndarray) -> tuple[int, float]:
    geometry = _grid_geometry(coords, truth_ch4)
    gx, gy = np.gradient(geometry["field_grid"], geometry["x"], geometry["y"], edge_order=1)
    gradient = np.hypot(gx, gy)
    flat_gradient = gradient[geometry["grid_index"][:, 0], geometry["grid_index"][:, 1]]
    index = int(np.argmax(np.nan_to_num(flat_gradient, nan=-np.inf)))
    return index, float(flat_gradient[index])


def select_physical_locations(
    coords: np.ndarray,
    truth_ch4: np.ndarray,
    signed_distance: np.ndarray,
    train_q10: float,
    train_q90: float,
    *,
    minimum_interface_distance: float,
) -> list[dict[str, Any]]:
    """Freeze C3 locations from truth/training quantities only.

    If a prescribed quantile has no eligible point, the location is explicitly
    returned as unavailable; no nearby or method-dependent substitute is made.
    """

    coords_array = np.asarray(coords, dtype=float)
    truth = np.asarray(truth_ch4, dtype=float)
    distance = np.asarray(signed_distance, dtype=float)
    if coords_array.shape[0] != truth.size or distance.shape != truth.shape:
        raise ValueError("C3 location inputs have inconsistent lengths")
    interface_index, gradient = _truth_interface_index(coords_array, truth)
    locations: list[dict[str, Any]] = [
        {
            "location_id": "interface_max_gradient",
            "status": "ok",
            "flat_index": interface_index,
            "x": float(coords_array[interface_index, 0]),
            "y": float(coords_array[interface_index, 1]),
            "truth_value": float(truth[interface_index]),
            "target_quantile": float("nan"),
            "target_quantile_value": float("nan"),
            "interface_gradient": gradient,
            "selection_rule": "truth_only_argmax_abs_gradient_Y_CH4; tie_break_lowest_flat_index",
            "interface_distance": float(abs(distance[interface_index])),
            "signed_interface_distance": float(distance[interface_index]),
        }
    ]
    for location_id, target, target_value, rule in (
        ("fuel_rich", 0.90, float(train_q90), "truth_Y_CH4_nearest_training_split_q90_with_distance_from_interface_at_least_0.05"),
        ("fuel_lean", 0.10, float(train_q10), "truth_Y_CH4_nearest_training_split_q10_with_distance_from_interface_at_least_0.05"),
    ):
        eligible = np.flatnonzero(np.abs(distance) >= float(minimum_interface_distance))
        if eligible.size == 0:
            locations.append(
                {
                    "location_id": location_id,
                    "status": "unavailable_no_eligible_point",
                    "flat_index": -1,
                    "x": float("nan"),
                    "y": float("nan"),
                    "truth_value": float("nan"),
                    "target_quantile": target,
                    "target_quantile_value": target_value,
                    "interface_gradient": float("nan"),
                    "selection_rule": rule,
                    "interface_distance": float("nan"),
                    "signed_interface_distance": float("nan"),
                    "quantile_degenerate_zero": bool(target_value == 0.0),
                }
            )
            continue
        order = np.lexsort((eligible, np.abs(truth[eligible] - target_value)))
        index = int(eligible[order[0]])
        locations.append(
            {
                "location_id": location_id,
                "status": "ok",
                "flat_index": index,
                "x": float(coords_array[index, 0]),
                "y": float(coords_array[index, 1]),
                "truth_value": float(truth[index]),
                "target_quantile": target,
                "target_quantile_value": target_value,
                "interface_gradient": float("nan"),
                "selection_rule": rule,
                "interface_distance": float(abs(distance[index])),
                "signed_interface_distance": float(distance[index]),
                "quantile_degenerate_zero": bool(target_value == 0.0),
            }
        )
    return locations


def _crossing_coordinate(x: np.ndarray, values: np.ndarray, threshold: float) -> float:
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    delta = values - float(threshold)
    for index in range(len(x) - 1):
        if delta[index] == 0:
            return float(x[index])
        if delta[index] * delta[index + 1] < 0:
            weight = abs(delta[index]) / (abs(delta[index]) + abs(delta[index + 1]))
            return float(x[index] + weight * (x[index + 1] - x[index]))
    return float("nan")


def reduce_interface_profile(
    coords: np.ndarray,
    truth_ch4: np.ndarray,
    draw_ch4: np.ndarray,
    threshold: float,
    bin_edges: Sequence[float],
    minimum_points: int,
    training_std: float,
    *,
    method: str,
    state: int,
    original_time_index: int,
) -> list[dict[str, Any]]:
    """Reduce one state/method ensemble to compact signed-distance rows."""

    truth = np.asarray(truth_ch4, dtype=float)
    draws = np.asarray(draw_ch4, dtype=float)
    if draws.ndim != 2 or draws.shape[1] != truth.size or draws.shape[0] < 2:
        raise ValueError("Interface reducer expects draws with shape (S, n_points)")
    signed_distance, _ = signed_distance_field(coords, truth, threshold)
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("Profile bin edges must be strictly increasing")
    bin_index = np.digitize(signed_distance, edges, right=False) - 1
    records: list[dict[str, Any]] = []
    truth_x: list[float] = []
    truth_y: list[float] = []
    draw_profile_rows: list[np.ndarray] = []
    bin_metadata: list[tuple[int, float, int]] = []
    for index in range(len(edges) - 1):
        selected = np.flatnonzero((bin_index == index) & np.isfinite(truth))
        if selected.size < int(minimum_points):
            continue
        point_truth = truth[selected]
        point_draws = draws[:, selected]
        truth_profile = float(np.mean(point_truth))
        draw_profiles = np.mean(point_draws, axis=1)
        truth_x.append(float(0.5 * (edges[index] + edges[index + 1])))
        truth_y.append(truth_profile)
        draw_profile_rows.append(draw_profiles)
        bin_metadata.append((index, truth_x[-1], int(selected.size)))
    if not draw_profile_rows:
        raise ValueError("No signed-distance profile bin meets the minimum point count")
    profile_draws = np.asarray(draw_profile_rows, dtype=float).T
    profile_truth = np.asarray(truth_y, dtype=float)
    quantiles = _empirical_quantiles(profile_draws, (0.05, 0.25, 0.50, 0.75, 0.95))
    scale = float(training_std)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Training CH4 standard deviation must be finite and positive")
    crps_by_bin = np.asarray([empirical_crps(profile_draws[:, i], np.asarray(profile_truth[i])) for i in range(profile_draws.shape[1])], dtype=float) / scale
    coverage_by_bin = ((profile_truth >= quantiles[0]) & (profile_truth <= quantiles[4])).astype(float)
    sharpness_by_bin = quantiles[4] - quantiles[0]
    truth_crossing = _crossing_coordinate(np.asarray(truth_x), profile_truth, threshold)
    mean_crossing = _crossing_coordinate(np.asarray(truth_x), np.mean(profile_draws, axis=0), threshold)
    front_bias = float(mean_crossing - truth_crossing) if np.isfinite(mean_crossing) and np.isfinite(truth_crossing) else float("nan")
    for i, (bin_number, center, point_count) in enumerate(bin_metadata):
        records.append(
            {
                "dataset": DATASET,
                "task": TASK,
                "condition": CONDITION,
                "method": method,
                "state": int(state),
                "original_time_index": int(original_time_index),
                "field": "Y_CH4",
                "threshold": float(threshold),
                "bin_index": int(bin_number),
                "signed_distance": float(center),
                "point_count": point_count,
                "truth_profile": float(profile_truth[i]),
                "mean_profile": float(np.mean(profile_draws[:, i])),
                "q05_profile": float(quantiles[0, i]),
                "q25_profile": float(quantiles[1, i]),
                "q50_profile": float(quantiles[2, i]),
                "q75_profile": float(quantiles[3, i]),
                "q95_profile": float(quantiles[4, i]),
                "normalized_profile_crps": float(crps_by_bin[i]),
                "profile_interval_coverage_90": float(coverage_by_bin[i]),
                "profile_sharpness_90": float(sharpness_by_bin[i]),
                "front_location_bias": front_bias,
                "truth_front_crossing": float(truth_crossing),
                "mean_front_crossing": float(mean_crossing),
                "training_std": scale,
                "aggregation": "equal_point_weight_within_bin",
                "draw_count": int(draws.shape[0]),
            }
        )
    return records


def _reduce_binned_profiles(
    bin_metadata: Sequence[tuple[int, float, int]],
    truth_profile: np.ndarray,
    draw_profiles: np.ndarray,
    threshold: float,
    training_std: float,
    *,
    method: str,
    state: int,
    original_time_index: int,
) -> list[dict[str, Any]]:
    """Reduce already-binned profiles without retaining any full fields."""

    values = np.asarray(draw_profiles, dtype=float)
    target = np.asarray(truth_profile, dtype=float)
    if values.ndim != 2 or values.shape[1] != target.size or values.shape[0] < 2 or len(bin_metadata) != target.size:
        raise ValueError("Binned profile arrays have inconsistent shapes")
    scale = float(training_std)
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Training CH4 standard deviation must be finite and positive")
    quantiles = _empirical_quantiles(values, (0.05, 0.25, 0.50, 0.75, 0.95))
    crps = np.maximum(np.asarray([empirical_crps(values[:, index], np.asarray(target[index])) for index in range(target.size)], dtype=float) / scale, 0.0)
    coverage = ((target >= quantiles[0]) & (target <= quantiles[4])).astype(float)
    sharpness = quantiles[4] - quantiles[0]
    centers = np.asarray([item[1] for item in bin_metadata], dtype=float)
    truth_crossing = _crossing_coordinate(centers, target, threshold)
    mean_crossing = _crossing_coordinate(centers, np.mean(values, axis=0), threshold)
    front_bias = float(mean_crossing - truth_crossing) if np.isfinite(mean_crossing) and np.isfinite(truth_crossing) else float("nan")
    rows: list[dict[str, Any]] = []
    for index, (bin_number, center, point_count) in enumerate(bin_metadata):
        rows.append(
            {
                "dataset": DATASET,
                "task": TASK,
                "condition": CONDITION,
                "method": method,
                "state": int(state),
                "original_time_index": int(original_time_index),
                "field": "Y_CH4",
                "threshold": float(threshold),
                "bin_index": int(bin_number),
                "signed_distance": float(center),
                "point_count": int(point_count),
                "truth_profile": float(target[index]),
                "mean_profile": float(np.mean(values[:, index])),
                "q05_profile": float(quantiles[0, index]),
                "q25_profile": float(quantiles[1, index]),
                "q50_profile": float(quantiles[2, index]),
                "q75_profile": float(quantiles[3, index]),
                "q95_profile": float(quantiles[4, index]),
                "normalized_profile_crps": float(crps[index]),
                "profile_interval_coverage_90": float(coverage[index]),
                "profile_sharpness_90": float(sharpness[index]),
                "front_location_bias": front_bias,
                "truth_front_crossing": float(truth_crossing),
                "mean_front_crossing": float(mean_crossing),
                "training_std": scale,
                "aggregation": "equal_point_weight_within_bin",
                "draw_count": int(values.shape[0]),
            }
        )
    return rows


def training_split_indices(validation_plan: Mapping[str, Any], total_time: int) -> np.ndarray:
    """Reproduce the frozen train split used by every adopted checkpoint."""

    split = validation_plan["dataset"]["split"]
    seed = int(split["seed"])
    ratio = float(split["train_ratio"])
    stride = int(split.get("time_stride", 1))
    if stride != 1:
        raise ValueError("Panel-C requires the frozen time_stride=1 split")
    indices = np.arange(int(total_time), dtype=np.int64)
    shuffled = np.random.default_rng(seed).permutation(indices)
    count = int(total_time * ratio)
    return np.sort(shuffled[:count])


def _training_scalar_values(dataset_path: Path, validation_plan: Mapping[str, Any]) -> tuple[float, float, float, float, float, int, float, float]:
    """Compute exact CH4 quantiles and streaming training moments.

    Quantiles are obtained from an on-disk temporary memmap and the memmap is
    removed in ``finally``.  The retained result is only the three scalar
    quantiles plus moments; no training fields are copied into the run.
    """

    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - inference environment provides h5py
        raise RuntimeError("h5py is required to audit the frozen training split") from exc
    with h5py.File(dataset_path, "r") as handle:
        fields = handle["fields"]
        if fields.ndim != 6 or int(fields.shape[0]) != 1 or int(fields.shape[-1]) < 5:
            raise ValueError(f"Unexpected HDF5 fields shape: {fields.shape}")
        total_time, point_count = int(fields.shape[1]), int(fields.shape[2])
        train_indices = training_split_indices(validation_plan, total_time)
        if len(train_indices) != int(total_time * float(validation_plan["dataset"]["split"]["train_ratio"])):
            raise ValueError("Frozen train split cardinality mismatch")
        temporary_path: str | None = None
        try:
            temporary = tempfile.NamedTemporaryFile(prefix="phycoflow_panel_c_train_ch4_", suffix=".bin", dir="/tmp", delete=False)
            temporary_path = temporary.name
            temporary.close()
            total_values = len(train_indices) * point_count
            with open(temporary_path, "r+b") as raw:
                raw.truncate(total_values * np.dtype("float32").itemsize)
            mmap = np.memmap(temporary_path, mode="r+", dtype="float32", shape=(total_values,))
            position = 0
            for index in train_indices:
                values = np.asarray(fields[0, int(index), :, 0, 0, 0], dtype=np.float32)
                mmap[position : position + point_count] = values
                position += point_count
            mmap.flush()
            q_indices: list[int] = []
            for probability in (0.10, 0.90, 0.99):
                rank = float(probability) * float(total_values - 1)
                q_indices.extend([int(math.floor(rank)), int(math.ceil(rank))])
            # ndarray.partition is in-place for the memmap and keeps memory
            # bounded by the operating system's page cache.
            mmap.partition(tuple(sorted(set(q_indices))))
            quantiles: list[float] = []
            for probability in (0.10, 0.90, 0.99):
                rank = probability * float(total_values - 1)
                lower, upper = int(math.floor(rank)), int(math.ceil(rank))
                weight = rank - lower
                quantiles.append(float((1.0 - weight) * mmap[lower] + weight * mmap[upper]))
            del mmap
        finally:
            if temporary_path is not None:
                try:
                    Path(temporary_path).unlink()
                except FileNotFoundError:
                    pass
        # Functional training statistics treat every frozen training time as
        # one cohort observation.  This avoids weighting long fields by their
        # number of grid points when standardizing C4b residuals.
        q10, q90, q99 = quantiles
        sums = np.zeros(2, dtype=np.float64)
        squares = np.zeros(2, dtype=np.float64)
        count = 0
        for index in train_indices:
            values = np.asarray(fields[0, int(index), :, 0, 0, 0], dtype=np.float64)
            functionals = np.asarray([np.mean(values), np.mean(values > 0.5 * q99)], dtype=np.float64)
            sums += functionals
            squares += functionals * functionals
            count += 1
        means = sums / count
        stds = np.sqrt(np.maximum(squares / count - means * means, 0.0))
    return q10, q90, q99, float(means[0]), float(stds[0]), int(count), float(means[1]), float(stds[1])


def _validate_pointwise_plan(config: Mapping[str, Any], validation_plan: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    if validation_plan.get("schema_version") != "figure5-validation-v1":
        raise ValueError("Pointwise inference requires validation_v1")
    dataset = _repo_path(validation_plan["dataset"]["path"], repo_root)
    if not dataset.is_file() or _sha256(dataset) != str(validation_plan["dataset"]["sha256"]):
        raise ValueError("Pointwise dataset identity does not match validation_v1")
    identity: dict[str, Any] = {"dataset": {"path": str(dataset), "sha256": _sha256(dataset), "pass": True}}
    for label, section in (("dataset_statistics", validation_plan["dataset_statistics"]), ("sensor_plan", validation_plan["sensor_plan"])):
        path = _repo_path(section["path"], repo_root)
        actual = _sha256(path)
        identity[label] = {"path": str(path), "expected_sha256": str(section["sha256"]), "actual_sha256": actual, "pass": actual == str(section["sha256"])}
        if not identity[label]["pass"]:
            raise ValueError(f"{label} identity mismatch")
    checkpoint_checks: dict[str, Any] = {}
    checkpoint_map = {str(row["method"]): row for row in validation_plan["checkpoints"]}
    for method in METHODS:
        row = checkpoint_map.get(method)
        if row is None:
            raise ValueError(f"Validation plan has no checkpoint for {method}")
        path = _repo_path(row["path"], repo_root)
        actual = _sha256(path)
        checkpoint_checks[method] = {"path": str(path), "expected_sha256": str(row["sha256"]), "actual_sha256": actual, "pass": actual == str(row["sha256"])}
        if not checkpoint_checks[method]["pass"]:
            raise ValueError(f"Checkpoint identity mismatch for {method}")
    identity["checkpoints"] = checkpoint_checks
    return identity


def _pointwise_method_runs(
    config: Mapping[str, Any],
    validation_plan: Mapping[str, Any],
    v3: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble per-method checkpoint/settings metadata for the manifest."""

    checkpoint_map = {str(row["method"]): row for row in validation_plan["checkpoints"]}
    method_runs: dict[str, Any] = {}
    for method in METHODS:
        expected = _pointwise_method_settings(config, method)
        audit = dict(v3["audit_settings"][method])
        checkpoint_row = checkpoint_map[method]
        method_runs[method] = {
            "checkpoint": {
                "path": str(identity["checkpoints"][method]["path"]),
                "checkpoint_name": str(checkpoint_row.get("checkpoint_name", "last.pt")),
                "condition": str(checkpoint_row.get("condition", CONDITION)),
                "expected_sha256": str(identity["checkpoints"][method]["expected_sha256"]),
                "actual_sha256": str(identity["checkpoints"][method]["actual_sha256"]),
                "identity_pass": bool(identity["checkpoints"][method]["pass"]),
            },
            "state": int(config["panel_c"]["fixed_state"]["test_index"]),
            "original_time_index": int(config["panel_c"]["fixed_state"]["original_hdf5_time_index"]),
            "draw_count": int(config["panel_c"]["pointwise_protocol"]["required_draws"]),
            "settings": {
                "n_steps": int(expected["n_steps"]),
                "solver_metadata": str(expected["solver"]),
                "runner_solver_argument": "euler",
                "measured_nfe": int(expected["measured_nfe"]),
                "obs_consistency_requested": str(expected["obs_consistency_requested"]),
                "obs_consistency_applied": str(expected["obs_consistency_applied"]),
                "execution_mode": str(expected["execution_mode"]),
                "cache_level": str(expected["cache_level"]),
            },
            "v3_method_draw_audit": audit,
            "reduction": "stream_one_state_method_at_a_time; reduce_immediately; no_full_draw_saved",
        }
    return method_runs


def _load_sensor_rows(validation_plan: Mapping[str, Any], states: Iterable[int], repo_root: Path) -> dict[int, list[dict[str, int]]]:
    import csv

    groups: dict[int, list[dict[str, int]]] = {int(state): [] for state in states}
    sensor_path = _repo_path(validation_plan["sensor_plan"]["path"], repo_root)
    with sensor_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            state = int(row["snapshot"])
            if row["condition"] == CONDITION and state in groups:
                groups[state].append({key: int(row[key]) for key in ("sensor_order", "field_index", "point_index", "sensor_seed")})
    for state, rows in groups.items():
        rows.sort(key=lambda row: row["sensor_order"])
        if len(rows) != 256 or {row["field_index"] for row in rows} != {2}:
            raise ValueError(f"Exact Cond_T sensor plan is incomplete for state {state}")
    return groups


def _physical_stats(validation_plan: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(validation_plan["dataset_statistics"]["mean"], dtype=np.float32)
    stds = np.asarray(validation_plan["dataset_statistics"]["std"], dtype=np.float32)
    if means.shape != (5,) or stds.shape != (5,) or not np.isfinite(means).all() or not np.isfinite(stds).all() or (stds <= 0).any():
        raise ValueError("Frozen physical normalization statistics are invalid")
    return means, stds


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _pointwise_method_settings(config: Mapping[str, Any], method: str) -> dict[str, Any]:
    """Build runner settings from the frozen requested value.

    ``obs_consistency_applied`` is metadata verified against the V3 audit; it
    is never passed back as the requested value.  Native baseline adapters may
    report ``native_not_applied`` while still receiving the canonical requested
    setting in the runner contract.
    """

    expected = _audit_expected_from_config(config, method)
    return {
        "n_steps": int(expected["n_steps"]),
        "solver": str(expected["solver"]),
        "measured_nfe": int(expected["measured_nfe"]),
        "obs_consistency": str(expected["obs_consistency_requested"]),
        "obs_consistency_requested": str(expected["obs_consistency_requested"]),
        "obs_consistency_applied": str(expected["obs_consistency_applied"]),
        "execution_mode": str(expected["execution_mode"]),
        "cache_level": str(expected["cache_level"]),
        "query_chunk_size": 8192,
        # benchmark_validation_v3.core_call owns the canonical execution path
        # and passes euler to PointCloudFFM.sample; ``solver`` above remains
        # the audited method-level metadata (heun_native for FFM-FNO).
        "runner_solver_argument": "euler",
    }


def _verify_postprocess_settings(config: Mapping[str, Any], postprocess: Mapping[str, Any], v3_audit: Mapping[str, Any]) -> dict[str, Any]:
    defaults = postprocess.get("defaults", {})
    overrides = postprocess.get("method_inference_overrides", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("Canonical postprocess config has no defaults mapping")
    checked: dict[str, Any] = {}
    for method in METHODS:
        override = overrides.get(method, {}) or {}
        expected = _pointwise_method_settings(config, method)
        canonical_steps = int(override.get("n_steps", defaults.get("n_steps", 2)))
        canonical_obs = str(override.get("obs_consistency", defaults.get("obs_consistency", "endpoint_smooth")))
        if canonical_steps != expected["n_steps"] or canonical_obs != expected["obs_consistency_requested"]:
            raise ValueError(f"Panel-C contract differs from canonical postprocess settings for {method}")
        audit_values = v3_audit[method]
        checked[method] = {
            "runner_requested": expected["obs_consistency_requested"],
            "audit_requested": audit_values["obs_consistency_requested"],
            "audit_applied": audit_values["obs_consistency_applied"],
            "n_steps": expected["n_steps"],
            "solver_metadata": expected["solver"],
            "measured_nfe": expected["measured_nfe"],
            "postprocess_config_match": True,
            "v3_audit_match": True,
        }
    return checked


def _read_truth_state(dataset_path: Path, original_time_index: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for pointwise C2--C4 preparation") from exc
    with h5py.File(dataset_path, "r") as handle:
        raw_coords = np.asarray(handle["coordinates"], dtype=np.float32)
        raw_coords = raw_coords.reshape(raw_coords.shape[0], -1)[:, :3]
        # Match common.helpers.normalize_coords exactly: the model and the
        # frozen pointwise distance threshold use min--max normalized x/y,
        # rather than the raw HDF5 coordinate units.
        coordinate_min = raw_coords.min(axis=0)
        coordinate_scale = np.maximum(raw_coords.max(axis=0) - coordinate_min, 1e-8)
        coords = (raw_coords - coordinate_min) / coordinate_scale
        truth = np.asarray(handle["fields"][0, int(original_time_index), :, 0, 0, :], dtype=np.float32)
    if coords.shape[0] != truth.shape[0] or truth.shape[1] < 5:
        raise ValueError("Truth state and coordinate query sizes do not match")
    return coords, truth[:, :5]


def _profile_bins(coords: np.ndarray, truth_ch4: np.ndarray, threshold: float, edges: Sequence[float], minimum_points: int) -> tuple[np.ndarray, list[tuple[int, float, int]], np.ndarray]:
    signed_distance, _ = signed_distance_field(coords, truth_ch4, threshold)
    edges_array = np.asarray(edges, dtype=float)
    memberships = np.digitize(signed_distance, edges_array, right=False) - 1
    metadata: list[tuple[int, float, int]] = []
    truth_profile: list[float] = []
    for index in range(len(edges_array) - 1):
        selected = np.flatnonzero(memberships == index)
        if len(selected) < int(minimum_points):
            continue
        metadata.append((index, float(0.5 * (edges_array[index] + edges_array[index + 1])), int(len(selected))))
        truth_profile.append(float(np.mean(truth_ch4[selected])))
    if not metadata:
        raise ValueError("No C2 profile bin meets the predeclared minimum point count")
    return memberships, metadata, np.asarray(truth_profile, dtype=float)


def _functional_values(ch4: np.ndarray, threshold: float) -> dict[str, float]:
    values = np.asarray(ch4, dtype=float)
    return {
        "methane_inventory": float(np.mean(values)),
        "methane_rich_area_fraction": float(np.mean(values > float(threshold))),
    }


def _append_functional_summary(
    rows: list[dict[str, Any]],
    *,
    method: str,
    state: int,
    original_time_index: int,
    draw_values: Mapping[str, Sequence[float]],
    truth_values: Mapping[str, float],
    training_stats: Mapping[str, tuple[float, float]],
    fixed: bool,
) -> None:
    for functional_id, values_raw in draw_values.items():
        values = np.asarray(values_raw, dtype=float)
        truth = float(truth_values[functional_id])
        quantiles = _empirical_quantiles(values, (0.05, 0.25, 0.50, 0.75, 0.95))
        mean = float(np.mean(values))
        training_mean, training_std = training_stats[functional_id]
        rows.append(
            {
                "dataset": DATASET,
                "task": TASK,
                "condition": CONDITION,
                "method": method,
                "state": int(state),
                "original_time_index": int(original_time_index),
                "summary_kind": "fixed_state_summary" if fixed else "state_summary",
                "functional_id": functional_id,
                "draw_id": -1,
                "value": mean,
                "truth": truth,
                "ensemble_mean": mean,
                "q05": float(quantiles[0]),
                "q25": float(quantiles[1]),
                "q50": float(quantiles[2]),
                "q75": float(quantiles[3]),
                "q95": float(quantiles[4]),
                "scalar_crps": float(empirical_crps(values, np.asarray(truth))),
                "mean_absolute_error": abs(mean - truth),
                "coverage_90": float(quantiles[0] <= truth <= quantiles[4]),
                "interval_width_90": float(quantiles[4] - quantiles[0]),
                "training_mean": float(training_mean),
                "training_std": float(training_std),
                "standardized_residual": float((mean - truth) / training_std) if training_std > 0 else float("nan"),
                "draw_count": int(len(values)),
                "weighting": "equal_point_weight_no_cell_area_source_available",
            }
        )


def _run_pointwise_inference(
    config: Mapping[str, Any],
    validation_plan: Mapping[str, Any],
    v3: Mapping[str, Any],
    run_dir: Path,
    repo_root: Path,
    *,
    device: str,
    include_cohort_functionals: bool,
) -> dict[str, Any]:
    """Run the gated compact pointwise reducer in the inference environment."""

    if not device.startswith("cuda:2"):
        raise RuntimeError("Panel-C pointwise inference is reserved for explicit cuda:2")
    # Lazy imports keep the audit/plot command usable in ``fig``.
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - selected only in inference env
        raise RuntimeError("Pointwise inference requires the phycoflow_env torch stack") from exc
    model_script_dir = repo_root / "0_demo_TurbulentCombustion" / "Save_TrainedModel" / "_TrainedModels" / "_Scripts"
    tools_dir = repo_root / "0_demo_TurbulentCombustion" / "tools"
    for path in (repo_root, model_script_dir, tools_dir):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from benchmark_validation_v3 import core_call, evaluation_context, prepare_state  # type: ignore
    from common.model_loader import load_model  # type: ignore

    if not torch.cuda.is_available() or torch.device(device).index != 2:
        raise RuntimeError("Requested pointwise device cuda:2 is unavailable")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    identity = _validate_pointwise_plan(config, validation_plan, repo_root)
    means, stds = _physical_stats(validation_plan)
    dataset_path = _repo_path(validation_plan["dataset"]["path"], repo_root)
    q10, q90, q99, train_m_mean, train_m_std, train_count, train_a_mean, train_a_std = _training_scalar_values(dataset_path, validation_plan)
    threshold = 0.5 * q99
    coords, truth_physical = _read_truth_state(dataset_path, int(config["fig4_state_original_time_index"]))
    truth_ch4 = truth_physical[:, 0]
    signed_distance, geometry = signed_distance_field(coords, truth_ch4, threshold)
    memberships, bin_metadata, truth_profile = _profile_bins(
        coords,
        truth_ch4,
        threshold,
        config["panel_c"]["ch4_interface"]["profile_bin_edges"],
        int(config["panel_c"]["ch4_interface"]["minimum_points_per_bin"]),
    )
    locations = select_physical_locations(
        coords,
        truth_ch4,
        signed_distance,
        q10,
        q90,
        minimum_interface_distance=float(config["panel_c"]["c3_locations"]["minimum_interface_distance"]),
    )
    active_locations = [item for item in locations if item.get("status") == "ok"]
    fixed_state = int(config["panel_c"]["fixed_state"]["test_index"])
    original_by_state = v3["table"].loc[v3["table"]["method"] == "DMF-Gen"].set_index("state")["original_time_index"].astype(int).to_dict()
    states = [fixed_state] + ([int(value) for value in v3["states"] if int(value) != fixed_state] if include_cohort_functionals else [])
    sensor_rows = _load_sensor_rows(validation_plan, states, repo_root)
    postprocess_path = model_script_dir / "postprocess_config.yaml"
    postprocess = yaml.safe_load(postprocess_path.read_text(encoding="utf-8"))
    settings_audit = _verify_postprocess_settings(config, postprocess, v3["audit_settings"])
    method_cfg = {str(row["name"]): row for row in postprocess["methods"]}
    profile_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    functional_rows: list[dict[str, Any]] = []
    training_functional_stats = {
        "methane_inventory": (train_m_mean, train_m_std),
        "methane_rich_area_fraction": (train_a_mean, train_a_std),
    }
    base_seed = int(validation_plan["generation_seeds"]["base"])
    for method in METHODS:
        settings = _pointwise_method_settings(config, method)
        loaded = load_model(method_cfg[method], CONDITION, checkpoint="last", split="test", device=device, n_steps=settings["n_steps"], ode_solver="euler")
        try:
            with evaluation_context(loaded), torch.no_grad():
                for state in states:
                    prepared = prepare_state(loaded, state, sensor_rows[state], 40300, method, retain_truth=True)
                    recon_truth = prepared["truth"][0].detach().cpu().numpy().astype(np.float32, copy=False) * stds[None, :] + means[None, :]
                    if state == fixed_state and not np.allclose(recon_truth[:, 0], truth_ch4, rtol=2e-5, atol=2e-5):
                        raise RuntimeError(f"{method} prepared truth does not match exact Figure 4 state original index {config['fig4_state_original_time_index']}")
                    draw_profile_values: list[np.ndarray] = []
                    draw_functionals: dict[str, list[float]] = {"methane_inventory": [], "methane_rich_area_fraction": []}
                    for draw_id in range(64):
                        seed = stable_seed(base_seed, "generation", "U2", state, draw_id)
                        torch.manual_seed(seed)
                        np.random.seed(seed % (2**32))
                        torch.cuda.manual_seed_all(seed)
                        reconstruction = core_call(loaded, prepared, settings, method)
                        physical = reconstruction[0].detach().cpu().numpy().astype(np.float32, copy=False) * stds[None, :] + means[None, :]
                        ch4 = np.asarray(physical[:, 0], dtype=np.float64)
                        # Reduce the full field immediately to profile bins,
                        # three fixed point values and two scalar functionals.
                        profile_values = []
                        for bin_number, _, _ in bin_metadata:
                            selected = np.flatnonzero(memberships == bin_number)
                            profile_values.append(float(np.mean(ch4[selected])))
                        draw_profile_values.append(np.asarray(profile_values, dtype=float))
                        if state == fixed_state:
                            for location in active_locations:
                                posterior_rows.append(
                                    {
                                        "dataset": DATASET,
                                        "task": TASK,
                                        "condition": CONDITION,
                                        "method": method,
                                        "state": state,
                                        "original_time_index": int(config["fig4_state_original_time_index"]),
                                        "location_id": location["location_id"],
                                        "flat_index": int(location["flat_index"]),
                                        "x": float(location["x"]),
                                        "y": float(location["y"]),
                                        "truth": float(location["truth_value"]),
                                        "draw_id": draw_id,
                                        "value": float(ch4[int(location["flat_index"])]),
                                        "target_quantile": float(location.get("target_quantile", np.nan)),
                                        "target_quantile_value": float(location.get("target_quantile_value", np.nan)),
                                        "selection_rule": location["selection_rule"],
                                        "quantile_degenerate_zero": bool(location.get("quantile_degenerate_zero", False)),
                                        "draw_count": 64,
                                    }
                                )
                        functionals = _functional_values(ch4, threshold)
                        for key, value in functionals.items():
                            draw_functionals[key].append(value)
                        del reconstruction
                        del physical
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    if state == fixed_state:
                        profile_rows.extend(
                            _reduce_binned_profiles(
                                bin_metadata,
                                truth_profile,
                                np.asarray(draw_profile_values, dtype=float),
                                threshold,
                                float(stds[0]),
                                method=method,
                                state=state,
                                original_time_index=int(config["fig4_state_original_time_index"]),
                            )
                        )
                        for key, values in draw_functionals.items():
                            truth_value = _functional_values(truth_ch4, threshold)[key]
                            for draw_id, value in enumerate(values):
                                functional_rows.append(
                                    {
                                        "dataset": DATASET,
                                        "task": TASK,
                                        "condition": CONDITION,
                                        "method": method,
                                        "state": state,
                                        "original_time_index": int(config["fig4_state_original_time_index"]),
                                        "summary_kind": "draw",
                                        "functional_id": key,
                                        "draw_id": draw_id,
                                        "value": float(value),
                                        "truth": float(truth_value),
                                        "ensemble_mean": float(np.mean(values)),
                                        "draw_count": 64,
                                        "weighting": "equal_point_weight_no_cell_area_source_available",
                                    }
                                )
                        _append_functional_summary(
                            functional_rows,
                            method=method,
                            state=state,
                            original_time_index=int(config["fig4_state_original_time_index"]),
                            draw_values=draw_functionals,
                            truth_values=_functional_values(truth_ch4, threshold),
                            training_stats=training_functional_stats,
                            fixed=True,
                        )
                    elif include_cohort_functionals:
                        truth_state = prepared["truth"][0].detach().cpu().numpy().astype(np.float32, copy=False) * stds[None, :] + means[None, :]
                        truth_values = _functional_values(truth_state[:, 0], threshold)
                        _append_functional_summary(
                            functional_rows,
                            method=method,
                            state=state,
                            original_time_index=int(original_by_state[state]),
                            draw_values=draw_functionals,
                            truth_values=truth_values,
                            training_stats=training_functional_stats,
                            fixed=False,
                        )
                    del prepared
        finally:
            loaded.close()
    _write_csv(run_dir / "interface_profile.csv", pd.DataFrame(profile_rows))
    _write_csv(run_dir / "pointwise_posterior.csv", pd.DataFrame(posterior_rows))
    _write_csv(run_dir / "derived_functionals.csv", pd.DataFrame(functional_rows))
    return {
        "status": "complete",
        "identity_checks": identity,
        "settings_audit": settings_audit,
        "threshold": threshold,
        "training_quantiles": {"q10": q10, "q90": q90, "q99": q99},
        "training_functional_stats": {"methane_inventory": training_functional_stats["methane_inventory"], "methane_rich_area_fraction": training_functional_stats["methane_rich_area_fraction"]},
        "training_count": train_count,
        "locations": locations,
        "profile_rows": len(profile_rows),
        "posterior_rows": len(posterior_rows),
        "functional_rows": len(functional_rows),
        "cohort_functionals": bool(include_cohort_functionals),
        "geometry": {
            "shape": list(geometry["shape"]),
            "spacing_metadata": list(geometry["spacing"]),
            "point_count": int(len(coords)),
            "distance_method": geometry["distance_method"],
            "contour_point_count": int(len(geometry["contour_points"])),
        },
    }


def _load_pointwise_tables(source_dir: Path) -> dict[str, pd.DataFrame]:
    required = {key: source_dir / filename for key, filename in {
        "profile": "interface_profile.csv",
        "posterior": "pointwise_posterior.csv",
        "functionals": "derived_functionals.csv",
    }.items()}
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Pointwise compact source is incomplete: {missing}")
    tables = {key: pd.read_csv(path) for key, path in required.items()}
    if tables["profile"].empty or tables["posterior"].empty or tables["functionals"].empty:
        raise ValueError("Pointwise compact source contains an empty table")
    for key, table in tables.items():
        if "method" not in table.columns or not set(table["method"].astype(str)).issubset(set(METHODS)):
            raise ValueError(f"Pointwise {key} table has unknown methods")
        numeric_columns = table.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if numeric_columns.size and not np.isfinite(numeric_columns).all():
            # NaN is valid only for an unavailable C3 target row, but that row
            # is not written to the posterior table.  Profile front bias can
            # be undefined when no crossing exists, so tolerate that one
            # annotation column and reject all other non-finite values.
            allowed = {"front_location_bias", "truth_front_crossing", "mean_front_crossing", "target_quantile", "target_quantile_value"}
            for column in table.columns:
                if column in allowed:
                    continue
                column_values = table[column]
                if key == "functionals" and "summary_kind" in table.columns and column in {"q05", "q25", "q50", "q75", "q95", "scalar_crps", "mean_absolute_error", "coverage_90", "interval_width_90", "training_mean", "training_std", "standardized_residual"}:
                    column_values = column_values.loc[table["summary_kind"].astype(str) != "draw"]
                if pd.api.types.is_numeric_dtype(table[column]) and not np.isfinite(column_values.to_numpy(float)).all():
                    raise ValueError(f"Pointwise {key} table has non-finite values in {column}")
    return tables


def _pointwise_info_from_tables(
    tables: Mapping[str, pd.DataFrame],
    source_dir: Path,
    run_dir: Path,
    *,
    config: Mapping[str, Any],
    repo_root: Path,
    v3: Mapping[str, Any],
    inference_command: str | None,
    render_command: str | None,
) -> dict[str, Any]:
    """Recover provenance when rendering a completed compact-source run.

    This is used after a renderer-only retry, so a failed render cannot erase
    the reducer's threshold/location/count metadata from the final manifest.
    """

    profile = tables["profile"]
    posterior = tables["posterior"]
    functionals = tables["functionals"]
    threshold = float(profile["threshold"].dropna().iloc[0])
    q10_values = posterior.loc[posterior["location_id"].astype(str) == "fuel_lean", "target_quantile_value"].dropna().unique()
    q90_values = posterior.loc[posterior["location_id"].astype(str) == "fuel_rich", "target_quantile_value"].dropna().unique()
    if len(q10_values) != 1 or len(q90_values) != 1:
        raise ValueError("Pointwise posterior does not contain one frozen q10 and q90 target")
    validation_plan = _load_validation_plan(config, repo_root)
    identity = _validate_pointwise_plan(config, validation_plan, repo_root)
    dataset_path = _repo_path(validation_plan["dataset"]["path"], repo_root)
    coords, truth_physical = _read_truth_state(dataset_path, int(config["panel_c"]["fixed_state"]["original_hdf5_time_index"]))
    signed_distance, geometry = signed_distance_field(coords, truth_physical[:, 0], threshold)
    locations = select_physical_locations(
        coords,
        truth_physical[:, 0],
        signed_distance,
        float(q10_values[0]),
        float(q90_values[0]),
        minimum_interface_distance=float(config["panel_c"]["c3_locations"]["minimum_interface_distance"]),
    )
    # Confirm that the compact posterior rows came from the same truth-only
    # locations; do not silently replace or reselect a point from predictions.
    posterior_locations = {str(value): group.iloc[0] for value, group in posterior.groupby("location_id", sort=False)}
    for location in locations:
        if location["status"] != "ok":
            continue
        row = posterior_locations.get(str(location["location_id"]))
        if row is None or int(row["flat_index"]) != int(location["flat_index"]):
            raise ValueError(f"Compact posterior location mismatch for {location['location_id']}")
        if not np.allclose([float(row["x"]), float(row["y"]), float(row["truth"])], [float(location["x"]), float(location["y"]), float(location["truth_value"])], rtol=0.0, atol=2e-7):
            raise ValueError(f"Compact posterior truth/location coordinates mismatch for {location['location_id']}")
    training_stats: dict[str, tuple[float, float]] = {}
    summaries = functionals.loc[functionals["summary_kind"].astype(str) == "fixed_state_summary"]
    for row in summaries.itertuples(index=False):
        training_stats[str(row.functional_id)] = (float(row.training_mean), float(row.training_std))
    source_is_this_run = source_dir.resolve() == run_dir.resolve()
    method_runs = _pointwise_method_runs(config, validation_plan, v3, identity)
    provenance = {
        "inference": {
            "status": "complete" if source_is_this_run else "external_compact_source",
            "requested": bool(source_is_this_run),
            "confirm_gpu2_free": bool(source_is_this_run),
            "device": "cuda:2" if source_is_this_run else "unknown_external_device",
            "command": inference_command or "not_recorded_for_external_source",
            "scope": "exact fixed test_index 0 / original HDF5 time index 5; five methods; 64 shared U2 draws; no cohort functionals",
            "purpose": "gated fixed-state pointwise inference; stream/reduce/delete full reconstruction immediately",
        },
        "render": {
            "status": "complete",
            "command": render_command or "not_recorded",
            "inference_executed_in_this_command": False,
            "purpose": "renderer-only compact-source retry after schema-validator repair; no scientific values regenerated",
        },
    }
    return {
        "status": "complete",
        "source_directory": str(source_dir),
        "requested": bool(source_is_this_run),
        "confirm_gpu2_free": bool(source_is_this_run),
        "device": "cuda:2" if source_is_this_run else "unknown_external_device",
        "inference_executed": bool(source_is_this_run),
        "rendered_from_compact_source": True,
        "execution_provenance": provenance,
        "identity_checks": identity,
        "settings_audit": v3["audit_settings"],
        "method_runs": method_runs,
        "full_draws_saved": False,
        "threshold": threshold,
        "training_quantiles": {
            "q10": float(q10_values[0]) if len(q10_values) else float("nan"),
            "q90": float(q90_values[0]) if len(q90_values) else float("nan"),
            "q99": float(2.0 * threshold),
        },
        "locations": locations,
        "profile_rows": int(len(profile)),
        "posterior_rows": int(len(posterior)),
        "functional_rows": int(len(functionals)),
        "training_functional_stats": training_stats,
        "training_count": int(len(training_split_indices(validation_plan, int(validation_plan["dataset"]["shape"][1])))),
        "cohort_functionals": bool((functionals["summary_kind"].astype(str) == "state_summary").any()),
        "geometry": {
            "shape": list(geometry["shape"]),
            "spacing_metadata": list(geometry["spacing"]),
            "distance_method": geometry["distance_method"],
            "contour_point_count": int(len(geometry["contour_points"])),
            "point_count": int(len(coords)),
        },
    }


def _profile_y_limits(source: pd.DataFrame) -> tuple[float, float]:
    values = source[["truth_profile", "q05_profile", "q95_profile"]].to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("No finite C2 profile values")
    span = max(float(np.ptp(values)), 1e-8)
    return float(np.min(values) - 0.06 * span), float(np.max(values) + 0.06 * span)


def _make_c2a(source: pd.DataFrame, config: Mapping[str, Any]) -> plt.Figure:
    apply_style(config["style"].get("font_family"))
    colors = method_colors(config)
    fig, axes = plt.subplots(len(METHODS), 1, figsize=(89 * MM, 132 * MM), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    ymin, ymax = _profile_y_limits(source)
    for axis, method in zip(axes, METHODS):
        rows = source.loc[source["method"] == method].sort_values("signed_distance")
        x = rows["signed_distance"].to_numpy(float)
        axis.fill_between(x, rows["q05_profile"], rows["q95_profile"], color=colors[method], alpha=0.11, linewidth=0)
        axis.fill_between(x, rows["q25_profile"], rows["q75_profile"], color=colors[method], alpha=0.24, linewidth=0)
        axis.plot(x, rows["truth_profile"], color="#111111", linewidth=0.9, label="truth" if method == METHODS[0] else None)
        axis.plot(x, rows["mean_profile"], color=colors[method], linewidth=1.05, label="ensemble mean" if method == METHODS[0] else None)
        crps = float(rows["normalized_profile_crps"].mean())
        bias = float(rows["front_location_bias"].iloc[0])
        bias_text = f"bias {bias:+.3f}" if np.isfinite(bias) else "bias n/a"
        axis.text(0.99, 0.82, f"{method} · CRPS {crps:.3f}\n{bias_text}", transform=axis.transAxes, ha="right", va="top", fontsize=5.6, color=colors[method])
        axis.axvline(0.0, color="#888888", linewidth=0.45, linestyle=(0, (2, 2)))
        axis.set_ylim(ymin, ymax)
        style_grid(axis, axis="y")
    axes[-1].set_xlabel("Signed distance from truth $Y_{CH_4}$ interface")
    axes[len(axes) // 2].set_ylabel("$Y_{CH_4}$")
    axes[0].legend(loc="upper left", ncol=2, handlelength=1.5, fontsize=5.8)
    fig.suptitle("Conditional reconstruction of an unobserved methane interface", x=0.11, y=0.985, ha="left", fontsize=8.5, fontweight="semibold")
    _add_figure_panel_label(fig, "C2a")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.07, top=0.965, hspace=0.15)
    return fig


def _make_c2b(source: pd.DataFrame, config: Mapping[str, Any]) -> plt.Figure:
    apply_style(config["style"].get("font_family"))
    colors = method_colors(config)
    comparator = str(config["panel_c"]["c2b_overlay_comparator"]["expected_method_from_frozen_source"])
    fig, axis = plt.subplots(figsize=(89 * MM, 67 * MM))
    for method in METHODS:
        rows = source.loc[source["method"] == method].sort_values("signed_distance")
        x = rows["signed_distance"].to_numpy(float)
        if method in {"DMF-Gen", comparator}:
            axis.fill_between(x, rows["q05_profile"], rows["q95_profile"], color=colors[method], alpha=0.10, linewidth=0)
            axis.fill_between(x, rows["q25_profile"], rows["q75_profile"], color=colors[method], alpha=0.22, linewidth=0)
        axis.plot(x, rows["mean_profile"], color=colors[method], linewidth=1.1, label=method)
    truth_rows = source.loc[source["method"] == METHODS[0]].sort_values("signed_distance")
    axis.plot(truth_rows["signed_distance"], truth_rows["truth_profile"], color="#111111", linewidth=1.0, label="truth")
    axis.axvline(0.0, color="#888888", linewidth=0.45, linestyle=(0, (2, 2)))
    ymin, ymax = _profile_y_limits(source)
    axis.set_ylim(ymin, ymax)
    axis.set_xlim(float(source["signed_distance"].min()), float(source["signed_distance"].max()))
    axis.set_xlabel("Signed distance from truth $Y_{CH_4}$ interface")
    axis.set_ylabel("$Y_{CH_4}$")
    axis.set_title("Conditional reconstruction of an unobserved methane interface", loc="left", pad=5, fontweight="semibold")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, handlelength=1.5, columnspacing=0.8)
    style_grid(axis, axis="both")
    _add_figure_panel_label(fig, "C2b")
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.30, top=0.84)
    return fig


def _column_limits(values: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    pooled = np.concatenate((np.asarray(values, dtype=float).ravel(), np.asarray(truth, dtype=float).ravel()))
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        raise ValueError("No finite pointwise values")
    low, high = np.quantile(pooled, [0.005, 0.995])
    span = max(float(high - low), 1e-8)
    return float(low - 0.05 * span), float(high + 0.05 * span)


def _draw_quantile_strip(axis: plt.Axes, values: np.ndarray, truth: float, y: float, color: str, *, label: str | None = None) -> tuple[float, float, float, float, float]:
    q05, q25, q50, q75, q95 = _empirical_quantiles(np.asarray(values, dtype=float), (0.05, 0.25, 0.50, 0.75, 0.95))
    axis.plot([q05, q95], [y, y], color=color, linewidth=1.0, solid_capstyle="round", label=label, zorder=2)
    axis.plot([q25, q75], [y, y], color=color, linewidth=3.2, solid_capstyle="round", zorder=3)
    axis.plot(q50, y, marker="o", markersize=3.0, color=color, zorder=4)
    axis.plot(truth, y, marker="|", markersize=8, markeredgewidth=1.0, color="#111111", zorder=5)
    return float(q05), float(q25), float(q50), float(q75), float(q95)


def _make_c3(source: pd.DataFrame, config: Mapping[str, Any], *, training_std: float) -> plt.Figure:
    apply_style(config["style"].get("font_family"))
    colors = method_colors(config)
    if not np.isfinite(training_std) or training_std <= 0:
        raise ValueError("C3 needs a finite positive frozen Y_CH4 training standard deviation")
    locations = [str(value) for value in config["panel_c"]["c3_locations"]["locations"] for value in ([value["id"]] if isinstance(value, Mapping) else [])]
    observed_locations = [value for value in locations if value in set(source["location_id"].astype(str))]
    if not observed_locations:
        observed_locations = list(dict.fromkeys(source["location_id"].astype(str)))
    fig, axes = plt.subplots(len(METHODS), len(observed_locations), figsize=(89 * MM, 102 * MM), sharex="col", sharey=True, squeeze=False)
    for column, location_id in enumerate(observed_locations):
        column_source = source.loc[source["location_id"].astype(str) == location_id]
        limits = _column_limits(column_source["value"].to_numpy(float), column_source["truth"].to_numpy(float))
        for row_index, method in enumerate(METHODS):
            axis = axes[row_index, column]
            values = column_source.loc[column_source["method"] == method, "value"].to_numpy(float)
            truth_values = column_source.loc[column_source["method"] == method, "truth"].to_numpy(float)
            if values.size == 0:
                axis.text(0.5, 0.5, "n/a", transform=axis.transAxes, ha="center", va="center", fontsize=6)
                continue
            truth = float(truth_values[0])
            _draw_quantile_strip(axis, values, truth, 0.5, colors[method])
            normalized_crps = empirical_crps(values, np.asarray(truth)) / float(training_std)
            axis.text(0.98, 0.12, f"nCRPS {normalized_crps:.3g}", transform=axis.transAxes, ha="right", va="bottom", fontsize=5.2, color=colors[method])
            axis.set_xlim(*limits)
            axis.set_ylim(0.2, 0.8)
            axis.set_yticks([])
            axis.set_xticks([] if row_index < len(METHODS) - 1 else axis.get_xticks())
            if column == 0:
                axis.set_ylabel(method, rotation=0, labelpad=20, va="center", fontsize=5.7, color=colors[method])
        axes[-1, column].set_xlabel(location_id.replace("_", "\n"), fontsize=5.7)
    fig.suptitle("Conditional distributions at physically distinct unobserved locations", x=0.11, y=0.985, ha="left", fontsize=8.4, fontweight="semibold")
    fig.text(0.012, 0.5, "$Y_{CH_4}$ value", rotation=90, va="center", fontsize=7.0)
    _add_figure_panel_label(fig, "C3")
    fig.subplots_adjust(left=0.20, right=0.99, bottom=0.08, top=0.93, wspace=0.20, hspace=0.24)
    return fig


def _make_c4a(source: pd.DataFrame, config: Mapping[str, Any]) -> plt.Figure:
    apply_style(config["style"].get("font_family"))
    colors = method_colors(config)
    functionals = [str(item["id"]) for item in config["panel_c"]["c4_functionals"]["functionals"]]
    source = source.loc[source["summary_kind"].astype(str) == "draw"]
    fig, axes = plt.subplots(len(functionals), 1, figsize=(89 * MM, 62 * MM), squeeze=False)
    axes = axes[:, 0]
    for axis, functional_id in zip(axes, functionals):
        subset = source.loc[source["functional_id"].astype(str) == functional_id]
        all_values = subset["value"].to_numpy(float)
        truth = float(subset["truth"].iloc[0])
        limits = _column_limits(all_values, np.asarray([truth]))
        for index, method in enumerate(METHODS):
            values = subset.loc[subset["method"] == method, "value"].to_numpy(float)
            if values.size:
                _draw_quantile_strip(axis, values, truth, float(len(METHODS) - 1 - index), colors[method])
            axis.text(0.0, float(len(METHODS) - 1 - index), method, transform=axis.get_yaxis_transform(), ha="right", va="center", fontsize=5.7, color=colors[method])
        axis.axvline(truth, color="#111111", linewidth=0.85, linestyle=(0, (2, 2)))
        axis.set_xlim(*limits)
        axis.set_yticks([])
        axis.set_ylim(-0.8, len(METHODS) - 0.2)
        axis.set_title(functional_id.replace("_", " "), loc="left", fontsize=7.0, pad=2)
        style_grid(axis, axis="x")
    axes[-1].set_xlabel("Predictive functional value; black tick = truth")
    fig.suptitle("Uncertainty propagated to physical field functionals", x=0.11, y=0.985, ha="left", fontsize=8.5, fontweight="semibold")
    _add_figure_panel_label(fig, "C4a")
    fig.subplots_adjust(left=0.20, right=0.98, bottom=0.14, top=0.91, hspace=0.40)
    return fig


def _make_c4b(source: pd.DataFrame, config: Mapping[str, Any]) -> plt.Figure:
    apply_style(config["style"].get("font_family"))
    colors = method_colors(config)
    functionals = [str(item["id"]) for item in config["panel_c"]["c4_functionals"]["functionals"]]
    source = source.loc[source["summary_kind"].astype(str) == "state_summary"]
    if source.empty:
        raise ValueError("C4b needs state_summary rows over the 200-state cohort")
    fig, axes = plt.subplots(1, len(functionals), figsize=(110 * MM, 55 * MM), squeeze=False)
    axes = axes[0]
    for axis, functional_id in zip(axes, functionals):
        subset = source.loc[source["functional_id"].astype(str) == functional_id]
        data = [subset.loc[subset["method"] == method, "standardized_residual"].to_numpy(float) for method in METHODS]
        box = axis.boxplot(data, positions=np.arange(len(METHODS)), widths=0.56, patch_artist=True, showfliers=False, medianprops={"color": "#111111", "linewidth": 0.7}, whiskerprops={"linewidth": 0.65}, capprops={"linewidth": 0.65})
        for patch, method in zip(box["boxes"], METHODS):
            patch.set_facecolor(colors[method])
            patch.set_alpha(0.58)
            patch.set_edgecolor(colors[method])
        axis.axhline(0.0, color="#777777", linewidth=0.55, linestyle=(0, (2, 2)))
        axis.set_title(functional_id.replace("_", " "), loc="left", fontsize=7.0, pad=2)
        axis.set_xticks(np.arange(len(METHODS)), [method.replace("-", "‑") for method in METHODS], rotation=50, ha="right")
        axis.set_ylabel("Standardized residual" if axis is axes[0] else "")
        style_grid(axis, axis="y")
    fig.suptitle("Uncertainty propagated to physical field functionals", x=0.11, y=0.985, ha="left", fontsize=8.5, fontweight="semibold")
    _add_figure_panel_label(fig, "C4b")
    fig.subplots_adjust(left=0.12, right=0.99, bottom=0.30, top=0.87, wspace=0.28)
    return fig


def _audit_svg(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    text_nodes = [node for node in root.iter() if node.tag.endswith("text")]
    image_nodes = [node for node in root.iter() if node.tag.endswith("image")]
    text = " ".join("".join(node.itertext()) for node in text_nodes)
    return {
        "parseable": True,
        "editable_text": bool(text_nodes),
        "raster_image_count": len(image_nodes),
        "fixed_width": bool(root.attrib.get("width")),
        "fixed_height": bool(root.attrib.get("height")),
        "contains_failure_prediction_language": "failure prediction" in text.lower(),
    }


def _profile_training_std(profile: pd.DataFrame) -> float:
    """Return and validate the frozen physical Y_CH4 profile scale."""

    if "training_std" not in profile.columns or profile.empty:
        raise ValueError("Pointwise profile is missing the frozen Y_CH4 training_std")
    values = pd.to_numeric(profile["training_std"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any() or not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
        raise ValueError("Pointwise profile has inconsistent or invalid Y_CH4 training_std values")
    return float(values[0])


def _c3_metrics_table(posterior: pd.DataFrame, training_std: float) -> pd.DataFrame:
    """Reduce retained C3 draws to auditable, normalized cell metrics."""

    scale = float(training_std)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("C3 metric scale must be finite and positive")
    records: list[dict[str, Any]] = []
    for method in METHODS:
        for location_id in dict.fromkeys(posterior["location_id"].astype(str)):
            rows = posterior.loc[(posterior["method"].astype(str) == method) & (posterior["location_id"].astype(str) == location_id)].sort_values("draw_id")
            if len(rows) != 64:
                raise ValueError(f"C3 requires exactly 64 draws for {method}/{location_id}; found {len(rows)}")
            values = rows["value"].to_numpy(dtype=float)
            truth_values = rows["truth"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or not np.isfinite(truth_values).all() or not np.allclose(truth_values, truth_values[0], rtol=0.0, atol=0.0):
                raise ValueError(f"C3 values/truth are invalid for {method}/{location_id}")
            truth = float(truth_values[0])
            less = int(np.count_nonzero(values < truth))
            equal = int(np.count_nonzero(values == truth))
            rank = float(less + ((equal + 1) / 2.0 if equal else 1.0))
            # Use the empirical CDF convention so an out-of-ensemble truth is
            # reported at exactly 0% or 100%, rather than producing a
            # misleading value above 100% when its insertion rank is n+1.
            percentile = float(100.0 * (less + 0.5 * equal) / len(values))
            records.append(
                {
                    "dataset": str(rows["dataset"].iloc[0]),
                    "task": str(rows["task"].iloc[0]),
                    "condition": str(rows["condition"].iloc[0]),
                    "method": method,
                    "state": int(rows["state"].iloc[0]),
                    "original_time_index": int(rows["original_time_index"].iloc[0]),
                    "location_id": location_id,
                    "truth": truth,
                    "draw_count": int(len(values)),
                    "training_std": scale,
                    "normalized_crps": float(empirical_crps(values, np.asarray(truth)) / scale),
                    "absolute_ensemble_mean_error": float(abs(np.mean(values) - truth)),
                    "truth_percentile": percentile,
                    "truth_rank": rank,
                }
            )
    return pd.DataFrame(records)


def _pointwise_companion(stem: str, tables: Mapping[str, pd.DataFrame], config: Mapping[str, Any], source_dir: Path, candidate: str) -> str:
    profile = tables["profile"]
    posterior = tables["posterior"]
    functionals = tables["functionals"]
    threshold = float(profile["threshold"].iloc[0]) if "threshold" in profile.columns else float("nan")
    training_std = _profile_training_std(profile)
    locations = list(dict.fromkeys(posterior["location_id"].astype(str)))
    lines = [
        f"# Figure 5 V5.1 panel-C candidate: {stem}",
        "",
        f"- Candidate: {candidate}.",
        f"- SVG: {stem}.svg",
        "- Task: turbulent-combustion Cond_T, 256 observed temperature sensors, native 40,300-point query.",
        "",
        "## Scientific role",
        "",
        "This candidate probes conditional uncertainty in an always-unobserved methane field using a fixed Figure 4 state. It complements, rather than replaces, the paired 200-state V3 accuracy/UQ statistics.",
        "",
        "## Frozen selections and reductions",
        "",
        f"- Exact state: test index {config['panel_c']['fixed_state']['test_index']} -> original HDF5 time index {config['panel_c']['fixed_state']['original_hdf5_time_index']}; no nearest-state substitution.",
        f"- Methane threshold: 0.5 x training-split Y_CH4 q99 = {threshold:.8g} physical units, audited before method outputs.",
        f"- C3 locations present: {', '.join(locations)}.",
        "- The prescribed training q90/q10 targets are retained even if a target is degenerate (including q90 = 0); the run manifest records that condition and never substitutes a nearby quantile.",
        "- Signed distance uses linearly interpolated truth threshold-edge crossings and nearest Euclidean distance in the actual normalized coordinates; positive sign is the truth Y_CH4 >= threshold side. The run manifest records the cKDTree backend (or the disclosed Euclidean brute-force fallback).",
        "- Each model ensemble is reduced state/method-at-a-time to profile bins, point values and scalar functionals; no full draw field is retained or written.",
        "",
        "## Compact-source metrics",
        "",
        f"- Frozen pointwise Y_CH4 scale: training-split physical standard deviation = `{training_std:.12g}`; pointwise nCRPS is empirical CRPS divided by this scale.",
    ]
    if not profile.empty:
        for method in METHODS:
            rows = profile.loc[profile["method"] == method]
            if rows.empty:
                continue
            bias = float(rows["front_location_bias"].iloc[0])
            bias_text = f"{bias:+.5g}" if np.isfinite(bias) else "n/a"
            lines.append(f"- **{method}:** profile CRPS={float(rows['normalized_profile_crps'].mean()):.5g}, 90% coverage={float(rows['profile_interval_coverage_90'].mean()):.3f}, front-location bias={bias_text}.")
    if not functionals.empty:
        summary = functionals.loc[functionals["summary_kind"].astype(str).isin(["fixed_state_summary", "state_summary"])]
        if not summary.empty:
            lines.extend(["", "Functional summary rows:"])
            for row in summary.itertuples(index=False):
                lines.append(f"- **{row.method} / {row.functional_id} / state {int(row.state)}:** scalar CRPS={float(row.scalar_crps):.5g}, 90% coverage={float(row.coverage_90):.3f}, interval width={float(row.interval_width_90):.5g}.")
    if candidate == "C3" and not posterior.empty:
        metrics = _c3_metrics_table(posterior, training_std)
        lines.extend(
            [
                "",
                "### C3 cell metrics",
                "",
                "nCRPS is the pointwise empirical CRPS divided by the frozen physical Y_CH4 training standard deviation. Truth percentile is the empirical CDF within the 64 retained draws (ties use half weight, and out-of-ensemble values remain at 0% or 100%). Truth rank is the 1-based insertion rank; rank 65 means the truth is above all 64 draws.",
                "",
                "| Method | Location | nCRPS | Absolute ensemble-mean error | Truth percentile | Truth rank / 64 |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in metrics.itertuples(index=False):
            lines.append(
                f"| {row.method} | {row.location_id} | {float(row.normalized_crps):.6g} | {float(row.absolute_ensemble_mean_error):.6g} | {float(row.truth_percentile):.3f}% | {float(row.truth_rank):.3g} |")
    lines.extend(
        [
            "",
            "## Sources and limits",
            "",
            f"- Compact pointwise source directory: {source_dir}",
            f"- C3 metrics are deterministically recomputed from: {source_dir / 'pointwise_posterior.csv'}",
            f"- Existing C0 (preserved for SI): {config['outputs']['c0_svg']} and {config['outputs']['c0_source']}",
            "- C3 density/ridgeline marks are empirical quantile strips; no synthetic draws are generated.",
            "- These illustrative profiles and pointwise distributions do not justify calling the functionals conserved quantities.",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_c0_reference(config: Mapping[str, Any], docs_dir: Path, c0: Mapping[str, Any]) -> Path:
    path = docs_dir / "C0_reference.md"
    path.write_text(
        "\n".join(
            [
                "# Figure 5 V5.1 panel-C C0 reference",
                "",
                "The accepted V5 spatial error-capture curve is preserved as an SI/reference candidate and is not rerun or copied into this exploration bundle.",
                "",
                f"- SVG: {c0['svg']}",
                f"- Compact source: {c0['source']}",
                f"- Source SHA256: {c0['source_sha256']}",
                "- Existing formal interpretation: SiT leads by C(0.20) and EC-AUC; this unfavorable-to-DMF evidence remains visible for audit.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _save_candidate(fig: plt.Figure, figure_dir: Path, docs_dir: Path, stem: str, companion: str) -> Path:
    path = figure_dir / f"{stem}.svg"
    save_svg(fig, path)
    docs_dir.mkdir(parents=True, exist_ok=True)
    (docs_dir / f"{path.stem}.md").write_text(companion, encoding="utf-8")
    return path


def _run_smoke(config: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise reducers on tiny synthetic arrays without reading model data."""

    fractions = np.asarray(config["panel_c"]["c1"]["coverage_fractions"], dtype=float)
    curve, full = _risk_curve(np.asarray([0.1, 0.2, 0.3, 0.4]), np.asarray([1.0, 2.0, 3.0, 4.0]), np.arange(4), fractions)
    if curve.shape != fractions.shape or not np.isfinite(curve).all() or full != 2.5:
        raise AssertionError("C1 smoke reducer failed")
    x = np.asarray([0.0, 0.1, 0.4, 0.9])
    y = np.asarray([0.0, 0.2, 0.8])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    coords = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    truth = coords[:, 0] + 0.2 * coords[:, 1]
    signed, _ = signed_distance_field(coords, truth, 0.35)
    if not np.isfinite(signed).all() or not (signed > 0).any() or not (signed < 0).any():
        raise AssertionError("Signed-distance smoke reducer failed")
    locations = select_physical_locations(coords, truth, signed, 0.1, 0.9, minimum_interface_distance=0.05)
    if locations[0]["status"] != "ok" or empirical_crps(np.asarray([[0.0, 1.0], [1.0, 2.0]]), np.asarray([0.5, 1.5])) < 0:
        raise AssertionError("Pointwise reducer smoke failed")
    if _pointwise_method_settings(config, "DMF-Gen")["obs_consistency"] != "default_hard" or _pointwise_method_settings(config, "SiT")["n_steps"] != 4:
        raise AssertionError("Method-settings smoke failed")
    return {"status": "pass", "checks": ["selective_risk", "nonuniform_contour_distance", "locations", "crps", "method_settings"]}


def _source_hashes(paths: Mapping[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, raw_path in paths.items():
        path = Path(raw_path)
        if path.is_file():
            hashes[str(key)] = _sha256(path)
    return hashes


def _write_final_manifest(path: Path, manifest: dict[str, Any], *, pointwise_info: Mapping[str, Any], source_paths: Mapping[str, Any], figures: Sequence[Path]) -> None:
    manifest = dict(manifest)
    manifest["status"] = "complete"
    manifest["formal"] = False
    manifest["completed_head"] = _git_value(REPO_ROOT, "rev-parse", "HEAD")
    manifest["pointwise_execution"] = dict(manifest.get("pointwise_execution", {}))
    manifest["pointwise_execution"].update(dict(pointwise_info))
    manifest["source_paths"] = {key: str(value) for key, value in source_paths.items()}
    manifest["source_sha256"] = _source_hashes(source_paths)
    manifest["figures"] = [str(path) for path in figures]
    manifest["storage_policy"] = {
        "full_ensemble_fields_saved": False,
        "per_draw_field_files_saved": False,
        "retained_compact_tables": ["selective_risk.csv"] + (["interface_profile.csv", "pointwise_posterior.csv", "derived_functionals.csv"] if pointwise_info.get("status") == "complete" else []),
        "temporary_preview_policy": "PNG previews, if any, must stay under /tmp and be removed after QA",
    }
    _write_json(path, manifest)


def _pointwise_table_qa(
    run_dir: Path,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    validation_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate compact pointwise rows, scales, locations and C3 metrics."""

    tables = _load_pointwise_tables(run_dir)
    profile, posterior, functionals = tables["profile"], tables["posterior"], tables["functionals"]
    expected_std = float(_physical_stats(validation_plan)[1][0])
    required_draws = int(config["panel_c"]["pointwise_protocol"]["required_draws"])
    fixed_state = int(config["panel_c"]["fixed_state"]["test_index"])
    original_time_index = int(config["panel_c"]["fixed_state"]["original_hdf5_time_index"])
    profile_bins = len(config["panel_c"]["ch4_interface"]["profile_bin_edges"]) - 1
    locations = set(posterior["location_id"].astype(str))
    profile_numeric = profile[["threshold", "signed_distance", "point_count", "truth_profile", "mean_profile", "q05_profile", "q25_profile", "q50_profile", "q75_profile", "q95_profile", "normalized_profile_crps", "profile_interval_coverage_90", "profile_sharpness_90", "training_std", "draw_count"]].to_numpy(dtype=float)
    posterior_numeric = posterior[["state", "original_time_index", "flat_index", "x", "y", "truth", "draw_id", "value", "draw_count"]].to_numpy(dtype=float)
    function_draw = functionals.loc[functionals["summary_kind"].astype(str) == "draw"]
    function_summary = functionals.loc[functionals["summary_kind"].astype(str) == "fixed_state_summary"]
    function_draw_numeric = function_draw[["state", "original_time_index", "draw_id", "value", "truth", "ensemble_mean", "draw_count"]].to_numpy(dtype=float)
    function_summary_numeric = function_summary[["state", "original_time_index", "draw_id", "value", "truth", "ensemble_mean", "q05", "q25", "q50", "q75", "q95", "scalar_crps", "mean_absolute_error", "coverage_90", "interval_width_90", "training_mean", "training_std", "standardized_residual", "draw_count"]].to_numpy(dtype=float)
    checks: dict[str, Any] = {
        "tables_present_and_nonempty": True,
        "profile_exact_shape": len(profile) == len(METHODS) * profile_bins,
        "profile_method_bin_coverage": all(len(profile.loc[profile["method"].astype(str) == method]["bin_index"].unique()) == profile_bins for method in METHODS),
        "profile_numeric_values_finite": bool(np.isfinite(profile_numeric).all()),
        "posterior_exact_fixed_state_rows": len(posterior) == len(METHODS) * len(locations) * required_draws,
        "posterior_three_truth_only_locations": locations == {"interface_max_gradient", "fuel_rich", "fuel_lean"},
        "posterior_group_draw_count": all(len(group) == required_draws for _, group in posterior.groupby(["method", "location_id"], sort=False)),
        "posterior_numeric_values_finite": bool(np.isfinite(posterior_numeric).all()),
        "posterior_exact_state_mapping": bool((posterior["state"] == fixed_state).all() and (posterior["original_time_index"] == original_time_index).all()),
        "functional_fixed_state_draw_rows": len(function_draw) == len(METHODS) * 2 * required_draws,
        "functional_fixed_state_summary_rows": len(function_summary) == len(METHODS) * 2,
        "functional_group_draw_count": all(len(group) == required_draws for _, group in function_draw.groupby(["method", "functional_id"], sort=False)),
        "functional_numeric_values_finite": bool(np.isfinite(function_draw_numeric).all() and np.isfinite(function_summary_numeric).all()),
        "functional_state_mapping": bool((functionals["state"] == fixed_state).all() and (functionals["original_time_index"] == original_time_index).all()),
        "interface_profile_scale_matches_validation_plan": bool(np.isfinite(profile["training_std"].to_numpy(dtype=float)).all() and np.allclose(profile["training_std"].to_numpy(dtype=float), expected_std, rtol=0.0, atol=1e-12)),
        "interface_profile_crps_finite_nonnegative": bool(np.isfinite(profile["normalized_profile_crps"].to_numpy(dtype=float)).all() and (profile["normalized_profile_crps"].to_numpy(dtype=float) >= -1e-10).all()),
        "metric_scale_declares_physical_Y_CH4_std0": "std[0]" in str(manifest.get("pointwise_metric_scales", {}).get("interface_profile_crps", "")),
    }
    metrics = _c3_metrics_table(posterior, expected_std)
    checks["c3_metrics_exact_shape"] = len(metrics) == len(METHODS) * len(locations)
    checks["c3_metrics_normalized_crps_recomputed"] = bool(
        set(metrics.columns) >= {"method", "location_id", "normalized_crps", "absolute_ensemble_mean_error", "truth_percentile", "truth_rank", "training_std", "draw_count"}
        and np.isfinite(metrics["normalized_crps"].to_numpy(dtype=float)).all()
        and (metrics["normalized_crps"].to_numpy(dtype=float) >= 0.0).all()
        and np.allclose(metrics["training_std"].to_numpy(dtype=float), expected_std, rtol=0.0, atol=1e-12)
        and (metrics["draw_count"].to_numpy(dtype=int) == required_draws).all()
    )
    checks["c3_metrics_absolute_error_recomputed"] = bool(np.isfinite(metrics["absolute_ensemble_mean_error"].to_numpy(dtype=float)).all() and (metrics["absolute_ensemble_mean_error"].to_numpy(dtype=float) >= 0.0).all())
    checks["c3_metrics_truth_rank_percentile_recomputed"] = bool(
        np.isfinite(metrics[["truth_percentile", "truth_rank"]].to_numpy(dtype=float)).all()
        and metrics["truth_percentile"].between(0.0, 100.0).all()
        and metrics["truth_rank"].between(1.0, required_draws + 1.0).all()
    )
    pointwise = manifest.get("pointwise_execution", {})
    provenance = pointwise.get("execution_provenance", {})
    inference_provenance = provenance.get("inference", {}) if isinstance(provenance, Mapping) else {}
    render_provenance = provenance.get("render", {}) if isinstance(provenance, Mapping) else {}
    method_runs = pointwise.get("method_runs", {})
    checks.update(
        {
            "provenance_gated_cuda2_confirmation": bool(pointwise.get("requested") is True and pointwise.get("confirm_gpu2_free") is True and pointwise.get("device") == "cuda:2" and inference_provenance.get("status") == "complete" and inference_provenance.get("requested") is True and inference_provenance.get("confirm_gpu2_free") is True and "--execute-pointwise" in str(inference_provenance.get("command", "")) and "--confirm-gpu2-free" in str(inference_provenance.get("command", ""))),
            "provenance_method_runs_complete": bool(set(method_runs) == set(METHODS) and all(int(method_runs[method].get("draw_count", -1)) == required_draws for method in METHODS)),
            "provenance_render_retry_distinguished": bool(pointwise.get("rendered_from_compact_source") is True and render_provenance.get("inference_executed_in_this_command") is False and "--pointwise-source" in str(render_provenance.get("command", ""))),
        }
    )
    location_rows = {str(item.get("location_id")): item for item in pointwise.get("locations", []) if isinstance(item, Mapping)}
    distance_floor = float(config["panel_c"]["c3_locations"]["minimum_interface_distance"])
    checks["c3_rich_lean_distances_finite"] = all(np.isfinite(float(location_rows.get(key, {}).get("interface_distance", np.nan))) for key in ("fuel_rich", "fuel_lean"))
    checks["c3_rich_lean_distances_meet_rule"] = all(float(location_rows.get(key, {}).get("interface_distance", np.nan)) >= distance_floor for key in ("fuel_rich", "fuel_lean"))
    return checks


def _write_qa(
    run_dir: Path,
    *,
    v3: Mapping[str, Any],
    selective: pd.DataFrame,
    figures: Sequence[Path],
    pointwise_status: str,
    config: Mapping[str, Any],
    validation_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    audits = {path.name: _audit_svg(path) for path in figures}
    manifest = _read_json(run_dir / "manifest.json")
    c0_reference = manifest.get("c0_reference", {})
    c0_svg = Path(str(c0_reference.get("svg", "")))
    c0_source = Path(str(c0_reference.get("source", "")))
    checks: dict[str, Any] = {
        "v3_formal_source_qa_pass": v3["qa"].get("status") == "pass",
        "v3_state_rows": len(v3["table"]) == 1000,
        "c1_row_count": len(selective) == 90,
        "c1_finite": bool(np.isfinite(selective[["risk", "ci_low", "ci_high"]].to_numpy(float)).all()),
        "c1_normalized_full_coverage_one": bool(np.allclose(selective.loc[(selective["risk_kind"] == "normalized") & np.isclose(selective["coverage_fraction"], 1.0), "risk"], 1.0)),
        "c0_preserved_source_exists": bool(c0_svg.is_file() and c0_source.is_file()),
        "pointwise_status_allowed": pointwise_status in {"pending_gpu2_inference", "complete"},
        "no_full_draw_files": not any(path.suffix.lower() in {".npy", ".npz", ".pt", ".pth"} for path in run_dir.rglob("*") if path.is_file()),
        "all_svg_parseable": all(item["parseable"] for item in audits.values()),
        "all_svg_editable_text": all(item["editable_text"] for item in audits.values()),
        "no_svg_raster_images": all(item["raster_image_count"] == 0 for item in audits.values()),
        "all_svg_fixed_dimensions": all(item["fixed_width"] and item["fixed_height"] for item in audits.values()),
        "no_failure_prediction_label": all(not item["contains_failure_prediction_language"] for item in audits.values()),
    }
    if pointwise_status == "complete" and validation_plan is not None:
        checks.update({f"pointwise_{key}": value for key, value in _pointwise_table_qa(run_dir, manifest, config, validation_plan).items()})
    payload = {
        "schema_version": QA_SCHEMA_VERSION,
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "svg_audits": audits,
        "pointwise_status": pointwise_status,
    }
    _write_json(run_dir / "qa.json", payload)
    if payload["status"] != "pass":
        raise RuntimeError(f"Panel-C QA failed: {checks}")
    return payload


def _load_validation_plan(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    path = _repo_path(config["validation_plan"], repo_root)
    plan = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or plan.get("schema_version") != "figure5-validation-v1":
        raise ValueError(f"Unexpected validation plan: {path}")
    return plan


def _render_pointwise_candidates(
    tables: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    figure_dir: Path,
    docs_dir: Path,
    source_dir: Path,
    run_dir: Path,
) -> list[Path]:
    paths: list[Path] = []
    profile = tables["profile"]
    posterior = tables["posterior"]
    functionals = tables["functionals"]
    training_std = _profile_training_std(profile)
    for stem, figure, candidate in (
        ("fig5_v51_c2a_methane_interface_small_multiples_20260902_1129", _make_c2a(profile, config), "C2a"),
        ("fig5_v51_c2b_methane_interface_overlay_20260902_1129", _make_c2b(profile, config), "C2b"),
        ("fig5_v51_c3_pointwise_posterior_atlas_20260902_1129", _make_c3(posterior, config, training_std=training_std), "C3"),
        ("fig5_v51_c4a_derived_functional_predictive_strips_20260902_1129", _make_c4a(functionals, config), "C4a"),
    ):
        paths.append(_save_candidate(figure, figure_dir, docs_dir, stem, _pointwise_companion(stem, tables, config, source_dir, candidate)))
    if not functionals.loc[functionals["summary_kind"].astype(str) == "state_summary"].empty:
        stem = "fig5_v51_c4b_derived_functional_residuals_20260902_1129"
        paths.append(_save_candidate(_make_c4b(functionals, config), figure_dir, docs_dir, stem, _pointwise_companion(stem, tables, config, source_dir, "C4b")))
    return paths


def _run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config_path = config_path.resolve()
    config = load_config(config_path)
    if args.smoke:
        result = _run_smoke(config)
        print(json.dumps(result, indent=2))
        return result
    run_id = str(args.run_id or config["outputs"]["result_run_id"])
    result_root = _repo_path(config["outputs"]["result_root"], repo_root)
    run_dir = result_root / run_id
    _safe_completed_run(
        run_dir,
        overwrite_failed=bool(args.overwrite_failed),
        allow_pointwise_extension=bool(args.execute_pointwise or args.pointwise_source is not None),
        allow_pointwise_render=bool(args.pointwise_source is not None),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite_failed:
        for filename in (
            "manifest.json",
            "qa.json",
            "selective_risk.csv",
            "interface_profile.csv",
            "pointwise_posterior.csv",
            "derived_functionals.csv",
        ):
            path = run_dir / filename
            if path.is_file():
                path.unlink()
    # Freeze and persist the protocol before reading the V3 table or any
    # optional model output. This is the audit boundary for C2--C4.
    manifest = _new_manifest(
        config,
        config_path,
        repo_root,
        run_id,
        pointwise_requested=bool(args.execute_pointwise),
        confirm_gpu2_free=bool(args.confirm_gpu2_free),
    )
    _write_json(run_dir / "manifest.json", manifest)
    v3 = _load_v3_sources(config, repo_root)
    fig4 = _validate_fig4_state(config, repo_root)
    c0 = _validate_c0(config, repo_root)
    selective = _selective_risk(v3["table"], config, bootstrap=True)
    _write_csv(run_dir / "selective_risk.csv", selective)
    source_paths: dict[str, Any] = {
        "config": config_path,
        "v3_manifest": v3["source_paths"]["v3_manifest"],
        "v3_qa": v3["source_paths"]["v3_qa"],
        "v3_state_table": v3["source_paths"]["v3_state_table"],
        "v3_method_draw_audit": v3["source_paths"]["v3_method_draw_audit"],
        "fig4_state_source": fig4["path"],
        "c0_svg": c0["svg"],
        "c0_source": c0["source"],
        "selective_risk": run_dir / "selective_risk.csv",
    }
    manifest["source_paths"] = {key: str(value) for key, value in source_paths.items()}
    manifest["v3_contract_audit"] = v3["audit_settings"]
    manifest["v3_source_rows"] = {"state_table": int(len(v3["table"])), "audit": int(len(v3["audit"]))}
    manifest["fig4_state_verification"] = {key: value for key, value in fig4.items() if key != "path"}
    manifest["c0_reference"] = {key: str(value) for key, value in c0.items()}
    figure_dir = _repo_path(config["outputs"]["figure_root"], repo_root)
    docs_dir = _repo_path(config["outputs"]["docs_root"], repo_root)
    figure_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    _write_c0_reference(config, docs_dir, c0)
    figures: list[Path] = []
    for stem, normalized in (
        ("fig5_v51_c1a_selective_risk_absolute_20260902_1129", False),
        ("fig5_v51_c1b_selective_risk_normalized_20260902_1129", True),
    ):
        candidate = "C1b" if normalized else "C1a"
        companion = _c1_companion(
            stem,
            selective,
            normalized=normalized,
            config=config,
            source_paths=source_paths,
        )
        figures.append(
            _save_candidate(
                _make_c1(selective, config, normalized=normalized),
                figure_dir,
                docs_dir,
                stem,
                companion,
            )
        )
    pointwise_info: dict[str, Any]
    pointwise_source_dir: Path | None
    pointwise_tables: dict[str, pd.DataFrame] | None = None
    validation_plan: Mapping[str, Any] | None = None
    if args.execute_pointwise:
        if not args.confirm_gpu2_free:
            raise RuntimeError(
                "Pointwise inference is gated: pass --confirm-gpu2-free only after GPU2 is explicitly released"
            )
        validation_plan = _load_validation_plan(config, repo_root)
        pointwise_info = _run_pointwise_inference(
            config,
            validation_plan,
            v3,
            run_dir,
            repo_root,
            device=str(args.device),
            include_cohort_functionals=bool(args.include_cohort_functionals),
        )
        pointwise_info["requested"] = True
        pointwise_info["confirm_gpu2_free"] = True
        pointwise_info["device"] = str(args.device)
        pointwise_info["execution_provenance"] = {
            "inference": {
                "status": "complete",
                "requested": True,
                "confirm_gpu2_free": True,
                "device": str(args.device),
                "command": str(args.inference_command or "command_not_recorded"),
                "scope": "exact fixed test_index 0 / original HDF5 time index 5; five methods; 64 shared U2 draws; no cohort functionals" if not args.include_cohort_functionals else "fixed state plus explicitly requested cohort functionals",
                "purpose": "gated fixed-state pointwise inference; stream/reduce/delete full reconstruction immediately",
            },
            "render": {
                "status": "same_command",
                "command": "inference_and_render_in_one_invocation",
                "inference_executed_in_this_command": True,
                "purpose": "render compact tables after inference",
            },
        }
        pointwise_info["method_runs"] = _pointwise_method_runs(config, validation_plan, v3, pointwise_info["identity_checks"])
        pointwise_source_dir = run_dir
    elif args.pointwise_source is not None:
        pointwise_source_dir = args.pointwise_source if args.pointwise_source.is_absolute() else repo_root / args.pointwise_source
        pointwise_source_dir = pointwise_source_dir.resolve()
        pointwise_tables = _load_pointwise_tables(pointwise_source_dir)
        validation_plan = _load_validation_plan(config, repo_root)
        pointwise_info = _pointwise_info_from_tables(
            pointwise_tables,
            pointwise_source_dir,
            run_dir,
            config=config,
            repo_root=repo_root,
            v3=v3,
            inference_command=args.source_inference_command,
            render_command=args.source_render_command,
        )
    else:
        pointwise_source_dir = None
        pointwise_info = {
            "status": "pending_gpu2_inference",
            "inference_executed": False,
            "reason": "C2-C4 require compact pointwise output; no GPU inference was run",
            "full_draws_saved": False,
        }
    if pointwise_source_dir is not None:
        tables = pointwise_tables or _load_pointwise_tables(pointwise_source_dir)
        figures.extend(
            _render_pointwise_candidates(
                tables,
                config,
                figure_dir,
                docs_dir,
                pointwise_source_dir,
                run_dir,
            )
        )
        source_paths.update(
            {
                "interface_profile": pointwise_source_dir / "interface_profile.csv",
                "pointwise_posterior": pointwise_source_dir / "pointwise_posterior.csv",
                "derived_functionals": pointwise_source_dir / "derived_functionals.csv",
            }
        )
    manifest["pointwise_execution"].update(pointwise_info)
    _write_final_manifest(
        run_dir / "manifest.json",
        manifest,
        pointwise_info=pointwise_info,
        source_paths=source_paths,
        figures=figures,
    )
    qa = _write_qa(
        run_dir,
        v3=v3,
        selective=selective,
        figures=figures,
        pointwise_status=str(pointwise_info["status"]),
        config=config,
        validation_plan=validation_plan,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "run_dir": str(run_dir),
                "figures": [str(path) for path in figures],
                "qa": qa["status"],
                "pointwise": pointwise_info["status"],
            },
            indent=2,
        )
    )
    return {
        "status": "complete",
        "run_dir": str(run_dir),
        "figures": [str(path) for path in figures],
        "pointwise": pointwise_info["status"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_v51_exploration.yaml")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke", action="store_true", help="run reducer smoke tests without reading model/data outputs")
    parser.add_argument("--execute-pointwise", action="store_true", help="execute compact C2-C4 inference; requires the explicit GPU2 gate")
    parser.add_argument("--confirm-gpu2-free", action="store_true", help="explicitly confirm that GPU2 is released for pointwise inference")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--include-cohort-functionals", action="store_true", help="also reduce C4b scalar functionals over the frozen 200-state cohort")
    parser.add_argument("--pointwise-source", type=Path, help="render C2-C4 from an existing compact result directory")
    parser.add_argument("--inference-command", help="exact shell command to record for a gated inference invocation")
    parser.add_argument("--source-inference-command", help="exact successful inference command that produced a renderer-only compact source")
    parser.add_argument("--source-render-command", help="exact renderer-only command used for a compact-source retry")
    parser.add_argument("--overwrite-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.confirm_gpu2_free and not args.execute_pointwise:
        raise ValueError("--confirm-gpu2-free has no effect without --execute-pointwise")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    _run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

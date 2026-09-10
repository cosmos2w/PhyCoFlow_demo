"""Shared constants and small, source-first reducers for Figure 5 V7.

The V7 source collector intentionally treats the saved evaluation products as
the authority.  This module contains only adapters, provenance decoration and
statistical helpers; it does not load a model, launch inference or copy cache
arrays.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


DATASET = "turbulent_combustion"
TASK = "missing_channel_reconstruction"
CONDITION = "Cond_T"
FIELDS = ("CH4", "CO", "T", "U1", "p")
UNOBSERVED_FIELDS = ("CH4", "CO", "U1", "p")
METHODS = ("A0", "A1", "A2", "A3", "A4", "A5")
STOCHASTIC_ORDER = ("A0", "A2", "A3", "A5", "A4")
POLICIES = ("last", "best")

DISPLAY_INFO: dict[str, dict[str, Any]] = {
    "A0": {
        "label": "Full model",
        "definition": "Full conditional DMF-Gen with global and local observation routes and spatial RFF prior.",
        "main_column": True,
    },
    "A1": {
        "label": "Deterministic regression",
        "definition": "Same backbone family with deterministic regression objective; retained as a separate SI control.",
        "main_column": False,
    },
    "A2": {
        "label": "No sensor feedback",
        "definition": "Global sensor feedback route removed; local query conditioning and stochastic prior retained.",
        "main_column": True,
    },
    "A3": {
        "label": "No local conditioning",
        "definition": "Local query conditioning route removed; global sensor feedback and stochastic prior retained.",
        "main_column": True,
    },
    "A4": {
        "label": "IID Gaussian prior",
        "definition": "Spatial RFF prior replaced by an IID Gaussian prior; other saved training configuration retained.",
        "main_column": True,
    },
    "A5": {
        "label": "Local-only conditioning",
        "definition": "Only local sensor-token Top-K/RBF observation route retained; global routes bypassed.",
        "main_column": True,
    },
}

PRIMARY_METRICS = {
    "physical_relative_l2",
    "normalized_relative_l2",
    "truth_fluctuation_normalized_l2",
    "physical_relative_l2_excluding_sensors",
    "normalized_relative_l2_excluding_sensors",
    "pointwise_correlation",
}
COUPLING_METRICS = {
    "coupling_correlation_abs_error",
    "coupling_reconstruction_correlation",
    "coupling_truth_correlation",
    "joint_pdf_jsd_base2",
    "joint_pdf_jsd_with_overflow_base2",
    "joint_pdf_reconstruction_retained_fraction",
    "joint_pdf_truth_retained_fraction",
}
ADMISSIBILITY_METRICS = {
    "reconstruction_fraction_below_minus_1e4",
    "reconstruction_minimum",
    "reconstruction_negative_l2_over_truth_l2",
    "reconstruction_negative_mean_magnitude",
    "reconstruction_nonphysical_fraction",
    "sensor_max_abs_normalized_error",
    "truth_minimum",
    "truth_nonphysical_fraction",
}
PROVENANCE_COLUMNS = (
    "evidence_package",
    "dataset",
    "task",
    "condition",
    "cohort_id",
    "cohort_identity",
    "policy",
    "checkpoint_policy",
    "checkpoint_name",
    "checkpoint_epoch",
    "checkpoint_sha256",
    "checkpoint_path",
    "source_path",
    "source_file",
    "source_sha256",
    "sensor_plan_path",
    "sensor_plan_sha256",
    "state_count",
    "measurement_count",
    "output_point_count",
    "split",
    "truth_sensor_time_identity",
    "metric_definition",
)


def sha256(path: Path) -> str:
    """Hash a file in bounded memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def relative_path(repo_root: Path, path: Path | str) -> str:
    """Return stable repo-relative paths while preserving external sources."""

    path = Path(path)
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def finite_columns(table: pd.DataFrame, columns: Iterable[str]) -> bool:
    columns = [column for column in columns if column in table.columns]
    return bool(columns) and bool(np.isfinite(table[columns].to_numpy(dtype=float)).all())


def circular_moving_block_bootstrap(
    values: Iterable[float] | np.ndarray,
    *,
    block_length: int = 20,
    resamples: int = 2000,
    seed: int = 20260910,
) -> tuple[float, float]:
    """Return a descriptive circular moving-block percentile interval.

    The production evaluation has already stored block-20 intervals.  This
    helper is kept for independent tests and for future source products.  The
    temporal unit is one state, never a spatial pixel.
    """

    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap input must be a non-empty finite vector")
    block_length = int(block_length)
    resamples = int(resamples)
    if block_length < 1 or resamples < 1:
        raise ValueError("block_length and resamples must be positive")
    n = array.size
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(resamples, int(np.ceil(n / block_length))))
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[..., None] + offsets) % n
    draws = array[indices].reshape(resamples, -1)[:, :n]
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def metric_definition(metric: str, target: str = "", *, high_frequency: bool = False) -> str:
    """Human-readable definition retained on every derived row."""

    metric = str(metric)
    if high_frequency:
        definitions = {
            "canonical_shellmean_high_energy_ratio": "Canonical retained-shell high-band reconstruction energy divided by truth high-band energy; shell-mean/trapezoidal estimator.",
            "canonical_spectral_lsd_db": "Canonical retained-shell log-spectral distance in dB over the complete retained spectrum.",
            "highband_error_relative_l2": "Square root of high-band residual mode energy divided by truth high-band mode energy.",
            "highband_error_energy_over_truth_high_energy": "High-band residual mode energy divided by truth high-band mode energy.",
            "highband_error_energy_over_truth_total_fluctuation_energy": "High-band residual mode energy divided by the full centered truth fluctuation energy.",
            "highband_error_energy_over_truth_retained_fluctuation_energy": "High-band residual mode energy divided by retained centered truth fluctuation energy.",
            "reconstruction_to_truth_high_energy_ratio": "Reconstruction high-band mode energy divided by truth high-band mode energy.",
            "truth_high_energy_fraction_total": "Truth high-band mode energy divided by full centered truth fluctuation energy.",
            "truth_high_energy_fraction_retained": "Truth high-band mode energy divided by retained centered truth fluctuation energy.",
        }
        if metric.startswith("hann_"):
            return definitions.get(metric.removeprefix("hann_"), "Hann-tapered companion diagnostic; mean-square correction applied.") + " Hann companion; no smoothing or clipping."
        return definitions.get(metric, "High-frequency population summary from the canonical retained-shell spectral evaluator.")
    definitions = {
        "physical_relative_l2": "Physical relative L2 error on the stated field or macro target.",
        "normalized_relative_l2": "Dataset-statistics normalized relative L2 error on the stated field or macro target.",
        "truth_fluctuation_normalized_l2": "Relative L2 error normalized by truth fluctuation magnitude.",
        "physical_relative_l2_excluding_sensors": "Physical relative L2 error evaluated after excluding observed sensor points.",
        "normalized_relative_l2_excluding_sensors": "Dataset-statistics normalized relative L2 error after excluding observed sensor points.",
        "pointwise_correlation": "Pearson pointwise correlation on the stated field or macro target.",
        "spectral_high_energy_ratio": "Canonical retained-shell high-frequency reconstruction energy ratio.",
        "spectral_lsd_db": "Canonical retained-shell log-spectral distance in dB.",
        "spectral_low_energy_ratio": "Canonical retained-shell low-frequency reconstruction energy ratio.",
        "spectral_mid_energy_ratio": "Canonical retained-shell mid-frequency reconstruction energy ratio.",
        "spectral_total_energy_ratio": "Canonical retained-shell total reconstruction energy ratio.",
    }
    if metric in definitions:
        return definitions[metric]
    if metric in COUPLING_METRICS:
        return "Coupling audit metric on the paired field target; paper-edge retention and overflow policy are retained in source metadata."
    if metric in ADMISSIBILITY_METRICS:
        return "Unclipped physical admissibility diagnostic evaluated on the full predicted field and sensor-excluded checks where stated."
    return f"Source evaluator metric {metric} for target {target}."


def source_file_record(repo_root: Path, path: Path, *, role: str, hash_file: bool = True, rows: int | None = None) -> dict[str, Any]:
    """Describe a source without copying it."""

    record: dict[str, Any] = {
        "role": role,
        "path": relative_path(repo_root, path),
        "absolute_path": str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() and hash_file else None,
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def first_nonempty(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and str(value) not in {"", "nan", "None"}:
            return value
    return default


def decorate_row(
    row: Mapping[str, Any],
    *,
    meta: Mapping[str, Any],
    source_path: Path,
    source_hash: str,
    repo_root: Path,
    high_frequency: bool = False,
    policy_override: str | None = None,
    checkpoint_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve all source columns and append a common provenance contract."""

    result = dict(row)
    method = str(result.get("method", meta.get("method", "")))
    metric = str(first_nonempty(result, "metric", "metric_name", default=""))
    target = str(first_nonempty(result, "target", "field", default=""))
    checkpoint = dict(checkpoint_override or {})
    policy = policy_override or str(meta.get("policy", result.get("policy", "")))
    checkpoint_name = first_nonempty(checkpoint, "checkpoint_name", "name", default=meta.get("checkpoint_name", ""))
    checkpoint_epoch = first_nonempty(checkpoint, "checkpoint_epoch", "epoch", default=meta.get("checkpoint_epoch"))
    checkpoint_hash = first_nonempty(checkpoint, "checkpoint_sha256", "sha256", default=meta.get("checkpoint_sha256", ""))
    checkpoint_path = first_nonempty(checkpoint, "checkpoint_path", "path", default=meta.get("checkpoint_path", ""))
    additions = {
        "evidence_package": "Package A",
        "dataset": DATASET,
        "task": TASK,
        "condition": CONDITION,
        "cohort_id": meta.get("cohort_id", "condT_test_1000_snapshots"),
        "cohort_identity": meta.get("cohort_identity", "Cond_T/test; 1,000 held-out snapshots; shared truth, sensor plan, time indices and generation seeds"),
        "policy": policy,
        "checkpoint_policy": policy,
        "checkpoint_name": checkpoint_name,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_path": checkpoint_path,
        "source_path": relative_path(repo_root, source_path),
        "source_file": relative_path(repo_root, source_path),
        "source_sha256": source_hash,
        "sensor_plan_path": meta.get("sensor_plan_path", ""),
        "sensor_plan_sha256": meta.get("sensor_plan_sha256", ""),
        "state_count": meta.get("state_count", 1000),
        "measurement_count": meta.get("measurement_count", 256),
        "output_point_count": meta.get("output_point_count", 40300),
        "split": meta.get("split", "test"),
        "truth_sensor_time_identity": meta.get("truth_sensor_time_identity", True),
        "metric_definition": metric_definition(metric, target, high_frequency=high_frequency),
    }
    # Keep source columns authoritative when a source already contains one of
    # the contract keys, except for explicit policy/checkpoint overrides.
    result.update(additions)
    return result


def ensure_provenance_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Order the contract columns after source columns for readable CSVs."""

    ordered = [column for column in table.columns if column not in PROVENANCE_COLUMNS]
    ordered += [column for column in PROVENANCE_COLUMNS if column in table.columns]
    return table.loc[:, ordered]


def as_jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(item) for item in value]
    return value

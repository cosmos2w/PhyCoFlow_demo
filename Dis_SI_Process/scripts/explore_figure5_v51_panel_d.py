#!/usr/bin/env python
"""Explore accuracy-first Figure 5 V5.1 panel-d layouts.

This script is intentionally additive.  It consumes the accepted V5 lifecycle
bundle and, when it exists, the separately produced common-batch-32 training
footprint bundle.  The two evidence modes are never mixed:

``existing_formal``
    frozen relative-L2, clean native inference latency, replay-equivalent
    model-core training GPU-hours, and required stage count.

``common_b32``
    frozen relative-L2, clean native inference latency, and the formal common
    batch-32 update/memory footprint.  The benchmark is a hard dependency;
    missing or incomplete data produce a strict wait and never fall back to
    the existing mode.

Only standalone SVG candidates are written.  The script does not assemble or
modify Figure 5 V2--V5 products.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.figure5_v41_style import (  # noqa: E402
    MM,
    apply_style,
    method_colors,
    method_markers,
    style_grid,
)


DATASET = "turbulent_combustion"
TASK = "missing_channel_reconstruction"
CONDITION = "Cond_T"
TIMING_BOUNDARY = "warm_model_core_geometry_persisted"
COMMON_B32_DEFAULT = Path("Dis_SI_Process/results/ValidationV51/TrainingFootprint/training_footprint_common_b32_v51")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


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


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Panel-D exploration config must be a mapping")
    if config.get("schema_version") == "figure5-validation-v5":
        return config
    # The additive V5.1 exploration contract is primarily a panel-c contract,
    # so it intentionally does not repeat the V5 lifecycle input paths.  When
    # callers pass that shared contract, borrow only the immutable V5 data
    # wiring and retain the V5.1 style declaration.  No result is copied or
    # modified by this translation.
    if config.get("schema_version") == "figure5-v51-panel-c-exploration-1":
        base_path = PACKAGE_ROOT / "configs" / "figure5_v5.yaml"
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if not isinstance(base, dict):
            raise ValueError("Accepted Figure 5 V5 config is not a mapping")
        style = dict(base.get("style", {}))
        for key in ("font_family", "method_colors", "method_markers"):
            if key in config.get("style", {}):
                if key in {"method_colors", "method_markers"} and isinstance(style.get(key), dict):
                    style[key] = {**style[key], **config["style"][key]}
                else:
                    style[key] = config["style"][key]
        base["style"] = style
        base["v51_exploration_contract"] = str(path)
        return base
    raise ValueError("Panel-D exploration expects the accepted Figure 5 V5 or V5.1 exploration config")


def load_existing_formal(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Load the accepted lifecycle table without recomputing or copying it."""

    formal = config["formal_inputs"]
    directory = _repo_path(formal["lifecycle_root"], repo_root) / str(formal["lifecycle_run_id"])
    manifest, qa = _required_formal(
        directory,
        "figure5-validation-v5-lifecycle-1",
        ("lifecycle_summary.csv", "lifecycle_stage_provenance.csv"),
    )
    summary = pd.read_csv(directory / "lifecycle_summary.csv")
    stages = pd.read_csv(directory / "lifecycle_stage_provenance.csv")
    methods = [str(value) for value in config["paper_contract"]["method_order"]]
    required_summary = {
        "method",
        "checkpoint_sha256",
        "native_latency_ms",
        "native_latency_q25_ms",
        "native_latency_q75_ms",
        "replay_equivalent_gpu_hours",
        "replay_equivalent_gpu_hours_low",
        "replay_equivalent_gpu_hours_high",
        "mean_unobserved_relative_l2",
        "mean_unobserved_relative_l2_ci_low",
        "mean_unobserved_relative_l2_ci_high",
        "stage_count",
        "status",
    }
    missing = sorted(required_summary.difference(summary.columns))
    if missing:
        raise ValueError(f"Lifecycle summary is missing columns: {missing}")
    if list(summary["method"].astype(str)) != methods:
        raise ValueError("Lifecycle method order does not match the V5 contract")
    if set(summary["status"].astype(str)) != {"ok"}:
        raise ValueError("Existing formal lifecycle source contains unavailable rows")
    numeric = summary[
        [
            "native_latency_ms",
            "native_latency_q25_ms",
            "native_latency_q75_ms",
            "replay_equivalent_gpu_hours",
            "replay_equivalent_gpu_hours_low",
            "replay_equivalent_gpu_hours_high",
            "mean_unobserved_relative_l2",
            "mean_unobserved_relative_l2_ci_low",
            "mean_unobserved_relative_l2_ci_high",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError("Existing formal lifecycle values must be finite and positive")
    expected_counts = {method: 2 if method == "Latent FM" else 1 for method in methods}
    observed_counts = stages.groupby(stages["method"].astype(str)).size().to_dict()
    if observed_counts != expected_counts:
        raise ValueError(f"Lifecycle stage cardinality mismatch: {observed_counts}")
    if "historical_training_wall_time" in stages and stages["historical_training_wall_time"].astype(bool).any():
        raise ValueError("Historical training wall-clock values are not admissible")
    if str(manifest.get("metric_label")) != "Replay-equivalent model-core training GPU-hours":
        raise ValueError("Unexpected lifecycle metric label")
    return {
        "mode": "existing_formal",
        "directory": directory,
        "manifest": manifest,
        "qa": qa,
        "summary": summary,
        "stages": stages,
        "methods": methods,
        "source_paths": {
            "lifecycle_summary": directory / "lifecycle_summary.csv",
            "lifecycle_stages": directory / "lifecycle_stage_provenance.csv",
        },
    }


def _resolve_benchmark_path(repo_root: Path, requested: Path) -> tuple[Path | None, str | None]:
    """Resolve the exact common-b32 run, waiting on ambiguity or absence."""

    direct = _repo_path(requested, repo_root)
    if direct.is_dir():
        return direct, None
    parent = direct.parent
    if not parent.is_dir():
        return None, f"benchmark directory is absent: {direct}"
    candidates = sorted(path for path in parent.glob(f"{direct.name}*") if path.is_dir())
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        return None, f"benchmark path is ambiguous; expected {direct}, found {[str(path) for path in candidates]}"
    return None, f"benchmark directory is absent: {direct}"


def _nested_values(payload: Any, key: str) -> list[Any]:
    values: list[Any] = []
    if isinstance(payload, dict):
        for name, value in payload.items():
            if str(name).lower() == key.lower():
                values.append(value)
            values.extend(_nested_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_nested_values(value, key))
    return values


def _first_column(df: pd.DataFrame, names: Iterable[str], *, default: Any = np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(default, index=df.index)


def _status_is_ok(value: Any) -> bool:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return token in {"ok", "success", "complete", "completed"}


def _status_is_oom(value: Any) -> bool:
    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return "oom" in token or "out_of_memory" in token or "cuda_oom" in token


def _normalise_common_stages(stage: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize likely benchmark column names without imputing metrics."""

    if "method" not in stage.columns:
        raise ValueError("Common-b32 stage table has no method column")
    result = pd.DataFrame(index=stage.index)
    result["method"] = stage["method"].astype(str)
    result["stage_ordinal"] = pd.to_numeric(
        _first_column(stage, ("stage_ordinal", "stage", "stage_index"), default=1), errors="coerce"
    ).fillna(1).astype(int)
    result["stage_id"] = _first_column(stage, ("stage_id", "stage_name", "stage_role"), default="stage").astype(str)
    result["status"] = _first_column(stage, ("status", "result", "outcome"), default="unavailable").astype(str)
    result["update_ms"] = pd.to_numeric(
        _first_column(
            stage,
            (
                "update_time_median_ms",
                "training_update_time_median_ms",
                "median_update_time_ms",
                "training_update_time_ms",
                "training_update_ms",
                "update_ms",
            ),
        ),
        errors="coerce",
    )
    result["update_q25_ms"] = pd.to_numeric(
        _first_column(stage, ("update_time_q25_ms", "training_update_time_q25_ms", "q25_update_ms")), errors="coerce"
    )
    result["update_q75_ms"] = pd.to_numeric(
        _first_column(stage, ("update_time_q75_ms", "training_update_time_q75_ms", "q75_update_ms")), errors="coerce"
    )
    result["allocated_mib"] = pd.to_numeric(
        _first_column(
            stage,
            (
                "peak_allocated_mib",
                "peak_allocated_memory_mib",
                "peak_allocated_mb",
                "allocated_mib",
            ),
        ),
        errors="coerce",
    )
    result["reserved_mib"] = pd.to_numeric(
        _first_column(
            stage,
            (
                "peak_reserved_mib",
                "peak_reserved_memory_mib",
                "peak_reserved_mb",
                "reserved_mib",
            ),
        ),
        errors="coerce",
    )
    result["training_target_mode"] = _first_column(stage, ("training_target_mode", "target_mode"), default="unknown").astype(str)
    result["n_training_targets"] = pd.to_numeric(
        _first_column(stage, ("n_training_targets", "n_query_points", "target_count")), errors="coerce"
    )

    # A benchmark may expose component metrics only in its method-level
    # summary.  Fill a stage row by exact method match, never by averaging or
    # extrapolating.  Missing values remain NaN and fail strict promotion.
    if "method" in summary.columns:
        method = summary.copy()
        method["method"] = method["method"].astype(str)
        method["_update"] = pd.to_numeric(
            _first_column(
                method,
                (
                    "update_time_median_ms",
                    "training_update_time_median_ms",
                    "median_update_time_ms",
                    "training_update_time_ms",
                    "training_update_ms",
                    "update_ms",
                ),
            ),
            errors="coerce",
        )
        method["_alloc"] = pd.to_numeric(
            _first_column(method, ("peak_allocated_mib", "peak_allocated_memory_mib", "allocated_mib")), errors="coerce"
        )
        method["_reserved"] = pd.to_numeric(
            _first_column(method, ("peak_reserved_mib", "peak_reserved_memory_mib", "reserved_mib")), errors="coerce"
        )
        method = method.drop_duplicates("method").set_index("method")
        for index, row in result.iterrows():
            key = row["method"]
            if key not in method.index:
                continue
            status = str(row["status"])
            # Copy the method-level status only when the stage table omitted
            # one.  A declared OOM remains an OOM boundary; it is never
            # converted into a successful numeric value.
            if status.lower() in {"unavailable", "nan", "none", ""}:
                status_columns = [name for name in ("status", "result", "outcome") if name in method.columns]
                if status_columns:
                    result.loc[index, "status"] = str(method.loc[key, status_columns[0]])
            if not np.isfinite(row["update_ms"]) and np.isfinite(method.loc[key, "_update"]):
                result.loc[index, "update_ms"] = method.loc[key, "_update"]
            if not np.isfinite(row["allocated_mib"]) and np.isfinite(method.loc[key, "_alloc"]):
                result.loc[index, "allocated_mib"] = method.loc[key, "_alloc"]
            if not np.isfinite(row["reserved_mib"]) and np.isfinite(method.loc[key, "_reserved"]):
                result.loc[index, "reserved_mib"] = method.loc[key, "_reserved"]
    return result


def load_common_b32(config: dict[str, Any], repo_root: Path, requested: Path = COMMON_B32_DEFAULT) -> dict[str, Any]:
    """Load common-batch evidence or return a strict-wait record."""

    directory, reason = _resolve_benchmark_path(repo_root, requested)
    if directory is None:
        return {"mode": "common_b32", "status": "strict_wait", "reason": reason, "requested": str(_repo_path(requested, repo_root))}
    required_files = ("training_footprint_summary.csv", "training_stage_summary.csv", "benchmark_repeats.csv")
    missing = [name for name in ("manifest.json", "qa.json", *required_files) if not (directory / name).is_file()]
    if missing:
        return {"mode": "common_b32", "status": "strict_wait", "reason": f"incomplete benchmark bundle; missing {missing}", "requested": str(directory)}
    manifest = _read_json(directory / "manifest.json")
    qa = _read_json(directory / "qa.json")
    if manifest.get("formal") is not True or manifest.get("status") != "complete":
        return {"mode": "common_b32", "status": "strict_wait", "reason": "benchmark manifest is not formal/complete", "requested": str(directory)}
    if qa.get("status") != "pass":
        return {"mode": "common_b32", "status": "strict_wait", "reason": "benchmark QA is not pass", "requested": str(directory)}
    # A pass status alone is insufficient for a training-footprint source:
    # the earlier SiT spike-state bundle was marked QA-pass while all measured
    # optimizer/EMA updates were skipped.  Require the explicit counter gates
    # before exposing any common-batch resource coordinate.
    counter_flags = ("optimizer_step_counters_pass", "ema_update_counters_pass")
    missing_counter_flags = [name for name in counter_flags if name not in qa]
    failed_counter_flags = [name for name in counter_flags if name in qa and qa.get(name) is not True]
    sit_counter_flags = ("sit_optimizer_step_successes_measured_100", "sit_ema_update_successes_measured_100")
    missing_sit_flags = [name for name in sit_counter_flags if name not in qa]
    failed_sit_flags = [name for name in sit_counter_flags if name in qa and qa.get(name) is not True]
    if missing_counter_flags or failed_counter_flags or missing_sit_flags or failed_sit_flags:
        details = missing_counter_flags + failed_counter_flags + missing_sit_flags + failed_sit_flags
        return {"mode": "common_b32", "status": "strict_wait", "reason": f"benchmark QA does not confirm optimizer/EMA success counters: {details}", "requested": str(directory)}
    batch_values = [value for value in _nested_values(manifest, "batch_size") if value is not None]
    if not batch_values:
        return {"mode": "common_b32", "status": "strict_wait", "reason": "benchmark manifest does not declare batch_size=32", "requested": str(directory)}
    if any(str(value) not in {"32", "32.0"} for value in batch_values):
        return {"mode": "common_b32", "status": "strict_wait", "reason": f"benchmark manifest has non-32 batch sizes: {batch_values}", "requested": str(directory)}
    dtype_values = [value for value in _nested_values(manifest, "dtype") if value is not None]
    if not dtype_values:
        return {"mode": "common_b32", "status": "strict_wait", "reason": "benchmark manifest does not declare float32", "requested": str(directory)}
    if any(str(value).lower() not in {"float32", "torch.float32"} for value in dtype_values):
        return {"mode": "common_b32", "status": "strict_wait", "reason": f"benchmark manifest is not float32: {dtype_values}", "requested": str(directory)}
    footprint = pd.read_csv(directory / "training_footprint_summary.csv")
    stage_raw = pd.read_csv(directory / "training_stage_summary.csv")
    repeats = pd.read_csv(directory / "benchmark_repeats.csv")
    for frame_name, frame in (("training_footprint_summary.csv", footprint), ("training_stage_summary.csv", stage_raw)):
        if "batch_size" in frame.columns:
            batch_column = pd.to_numeric(frame["batch_size"], errors="coerce").dropna()
            if len(batch_column) == 0 or not (batch_column == 32).all():
                return {"mode": "common_b32", "status": "strict_wait", "reason": f"{frame_name} does not contain batch size 32 throughout", "requested": str(directory)}
        if "dtype" in frame.columns:
            dtype_column = frame["dtype"].dropna().astype(str).str.lower()
            if len(dtype_column) and not dtype_column.isin({"float32", "torch.float32"}).all():
                return {"mode": "common_b32", "status": "strict_wait", "reason": f"{frame_name} is not float32 throughout", "requested": str(directory)}
    stages = _normalise_common_stages(stage_raw, footprint)
    methods = [str(value) for value in config["paper_contract"]["method_order"]]
    if set(stages["method"]) != set(methods):
        return {"mode": "common_b32", "status": "strict_wait", "reason": f"benchmark methods differ: {sorted(set(stages['method']))}", "requested": str(directory)}
    if len(repeats) == 0:
        return {"mode": "common_b32", "status": "strict_wait", "reason": "benchmark repeat table is empty", "requested": str(directory)}
    expected_counts = {method: 2 if method == "Latent FM" else 1 for method in methods}
    observed_counts = stages.groupby("method").size().to_dict()
    if observed_counts != expected_counts:
        return {"mode": "common_b32", "status": "strict_wait", "reason": f"common-b32 stage cardinality mismatch: {observed_counts}", "requested": str(directory)}
    # Every successful stage must expose all plotted resource quantities.  OOM
    # rows remain valid boundaries, but no numeric value is invented for them.
    for row in stages.itertuples(index=False):
        if _status_is_ok(row.status):
            metrics = np.asarray([row.update_ms, row.allocated_mib, row.reserved_mib], dtype=float)
            if not np.isfinite(metrics).all() or (metrics <= 0).any():
                return {"mode": "common_b32", "status": "strict_wait", "reason": f"successful stage {row.method}/{row.stage_ordinal} lacks finite positive timing/memory", "requested": str(directory)}
        elif not _status_is_oom(row.status):
            return {"mode": "common_b32", "status": "strict_wait", "reason": f"unsupported stage status {row.status!r} for {row.method}", "requested": str(directory)}
    return {
        "mode": "common_b32",
        "status": "ready",
        "directory": directory,
        "manifest": manifest,
        "qa": qa,
        "footprint": footprint,
        "stages": stages,
        "repeats": repeats,
        "methods": methods,
        "batch_sizes": [int(float(value)) for value in batch_values],
        "dtype_values": [str(value) for value in dtype_values],
        "source_paths": {
            "manifest": directory / "manifest.json",
            "qa": directory / "qa.json",
            "footprint": directory / "training_footprint_summary.csv",
            "stages": directory / "training_stage_summary.csv",
            "repeats": directory / "benchmark_repeats.csv",
        },
    }


def _finite_range(values: Iterable[float], *, log: bool = False, pad: float = 1.3) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array) & (array > 0 if log else np.ones_like(array, dtype=bool))]
    if len(array) == 0:
        return (1.0, 10.0)
    low, high = float(array.min()), float(array.max())
    if log:
        return low / pad, high * pad
    span = max(high - low, high * 0.1, 1e-6)
    return low - 0.08 * span, high + 0.08 * span


def _error_limits(summary: pd.DataFrame) -> tuple[float, float]:
    low = float(summary["mean_unobserved_relative_l2_ci_low"].min())
    high = float(summary["mean_unobserved_relative_l2_ci_high"].max())
    span = max(high - low, 0.02)
    return max(0.0, low - 0.18 * span), high + 0.18 * span


def _ordered_summary(data: dict[str, Any]) -> pd.DataFrame:
    summary = data["summary"].copy()
    return summary.sort_values("mean_unobserved_relative_l2", ascending=True).reset_index(drop=True)


def _add_method_legend(fig: plt.Figure, config: dict[str, Any], methods: Iterable[str], *, y: float = 0.01) -> None:
    colors, markers = method_colors(config), method_markers(config)
    handles = [
        plt.Line2D([], [], color=colors[method], marker=markers[method], linestyle="-", linewidth=1.0, markersize=4.0, label=method)
        for method in methods
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, y), ncol=min(4, len(handles)), frameon=False, fontsize=6.0, handlelength=1.4, columnspacing=1.0)


def _add_figure_tag(fig: plt.Figure, label: str) -> None:
    """Place the candidate tag in the figure margin, away from headings."""

    artist = fig.text(0.015, 0.997, label, ha="left", va="top", fontsize=9.5, fontweight="bold", color="#202020")
    artist.set_gid(f"font-role:panel_label:{label}")


def _highlight_rows(axes: Iterable[plt.Axes], y: float, *, height: float = 0.82) -> None:
    for ax in axes:
        ax.axhspan(y - height / 2, y + height / 2, color="#E63946", alpha=0.055, zorder=0)


def _format_method_tick(ax: plt.Axes, methods: list[str], colors: dict[str, str]) -> None:
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    for label, method in zip(ax.get_yticklabels(), methods):
        label.set_color(colors[method])
        if method == "DMF-Gen":
            label.set_fontweight("bold")


def _plot_error_rows(ax: plt.Axes, summary: pd.DataFrame, methods: list[str], colors: dict[str, str], *, show_labels: bool = True) -> None:
    for y, row in enumerate(summary.itertuples(index=False)):
        method = str(row.method)
        if method == "DMF-Gen":
            ax.axhspan(y - 0.42, y + 0.42, color="#E63946", alpha=0.055, zorder=0)
        ax.hlines(y, float(row.mean_unobserved_relative_l2_ci_low), float(row.mean_unobserved_relative_l2_ci_high), color=colors[method], linewidth=1.25, zorder=3)
        ax.vlines([float(row.mean_unobserved_relative_l2_ci_low), float(row.mean_unobserved_relative_l2_ci_high)], y - 0.10, y + 0.10, color=colors[method], linewidth=0.75, zorder=3)
        ax.plot(float(row.mean_unobserved_relative_l2), y, marker="o", markersize=5.2 if method == "DMF-Gen" else 4.4, markerfacecolor="white", markeredgecolor=colors[method], markeredgewidth=0.9, linestyle="none", zorder=4)
        if show_labels:
            ax.text(float(row.mean_unobserved_relative_l2_ci_high), y + 0.20, f"{float(row.mean_unobserved_relative_l2):.3f}", color=colors[method], fontsize=5.8, ha="left", va="bottom")
    ax.set_ylim(-0.58, len(methods) - 0.42)
    ax.set_xlim(*_error_limits(summary))
    _format_method_tick(ax, methods, colors)
    ax.set_xlabel("Mean unobserved-field relative $L_2$\n(lower is better)")
    style_grid(ax, axis="x")


def _native_summary_row(summary: pd.DataFrame, method: str) -> pd.Series:
    return summary.loc[summary["method"].astype(str).eq(method)].iloc[0]


def _plot_cost_rows(
    ax: plt.Axes,
    summary: pd.DataFrame,
    methods: list[str],
    colors: dict[str, str],
    *,
    value: str,
    low: str | None,
    high: str | None,
    xlabel: str,
    log: bool = True,
    annotate: bool = True,
) -> None:
    values: list[float] = []
    for y, method in enumerate(methods):
        row = _native_summary_row(summary, method)
        value_float = float(row[value])
        values.append(value_float)
        if method == "DMF-Gen":
            ax.axhspan(y - 0.42, y + 0.42, color="#E63946", alpha=0.055, zorder=0)
        if low and high:
            ax.hlines(y, float(row[low]), float(row[high]), color=colors[method], linewidth=1.1, zorder=3)
            ax.vlines([float(row[low]), float(row[high])], y - 0.09, y + 0.09, color=colors[method], linewidth=0.7, zorder=3)
        ax.plot(value_float, y, marker="o", markersize=4.6, markerfacecolor="white", markeredgecolor=colors[method], markeredgewidth=0.9, linestyle="none", zorder=4)
        if annotate:
            label = f"{value_float:.2f}" if value_float < 100 else f"{value_float:.0f}"
            ax.text(value_float, y + 0.20, label, fontsize=5.5, color=colors[method], ha="center", va="bottom")
    ax.set_ylim(-0.58, len(methods) - 0.42)
    if log:
        ax.set_xscale("log")
    ax.set_xlim(*_finite_range(values, log=log))
    ax.set_yticks(np.arange(len(methods)))
    ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_xlabel(xlabel)
    style_grid(ax, axis="x")


def _stage_records(common: dict[str, Any], method: str) -> pd.DataFrame:
    return common["stages"].loc[common["stages"]["method"].eq(method)].sort_values("stage_ordinal").reset_index(drop=True)


def _common_limits(common: dict[str, Any], column: str) -> tuple[float, float]:
    values = common["stages"].loc[common["stages"][column].notna(), column].astype(float).tolist()
    return _finite_range(values, log=True)


def _draw_common_stage_cost(
    ax: plt.Axes,
    common: dict[str, Any],
    methods: list[str],
    colors: dict[str, str],
    *,
    column: str,
    q25_column: str | None,
    q75_column: str | None,
    xlabel: str,
    annotate: bool = True,
) -> None:
    finite = common["stages"][column].dropna().astype(float).tolist()
    xlim = _finite_range(finite, log=True)
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    boundary_x = xlim[1] / 1.08
    for y, method in enumerate(methods):
        if method == "DMF-Gen":
            ax.axhspan(y - 0.42, y + 0.42, color="#E63946", alpha=0.055, zorder=0)
        rows = _stage_records(common, method)
        successful = rows.loc[rows["status"].map(_status_is_ok)]
        if successful.empty:
            ax.plot(boundary_x, y, marker="|", markersize=11, markeredgewidth=1.35, color=colors[method], linestyle="none", zorder=4)
            ax.text(boundary_x, y + 0.20, "OOM", fontsize=5.3, color=colors[method], ha="right", va="bottom")
            continue
        offsets = np.linspace(-0.12, 0.12, max(len(successful), 1)) if len(successful) > 1 else [0.0]
        for index, (offset, row) in enumerate(zip(offsets, successful.itertuples(index=False))):
            value = float(getattr(row, column))
            low = float(getattr(row, q25_column)) if q25_column and np.isfinite(getattr(row, q25_column)) else value
            high = float(getattr(row, q75_column)) if q75_column and np.isfinite(getattr(row, q75_column)) else value
            ax.hlines(y + offset, low, high, color=colors[method], linewidth=1.0, zorder=3)
            ax.plot(value, y + offset, marker="o", markersize=4.1, markerfacecolor="white", markeredgecolor=colors[method], markeredgewidth=0.8, linestyle="none", zorder=4)
            if annotate:
                label = f"S{int(row.stage_ordinal)} {value:.1f}" if len(successful) > 1 else f"{value:.1f}"
                # The two Latent-FM stage values are close on the log x scale;
                # place their labels on opposite sides of the row centre.
                label_delta = -0.20 if len(successful) > 1 and index == 0 else 0.20
                ax.text(value, y + offset + label_delta, label, fontsize=5.0, color=colors[method], ha="center", va="bottom" if label_delta > 0 else "top")
        failed = rows.loc[~rows["status"].map(_status_is_ok)]
        if not failed.empty:
            ax.plot(boundary_x, y - 0.20, marker="|", markersize=8, markeredgewidth=1.0, color=colors[method], linestyle="none", zorder=4)
    ax.set_ylim(-0.58, len(methods) - 0.42)
    ax.set_yticks(np.arange(len(methods)))
    ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_xlabel(xlabel)
    style_grid(ax, axis="x")


def _common_row_status(common: dict[str, Any], method: str) -> str:
    rows = _stage_records(common, method)
    if any(_status_is_oom(value) for value in rows["status"]):
        return "OOM at B=32"
    return "ok"


def make_d1(data: dict[str, Any], config: dict[str, Any], mode: str) -> plt.Figure:
    summary = _ordered_summary(data)
    methods = summary["method"].astype(str).tolist()
    colors = method_colors(config)
    if mode == "existing_formal":
        fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 79 * MM), gridspec_kw={"width_ratios": [2.65, 1.25, 1.55, 0.75], "wspace": 0.28}, sharey=True)
        _plot_error_rows(axes[0], summary, methods, colors)
        _plot_cost_rows(axes[1], summary, methods, colors, value="native_latency_ms", low="native_latency_q25_ms", high="native_latency_q75_ms", xlabel="Native inference\nlatency (ms)")
        _plot_cost_rows(axes[2], summary, methods, colors, value="replay_equivalent_gpu_hours", low="replay_equivalent_gpu_hours_low", high="replay_equivalent_gpu_hours_high", xlabel="Replay-equivalent\ntraining (GPU h)")
        axes[3].set_xlim(0.5, 2.55)
        axes[3].set_xticks([1, 2])
        axes[3].set_xlabel("Required\nstages (count)")
        axes[3].set_ylim(-0.58, len(methods) - 0.42)
        axes[3].set_yticks(np.arange(len(methods)))
        axes[3].tick_params(axis="y", labelleft=False, length=0)
        for y, row in enumerate(summary.itertuples(index=False)):
            method = str(row.method)
            axes[3].axhspan(y - 0.42, y + 0.42, color="#E63946", alpha=0.055 if method == "DMF-Gen" else 0.0, zorder=0)
            axes[3].plot(float(row.stage_count), y, marker=method_markers(config)[method], markersize=4.3, color=colors[method], markerfacecolor="white", markeredgewidth=0.8, linestyle="none", zorder=3)
            axes[3].text(float(row.stage_count), y + 0.20, str(int(row.stage_count)), fontsize=5.4, color=colors[method], ha="center", va="bottom")
        title = "Accuracy and lifecycle resource footprint"
        subtitle = "Evidence mode: existing formal · accuracy + clean latency + replay-equivalent training GPU-hours"
    else:
        common = data["common"]
        fig, axes = plt.subplots(1, 5, figsize=(183 * MM, 82 * MM), gridspec_kw={"width_ratios": [2.65, 1.1, 1.3, 1.45, 0.55], "wspace": 0.30}, sharey=True)
        _plot_error_rows(axes[0], summary, methods, colors)
        _plot_cost_rows(axes[1], summary, methods, colors, value="native_latency_ms", low="native_latency_q25_ms", high="native_latency_q75_ms", xlabel="Native inference\nlatency (ms)")
        _draw_common_stage_cost(axes[2], common, methods, colors, column="update_ms", q25_column="update_q25_ms", q75_column="update_q75_ms", xlabel="Common B=32\nupdate (ms)")
        _draw_common_stage_cost(axes[3], common, methods, colors, column="allocated_mib", q25_column=None, q75_column=None, xlabel="Common B=32\npeak allocated (MiB)")
        axes[4].set_xlim(0.5, 2.55)
        axes[4].set_xticks([1, 2])
        axes[4].set_xlabel("Stages\n(count)")
        axes[4].set_ylim(-0.58, len(methods) - 0.42)
        axes[4].set_yticks(np.arange(len(methods)))
        axes[4].tick_params(axis="y", labelleft=False, length=0)
        for y, row in enumerate(summary.itertuples(index=False)):
            method = str(row.method)
            axes[4].plot(float(row.stage_count), y, marker=method_markers(config)[method], markersize=4.2, color=colors[method], markerfacecolor="white", markeredgewidth=0.8, linestyle="none", zorder=3)
            axes[4].text(float(row.stage_count), y + 0.20, str(int(row.stage_count)), fontsize=5.3, color=colors[method], ha="center", va="bottom")
            if _common_row_status(common, method) != "ok":
                axes[4].text(2.25, y - 0.22, "OOM", fontsize=4.9, color=colors[method], ha="right", va="top")
        title = "Accuracy and lifecycle resource footprint"
        subtitle = "Evidence mode: common-batch-32 formal · accuracy + clean latency + synchronized update/memory"
    # Keep the compact two-line headings declared by each column.  Flattening
    # them into one line makes the narrow latency/training/stage columns
    # collide at print size.
    axis_titles = [str(ax.get_xlabel()) for ax in axes]
    for ax, axis_title in zip(axes, axis_titles):
        ax.set_title(axis_title, loc="left", fontsize=5.8, pad=3)
        ax.set_xlabel("")
        ax.tick_params(axis="x", pad=2)
    axes[0].set_title("Reconstruction\naccuracy", loc="left", fontsize=5.8, pad=3)
    axes[0].set_ylabel("")
    fig.suptitle(title, x=0.055, y=0.985, ha="left", fontsize=9.3, fontweight="semibold")
    fig.text(0.055, 0.925, subtitle, ha="left", va="top", fontsize=6.0, color="#555555")
    _add_figure_tag(fig, "D1")
    fig.subplots_adjust(left=0.17, right=0.975, bottom=0.21, top=0.80)
    return fig


def _plane_points(
    ax: plt.Axes,
    summary: pd.DataFrame,
    methods: list[str],
    colors: dict[str, str],
    markers: dict[str, str],
    *,
    x: str,
    xlow: str | None,
    xhigh: str | None,
    xlabel: str,
    logx: bool = True,
    title: str = "",
    common: dict[str, Any] | None = None,
    column: str | None = None,
    q25_column: str | None = None,
    q75_column: str | None = None,
) -> None:
    error_low, error_high = _error_limits(summary)
    ax.set_ylim(error_low, error_high)
    ax.set_yscale("linear")
    for method in methods:
        row = _native_summary_row(summary, method)
        y = float(row["mean_unobserved_relative_l2"])
        ylo = float(row["mean_unobserved_relative_l2_ci_low"])
        yhi = float(row["mean_unobserved_relative_l2_ci_high"])
        if method == "DMF-Gen":
            ax.axhspan(ylo, yhi, color="#E63946", alpha=0.05, zorder=0)
        if common is not None and column is not None:
            rows = _stage_records(common, method)
            success = rows.loc[rows["status"].map(_status_is_ok)]
            values = success[column].dropna().astype(float).tolist()
            if values:
                offsets = np.linspace(-0.004, 0.004, len(values)) if len(values) > 1 else [0.0]
                for offset, value in zip(offsets, values):
                    ax.plot(value, y + offset, marker=markers[method], markersize=5.0, markerfacecolor="white", markeredgecolor=colors[method], markeredgewidth=0.9, linestyle="none", zorder=4)
                if len(values) == 1:
                    qlow = float(success.iloc[0][q25_column]) if q25_column and np.isfinite(success.iloc[0][q25_column]) else values[0]
                    qhigh = float(success.iloc[0][q75_column]) if q75_column and np.isfinite(success.iloc[0][q75_column]) else values[0]
                    ax.hlines(y, qlow, qhigh, color=colors[method], linewidth=1.0, zorder=3)
            else:
                xlim = _finite_range(common["stages"][column].dropna().astype(float), log=True)
                ax.plot(xlim[1] / 1.08, y, marker="|", markersize=12, markeredgewidth=1.4, color=colors[method], linestyle="none", zorder=4)
                ax.text(xlim[1] / 1.08, y, "OOM", color=colors[method], fontsize=5.2, ha="right", va="bottom")
        else:
            value = float(row[x])
            ax.plot(value, y, marker=markers[method], markersize=5.2, markerfacecolor="white", markeredgecolor=colors[method], markeredgewidth=0.9, linestyle="none", zorder=4)
            if xlow and xhigh:
                ax.hlines(y, float(row[xlow]), float(row[xhigh]), color=colors[method], linewidth=1.0, zorder=3)
        ax.text(float(np.nanmedian([float(row["mean_unobserved_relative_l2_ci_high"]), yhi])), y, method, color=colors[method], fontsize=5.0, ha="left", va="center", clip_on=True)
    if logx:
        ax.set_xscale("log")
    if common is not None and column is not None:
        ax.set_xlim(*_common_limits(common, column))
    else:
        ax.set_xlim(*_finite_range(summary[x].astype(float), log=logx))
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontsize=7.0, fontweight="semibold", pad=3)
    style_grid(ax)


def make_d2(data: dict[str, Any], config: dict[str, Any], mode: str) -> plt.Figure:
    summary = data["summary"]
    methods = [str(value) for value in config["paper_contract"]["method_order"]]
    colors = method_colors(config)
    markers = method_markers(config)
    fig, axes = plt.subplots(1, 3, figsize=(183 * MM, 73 * MM), sharey=True, gridspec_kw={"wspace": 0.23})
    if mode == "existing_formal":
        specs = [
            ("native_latency_ms", "native_latency_q25_ms", "native_latency_q75_ms", "Warm native latency (ms)", "Inference latency"),
            ("replay_equivalent_gpu_hours", "replay_equivalent_gpu_hours_low", "replay_equivalent_gpu_hours_high", "Replay-equivalent training (GPU h)", "Training compute"),
            ("stage_count", None, None, "Required training stages (count)", "Lifecycle stages"),
        ]
        for ax, (value, low, high, xlabel, title) in zip(axes, specs):
            if value == "stage_count":
                for method in methods:
                    row = _native_summary_row(summary, method)
                    y = float(row["mean_unobserved_relative_l2"])
                    ax.plot(float(row[value]), y, marker=method_markers(config)[method], markersize=5.0, markerfacecolor="white", markeredgecolor=colors[method], markeredgewidth=0.9, linestyle="none")
                    ax.text(float(row[value]), y, method, fontsize=5.0, color=colors[method], ha="left", va="center")
                ax.set_xlim(0.5, 2.5)
                ax.set_xticks([1, 2])
                ax.set_xlabel(xlabel)
                ax.set_title(title, loc="left", fontsize=7.0, fontweight="semibold", pad=3)
                style_grid(ax)
            else:
                _plane_points(ax, summary, methods, colors, markers, x=value, xlow=low, xhigh=high, xlabel=xlabel, title=title)
    else:
        common = data["common"]
        specs = [
            ("native_latency_ms", "native_latency_q25_ms", "native_latency_q75_ms", "Warm native\nlatency (ms)", "Inference latency", None),
            ("update_ms", "update_q25_ms", "update_q75_ms", "Common B=32\nupdate (ms)", "Training update", common),
            ("allocated_mib", None, None, "Common B=32 peak\nallocated (MiB)", "Training memory", common),
        ]
        for ax, (value, low, high, xlabel, title, common_arg) in zip(axes, specs):
            _plane_points(ax, summary, methods, colors, markers, x=value, xlow=low, xhigh=high, xlabel=xlabel, title=title, common=common_arg, column=value if common_arg is not None else None, q25_column=low if common_arg is not None else None, q75_column=high if common_arg is not None else None)
    axes[0].set_ylabel("Mean unobserved-field relative $L_2$\n(lower is better)")
    axes[0].set_ylim(*_error_limits(summary))
    title = "Accuracy–cost trade-offs across the model lifecycle"
    subtitle = "Evidence mode: existing formal · three views share the exact frozen accuracy coordinate" if mode == "existing_formal" else "Evidence mode: common-batch-32 formal · three real-unit resource views share the exact frozen accuracy coordinate"
    fig.suptitle(title, x=0.055, y=0.985, ha="left", fontsize=9.3, fontweight="semibold")
    fig.text(0.055, 0.925, subtitle, ha="left", va="top", fontsize=6.0, color="#555555")
    _add_method_legend(fig, config, methods, y=0.012)
    _add_figure_tag(fig, "D2")
    fig.subplots_adjust(left=0.14, right=0.992, bottom=0.34, top=0.79)
    return fig


def _draw_lollipop_column(ax: plt.Axes, summary: pd.DataFrame, methods: list[str], colors: dict[str, str], value: str, low: str | None = None, high: str | None = None, *, log: bool = True, label_fmt: str = ".1f") -> None:
    vals = []
    for y, method in enumerate(methods):
        row = _native_summary_row(summary, method)
        value_float = float(row[value])
        vals.append(value_float)
        if low and high:
            ax.hlines(y, float(row[low]), float(row[high]), color=colors[method], linewidth=1.0)
        ax.plot(value_float, y, marker="o", markersize=4.1, markerfacecolor=colors[method], markeredgecolor="white", markeredgewidth=0.6)
        ax.text(value_float, y + 0.22, format(value_float, label_fmt), color=colors[method], fontsize=5.0, ha="center", va="bottom")
    ax.set_ylim(-0.58, len(methods) - 0.42)
    if log:
        ax.set_xscale("log")
    ax.set_xlim(*_finite_range(vals, log=log))
    ax.set_yticks(np.arange(len(methods)))
    ax.tick_params(axis="y", labelleft=False, length=0)
    style_grid(ax, axis="x")


def _draw_bar_column(ax: plt.Axes, summary: pd.DataFrame, methods: list[str], colors: dict[str, str], value: str, low: str | None = None, high: str | None = None, *, label_fmt: str = ".1f") -> None:
    vals = []
    for y, method in enumerate(methods):
        row = _native_summary_row(summary, method)
        value_float = float(row[value])
        vals.append(value_float)
        if method == "DMF-Gen":
            ax.axhspan(y - 0.42, y + 0.42, color="#E63946", alpha=0.055, zorder=0)
        if low and high:
            ax.hlines(y, float(row[low]), float(row[high]), color=colors[method], linewidth=1.0, zorder=3)
        ax.barh(y, value_float, color=colors[method], alpha=0.18, edgecolor=colors[method], linewidth=0.8, height=0.36, zorder=2)
        ax.text(value_float, y, format(value_float, label_fmt), color=colors[method], fontsize=5.0, ha="left", va="center", clip_on=False)
    ax.set_ylim(-0.58, len(methods) - 0.42)
    ax.set_xscale("log")
    ax.set_xlim(_finite_range(vals, log=True)[0] / 2, _finite_range(vals, log=True)[1] * 1.2)
    ax.set_yticks(np.arange(len(methods)))
    ax.tick_params(axis="y", labelleft=False, length=0)
    style_grid(ax, axis="x")


def _draw_common_bar_column(ax: plt.Axes, common: dict[str, Any], methods: list[str], colors: dict[str, str], value: str, xlabel: str, *, memory: bool = False) -> None:
    finite = common["stages"][value].dropna().astype(float).tolist()
    xlim = _finite_range(finite, log=True)
    ax.set_xscale("log")
    ax.set_xlim(xlim[0] / 2, xlim[1] * 1.18)
    for y, method in enumerate(methods):
        rows = _stage_records(common, method)
        success = rows.loc[rows["status"].map(_status_is_ok)]
        if success.empty:
            ax.plot(xlim[1] / 1.08, y, marker="|", markersize=11, markeredgewidth=1.35, color=colors[method], linestyle="none")
            ax.text(xlim[1] / 1.08, y, "OOM", color=colors[method], fontsize=5.2, ha="right", va="center")
            continue
        for index, row in enumerate(success.itertuples(index=False)):
            val = float(getattr(row, value))
            yoffset = y + (index - (len(success) - 1) / 2) * 0.15
            ax.barh(yoffset, val, color=colors[method], alpha=0.18, edgecolor=colors[method], linewidth=0.8, height=0.24)
            label = f"S{int(row.stage_ordinal)} {val:.1f}" if len(success) > 1 else f"{val:.1f}"
            # Separate close stage labels vertically instead of placing both
            # at the bar centre (the Latent-FM pair is especially close).
            label_delta = -0.17 if len(success) > 1 and index == 0 else 0.17
            ax.text(val, yoffset + label_delta, label, color=colors[method], fontsize=4.9, ha="left", va="bottom" if label_delta > 0 else "top", clip_on=False)
    ax.set_ylim(-0.58, len(methods) - 0.42)
    ax.set_yticks(np.arange(len(methods)))
    ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_xlabel(xlabel)
    style_grid(ax, axis="x")


def make_d3(data: dict[str, Any], config: dict[str, Any], mode: str) -> plt.Figure:
    summary = _ordered_summary(data)
    methods = summary["method"].astype(str).tolist()
    colors = method_colors(config)
    if mode == "existing_formal":
        fig, axes = plt.subplots(1, 4, figsize=(183 * MM, 80 * MM), sharey=True, gridspec_kw={"width_ratios": [2.8, 1.5, 1.8, 0.8], "wspace": 0.28})
        _plot_error_rows(axes[0], summary, methods, colors)
        axes[0].set_title("Error", loc="left", fontsize=7.0, fontweight="semibold", pad=3)
        _draw_lollipop_column(axes[1], summary, methods, colors, "native_latency_ms", "native_latency_q25_ms", "native_latency_q75_ms", label_fmt=".1f")
        axes[1].set_xlabel("Native latency (ms)")
        _draw_bar_column(axes[2], summary, methods, colors, "replay_equivalent_gpu_hours", "replay_equivalent_gpu_hours_low", "replay_equivalent_gpu_hours_high", label_fmt=".1f")
        axes[2].set_xlabel("Replay-equivalent\ntraining (GPU h)")
        axes[3].set_xlim(0.5, 2.5)
        axes[3].set_xticks([1, 2])
        axes[3].set_xlabel("Stages")
        axes[3].set_ylim(-0.58, len(methods) - 0.42)
        axes[3].set_yticks(np.arange(len(methods)))
        axes[3].tick_params(axis="y", labelleft=False, length=0)
        for y, method in enumerate(methods):
            row = _native_summary_row(summary, method)
            axes[3].plot(float(row.stage_count), y, marker=method_markers(config)[method], markersize=4.1, markerfacecolor="white", markeredgecolor=colors[method], markeredgewidth=0.8, linestyle="none")
            axes[3].text(float(row.stage_count), y + 0.20, str(int(row.stage_count)), color=colors[method], fontsize=5.0, ha="center", va="bottom")
        subtitle = "Evidence mode: existing formal · graphical table with actual units and required stage count"
    else:
        common = data["common"]
        fig, axes = plt.subplots(1, 5, figsize=(183 * MM, 82 * MM), sharey=True, gridspec_kw={"width_ratios": [2.8, 1.35, 1.55, 1.65, 0.8], "wspace": 0.27})
        _plot_error_rows(axes[0], summary, methods, colors)
        axes[0].set_title("Error", loc="left", fontsize=7.0, fontweight="semibold", pad=3)
        axes[0].set_xlabel("Relative $L_2$\nerror (lower is better)", fontsize=6.4, labelpad=2)
        _draw_lollipop_column(axes[1], summary, methods, colors, "native_latency_ms", "native_latency_q25_ms", "native_latency_q75_ms", label_fmt=".1f")
        axes[1].set_xlabel("Native\nlatency (ms)", fontsize=6.4, labelpad=2)
        _draw_common_bar_column(axes[2], common, methods, colors, "update_ms", "B=32 update (ms)")
        axes[2].set_xlabel("B=32\nupdate (ms)", fontsize=6.4, labelpad=2)
        _draw_common_bar_column(axes[3], common, methods, colors, "allocated_mib", "B=32 peak allocated (MiB)", memory=True)
        axes[3].set_xlabel("B=32 peak\nallocated (MiB)", fontsize=6.4, labelpad=2)
        axes[4].set_xlim(0.5, 2.5)
        axes[4].set_xticks([1, 2])
        axes[4].set_xlabel("Stages", fontsize=6.4, labelpad=2)
        axes[4].set_ylim(-0.58, len(methods) - 0.42)
        axes[4].set_yticks(np.arange(len(methods)))
        axes[4].tick_params(axis="y", labelleft=False, length=0)
        for y, method in enumerate(methods):
            row = _native_summary_row(summary, method)
            axes[4].plot(float(row.stage_count), y, marker=method_markers(config)[method], markersize=4.1, markerfacecolor="white", markeredgecolor=colors[method], markeredgewidth=0.8, linestyle="none")
            axes[4].text(float(row.stage_count), y + 0.20, str(int(row.stage_count)), color=colors[method], fontsize=5.0, ha="center", va="bottom")
            if _common_row_status(common, method) != "ok":
                axes[4].text(2.4, y - 0.2, "OOM", fontsize=4.8, color=colors[method], ha="right", va="top")
        subtitle = "Evidence mode: common-batch-32 formal · graphical table with actual units, stage markers and OOM boundary"
    fig.suptitle("Adopted-checkpoint accuracy and lifecycle costs", x=0.055, y=0.985, ha="left", fontsize=9.3, fontweight="semibold")
    fig.text(0.055, 0.925, subtitle, ha="left", va="top", fontsize=6.0, color="#555555")
    _add_figure_tag(fig, "D3")
    fig.subplots_adjust(left=0.17, right=0.992, bottom=0.255 if mode == "common_b32" else 0.21, top=0.80)
    return fig


def _draw_strip(ax: plt.Axes, methods: list[str], colors: dict[str, str], values: dict[str, list[tuple[str, float]]], *, title: str, unit: str, log: bool = True) -> None:
    all_values = [value for rows in values.values() for _, value in rows if np.isfinite(value) and value > 0]
    xlim = _finite_range(all_values, log=True)
    ax.set_xscale("log" if log else "linear")
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.58, len(methods) - 0.42)
    ax.set_yticks(np.arange(len(methods)))
    ax.tick_params(axis="y", labelleft=False, length=0)
    ax.set_title(title, loc="left", fontsize=7.0, fontweight="semibold", pad=3)
    ax.set_xlabel(unit)
    for y, method in enumerate(methods):
        if method == "DMF-Gen":
            ax.axhspan(y - 0.42, y + 0.42, color="#E63946", alpha=0.055, zorder=0)
        rows = values[method]
        finite_rows = [(label, value) for label, value in rows if np.isfinite(value) and value > 0]
        if not finite_rows:
            ax.plot(xlim[1] / 1.08, y, marker="|", markersize=11, markeredgewidth=1.35, color=colors[method], linestyle="none")
            ax.text(xlim[1] / 1.08, y, "OOM", fontsize=5.1, color=colors[method], ha="right", va="center")
            continue
        offsets = np.linspace(-0.14, 0.14, len(finite_rows)) if len(finite_rows) > 1 else [0.0]
        for index, (offset, (label, value)) in enumerate(zip(offsets, finite_rows)):
            ax.plot(value, y + offset, marker="s", markersize=4.3, markerfacecolor=colors[method], markeredgecolor="white", markeredgewidth=0.55, linestyle="none", zorder=3)
            # Keep two stage labels readable when their x values are close;
            # the first label sits below its marker and the second above.
            label_delta = -0.20 if len(finite_rows) > 1 and index == 0 else 0.20
            ax.text(value, y + offset + label_delta, f"{label} {value:.1f}", fontsize=4.8, color=colors[method], ha="center", va="bottom" if label_delta > 0 else "top")
    style_grid(ax, axis="x")


def make_d4(data: dict[str, Any], config: dict[str, Any], mode: str) -> plt.Figure:
    summary = _ordered_summary(data)
    methods = summary["method"].astype(str).tolist()
    colors = method_colors(config)
    if mode == "existing_formal":
        stage = data["stages"].copy()
        train_values: dict[str, list[tuple[str, float]]] = {}
        for method in methods:
            rows = stage.loc[stage["method"].astype(str).eq(method)]
            if method == "Latent FM":
                train_values[method] = [(f"S{int(row.stage_ordinal)}", float(row.replay_equivalent_gpu_hours)) for row in rows.itertuples(index=False)]
            else:
                total = float(_native_summary_row(summary, method)["replay_equivalent_gpu_hours"])
                train_values[method] = [("total", total)]
        inference_values = {method: [("", float(_native_summary_row(summary, method)["native_latency_ms"]))] for method in methods}
        unit_train, title_train = "Replay-equivalent training (GPU h)", "Training footprint"
        subtitle = "Evidence mode: existing formal · separate training and inference strips; endpoint error is direct"
    else:
        common = data["common"]
        train_values = {}
        for method in methods:
            rows = _stage_records(common, method).loc[_stage_records(common, method)["status"].map(_status_is_ok)]
            train_values[method] = [(f"S{int(row.stage_ordinal)}", float(row.update_ms)) for row in rows.itertuples(index=False)]
        inference_values = {method: [("", float(_native_summary_row(summary, method)["native_latency_ms"]))] for method in methods}
        unit_train, title_train = "Common B=32 update (ms)", "Training update"
        subtitle = "Evidence mode: common-batch-32 formal · update and inference scales remain in separate labelled strips"
    fig = plt.figure(figsize=(183 * MM, 73 * MM))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.9, 1.9, 2.4], wspace=0.27)
    ax_train = fig.add_subplot(grid[0, 0])
    ax_infer = fig.add_subplot(grid[0, 1], sharey=ax_train)
    ax_error = fig.add_subplot(grid[0, 2], sharey=ax_train)
    _draw_strip(ax_train, methods, colors, train_values, title=title_train, unit=unit_train)
    _draw_strip(ax_infer, methods, colors, inference_values, title="Native inference", unit="Clean warm latency (ms)")
    _plot_error_rows(ax_error, summary, methods, colors)
    ax_error.set_title("Reconstruction endpoint", loc="left", fontsize=7.0, fontweight="semibold", pad=3)
    ax_error.set_xlabel("Relative $L_2$\nerror (lower is better)" if mode == "common_b32" else "Mean unobserved-field relative $L_2$")
    ax_error.set_yticks(np.arange(len(methods)))
    ax_error.tick_params(axis="y", labelleft=True)
    ax_error.set_yticklabels(methods)
    for label, method in zip(ax_error.get_yticklabels(), methods):
        label.set_color(colors[method])
        if method == "DMF-Gen":
            label.set_fontweight("bold")
    fig.suptitle("From training to sparse-field reconstruction", x=0.055, y=0.985, ha="left", fontsize=9.3, fontweight="semibold")
    fig.text(0.055, 0.925, subtitle, ha="left", va="top", fontsize=6.0, color="#555555")
    _add_figure_tag(fig, "D4")
    fig.subplots_adjust(left=0.17, right=0.992, bottom=0.21, top=0.80)
    return fig


def _cell_text(value: Any, *, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def _draw_matrix(ax: plt.Axes, summary: pd.DataFrame, config: dict[str, Any], mode: str, common: dict[str, Any] | None = None) -> None:
    methods = summary["method"].astype(str).tolist()
    colors = method_colors(config)
    if mode == "existing_formal":
        columns = [
            ("Method", "method", 1.65),
            ("Error\nrelative $L_2$", "mean_unobserved_relative_l2", 1.7),
            ("Native\nlatency (ms)", "native_latency_ms", 1.35),
            ("Replay training\nGPU h", "replay_equivalent_gpu_hours", 1.55),
            ("Stages\n(count)", "stage_count", 0.8),
        ]
    else:
        columns = [
            ("Method", "method", 1.55),
            ("Error\nrelative $L_2$", "mean_unobserved_relative_l2", 1.85),
            ("Native\nlatency (ms)", "native_latency_ms", 1.25),
            ("B=32 update\n(ms)", "update_ms", 1.35),
            ("B=32 peak\nallocated (MiB)", "allocated_mib", 1.65),
            ("Stages\n(count)", "stage_count", 0.8),
        ]
    widths = np.asarray([column[2] for column in columns], dtype=float)
    edges = np.concatenate(([0.0], np.cumsum(widths) / widths.sum()))
    row_height = 1.0 / (len(methods) + 1)
    for x0, x1, (header, key, _) in zip(edges[:-1], edges[1:], columns):
        ax.add_patch(plt.Rectangle((x0, 1 - row_height), x1 - x0, row_height, facecolor="#F2F2F2", edgecolor="white", linewidth=0.6))
        ax.text((x0 + x1) / 2, 1 - row_height / 2, header, ha="center", va="center", fontsize=5.6, fontweight="semibold")
    for row_index, row in enumerate(summary.itertuples(index=False), start=1):
        y0 = 1 - (row_index + 1) * row_height
        method = str(row.method)
        if method == "DMF-Gen":
            ax.add_patch(plt.Rectangle((0.0, y0), 1.0, row_height, facecolor="#E63946", alpha=0.055, edgecolor="none"))
        for x0, x1, (_, key, _) in zip(edges[:-1], edges[1:], columns):
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, row_height, facecolor="none", edgecolor="#E0E0E0", linewidth=0.45))
            if key == "method":
                text = method
                color = colors[method]
                weight = "bold" if method == "DMF-Gen" else "normal"
            elif key in {"mean_unobserved_relative_l2", "native_latency_ms", "replay_equivalent_gpu_hours", "stage_count"}:
                text = _cell_text(getattr(row, key), digits=3 if key == "mean_unobserved_relative_l2" else (0 if key == "stage_count" else 2))
                color = colors[method]
                weight = "bold" if key == "mean_unobserved_relative_l2" else "normal"
            else:
                records = _stage_records(common, method) if common is not None else pd.DataFrame()
                ok = records.loc[records["status"].map(_status_is_ok)]
                if ok.empty:
                    text, color = "OOM at B=32", colors[method]
                else:
                    text = "/".join(_cell_text(value, digits=1) for value in ok[key].astype(float))
                    color = colors[method]
                weight = "bold" if key == "update_ms" and method == "DMF-Gen" else "normal"
            ax.text((x0 + x1) / 2, y0 + row_height / 2, text, ha="center", va="center", fontsize=5.7 if key == "method" else 5.3, color=color, fontweight=weight)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def make_d5(data: dict[str, Any], config: dict[str, Any], mode: str) -> plt.Figure:
    summary = _ordered_summary(data)
    fig, ax = plt.subplots(figsize=(183 * MM, 68 * MM))
    _draw_matrix(ax, summary, config, mode, data.get("common"))
    subtitle = "Evidence mode: existing formal · actual values printed; no composite score" if mode == "existing_formal" else "Evidence mode: common-batch-32 formal · actual values printed; stage pairs and OOM remain explicit"
    fig.suptitle("Lifecycle matrix with accuracy first", x=0.055, y=0.985, ha="left", fontsize=9.3, fontweight="semibold")
    fig.text(0.055, 0.925, subtitle, ha="left", va="top", fontsize=6.0, color="#555555")
    _add_figure_tag(fig, "D5")
    fig.subplots_adjust(left=0.02, right=0.995, bottom=0.02, top=0.80)
    return fig


def _candidate_specs(mode: str) -> list[tuple[str, Any, str]]:
    suffix = "existing_formal" if mode == "existing_formal" else "common_b32"
    return [
        (f"fig5_v51_d1_accuracy_first_scorecard_{suffix}", make_d1, "Accuracy-first lifecycle scorecard"),
        (f"fig5_v51_d2_accuracy_cost_triptych_{suffix}", make_d2, "Three aligned accuracy–cost/lifecycle views"),
        (f"fig5_v51_d3_graphical_lifecycle_table_{suffix}", make_d3, "Graphical lifecycle table"),
        (f"fig5_v51_d4_workflow_strips_{suffix}", make_d4, "End-to-end workflow strips"),
        (f"fig5_v51_d5_lifecycle_matrix_{suffix}", make_d5, "Lifecycle matrix"),
    ]


def _write_source_tables(data: dict[str, Any], derived: Path, mode: str) -> dict[str, str]:
    derived.mkdir(parents=True, exist_ok=True)
    suffix = "existing_formal" if mode == "existing_formal" else "common_b32"
    source = derived / f"panel_d_plot_source_{suffix}.csv"
    summary = data["summary"].copy()
    summary.insert(0, "evidence_mode", mode)
    if mode == "common_b32":
        common = data["common"]
        values: list[dict[str, Any]] = []
        for method in summary["method"].astype(str):
            rows = _stage_records(common, method)
            success = rows.loc[rows["status"].map(_status_is_ok)]
            values.append(
                {
                    "method": method,
                    "common_b32_status": _common_row_status(common, method),
                    "common_b32_update_ms": "/".join(f"{value:.8g}" for value in success["update_ms"].astype(float)) if not success.empty else "",
                    "common_b32_peak_allocated_mib": "/".join(f"{value:.8g}" for value in success["allocated_mib"].astype(float)) if not success.empty else "",
                    "common_b32_peak_reserved_mib": "/".join(f"{value:.8g}" for value in success["reserved_mib"].astype(float)) if not success.empty else "",
                }
            )
        summary = summary.merge(pd.DataFrame(values), on="method", how="left", validate="one_to_one")
    summary.to_csv(source, index=False)
    stages_source = derived / f"panel_d_stage_source_{suffix}.csv"
    if mode == "existing_formal":
        # Keep the existing-mode provenance compact and explicitly exclude
        # the unequal-batch V4.1 memory fields.  They are not an evidence
        # metric for these candidates, even though they exist in the source
        # replay table.
        allowed = [
            "method",
            "stage_ordinal",
            "stage_id",
            "stage_name",
            "checkpoint_sha256",
            "update_count",
            "gpu_count",
            "canonical_update_time_q25_ms",
            "canonical_update_time_median_ms",
            "canonical_update_time_q75_ms",
            "replay_equivalent_gpu_hours_low",
            "replay_equivalent_gpu_hours",
            "replay_equivalent_gpu_hours_high",
            "timing_source",
            "timing_boundary",
            "historical_training_wall_time",
        ]
        stage = data["stages"][[column for column in allowed if column in data["stages"].columns]].copy()
    else:
        stage = data["common"]["stages"].copy()
    stage.to_csv(stages_source, index=False)
    return {"plot_source": str(source), "stage_source": str(stages_source)}


def _companion_text(stem: str, title: str, data: dict[str, Any], mode: str, source_paths: dict[str, str]) -> str:
    summary = _ordered_summary(data)
    lines = [
        f"# Figure 5 V5.1 panel d candidate: {title}",
        "",
        f"- SVG: `{stem}.svg`",
        f"- Evidence mode: `{mode}`",
        "- Task: turbulent-combustion `Cond_T`, 256 observed temperature sensors, native 40,300-point reconstruction.",
        "- Accuracy coordinate: frozen 1,000-state mean unobserved-field relative $L_2$ over `Y_CH4`, `Y_CO`, `U1`, and `p`, with the accepted temporal-bootstrap interval.",
        "",
        "## Scientific role",
        "",
        "This standalone candidate keeps reconstruction accuracy as a direct graphical quantity and separates lifecycle resources into explicitly labelled real-unit views. It is descriptive evidence for the adopted checkpoints, not a matched-budget causal efficiency comparison.",
        "",
        "## Values",
        "",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"- **{row.method}:** relative $L_2$={float(row.mean_unobserved_relative_l2):.4f} "
            f"[{float(row.mean_unobserved_relative_l2_ci_low):.4f}, {float(row.mean_unobserved_relative_l2_ci_high):.4f}], "
            f"native latency={float(row.native_latency_ms):.2f} ms "
            f"[{float(row.native_latency_q25_ms):.2f}, {float(row.native_latency_q75_ms):.2f}], "
            f"stages={int(row.stage_count)}"
            + (f", replay-equivalent training={float(row.replay_equivalent_gpu_hours):.2f} GPU h." if mode == "existing_formal" else ".")
        )
    if mode == "existing_formal":
        lines.extend(
            [
                "",
                "Training column: replay-equivalent model-core training GPU-hours, derived from the accepted canonical update replays; Latent FM includes both sequential stages. The previous unequal-batch training-memory comparison is intentionally excluded.",
                "",
                "Geo-FNO is represented by the accepted lifecycle replay coordinate (two-GPU DDP provenance), not by a common-batch memory claim.",
            ]
        )
    else:
        common = data["common"]
        lines.extend(["", "Common-batch footprint:"])
        for method in data["methods"]:
            rows = _stage_records(common, method)
            if rows["status"].map(_status_is_oom).any() and rows.loc[rows["status"].map(_status_is_ok)].empty:
                lines.append(f"- **{method}:** OOM at batch 32; no numeric common-batch value is shown.")
            else:
                ok = rows.loc[rows["status"].map(_status_is_ok)]
                values = "; ".join(f"stage {int(row.stage_ordinal)}: {float(row.update_ms):.2f} ms/update, {float(row.allocated_mib):.1f} MiB allocated ({float(row.reserved_mib):.1f} MiB reserved)" for row in ok.itertuples(index=False))
                lines.append(f"- **{method}:** {values}.")
        lines.append("- Batch size is exactly 32, float32, one clean GPU, and method-native target support as recorded by the formal benchmark.")
    lines.extend(
        [
            "",
            "## Sources",
            "",
            f"- Plot source: `{source_paths['plot_source']}`",
            f"- Stage source: `{source_paths['stage_source']}`",
            f"- Lifecycle formal source: `{data['source_paths'].get('lifecycle_summary', '')}`",
            "",
            "## Interpretation limits",
            "",
            "No weighted composite score, rank average, bubble-area-only accuracy encoding, or outcome-driven normalization is used. Latent FM stages remain separate wherever stage-level common-batch values are available. A common-batch OOM is a boundary, not a fabricated coordinate.",
        ]
    )
    if mode == "common_b32":
        lines.extend(["", "- Common-batch benchmark sources:"])
        for name, path in data["common"]["source_paths"].items():
            lines.append(f"  - `{name}`: `{path}`")
    return "\n".join(lines) + "\n"


def _audit_svg(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    text_nodes = [node for node in root.iter() if node.tag.endswith("text")]
    image_nodes = [node for node in root.iter() if node.tag.endswith("image")]
    text = " ".join("".join(node.itertext()) for node in text_nodes)
    forbidden = [token for token in ("bubble area", "weighted composite", "unequal-batch") if token in text.lower()]
    return {
        "parseable": True,
        "editable_text": bool(text_nodes),
        "raster_image_count": len(image_nodes),
        "forbidden_text": forbidden,
        "fixed_width": bool(root.attrib.get("width")),
        "fixed_height": bool(root.attrib.get("height")),
    }


def _write_manifest_and_qa(
    derived: Path,
    config_path: Path,
    data: dict[str, Any],
    mode: str,
    source_paths: dict[str, str],
    figure_paths: list[Path],
    *,
    start_head: str,
) -> None:
    all_source_paths: dict[str, str] = {
        **source_paths,
        **{key: str(path) for key, path in data["source_paths"].items()},
        "config": str(config_path),
    }
    if mode == "common_b32":
        all_source_paths.update({f"common_{key}": str(path) for key, path in data["common"]["source_paths"].items()})
    source_hashes: dict[str, str] = {}
    for name, value in all_source_paths.items():
        path = Path(value)
        if path.is_file():
            source_hashes[name] = _sha256(path)
    manifest = {
        "schema_version": "figure5-validation-v5.1-panel-d-exploration-1",
        "status": "complete",
        "formal": mode == "existing_formal" or data.get("common", {}).get("status") == "ready",
        "generated_mode": mode,
        "starting_head": start_head,
        "dataset": DATASET,
        "task": TASK,
        "condition": CONDITION,
        "accuracy_metric": "frozen mean unobserved-field relative L2 with temporal bootstrap CI",
        "evidence_metrics": (
            ["mean_unobserved_relative_l2", "native_latency_ms", "replay_equivalent_model_core_training_gpu_hours", "stage_count"]
            if mode == "existing_formal"
            else ["mean_unobserved_relative_l2", "native_latency_ms", "common_b32_update_time_ms", "common_b32_peak_allocated_mib", "common_b32_peak_reserved_mib", "stage_count"]
        ),
        "timing_boundary": TIMING_BOUNDARY,
        "config_path": str(config_path),
        "source_paths": all_source_paths,
        "source_sha256": source_hashes,
        "figures": [str(path) for path in figure_paths],
        "no_weighted_composite": True,
        "no_unequal_batch_memory_fallback": True,
        "latent_stages_separate": True,
    }
    (derived / f"panel_d_manifest_{mode}.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    audits = {path.name: _audit_svg(path) for path in figure_paths}
    checks = {
        "formal_source_qa_pass": data["qa"].get("status") == "pass",
        "method_count": len(data["summary"]) == 8,
        "all_accuracy_finite": bool(np.isfinite(data["summary"]["mean_unobserved_relative_l2"].to_numpy(float)).all()),
        "all_svg_parseable": all(item["parseable"] for item in audits.values()),
        "all_svg_editable_text": all(item["editable_text"] for item in audits.values()),
        "no_svg_raster_images": all(item["raster_image_count"] == 0 for item in audits.values()),
        "no_forbidden_text": all(not item["forbidden_text"] for item in audits.values()),
        "all_svg_fixed_dimensions": all(item["fixed_width"] and item["fixed_height"] for item in audits.values()),
        "mode_is_explicit": mode in {"existing_formal", "common_b32"},
    }
    if mode == "existing_formal":
        checks["replay_metric_label_exact"] = data["manifest"].get("metric_label") == "Replay-equivalent model-core training GPU-hours"
        checks["latent_two_stages"] = int(data["summary"].set_index("method").loc["Latent FM", "stage_count"]) == 2
    else:
        checks["common_b32_ready"] = data.get("common", {}).get("status") == "ready"
        checks["common_batch_exact"] = all(int(value) == 32 for value in data["common"].get("batch_sizes", [32]))
    qa = {
        "schema_version": "figure5-validation-v5.1-panel-d-exploration-qa-1",
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "svg_audits": audits,
    }
    (derived / f"panel_d_qa_{mode}.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    if qa["status"] != "pass":
        raise RuntimeError(f"Panel-D QA failed: {checks}")


def _write_wait(derived: Path, wait: dict[str, Any], *, timestamp: str, start_head: str) -> Path:
    derived.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "figure5-validation-v5.1-panel-d-common-b32-wait-1",
        "status": "strict_wait",
        "timestamp": timestamp,
        "starting_head": start_head,
        **wait,
        "policy": "No fallback or invented common-batch values; rerun after the formal benchmark bundle is complete and QA-pass.",
    }
    path = derived / "panel_d_common_b32_wait.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def generate_mode(config: dict[str, Any], config_path: Path, repo_root: Path, timestamp: str, mode: str, *, benchmark_path: Path = COMMON_B32_DEFAULT) -> dict[str, Any]:
    existing = load_existing_formal(config, repo_root)
    data: dict[str, Any] = {**existing}
    if mode == "common_b32":
        benchmark = load_common_b32(config, repo_root, benchmark_path)
        if benchmark.get("status") != "ready":
            # Do not expose the loaded formal data in the wait report (it
            # contains Path objects and, more importantly, must not be
            # mistaken for a common-batch fallback).
            return {"status": "strict_wait", "mode": mode, "wait": benchmark}
        data["common"] = benchmark
        data["methods"] = existing["methods"]
    elif mode == "existing_formal":
        data["methods"] = existing["methods"]
    else:
        raise ValueError(mode)
    output_dir = repo_root / "Dis_SI_Process" / "figures" / "exploration" / "figure5_v51" / timestamp / "panel_d_candidates"
    docs_dir = repo_root / "Dis_SI_Process" / "docs" / "exploration" / "figure5_v51" / timestamp / "panel_d_candidates"
    derived = repo_root / "Dis_SI_Process" / "results" / "ValidationV51" / "Derived" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    source_paths = _write_source_tables(data, derived, mode)
    apply_style(config["style"].get("font_family"))
    figure_paths: list[Path] = []
    for stem_base, renderer, title in _candidate_specs(mode):
        fig = renderer(data, config, mode)
        path = output_dir / f"{stem_base}_{timestamp}.svg"
        fig.savefig(path, format="svg", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        figure_paths.append(path)
        companion = docs_dir / f"{path.stem}.md"
        companion.write_text(_companion_text(path.stem, title, data, mode, source_paths), encoding="utf-8")
    _write_manifest_and_qa(derived, config_path, data, mode, source_paths, figure_paths, start_head=_git_value(repo_root, "rev-parse", "HEAD"))
    return {"status": "complete", "mode": mode, "figures": [str(path) for path in figure_paths], "docs": [str(docs_dir / f"{path.stem}.md") for path in figure_paths], "source_paths": source_paths}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_v5.yaml")
    parser.add_argument("--timestamp", default="20260902_1129")
    parser.add_argument("--mode", choices=("auto", "existing_formal", "common_b32"), default="auto")
    parser.add_argument("--benchmark-path", type=Path, default=COMMON_B32_DEFAULT)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = load_config(config_path)
    start_head = _git_value(repo_root, "rev-parse", "HEAD")
    derived = repo_root / "Dis_SI_Process" / "results" / "ValidationV51" / "Derived" / args.timestamp
    requested_modes = ["existing_formal", "common_b32"] if args.mode == "auto" else [args.mode]
    reports: list[dict[str, Any]] = []
    for mode in requested_modes:
        report = generate_mode(config, config_path, repo_root, args.timestamp, mode, benchmark_path=args.benchmark_path)
        if report.get("status") == "strict_wait":
            wait_path = _write_wait(derived, report["wait"], timestamp=args.timestamp, start_head=start_head)
            report["wait_file"] = str(wait_path)
            print(f"STRICT_WAIT[{mode}]: {report['wait']['reason']}")
        else:
            print(f"COMPLETE[{mode}]: {len(report['figures'])} standalone SVG candidates")
        reports.append(report)
    print(json.dumps({"timestamp": args.timestamp, "reports": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict data adapters and compact reducers for Figure 5 V5."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .figure5_v42_data import load_figure5_v42_data


DATASET = "turbulent_combustion"
TASK = "missing_channel_reconstruction"
CONDITION = "Cond_T"


def error_capture_curve(
    uncertainty: np.ndarray,
    absolute_error: np.ndarray,
    fractions: tuple[float, ...],
) -> np.ndarray:
    """Return cumulative absolute-error capture under descending uncertainty."""

    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    absolute_error = np.asarray(absolute_error, dtype=np.float64).reshape(-1)
    if uncertainty.shape != absolute_error.shape or not np.isfinite(uncertainty).all() or not np.isfinite(absolute_error).all():
        raise ValueError("Error-capture inputs must be same-shape finite arrays")
    order = np.argsort(-uncertainty, kind="mergesort")
    cumulative = np.cumsum(absolute_error[order])
    denominator = max(float(cumulative[-1]), np.finfo(np.float64).eps)
    values = []
    for fraction in fractions:
        count = min(len(order), max(1, int(np.ceil(float(fraction) * len(order)))))
        values.append(float(cumulative[count - 1] / denominator))
    values[-1] = 1.0
    return np.asarray(values, dtype=float)


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_formal(directory: Path, schema: str, files: tuple[str, ...]) -> tuple[dict[str, Any], dict[str, Any]]:
    missing = [name for name in ("manifest.json", "qa.json", *files) if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing formal source files under {directory}: {missing}")
    manifest, qa = _json(directory / "manifest.json"), _json(directory / "qa.json")
    if manifest.get("schema_version") != schema or manifest.get("formal") is not True or manifest.get("status") != "complete":
        raise ValueError(f"Source is not formal/complete {schema}: {directory}")
    if qa.get("status") != "pass":
        raise ValueError(f"Source QA did not pass: {directory}")
    return manifest, qa


def derive_lifecycle_v5(config: dict[str, Any], repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Convert accepted canonical update timings to replay-equivalent GPU-hours."""

    formal = config["formal_inputs"]
    training_dir = _repo_path(repo_root, formal["training_cost_root"]) / str(formal["training_cost_run_id"])
    geo_dir = _repo_path(repo_root, formal["geofno_timing_root"]) / str(formal["geofno_timing_run_id"])
    native_dir = _repo_path(repo_root, formal["cost_root"]) / str(formal["cost_run_id"])
    training_manifest, training_qa = _require_formal(
        training_dir,
        "figure5-validation-v4-training-cost-1",
        ("training_stage_summary.csv", "training_cost_summary.csv"),
    )
    geo_manifest, geo_qa = _require_formal(
        geo_dir,
        "figure5-validation-v4.2-geofno-ddp-timing-1",
        ("geofno_ddp_summary.csv",),
    )
    native_manifest, native_qa = _require_formal(
        native_dir,
        "figure5-validation-v3-cost-1",
        ("native_summary.csv", "timing_boundary_audit.csv"),
    )
    if training_manifest.get("historical_wall_clock_used") is not False or training_manifest.get("filesystem_mtime_used") is not False:
        raise ValueError("Lifecycle reducer rejects historical wall time and filesystem timestamps")
    if geo_manifest.get("protocol", {}).get("wall_timing_admissible") is not True:
        raise ValueError("Geo-FNO DDP wall timing is not formally admissible")

    stage = pd.read_csv(training_dir / "training_stage_summary.csv")
    geo = pd.read_csv(geo_dir / "geofno_ddp_summary.csv")
    native = pd.read_csv(native_dir / "native_summary.csv")
    methods = list(config["paper_contract"]["method_order"])
    if list(native["method"].astype(str)) != methods:
        raise ValueError("Native method order differs from the Figure 5 V5 contract")
    if len(geo) != 1 or str(geo.iloc[0]["method"]) != "Geo-FNO":
        raise ValueError("Geo-FNO formal DDP summary must contain one row")
    if int(geo.iloc[0]["device_count"]) != 2:
        raise ValueError("Geo-FNO lifecycle compute requires the formal two-GPU replay")

    rows: list[dict[str, Any]] = []
    for item in stage.itertuples(index=False):
        method = str(item.method)
        gpu_count = 1
        status = str(item.status).lower()
        q25 = float(item.update_time_q25_ms) if status == "ok" else np.nan
        median = float(item.update_time_median_ms) if status == "ok" else np.nan
        q75 = float(item.update_time_q75_ms) if status == "ok" else np.nan
        source = training_dir / "training_stage_summary.csv"
        timing_source = "inherited_formal_v4_single_gpu_replay"
        if method == "Geo-FNO":
            measured = geo.iloc[0]
            status = "ok"
            gpu_count = 2
            q25 = float(measured["wall_time_q25_ms"])
            median = float(measured["wall_time_median_ms"])
            q75 = float(measured["wall_time_q75_ms"])
            source = geo_dir / "geofno_ddp_summary.csv"
            timing_source = "inherited_formal_v42_two_gpu_ddp_replay"
        update_count = int(item.update_count)
        if status != "ok" or update_count <= 0 or not np.isfinite([q25, median, q75]).all():
            raise ValueError(f"No admissible lifecycle timing/update count for {method} stage {item.stage_id}")
        checkpoint_sha = str(item.checkpoint_sha256)
        if method == "Geo-FNO" and checkpoint_sha != str(geo.iloc[0]["checkpoint_sha256"]):
            raise ValueError("Geo-FNO stage and DDP checkpoint identities differ")
        divisor = float(config["formal_protocol"]["training_compute"]["divisor_ms_per_hour"])
        rows.append(
            {
                "dataset": DATASET,
                "task": TASK,
                "condition": CONDITION,
                "method": method,
                "stage_ordinal": int(item.stage_ordinal),
                "stage_id": str(item.stage_id),
                "stage_name": str(item.stage_name),
                "checkpoint_sha256": checkpoint_sha,
                "update_count": update_count,
                "gpu_count": gpu_count,
                "canonical_update_time_q25_ms": q25,
                "canonical_update_time_median_ms": median,
                "canonical_update_time_q75_ms": q75,
                "replay_equivalent_gpu_hours_low": q25 * update_count * gpu_count / divisor,
                "replay_equivalent_gpu_hours": median * update_count * gpu_count / divisor,
                "replay_equivalent_gpu_hours_high": q75 * update_count * gpu_count / divisor,
                "timing_source": timing_source,
                "source_file": str(source),
                "timing_boundary": "model_core_update_preloaded_batch",
                "historical_training_wall_time": False,
            }
        )
    stages = pd.DataFrame(rows)
    expected_counts = {method: (2 if method == "Latent FM" else 1) for method in methods}
    observed_counts = stages.groupby("method").size().to_dict()
    if observed_counts != expected_counts:
        raise ValueError(f"Lifecycle stage cardinality mismatch: {observed_counts}")

    totals = (
        stages.groupby("method", sort=False)
        .agg(
            replay_equivalent_gpu_hours=("replay_equivalent_gpu_hours", "sum"),
            replay_equivalent_gpu_hours_low=("replay_equivalent_gpu_hours_low", "sum"),
            replay_equivalent_gpu_hours_high=("replay_equivalent_gpu_hours_high", "sum"),
            stage_count=("stage_id", "count"),
            total_update_count=("update_count", "sum"),
        )
        .reset_index()
    )
    lifecycle = native.merge(totals, on="method", how="left", validate="one_to_one")
    lifecycle = lifecycle.rename(
        columns={
            "median_latency_ms": "native_latency_ms",
            "latency_q25_ms": "native_latency_q25_ms",
            "latency_q75_ms": "native_latency_q75_ms",
            "error": "mean_unobserved_relative_l2",
            "error_ci_low": "mean_unobserved_relative_l2_ci_low",
            "error_ci_high": "mean_unobserved_relative_l2_ci_high",
        }
    )
    lifecycle.insert(0, "dataset", DATASET)
    lifecycle.insert(1, "task", TASK)
    lifecycle.insert(2, "condition", CONDITION)
    lifecycle["cohort_id"] = "figure4_frozen_1000_states"
    lifecycle["metric_name"] = "replay_equivalent_model_core_training_gpu_hours"
    lifecycle["metric_value"] = lifecycle["replay_equivalent_gpu_hours"]
    lifecycle["status"] = "ok"
    keep = [
        "dataset",
        "task",
        "condition",
        "method",
        "checkpoint_sha256",
        "cohort_id",
        "metric_name",
        "metric_value",
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
        "total_update_count",
        "N",
        "sensor_count",
        "batch_size",
        "dtype",
        "timing_boundary",
        "status",
    ]
    lifecycle = lifecycle[keep]
    numeric = lifecycle[
        [
            "native_latency_ms",
            "replay_equivalent_gpu_hours",
            "mean_unobserved_relative_l2",
        ]
    ].to_numpy(dtype=float)
    checks = {
        "formal_v3_native_source": native_manifest.get("formal") is True and native_qa.get("status") == "pass",
        "formal_v4_training_source": training_manifest.get("formal") is True and training_qa.get("status") == "pass",
        "formal_v42_geofno_source": geo_manifest.get("formal") is True and geo_qa.get("status") == "pass",
        "all_eight_methods": list(lifecycle["method"].astype(str)) == methods,
        "all_values_finite_positive": bool(np.isfinite(numeric).all() and (numeric > 0).all()),
        "latent_fm_two_stages_summed": int(lifecycle.set_index("method").loc["Latent FM", "stage_count"]) == 2,
        "geofno_two_gpu_ddp": int(stages.set_index("method").loc["Geo-FNO", "gpu_count"]) == 2,
        "no_historical_wall_time": not stages["historical_training_wall_time"].astype(bool).any(),
        "checkpoint_identity_joined": not lifecycle["checkpoint_sha256"].astype(str).str.strip().eq("").any(),
    }
    qa = {"status": "pass" if all(checks.values()) else "fail", "checks": checks}
    manifest = {
        "schema_version": "figure5-validation-v5-lifecycle-1",
        "status": "complete" if qa["status"] == "pass" else "qa_failed",
        "formal": qa["status"] == "pass",
        "run_id": str(formal["lifecycle_run_id"]),
        "metric_name": "replay_equivalent_model_core_training_gpu_hours",
        "metric_label": "Replay-equivalent model-core training GPU-hours",
        "definition": "sum(stage median canonical update ms * adopted update count * active GPU count) / 3.6e6",
        "historical_training_wall_time": False,
        "sources": {
            "native": str(native_dir / "native_summary.csv"),
            "training_stages": str(training_dir / "training_stage_summary.csv"),
            "geofno_ddp": str(geo_dir / "geofno_ddp_summary.csv"),
        },
        "source_sha256": {
            "native": _sha(native_dir / "native_summary.csv"),
            "training_stages": _sha(training_dir / "training_stage_summary.csv"),
            "geofno_ddp": _sha(geo_dir / "geofno_ddp_summary.csv"),
        },
        "method_count": len(lifecycle),
        "stage_count": len(stages),
        "unavailable_methods": [],
    }
    return lifecycle, stages, manifest, qa


def materialize_lifecycle_v5(config: dict[str, Any], repo_root: Path) -> Path:
    formal = config["formal_inputs"]
    directory = _repo_path(repo_root, formal["lifecycle_root"]) / str(formal["lifecycle_run_id"])
    lifecycle, stages, manifest, qa = derive_lifecycle_v5(config, repo_root)
    directory.mkdir(parents=True, exist_ok=True)
    expected = {
        "manifest.json": json.dumps(manifest, indent=2) + "\n",
        "qa.json": json.dumps(qa, indent=2) + "\n",
        "lifecycle_summary.csv": lifecycle.to_csv(index=False),
        "lifecycle_stage_provenance.csv": stages.to_csv(index=False),
    }
    if (directory / "manifest.json").is_file():
        existing = _json(directory / "manifest.json")
        if existing.get("status") == "complete":
            for name, content in expected.items():
                if not (directory / name).is_file() or (directory / name).read_text(encoding="utf-8") != content:
                    raise ValueError(f"Completed lifecycle bundle differs from current formal derivation: {directory / name}")
            return directory
    for name, content in expected.items():
        temporary = (directory / name).with_suffix((directory / name).suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(directory / name)
    return directory


def _load_localization(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    formal = config["formal_inputs"]
    directory = _repo_path(repo_root, formal["uq_localization_root"]) / str(formal["uq_localization_run_id"])
    manifest, qa = _require_formal(
        directory,
        "figure5-validation-v5-uq-localization-1",
        ("error_capture_curves.csv", "error_capture_summary.csv"),
    )
    curves = pd.read_csv(directory / "error_capture_curves.csv")
    summary = pd.read_csv(directory / "error_capture_summary.csv")
    methods = list(config["paper_contract"]["generative_method_order"])
    fractions = [float(value) for value in config["formal_protocol"]["localization"]["spatial_fractions"]]
    macro = summary.loc[summary["field"].astype(str).eq("macro_unobserved")].copy()
    checks = {
        "method_set": set(macro["method"].astype(str)) == set(methods),
        "fraction_grid": set(np.round(macro["spatial_fraction"].astype(float), 8)) == set(np.round(fractions, 8)),
        "state_rows": len(curves) == 200 * len(methods),
        "draw_count": set(curves["draw_count"].astype(int)) == {64},
        "dataset": set(curves["dataset"].astype(str)) == {DATASET},
        "finite": bool(np.isfinite(macro[["metric_value", "ci_low", "ci_high", "ec_auc"]].to_numpy(dtype=float)).all()),
    }
    if not all(checks.values()):
        raise ValueError(f"Localization table failed V5 adapter checks: {checks}")
    return {"directory": directory, "manifest": manifest, "qa": qa, "curves": curves, "summary": summary, "macro": macro}


def _load_lifecycle(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    formal = config["formal_inputs"]
    directory = _repo_path(repo_root, formal["lifecycle_root"]) / str(formal["lifecycle_run_id"])
    manifest, qa = _require_formal(
        directory,
        "figure5-validation-v5-lifecycle-1",
        ("lifecycle_summary.csv", "lifecycle_stage_provenance.csv"),
    )
    summary = pd.read_csv(directory / "lifecycle_summary.csv")
    stages = pd.read_csv(directory / "lifecycle_stage_provenance.csv")
    methods = list(config["paper_contract"]["method_order"])
    if list(summary["method"].astype(str)) != methods or len(stages) != 9:
        raise ValueError("Lifecycle bundle does not contain eight methods and nine required stages")
    return {"directory": directory, "manifest": manifest, "qa": qa, "summary": summary, "stages": stages}


def load_figure5_v5_data(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    base, _ = load_figure5_v42_data(config, repo_root)
    localization = _load_localization(config, repo_root)
    lifecycle = _load_lifecycle(config, repo_root)
    uq_dir = base["run_metadata"]["uq"]["directory"]
    reliability = pd.read_csv(uq_dir / "reliability_si.csv")
    nfe_path = _repo_path(repo_root, config["formal_inputs"]["nfe_source"])
    if not nfe_path.is_file():
        raise FileNotFoundError(f"Missing formal NFE SI source: {nfe_path}")
    nfe = pd.read_csv(nfe_path)
    return {
        **base,
        "localization": localization,
        "lifecycle": lifecycle,
        "reliability_si": reliability,
        "nfe_si": nfe,
        "nfe_source": nfe_path,
        "modes_v5": {"a": "formal_reused", "b": "formal_reused", "c": "formal_new", "d": "formal_derived"},
    }

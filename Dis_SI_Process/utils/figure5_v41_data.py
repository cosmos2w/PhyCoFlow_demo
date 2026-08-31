"""Strict source adapter and deterministic derived tables for Figure 5 V4.1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .figure5_v4_data import SourceRecord, load_figure5_v4_data


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (base, *parts))).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) & 0x7FFFFFFF


def _moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n, size=int(np.ceil(n / block)))
    return np.concatenate([(np.arange(block) + start) % n for start in starts])[:n]


def _uq_distributions(run: dict[str, Any], methods: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    states = run["states"].copy()
    spec = run["manifest"]["bootstrap"]
    crps_rows: list[dict[str, Any]] = []
    rho_rows: list[dict[str, Any]] = []
    alpha = (1.0 - float(spec["confidence_level"])) / 2.0
    for method in methods:
        group = states.loc[states["method"].astype(str).eq(method)].sort_values("original_time_index")
        if len(group) != 200:
            raise ValueError(f"V4.1 requires exactly 200 paired UQ states for {method}")
        for row in group.itertuples():
            crps_rows.append(
                {
                    "method": method,
                    "state": int(row.state),
                    "original_time_index": int(row.original_time_index),
                    "normalized_crps": float(row.macro_normalized_crps),
                    "sample_kind": "paired_held_out_state",
                }
            )
        x = group["macro_normalized_spread"].to_numpy(dtype=float)
        y = group["macro_ensemble_mean_relative_l2"].to_numpy(dtype=float)
        rng = np.random.default_rng(_stable_seed(int(spec["seed"]), f"v3|spearman|{method}"))
        block = min(int(spec["block_length"]), len(group))
        values = np.empty(int(spec["replicates"]), dtype=float)
        for replicate in range(len(values)):
            selected = _moving_block_indices(len(group), block, rng)
            values[replicate] = float(spearmanr(x[selected], y[selected]).statistic)
            rho_rows.append(
                {
                    "method": method,
                    "replicate": replicate,
                    "spearman_rho": values[replicate],
                    "sample_kind": "temporal_moving_block_bootstrap",
                    "block_length": block,
                    "bootstrap_seed": int(spec["seed"]),
                }
            )
        expected = run["spread"].loc[run["spread"]["method"].astype(str).eq(method)].iloc[0]
        lo, hi = np.nanquantile(values, [alpha, 1.0 - alpha])
        if not np.allclose(
            [lo, hi],
            [float(expected["spearman_ci_low"]), float(expected["spearman_ci_high"])],
            rtol=0.0,
            atol=5.0e-13,
        ):
            raise ValueError(f"Rebuilt Spearman bootstrap does not reproduce the formal V3 CI for {method}")
    return pd.DataFrame(crps_rows), pd.DataFrame(rho_rows)


def _load_geofno_ddp(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    formal = config["formal_inputs"]
    directory = _repo_path(repo_root, formal["geofno_multigpu_root"]) / str(formal["geofno_multigpu_run_id"])
    errors: list[str] = []
    for name in ("manifest.json", "qa.json", "geofno_ddp_summary.csv"):
        if not (directory / name).is_file():
            errors.append(f"missing {directory / name}")
    if errors:
        return None, errors
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    qa = json.loads((directory / "qa.json").read_text(encoding="utf-8"))
    table = pd.read_csv(directory / "geofno_ddp_summary.csv")
    if manifest.get("schema_version") != "figure5-validation-v4.1-geofno-ddp-memory-1":
        errors.append("Geo-FNO DDP schema mismatch")
    if manifest.get("status") != "complete" or manifest.get("formal") is not True:
        errors.append("Geo-FNO DDP manifest is not formal/complete")
    if qa.get("status") != "pass":
        errors.append("Geo-FNO DDP QA did not pass")
    for key in (
        "global_batch_exact",
        "two_distinct_gpus",
        "checkpoint_identity_pass",
        "memory_repeatability_pass",
        "process_local_allocated_metric",
    ):
        if qa.get(key) is not True:
            errors.append(f"Geo-FNO DDP QA does not pass {key}")
    if len(table) != 1 or str(table.iloc[0].get("method")) != "Geo-FNO":
        errors.append("Geo-FNO DDP summary must contain exactly one Geo-FNO row")
    required = [
        "peak_allocated_mib_per_device_max",
        "peak_allocated_mib_total",
        "device_count",
        "global_batch_size",
    ]
    if any(column not in table.columns for column in required):
        errors.append("Geo-FNO DDP summary lacks required timing/memory columns")
    elif not np.isfinite(table[required].to_numpy(dtype=float)).all():
        errors.append("Geo-FNO DDP summary contains non-finite values")
    else:
        row = table.iloc[0]
        if int(row["device_count"]) != 2 or int(row["global_batch_size"]) != 192:
            errors.append("Geo-FNO DDP replay is not the canonical two-GPU global batch 192")
    if errors:
        return None, errors
    return {"directory": directory, "manifest": manifest, "qa": qa, "summary": table}, []


def _merge_training_cost(
    base: pd.DataFrame,
    geofno: dict[str, Any] | None,
    stage_table: pd.DataFrame,
) -> pd.DataFrame:
    table = base.copy()
    stage_ok = stage_table.loc[stage_table["status"].astype(str).str.lower().eq("ok")].copy()
    memory = stage_ok.groupby("method")["peak_allocated_mib"].max()
    table["device_count"] = 1.0
    table["peak_allocated_mib_per_device_max"] = table["method"].map(memory)
    table["peak_allocated_mib_total"] = table["peak_allocated_mib_per_device_max"]
    table["cost_value"] = table["peak_allocated_mib_total"]
    table["cost_low"] = table["cost_value"]
    table["cost_high"] = table["cost_value"]
    table["cost_metric"] = "peak_allocated_training_memory_mib"
    table["cost_unit"] = "MiB"
    # Latent FM has two required stages. Peak capacity is the maximum stage
    # allocation, not their sum, because the stages do not execute concurrently.
    latent = table["method"].astype(str).eq("Latent FM")
    if table.loc[latent, "cost_value"].notna().all():
        table.loc[latent, "status"] = "ok"
        table.loc[latent, "unavailable_reason"] = ""
    if geofno is not None:
        measured = geofno["summary"].iloc[0]
        mask = table["method"].astype(str).eq("Geo-FNO")
        if mask.sum() != 1:
            raise ValueError("Expected one Geo-FNO row in the V4 training table")
        for column, value in {
            "status": "ok",
            "cost_value": float(measured["peak_allocated_mib_total"]),
            "cost_low": float(measured["peak_allocated_mib_total"]),
            "cost_high": float(measured["peak_allocated_mib_total"]),
            "device_count": 2,
            "peak_allocated_mib_per_device_max": float(measured["peak_allocated_mib_per_device_max"]),
            "peak_allocated_mib_total": float(measured["peak_allocated_mib_total"]),
            "training_cost_basis": "two-GPU DDP canonical global-batch replay; summed simultaneous rank peaks",
            "unavailable_reason": "",
        }.items():
            table.loc[mask, column] = value
    ok = table["status"].astype(str).str.lower().eq("ok")
    if table.loc[ok, "cost_value"].isna().any():
        raise ValueError("Training-memory table contains a promoted row without measured peak allocation")
    return table


def _load_zeroh(config: dict[str, Any], repo_root: Path) -> tuple[pd.DataFrame | None, dict[str, Any] | None, list[str]]:
    formal = config["formal_inputs"]
    source = _repo_path(repo_root, formal["zeroh_per_snapshot"])
    summary_path = _repo_path(repo_root, formal["zeroh_summary"])
    audit_path = _repo_path(repo_root, formal["zeroh_audit"])
    manifest_path = _repo_path(repo_root, formal["zeroh_manifest"])
    missing = [str(path) for path in (source, summary_path, audit_path, manifest_path) if not path.is_file()]
    if missing:
        return None, None, [f"missing Zero-H source: {path}" for path in missing]
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if audit.get("passed") is not True or any(check.get("passed") is not True for check in audit.get("checks", [])):
        errors.append("Zero-H unified audit did not pass")
    if manifest.get("data_run_id") != "2026-08-06_11-24":
        errors.append("Zero-H data run ID is not the audited 2026-08-06 refresh")
    if not any(str(item).endswith(summary_path.name) for item in manifest.get("sources", [])):
        errors.append("Zero-H summary source is not joined by the audited manifest")
    table = pd.read_csv(source)
    summary = pd.read_csv(summary_path)
    recipe = str(config["formal_protocol"]["zeroh"]["recipe"])
    table = table.loc[table["recipe"].astype(str).eq(recipe)].copy()
    table = table.loc[table["status"].astype(str).str.lower().eq("ok")].copy()
    methods = list(config["paper_contract"]["zeroh_method_order"])
    if set(table["model_label"].astype(str)) != set(methods):
        errors.append("Zero-H model set does not match the four audited backup methods")
    expected = int(config["formal_protocol"]["zeroh"]["snapshots_per_method"])
    counts = table.groupby("model_label").size().to_dict()
    if any(int(counts.get(method, 0)) != expected for method in methods):
        errors.append(f"Zero-H source does not contain exactly {expected} valid rows per method")
    if set(table["sensor_count"].astype(int)) != {int(config["formal_protocol"]["zeroh"]["sensor_count"])}:
        errors.append("Zero-H source does not use the canonical 256-sensor protocol")
    metrics = ["physical_rel_l2", "gradient_rel_l2", "physical_rel_l2_sensor_excluded", "normalized_rel_l2"]
    if any(metric not in table.columns for metric in metrics) or not np.isfinite(table[metrics].to_numpy(dtype=float)).all():
        errors.append("Zero-H source lacks finite backup metrics")
    audited_summary = summary.loc[
        summary["recipe"].astype(str).eq(recipe) & summary["model_label"].astype(str).isin(methods)
    ]
    if len(audited_summary) != len(methods):
        errors.append("Audited Zero-H summary lacks one row per backup method")
    else:
        observed = table.groupby("model_label")["physical_rel_l2"].mean()
        expected_means = audited_summary.set_index("model_label")["mean"].astype(float)
        if not np.allclose(observed.loc[methods], expected_means.loc[methods], rtol=0.0, atol=5.0e-13):
            errors.append("Zero-H per-snapshot means do not reproduce the audited summary")
    if errors:
        return None, None, errors
    table["method"] = table["model_label"].astype(str)
    table["recipe_provenance"] = recipe
    return table, {"audit": audit, "manifest": manifest, "source": source, "summary": summary_path, "audit_path": audit_path, "manifest_path": manifest_path}, []


def load_figure5_v41_data(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any], list[SourceRecord]]:
    """Load V4 sources, add V4.1 distributions, DDP evidence, and Zero-H backup."""

    data, records = load_figure5_v4_data(config, repo_root)
    errors = {key: list(value) for key, value in data["source_errors"].items()}
    if data["run_metadata"]["uq"] is not None:
        crps_samples, rho_bootstrap = _uq_distributions(
            data["run_metadata"]["uq"], list(config["paper_contract"]["generative_method_order"])
        )
    else:
        crps_samples, rho_bootstrap = None, None

    geofno, geofno_errors = _load_geofno_ddp(config, repo_root)
    errors["d"].extend(geofno_errors)
    training = None
    if data["training_cost"] is not None:
        stage_path = data["run_metadata"]["training"]["directory"] / "training_stage_summary.csv"
        training = _merge_training_cost(data["training_cost"], geofno, pd.read_csv(stage_path))
    if geofno is None:
        data["modes"]["d"] = "pending"

    zeroh, zeroh_metadata, zeroh_errors = _load_zeroh(config, repo_root)
    data.update(
        {
            "uq_crps_samples": crps_samples,
            "uq_spearman_bootstrap": rho_bootstrap,
            "training_cost": training,
            "training_metric": "peak_allocated_training_memory_mib" if training is not None else None,
            "training_metric_label": "Peak allocated training memory (total MiB)",
            "geofno_multigpu": geofno,
            "zeroh": zeroh,
            "zeroh_metadata": zeroh_metadata,
            "zeroh_errors": zeroh_errors,
            "source_errors": errors,
        }
    )
    data["run_metadata"]["geofno_multigpu"] = geofno
    record_by_panel = {record.panel: record for record in records}
    if geofno is not None:
        source = str(geofno["directory"] / "geofno_ddp_summary.csv")
        record_by_panel["d"] = SourceRecord(
            "d",
            "formal",
            "available",
            source,
            "V4 clean replay peak allocations merged with a formal two-GPU Geo-FNO memory replay; x is total simultaneous allocated MiB.",
        )
        data["modes"]["d"] = "formal"
    else:
        record_by_panel["d"] = SourceRecord(
            "d",
            "pending",
            "missing",
            str(_repo_path(repo_root, config["formal_inputs"]["geofno_multigpu_root"]) / str(config["formal_inputs"]["geofno_multigpu_run_id"])),
            "V4.1 requires a passing two-GPU canonical Geo-FNO replay; no one-GPU OOM proxy is plotted.",
        )
    return data, [record_by_panel[panel] for panel in "abcde"]

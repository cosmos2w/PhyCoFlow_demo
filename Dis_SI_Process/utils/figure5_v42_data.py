"""Strict V4.2 adapter: canonical training-update time remains panel d's x metric."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .figure5_v4_data import SourceRecord, load_figure5_v4_data
from .figure5_v41_data import _uq_distributions


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_geofno_timing(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    formal = config["formal_inputs"]
    directory = _repo_path(repo_root, formal["geofno_timing_root"]) / str(formal["geofno_timing_run_id"])
    errors: list[str] = []
    for name in ("manifest.json", "qa.json", "geofno_ddp_summary.csv", "geofno_ddp_updates.csv"):
        if not (directory / name).is_file():
            errors.append(f"missing {directory / name}")
    if errors:
        return None, errors
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    qa = json.loads((directory / "qa.json").read_text(encoding="utf-8"))
    table = pd.read_csv(directory / "geofno_ddp_summary.csv")
    if manifest.get("schema_version") != "figure5-validation-v4.2-geofno-ddp-timing-1":
        errors.append("Geo-FNO DDP timing schema mismatch")
    if manifest.get("status") != "complete" or manifest.get("formal") is not True:
        errors.append("Geo-FNO DDP timing manifest is not formal/complete")
    if manifest.get("protocol", {}).get("wall_timing_admissible") is not True:
        errors.append("Geo-FNO DDP wall timing is not admissible")
    if manifest.get("protocol", {}).get("promoted_metric") != "synchronized wall ms/global optimizer update":
        errors.append("Geo-FNO DDP promoted metric is not synchronized wall ms/global update")
    for key in (
        "global_batch_exact", "two_distinct_gpus", "checkpoint_identity_pass",
        "all_rank_stability_pass", "gpu_clean_before", "gpu_clean_after",
        "formal_sample_count_exact", "no_archive_write",
    ):
        if qa.get(key) is not True:
            errors.append(f"Geo-FNO DDP QA does not pass {key}")
    required = [
        "wall_time_q25_ms", "wall_time_median_ms", "wall_time_q75_ms",
        "device_count", "global_batch_size", "local_batch_size",
        "peak_allocated_mib_per_device_max", "error", "error_ci_low", "error_ci_high",
    ]
    if len(table) != 1 or str(table.iloc[0].get("method")) != "Geo-FNO":
        errors.append("Geo-FNO timing summary must contain exactly one Geo-FNO row")
    elif any(column not in table.columns for column in required):
        errors.append("Geo-FNO timing summary lacks required columns")
    elif not np.isfinite(table[required].to_numpy(dtype=float)).all():
        errors.append("Geo-FNO timing summary contains non-finite values")
    else:
        row = table.iloc[0]
        if int(row["device_count"]) != 2 or int(row["global_batch_size"]) != 192 or int(row["local_batch_size"]) != 96:
            errors.append("Geo-FNO timing is not the canonical two-GPU global batch 192 replay")
        if not float(row["wall_time_q25_ms"]) <= float(row["wall_time_median_ms"]) <= float(row["wall_time_q75_ms"]):
            errors.append("Geo-FNO wall-time quartiles are not ordered")
    if errors:
        return None, errors
    return {"directory": directory, "manifest": manifest, "qa": qa, "summary": table}, []


def _merge_training_time(base: pd.DataFrame, geofno: dict[str, Any] | None) -> pd.DataFrame:
    table = base.copy()
    table["device_count"] = 1
    table["parallelism"] = "single_gpu"
    table["cost_metric"] = "training_update_time_ms"
    table["cost_unit"] = "ms/update"
    if geofno is not None:
        measured = geofno["summary"].iloc[0]
        mask = table["method"].astype(str).eq("Geo-FNO")
        if mask.sum() != 1:
            raise ValueError("Expected exactly one Geo-FNO row in the V4 training table")
        updates = {
            "status": "ok",
            "cost_value": float(measured["wall_time_median_ms"]),
            "cost_low": float(measured["wall_time_q25_ms"]),
            "cost_high": float(measured["wall_time_q75_ms"]),
            "training_update_time_ms": float(measured["wall_time_median_ms"]),
            "device_count": 2,
            "parallelism": "two_gpu_ddp",
            "global_batch_size": 192,
            "local_batch_size": 96,
            "peak_allocated_mib_per_device_max": float(measured["peak_allocated_mib_per_device_max"]),
            "training_cost_basis": "median synchronized max-rank wall time for one canonical global-batch-192 DDP optimizer update",
            "unavailable_reason": "",
        }
        for column, value in updates.items():
            table.loc[mask, column] = value
    ok = table["status"].astype(str).str.lower().eq("ok")
    if table.loc[ok, "cost_value"].isna().any() or (table.loc[ok, "cost_value"].astype(float) <= 0).any():
        raise ValueError("V4.2 training-time table contains an invalid promoted coordinate")
    return table


def load_figure5_v42_data(config: dict[str, Any], repo_root: Path) -> tuple[dict[str, Any], list[SourceRecord]]:
    data, records = load_figure5_v4_data(config, repo_root)
    if data["run_metadata"]["uq"] is not None:
        crps_samples, rho_bootstrap = _uq_distributions(
            data["run_metadata"]["uq"], list(config["paper_contract"]["generative_method_order"])
        )
    else:
        crps_samples, rho_bootstrap = None, None

    geofno, errors = _load_geofno_timing(config, repo_root)
    base_training = data["training_cost"]
    training = None if base_training is None else _merge_training_time(base_training, geofno)
    data["source_errors"]["d"].extend(errors)
    if geofno is None:
        data["modes"]["d"] = "pending"
    data.update(
        {
            "uq_crps_samples": crps_samples,
            "uq_spearman_bootstrap": rho_bootstrap,
            "training_cost": training,
            "training_metric": "training_update_time_ms" if training is not None else None,
            "training_metric_label": "Canonical training update time (ms/update)",
            "geofno_timing": geofno,
        }
    )
    revised: list[SourceRecord] = []
    for record in records:
        if record.panel != "d":
            revised.append(record)
            continue
        revised.append(
            SourceRecord(
                panel="d",
                mode=data["modes"]["d"],
                status="available" if data["modes"]["d"] == "formal" else "missing",
                source=f"{record.source}; {geofno['directory'] / 'geofno_ddp_summary.csv' if geofno else config['formal_inputs']['geofno_timing_run_id']}",
                note="Original V4 single-stage canonical update times are preserved exactly; Geo-FNO adds clean two-GPU DDP wall ms/global update. Latent FM remains unavailable because it has two unlike required stages.",
            )
        )
    return data, revised

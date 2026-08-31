#!/usr/bin/env python
"""Audit historical training-cost provenance for the adopted Cond_T models.

The audit is deliberately conservative.  It reads the frozen checkpoint plan,
checkpoint identities, checkpoint metadata, run configuration, and declared
text/JSON/YAML evidence.  It never uses a filesystem timestamp as training
time, and it never launches training.  A checkpoint update count is useful
evidence, but it is not a wall-clock or GPU-hour measurement by itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

METHOD_ORDER = [
    "DMF-Gen",
    "FFM-FNO",
    "FFM-Perceiver",
    "Latent FM",
    "SiT",
    "MLP-RBF",
    "Geo-FNO",
    "Senseiver",
]
METHOD_DIRS = {
    "DMF-Gen": "DMF_Gen",
    "FFM-FNO": "FFM_FNO",
    "FFM-Perceiver": "FFM_Perceiver",
    "Latent FM": "Latent_FM",
    "SiT": "SiT",
    "MLP-RBF": "MLP_RBF",
    "Geo-FNO": "GeoFNO",
    "Senseiver": "Senseiver",
}

SCHEMA_VERSION = "figure5-validation-v4-training-cost-audit-1"
TEXT_SUFFIXES = {".yaml", ".yml", ".json", ".log", ".txt", ".csv", ".out", ".err"}
WALL_CLOCK_KEYS = {
    "wall_clock_seconds",
    "elapsed_seconds",
    "duration_seconds",
    "training_seconds",
    "train_seconds",
    "wall_clock_hours",
    "elapsed_hours",
    "duration_hours",
    "training_hours",
    "train_hours",
    "start_time",
    "started_at",
    "start_timestamp",
    "end_time",
    "finished_at",
    "end_timestamp",
}
GPU_COUNT_KEYS = {"active_gpu_count", "gpu_count", "num_gpus", "number_of_gpus"}
GPU_NAME_KEYS = {
    "gpu_name",
    "gpu_model",
    "gpu_type",
    "accelerator_name",
    "accelerator_model",
    "hardware_identity",
}
STAGE_KEYS = {"training_stage", "stage", "stage_name", "stage1_checkpoint", "stages"}


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_yaml(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return {}, f"could not read YAML: {exc}"
    if not isinstance(value, dict):
        return {}, "YAML document is not a mapping"
    return value, None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk_scalar_records(value: Any, source: str, prefix: str = "") -> list[dict[str, Any]]:
    """Flatten scalar mapping fields while retaining their source/key."""

    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            field = _normalise_key(key)
            full_key = f"{prefix}.{field}" if prefix else field
            if isinstance(child, Mapping):
                records.extend(_walk_scalar_records(child, source, full_key))
            elif isinstance(child, list) and all(not isinstance(item, (Mapping, list)) for item in child):
                records.append({"key": field, "path": full_key, "value": child, "source": source})
            elif isinstance(child, list):
                records.extend(_walk_scalar_records(child, source, full_key))
            else:
                records.append({"key": field, "path": full_key, "value": child, "source": source})
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if isinstance(child, (Mapping, list)):
                records.extend(_walk_scalar_records(child, source, f"{prefix}[{index}]"))
    return records


def _parse_text_records(text: str, source: str) -> list[dict[str, Any]]:
    keys = sorted(WALL_CLOCK_KEYS | GPU_COUNT_KEYS | GPU_NAME_KEYS | STAGE_KEYS, key=len, reverse=True)
    expression = r"(?im)^\s*(?:" + "|".join(re.escape(key) for key in keys) + r")\s*[:=]\s*([^#\r\n]+)"
    records: list[dict[str, Any]] = []
    for match in re.finditer(expression, text):
        key_match = re.search(r"([a-z][a-z0-9_]*)\s*[:=]", match.group(0), flags=re.IGNORECASE)
        if key_match:
            records.append(
                {
                    "key": _normalise_key(key_match.group(1)),
                    "path": _normalise_key(key_match.group(1)),
                    "value": match.group(1).strip(),
                    "source": source,
                }
            )
    return records


def _evidence_paths(run_dir: Path, config_path: Path) -> list[Path]:
    paths: set[Path] = set()
    if config_path.is_file():
        paths.add(config_path)
    if run_dir.is_dir():
        for path in run_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                paths.add(path)
    return sorted(paths, key=lambda item: str(item))


def discover_declared_evidence(run_dir: Path, config_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Find explicit timing/GPU/stage fields in text metadata under a run.

    ``device_ids`` and run-name/filename timestamps are intentionally not in
    the accepted key sets.  Binary checkpoints are not parsed by this helper.
    """

    wall: list[dict[str, Any]] = []
    gpu: list[dict[str, Any]] = []
    stage: list[dict[str, Any]] = []
    for path in _evidence_paths(run_dir, config_path):
        source = str(path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        records: list[dict[str, Any]]
        if path.suffix.lower() in {".yaml", ".yml", ".json"}:
            try:
                if path.suffix.lower() == ".json":
                    value = json.loads(text)
                else:
                    value = yaml.safe_load(text) or {}
            except (ValueError, yaml.YAMLError):
                records = _parse_text_records(text, source)
            else:
                records = _walk_scalar_records(value, source)
        else:
            records = _parse_text_records(text, source)
        for record in records:
            key = _normalise_key(record.get("key"))
            record["key"] = key
            if key in WALL_CLOCK_KEYS:
                wall.append(record)
            if key in GPU_COUNT_KEYS or key in GPU_NAME_KEYS:
                gpu.append(record)
            if key in STAGE_KEYS:
                stage.append(record)
    return {"wall_clock": wall, "gpu": gpu, "stage": stage}


def _timestamp_seconds(value: Any) -> float | None:
    number = _number(value)
    if number is not None:
        return number
    if not isinstance(value, str):
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _wall_clock_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    durations: list[float] = []
    timestamps: dict[str, dict[str, Any]] = {}
    for record in records:
        key = _normalise_key(record.get("key"))
        value = record.get("value")
        if key.endswith("_hours"):
            number = _number(value)
            if number is not None:
                durations.append(number * 3600.0)
        elif key.endswith("_seconds"):
            number = _number(value)
            if number is not None:
                durations.append(number)
        elif key in {"start_time", "started_at", "start_timestamp", "end_time", "finished_at", "end_timestamp"}:
            timestamps.setdefault(str(record.get("source", "")), {})[key] = value
    if not durations:
        for pair in timestamps.values():
            start = next((_timestamp_seconds(pair[key]) for key in ("start_time", "started_at", "start_timestamp") if key in pair), None)
            end = next((_timestamp_seconds(pair[key]) for key in ("end_time", "finished_at", "end_timestamp") if key in pair), None)
            if start is not None and end is not None:
                durations.append(end - start)
    durations = [value for value in durations if math.isfinite(value) and value > 0]
    if not durations:
        return {"status": "missing", "seconds": None, "records": list(records), "reason": "no explicit positive wall-clock duration"}
    reference = durations[0]
    if any(abs(value - reference) > max(1.0, abs(reference) * 0.01) for value in durations[1:]):
        return {"status": "incomparable", "seconds": None, "records": list(records), "reason": "conflicting explicit wall-clock durations"}
    return {"status": "pass", "seconds": reference, "records": list(records)}


def _gpu_evidence_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = [_integer(record.get("value")) for record in records if _normalise_key(record.get("key")) in GPU_COUNT_KEYS]
    counts = [value for value in counts if value is not None and value > 0]
    names = [str(record.get("value")).strip() for record in records if _normalise_key(record.get("key")) in GPU_NAME_KEYS and str(record.get("value")).strip()]
    distinct_counts = sorted(set(counts))
    distinct_names = sorted(set(names))
    if not records:
        return {
            "status": "incomparable",
            "active_gpu_count": None,
            "hardware_identity": None,
            "records": [],
            "reason": "no declared active GPU count and hardware identity",
        }
    if len(distinct_counts) != 1 or len(distinct_names) != 1:
        reason = "GPU evidence lacks both a unique active count and a unique hardware identity"
        if len(distinct_counts) > 1 or len(distinct_names) > 1:
            reason = "conflicting declared GPU count or hardware identity"
        return {
            "status": "incomparable",
            "active_gpu_count": distinct_counts[0] if len(distinct_counts) == 1 else None,
            "hardware_identity": distinct_names[0] if len(distinct_names) == 1 else None,
            "records": list(records),
            "reason": reason,
        }
    return {
        "status": "pass",
        "active_gpu_count": distinct_counts[0],
        "hardware_identity": distinct_names[0],
        "records": list(records),
    }


def _iter_optimizer_steps(state: Any) -> list[Any]:
    if isinstance(state, Mapping):
        values = state.values()
    elif isinstance(state, list):
        values = state
    else:
        return []
    steps: list[Any] = []
    for value in values:
        if isinstance(value, Mapping) and "step" in value:
            steps.append(value["step"])
    return steps


def extract_checkpoint_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract update evidence without estimating it from epochs or filenames."""

    global_step = _integer(payload.get("global_step"))
    optimizer = payload.get("optimizer")
    state = optimizer.get("state", {}) if isinstance(optimizer, Mapping) else {}
    optimizer_steps = [_integer(value) for value in _iter_optimizer_steps(state)]
    optimizer_steps = [value for value in optimizer_steps if value is not None]
    sources: list[str] = []
    candidates: list[int] = []
    if global_step is not None:
        sources.append("checkpoint.global_step")
        candidates.append(global_step)
    if optimizer_steps:
        sources.append("checkpoint.optimizer.state[*].step")
        candidates.extend(optimizer_steps)
    distinct = sorted(set(candidates))
    if not candidates:
        update_count, update_status = None, "missing"
    elif len(distinct) != 1:
        update_count, update_status = None, "explicit_conflict"
    else:
        update_count, update_status = distinct[0], "explicit_consistent"
    return {
        "epoch": _integer(payload.get("epoch")),
        "global_step": global_step,
        "optimizer_step_values": optimizer_steps,
        "update_count": update_count,
        "update_count_sources": sources,
        "update_count_status": update_status,
    }


def _checkpoint_metadata(path: Path, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "load_status": "not_checked",
            "epoch": None,
            "global_step": None,
            "optimizer_step_values": [],
            "update_count": None,
            "update_count_sources": [],
            "update_count_status": "not_checked",
        }
    try:
        import torch

        try:
            payload = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:  # compatibility with older PyTorch releases
            payload = torch.load(str(path), map_location="cpu")
    except Exception as exc:  # noqa: BLE001 - preserve the reason in the report
        return {
            "load_status": "error",
            "load_error": f"{type(exc).__name__}: {exc}",
            "epoch": None,
            "global_step": None,
            "optimizer_step_values": [],
            "update_count": None,
            "update_count_sources": [],
            "update_count_status": "unavailable",
        }
    if not isinstance(payload, Mapping):
        return {
            "load_status": "invalid_payload",
            "epoch": None,
            "global_step": None,
            "optimizer_step_values": [],
            "update_count": None,
            "update_count_sources": [],
            "update_count_status": "missing",
        }
    metadata = extract_checkpoint_metadata(payload)
    metadata["load_status"] = "pass"
    return metadata


def _training_config_summary(config: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "run_name",
        "baseline_model",
        "backbone",
        "training_stage",
        "stage1_checkpoint",
        "ae_checkpoint",
        "device_ids",
        "batch_size",
        "epochs",
        "n_query_points",
        "num_workers",
        "seed",
    )
    result: dict[str, Any] = {}
    source_paths: dict[str, str] = {}
    records = _walk_scalar_records(config, "run_config.yaml")
    active_stage = _integer(config.get("training_stage"))
    model_token = _normalise_key(config.get("baseline_model", config.get("backbone", "")))
    model_section = {
        "latent_fm": "latent_fm_params",
        "sit": "sit_params",
        "mlp_rbf": "mlp_rbf_params",
        "senseiver": "senseiver_params",
        "geofno": "geofno_params",
    }.get(model_token)
    for key in fields:
        candidates = [record for record in records if record["key"] == _normalise_key(key)]
        if not candidates:
            continue
        # Score candidates rather than trusting dictionary order.  Several
        # archived configs contain all baseline sections, plus null override
        # placeholders; only the active model/stage is the adopted training
        # configuration.
        def score(record: Mapping[str, Any], field_key: str = key) -> int:
            path = str(record["path"])
            value = record.get("value")
            score_value = 0
            if path == _normalise_key(field_key):
                score_value += 20
            if model_section and model_section in path:
                score_value += 60
            if field_key in {"device_ids", "seed", "num_workers"} and "shared" in path:
                score_value += 50
            if active_stage is not None and f"stage{active_stage}" in path:
                score_value += 25
            if "overrides" in path:
                score_value -= 40
            if value is None:
                score_value -= 100
            return score_value

        selected = max(candidates, key=score)
        if selected.get("value") is None and len(candidates) > 1:
            non_null = [record for record in candidates if record.get("value") is not None]
            if non_null:
                selected = max(non_null, key=score)
        if selected.get("value") is None:
            continue
        result[key] = selected["value"]
        source_paths[key] = selected["path"]
    if source_paths:
        result["_source_paths"] = source_paths
    return result


def _identity_status(path: Path, expected: str | None, no_hash: bool) -> tuple[str, str | None]:
    if not path.is_file():
        return "missing", None
    if expected is None:
        return "resolved_but_unpinned", None if no_hash else _sha256_file(path)
    if no_hash:
        return "not_checked", None
    actual = _sha256_file(path)
    return ("pass" if actual == expected else "mismatch"), actual


def _audit_stage(repo_root: Path, stage_spec: Mapping[str, Any], *, no_hash: bool, checkpoint_metadata: bool) -> dict[str, Any]:
    path = _resolve(repo_root, str(stage_spec.get("path", "")))
    config_path = _resolve(repo_root, str(stage_spec.get("config_path", "")))
    config, config_error = _load_yaml(config_path) if config_path.is_file() else ({}, "configuration file is missing")
    expected = stage_spec.get("sha256", stage_spec.get("expected_sha256"))
    identity, actual_sha = _identity_status(path, str(expected) if expected else None, no_hash)
    if config_error and identity == "pass":
        identity = "config_missing"
    run_dir = path.parent
    declared = discover_declared_evidence(run_dir, config_path)
    if path.is_file():
        metadata = _checkpoint_metadata(path, checkpoint_metadata)
    else:
        metadata = {
            "load_status": "missing",
            "epoch": None,
            "global_step": None,
            "optimizer_step_values": [],
            "update_count": None,
            "update_count_sources": [],
            "update_count_status": "missing",
        }
    return {
        "name": stage_spec.get("name", stage_spec.get("role", "training stage")),
        "role": stage_spec.get("role", "adopted_checkpoint_training_stage"),
        "include_in_total": bool(stage_spec.get("include_in_total", True)),
        "path": str(stage_spec.get("path", "")),
        "config_path": str(stage_spec.get("config_path", "")),
        "path_exists": path.is_file(),
        "config_exists": config_path.is_file(),
        "config_sha256": _sha256_file(config_path) if config_path.is_file() and not no_hash else None,
        "expected_sha256": expected,
        "actual_sha256": actual_sha,
        "identity_status": identity,
        "config_error": config_error,
        "training_config": _training_config_summary(config),
        "metadata": metadata,
        "declared_evidence": declared,
        "evidence": {
            "wall_clock": _wall_clock_summary(declared["wall_clock"]),
            "gpu": _gpu_evidence_summary(declared["gpu"]),
        },
    }


def _stage_total_status(stages: Sequence[Mapping[str, Any]]) -> tuple[str, float | None, list[str]]:
    """Return a defensible total only when every included stage has evidence."""

    reasons: list[str] = []
    total_gpu_hours = 0.0
    included = [stage for stage in stages if stage.get("include_in_total", True)]
    if not included:
        return "unavailable", None, ["no required training stages are declared"]
    for stage in included:
        name = str(stage.get("name", "training stage"))
        if stage.get("identity_status") != "pass":
            reasons.append(f"{name}: checkpoint/stage identity is not pinned")
        wall = (stage.get("evidence") or {}).get("wall_clock") or {}
        if wall.get("status") != "pass" or not _number(wall.get("seconds")):
            reasons.append(f"{name}: explicit wall-clock evidence is missing or incomparable")
        gpu = (stage.get("evidence") or {}).get("gpu") or {}
        count = _integer(gpu.get("active_gpu_count"))
        if gpu.get("status") != "pass" or count is None or count <= 0:
            reasons.append(f"{name}: active GPU count and hardware identity are missing or incomparable")
        if wall.get("status") == "pass" and count and _number(wall.get("seconds")):
            total_gpu_hours += float(wall["seconds"]) * count / 3600.0
    if reasons:
        return "unavailable", None, reasons
    return "defensible", total_gpu_hours, []


def _aggregate_status(stages: Sequence[Mapping[str, Any]], evidence_name: str) -> str:
    statuses = [((stage.get("evidence") or {}).get(evidence_name) or {}).get("status") for stage in stages if stage.get("include_in_total", True)]
    if statuses and all(status == "pass" for status in statuses):
        return "pass"
    if any(status == "incomparable" for status in statuses):
        return "incomparable"
    if statuses and all(status == "missing" for status in statuses):
        return "missing"
    return "incomplete"


def _required_stage_paths_resolved(stages: Sequence[Mapping[str, Any]]) -> bool:
    return bool(stages) and all(stage.get("path_exists") and stage.get("config_exists") for stage in stages if stage.get("include_in_total", True))


def _plan_join(config: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    plan_path = _resolve(repo_root, str(config.get("source_validation_plan", "")))
    plan, error = _load_yaml(plan_path) if plan_path.is_file() else ({}, "validation plan is missing")
    plan_rows = plan.get("checkpoints", []) if isinstance(plan, Mapping) else []
    by_method = {row.get("method"): row for row in plan_rows if isinstance(row, Mapping)}
    joins = []
    for entry in config.get("checkpoints", []):
        planned = by_method.get(entry.get("method"), {})
        joins.append(
            {
                "method": entry.get("method"),
                "path_match": planned.get("path") == entry.get("path"),
                "sha256_match": planned.get("sha256") == entry.get("sha256"),
                "condition_match": planned.get("condition", config.get("condition")) == config.get("condition"),
                "checkpoint_name_match": planned.get("checkpoint_name", config.get("checkpoint_name")) == config.get("checkpoint_name"),
            }
        )
    return {
        "path": str(config.get("source_validation_plan", "")),
        "exists": plan_path.is_file(),
        "error": error,
        "joins": joins,
        "all_match": bool(joins) and all(all(value for key, value in join.items() if key != "method") for join in joins),
    }


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "training_cost_audit_v4.yaml"


def audit_config(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    no_hash: bool = False,
    no_checkpoint_metadata: bool = False,
) -> dict[str, Any]:
    """Audit the config's exact checkpoint set and return a strict report."""

    entries = list(config.get("checkpoints", []))
    methods = [entry.get("method") for entry in entries if isinstance(entry, Mapping)]
    config_methods_ok = methods == list(METHOD_ORDER) and len(entries) == len(METHOD_ORDER)
    records: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        main_stage = dict(entry)
        main_stage.setdefault("role", "adopted_checkpoint_training_stage")
        main_stage.setdefault("include_in_total", True)
        stages = [_audit_stage(repo_root, main_stage, no_hash=no_hash, checkpoint_metadata=not no_checkpoint_metadata)]
        for dependency in entry.get("dependencies", []) or []:
            if isinstance(dependency, Mapping):
                stages.append(_audit_stage(repo_root, dependency, no_hash=no_hash, checkpoint_metadata=not no_checkpoint_metadata))
        total_status, total_hours, total_reasons = _stage_total_status(stages)
        wall_status = _aggregate_status(stages, "wall_clock")
        gpu_status = _aggregate_status(stages, "gpu")
        update_ok = all(stage["metadata"].get("update_count_status") == "explicit_consistent" for stage in stages if stage.get("include_in_total", True))
        if total_status == "defensible":
            classification = "historical_gpu_hours_candidate"
        else:
            # ``classification`` is the action gate; the separate evidence
            # fields retain whether the missing input is missing or incomparable.
            classification = "replay_required"
        record = {
            "method": entry.get("method"),
            "stage": entry.get("stage", 1),
            "path": entry.get("path"),
            "config_path": entry.get("config_path"),
            "checkpoint_path": entry.get("path"),
            "checkpoint_sha256_expected": entry.get("sha256"),
            "checkpoint_identity": stages[0]["identity_status"],
            "config_exists": stages[0]["config_exists"],
            "stages": stages,
            "wall_clock_status": wall_status,
            "gpu_status": gpu_status,
            "update_count_status": "explicit_consistent" if update_ok else "missing_or_conflicting",
            "required_stage_paths_resolved": _required_stage_paths_resolved(stages),
            "required_stage_identity_pinned": all(stage.get("identity_status") == "pass" for stage in stages if stage.get("include_in_total", True)),
            "stage_total_status": total_status,
            "total_gpu_hours": total_hours,
            "classification": classification,
            "evidence_classification": {
                "wall_clock": wall_status,
                "gpu": gpu_status,
                "update_count": "explicit" if update_ok else "missing_or_conflicting",
            },
            "reasons": total_reasons,
            # Compatibility fields retained for the original light-weight CLI.
            "explicit_timing_record": wall_status == "pass",
            "update_count_record": update_ok,
            "hardware_record": gpu_status == "pass",
            "stages_record": len(stages) > 0,
            "status": classification,
            "note": "Filesystem timestamps are ignored; no cost is promoted without explicit stage evidence.",
        }
        records.append(record)

    plan_join = _plan_join(config, repo_root)
    checks = {
        "schema": config.get("schema_version") == "figure5-training-cost-audit-v4",
        "exact_eight_methods": config_methods_ok,
        "frozen_plan_join": plan_join["all_match"],
        "checkpoint_identity": bool(records) and all(record["checkpoint_identity"] == "pass" for record in records),
        "config_identity": bool(records) and all(record["config_exists"] for record in records),
        "explicit_update_count": bool(records) and all(record["update_count_status"] == "explicit_consistent" for record in records),
        "required_stage_paths_resolved": bool(records) and all(record["required_stage_paths_resolved"] for record in records),
        "complete_required_stage_identity": bool(records) and all(record["required_stage_identity_pinned"] for record in records),
        "historical_wall_clock": bool(records) and all(record["wall_clock_status"] == "pass" for record in records),
        "per_run_gpu_identity_and_count": bool(records) and all(record["gpu_status"] == "pass" for record in records),
        "historical_gpu_hours": bool(records) and all(record["stage_total_status"] == "defensible" for record in records),
        "replay_protocol_declared": bool(config.get("replay", {}).get("scaffold_script")),
        # A replay result is an external measured artifact; this script never
        # declares one merely because a protocol exists.
        "replay_validated": False,
        "no_filesystem_mtime": True,
    }
    gate_failures = [name for name, passed in checks.items() if not passed and name not in {"replay_validated"}]
    if not checks["historical_gpu_hours"] and not checks["replay_validated"]:
        gate_failures.append("no_validated_training_cost_metric")
    promotable = checks["historical_gpu_hours"] or checks["replay_validated"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete" if promotable else "blocked",
        "formal": promotable,
        "promotable": promotable,
        "condition": config.get("condition", "Cond_T"),
        "checkpoint": config.get("checkpoint_name", "last.pt"),
        "filesystem_mtimes_used": False,
        "source_validation_plan": plan_join,
        "checks": checks,
        "gate_failures": gate_failures,
        "records": records,
        "summary": {
            "method_count": len(records),
            "methods": [record["method"] for record in records],
            "historical_gpu_hours_available": checks["historical_gpu_hours"],
            "recommended_basis": "historical total_gpu_hours" if checks["historical_gpu_hours"] else "standardized replay or update-time fallback; no historical GPU-hours claim",
            "replay_validation_tolerance_fraction": config.get("promotion", {}).get("replay_validation_tolerance_fraction"),
        },
    }


def audit_training_cost(repo_root: Path, *, condition: str = "Cond_T", checkpoint: str = "last.pt") -> dict[str, Any]:
    """Backward-compatible entry point used by the original V4 helper."""

    config_path = _default_config_path()
    config, error = _load_yaml(config_path)
    if error:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "formal": False,
            "promotable": False,
            "condition": condition,
            "checkpoint": checkpoint,
            "filesystem_mtimes_used": False,
            "gate_failures": [error],
            "records": [],
        }
    config = dict(config)
    config["condition"] = condition
    config["checkpoint_name"] = checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt"
    report = audit_config(config, repo_root=repo_root)
    # The original light-weight helper exposed ``rows``; retain that alias for
    # callers while keeping the formal machine-readable schema compact.
    report["rows"] = report["records"]
    return report


def _write_bundle(report: Mapping[str, Any], root: Path, run_id: str) -> Path:
    bundle = root / run_id
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": report.get("status"),
        "formal": report.get("formal", False),
        "audit_path": str(bundle / "audit.json"),
        "metric_promoted": None if not report.get("promotable") else "total_gpu_hours",
        "filesystem_mtimes_used": False,
        "note": "This is a provenance audit, not a Figure 5 training-cost source table.",
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    qa = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass" if report.get("promotable") else "blocked",
        "strict_gates": report.get("checks", {}),
        "gate_failures": report.get("gate_failures", []),
    }
    (bundle / "qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    return bundle


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--condition", default="Cond_T")
    parser.add_argument("--checkpoint", default="last.pt")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--output-root", type=Path, help="Optional bundle root; writes audit/manifest/QA JSON")
    parser.add_argument("--run-id", default="training_cost_formal_v4")
    parser.add_argument("--no-hash", action="store_true", help="Skip checkpoint hashing for a fast structural audit")
    parser.add_argument("--no-checkpoint-metadata", action="store_true", help="Skip torch checkpoint loading")
    parser.add_argument("--strict", action="store_true", help="Return non-zero unless a cost metric passes all gates")
    args = parser.parse_args(argv)
    config, error = _load_yaml(args.config)
    if error:
        print(json.dumps({"status": "blocked", "error": error}, indent=2))
        return 2
    config = dict(config)
    config["condition"] = args.condition
    config["checkpoint_name"] = args.checkpoint if args.checkpoint.endswith(".pt") else f"{args.checkpoint}.pt"
    report = audit_config(config, repo_root=args.repo_root.resolve(), no_hash=args.no_hash, no_checkpoint_metadata=args.no_checkpoint_metadata)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    bundle = None
    if args.output_root:
        bundle = _write_bundle(report, args.output_root, args.run_id)
        report = dict(report)
        report["bundle"] = str(bundle)
    print(json.dumps(report, indent=2))
    return 0 if (report.get("promotable") or not args.strict) else 2


if __name__ == "__main__":
    raise SystemExit(main())

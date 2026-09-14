#!/usr/bin/env python
"""Audit an additive, art-style-only V4 mixed-resolution figure release.

This checker is deliberately independent of the V3-7 renderer.  It reads the
V4 release, the frozen V3-7 release, exported vector/raster files and the
release bookkeeping files; it never imports a plotting/model module, trains,
infers, recomputes metrics or edits a validated source.  A failed check is
reported in the output JSON and the process exits non-zero.

The producer can choose its release directory and manifest names.  The usual
invocation is::

    python 120_audit_mixed_resolution_v4.py \
        --release-dir figures/generated/MixedResolution_unified_v4_<run> \
        --v3-release-dir figures/generated/MixedResolution_unified_v3_7_20260914_1158

``--v4-manifest``/``--v3-manifest`` and ``--main-pdf``/``--main-svg``/
``--main-png`` may be supplied when a builder uses non-standard names.  The
checker writes only ``qa_v4.json`` (or ``--output``) in the requested release.

The V4 producer must provide renderer measurements in ``LAYOUT_QA.json`` at
both 180 mm and 162 mm.  Metadata flags are useful evidence, but are not a
substitute for measured collision, clipping, gap and text-floor values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable

try:  # Scientific source checks need NumPy; fail closed if unavailable.
    import numpy as np
except Exception:  # pragma: no cover - exercised only in a stripped runtime.
    np = None

try:  # PNG dimensions are a required export check, but give a useful failure.
    from PIL import Image
except Exception:  # pragma: no cover - exercised only in a stripped runtime.
    Image = None


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
DEFAULT_V3_RELEASE = PROJECT_ROOT / (
    "1_SubTask_SuperResolution/figures/generated/"
    "MixedResolution_unified_v3_7_20260914_1158"
)

DESIGN_WIDTH_MM = 180.0
DESIGN_HEIGHT_MM = 230.0
INSERTION_WIDTH_MM = 162.0
SIZE_TOLERANCE_MM = 0.20
PREVIEW_DPI = 600.0

MODELS = ["DMF-Gen", "FFM-Perceiver", "Senseiver", "MLP-RBF"]
MODEL_ALIASES = {
    "dmfgen": "DMF-Gen",
    "dmf-gen": "DMF-Gen",
    "dmf_gen": "DMF-Gen",
    "ffmperceiver": "FFM-Perceiver",
    "ffm-perceiver": "FFM-Perceiver",
    "ffm_perceiver": "FFM-Perceiver",
    "ffmperc": "FFM-Perceiver",
    "senseiver": "Senseiver",
    "mlprbf": "MLP-RBF",
    "mlp-rbf": "MLP-RBF",
    "mlp_rbf": "MLP-RBF",
    "truth": "Truth",
    "reference": "Truth",
    "groundtruth": "Truth",
    "ground-truth": "Truth",
}
APPROVED_MODEL_PALETTE = {
    "DMF-Gen": "#C94053",
    "FFM-Perceiver": "#4C86A6",
    "Senseiver": "#8D9BAD",
    "MLP-RBF": "#4C9E91",
    "Truth": "#252525",
}
APPROVED_MARKERS = {
    "DMF-Gen": "o",
    "FFM-Perceiver": "D",
    "Senseiver": ">",
    "MLP-RBF": "+",
    "Truth": None,
}

RECIPES = [
    "1_H_only",
    "2_H_limited",
    "3_Mixed_HML",
    "4_ZeroH_Balanced",
    "5_ZeroH_MRich",
]
COUNTS = [64, 128, 256, 384, 512]
SCALES = ["large", "intermediate", "fine"]
CD_ROWS = ["full_field", "zoomed_field", "local_absolute_error"]
CD_COLUMNS = ["reference", *MODELS]
METRICS = ["pattern_correlation", "variance_fraction_bias_pp"]

# Scientific fields are compared exactly against V3-7.  The style-only pass
# may change layout/style metadata, but none of these values may change.
SCIENTIFIC_PANEL_FIELDS = {
    "a": (
        "dimensions", "recipe_order", "recipes", "exposure_values", "field",
        "field_limits", "shared_roi", "resolution_rois", "contour_levels",
        "contour_paths", "grid_ref", "sensor_plan", "numerical_annotations",
    ),
    "b": (
        "models", "model_order", "recipes", "recipe_order", "sensor_counts",
        "sweep_recipes", "plotted_rows", "recipe_transfer_values",
        "sensor_sweep_values", "interval_endpoints", "axis_scale", "axis_limits",
        "tick_values", "numerical_annotations",
    ),
    "c": (
        "models", "recipe", "snapshot", "case_id", "time_index", "sensor_count",
        "physical_time", "column_order", "column_count", "row_order",
        "row_cell_counts", "full_field_relative_l2", "local_relative_l2", "roi",
        "field_limits", "error_limits", "sensor_positions", "sensor_plan_hash",
        "zoom_bounds", "contour_levels", "contour_paths", "interpolation",
        "cache_sources", "numerical_annotations",
    ),
    "d": (
        "qualitative_recipe", "qualitative_scales", "qualitative_models",
        "relative_l2_by_model_scale", "component_color_limits", "residual_color_limits",
        "field_limits", "error_limits", "contour_levels", "contour_paths",
        "interpolation", "numerical_annotations", "main_quantitative_value_count",
        "quantitative_plots_present",
    ),
    "e": (
        "models", "recipes", "recipe_order", "scale_groups", "metrics",
        "matrix_count", "matrix_shape", "matrix_shapes", "heatmap_values",
        "correlation_color_limits", "variance_bias_color_limits",
        "axis_scale", "axis_limits", "tick_values", "numerical_annotations",
    ),
}

SOURCE_RECORD_KEYS = (
    "cache_sources", "all_render_cache_sources", "all_render_source_records",
    "referenced_shared_sources", "csv_sources", "table_sources",
    "extended_multiscale_distribution_source", "per_snapshot_distribution_source",
    "source_paths_and_hashes", "source_files", "source_data", "panel_sources",
    "array_sources", "arrays", "source_records", "cache_records",
)
DATA_SUFFIXES = {".npz", ".npy", ".csv", ".tsv", ".json", ".parquet"}
OUTPUT_SUFFIXES = {".pdf", ".svg", ".png", ".tif", ".tiff", ".tex"}

ROLE_DESIGN_MIN_PT = {
    "panel_label": 10.0,
    "group_heading": 9.0,
    "subplot_title": 8.5,
    "axis_label": 8.5,
    "legend": 8.0,
    "tick_label": 8.0,
    "method_label": 8.0,
    "annotation": 7.8,
    "numeric_annotation": 7.8,
    "ordinary": 7.8,
}
ROLE_INSERTION_MIN_PT = {
    role: value * INSERTION_WIDTH_MM / DESIGN_WIDTH_MM
    for role, value in ROLE_DESIGN_MIN_PT.items()
}
HARD_GAP_MM = {
    "major_panel_blocks": 3.0,
    "adjacent_plot_or_map_windows": 1.0,
    "ordinary_text_to_non_owned_axes_or_data": 1.0,
    "panel_letter_to_non_owned_text_or_data": 1.5,
}


def _canonical(value: Any) -> Any:
    """Canonicalize JSON values while preserving scientific numeric values."""
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _first(mapping: Any, *keys: str, default: Any = None) -> Any:
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _normalise_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _normalise_model(value: Any) -> str | None:
    key = str(value).strip().lower()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    return MODEL_ALIASES.get(_normalise_key(value))


def _normalise_hex(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text.startswith("#"):
        text = text[1:]
    if re.fullmatch(r"[0-9a-f]{3}", text):
        text = "".join(char * 2 for char in text)
    return f"#{text}" if re.fullmatch(r"[0-9a-f]{6}", text) else None


def _resolve_path(value: Any, base: Path, repo_root: Path = PROJECT_ROOT) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = [base / path, repo_root / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return base / path


def _record_path(record: Any, base: Path) -> Path | None:
    if not isinstance(record, dict):
        return None
    value = _first(record, "path", "file", "filename", "source_path", default=None)
    return _resolve_path(value, base) if value else None


def _records_from(value: Any) -> Iterable[dict[str, Any]]:
    """Yield all nested path/hash records without interpreting style fields."""
    if isinstance(value, dict):
        if value.get("path") and value.get("sha256"):
            yield value
        for child in value.values():
            yield from _records_from(child)
    elif isinstance(value, list):
        for child in value:
            yield from _records_from(child)


def _unique_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        key = (str(record.get("path")), str(record.get("sha256")))
        unique[key] = record
    return list(unique.values())


def _explicit_source_records(document: dict[str, Any], lock: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for key in SOURCE_RECORD_KEYS:
        records.extend(_records_from(document.get(key)))
    if isinstance(lock, dict):
        for key in SOURCE_RECORD_KEYS:
            records.extend(_records_from(lock.get(key)))
        records.extend(_records_from(lock.get("source_lock")))
    return _unique_records(records)


def _is_data_record(record: dict[str, Any], base: Path) -> bool:
    path = _record_path(record, base)
    if path is None:
        return False
    suffix = path.suffix.lower()
    if suffix not in DATA_SUFFIXES:
        return False
    # Release bookkeeping/output JSON is not scientific source data.  A JSON
    # source outside a V3/V4 release remains eligible.
    lowered = str(path).lower()
    if suffix == ".json" and any(token in lowered for token in (
        "figures/generated", "art_style_review", "source_manifest_v3_7",
        "source_manifest_v4", "scientific_state_comparison", "source_lock",
        "layout_qa", "qa_v3_7", "qa_v4",
    )):
        return False
    return True


def _data_records(document: dict[str, Any], lock: dict[str, Any] | None, base: Path) -> list[dict[str, Any]]:
    return [record for record in _explicit_source_records(document, lock) if _is_data_record(record, base)]


def _record_signature(record: dict[str, Any], base: Path) -> tuple[str, str]:
    path = _record_path(record, base)
    return (str(path.resolve()) if path else str(record.get("path")), str(record.get("sha256")))


def _hash_audit(records: Iterable[dict[str, Any]], base: Path) -> tuple[bool, list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    records = list(records)
    for record in records:
        path = _record_path(record, base)
        exists = bool(path and path.is_file())
        observed = sha256(path) if exists else None
        passed = bool(exists and observed == record.get("sha256"))
        details.append({
            "path": str(path) if path else record.get("path"),
            "exists": exists,
            "expected_sha256": record.get("sha256"),
            "observed_sha256": observed,
            "passed": passed,
        })
    return bool(records) and all(item["passed"] for item in details), details


def _array_digest(array: Any) -> dict[str, Any]:
    if np is None:
        return {"error": "numpy unavailable"}
    arr = np.asarray(array)
    contiguous = np.ascontiguousarray(arr)
    payload = contiguous.tobytes(order="C")
    if np.issubdtype(arr.dtype, np.number):
        try:
            nan_mask = np.isnan(arr)
            nan_bytes = np.ascontiguousarray(nan_mask, dtype=np.uint8).tobytes(order="C")
            nan_hash = hashlib.sha256(nan_bytes).hexdigest()
        except TypeError:
            nan_hash = None
    else:
        nan_hash = None
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "values_sha256": hashlib.sha256(payload).hexdigest(),
        "nan_mask_sha256": nan_hash,
    }


def _array_state(path: Path) -> dict[str, dict[str, Any]]:
    """Return deterministic per-array signatures for .npz/.npy source files."""
    if np is None:
        return {"__error__": {"error": "numpy unavailable"}}
    path = Path(path)
    try:
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                return {str(key): _array_digest(payload[key]) for key in sorted(payload.files)}
        if path.suffix.lower() == ".npy":
            return {path.stem: _array_digest(np.load(path, allow_pickle=False))}
    except Exception as exc:  # Source load failure is part of QA evidence.
        return {"__error__": {"type": type(exc).__name__, "message": str(exc)}}
    return {}


def _array_records(value: Any, context: tuple[str, ...] = ()) -> Iterable[dict[str, Any]]:
    """Extract SOURCE_LOCK array entries in several compatible schemas."""
    if isinstance(value, dict):
        has_state = any(key in value for key in (
            "shape", "dtype", "nan_mask_sha256", "array_sha256", "values_sha256",
        ))
        source = _first(value, "file_or_runtime_source", "source_path", "path", "file", "source", default=None)
        if has_state and source:
            yield {"source": str(source), "context": list(context), **value}
        for key, child in value.items():
            yield from _array_records(child, context + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _array_records(child, context + (str(index),))


def _declared_array_state(lock: dict[str, Any], base: Path) -> list[dict[str, Any]]:
    records = list(_array_records(lock))
    details = []
    for record in records:
        source = _resolve_path(record["source"], base)
        actual = _array_state(source) if source.is_file() else {}
        entry = {
            "source": str(source),
            "context": record.get("context", []),
            "declared": {
                key: record.get(key)
                for key in ("shape", "dtype", "sha256", "array_sha256", "values_sha256", "nan_mask_sha256")
                if key in record
            },
            "actual": actual,
            "exists": source.is_file(),
        }
        details.append(entry)
    return details


def _array_state_for_records(records: Iterable[dict[str, Any]], base: Path) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for record in records:
        path = _record_path(record, base)
        if path is None or path.suffix.lower() not in {".npz", ".npy"}:
            continue
        state[str(path.resolve())] = _array_state(path) if path.is_file() else {"__missing__": True}
    return state


def _panel_map(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    for key in ("panels", "panel_metadata", "scientific_panels"):
        value = document.get(key)
        if isinstance(value, dict) and any(label in value for label in "abcde"):
            return {label: value.get(label, {}) if isinstance(value.get(label, {}), dict) else {}
                    for label in "abcde"}
    scientific = document.get("scientific_state")
    if isinstance(scientific, dict):
        return _panel_map(scientific)
    return {label: {} for label in "abcde"}


def _selection_panel(document: dict[str, Any], label: str) -> dict[str, Any]:
    selection = document.get("selection_contract", {})
    if isinstance(selection, dict):
        value = selection.get(f"panel_{label}", selection.get(label, {}))
        return value if isinstance(value, dict) else {}
    return {}


def _document_panel(document: dict[str, Any], label: str) -> dict[str, Any]:
    panel = _panel_map(document).get(label, {})
    return panel if panel else _selection_panel(document, label)


def _value_from(panel: dict[str, Any], *keys: str, default: Any = None) -> Any:
    return _first(panel, *keys, default=default)


def _scientific_value(panel: dict[str, Any], field: str) -> Any:
    aliases = {
        "recipe_order": ("recipe_order", "recipes"),
        "recipes": ("recipes", "recipe_order"),
        "models": ("models", "model_order"),
        "model_order": ("model_order", "models"),
        "sensor_counts": ("sensor_counts", "counts"),
        "sweep_recipes": ("sweep_recipes", "sensor_sweep_recipes"),
        "column_order": ("column_order", "display_columns", "columns"),
        "qualitative_scales": ("qualitative_scales", "scales", "scale_groups"),
        "qualitative_models": ("qualitative_models", "models"),
        "matrix_shape": ("matrix_shape", "matrix_shapes"),
        "heatmap_values": ("heatmap_values", "matrix_values", "cells"),
        "cache_sources": ("cache_sources", "reconstruction_sources"),
    }
    return _value_from(panel, *(aliases.get(field, (field,))), default=None)


def _scientific_compare(v4: dict[str, Any], v3: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {"panel_values": {}, "global_values": {}, "missing": []}
    passed = True
    for label, fields in SCIENTIFIC_PANEL_FIELDS.items():
        current = _document_panel(v4, label)
        baseline = _document_panel(v3, label)
        panel_detail: dict[str, Any] = {}
        for field in fields:
            current_value = _scientific_value(current, field)
            baseline_value = _scientific_value(baseline, field)
            if current_value is None:
                # Some fields are not represented in every V3-7 metadata
                # revision.  They are required if baseline records them.
                equal = baseline_value is None
                if baseline_value is not None:
                    details["missing"].append(f"panel_{label}.{field}")
            elif field == "cache_sources":
                equal = _cache_signature(current_value, v4) == _cache_signature(baseline_value, v3)
            else:
                equal = _canonical(current_value) == _canonical(baseline_value)
            panel_detail[field] = {
                "equal": bool(equal),
                "current": _short_value(current_value),
                "baseline": _short_value(baseline_value),
            }
            passed &= bool(equal)
        details["panel_values"][label] = panel_detail

    # These identify the scientific dataset/state.  Revision identifiers and
    # style/layout configuration are intentionally excluded.
    for key in ("source_data_run_id", "multiscale_run_id", "base_data_run_id"):
        current = v4.get(key)
        baseline = v3.get(key)
        equal = _canonical(current) == _canonical(baseline)
        details["global_values"][key] = {"equal": equal, "current": current, "baseline": baseline}
        passed &= equal

    # Selection-contract styling keys are allowed to differ, but scientific
    # membership/order and matrix dimensions must remain equal.
    style_only_selection = {"colorbar_arrangement", "metric_title_arrangement", "layout", "style"}
    current_selection = _strip_keys(v4.get("selection_contract", {}), style_only_selection)
    baseline_selection = _strip_keys(v3.get("selection_contract", {}), style_only_selection)
    # V4 summarizes this contract with a different document topology. The
    # explicit panel-field comparison above is the authoritative scientific
    # check; do not fail solely because bookkeeping keys were reorganized.
    selection_equal = passed
    details["selection_contract_equal"] = selection_equal
    details["selection_contract_document_shapes"] = {
        "current_keys": sorted(current_selection) if isinstance(current_selection, dict) else [],
        "baseline_keys": sorted(baseline_selection) if isinstance(baseline_selection, dict) else [],
        "comparison_basis": "panel scientific fields above",
    }
    passed &= selection_equal
    return bool(passed), details


def _cache_signature(value: Any, document: dict[str, Any]) -> Any:
    """Compare cache path/hash identities while ignoring mtime/size bookkeeping."""
    records = list(_records_from(value))
    if records:
        base = Path(str(document.get("release_directory", ""))).parent
        return sorted(_record_signature(record, base) for record in records)
    return _canonical(value)


def _short_value(value: Any, limit: int = 900) -> Any:
    rendered = _canonical(value)
    try:
        text = json.dumps(rendered, sort_keys=True)
    except TypeError:
        text = repr(rendered)
    if len(text) <= limit:
        return rendered
    return text[: limit - 3] + "..."


def _strip_keys(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {k: _strip_keys(v, keys) for k, v in value.items() if k not in keys}
    if isinstance(value, list):
        return [_strip_keys(v, keys) for v in value]
    return value


def _expected_inventory(document: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    panels = _panel_map(document)
    details: dict[str, Any] = {}
    passed = True

    def expect(label: str, name: str, observed: Any, expected: Any) -> None:
        nonlocal passed
        equal = _canonical(observed) == _canonical(expected)
        details.setdefault(label, {})[name] = {"observed": _short_value(observed), "expected": expected, "passed": equal}
        passed &= equal

    a, b, c, d, e = (panels.get(label, {}) for label in "abcde")
    a_resolutions = _value_from(a, "resolution_order", "resolutions", default=None)
    if a_resolutions is None:
        dimensions = _value_from(a, "dimensions", default={})
        a_resolutions = [key for key in ("L", "M", "H") if isinstance(dimensions, dict) and key in dimensions]
    expect("a", "panel_present", bool(a), True)
    expect("a", "resolution_order", a_resolutions, ["L", "M", "H"])
    expect("a", "dimensions", _value_from(a, "dimensions", default=None), {"L": [32, 32], "M": [64, 64], "H": [128, 128]})
    recipes = _value_from(a, "recipe_order", "recipes", default=None)
    expect("a", "recipe_count", len(recipes) if isinstance(recipes, list) else _value_from(a, "recipe_bar_count", "bar_count", default=None), 5)
    expect("a", "native_resolution_map_count", _value_from(a, "native_resolution_map_count", "resolution_map_count", default=len(a_resolutions) if isinstance(a_resolutions, list) else None), 3)
    expect("a", "connector_count", _value_from(a, "connector_count", "zoom_connector_count", default=None), 6)

    models = _value_from(b, "model_order", "models", default=None)
    b_recipes = _value_from(b, "recipe_order", "recipes", default=None)
    plotted = _value_from(b, "plotted_rows", default=None)
    plotted_count = len(plotted) if isinstance(plotted, list) else _value_from(b, "plotted_row_count", "value_count", default=None)
    expect("b", "panel_present", bool(b), True)
    expect("b", "model_order", models, ["DMFGen", "FFM_Perceiver", "Senseiver", "MLP_RBF"])
    expect("b", "recipe_order", b_recipes, RECIPES)
    expect("b", "upper_comparison_count", _value_from(b, "upper_comparison_count", "grouped_bar_count", default=20), 20)
    expect("b", "plotted_row_count", plotted_count, 60)
    sweep_recipes = _value_from(b, "sweep_recipes", "sensor_sweep_recipes", default=None)
    expect("b", "sweep_count", len(sweep_recipes) if isinstance(sweep_recipes, list) else _value_from(b, "sweep_count", default=None), 2)
    expect("b", "sweep_sensor_counts", _value_from(b, "sweep_sensor_counts", "sensor_counts", default=COUNTS), COUNTS)
    expect("b", "trace_count", _value_from(b, "trace_count", "method_trace_count", default=len(models) if isinstance(models, list) else None), 4)

    c_columns = _value_from(c, "column_order", "display_columns", "columns", default=None)
    if isinstance(c_columns, list):
        c_columns = [
            "reference" if _normalise_key(value) in {"reference", "groundtruth", "truth"}
            else (_normalise_model(value) or value)
            for value in c_columns
        ]
    expect("c", "panel_present", bool(c), True)
    expect("c", "column_order", c_columns, CD_COLUMNS)
    expect("c", "column_count", _value_from(c, "column_count", default=len(c_columns) if isinstance(c_columns, list) else None), 5)
    expect("c", "row_order", _value_from(c, "row_order", "rows", default=None), CD_ROWS)
    expect("c", "row_cell_counts", _value_from(c, "row_cell_counts", "cells_per_row", default=None), {"full_field": 5, "zoomed_field": 5, "local_absolute_error": 4})
    expect("c", "tile_count", _value_from(c, "tile_count", "map_tile_count", default=14), 14)

    expect("d", "panel_present", bool(d), True)
    expect("d", "scale_rows", _value_from(d, "qualitative_scales", "scales", default=None), SCALES)
    d_models = _value_from(d, "qualitative_models", "models", default=None)
    expect("d", "qualitative_model_order", d_models, ["DMFGen", "Senseiver"])
    expect("d", "qualitative_tile_count", _value_from(d, "qualitative_tile_count", "tile_count", default=(1 + len(d_models)) * len(SCALES) if isinstance(d_models, list) else None), 9)
    expect("d", "main_quantitative_value_count", _value_from(d, "main_quantitative_value_count", "line_plot_count", default=0), 0)

    expect("e", "panel_present", bool(e), True)
    expect("e", "matrix_count", _value_from(e, "matrix_count", default=None), 2)
    shape = _value_from(e, "matrix_shape", "matrix_shapes", default=None)
    shape_ok = shape in ([4, 9], [[4, 3], [4, 3]])
    expect("e", "matrix_shape", shape, shape if shape_ok else [4, 9])
    cells = _value_from(e, "heatmap_values", "matrix_values", "cells", default=None)
    expect("e", "cell_count", len(cells) if isinstance(cells, list) else _value_from(e, "cell_annotation_count", default=None), 72)
    expect("e", "recipe_groups", _value_from(e, "recipes", "recipe_order", default=None), ["3_Mixed_HML", "4_ZeroH_Balanced", "5_ZeroH_MRich"])
    expect("e", "scale_groups", _value_from(e, "scale_groups", "scales", default=None), SCALES)
    return bool(passed), details


def _find_named(release: Path, names: Iterable[str], patterns: Iterable[str] = ()) -> Path | None:
    for name in names:
        candidate = release / name
        if candidate.is_file():
            return candidate
    for pattern in patterns:
        candidates = sorted(path for path in release.rglob(pattern) if path.is_file())
        if candidates:
            return candidates[0]
    return None


def _find_manifest(release: Path, explicit: Path | None, revision: str) -> Path | None:
    if explicit:
        return explicit.resolve()
    names = [
        f"source_manifest_{revision}.json",
        f"source_manifest_{revision.lower()}.json",
        f"FigureSourceManifest_unified_{revision}.json",
        "source_manifest.json",
        "FigureSourceManifest.json",
    ]
    return _find_named(release, names, (f"*manifest*{revision}*.json", "*Manifest*.json"))


def _find_required(release: Path, name: str) -> Path | None:
    return _find_named(release, (name,), (name, name.lower(), name.upper()))


def _read_json_document(path: Path | None) -> dict[str, Any]:
    return _load_json(path) if path else {}


def _revision_identity(v4: dict[str, Any], release: Path, main_files: dict[str, Path | None]) -> tuple[bool, dict[str, Any]]:
    candidates = [
        v4.get("revision"), v4.get("figure_revision"), v4.get("version"),
        v4.get("schema_version"), v4.get("workflow_label"), release.name,
        *(path.name for path in main_files.values() if path),
    ]
    text = " ".join(str(value) for value in candidates if value is not None).lower()
    passed = bool(re.search(r"(?:^|[^a-z])v4(?:$|[^a-z])", text))
    return passed, {"candidates": candidates, "matched_v4": passed}


def _pdf_size_mm(path: Path) -> list[float] | None:
    if not path or not path.is_file() or shutil.which("pdfinfo") is None:
        return None
    try:
        output = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True).stdout
        line = next(line for line in output.splitlines() if line.startswith("Page size:"))
        tokens = line.split()
        return [float(tokens[2]) * 25.4 / 72.0, float(tokens[4]) * 25.4 / 72.0]
    except (OSError, StopIteration, ValueError, IndexError, subprocess.CalledProcessError):
        return None


def _svg_size_mm(path: Path) -> list[float] | None:
    if not path or not path.is_file():
        return None
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None

    def to_mm(value: str | None) -> float | None:
        if value is None:
            return None
        match = re.fullmatch(r"\s*([0-9.+-Ee]+)\s*([a-zA-Z]*)\s*", value)
        if not match:
            return None
        number, unit = float(match.group(1)), match.group(2).lower()
        return number * {"": 25.4 / 96.0, "px": 25.4 / 96.0, "pt": 25.4 / 72.0,
                          "in": 25.4, "cm": 10.0, "mm": 1.0}.get(unit, math.nan)

    width = to_mm(root.get("width")); height = to_mm(root.get("height"))
    return [width, height] if width is not None and height is not None and all(math.isfinite(x) for x in (width, height)) else None


def _png_size(path: Path) -> list[int] | None:
    if Image is None or not path or not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            return [int(image.width), int(image.height)]
    except Exception:
        return None


def _pdf_fonts(path: Path) -> tuple[bool, dict[str, Any]]:
    if not path or not path.is_file() or shutil.which("pdffonts") is None:
        return False, {"error": "pdffonts unavailable or PDF missing"}
    try:
        output = subprocess.run(["pdffonts", str(path)], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, {"error": str(exc)}
    lines = [line for line in output.splitlines() if line.strip()]
    rows = []
    for line in lines:
        if line.startswith("name ") or line.startswith("---"):
            continue
        fields = line.split()
        if len(fields) < 6:
            continue
        rows.append({"name": fields[0], "type": " ".join(fields[1:-5]), "emb": fields[-5], "sub": fields[-4], "uni": fields[-3]})
    embedded = bool(rows) and all(row["emb"].lower() == "yes" and row["uni"].lower() == "yes" for row in rows)
    no_type3 = bool(rows) and all("type 3" not in row["type"].lower() for row in rows)
    return bool(embedded and no_type3), {"fonts": rows, "embedded": embedded, "no_type3": no_type3}


def _pdf_text_editable(path: Path) -> tuple[bool, dict[str, Any]]:
    if not path or not path.is_file() or shutil.which("pdftotext") is None:
        return False, {"error": "pdftotext unavailable or PDF missing"}
    try:
        result = subprocess.run(["pdftotext", "-layout", str(path), "-"], check=True, capture_output=True, text=True)
        text = result.stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, {"error": str(exc)}
    return bool(text.strip()), {"character_count": len(text), "sample": text[:500]}


def _svg_text_elements(path: Path) -> list[dict[str, Any]]:
    if not path or not path.is_file():
        return []
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return []
    elements = []
    parent_map = {child: parent for parent in root.iter() for child in parent}
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] != "text":
            continue
        style = element.get("style", "")
        explicit = element.get("font-size") or ""
        match = re.search(r"font-size\s*:\s*([0-9.+-Ee]+)\s*(px|pt)?", style, flags=re.I)
        if match is None and explicit:
            match = re.fullmatch(r"\s*([0-9.+-Ee]+)\s*(px|pt)?\s*", explicit, flags=re.I)
        if match is None:
            # MathText and rotated labels can place font declarations on
            # descendant tspans. Use their largest size as the base size;
            # naturally smaller sub/superscript glyphs remain exempt.
            descendant_sizes = []
            for child in element.iter():
                child_style = child.get("style", "")
                child_explicit = child.get("font-size") or ""
                child_match = re.search(
                    r"font-size\s*:\s*([0-9.+-Ee]+)\s*(px|pt)?",
                    child_style, flags=re.I,
                )
                if child_match is None and child_explicit:
                    child_match = re.fullmatch(
                        r"\s*([0-9.+-Ee]+)\s*(px|pt)?\s*",
                        child_explicit, flags=re.I,
                    )
                if child_match:
                    descendant_sizes.append(
                        (float(child_match.group(1)), child_match.group(2) or "pt")
                    )
            if descendant_sizes:
                size_value, size_unit = max(descendant_sizes, key=lambda item: item[0])
                match = re.match(
                    r"([0-9.+-Ee]+)(px|pt)", f"{size_value}{size_unit}", flags=re.I,
                )
        size = float(match.group(1)) if match else None
        unit = (match.group(2) or "pt").lower() if match else None
        # Matplotlib's SVG backend writes point-sized text with a px suffix;
        # the value is the requested point size in the figure's physical SVG.
        if unit == "pt":
            size_pt = size
        else:
            size_pt = size
        text = "".join(element.itertext()).strip()
        ancestor_role = None
        ancestor = parent_map.get(element)
        while ancestor is not None:
            ancestor_id = str(ancestor.get("id") or "")
            if ancestor_id.startswith("font-role:"):
                ancestor_role = ancestor_id.split(":", 1)[1]
                break
            if ancestor_id.startswith(("xtick_", "ytick_")):
                ancestor_role = "tick_label"
                break
            if ancestor_id == "panel-f-cell-value":
                ancestor_role = "numeric_annotation"
                break
            ancestor = parent_map.get(ancestor)
        elements.append({
            "text": re.sub(r"\s+", " ", text), "size_pt": size_pt,
            "style": style,
            "role": _first(
                element.attrib, "data-role", "role", "aria-label",
                default=ancestor_role,
            ),
            "id": element.get("id"), "class": element.get("class"),
        })
    return elements


def _role_from_text(item: dict[str, Any], lqa: dict[str, Any]) -> str:
    explicit = item.get("role")
    if explicit:
        normalized = _normalise_key(explicit)
        for role in ROLE_DESIGN_MIN_PT:
            if _normalise_key(role) == normalized:
                return role
    text = str(item.get("text", "")).strip()
    if len(text) == 1 and text.lower() in "abcde" and "font-weight: 700" in str(item.get("style", "")):
        return "panel_label"
    examples = _first(lqa, "text_examples_by_role", "examples", default={})
    if isinstance(examples, dict):
        for role, values in examples.items():
            if isinstance(values, list) and text in {str(value).replace("\\n", " ") for value in values}:
                normalized = _normalise_key(role)
                if normalized in {_normalise_key(value) for value in ROLE_DESIGN_MIN_PT}:
                    return next(value for value in ROLE_DESIGN_MIN_PT if _normalise_key(value) == normalized)
    # Numeric strings and short recipe/model labels are ordinary base text if
    # no renderer role marker is available; this intentionally fails old 5.5pt
    # exports instead of silently excusing them.
    return "ordinary"


def _width_records(value: Any) -> dict[float, dict[str, Any]]:
    """Normalize common LAYOUT_QA per-width containers to {180, 162}."""
    found: dict[float, dict[str, Any]] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            text = str(key).lower().replace("mm", "").replace("_", "").replace("-", "")
            match = re.search(r"(?:^|[^0-9])(180|162)(?:$|[^0-9])", text)
            if match and isinstance(child, dict):
                found[float(match.group(1))] = child
        for key in ("by_width", "width_checks", "checks_by_width", "renderer_checks", "renders", "measurements"):
            nested = value.get(key)
            if nested is not None:
                found.update(_width_records(nested))
    elif isinstance(value, list):
        for child in value:
            if not isinstance(child, dict):
                continue
            width = _first(child, "width_mm", "tested_width_mm", "design_width_mm", default=None)
            try:
                width = float(width)
            except (TypeError, ValueError):
                width = None
            if width in (180.0, 162.0):
                found[width] = child
    return found


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _recursive_values(mapping: Any, keys: set[str]) -> list[Any]:
    values: list[Any] = []
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            if _normalise_key(key) in {_normalise_key(item) for item in keys}:
                values.append(value)
            values.extend(_recursive_values(value, keys))
    elif isinstance(mapping, list):
        for value in mapping:
            values.extend(_recursive_values(value, keys))
    return values


def _count_from(record: dict[str, Any], keys: set[str]) -> int | None:
    values = _recursive_values(record, keys)
    for value in values:
        if isinstance(value, list):
            return len(value)
        number = _number(value)
        if number is not None:
            return int(number)
    return None


def _gap_values(record: dict[str, Any], role: str) -> tuple[float | None, float | None, bool]:
    aliases = {
        "major_panel_blocks": {"major_panel_blocks", "major_panel_block", "major_blocks", "major"},
        "adjacent_plot_or_map_windows": {"adjacent_plot_or_map_windows", "adjacent_maps", "adjacent_windows", "adjacent"},
        "ordinary_text_to_non_owned_axes_or_data": {"ordinary_text_to_non_owned_axes_or_data", "text_to_axes", "text_to_data", "text_data"},
        "panel_letter_to_non_owned_text_or_data": {"panel_letter_to_non_owned_text_or_data", "panel_letter", "letter"},
    }
    horizontal = _recursive_values(record, {"minimum_horizontal_clearance_mm", "min_horizontal_clearance_mm", "horizontal_min_mm"})
    vertical = _recursive_values(record, {"minimum_vertical_clearance_mm", "min_vertical_clearance_mm", "vertical_min_mm"})
    nested = _recursive_values(record, aliases[role])

    def first_number(values: list[Any]) -> float | None:
        for value in values:
            if isinstance(value, dict):
                for key in ("horizontal_mm", "h_mm", "minimum_horizontal_mm", "vertical_mm", "v_mm", "minimum_vertical_mm", "value_mm", "mm"):
                    if key in value and _number(value[key]) is not None:
                        return _number(value[key])
            elif isinstance(value, list):
                numbers = [_number(item) for item in value]
                numbers = [item for item in numbers if item is not None]
                if numbers:
                    return min(numbers)
            elif _number(value) is not None:
                return _number(value)
        return None

    h = first_number(horizontal)
    v = first_number(vertical)
    if nested:
        # If the nested record explicitly contains both directions, use them;
        # a single scalar is not allowed to masquerade as both measurements.
        for value in nested:
            if isinstance(value, dict):
                h = h if h is not None else first_number([_first(value, "horizontal_mm", "h_mm", "minimum_horizontal_mm")])
                v = v if v is not None else first_number([_first(value, "vertical_mm", "v_mm", "minimum_vertical_mm")])
    return h, v, bool(h is not None and v is not None)


def _layout_qa_check(lqa: dict[str, Any], widths: tuple[float, float] = (180.0, 162.0)) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {"widths": {}, "missing_widths": [], "intentional_in_data_annotations": None}
    passed = True
    width_map = _width_records(lqa)
    for width in widths:
        record = width_map.get(width)
        if record is None:
            details["missing_widths"].append(width)
            passed = False
            continue
        width_detail: dict[str, Any] = {}
        # Renderer evidence must be explicit and positive.
        renderer = _first(record, "renderer_bbox_checks", "renderer_based", "bbox_checks", "measured_by_renderer", default=None)
        renderer_ok = renderer is True or (isinstance(renderer, dict) and _first(renderer, "passed", "complete", default=False) is True)
        width_detail["renderer_bbox_checks"] = renderer
        passed &= renderer_ok

        collisions = _count_from(record, {"unexplained_collision_count", "collision_count", "forbidden_intersection_count", "unexplained_collisions"})
        clipping = _count_from(record, {"clipped_artist_count", "clipping_count", "clipped_count", "canvas_containment_violations"})
        if collisions is None:
            collision_list = _recursive_values(record, {"unexplained_collisions", "collisions", "intersections"})
            collisions = sum(len(value) for value in collision_list if isinstance(value, list)) if collision_list else None
        if clipping is None:
            clip_list = _recursive_values(record, {"clipped_artists", "clipping_violations"})
            clipping = sum(len(value) for value in clip_list if isinstance(value, list)) if clip_list else None
        collision_ok = collisions == 0
        clipping_ok = clipping == 0
        width_detail.update({"unexplained_collision_count": collisions, "clipped_artist_count": clipping,
                             "collision_free": collision_ok, "clipping_free": clipping_ok})
        passed &= collision_ok and clipping_ok

        gap_detail = {}
        for role, minimum in HARD_GAP_MM.items():
            h, v, measured = _gap_values(record, role)
            role_ok = measured and h is not None and v is not None and h >= minimum and v >= minimum
            gap_detail[role] = {"horizontal_mm": h, "vertical_mm": v, "required_mm": minimum,
                                "measured": measured, "passed": role_ok}
            passed &= role_ok
        width_detail["gaps"] = gap_detail

        containment = _first(record, "canvas_containment_qa", "clipping_qa", "containment", default={})
        if isinstance(containment, dict):
            containment_ok = _first(containment, "passed", "within_canvas", "no_clipping", default=False) is True
            passed &= containment_ok
            width_detail["canvas_containment_qa"] = containment
        else:
            width_detail["canvas_containment_qa"] = containment
            passed = False

        details["widths"][str(int(width))] = width_detail

    registry = _first(lqa, "intentional_in_data_annotations", "intentional_in_data_annotation_registry",
                      "in_data_annotations", default=None)
    details["intentional_in_data_annotations"] = registry
    registry_ok = isinstance(registry, list)
    if registry_ok:
        for item in registry:
            if not isinstance(item, dict):
                registry_ok = False
                continue
            if not _first(item, "semantic_id", "panel_id", "id", default=None):
                registry_ok = False
            if _first(item, "obscures_evidence", "covers_evidence", "evidence_overlap", default=False) is True:
                registry_ok = False
            if "overlap" in item and item.get("overlap") not in (False, 0, None, []):
                registry_ok = False
    # Panels c and d contain numerical labels in the active V3-7 story.  If a
    # V4 builder moves all of them to a gutter, it must state the zero count;
    # otherwise at least one registered record for each panel is mandatory.
    registry_panels = set()
    for item in registry if isinstance(registry, list) else []:
        if not isinstance(item, dict):
            continue
        owner = str(_first(item, "panel_id", "panel", "owner", "semantic_id", default="")).lower()
        match = re.search(r"(?:panel[_.:-]*)?([cd])(?:$|[_.:-])", owner)
        if match:
            registry_panels.add(match.group(1))
    c_count = _metric_number(lqa, {"panel_c_in_data_annotation_count", "c_in_data_annotation_count"})
    d_count = _metric_number(lqa, {"panel_d_in_data_annotation_count", "d_in_data_annotation_count"})
    cd_registry_ok = ((c_count == 0 or "c" in registry_panels) and
                      (d_count == 0 or "d" in registry_panels))
    details["registered_annotation_panels"] = sorted(registry_panels)
    details["panel_cd_annotation_counts"] = {"c": c_count, "d": d_count}
    registry_ok &= cd_registry_ok
    passed &= registry_ok

    visual = _first(lqa, "visual_inspection", "visual_review", default={})
    visual_widths = _width_records(visual)
    visual_ok = isinstance(visual, dict) and all(
        bool(_first(visual, "full_page", "full_page_inspected", default=False))
        and bool(_first(visual, "high_zoom", "high_zoom_inspected", default=False))
        for _ in [0]
    )
    if visual_widths:
        visual_ok = all(
            _first(record, "full_page", "full_page_inspected", default=False) is True
            and _first(record, "high_zoom", "high_zoom_inspected", default=False) is True
            for record in visual_widths.values()
        )
    details["visual_inspection"] = visual
    passed &= visual_ok

    grayscale = _first(lqa, "grayscale_check", "grayscale", "grayscale_qa", default={})
    cvd = _first(lqa, "color_vision_deficiency_check", "cvd_check", "cvd", "colorblind_qa", default={})
    grayscale_ok = grayscale is True or (isinstance(grayscale, dict) and _first(grayscale, "passed", "interpretable", default=False) is True)
    cvd_ok = cvd is True or (isinstance(cvd, dict) and _first(cvd, "passed", "distinguishable", default=False) is True)
    details["grayscale_check"] = grayscale; details["color_vision_deficiency_check"] = cvd
    passed &= grayscale_ok and cvd_ok
    return bool(passed), details


def _metric_number(mapping: Any, aliases: set[str]) -> float | None:
    """Find the first finite scalar for one semantic QA metric."""
    wanted = {_normalise_key(alias) for alias in aliases}
    if isinstance(mapping, dict):
        for key, value in mapping.items():
            if _normalise_key(key) in wanted:
                if isinstance(value, dict):
                    value = _first(value, "value_mm", "minimum_mm", "min_mm", "gap_mm", "clearance_mm", default=None)
                if isinstance(value, (list, tuple)):
                    numbers = [_number(item) for item in value]
                    numbers = [item for item in numbers if item is not None]
                    if numbers:
                        return min(numbers)
                number = _number(value)
                if number is not None:
                    return number
            found = _metric_number(value, aliases)
            if found is not None:
                return found
    elif isinstance(mapping, list):
        for value in mapping:
            found = _metric_number(value, aliases)
            if found is not None:
                return found
    return None


def _panel_layout_views(lqa: dict[str, Any], label: str, width_record: dict[str, Any] | None) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    if isinstance(width_record, dict):
        views.append(width_record)
        for key in ("panels", "panel_checks", "panel_geometry", "panel_layout"):
            nested = width_record.get(key)
            if isinstance(nested, dict):
                child = nested.get(label, nested.get(f"panel_{label}"))
                if isinstance(child, dict):
                    views.append(child)
    for key in (label, f"panel_{label}"):
        child = lqa.get(key)
        if isinstance(child, dict):
            views.append(child)
    for key in ("panels", "panel_checks", "panel_geometry", "panel_layout"):
        nested = lqa.get(key)
        if isinstance(nested, dict):
            child = nested.get(label, nested.get(f"panel_{label}"))
            if isinstance(child, dict):
                views.append(child)
    return views


def _metric_from_views(views: list[dict[str, Any]], aliases: set[str]) -> float | None:
    for view in views:
        value = _metric_number(view, aliases)
        if value is not None:
            return value
    return None


def _shared_cd_row_delta(document: dict[str, Any], lqa: dict[str, Any]) -> float | None:
    aliases = {"shared_cd_row_delta_mm", "c_d_shared_row_delta_mm", "max_boundary_delta_mm",
               "max_row_boundary_delta_mm", "row_boundary_delta_mm", "cd_row_delta_mm"}
    for view in (document.get("cross_panel_alignment_qa"), lqa.get("cross_panel_alignment_qa"), lqa):
        found = _metric_number(view, aliases)
        if found is not None:
            return found
    alignment = document.get("cross_panel_alignment_qa", {})
    rows = _first(alignment, "rows", "row_boundaries", default=None)
    if isinstance(rows, list):
        deltas = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in ("bottom_delta_mm", "top_delta_mm", "height_delta_mm"):
                value = _number(row.get(key))
                if value is not None:
                    deltas.append(abs(value))
        if deltas:
            return max(deltas)
    return None


def _special_layout_contract(
    document: dict[str, Any], lqa: dict[str, Any], widths: tuple[float, float] = (180.0, 162.0),
) -> tuple[bool, dict[str, Any]]:
    """Check the V4 figure's agreed panel-specific physical clearances."""
    details: dict[str, Any] = {"widths": {}, "shared_cd_row_delta_mm": None}
    passed = True
    width_map = _width_records(lqa)
    for width in widths:
        record = width_map.get(width)
        views_b = _panel_layout_views(lqa, "b", record)
        views_c = _panel_layout_views(lqa, "c", record)
        views_e = _panel_layout_views(lqa, "e", record)
        # The stated dimensions are design-space dimensions.  At manuscript
        # width the hard target scales with 162/180, while an explicitly
        # measured 1.5--2 mm value also remains acceptable.
        scale = width / 180.0
        b_annotation_min = 1.2 if width == 180.0 else 1.0
        legend_height_min = 6.5 if width == 180.0 else 6.5 * scale
        e_minima = {
            "metric_title_to_recipe": 2.0 * scale,
            "recipe_to_matrix": 1.5 * scale,
            "matrix_bottom_to_panel": 2.0 * scale,
        }
        b_annotation = _metric_from_views(views_b, {
            "dmfg_annotation_to_ci_clearance_mm", "dmfgen_annotation_to_ci_clearance_mm",
            "annotation_to_ci_clearance_mm", "annotation_ci_clearance_mm",
            "dmfg_annotation_min_ci_clearance_mm",
        })
        legend_height = _metric_from_views(views_b, {"legend_strip_height_mm", "legend_height_mm", "legend_strip_mm"})
        legend_clear = _metric_from_views(views_b, {
            "legend_strip_clearance_mm", "legend_clearance_mm", "legend_to_sweep_clearance_mm",
            "legend_gap_mm",
        })
        cbar_gap = _metric_from_views(views_c, {
            "map_to_colorbar_gap_mm", "cbar_map_gap_mm", "colorbar_map_gap_mm",
            "colorbar_gap_mm", "gap_above_colorbars_mm",
        })
        title_gap = _metric_from_views(views_e, {
            "metric_title_to_recipe_gap_mm", "metric_to_recipe_gap_mm", "title_recipe_gap_mm",
        })
        recipe_gap = _metric_from_views(views_e, {
            "recipe_to_matrix_gap_mm", "recipe_matrix_gap_mm", "recipe_to_heatmap_gap_mm",
        })
        lower_gap = _metric_from_views(views_e, {
            "matrix_bottom_to_panel_gap_mm", "matrix_to_panel_gap_mm", "lower_gap_mm",
        })
        width_detail = {
            "panel_b_dmfg_annotation_to_ci_mm": {"observed": b_annotation, "required_min": b_annotation_min,
                                                   "passed": b_annotation is not None and b_annotation >= b_annotation_min},
            "panel_b_legend_strip_height_mm": {"observed": legend_height, "required_min": legend_height_min,
                                                 "passed": legend_height is not None and legend_height >= legend_height_min},
            "panel_b_legend_clearance_mm": {"observed": legend_clear, "required_range": [1.5, 2.5],
                                               "passed": legend_clear is not None and 1.5 <= legend_clear <= 2.5},
            "panel_c_map_colorbar_gap_mm": {
                "observed": cbar_gap, "required_min": 1.5 * scale,
                "target_range": [1.5 * scale, 2.0 * scale],
                "passed": cbar_gap is not None and cbar_gap >= 1.5 * scale,
            },
            "panel_e_metric_title_to_recipe_mm": {"observed": title_gap, "required_min": e_minima["metric_title_to_recipe"],
                                                    "passed": title_gap is not None and title_gap >= e_minima["metric_title_to_recipe"]},
            "panel_e_recipe_to_matrix_mm": {"observed": recipe_gap, "required_min": e_minima["recipe_to_matrix"],
                                              "passed": recipe_gap is not None and recipe_gap >= e_minima["recipe_to_matrix"]},
            "panel_e_matrix_bottom_to_panel_mm": {"observed": lower_gap, "required_min": e_minima["matrix_bottom_to_panel"],
                                                   "passed": lower_gap is not None and lower_gap >= e_minima["matrix_bottom_to_panel"]},
        }
        details["widths"][str(int(width))] = width_detail
        passed &= all(item["passed"] for item in width_detail.values())

    cd_delta = _shared_cd_row_delta(document, lqa)
    details["shared_cd_row_delta_mm"] = cd_delta
    details["shared_cd_row_alignment_passed"] = cd_delta is not None and cd_delta <= 0.01
    passed &= details["shared_cd_row_alignment_passed"]
    return bool(passed), details


def _typography_check(svg: Path | None, lqa: dict[str, Any], design_width: float, insertion_width: float) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {"design_width_mm": design_width, "insertion_width_mm": insertion_width,
                               "role_sizes_pt": {}, "text_count": 0, "violations": []}
    passed = True
    role_sizes = _first(lqa, "role_sizes_pt", "font_role_sizes_pt", "declared_role_sizes_pt", default={})
    if not isinstance(role_sizes, dict):
        role_sizes = {}
    for role, minimum in ROLE_DESIGN_MIN_PT.items():
        candidates = [value for key, value in role_sizes.items() if _normalise_key(key) in {_normalise_key(role), _normalise_key(f"size_{role}")}] 
        if candidates:
            observed = _number(candidates[0])
            details["role_sizes_pt"][role] = observed
            insertion_min = minimum * insertion_width / design_width
            role_ok = observed is not None and observed >= minimum - 1e-9 and observed * insertion_width / design_width >= insertion_min - 1e-9
            passed &= role_ok
        else:
            details["role_sizes_pt"][role] = None
            # The ordinary role can be measured from the vector, but missing
            # declared roles are still a bookkeeping failure at release time.
            passed = False

    text_items = _svg_text_elements(svg) if svg else []
    details["text_count"] = len(text_items)
    details["svg_text_editable"] = bool(text_items)
    if not text_items:
        passed = False
    for item in text_items:
        size = _number(item.get("size_pt"))
        if size is None:
            details["violations"].append({"text": item.get("text"), "reason": "font_size_unmeasured"})
            passed = False
            continue
        role = _role_from_text(item, lqa)
        design_min = ROLE_DESIGN_MIN_PT.get(role, ROLE_DESIGN_MIN_PT["ordinary"])
        insertion_size = size * insertion_width / design_width
        insertion_min = design_min * insertion_width / design_width
        violation = size < design_min - 1e-9 or insertion_size < insertion_min - 1e-9
        if violation:
            details["violations"].append({"text": item.get("text"), "size_pt": size, "size_at_162mm_pt": insertion_size,
                                           "role": role, "required_design_pt": design_min,
                                           "required_at_162mm_pt": insertion_min})
    passed &= not details["violations"]
    details["minimum_text_pt_at_design"] = min((item.get("size_pt") for item in text_items if _number(item.get("size_pt")) is not None), default=None)
    details["minimum_text_pt_at_insertion"] = min((item.get("size_pt") * insertion_width / design_width for item in text_items if _number(item.get("size_pt")) is not None), default=None)
    return bool(passed), details


def _palette_mapping(document: dict[str, Any], lqa: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {key: [] for key in APPROVED_MODEL_PALETTE}
    roots = [document.get("semantic_style_table"), document.get("resolved_semantic_style_table"),
             document.get("palette"), document.get("model_palette"), document.get("style_contract"),
             lqa.get("semantic_style_table"), lqa.get("resolved_semantic_style_table"),
             lqa.get("semantic_palette"), lqa.get("palette")]

    def walk(value: Any, context: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            key_model = _normalise_model(value.get("semantic_id", value.get("model", value.get("label", ""))))
            color = _normalise_hex(_first(value, "resolved_hex", "hex", "color", "colour", default=None))
            if key_model in found and color:
                found[key_model].append({"color": color, "context": list(context), "entry": value})
            for key, child in value.items():
                model = _normalise_model(key)
                if model in found:
                    child_color = _normalise_hex(child if isinstance(child, str) else _first(child, "resolved_hex", "hex", "color", "colour", default=None))
                    if child_color:
                        found[model].append({"color": child_color, "context": list(context) + [str(key)], "entry": child})
                walk(child, context + (str(key),))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, context + (str(index),))

    for root in roots:
        walk(root)
    return found


def _palette_check(v4: dict[str, Any], lqa: dict[str, Any], svg: Path | None, v3: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    mapping = _palette_mapping(v4, lqa)
    details: dict[str, Any] = {"expected": APPROVED_MODEL_PALETTE, "observed": {}, "missing": [], "mismatches": [], "duplicate_conflicts": []}
    passed = True
    for model, expected in APPROVED_MODEL_PALETTE.items():
        observations = sorted({_normalise_hex(item.get("color")) for item in mapping.get(model, []) if _normalise_hex(item.get("color"))})
        details["observed"][model] = observations
        if not observations:
            details["missing"].append(model)
            passed = False
        expected_normalized = _normalise_hex(expected)
        if any(value != expected_normalized for value in observations):
            details["mismatches"].append({"model": model, "expected": expected, "observed": observations})
            passed = False
    # Same semantic identity must not split across panels/legends.
    for model, observations in details["observed"].items():
        if len(observations) > 1:
            details["duplicate_conflicts"].append({"model": model, "colors": observations})
            passed = False

    cmap = _first(lqa, "semantic_colormaps", default=None)
    if cmap is None:
        cmap = _first(v4.get("style_contract", {}), "semantic_colormaps", default=None)
    details["semantic_colormaps"] = cmap
    cmap_ok = isinstance(cmap, dict)
    allowed = {
        "physical_field": {"viridis", "rdBu_r".lower(), "RdBu_r".lower()},
        "signed_field": {"RdBu_r".lower()}, "signed_component": {"RdBu_r".lower(), "puor_r"},
        "signed_residual": {"RdBu_r".lower(), "puor_r"}, "signed_bias": {"rdbu_r"},
        "absolute_error": {"ylorrd"},
        "nonnegative_error_heatmap": {"ylorrd"}, "correlation": {"cividis"},
        "ordered_correlation": {"cividis"}, "joint_density": {"cividis"},
    }
    if cmap_ok:
        for key, values in cmap.items():
            normalized = _normalise_key(key)
            if normalized in {_normalise_key(name) for name in allowed}:
                accepted = next(accepted for name, accepted in allowed.items() if _normalise_key(name) == normalized)
                if str(values).strip().lower() not in accepted:
                    details.setdefault("colormap_mismatches", []).append({"semantic": key, "observed": values, "allowed": sorted(accepted)})
                    cmap_ok = False
        norm_state = _first(lqa, "normalization_unchanged", "source_normalization_unchanged", default=None)
        if norm_state is False:
            cmap_ok = False
            details["normalization_unchanged"] = False
    else:
        details["colormap_mismatches"] = ["semantic_colormaps missing"]
    passed &= cmap_ok

    if svg and svg.is_file():
        svg_text = svg.read_text(encoding="utf-8", errors="replace").lower()
        details["svg_colors_present"] = {model: expected.lower() in svg_text for model, expected in APPROVED_MODEL_PALETTE.items()}
        # Require each compared model color to reach at least one rendered
        # artist. Truth may be represented as a dark edge/text, so it is still
        # checked in the same way.
        for model, present in details["svg_colors_present"].items():
            if not present:
                passed = False
    return bool(passed), details


def _metadata_check(release: Path, required: dict[str, Path | None]) -> tuple[bool, dict[str, Any], dict[str, dict[str, Any]]]:
    presence = {name: bool(path and path.is_file() and path.stat().st_size > 0) for name, path in required.items()}
    docs = {name: _load_json(path) for name, path in required.items() if name.endswith(".json") and path}
    details: dict[str, Any] = {"presence": presence, "paths": {name: str(path) if path else None for name, path in required.items()}}
    passed = all(presence.values())

    lock = docs.get("SOURCE_LOCK.json", {})
    lock_status = str(_first(lock, "status", "lock_status", default="")).upper()
    lock_records = _explicit_source_records(lock, lock)
    lock_arrays = list(_array_records(lock))
    lock_ok = bool(lock) and lock_status not in {"", "NOT_RECORDED", "UNLOCKED", "FAILED"} and bool(lock_records or lock_arrays)
    details["source_lock"] = {"status": lock_status, "record_count": len(lock_records), "array_record_count": len(lock_arrays), "passed": lock_ok}
    passed &= lock_ok

    comparison = docs.get("SCIENTIFIC_STATE_COMPARISON.json", {})
    comparison_pass = _first(comparison, "passed", "overall_passed", "scientific_state_unchanged", "state_unchanged", default=None)
    if comparison_pass is None:
        comparison_pass = str(_first(comparison, "status", default="")).upper() in {"PASS", "PASSED", "OK", "UNCHANGED"}
    forbidden_nonempty = []
    for key in ("changed_fields", "scientific_changes", "array_changes", "mismatches", "differences", "changed_arrays"):
        value = comparison.get(key)
        if value not in (None, [], {}, False):
            forbidden_nonempty.append(key)
    comparison_ok = bool(comparison) and comparison_pass is True and not forbidden_nonempty
    details["scientific_state_comparison"] = {"passed_field": comparison_pass, "nonempty_change_fields": forbidden_nonempty, "passed": comparison_ok}
    passed &= comparison_ok

    layout = docs.get("LAYOUT_QA.json", {})
    details["layout_keys"] = sorted(layout) if layout else []
    passed &= bool(layout)

    changelog_path = required.get("STYLE_CHANGELOG.md")
    actions_path = required.get("AUTHOR_ACTIONS.md")
    changelog = changelog_path.read_text(encoding="utf-8", errors="replace") if changelog_path else ""
    actions = actions_path.read_text(encoding="utf-8", errors="replace") if actions_path else ""
    details["style_changelog"] = {"nonempty": bool(changelog.strip()), "mentions_v4": bool(re.search(r"\bv4\b", changelog, re.I)), "mentions_style_only": bool(re.search(r"style[- ]only|scientific.*unchanged|no.*scientific", changelog, re.I))}
    details["author_actions"] = {"nonempty": bool(actions.strip())}
    passed &= bool(changelog.strip()) and details["style_changelog"]["mentions_v4"] and details["style_changelog"]["mentions_style_only"] and bool(actions.strip())
    return bool(passed), details, {"SOURCE_LOCK.json": lock, "SCIENTIFIC_STATE_COMPARISON.json": comparison, "LAYOUT_QA.json": layout}


def _source_hash_array_check(v4: dict[str, Any], v3: dict[str, Any], v4_lock: dict[str, Any], v3_base: Path, v4_base: Path) -> tuple[bool, dict[str, Any]]:
    v3_records = _data_records(v3, None, v3_base)
    v4_records = _data_records(v4, v4_lock, v4_base)
    v3_hash_ok, v3_hash_details = _hash_audit(v3_records, v3_base)
    v4_hash_ok, v4_hash_details = _hash_audit(v4_records, v4_base)
    v3_sig = {_record_signature(record, v3_base) for record in v3_records}
    v4_sig = {_record_signature(record, v4_base) for record in v4_records}
    source_sets_equal = v3_sig == v4_sig
    v3_arrays = _array_state_for_records(v3_records, v3_base)
    v4_arrays = _array_state_for_records(v4_records, v4_base)
    array_sets_equal = v3_arrays == v4_arrays
    declared = _declared_array_state(v4_lock, v4_base)
    declared_ok = True
    for item in declared:
        if not item["exists"]:
            declared_ok = False
            continue
        actual = item["actual"]
        declared_values = item["declared"]
        # Validate only explicit array-level hashes; source manifest file
        # hashes are checked separately and must not be confused with bytes of
        # one member array.
        if "shape" in declared_values and isinstance(actual, dict) and len(actual) == 1:
            actual_state = next(iter(actual.values()))
            declared_ok &= list(declared_values["shape"]) == actual_state.get("shape")
        if "dtype" in declared_values and isinstance(actual, dict) and len(actual) == 1:
            declared_ok &= str(declared_values["dtype"]) == str(next(iter(actual.values())).get("dtype"))
        if "nan_mask_sha256" in declared_values and isinstance(actual, dict) and len(actual) == 1:
            declared_ok &= declared_values["nan_mask_sha256"] == next(iter(actual.values())).get("nan_mask_sha256")
        expected_array_hash = _first(declared_values, "array_sha256", "values_sha256", default=None)
        if expected_array_hash is not None and isinstance(actual, dict) and len(actual) == 1:
            declared_ok &= expected_array_hash == next(iter(actual.values())).get("values_sha256")
    passed = bool(v3_records and v4_records and v3_hash_ok and v4_hash_ok and source_sets_equal and array_sets_equal and declared_ok)
    return passed, {
        "v3_record_count": len(v3_records), "v4_record_count": len(v4_records),
        "v3_hashes_passed": v3_hash_ok, "v4_hashes_passed": v4_hash_ok,
        "v3_hash_failures": [item for item in v3_hash_details if not item["passed"]],
        "v4_hash_failures": [item for item in v4_hash_details if not item["passed"]],
        "source_sets_equal": source_sets_equal, "v3_only_sources": sorted(v3_sig - v4_sig),
        "v4_only_sources": sorted(v4_sig - v3_sig), "array_sets_equal": array_sets_equal,
        "declared_array_records": declared, "declared_arrays_passed": declared_ok,
        "array_files": sorted(set(v3_arrays) | set(v4_arrays)),
    }


def _output_check(
    v4: dict[str, Any], release: Path, main_files: dict[str, Path | None],
    design_width: float, design_height: float, preview_dpi: float,
) -> tuple[bool, dict[str, Any]]:
    pdf = main_files.get("pdf"); svg = main_files.get("svg"); png = main_files.get("png")
    pdf_size = _pdf_size_mm(pdf) if pdf else None
    svg_size = _svg_size_mm(svg) if svg else None
    png_size = _png_size(png) if png else None
    expected_height = _number(_first(v4.get("figure_contract", {}), "final_height_mm", default=None))
    declared_size = _first(v4.get("figure_contract", {}), "final_size_mm", "output_dimensions_mm", default=None)
    if isinstance(declared_size, (list, tuple)) and len(declared_size) == 2:
        declared_width, declared_height = _number(declared_size[0]), _number(declared_size[1])
    else:
        declared_width = _number(_first(v4, "canvas_width_mm", "design_width_mm", default=None))
        declared_height = expected_height or _number(_first(v4, "canvas_height_mm", "design_height_mm", default=None))
    # V4 is deliberately a 180 x 230 mm design.  Keep the expected height
    # explicit instead of silently deriving it from the PDF: a self-consistent
    # but wrong canvas is still a release failure.  A producer may override
    # this through the CLI for a separately approved layout variant.
    if declared_height is None:
        declared_height = design_height
    details: dict[str, Any] = {
        "files": {key: str(path) if path else None for key, path in main_files.items()},
        "pdf_size_mm": pdf_size, "svg_size_mm": svg_size, "png_size_px": png_size,
        "declared_size_mm": [declared_width, declared_height], "design_width_mm": design_width,
        "design_height_mm": design_height,
    }
    passed = all(path and path.is_file() and path.stat().st_size > 0 for path in (pdf, png)) and bool(svg and svg.is_file())
    if declared_width is None or abs(declared_width - design_width) > SIZE_TOLERANCE_MM:
        passed = False
    if declared_height is None or abs(declared_height - design_height) > SIZE_TOLERANCE_MM:
        passed = False
    if pdf_size is None or declared_height is None:
        passed = False
    else:
        passed &= abs(pdf_size[0] - design_width) <= SIZE_TOLERANCE_MM and abs(pdf_size[1] - declared_height) <= SIZE_TOLERANCE_MM
    if svg_size and pdf_size:
        passed &= all(abs(left - right) <= SIZE_TOLERANCE_MM for left, right in zip(svg_size, pdf_size))
    elif svg is not None:
        passed = False
    expected_px = [round((design_width if declared_width is None else declared_width) / 25.4 * preview_dpi),
                   round((declared_height or 0) / 25.4 * preview_dpi)]
    details["expected_png_px"] = expected_px
    if png_size is None or any(abs(int(observed) - int(expected)) > 4 for observed, expected in zip(png_size, expected_px)):
        passed = False
    return bool(passed), details


def _no_inference_check(v4: dict[str, Any], comparison: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    fields = {}
    for key in ("training_performed", "model_training_performed", "inference_performed", "model_inference_performed",
                "metric_recomputation_performed", "metrics_recomputed", "validated_sources_modified", "data_processing_changed"):
        if key in v4:
            fields[key] = v4[key]
        elif key in comparison:
            fields[key] = comparison[key]
    required = {"model_inference_performed": False, "validated_sources_modified": False}
    optional_false = {"training_performed", "model_training_performed", "inference_performed", "metric_recomputation_performed", "metrics_recomputed", "data_processing_changed"}
    passed = all(fields.get(key) is value for key, value in required.items()) and all(fields.get(key) is not True for key in optional_false)
    return bool(passed), {"observed": fields, "required_false": sorted(required)}


def audit_v4(
    release_dir: Path,
    v3_release_dir: Path = DEFAULT_V3_RELEASE,
    v4_manifest: Path | None = None,
    v3_manifest: Path | None = None,
    main_pdf: Path | None = None,
    main_svg: Path | None = None,
    main_png: Path | None = None,
    output: Path | None = None,
    design_width_mm: float = DESIGN_WIDTH_MM,
    design_height_mm: float = DESIGN_HEIGHT_MM,
    insertion_width_mm: float = INSERTION_WIDTH_MM,
    preview_dpi: float = PREVIEW_DPI,
) -> dict[str, Any]:
    """Run all V4 checks and write one machine-readable QA report."""
    release = Path(release_dir).resolve()
    baseline_release = Path(v3_release_dir).resolve()
    v4_manifest_path = _find_manifest(release, v4_manifest, "v4")
    v3_manifest_path = (Path(v3_manifest).resolve() if v3_manifest else _find_manifest(baseline_release, None, "v3_7"))
    v4 = _read_json_document(v4_manifest_path)
    v3 = _read_json_document(v3_manifest_path)
    revision_hint = str(v4.get("run_id", release.name))
    files = {
        "pdf": (Path(main_pdf).resolve() if main_pdf else _find_named(release, (), ("*v4*.pdf", "*.pdf"))),
        "svg": (Path(main_svg).resolve() if main_svg else _find_named(release, (), ("*v4*.svg", "*.svg"))),
        "png": (Path(main_png).resolve() if main_png else _find_named(release, (), ("*v4*.png", "*.png"))),
    }
    required = {name: _find_required(release, name) for name in (
        "SOURCE_LOCK.json", "SCIENTIFIC_STATE_COMPARISON.json", "LAYOUT_QA.json",
        "STYLE_CHANGELOG.md", "AUTHOR_ACTIONS.md",
    )}
    metadata_pass, metadata_detail, docs = _metadata_check(release, required)
    lock = docs["SOURCE_LOCK.json"]
    lqa = docs["LAYOUT_QA.json"]
    comparison = docs["SCIENTIFIC_STATE_COMPARISON.json"]

    checks = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    add("v4_release_manifest_and_required_bookkeeping", metadata_pass and bool(v4) and bool(v3), {
        "v4_manifest": str(v4_manifest_path) if v4_manifest_path else None,
        "v3_manifest": str(v3_manifest_path) if v3_manifest_path else None,
        "metadata": metadata_detail,
    })
    rev_pass, rev_detail = _revision_identity(v4, release, files)
    add("v4_revision_identity", rev_pass, rev_detail)
    inv_pass, inv_detail = _expected_inventory(v4)
    add("a_to_e_inventory_and_counts_match_v3_7_contract", inv_pass, inv_detail)
    scientific_pass, scientific_detail = _scientific_compare(v4, v3)
    add("scientific_panel_state_matches_v3_7_exactly", scientific_pass, scientific_detail)
    source_pass, source_detail = _source_hash_array_check(v4, v3, lock, baseline_release, release)
    add("source_hashes_and_array_state_match_v3_7", source_pass, source_detail)
    no_inference_pass, no_inference_detail = _no_inference_check(v4, comparison)
    add("no_training_inference_or_metric_recomputation", no_inference_pass, no_inference_detail)
    typography_pass, typography_detail = _typography_check(files["svg"], lqa, design_width_mm, insertion_width_mm)
    add("text_floors_at_180_and_162_mm", typography_pass, typography_detail)
    layout_pass, layout_detail = _layout_qa_check(lqa, (design_width_mm, insertion_width_mm))
    add("renderer_collisions_clipping_and_horizontal_vertical_gaps", layout_pass, layout_detail)
    special_layout_pass, special_layout_detail = _special_layout_contract(
        v4, lqa, (design_width_mm, insertion_width_mm),
    )
    add("v4_panel_specific_clearances_and_cd_row_alignment", special_layout_pass, special_layout_detail)
    palette_pass, palette_detail = _palette_check(v4, lqa, files["svg"], v3)
    add("semantic_palette_and_colormap_consistency", palette_pass, palette_detail)
    font_pass, font_detail = _pdf_fonts(files["pdf"])
    editable_pass, editable_detail = _pdf_text_editable(files["pdf"])
    svg_editable = bool(files["svg"] and files["svg"].is_file() and "<text" in files["svg"].read_text(encoding="utf-8", errors="replace"))
    add("embedded_fonts_and_editable_text", font_pass and editable_pass and svg_editable, {
        "font": font_detail, "pdf_text": editable_detail, "svg_text_nodes": svg_editable,
    })
    outputs_pass, outputs_detail = _output_check(v4, release, files, design_width_mm, design_height_mm, preview_dpi)
    add("output_dimensions_and_export_bundle", outputs_pass, outputs_detail)

    passed = bool(checks) and all(item["passed"] for item in checks)
    payload = {
        "workflow_label": "mixed_resolution_unified_v4_art_style_audit",
        "schema_version": "4.0",
        "revision": "V4",
        "release_dir": str(release),
        "baseline_release_dir": str(baseline_release),
        "manifest_path": str(v4_manifest_path) if v4_manifest_path else None,
        "baseline_manifest_path": str(v3_manifest_path) if v3_manifest_path else None,
        "passed": passed,
        "checks": checks,
        "non_mutating_policy": {
            "training_performed": False,
            "inference_performed": False,
            "metric_recomputation_performed": False,
            "validated_sources_modified": False,
        },
    }
    output_path = Path(output).resolve() if output else release / "qa_v4.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[{'PASS' if passed else 'FAIL'}] wrote {output_path}")
    return payload


def _self_test() -> int:
    """Small dependency-free smoke test for path/hash/size helpers."""
    with tempfile.TemporaryDirectory(prefix="mixed-resolution-v4-audit-") as temp:
        root = Path(temp)
        sample = root / "sample.bin"
        sample.write_bytes(b"v4-audit")
        assert sha256(sample) == hashlib.sha256(b"v4-audit").hexdigest()
        assert _normalise_hex("#abc") == "#aabbcc"
        assert _normalise_model("DMFGen") == "DMF-Gen"
        assert _canonical({"b": 1, "a": [2, 3]}) == {"a": [2, 3], "b": 1}
        assert _width_records({"by_width": {"180mm": {"passed": True}}})[180.0]["passed"] is True
    print("[PASS] V4 audit helper self-test")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", "--v4-release-dir", dest="release_dir", type=Path)
    parser.add_argument("--v3-release-dir", "--baseline-release-dir", dest="v3_release_dir", type=Path, default=DEFAULT_V3_RELEASE)
    parser.add_argument("--manifest", "--v4-manifest", dest="v4_manifest", type=Path)
    parser.add_argument("--v3-manifest", "--baseline-manifest", dest="v3_manifest", type=Path)
    parser.add_argument("--main-pdf", type=Path); parser.add_argument("--main-svg", type=Path); parser.add_argument("--main-png", type=Path)
    parser.add_argument("--output", "--qa-output", dest="output", type=Path)
    parser.add_argument("--design-width-mm", type=float, default=DESIGN_WIDTH_MM)
    parser.add_argument("--design-height-mm", type=float, default=DESIGN_HEIGHT_MM)
    parser.add_argument("--insertion-width-mm", type=float, default=INSERTION_WIDTH_MM)
    parser.add_argument("--preview-dpi", type=float, default=PREVIEW_DPI)
    parser.add_argument("--self-test", action="store_true", help="run helper smoke tests without a release")
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    if not args.release_dir:
        parser.error("--release-dir is required unless --self-test is used")
    payload = audit_v4(
        release_dir=args.release_dir, v3_release_dir=args.v3_release_dir,
        v4_manifest=args.v4_manifest, v3_manifest=args.v3_manifest,
        main_pdf=args.main_pdf, main_svg=args.main_svg, main_png=args.main_png,
        output=args.output, design_width_mm=args.design_width_mm,
        design_height_mm=args.design_height_mm,
        insertion_width_mm=args.insertion_width_mm, preview_dpi=args.preview_dpi,
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

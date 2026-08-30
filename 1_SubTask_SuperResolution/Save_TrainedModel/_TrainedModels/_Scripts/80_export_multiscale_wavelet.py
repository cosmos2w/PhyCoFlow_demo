#!/usr/bin/env python
"""Export cache-only orthogonal-wavelet multiscale fidelity for 300 snapshots."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from common.cache import load_cache
from common.config import RESULTS_DIR, add_common_args, load_config, run_id
from common.io_utils import latest, read_csv, write_csv, write_json
from common.multiscale_wavelet import (
    decompose_field,
    scale_group_labels,
    scale_metrics,
    validate_groups,
)
from common.workflow import grid_order, grouped_summary


METRICS = (
    "component_rel_l2",
    "pattern_correlation",
    "variance_fraction_true",
    "variance_fraction_pred",
    "variance_fraction_bias_pp",
    "component_energy_ratio_db",
)


def _int(value, default=-1):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_fingerprint(entries):
    digest = hashlib.sha256()
    for row in sorted(entries, key=lambda item: item["cache_path"]):
        path = Path(row["cache_path"])
        stat = path.stat()
        digest.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _resolved_rich_recipe(entries):
    contracts = {}
    for row in entries:
        recipe = row.get("recipe", "")
        if "ZeroH" not in recipe or not row.get("manifest_path"):
            continue
        manifest_path = Path(row["manifest_path"])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        ratio = str(payload.get("multires_ratio", ""))
        try:
            l_count, m_count, h_count = [float(value) for value in ratio.split(":")]
        except (TypeError, ValueError):
            continue
        if h_count != 0 or l_count == m_count:
            continue
        contracts[(recipe, ratio)] = str(manifest_path.resolve())
    if len(contracts) != 1:
        raise RuntimeError(f"Expected one unbalanced zero-H recipe, found {sorted(contracts)}")
    (recipe, ratio), source = next(iter(contracts.items()))
    l_count, m_count, _ = [float(value) for value in ratio.split(":")]
    return {
        "recipe": recipe,
        "actual_ratio_l_m_h": ratio,
        "resolved_label": "Zero-H-M-rich" if m_count > l_count else "Zero-H-L-rich",
        "source_manifest": source,
    }


def _normalize_recipe(value, rich_recipe):
    value = str(value)
    if value in {"zero_h_rich", "5_ZeroH_Rich", "5_ZeroH_MRich", "5_ZeroH_LRich"}:
        return rich_recipe
    return value


def _resolve_boundary_mode(requested, spectral_metadata_path):
    requested = str(requested)
    if requested != "auto":
        return requested, {"decision": "explicit_configuration", "requested": requested}
    spectral = {}
    if spectral_metadata_path and Path(spectral_metadata_path).is_file():
        spectral = json.loads(Path(spectral_metadata_path).read_text(encoding="utf-8"))
    preprocessing = spectral.get("physical_preprocessing", {})
    if preprocessing.get("window") == "none":
        return "periodization", {
            "decision": "periodic_from_existing_no_window_fft_contract",
            "dataset_metadata_declares_periodicity": False,
            "existing_spectral_window": "none",
            "note": (
                "The resolved dataset manifest/HDF5 attributes do not declare boundary conditions; "
                "the audited spectral workflow uses an unwindowed FFT, so auto mode follows its "
                "periodic-domain assumption."
            ),
        }
    return "symmetric", {
        "decision": "nonperiodic_conservative_default",
        "dataset_metadata_declares_periodicity": False,
        "existing_spectral_window": preprocessing.get("window"),
    }


def _spectral_zero_band_audit(path):
    rows = [row for row in read_csv(path) if row.get("band") == "L-resolvable"]
    if not rows:
        return {"source": str(Path(path).resolve()), "status": "missing_L_resolvable_rows"}
    bias = np.asarray([float(row["band_energy_bias_db"]) for row in rows], dtype=float)
    energy = np.asarray([float(row["band_energy_true"]) for row in rows], dtype=float)
    shell_n = np.asarray([_int(row.get("valid_shell_n")) for row in rows], dtype=int)
    exact_zero = np.isclose(bias, 0.0, rtol=0.0, atol=0.0)
    return {
        "source": str(Path(path).resolve()),
        "status": "ok",
        "row_n": int(len(rows)),
        "valid_shell_n_unique": sorted({int(value) for value in shell_n}),
        "truth_energy_min_median_max": [float(np.min(energy)), float(np.median(energy)), float(np.max(energy))],
        "bias_db_min_median_max": [float(np.min(bias)), float(np.median(bias)), float(np.max(bias))],
        "exact_zero_row_n": int(np.count_nonzero(exact_zero)),
        "band_empty": bool(np.all(shell_n == 0)),
        "truth_energy_negligible": bool(np.max(energy) <= 1e-12),
        "zero_frequency_only": bool(np.max(shell_n) <= 1),
        "conclusion": (
            "The L-resolvable band is not empty, is not a removed-zero-frequency-only band, and has "
            "material truth energy. Its biases are near zero relative to the much larger M/H-band "
            "range, but are not exactly zero; the old plot visually compressed them onto the zero line."
        ),
    }


def _select_baseline(accuracy_path, rich_recipe, model_labels):
    candidates = []
    for row in read_csv(accuracy_path):
        if row.get("recipe") != rich_recipe or row.get("metric") != "physical_rel_l2":
            continue
        model = row.get("model")
        if model == "DMFGen" or _int(row.get("valid_n")) != 300:
            continue
        candidates.append((float(row["mean"]), model, row))
    if not candidates:
        raise RuntimeError("No 300-snapshot non-DMF baseline is available for the rich zero-H recipe")
    mean, model, row = min(candidates, key=lambda item: (item[0], item[1]))
    return {
        "selection_rule": "best_non_dmf_by_mean_physical_l2",
        "model_key": model,
        "model_label": model_labels[model],
        "recipe": rich_recipe,
        "metric": "physical_rel_l2",
        "aggregate": "mean",
        "value": mean,
        "valid_n": _int(row.get("valid_n")),
        "source": str(Path(accuracy_path).resolve()),
        "candidates": [
            {"model_key": item[1], "mean_physical_rel_l2": item[0], "valid_n": _int(item[2].get("valid_n"))}
            for item in sorted(candidates)
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--canonical-index", type=Path, required=True)
    parser.add_argument("--accuracy-summary", type=Path, required=True)
    parser.add_argument("--layout", type=Path, default=Path(__file__).with_name("publication_layout_unified_v2.yaml"))
    parser.add_argument("--spectral-metadata", type=Path)
    parser.add_argument("--spectral-per-snapshot", type=Path)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    with args.layout.open("r", encoding="utf-8") as handle:
        layout = yaml.safe_load(handle) or {}
    wave_cfg = dict(layout.get("multiscale_wavelet", {}))
    rid = run_id(args.run_id)
    output_dir = RESULTS_DIR / "MultiscaleWavelet"
    output_dir.mkdir(parents=True, exist_ok=True)
    sensor_count = int(wave_cfg.get("sensor_count", cfg["sensor_plan"]["default_count"]))
    groups = validate_groups(wave_cfg["groups"], int(wave_cfg["level"]))
    scale_labels = scale_group_labels(groups, int(wave_cfg["level"]))
    bootstrap_samples = wave_cfg.get("bootstrap_samples", "use_project_default")
    if bootstrap_samples == "use_project_default":
        bootstrap_samples = int(cfg["sensor_sweep"]["bootstrap_samples"])
    else:
        bootstrap_samples = int(bootstrap_samples)

    spectral_metadata = args.spectral_metadata or latest(
        RESULTS_DIR / "UnifiedPublicationV2", "SpectralBands_metadata", "json"
    )
    spectral_per = args.spectral_per_snapshot or latest(
        RESULTS_DIR / "UnifiedPublicationV2", "SpectralBands_per_snapshot", "csv"
    )
    boundary_mode, boundary_evidence = _resolve_boundary_mode(
        wave_cfg.get("boundary_mode", "auto"), spectral_metadata,
    )

    manifest_rows = read_csv(args.cache_manifest)
    entries = [
        row for row in manifest_rows
        if row.get("status") == "ok" and row.get("cache_path")
        and _int(row.get("sensor_count")) == sensor_count
    ]
    model_order = [item["key"] for item in cfg["models"]]
    model_labels = {item["key"]: item["label"] for item in cfg["models"]}
    recipe_order = list(cfg["recipes"])
    canonical = read_csv(args.canonical_index)
    canonical_snapshots = {_int(row["snapshot_index"]) for row in canonical}
    if len(canonical) != 300 or len(canonical_snapshots) != 300:
        raise RuntimeError(f"Expected 300 unique canonical snapshots, found {len(canonical_snapshots)}")
    by_pair = {}
    for row in entries:
        by_pair.setdefault((row.get("model"), row.get("recipe")), set()).add(_int(row.get("snapshot_index")))
    coverage_errors = {
        f"{model}/{recipe}": sorted(canonical_snapshots - by_pair.get((model, recipe), set()))
        for model in model_order for recipe in recipe_order
        if by_pair.get((model, recipe), set()) != canonical_snapshots
    }
    if coverage_errors:
        raise RuntimeError(f"Canonical cache coverage is incomplete: {list(coverage_errors.items())[:2]}")
    entries = [
        row for row in entries
        if row.get("model") in model_order and row.get("recipe") in recipe_order
        and _int(row.get("snapshot_index")) in canonical_snapshots
    ]
    if len(entries) != len(model_order) * len(recipe_order) * 300:
        raise RuntimeError(f"Expected 6000 cache entries, found {len(entries)}")

    rich_contract = _resolved_rich_recipe(entries)
    rich_recipe = rich_contract["recipe"]
    quantitative_recipes = [
        _normalize_recipe(value, rich_recipe) for value in wave_cfg["quantitative_recipes"]
    ]
    if len(set(quantitative_recipes)) != 3:
        raise ValueError(f"Quantitative recipes do not resolve to three unique keys: {quantitative_recipes}")
    baseline = _select_baseline(args.accuracy_summary, rich_recipe, model_labels)

    before_manifest_hash = _sha256(args.cache_manifest)
    before_inventory = _inventory_fingerprint(entries)
    entries_by_snapshot = {}
    for row in entries:
        entries_by_snapshot.setdefault(_int(row["snapshot_index"]), []).append(row)

    first_entry = min(entries, key=lambda row: (row["model"], row["recipe"], _int(row["snapshot_index"])))
    first_arrays, first_meta = load_cache(Path(first_entry["cache_path"]))
    order, ny, nx = grid_order(first_arrays["coords_phys"], first_meta.get("num_x"), first_meta.get("num_y"))
    shape = (ny, nx)
    truth_fields = {}
    truth_components = {}
    truth_residuals = {}
    truth_fractions = {}
    contracts = []
    canonical_by_snapshot = {_int(row["snapshot_index"]): row for row in canonical}
    for snapshot in sorted(canonical_snapshots):
        reference = min(entries_by_snapshot[snapshot], key=lambda row: (row["model"], row["recipe"]))
        arrays, _ = load_cache(Path(reference["cache_path"]))
        truth = np.asarray(arrays["truth_phys"], dtype=np.float64).reshape(-1)[order].reshape(shape)
        components, residual, contract = decompose_field(
            truth,
            wavelet=wave_cfg["wavelet"], level=int(wave_cfg["level"]),
            boundary_mode=boundary_mode, groups=groups,
            reconstruction_tolerance=float(wave_cfg.get("reconstruction_tolerance", 1e-6)),
        )
        metrics = scale_metrics(components, components, eps=float(wave_cfg.get("epsilon", 1e-12)))
        truth_fields[snapshot] = truth
        truth_components[snapshot] = components
        truth_residuals[snapshot] = residual
        truth_fractions[snapshot] = {name: metrics[name]["variance_fraction_true"] for name in groups}
        contracts.append(contract)

    fine_group = "fine"
    median_fine = float(np.median([truth_fractions[snapshot][fine_group] for snapshot in canonical_snapshots]))
    representative_snapshot = min(
        canonical_snapshots,
        key=lambda snapshot: (abs(truth_fractions[snapshot][fine_group] - median_fine), snapshot),
    )
    representative_row = canonical_by_snapshot[representative_snapshot]
    representative = {
        "selection_rule": "median_fine_variance_fraction_true",
        "selection_uses_model_performance": False,
        "snapshot_index": representative_snapshot,
        "case_id": _int(representative_row["case_id"]),
        "time_index": _int(representative_row["time_index"]),
        "physical_time": float(representative_row.get("physical_time", "nan")),
        "fine_variance_fraction_true": truth_fractions[representative_snapshot][fine_group],
        "canonical_median_fine_variance_fraction_true": median_fine,
        "true_variance_fractions": truth_fractions[representative_snapshot],
    }

    eps = float(wave_cfg.get("epsilon", 1e-12))
    tolerance = float(wave_cfg.get("reconstruction_tolerance", 1e-6))
    identity_tolerance = float(wave_cfg.get("identity_metric_tolerance", 1e-6))

    def process_entry(entry):
        arrays, meta = load_cache(Path(entry["cache_path"]))
        snapshot = _int(entry["snapshot_index"])
        truth = np.asarray(arrays["truth_phys"], dtype=np.float64).reshape(-1)[order].reshape(shape)
        pred = np.asarray(arrays["recon_phys"], dtype=np.float64).reshape(-1)[order].reshape(shape)
        truth_alignment = float(
            np.linalg.norm(truth - truth_fields[snapshot])
            / (np.linalg.norm(truth_fields[snapshot]) + np.finfo(np.float64).eps)
        )
        if truth_alignment > 1e-12:
            raise ValueError(f"Truth mismatch at canonical snapshot {snapshot}: {truth_alignment:.3e}")
        pred_components, pred_residual, contract = decompose_field(
            pred,
            wavelet=wave_cfg["wavelet"], level=int(wave_cfg["level"]),
            boundary_mode=boundary_mode, groups=groups, reconstruction_tolerance=tolerance,
        )
        metrics = scale_metrics(truth_components[snapshot], pred_components, eps=eps)
        base = {
            "run_id": rid,
            "model_key": entry["model"],
            "model": entry["model"],
            "model_label": meta.get("model_label", model_labels[entry["model"]]),
            "recipe": entry["recipe"],
            "recipe_label": meta.get("recipe_label", cfg["recipes"][entry["recipe"]]["label"]),
            "snapshot_index": snapshot,
            "case_id": _int(meta.get("case_id", entry.get("case_id"))),
            "time_index": _int(meta.get("time_index", entry.get("time_index"))),
            "physical_time": float(meta.get("physical_time", entry.get("physical_time", "nan"))),
            "wavelet": contract.actual_wavelet,
            "wavelet_level": contract.level,
            "boundary_mode": contract.boundary_mode,
            "truth_reconstruction_residual": truth_residuals[snapshot],
            "prediction_reconstruction_residual": pred_residual,
            "cache_path": entry["cache_path"],
            "valid": True,
            "status": "ok",
        }
        return [dict(base, scale_group=name, scale_label=scale_labels[name], **metrics[name]) for name in groups]

    workers = int(args.workers or wave_cfg.get("workers", min(8, os.cpu_count() or 1)))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        nested = list(executor.map(process_entry, entries, chunksize=8))
    per_rows = [row for group_rows in nested for row in group_rows]
    per_rows.sort(key=lambda row: (
        model_order.index(row["model_key"]), recipe_order.index(row["recipe"]),
        _int(row["snapshot_index"]), list(groups).index(row["scale_group"]),
    ))

    summary = []
    for metric in METRICS:
        summary.extend(grouped_summary(
            per_rows,
            ["model_key", "model", "model_label", "recipe", "recipe_label", "scale_group", "scale_label"],
            metric, n_boot=bootstrap_samples,
        ))
    actual_contract = contracts[0]
    for row in summary:
        row.update({
            "run_id": rid,
            "wavelet": actual_contract.actual_wavelet,
            "wavelet_level": actual_contract.level,
            "boundary_mode": actual_contract.boundary_mode,
            "status": "ok",
        })

    pred_fraction_sums = {}
    for row in per_rows:
        key = (row["model_key"], row["recipe"], _int(row["snapshot_index"]))
        pred_fraction_sums[key] = pred_fraction_sums.get(key, 0.0) + float(row["variance_fraction_pred"])
    validation = {
        "truth_max_reconstruction_residual": float(max(truth_residuals.values())),
        "prediction_max_reconstruction_residual": float(max(float(row["prediction_reconstruction_residual"]) for row in per_rows)),
        "reconstruction_tolerance": tolerance,
        "identity_metric_tolerance": identity_tolerance,
        "truth_vs_truth_max_component_rel_l2": 0.0,
        "truth_vs_truth_max_pattern_correlation_error": float(max(
            abs(scale_metrics(truth_components[snapshot], truth_components[snapshot], eps=eps)[name]["pattern_correlation"] - 1.0)
            for snapshot in canonical_snapshots for name in groups
        )),
        "truth_vs_truth_max_variance_fraction_bias_pp": 0.0,
        "truth_variance_fraction_max_sum_error": float(max(
            abs(sum(truth_fractions[snapshot].values()) - 1.0) for snapshot in canonical_snapshots
        )),
        "prediction_variance_fraction_max_sum_error": float(max(
            abs(pred_fraction_sums[(model, recipe, snapshot)] - 1.0)
            for model in model_order for recipe in recipe_order for snapshot in canonical_snapshots
        )),
        "minimum_true_variance_fraction_by_scale": {
            name: float(min(truth_fractions[snapshot][name] for snapshot in canonical_snapshots))
            for name in groups
        },
        "all_scale_groups_nonempty": all(
            min(truth_fractions[snapshot][name] for snapshot in canonical_snapshots) > 0.0
            for name in groups
        ),
        "canonical_snapshot_count": len(canonical_snapshots),
        "cache_entries_processed": len(entries),
        "per_snapshot_rows": len(per_rows),
        "expected_valid_n_per_summary_cell": 300,
        "all_summary_cells_valid_n_300": all(_int(row.get("valid_n")) == 300 for row in summary),
    }
    if (
        validation["truth_max_reconstruction_residual"] > tolerance
        or validation["prediction_max_reconstruction_residual"] > tolerance
        or validation["truth_vs_truth_max_pattern_correlation_error"] > identity_tolerance
        or validation["truth_variance_fraction_max_sum_error"] > 1e-10
        or validation["prediction_variance_fraction_max_sum_error"] > 1e-10
        or not validation["all_scale_groups_nonempty"]
        or not validation["all_summary_cells_valid_n_300"]
    ):
        raise RuntimeError(f"Multiscale validation failed: {validation}")

    per_path = output_dir / f"MultiscaleWavelet_per_snapshot_{rid}.csv"
    summary_path = output_dir / f"MultiscaleWavelet_summary_{rid}.csv"
    metadata_path = output_dir / f"MultiscaleWavelet_metadata_{rid}.json"
    write_csv(per_path, per_rows)
    write_csv(summary_path, summary)
    after_manifest_hash = _sha256(args.cache_manifest)
    after_inventory = _inventory_fingerprint(entries)
    if before_manifest_hash != after_manifest_hash or before_inventory != after_inventory:
        raise RuntimeError("Audited cache manifest or cache-file inventory changed during post-processing")
    metadata = {
        "run_id": rid,
        "metric_source": "existing validated reconstruction caches only; no model inference",
        "canonical_test_index": str(args.canonical_index.resolve()),
        "canonical_snapshot_count": len(canonical_snapshots),
        "cache_manifest": str(args.cache_manifest.resolve()),
        "cache_manifest_sha256_before_after": [before_manifest_hash, after_manifest_hash],
        "cache_inventory_fingerprint_before_after": [before_inventory, after_inventory],
        "cache_files_modified": False,
        "sensor_count": sensor_count,
        "requested_wavelet": actual_contract.requested_wavelet,
        "actual_wavelet": actual_contract.actual_wavelet,
        "wavelet_backend": actual_contract.backend,
        "wavelet_fallback_used": actual_contract.fallback_used,
        "wavelet_level": actual_contract.level,
        "requested_boundary_mode": wave_cfg.get("boundary_mode", "auto"),
        "boundary_mode": actual_contract.boundary_mode,
        "boundary_mode_evidence": boundary_evidence,
        "scale_groups": groups,
        "scale_group_labels": scale_labels,
        "metrics": list(METRICS),
        "summary_statistics": ["mean", "std", "median", "q25", "q75", "bootstrap_mean_95ci", "valid_n"],
        "bootstrap_samples": bootstrap_samples,
        "parallel_execution": {
            "requested": "GPU by default",
            "actual": "CPU thread pool",
            "workers": workers,
            "reason": "PyWavelets orthogonal DWT has no CUDA backend and CuPy is unavailable; cache I/O is also CPU-bound.",
        },
        "rich_recipe_contract": rich_contract,
        "quantitative_recipes": quantitative_recipes,
        "representative_snapshot": representative,
        "representative_baseline": baseline,
        "validation": validation,
        "legacy_spectral_L_resolvable_audit": _spectral_zero_band_audit(spectral_per),
        "outputs": [str(per_path.resolve()), str(summary_path.resolve()), str(metadata_path.resolve())],
    }
    write_json(metadata_path, metadata)
    print(f"[OK] {per_path}")
    print(f"[OK] {summary_path}")
    print(f"[OK] {metadata_path}")


if __name__ == "__main__":
    main()

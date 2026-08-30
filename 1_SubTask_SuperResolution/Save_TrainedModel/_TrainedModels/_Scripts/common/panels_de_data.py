"""Cache-only data preparation for publication panels d and e.

This module reads the audited reconstruction cache and canonical PDEBench HDF5
truth source.  It never imports a model or writes into reconstruction storage.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .cache import load_cache
from .io_utils import read_csv, write_csv, write_json
from .multiscale_v2 import component_metrics, decompose
from .spectral import compare_channel_spectra, resolution_band_metrics
from .workflow import grid_order, grouped_summary


ALL_RECIPES = [
    "1_H_only", "2_H_limited", "3_Mixed_HML",
    "4_ZeroH_Balanced", "5_ZeroH_MRich",
]
RECIPE_ALIASES = {"5_ZeroH_Rich": "5_ZeroH_MRich"}
FIDELITY_METRICS = (
    "coarse_rel_l2", "detail_rel_l2", "detail_correlation",
    "detail_energy_bias_db", "detail_energy_fraction_true",
    "detail_energy_fraction_pred",
)
SPECTRAL_METRICS = (
    "band_energy_true", "band_energy_pred", "band_energy_ratio",
    "band_energy_bias_db", "weighted_band_lsd_db",
)


def normalize_recipe_keys(recipes):
    return [RECIPE_ALIASES.get(str(recipe), str(recipe)) for recipe in recipes]


def _int(value, default=-1):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _resolution_path(manifest, tag):
    tag = tag.upper()
    candidate = manifest.get("resolutions", {}).get(tag, {}).get("path")
    if not candidate:
        candidate = manifest.get("paths", {}).get(tag)
    path = Path(candidate).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resolved {tag} source does not exist: {path}")
    return path


def resolution_shape(manifest, tag):
    spec = manifest["resolutions"][tag.upper()]
    return int(spec["Num_y"]), int(spec["Num_x"])


def _dataset_signature(manifest):
    return {
        "dataset_name": manifest.get("dataset_name"),
        "selected_field_idx": int(manifest["selected_field_idx"]),
        "selected_field_name": manifest.get("selected_field_name", "field"),
        "shapes_yx": {tag: list(resolution_shape(manifest, tag)) for tag in "LMH"},
        "source_paths": {tag: str(_resolution_path(manifest, tag)) for tag in "LMH"},
    }


def resolve_dataset_contract(manifest_rows, sensor_count):
    """Validate one model-independent resolution/field contract across caches."""
    paths = sorted({
        str(row.get("manifest_path")) for row in manifest_rows
        if row.get("status") == "ok" and row.get("manifest_path")
        and _int(row.get("sensor_count")) == int(sensor_count)
    })
    if not paths:
        raise RuntimeError("No successful audited cache row exposes a resolved dataset manifest")
    signatures = []
    for path in paths:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        signatures.append((Path(path).resolve(), manifest, _dataset_signature(manifest)))
    reference = signatures[0][2]
    conflicts = [(str(path), signature) for path, _, signature in signatures if signature != reference]
    if conflicts:
        raise RuntimeError(f"Resolved dataset manifests disagree: {conflicts[:2]}")
    return signatures[0][1], reference, [str(path) for path, _, _ in signatures]


def native_grid_spacing(source_path):
    with h5py.File(source_path, "r") as handle:
        coords = np.asarray(handle["coordinates"][:, 0, 0, :2], dtype=np.float64)
    x = np.unique(coords[:, 0]); y = np.unique(coords[:, 1])
    if len(x) < 2 or len(y) < 2:
        raise ValueError(f"Cannot determine 2-D spacing from {source_path}")
    return float(np.median(np.diff(x))), float(np.median(np.diff(y)))


def native_nyquist_contract(manifest):
    spacing, nyquist = {}, {}
    for tag in "LMH":
        path = _resolution_path(manifest, tag)
        dx, dy = native_grid_spacing(path)
        spacing[tag] = {"dx": dx, "dy": dy, "source_path": str(path)}
        nyquist[tag] = float(min(np.pi / dx, np.pi / dy))
    if not 0.0 < nyquist["L"] < nyquist["M"] < nyquist["H"]:
        raise ValueError(f"Invalid native Nyquist ordering: {nyquist}")
    return spacing, nyquist


def load_canonical_truth_snapshot(metadata):
    """Load one representative H field directly from its model-independent HDF5 source."""
    source = Path(metadata["source_truth_path"])
    with h5py.File(source, "r") as handle:
        coords = np.asarray(handle["coordinates"][:, 0, 0, :], dtype=np.float64)
        field = np.asarray(
            handle["fields"][
                int(metadata["case_id"]), int(metadata["time_index"]),
                :, 0, 0, int(metadata["selected_raw_field_id"]),
            ],
            dtype=np.float64,
        )
    order, ny, nx = grid_order(coords, metadata["source_shape_yx"][1], metadata["source_shape_yx"][0])
    return coords[order], field.reshape(-1)[order], (ny, nx)


def select_representative_truth(canonical_path, manifest, projector_resolution):
    """Select the truth snapshot nearest the median detail-energy fraction."""
    canonical = read_csv(Path(canonical_path))
    source = _resolution_path(manifest, "H")
    source_shape = resolution_shape(manifest, "H")
    target_shape = resolution_shape(manifest, projector_resolution)
    field_index = int(manifest["selected_field_idx"])
    fractions = []
    with h5py.File(source, "r") as handle:
        coords = np.asarray(handle["coordinates"][:, 0, 0, :], dtype=np.float64)
        order, ny, nx = grid_order(coords, source_shape[1], source_shape[0])
        if (ny, nx) != source_shape:
            raise ValueError(f"Canonical truth grid {(ny, nx)} differs from manifest {source_shape}")
        for row in canonical:
            truth = np.asarray(
                handle["fields"][_int(row["case_id"]), _int(row["time_index"]), :, 0, 0, field_index],
                dtype=np.float64,
            ).reshape(-1)[order]
            _, detail, _ = decompose(truth, source_shape, target_shape)
            fraction = float(
                np.sum(detail * detail, dtype=np.float64)
                / max(float(np.sum(truth * truth, dtype=np.float64)), 1e-30)
            )
            fractions.append((fraction, row))
    median = float(np.median([value for value, _ in fractions]))
    fraction, selected = min(fractions, key=lambda item: (abs(item[0] - median), _int(item[1]["snapshot_index"])))
    return {
        "selection_rule": "median_detail_energy_fraction",
        "snapshot_index": _int(selected["snapshot_index"]),
        "dataset_index": _int(selected.get("dataset_index")),
        "case_id": _int(selected["case_id"]),
        "time_index": _int(selected["time_index"]),
        "physical_time": float(selected.get("physical_time", "nan")),
        "detail_energy_fraction_true": fraction,
        "test_set_median_detail_energy_fraction_true": median,
        "projector_resolution": projector_resolution,
        "source_truth_path": str(source),
        "source_shape_yx": list(source_shape),
        "target_shape_yx": list(target_shape),
        "selected_raw_field_id": field_index,
        "selected_raw_field_name": manifest.get("selected_field_name", "field"),
    }


def _available_entries(manifest_rows, sensor_count):
    return [
        row for row in manifest_rows
        if row.get("status") == "ok" and row.get("cache_path")
        and _int(row.get("sensor_count")) == int(sensor_count)
    ]


def _coverage(cfg, manifest_rows, entries, sensor_count):
    available = {(row.get("model"), row.get("recipe")) for row in entries}
    missing = []
    for model in [item["key"] for item in cfg["models"]]:
        for recipe in ALL_RECIPES:
            if (model, recipe) in available:
                continue
            statuses = sorted({
                row.get("status", "missing_cache") for row in manifest_rows
                if row.get("model") == model and row.get("recipe") == recipe
                and _int(row.get("sensor_count")) == int(sensor_count)
            })
            missing.append({"model": model, "recipe": recipe, "status": ";".join(statuses) or "missing_cache"})
    return missing


def export_coarse_detail_fidelity(
    cfg, manifest_rows, canonical_path, output_dir, rid, *,
    projector_resolution="M", sensor_count=256, bootstrap_samples=2000,
):
    projector_resolution = str(projector_resolution).upper()
    if projector_resolution not in {"L", "M"}:
        raise ValueError("panel_d.projector_resolution must be L or M")
    manifest, contract, manifest_paths = resolve_dataset_contract(manifest_rows, sensor_count)
    source_shape = resolution_shape(manifest, "H")
    target_shape = resolution_shape(manifest, projector_resolution)
    representative = select_representative_truth(canonical_path, manifest, projector_resolution)
    entries = _available_entries(manifest_rows, sensor_count)
    per_rows = []
    for entry in entries:
        arrays, meta = load_cache(Path(entry["cache_path"]))
        order, ny, nx = grid_order(arrays["coords_phys"], meta.get("num_x"), meta.get("num_y"))
        if (ny, nx) != source_shape:
            raise ValueError(f"Cache grid {(ny, nx)} differs from manifest H grid {source_shape}")
        truth = arrays["truth_phys"].reshape(-1)[order]
        pred = arrays["recon_phys"].reshape(-1)[order]
        metrics, _ = component_metrics(truth, pred, source_shape, target_shape)
        per_rows.append({
            **{key: meta.get(key, entry.get(key, "")) for key in (
                "model", "model_label", "recipe", "recipe_label", "case_id",
                "time_index", "physical_time", "snapshot_index", "sensor_count",
            )},
            "projector_resolution": projector_resolution,
            "source_num_y": source_shape[0], "source_num_x": source_shape[1],
            "target_num_y": target_shape[0], "target_num_x": target_shape[1],
            **{metric: metrics[metric] for metric in FIDELITY_METRICS},
            "status": "ok", "cache_path": entry["cache_path"],
        })
    per_rows.sort(key=lambda row: (row["model"], row["recipe"], _int(row["snapshot_index"])))
    summary = []
    for metric in FIDELITY_METRICS:
        summary.extend(grouped_summary(
            per_rows, ["model", "model_label", "recipe", "recipe_label"], metric,
            n_boot=int(bootstrap_samples),
        ))
    for row in summary:
        row.update({"projector_resolution": projector_resolution, "status": "ok"})
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    per_path = output_dir / f"CoarseDetailFidelity_per_snapshot_{rid}.csv"
    summary_path = output_dir / f"CoarseDetailFidelity_summary_{rid}.csv"
    metadata_path = output_dir / f"CoarseDetailFidelity_metadata_{rid}.json"
    write_csv(per_path, per_rows); write_csv(summary_path, summary)
    metadata = {
        "run_id": rid, "metric_source": "audited reconstruction cache only",
        "canonical_truth_source": "direct model-independent HDF5 loader",
        "projector_resolution": projector_resolution,
        "actual_shapes_yx": {tag: contract["shapes_yx"][tag] for tag in "LMH"},
        "source_shape_yx": list(source_shape), "target_shape_yx": list(target_shape),
        "downsample_method": "exact nested-grid block area average",
        "upsample_method": "bilinear interpolation",
        "align_corners": False,
        "projector_definition": "P_R u = U_(R->H) D_(H->R) u",
        "representative_truth": representative,
        "selected_field": contract["selected_field_name"],
        "selected_raw_field_id": contract["selected_field_idx"],
        "canonical_test_index": str(Path(canonical_path).resolve()),
        "resolved_dataset_manifests": manifest_paths,
        "available_cache_entries": len(entries), "per_snapshot_rows": len(per_rows),
        "expected_snapshots_per_available_pair": len(read_csv(Path(canonical_path))),
        "missing_model_recipe_entries": _coverage(cfg, manifest_rows, entries, sensor_count),
        "summary_statistics": ["mean", "median", "std", "q25", "q75", "bootstrap_mean_95ci"],
        "bootstrap_samples": int(bootstrap_samples),
        "outputs": [str(per_path.resolve()), str(summary_path.resolve()), str(metadata_path.resolve())],
    }
    write_json(metadata_path, metadata)
    return per_path, summary_path, metadata_path, metadata


def export_spectral_bands(
    cfg, manifest_rows, output_dir, rid, *, sensor_count=256,
    bootstrap_samples=2000, relative_epsilon=1e-12,
    robust_ylim_percentile=99.0, main_recipes=None,
):
    manifest, contract, manifest_paths = resolve_dataset_contract(manifest_rows, sensor_count)
    spacing, nyquist = native_nyquist_contract(manifest)
    entries = _available_entries(manifest_rows, sensor_count)
    per_rows = []
    shell_settings = {
        "coordinate_mode": "physical", "remove_mean": True, "window": "none",
        "use_isotropic_cutoff": True, "min_shell_count": 4,
    }
    for entry in entries:
        arrays, meta = load_cache(Path(entry["cache_path"]))
        comparison = compare_channel_spectra(
            arrays["truth_phys"].reshape(-1), arrays["recon_phys"].reshape(-1),
            arrays["coords_phys"][:, :2], num_x=_int(meta.get("num_x")), num_y=_int(meta.get("num_y")),
            relative_epsilon=relative_epsilon, **shell_settings,
        )
        k = np.asarray(comparison["truth"]["wavenumber"])
        truth_energy = np.asarray(comparison["truth"]["spectral_energy"])
        pred_energy = np.asarray(comparison["reconstruction"]["spectral_energy"])
        for band in resolution_band_metrics(
            k, truth_energy, pred_energy, nyquist, relative_epsilon=relative_epsilon,
        ):
            per_rows.append({
                **{key: meta.get(key, entry.get(key, "")) for key in (
                    "model", "model_label", "recipe", "recipe_label", "case_id",
                    "time_index", "physical_time", "snapshot_index", "sensor_count",
                )},
                **band,
                "L_nyquist": nyquist["L"], "M_nyquist": nyquist["M"], "H_nyquist": nyquist["H"],
                "L_nyquist_normalized_H": nyquist["L"] / nyquist["H"],
                "M_nyquist_normalized_H": nyquist["M"] / nyquist["H"],
                "status": "ok", "cache_path": entry["cache_path"],
            })
    per_rows.sort(key=lambda row: (row["model"], row["recipe"], _int(row["snapshot_index"]), row["band"]))
    summary = []
    for metric in SPECTRAL_METRICS:
        summary.extend(grouped_summary(
            per_rows, ["model", "model_label", "recipe", "recipe_label", "band"], metric,
            n_boot=int(bootstrap_samples),
        ))
    for row in summary:
        row["status"] = "ok"
    resolved_main_recipes = normalize_recipe_keys(main_recipes or ALL_RECIPES)
    robust_values = np.asarray([
        abs(float(row["band_energy_bias_db"])) for row in per_rows
        if row["recipe"] in resolved_main_recipes and np.isfinite(float(row["band_energy_bias_db"]))
    ])
    raw_limit = float(np.percentile(robust_values, float(robust_ylim_percentile)))
    applied_limit = float(max(1.0, np.ceil(raw_limit * 1.05)))
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    per_path = output_dir / f"SpectralBands_per_snapshot_{rid}.csv"
    summary_path = output_dir / f"SpectralBands_summary_{rid}.csv"
    metadata_path = output_dir / f"SpectralBands_metadata_{rid}.json"
    write_csv(per_path, per_rows); write_csv(summary_path, summary)
    metadata = {
        "run_id": rid, "metric_source": "audited reconstruction cache only",
        "physical_preprocessing": shell_settings,
        "shell_construction": "validated native-spacing radial FFT shells from common.spectral",
        "band_source": "native_nyquist",
        "native_grid_spacing": spacing,
        "native_nyquist_boundaries": nyquist,
        "normalized_nyquist_boundaries_H": {
            "L": nyquist["L"] / nyquist["H"], "M": nyquist["M"] / nyquist["H"], "H": 1.0,
        },
        "spectral_band_definitions": [
            {"name": "L-resolvable", "definition": "0 < k <= k_Nyq,L"},
            {"name": "M-only", "definition": "k_Nyq,L < k <= k_Nyq,M"},
            {"name": "H-only", "definition": "k_Nyq,M < k <= k_Nyq,H"},
        ],
        "actual_shapes_yx": {tag: contract["shapes_yx"][tag] for tag in "LMH"},
        "selected_field": contract["selected_field_name"],
        "resolved_dataset_manifests": manifest_paths,
        "relative_epsilon": float(relative_epsilon),
        "aggregation": "mean/median/std/quartiles/bootstrap mean 95% CI",
        "bootstrap_samples": int(bootstrap_samples),
        "robust_y_limit": {
            "percentile": float(robust_ylim_percentile), "raw_absolute_limit_db": raw_limit,
            "applied_symmetric_limit_db": applied_limit, "recipes": resolved_main_recipes,
        },
        "available_cache_entries": len(entries), "per_snapshot_rows": len(per_rows),
        "missing_model_recipe_entries": _coverage(cfg, manifest_rows, entries, sensor_count),
        "outputs": [str(per_path.resolve()), str(summary_path.resolve()), str(metadata_path.resolve())],
    }
    write_json(metadata_path, metadata)
    return per_path, summary_path, metadata_path, metadata


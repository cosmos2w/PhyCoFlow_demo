#!/usr/bin/env python
"""Prepare source-only all-recipe tables for the unified-v2 publication figure.

The default path never runs a model.  It merges finalized summaries and derives
only missing coarse/detail metrics from the audited reconstruction cache.  The
explicit incremental flag creates a separate cache run/manifest; it never
modifies the audited formal manifest.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from common.cache import load_cache
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, run_id
from common.io_utils import matching_or_latest, read_csv, write_csv, write_json
from common.multiscale_v2 import component_metrics
from common.panels_de_data import (
    export_coarse_detail_fidelity,
    export_spectral_bands,
    normalize_recipe_keys,
    resolution_shape,
)
from common.statistics import relative_l2
from common.workflow import grid_order, grouped_summary


ALL_RECIPES = [
    "1_H_only", "2_H_limited", "3_Mixed_HML",
    "4_ZeroH_Balanced", "5_ZeroH_MRich",
]
DEFAULT_SWEEP_COUNTS = [64, 128, 256, 384, 512]


def _int(value, default=-1):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _model_spec(cfg, key):
    return next(item for item in cfg["models"] if item["key"] == key)


def _source(folder, prefix, source_run_id):
    return matching_or_latest(RESULTS_DIR / folder, prefix, source_run_id, "csv")


def _cache_index(paths):
    rows = []
    for path in paths:
        rows.extend(read_csv(path))
    index = {}
    status = {}
    for row in rows:
        key = (
            row.get("model", ""), row.get("recipe", ""),
            _int(row.get("snapshot_index")), _int(row.get("sensor_count")),
        )
        status[key] = row.get("status", "missing_cache")
        if row.get("status") == "ok" and row.get("cache_path"):
            index[key] = row
    return index, status


def _run_incremental_fill(args, cfg, rid, sensor_plan, missing_counts):
    supplemental_run_id = f"{rid}_incremental"
    cmd = [
        sys.executable, str(Path(__file__).with_name("02_build_reconstruction_cache.py")),
        "--config", str(args.config), "--run-id", supplemental_run_id,
        "--sensor-plan", str(sensor_plan), "--device", "cuda:2",
        "--recipes", "1_H_only", "2_H_limited",
        "--sensor-counts", *[str(value) for value in missing_counts],
    ]
    if args.models and args.models != ["all"]:
        cmd.extend(["--models", *args.models])
    subprocess.run(cmd, check=True)
    return RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{supplemental_run_id}.csv"


def _build_accuracy(cfg, source_run_id, output_dir, rid):
    qa_path = _source("QuestionA_DataBenefit", "QuestionA_summary", source_run_id)
    qb_path = _source("QuestionB_ZeroH", "QuestionB_summary", source_run_id)
    merged = {}
    conflicts = []
    for path in (qa_path, qb_path):
        for row in read_csv(path):
            if row.get("metric") != "physical_rel_l2":
                continue
            key = (row.get("model"), row.get("recipe"))
            if key in merged and _finite(row.get("mean")) and _finite(merged[key].get("mean")):
                if not np.isclose(float(row["mean"]), float(merged[key]["mean"]), rtol=1e-10, atol=1e-12):
                    conflicts.append({"key": key, "first": merged[key], "second": row})
            merged[key] = dict(row)
    if conflicts:
        raise RuntimeError(f"Conflicting finalized accuracy rows: {conflicts[:2]}")
    rows = []
    coverage = []
    for model in [item["key"] for item in cfg["models"]]:
        for recipe in ALL_RECIPES:
            row = merged.get((model, recipe))
            if row is None:
                spec = _model_spec(cfg, model)
                rows.append({
                    "model": model, "model_label": spec["label"], "recipe": recipe,
                    "recipe_label": cfg["recipes"][recipe]["label"], "metric": "physical_rel_l2",
                    "mean": np.nan, "ci95_low": np.nan, "ci95_high": np.nan,
                    "valid_n": 0, "status": "missing_finalized_result",
                })
                coverage.append({"model": model, "recipe": recipe, "status": "missing_finalized_result"})
            else:
                row["status"] = "ok"
                rows.append(row)
                coverage.append({"model": model, "recipe": recipe, "status": "ok", "valid_n": row.get("valid_n")})
    output = output_dir / f"AllRecipeAccuracy_summary_{rid}.csv"
    write_csv(output, rows)
    return output, coverage, [qa_path, qb_path]


def _base_coarse_rows(source_run_id):
    path = _source("CoarseDetail", "CoarseDetail_per_snapshot", source_run_id)
    return path, [row for row in read_csv(path) if row.get("status") == "ok"]


def _derive_coarse_detail(
    cfg, source_run_id, output_dir, rid, cache_index, cache_status, canonical_path,
    projector_resolution="M", refresh_models=None,
):
    base_path, rows = _base_coarse_rows(source_run_id)
    refresh_models = set(refresh_models or [])
    if refresh_models:
        rows = [row for row in rows if row.get("model") not in refresh_models]
    per_path = output_dir / f"CoarseDetailAllRecipes_per_snapshot_{rid}.csv"
    if per_path.exists():
        rows = [row for row in read_csv(per_path) if row.get("status") == "ok"]
    known = {(r.get("model"), r.get("recipe"), _int(r.get("snapshot_index"))) for r in rows}
    canonical = read_csv(canonical_path)
    snapshots = sorted({_int(row.get("snapshot_index")) for row in canonical if _int(row.get("snapshot_index")) >= 0})
    cache_sources = {
        row.get("metadata") for row in rows
        if row.get("metadata") and str(row.get("metadata")).endswith(".npz")
    }
    coverage = []
    formal = int(cfg["sensor_plan"]["default_count"])
    shape_cache = {}
    for model_spec in cfg["models"]:
        model = model_spec["key"]
        for recipe in ALL_RECIPES:
            valid = 0
            missing_reasons = set()
            for snapshot in snapshots:
                if (model, recipe, snapshot) in known:
                    valid += 1
                    continue
                key = (model, recipe, snapshot, formal)
                entry = cache_index.get(key)
                if entry is None:
                    missing_reasons.add(cache_status.get(key, "missing_cache"))
                    continue
                arrays, meta = load_cache(Path(entry["cache_path"]))
                order, ny, nx = grid_order(arrays["coords_phys"], meta.get("num_x"), meta.get("num_y"))
                truth = arrays["truth_phys"].reshape(-1)[order]
                pred = arrays["recon_phys"].reshape(-1)[order]
                manifest_path = str(meta["manifest_path"])
                if manifest_path not in shape_cache:
                    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                    shape_cache[manifest_path] = resolution_shape(manifest, projector_resolution)
                target_shape = shape_cache[manifest_path]
                metrics, _ = component_metrics(truth, pred, (ny, nx), target_shape)
                rows.append({
                    "model": model, "model_label": model_spec["label"], "recipe": recipe,
                    "recipe_label": cfg["recipes"][recipe]["label"],
                    "case_id": meta.get("case_id"), "time_index": meta.get("time_index"),
                    "snapshot_index": snapshot, "sensor_count": formal,
                    "cutoff": projector_resolution, **metrics, "status": "ok", "metadata": entry["cache_path"],
                })
                cache_sources.add(entry["cache_path"])
                valid += 1
            coverage.append({
                "model": model, "recipe": recipe, "sensor_count": formal,
                "expected_n": len(snapshots), "valid_n": valid,
                "status": "ok" if valid == len(snapshots) else "missing_cache",
                "missing_reason": ";".join(sorted(missing_reasons)),
            })
    rows.sort(key=lambda r: (r.get("model", ""), r.get("recipe", ""), _int(r.get("snapshot_index"))))
    write_csv(per_path, rows)
    base_summary_path = _source("CoarseDetail", "CoarseDetail_summary", source_run_id)
    summaries = [dict(row, status="ok") for row in read_csv(base_summary_path)]
    if refresh_models:
        summaries = [row for row in summaries if row.get("model") not in refresh_models]
    existing_summary_keys = {(row["model"], row["recipe"], row["metric"]) for row in summaries}
    valid_rows = [row for row in rows if row.get("status") == "ok"]
    for metric in ("coarse_rel_l2", "detail_rel_l2", "full_rel_l2"):
        derived = grouped_summary(
            [row for row in valid_rows if (row.get("model"), row.get("recipe"), metric) not in existing_summary_keys],
            ["model", "model_label", "recipe", "recipe_label"], metric,
            n_boot=int(cfg["coarse_detail"].get("bootstrap_samples", 2000)),
        )
        summaries.extend(derived)
    summary_index = {(r["model"], r["recipe"], r["metric"]): r for r in summaries}
    for model_spec in cfg["models"]:
        for recipe in ALL_RECIPES:
            for metric in ("coarse_rel_l2", "detail_rel_l2", "full_rel_l2"):
                key = (model_spec["key"], recipe, metric)
                if key not in summary_index:
                    summaries.append({
                        "model": model_spec["key"], "model_label": model_spec["label"],
                        "recipe": recipe, "recipe_label": cfg["recipes"][recipe]["label"],
                        "metric": metric, "mean": np.nan, "ci95_low": np.nan,
                        "ci95_high": np.nan, "valid_n": 0, "status": "missing_cache",
                    })
    for row in summaries:
        row.setdefault("status", "ok")
    summary_path = output_dir / f"CoarseDetailAllRecipes_summary_{rid}.csv"
    coverage_path = output_dir / f"CoarseDetailAllRecipes_coverage_{rid}.csv"
    write_csv(summary_path, summaries)
    write_csv(coverage_path, coverage)
    return per_path, summary_path, coverage_path, coverage, [base_path, base_summary_path], sorted(cache_sources)


def _build_sensor_table(
    cfg, source_run_id, output_dir, rid, cache_index, cache_status,
    incremental_used, qa_source_run_id=None,
):
    sweep_path = _source("SensorSweep", "SensorSweep_summary", source_run_id)
    qa_path = _source(
        "QuestionA_DataBenefit", "QuestionA_summary",
        qa_source_run_id or source_run_id,
    )
    counts = DEFAULT_SWEEP_COUNTS
    rows = [
        dict(row) for row in read_csv(sweep_path)
        if row.get("metric") == "physical_rel_l2"
        and row.get("recipe") in ALL_RECIPES
        and _int(row.get("sensor_count")) in counts
    ]
    by_key = {(r.get("model"), r.get("recipe"), _int(r.get("sensor_count"))): r for r in rows}
    formal = int(cfg["sensor_plan"]["default_count"])
    for row in read_csv(qa_path):
        if row.get("metric") != "physical_rel_l2" or row.get("recipe") not in {"1_H_only", "2_H_limited"}:
            continue
        copied = dict(row)
        copied["sensor_count"] = formal
        copied["evaluation_grid_points"] = 128 * 128
        copied["status"] = "ok"
        by_key[(copied.get("model"), copied.get("recipe"), formal)] = copied
    # Derive H-only/H-limited sweep summaries whenever the requested cache rows
    # are already present, regardless of whether they came from the audited
    # manifest or a supplemental incremental fill.  Restricting this path to
    # incremental fills incorrectly reported existing canonical cache rows as
    # missing in the all-recipe accuracy panel.
    per_snapshot = []
    for (model, recipe, snapshot, count), entry in cache_index.items():
        if recipe not in {"1_H_only", "2_H_limited"} or count not in counts or count == formal:
            continue
        arrays, meta = load_cache(Path(entry["cache_path"]))
        per_snapshot.append({
            "model": model, "model_label": meta.get("model_label", _model_spec(cfg, model)["label"]),
            "recipe": recipe, "recipe_label": meta.get("recipe_label", cfg["recipes"][recipe]["label"]),
            "sensor_count": count,
            "physical_rel_l2": relative_l2(
                arrays["truth_phys"].reshape(-1), arrays["recon_phys"].reshape(-1)
            ),
            "evaluation_grid_points": int(meta.get("num_x", 128)) * int(meta.get("num_y", 128)),
        })
    derived = grouped_summary(
        per_snapshot, ["model", "model_label", "recipe", "recipe_label", "sensor_count"],
        "physical_rel_l2", n_boot=int(cfg["sensor_sweep"]["bootstrap_samples"]),
    )
    for row in derived:
        row["evaluation_grid_points"] = 128 * 128
        row["status"] = "ok"
        by_key[(row["model"], row["recipe"], _int(row["sensor_count"]))] = row
    coverage = []
    for model_spec in cfg["models"]:
        for recipe in ALL_RECIPES:
            for count in counts:
                key = (model_spec["key"], recipe, count)
                if key in by_key and _finite(by_key[key].get("mean")):
                    status = "ok"
                else:
                    sample_statuses = {
                        cache_status.get((model_spec["key"], recipe, snap, count), "missing_cache")
                        for snap in range(int(cfg["canonical_test"]["max_snapshots"]))
                    }
                    status = ";".join(sorted(sample_statuses))
                coverage.append({"model": model_spec["key"], "recipe": recipe, "sensor_count": count, "status": status})
    summary_path = output_dir / f"SensorSweepAllRecipes_summary_{rid}.csv"
    coverage_path = output_dir / f"SensorSweepAllRecipes_coverage_{rid}.csv"
    write_csv(summary_path, list(by_key.values()))
    write_csv(coverage_path, coverage)
    return summary_path, coverage_path, coverage, [sweep_path, qa_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument(
        "--supplemental-cache-manifest", type=Path,
        help="Existing supplemental cache manifest to merge without running inference.",
    )
    parser.add_argument("--sensor-plan", type=Path)
    parser.add_argument("--data-run-id", default="2026-07-13_14-55")
    parser.add_argument(
        "--sensor-data-run-id",
        help="Optional sensor-sweep source run; defaults to --data-run-id.",
    )
    parser.add_argument("--canonical-run-id", default="formal_20260712")
    parser.add_argument(
        "--layout", default=str(Path(__file__).with_name("publication_layout_unified_v2.yaml")),
        help="Unified plotting YAML; panel d/e metric definitions are read from it.",
    )
    parser.add_argument("--allow-incremental-cache-fill", action="store_true")
    parser.add_argument(
        "--refresh-models", nargs="+", default=[],
        help="Recompute these models from cache instead of reusing finalized coarse/detail rows.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    with Path(args.layout).open("r", encoding="utf-8") as handle:
        layout = yaml.safe_load(handle) or {}
    panel_d_cfg = layout.get("panel_d", {})
    panel_e_cfg = layout.get("panel_e", {})
    ensure_output_dirs()
    rid = run_id(args.run_id)
    output_dir = RESULTS_DIR / "UnifiedPublicationV2"
    output_dir.mkdir(parents=True, exist_ok=True)
    audited_manifest = args.cache_manifest or RESULTS_DIR / "ReconstructionCache" / "ReconstructionCache_manifest_formal_20260712.csv"
    sensor_plan = args.sensor_plan or RESULTS_DIR / "SensorPlans" / "SensorPlan_formal_20260712.csv"
    canonical_path = RESULTS_DIR / "CanonicalTestIndex" / f"CanonicalTestIndex_{args.canonical_run_id}.csv"
    manifest_paths = [Path(audited_manifest)]
    supplemental = args.supplemental_cache_manifest
    if supplemental is not None:
        supplemental = supplemental.resolve()
        if not supplemental.is_file():
            raise FileNotFoundError(supplemental)
        manifest_paths.append(supplemental)
    missing_counts = [count for count in DEFAULT_SWEEP_COUNTS if count != int(cfg["sensor_plan"]["default_count"])]
    if args.allow_incremental_cache_fill:
        if supplemental is not None:
            raise ValueError("Use either --supplemental-cache-manifest or --allow-incremental-cache-fill, not both")
        supplemental = _run_incremental_fill(args, cfg, rid, sensor_plan, missing_counts)
        manifest_paths.append(supplemental)
    cache_index, cache_status = _cache_index(manifest_paths)

    accuracy_path, accuracy_coverage, accuracy_sources = _build_accuracy(cfg, args.data_run_id, output_dir, rid)
    cd_per, cd_summary, cd_coverage_path, cd_coverage, cd_sources, cd_cache_sources = _derive_coarse_detail(
        cfg, args.data_run_id, output_dir, rid, cache_index, cache_status, canonical_path,
        projector_resolution=panel_d_cfg.get("projector_resolution", "M"),
        refresh_models=args.refresh_models,
    )
    manifest_rows = [row for path in manifest_paths for row in read_csv(path)]
    fidelity_per = output_dir / f"CoarseDetailFidelity_per_snapshot_{rid}.csv"
    fidelity_summary = output_dir / f"CoarseDetailFidelity_summary_{rid}.csv"
    fidelity_metadata = output_dir / f"CoarseDetailFidelity_metadata_{rid}.json"
    if all(path.exists() for path in (fidelity_per, fidelity_summary, fidelity_metadata)):
        fidelity_payload = json.loads(fidelity_metadata.read_text(encoding="utf-8"))
    else:
        fidelity_per, fidelity_summary, fidelity_metadata, fidelity_payload = export_coarse_detail_fidelity(
            cfg, manifest_rows, canonical_path, output_dir, rid,
            projector_resolution=panel_d_cfg.get("projector_resolution", "M"),
            sensor_count=int(cfg["sensor_plan"]["default_count"]),
            bootstrap_samples=int(cfg["coarse_detail"].get("bootstrap_samples", 2000)),
        )
    spectral_per = output_dir / f"SpectralBands_per_snapshot_{rid}.csv"
    spectral_summary = output_dir / f"SpectralBands_summary_{rid}.csv"
    spectral_metadata = output_dir / f"SpectralBands_metadata_{rid}.json"
    if all(path.exists() for path in (spectral_per, spectral_summary, spectral_metadata)):
        spectral_payload = json.loads(spectral_metadata.read_text(encoding="utf-8"))
    else:
        spectral_per, spectral_summary, spectral_metadata, spectral_payload = export_spectral_bands(
            cfg, manifest_rows, output_dir, rid,
            sensor_count=int(cfg["sensor_plan"]["default_count"]),
            bootstrap_samples=int(cfg["frequency_error"].get("bootstrap_samples", 2000)),
            relative_epsilon=float(cfg["frequency_error"].get("epsilon", 1e-12)),
            robust_ylim_percentile=float(panel_e_cfg.get("robust_ylim_percentile", 99.0)),
            main_recipes=normalize_recipe_keys(panel_e_cfg.get("recipes", ALL_RECIPES)),
        )
    sensor_data_run_id = args.sensor_data_run_id or args.data_run_id
    sweep_summary, sweep_coverage_path, sweep_coverage, sweep_sources = _build_sensor_table(
        cfg, sensor_data_run_id, output_dir, rid, cache_index, cache_status,
        supplemental is not None, qa_source_run_id=args.data_run_id,
    )
    manifest = {
        "workflow_label": "mixed_resolution_unified_v2",
        "run_id": rid,
        "data_run_id": args.data_run_id,
        "sensor_data_run_id": sensor_data_run_id,
        "canonical_run_id": args.canonical_run_id,
        "audited_cache_manifest": str(Path(audited_manifest).resolve()),
        "audited_cache_manifest_modified": False,
        "supplemental_cache_manifest": str(supplemental.resolve()) if supplemental else None,
        "incremental_cache_fill_used": bool(args.allow_incremental_cache_fill),
        "supplemental_cache_reused": bool(args.supplemental_cache_manifest),
        "incremental_device": "cuda:2" if args.allow_incremental_cache_fill else None,
        "refreshed_models": sorted(set(args.refresh_models)),
        "sources": [str(Path(p).resolve()) for p in accuracy_sources + cd_sources + sweep_sources],
        "canonical_test_index": str(canonical_path.resolve()),
        "sensor_plan": str(sensor_plan.resolve()),
        "outputs": [str(p.resolve()) for p in (
            accuracy_path, cd_per, cd_summary, cd_coverage_path,
            fidelity_per, fidelity_summary, fidelity_metadata,
            spectral_per, spectral_summary, spectral_metadata,
            sweep_summary, sweep_coverage_path,
        )],
        "panels_de": {
            "layout_configuration": str(Path(args.layout).resolve()),
            "coarse_detail_fidelity": fidelity_payload,
            "spectral_bands": spectral_payload,
        },
        "coarse_detail_cache_sources": cd_cache_sources,
        "coverage": {
            "accuracy": accuracy_coverage,
            "coarse_detail": cd_coverage,
            "sensor_sweep": sweep_coverage,
        },
    }
    manifest_path = output_dir / f"UnifiedV2DataManifest_{rid}.json"
    write_json(manifest_path, manifest)
    for path in manifest["outputs"]:
        print(f"[OK] {path}")
    print(f"[OK] {manifest_path}")


if __name__ == "__main__":
    main()

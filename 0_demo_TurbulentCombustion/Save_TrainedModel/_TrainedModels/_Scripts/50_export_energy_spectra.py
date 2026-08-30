#!/usr/bin/env python
"""Export representative cache-only channel-wise spectral-energy curves."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common.cache import load_cache
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, run_id
from common.io_utils import artifact_name, latest, read_csv, write_csv, write_json
from common.spectral import SpectralUnsupportedError, compare_channel_spectra


def _field_specs(cfg: dict, requested: list[str] | None) -> list[dict]:
    if requested is None:
        return [field for field in cfg["fields"] if field["index"] in cfg["spectral"]["channels"]]
    normalize = lambda value: "".join(char for char in str(value).lower() if char.isalnum())
    keys = {normalize(item) for item in requested}
    selected = []
    for field in cfg["fields"]:
        aliases = {normalize(field["key"]), normalize(field["label"]), normalize(field["index"])}
        if keys & aliases:
            selected.append(field)
    if not selected:
        raise ValueError(f"No configured fields match --channels {requested}.")
    return selected


def _manifest_path(args, run_identifier: str) -> Path:
    exact = RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{args.run_id}.csv" if args.run_id else None
    return args.cache_manifest or (exact if exact and exact.exists() else latest(RESULTS_DIR / "ReconstructionCache", "ReconstructionCache_manifest", "csv"))


def _spectral_options(cfg: dict) -> dict[str, Any]:
    spec = cfg["spectral"]
    return {
        "coordinate_mode": spec["preprocessing"]["coordinate_mode"],
        "spacing_tolerance": spec["preprocessing"]["spacing_tolerance"],
        "remove_mean": spec["preprocessing"]["remove_mean"],
        "window": spec["preprocessing"]["window"],
        "use_isotropic_cutoff": spec["radial_average"]["use_isotropic_cutoff"],
        "min_shell_count": spec["radial_average"]["min_shell_count"],
        "relative_epsilon": spec["lsd"]["relative_epsilon"],
    }


def _grid_metadata(grid: dict[str, Any]) -> dict[str, Any]:
    """Keep audit metadata concise; point-order arrays stay in the cache."""
    return {key: value for key, value in grid.items() if key != "sort_idx"}


def _curve_rows(run_identifier: str, method_key: str, method_label: str, family: str, condition: str,
                snapshot: int, field: dict, source: str, spectrum: dict | None, status: str) -> list[dict]:
    common = {"run_id": run_identifier, "model_key": method_key, "model_label": method_label,
              "model_family": family, "condition": condition, "snapshot_index": snapshot,
              "field_id": field["index"], "field_name": field["key"], "source": source, "status": status}
    if spectrum is None:
        return [{**common, "wavenumber_index": np.nan, "wavenumber": np.nan, "spectral_energy": np.nan,
                 "normalized_spectral_energy": np.nan, "shell_count": np.nan}]
    return [{**common, "wavenumber_index": int(index), "wavenumber": float(k), "spectral_energy": float(energy),
             "normalized_spectral_energy": float(normalized), "shell_count": int(count)}
            for index, k, energy, normalized, count in zip(spectrum["wavenumber_index"], spectrum["wavenumber"],
                                                            spectrum["spectral_energy"], spectrum["normalized_spectral_energy"], spectrum["shell_count"])]


def _metric_row(run_identifier: str, method: dict, condition: str, snapshot: int, field: dict,
                observed: bool, entry: dict, result: dict | None, status: str) -> dict:
    row = {"run_id": run_identifier, "model_key": method["directory"], "model_label": method["name"],
           "model_family": entry.get("family", ""), "condition": condition, "snapshot_index": snapshot,
           "field_id": field["index"], "field_name": field["key"], "is_observed": observed,
           "lsd_db": np.nan, "lsd_loge": np.nan, "total_energy_true": np.nan, "total_energy_pred": np.nan,
           "total_energy_ratio": np.nan, "low_band_ratio": np.nan, "mid_band_ratio": np.nan,
           "high_band_ratio": np.nan, "low_band_rel_error": np.nan, "mid_band_rel_error": np.nan,
           "high_band_rel_error": np.nan, "spectral_coord_mode": "", "window": "", "status": status}
    if result is not None:
        row.update({"lsd_db": result["lsd_db"], "lsd_loge": result["lsd_loge"],
                    "total_energy_true": result["truth"]["total_energy"], "total_energy_pred": result["reconstruction"]["total_energy"],
                    "total_energy_ratio": result["total_energy_ratio"], "low_band_ratio": result["band_energy_ratio"][0],
                    "mid_band_ratio": result["band_energy_ratio"][1], "high_band_ratio": result["band_energy_ratio"][2],
                    "low_band_rel_error": result["band_rel_error"][0], "mid_band_rel_error": result["band_rel_error"][1],
                    "high_band_rel_error": result["band_rel_error"][2],
                    "spectral_coord_mode": result["grid"]["coordinate_mode_used"], "window": result["grid"].get("window", "")})
        row["window"] = result.get("window", row["window"])
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--condition", default=None)
    parser.add_argument("--snapshot-index", type=int, default=None)
    parser.add_argument("--channels", nargs="+")
    args = parser.parse_args()
    cfg, rid = load_config(args.config), run_id(args.run_id)
    ensure_output_dirs()
    condition = args.condition or cfg["spectral"]["representative_condition"]
    snapshot = int(args.snapshot_index if args.snapshot_index is not None else cfg["spectral"]["representative_snapshot"])
    fields, options = _field_specs(cfg, args.channels), _spectral_options(cfg)
    manifest_path = _manifest_path(args, rid)
    split = args.split or cfg["defaults"]["split"]
    manifest = [row for row in read_csv(manifest_path) if row.get("split", split) == split]
    lookup = {(row["method"], row["condition"], int(row["snapshot"])): row for row in manifest}
    methods = list(method_items(cfg, args.models))
    entries = [lookup.get((method["name"], condition, snapshot), {"status": "missing cache", "cache_path": ""}) for method in methods]
    truth_entry = next((entry for entry in entries if entry.get("status") == "ok" and entry.get("cache_path")), None)
    truth_arrays = truth_meta = None
    if truth_entry is not None:
        try:
            truth_arrays, truth_meta = load_cache(Path(truth_entry["cache_path"]))
        except Exception:
            truth_arrays, truth_meta, truth_entry = None, None, None

    curves, metrics, grid_decisions = [], [], {}
    observed_fields = set(cfg["conditions"][condition]["cond_fields"])
    truth_results: dict[int, dict] = {}
    for field in fields:
        field_id = field["index"]
        if truth_arrays is None:
            curves.extend(_curve_rows(rid, "truth", "Ground truth", "truth", condition, snapshot, field, "truth", None, "missing cache"))
            truth_status = "missing cache"
        else:
            try:
                truth_result = compare_channel_spectra(
                    truth_arrays["truth_phys"][:, field_id], truth_arrays["truth_phys"][:, field_id], truth_arrays["coords_phys"][:, :2],
                    num_x=truth_meta.get("num_x"), num_y=truth_meta.get("num_y"), **options
                )
                truth_results[field_id] = truth_result
                grid_decisions[field["key"]] = _grid_metadata(truth_result["grid"])
                curves.extend(_curve_rows(rid, "truth", "Ground truth", "truth", condition, snapshot, field, "truth", truth_result["truth"], "ok"))
                truth_status = "ok"
            except SpectralUnsupportedError as exc:
                curves.extend(_curve_rows(rid, "truth", "Ground truth", "truth", condition, snapshot, field, "truth", None, "unsupported grid"))
                grid_decisions[field["key"]] = {"status": "unsupported grid", "detail": str(exc)}
                truth_status = "unsupported grid"

        for method, entry in zip(methods, entries):
            status, arrays = entry.get("status", "missing cache"), None
            if status == "ok" and entry.get("cache_path"):
                try:
                    arrays, cache_metadata = load_cache(Path(entry["cache_path"]))
                except Exception as exc:
                    status = "inference error"
            result = None
            if arrays is not None and truth_status == "ok":
                try:
                    result = compare_channel_spectra(
                        arrays["truth_phys"][:, field_id], arrays["recon_phys"][:, field_id], arrays["coords_phys"][:, :2],
                        num_x=cache_metadata.get("num_x"), num_y=cache_metadata.get("num_y"), **options,
                    )
                    status = "ok"
                except SpectralUnsupportedError:
                    status = "unsupported grid"
            curves.extend(_curve_rows(rid, method["directory"], method["name"], entry.get("family", ""), condition, snapshot,
                                      field, "reconstruction", result["reconstruction"] if result else None, status))
            metric = _metric_row(rid, method, condition, snapshot, field, field_id in observed_fields, entry, result, status)
            metric["window"] = options["window"]
            metrics.append(metric)

    out = RESULTS_DIR / "Spectral" / "EnergySpectra"
    curve_path = out / artifact_name("EnergySpectra_snapshot", rid, "csv")
    metric_path = out / artifact_name("EnergySpectra_snapshot_metrics", rid, "csv")
    metadata_path = out / artifact_name("EnergySpectra_metadata", rid, "json")
    write_csv(curve_path, curves, fieldnames=[
        "run_id", "model_key", "model_label", "model_family", "condition", "snapshot_index", "field_id", "field_name",
        "source", "wavenumber_index", "wavenumber", "spectral_energy", "normalized_spectral_energy", "shell_count", "status",
    ])
    write_csv(metric_path, metrics, fieldnames=[
        "run_id", "model_key", "model_label", "model_family", "condition", "snapshot_index", "field_id", "field_name",
        "is_observed", "lsd_db", "lsd_loge", "total_energy_true", "total_energy_pred", "total_energy_ratio",
        "low_band_ratio", "mid_band_ratio", "high_band_ratio", "low_band_rel_error", "mid_band_rel_error",
        "high_band_rel_error", "spectral_coord_mode", "window", "status",
    ])
    write_json(metadata_path, {"run_id": rid, "cache_manifest": str(manifest_path), "split": split, "condition": condition, "snapshot_index": snapshot,
                               "spectral_options": options, "grid_decisions": grid_decisions,
                               "truth_storage": "one curve per condition/snapshot/field"})
    print(f"[OK] {curve_path}\n[OK] {metric_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

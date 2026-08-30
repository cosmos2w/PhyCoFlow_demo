#!/usr/bin/env python
"""Validate the cache-only panel d/e refinement and cache immutability."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from common.cache import load_cache
from common.config import FIGURES_DIR, RESULTS_DIR
from common.io_utils import read_csv, write_json
from common.multiscale_v2 import component_metrics, decompose
from common.panels_de_data import native_nyquist_contract, resolve_dataset_contract, resolution_shape
from common.spectral import compare_channel_spectra, resolution_band_masks, resolution_band_metrics
from common.workflow import grid_order


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _close(a, b, rtol=2e-6, atol=2e-9):
    return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--legacy-run-id", default="2026-07-15_11-32")
    parser.add_argument("--cache-manifest", type=Path, default=RESULTS_DIR / "ReconstructionCache" / "ReconstructionCache_manifest_formal_20260712.csv")
    parser.add_argument("--cache-baseline", type=Path, required=True)
    parser.add_argument("--composite", type=Path)
    args = parser.parse_args()
    manifest_rows = read_csv(args.cache_manifest)
    available = [row for row in manifest_rows if row.get("status") == "ok" and row.get("cache_path") and int(row.get("sensor_count", -1)) == 256]
    manifest, _, _ = resolve_dataset_contract(manifest_rows, 256)
    entry = available[0]
    arrays, meta = load_cache(Path(entry["cache_path"]))
    order, ny, nx = grid_order(arrays["coords_phys"], meta.get("num_x"), meta.get("num_y"))
    truth = arrays["truth_phys"].reshape(-1)[order]
    pred = arrays["recon_phys"].reshape(-1)[order]
    shape_h = resolution_shape(manifest, "H")
    shape_m = resolution_shape(manifest, "M")
    shape_l = resolution_shape(manifest, "L")
    metrics_m, _ = component_metrics(truth, pred, shape_h, shape_m)
    metrics_l, _ = component_metrics(truth, pred, shape_h, shape_l)

    legacy_path = RESULTS_DIR / "UnifiedPublicationV2" / f"CoarseDetailAllRecipes_per_snapshot_{args.legacy_run_id}.csv"
    new_path = RESULTS_DIR / "UnifiedPublicationV2" / f"CoarseDetailFidelity_per_snapshot_{args.run_id}.csv"
    legacy = {
        (row["model"], row["recipe"], int(row["snapshot_index"])): row
        for row in read_csv(legacy_path) if row.get("status") == "ok"
    }
    new = {
        (row["model"], row["recipe"], int(row["snapshot_index"])): row
        for row in read_csv(new_path) if row.get("status") == "ok"
    }
    comparison_keys = [key for key in sorted(new) if key in legacy][:3]
    comparison_keys += [key for key in sorted(new) if key in legacy][-3:]
    legacy_checks = []
    for key in comparison_keys:
        legacy_checks.append({
            "key": key,
            "coarse_rel_l2_match": _close(new[key]["coarse_rel_l2"], legacy[key]["coarse_rel_l2"]),
            "detail_rel_l2_match": _close(new[key]["detail_rel_l2"], legacy[key]["detail_rel_l2"]),
            "new_detail_rel_l2": float(new[key]["detail_rel_l2"]),
            "legacy_detail_rel_l2": float(legacy[key]["detail_rel_l2"]),
        })
    entry_key = (entry["model"], entry["recipe"], int(entry["snapshot_index"]))
    truth_metrics, _ = component_metrics(truth, truth, shape_h, shape_m)

    spacing, nyquist = native_nyquist_contract(manifest)
    comparison = compare_channel_spectra(
        arrays["truth_phys"].reshape(-1), arrays["recon_phys"].reshape(-1),
        arrays["coords_phys"][:, :2], num_x=nx, num_y=ny,
        coordinate_mode="physical", relative_epsilon=1e-12,
    )
    k = np.asarray(comparison["truth"]["wavenumber"])
    et = np.asarray(comparison["truth"]["spectral_energy"])
    masks = resolution_band_masks(k, nyquist)
    union = masks[0] | masks[1] | masks[2]
    valid = (k > 0) & (k <= nyquist["H"])
    overlaps = (masks[0] & masks[1]) | (masks[0] & masks[2]) | (masks[1] & masks[2])
    truth_bands = resolution_band_metrics(k, et, et, nyquist)
    synthetic_energy = np.ones_like(et)
    excess_bands = resolution_band_metrics(k, synthetic_energy, 4.0 * synthetic_energy, nyquist)
    deficient_bands = resolution_band_metrics(k, synthetic_energy, .25 * synthetic_energy, nyquist)

    baseline = json.loads(args.cache_baseline.read_text(encoding="utf-8"))
    cache_unchanged = []
    for row in baseline["cache_files"]:
        path = Path(row["path"]); stat = path.stat()
        cache_unchanged.append(
            stat.st_size == row["size"] and stat.st_mtime_ns == row["mtime_ns"] and _sha256(path) == row["sha256"]
        )
    manifest_unchanged = _sha256(Path(baseline["manifest"]["path"])) == baseline["manifest"]["sha256"]
    status_unchanged = baseline.get("status") is None or _sha256(Path(baseline["status"]["path"])) == baseline["status"]["sha256"]

    standalone = [
        FIGURES_DIR / "PublicationPanels" / "Panel_D" / f"Panel_d_ScaleSeparatedFidelity_{args.run_id}.{suffix}"
        for suffix in ("pdf", "svg", "png")
    ] + [
        FIGURES_DIR / "PublicationPanels" / "Panel_E" / f"Panel_e_BandSpectralRecovery_{args.run_id}.{suffix}"
        for suffix in ("pdf", "svg", "png")
    ]
    payload = {
        "run_id": args.run_id,
        "no_model_inference": True,
        "projector_M_matches_legacy_metrics": all(
            item["coarse_rel_l2_match"] and item["detail_rel_l2_match"] for item in legacy_checks
        ),
        "projector_L_changes_decomposition": not _close(metrics_l["detail_rel_l2"], metrics_m["detail_rel_l2"], rtol=1e-8, atol=1e-12),
        "legacy_metric_comparisons": legacy_checks,
        "truth_vs_truth_detail_correlation": truth_metrics["detail_correlation"],
        "truth_vs_truth_detail_energy_bias_db": truth_metrics["detail_energy_bias_db"],
        "truth_identity_checks_pass": truth_metrics["detail_correlation"] == 1.0 and truth_metrics["detail_energy_bias_db"] == 0.0,
        "spectral_bands_non_overlapping": not bool(np.any(overlaps)),
        "spectral_bands_cover_valid_nonzero_H_range": bool(np.array_equal(union, valid)),
        "truth_vs_truth_band_bias_db": {row["band"]: row["band_energy_bias_db"] for row in truth_bands},
        "truth_spectral_identity_checks_pass": all(row["band_energy_bias_db"] == 0.0 for row in truth_bands),
        "scaled_field_bias_db": {
            "2x_amplitude_excess": {row["band"]: row["band_energy_bias_db"] for row in excess_bands},
            "0.5x_amplitude_deficit": {row["band"]: row["band_energy_bias_db"] for row in deficient_bands},
        },
        "signed_bias_direction_checks_pass": all(row["band_energy_bias_db"] > 0 for row in excess_bands) and all(row["band_energy_bias_db"] < 0 for row in deficient_bands),
        "native_spacing": spacing, "native_nyquist": nyquist,
        "standalone_outputs_exist": all(path.is_file() for path in standalone),
        "standalone_outputs": [str(path.resolve()) for path in standalone],
        "composite_output_exists": bool(args.composite and args.composite.is_file()),
        "composite_output": str(args.composite.resolve()) if args.composite else None,
        "audited_cache_file_count": len(cache_unchanged),
        "audited_cache_files_unchanged": all(cache_unchanged),
        "audited_cache_manifest_unchanged": manifest_unchanged,
        "audited_cache_status_unchanged": status_unchanged,
    }
    payload["all_checks_pass"] = all([
        payload["projector_M_matches_legacy_metrics"], payload["projector_L_changes_decomposition"],
        payload["truth_identity_checks_pass"], payload["spectral_bands_non_overlapping"],
        payload["spectral_bands_cover_valid_nonzero_H_range"], payload["truth_spectral_identity_checks_pass"],
        payload["signed_bias_direction_checks_pass"], payload["standalone_outputs_exist"],
        payload["composite_output_exists"], payload["audited_cache_files_unchanged"],
        payload["audited_cache_manifest_unchanged"], payload["audited_cache_status_unchanged"],
    ])
    output = RESULTS_DIR / "UnifiedPublicationV2" / f"PanelDEValidation_{args.run_id}.json"
    write_json(output, payload)
    print(json.dumps(payload, indent=2, default=str))
    print(f"[OK] {output}")
    raise SystemExit(0 if payload["all_checks_pass"] else 1)


if __name__ == "__main__":
    main()

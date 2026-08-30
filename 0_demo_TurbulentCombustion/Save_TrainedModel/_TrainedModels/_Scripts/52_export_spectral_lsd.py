#!/usr/bin/env python
"""Export cache-only channel-wise dB and natural-log spectral LSD statistics."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.cache import load_cache
from common.coverage import aggregate_status, expected_snapshots_by_condition
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, run_id
from common.io_utils import artifact_name, latest, read_csv, write_csv, write_json
from common.spectral import SpectralUnsupportedError, compare_channel_spectra_batch, resolve_spectral_device
from common.statistics import summarize
from importlib import import_module

_energy = import_module("50_export_energy_spectra")


def _manifest_path(args) -> Path:
    exact = RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{args.run_id}.csv" if args.run_id else None
    return args.cache_manifest or (exact if exact and exact.exists() else latest(RESULTS_DIR / "ReconstructionCache", "ReconstructionCache_manifest", "csv"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--channels", nargs="+")
    parser.add_argument("--snapshots", nargs="+", type=int)
    parser.add_argument("--max-snapshots", type=int)
    parser.add_argument("--device", default=None, help="Spectral compute device: auto, cpu, or CUDA device (for example cuda:1).")
    args = parser.parse_args()
    cfg, rid = load_config(args.config), run_id(args.run_id)
    ensure_output_dirs()
    fields = _energy._field_specs(cfg, args.channels)
    options = _energy._spectral_options(cfg)
    spectral_device = resolve_spectral_device(args.device or cfg["spectral"].get("compute_device", "auto"))
    print(f"[SPECTRAL] device={spectral_device} | batched_channels={len(fields)}", flush=True)
    conditions = args.conditions or cfg["spectral"]["conditions"]
    manifest_path = _manifest_path(args)
    split = args.split or cfg["defaults"]["split"]
    # Failed cache entries intentionally have no split-specific cache payload.
    # Keep those rows so their explicit inventory status (for example
    # ``missing config`` or ``load error``) propagates to the NaN placeholders
    # instead of being reduced to the less-informative ``missing cache``.
    manifest = [row for row in read_csv(manifest_path)
                if row.get("status") != "ok" or row.get("split", split) == split]
    methods = list(method_items(cfg, args.models))
    lookup = {(row["method"], row["condition"], int(row["snapshot"])): row for row in manifest}
    expected_by_condition = expected_snapshots_by_condition(manifest, conditions)
    snapshots_by_condition = {}
    for condition in conditions:
        available = expected_by_condition[condition]
        if args.snapshots is not None:
            available = [value for value in available if value in args.snapshots]
        if args.max_snapshots is not None:
            available = available[:args.max_snapshots]
        snapshots_by_condition[condition] = available

    rows, grid_status = [], defaultdict(lambda: defaultdict(int))
    for condition in conditions:
        observed_fields = set(cfg["conditions"][condition]["cond_fields"])
        for snapshot in snapshots_by_condition[condition]:
            for method in methods:
                entry = lookup.get((method["name"], condition, snapshot), {"status": "missing cache", "cache_path": ""})
                arrays = metadata = None
                channel_results = None
                status = entry.get("status", "missing cache")
                if status == "ok" and entry.get("cache_path"):
                    try:
                        arrays, metadata = load_cache(Path(entry["cache_path"]))
                    except Exception:
                        status = "inference error"
                if arrays is not None:
                    try:
                        field_indices = [field["index"] for field in fields]
                        channel_results = compare_channel_spectra_batch(
                            arrays["truth_phys"][:, field_indices], arrays["recon_phys"][:, field_indices], arrays["coords_phys"][:, :2],
                            num_x=metadata.get("num_x"), num_y=metadata.get("num_y"), **options, device=spectral_device,
                        )
                    except SpectralUnsupportedError:
                        status = "unsupported grid"
                    except RuntimeError as exc:
                        raise RuntimeError(f"Spectral computation failed on {spectral_device}: {exc}") from exc
                for field in fields:
                    field_id = field["index"]
                    row = {"run_id": rid, "model_key": method["directory"], "model_label": method["name"],
                           "model_family": entry.get("family", ""), "condition": condition, "snapshot_index": snapshot,
                           "field_id": field_id, "field_name": field["key"], "is_observed": field_id in observed_fields,
                           "lsd_db": np.nan, "lsd_loge": np.nan, "total_energy_ratio": np.nan,
                           "low_band_ratio": np.nan, "mid_band_ratio": np.nan, "high_band_ratio": np.nan,
                           "checkpoint_kind": "", "n_steps": "", "consistency_mode_applied": "",
                           "spectral_coord_mode": "", "window": options["window"], "status": status}
                    if metadata is not None:
                        row.update({"checkpoint_kind": metadata.get("checkpoint_name", ""), "n_steps": metadata.get("n_steps", ""),
                                    "consistency_mode_applied": metadata.get("obs_consistency_applied", "")})
                    if channel_results is not None:
                        result = channel_results[fields.index(field)]
                        row.update({"lsd_db": result["lsd_db"], "lsd_loge": result["lsd_loge"],
                                    "total_energy_ratio": result["total_energy_ratio"],
                                    "low_band_ratio": result["band_energy_ratio"][0], "mid_band_ratio": result["band_energy_ratio"][1],
                                    "high_band_ratio": result["band_energy_ratio"][2],
                                    "spectral_coord_mode": result["grid"]["coordinate_mode_used"], "status": "ok"})
                        grid_status[condition][result["grid"]["coordinate_mode_used"]] += 1
                    elif status == "unsupported grid":
                        grid_status[condition]["unsupported grid"] += 1
                    rows.append(row)

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model_key"], row["model_label"], row["condition"], row["field_id"], row["field_name"], row["is_observed"])].append(row)
    summaries = []
    for (key, label, condition, field_id, field_name, observed), values in grouped.items():
        lsd = summarize([value["lsd_db"] for value in values], seed=cfg["defaults"]["seed"])
        energy = summarize([value["total_energy_ratio"] for value in values], seed=cfg["defaults"]["seed"])
        expected_n = len(snapshots_by_condition[condition])
        summaries.append({"run_id": rid, "model_key": key, "model_label": label, "condition": condition,
                          "field_id": field_id, "field_name": field_name, "is_observed": observed,
                          "n_expected_snapshots": expected_n, "n_valid_snapshots": lsd["valid_n"], "mean_lsd_db": lsd["mean"], "std_lsd_db": lsd["std"],
                          "median_lsd_db": lsd["median"], "q25_lsd_db": lsd["q25"], "q75_lsd_db": lsd["q75"],
                          "ci95_low_lsd_db": lsd["ci95_low"], "ci95_high_lsd_db": lsd["ci95_high"],
                          "mean_lsd_loge": summarize([value["lsd_loge"] for value in values], seed=cfg["defaults"]["seed"])["mean"],
                          "mean_energy_ratio": energy["mean"], "std_energy_ratio": energy["std"],
                          "status": aggregate_status(values, lsd["valid_n"], expected_n)})

    out = RESULTS_DIR / "Spectral" / "SpectralLSD"
    per_path = out / artifact_name("SpectralLSD_per_snapshot", rid, "csv")
    summary_path = out / artifact_name("SpectralLSD_summary", rid, "csv")
    metadata_path = out / artifact_name("SpectralLSD_metadata", rid, "json")
    write_csv(per_path, rows, fieldnames=[
        "run_id", "model_key", "model_label", "model_family", "condition", "snapshot_index", "field_id", "field_name",
        "is_observed", "lsd_db", "lsd_loge", "total_energy_ratio", "low_band_ratio", "mid_band_ratio", "high_band_ratio",
        "checkpoint_kind", "n_steps", "consistency_mode_applied", "spectral_coord_mode", "window", "status",
    ])
    write_csv(summary_path, summaries, fieldnames=[
        "run_id", "model_key", "model_label", "condition", "field_id", "field_name", "is_observed", "n_expected_snapshots", "n_valid_snapshots",
        "mean_lsd_db", "std_lsd_db", "median_lsd_db", "q25_lsd_db", "q75_lsd_db", "ci95_low_lsd_db", "ci95_high_lsd_db",
        "mean_lsd_loge", "mean_energy_ratio", "std_energy_ratio", "status",
    ])
    write_json(metadata_path, {"run_id": rid, "cache_manifest": str(manifest_path), "split": split, "conditions": conditions,
                               "snapshots_by_condition": snapshots_by_condition, "spectral_options": options,
                               "spectral_device": spectral_device, "channel_batching": True,
                               "grid_status": {key: dict(value) for key, value in grid_status.items()}, "bootstrap_resamples": 2000,
                               "coverage_policy": "Summary status is ok only when all requested sensor-plan snapshots are valid."})
    print(f"[OK] {per_path}\n[OK] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

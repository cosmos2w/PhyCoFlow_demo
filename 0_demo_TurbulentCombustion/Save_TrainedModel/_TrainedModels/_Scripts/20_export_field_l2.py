#!/usr/bin/env python
"""Export per-snapshot and bootstrap-summary fieldwise reconstruction errors."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from common.cache import cache_manifest, load_cache
from common.coverage import aggregate_status, expected_snapshots_by_condition
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, run_id
from common.io_utils import artifact_name, latest, read_csv, write_csv, write_json
from common.statistics import relative_l2, summarize


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p); p.add_argument("--cache-manifest", type=Path); args = p.parse_args()
    cfg = load_config(args.config); rid = run_id(args.run_id); ensure_output_dirs()
    exact = RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{args.run_id}.csv" if args.run_id else None
    manifest_path = args.cache_manifest or (exact if exact and exact.exists() else latest(RESULTS_DIR / "ReconstructionCache", "ReconstructionCache_manifest", "csv"))
    manifest = read_csv(manifest_path); lookup = {(r["method"], r["condition"], int(r["snapshot"])): r for r in manifest}
    expected = expected_snapshots_by_condition(manifest, cfg["conditions"])
    rows = []
    for method in method_items(cfg, args.models):
        for condition, spec in cfg["conditions"].items():
            observed = set(spec["cond_fields"])
            for snapshot in expected[condition]:
                entry = lookup.get((method["name"], condition, snapshot), {"status": "missing cache", "cache_path": ""})
                arrays = meta = None
                if entry.get("status") == "ok" and entry.get("cache_path"):
                    try: arrays, meta = load_cache(Path(entry["cache_path"]))
                    except Exception as exc: entry = {**entry, "status": "inference error", "detail": str(exc)}
                unobserved_values = []
                for field in cfg["fields"]:
                    c = field["index"]; status = entry.get("status", "missing directory")
                    result = {"run_id": rid, "method": method["name"], "condition": condition, "snapshot": snapshot,
                        "field_index": c, "field": field["key"], "observed": c in observed,
                        "sensor_count": spec["n_obs"][spec["cond_fields"].index(c)] if c in observed else 0,
                        "physical_rel_l2": np.nan, "physical_rel_l2_excluding_observed_entries": np.nan,
                        "normalized_rel_l2": np.nan, "sensor_consistency": np.nan, "status": status,
                        "checkpoint": entry.get("checkpoint_name", ""), "family": entry.get("family", ""),
                        "backbone": entry.get("backbone", ""), "n_steps": entry.get("n_steps", ""),
                        "ode_solver": entry.get("ode_solver", ""), "obs_consistency_applied": entry.get("obs_consistency_applied", "")}
                    if arrays is not None:
                        t, q = arrays["truth_phys"][:, c], arrays["recon_phys"][:, c]
                        result["physical_rel_l2"] = relative_l2(t, q)
                        keep = np.ones(t.shape, bool)
                        field_sensor = arrays["obs_indices"][arrays["obs_field_ids"] == c]
                        keep[field_sensor] = False
                        result["physical_rel_l2_excluding_observed_entries"] = relative_l2(t, q, keep)
                        result["normalized_rel_l2"] = relative_l2(arrays["truth_norm"][:, c], arrays["recon_norm"][:, c])
                        result["sensor_consistency"] = relative_l2(t[field_sensor], q[field_sensor]) if field_sensor.size else np.nan
                        result["status"] = "ok"
                        if c not in observed: unobserved_values.append(result["physical_rel_l2"])
                    rows.append(result)
                rows.append({"run_id": rid, "method": method["name"], "condition": condition, "snapshot": snapshot,
                    "field_index": -1, "field": "Unobserved_mean", "observed": False, "sensor_count": 0,
                    "physical_rel_l2": float(np.mean(unobserved_values)) if unobserved_values else np.nan,
                    "physical_rel_l2_excluding_observed_entries": np.nan, "normalized_rel_l2": np.nan,
                    "sensor_consistency": np.nan, "status": "ok" if unobserved_values else entry.get("status", "missing directory")})
    out = RESULTS_DIR / "FieldL2"; per = out / artifact_name("FieldL2_per_snapshot", rid, "csv"); write_csv(per, rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["condition"], row["field"])].append(row)
    summary = []
    for (method, condition, field), values in groups.items():
        stats = summarize([v["physical_rel_l2"] for v in values], seed=cfg["defaults"]["seed"])
        expected_n = len(expected[condition])
        summary.append({"run_id": rid, "method": method, "condition": condition, "field": field,
                        "n_expected_snapshots": expected_n, **stats,
                        "status": aggregate_status(values, stats["valid_n"], expected_n)})
    summ = out / artifact_name("FieldL2_summary", rid, "csv"); write_csv(summ, summary)
    write_json(out / artifact_name("FieldL2_metadata", rid, "json"), {"run_id": rid, "cache_manifest": str(manifest_path),
        "metric": "physical relative L2 over spatial points", "bootstrap_resamples": 2000,
        "split": cfg["defaults"]["split"], "expected_snapshots_by_condition": expected,
        "coverage_policy": "Summary status is ok only when all sensor-plan snapshots are valid."})
    print(f"[OK] {per}\n[OK] {summ}"); return 0


if __name__ == "__main__": raise SystemExit(main())

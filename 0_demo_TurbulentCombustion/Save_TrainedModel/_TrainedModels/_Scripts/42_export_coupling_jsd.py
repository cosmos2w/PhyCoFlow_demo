#!/usr/bin/env python
"""Export cache-only JSD statistics for the three publication coupling pairs.

The exporter reuses finalized T--U1 and T--CO JSD rows where possible.  It
loads reconstruction caches only for the new CO--U1 coupling, never loads a
model or regenerates a prediction.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.cache import load_cache
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, run_id
from common.coverage import aggregate_status, expected_snapshots_by_condition
from common.io_utils import artifact_name, latest, read_csv, write_csv
from common.pdf_utils import PAIR_FIELDS, global_edges, histogram
from common.statistics import jsd_base2, summarize


COUPLING_PAIRS = ("T-U1", "CO-U1", "CO-T")
# JSD is invariant to transposing both truth and reconstruction histograms.
# Therefore legacy T--CO values remain mathematically identical when displayed
# as CO--T; retaining them avoids a needless full-cache pass.
LEGACY_PAIR = {"T-U1": "T-U1", "CO-T": "T-CO"}


def _manifest_path(args, rid: str) -> Path:
    if args.cache_manifest:
        return args.cache_manifest
    exact = RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{args.run_id}.csv" if args.run_id else None
    return exact if exact and exact.exists() else latest(RESULTS_DIR / "ReconstructionCache", "ReconstructionCache_manifest", "csv")


def _legacy_path(args) -> Path:
    if args.legacy_jsd:
        return args.legacy_jsd
    exact = RESULTS_DIR / "JointPDF_JSD" / f"JointPDF_JSD_per_snapshot_{args.run_id}.csv" if args.run_id else None
    return exact if exact and exact.exists() else latest(RESULTS_DIR / "JointPDF_JSD", "JointPDF_JSD_per_snapshot", "csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--cache-manifest", type=Path, help="Finalized reconstruction-cache manifest.")
    parser.add_argument("--legacy-jsd", type=Path, help="Existing T--U1/T--CO per-snapshot JSD CSV.")
    parser.add_argument("--conditions", nargs="+", default=None, help="Condition keys; defaults to configured order.")
    parser.add_argument("--bins", type=int, help="Bin count used only for the new CO--U1 metric.")
    args = parser.parse_args()
    cfg, rid = load_config(args.config), run_id(args.run_id)
    ensure_output_dirs()
    conditions = list(args.conditions or cfg["conditions"])
    methods = list(method_items(cfg, args.models))
    method_names = {item["name"] for item in methods}
    manifest_path = _manifest_path(args, rid)
    manifest = [row for row in read_csv(manifest_path) if row.get("method") in method_names and row.get("condition") in conditions]
    expected = expected_snapshots_by_condition(manifest, conditions)
    cache_lookup = {(row["method"], row["condition"], int(row["snapshot"])): row for row in manifest}

    legacy_path = _legacy_path(args)
    legacy = {
        (row["method"], row["condition"], int(row["snapshot"]), row["pair"]): row
        for row in read_csv(legacy_path)
        if row.get("method") in method_names and row.get("condition") in conditions
    }
    # Global physical CO--U1 bins are truth-only and invariant across methods.
    co_u1_edges = global_edges(manifest, ["CO-U1"], args.bins or cfg["defaults"]["pdf_bins"], cfg["defaults"]["robust_quantiles"])["CO-U1"]

    rows: list[dict] = []
    for method in methods:
        for condition in conditions:
            for snapshot in expected[condition]:
                cache_entry = cache_lookup.get((method["name"], condition, snapshot), {})
                for pair in COUPLING_PAIRS:
                    legacy_pair = LEGACY_PAIR.get(pair)
                    legacy_row = legacy.get((method["name"], condition, snapshot, legacy_pair)) if legacy_pair else None
                    status = cache_entry.get("status", "missing cache")
                    value = np.nan
                    source = "cache_CO-U1"
                    if legacy_row is not None:
                        status = legacy_row.get("status", status)
                        value = float(legacy_row["jsd_base2"]) if status == "ok" else np.nan
                        source = f"reused_{legacy_pair}"
                    elif pair == "CO-U1" and status == "ok" and cache_entry.get("cache_path"):
                        try:
                            arrays, _ = load_cache(Path(cache_entry["cache_path"]))
                            a, b = PAIR_FIELDS[pair]
                            truth = histogram(arrays["truth_phys"][:, a], arrays["truth_phys"][:, b], co_u1_edges)
                            recon = histogram(arrays["recon_phys"][:, a], arrays["recon_phys"][:, b], co_u1_edges)
                            value = jsd_base2(truth, recon, cfg["defaults"]["pdf_pseudocount"])
                            status = "ok"
                        except Exception as exc:  # keep every expected table slot
                            status = "inference error"
                            source = f"cache_error:{type(exc).__name__}"
                    rows.append({
                        "run_id": rid, "method": method["name"], "condition": condition, "snapshot": snapshot,
                        "pair": pair, "jsd_base2": value, "status": status, "value_source": source,
                        "checkpoint": cache_entry.get("checkpoint_name", ""), "family": cache_entry.get("family", ""),
                        "n_steps": cache_entry.get("n_steps", ""), "ode_solver": cache_entry.get("ode_solver", ""),
                    })

    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["method"], row["condition"], row["pair"])].append(row)
    summary = []
    for (method, condition, pair), items in groups.items():
        stats = summarize([item["jsd_base2"] for item in items], seed=cfg["defaults"]["seed"])
        expected_n = len(expected[condition])
        summary.append({
            "run_id": rid, "method": method, "condition": condition, "pair": pair,
            "n_expected_snapshots": expected_n, **stats,
            "status": aggregate_status(items, stats["valid_n"], expected_n),
        })

    out = RESULTS_DIR / "JointPDF_JSD"
    per_path = out / artifact_name("CouplingJSD_per_snapshot", rid, "csv")
    summary_path = out / artifact_name("CouplingJSD_summary", rid, "csv")
    edges_path = out / artifact_name("CouplingJSD_bin_edges", rid, "csv")
    write_csv(per_path, rows)
    write_csv(summary_path, summary)
    edge_rows = [
        {"run_id": rid, "pair": "CO-U1", "axis": axis, "edge_index": index, "edge": edge}
        for axis, values in zip(("x", "y"), co_u1_edges) for index, edge in enumerate(values)
    ]
    write_csv(edges_path, edge_rows)
    print(f"[OK] cache-only coupling JSD: {per_path}\n[OK] {summary_path}\n[OK] {edges_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

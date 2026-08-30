#!/usr/bin/env python
"""Export cache-only U1--p flow-consistency JSD diagnostics for all conditions.

This diagnostic exporter reads finalized reconstruction caches but never loads a
model or writes/regenerates a cache.  It is deliberately separate from the
main manuscript coupling artifacts.
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


PAIR = "U1-p"


def _manifest_path(args) -> Path:
    if args.cache_manifest:
        return args.cache_manifest
    exact = RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{args.run_id}.csv" if args.run_id else None
    return exact if exact and exact.exists() else latest(RESULTS_DIR / "ReconstructionCache", "ReconstructionCache_manifest", "csv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--cache-manifest", type=Path, help="Finalized reconstruction-cache manifest.")
    parser.add_argument("--conditions", nargs="+", default=None, help="Condition keys; defaults to all configured conditions.")
    parser.add_argument("--bins", type=int, help="Shared truth-derived bin count.")
    args = parser.parse_args()
    cfg, rid = load_config(args.config), run_id(args.run_id)
    ensure_output_dirs()
    conditions = list(args.conditions or cfg["conditions"])
    methods = list(method_items(cfg, args.models))
    method_names = {method["name"] for method in methods}
    manifest = [
        row for row in read_csv(_manifest_path(args))
        if row.get("method") in method_names and row.get("condition") in conditions
    ]
    expected = expected_snapshots_by_condition(manifest, conditions)
    cache_lookup = {(row["method"], row["condition"], int(row["snapshot"])): row for row in manifest}
    edges = global_edges(manifest, [PAIR], args.bins or cfg["defaults"]["pdf_bins"], cfg["defaults"]["robust_quantiles"])[PAIR]

    rows: list[dict] = []
    for method in methods:
        for condition in conditions:
            for snapshot in expected[condition]:
                entry = cache_lookup.get((method["name"], condition, snapshot), {})
                status, value = entry.get("status", "missing cache"), np.nan
                if status == "ok" and entry.get("cache_path"):
                    try:
                        arrays, _ = load_cache(Path(entry["cache_path"]))
                        first, second = PAIR_FIELDS[PAIR]
                        truth = histogram(arrays["truth_phys"][:, first], arrays["truth_phys"][:, second], edges)
                        reconstruction = histogram(arrays["recon_phys"][:, first], arrays["recon_phys"][:, second], edges)
                        value = jsd_base2(truth, reconstruction, cfg["defaults"]["pdf_pseudocount"])
                    except Exception as exc:
                        status = "inference error"
                        detail = type(exc).__name__
                    else:
                        detail = "cache-only U1-p"
                else:
                    detail = ""
                rows.append({
                    "run_id": rid, "method": method["name"], "condition": condition, "snapshot": snapshot,
                    "pair": PAIR, "jsd_base2": value, "status": status, "value_source": "cache_U1-p",
                    "checkpoint": entry.get("checkpoint_name", ""), "family": entry.get("family", ""),
                    "n_steps": entry.get("n_steps", ""), "ode_solver": entry.get("ode_solver", ""), "detail": detail,
                })

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["condition"])].append(row)
    summary = []
    for (method, condition), items in grouped.items():
        stats = summarize([row["jsd_base2"] for row in items], seed=cfg["defaults"]["seed"])
        summary.append({
            "run_id": rid, "method": method, "condition": condition, "pair": PAIR,
            "n_expected_snapshots": len(expected[condition]), **stats,
            "status": aggregate_status(items, stats["valid_n"], len(expected[condition])),
        })

    output = RESULTS_DIR / "JointPDF_JSD"
    per_path = output / artifact_name("FlowConsistencyJSD_per_snapshot", rid, "csv")
    summary_path = output / artifact_name("FlowConsistencyJSD_summary", rid, "csv")
    edges_path = output / artifact_name("FlowConsistencyJSD_bin_edges", rid, "csv")
    write_csv(per_path, rows); write_csv(summary_path, summary)
    write_csv(edges_path, [
        {"run_id": rid, "pair": PAIR, "axis": axis, "edge_index": index, "edge": edge}
        for axis, values in zip(("x", "y"), edges) for index, edge in enumerate(values)
    ])
    print(f"[OK] cache-only flow-consistency JSD: {per_path}\n[OK] {summary_path}\n[OK] {edges_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

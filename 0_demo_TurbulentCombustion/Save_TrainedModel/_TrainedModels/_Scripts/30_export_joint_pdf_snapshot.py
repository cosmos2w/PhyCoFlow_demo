#!/usr/bin/env python
"""Export representative fixed-bin truth/reconstruction joint PDFs and JSD."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from common.cache import load_cache
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, run_id
from common.io_utils import artifact_name, latest, read_csv, write_csv
from common.pdf_utils import PAIR_FIELDS, global_edges, histogram
from common.statistics import jsd_base2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p); p.add_argument("--cache-manifest", type=Path); p.add_argument("--condition", default="Cond_T")
    p.add_argument("--snapshot", type=int, default=0); p.add_argument("--pairs", nargs="+", default=["T-CO", "T-U1"])
    p.add_argument("--bins", type=int); args = p.parse_args(); cfg = load_config(args.config); rid = run_id(args.run_id); ensure_output_dirs()
    exact = RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{args.run_id}.csv" if args.run_id else None
    manifest_path = args.cache_manifest or (exact if exact and exact.exists() else latest(RESULTS_DIR / "ReconstructionCache", "ReconstructionCache_manifest", "csv"))
    manifest = read_csv(manifest_path); bins = args.bins or cfg["defaults"]["pdf_bins"]
    edges = global_edges(manifest, args.pairs, bins, cfg["defaults"]["robust_quantiles"])
    lookup = {(r["method"], r["condition"], int(r["snapshot"])): r for r in manifest}; long_rows = []; metrics = []
    # Truth is model independent.  Recover it once from the first compatible
    # reconstruction cache rather than repeating the same histogram under each
    # method name.  This also makes the contact sheet's reference column explicit.
    truth_arrays = None
    truth_status = "missing cache"
    for method in method_items(cfg, args.models):
        candidate = lookup.get((method["name"], args.condition, args.snapshot))
        if not candidate or candidate.get("status") != "ok" or not candidate.get("cache_path"):
            continue
        try:
            truth_arrays, _ = load_cache(Path(candidate["cache_path"]))
            truth_status = "ok"
            break
        except Exception:
            truth_status = "inference error"
    for pair in args.pairs:
        ex, ey = edges[pair]
        if truth_arrays is not None:
            a, b = PAIR_FIELDS[pair]
            truth_hist = histogram(truth_arrays["truth_phys"][:, a], truth_arrays["truth_phys"][:, b], (ex, ey))
        else:
            truth_hist = np.full((bins, bins), np.nan)
        for i in range(bins):
            for j in range(bins):
                long_rows.append({"run_id": rid, "method": "truth", "condition": args.condition, "snapshot": args.snapshot,
                    "pair": pair, "source": "truth", "bin_x": i, "bin_y": j, "x_left": ex[i], "x_right": ex[i+1],
                    "y_left": ey[j], "y_right": ey[j+1], "probability": truth_hist[i, j], "status": truth_status})
    for method in method_items(cfg, args.models):
        entry = lookup.get((method["name"], args.condition, args.snapshot), {"status": "missing directory"})
        arrays = None
        if entry.get("status") == "ok" and entry.get("cache_path"):
            try: arrays, _ = load_cache(Path(entry["cache_path"]))
            except Exception: entry["status"] = "inference error"
        for pair in args.pairs:
            ex, ey = edges[pair]; status = entry.get("status", "missing directory"); jsd = np.nan
            if arrays is not None:
                a, b = PAIR_FIELDS[pair]
                gt = histogram(arrays["truth_phys"][:, a], arrays["truth_phys"][:, b], (ex, ey))
                rec = histogram(arrays["recon_phys"][:, a], arrays["recon_phys"][:, b], (ex, ey))
                jsd = jsd_base2(gt, rec, cfg["defaults"]["pdf_pseudocount"]); status = "ok"
            else:
                rec = np.full((bins, bins), np.nan)
            metrics.append({"run_id": rid, "method": method["name"], "condition": args.condition, "snapshot": args.snapshot, "pair": pair, "jsd_base2": jsd, "status": status})
            for i in range(bins):
                for j in range(bins):
                    long_rows.append({"run_id": rid, "method": method["name"], "condition": args.condition, "snapshot": args.snapshot,
                        "pair": pair, "source": "reconstruction", "bin_x": i, "bin_y": j, "x_left": ex[i], "x_right": ex[i+1],
                        "y_left": ey[j], "y_right": ey[j+1], "probability": rec[i, j], "status": status})
    out = RESULTS_DIR / "JointPDF"; long_path = out / artifact_name("JointPDF_snapshot", rid, "csv"); metric_path = out / artifact_name("JointPDF_snapshot_metrics", rid, "csv")
    write_csv(long_path, long_rows); write_csv(metric_path, metrics)
    edge_rows = [{"run_id": rid, "pair": pair, "axis": axis, "edge_index": i, "edge": value} for pair, edge_pair in edges.items() for axis, edge in zip(("x", "y"), edge_pair) for i, value in enumerate(edge)]
    write_csv(out / artifact_name("JointPDF_bin_edges", rid, "csv"), edge_rows)
    print(f"[OK] {long_path}\n[OK] {metric_path}"); return 0


if __name__ == "__main__": raise SystemExit(main())

#!/usr/bin/env python
"""Export fixed-global-bin base-2 JSD for all cached test snapshots."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from common.cache import load_cache
from common.coverage import aggregate_status, expected_snapshots_by_condition
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, run_id
from common.io_utils import artifact_name, latest, read_csv, write_csv
from common.pdf_utils import PAIR_FIELDS, global_edges, histogram
from common.statistics import jsd_base2, summarize


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter); add_common_args(p)
    p.add_argument("--cache-manifest", type=Path); p.add_argument("--pairs", nargs="+", default=["T-CO","T-U1"]); p.add_argument("--bins", type=int); args=p.parse_args()
    cfg=load_config(args.config); rid=run_id(args.run_id); ensure_output_dirs(); exact=RESULTS_DIR/"ReconstructionCache"/f"ReconstructionCache_manifest_{args.run_id}.csv" if args.run_id else None; manifest_path=args.cache_manifest or (exact if exact and exact.exists() else latest(RESULTS_DIR/"ReconstructionCache","ReconstructionCache_manifest","csv"))
    manifest=read_csv(manifest_path); methods=list(method_items(cfg,args.models)); method_names={m["name"] for m in methods}; manifest=[r for r in manifest if r["method"] in method_names]
    expected=expected_snapshots_by_condition(manifest,cfg["conditions"]); lookup={(r["method"],r["condition"],int(r["snapshot"])):r for r in manifest}
    edges=global_edges(manifest,args.pairs,args.bins or cfg["defaults"]["pdf_bins"],cfg["defaults"]["robust_quantiles"]); rows=[]
    for method in methods:
        for condition in cfg["conditions"]:
            for snapshot in expected[condition]:
                entry=lookup.get((method["name"],condition,snapshot),{"method":method["name"],"condition":condition,"snapshot":snapshot,"status":"missing cache","cache_path":""})
                arrays=None; status=entry.get("status","missing cache")
                if status=="ok" and entry.get("cache_path"):
                    try: arrays,_=load_cache(Path(entry["cache_path"]))
                    except Exception: status="inference error"
                for pair in args.pairs:
                    value=np.nan
                    if arrays is not None:
                        a,b=PAIR_FIELDS[pair]; value=jsd_base2(histogram(arrays["truth_phys"][:,a],arrays["truth_phys"][:,b],edges[pair]),histogram(arrays["recon_phys"][:,a],arrays["recon_phys"][:,b],edges[pair]),cfg["defaults"]["pdf_pseudocount"]); status="ok"
                    rows.append({"run_id":rid,"method":method["name"],"condition":condition,"snapshot":snapshot,"pair":pair,"jsd_base2":value,"status":status,
                                 "checkpoint":entry.get("checkpoint_name",""),"family":entry.get("family",""),"n_steps":entry.get("n_steps",""),"ode_solver":entry.get("ode_solver","")})
    groups=defaultdict(list)
    for r in rows: groups[(r["method"],r["condition"],r["pair"])].append(r)
    summary=[]
    for key,items in groups.items():
        stats=summarize([x["jsd_base2"] for x in items],seed=cfg["defaults"]["seed"]); expected_n=len(expected[key[1]])
        summary.append({"run_id":rid,"method":key[0],"condition":key[1],"pair":key[2],"n_expected_snapshots":expected_n,**stats,"status":aggregate_status(items,stats["valid_n"],expected_n)})
    out=RESULTS_DIR/"JointPDF_JSD"; per=out/artifact_name("JointPDF_JSD_per_snapshot",rid,"csv"); summ=out/artifact_name("JointPDF_JSD_summary",rid,"csv"); write_csv(per,rows); write_csv(summ,summary)
    edge_rows=[{"run_id":rid,"pair":pair,"axis":axis,"edge_index":i,"edge":v} for pair,e in edges.items() for axis,edge in zip(("x","y"),e) for i,v in enumerate(edge)]; write_csv(out/artifact_name("JointPDF_JSD_bin_edges",rid,"csv"),edge_rows)
    print(f"[OK] {per}\n[OK] {summ}"); return 0
if __name__=="__main__": raise SystemExit(main())

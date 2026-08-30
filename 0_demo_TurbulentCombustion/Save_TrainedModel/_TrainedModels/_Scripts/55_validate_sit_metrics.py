#!/usr/bin/env python
"""Audit SiT spatial, spectral, and T--U1 distribution metrics from final CSVs.

This cache-only diagnostic does not alter formal metrics.  It records the
per-snapshot L2/LSD association and the cache identity invariants used by the
publication figure.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from common.config import RESULTS_DIR, add_common_args, run_id
from common.io_utils import read_csv, write_csv, write_json

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p, models=False); p.add_argument("--cache-manifest", type=Path); args=p.parse_args(); rid=run_id(args.run_id)
    manifest=args.cache_manifest or RESULTS_DIR/"ReconstructionCache"/f"ReconstructionCache_manifest_{rid}.csv"
    l2=read_csv(RESULTS_DIR/"FieldL2"/f"FieldL2_per_snapshot_{rid}.csv")
    lsd=read_csv(RESULTS_DIR/"Spectral"/"SpectralLSD"/f"SpectralLSD_per_snapshot_{rid}.csv")
    jsd=read_csv(RESULTS_DIR/"JointPDF_JSD"/f"JointPDF_JSD_per_snapshot_{rid}.csv")
    ok={(r["condition"],r["snapshot"]):r for r in read_csv(manifest) if r["method"]=="SiT" and r["status"]=="ok"}
    l2k={(r["condition"],r["snapshot"],r["field"]):r for r in l2 if r["method"]=="SiT" and r["status"]=="ok"}
    rows=[]
    for r in lsd:
        if r["model_label"]!="SiT" or r["status"]!="ok" or r["field_name"] not in {"CH4","CO","U1"}: continue
        key=(r["condition"],r["snapshot_index"],r["field_name"])
        a=l2k.get(key); m=ok.get((r["condition"],r["snapshot_index"]))
        rows.append({"condition":r["condition"],"snapshot":r["snapshot_index"],"field":r["field_name"],"physical_l2":a.get("physical_rel_l2",np.nan) if a else np.nan,"lsd_db":r["lsd_db"],"cache_identity":m.get("cache_identity","") if m else "","status":"ok" if a and m else "missing linkage"})
    out=RESULTS_DIR/"Validation"; out.mkdir(parents=True,exist_ok=True)
    write_csv(out/f"SiT_metric_validation_{rid}.csv",rows)
    correlations={}
    for field in ("CH4","CO","U1"):
        vals=[(float(x["physical_l2"]),float(x["lsd_db"])) for x in rows if x["field"]==field and np.isfinite(float(x["physical_l2"]))]
        correlations[field]=float(np.corrcoef(np.asarray(vals).T)[0,1]) if len(vals)>1 else np.nan
    write_json(out/f"SiT_metric_validation_{rid}.json",{"cache_manifest":str(manifest),"n_rows":len(rows),"l2_lsd_correlation":correlations,"t_u1_jsd_rows":sum(r["method"]=="SiT" and r["pair"]=="T-U1" and r["status"]=="ok" for r in jsd),"interpretation":"Radial LSD and joint-PDF JSD are insensitive to spatial phase/location; they must not be interpreted as pointwise L2."})
    print(f"[OK] {out}"); return 0
if __name__=="__main__": raise SystemExit(main())

#!/usr/bin/env python
"""Export paired H-limited versus Mixed-HML reconstruction metrics."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, run_id, stable_seed
from common.io_utils import artifact_name, write_csv
from common.workflow import cache_entries, grouped_summary, metric_row

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p); p.add_argument("--cache-manifest",type=Path); p.add_argument("--snapshots",nargs="+",type=int); p.add_argument("--sensor-count",type=int); args=p.parse_args()
    cfg=load_config(args.config); rid=run_id(args.run_id); ensure_output_dirs(); recipes=[*cfg["questionA"]["recipes"],cfg["questionA"]["reference_recipe"]]; count=args.sensor_count or cfg["sensor_plan"]["default_count"]
    entries=cache_entries(args.cache_manifest,models=args.models,recipes=recipes,sensor_count=count,snapshots=args.snapshots); rows=[]
    for e in entries:
        try: rows.append(metric_row(e))
        except Exception as exc: rows.append({"model":e.get("model"),"recipe":e.get("recipe"),"snapshot_index":e.get("snapshot_index"),"physical_rel_l2":np.nan,"status":f"metric_error:{exc}"})
    expected_models=[m["key"] for m in method_items(cfg,args.models)]; expected_snapshots=args.snapshots or sorted({int(e["snapshot_index"]) for e in entries}); present={(r.get("model"),r.get("recipe"),int(r.get("snapshot_index",-1))) for r in rows}
    for model in expected_models:
        for recipe in recipes:
            for snapshot in expected_snapshots:
                if (model,recipe,snapshot) not in present: rows.append({"model":model,"recipe":recipe,"snapshot_index":snapshot,"sensor_count":count,"physical_rel_l2":np.nan,"physical_rel_l2_sensor_excluded":np.nan,"normalized_rel_l2":np.nan,"SSIM":np.nan,"gradient_rel_l2":np.nan,"status":"missing_cache","metadata":""})
    per=RESULTS_DIR/"QuestionA_DataBenefit"/artifact_name("QuestionA_per_snapshot",rid,"csv"); write_csv(per,rows)
    summary=grouped_summary([r for r in rows if r["status"]=="ok"],["model","model_label","recipe","recipe_label"],"physical_rel_l2",n_boot=cfg["questionA"]["bootstrap_samples"])
    sp=RESULTS_DIR/"QuestionA_DataBenefit"/artifact_name("QuestionA_summary",rid,"csv"); write_csv(sp,summary)
    by=defaultdict(dict)
    for r in rows:
        if r["status"]=="ok": by[(r["model"],int(r["snapshot_index"]))][r["recipe"]]=r
    paired=[]; a,b=cfg["questionA"]["recipes"]
    for (model,snapshot),d in by.items():
        if a in d and b in d:
            old,new=float(d[a]["physical_rel_l2"]),float(d[b]["physical_rel_l2"]); paired.append({"model":model,"snapshot_index":snapshot,"recipe_from":a,"recipe_to":b,"paired_difference":new-old,"relative_improvement_percent":100*(old-new)/(old+1e-12),"status":"ok"})
    pp=RESULTS_DIR/"QuestionA_DataBenefit"/artifact_name("QuestionA_paired",rid,"csv"); write_csv(pp,paired); print(f"[OK] {per}\n[OK] {sp}\n[OK] {pp}")
if __name__=="__main__": main()

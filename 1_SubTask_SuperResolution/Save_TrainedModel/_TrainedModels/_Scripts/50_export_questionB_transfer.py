#!/usr/bin/env python
"""Export zero-H transfer errors and penalties relative to H-only."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, run_id
from common.io_utils import artifact_name, write_csv
from common.workflow import cache_entries, grouped_summary, metric_row

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p); p.add_argument("--cache-manifest",type=Path); p.add_argument("--snapshots",nargs="+",type=int); p.add_argument("--sensor-count",type=int); args=p.parse_args(); cfg=load_config(args.config); rid=run_id(args.run_id); ensure_output_dirs(); recipes=cfg["questionB"]["recipes"]; count=args.sensor_count or cfg["sensor_plan"]["default_count"]
    rows=[]
    for e in cache_entries(args.cache_manifest,models=args.models,recipes=recipes,sensor_count=count,snapshots=args.snapshots):
        try: rows.append(metric_row(e))
        except Exception as exc: rows.append({"model":e.get("model"),"recipe":e.get("recipe"),"snapshot_index":e.get("snapshot_index"),"physical_rel_l2":np.nan,"status":f"error:{exc}"})
    expected_models=[m["key"] for m in method_items(cfg,args.models)]; expected_snapshots=args.snapshots or sorted({int(r["snapshot_index"]) for r in rows if str(r.get("snapshot_index","")).isdigit()}); present={(r.get("model"),r.get("recipe"),int(r.get("snapshot_index",-1))) for r in rows}
    for model in expected_models:
        for recipe in recipes:
            for snapshot in expected_snapshots:
                if (model,recipe,snapshot) not in present: rows.append({"model":model,"recipe":recipe,"snapshot_index":snapshot,"sensor_count":count,"physical_rel_l2":np.nan,"physical_rel_l2_sensor_excluded":np.nan,"normalized_rel_l2":np.nan,"SSIM":np.nan,"gradient_rel_l2":np.nan,"status":"missing_cache","metadata":""})
    by=defaultdict(dict)
    for r in rows:
        if r["status"]=="ok": by[(r["model"],int(r["snapshot_index"]))][r["recipe"]]=r
    ref=cfg["questionB"]["reference_recipe"]
    for (_, _),d in by.items():
        if ref in d:
            baseline=float(d[ref]["physical_rel_l2"])
            for r in d.values(): r["transfer_penalty_abs"]=float(r["physical_rel_l2"])-baseline; r["transfer_penalty_percent"]=100*(float(r["physical_rel_l2"])-baseline)/(baseline+1e-12)
    per=RESULTS_DIR/"QuestionB_ZeroH"/artifact_name("QuestionB_per_snapshot",rid,"csv"); write_csv(per,rows); summary=grouped_summary([r for r in rows if r["status"]=="ok"],["model","model_label","recipe","recipe_label"],"physical_rel_l2",n_boot=cfg["questionB"]["bootstrap_samples"]); sp=RESULTS_DIR/"QuestionB_ZeroH"/artifact_name("QuestionB_summary",rid,"csv"); write_csv(sp,summary); print(f"[OK] {per}\n[OK] {sp}")
if __name__=="__main__": main()

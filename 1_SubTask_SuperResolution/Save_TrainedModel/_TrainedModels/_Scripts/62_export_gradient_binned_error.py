#!/usr/bin/env python
"""Export pooled H-truth gradient-quantile normalized RMSE/MAE."""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from common.cache import load_cache
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, run_id
from common.io_utils import artifact_name, write_csv
from common.workflow import cache_entries, grid_order

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p); p.add_argument("--cache-manifest",type=Path); p.add_argument("--snapshots",nargs="+",type=int); p.add_argument("--sensor-count",type=int); args=p.parse_args(); cfg=load_config(args.config); rid=run_id(args.run_id); ensure_output_dirs(); entries=cache_entries(args.cache_manifest,models=args.models,recipes=cfg["gradient_error"]["recipes"],sensor_count=args.sensor_count or cfg["sensor_plan"]["default_count"],snapshots=args.snapshots)
    unique={}; loaded=[]
    for e in entries:
        arrays,meta=load_cache(Path(e["cache_path"])); order,ny,nx=grid_order(arrays["coords_phys"],meta.get("num_x"),meta.get("num_y")); truth=arrays["truth_phys"].reshape(-1)[order].reshape(ny,nx); pred=arrays["recon_phys"].reshape(-1)[order].reshape(ny,nx); grad=np.hypot(*np.gradient(truth)); unique.setdefault((meta["case_id"],meta["time_index"]),(grad,truth)); loaded.append((meta,grad,truth,pred))
    if not loaded:
        per=RESULTS_DIR/"GradientError"/artifact_name("GradientError_per_snapshot",rid,"csv"); write_csv(per,[],fieldnames=["model","recipe","snapshot_index","bin_index","normalized_rmse","normalized_mae","status"])
        sp=RESULTS_DIR/"GradientError"/artifact_name("GradientError_summary",rid,"csv"); write_csv(sp,[],fieldnames=["model","model_label","recipe","recipe_label","bin_index","normalized_rmse","normalized_mae","valid_n"])
        print(f"[OK] {per}\n[OK] {sp}"); return
    pooled=np.concatenate([v[0].reshape(-1) for v in unique.values()]); q=np.linspace(0,1,int(cfg["gradient_error"]["quantile_bins"])+1); edges=np.quantile(pooled,q); scale=np.std(np.concatenate([v[1].reshape(-1) for v in unique.values()]))+1e-12; rows=[]
    for meta,grad,truth,pred in loaded:
        err=pred-truth
        for i in range(len(edges)-1):
            mask=(grad>=edges[i])&(grad<edges[i+1] if i<len(edges)-2 else grad<=edges[i+1]); point_count=int(mask.sum())
            rmse=float(np.sqrt(np.mean(err[mask]**2))/scale) if point_count else float("nan")
            mae=float(np.mean(np.abs(err[mask]))/scale) if point_count else float("nan")
            rows.append({**{k:meta.get(k) for k in ("model","model_label","recipe","recipe_label","case_id","time_index","snapshot_index","sensor_count")},"bin_index":i,"quantile_low":q[i],"quantile_high":q[i+1],"gradient_low":edges[i],"gradient_high":edges[i+1],"normalized_rmse":rmse,"normalized_mae":mae,"point_count":point_count,"normalization":"pooled_truth_standard_deviation","status":"ok" if point_count else "empty_bin"})
    per=RESULTS_DIR/"GradientError"/artifact_name("GradientError_per_snapshot",rid,"csv"); write_csv(per,rows); groups=defaultdict(list)
    for r in rows: groups[(r["model"],r["model_label"],r["recipe"],r["recipe_label"],r["bin_index"],r["quantile_low"],r["quantile_high"])].append(r)
    summary=[]
    for key,vals in groups.items():
        valid=[v for v in vals if v["status"]=="ok" and np.isfinite(float(v["normalized_rmse"]))]
        summary.append({"model":key[0],"model_label":key[1],"recipe":key[2],"recipe_label":key[3],"bin_index":key[4],"quantile_low":key[5],"quantile_high":key[6],"normalized_rmse":np.mean([float(v["normalized_rmse"]) for v in valid]) if valid else float("nan"),"normalized_mae":np.mean([float(v["normalized_mae"]) for v in valid]) if valid else float("nan"),"valid_n":len(valid),"total_n":len(vals),"status":"ok" if valid else "empty_bin"})
    sp=RESULTS_DIR/"GradientError"/artifact_name("GradientError_summary",rid,"csv"); write_csv(sp,summary); print(f"[OK] {per}\n[OK] {sp}")
if __name__=="__main__": main()

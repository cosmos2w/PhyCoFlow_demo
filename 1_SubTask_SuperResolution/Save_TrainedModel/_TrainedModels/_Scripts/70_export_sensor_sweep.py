#!/usr/bin/env python
"""Export nested-sensor-count reconstruction metrics from cache."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from common.cache import load_cache
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, run_id
from common.io_utils import artifact_name, read_csv, write_csv
from common.statistics import relative_l2
from common.workflow import cache_entries, grouped_summary


def sensor_metric_row(entry):
    """Compute only the two sweep metrics; avoid unused SSIM/gradient work."""
    arrays, meta = load_cache(Path(entry["cache_path"]))
    truth = arrays["truth_phys"].reshape(-1)
    pred = arrays["recon_phys"].reshape(-1)
    excluded = np.ones(truth.size, dtype=bool)
    excluded[arrays["obs_indices"].astype(int)] = False
    return {
        **{key: meta.get(key, entry.get(key, "")) for key in (
            "model", "model_label", "recipe", "recipe_label", "case_id", "time_index",
            "physical_time", "snapshot_index", "sensor_count", "sensor_plan_id",
            "checkpoint_kind", "checkpoint_hash", "nfe", "ode_solver", "consistency_mode",
            "generation_seed",
        )},
        "num_x": meta.get("num_x"), "num_y": meta.get("num_y"),
        "physical_rel_l2": relative_l2(truth, pred),
        "physical_rel_l2_sensor_excluded": relative_l2(truth, pred, excluded),
        "status": "ok", "metadata": entry["cache_path"],
    }

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p); p.add_argument("--cache-manifest",type=Path); p.add_argument("--recipes",nargs="+"); p.add_argument("--sensor-counts",nargs="+",type=int); p.add_argument("--snapshots",nargs="+",type=int); p.add_argument("--base-per-snapshot",nargs="+",type=Path,help="Reuse matching finalized rows before computing missing counts."); args=p.parse_args(); cfg=load_config(args.config); rid=run_id(args.run_id); ensure_output_dirs(); recipes=args.recipes or cfg["sensor_sweep"]["recipes"]; counts=args.sensor_counts or cfg["sensor_sweep"]["counts"]; rows=[]
    base={}
    for path in args.base_per_snapshot or []:
        for row in read_csv(path):
            if row.get("status") != "ok": continue
            try: key=(row["model"],row["recipe"],int(row["snapshot_index"]),int(row["sensor_count"]))
            except (KeyError,ValueError): continue
            base[key]=row
    for count in counts:
        for e in cache_entries(args.cache_manifest,models=args.models,recipes=recipes,sensor_count=count,snapshots=args.snapshots):
            key=(e.get("model"),e.get("recipe"),int(e.get("snapshot_index",-1)),int(count))
            if key in base:
                reused=dict(base[key]); reused.setdefault("num_x",e.get("num_x")); reused.setdefault("num_y",e.get("num_y")); rows.append(reused); continue
            try: rows.append(sensor_metric_row(e))
            except Exception as exc: rows.append({"model":e.get("model"),"recipe":e.get("recipe"),"snapshot_index":e.get("snapshot_index"),"sensor_count":count,"physical_rel_l2":np.nan,"status":f"error:{exc}"})
    per=RESULTS_DIR/"SensorSweep"/artifact_name("SensorSweep_per_snapshot",rid,"csv"); write_csv(per,rows); summary=[]
    for metric in ("physical_rel_l2","physical_rel_l2_sensor_excluded"):
        summary+=grouped_summary([r for r in rows if r.get("status")=="ok"],["model","model_label","recipe","recipe_label","sensor_count"],metric,n_boot=cfg["sensor_sweep"]["bootstrap_samples"])
    grid_points=max((int(float(r.get("num_x",0)))*int(float(r.get("num_y",0))) for r in rows if r.get("num_x") and r.get("num_y")),default=0)
    for row in summary: row["evaluation_grid_points"]=grid_points
    sp=RESULTS_DIR/"SensorSweep"/artifact_name("SensorSweep_summary",rid,"csv"); write_csv(sp,summary); print(f"[OK] {per}\n[OK] {sp}")
if __name__=="__main__": main()

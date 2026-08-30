#!/usr/bin/env python
"""Verify cached sensor indices against the current plan and refresh its metadata."""
from __future__ import annotations
import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np

from common.cache import cache_manifest, load_cache_metadata, update_cache_metadata
from common.config import RESULTS_DIR, add_common_args, load_config, run_id
from common.io_utils import matching_or_latest, read_csv

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p,models=False); p.add_argument("--sensor-plan",type=Path); args=p.parse_args(); cfg=load_config(args.config); rid=run_id(args.run_id); plan=args.sensor_plan or matching_or_latest(RESULTS_DIR/"SensorPlans","SensorPlan",args.run_id,"csv"); plan_hash=hashlib.sha256(plan.read_bytes()).hexdigest(); groups=defaultdict(list)
    with plan.open(newline="",encoding="utf-8") as handle:
        for row in csv.DictReader(handle): groups[int(row["snapshot_index"])].append((int(row["sensor_order"]),int(row["point_index"])))
    expected={snap:np.asarray([p for _,p in sorted(rows)],dtype=np.int64) for snap,rows in groups.items()}; manifest=cache_manifest(rid); rows=read_csv(manifest); updated=verified=0
    for row in rows:
        if row.get("status")!="ok" or not row.get("cache_path"): continue
        path=Path(row["cache_path"]); snap=int(row["snapshot_index"]); count=int(row["sensor_count"])
        with np.load(path,allow_pickle=False) as data: actual=data["obs_indices"].astype(np.int64)
        if not np.array_equal(actual,expected[snap][:count]): raise RuntimeError(f"sensor_plan_mismatch: {row['model']} / {row['recipe']} / s{snap} / n{count}")
        verified+=1; meta=load_cache_metadata(path)
        if meta.get("sensor_plan_hash")!=plan_hash or meta.get("sensor_plan_path")!=str(plan):
            update_cache_metadata(path,{"sensor_plan_hash":plan_hash,"sensor_plan_path":str(plan)}); updated+=1
    print(f"[OK] sensor metadata reconciled | verified={verified} updated={updated} plan={plan}")

if __name__=="__main__": main()

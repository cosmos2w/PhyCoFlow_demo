#!/usr/bin/env python
"""Report formal cache coverage without loading models or dense arrays."""
from __future__ import annotations
import argparse
import csv
import time
from collections import Counter
from pathlib import Path

from common.cache import cache_manifest
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, recipe_items, run_id
from common.io_utils import artifact_name, read_csv, write_csv


def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p,models=False); p.add_argument("--expected",type=int); args=p.parse_args()
    cfg=load_config(args.config); rid=run_id(args.run_id); ensure_output_dirs(); path=cache_manifest(rid)
    rows=[]
    if path.exists():
        for attempt in range(5):
            try: rows=read_csv(path); break
            except (csv.Error, UnicodeDecodeError):
                if attempt==4: raise
                time.sleep(.2)
    expected=args.expected or int(cfg["canonical_test"]["max_snapshots"]); observed=Counter(); unique={}
    for row in rows:
        try: key=(row.get("model"),row.get("recipe"),int(row.get("snapshot_index") or -1),int(row.get("sensor_count") or -1))
        except ValueError: continue
        unique[key]=row
    for row in unique.values(): observed[(row.get("model"),row.get("recipe"),int(row.get("sensor_count") or -1),row.get("status"))]+=1
    report=[]
    for model in method_items(cfg):
        for recipe,_ in recipe_items(cfg):
            counts=cfg["sensor_sweep"]["counts"] if recipe in cfg["sensor_sweep"]["recipes"] else [cfg["sensor_plan"]["default_count"]]
            for count in counts:
                ok=observed[(model["key"],recipe,int(count),"ok")]; missing=sum(v for (m,r,c,s),v in observed.items() if (m,r,c)==(model["key"],recipe,int(count)) and s.startswith("missing")); errors=sum(v for (m,r,c,s),v in observed.items() if (m,r,c)==(model["key"],recipe,int(count)) and s not in {"ok"} and not s.startswith("missing")); report.append({"model":model["key"],"recipe":recipe,"sensor_count":count,"expected":expected,"ok":ok,"missing":missing,"errors":errors,"remaining":max(0,expected-ok-missing-errors),"complete":ok+missing+errors>=expected})
    out=RESULTS_DIR/"ReconstructionCache"/artifact_name("CacheStatus",rid,"csv"); write_csv(out,report)
    for row in report:
        if row["sensor_count"]==cfg["sensor_plan"]["default_count"]: print(f"{row['model']:<16} {row['recipe']:<20} n={row['sensor_count']:<3} ok={row['ok']:<3} missing={row['missing']:<3} errors={row['errors']:<3} remaining={row['remaining']:<3}")
    print(f"[OK] {out}")

if __name__=="__main__": main()

#!/usr/bin/env python
"""Remove cache files not referenced by the rebuilt manifest (dry-run by default)."""
from __future__ import annotations
import argparse
from pathlib import Path

from common.cache import cache_manifest
from common.config import RESULTS_DIR
from common.io_utils import read_csv

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run-id",required=True); p.add_argument("--apply",action="store_true")
    p.add_argument("--models", nargs="+"); p.add_argument("--recipes", nargs="+")
    args=p.parse_args(); manifest=cache_manifest(args.run_id)
    if not manifest.exists(): raise FileNotFoundError(manifest)
    referenced={str(Path(r["cache_path"]).resolve()) for r in read_csv(manifest) if r.get("cache_path")}
    root=RESULTS_DIR/"ReconstructionCache"/args.run_id
    candidates=[]
    model_roots = [root / name for name in args.models] if args.models else [root]
    for model_root in model_roots:
        if args.recipes:
            candidates.extend(path for recipe in args.recipes for path in (model_root / recipe).rglob("*.npz"))
        else:
            candidates.extend(model_root.rglob("*.npz"))
    orphans=[path for path in candidates if str(path.resolve()) not in referenced]
    shared_referenced=set()
    for manifest_path in (RESULTS_DIR/"ReconstructionCache").glob("ReconstructionCache_manifest_*.csv"):
        for row in read_csv(manifest_path):
            for key in ("truth_ref","grid_ref"):
                if row.get(key): shared_referenced.add(str(Path(row[key]).resolve()))
    shared_root=RESULTS_DIR/"ReconstructionCache"/"Shared"
    # Shared arrays are global rather than model/recipe scoped.  Leave them
    # untouched during a targeted prune.
    shared_orphans=[] if args.models or args.recipes else [p for p in shared_root.rglob("*.npz") if str(p.resolve()) not in shared_referenced]
    size=sum(p.stat().st_size for p in [*orphans,*shared_orphans])
    print(f"[PRUNE] run_id={args.run_id} | models={args.models or 'all'} | recipes={args.recipes or 'all'} | orphan_files={len(orphans)} | shared_orphans={len(shared_orphans)} | reclaim={size/1024**2:.2f} MiB | apply={args.apply}")
    if args.apply:
        for path in orphans: path.unlink()
        for path in shared_orphans: path.unlink()
        for directory in sorted((p for p in root.rglob("*") if p.is_dir()),reverse=True):
            try: directory.rmdir()
            except OSError: pass
        print("[OK] orphan cache files removed")

if __name__=="__main__": main()

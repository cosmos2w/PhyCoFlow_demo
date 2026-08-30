#!/usr/bin/env python
"""Rebuild a cache manifest from resumable NPZ entries without loading models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common.cache import cache_manifest, update_cache_metadata
from common.config import RESULTS_DIR
from common.io_utils import read_csv, write_csv


def _metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"].item()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--run-id", required=True, help="Cache run ID to scan.")
    args = parser.parse_args()
    path = cache_manifest(args.run_id)
    rows = {}
    chosen_mtime = {}
    plan_by_hash = {
        hashlib.sha256(plan.read_bytes()).hexdigest(): str(plan.resolve())
        for plan in (RESULTS_DIR / "SensorPlans").glob("SensorPlan_*.csv")
    }
    if path.exists():
        for row in read_csv(path):
            try:
                key=(row["model"], row["recipe"], int(row["snapshot_index"]), int(row["sensor_count"]))
            except (KeyError,ValueError):
                print(f"[SKIP] malformed manifest row: model={row.get('model')} recipe={row.get('recipe')}")
                continue
            rows[key]=row
            existing=Path(row.get("cache_path", ""))
            chosen_mtime[key]=existing.stat().st_mtime_ns if existing.is_file() else -1
    root = RESULTS_DIR / "ReconstructionCache" / args.run_id
    for npz_path in root.rglob("*.npz"):
        try:
            meta = _metadata(npz_path)
            if not meta.get("sensor_plan_path") and meta.get("sensor_plan_hash") in plan_by_hash:
                meta = update_cache_metadata(
                    npz_path, {"sensor_plan_path": plan_by_hash[meta["sensor_plan_hash"]]},
                )
            key = (meta["model"], meta["recipe"], int(meta["snapshot_index"]), int(meta["sensor_count"]))
            mtime=npz_path.stat().st_mtime_ns
            if mtime >= chosen_mtime.get(key,-1):
                rows[key] = {**meta, "cache_path": str(npz_path), "cache_identity": npz_path.stem.split("_")[-1]}
                chosen_mtime[key]=mtime
        except Exception as exc:
            print(f"[SKIP] unreadable cache {npz_path}: {exc}")
    ordered = [rows[key] for key in sorted(rows, key=lambda value: (value[0], value[1], value[2], value[3]))]
    write_csv(path, ordered)
    print(f"[OK] {len(ordered)} entries: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

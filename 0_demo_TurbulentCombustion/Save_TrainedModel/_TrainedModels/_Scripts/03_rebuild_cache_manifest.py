#!/usr/bin/env python
"""Rebuild a cache manifest from resumable NPZ entries without loading models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.cache import cache_manifest
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
    if path.exists():
        for row in read_csv(path):
            rows[(row["method"], row["condition"], int(row["snapshot"]))] = row
    root = RESULTS_DIR / "ReconstructionCache" / args.run_id
    for npz_path in root.rglob("*.npz"):
        try:
            meta = _metadata(npz_path)
            key = (meta["method"], meta["condition"], int(meta["snapshot"]))
            rows[key] = {**meta, "cache_path": str(npz_path), "cache_identity": npz_path.stem.split("_")[-1]}
        except Exception as exc:
            print(f"[SKIP] unreadable cache {npz_path}: {exc}")
    ordered = [rows[key] for key in sorted(rows, key=lambda value: (value[0], value[1], value[2]))]
    write_csv(path, ordered)
    print(f"[OK] {len(ordered)} entries: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Build deterministic per-condition/per-snapshot sparse-sensor plans."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from common.config import ARCHIVE_DIR, RESULTS_DIR, SRC_DIR, add_common_args, ensure_output_dirs, load_config, run_id, select_snapshots, stable_seed
from common.io_utils import artifact_name, write_csv, write_json

if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))
import evaluate_coherence as canonical
import model_baseline as baseline_lib
from helpers import TurbulentCombustionH5Dataset, build_sparse_condition


def reference_dataset(cfg, split):
    """Open data using the first complete config; no checkpoint/model is loaded."""
    for method in cfg["methods"]:
        for cond in cfg["conditions"]:
            run_dir = ARCHIVE_DIR / method["directory"] / cond
            path = run_dir / "run_config.yaml"
            if not path.exists(): continue
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            stats = run_dir / "dataset_stats.pt"
            if raw.get("baseline_model"):
                norm = baseline_lib.validate_and_normalize_config(raw)
                return baseline_lib.build_dataset(norm, split=split, stats_path=stats)
            data = canonical.resolve_input_path(str(raw["data"]), label="Dataset", extra_base_dirs=[run_dir, canonical.DEMO_DIR])
            return TurbulentCombustionH5Dataset(str(data), split=split,
                train_ratio=float(raw.get("train_ratio", .9)), seed=int(raw.get("seed", 42)),
                time_stride=int(raw.get("time_stride", 1)), field_names=raw.get("FIELD_NAMES", raw.get("field_names")),
                stats_path=str(stats) if stats.exists() else None)
    raise FileNotFoundError("No complete run_config.yaml was found to locate the dataset.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p, models=False); p.add_argument("--split", default=None, choices=["train", "val", "test"])
    p.add_argument("--snapshots", nargs="+", type=int); p.add_argument("--max-snapshots", type=int)
    p.add_argument("--seed", type=int, default=None); args = p.parse_args()
    cfg = load_config(args.config); rid = run_id(args.run_id); ensure_output_dirs()
    split = args.split or cfg["defaults"]["split"]; base_seed = args.seed or cfg["defaults"]["seed"]
    ds = reference_dataset(cfg, split); snapshots = select_snapshots(len(ds), args.snapshots, args.max_snapshots)
    rows = []
    for condition, spec in cfg["conditions"].items():
        for snapshot in snapshots:
            seed = stable_seed(base_seed, split, condition, snapshot)
            torch.manual_seed(seed); np.random.seed(seed & 0xFFFFFFFF)
            sample = ds[snapshot]; coords = sample["coords"].unsqueeze(0); truth = sample["fields"].unsqueeze(0)
            _, values, mask, indices, fields = build_sparse_condition(coords, truth, spec["cond_fields"], spec["n_obs"], spec["n_obs"])
            valid = mask[0].bool()
            for order, (idx, field, value) in enumerate(zip(indices[0, valid], fields[0, valid], values[0, valid, 0])):
                rows.append({"run_id": rid, "split": split, "condition": condition, "snapshot": snapshot,
                             "sensor_seed": seed, "sensor_order": order, "field_index": int(field),
                             "point_index": int(idx), "normalized_value": float(value)})
    path = RESULTS_DIR / "SensorPlans" / artifact_name("SensorPlan", rid, "csv")
    write_csv(path, rows)
    write_json(RESULTS_DIR / "SensorPlans" / artifact_name("SensorPlan_metadata", rid, "json"),
               {"run_id": rid, "split": split, "base_seed": base_seed, "snapshots": snapshots,
                "dataset_length": len(ds), "conditions": cfg["conditions"]})
    print(f"[OK] {len(rows)} sensor entries: {path}")
    return 0


if __name__ == "__main__": raise SystemExit(main())


#!/usr/bin/env python
"""Build one canonical H test index and deterministic nested H-grid sensors."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np

from common.config import ARCHIVE_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, recipe_items, run_id, stable_seed
from common.dataset_loader import build_run_dataset, canonical_rows, find_snapshot, locate_or_rebuild_manifest
from common.io_utils import artifact_name, read_csv, write_csv, write_json
from common.recipe_registry import resolve_recipe_dir


def reference(cfg):
    for model in method_items(cfg):
        for recipe_key, recipe_spec in recipe_items(cfg):
            run_dir = resolve_recipe_dir(ARCHIVE_DIR / model["directory"], recipe_key, recipe_spec)
            if (run_dir / "run_config.yaml").exists():
                return build_run_dataset(run_dir, model["key"], recipe_key, split=cfg["dataset"]["split"],
                                         eval_resolution=cfg["dataset"]["eval_resolution"])
    raise FileNotFoundError("No model run_config.yaml is available for the canonical dataset.")


def common_time_window(cfg):
    windows=[]
    for model in method_items(cfg):
        for recipe_key,recipe_spec in recipe_items(cfg):
            run_dir=resolve_recipe_dir(ARCHIVE_DIR/model["directory"],recipe_key,recipe_spec)
            if not (run_dir/"run_config.yaml").exists(): continue
            try:
                _,manifest=locate_or_rebuild_manifest(run_dir,RESULTS_DIR/"DatasetStats"/model["key"]/recipe_key)
                start=int(manifest.get("time_start_idx",0)); windows.append(set(range(start,start+int(manifest["n_time"]))))
            except Exception as exc: print(f"[WARN] common-time scan skipped {model['key']} / {recipe_key}: {exc}")
    if not windows: raise RuntimeError("No manifest time windows are available")
    common=set.intersection(*windows)
    if not common: raise RuntimeError("Configured run manifests have no common usable time indices")
    return common


def preserve_existing_cases(dataset,existing,allowed_times,seed):
    rows=[]; allowed=sorted(allowed_times)
    by_case={}
    for i,(_,case,time_index) in enumerate(dataset.entries):
        if int(time_index) in allowed_times: by_case.setdefault(int(case),[]).append((i,int(time_index)))
    for old in sorted(existing,key=lambda r:int(r["snapshot_index"])):
        case_id=int(old["case_id"]); old_time=int(old["time_index"]); choices=by_case[case_id]
        if old_time in allowed_times: dataset_index=find_snapshot(dataset,case_id,old_time); time_index=old_time
        else:
            pick=stable_seed(seed,"common_time_replacement",case_id)%len(choices); dataset_index,time_index=choices[pick]
        rows.append({"snapshot_index":int(old["snapshot_index"]),"dataset_index":dataset_index,"case_id":case_id,
                     "time_index":time_index,"physical_time":float(dataset.times[time_index]),"eval_resolution":dataset.output_resolution or "H",
                     "selection_strategy":"stratified_unique_cases_common_time","selection_seed":int(seed)})
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); add_common_args(p, models=False)
    p.add_argument("--max-snapshots", type=int); p.add_argument("--snapshots", nargs="+", type=int)
    p.add_argument("--canonical-index", type=Path,
                   help="Reuse an authoritative canonical index instead of selecting new sample times.")
    p.add_argument("--counts", nargs="+", type=int); p.add_argument("--seed", type=int)
    args = p.parse_args(); cfg = load_config(args.config); rid = run_id(args.run_id); ensure_output_dirs()
    ds, manifest_path, manifest, _ = reference(cfg)
    maximum = args.max_snapshots if args.max_snapshots is not None else cfg["canonical_test"]["max_snapshots"]
    selection_seed=int(cfg["canonical_test"].get("selection_seed",42)); allowed_times=common_time_window(cfg) if cfg["canonical_test"].get("require_common_time_window",False) else None
    index_path = RESULTS_DIR / "CanonicalTestIndex" / artifact_name("CanonicalTestIndex", rid, "csv")
    existing=read_csv(args.canonical_index) if args.canonical_index else (read_csv(index_path) if index_path.exists() else [])
    if args.canonical_index:
        index_rows = existing
        if len(index_rows) != maximum:
            raise ValueError(f"Authoritative canonical index has {len(index_rows)} rows; expected {maximum}")
        if allowed_times is not None and any(int(row["time_index"]) not in allowed_times for row in index_rows):
            raise ValueError("Authoritative canonical index includes a time outside the common usable window")
    elif existing and len(existing)==maximum and len({r["case_id"] for r in existing})==maximum and allowed_times is not None:
        index_rows=preserve_existing_cases(ds,existing,allowed_times,selection_seed)
    else:
        index_rows = canonical_rows(ds, maximum, strategy=cfg["canonical_test"].get("selection_strategy", "sequential"),
                                    seed=selection_seed,allowed_time_indices=allowed_times)
    if args.snapshots is not None:
        wanted = set(args.snapshots); index_rows = [r for r in index_rows if r["snapshot_index"] in wanted]
    write_csv(index_path, index_rows)
    counts = sorted(set(args.counts or cfg["sensor_plan"]["counts"])); max_count = max(counts)
    base_seed = args.seed if args.seed is not None else cfg["sensor_plan"]["seed"]
    plan_id = cfg["sensor_plan"]["plan_id"]; rows = []
    coords_norm = ds.coords_by_res["H"].cpu().numpy(); coords_phys = ds.coords_raw_by_res["H"].cpu().numpy()
    for identity in index_rows:
        seed = stable_seed(base_seed, plan_id, identity["case_id"], identity["time_index"])
        order = np.random.default_rng(seed).permutation(len(coords_phys))[:max_count]
        for sensor_order, point_index in enumerate(order):
            membership = {f"in_S{count}": sensor_order < count for count in counts}
            rows.append({
                **identity, "sensor_plan_id": plan_id, "sensor_seed": seed, "sensor_order": sensor_order,
                "point_index": int(point_index), **membership,
                "x_norm": float(coords_norm[point_index, 0]), "y_norm": float(coords_norm[point_index, 1]),
                "x_phys": float(coords_phys[point_index, 0]), "y_phys": float(coords_phys[point_index, 1]),
            })
    plan_path = RESULTS_DIR / "SensorPlans" / artifact_name("SensorPlan", rid, "csv")
    write_csv(plan_path, rows)
    plan_hash = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    write_json(RESULTS_DIR / "SensorPlans" / artifact_name("SensorPlan_metadata", rid, "json"), {
        "run_id": rid, "sensor_plan_id": plan_id, "sensor_plan_hash": plan_hash, "counts": counts,
        "nested": True, "canonical_test_index": str(index_path), "manifest": str(manifest_path),
        "selected_raw_field_id": manifest["selected_field_idx"], "selected_raw_field_name": manifest["selected_field_name"],
        "common_time_indices": sorted(allowed_times) if allowed_times is not None else None,
    })
    print(f"[OK] canonical index: {index_path}"); print(f"[OK] nested sensor plan: {plan_path}"); return 0


if __name__ == "__main__": raise SystemExit(main())

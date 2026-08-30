#!/usr/bin/env python
"""Build resumable H-resolution reconstruction caches from a saved sensor plan."""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np

from common.cache import cache_identity, cache_manifest, cache_path, load_cache_metadata, save_cache, save_compact_cache
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, recipe_items, run_id, stable_seed
from common.dataset_loader import find_snapshot
from common.dataset_loader import resolve_run_config_path
from common.io_utils import latest, read_csv, write_csv
from common.model_loader import checkpoint_digest, inspect_artifacts, load_model, status_from_exception
from common.storage import acquire_cache_lock, assert_cache_storage, device_preflight


def sensor_groups(path: Path, counts: set[int], snapshots: set[int] | None):
    grouped = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            snap = int(row["snapshot_index"])
            if snapshots is not None and snap not in snapshots: continue
            grouped.setdefault(snap, []).append(row)
    for snap, all_rows in sorted(grouped.items()):
        all_rows.sort(key=lambda r: int(r["sensor_order"]))
        for count in sorted(counts):
            if len(all_rows) < count:
                raise ValueError(
                    f"Sensor plan {path} contains only {len(all_rows)} rows for snapshot {snap}, "
                    f"but sensor_count={count} was requested. Rebuild the sensor plan with at least "
                    f"{count} nested sensors before generating this cache slice."
                )
            yield snap, count, all_rows[:count]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__); add_common_args(p)
    p.add_argument("--recipes", nargs="+", default=["all"]); p.add_argument("--sensor-plan", type=Path)
    p.add_argument("--sensor-counts", nargs="+", type=int); p.add_argument("--snapshots", nargs="+", type=int)
    p.add_argument("--checkpoint", choices=["last", "best"]); p.add_argument("--allow-checkpoint-fallback", action="store_true")
    p.add_argument("--device"); p.add_argument("--n-steps", type=int); p.add_argument("--ode-solver", choices=["euler", "heun"])
    p.add_argument("--consistency-mode"); p.add_argument("--force", action="store_true")
    p.add_argument(
        "--lock-scope",
        help=(
            "Optional disjoint slice name for concurrent cache generation. "
            "Only use distinct scopes for non-overlapping model/recipe slices; "
            "rebuild the canonical manifest after all slices finish."
        ),
    )
    p.add_argument("--allow-local-cache", action="store_true"); p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(); cfg = load_config(args.config); rid = run_id(args.run_id); ensure_output_dirs()
    defaults = cfg["cache"]
    kind = args.checkpoint or defaults["checkpoint"]; n_steps = args.n_steps or defaults["n_steps"]
    solver = args.ode_solver or defaults["ode_solver"]
    device = args.device or defaults["device"]
    storage_info = assert_cache_storage(cfg, allow_local=args.allow_local_cache)
    device_info = device_preflight(device, cfg)
    if args.preflight_only:
        print("[OK] cache preflight passed"); return 0
    lock_id = f"{rid}__{args.lock_scope}" if args.lock_scope else rid
    cache_lock = acquire_cache_lock(lock_id)
    plan = args.sensor_plan or latest(RESULTS_DIR / "SensorPlans", "SensorPlan", "csv")
    plan_hash = hashlib.sha256(plan.read_bytes()).hexdigest()
    counts = set(args.sensor_counts or [cfg["sensor_plan"]["default_count"]]); snapshots = None if args.snapshots is None else set(args.snapshots)
    # Concurrent disjoint slices must not share write_csv's fixed ``.tmp`` path.
    # Their cache files still live under ``rid``; only the transient manifests
    # are scoped and the canonical manifest is rebuilt once all slices finish.
    manifest_run_id = f"{rid}__{args.lock_scope}" if args.lock_scope else rid
    manifest_path = cache_manifest(manifest_run_id)
    existing_rows = read_csv(manifest_path) if manifest_path.exists() else []
    def row_key(row):
        def integer(value):
            try: return int(value)
            except (TypeError, ValueError): return -1
        return (str(row.get("model", "")), str(row.get("recipe", "")),
                integer(row.get("snapshot_index")), integer(row.get("sensor_count")))
    merged = {}
    for row in existing_rows:
        merged[row_key(row)] = row
    processed_since_flush = 0
    flush_every = max(1, int(defaults.get("manifest_flush_every", 25)))

    def flush_manifest():
        write_csv(manifest_path, [merged[k] for k in sorted(merged, key=lambda x: tuple(map(str, x)))])

    def record(row):
        nonlocal processed_since_flush
        merged[row_key(row)] = row; processed_since_flush += 1
        if processed_since_flush >= flush_every:
            flush_manifest(); processed_since_flush = 0
    groups = list(sensor_groups(plan, counts, snapshots))
    for model_spec in method_items(cfg, args.models):
        consistency = (
            args.consistency_mode
            or defaults.get("consistency_mode_overrides", {}).get(model_spec["key"])
            or defaults["consistency_mode"]
        )
        for recipe_key, recipe_spec in recipe_items(cfg, args.recipes):
            artifact = inspect_artifacts(model_spec, recipe_key, recipe_spec, kind, args.allow_checkpoint_fallback)
            if artifact["status"] != "ok":
                print(f"[SKIP] {model_spec['label']} / {recipe_key} | {artifact['status']}")
                for snap, count, rows in groups:
                    record({"run_id": rid, "model": model_spec["key"], "recipe": recipe_key,
                            "snapshot_index": snap, "dataset_index": rows[0].get("dataset_index", snap),
                            "sensor_count": count, "status": artifact["status"], "cache_path": ""})
                continue
            loaded = None
            try:
                loaded = load_model(model_spec, recipe_key, recipe_spec, checkpoint=kind,
                                    allow_fallback=args.allow_checkpoint_fallback, split=cfg["dataset"]["split"],
                                    eval_resolution=cfg["dataset"]["eval_resolution"], device=device)
                ck_hash = checkpoint_digest(loaded.checkpoint_path)
                mean = loaded.dataset.mean.cpu().numpy(); std = loaded.dataset.std.cpu().numpy()
                for snap, count, rows in groups:
                    first = rows[0]
                    # One stochastic draw per physical sample, shared across
                    # models, recipes, and nested sensor counts for paired comparisons.
                    generation_seed = stable_seed(defaults["generation_seed"], "generation",
                                                  first["case_id"], first["time_index"])
                    dataset_index = find_snapshot(loaded.dataset,int(first["case_id"]),int(first["time_index"]))
                    meta = {
                        "run_id": rid, "model": model_spec["key"], "model_label": model_spec["label"], "recipe": recipe_key,
                        "recipe_label": recipe_spec["label"], "case_id": int(first["case_id"]), "time_index": int(first["time_index"]),
                        "physical_time": float(first["physical_time"]), "snapshot_index": snap, "eval_resolution": cfg["dataset"]["eval_resolution"],
                        "dataset_index": dataset_index, "canonical_dataset_index": int(first.get("dataset_index",snap)),
                        "dataset_name": loaded.manifest["dataset_name"],
                        "sensor_count": count, "sensor_plan_id": first["sensor_plan_id"], "sensor_plan_hash": plan_hash,
                        "sensor_plan_path": str(plan.resolve()),
                        "checkpoint_kind": kind, "checkpoint_path": str(loaded.checkpoint_path), "checkpoint_hash": ck_hash,
                        "checkpoint_mtime_ns": loaded.checkpoint_path.stat().st_mtime_ns,
                        "config_path": str(resolve_run_config_path(loaded.checkpoint_path.parent)),
                        "manifest_path": loaded.dataset.manifest_path, "family": loaded.family, "backbone": loaded.backbone,
                        "generation_seed": generation_seed, "n_steps": n_steps, "nfe": n_steps if loaded.family == "pointcloud_ffm" else 1,
                        "ode_solver": solver if loaded.family == "pointcloud_ffm" else "native", "consistency_mode": consistency,
                        "selected_raw_field_id": loaded.manifest["selected_field_idx"], "selected_raw_field_name": loaded.manifest["selected_field_name"],
                        "native_resolution": "H", "normalization_mean": float(mean.reshape(-1)[0]),
                        "normalization_std": float(std.reshape(-1)[0]), "storage_root": storage_info["target"],
                        "requested_device": device, "device_utilization_at_start": device_info.get("utilization", ""), "status": "ok",
                    }
                    identity = cache_identity(meta); path = cache_path(rid, model_spec["key"], recipe_key, snap, count, identity)
                    if path.exists() and not args.force:
                        cached_meta = load_cache_metadata(path); meta.update(cached_meta); print(f"[CACHE] {model_spec['label']} / {recipe_key} / s{snap} / n{count}")
                    else:
                        try:
                            out = loaded.reconstruct(dataset_index, rows, n_steps=n_steps, ode_solver=solver,
                                                     consistency_mode=consistency, generation_seed=generation_seed)
                        except RuntimeError as exc:
                            if "out of memory" not in str(exc).lower() or not defaults.get("cpu_fallback", True) or loaded.device.type == "cpu":
                                raise
                            print(f"[FALLBACK] {model_spec['label']} / {recipe_key} | GPU OOM -> cpu")
                            loaded.close()
                            loaded = load_model(model_spec, recipe_key, recipe_spec, checkpoint=kind,
                                                allow_fallback=args.allow_checkpoint_fallback, split=cfg["dataset"]["split"],
                                                eval_resolution=cfg["dataset"]["eval_resolution"], device="cpu")
                            mean = loaded.dataset.mean.cpu().numpy(); std = loaded.dataset.std.cpu().numpy()
                            out = loaded.reconstruct(dataset_index, rows, n_steps=n_steps, ode_solver=solver,
                                                     consistency_mode=consistency, generation_seed=generation_seed)
                        arrays = {k: v for k, v in out.items() if isinstance(v, np.ndarray)}
                        truth_phys = arrays["truth_norm"] * std + mean
                        recon_phys = arrays["recon_norm"] * std + mean
                        meta["obs_consistency_applied"] = out["obs_consistency_applied"]
                        res = loaded.manifest["resolutions"]["H"]; meta["num_x"] = res["Num_x"]; meta["num_y"] = res["Num_y"]
                        if defaults.get("storage_mode") == "compact_shared_v1":
                            save_compact_cache(path, recon_phys=recon_phys, truth_phys=truth_phys,
                                               coords_norm=arrays["coords"], coords_phys=arrays["coords_phys"],
                                               obs_indices=arrays["obs_indices"], metadata=meta)
                            meta = load_cache_metadata(path)
                        else:
                            arrays["truth_phys"] = truth_phys; arrays["recon_phys"] = recon_phys
                            save_cache(path, arrays, meta)
                    record({**meta, "cache_path": str(path), "cache_identity": identity})
            except Exception as exc:
                status = status_from_exception(exc); print(f"[ERROR] {model_spec['label']} / {recipe_key} | {type(exc).__name__}: {exc}")
                record({"run_id": rid, "model": model_spec["key"], "recipe": recipe_key, "status": status,
                        "detail": f"{type(exc).__name__}: {exc}", "cache_path": ""})
            finally:
                if loaded is not None: loaded.close()
    flush_manifest(); print(f"[OK] cache manifest: {manifest_path}"); return 0


if __name__ == "__main__": raise SystemExit(main())

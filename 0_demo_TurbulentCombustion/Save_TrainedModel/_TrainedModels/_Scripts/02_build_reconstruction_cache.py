#!/usr/bin/env python
"""Run each available model once per group and write auditable NPZ caches."""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path
import numpy as np
from common.cache import cache_identity, cache_manifest, cache_path, load_cache, save_cache
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, run_id, stable_seed
from common.io_utils import latest, read_csv, write_csv


def _structured_grid_dimensions(cfg: dict) -> tuple[int | None, int | None]:
    """Recover optional explicit grid metadata for cache-only spectral analysis."""
    shared_data = cfg.get("shared", {}).get("data", {}) if isinstance(cfg, dict) else {}
    nx = cfg.get("Num_x", shared_data.get("num_x")) if isinstance(cfg, dict) else None
    ny = cfg.get("Num_y", shared_data.get("num_y")) if isinstance(cfg, dict) else None
    try:
        return int(nx) if nx is not None else None, int(ny) if ny is not None else None
    except (TypeError, ValueError):
        return None, None


def _sensor_plan_info(plan_path: Path) -> tuple[str, dict[str, list[int]]]:
    """Read only split and snapshot IDs; keep large full-test plans out of RAM."""
    snapshots: dict[str, list[int]] = {}
    split = "test"
    seen: dict[str, set[int]] = {}
    with plan_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row.get("split", split)
            condition, snapshot = row["condition"], int(row["snapshot"])
            if snapshot not in seen.setdefault(condition, set()):
                seen[condition].add(snapshot)
                snapshots.setdefault(condition, []).append(snapshot)
    return split, {condition: sorted(values) for condition, values in snapshots.items()}


def _iter_sensor_groups(plan_path: Path, condition: str, snapshots: set[int]):
    """Yield one deterministic sensor group at a time from sorted plan CSV."""
    current_snapshot = None
    current_rows: list[dict] = []
    with plan_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] != condition:
                continue
            snapshot = int(row["snapshot"])
            if snapshot not in snapshots:
                continue
            if current_snapshot is not None and snapshot != current_snapshot:
                yield current_snapshot, current_rows
                current_rows = []
            current_snapshot = snapshot
            current_rows.append(row)
    if current_snapshot is not None:
        yield current_snapshot, current_rows


def _existing_group_is_compatible(
    entries: list[dict], *, run_id_value: str, plan_hash: str, split: str,
    checkpoint: str, n_steps: int, solver: str, consistency: str,
) -> bool:
    """Return whether a complete requested group can be reused without loading a model.

    Successful entries must still point at an NPZ cache generated from the same
    run settings and sensor plan.  Recorded unavailable entries are reusable as
    well: they preserve the prior explicit ``missing config``/``load error``
    status, rather than spending time attempting the same unavailable model.
    ``--force-regenerate`` bypasses this fast path.
    """
    if not entries:
        return False
    for entry in entries:
        if entry.get("run_id", run_id_value) not in ("", run_id_value):
            return False
        if entry.get("status") != "ok":
            continue
        cache = Path(entry.get("cache_path", ""))
        if not cache.is_file():
            return False
        if entry.get("sensor_plan_hash") != plan_hash:
            return False
        if entry.get("split", split) != split:
            return False
        if entry.get("checkpoint_name") not in ("", f"{checkpoint}.pt"):
            return False
        if str(entry.get("n_steps")) != str(n_steps):
            return False
        if entry.get("ode_solver") not in ("", solver):
            return False
        if entry.get("obs_consistency") not in ("", consistency):
            return False
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p); p.add_argument("--sensor-plan", type=Path); p.add_argument("--checkpoint", choices=["last", "best"])
    p.add_argument("--allow-checkpoint-fallback", action="store_true"); p.add_argument("--device")
    p.add_argument("--n-steps", type=int); p.add_argument("--ode-solver", choices=["euler", "heun"])
    p.add_argument("--obs-consistency", choices=["none", "default_hard", "endpoint", "endpoint_smooth"])
    p.add_argument("--conditions", nargs="+", default=["all"], help="Condition names or 'all'.")
    p.add_argument("--snapshots", nargs="+", type=int, help="Subset of snapshots present in the sensor plan.")
    p.add_argument("--manifest-flush-every", type=int, default=25, help="Persist resumable manifest progress after this many snapshots.")
    p.add_argument("--force-regenerate", "--force", dest="force", action="store_true",
                   help="Ignore compatible existing cache entries and rerun reconstruction.")
    args = p.parse_args()
    cfg = load_config(args.config); rid = run_id(args.run_id); ensure_output_dirs(); defaults = cfg["defaults"]
    plan_path = args.sensor_plan or latest(RESULTS_DIR / "SensorPlans", "SensorPlan", "csv")
    plan_bytes = plan_path.read_bytes(); plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    split, plan_snapshots = _sensor_plan_info(plan_path)
    checkpoint = args.checkpoint or defaults["checkpoint"]
    default_n_steps = int(args.n_steps if args.n_steps is not None else defaults["n_steps"])
    solver = args.ode_solver or defaults["ode_solver"]
    # An explicit CLI value wins globally.  Otherwise a method override can
    # safely change one family (DMF-Gen -> default_hard) without altering the
    # established inference policy of the other point-cloud models.
    requested_consistency = args.obs_consistency or defaults["obs_consistency"]
    manifest_path = cache_manifest(rid)
    manifest_by_key = {}
    if manifest_path.exists():
        for existing in read_csv(manifest_path):
            try:
                manifest_by_key[(existing["method"], existing["condition"], int(existing["snapshot"]))] = existing
            except (KeyError, ValueError):
                continue

    def record(row: dict) -> None:
        manifest_by_key[(row["method"], row["condition"], int(row["snapshot"]))] = row

    def flush_manifest() -> None:
        ordered = [manifest_by_key[key] for key in sorted(manifest_by_key, key=lambda value: (value[0], value[1], value[2]))]
        write_csv(manifest_path, ordered)

    processed_since_flush = 0
    for method in method_items(cfg, args.models):
        method_n_steps = int(
            args.n_steps if args.n_steps is not None
            else cfg.get("method_inference_overrides", {}).get(method["name"], {}).get("n_steps", default_n_steps)
        )
        method_consistency = str(
            args.obs_consistency if args.obs_consistency is not None
            else cfg.get("method_inference_overrides", {}).get(method["name"], {}).get("obs_consistency", requested_consistency)
        )
        for condition, condition_cfg in cfg["conditions"].items():
            if "all" not in args.conditions and condition not in args.conditions:
                continue
            snapshots = [value for value in plan_snapshots.get(condition, []) if args.snapshots is None or value in args.snapshots]
            existing_entries = [manifest_by_key.get((method["name"], condition, snapshot)) for snapshot in snapshots]
            if not args.force and all(entry is not None for entry in existing_entries) and _existing_group_is_compatible(
                existing_entries, run_id_value=rid, plan_hash=plan_hash, split=split,
                checkpoint=checkpoint, n_steps=method_n_steps, solver=solver, consistency=method_consistency,
            ):
                ok_count = sum(entry.get("status") == "ok" for entry in existing_entries)
                unavailable = len(existing_entries) - ok_count
                print(f"[FOUND] existing reconstruction cache | {method['name']} / {condition} | "
                      f"reusing {ok_count}/{len(existing_entries)} NPZ entries"
                      + (f", {unavailable} recorded unavailable" if unavailable else ""))
                continue

            loaded = None
            try:
                # Keep cache-only runs independent of PyTorch and optional
                # model-family dependencies.  These imports are needed only
                # after the preflight above finds a real cache miss.
                from common.model_loader import inspect_artifacts, load_model, status_from_exception
                artifact = inspect_artifacts(method, condition, checkpoint, args.allow_checkpoint_fallback)
                if artifact["status"] != "ok":
                    print(f"[SKIP] {method['name']} / {condition} | {artifact['status']}")
                    for snapshot in snapshots:
                        record({"run_id": rid, "method": method["name"], "condition": condition,
                                "snapshot": snapshot, "status": artifact["status"], "detail": artifact["detail"], "cache_path": ""})
                    flush_manifest()
                    continue
                loaded = load_model(method, condition, checkpoint=checkpoint, allow_fallback=args.allow_checkpoint_fallback,
                                    split=split, device=args.device, n_steps=method_n_steps, ode_solver=solver)
                for snapshot, rows in _iter_sensor_groups(plan_path, condition, set(snapshots)):
                    gen_seed = stable_seed(defaults["seed"], "generation", method["name"], condition, snapshot)
                    num_x, num_y = _structured_grid_dimensions(loaded.config)
                    meta = {"run_id": rid, "method": method["name"], "directory": method["directory"], "condition": condition,
                        "family": loaded.family, "backbone": loaded.backbone, "checkpoint_path": str(loaded.checkpoint_path),
                        "checkpoint_name": loaded.checkpoint_name, "checkpoint_mtime": loaded.checkpoint_path.stat().st_mtime_ns,
                        "sensor_plan": str(plan_path), "sensor_plan_hash": plan_hash, "split": split, "snapshot": snapshot,
                        "sensor_seed": int(rows[0]["sensor_seed"]), "generation_seed": gen_seed, "n_steps": method_n_steps,
                        "ode_solver": solver, "obs_consistency": method_consistency, "num_x": num_x, "num_y": num_y, "status": "ok"}
                    identity = cache_identity(meta); path = cache_path(rid, method["name"], condition, snapshot, identity)
                    reuse_cache = path.exists() and not args.force
                    if reuse_cache:
                        print(f"[CACHE] {method['name']} / {condition} / s{snapshot:04d}")
                        # Preserve the exact inference metadata originally
                        # recorded with the cache while rebuilding a manifest.
                        try:
                            _, cached_meta = load_cache(path)
                            meta.update(cached_meta)
                        except Exception as exc:
                            # A prior interrupted run can only leave a stale
                            # cache at this identity.  Rebuild that snapshot
                            # instead of letting one bad NPZ abort the full
                            # model-condition group.
                            reuse_cache = False
                            print(f"[REBUILD] {method['name']} / {condition} / s{snapshot:04d} | unreadable cache: {exc}")
                    if not reuse_cache:
                        out = loaded.reconstruct(snapshot, condition_cfg, rows, n_steps=method_n_steps, ode_solver=solver,
                                                 obs_consistency=method_consistency, generation_seed=gen_seed)
                        sample = loaded.dataset[snapshot]; mean = loaded.dataset.mean.cpu().numpy(); std = loaded.dataset.std.cpu().numpy()
                        truth = out["truth"][0].detach().cpu().numpy(); recon = out["recon"][0].detach().cpu().numpy()
                        valid = out["obs_mask"][0].bool()
                        arrays = {"truth_norm": truth, "recon_norm": recon, "truth_phys": truth * std + mean,
                            "recon_phys": recon * std + mean, "coords_norm": out["coords"][0].detach().cpu().numpy(),
                            "coords_phys": sample.get("coords_raw", sample["coords"]).cpu().numpy(),
                            "obs_indices": out["obs_indices"][0, valid].detach().cpu().numpy(),
                            "obs_field_ids": out["obs_field_ids"][0, valid].detach().cpu().numpy(),
                            "obs_values_norm": out["obs_values"][0, valid, 0].detach().cpu().numpy()}
                        meta["obs_consistency_applied"] = out["obs_consistency_applied"]
                        save_cache(path, arrays, meta)
                    record({**meta, "cache_path": str(path), "cache_identity": identity})
                    processed_since_flush += 1
                    if processed_since_flush >= max(1, args.manifest_flush_every):
                        flush_manifest()
                        processed_since_flush = 0
            except Exception as exc:
                status = status_from_exception(exc) if "status_from_exception" in locals() else "load error"
                print(f"[ERROR] {method['name']} / {condition} | {type(exc).__name__}: {exc}")
                for snapshot in snapshots:
                    record({"run_id": rid, "method": method["name"], "condition": condition,
                            "snapshot": snapshot, "status": status, "detail": f"{type(exc).__name__}: {exc}", "cache_path": ""})
            finally:
                if loaded is not None: loaded.close()
            flush_manifest()
    flush_manifest()
    print(f"[OK] cache manifest: {manifest_path}"); return 0


if __name__ == "__main__": raise SystemExit(main())

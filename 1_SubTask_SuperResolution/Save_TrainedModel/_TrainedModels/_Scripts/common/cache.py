"""Reconstruction-cache storage, indexing, and invalidation."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from .config import RESULTS_DIR, short_hash


def cache_identity(meta: dict[str, Any]) -> str:
    keys = ("model", "recipe", "case_id", "time_index", "snapshot_index", "eval_resolution",
            "sensor_count", "sensor_plan_id", "sensor_plan_hash", "checkpoint_kind", "checkpoint_hash",
            "checkpoint_mtime_ns", "n_steps", "ode_solver", "consistency_mode", "generation_seed")
    return short_hash({k: meta.get(k) for k in keys})


def cache_path(run_id: str, model: str, recipe: str, snapshot: int, sensor_count: int, identity: str) -> Path:
    safe = model.replace(" ", "_").replace("/", "-")
    return RESULTS_DIR / "ReconstructionCache" / run_id / safe / recipe / f"RecCache_s{snapshot:04d}_n{sensor_count}_{identity}.npz"


def save_cache(path: Path, arrays: dict[str, Any], metadata: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays, metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    return path


def _atomic_savez(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def shared_grid_path(meta: dict[str, Any]) -> Path:
    identity = short_hash({k: meta.get(k) for k in ("dataset_name", "eval_resolution", "num_x", "num_y")})
    return RESULTS_DIR / "ReconstructionCache" / "Shared" / "Grid" / f"Grid_{meta.get('eval_resolution','H')}_{identity}.npz"


def shared_truth_path(meta: dict[str, Any]) -> Path:
    identity = short_hash({k: meta.get(k) for k in (
        "dataset_name", "selected_raw_field_id", "case_id", "time_index", "eval_resolution"
    )})
    return RESULTS_DIR / "ReconstructionCache" / "Shared" / "Truth" / f"Truth_s{int(meta['snapshot_index']):04d}_{identity}.npz"


def save_compact_cache(path: Path, *, recon_phys: np.ndarray, truth_phys: np.ndarray,
                       coords_norm: np.ndarray, coords_phys: np.ndarray,
                       obs_indices: np.ndarray, metadata: dict[str, Any]) -> Path:
    """Store one reconstruction plus references to shared grid/truth arrays.

    Formal fields remain float32.  Hydration in :func:`load_cache` restores the
    legacy array interface, so analysis scripts do not duplicate storage logic.
    """
    grid_path = shared_grid_path(metadata)
    truth_path = shared_truth_path(metadata)
    if not grid_path.exists():
        _atomic_savez(grid_path, {
            "coords_norm": np.asarray(coords_norm, dtype=np.float32),
            "coords_phys": np.asarray(coords_phys, dtype=np.float32),
        })
    if not truth_path.exists():
        _atomic_savez(truth_path, {"truth_phys": np.asarray(truth_phys, dtype=np.float32)})
    metadata = dict(metadata)
    metadata.update({
        "storage_mode": "compact_shared_v1", "grid_ref": str(grid_path),
        "truth_ref": str(truth_path), "array_dtype": "float32",
    })
    _atomic_savez(path, {
        "recon_phys": np.asarray(recon_phys, dtype=np.float32),
        "obs_indices": np.asarray(obs_indices, dtype=np.uint16),
        "metadata_json": np.array(json.dumps(metadata, sort_keys=True)),
    })
    return path


def load_cache_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return json.loads(str(data["metadata_json"].item()))


def update_cache_metadata(path: Path, updates: dict[str, Any]) -> dict:
    """Atomically update metadata while preserving compact/full cache arrays."""
    with np.load(path,allow_pickle=False) as data:
        arrays={k:data[k] for k in data.files if k!="metadata_json"}
        meta=json.loads(str(data["metadata_json"].item()))
    meta.update(updates); arrays["metadata_json"]=np.array(json.dumps(meta,sort_keys=True)); _atomic_savez(path,arrays); return meta


def load_cache(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "metadata_json"}
        meta = json.loads(str(data["metadata_json"].item()))
    if meta.get("storage_mode") == "compact_shared_v1":
        with np.load(meta["grid_ref"], allow_pickle=False) as grid:
            coords_norm = grid["coords_norm"].astype(np.float32, copy=False)
            coords_phys = grid["coords_phys"].astype(np.float32, copy=False)
        with np.load(meta["truth_ref"], allow_pickle=False) as truth_file:
            truth_phys = truth_file["truth_phys"].astype(np.float32, copy=False)
        mean = float(meta["normalization_mean"]); std = float(meta["normalization_std"])
        recon_phys = arrays["recon_phys"].astype(np.float32, copy=False)
        obs_indices = arrays["obs_indices"].astype(np.int64, copy=False)
        arrays.update({
            "coords": coords_norm, "coords_norm": coords_norm, "coords_phys": coords_phys,
            "truth_phys": truth_phys, "truth_norm": (truth_phys - mean) / std,
            "recon_norm": (recon_phys - mean) / std, "obs_indices": obs_indices,
            "obs_coords_norm": coords_norm[obs_indices], "obs_coords_phys": coords_phys[obs_indices],
            "obs_values_phys": truth_phys.reshape(-1)[obs_indices],
            "obs_values_norm": ((truth_phys.reshape(-1)[obs_indices] - mean) / std),
        })
    return arrays, meta


def cache_manifest(run_id: str) -> Path:
    return RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{run_id}.csv"

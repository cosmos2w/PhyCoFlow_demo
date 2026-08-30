"""Reconstruction-cache storage, indexing, and invalidation."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
from .config import RESULTS_DIR, short_hash


def cache_identity(meta: dict[str, Any]) -> str:
    keys = ("method", "condition", "checkpoint_path", "checkpoint_mtime", "sensor_plan_hash",
            "split", "snapshot", "n_steps", "ode_solver", "obs_consistency", "generation_seed")
    return short_hash({k: meta.get(k) for k in keys})


def cache_path(run_id: str, method: str, condition: str, snapshot: int, identity: str) -> Path:
    safe = method.replace(" ", "_").replace("/", "-")
    return RESULTS_DIR / "ReconstructionCache" / run_id / safe / condition / f"RecCache_s{snapshot:04d}_{identity}.npz"


def save_cache(path: Path, arrays: dict[str, Any], metadata: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write-then-replace prevents a cancelled long cache run from leaving a
    # partially written NPZ at the final identity path.
    temporary = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(temporary, **arrays, metadata_json=np.array(json.dumps(metadata, sort_keys=True)))
    os.replace(temporary, path)
    return path


def load_cache(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "metadata_json"}
        meta = json.loads(str(data["metadata_json"].item()))
    return arrays, meta


def cache_manifest(run_id: str) -> Path:
    return RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{run_id}.csv"

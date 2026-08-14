"""Atomic raw trajectory storage and provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np


SCHEMA_VERSION = "1.0"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def git_commit(repository_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_versions() -> dict[str, str]:
    import matplotlib
    import scipy

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "h5py": h5py.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["torch_cuda"] = str(torch.version.cuda)
    except ImportError:
        pass
    return versions


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def trajectory_paths(raw_dir: Path, trajectory_id: int) -> tuple[Path, Path]:
    stem = f"trajectory_{trajectory_id:06d}"
    trajectory_dir = raw_dir / "trajectories"
    return trajectory_dir / f"{stem}.npz", trajectory_dir / f"{stem}.json"


def raw_trajectory_is_complete(raw_dir: Path, trajectory_id: int) -> bool:
    npz_path, json_path = trajectory_paths(raw_dir, trajectory_id)
    if not npz_path.is_file() or not json_path.is_file():
        return False
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        with np.load(npz_path, allow_pickle=False) as payload:
            required = {"state", "time", "step", "x"}
            if not required.issubset(payload.files):
                return False
            if payload["state"].shape[0] != payload["time"].shape[0]:
                return False
        return metadata.get("sha256") == sha256_file(npz_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def write_raw_trajectory(
    raw_dir: Path,
    trajectory_id: int,
    result: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    npz_path, json_path = trajectory_paths(raw_dir, trajectory_id)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if (npz_path.exists() or json_path.exists()) and not overwrite:
        raise FileExistsError(
            f"trajectory {trajectory_id} already exists; use --resume or --overwrite intentionally"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{npz_path.stem}.", suffix=".npz.tmp", dir=npz_path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **result)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, npz_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise

    metadata = dict(metadata)
    metadata["schema_version"] = SCHEMA_VERSION
    metadata["trajectory_id"] = trajectory_id
    metadata["array_shapes"] = {name: list(value.shape) for name, value in result.items()}
    metadata["array_dtypes"] = {name: str(value.dtype) for name, value in result.items()}
    metadata["sha256"] = sha256_file(npz_path)
    atomic_write_json(json_path, metadata)
    return npz_path, json_path


def load_raw_trajectory(npz_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if npz_path.suffix != ".npz":
        raise ValueError(f"expected an NPZ trajectory, got {npz_path}")
    json_path = npz_path.with_suffix(".json")
    if not json_path.is_file():
        raise FileNotFoundError(f"missing trajectory metadata: {json_path}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("sha256") != sha256_file(npz_path):
        raise ValueError(f"checksum mismatch for {npz_path}")
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    return arrays, metadata


def list_raw_trajectories(raw_dir: Path) -> list[Path]:
    paths = sorted((raw_dir / "trajectories").glob("trajectory_*.npz"))
    if not paths:
        raise FileNotFoundError(f"no raw trajectories found under {raw_dir / 'trajectories'}")
    return paths

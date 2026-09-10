#!/usr/bin/env python
"""Collect the cache-only source package for Figure 5 V7R2.

The collector is deliberately separate from the V7 release.  It consumes the
frozen Cond_T evaluation tables, the six saved ablation cache sets, and the
same-cohort Senseiver Cond_T/last.pt cache.  It never loads a model or runs
inference.  All field spectra and the Senseiver high-band diagnostics are
computed with the checked-in ``common.spectral`` estimator and the same native
structured-grid geometry used by the evaluation package.

The output directory is a release boundary.  A populated directory from a
different release is rejected, while rerunning the same collector release is
allowed so a failed interrupted write can be repaired without touching older
outputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import traceback
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
DEMO = REPO / "0_demo_TurbulentCombustion"
SCRIPTS = DEMO / "Save_TrainedModel/_TrainedModels/_Scripts"
SPECTRAL_ANALYSIS = DEMO / "src/analyze_ablation_high_frequency.py"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common.cache import load_cache  # noqa: E402
from common.spectral import band_energy_breakdown, radial_spectrum, recover_structured_grid  # noqa: E402


FIELDS = ("CH4", "CO", "T", "U1", "p")
UNOBSERVED = ("CH4", "CO", "U1", "p")
ABLATIONS = ("A0", "A1", "A2", "A3", "A4", "A5")
MAIN_METHODS = ("A0", "A2", "A3", "A5", "A4", "Senseiver")
ALL_METHODS = (*ABLATIONS, "Senseiver")
EXPECTED_STATES = 1000
NX, NY = 403, 100
SENSOR_HASH = "c4c3233b2a5d87c8ae59f0673c85368d33d78df5b07adf4240654f5f7fb0b5aa"
BLOCK_LENGTH = 20
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260910
REPORT_SHA256 = "84d5c9fc708988f51f95d4668ef84786822367570e19bd20f71c466e632a1138"
SENSEIVER_CHECKPOINT_SHA256 = "b055ccc8ad86ab2d4d44a527620f4c0db06009c26de26d8d52db44065fc48503"
SENSEIVER_EPOCH = 5000
SENSEIVER_UNOBSERVED_ANCHOR = 0.1429897546212702
A0_LAST_CHECKPOINT_SHA256 = "71a9e1004010558f7137f4140eb976d51d367d39e9bc3473334912257cbaa541"
A0_LAST_EPOCH = 7520

PUBLISHED_LABELS = {
    "Truth": "Truth",
    "A0": "Full model",
    "A1": "Deterministic regression",
    "A2": "No sensor feedback",
    "A3": "No local conditioning",
    "A4": "IID Gaussian prior",
    "A5": "Local-only conditioning",
    "Senseiver": "Senseiver",
}

EVALUATION = DEMO / "Save_TrainedModel/ablation_condT/evaluation_20260910"
REPORT = EVALUATION / "Ablation_CondT_Evaluation_20260910.md"
HF_PER_STATE = EVALUATION / "high_frequency/per_state_high_frequency.csv"
HF_SUMMARY = EVALUATION / "high_frequency/summary_high_frequency.csv"
RECON_PER_STATE = EVALUATION / "last/metrics/per_state_metrics.csv"
RECON_SUMMARY = EVALUATION / "last/metrics/summary_metrics.csv"
SPECTRAL_U1 = EVALUATION / "high_frequency/per_state_U1_spectra.csv"
SPECTRAL_SNAPSHOT = EVALUATION / "last/metrics/snapshot0_spectra.csv"
INHERITED_V7_SOURCE_MANIFEST = ROOT / "results/derived/20260910_1540/source_manifest.json"
INHERITED_V7_BUILD_MANIFEST = ROOT / "results/derived/20260910_1540/build_manifest.json"
INHERITED_V6_BUILD_MANIFEST = ROOT / "results/derived/20260904_1200/build_manifest.json"
INHERITED_PANEL_FILES = {
    "a": (
        ROOT / "results/derived/20260910_1540/benchmark_main_a_samples.csv",
        ROOT / "results/derived/20260910_1540/benchmark_main_a_summary.csv",
    ),
    "b": (
        ROOT / "results/derived/20260910_1540/benchmark_main_b_samples.csv",
        ROOT / "results/derived/20260910_1540/benchmark_main_b_summary.csv",
    ),
    "c": (ROOT / "results/derived/20260910_1540/benchmark_main_d.csv",),
    "f": (ROOT / "results/derived/20260910_1540/benchmark_main_f.csv",),
}
SENSEIVER_MANIFEST = Path(
    "/data/wanglz/Cache/PhyCoFlow_TurbulentCombustion_Process_Results/"
    "ReconstructionCache/ReconstructionCache_manifest_paper_full_20260711.csv"
)
SENSEIVER_FIELDL2 = Path(
    "/data/wanglz/Cache/PhyCoFlow_TurbulentCombustion_Process_Results/"
    "FieldL2/FieldL2_per_snapshot_paper_full_20260711.csv"
)
# The figure environment intentionally contains the lightweight cache/spectral
# dependencies and does not install torch.  Checkpoint metadata is read in the
# repository's training environment through a short CPU-only subprocess; this
# still loads metadata only and never constructs a model or runs inference.
CHECKPOINT_METADATA_RUNTIME = Path("/home/wanglz/miniconda3/envs/phycoflow_env/bin/python")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=json_default) + "\n", encoding="utf-8")


def parse_snapshot(path: Path) -> int:
    match = re.search(r"RecCache_s(\d+)(?:_|\.)", path.name)
    if match is None:
        raise ValueError(f"Cache filename has no snapshot identity: {path}")
    return int(match.group(1))


def require_file(path: Path, label: str, errors: list[str]) -> None:
    if not path.is_file():
        errors.append(f"missing required {label}: {path}")


def checkpoint_epoch(path: Path) -> tuple[int | None, str | None]:
    """Read the recorded epoch from a saved checkpoint, without inference.

    The plotting environment has no torch dependency.  Use the repository's
    existing training environment and return the runtime alongside the value
    so the provenance record makes this source route explicit.
    """

    if not CHECKPOINT_METADATA_RUNTIME.is_file():
        return None, None
    probe = r"""
import json
import sys
import torch

path = sys.argv[1]
try:
    payload = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    payload = torch.load(path, map_location="cpu")
epoch = payload.get("epoch") if isinstance(payload, dict) else None
if epoch is None and isinstance(payload, dict):
    for key in ("global_epoch", "current_epoch"):
        if payload.get(key) is not None:
            epoch = payload[key]
            break
if epoch is None:
    raise SystemExit("checkpoint payload has no recorded epoch")
print(json.dumps({"epoch": int(epoch)}))
"""
    try:
        result = subprocess.run(
            [str(CHECKPOINT_METADATA_RUNTIME), "-c", probe, str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, str(CHECKPOINT_METADATA_RUNTIME)
    if result.returncode != 0:
        return None, str(CHECKPOINT_METADATA_RUNTIME)
    try:
        return int(json.loads(result.stdout)["epoch"]), str(CHECKPOINT_METADATA_RUNTIME)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None, str(CHECKPOINT_METADATA_RUNTIME)


def _float_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Expected a non-empty finite numeric vector")
    return array


def block_bootstrap_ci(values: Iterable[float], *, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """Circular moving-block mean interval over states, matching the evaluator."""

    array = _float_array(values)
    block = min(BLOCK_LENGTH, array.size)
    n_blocks = int(np.ceil(array.size / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, array.size, size=(BOOTSTRAP_RESAMPLES, n_blocks))
    offsets = np.arange(block, dtype=np.int64)
    indices = ((starts[..., None] + offsets) % array.size).reshape(BOOTSTRAP_RESAMPLES, -1)
    draws = array[indices[:, : array.size]]
    means = draws.mean(axis=1)
    return tuple(float(item) for item in np.quantile(means, [0.025, 0.975]))


def describe(values: Iterable[float], *, bootstrap: bool = True, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    array = _float_array(values)
    result: dict[str, Any] = {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
        "p95": float(np.quantile(array, 0.95)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }
    if bootstrap:
        result["block20_ci95_low"], result["block20_ci95_high"] = block_bootstrap_ci(array, seed=seed)
        result["ci_method"] = "circular moving-block bootstrap; block_length=20; resamples=2000"
    return result


def read_state_index(errors: list[str]) -> pd.DataFrame:
    path = EVALUATION / "last/manifest_A0.csv"
    require_file(path, "A0 last manifest", errors)
    if not path.is_file():
        return pd.DataFrame(columns=["snapshot", "time_index"])
    frame = pd.read_csv(path, usecols=["snapshot", "time_index", "sensor_plan_hash", "sensor_seed", "condition", "split"])
    required = {"snapshot", "time_index", "sensor_plan_hash", "sensor_seed", "condition", "split"}
    if not required.issubset(frame.columns):
        errors.append(f"A0 manifest lacks required columns: {sorted(required - set(frame.columns))}")
        return pd.DataFrame(columns=["snapshot", "time_index"])
    if len(frame) != EXPECTED_STATES or set(frame.snapshot.astype(int)) != set(range(EXPECTED_STATES)):
        errors.append(f"A0 state manifest must contain snapshots 0..999 exactly; rows={len(frame)}")
    if frame.snapshot.duplicated().any() or frame.time_index.duplicated().any():
        errors.append("A0 state manifest has duplicate snapshot or time_index identities")
    bad = frame[frame.sensor_plan_hash.astype(str) != SENSOR_HASH]
    if not bad.empty:
        errors.append(f"A0 state manifest has {len(bad)} sensor-plan identity mismatches")
    if not np.isfinite(frame.sensor_seed.astype(float)).all():
        errors.append("A0 state manifest has non-finite per-state sensor seeds")
    if set(frame.condition.astype(str)) != {"Cond_T"} or set(frame.split.astype(str)) != {"test"}:
        errors.append("A0 state manifest condition/split is not exactly Cond_T/test")
    return frame[["snapshot", "time_index", "sensor_plan_hash", "sensor_seed", "condition", "split"]].sort_values("snapshot").reset_index(drop=True)


def read_method_manifests(errors: list[str], state_index: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    manifests: dict[str, pd.DataFrame] = {}
    usecols = [
        "cache_path", "checkpoint_epoch", "checkpoint_name", "checkpoint_path", "checkpoint_sha256",
        "code_sha256", "config_sha256", "stats_sha256", "sensor_plan", "generation_seed_policy", "variant",
        "condition", "method", "num_x", "num_y", "obs_consistency_applied", "sensor_plan_hash",
        "sensor_seed", "snapshot", "split", "time_index",
    ]
    for method in ABLATIONS:
        path = EVALUATION / f"last/manifest_{method}.csv"
        require_file(path, f"{method} last manifest", errors)
        if not path.is_file():
            continue
        frame = pd.read_csv(path, usecols=lambda col: col in usecols)
        missing = set(usecols) - set(frame.columns)
        if missing:
            errors.append(f"{method} manifest lacks required columns: {sorted(missing)}")
            continue
        frame["snapshot"] = frame.snapshot.astype(int)
        for name in ("sensor_seed", "num_x", "num_y"):
            frame[name] = frame[name].astype(int)
        if len(frame) != EXPECTED_STATES or set(frame.snapshot) != set(range(EXPECTED_STATES)):
            errors.append(f"{method} manifest must contain snapshots 0..999 exactly; rows={len(frame)}")
        if frame.snapshot.duplicated().any():
            errors.append(f"{method} manifest has duplicate snapshot identities")
        checks = {
            "method": method,
            "condition": "Cond_T",
            "num_x": NX,
            "num_y": NY,
            "sensor_plan_hash": SENSOR_HASH,
            "split": "test",
            "checkpoint_name": "last.pt",
            "obs_consistency_applied": "default_hard",
        }
        for column, expected in checks.items():
            if set(frame[column].astype(str)) != {str(expected)}:
                errors.append(f"{method} manifest has ambiguous {column}: {sorted(frame[column].astype(str).unique())[:5]}")
        if state_index is not None and not state_index.empty:
            ordered = frame.sort_values("snapshot").reset_index(drop=True)
            reference = state_index.sort_values("snapshot").reset_index(drop=True)
            for column in ("time_index", "sensor_plan_hash", "sensor_seed"):
                left = ordered[column].astype(str).to_numpy()
                right = reference[column].astype(str).to_numpy()
                if not np.array_equal(left, right):
                    errors.append(f"{method} per-state {column} sequence differs from A0 state identity")
        manifests[method] = frame.sort_values("snapshot").reset_index(drop=True)
    return manifests


def read_senseiver_manifest(errors: list[str]) -> tuple[pd.DataFrame, Path | None]:
    require_file(SENSEIVER_MANIFEST, "Senseiver reconstruction-cache manifest", errors)
    if not SENSEIVER_MANIFEST.is_file():
        return pd.DataFrame(), None
    all_rows = pd.read_csv(SENSEIVER_MANIFEST)
    required = {
        "cache_path", "checkpoint_name", "checkpoint_path", "condition", "method", "num_x", "num_y",
        "obs_consistency_applied", "sensor_plan_hash", "sensor_seed", "snapshot", "split", "status",
    }
    if not required.issubset(all_rows.columns):
        errors.append(f"Senseiver manifest lacks required columns: {sorted(required - set(all_rows.columns))}")
        return pd.DataFrame(), None
    frame = all_rows[(all_rows.method.astype(str) == "Senseiver") & (all_rows.condition.astype(str) == "Cond_T")].copy()
    if len(frame) != EXPECTED_STATES or set(frame.snapshot.astype(int)) != set(range(EXPECTED_STATES)):
        errors.append(f"Senseiver Cond_T manifest must contain snapshots 0..999 exactly; rows={len(frame)}")
    for column, expected in {
        "checkpoint_name": "last.pt", "condition": "Cond_T", "method": "Senseiver", "num_x": NX,
        "num_y": NY, "sensor_plan_hash": SENSOR_HASH, "split": "test", "status": "ok",
    }.items():
        if column not in frame or set(frame[column].astype(str)) != {str(expected)}:
            errors.append(f"Senseiver manifest has ambiguous {column}: {sorted(frame[column].astype(str).unique())[:5] if column in frame else []}")
    if frame.snapshot.duplicated().any():
        errors.append("Senseiver manifest has duplicate snapshot identities")
    frame["snapshot"] = frame.snapshot.astype(int)
    frame = frame.sort_values("snapshot").reset_index(drop=True)
    cache_paths = [Path(str(path)) for path in frame.cache_path]
    if not all(path.is_file() for path in cache_paths):
        # A missing explicitly listed path is a hard source gate.  Do not
        # silently switch to a mirror or adjacent cache root.
        missing = [str(path) for path in cache_paths if not path.is_file()]
        errors.append(f"Senseiver manifest has missing explicitly listed cache paths (first five): {missing[:5]}")
    checkpoint = Path(str(frame.checkpoint_path.iloc[0])) if len(frame) else None
    return frame, checkpoint


def cache_paths_from_manifest(frame: pd.DataFrame, method: str, errors: list[str]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for row in frame.itertuples(index=False):
        snapshot = int(row.snapshot)
        path = Path(str(row.cache_path))
        if not path.is_file():
            errors.append(f"{method} snapshot {snapshot} cache missing: {path}")
        if snapshot in paths:
            errors.append(f"{method} duplicate cache snapshot {snapshot}")
        paths[snapshot] = path
    if set(paths) != set(range(EXPECTED_STATES)):
        errors.append(f"{method} cache path coverage is {len(paths)}/1000")
    return paths


def make_geometry(example_path: Path) -> dict[str, Any]:
    arrays, _ = load_cache(example_path)
    truth = np.asarray(arrays["truth_phys"], dtype=float)
    coords = np.asarray(arrays["coords_phys"], dtype=float)
    grid = recover_structured_grid(coords, num_x=NX, num_y=NY, coordinate_mode="auto", spacing_tolerance=0.02)
    ordered = truth[grid["sort_idx"]].T.reshape(len(FIELDS), NY, NX)
    canonical = radial_spectrum(
        ordered[0], dx=grid["used_dx"], dy=grid["used_dy"], remove_mean=True,
        window="none", use_isotropic_cutoff=True, min_shell_count=4,
    )
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(NX, d=grid["used_dx"]))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(NY, d=grid["used_dy"]))
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_mag = np.hypot(kx_grid, ky_grid)
    dk = max(float(min(np.min(np.abs(np.diff(np.unique(kx)))), np.min(np.abs(np.diff(np.unique(ky)))))), 1e-30)
    shell_id = np.rint(k_mag / dk).astype(np.int64)
    shell_count = np.bincount(shell_id.ravel())
    shell_k_sum = np.bincount(shell_id.ravel(), weights=k_mag.ravel())
    shell_k = shell_k_sum / np.maximum(shell_count, 1)
    valid_shells = np.asarray(canonical["wavenumber_index"], dtype=np.int64)
    valid_mask = np.isin(shell_id, valid_shells)
    canonical_wavenumber = np.asarray(canonical["wavenumber"], dtype=float)
    kmax = float(np.max(canonical_wavenumber))
    high_shells = valid_shells[canonical_wavenumber > (2.0 * kmax / 3.0)]
    high_mask = np.isin(shell_id, high_shells)
    return {
        "nx": NX, "ny": NY, "sort_idx": np.asarray(grid["sort_idx"]), "dx": float(grid["used_dx"]),
        "dy": float(grid["used_dy"]), "coordinate_mode_used": grid["coordinate_mode_used"],
        "shell_id": shell_id, "shell_count": shell_count, "shell_k": shell_k,
        "valid_shells": valid_shells, "valid_mask": valid_mask, "high_shells": high_shells,
        "high_mask": high_mask, "canonical_wavenumber": canonical_wavenumber, "kmax": kmax,
        "high_k_min": float(np.min(shell_k[high_shells])), "high_k_max": float(np.max(shell_k[high_shells])),
        "retained_mode_count": int(np.count_nonzero(valid_mask)), "high_mode_count": int(np.count_nonzero(high_mask)),
    }


def prepared_fft(field_grid: np.ndarray, geometry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    centered = np.asarray(field_grid, dtype=np.float64) - float(np.mean(field_grid, dtype=np.float64))
    fft = np.fft.fftshift(np.fft.fft2(centered))
    psd = (np.abs(fft) ** 2) / float(geometry["nx"] * geometry["ny"])
    return centered, psd


def shell_means(psd: np.ndarray, geometry: dict[str, Any]) -> np.ndarray:
    sums = np.bincount(
        geometry["shell_id"].ravel(), weights=np.asarray(psd).ravel(), minlength=len(geometry["shell_count"])
    )
    means = sums / np.maximum(geometry["shell_count"], 1)
    return means[geometry["valid_shells"]]


def cache_spectral_metrics(truth: np.ndarray, recon: np.ndarray, geometry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    truth_grids = truth[geometry["sort_idx"]].T.reshape(len(FIELDS), NY, NX)
    recon_grids = recon[geometry["sort_idx"]].T.reshape(len(FIELDS), NY, NX)
    truth_shells = np.empty((len(FIELDS), len(geometry["valid_shells"])), dtype=float)
    recon_shells = np.empty_like(truth_shells)
    high_l2 = np.empty(len(FIELDS), dtype=float)
    canonical_ratio = np.empty(len(FIELDS), dtype=float)
    canonical_lsd = np.empty(len(FIELDS), dtype=float)
    for index in range(len(FIELDS)):
        truth_centered, truth_psd = prepared_fft(truth_grids[index], geometry)
        recon_centered, recon_psd = prepared_fft(recon_grids[index], geometry)
        truth_shell = shell_means(truth_psd, geometry)
        recon_shell = shell_means(recon_psd, geometry)
        truth_shells[index] = truth_shell
        recon_shells[index] = recon_shell
        residual = recon_centered - truth_centered
        residual_fft = np.fft.fftshift(np.fft.fft2(residual))
        residual_psd = (np.abs(residual_fft) ** 2) / float(NX * NY)
        truth_high = float(np.sum(truth_psd[geometry["high_mask"]]))
        error_high = float(np.sum(residual_psd[geometry["high_mask"]]))
        high_l2[index] = np.sqrt(error_high / max(truth_high, 1e-30))
        truth_band = band_energy_breakdown(geometry["canonical_wavenumber"], truth_shell)
        recon_band = band_energy_breakdown(geometry["canonical_wavenumber"], recon_shell)
        epsilon = max(1e-30, 1e-12 * float(np.max(truth_shell)))
        canonical_ratio[index] = float(recon_band["energy"][2] / max(float(truth_band["energy"][2]), epsilon))
        canonical_lsd[index] = float(np.sqrt(np.mean((10.0 * np.log10((recon_shell + epsilon) / (truth_shell + epsilon))) ** 2)))
    return truth_shells, recon_shells, {
        "highband_error_relative_l2": high_l2,
        "canonical_shellmean_high_energy_ratio": canonical_ratio,
        "canonical_spectral_lsd_db": canonical_lsd,
    }


def physical_errors(truth: np.ndarray, recon: np.ndarray) -> dict[str, float]:
    per_field = np.linalg.norm(recon - truth, axis=0) / np.maximum(np.linalg.norm(truth, axis=0), 1e-30)
    result = {field: float(per_field[index]) for index, field in enumerate(FIELDS)}
    result["Unobserved_mean"] = float(np.mean([per_field[FIELDS.index(field)] for field in UNOBSERVED]))
    result["All_fields_mean"] = float(np.mean(per_field))
    return result


def metadata_identity(meta: dict[str, Any], *, method: str, snapshot: int, expected_sensor_seed: int | None, errors: list[str], path: Path) -> None:
    checks = {
        "method": method, "condition": "Cond_T", "num_x": NX, "num_y": NY,
        "sensor_plan_hash": SENSOR_HASH, "snapshot": snapshot,
    }
    for column, expected in checks.items():
        if str(meta.get(column)) != str(expected):
            errors.append(f"{path}: metadata {column}={meta.get(column)!r}, expected {expected!r}")
    if expected_sensor_seed is not None and str(meta.get("sensor_seed")) != str(expected_sensor_seed):
        errors.append(f"{path}: metadata sensor_seed={meta.get('sensor_seed')!r}, expected per-state seed {expected_sensor_seed!r}")
    if meta.get("checkpoint_name") != "last.pt":
        errors.append(f"{path}: checkpoint is not last.pt ({meta.get('checkpoint_name')!r})")


def validate_arrays(arrays: dict[str, Any], path: Path, errors: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {"truth_phys", "recon_phys", "coords_phys", "obs_indices", "obs_field_ids", "obs_values_norm"}
    missing = required - set(arrays)
    if missing:
        errors.append(f"{path}: cache lacks arrays {sorted(missing)}")
        return np.empty((0, 5)), np.empty((0, 5)), np.empty((0, 3))
    truth = np.asarray(arrays["truth_phys"], dtype=float)
    recon = np.asarray(arrays["recon_phys"], dtype=float)
    coords = np.asarray(arrays["coords_phys"], dtype=float)
    if truth.shape != (NX * NY, len(FIELDS)) or recon.shape != truth.shape or coords.shape[0] != NX * NY:
        errors.append(f"{path}: unexpected cache shapes truth={truth.shape}, recon={recon.shape}, coords={coords.shape}")
    for name in required:
        if not np.isfinite(np.asarray(arrays[name], dtype=float)).all():
            errors.append(f"{path}: non-finite values in {name}")
    return truth, recon, coords


def collect_reconstruction_states(state_index: pd.DataFrame, manifests: dict[str, pd.DataFrame], errors: list[str]) -> pd.DataFrame:
    targets = (*FIELDS, "Unobserved_mean", "All_fields_mean")
    source = pd.read_csv(RECON_PER_STATE, usecols=["method", "snapshot", "time_index", "checkpoint", "cache_path", "metric", "target", "value"])
    source = source[(source.metric == "physical_relative_l2") & source.method.isin(ABLATIONS) & source.target.isin(targets)].copy()
    expected = EXPECTED_STATES * len(ABLATIONS) * len(targets)
    if len(source) != expected:
        errors.append(f"physical reconstruction source coverage is {len(source)} rows; expected {expected}")
    source["snapshot"] = source.snapshot.astype(int)
    source["time_index"] = source.time_index.astype(int)
    if source.duplicated(["method", "snapshot", "target"]).any():
        errors.append("physical reconstruction source has duplicate method/snapshot/target rows")
    rows: list[dict[str, Any]] = []
    manifest_map = {method: frame.set_index("snapshot") for method, frame in manifests.items()}
    for item in source.itertuples(index=False):
        manifest = manifest_map.get(str(item.method))
        checkpoint_path = str(manifest.loc[int(item.snapshot), "checkpoint_path"]) if manifest is not None and int(item.snapshot) in manifest.index else ""
        checkpoint_sha = str(manifest.loc[int(item.snapshot), "checkpoint_sha256"]) if manifest is not None and int(item.snapshot) in manifest.index else ""
        rows.append({
            "method": str(item.method), "field": str(item.target), "snapshot": int(item.snapshot),
            "time_index": int(item.time_index), "value": float(item.value), "policy": "last",
            "checkpoint_name": str(item.checkpoint), "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha, "source_path": str(RECON_PER_STATE),
            "source_kind": "accepted_ablation_evaluation", "metric": "physical_relative_l2",
        })
    frame = pd.DataFrame(rows)
    if not frame.empty and not np.isfinite(frame.value.to_numpy(float)).all():
        errors.append("physical reconstruction source has non-finite values")
    return frame


def collect_accepted_highband(errors: list[str]) -> pd.DataFrame:
    usecols = ["policy", "method", "snapshot", "time_index", "field", "metric", "value"]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(HF_PER_STATE, usecols=usecols, chunksize=200_000):
        piece = chunk[
            (chunk.policy.astype(str) == "last") & chunk.method.isin(ABLATIONS)
            & chunk.field.isin(FIELDS)
            & chunk.metric.isin([
                "highband_error_relative_l2", "canonical_shellmean_high_energy_ratio",
                "canonical_spectral_lsd_db",
            ])
        ]
        if not piece.empty:
            pieces.append(piece)
    source = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=usecols)
    expected = EXPECTED_STATES * len(ABLATIONS) * len(FIELDS) * 3
    if len(source) != expected:
        errors.append(f"high-band source coverage is {len(source)} rows; expected {expected}")
    pivot = source.pivot_table(index=["method", "snapshot", "time_index", "field"], columns="metric", values="value", aggfunc="first").reset_index()
    required = {"highband_error_relative_l2", "canonical_shellmean_high_energy_ratio", "canonical_spectral_lsd_db"}
    if not required.issubset(pivot.columns):
        errors.append(f"high-band source lacks metrics {sorted(required - set(pivot.columns))}")
    if len(pivot) != EXPECTED_STATES * len(ABLATIONS) * len(FIELDS):
        errors.append(f"high-band source pivot coverage is {len(pivot)} rows")
    return pivot


def summary_rows(frame: pd.DataFrame, value_column: str = "value", group_columns: tuple[str, ...] = ("method", "field"), *, bootstrap: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(group_columns), sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row.update(describe(group[value_column].to_numpy(float), bootstrap=bootstrap))
        rows.append(row)
    return pd.DataFrame(rows)


def apply_accepted_summary(
    computed: pd.DataFrame,
    accepted: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    source_label: str,
) -> pd.DataFrame:
    """Use the frozen evaluator summary rows wherever that metric exists.

    The new Senseiver rows have no accepted summary and therefore retain the
    fresh block-20 bootstrap calculated by this collector.  This distinction
    is kept in ``ci_method`` so a reviewer can see which intervals were reused.
    """

    if computed.empty or accepted.empty:
        return computed
    result = computed.copy()
    statistic_columns = (
        "n", "mean", "std", "median", "q25", "q75", "p95", "min", "max",
        "iid_ci95_low", "iid_ci95_high", "block20_ci95_low", "block20_ci95_high",
    )
    for index, row in result.iterrows():
        mask = np.ones(len(accepted), dtype=bool)
        for column in key_columns:
            mask &= accepted[column].astype(str).to_numpy() == str(row[column])
        matches = accepted.loc[mask]
        if len(matches) != 1:
            continue
        match = matches.iloc[0]
        for column in statistic_columns:
            if column in match and pd.notna(match[column]):
                result.at[index, column] = float(match[column]) if column != "n" else int(match[column])
        result.at[index, "ci_method"] = source_label
    return result


def append_provenance(frame: pd.DataFrame, *, manifests: dict[str, pd.DataFrame], source_kind: str, metric: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["policy"] = frame.get("policy", "last")
    frame["metric"] = frame.get("metric", metric)
    frame["source_kind"] = frame.get("source_kind", source_kind)
    checkpoints: dict[str, tuple[str, str, str]] = {}
    for method, manifest in manifests.items():
        if len(manifest):
            row = manifest.iloc[0]
            checkpoints[method] = (str(row.checkpoint_name), str(row.checkpoint_path), str(row.checkpoint_sha256))
    checkpoints["Senseiver"] = ("last.pt", "", "")
    frame["checkpoint_name"] = frame.apply(lambda row: checkpoints.get(str(row.method), ("", "", ""))[0], axis=1)
    frame["checkpoint_path"] = frame.apply(lambda row: checkpoints.get(str(row.method), ("", "", ""))[1], axis=1)
    frame["checkpoint_sha256"] = frame.apply(lambda row: checkpoints.get(str(row.method), ("", "", ""))[2], axis=1)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--strict-formal", action="store_true", help="return nonzero when the source gate is blocked")
    args = parser.parse_args()
    out = ROOT / "results/derived" / args.timestamp
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "source_manifest.json"
    if marker.is_file():
        try:
            old = json.loads(marker.read_text(encoding="utf-8"))
            if old.get("release") not in (None, "figure5-v7r2-source") or old.get("timestamp") not in (None, args.timestamp):
                raise RuntimeError(f"refusing to overwrite prior release source directory: {out}")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"refusing to overwrite unreadable source manifest: {marker}") from exc

    errors: list[str] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []
    source_files = [
        REPORT, RECON_PER_STATE, RECON_SUMMARY, HF_PER_STATE, HF_SUMMARY,
        SPECTRAL_U1, SPECTRAL_SNAPSHOT, SENSEIVER_MANIFEST,
        SPECTRAL_ANALYSIS, SCRIPTS / "common/spectral.py",
        INHERITED_V7_SOURCE_MANIFEST, INHERITED_V7_BUILD_MANIFEST, INHERITED_V6_BUILD_MANIFEST,
        *(path for paths in INHERITED_PANEL_FILES.values() for path in paths),
    ]
    for path in source_files:
        require_file(path, path.name, errors)
    if REPORT.is_file() and sha256(REPORT) != REPORT_SHA256:
        errors.append(f"evaluation report hash changed: expected {REPORT_SHA256}, got {sha256(REPORT)}")
    state_index = read_state_index(errors)
    manifests = read_method_manifests(errors, state_index)
    senseiver_manifest, senseiver_checkpoint = read_senseiver_manifest(errors)
    if not state_index.empty and not senseiver_manifest.empty:
        ordered = senseiver_manifest.sort_values("snapshot").reset_index(drop=True)
        reference = state_index.sort_values("snapshot").reset_index(drop=True)
        for column in ("sensor_plan_hash", "sensor_seed"):
            if not np.array_equal(ordered[column].astype(str).to_numpy(), reference[column].astype(str).to_numpy()):
                errors.append(f"Senseiver per-state {column} sequence differs from A0 state identity")
    senseiver_checkpoint_sha = (
        sha256(senseiver_checkpoint)
        if senseiver_checkpoint is not None and senseiver_checkpoint.is_file()
        else ""
    )
    senseiver_epoch_runtime: str | None = None
    observed_senseiver_epoch: int | None = None
    cache_maps = {method: cache_paths_from_manifest(frame, method, errors) for method, frame in manifests.items()}
    senseiver_paths = cache_paths_from_manifest(senseiver_manifest, "Senseiver", errors) if not senseiver_manifest.empty else {}
    if senseiver_checkpoint is not None:
        require_file(senseiver_checkpoint, "Senseiver last.pt checkpoint", errors)
        if senseiver_checkpoint_sha and senseiver_checkpoint_sha != SENSEIVER_CHECKPOINT_SHA256:
            errors.append(
                f"Senseiver last.pt hash mismatch: expected {SENSEIVER_CHECKPOINT_SHA256}, "
                f"got {senseiver_checkpoint_sha}"
            )
        if senseiver_checkpoint_sha:
            observed_senseiver_epoch, senseiver_epoch_runtime = checkpoint_epoch(senseiver_checkpoint)
            if observed_senseiver_epoch is None:
                errors.append("Senseiver last.pt checkpoint does not expose a recorded epoch")
            elif observed_senseiver_epoch != SENSEIVER_EPOCH:
                errors.append(f"Senseiver last.pt epoch mismatch: expected {SENSEIVER_EPOCH}, got {observed_senseiver_epoch}")
            else:
                checks.append({
                    "name": "Senseiver_checkpoint_epoch", "status": "pass",
                    "epoch": observed_senseiver_epoch, "sha256": senseiver_checkpoint_sha,
                    "metadata_runtime": senseiver_epoch_runtime,
                })
    a0_manifest = manifests.get("A0")
    if a0_manifest is not None and len(a0_manifest):
        a0 = a0_manifest.iloc[0]
        if int(a0.checkpoint_epoch) != A0_LAST_EPOCH or str(a0.checkpoint_sha256) != A0_LAST_CHECKPOINT_SHA256:
            errors.append(
                f"A0 last.pt anchor mismatch: expected epoch/hash {A0_LAST_EPOCH}/{A0_LAST_CHECKPOINT_SHA256}, "
                f"got {a0.checkpoint_epoch}/{a0.checkpoint_sha256}"
            )
        else:
            checks.append({"name": "A0_checkpoint_anchor", "status": "pass", "epoch": A0_LAST_EPOCH, "sha256": A0_LAST_CHECKPOINT_SHA256})

    all_cache_maps = {**cache_maps, "Senseiver": senseiver_paths}
    if not state_index.empty and not errors:
        try:
            geometry = make_geometry(all_cache_maps["A0"][0])
        except Exception as exc:
            errors.append(f"canonical spectral geometry failed: {exc}")
            geometry = None
    else:
        geometry = None

    reconstruction = pd.DataFrame()
    highband = pd.DataFrame()
    accepted_recon_summary = pd.DataFrame()
    accepted_high_summary = pd.DataFrame()
    spectra_values: dict[str, np.ndarray] = {}
    truth_values: np.ndarray | None = None
    senseiver_physical: list[dict[str, Any]] = []
    senseiver_highband: list[dict[str, Any]] = []
    derived_highband_checks: list[dict[str, Any]] = []
    identity_counts = {"truth_phys": 0, "coords_phys": 0, "obs_indices": 0, "obs_field_ids": 0, "obs_values_norm": 0}
    if geometry is not None and not errors:
        n_shells = len(geometry["valid_shells"])
        for method in ALL_METHODS:
            spectra_values[method] = np.full((EXPECTED_STATES, len(FIELDS), n_shells), np.nan, dtype=float)
        truth_values = np.full((EXPECTED_STATES, len(FIELDS), n_shells), np.nan, dtype=float)
        state_frame = state_index.set_index("snapshot")
        state_time = state_frame.time_index.to_dict()
        state_seeds = state_frame.sensor_seed.to_dict()
        for snapshot in range(EXPECTED_STATES):
            loaded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], Path, dict[str, Any]]] = {}
            for method in ALL_METHODS:
                path = all_cache_maps.get(method, {}).get(snapshot)
                if path is None or not path.is_file():
                    continue
                try:
                    arrays, meta = load_cache(path)
                    truth, recon, coords = validate_arrays(arrays, path, errors)
                    metadata_identity(meta, method=method, snapshot=snapshot, expected_sensor_seed=int(state_seeds[snapshot]), errors=errors, path=path)
                    loaded[method] = (truth, recon, coords, meta, path, arrays)
                except Exception as exc:
                    errors.append(f"failed loading {method} snapshot {snapshot}: {exc}")
            base = loaded.get("A0")
            if base is None:
                continue
            base_truth, _, base_coords, _, _, base_arrays = base
            for method, item in loaded.items():
                truth, recon, coords, meta, path, arrays = item
                if method != "A0":
                    if not np.array_equal(truth, base_truth):
                        errors.append(f"{method} snapshot {snapshot}: truth_phys differs from A0 cohort payload")
                    else:
                        identity_counts["truth_phys"] += 1
                    if not np.array_equal(coords, base_coords):
                        errors.append(f"{method} snapshot {snapshot}: coords_phys differs from A0 cohort payload")
                    else:
                        identity_counts["coords_phys"] += 1
                    base_obs = tuple(np.asarray(base_arrays.get(name, [])) for name in ("obs_indices", "obs_field_ids", "obs_values_norm"))
                    for name, value, expected in zip(("obs_indices", "obs_field_ids", "obs_values_norm"),
                                                      (arrays.get("obs_indices"), arrays.get("obs_field_ids"), arrays.get("obs_values_norm")), base_obs):
                        if value is None or not np.array_equal(np.asarray(value), expected):
                            errors.append(f"{method} snapshot {snapshot}: {name} differs from A0 cohort payload")
                        else:
                            identity_counts[name] += 1
                truth_shell, recon_shell, hf = cache_spectral_metrics(truth, recon, geometry)
                spectra_values[method][snapshot] = recon_shell
                derived_highband_checks.extend({
                    "method": method, "field": field, "snapshot": snapshot,
                    "highband_error_relative_l2": float(hf["highband_error_relative_l2"][index]),
                    "canonical_shellmean_high_energy_ratio": float(hf["canonical_shellmean_high_energy_ratio"][index]),
                    "canonical_spectral_lsd_db": float(hf["canonical_spectral_lsd_db"][index]),
                } for index, field in enumerate(FIELDS))
                if method == "A0":
                    truth_values[snapshot] = truth_shell
                if method == "Senseiver":
                    physical = physical_errors(truth, recon)
                    for field, value in physical.items():
                        senseiver_physical.append({
                            "method": method, "field": field, "snapshot": snapshot,
                            "time_index": int(state_time.get(snapshot, snapshot)), "value": value,
                            "policy": "last", "checkpoint_name": "last.pt", "checkpoint_path": str(meta.get("checkpoint_path", "")),
                            "checkpoint_sha256": SENSEIVER_CHECKPOINT_SHA256, "source_path": str(path), "source_kind": "derived_from_saved_cache",
                            "metric": "physical_relative_l2",
                        })
                    for index, field in enumerate(FIELDS):
                        senseiver_highband.append({
                            "method": method, "field": field, "snapshot": snapshot,
                            "time_index": int(state_time.get(snapshot, snapshot)), "value": float(hf["highband_error_relative_l2"][index]),
                            "canonical_power_ratio": float(hf["canonical_shellmean_high_energy_ratio"][index]),
                            "canonical_lsd_db": float(hf["canonical_spectral_lsd_db"][index]),
                            "policy": "last", "checkpoint_name": "last.pt", "checkpoint_path": str(meta.get("checkpoint_path", "")),
                            "checkpoint_sha256": SENSEIVER_CHECKPOINT_SHA256, "source_path": str(path), "source_kind": "derived_from_saved_cache",
                            "metric": "highband_error_relative_l2",
                        })
        if truth_values is not None and not np.isfinite(truth_values).all():
            errors.append("absolute truth spectrum population has incomplete/non-finite state coverage")
        for method, values in spectra_values.items():
            if not np.isfinite(values).all():
                errors.append(f"absolute reconstruction spectrum population for {method} has incomplete/non-finite state coverage")

    if not errors:
        reconstruction = collect_reconstruction_states(state_index, manifests, errors)
        if senseiver_physical:
            reconstruction = pd.concat([reconstruction, pd.DataFrame(senseiver_physical)], ignore_index=True)
        highband_accepted = collect_accepted_highband(errors)
        if not highband_accepted.empty:
            highband_accepted = highband_accepted.rename(columns={
                "highband_error_relative_l2": "value", "canonical_shellmean_high_energy_ratio": "canonical_power_ratio",
                "canonical_spectral_lsd_db": "canonical_lsd_db",
            })
            highband_accepted["policy"] = "last"
            highband_accepted["source_path"] = str(HF_PER_STATE)
            highband_accepted["source_kind"] = "accepted_ablation_evaluation"
            highband_accepted["metric"] = "highband_error_relative_l2"
            highband = highband_accepted
        if senseiver_highband:
            highband = pd.concat([highband, pd.DataFrame(senseiver_highband)], ignore_index=True)
        # The source evaluator already published confidence intervals for the
        # six ablation methods.  Keep them as the authoritative intervals and
        # use fresh bootstrap intervals only for the newly derived Senseiver.
        accepted_recon_summary = pd.read_csv(RECON_SUMMARY)
        accepted_recon_summary = accepted_recon_summary[
            (accepted_recon_summary.metric == "physical_relative_l2")
            & accepted_recon_summary.method.isin(ABLATIONS)
            & accepted_recon_summary.target.isin((*FIELDS, "Unobserved_mean", "All_fields_mean"))
        ].copy()
        if len(accepted_recon_summary) != len(ABLATIONS) * 7:
            errors.append(f"accepted physical summary coverage is {len(accepted_recon_summary)} rows; expected {len(ABLATIONS) * 7}")
        accepted_high_summary = pd.read_csv(HF_SUMMARY)
        accepted_high_summary = accepted_high_summary[
            (accepted_high_summary.policy.astype(str) == "last")
            & accepted_high_summary.method.isin(ABLATIONS)
            & accepted_high_summary.field.isin(FIELDS)
            & accepted_high_summary.metric.isin([
                "highband_error_relative_l2", "canonical_shellmean_high_energy_ratio", "canonical_spectral_lsd_db",
            ])
        ].copy()
        if len(accepted_high_summary) != len(ABLATIONS) * len(FIELDS) * 3:
            errors.append(f"accepted high-band summary coverage is {len(accepted_high_summary)} rows; expected {len(ABLATIONS) * len(FIELDS) * 3}")
        derived_map = {
            (item["method"], int(item["snapshot"]), item["field"]): item for item in derived_highband_checks
        }
        accepted_map = {
            (str(item.method), int(item.snapshot), str(item.field)): item
            for item in highband_accepted.itertuples(index=False)
        }
        parity = []
        for key, derived in derived_map.items():
            accepted = accepted_map.get(key)
            if accepted is None:
                continue
            parity.append({
                "highband_error_relative_l2": abs(derived["highband_error_relative_l2"] - float(accepted.value)),
                "canonical_shellmean_high_energy_ratio": abs(derived["canonical_shellmean_high_energy_ratio"] - float(accepted.canonical_power_ratio)),
                "canonical_spectral_lsd_db": abs(derived["canonical_spectral_lsd_db"] - float(accepted.canonical_lsd_db)),
            })
        if len(parity) != EXPECTED_STATES * len(ABLATIONS) * len(FIELDS):
            errors.append(f"cache recomputation parity coverage is {len(parity)} rows")
        elif parity:
            max_parity = {metric: max(item[metric] for item in parity) for metric in parity[0]}
            if any(value > 2e-5 for value in max_parity.values()):
                errors.append(f"cache recomputation differs from accepted high-band values: {max_parity}")
            checks.append({"name": "ablation_highband_recomputation_parity", "status": "pass", "max_abs_difference": max_parity})
        if not reconstruction.empty:
            senseiver_values = reconstruction[(reconstruction.method == "Senseiver") & (reconstruction.field == "Unobserved_mean")].value.to_numpy(float)
            if len(senseiver_values) != EXPECTED_STATES:
                errors.append(f"Senseiver unobserved physical coverage is {len(senseiver_values)} rows")
            elif abs(float(np.mean(senseiver_values)) - SENSEIVER_UNOBSERVED_ANCHOR) > 5e-6:
                errors.append(f"Senseiver unobserved physical anchor mismatch: {float(np.mean(senseiver_values))}")
            else:
                checks.append({"name": "Senseiver_unobserved_physical_anchor", "status": "pass", "mean": float(np.mean(senseiver_values)), "expected": SENSEIVER_UNOBSERVED_ANCHOR})

    # Output tables are written only after all strict source checks have passed.
    status = "pass" if not errors else "blocked"
    manifest: dict[str, Any] = {
        "release": "figure5-v7r2-source", "schema_version": "figure5-v7r2-source-1", "timestamp": args.timestamp,
        "status": status, "evaluation_report_sha256": REPORT_SHA256, "checkpoint_policy": "last_only",
        "cohort": {"condition": "Cond_T", "split": "test", "snapshots": EXPECTED_STATES, "points_per_state": NX * NY,
                   "fields": list(FIELDS), "sensor_plan_sha256": SENSOR_HASH,
                   "sensor_seed_policy": "per-state seeds retained from A0 manifest and matched across all caches",
                   "truth_payload_identity": "exact array equality against A0 for every method and Senseiver cache"},
        "methods": list(ALL_METHODS), "main_methods": list(MAIN_METHODS), "si_control": ["A1"],
        "spectral_estimator": {"implementation": "common.spectral", "version": "condT-high-frequency-v2",
                               "coordinate_mode": geometry.get("coordinate_mode_used") if geometry else None,
                               "retained_shells": int(len(geometry["valid_shells"])) if geometry else None,
                               "high_shells": int(len(geometry["high_shells"])) if geometry else None,
                               "high_k_min": geometry.get("high_k_min") if geometry else None,
                               "high_k_max": geometry.get("high_k_max") if geometry else None,
                               "normalization": "|FFT(centered)|^2/(nx*ny); native shell means; no window"},
        "bootstrap": {"state_block_length": BLOCK_LENGTH, "state_resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED,
                       "spectral_shell_ci": "not estimated; spectra_population reports median and IQR"},
        "summary_sources": {
            "ablation_reconstruction": {"path": str(RECON_SUMMARY), "intervals": "reused accepted source intervals", "bootstrap_seed": 20260906},
            "ablation_highband": {"path": str(HF_SUMMARY), "intervals": "reused accepted source intervals", "bootstrap_seed": 20260910},
            "Senseiver_reconstruction_and_highband": {"intervals": "new circular moving-block bootstrap", "bootstrap_seed": BOOTSTRAP_SEED},
        },
        "identity_proof": {"comparison_reference": "A0 last cache payload at each snapshot",
                            "compared_nonreference_method_states": EXPECTED_STATES * (len(ALL_METHODS) - 1),
                            "array_exact_match_counts": identity_counts,
                            "array_exact_mismatch_counts": {key: EXPECTED_STATES * (len(ALL_METHODS) - 1) - value for key, value in identity_counts.items()}},
        "method_provenance": {},
        "checkpoint_metadata": {
            "Senseiver": {
                "observed_epoch": observed_senseiver_epoch,
                "expected_epoch": SENSEIVER_EPOCH,
                "runtime": senseiver_epoch_runtime,
                "source": "checkpoint payload key epoch",
            }
        },
        "inherited_panel_sources": {},
        "sources": [], "checks": checks, "warnings": warnings, "errors": errors,
    }
    for panel, paths in INHERITED_PANEL_FILES.items():
        manifest["inherited_panel_sources"][panel] = [
            {"path": str(path), "sha256": sha256(path) if path.is_file() else "", "role": path.name,
             "evidence_package": "PackageB"}
            for path in paths
        ]
    manifest["inherited_lineage"] = [
        {"path": str(path), "sha256": sha256(path) if path.is_file() else "", "role": "accepted_style_or_geometry_lineage"}
        for path in (INHERITED_V7_SOURCE_MANIFEST, INHERITED_V7_BUILD_MANIFEST, INHERITED_V6_BUILD_MANIFEST)
    ]
    for method, frame in manifests.items():
        if len(frame):
            row = frame.iloc[0]
            manifest["method_provenance"][method] = {
                "manifest": str(EVALUATION / f"last/manifest_{method}.csv"),
                "manifest_sha256": sha256(EVALUATION / f"last/manifest_{method}.csv"),
                "code_sha256": str(row.code_sha256), "config_sha256": str(row.config_sha256),
                "stats_sha256": str(row.stats_sha256), "sensor_plan": str(row.sensor_plan),
                "generation_seed_policy": str(row.generation_seed_policy), "checkpoint_policy": "last_only",
                "cache_coverage": int(len(frame)),
            }
    manifest["method_provenance"]["Senseiver"] = {
        "manifest": str(SENSEIVER_MANIFEST), "manifest_sha256": sha256(SENSEIVER_MANIFEST) if SENSEIVER_MANIFEST.is_file() else "",
        "checkpoint_sha256": senseiver_checkpoint_sha, "checkpoint_epoch": SENSEIVER_EPOCH,
        "checkpoint_epoch_observed": observed_senseiver_epoch,
        "checkpoint_metadata_runtime": senseiver_epoch_runtime,
        "cache_coverage": int(len(senseiver_manifest)), "obs_consistency_applied": "native_not_applied",
        "cache_source": "same-cohort paper_full_20260711 reconstruction cache",
    }
    for path in source_files:
        if path.is_file():
            manifest["sources"].append({"path": str(path), "sha256": sha256(path), "role": path.name})
    for method, frame in manifests.items():
        if len(frame):
            row = frame.iloc[0]
            manifest["sources"].append({"path": str(row.checkpoint_path), "sha256": str(row.checkpoint_sha256), "role": f"{method}_last_checkpoint"})
    if senseiver_checkpoint is not None and senseiver_checkpoint.is_file():
        manifest["sources"].append({"path": str(senseiver_checkpoint), "sha256": sha256(senseiver_checkpoint), "role": "Senseiver_last_checkpoint"})
    for path, role in (
        (SPECTRAL_ANALYSIS, "accepted_spectral_analysis_source"),
        (SCRIPTS / "common/spectral.py", "accepted_common_spectral_estimator"),
    ):
        require_file(path, role, errors)
        if path.is_file():
            manifest["sources"].append({"path": str(path), "sha256": sha256(path), "role": role})
    if SENSEIVER_FIELDL2.is_file():
        manifest["sources"].append({"path": str(SENSEIVER_FIELDL2), "sha256": sha256(SENSEIVER_FIELDL2), "role": "optional_Senseiver_FieldL2_crosscheck"})
    if status == "pass":
        # Add standard summaries and provenance columns.
        reconstruction = reconstruction.sort_values(["method", "field", "snapshot"]).reset_index(drop=True)
        recon_summary = summary_rows(reconstruction, group_columns=("method", "field"), bootstrap=True)
        recon_summary = apply_accepted_summary(
            recon_summary,
            accepted_recon_summary.rename(columns={"target": "field"}),
            key_columns=("method", "field"),
            source_label="accepted last/metrics/summary_metrics.csv; bootstrap seed 20260906",
        )
        recon_summary["policy"] = "last"
        recon_summary["metric"] = "physical_relative_l2"
        recon_summary["source_kind"] = "accepted_ablation_evaluation_or_derived_saved_cache"
        highband = highband.sort_values(["method", "field", "snapshot"]).reset_index(drop=True)
        high_summary = summary_rows(highband, group_columns=("method", "field"), bootstrap=True)
        high_summary = apply_accepted_summary(
            high_summary,
            accepted_high_summary[accepted_high_summary.metric == "highband_error_relative_l2"],
            key_columns=("method", "field"),
            source_label="accepted high_frequency/summary_high_frequency.csv; bootstrap seed 20260910",
        )
        power_summary = summary_rows(highband.rename(columns={"canonical_power_ratio": "value_power"}), value_column="value_power", group_columns=("method", "field"), bootstrap=True)
        power_summary = apply_accepted_summary(
            power_summary,
            accepted_high_summary[accepted_high_summary.metric == "canonical_shellmean_high_energy_ratio"],
            key_columns=("method", "field"),
            source_label="accepted high_frequency/summary_high_frequency.csv; bootstrap seed 20260910",
        )
        power_summary = power_summary.rename(columns={column: f"canonical_power_{column}" for column in ("n", "mean", "std", "median", "q25", "q75", "p95", "min", "max", "iid_ci95_low", "iid_ci95_high", "block20_ci95_low", "block20_ci95_high", "ci_method")})
        lsd_summary = summary_rows(highband.rename(columns={"canonical_lsd_db": "value_lsd"}), value_column="value_lsd", group_columns=("method", "field"), bootstrap=True)
        lsd_summary = apply_accepted_summary(
            lsd_summary,
            accepted_high_summary[accepted_high_summary.metric == "canonical_spectral_lsd_db"],
            key_columns=("method", "field"),
            source_label="accepted high_frequency/summary_high_frequency.csv; bootstrap seed 20260910",
        )
        lsd_summary = lsd_summary.rename(columns={column: f"canonical_lsd_{column}" for column in ("n", "mean", "std", "median", "q25", "q75", "p95", "min", "max", "iid_ci95_low", "iid_ci95_high", "block20_ci95_low", "block20_ci95_high", "ci_method")})
        high_summary["policy"] = "last"
        high_summary["metric"] = "highband_error_relative_l2"
        high_summary["source_kind"] = "accepted_ablation_evaluation_or_derived_saved_cache"
        high_summary = high_summary.merge(power_summary, on=["method", "field"], how="left")
        high_summary = high_summary.merge(lsd_summary, on=["method", "field"], how="left")

        spectra_rows: list[dict[str, Any]] = []
        assert truth_values is not None and geometry is not None
        for method in ("Truth", *ALL_METHODS):
            values_by_state = truth_values if method == "Truth" else spectra_values[method]
            for field_index, field in enumerate(FIELDS):
                for shell_position, shell_index in enumerate(geometry["valid_shells"]):
                    stats = describe(values_by_state[:, field_index, shell_position], bootstrap=False)
                    spectra_rows.append({
                        "method": method, "field": field, "shell_index": int(shell_index),
                        "wavenumber": float(geometry["canonical_wavenumber"][shell_position]),
                        "high_band": bool(shell_index in set(geometry["high_shells"])),
                        "energy_definition": "absolute physical shell-mean spectral energy",
                        **stats,
                        "source_kind": "derived_from_saved_cache" if method != "Truth" else "truth_from_A0_same_cohort_cache",
                    })
        spectra = pd.DataFrame(spectra_rows)
        output_tables = {
            "reconstruction_states": reconstruction,
            "reconstruction_summary": recon_summary,
            "highband_states": highband,
            "highband_summary": high_summary,
            "spectra_population": spectra,
        }
        # A compact diagnostic table makes the exact high-band estimator and
        # source route inspectable without storing another copy of all states.
        diagnostics = pd.DataFrame([{
            "method": method, "field": field,
            "shell_count": int(len(geometry["valid_shells"])), "high_shell_count": int(len(geometry["high_shells"])),
            "retained_mode_count": geometry["retained_mode_count"], "high_mode_count": geometry["high_mode_count"],
            "kmax": geometry["kmax"], "high_k_min": geometry["high_k_min"], "high_k_max": geometry["high_k_max"],
            "coordinate_mode_used": geometry["coordinate_mode_used"], "window": "none",
            "estimator_version": "condT-high-frequency-v2",
        } for method in ALL_METHODS for field in FIELDS])
        # Join the scalar canonical-power and LSD summaries into a diagnostic
        # table keyed by method and field.  Truth is a reference spectrum and
        # consequently has no reconstruction diagnostic values.
        high_summary_index = high_summary.set_index(["method", "field"])
        for index, row in diagnostics.iterrows():
            key = (row["method"], row["field"])
            if key in high_summary_index.index:
                summary = high_summary_index.loc[key]
                diagnostics.at[index, "canonical_power_mean"] = summary.get("canonical_power_mean", np.nan)
                diagnostics.at[index, "canonical_power_block20_ci95_low"] = summary.get("canonical_power_block20_ci95_low", np.nan)
                diagnostics.at[index, "canonical_power_block20_ci95_high"] = summary.get("canonical_power_block20_ci95_high", np.nan)
                diagnostics.at[index, "canonical_lsd_mean"] = summary.get("canonical_lsd_mean", np.nan)
                diagnostics.at[index, "canonical_lsd_block20_ci95_low"] = summary.get("canonical_lsd_block20_ci95_low", np.nan)
                diagnostics.at[index, "canonical_lsd_block20_ci95_high"] = summary.get("canonical_lsd_block20_ci95_high", np.nan)
        output_tables["spectral_diagnostics"] = diagnostics
        for name, frame in output_tables.items():
            if "method" in frame.columns:
                frame.insert(1, "display_label", frame["method"].map(PUBLISHED_LABELS))
        for name, frame in output_tables.items():
            frame.to_csv(out / f"{name}.csv", index=False, float_format="%.17g")

        checkpoint_rows: list[dict[str, Any]] = []
        for method, frame in manifests.items():
            row = frame.iloc[0]
            checkpoint_rows.append({"method": method, "display_label": PUBLISHED_LABELS[method], "policy": "last", "epoch": int(row.checkpoint_epoch), "path": str(row.checkpoint_path), "sha256": str(row.checkpoint_sha256), "source": "ablation_evaluation_manifest"})
        if len(senseiver_manifest):
            row = senseiver_manifest.iloc[0]
            checkpoint_rows.append({
                "method": "Senseiver", "display_label": PUBLISHED_LABELS["Senseiver"], "policy": "last",
                "epoch": SENSEIVER_EPOCH, "path": str(row.checkpoint_path), "sha256": senseiver_checkpoint_sha,
                "source": "ReconstructionCache_manifest_paper_full_20260711",
                "epoch_source": "checkpoint payload key epoch",
                "metadata_runtime": senseiver_epoch_runtime,
            })
        pd.DataFrame(checkpoint_rows).to_csv(out / "checkpoint_provenance.csv", index=False)
        manifest["outputs"] = {path.name: {"path": str(path), "sha256": sha256(path)} for path in sorted(out.glob("*.csv"))}
        write_json(marker, manifest)
    else:
        manifest["stop_reason"] = "required source is missing, ambiguous, or failed exact cohort/estimator validation"
        write_json(marker, manifest)
    qa = {
        "release": "figure5-v7r2-source", "schema_version": "figure5-v7r2-source-1", "timestamp": args.timestamp,
        "status": status, "errors": errors, "warnings": warnings,
        "checks": [*checks, {"name": "source_gate", "status": status, "detail": "strict cache identity and schema gate"}],
        "coverage": {"ablation_methods": list(ABLATIONS), "main_methods": list(MAIN_METHODS), "senseiver_states": len(senseiver_paths),
                     "reconstruction_rows": int(len(reconstruction)), "highband_rows": int(len(highband))},
        "failed_or_unresolved_checks": errors,
    }
    write_json(out / "source_qa.json", qa)
    print(json.dumps({"status": status, "output": str(out), "errors": len(errors), "reconstruction_rows": len(reconstruction), "highband_rows": len(highband)}))
    return 1 if args.strict_formal and status != "pass" else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # A source failure must remain reviewable in the release directory.
        # The normal path writes the detailed manifest/QA; this fallback is
        # reserved for argument/path failures before those structures exist.
        traceback.print_exc()
        raise

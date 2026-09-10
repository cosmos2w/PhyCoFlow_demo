#!/usr/bin/env python
"""Quantify high-frequency reconstruction behavior from Cond_T cache files.

This is a CPU-only, cache-only diagnostic.  It reuses the same structured-grid
and radial-shell definitions as ``common.spectral`` but reports Fourier-mode
energy sums for the high band.  Mode sums are kept separate from the existing
shell-mean/trapezoid ``spectral_high_energy_ratio`` metric because the two
quantities have different weights.

The primary diagnostic is U1, but ``--fields`` can request any subset of the
five physical channels.  A complete run expects six methods, both checkpoint
policies, and 1,000 snapshot identities per policy.  A source evaluation root
can be supplied when A1--A5 caches are deliberately reused by the new
evaluation; source paths and cache metadata are recorded in the output.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import pandas as pd


DEMO = Path(__file__).resolve().parents[1]
ARCHIVE = DEMO / "Save_TrainedModel/_TrainedModels"
SCRIPTS = ARCHIVE / "_Scripts"
sys.path.insert(0, str(SCRIPTS))
from common.cache import load_cache
from common.spectral import band_energy_breakdown, radial_spectrum, recover_structured_grid


FIELDS = ("CH4", "CO", "T", "U1", "p")
METHODS = tuple(f"A{i}" for i in range(6))
POLICIES = ("last", "best")
HF_METRICS = (
    "canonical_shellmean_high_energy_ratio",
    "truth_high_energy_fraction_total",
    "truth_high_energy_fraction_retained",
    "reconstruction_high_energy_fraction_total",
    "reconstruction_high_energy_fraction_retained",
    "reconstruction_high_energy_over_truth_total",
    "reconstruction_high_energy_over_truth_retained",
    "reconstruction_to_truth_high_energy_ratio",
    "highband_error_energy_over_truth_high_energy",
    "highband_error_relative_l2",
    "highband_error_energy_over_truth_total_fluctuation_energy",
    "highband_error_energy_over_truth_retained_fluctuation_energy",
    "canonical_spectral_lsd_db",
)
HANN_METRICS = (
    "hann_truth_high_energy_fraction_total",
    "hann_truth_high_energy_fraction_retained",
    "hann_reconstruction_high_energy_fraction_total",
    "hann_reconstruction_high_energy_fraction_retained",
    "hann_reconstruction_high_energy_over_truth_total",
    "hann_reconstruction_high_energy_over_truth_retained",
    "hann_reconstruction_to_truth_high_energy_ratio",
    "hann_highband_error_energy_over_truth_high_energy",
    "hann_highband_error_relative_l2",
    "hann_highband_error_energy_over_truth_total_fluctuation_energy",
    "hann_highband_error_energy_over_truth_retained_fluctuation_energy",
)
METRIC_VERSION = "condT-high-frequency-v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path, default=None):
    if not path.is_file():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def block_bootstrap_means(values, *, block: int, n_boot: int, seed: int):
    """Circular moving-block bootstrap of temporally sorted values."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Bootstrap requires a finite nonempty vector")
    block = min(max(int(block), 1), len(values))
    n_blocks = (len(values) + block - 1) // block
    starts = np.random.default_rng(seed).integers(0, len(values), size=(n_boot, n_blocks))
    indices = ((starts[:, :, None] + np.arange(block)) % len(values)).reshape(n_boot, -1)[:, : len(values)]
    return values[indices].mean(axis=1)


def describe(values, *, n_boot: int, seed: int) -> dict[str, float | int | bool]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Cannot summarize empty or nonfinite high-frequency values")
    result = {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "fraction_gt_1": float(np.mean(values > 1.0)),
        "fraction_gt_2": float(np.mean(values > 2.0)),
    }
    for block, label in ((1, "iid"), (20, "block20")):
        boot = block_bootstrap_means(values, block=block, n_boot=n_boot, seed=seed)
        result[f"{label}_ci95_low"], result[f"{label}_ci95_high"] = map(
            float, np.quantile(boot, [0.025, 0.975])
        )
    return result


def resolve_cache_paths(root: Path, source_root: Path | None, policy: str, method: str, expected: int):
    """Use the new root first, then an explicitly authorized reuse root."""
    def entries(directory: Path):
        valid = []
        if not directory.is_dir():
            return valid
        for path in sorted(directory.glob("RecCache_s*.npz")):
            stem = path.stem
            try:
                snapshot = int(stem.split("_s", 1)[1])
            except (IndexError, ValueError):
                continue
            if 0 <= snapshot < expected:
                valid.append((snapshot, path.resolve()))
        return valid

    local_entries = entries(root / policy / method)
    source_entries = entries(source_root / policy / method) if source_root is not None else []
    # A complete explicit reuse root is preferred over a partially materialized
    # local directory.  This avoids silently analyzing only the prefix of a
    # cache set while an evaluation is still being copied.
    paths = local_entries
    if len(source_entries) == expected and len(local_entries) != expected:
        paths = source_entries
    by_snapshot = {snapshot: path for snapshot, path in paths}
    return [(snapshot, by_snapshot[snapshot]) for snapshot in sorted(by_snapshot)]


def discover_jobs(root: Path, source_root: Path | None, expected: int, allow_partial: bool):
    jobs = []
    source_labels = {}
    expected_snapshots = set(range(expected))
    for policy in POLICIES:
        for method in METHODS:
            entries = resolve_cache_paths(root, source_root, policy, method, expected)
            snapshots = {snapshot for snapshot, _ in entries}
            if not allow_partial and snapshots != expected_snapshots:
                raise RuntimeError(
                    f"{policy}/{method}: expected {expected} caches, found {len(snapshots)}; "
                    "finish evaluation or pass --allow-partial for a smoke diagnostic"
                )
            for snapshot, path in entries:
                jobs.append((policy, method, snapshot, str(path)))
                source_labels[(policy, method)] = str(path.parent.parent.parent)
    if not jobs:
        raise FileNotFoundError(f"No Cond_T caches found under {root}")
    return jobs, source_labels


def field_indices(fields):
    unknown = [field for field in fields if field not in FIELDS]
    if unknown:
        raise ValueError(f"Unknown fields {unknown}; choose from {FIELDS}")
    return [(field, FIELDS.index(field)) for field in fields]


def build_geometry(example_path: Path, num_x: int | None, num_y: int | None):
    arrays, meta = load_cache(example_path)
    truth = np.asarray(arrays["truth_phys"])
    recon = np.asarray(arrays["recon_phys"])
    coords = np.asarray(arrays["coords_phys"])
    if truth.ndim != 2 or truth.shape != recon.shape or truth.shape[1] != len(FIELDS):
        raise ValueError(f"Unexpected cache fields at {example_path}: {truth.shape}, {recon.shape}")
    grid = recover_structured_grid(
        coords,
        num_x=num_x,
        num_y=num_y,
        coordinate_mode="auto",
        spacing_tolerance=0.02,
    )
    ordered_truth = truth[grid["sort_idx"]].T.reshape(-1, grid["ny"], grid["nx"])
    ordered_recon = recon[grid["sort_idx"]].T.reshape(-1, grid["ny"], grid["nx"])
    # This call is the canonical API contract. The custom mode-sum calculation
    # below is checked against its shell means and retained shell indices.
    canonical = radial_spectrum(
        ordered_truth[0],
        dx=grid["used_dx"],
        dy=grid["used_dy"],
        remove_mean=True,
        window="none",
        use_isotropic_cutoff=True,
        min_shell_count=4,
    )
    canonical_recon = radial_spectrum(
        ordered_recon[0],
        dx=grid["used_dx"],
        dy=grid["used_dy"],
        remove_mean=True,
        window="none",
        use_isotropic_cutoff=True,
        min_shell_count=4,
    )
    geometry = fft_geometry(grid, canonical)
    if not np.array_equal(canonical["wavenumber_index"], canonical_recon["wavenumber_index"]):
        raise ValueError("Truth/reconstruction retained FFT shells differ in the example cache")
    canonical_shell_mean, canonical_total = shell_statistics(ordered_truth[0], geometry, window="none")
    if not np.allclose(canonical_shell_mean, canonical["spectral_energy"], rtol=1e-10, atol=1e-12):
        raise AssertionError("Mode-sum geometry does not reproduce common.spectral shell means")
    if not np.isclose(canonical_total, canonical["total_energy"], rtol=1e-10, atol=1e-8):
        raise AssertionError("Mode-sum geometry does not reproduce common.spectral total energy")
    geometry["canonical_example"] = {
        "snapshot": int(meta.get("snapshot", 0)),
        "path": str(example_path),
        "coordinate_mode_used": grid["coordinate_mode_used"],
        "canonical_shell_count": int(len(canonical["wavenumber"])),
    }
    geometry["sort_idx"] = grid["sort_idx"]
    geometry["coordinate_mode_used"] = grid["coordinate_mode_used"]
    return geometry, meta, grid


def fft_geometry(grid: dict, canonical: dict):
    nx, ny = int(grid["nx"]), int(grid["ny"])
    dx, dy = float(grid["used_dx"]), float(grid["used_dy"])
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_mag = np.hypot(kx_grid, ky_grid)
    dkx = float(np.min(np.abs(np.diff(np.unique(kx)))))
    dky = float(np.min(np.abs(np.diff(np.unique(ky)))))
    dk = max(min(dkx, dky), 1e-30)
    shell_id = np.rint(k_mag / dk).astype(np.int64)
    n_shells = int(shell_id.max()) + 1
    shell_count = np.bincount(shell_id.ravel(), minlength=n_shells)
    shell_k_sum = np.bincount(shell_id.ravel(), weights=k_mag.ravel(), minlength=n_shells)
    shell_k = shell_k_sum / np.maximum(shell_count, 1)
    valid_shells = np.asarray(canonical["wavenumber_index"], dtype=np.int64)
    valid_mask = np.isin(shell_id, valid_shells)
    kmax = float(np.max(canonical["wavenumber"]))
    high_shells = valid_shells[np.asarray(canonical["wavenumber"]) > (2.0 * kmax / 3.0)]
    high_mask = np.isin(shell_id, high_shells)
    return {
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "shell_id": shell_id,
        "shell_count": shell_count,
        "shell_k": shell_k,
        "valid_shells": valid_shells,
        "valid_mask": valid_mask,
        "high_shells": high_shells,
        "high_mask": high_mask,
        "kmax": kmax,
        "high_k_min": float(np.min(shell_k[high_shells])) if len(high_shells) else np.nan,
        "high_k_max": float(np.max(shell_k[high_shells])) if len(high_shells) else np.nan,
        "retained_mode_count": int(np.count_nonzero(valid_mask)),
        "high_mode_count": int(np.count_nonzero(high_mask)),
        "canonical_wavenumber": np.asarray(canonical["wavenumber"], dtype=float),
    }


def _prepared_fft(field_grid: np.ndarray, geometry: dict, *, window: str):
    values = np.asarray(field_grid, dtype=np.float64)
    values = values - np.mean(values, dtype=np.float64)
    correction = 1.0
    if window == "hann":
        win = np.outer(np.hanning(geometry["ny"]), np.hanning(geometry["nx"]))
        correction = float(np.mean(win**2))
        values = values * win
    elif window != "none":
        raise ValueError(f"Unsupported window {window}")
    fft = np.fft.fftshift(np.fft.fft2(values))
    psd = (np.abs(fft) ** 2) / (geometry["nx"] * geometry["ny"] * correction)
    return values, psd


def shell_statistics(field_grid: np.ndarray, geometry: dict, *, window: str):
    _, psd = _prepared_fft(field_grid, geometry, window=window)
    shell_id = geometry["shell_id"]
    shell_count = geometry["shell_count"]
    sums = np.bincount(shell_id.ravel(), weights=psd.ravel(), minlength=len(shell_count))
    means = sums / np.maximum(shell_count, 1)
    valid = geometry["valid_shells"]
    return means[valid], float(np.sum(psd[geometry["valid_mask"]]))


def _mode_metrics(truth_grid: np.ndarray, recon_grid: np.ndarray, geometry: dict, *, window: str):
    truth_centered, truth_psd = _prepared_fft(truth_grid, geometry, window=window)
    recon_centered, recon_psd = _prepared_fft(recon_grid, geometry, window=window)
    residual = recon_centered - truth_centered
    residual_fft = np.fft.fftshift(np.fft.fft2(residual))
    correction = 1.0
    if window == "hann":
        correction = float(np.mean(np.outer(np.hanning(geometry["ny"]), np.hanning(geometry["nx"])) ** 2))
    residual_psd = (np.abs(residual_fft) ** 2) / (geometry["nx"] * geometry["ny"] * correction)
    valid = geometry["valid_mask"]
    high = geometry["high_mask"]
    truth_total = float(np.sum(truth_psd[valid]))
    recon_total = float(np.sum(recon_psd[valid]))
    # The full centered FFT budget is the exact spatial fluctuation energy by
    # Parseval.  Retained-shell energy is also reported because the canonical
    # radial evaluator excludes under-populated/corner shells.
    truth_total_full = float(np.sum(truth_psd))
    recon_total_full = float(np.sum(recon_psd))
    truth_high = float(np.sum(truth_psd[high]))
    recon_high = float(np.sum(recon_psd[high]))
    error_high = float(np.sum(residual_psd[high]))
    denominator_total = max(truth_total, 1e-30)
    denominator_total_full = max(truth_total_full, 1e-30)
    denominator_high = max(truth_high, 1e-30)
    return {
        "truth_total": truth_total,
        "recon_total": recon_total,
        "truth_total_full": truth_total_full,
        "recon_total_full": recon_total_full,
        "truth_high": truth_high,
        "recon_high": recon_high,
        "error_high": error_high,
        "truth_high_fraction_total": truth_high / denominator_total_full,
        "truth_high_fraction_retained": truth_high / denominator_total,
        "recon_high_fraction_total": recon_high / max(recon_total_full, 1e-30),
        "recon_high_fraction_retained": recon_high / max(recon_total, 1e-30),
        "recon_high_over_truth_total": recon_high / denominator_total_full,
        "recon_high_over_truth_retained": recon_high / denominator_total,
        "recon_to_truth_high_ratio": recon_high / denominator_high,
        "error_high_over_truth_high": error_high / denominator_high,
        "error_high_relative_l2": float(np.sqrt(error_high / denominator_high)),
        "error_high_over_truth_total": error_high / denominator_total_full,
        "error_high_over_truth_retained": error_high / denominator_total,
    }


def _init_worker(geometry, selected_fields):
    global _WORKER_GEOMETRY, _WORKER_FIELDS
    _WORKER_GEOMETRY = geometry
    _WORKER_FIELDS = selected_fields


def process_job(job):
    policy, method, snapshot, path_text = job
    path = Path(path_text)
    arrays, meta = load_cache(path)
    truth = np.asarray(arrays["truth_phys"])
    recon = np.asarray(arrays["recon_phys"])
    if truth.shape != recon.shape or truth.ndim != 2 or truth.shape[1] != len(FIELDS):
        raise ValueError(f"{path}: expected matching [N,5] physical arrays, got {truth.shape}/{recon.shape}")
    if not np.isfinite(truth).all() or not np.isfinite(recon).all():
        raise ValueError(f"{path}: nonfinite physical cache arrays")
    geometry = _WORKER_GEOMETRY
    if truth.shape[0] != geometry["nx"] * geometry["ny"]:
        raise ValueError(f"{path}: point count differs from the canonical grid")
    cache_snapshot = int(meta.get("snapshot", snapshot))
    if cache_snapshot != int(snapshot):
        raise ValueError(f"{path}: metadata snapshot {cache_snapshot} disagrees with filename snapshot {snapshot}")
    rows = []
    spectra = []
    sorted_truth = truth[meta_sort_idx(geometry, arrays)].T.reshape(-1, geometry["ny"], geometry["nx"])
    sorted_recon = recon[meta_sort_idx(geometry, arrays)].T.reshape(-1, geometry["ny"], geometry["nx"])
    # The sort index is kept in the geometry by main; meta_sort_idx avoids
    # relying on cache point order or any implicit reshape convention.
    for field, index in _WORKER_FIELDS:
        truth_grid = sorted_truth[index]
        recon_grid = sorted_recon[index]
        truth_shell, truth_total_canonical = shell_statistics(truth_grid, geometry, window="none")
        recon_shell, recon_total_canonical = shell_statistics(recon_grid, geometry, window="none")
        canonical_eps = max(1e-30, 1e-12 * float(np.max(truth_shell)))
        log_ratio_db = 10.0 * np.log10((recon_shell + canonical_eps) / (truth_shell + canonical_eps))
        canonical_lsd_db = float(np.sqrt(np.mean(log_ratio_db**2)))
        canonical_bands_truth = band_energy_breakdown(
            geometry["canonical_wavenumber"], truth_shell
        )
        canonical_bands_recon = band_energy_breakdown(
            geometry["canonical_wavenumber"], recon_shell
        )
        # Match the existing evaluator's canonical metric exactly: the ratio
        # of high-band trapezoidal shell energies.  The ratio of normalized
        # band fractions is a different quantity and is intentionally not used.
        canonical_ratio = float(
            canonical_bands_recon["energy"][2]
            / max(float(canonical_bands_truth["energy"][2]), canonical_eps)
        )
        no_window = _mode_metrics(truth_grid, recon_grid, geometry, window="none")
        hann = _mode_metrics(truth_grid, recon_grid, geometry, window="hann")
        values = {
            "canonical_shellmean_high_energy_ratio": canonical_ratio,
            "truth_high_energy_fraction_total": no_window["truth_high_fraction_total"],
            "truth_high_energy_fraction_retained": no_window["truth_high_fraction_retained"],
            "reconstruction_high_energy_fraction_total": no_window["recon_high_fraction_total"],
            "reconstruction_high_energy_fraction_retained": no_window["recon_high_fraction_retained"],
            "reconstruction_high_energy_over_truth_total": no_window["recon_high_over_truth_total"],
            "reconstruction_high_energy_over_truth_retained": no_window["recon_high_over_truth_retained"],
            "reconstruction_to_truth_high_energy_ratio": no_window["recon_to_truth_high_ratio"],
            "highband_error_energy_over_truth_high_energy": no_window["error_high_over_truth_high"],
            "highband_error_relative_l2": no_window["error_high_relative_l2"],
            "highband_error_energy_over_truth_total_fluctuation_energy": no_window["error_high_over_truth_total"],
            "highband_error_energy_over_truth_retained_fluctuation_energy": no_window["error_high_over_truth_retained"],
            "canonical_spectral_lsd_db": canonical_lsd_db,
            "hann_truth_high_energy_fraction_total": hann["truth_high_fraction_total"],
            "hann_truth_high_energy_fraction_retained": hann["truth_high_fraction_retained"],
            "hann_reconstruction_high_energy_fraction_total": hann["recon_high_fraction_total"],
            "hann_reconstruction_high_energy_fraction_retained": hann["recon_high_fraction_retained"],
            "hann_reconstruction_high_energy_over_truth_total": hann["recon_high_over_truth_total"],
            "hann_reconstruction_high_energy_over_truth_retained": hann["recon_high_over_truth_retained"],
            "hann_reconstruction_to_truth_high_energy_ratio": hann["recon_to_truth_high_ratio"],
            "hann_highband_error_energy_over_truth_high_energy": hann["error_high_over_truth_high"],
            "hann_highband_error_relative_l2": hann["error_high_relative_l2"],
            "hann_highband_error_energy_over_truth_total_fluctuation_energy": hann["error_high_over_truth_total"],
            "hann_highband_error_energy_over_truth_retained_fluctuation_energy": hann["error_high_over_truth_retained"],
        }
        identity = {
            "policy": policy,
            "method": method,
            "snapshot": cache_snapshot,
            "time_index": int(meta.get("time_index", snapshot)),
            "checkpoint": meta.get("checkpoint_name", ""),
            "checkpoint_epoch": meta.get("checkpoint_epoch", ""),
            "generation_seed": meta.get("generation_seed", ""),
            "cache_path": str(path),
            "field": field,
            "coordinate_mode_used": geometry["coordinate_mode_used"],
            "nx": geometry["nx"],
            "ny": geometry["ny"],
            "high_shell_count": len(geometry["high_shells"]),
            "high_mode_count": geometry["high_mode_count"],
            "high_k_min": geometry["high_k_min"],
            "high_k_max": geometry["high_k_max"],
            "kmax": geometry["kmax"],
            "truth_total_fluctuation_energy": no_window["truth_total_full"],
            "truth_retained_fluctuation_energy": no_window["truth_total"],
            "truth_full_fluctuation_energy": no_window["truth_total_full"],
            "truth_high_energy": no_window["truth_high"],
            "reconstruction_total_fluctuation_energy": no_window["recon_total_full"],
            "reconstruction_retained_fluctuation_energy": no_window["recon_total"],
            "reconstruction_full_fluctuation_energy": no_window["recon_total_full"],
            "reconstruction_high_energy": no_window["recon_high"],
            "highband_error_energy": no_window["error_high"],
        }
        rows.append({**identity, "metric": values})
        if field == "U1":
            valid_shells = geometry["valid_shells"]
            truth_shell_safe = np.maximum(truth_shell, canonical_eps)
            spectra.extend(
                {
                    "policy": policy,
                    "method": method,
                    "snapshot": cache_snapshot,
                    "time_index": int(meta.get("time_index", snapshot)),
                    "field": field,
                    "shell_index": int(shell_index),
                    "wavenumber": float(k),
                    "kmax": float(geometry["kmax"]),
                    "high_band": bool(float(k) > (2.0 * float(geometry["kmax"]) / 3.0)),
                    "truth_shell_mean_energy": float(t),
                    "reconstruction_shell_mean_energy": float(r),
                    "reconstruction_to_truth_shell_mean_ratio": float((r + canonical_eps) / (t + canonical_eps)),
                    "log10_reconstruction_to_truth_shell_mean_ratio": float(np.log10((r + canonical_eps) / (t + canonical_eps))),
                }
                for shell_index, k, t, r in zip(valid_shells, geometry["canonical_wavenumber"], truth_shell, recon_shell)
            )
    return rows, spectra


def meta_sort_idx(geometry, arrays):
    # The sort index is carried in the process-global geometry. This helper
    # exists to make the cache order dependency explicit at the call site.
    return geometry["sort_idx"]


def flatten_metric_rows(rows):
    flat = []
    for row in rows:
        values = row.pop("metric")
        for metric, value in values.items():
            flat.append({**row, "metric": metric, "value": float(value)})
    return flat


def summarize_metric_rows(frame: pd.DataFrame, *, n_boot: int, seed: int):
    rows = []
    for (policy, method, field, metric), group in frame.groupby(["policy", "method", "field", "metric"], sort=True):
        stats = describe(group.sort_values(["time_index", "snapshot"]).value.to_numpy(), n_boot=n_boot, seed=seed)
        rows.append({"policy": policy, "method": method, "field": field, "metric": metric, **stats})
    return pd.DataFrame(rows)


def paired_summary(frame: pd.DataFrame, *, n_boot: int, seed: int):
    """Write both ablation-vs-A0 and best-vs-last paired uncertainty."""
    key = ["field", "metric", "snapshot", "time_index"]
    output = []
    for policy in POLICIES:
        current = frame[frame.policy == policy]
        for method in METHODS[1:]:
            left = current[current.method == method][key + ["value"]].rename(columns={"value": "method_value"})
            right = current[current.method == "A0"][key + ["value"]].rename(columns={"value": "baseline_value"})
            matched = left.merge(right, on=key, how="inner", validate="one_to_one")
            for (field, metric), group in matched.groupby(["field", "metric"], sort=True):
                group = group.sort_values(["time_index", "snapshot"])
                delta = group.method_value.to_numpy(float) - group.baseline_value.to_numpy(float)
                output.append(paired_row(policy, "ablation_vs_A0", method, "A0", field, metric, delta, group.baseline_value.to_numpy(float), n_boot, seed))
    for method in METHODS:
        left = frame[(frame.policy == "best") & (frame.method == method)][key + ["value"]].rename(columns={"value": "best_value"})
        right = frame[(frame.policy == "last") & (frame.method == method)][key + ["value"]].rename(columns={"value": "last_value"})
        matched = left.merge(right, on=key, how="inner", validate="one_to_one")
        for (field, metric), group in matched.groupby(["field", "metric"], sort=True):
            group = group.sort_values(["time_index", "snapshot"])
            delta = group.best_value.to_numpy(float) - group.last_value.to_numpy(float)
            output.append(paired_row("best_vs_last", "checkpoint_sensitivity", method, "last", field, metric, delta, group.last_value.to_numpy(float), n_boot, seed))
    return pd.DataFrame(output)


def paired_row(policy, comparison, method, baseline, field, metric, delta, baseline_values, n_boot, seed):
    result = {
        "policy": policy,
        "comparison": comparison,
        "method": method,
        "baseline": baseline,
        "field": field,
        "metric": metric,
        "n_paired": int(len(delta)),
        "method_mean_minus_baseline": float(np.mean(delta)),
        "baseline_mean": float(np.mean(baseline_values)),
        "relative_mean_change_percent": float(100 * np.mean(delta) / np.mean(baseline_values)) if np.mean(baseline_values) else np.nan,
        "fraction_method_less_than_baseline": float(np.mean(delta < 0)),
        "fraction_method_equal_baseline": float(np.mean(delta == 0)),
    }
    for block in (5, 20, 50):
        boot = block_bootstrap_means(delta, block=block, n_boot=n_boot, seed=seed)
        low, high = np.quantile(boot, [0.025, 0.975])
        result[f"block{block}_ci95_low"], result[f"block{block}_ci95_high"] = float(low), float(high)
        result[f"block{block}_excludes_zero"] = bool(low > 0 or high < 0)
    return result


def population_spectra_summary(spectra: pd.DataFrame, *, n_boot: int, seed: int):
    if spectra.empty:
        return pd.DataFrame()
    rows = []
    for (policy, method, field, shell_index), group in spectra.groupby(["policy", "method", "field", "shell_index"], sort=True):
        values = group.sort_values(["time_index", "snapshot"])["reconstruction_to_truth_shell_mean_ratio"].to_numpy(float)
        stats = describe(values, n_boot=n_boot, seed=seed)
        rows.append({
            "policy": policy,
            "method": method,
            "field": field,
            "shell_index": int(shell_index),
            "wavenumber": float(group.wavenumber.iloc[0]),
            "kmax": float(group.kmax.iloc[0]) if "kmax" in group.columns else np.nan,
            **stats,
        })
    return pd.DataFrame(rows)


def population_spectra_summary_array(
    values: np.ndarray,
    geometry: dict,
    *,
    n_boot: int,
    seed: int,
):
    """Summarize streamed U1 shell ratios without retaining millions of rows."""
    rows = []
    for policy_index, policy in enumerate(POLICIES):
        for method_index, method in enumerate(METHODS):
            for shell_position, (shell_index, wavenumber) in enumerate(
                zip(geometry["valid_shells"], geometry["canonical_wavenumber"])
            ):
                shell_values = values[policy_index, method_index, :, shell_position]
                shell_values = shell_values[np.isfinite(shell_values)]
                if shell_values.size == 0:
                    continue
                stats = describe(shell_values, n_boot=n_boot, seed=seed)
                rows.append({
                    "policy": policy,
                    "method": method,
                    "field": "U1",
                    "shell_index": int(shell_index),
                    "wavenumber": float(wavenumber),
                    "kmax": float(geometry["kmax"]),
                    "high_band": bool(float(wavenumber) > (2.0 * float(geometry["kmax"]) / 3.0)),
                    **stats,
                })
    return pd.DataFrame(rows)


def hann_robustness_summary(summary: pd.DataFrame):
    """Return the compact U1 no-window/Hann companion table."""
    if summary.empty:
        return pd.DataFrame()
    metric_map = {
        "reconstruction_to_truth_high_energy_ratio": ("none", "reconstruction_to_truth_high_energy_ratio"),
        "highband_error_relative_l2": ("none", "highband_error_relative_l2"),
        "highband_error_energy_over_truth_total_fluctuation_energy": (
            "none", "highband_error_energy_over_truth_total_fluctuation_energy"
        ),
        "hann_reconstruction_to_truth_high_energy_ratio": ("hann", "reconstruction_to_truth_high_energy_ratio"),
        "hann_highband_error_relative_l2": ("hann", "highband_error_relative_l2"),
        "hann_highband_error_energy_over_truth_total_fluctuation_energy": (
            "hann", "highband_error_energy_over_truth_total_fluctuation_energy"
        ),
    }
    table = summary[(summary.field == "U1") & summary.metric.isin(metric_map)].copy()
    if table.empty:
        return pd.DataFrame()
    table[["window", "diagnostic"]] = table.metric.map(metric_map).apply(pd.Series)
    table.drop(columns=["metric"], inplace=True)
    columns = ["policy", "method", "field", "window", "diagnostic"] + [
        column for column in summary.columns if column not in {"policy", "method", "field", "metric"}
    ]
    return table[columns].sort_values(["policy", "method", "diagnostic", "window"])


def validate_per_state(frame: pd.DataFrame, expected: int, allow_partial: bool):
    required = {"policy", "method", "snapshot", "time_index", "field", "metric", "value"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Generated high-frequency rows lack {required - set(frame.columns)}")
    identity = ["policy", "method", "snapshot", "field", "metric"]
    if frame.duplicated(identity).any():
        raise ValueError("Duplicate high-frequency per-state identities")
    if not allow_partial:
        for policy in POLICIES:
            for method in METHODS:
                subset = frame[(frame.policy == policy) & (frame.method == method)]
                if set(subset.snapshot) != set(range(expected)):
                    raise ValueError(f"{policy}/{method}: high-frequency metric coverage is incomplete")
    if not np.isfinite(frame.value.to_numpy(float)).all():
        raise ValueError("High-frequency metrics contain nonfinite values")


def validate_against_existing_metrics(
    per_state: pd.DataFrame,
    root: Path,
    source_root: Path | None,
    *,
    expected: int,
    fields: list[str] | tuple[str, ...],
    allow_partial: bool,
    rtol: float = 2e-5,
    atol: float = 2e-6,
):
    """Assert canonical high-energy and LSD rows reproduce the main evaluator.

    The evaluator CSVs are rounded to six decimal places, so the comparison
    tolerance is deliberately small but above that serialization precision.
    """
    existing_rows = []
    source_files = {}
    for policy in POLICIES:
        candidates = [root / policy / "metrics" / "per_state_metrics.csv"]
        if source_root is not None:
            candidates.append(source_root / policy / "metrics" / "per_state_metrics.csv")
        metric_path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if metric_path is None:
            if allow_partial:
                continue
            raise FileNotFoundError(
                f"Missing main evaluator per-state metrics for {policy}; expected "
                f"{candidates[0]} or an explicit source-evaluation equivalent"
            )
        source_files[policy] = str(metric_path.resolve())
        existing = pd.read_csv(metric_path, usecols=["method", "snapshot", "time_index", "metric", "target", "value"])
        existing = existing[
            existing.method.isin(METHODS)
            & existing.metric.isin(["spectral_high_energy_ratio", "spectral_lsd_db"])
            & existing.target.isin(FIELDS)
        ].copy()
        existing.insert(0, "policy", policy)
        existing.rename(columns={"target": "field", "value": "existing_value"}, inplace=True)
        existing_rows.append(existing[["policy", "method", "snapshot", "time_index", "field", "metric", "existing_value"]])
    if not existing_rows:
        return {"status": "skipped", "reason": "no evaluator metrics found", "files": source_files}

    existing = pd.concat(existing_rows, ignore_index=True)
    generated = per_state[
        per_state.metric.isin(["canonical_shellmean_high_energy_ratio", "canonical_spectral_lsd_db"])
    ].copy()
    generated["metric"] = generated.metric.map({
        "canonical_shellmean_high_energy_ratio": "spectral_high_energy_ratio",
        "canonical_spectral_lsd_db": "spectral_lsd_db",
    })
    generated.rename(columns={"value": "generated_value"}, inplace=True)
    keys = ["policy", "method", "snapshot", "time_index", "field", "metric"]
    generated = generated[keys + ["generated_value"]]
    matched = generated.merge(existing, on=keys, how="inner", validate="one_to_one")
    expected_matches = 2 * len(POLICIES) * len(METHODS) * expected * len(fields)
    if not allow_partial and len(matched) != expected_matches:
        raise AssertionError(
            f"Canonical evaluator validation matched {len(matched)} rows; expected "
            f"{expected_matches}"
        )
    if matched.empty:
        raise AssertionError("No canonical rows matched the main evaluator metrics")
    generated_values = matched.generated_value.to_numpy(float)
    existing_values = matched.existing_value.to_numpy(float)
    abs_error = np.abs(generated_values - existing_values)
    scale = np.maximum(np.abs(existing_values), atol)
    rel_error = abs_error / scale
    if not np.all(np.isfinite(abs_error)) or not np.all(np.isfinite(rel_error)):
        raise AssertionError("Canonical evaluator validation produced nonfinite differences")
    if not np.allclose(generated_values, existing_values, rtol=rtol, atol=atol):
        worst = int(np.argmax(abs_error))
        raise AssertionError(
            "Canonical evaluator mismatch: "
            f"{keys}={matched.iloc[worst][keys].to_dict()}, "
            f"generated={generated_values[worst]:.9g}, existing={existing_values[worst]:.9g}, "
            f"abs_error={abs_error[worst]:.3g}"
        )
    return {
        "status": "passed",
        "files": source_files,
        "n_matched": int(len(matched)),
        "max_absolute_error": float(np.max(abs_error)),
        "max_relative_error": float(np.max(rel_error)),
        "rtol": rtol,
        "atol": atol,
        "metrics": ["spectral_high_energy_ratio", "spectral_lsd_db"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=DEMO / "Save_TrainedModel/ablation_condT/evaluation_20260910")
    parser.add_argument("--source-evaluation-dir", type=Path, help="Optional cache-reuse root for missing policy/method directories")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <evaluation-dir>/high_frequency")
    parser.add_argument("--fields", nargs="+", default=list(FIELDS), choices=FIELDS)
    parser.add_argument("--expected-snapshots", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--num-x", type=int, default=403)
    parser.add_argument("--num-y", type=int, default=100)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    root = args.evaluation_dir.resolve()
    source_root = args.source_evaluation_dir.resolve() if args.source_evaluation_dir else None
    output = (args.output_dir or root / "high_frequency").resolve()
    output.mkdir(parents=True, exist_ok=True)
    jobs, source_labels = discover_jobs(root, source_root, args.expected_snapshots, args.allow_partial)
    first_path = Path(jobs[0][3])
    geometry, example_meta, grid = build_geometry(first_path, args.num_x, args.num_y)
    selected_fields = field_indices(args.fields)
    all_rows = []
    spectrum_path = output / "per_state_U1_spectra.csv"
    spectrum_columns = [
        "policy", "method", "snapshot", "time_index", "field", "shell_index",
        "wavenumber", "kmax", "high_band", "truth_shell_mean_energy",
        "reconstruction_shell_mean_energy", "reconstruction_to_truth_shell_mean_ratio",
        "log10_reconstruction_to_truth_shell_mean_ratio",
    ]
    spectrum_array = None
    if "U1" in args.fields:
        spectrum_array = np.full(
            (len(POLICIES), len(METHODS), args.expected_snapshots, len(geometry["valid_shells"])),
            np.nan,
            dtype=np.float64,
        )
    policy_index = {policy: index for index, policy in enumerate(POLICIES)}
    method_index = {method: index for index, method in enumerate(METHODS)}
    shell_position = {int(shell): index for index, shell in enumerate(geometry["valid_shells"])}

    def consume_result(rows, spectra, writer):
        all_rows.extend(rows)
        for record in spectra:
            writer.writerow({key: record.get(key, "") for key in spectrum_columns})
            if spectrum_array is not None:
                spectrum_array[
                    policy_index[record["policy"]],
                    method_index[record["method"]],
                    int(record["snapshot"]),
                    shell_position[int(record["shell_index"])],
                ] = float(record["reconstruction_to_truth_shell_mean_ratio"])

    with spectrum_path.open("w", newline="", encoding="utf-8") as spectrum_stream:
        spectrum_writer = csv.DictWriter(spectrum_stream, fieldnames=spectrum_columns)
        spectrum_writer.writeheader()
        if args.workers > 1:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(geometry, selected_fields),
            ) as pool:
                iterator = pool.map(process_job, jobs, chunksize=2)
                for index, (rows, spectra) in enumerate(iterator, 1):
                    consume_result(rows, spectra, spectrum_writer)
                    if index % 100 == 0 or index == len(jobs):
                        print(f"[high-frequency] {index}/{len(jobs)}", flush=True)
        else:
            _init_worker(geometry, selected_fields)
            for index, job in enumerate(jobs, 1):
                rows, spectra = process_job(job)
                consume_result(rows, spectra, spectrum_writer)
                if index % 100 == 0 or index == len(jobs):
                    print(f"[high-frequency] {index}/{len(jobs)}", flush=True)
    flat = flatten_metric_rows(all_rows)
    per_state = pd.DataFrame(flat)
    validate_per_state(per_state, args.expected_snapshots, args.allow_partial)
    existing_validation = validate_against_existing_metrics(
        per_state,
        root,
        source_root,
        expected=args.expected_snapshots,
        fields=args.fields,
        allow_partial=args.allow_partial,
    )
    per_state_path = output / "per_state_high_frequency.csv"
    per_state.to_csv(per_state_path, index=False)
    summary = summarize_metric_rows(per_state, n_boot=args.bootstrap_resamples, seed=args.seed)
    summary_path = output / "summary_high_frequency.csv"
    summary.to_csv(summary_path, index=False)
    hann_summary = hann_robustness_summary(summary)
    hann_summary.to_csv(output / "U1_hann_robustness_summary.csv", index=False)
    if spectrum_array is None:
        spectra_summary = pd.DataFrame()
    else:
        if not args.allow_partial and not np.isfinite(spectrum_array).all():
            raise AssertionError("U1 shell spectrum coverage is incomplete")
        spectra_summary = population_spectra_summary_array(
            spectrum_array,
            geometry,
            n_boot=args.bootstrap_resamples,
            seed=args.seed,
        )
    spectra_summary_path = output / "population_U1_spectra.csv"
    spectra_summary.to_csv(spectra_summary_path, index=False)
    paired = paired_summary(per_state, n_boot=args.bootstrap_resamples, seed=args.seed)
    paired_path = output / "paired_high_frequency.csv"
    paired.to_csv(paired_path, index=False)

    # A small numerical validation record is kept separate from per-state rows.
    example_arrays, _ = load_cache(first_path)
    example_truth = np.asarray(example_arrays["truth_phys"])[geometry["sort_idx"], FIELDS.index(args.fields[0])].reshape(geometry["ny"], geometry["nx"])
    example_recon = np.asarray(example_arrays["recon_phys"])[geometry["sort_idx"], FIELDS.index(args.fields[0])].reshape(geometry["ny"], geometry["nx"])
    _, example_truth_psd = _prepared_fft(example_truth, geometry, window="none")
    _, example_recon_psd = _prepared_fft(example_recon, geometry, window="none")
    parseval_truth = float(np.sum(example_truth_psd) / max(np.sum((example_truth - np.mean(example_truth)) ** 2), 1e-30))
    parseval_recon = float(np.sum(example_recon_psd) / max(np.sum((example_recon - np.mean(example_recon)) ** 2), 1e-30))
    metadata = {
        "metric_implementation_version": METRIC_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": str(Path(__file__).resolve()),
        "analysis_script_sha256": sha256(Path(__file__).resolve()),
        "evaluation_dir": str(root),
        "source_evaluation_dir": str(source_root) if source_root else None,
        "policies": list(POLICIES),
        "methods": list(METHODS),
        "fields": list(args.fields),
        "expected_snapshots_per_method": args.expected_snapshots,
        "jobs_processed": len(jobs),
        "snapshots_by_policy_method": {
            f"{policy}/{method}": int(sum(1 for item in jobs if item[0] == policy and item[1] == method))
            for policy in POLICIES for method in METHODS
        },
        "cache_roots": {f"{policy}/{method}": path for (policy, method), path in source_labels.items()},
        "cache_metadata_example": example_meta,
        "grid": {
            "nx": geometry["nx"], "ny": geometry["ny"], "dx": geometry["dx"], "dy": geometry["dy"],
            "coordinate_mode_requested": "auto", "coordinate_mode_used": geometry["coordinate_mode_used"],
            "valid_shell_count": int(len(geometry["valid_shells"])),
            "high_shell_count": int(len(geometry["high_shells"])),
            "retained_mode_count": geometry["retained_mode_count"],
            "high_mode_count": geometry["high_mode_count"],
            "kmax": geometry["kmax"], "high_k_min": geometry["high_k_min"], "high_k_max": geometry["high_k_max"],
            "high_band_rule": "canonical retained shells with k > 2/3 * max(retained wavenumber)",
            "isotropic_cutoff": True,
            "min_shell_count": 4,
            "window": "none for primary metrics; Hann companion uses canonical mean-square correction",
            "canonical_spectral_api": "common.spectral.recover_structured_grid/radial_spectrum",
            "canonical_shell_weighting": "shell-mean energy and band_energy_breakdown are retained separately from mode sums",
        },
        "normalization": {
            "energy_definition": "sum(|FFT|^2 / (nx*ny)) over FFT modes; retained metrics sum canonical valid modes and total metrics sum the full centered FFT",
            "full_fluctuation_budget": "sum of all centered FFT modes; equal to spatial sum of squared fluctuations by Parseval",
            "retained_fluctuation_budget": "sum of canonical retained individual modes; excludes shell 0, under-populated shells, and isotropic-cutoff corners",
            "truth_high_energy_fraction_total": "truth high-band mode energy / truth full centered fluctuation energy",
            "truth_high_energy_fraction_retained": "truth high-band mode energy / truth retained centered fluctuation energy",
            "reconstruction_high_energy_fraction_total": "reconstruction high-band mode energy / reconstruction full centered fluctuation energy",
            "reconstruction_high_energy_fraction_retained": "reconstruction high-band mode energy / reconstruction retained centered fluctuation energy",
            "reconstruction_high_energy_over_truth_total": "reconstruction high-band mode energy / truth full centered fluctuation energy",
            "reconstruction_high_energy_over_truth_retained": "reconstruction high-band mode energy / truth retained centered fluctuation energy",
            "highband_error_relative_l2": "sqrt(high-band residual mode energy / truth high-band mode energy)",
            "highband_error_energy_over_truth_total_fluctuation_energy": "high-band residual mode energy / truth full centered fluctuation energy",
            "highband_error_energy_over_truth_retained_fluctuation_energy": "high-band residual mode energy / truth retained centered fluctuation energy",
            "phase_sensitive": "residual FFT is computed after per-field spatial-mean removal; no smoothing or clipping",
            "hann_companion": "same metrics after a separable Hann window, corrected by mean(window^2); reported for boundary-leakage sensitivity only",
        },
        "statistics": {
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
            "summary": "mean, std, median, q25, q75, p95, p99, max, fractions >1 and >2, IID/block20 CIs",
            "paired": "A1-A5 vs A0 within each policy and best-minus-last within method; circular blocks 5/20/50",
            "unit": "one held-out snapshot; no spatial pixels treated as independent replicates",
        },
        "validation": {
            "canonical_mode_sum_shell_mean_match": True,
            "canonical_mode_sum_total_energy_match": True,
            "existing_main_metric_match": existing_validation,
            "example_parseval_full_fft_truth_ratio": parseval_truth,
            "example_parseval_full_fft_reconstruction_ratio": parseval_recon,
            "finite_metric_rows": bool(np.isfinite(per_state.value.to_numpy(float)).all()),
            "full_population_required_unless_allow_partial": not args.allow_partial,
        },
        "source_hashes": {
            str(path): sha256(path)
            for path in (SCRIPTS / "common/spectral.py", SCRIPTS / "common/cache.py")
        },
    }
    (output / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"[OK] {output}", flush=True)


if __name__ == "__main__":
    main()

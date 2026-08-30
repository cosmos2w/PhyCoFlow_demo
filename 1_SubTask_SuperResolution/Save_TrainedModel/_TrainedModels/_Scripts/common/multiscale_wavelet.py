"""Orthogonal 2-D wavelet decomposition for scale-resolved field fidelity.

PyWavelets is used when available.  A self-contained orthonormal Haar
implementation is retained as an explicit fallback; callers receive the
actual wavelet/backend in the returned metadata and therefore cannot silently
label Haar results as db2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

try:  # pragma: no cover - exercised in the configured ``fig`` environment.
    import pywt
except ImportError:  # pragma: no cover - validated by the fallback unit checks.
    pywt = None


@dataclass(frozen=True)
class WaveletContract:
    requested_wavelet: str
    actual_wavelet: str
    level: int
    boundary_mode: str
    backend: str
    fallback_used: bool


def resolve_contract(wavelet: str, level: int, boundary_mode: str) -> WaveletContract:
    """Resolve an explicit orthogonal transform contract."""
    requested = str(wavelet)
    level = int(level)
    if level < 1:
        raise ValueError("wavelet level must be positive")
    if pywt is None:
        return WaveletContract(requested, "haar", level, str(boundary_mode), "self_contained_haar", True)
    resolved = pywt.Wavelet(requested)
    if not resolved.orthogonal:
        raise ValueError(f"Configured wavelet must be orthogonal: {requested}")
    if boundary_mode not in pywt.Modes.modes:
        raise ValueError(f"Unsupported PyWavelets boundary mode: {boundary_mode}")
    return WaveletContract(requested, resolved.name, level, str(boundary_mode), "PyWavelets", False)


def validate_groups(groups: Mapping[str, Mapping], level: int) -> dict[str, dict]:
    """Validate a complete, non-overlapping partition of approximation/details."""
    normalized = {}
    assigned = []
    approximation_groups = []
    for name, spec in groups.items():
        details = sorted({int(value) for value in spec.get("detail_levels", [])})
        approximation = bool(spec.get("approximation", False))
        if not approximation and not details:
            raise ValueError(f"Scale group {name!r} is empty")
        invalid = [value for value in details if value < 1 or value > int(level)]
        if invalid:
            raise ValueError(f"Scale group {name!r} has invalid detail levels: {invalid}")
        assigned.extend(details)
        if approximation:
            approximation_groups.append(name)
        normalized[str(name)] = {"approximation": approximation, "detail_levels": details}
    if len(approximation_groups) != 1:
        raise ValueError(f"Exactly one group must contain the approximation; found {approximation_groups}")
    if sorted(assigned) != list(range(1, int(level) + 1)):
        raise ValueError(
            "Detail levels must form a non-overlapping complete partition: "
            f"assigned={sorted(assigned)}, expected={list(range(1, int(level) + 1))}"
        )
    return normalized


def scale_group_labels(groups: Mapping[str, Mapping], level: int) -> dict[str, str]:
    """Derive approximate grid-cell support labels from dyadic detail levels."""
    normalized = validate_groups(groups, level)
    labels = {}
    for name, spec in normalized.items():
        details = spec["detail_levels"]
        if spec["approximation"]:
            threshold_level = min(details) if details else int(level)
            labels[name] = f"structures approximately >= {2 ** threshold_level} H-grid cells"
        else:
            labels[name] = (
                f"approximately {2 ** min(details)}-{2 ** (max(details) + 1)} H-grid cells"
            )
    return labels


def _haar_wavedec2(field: np.ndarray, level: int):
    """Minimal orthonormal Haar analysis for even dyadic grids."""
    current = np.asarray(field, dtype=np.float64)
    details = []
    root2 = np.sqrt(2.0)
    for _ in range(int(level)):
        if current.shape[0] % 2 or current.shape[1] % 2:
            raise ValueError("Haar fallback requires even dimensions at every requested level")
        low_x = (current[:, 0::2] + current[:, 1::2]) / root2
        high_x = (current[:, 0::2] - current[:, 1::2]) / root2
        ll = (low_x[0::2] + low_x[1::2]) / root2
        lh = (low_x[0::2] - low_x[1::2]) / root2
        hl = (high_x[0::2] + high_x[1::2]) / root2
        hh = (high_x[0::2] - high_x[1::2]) / root2
        details.append((lh, hl, hh))
        current = ll
    return [current, *details[::-1]]


def _haar_waverec2(coeffs):
    """Inverse of :func:`_haar_wavedec2` for the PyWavelets-like layout."""
    current = np.asarray(coeffs[0], dtype=np.float64)
    root2 = np.sqrt(2.0)
    for lh, hl, hh in coeffs[1:]:
        lh = np.asarray(lh, dtype=np.float64)
        hl = np.asarray(hl, dtype=np.float64)
        hh = np.asarray(hh, dtype=np.float64)
        low_x = np.empty((current.shape[0] * 2, current.shape[1]), dtype=np.float64)
        high_x = np.empty_like(low_x)
        low_x[0::2] = (current + lh) / root2
        low_x[1::2] = (current - lh) / root2
        high_x[0::2] = (hl + hh) / root2
        high_x[1::2] = (hl - hh) / root2
        restored = np.empty((low_x.shape[0], low_x.shape[1] * 2), dtype=np.float64)
        restored[:, 0::2] = (low_x + high_x) / root2
        restored[:, 1::2] = (low_x - high_x) / root2
        current = restored
    return current


def _zeros_like_coeffs(coeffs):
    return [
        np.zeros_like(item) if index == 0 else tuple(np.zeros_like(value) for value in item)
        for index, item in enumerate(coeffs)
    ]


def decompose_field(
    field: np.ndarray,
    *,
    wavelet: str,
    level: int,
    boundary_mode: str,
    groups: Mapping[str, Mapping],
    reconstruction_tolerance: float = 1e-6,
):
    """Reconstruct configured full-size scale groups and validate closure."""
    grid = np.asarray(field, dtype=np.float64)
    if grid.ndim != 2 or not np.all(np.isfinite(grid)):
        raise ValueError("Wavelet decomposition requires one finite 2-D field")
    contract = resolve_contract(wavelet, level, boundary_mode)
    normalized_groups = validate_groups(groups, contract.level)
    if contract.backend == "PyWavelets":
        max_level = pywt.dwtn_max_level(grid.shape, contract.actual_wavelet)
        if contract.level > max_level:
            raise ValueError(
                f"Requested level {contract.level} exceeds maximum {max_level} for {grid.shape}"
            )
        coeffs = pywt.wavedec2(
            grid, wavelet=contract.actual_wavelet, mode=contract.boundary_mode,
            level=contract.level,
        )
        reconstruct = lambda value: pywt.waverec2(
            value, wavelet=contract.actual_wavelet, mode=contract.boundary_mode,
        )
    else:
        if contract.boundary_mode != "periodization":
            raise ValueError("The self-contained Haar fallback supports periodization only")
        coeffs = _haar_wavedec2(grid, contract.level)
        reconstruct = _haar_waverec2

    components = {}
    for name, spec in normalized_groups.items():
        selected = _zeros_like_coeffs(coeffs)
        if spec["approximation"]:
            selected[0] = np.array(coeffs[0], copy=True)
        for detail_level in spec["detail_levels"]:
            coefficient_index = contract.level - int(detail_level) + 1
            selected[coefficient_index] = tuple(
                np.array(value, copy=True) for value in coeffs[coefficient_index]
            )
        component = np.asarray(reconstruct(selected), dtype=np.float64)
        components[name] = component[: grid.shape[0], : grid.shape[1]]

    reconstructed = np.sum(np.stack(list(components.values()), axis=0), axis=0)
    residual = float(
        np.linalg.norm(reconstructed - grid) / (np.linalg.norm(grid) + np.finfo(np.float64).eps)
    )
    if not np.isfinite(residual) or residual > float(reconstruction_tolerance):
        raise ValueError(
            f"Wavelet scale groups do not reconstruct the field: residual={residual:.3e}, "
            f"tolerance={float(reconstruction_tolerance):.3e}"
        )
    return components, residual, contract


def scale_metrics(truth_components, pred_components, eps=1e-12):
    """Return the requested structural and variance-allocation metrics."""
    names = list(truth_components)
    if set(names) != set(pred_components):
        raise ValueError("Truth and prediction scale groups differ")
    truth_energy = {
        name: float(np.sum(np.asarray(truth_components[name], dtype=np.float64) ** 2, dtype=np.float64))
        for name in names
    }
    pred_energy = {
        name: float(np.sum(np.asarray(pred_components[name], dtype=np.float64) ** 2, dtype=np.float64))
        for name in names
    }
    truth_total = float(sum(value + eps for value in truth_energy.values()))
    pred_total = float(sum(value + eps for value in pred_energy.values()))
    rows = {}
    for name in names:
        truth = np.asarray(truth_components[name], dtype=np.float64)
        pred = np.asarray(pred_components[name], dtype=np.float64)
        truth_norm = float(np.sqrt(truth_energy[name]))
        pred_norm = float(np.sqrt(pred_energy[name]))
        difference_norm = float(np.sqrt(np.sum((pred - truth) ** 2, dtype=np.float64)))
        truth_fraction = float((truth_energy[name] + eps) / truth_total)
        pred_fraction = float((pred_energy[name] + eps) / pred_total)
        rows[name] = {
            "component_rel_l2": float(difference_norm / (truth_norm + eps)),
            "pattern_correlation": float(np.sum(pred * truth, dtype=np.float64) / (pred_norm * truth_norm + eps)),
            "variance_fraction_true": truth_fraction,
            "variance_fraction_pred": pred_fraction,
            "variance_fraction_bias_pp": float(100.0 * (pred_fraction - truth_fraction)),
            "component_energy_ratio_db": float(10.0 * np.log10((pred_energy[name] + eps) / (truth_energy[name] + eps))),
        }
    return rows

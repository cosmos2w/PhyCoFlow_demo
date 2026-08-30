"""NumPy H-to-M/L projector matching the canonical PyTorch implementation.

This plotting-safe implementation avoids importing PyTorch in the ``fig``
environment.  It uses exact block area averages for the nested PDEBench grids
and the align-corners-false bilinear coordinate convention on return to H.
"""
from __future__ import annotations

import numpy as np


def _resize_linear_axis(values, output_length, axis):
    values = np.asarray(values, dtype=np.float64)
    input_length = values.shape[axis]
    source = (np.arange(output_length, dtype=np.float64) + .5) * input_length / output_length - .5
    source = np.clip(source, 0.0, input_length - 1.0)
    lower = np.floor(source).astype(int)
    upper = np.minimum(lower + 1, input_length - 1)
    weight = source - lower
    shape = [1] * values.ndim
    shape[axis] = output_length
    return (
        np.take(values, lower, axis=axis) * (1.0 - weight.reshape(shape))
        + np.take(values, upper, axis=axis) * weight.reshape(shape)
    )


def project_grid(field_flat, src_shape, target_shape):
    """Area-average to target and bilinearly return to the source grid."""
    src_y, src_x = map(int, src_shape)
    target_y, target_x = map(int, target_shape)
    if src_y % target_y or src_x % target_x:
        raise ValueError(f"Nested-grid projector requires integer factors: {src_shape} -> {target_shape}")
    grid = np.asarray(field_flat, dtype=np.float64).reshape(src_y, src_x)
    factor_y, factor_x = src_y // target_y, src_x // target_x
    low = grid.reshape(target_y, factor_y, target_x, factor_x).mean(axis=(1, 3))
    up_x = _resize_linear_axis(low, src_x, axis=1)
    up = _resize_linear_axis(up_x, src_y, axis=0)
    return up.reshape(-1), low.reshape(-1)


def decompose(field_flat, src_shape, target_shape):
    coarse, native_low = project_grid(field_flat, src_shape, target_shape)
    field = np.asarray(field_flat, dtype=np.float64).reshape(-1)
    return coarse, field - coarse, native_low


def component_metrics(truth, pred, src_shape, target_shape, eps=1e-30):
    """Return scale-separated fidelity metrics for one aligned field pair.

    ``detail_energy_ratio`` is retained as a compatibility alias for the
    truth detail-energy fraction used by the original exporter.
    """
    truth_coarse, truth_detail, _ = decompose(truth, src_shape, target_shape)
    pred_coarse, pred_detail, _ = decompose(pred, src_shape, target_shape)
    norm = lambda value: float(np.sqrt(np.sum(np.asarray(value, dtype=np.float64) ** 2, dtype=np.float64)))
    rel = lambda a, b: float(norm(np.asarray(b) - np.asarray(a)) / (norm(a) + eps))
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    truth_detail_energy = float(np.sum(truth_detail * truth_detail, dtype=np.float64))
    pred_detail_energy = float(np.sum(pred_detail * pred_detail, dtype=np.float64))
    detail_dot = float(np.sum(truth_detail * pred_detail, dtype=np.float64))
    detail_norm_product = float(norm(truth_detail) * norm(pred_detail))
    truth_detail_fraction = truth_detail_energy / (float(np.sum(truth * truth, dtype=np.float64)) + eps)
    return {
        "coarse_rel_l2": rel(truth_coarse, pred_coarse),
        "detail_rel_l2": rel(truth_detail, pred_detail),
        "detail_correlation": float(detail_dot / (detail_norm_product + eps)),
        "detail_energy_bias_db": float(10.0 * np.log10((pred_detail_energy + eps) / (truth_detail_energy + eps))),
        "detail_energy_fraction_true": float(truth_detail_fraction),
        "detail_energy_fraction_pred": float(
            pred_detail_energy / (float(np.sum(pred * pred, dtype=np.float64)) + eps)
        ),
        "detail_energy_ratio": float(truth_detail_fraction),
        "full_rel_l2": rel(truth, pred),
    }, (truth_coarse, truth_detail, pred_coarse, pred_detail)

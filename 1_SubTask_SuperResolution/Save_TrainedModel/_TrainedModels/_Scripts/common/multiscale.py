"""Resolution-aligned H-to-M/L projectors and coarse/detail metrics."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def project_grid(field_flat, src_shape: tuple[int, int], target_shape: tuple[int, int]):
    """Area-average downsample then bilinear upsample back to the source grid."""
    a = torch.as_tensor(np.asarray(field_flat), dtype=torch.float32).reshape(1, 1, *src_shape)
    low = F.interpolate(a, size=target_shape, mode="area")
    up = F.interpolate(low, size=src_shape, mode="bilinear", align_corners=False)
    return up.reshape(-1).cpu().numpy(), low.reshape(-1).cpu().numpy()


def decompose(field_flat, src_shape, target_shape):
    coarse, native_low = project_grid(field_flat, src_shape, target_shape)
    field = np.asarray(field_flat).reshape(-1)
    return coarse, field - coarse, native_low


def component_metrics(truth, pred, src_shape, target_shape, eps=1e-12):
    tc, td, _ = decompose(truth, src_shape, target_shape)
    pc, pd, _ = decompose(pred, src_shape, target_shape)
    rel = lambda a, b: float(np.linalg.norm(np.asarray(b)-np.asarray(a)) / (np.linalg.norm(a)+eps))
    return {
        "coarse_rel_l2": rel(tc, pc), "detail_rel_l2": rel(td, pd),
        "detail_energy_ratio": float(np.sum(td**2) / (np.sum(np.asarray(truth)**2)+eps)),
        "full_rel_l2": rel(truth, pred),
    }, (tc, td, pc, pd)

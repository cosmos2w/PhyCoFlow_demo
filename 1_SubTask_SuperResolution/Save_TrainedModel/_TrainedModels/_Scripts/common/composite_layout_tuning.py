"""Centralized vertical tuning for the unified publication composite.

The user-facing controls live under ``assembly.vertical_tuning`` in
``publication_layout_unified_v2.yaml``.  This module converts those controls
to the effective row heights and Matplotlib SubFigure spacing used by the
assembler and audit code.
"""
from __future__ import annotations

import copy
import math


ROW_KEYS = ("a_b", "c", "d", "e_f")


def apply_composite_vertical_tuning(layout: dict) -> dict:
    """Return a tuned layout with validated physical row clearances."""
    out = copy.deepcopy(layout)
    assembly = out["assembly"]
    tuning = assembly.get("vertical_tuning", {})
    base_heights = [float(value) for value in assembly["compact_row_heights_mm"]]
    if len(base_heights) != len(ROW_KEYS):
        raise ValueError("assembly.compact_row_heights_mm must contain four row heights")

    global_scale = float(tuning.get("global_height_scale", 1.0))
    if not math.isfinite(global_scale) or global_scale <= 0:
        raise ValueError("assembly.vertical_tuning.global_height_scale must be positive")
    multipliers = tuning.get("row_height_multipliers", {})
    unknown = sorted(set(multipliers) - set(ROW_KEYS))
    if unknown:
        raise ValueError(f"Unknown vertical-tuning row keys: {unknown}; use {ROW_KEYS}")
    effective_heights = []
    for key, base in zip(ROW_KEYS, base_heights):
        multiplier = float(multipliers.get(key, 1.0))
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError(f"Row-height multiplier for {key} must be positive")
        effective_heights.append(base * global_scale * multiplier)

    requested_gap = float(tuning.get("inter_row_gap_mm", assembly.get("row_gap_mm", .8)))
    minimum_gap = float(tuning.get("minimum_inter_row_gap_mm", .6))
    if not all(math.isfinite(value) and value >= 0 for value in (requested_gap, minimum_gap)):
        raise ValueError("Composite row gaps must be finite and non-negative")
    if requested_gap < minimum_gap:
        raise ValueError(
            "Requested composite inter-row gap is below the overlap guard: "
            f"{requested_gap:.3f} mm < {minimum_gap:.3f} mm"
        )

    # Matplotlib defines SubFigure hspace relative to the mean row height.
    # This conversion keeps the user-facing parameter in physical millimetres.
    mean_row_height = sum(effective_heights) / len(effective_heights)
    subfigure_hspace = requested_gap / mean_row_height
    canvas_extra = float(tuning.get("canvas_extra_height_mm", .5))
    if not math.isfinite(canvas_extra) or canvas_extra < 0:
        raise ValueError("assembly.vertical_tuning.canvas_extra_height_mm must be non-negative")

    assembly["compact_row_heights_mm"] = effective_heights
    assembly["row_gap_mm"] = requested_gap
    assembly["subfigure_hspace"] = subfigure_hspace
    assembly["canvas_extra_height_mm"] = canvas_extra
    assembly["minimum_inter_row_gap_mm"] = minimum_gap
    assembly["vertical_tuning_resolved"] = {
        "row_keys": list(ROW_KEYS),
        "base_row_heights_mm": base_heights,
        "global_height_scale": global_scale,
        "row_height_multipliers": {
            key: float(multipliers.get(key, 1.0)) for key in ROW_KEYS
        },
        "effective_row_heights_mm": effective_heights,
        "requested_inter_row_gap_mm": requested_gap,
        "minimum_inter_row_gap_mm": minimum_gap,
        "effective_subfigure_hspace": subfigure_hspace,
        "canvas_extra_height_mm": canvas_extra,
    }
    return out

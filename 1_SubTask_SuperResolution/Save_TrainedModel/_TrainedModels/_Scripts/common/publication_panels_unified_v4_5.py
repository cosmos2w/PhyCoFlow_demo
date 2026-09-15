"""Typography-only layer for mixed-resolution Figure V4_5.

All panel drawers and scientific artists come from V4_4.  This module only
assigns the approved semantic font roles, removes obsolete local-size
overrides, normalizes weights, and reasserts contrast-aware matrix text.
"""
from __future__ import annotations

import matplotlib
import numpy as np

from . import publication_panels_unified_v4_4 as v44


PANEL_OUTPUT_NAMES = {
    key: value.replace("V4_4", "V4_5") for key, value in v44.PANEL_OUTPUT_NAMES.items()
}

MAJOR_TITLES = {
    "Contains H-resolution training fields",
    "Zero-H training",
    "High resolution reconstruction (512 sensors)",
    "Spatial pattern correlation",
    "Variance allocation bias [pp]",
}


def _mark(metadata: dict, label: str) -> dict:
    result = dict(metadata)
    result.update({
        "figure_revision": "V4_5",
        "source_visual_revision": "V4_4",
        "source_scientific_revision": "V3-7",
        "panel_id": f"fig3.{label}",
        "underlying_source_arrays_preserved": True,
        "data_mapping_preserved_from_v4_4": True,
        "model_training_or_inference": False,
    })
    return result


def draw_panel(label: str, parent, ctx, **kwargs):
    return _mark(v44.draw_panel(label, parent, ctx, **kwargs), label)


def _relative_luminance(rgb) -> float:
    values = np.asarray(rgb[:3], dtype=float)
    values = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    return float(values @ np.asarray([0.2126, 0.7152, 0.0722]))


def apply_v4_5_typography(fig) -> dict:
    """Assign the strict V4_5 roles and prove matrix contrast decisions."""
    texts = fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text))
    for item in texts:
        if hasattr(item, "_global_font_size_pt"):
            delattr(item, "_global_font_size_pt")
        role = getattr(item, "_global_font_role", None)
        gid = str(item.get_gid() or "")
        if role == "panel_label" or gid == "font-role:panel_label":
            item._global_font_role = "panel_label"
            item.set_fontweight("bold")
        elif item.get_text() in MAJOR_TITLES:
            item._global_font_role = "major_title"
            item.set_fontweight("normal")
        else:
            item.set_fontweight("normal")

    matrix_records = []
    axes = fig.findobj(match=lambda item: isinstance(item, matplotlib.axes.Axes))
    for axis in axes:
        if str(axis.get_gid() or "") != "panel-e-metric-matrix" or len(axis.images) != 1:
            continue
        image = axis.images[0]
        values = np.ma.asarray(image.get_array())
        for item in axis.texts:
            x, y = map(float, item.get_position())
            row, column = int(round(y)), int(round(x))
            if not (values.ndim == 2 and 0 <= row < values.shape[0] and 0 <= column < values.shape[1]):
                continue
            rgba = image.get_cmap()(image.norm(float(values[row, column])))
            luminance = _relative_luminance(rgba)
            black_contrast = (luminance + 0.05) / 0.05
            white_contrast = 1.05 / (luminance + 0.05)
            color = "#111111" if black_contrast >= white_contrast else "#FFFFFF"
            contrast = max(black_contrast, white_contrast)
            item.set_color(color)
            item._global_font_role = "annotation"
            item.set_fontweight("normal")
            matrix_records.append({
                "text": item.get_text(), "row": row, "column": column,
                "color": color, "contrast_ratio": float(contrast),
            })

    visible = [item for item in texts if item.get_visible() and item.get_text()]
    unexpected_bold = []
    for item in visible:
        weight = item.get_fontweight()
        bold = weight in {"bold", "semibold", "demibold", 600, 700, 800, 900}
        role = getattr(item, "_global_font_role", None)
        if bold and role != "panel_label":
            unexpected_bold.append(item.get_text())
    major = [item.get_text() for item in visible if getattr(item, "_global_font_role", None) == "major_title"]
    return {
        "major_title_text": major,
        "major_title_count": len(major),
        "unexpected_bold_text": unexpected_bold,
        "only_panel_labels_bold": not unexpected_bold,
        "local_size_override_count": sum(hasattr(item, "_global_font_size_pt") for item in visible),
        "matrix_annotation_count": len(matrix_records),
        "matrix_annotation_min_contrast_ratio": min(
            (item["contrast_ratio"] for item in matrix_records), default=None,
        ),
        "matrix_annotation_colors": sorted({item["color"] for item in matrix_records}),
        "matrix_annotation_records": matrix_records,
    }


panel_label = v44.panel_label
center_panel_d_headers = v44.center_panel_d_headers
measure_panel_b_legend_clearance = v44.measure_panel_b_legend_clearance
measure_panel_d_header_alignment = v44.measure_panel_d_header_alignment
measure_major_content_gaps = v44.measure_major_content_gaps
record_panel_e_v4_3_tick_settings = v44.record_panel_e_v4_3_tick_settings

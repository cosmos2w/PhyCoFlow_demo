"""User-facing micro-adjustment API for publication Panel C.

Edit only the clearly marked block below for routine visual experiments.  Both
the standalone Panel C exporter and the native master assembler import these
values, so a single edit updates both outputs without touching cached data.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .figure_style import (
    SIZE_AXIS_LABEL,
    SIZE_SUBPLOT_TITLE,
    SIZE_TICK_LABEL,
)


# =============================================================================
# PANEL C USER TUNING API
# =============================================================================
# Dataset selection -----------------------------------------------------------
# SNAPSHOT_INDEX:
#   * None preserves the representative snapshot selected by the audited CSV.
#   * Any integer from 0 through 299 selects that canonical cached test frame.
#
# SENSOR_COUNT:
#   * Version 2 valid choices: 64, 128, 256, 384, 512, 768, or 1024.
#   * Version 1 has complete all-model/all-recipe coverage only at 256.
# These settings only select existing cache entries; they never run inference.
# None inherits representative_snapshots.shared_snapshot_index from the layout.
# Set an integer here only as a deliberate Panel-C-only cache selection override.
SNAPSHOT_INDEX = None
SENSOR_COUNT = 512

# Sensor-overlay visibility ---------------------------------------------------
# False hides sensors only on the first-row/first-column full ground-truth map.
# The zoomed ground-truth ROI and "Observed values" row still show sensors.
# Set True to restore sensors on the full ground-truth map.
SHOW_SENSORS_ON_FULL_GROUND_TRUTH = False

# Zoomed-in region ------------------------------------------------------------
# ROI_MODE valid choices:
#   * "automatic": maximum integrated ground-truth gradient (current default).
#   * "manual_square": use ROI_CENTER_PHYS and ROI_SIDE_LENGTH_PHYS below.
#
# The audited CFD field has physical x/y ranges [-1.0, 1.0].  For a manual ROI:
#   * center coordinates must lie inside [-1.0, 1.0] x [-1.0, 1.0];
#   * side length must satisfy 0 < side <= 2.0;
#   * the complete square must remain in the domain, i.e.
#       -1 <= center_x +/- side/2 <= 1 and likewise for center_y.
# Runtime validation uses the selected cache coordinates, rather than assuming
# the domain, and reports a clear error for an invalid box.
ROI_MODE = "automatic"
ROI_CENTER_PHYS = (0.5, 0.0)
ROI_SIDE_LENGTH_PHYS = 0.50

# Compact layout --------------------------------------------------------------
# The main model-header y positions remain fixed.  Increasing GRID_TOP_COMPOSITE
# moves the scheme titles and every image row upward, reducing the header gap.
# GRID_RIGHT remains fixed so the right edge of the error colorbar does not move.
# Reducing GRID_LEFT expands the grid toward the page margin; COLUMN_GAP then
# distributes the reclaimed width evenly between columns.  Every field axis is
# still forced to equal physical aspect, so no field is stretched.
GRID_LEFT = 0.005
GRID_RIGHT = 0.970
GRID_BOTTOM = 0.036
GRID_TOP_STANDALONE = 0.880
GRID_TOP_COMPOSITE = 0.880
COLUMN_GAP = 0.012
COLORBAR_WIDTH_RATIO = 0.060
# Independent colorbar-length controls. Both are fractions of their available
# row span and must remain in (0, 1]. FIELD applies to rows 1+2; ERROR applies
# only to the third row. The third-row bar keeps its bottom aligned with the
# image and reserves only the small top cap needed by the scientific multiplier.
FIELD_COLORBAR_LENGTH_RATIO = 1.00
ERROR_COLORBAR_LENGTH_RATIO = 1.00

# Right colorbar typography ---------------------------------------------------
# COLORBAR_LABEL_FONTSIZE_PT controls the vertical texts "Field value" and
# "Absolute error". COLORBAR_TICK_FONTSIZE_PT controls their numeric labels.
COLORBAR_LABEL_FONTSIZE_PT = SIZE_AXIS_LABEL
COLORBAR_TICK_FONTSIZE_PT = SIZE_TICK_LABEL
COLORBAR_LABEL_PAD_PT = 7.0
# Automatic horizontal redistribution ---------------------------------------
# Reducing the larger colorbar font below the reference size distributes the
# reclaimed allowance uniformly across every inter-column gap.  GRID_RIGHT is
# never changed, so the right edge of the error colorbar stays fixed.
#
# Example with the defaults:
#   effective gap = 0.080 + (COLORBAR_SPACING_REFERENCE_PT - COLORBAR_LABEL_FONTSIZE_PT) 
#                                                   * COLUMN_GAP_GAIN_PER_REDUCED_FONT_PT = ...
# Set COLUMN_GAP_GAIN_PER_REDUCED_FONT_PT to 0.0 for manual spacing only.
COLORBAR_SPACING_REFERENCE_PT = 5.0
COLUMN_GAP_GAIN_PER_REDUCED_FONT_PT = 0.008

# Internal text and row spacing ------------------------------------------------
# FIRST_ROW_HEADER_FONTSIZE_PT controls "Reference", "DMF-Gen",
# "FFM-Perceiver", and "Senseiver".
# SECOND_ROW_HEADER_FONTSIZE_PT controls the single-line recipe labels above
# the first image row.  Keeping them on one line prevents their bounding boxes
# from intruding into the model-header row on the compact master canvas.
FIRST_ROW_HEADER_FONTSIZE_PT = SIZE_SUBPLOT_TITLE
SECOND_ROW_HEADER_FONTSIZE_PT = SIZE_TICK_LABEL

# LEFT_ROW_LABEL_FONTSIZE_PT controls the three labels at the far left:
# "Full H-resolution field", "Zoomed-in region", and "Error / sensors".
LEFT_ROW_LABEL_FONTSIZE_PT = SIZE_AXIS_LABEL

# IMAGE_ROW_GAP is measured relative to one image-row height. Increasing it
# separates the three rows without resizing the image cells: panel C's physical
# height, and therefore the master-canvas height, expand automatically. The
# YAML height of panel C is the reference height at the default value below.
# A practical tuning range for this layout is approximately 0.04 to 0.40.
IMAGE_ROW_GAP = 0.18

# Internal geometry reference. Do not tune this alongside IMAGE_ROW_GAP; it
# records the gap for which physical_layout.rows[panel c].height_mm was chosen.
_REFERENCE_IMAGE_ROW_GAP = 0.18
_IMAGE_ROW_COUNT = 3

# COLUMN_HEADER_PAD_PT is the Matplotlib title padding above the first image row.
# Increasing it moves "Ground truth" / recipe labels upward and therefore closer
# to the fixed model-header row; decreasing it moves them toward the images and
# increases the gap from "Reference" / model names.  Keep enough clearance to
# avoid contact with the model headers.  
# A practical range is 0.0 to 4.0 pt.
# For coarse adjustment of this same gap, change GRID_TOP_STANDALONE or
# GRID_TOP_COMPOSITE; use COLUMN_HEADER_PAD_PT for the final fine adjustment.
COLUMN_HEADER_PAD_PT = 2.0

# Vertical grey model-separation lines ---------------------------------------
# Values use Panel C parent coordinates: 0.0 is the bottom and 1.0 is the top.
# The visible line length is VERTICAL_DIVIDER_Y_MAX - VERTICAL_DIVIDER_Y_MIN.
# Increase Y_MIN to shorten from the bottom; decrease Y_MAX to shorten from the
# top.  For a centered shorter line, change both by the same amount, e.g.
# Y_MIN=0.10 and Y_MAX=0.90.  Required: 0 <= Y_MIN < Y_MAX <= 1.
VERTICAL_DIVIDER_Y_MIN = 0.005
VERTICAL_DIVIDER_Y_MAX = 0.995

# Header coordinates are independent controls.  Keep these fixed when you want
# only the column-title/image block to move upward or downward.
MAIN_HEADER_Y_STANDALONE = 0.985
MAIN_HEADER_Y_COMPOSITE = 0.985
RECIPE_HEADER_Y_STANDALONE = 0.940
RECIPE_HEADER_Y_COMPOSITE = 0.940

# Panel C physical size is deliberately not duplicated here. Edit the
# height_mm of the physical_layout row containing [c] in the publication layout;
# that one value controls both the standalone page and composite rectangle.
# =============================================================================
# END PANEL C USER TUNING API
# =============================================================================


VALID_SNAPSHOT_MIN = 0
VALID_SNAPSHOT_MAX = 299
VALID_SENSOR_COUNTS_BY_VERSION = {
    1: (256,),
    2: (64, 128, 256, 384, 512, 768, 1024),
}


def apply_panel_c_tuning(layout: dict) -> dict:
    """Apply the shared API values to a loaded unified-v2 layout in place."""
    panel_c = layout["panel_c"]
    panel_c["show_sensors_on_full_ground_truth"] = bool(
        SHOW_SENSORS_ON_FULL_GROUND_TRUTH
    )
    compact = panel_c.setdefault("compact_layout", {})
    largest_colorbar_font = max(
        float(COLORBAR_LABEL_FONTSIZE_PT),
        float(COLORBAR_TICK_FONTSIZE_PT),
    )
    font_reduction = max(
        0.0,
        float(COLORBAR_SPACING_REFERENCE_PT) - largest_colorbar_font,
    )
    effective_column_gap = (
        float(COLUMN_GAP)
        + font_reduction * float(COLUMN_GAP_GAIN_PER_REDUCED_FONT_PT)
    )
    divider_y_min = float(VERTICAL_DIVIDER_Y_MIN)
    divider_y_max = float(VERTICAL_DIVIDER_Y_MAX)
    if not 0.0 <= divider_y_min < divider_y_max <= 1.0:
        raise ValueError(
            "Panel C divider endpoints must satisfy "
            "0 <= VERTICAL_DIVIDER_Y_MIN < VERTICAL_DIVIDER_Y_MAX <= 1."
        )
    compact.update({
        "grid_left": float(GRID_LEFT),
        "grid_right": float(GRID_RIGHT),
        "grid_bottom": float(GRID_BOTTOM),
        "grid_top_standalone": float(GRID_TOP_STANDALONE),
        "grid_top_composite": float(GRID_TOP_COMPOSITE),
        "colorbar_width_ratio": float(COLORBAR_WIDTH_RATIO),
        "field_colorbar_length_ratio": float(FIELD_COLORBAR_LENGTH_RATIO),
        "error_colorbar_length_ratio": float(ERROR_COLORBAR_LENGTH_RATIO),
        "column_gap": effective_column_gap,
        "column_gap_base": float(COLUMN_GAP),
        "colorbar_label_fontsize": float(COLORBAR_LABEL_FONTSIZE_PT),
        "colorbar_tick_fontsize": float(COLORBAR_TICK_FONTSIZE_PT),
        "colorbar_labelpad": float(COLORBAR_LABEL_PAD_PT),
        "colorbar_spacing_reference_fontsize": float(COLORBAR_SPACING_REFERENCE_PT),
        "column_gap_gain_per_reduced_font_pt": float(COLUMN_GAP_GAIN_PER_REDUCED_FONT_PT),
        "row_gap": float(IMAGE_ROW_GAP),
        "model_header_fontsize": float(FIRST_ROW_HEADER_FONTSIZE_PT),
        "scheme_header_fontsize": float(SECOND_ROW_HEADER_FONTSIZE_PT),
        "side_label_fontsize": float(LEFT_ROW_LABEL_FONTSIZE_PT),
        "scheme_header_pad": float(COLUMN_HEADER_PAD_PT),
        "vertical_divider_y_min": divider_y_min,
        "vertical_divider_y_max": divider_y_max,
        "model_header_y": float(MAIN_HEADER_Y_STANDALONE),
        "model_header_y_composite": float(MAIN_HEADER_Y_COMPOSITE),
        "recipe_header_y": float(RECIPE_HEADER_Y_STANDALONE),
        "recipe_header_y_composite": float(RECIPE_HEADER_Y_COMPOSITE),
    })
    panel_c_rows = [
        row for row in layout["physical_layout"]["rows"]
        if list(row.get("panels", ())) == ["c"]
    ]
    if len(panel_c_rows) != 1:
        raise ValueError("Expected exactly one single-panel physical-layout row for panel C.")
    panel_c_row = panel_c_rows[0]
    reference_height_mm = float(panel_c_row.setdefault(
        "_height_mm_at_reference_image_row_gap", panel_c_row["height_mm"],
    ))
    if reference_height_mm <= 0.0 or float(IMAGE_ROW_GAP) < 0.0:
        raise ValueError("Panel C height must be positive and IMAGE_ROW_GAP cannot be negative.")
    reference_units = _IMAGE_ROW_COUNT + (_IMAGE_ROW_COUNT - 1) * _REFERENCE_IMAGE_ROW_GAP
    requested_units = _IMAGE_ROW_COUNT + (_IMAGE_ROW_COUNT - 1) * float(IMAGE_ROW_GAP)
    panel_c_row["height_mm"] = reference_height_mm * requested_units / reference_units
    compact["physical_height_reference_mm"] = reference_height_mm
    compact["physical_height_reference_row_gap"] = _REFERENCE_IMAGE_ROW_GAP
    compact["derived_panel_height_mm"] = float(panel_c_row["height_mm"])
    return layout


def resolve_panel_c_selection(default_snapshot: int, default_sensor_count: int, version: int) -> tuple[int, int]:
    """Resolve and validate the cache-only snapshot and sensor selection."""
    snapshot = int(default_snapshot if SNAPSHOT_INDEX is None else SNAPSHOT_INDEX)
    count = int(SENSOR_COUNT if SENSOR_COUNT is not None else default_sensor_count)
    if not VALID_SNAPSHOT_MIN <= snapshot <= VALID_SNAPSHOT_MAX:
        raise ValueError(
            f"Panel C SNAPSHOT_INDEX={snapshot} is invalid; choose an integer "
            f"from {VALID_SNAPSHOT_MIN} through {VALID_SNAPSHOT_MAX}."
        )
    valid_counts = VALID_SENSOR_COUNTS_BY_VERSION.get(int(version))
    if valid_counts is None:
        raise ValueError(f"Unsupported Panel C qualitative version: {version}")
    if count not in valid_counts:
        raise ValueError(
            f"Panel C SENSOR_COUNT={count} is not fully cached for Version {version}; "
            f"choose one of {valid_counts}."
        )
    return snapshot, count


def resolve_panel_c_roi(
    coords: np.ndarray,
    truth: np.ndarray,
    automatic_roi,
    *,
    automatic_fraction: float,
) -> tuple[list[float], dict]:
    """Return an automatic or validated manual square ROI and its provenance."""
    mode = str(ROI_MODE).strip().lower()
    if mode == "automatic":
        roi = automatic_roi(coords, truth, fraction=float(automatic_fraction))
        return [float(value) for value in roi], {
            "selection": "maximum integrated ground-truth gradient magnitude",
            "mode": "automatic",
        }
    if mode != "manual_square":
        raise ValueError("Panel C ROI_MODE must be 'automatic' or 'manual_square'.")

    center: Sequence[float] = ROI_CENTER_PHYS
    if len(center) != 2:
        raise ValueError("Panel C ROI_CENTER_PHYS must contain exactly (center_x, center_y).")
    center_x, center_y = map(float, center)
    side = float(ROI_SIDE_LENGTH_PHYS)
    xy = np.asarray(coords, dtype=float)[:, :2]
    xmin, ymin = np.min(xy, axis=0)
    xmax, ymax = np.max(xy, axis=0)
    if not np.isfinite([center_x, center_y, side]).all() or side <= 0:
        raise ValueError("Panel C manual ROI center and side must be finite, with side > 0.")
    roi = [center_x - side / 2, center_x + side / 2,
           center_y - side / 2, center_y + side / 2]
    tolerance = 1e-9
    if (roi[0] < xmin - tolerance or roi[1] > xmax + tolerance
            or roi[2] < ymin - tolerance or roi[3] > ymax + tolerance):
        raise ValueError(
            "Panel C manual ROI lies outside the selected cache domain: "
            f"requested x=[{roi[0]:.4g}, {roi[1]:.4g}], y=[{roi[2]:.4g}, {roi[3]:.4g}]; "
            f"available x=[{xmin:.4g}, {xmax:.4g}], y=[{ymin:.4g}, {ymax:.4g}]."
        )
    return roi, {
        "selection": "manual ground-truth-coordinate square",
        "mode": "manual_square",
        "center_x": center_x,
        "center_y": center_y,
        "side_length": side,
    }

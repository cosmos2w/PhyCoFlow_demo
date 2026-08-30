#!/usr/bin/env python
"""Native one-canvas Nature-style coupled-field reconstruction figure.

The assembler uses finalized CSVs for all quantitative panels and only reads
the existing reconstruction cache for the qualitative physical-field maps. It
never loads a checkpoint or recomputes a metric.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedFormatter, FixedLocator, LogFormatterMathtext, NullLocator
import numpy as np
import seaborn as sns
import yaml
import global_style as global_style_contract

from common.cache import load_cache
from common.config import FIGURES_DIR, RESULTS_DIR, SCRIPT_DIR, add_common_args, load_config, method_items, run_id
from common.figure_style import apply_style, mark_missing, method_line_style
from common.io_utils import latest, read_csv
from common.statistics import relative_l2
from common.statistics import jsd_base2
from common.pdf_utils import PAIR_FIELDS, histogram
from global_style import (
    AUTO_REFLOW_RIGHT_COLUMN_FOR_FONTS,
    AUTO_REFLOW_SAFETY_PAD_IN,
    COLOR_DIVIDER,
    COLOR_GROUND_TRUTH,
    COLOR_GRID,
    COLOR_MISSING_TEXT,
    COMPOSITE_MARGIN_BOTTOM_IN,
    COMPOSITE_MARGIN_LEFT_IN,
    COMPOSITE_MARGIN_RIGHT_IN,
    COMPOSITE_MARGIN_TOP_IN,
    COMPOSITE_WIDTH_IN,
    DEFAULT_LOWER_ROW_HEIGHT_IN,
    FONT_SIZES,
    GLOBAL_STYLE_VERSION,
    INTRA_PANEL_HSPACE,
    INTRA_PANEL_WSPACE,
    LOWER_ROW_MARGIN_LEFT_IN,
    LOWER_ROW_MARGIN_RIGHT_IN,
    LOWER_ROW_CANVAS_CLEARANCE_IN,
    LW_AXIS_SPINE,
    LW_DIVIDER,
    LW_ERRORBAR,
    LW_GRID,
    LW_LINE_PLOT,
    LW_LINE_SECONDARY,
    LINE_WIDTHS,
    MODEL_ALPHAS,
    MODEL_COLORS,
    PANEL_B_TO_RIGHT_TEXT_CLEARANCE_IN,
    PANEL_C_D_GAP_IN,
    PANEL_C_D_HEIGHT_RATIOS,
    PANEL_C_D_TEXT_CLEARANCE_IN,
    PANEL_C_INTERNAL_HEIGHT_RATIOS,
    PANEL_HSPACE_IN,
    PANEL_WSPACE_IN,
    RIGHT_COLUMN_MANUAL_SHIFT_IN,
    RIGHT_COLUMN_CANVAS_CLEARANCE_IN,
    SIZE_ANNOTATION,
    SIZE_AXIS_LABEL,
    SIZE_LEGEND,
    SIZE_PANEL_LABEL,
    SIZE_SUBPLOT_TITLE,
    SIZE_TICK_LABEL,
    adaptive_composite_height,
    gridspec_space_from_inches,
    model_alpha,
    model_color,
    save_publication_figure,
)

# ---------------------------------------------------------------------------
# PANEL-B / PANEL-D COLORMAP TEST CONTROLS
# ---------------------------------------------------------------------------
# Change only each ``selected`` value to compare the approved built-in maps.
# Startup validation rejects typos and prevents accidental use of an
# unreviewed map. Scientific normalization remains panel-specific:
#   * Panel b linearly normalizes every physical-field error column.
#   * Panel d uses one shared LogNorm over all positive 25-state pooled PDF bins;
#     exact zeros are masked and explicitly rendered pure white.
COLORMAP_CONFIG = {
    "panel_a": {

        # SHARED SEABORN PALETTE AUTHORITY. Change only ``selected`` between
        # "mako" and "crest" to restyle all Panel-a/SI physical-value and absolute-error contours. 
        # ``apply_to`` can exclude either role if a future chapter needs field-specific or error-specific palettes.
        # "seaborn_palette": {
        #     "selected": "mako",
        #     "options": ("mako", "crest"),
        #     "apply_to": ("field_values", "absolute_error"),
        # },
        
        # One physical-value map per available field. These assignments apply
        # identically when a field is moved between Panel a and an SI panel.
        # They remain as portable Matplotlib fallbacks when ``field_values`` is
        # removed from the seaborn ``apply_to`` tuple above.
        "field_values": {
            "CH4": "crest", # YlOrRd
            "CO": "YlGnBu",
            "T": "inferno",
            "U1": "RdBu_r", # coolwarm
            "p": "viridis",
        },
        # All absolute-error maps share one magnitude palette so error texture
        # remains directly comparable across main and supplementary fields.
        "absolute_error": "OrRd", # magma, or seaborn_spec like crest_r / flare_r
    },
    "panel_b": {
        "selected": "Reds",            # default: intuitive high-error emphasis
        "options": ("YlOrRd", "Reds", "Greys"),
    },
    "panel_d": {
        "selected": "mako_r",            # default: stronger density topology contrast
        "options": ("mako_r", "PuBu", "PuRd", "crest_r"),
        "zero_density_color": "#FFFFFF", # exact zero / under-range background
        "positive_vmin_floor": 1.0e-8,    # LogNorm must remain strictly positive
    },
}


def _configured_colormap(panel_key: str):
    """Return a validated, mutable Matplotlib or seaborn colormap."""
    if panel_key not in COLORMAP_CONFIG:
        raise KeyError(f"Unknown colormap configuration {panel_key!r}.")
    spec = COLORMAP_CONFIG[panel_key]
    selected = str(spec["selected"])
    if selected not in spec["options"]:
        raise ValueError(
            f"COLORMAP_CONFIG[{panel_key!r}]['selected']={selected!r} is not one "
            f"of the approved options {spec['options']}."
        )
    if selected in plt.colormaps:
        return plt.colormaps[selected].copy()
    try:
        # Supports seaborn-only and reversed names such as mako_r/crest_r.
        return sns.color_palette(selected, as_cmap=True).copy()
    except ValueError as exc:
        raise ValueError(
            f"Neither Matplotlib nor seaborn provides colormap {selected!r}."
        ) from exc


def _panel_a_colormap(field_key: str | None = None, *, error: bool = False):
    """Return a validated seaborn or Matplotlib Panel-a/SI colormap."""
    spec = COLORMAP_CONFIG["panel_a"]
    role = "absolute_error" if error else "field_values"
    if error:
        selected = str(spec["absolute_error"])
    else:
        if field_key not in spec["field_values"]:
            raise KeyError(
                f"No Panel-a field colormap is configured for {field_key!r}; "
                f"choose from {tuple(spec['field_values'])}."
            )
        selected = str(spec["field_values"][field_key])
    if selected in plt.colormaps:
        return plt.colormaps[selected].copy()
    try:
        # Panel a intentionally permits seaborn-only names such as
        # ``crest``, ``mako``, and their reversed ``*_r`` variants.
        return sns.color_palette(selected, as_cmap=True).copy()
    except ValueError as exc:
        raise ValueError(
            f"Neither Matplotlib nor seaborn provides Panel-a colormap {selected!r}."
        ) from exc

# ---------------------------------------------------------------------------
# PANEL-A MANUAL LAYOUT CONTROLS
# ---------------------------------------------------------------------------
# Tune these values while iterating only Panel a with `--panel a`.
#
# Header locations are offsets in *figure coordinates*, measured relative to
# the top edge of the first contour-grid cell:
#   - more positive `*_y_offset` moves text upward;
#   - more negative values move it downward, toward the contour images.
# `header_band_ratio` reserves a dedicated header strip above that grid.  A
# larger value compresses every contour row vertically and gives the two text
# rows more protected space inside Panel a (rather than above the page).
#
# Grid spacing uses Matplotlib GridSpec units:
#   - smaller `grid_wspace` makes contour columns wider;
#   - `prediction_error_hspace` tightly couples each prediction/error pair;
#   - `field_group_spacer_ratio` separates successive physical variables.
# The panel itself keeps the same allocated height; only internal geometry
# changes when these controls are adjusted.
# `standalone_bottom_margin` is a figure-height fraction.  Keep it <= 0.05 so
# Panel-A-only exports do not retain an oversized bottom white band.
#
# Colourbar height:
#   - `colorbar_height_fraction` is the direct Panel-A colourbar-height
#     control.  It applies independently to every reconstruction and absolute-
#     error colourbar, and centres each one vertically in its own GridSpec row.
#   - `1.00` fills the matching image-row height; `0.85` leaves a modest,
#     balanced gap above and below; `0.70` makes the bar visibly shorter.
#   - This parameter changes only the bar height, not its mapped data range,
#     width, tick locations, or the contour axes.
#   - `colorbar_width_ratio` controls thickness separately.  Decrease it for
#     a slimmer bar, or increase it if tick labels need more room.
#
# Horizontal placement:
#   - Standalone panel exports intentionally inherit the *physical panel box*
#     from the full composite.  Do not give them independent outer margins:
#     that would make the inspection panel a different size from the version
#     embedded in the publication canvas.
#   - In the full composite, set `composite_right_pad_ratio` larger than the
#     left pad to shift Panel a left without changing panels b--e.
#
# Set `subplot_width_to_height` to a positive value (for example `3.0`) to
# impose that width:height ratio on each contour box.  Use `None` to let the
# GridSpec cell geometry determine the ratio.  Ground truth automatically
# matches the reconstruction-box geometry in either mode.
#
# Optional light-grey value-contour overlay:
#   - `show_value_contour_lines` enables isolines on Ground truth and
#     Reconstruction maps only; Absolute-error maps deliberately remain
#     colour-filled without extra lines.
#   - `value_contour_line_levels` is independent of the filled-contour levels
#     in the YAML.  The default 20 lines is normally detailed enough to show
#     physical structure without competing with the small-panel annotations.
#   - Tune colour, width, or alpha here when checking Panel a.  Set the switch
#     to `False` for the previous fill-only rendering.
#
# Optional SiT comparison column:
#   - Set `qualitative.include_sit_comparison: true` in the publication layout
#     YAML to insert SiT immediately after the representative generative model
#     in the T-only comparison block.
#   - The compact values below apply only when that extra image column is
#     enabled.  The extra GridSpec column automatically makes every cloud map
#     narrower while preserving the Panel-A height; the smaller wspace and
#     header font prevent the eight-column version from becoming crowded.
#   - With the YAML flag false, the original seven-image-column geometry is
#     used exactly, retaining the current publication layout.

# Standalone-panel exports retain each major panel's exact composite drawing
# box, plus only this protective surrounding gutter for headers, panel letters,
# tick labels, and colourbar text.  These are canvas margins in inches: they
# never scale or shrink the panel artwork itself.
STANDALONE_PANEL_GUTTER_IN = {
    "left": 0.18,
    "right": 0.12,
    "bottom": 0.11,
    "top": 0.18,
}
# Narrow panels with long method labels need additional *external* left room
# when exported alone.  These overrides never alter the physical panel slot
# measured from the composite; they merely prevent cropped labels in a
# standalone PDF/SVG/PNG.
STANDALONE_PANEL_GUTTER_OVERRIDES = {
    # Panel b's deliberately padded third-row tick labels extend farther than
    # the generic canvas gutter; reserve that space outside the fixed panel.
    "b": {"left": 0.62, "bottom": 0.18},
    "c": {"left": 0.37, "bottom": 0.20},
    "d": {"left": 0.62, "right": 0.18, "bottom": 0.19, "top": 0.25},
}

PANEL_A_LAYOUT = {
    # MAIN/SI FIELD ALLOCATION AUTHORITY. Panel a is benchmark-locked to three
    # physical fields; every configured field not listed here is generated as
    # a one-field SI panel using the same renderer. Reorder this tuple to
    # reorder Panel-a rows. Current manuscript selection: CH4, p, U1; CO and T
    # are therefore assigned automatically to SI panels S1 and S2.
    "main_fields": ("CH4", "p", "U1"),

    # MASTER/standalone geometry lock.  These are the measured physical
    # dimensions of the approved standalone Panel-a benchmark.  Both the
    # composite and standalone exports use them, so lower-panel changes can no
    # longer stretch Panel a.
    "content_width_in": COMPOSITE_WIDTH_IN - COMPOSITE_MARGIN_LEFT_IN - COMPOSITE_MARGIN_RIGHT_IN,
    "content_height_in": 2.5,
    "master_top_margin_in": COMPOSITE_MARGIN_TOP_IN,
    # MAJOR-PANEL VERTICAL GAP: increase this to move Panels B/C farther below
    # Panel a.  It is a physical inch value and therefore does not scale with
    # the master canvas height.
    "master_lower_gap_in": PANEL_HSPACE_IN,

    # The Panel-a header band is computed automatically from the requested
    # group-to-column gap below.  Reducing that gap therefore pulls the column
    # titles and all map rows upward instead of leaving meaningless white space.
    # This physical offset keeps the group-title/panel-letter baseline nearly
    # flush with the Panel-a top while retaining the approved alignment.
    "group_header_baseline_above_slot_in": 0.005,
    # Minimum clearance between the tallest top text and the canvas boundary.
    # Geometry validation raises an error instead of exporting clipped text.
    "top_text_safety_in": 0.002,

    # NON-UNIFORM VERTICAL CHUNKING. Prediction/error rows are one visual pair,
    # while successive physical variables receive an explicit spacer row.
    # `prediction_error_hspace` is GridSpec hspace within/across the structured rows; 
    # `field_group_spacer_ratio` is the empty-row height relative to one reconstruction/error map row. 
    # This keeps each pair tight without merging successive configured physical fields into one undifferentiated block.
    "prediction_error_hspace": 0.075,
    "field_group_spacer_ratio": 0.25,
    "grid_wspace": 0.050,                # horizontal gap between contour columns
    "sit_enabled_grid_wspace": 0.035,    # compact horizontal gap only with optional SiT column

    # Dedicated right-margin colourbar grid: six independently padded axes
    # avoid scientific-exponent/tick collisions between neighbouring rows.
    "colorbar_width_ratio": 0.010,       # dedicated colourbar-column width / map-grid width
    "colorbar_height_fraction": 0.75,    # readable, yet shorter than the matching map row
    "colorbar_title_pad": 1.0,           # pt; multiplier is the bar's anchored title
    "colorbar_extend": "both",           # baseline benchmark; revision layouts may request rectangular bars
    "colorbar_title_loc": "center",      # baseline benchmark; right avoids multiplier/tick contact

    # CONDITIONING SENSOR GLYPHS. The white fill and translucent grey outline
    # reveal sampling locations without masking the underlying flow contours.
    "sensor_marker_size": 1.2,
    "sensor_facecolor": "white",
    "sensor_edgecolor": "#808080",
    "sensor_linewidth": 0.35,
    "sensor_alpha": 0.7,

    # L2 annotation treatment on absolute-error maps.
    "l2_text_x": 0.02,
    "l2_text_y": 0.05,
    "l2_bbox_alpha": 0.50,
    "l2_bbox_pad": 0.20,

    # PANEL-A HEADER SPACING (physical inches, identical standalone/composite):
    # - column_header_gap_above_grid_in moves Ground truth / T only / ...
    #   upward from the first map-row boundary;
    # - group_to_column_header_gap_in is the baseline-to-baseline separation
    #   between the two bold group titles and the column-title row.  The value
    #   below reproduces the approved standalone Panel-a spacing.
    "column_header_gap_above_grid_in": 0.05,
    "group_to_column_header_gap_in": 0.15,
    # Used only by non-qualitative panel letters (b/c/d).  Panel a and its SI
    # variants now share the exact physical group-header baseline instead.
    "panel_letter_y_offset": 0.004,

    "column_header_fontsize": SIZE_SUBPLOT_TITLE,
    "sit_enabled_column_header_fontsize": SIZE_SUBPLOT_TITLE,
    "group_header_fontsize": SIZE_SUBPLOT_TITLE,
    "panel_letter_fontsize": SIZE_PANEL_LABEL,

    "composite_left_pad_ratio": -0.10,    # internal left pad in complete figure
    "composite_right_pad_ratio": 0.035,   # increase this to shift Panel a left

    # Explicit benchmark map shape prevents anisotropic stretching if any
    # containing GridSpec changes later.
    "subplot_width_to_height": 2.25,
    "field_label_x": 0.0,             # more negative = field labels move left

    "show_value_contour_lines": True,     # truth/reconstruction only; False restores fill-only maps
    "value_contour_line_levels": 15,      # tunable number of light-grey isoline levels
    "value_contour_line_color": "#D0D0D0",  # light grey, field-independent for clear overlays
    "value_contour_line_width": 0.10,     # points; keep fine for small publication panels
    "value_contour_line_alpha": 0.75,     # reduce if the lines compete with the colour field

    # Vertical separator between conditioning progression and T-only methods.
    # Width is in points.  Use "--" for dashed or ":" for dotted.  The two
    # trims are fractions of the full contour-grid height removed independently
    # from the bottom and top; their sum must remain below 1.0.
    "divider_line_width": LW_DIVIDER,
    "divider_line_style": "--",
    "divider_bottom_trim": 0.0,
    "divider_top_trim": 0.0,
}

# ---------------------------------------------------------------------------
# PANEL-A/B/C DATA-LIMIT AND INTERNAL-LAYOUT CONTROLS
# ---------------------------------------------------------------------------
# These controls deliberately use relative GridSpec geometry and artist
# padding. They never move an axes to an absolute figure-coordinate position.
PANEL_A_VALUE_LIMITS = {
    # Pressure retains its physical offset: its lower colour limit is the
    # actual finite truth minimum rounded to one significant digit, rather
    # than being reset to zero like concentration-style positive channels.
    "actual_min_rounded_fields": ("p",),
    "actual_min_significant_digits": 1,
}

PANEL_B_LAYOUT = {
    # Dimensionless vertical spacing between the three condition heatmaps.
    "condition_hspace": INTRA_PANEL_HSPACE,
    # Points of padding between the third heatmap's x tick labels and axes.
    "bottom_xtick_label_pad": 4.0,
    # In the composite, reuse Panel a's exact figure-coordinate label x value.
    # Standalone Panel b remains positioned relative to its own content box.
    "align_panel_label_x_with_panel_a": True,
}

PANEL_C_LAYOUT = {
    # Borderless shared legend fills the dedicated row from edge to edge.
    # Matplotlib's expand mode justifies the four columns horizontally while
    # retaining two readable rows for the eight model names.
    "legend_bbox_to_anchor": (0.0, 0.04, 1.0, 0.88),
    "legend_ncol": 4,
    "legend_mode": "expand",
    # Dimensionless GridSpec hspace shared by legend/bar and bar/spectrum gaps.
    # The outer Panel-c slot remains capped by the global 40% allocation.
    "grid_hspace": 0.25,
    # A shallow legend row followed by strict equal-height evidence rows.
    "height_ratios": PANEL_C_INTERNAL_HEIGHT_RATIOS,
    # Scatter-only visual trim. Bars, confidence intervals, and every summary
    # statistic continue to use the complete finalized dataset.
    "scatter_upper_percentile": 90.0,
}


def _panel_c_field_keys(layout: dict) -> list[str]:
    """Return one validated paired bar/spectrum field selection for Panel c."""
    spectral = layout["spectral"]
    options = spectral.get("field_options")
    if options is None:
        # Compatibility with older layout files that directly specified fields.
        fields = list(spectral.get("fields", []))
    else:
        selection = spectral.get("field_selection", "main")
        if selection not in options:
            raise KeyError(
                f"Unknown Panel-c field selection {selection!r}; choose from {list(options)}."
            )
        fields = list(options[selection])
    if len(fields) != 3 or len(set(fields)) != 3:
        raise ValueError("Panel c requires exactly three unique fields.")
    return fields


def _panel_c_scatter_values(values: list[float]) -> np.ndarray:
    """Return values below the visual-only upper percentile cutoff."""
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if finite.size == 0:
        return finite
    percentile = float(PANEL_C_LAYOUT["scatter_upper_percentile"])
    if not 0.0 < percentile <= 100.0:
        raise ValueError("Panel-c scatter percentile must be in (0, 100].")
    if percentile == 100.0:
        return finite
    cutoff = float(np.percentile(finite, percentile))
    trimmed = finite[finite < cutoff]
    # Degenerate equal-valued inputs should remain visible rather than
    # disappearing because every value equals the percentile threshold.
    return trimmed if trimmed.size else finite


# ---------------------------------------------------------------------------
# QUALITATIVE SUPPLEMENTARY PANELS (same renderer and styling as Panel a)
# ---------------------------------------------------------------------------
# CLI keys are generated from the complement of ``PANEL_A_LAYOUT['main_fields']``.
# Every SI panel preserves Panel a's physical map-row height, headers,
# colourbars, contour overlays, method order, margins, and fixed map aspect.
def _qualitative_si_panel_registry() -> dict[str, dict[str, str]]:
    all_fields = tuple(COLORMAP_CONFIG["panel_a"]["field_values"])
    main_fields = tuple(PANEL_A_LAYOUT["main_fields"])
    if len(main_fields) != 3 or len(set(main_fields)) != 3:
        raise ValueError("PANEL_A_LAYOUT['main_fields'] must contain exactly three unique fields.")
    unknown = [field for field in main_fields if field not in all_fields]
    if unknown:
        raise ValueError(
            f"Unknown Panel-a fields {unknown}; configure each field in "
            "COLORMAP_CONFIG['panel_a']['field_values']."
        )
    supplementary = [field for field in all_fields if field not in main_fields]
    return {
        f"si-{field.lower()}": {
            "field": field,
            "panel_label": f"S{index}",
            "output_directory": f"Panel_SI_{field}_Qualitative",
            "output_name": f"Panel_SI_{field}_Qualitative",
        }
        for index, field in enumerate(supplementary, start=1)
    }


QUALITATIVE_SI_PANELS = _qualitative_si_panel_registry()

# ---------------------------------------------------------------------------
# PANEL-D / SI SUBPLOT SELECTION API (EDIT HERE)
# ---------------------------------------------------------------------------
# This is the only block you normally need to edit to choose the nine possible
# coupling plots.  Each item is one explicit ``{coupling pair, condition}``
# selection.  Reorder items to reorder columns; exchange any pair/condition to
# move a validated result between the main manuscript and either SI panel.
#
# Valid pairs:      "T-U1", "CH4-U1", "CO-U1", "CO-T", "U1-p", "p-U1"
# Valid conditions: "Cond_T", "Cond_TU1", "Cond_COTU1P"
#
# The main publication slot is structurally locked to three columns so it
# remains vertically aligned with Panel C.  SI panels can contain one to three
# selections; the renderer creates the required number of columns automatically.
COUPLING_PAIR_LABELS = {
    "T-U1": "$T$–$U_1$\nthermal–flow",
    "CH4-T": "CH$_4$–$T$\nthermal–chemistry",
    "CH4-U1": "CH$_4$–$U_1$\nchemistry–flow",
    "CO-U1": "CO–$U_1$\nchemistry–flow",
    "CO-T": "CO–$T$\nthermal–chemistry",
    "U1-p": "$U_1$–$p$\nFlow field consistency",
    "p-U1": "$p$–$U_1$\nFlow field consistency",
}
# Compact labels used only by the main Panel-d header.  Keeping this mapping
# separate preserves the descriptive two-line titles used by standalone/SI
# violin figures while removing their unnecessary title band in the composite.
COUPLING_PAIR_MATH_LABELS = {
    "T-U1": "$T$–$U_1$",
    "CH4-T": "CH$_4$–$T$",
    "CH4-U1": "CH$_4$–$U_1$",
    "CO-U1": "CO–$U_1$",
    "CO-T": "CO–$T$",
    "U1-p": "$U_1$–$p$",
    "p-U1": "$p$–$U_1$",
}
# Change only ``selected`` to restore the previous CO–U1 middle column.  Both
# the ground-truth joint PDF and matching JSD violin read this single choice.
PANEL_D_MIDDLE_PAIR_OPTION = {
    "selected": "CH4-U1",
    "backup": "CO-U1",
}
PANEL_D_FIGURE_SELECTIONS = {
    "main": {
        "panel_letter": "d",
        "output_tag": "Main",
        "subplots": [
            {"pair": "T-U1", "condition": "Cond_T"},
            {"pair": PANEL_D_MIDDLE_PAIR_OPTION["selected"], "condition": "Cond_T"},
            {"pair": "p-U1", "metric_pair": "U1-p", "condition": "Cond_T"},
        ],
    },
    "si_1": {
        "panel_letter": "S1",
        "output_tag": "SI1",
        "subplots": [
            {"pair": "T-U1", "condition": "Cond_TU1"},
            {"pair": "CO-U1", "condition": "Cond_TU1"},
            {"pair": "U1-p", "condition": "Cond_TU1"},
        ],
    },
    "si_2": {
        "panel_letter": "S2",
        "output_tag": "SI2",
        "subplots": [
            {"pair": "T-U1", "condition": "Cond_COTU1P"},
            {"pair": "CO-U1", "condition": "Cond_COTU1P"},
            {"pair": "U1-p", "condition": "Cond_COTU1P"},
        ],
    },
}

# Fine controls for every selected Panel-D violin subplot.  Means are drawn
# beside their model distributions using exactly two significant figures.
PANEL_D_STYLE = {
    # Two-row architecture: compact ground-truth PDFs above tall log-JSD
    # violins. Coupling titles are figure-level text aligned with panel label
    # "d", so they consume no GridSpec row. The PDF height is a tunable
    # fraction of physical column width and all remainder goes to the violins.
    "row_hspace": 0.175,
    "pdf_height_scale": 0.70,  # physical PDF height / physical PDF width
    "pdf_bins": 64,
    "pdf_quantiles": (0.005, 0.995),
    # Load exactly this many cached truth frames, selected deterministically
    # across the full available snapshot range, before computing each PDF.
    "pdf_ensemble_frame_count": 25,
    "pdf_axis_labelpad": 6.0,
    "violin_log_lower_factor": 0.75,
    "annotate_means": True,
    "mean_prefix": "μ=",
    "mean_fontsize": SIZE_ANNOTATION,
    "mean_color": "#303030",
    "dmf_first_at_top": False,
    # On the final log axis, all KDE support ends by this axes fraction and μ
    # begins at the next fraction. This guarantees a true empty text column;
    # a simple multiplicative xmax is visually unreliable on logarithmic axes.
    "violin_data_right_fraction": 0.70,
    "mean_x_axes": 0.73,
    "mean_xlim_min_factor": 1.20,
    # Preserve the approved density landmarks irrespective of the annotation
    # reserve. Add a mapping entry when introducing a new coupling pair.
    "retained_ticks": {
        "T-U1": (0.2, 0.4, 0.6),
        "CH4-T": (0.2, 0.4, 0.6),
        "CH4-U1": (0.2, 0.4, 0.6),
        "CO-T": (0.2, 0.4, 0.6),
        "CO-U1": (0.2, 0.4, 0.6),
        "U1-p": (0.5, 1.0),
        "p-U1": (0.5, 1.0),
    },
    "violin_width": 0.65,
    "condition_xlabels": {
        "Cond_T": "JSD of joint PDF, all conditioned on 256 T sensors only",
    },
    "x_label_template": "JSD of joint PDF, all conditioned on {condition_label} sensors",
    "x_label_fontsize": SIZE_AXIS_LABEL,
    "x_label_pad": 0.20,
}

# CH4--U1 is computed transiently from read-only reconstruction caches because
# the established metric tables do not contain this pair.
PANEL_D_GPU_ON_THE_FLY_PAIRS = {"CH4-U1"}


def _apply_revision_overrides(layout: dict) -> None:
    """Apply narrowly scoped layout-driven typography and gap revisions.

    Defaults remain the exact baseline contract; dedicated revision YAML files
    opt in without making earlier reproduction configurations drift.
    """
    typography = dict(layout.get("typography_overrides_pt", {}))
    font_names = {
        "size_panel_label": "SIZE_PANEL_LABEL",
        "size_subplot_title": "SIZE_SUBPLOT_TITLE",
        "size_axis_label": "SIZE_AXIS_LABEL",
        "size_tick_label": "SIZE_TICK_LABEL",
        "size_legend": "SIZE_LEGEND",
        "size_annotation": "SIZE_ANNOTATION",
    }
    unsupported = set(typography) - set(font_names)
    if unsupported:
        raise ValueError("Unsupported typography_overrides_pt: " + ", ".join(sorted(unsupported)))
    for key, global_name in font_names.items():
        if key not in typography:
            continue
        value = float(typography[key])
        if value <= 0.0:
            raise ValueError(f"{key} must be positive.")
        globals()[global_name] = value
        setattr(global_style_contract, global_name, value)
        FONT_SIZES[key] = value
        global_style_contract.FONT_SIZES[key] = value
    if typography:
        hierarchy = [
            SIZE_PANEL_LABEL,
            SIZE_SUBPLOT_TITLE,
            SIZE_AXIS_LABEL,
            SIZE_TICK_LABEL,
            SIZE_ANNOTATION,
        ]
        if any(left <= right for left, right in zip(hierarchy, hierarchy[1:])):
            raise ValueError(
                "Typography hierarchy must satisfy panel label > subplot title > "
                "axis label > tick label > annotation."
            )
        if not SIZE_SUBPLOT_TITLE > SIZE_LEGEND > SIZE_ANNOTATION:
            raise ValueError("Legend size must remain between subplot titles and annotations.")
        PANEL_A_LAYOUT.update({
            "column_header_fontsize": SIZE_SUBPLOT_TITLE,
            "sit_enabled_column_header_fontsize": SIZE_SUBPLOT_TITLE,
            "group_header_fontsize": SIZE_SUBPLOT_TITLE,
            "panel_letter_fontsize": SIZE_PANEL_LABEL,
        })
        PANEL_D_STYLE.update({
            "mean_fontsize": SIZE_ANNOTATION,
            "x_label_fontsize": SIZE_AXIS_LABEL,
        })

    composite = dict(layout.get("composite_layout_overrides", {}))
    unsupported = set(composite) - {
        "panel_wspace_in", "bottom_margin_in", "composite_margin_left_in",
        "composite_margin_right_in", "panel_c_d_gap_in",
    }
    if unsupported:
        raise ValueError("Unsupported composite_layout_overrides: " + ", ".join(sorted(unsupported)))
    if "panel_wspace_in" in composite:
        value = float(composite["panel_wspace_in"])
        if value <= 0.0:
            raise ValueError("panel_wspace_in must be positive.")
        globals()["PANEL_WSPACE_IN"] = value
    if "bottom_margin_in" in composite:
        value = float(composite["bottom_margin_in"])
        if value <= 0.0:
            raise ValueError("bottom_margin_in must be positive.")
        globals()["COMPOSITE_MARGIN_BOTTOM_IN"] = value
        global_style_contract.COMPOSITE_MARGIN_BOTTOM_IN = value
    for key, global_name in (
        ("composite_margin_left_in", "COMPOSITE_MARGIN_LEFT_IN"),
        ("composite_margin_right_in", "COMPOSITE_MARGIN_RIGHT_IN"),
        ("panel_c_d_gap_in", "PANEL_C_D_GAP_IN"),
    ):
        if key not in composite:
            continue
        value = float(composite[key])
        if value < 0.0:
            raise ValueError(f"{key} must be non-negative.")
        globals()[global_name] = value
        setattr(global_style_contract, global_name, value)
    PANEL_A_LAYOUT["content_width_in"] = (
        COMPOSITE_WIDTH_IN - COMPOSITE_MARGIN_LEFT_IN - COMPOSITE_MARGIN_RIGHT_IN
    )

    panel_b = dict(layout.get("panel_b_layout_overrides", {}))
    unsupported = set(panel_b) - {"condition_hspace"}
    if unsupported:
        raise ValueError("Unsupported panel_b_layout_overrides: " + ", ".join(sorted(unsupported)))
    if "condition_hspace" in panel_b:
        value = float(panel_b["condition_hspace"])
        if value < 0.0:
            raise ValueError("Panel-b condition_hspace must be non-negative.")
        PANEL_B_LAYOUT["condition_hspace"] = value

    panel_c = dict(layout.get("panel_c_layout_overrides", {}))
    unsupported = set(panel_c) - {"grid_hspace"}
    if unsupported:
        raise ValueError("Unsupported panel_c_layout_overrides: " + ", ".join(sorted(unsupported)))
    if "grid_hspace" in panel_c:
        value = float(panel_c["grid_hspace"])
        if value < 0.0:
            raise ValueError("Panel-c grid_hspace must be non-negative.")
        PANEL_C_LAYOUT["grid_hspace"] = value

    panel_d = dict(layout.get("panel_d_layout_overrides", {}))
    unsupported = set(panel_d) - {
        "x_label_pad", "row_hspace", "cond_t_xlabel", "dmf_first_at_top",
    }
    if unsupported:
        raise ValueError("Unsupported panel_d_layout_overrides: " + ", ".join(sorted(unsupported)))
    if "x_label_pad" in panel_d:
        value = float(panel_d["x_label_pad"])
        if value < 0.0:
            raise ValueError("Panel-d x_label_pad must be non-negative.")
        PANEL_D_STYLE["x_label_pad"] = value
    if "row_hspace" in panel_d:
        value = float(panel_d["row_hspace"])
        if value < 0.0:
            raise ValueError("Panel-d row_hspace must be non-negative.")
        PANEL_D_STYLE["row_hspace"] = value
    if "cond_t_xlabel" in panel_d:
        PANEL_D_STYLE["condition_xlabels"]["Cond_T"] = str(panel_d["cond_t_xlabel"])
    if "dmf_first_at_top" in panel_d:
        PANEL_D_STYLE["dmf_first_at_top"] = bool(panel_d["dmf_first_at_top"])


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _csv_path(folder: Path, prefix: str, rid_arg: str | None) -> Path:
    exact = folder / f"{prefix}_{rid_arg}.csv" if rid_arg else None
    return exact if exact and exact.exists() else latest(folder, prefix, "csv")


def _field_lookup(cfg: dict) -> dict[str, dict]:
    return {field["key"]: field for field in cfg["fields"]}


def _method_lookup(cfg: dict) -> dict[str, dict]:
    return {method["name"]: method for method in method_items(cfg, None)}


def _clean_axis(ax, *, ticks: bool = False) -> None:
    if not ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(LW_DIVIDER)


def _panel_a_target_box(reference_box, fig) -> tuple[float, float]:
    """Return a contour-box size fitted inside a GridSpec cell.

    ``subplot_width_to_height`` is intentionally expressed as width/height,
    which is easier to tune visually than Matplotlib's height/width box aspect.
    """
    ratio = PANEL_A_LAYOUT["subplot_width_to_height"]
    if ratio is None:
        return reference_box.width, reference_box.height
    ratio = float(ratio)
    if ratio <= 0:
        raise ValueError("PANEL_A_LAYOUT['subplot_width_to_height'] must be positive or None.")
    fig_width, fig_height = (float(value) for value in fig.get_size_inches())
    physical_ratio = (reference_box.width * fig_width) / (reference_box.height * fig_height)
    if physical_ratio >= ratio:
        height = reference_box.height
        return ratio * height * fig_height / fig_width, height
    width = reference_box.width
    return width, width * fig_width / (ratio * fig_height)


def _qualitative_header_height_in() -> float:
    """Return the compact physical header-band height for Panel a.

    The group-title baseline stays at a fixed physical offset above the panel
    slot.  Consequently, decreasing the group-to-column gap decreases this
    reserved band by the same amount and shifts every lower artist upward.
    """
    column_gap = float(PANEL_A_LAYOUT["column_header_gap_above_grid_in"])
    header_gap = float(PANEL_A_LAYOUT["group_to_column_header_gap_in"])
    baseline_above_slot = float(PANEL_A_LAYOUT["group_header_baseline_above_slot_in"])
    header_height = column_gap + header_gap - baseline_above_slot
    if header_height <= 0.0:
        raise ValueError(
            "column_header_gap_above_grid_in + group_to_column_header_gap_in "
            "must exceed group_header_baseline_above_slot_in."
        )
    return header_height


def _qualitative_row_structure(field_count: int, pair_ratios: list[float]) -> tuple[list[float], list[tuple[int, int]]]:
    """Return non-uniform GridSpec ratios and reconstruction/error row pairs."""
    if field_count < 1 or len(pair_ratios) != 2 or any(float(value) <= 0 for value in pair_ratios):
        raise ValueError("Qualitative rows require fields and two positive prediction/error ratios.")
    spacer = float(PANEL_A_LAYOUT["field_group_spacer_ratio"])
    if spacer < 0:
        raise ValueError("PANEL_A_LAYOUT['field_group_spacer_ratio'] must be non-negative.")
    ratios: list[float] = []
    row_pairs: list[tuple[int, int]] = []
    for field_index in range(field_count):
        rec_row = len(ratios)
        ratios.extend(float(value) for value in pair_ratios)
        row_pairs.append((rec_row, rec_row + 1))
        if field_index < field_count - 1:
            ratios.append(spacer)
    return ratios, row_pairs


def _format_l2_mathtext(value: float) -> str:
    """Format L2 as clean manuscript scientific notation, never console e-notation."""
    value = float(value)
    if not np.isfinite(value):
        return r"$L_2 = \mathrm{NaN}$"
    if value == 0.0:
        return r"$L_2 = 0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    return rf"$L_2 = {mantissa:.2f} \times 10^{{{exponent}}}$"


def _center_panel_a_axes(ax, container_box, reference_box) -> None:
    """Center one manually sized Panel-A axes inside its allocated cell."""
    width, height = _panel_a_target_box(reference_box, ax.figure)
    x0 = container_box.x0 + (container_box.width - width) / 2.0
    y0 = container_box.y0 + (container_box.height - height) / 2.0
    ax.set_position([x0, y0, width, height], which="both")
    ax.set_anchor("C")


def _shorten_panel_a_colorbar(cax, grid_cell) -> None:
    """Center a Panel-A colourbar at a configurable fraction of its row height."""
    fraction = float(PANEL_A_LAYOUT["colorbar_height_fraction"])
    if not 0.0 < fraction <= 1.0:
        raise ValueError("PANEL_A_LAYOUT['colorbar_height_fraction'] must be in (0, 1].")
    box = grid_cell.get_position(cax.figure)
    height = box.height * fraction
    y0 = box.y0 + (box.height - height) / 2.0
    cax.set_position([box.x0, y0, box.width, height], which="both")


def _panel_letter(fig, slot, label: str, *, x: float | None = None,
                  y: float | None = None):
    """Place a lowercase panel letter relative to a major GridSpec bound."""
    box = slot.get_position(fig)
    letter_x = box.x0 - 0.020 if x is None else float(x)
    letter_y = box.y1 + PANEL_A_LAYOUT["panel_letter_y_offset"] if y is None else float(y)
    artist = fig.text(
        letter_x, letter_y, label,
        fontsize=float(PANEL_A_LAYOUT["panel_letter_fontsize"]),
        fontweight="bold", ha="left", va="bottom",
    )
    registry = getattr(fig, "_panel_letter_artists", {})
    registry[label] = artist
    fig._panel_letter_artists = registry  # type: ignore[attr-defined]
    return artist


def _load_qualitative_payloads(cfg: dict, rid: str, layout: dict) -> tuple[dict, dict, dict]:
    """Load selected cache arrays and choose available displayed methods."""
    q = layout["qualitative"]
    manifest_path = RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{rid}.csv"
    manifest = read_csv(manifest_path)
    snapshot = int(q["snapshot_index"])
    by_key = {(row["method"], row["condition"], int(row["snapshot"])): row for row in manifest}

    def available(name: str, condition: str = "Cond_T") -> bool:
        row = by_key.get((name, condition, snapshot), {})
        return row.get("status") == "ok" and Path(row.get("cache_path", "")).is_file()

    proposed = q["proposed_model"]
    if not available(proposed):
        raise RuntimeError(f"Required qualitative model {proposed!r} has no cache at snapshot {snapshot}.")
    generative = next((name for name in q["generative_preference"] if available(name)), None)
    deterministic = next((name for name in q["deterministic_preference"] if available(name)), None)
    if generative is None:
        generative = proposed
    if deterministic is None:
        deterministic = proposed
    # FFM-FNO occupies a fixed comparison slot even when its cache is absent.
    # Keeping it in the payload request lets the panel distinguish a genuine
    # missing result from a layout omission.  SiT is an optional, fixed
    # T-only comparison immediately after the selected generative model.  Do
    # not duplicate it if it is already the generative fallback.
    sit_enabled = bool(q.get("include_sit_comparison", False))
    comparison_headers = ["FFM-FNO", generative]
    if sit_enabled and "SiT" not in comparison_headers:
        comparison_headers.append("SiT")
    if deterministic not in comparison_headers:
        comparison_headers.append(deterministic)
    comparison_methods = [proposed, *comparison_headers]
    payloads: dict[tuple[str, str], tuple[dict, dict]] = {}
    cache_paths: dict[str, str] = {}
    for method in dict.fromkeys(comparison_methods):
        for condition in q["conditions"]:
            row = by_key.get((method, condition, snapshot), {})
            if row.get("status") != "ok" or not Path(row.get("cache_path", "")).is_file():
                continue
            try:
                payloads[(method, condition)] = load_cache(Path(row["cache_path"]))
                cache_paths[f"{method}|{condition}"] = row["cache_path"]
            except Exception:
                continue
    selection = {"proposed": proposed, "generative": generative, "deterministic": deterministic,
                 "comparison_methods": comparison_headers, "sit_comparison_enabled": sit_enabled,
                 "manifest": str(manifest_path), "snapshot_index": snapshot}
    return payloads, selection, cache_paths


def _load_panel_d_truth_ensemble(rid: str, proposed_model: str,
                                 subplot_specs: list[dict]) -> tuple[dict, dict]:
    """Load 25 evenly distributed cached truth frames for Panel-d PDFs.

    This is deliberately cache-only: no checkpoint inference or metric
    recomputation occurs during figure assembly. For each required condition,
    valid snapshots are sorted, 25 indices are spaced across the full range,
    and only ``truth_phys`` is retained from each NPZ payload.
    """
    manifest_path = RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{rid}.csv"
    manifest = read_csv(manifest_path)
    frame_count = int(PANEL_D_STYLE["pdf_ensemble_frame_count"])
    ensembles: dict[str, list[dict]] = {}
    selected_by_condition: dict[str, list[int]] = {}
    for condition in dict.fromkeys(spec["condition"] for spec in subplot_specs):
        candidates = sorted(
            (
                row for row in manifest
                if row.get("method") == proposed_model
                and row.get("condition") == condition
                and row.get("status") == "ok"
                and Path(row.get("cache_path", "")).is_file()
            ),
            key=lambda row: int(row["snapshot"]),
        )
        if len(candidates) < frame_count:
            raise RuntimeError(
                f"Panel d requires {frame_count} cached truth frames for "
                f"{proposed_model}/{condition}, but only {len(candidates)} are available."
            )
        # With candidate_count >= frame_count, these integer indices are
        # unique and span the first through last available dataset frames.
        indices = np.linspace(0, len(candidates) - 1, frame_count, dtype=int)
        selected = [candidates[int(index)] for index in indices]
        frames = []
        for row in selected:
            arrays, _ = load_cache(Path(row["cache_path"]))
            truth = np.asarray(arrays.get("truth_phys"), dtype=np.float32)
            if truth.ndim != 2 or truth.shape[0] == 0:
                raise RuntimeError(
                    f"Invalid truth_phys cache for snapshot {row['snapshot']} "
                    f"({proposed_model}/{condition})."
                )
            frames.append({"snapshot": int(row["snapshot"]), "truth_phys": truth})
        ensembles[condition] = frames
        selected_by_condition[condition] = [frame["snapshot"] for frame in frames]
    metadata = {
        "manifest": str(manifest_path),
        "selection": "deterministic_even_spacing_across_available_snapshots",
        "frame_count": frame_count,
        "snapshots_by_condition": selected_by_condition,
    }
    return ensembles, metadata


def _qualitative_limits(payloads: dict, fields: list[dict], value_percentiles: list[float],
                        error_percentile: float) -> tuple[dict, dict]:
    """Return finite, percentile-clipped physical-value and error bounds.

    The qualitative plate must be robust to malformed prediction values and
    isolated generative outliers.  Truth limits are therefore computed only
    from sanitized physical truth values, while error limits pool sanitized
    absolute errors across the displayed comparisons.  No raw extrema are
    used for rendering bounds.
    """
    if len(value_percentiles) != 2 or not 0 <= value_percentiles[0] < value_percentiles[1] <= 100:
        raise ValueError("qualitative.value_percentiles must be two increasing values in [0, 100].")
    values, errors, actual_truth_minima = defaultdict(list), defaultdict(list), defaultdict(list)
    for arrays, _ in payloads.values():
        for field in fields:
            index = field["index"]
            raw_truth = np.asarray(arrays["truth_phys"][:, index], dtype=float)
            finite_truth = raw_truth[np.isfinite(raw_truth)]
            if finite_truth.size:
                actual_truth_minima[field["key"]].append(float(np.min(finite_truth)))
            truth = np.nan_to_num(raw_truth, nan=0.0, posinf=0.0, neginf=0.0)
            reconstruction = np.nan_to_num(np.asarray(arrays["recon_phys"][:, index], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
            values[field["key"]].append(truth)
            errors[field["key"]].append(np.abs(reconstruction - truth))
    field_limits, error_limits = {}, {}
    for field in fields:
        key = field["key"]
        pooled = np.concatenate(values[key]) if values[key] else np.array([0.0, 1.0])
        lo, hi = np.percentile(pooled, value_percentiles)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            center = float(np.median(pooled))
            padding = max(abs(center) * 0.05, 1.0e-12)
            lo, hi = center - padding, center + padding
        if key in PANEL_A_VALUE_LIMITS["actual_min_rounded_fields"] and actual_truth_minima[key]:
            actual_minimum = min(actual_truth_minima[key])
            digits = int(PANEL_A_VALUE_LIMITS["actual_min_significant_digits"])
            if actual_minimum == 0:
                lo = 0.0
            else:
                decimal_places = digits - 1 - int(np.floor(np.log10(abs(actual_minimum))))
                lo = float(np.round(actual_minimum, decimal_places))
        # Other positive physical channels retain a zero-origin display range.
        elif lo >= 0:
            lo = 0.0
        field_limits[key] = [float(lo), float(hi)]
        pooled_error = np.concatenate(errors[key]) if errors[key] else np.array([1.0])
        high = float(np.percentile(pooled_error, error_percentile))
        error_limits[key] = [0.0, high if np.isfinite(high) and high > 0 else 1.0e-12]
    return field_limits, error_limits


def _compact_colorbar(fig, cax, *, limits: list[float], cmap: str, label: str | None = None, signed: bool = False) -> None:
    """Draw a narrow, non-overlapping two-tick scientific colourbar.

    The visual labels are intentionally rounded for page-scale legibility.  The
    mapped limits themselves remain the exact physical limits stored in the
    source manifest.
    """
    lo, hi = map(float, limits)
    magnitude = max(abs(lo), abs(hi), np.finfo(float).tiny)
    exponent = int(np.floor(np.log10(magnitude)))
    scale = 10.0 ** exponent
    if signed and lo < 0:
        ticks = [lo, hi]
        tick_labels = [f"{lo / scale:.2g}", f"{hi / scale:.2g}"]
    else:
        # A .5/integer ceiling keeps the displayed upper bound compact without
        # claiming that it is the exact clipped physical maximum.
        rounded_hi = max(0.5, np.ceil((hi / scale) * 2.0) / 2.0)
        display_hi = rounded_hi * scale
        # Locate the upper label at the mapped maximum, but show the compact
        # rounded display level requested for the publication layout.
        ticks = [0.0, hi]
        tick_labels = ["0.0", f"{display_hi / scale:.2g}"]
    scalar = plt.cm.ScalarMappable(norm=Normalize(lo, hi), cmap=cmap)
    cb = fig.colorbar(
        scalar, cax=cax, ticks=ticks,
        extend=str(PANEL_A_LAYOUT["colorbar_extend"]),
    )
    cb.ax.set_yticklabels(tick_labels)
    cb.ax.tick_params(labelsize=SIZE_TICK_LABEL, length=1.15, pad=.7)
    # Bind the multiplier to the colorbar itself as its title. This is more
    # rigorous than a free figure-coordinate label and cannot be mistaken for
    # an exponent belonging to a neighboring subplot or physical variable.
    cb.ax.set_title(
        rf"$\times 10^{{{exponent}}}$", fontsize=SIZE_ANNOTATION,
        pad=float(PANEL_A_LAYOUT["colorbar_title_pad"]),
        loc=str(PANEL_A_LAYOUT["colorbar_title_loc"]),
    )
    if label:
        cb.ax.text(.5, -.075, label, transform=cb.ax.transAxes, ha="center", va="top", fontsize=SIZE_AXIS_LABEL)


def _draw_cloud(ax, arrays: dict | None, field: dict, condition: str, kind: str, *,
                cfg: dict, x_compression: float, value_limits: dict, error_limits: dict,
                levels: int, show_sensors: bool, show_l2: bool, aspect: str = "equal") -> Any | None:
    """Draw one physical-field cloud, optionally with value isolines.

    The independently tunable light-grey isolines are limited to truth and
    reconstruction maps.  They use the same clipped physical limits as the
    filled map so all methods in a field share identical level positions.
    """
    if arrays is None:
        mark_missing(ax, "Missing", cfg)
        return None
    index = field["index"]
    xy = np.nan_to_num(np.asarray(arrays["coords_phys"][:, :2], dtype=float), nan=0.0, posinf=0.0, neginf=0.0).copy()
    xy[:, 0] = np.nanmin(xy[:, 0]) + x_compression * (xy[:, 0] - np.nanmin(xy[:, 0]))
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
    # Sanitise only the rendered/annotated arrays.  The cache itself remains
    # untouched, while NaN/Inf defects and extreme outliers cannot create
    # white triangulation artefacts in the publication raster.
    truth = np.nan_to_num(np.asarray(arrays["truth_phys"][:, index], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    reconstruction = np.nan_to_num(np.asarray(arrays["recon_phys"][:, index], dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if kind == "truth":
        data, cmap, limits = truth, field["cmap"], value_limits[field["key"]]
    elif kind == "reconstruction":
        data, cmap, limits = reconstruction, field["cmap"], value_limits[field["key"]]
    else:
        data, cmap, limits = (
            np.abs(reconstruction - truth), _panel_a_colormap(error=True), error_limits[field["key"]]
        )
    lo, hi = limits
    contour = ax.tricontourf(
        tri, data, levels=np.linspace(lo, hi, max(2, levels)), cmap=cmap,
        norm=Normalize(vmin=lo, vmax=hi), extend="both",
    )
    # The triangulated clouds are the only dense image artists.  Rasterizing
    # them at export DPI keeps PDF/SVG practical while all labels and axes stay
    # editable.  Matplotlib emits a harmless version-specific warning here.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Rasterization of .* will be ignored")
        contour.set_rasterized(True)
    # Keep the value-field overlays vector and deliberately omit them from
    # absolute-error panels: the error palette already carries the relevant
    # magnitude information and the extra isolines would reduce readability.
    if (kind in {"truth", "reconstruction"}
            and PANEL_A_LAYOUT["show_value_contour_lines"]
            and np.ptp(data) > np.finfo(float).eps):
        line_levels = int(PANEL_A_LAYOUT["value_contour_line_levels"])
        if line_levels < 2:
            raise ValueError("PANEL_A_LAYOUT['value_contour_line_levels'] must be at least 2.")
        ax.tricontour(
            tri, data, levels=np.linspace(lo, hi, line_levels),
            colors=PANEL_A_LAYOUT["value_contour_line_color"],
            linewidths=float(PANEL_A_LAYOUT["value_contour_line_width"]),
            alpha=float(PANEL_A_LAYOUT["value_contour_line_alpha"]),
            zorder=3.2,
        )
    if show_sensors and index in cfg["conditions"][condition]["cond_fields"]:
        observed = arrays["obs_indices"][arrays["obs_field_ids"] == index]
        ax.scatter(
            xy[observed, 0], xy[observed, 1],
            s=float(PANEL_A_LAYOUT["sensor_marker_size"]), marker="o",
            facecolors=PANEL_A_LAYOUT["sensor_facecolor"],
            edgecolors=PANEL_A_LAYOUT["sensor_edgecolor"],
            linewidths=float(PANEL_A_LAYOUT["sensor_linewidth"]),
            alpha=float(PANEL_A_LAYOUT["sensor_alpha"]), zorder=4,
        )
    if show_l2 and kind == "error":
        ax.text(
            float(PANEL_A_LAYOUT["l2_text_x"]), float(PANEL_A_LAYOUT["l2_text_y"]),
            _format_l2_mathtext(relative_l2(truth, reconstruction)), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=SIZE_ANNOTATION, color="white",
            bbox={
                "facecolor": "black", "alpha": float(PANEL_A_LAYOUT["l2_bbox_alpha"]),
                "edgecolor": "none",
                "boxstyle": f"round,pad={float(PANEL_A_LAYOUT['l2_bbox_pad'])}",
            },
        )
    ax.set_aspect(aspect, adjustable="box")
    _clean_axis(ax)
    # Contour plates are frameless image evidence; remove the residual left and
    # bottom axes lines inherited from the global Cartesian-plot style.
    for spine in ax.spines.values():
        spine.set_visible(False)
    return contour


def draw_qualitative_panel(fig, slot, cfg: dict, layout: dict, payloads: dict, selection: dict,
                           value_limits: dict, error_limits: dict, *, header_offset_scale: float = 1.0) -> float:
    """Draw consolidated conditioning progression and method comparison panel a.

    Manual positions and contour-grid gaps are controlled by
    :data:`PANEL_A_LAYOUT` near the top of this module.
    """
    q = layout["qualitative"]
    fields = [_field_lookup(cfg)[key] for key in q["fields"]]
    ratios, field_row_pairs = _qualitative_row_structure(
        len(fields), list(q["reconstruction_to_error_height"]),
    )
    # In the full composite an asymmetric right pad is the fastest way to
    # shift only Panel a left.  Panel-a-only margins are handled in
    # `build_figure` below.
    left_pad = float(PANEL_A_LAYOUT["composite_left_pad_ratio"])
    right_pad = float(PANEL_A_LAYOUT["composite_right_pad_ratio"])
    content_slot = slot
    if left_pad > 0 or right_pad > 0:
        padded = slot.subgridspec(1, 3, width_ratios=[max(left_pad, 1e-9), 1.0, max(right_pad, 1e-9)], wspace=0.0)
        content_slot = padded[0, 1]

    # Reserve exactly the physical height required by the two header baselines.
    # Because this height is derived from group_to_column_header_gap_in, making
    # the headers closer automatically expands the contour region upward and
    # does not leave dead space at the top of Panel a.
    slot_height_in = slot.get_position(fig).height * float(fig.get_size_inches()[1])
    header_height_in = _qualitative_header_height_in()
    if header_height_in >= slot_height_in:
        raise ValueError("Panel-a header band is taller than the available panel slot.")
    contour_height_in = slot_height_in - header_height_in
    panel_grid = content_slot.subgridspec(
        2, 1,
        height_ratios=[header_height_in / contour_height_in, 1.0],
        hspace=0.0,
    )
    contour_slot = panel_grid[1, 0]

    # The contour maps and their colourbars use sibling GridSpecs.  This is
    # deliberately not one shared bounding box: the dedicated right-margin
    # grid gives each of the six bars its own padded vertical cell.
    condition_headers = ["Ground truth", "T only", "T + U1", "CO + T + U1 + p"]
    method_headers = selection["comparison_methods"]
    method_col_start = len(condition_headers)
    map_col_count = method_col_start + len(method_headers)
    active_wspace = (PANEL_A_LAYOUT["sit_enabled_grid_wspace"] if selection["sit_comparison_enabled"]
                     else PANEL_A_LAYOUT["grid_wspace"])
    split_grid = contour_slot.subgridspec(
        1, 2, width_ratios=[1.0, PANEL_A_LAYOUT["colorbar_width_ratio"]], wspace=.025,
    )
    map_slot, colorbar_slot = split_grid[0, 0], split_grid[0, 1]
    grid = map_slot.subgridspec(
        len(ratios), map_col_count, height_ratios=ratios,
        hspace=PANEL_A_LAYOUT["prediction_error_hspace"],
        wspace=active_wspace,
    )
    colorbar_grid = colorbar_slot.subgridspec(
        len(ratios), 1, height_ratios=ratios,
        hspace=PANEL_A_LAYOUT["prediction_error_hspace"],
    )
    header_fontsize = (PANEL_A_LAYOUT["sit_enabled_column_header_fontsize"] if selection["sit_comparison_enabled"]
                       else PANEL_A_LAYOUT["column_header_fontsize"])
    # Column headings remain attached to the contour grid.  The two group
    # titles instead share the exact figure-coordinate baseline of panel label
    # ``a``; this prevents independent title padding from drifting or clipping.
    # Define both header rows in physical inches from one common map-grid
    # anchor.  This removes the former figure-height-dependent drift between
    # standalone and composite exports.  ``header_offset_scale`` remains in
    # the signature for backward-compatible callers but is intentionally not
    # used by the physical-distance layout.
    del header_offset_scale
    figure_height_in = float(fig.get_size_inches()[1])
    grid_top = grid[0, 0].get_position(fig).y1
    column_header_y = (
        grid_top + float(PANEL_A_LAYOUT["column_header_gap_above_grid_in"]) / figure_height_in
    )
    group_header_y = (
        column_header_y
        + float(PANEL_A_LAYOUT["group_to_column_header_gap_in"]) / figure_height_in
    )
    # The panel letter is taller than the group titles and therefore defines
    # the strict upper-bound safety requirement for this shared baseline.
    top_clearance_in = (1.0 - group_header_y) * figure_height_in
    required_clearance_in = (
        float(PANEL_A_LAYOUT["panel_letter_fontsize"]) / 72.0
        + float(PANEL_A_LAYOUT["top_text_safety_in"])
    )
    if top_clearance_in < required_clearance_in:
        raise ValueError(
            f"Panel-a top clearance is {top_clearance_in:.4f} in but at least "
            f"{required_clearance_in:.4f} in is required; increase "
            "master_top_margin_in/STANDALONE_PANEL_GUTTER_IN['top'] or reduce "
            "group_header_baseline_above_slot_in."
        )
    for col, header in enumerate(condition_headers):
        box = grid[0, col].get_position(fig)
        fig.text((box.x0 + box.x1) / 2, column_header_y, header, ha="center", va="bottom",
                 fontsize=header_fontsize)
    for col, header in enumerate(method_headers):
        box = grid[0, col + method_col_start].get_position(fig)
        fig.text((box.x0 + box.x1) / 2, column_header_y, header, ha="center", va="bottom",
                 fontsize=header_fontsize)
    # Group labels are direct; sensor overlays are self-explanatory at this scale.
    # The conditioning heading intentionally excludes Ground truth: it is the
    # exact midpoint of the three DMF-Gen conditioning columns only.
    conditioning_first = grid[0, 1].get_position(fig)
    conditioning_last = grid[0, method_col_start - 1].get_position(fig)
    right_first, right_last = grid[0, method_col_start].get_position(fig), grid[0, map_col_count - 1].get_position(fig)
    fig.text((conditioning_first.x0 + conditioning_last.x1) / 2, group_header_y, "Conditioning progression (DMF-Gen)",
             fontsize=PANEL_A_LAYOUT["group_header_fontsize"], fontweight="bold", ha="center")
    fig.text((right_first.x0 + right_last.x1) / 2, group_header_y, "Baseline comparisons: conditioned on T only",
             fontsize=PANEL_A_LAYOUT["group_header_fontsize"], fontweight="bold", ha="center")
    # A figure-coordinate artist separates the experimental conditioning
    # progression from the T-only baseline comparison without changing any
    # GridSpec cells.  It spans precisely the contour rows, not the headers.
    divider_x = (grid[0, method_col_start - 1].get_position(fig).x1 + grid[0, method_col_start].get_position(fig).x0) / 2.0
    divider_y0 = grid[len(ratios) - 1, 0].get_position(fig).y0
    # divider_y1 = grid[0, 0].get_position(fig).y1
    divider_y1 = group_header_y
    divider_bottom_trim = float(PANEL_A_LAYOUT["divider_bottom_trim"])
    divider_top_trim = float(PANEL_A_LAYOUT["divider_top_trim"])
    if (divider_bottom_trim < 0.0 or divider_top_trim < 0.0
            or divider_bottom_trim + divider_top_trim >= 1.0):
        raise ValueError(
            "Panel-a divider trims must be non-negative and sum to less than 1.0."
        )
    divider_height = divider_y1 - divider_y0
    divider_y0 += divider_height * divider_bottom_trim
    divider_y1 -= divider_height * divider_top_trim
    fig.add_artist(Line2D([divider_x, divider_x], [divider_y0, divider_y1], transform=fig.transFigure,
                          linestyle=PANEL_A_LAYOUT["divider_line_style"], color=COLOR_DIVIDER,
                          linewidth=float(PANEL_A_LAYOUT["divider_line_width"]),
                          zorder=20, clip_on=False))
    for field_index, base_field in enumerate(fields):
        field = dict(base_field)
        # Centralized field palettes apply equally to Panel a and every SI
        # panel; the YAML no longer overrides this manuscript-level identity.
        field["cmap"] = _panel_a_colormap(field["key"])
        rec_row, err_row = field_row_pairs[field_index]
        # The ground-truth axes spans the two-row container only to obtain a
        # centering region.  Its active box is then constrained to the exact
        # width:height ratio of one model reconstruction cell, so it is never
        # stretched vertically to fill both reconstruction/error sub-rows.
        truth_arrays = payloads.get((selection["proposed"], q["conditions"][0]), (None, {}))[0]
        truth_ax = fig.add_subplot(grid[rec_row:err_row + 1, 0])
        reference_box = grid[rec_row, 1].get_position(fig)
        truth_container = grid[rec_row:err_row + 1, 0].get_position(fig)
        truth_contour = _draw_cloud(truth_ax, truth_arrays, field, q["conditions"][0], "truth", cfg=cfg,
                                    x_compression=q["x_compression"], value_limits=value_limits, error_limits=error_limits,
                                    levels=q["contour_levels"], show_sensors=False, show_l2=False, aspect="auto")
        # Ground truth is a one-subplot-sized centered image inside the two-row
        # container, and therefore matches a reconstruction plot exactly.
        _center_panel_a_axes(truth_ax, truth_container, reference_box)
        truth_ax.text(PANEL_A_LAYOUT["field_label_x"], .5, field["label"], transform=truth_ax.transAxes,
                      rotation=90, ha="right", va="center", fontsize=SIZE_AXIS_LABEL)
        for col, condition in enumerate(q["conditions"], start=1):
            arrays = payloads.get((selection["proposed"], condition), (None, {}))[0]
            rec_ax, err_ax = fig.add_subplot(grid[rec_row, col]), fig.add_subplot(grid[err_row, col])
            _draw_cloud(rec_ax, arrays, field, condition, "reconstruction", cfg=cfg, x_compression=q["x_compression"],
                        value_limits=value_limits, error_limits=error_limits, levels=q["contour_levels"], show_sensors=True, show_l2=False, aspect="auto")
            _draw_cloud(err_ax, arrays, field, condition, "error", cfg=cfg, x_compression=q["x_compression"],
                        value_limits=value_limits, error_limits=error_limits, levels=q["contour_levels"], show_sensors=False, show_l2=True, aspect="auto")
            _center_panel_a_axes(rec_ax, grid[rec_row, col].get_position(fig), grid[rec_row, col].get_position(fig))
            _center_panel_a_axes(err_ax, grid[err_row, col].get_position(fig), grid[err_row, col].get_position(fig))
        for col, method in enumerate(method_headers):
            arrays = payloads.get((method, q["conditions"][0]), (None, {}))[0]
            grid_col = col + method_col_start
            rec_ax, err_ax = fig.add_subplot(grid[rec_row, grid_col]), fig.add_subplot(grid[err_row, grid_col])
            _draw_cloud(rec_ax, arrays, field, q["conditions"][0], "reconstruction", cfg=cfg, x_compression=q["x_compression"],
                        value_limits=value_limits, error_limits=error_limits, levels=q["contour_levels"], show_sensors=True, show_l2=False, aspect="auto")
            _draw_cloud(err_ax, arrays, field, q["conditions"][0], "error", cfg=cfg, x_compression=q["x_compression"],
                        value_limits=value_limits, error_limits=error_limits, levels=q["contour_levels"], show_sensors=False, show_l2=True, aspect="auto")
            _center_panel_a_axes(rec_ax, grid[rec_row, grid_col].get_position(fig), grid[rec_row, grid_col].get_position(fig))
            _center_panel_a_axes(err_ax, grid[err_row, grid_col].get_position(fig), grid[err_row, grid_col].get_position(fig))
        # Two narrow colourbars per field: the top one uses physical-value
        # limits for reconstructions; the lower one uses absolute-error limits.
        # Each is aligned strictly to its own GridSpec sub-row.
        rec_cax = fig.add_subplot(colorbar_grid[rec_row, 0])
        err_cax = fig.add_subplot(colorbar_grid[err_row, 0])
        _shorten_panel_a_colorbar(rec_cax, colorbar_grid[rec_row, 0])
        _shorten_panel_a_colorbar(err_cax, colorbar_grid[err_row, 0])
        _compact_colorbar(fig, rec_cax, limits=value_limits[field["key"]], cmap=field["cmap"],
                          signed=value_limits[field["key"]][0] < 0)
        _compact_colorbar(
            fig, err_cax, limits=error_limits[field["key"]],
            cmap=_panel_a_colormap(error=True), signed=False,
        )
    return group_header_y


def _qualitative_content_size(field_count: int, pair_ratios: list[float]) -> tuple[float, float]:
    """Return benchmark-locked qualitative content dimensions in inches.

    SI panels preserve the main panel's absolute header height, map-row height,
    and inter-row gap.  Only unused physical-field rows are removed.
    """
    if field_count < 1:
        raise ValueError("A qualitative panel requires at least one physical field.")
    width = float(PANEL_A_LAYOUT["content_width_in"])
    main_height = float(PANEL_A_LAYOUT["content_height_in"])
    main_header_height = _qualitative_header_height_in()
    if main_header_height >= main_height:
        raise ValueError("The qualitative header band exceeds the main Panel-a height.")
    main_contour_height = main_height - main_header_height
    hspace = float(PANEL_A_LAYOUT["prediction_error_hspace"])

    def effective_units(ratios: list[float]) -> float:
        # GridSpec hspace is a fraction of average row height. Expressing it in
        # ratio units keeps SI map rows physically identical to main Panel a.
        count = len(ratios)
        ratio_sum = sum(ratios)
        return ratio_sum + (count - 1) * hspace * ratio_sum / count

    main_ratios, _ = _qualitative_row_structure(3, pair_ratios)
    target_ratios, _ = _qualitative_row_structure(field_count, pair_ratios)
    ratio_unit_height = main_contour_height / effective_units(main_ratios)
    contour_height = ratio_unit_height * effective_units(target_ratios)
    return width, main_header_height + contour_height


def draw_l2_heatmap_panel(fig, slot, cfg: dict, rows: list[dict], layout: dict) -> None:
    """Draw compact vertical condition heatmaps on a supplied SubplotSpec."""
    methods = [method["name"] for method in method_items(cfg, None)]
    fields = [field["key"] for field in cfg["fields"]] + ["Unobserved_mean"]
    labels = [field["label"] for field in cfg["fields"]] + ["Unobs."]
    lookup = {(r["method"], r["condition"], r["field"]): _float(r["mean"]) if r["status"] == "ok" else np.nan for r in rows}
    matrices = [np.asarray([[lookup.get((method, condition, field), np.nan) for field in fields] for method in methods]) for condition in cfg["conditions"]]
    norms = []
    for column in range(len(fields)):
        finite = np.concatenate([matrix[:, column][np.isfinite(matrix[:, column])] for matrix in matrices])
        norms.append(Normalize(np.nanmin(finite) if finite.size else 0, np.nanmax(finite) if finite.size else 1))
    grid = slot.subgridspec(3, 1, hspace=float(PANEL_B_LAYOUT["condition_hspace"]))
    # Pass the centralized Panel-b selection directly into the manual RGBA
    # heatmap. All approved options increase monotonically from light/low error
    # to dark or saturated/high error.
    cmap = _configured_colormap("panel_b")
    cmap.set_bad(cfg["style"]["missing"]["facecolor"])
    image = None
    for index, ((condition, spec), matrix) in enumerate(zip(cfg["conditions"].items(), matrices)):
        ax = fig.add_subplot(grid[index, 0])
        rgba = np.zeros((*matrix.shape, 4))
        for column, norm in enumerate(norms):
            # Preserve the mask so missing entries use ``cmap.set_bad`` rather
            # than being misrepresented as the smallest finite error.
            values = np.ma.masked_invalid(matrix[:, column])
            rgba[:, column] = cmap(norm(values))
        image = ax.imshow(rgba, aspect="auto")
        ax.set_title("Conditioned on " + spec["label"], loc="center", fontsize=SIZE_SUBPLOT_TITLE, pad=3)
        ax.set_xticks(range(len(fields)), labels if index == 2 else [], fontsize=SIZE_TICK_LABEL)
        ax.set_yticks(range(len(methods)), methods, fontsize=SIZE_TICK_LABEL)
        ax.tick_params(
            axis="x", length=0,
            pad=float(PANEL_B_LAYOUT["bottom_xtick_label_pad"]) if index == 2 else 1.0,
        )
        ax.tick_params(axis="y", length=0, pad=1.0)
        for row_index, method in enumerate(methods):
            if not np.isfinite(matrix[row_index]).any():
                ax.text((len(fields) - 1) / 2, row_index, "Missing", ha="center", va="center", fontsize=SIZE_ANNOTATION, color=COLOR_MISSING_TEXT, fontweight="bold")
            for col_index, field in enumerate(fields):
                value = matrix[row_index, col_index]
                if np.isfinite(value):
                    ax.text(col_index, row_index, f"{value:.2g}", ha="center", va="center", fontsize=SIZE_ANNOTATION,
                            color="white" if (0.2126*np.array(cmap(norms[col_index](value))[:3]) @ np.array([1,1,1])) < .48 else "#272727")
                elif np.isfinite(matrix[row_index]).any():
                    ax.text(col_index, row_index, "—", ha="center", va="center", fontsize=SIZE_ANNOTATION, color=COLOR_MISSING_TEXT)
                if col_index < len(cfg["fields"]) and cfg["fields"][col_index]["index"] in spec["cond_fields"]:
                    ax.add_patch(Rectangle((col_index - .48, row_index - .48), .96, .06, facecolor="#22A6B3", edgecolor="none"))
        for spine in ax.spines.values(): spine.set_visible(False)
        for tick in ax.get_yticklabels():
            if tick.get_text() == "DMF-Gen": tick.set_fontweight("bold")


def draw_spectral_panel(fig, slot, cfg: dict, layout: dict, energy_rows: list[dict], lsd_summary: list[dict], lsd_per: list[dict], selection: dict,
                        *, legend_slot=None, bar_slots=None, spectrum_slots=None) -> None:
    """Draw LSD bars and spectra, optionally into externally shared columns.

    The composite passes one shared parent GridSpec for panels C and D.  This
    makes the field columns in Panel C identical to the condition columns in
    Panel D, while standalone Panel C retains its local self-contained grid.
    """
    s = layout["spectral"]
    fields = [_field_lookup(cfg)[key] for key in _panel_c_field_keys(layout)]
    methods = list(method_items(cfg, None))
    if legend_slot is None or bar_slots is None or spectrum_slots is None:
        # After the shallow legend band, bars and spectra receive a strict 1:1
        # physical height allocation in both standalone and composite modes.
        ratios = tuple(float(value) for value in PANEL_C_LAYOUT["height_ratios"])
        if len(ratios) != 3 or not np.isclose(ratios[1], ratios[2]):
            raise ValueError("Panel-c bar and spectrum height ratios must remain exactly 1:1.")
        grid = slot.subgridspec(
            3, len(fields), height_ratios=ratios,
            hspace=float(PANEL_C_LAYOUT["grid_hspace"]), wspace=INTRA_PANEL_WSPACE,
        )
        legend_slot = grid[0, :]
        bar_slots = [grid[1, col] for col in range(len(fields))]
        spectrum_slots = [grid[2, col] for col in range(len(fields))]
    legend_ax = fig.add_subplot(legend_slot); legend_ax.set_axis_off()
    handles = [Patch(
        facecolor=(cfg["spectral"]["plotting"]["dmf_gen_accent"] if m["name"] == "DMF-Gen" else m["color"]),
        alpha=model_alpha(m["name"]), label=m["name"],
    ) for m in methods]
    # Eight methods in four edge-justified columns produce two airy rows that
    # span the complete Panel-c width instead of clustering at its centre.
    legend_ax.legend(handles=handles, ncol=PANEL_C_LAYOUT["legend_ncol"],
                     mode=PANEL_C_LAYOUT["legend_mode"], loc="lower left",
                     bbox_to_anchor=PANEL_C_LAYOUT["legend_bbox_to_anchor"],
                     bbox_transform=legend_ax.transAxes,
                     fontsize=SIZE_LEGEND, columnspacing=.55, handletextpad=.32,
                     handlelength=1.15, handleheight=.80, borderaxespad=0.0,
                     labelspacing=.20, frameon=False)
    summary = {(row["model_key"], row["condition"], row["field_name"]): row for row in lsd_summary}
    per = defaultdict(list)
    for row in lsd_per:
        if row["status"] == "ok": per[(row["model_key"], row["condition"], row["field_name"])].append(_float(row["lsd_db"]))
    curves = defaultdict(list)
    for row in energy_rows:
        if row["condition"] == s["condition"] and int(row["snapshot_index"]) == int(s["snapshot_index"]):
            curves[(row["model_key"], row["field_name"], row["source"])].append(row)
    rng = np.random.default_rng(cfg["defaults"]["seed"])
    selected_names = ["DMF-Gen", selection["generative"], selection["deterministic"]]
    selected_methods = [next(method for method in methods if method["name"] == name) for name in dict.fromkeys(selected_names)]
    for col, field in enumerate(fields):
        bar_ax, spectrum_ax = fig.add_subplot(bar_slots[col]), fig.add_subplot(spectrum_slots[col])
        for index, method in enumerate(methods):
            row = summary.get((method["directory"], s["condition"], field["key"]))
            mean = _float(row["mean_lsd_db"]) if row and row["status"] == "ok" else np.nan
            if not np.isfinite(mean):
                bar_ax.text(index, .02, "Missing", transform=bar_ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=SIZE_ANNOTATION, color=COLOR_MISSING_TEXT)
                continue
            low, high = _float(row["ci95_low_lsd_db"]), _float(row["ci95_high_lsd_db"])
            color = cfg["spectral"]["plotting"]["dmf_gen_accent"] if method["name"] == "DMF-Gen" else method["color"]
            bar_ax.bar(index, mean, color=color, alpha=model_alpha(method["name"]),
                       width=.74, edgecolor=COLOR_GROUND_TRUTH, linewidth=LW_DIVIDER)
            bar_ax.errorbar(index, mean, yerr=[[max(mean-low, 0)], [max(high-mean, 0)]], color=COLOR_GROUND_TRUTH, linewidth=LW_ERRORBAR, capsize=1.5)
            values = [value for value in per[(method["directory"], s["condition"], field["key"])] if np.isfinite(value)]
            scatter_values = _panel_c_scatter_values(values)
            bar_ax.scatter(
                index + rng.uniform(-.13, .13, len(scatter_values)), scatter_values,
                s=1.3, color="#9A9A9A", alpha=.32, linewidths=0, rasterized=True,
            )
        observed = field["index"] in cfg["conditions"][s["condition"]]["cond_fields"]
        bar_ax.set_title(f"{field['label']} ({'obs.' if observed else 'unobs.'})", fontsize=SIZE_SUBPLOT_TITLE, pad=2)
        bar_ax.set_xticks(range(len(methods)), [])
        bar_ax.set_ylabel("LSD (dB)" if col == 0 else "", fontsize=SIZE_AXIS_LABEL)
        bar_ax.grid(axis="y", lw=LW_GRID, alpha=.25); bar_ax.tick_params(axis="y", labelsize=SIZE_TICK_LABEL, length=1.5)
        truth = curves.get(("truth", field["key"], "truth"), [])
        def valid(rows):
            return [(_float(r["wavenumber"]), _float(r["spectral_energy"])) for r in rows
                    if r["status"] == "ok" and _float(r["wavenumber"]) > 0 and _float(r["spectral_energy"]) > 0]
        true = valid(truth)
        if true: spectrum_ax.plot(*np.asarray(true).T, color=COLOR_GROUND_TRUTH, lw=LW_LINE_PLOT, label="GT")
        for index, method in enumerate(selected_methods):
            data = valid(curves.get((method["directory"], field["key"], "reconstruction"), []))
            if data:
                color = cfg["spectral"]["plotting"]["dmf_gen_accent"] if method["name"] == "DMF-Gen" else method["color"]
                spectrum_ax.plot(
                    *np.asarray(data).T, color=color, alpha=model_alpha(method["name"]),
                    lw=LW_LINE_PLOT if method["name"] == "DMF-Gen" else LW_LINE_SECONDARY,
                    linestyle=method_line_style(index), label=method["name"],
                )
        if not true:
            mark_missing(spectrum_ax, "Missing", cfg)
        else:
            spectrum_ax.set_xscale("log"); spectrum_ax.set_yscale("log")
            spectrum_ax.grid(which="both", lw=LW_GRID, alpha=.18)
            spectrum_ax.tick_params(labelsize=SIZE_TICK_LABEL, length=1.5)
            # Default LogLocator may retain a labelled decade beyond the data
            # limits; on the rightmost subplot that ghost label can be clipped
            # by the fixed 7.2-in page.  Keep only decades inside the view.
            xmin, xmax = spectrum_ax.get_xlim()
            x_low_exp = int(np.ceil(np.log10(xmin)))
            x_high_exp = int(np.floor(np.log10(xmax)))
            x_ticks = 10.0 ** np.arange(x_low_exp, x_high_exp + 1)
            spectrum_ax.xaxis.set_major_locator(FixedLocator(x_ticks))
            spectrum_ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
            ymin, ymax = spectrum_ax.get_ylim()
            low_exp = int(np.floor(np.log10(ymin) / 2.0) * 2)
            high_exp = int(np.ceil(np.log10(ymax) / 2.0) * 2)
            spectrum_ax.set_yticks(10.0 ** np.arange(low_exp, high_exp + 1, 2))
            spectrum_ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
            spectrum_ax.yaxis.set_minor_locator(NullLocator())
            if col == 0: spectrum_ax.set_ylabel("spectral energy", fontsize=SIZE_AXIS_LABEL)
            xlabel = spectrum_ax.set_xlabel(
                str(s.get("x_label", "wavenumber")) if col == 1 else "",
                fontsize=SIZE_AXIS_LABEL, labelpad=1.5,
            )
            if col == 1:
                labels = getattr(fig, "_panel_c_bottom_xlabel_artists", [])
                labels.append(xlabel)
                fig._panel_c_bottom_xlabel_artists = labels  # type: ignore[attr-defined]
            if col == 0:
                spectrum_ax.legend(fontsize=SIZE_LEGEND, loc="lower left", handlelength=1.4, borderpad=.25)


def _panel_d_specification(key: str) -> dict:
    """Return and validate one header-defined main/SI selection record."""
    if key not in PANEL_D_FIGURE_SELECTIONS:
        raise KeyError(f"Unknown Panel-D selection {key!r}; choose from {list(PANEL_D_FIGURE_SELECTIONS)}")
    selection = PANEL_D_FIGURE_SELECTIONS[key]
    subplots = selection.get("subplots", [])
    if not 1 <= len(subplots) <= 3:
        raise ValueError(f"Panel-D selection {key!r} needs one to three subplots; got {len(subplots)}.")
    for subplot in subplots:
        if subplot.get("pair") not in COUPLING_PAIR_LABELS:
            raise ValueError(f"Unknown coupling pair in {key!r}: {subplot.get('pair')!r}")
        metric_pair = subplot.get("metric_pair", subplot.get("pair"))
        if metric_pair not in COUPLING_PAIR_LABELS:
            raise ValueError(f"Unknown metric pair in {key!r}: {metric_pair!r}")
        if subplot.get("condition") not in {"Cond_T", "Cond_TU1", "Cond_COTU1P"}:
            raise ValueError(f"Unknown conditioning key in {key!r}: {subplot.get('condition')!r}")
    return selection


def _format_two_significant_figures(value: float) -> str:
    """Format a finite numeric mean with two visible significant figures."""
    if not np.isfinite(value):
        return "NaN"
    if value == 0:
        return "0.0"
    exponent = int(np.floor(np.log10(abs(value))))
    if -3 <= exponent < 2:
        return f"{value:.{max(0, 1 - exponent)}f}"
    return f"{value:.1e}"


def _panel_d_grid_slots(fig, slot, column_count: int) -> dict[str, list[Any]]:
    """Create compact-PDF/tall-violin rows with shared physical widths."""
    if column_count < 1:
        raise ValueError("Panel d requires at least one coupling column.")
    box = slot.get_position(fig)
    fig_width, fig_height = (float(value) for value in fig.get_size_inches())
    total_width_in = box.width * fig_width
    total_height_in = box.height * fig_height
    wspace = float(INTRA_PANEL_WSPACE)
    column_width_in = total_width_in / (column_count + (column_count - 1) * wspace)
    hspace = float(PANEL_D_STYLE["row_hspace"])
    # For two GridSpec rows, one gap occupies hspace times the average row
    # height. Solve for the physical height available to the two axes rows.
    row_height_total_in = total_height_in / (1.0 + hspace / 2.0)
    pdf_height_scale = float(PANEL_D_STYLE["pdf_height_scale"])
    if not 0.0 < pdf_height_scale <= 1.0:
        raise ValueError("PANEL_D_STYLE['pdf_height_scale'] must be in (0, 1].")
    pdf_height_in = column_width_in * pdf_height_scale
    violin_height_in = row_height_total_in - pdf_height_in
    if violin_height_in <= 0.45:
        raise ValueError(
            f"Panel d leaves only {violin_height_in:.3f} in for the violin row; "
            "reduce PANEL_C_D_GAP_IN/row_hspace or widen the right column."
        )
    grid = slot.subgridspec(
        2, column_count,
        height_ratios=[pdf_height_in, violin_height_in],
        hspace=hspace, wspace=wspace,
    )
    return {
        "pdfs": [grid[0, col] for col in range(column_count)],
        "violins": [grid[1, col] for col in range(column_count)],
    }


def _ground_truth_joint_pdf_data(cfg: dict, truth_ensemble: dict,
                                 subplot_specs: list[dict],
                                 precomputed_pdfs: dict | None = None) -> tuple[list[dict], LogNorm]:
    """Build normalized truth histograms pooled over the 25-frame ensemble."""
    field_by_index = {field["index"]: field for field in cfg["fields"]}
    results, positive = [], []
    bins = int(PANEL_D_STYLE["pdf_bins"])
    qlo, qhi = (float(value) for value in PANEL_D_STYLE["pdf_quantiles"])
    for subplot in subplot_specs:
        pair, condition = subplot["pair"], subplot["condition"]
        precomputed = (precomputed_pdfs or {}).get((pair, condition))
        if precomputed is not None:
            matrix = np.asarray(precomputed["matrix"], dtype=float)
            positive.extend(matrix[matrix > 0].tolist())
            x_index, y_index = PAIR_FIELDS[pair]
            results.append({
                "pair": pair,
                "condition": condition,
                "matrix": matrix,
                "frame_count": int(precomputed["frame_count"]),
                "extent": [float(value) for value in precomputed["extent"]],
                "x_label": field_by_index[x_index]["label"],
                "y_label": field_by_index[y_index]["label"],
            })
            continue
        frames = truth_ensemble.get(condition, [])
        if not frames:
            results.append({"pair": pair, "condition": condition, "matrix": None})
            continue
        x_index, y_index = PAIR_FIELDS[pair]
        # Superimpose samples from all selected frames before determining the
        # robust global bounds and histogram probabilities.
        x = np.concatenate([
            np.asarray(frame["truth_phys"][:, x_index], dtype=float) for frame in frames
        ])
        y = np.concatenate([
            np.asarray(frame["truth_phys"][:, y_index], dtype=float) for frame in frames
        ])
        finite = np.isfinite(x) & np.isfinite(y)
        x, y = x[finite], y[finite]
        if x.size == 0:
            results.append({"pair": pair, "condition": condition, "matrix": None})
            continue
        xlim, ylim = np.quantile(x, [qlo, qhi]), np.quantile(y, [qlo, qhi])
        if xlim[1] <= xlim[0]: xlim = np.array([xlim[0] - .5, xlim[1] + .5])
        if ylim[1] <= ylim[0]: ylim = np.array([ylim[0] - .5, ylim[1] + .5])
        x_edges = np.linspace(xlim[0], xlim[1], bins + 1)
        y_edges = np.linspace(ylim[0], ylim[1], bins + 1)
        matrix = histogram(x, y, (x_edges, y_edges))
        positive.extend(matrix[matrix > 0].tolist())
        results.append({
            "pair": pair, "condition": condition, "matrix": matrix,
            "frame_count": len(frames),
            "extent": [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            "x_label": field_by_index[x_index]["label"],
            "y_label": field_by_index[y_index]["label"],
        })
    # LogNorm cannot include zero. Use the smallest observed positive bin
    # (subject only to a numerical floor), while the renderer masks exact
    # zeros and assigns them the configured white background below.
    vmin_floor = float(COLORMAP_CONFIG["panel_d"]["positive_vmin_floor"])
    vmin = max(min(positive, default=vmin_floor), vmin_floor)
    vmax = max(positive, default=1.0)
    return results, LogNorm(vmin=vmin, vmax=vmax)


def _draw_ground_truth_joint_pdf_row(fig, cfg: dict, truth_ensemble: dict,
                                     subplot_specs: list[dict], panel_slot,
                                     pdf_slots: list,
                                     precomputed_pdfs: dict | None = None) -> list:
    """Draw compact headers and square, tick-free truth PDFs for Panel d."""
    pdf_data, norm = _ground_truth_joint_pdf_data(
        cfg, truth_ensemble, subplot_specs, precomputed_pdfs,
    )
    panel_box = panel_slot.get_position(fig)
    # Use exactly the same baseline formula as ``_panel_letter`` so the three
    # mathematical coupling titles and the panel label "d" align perfectly.
    title_y = panel_box.y1 + PANEL_A_LAYOUT["panel_letter_y_offset"]
    axes = []
    for subplot, pdf_slot, item in zip(subplot_specs, pdf_slots, pdf_data):
        pdf_box = pdf_slot.get_position(fig)
        title_artist = fig.text(
            (pdf_box.x0 + pdf_box.x1) / 2.0, title_y,
            COUPLING_PAIR_MATH_LABELS[subplot["pair"]],
            ha="center", va="bottom", fontsize=SIZE_SUBPLOT_TITLE,
        )
        titles = getattr(fig, "_panel_d_title_artists", [])
        titles.append(title_artist)
        fig._panel_d_title_artists = titles  # type: ignore[attr-defined]
        ax = fig.add_subplot(pdf_slot); axes.append(ax)
        if item.get("matrix") is None:
            mark_missing(ax, "Missing", cfg)
        else:
            # Pass the centralized Panel-d selection directly to imshow.
            # Masked zero bins and any under-range numerical residue are white;
            # all positive bins use the shared logarithmic normalization.
            pdf_cmap = _configured_colormap("panel_d")
            zero_color = COLORMAP_CONFIG["panel_d"]["zero_density_color"]
            pdf_cmap.set_bad(zero_color)
            pdf_cmap.set_under(zero_color)
            ax.imshow(
                np.ma.masked_less_equal(item["matrix"].T, 0.0),
                origin="lower", extent=item["extent"],
                aspect="auto", cmap=pdf_cmap, norm=norm,
                interpolation="nearest", rasterized=True,
            )
            ax.set_xlabel(item["x_label"], fontsize=SIZE_AXIS_LABEL,
                          labelpad=PANEL_D_STYLE["pdf_axis_labelpad"])
            ax.set_ylabel(item["y_label"], fontsize=SIZE_AXIS_LABEL,
                          labelpad=PANEL_D_STYLE["pdf_axis_labelpad"])
            # The small PDFs communicate joint-distribution shape, not exact
            # coordinate lookup. Keep physical field titles but remove both
            # numerical labels and tick marks to maximize the data-ink ratio.
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
            frame_counts = getattr(fig, "_panel_d_pdf_frame_counts", {})
            frame_counts[f"{subplot['pair']}|{subplot['condition']}"] = int(item["frame_count"])
            fig._panel_d_pdf_frame_counts = frame_counts  # type: ignore[attr-defined]
        ax.set_box_aspect(float(PANEL_D_STYLE["pdf_height_scale"]))
        ax.set_anchor("C")
    return axes


def draw_coupling_jsd_panel(fig, slot, cfg: dict, rows: list[dict], summary_rows: list[dict], *,
                            subplot_specs: list[dict], slots=None,
                            show_method_labels: bool = True, show_titles: bool = True,
                            center_xlabel_only: bool = False) -> list:
    """Draw selected channel-coupling JSD distributions from CSV rows only.

    Each member of ``subplot_specs`` chooses one of the nine available results
    with ``{"pair": ..., "condition": ...}``.  This makes the main and SI
    panel selection independent of the rendering/layout implementation.
    """
    methods = list(method_items(cfg, None))
    complete = {(r["method"], r["condition"], r["pair"]): r.get("status") == "ok" for r in summary_rows}
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "ok" and complete.get((row["method"], row["condition"], row["pair"]), False):
            groups[(row["method"], row["condition"], row["pair"])].append(_float(row["jsd_base2"]))
    if slots is None:
        grid = slot.subgridspec(1, len(subplot_specs), hspace=0.0, wspace=INTRA_PANEL_WSPACE)
        slots = [grid[0, col] for col in range(len(subplot_specs))]
    if len(slots) != len(subplot_specs):
        raise ValueError(f"Got {len(slots)} axes slots for {len(subplot_specs)} selected coupling subplots.")
    if float(PANEL_D_STYLE["mean_x_axes"]) <= float(PANEL_D_STYLE["violin_data_right_fraction"]):
        raise ValueError("Panel-d mean annotation column must begin to the right of all violin data.")
    axes = []
    for col_index, subplot in enumerate(subplot_specs):
            pair, condition = subplot["pair"], subplot["condition"]
            metric_pair = subplot.get("metric_pair", pair)
            # Use an explicitly supplied diagnostic title before consulting
            # the manuscript pair-label registry.
            pair_label = subplot.get("title") or COUPLING_PAIR_LABELS[pair]
            ax = fig.add_subplot(slots[col_index])
            axes.append(ax)
            datasets, positions, colors, alphas = [], [], [], []
            for method_index, method in enumerate(methods):
                vals = [value for value in groups[(method["name"], condition, metric_pair)] if np.isfinite(value)]
                if vals:
                    datasets.append(vals); positions.append(method_index); colors.append(method["color"])
                    alphas.append(model_alpha(method["name"]))
                else: ax.text(.98, method_index, "Missing", transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=SIZE_ANNOTATION, color=COLOR_MISSING_TEXT)
            if datasets:
                violin = ax.violinplot(datasets, positions=positions, vert=False,
                                       widths=PANEL_D_STYLE["violin_width"], showextrema=False)
                for body, color, alpha in zip(violin["bodies"], colors, alphas):
                    body.set_facecolor(color); body.set_edgecolor(COLOR_GROUND_TRUTH)
                    body.set_alpha(alpha); body.set_linewidth(LW_DIVIDER)
                for values, position in zip(datasets, positions):
                    q25, med, q75 = np.quantile(values, [.25, .5, .75])
                    ax.plot([q25, q75], [position, position], color=COLOR_GROUND_TRUTH, lw=LW_AXIS_SPINE)
                    ax.plot(med, position, marker="|", color="white", ms=3.5, mew=.8)
            ax.set_title(pair_label if show_titles else "", fontsize=SIZE_SUBPLOT_TITLE, pad=2.2, linespacing=1.05)
            ax.set_yticks(range(len(methods)), [m["name"] for m in methods] if show_method_labels and col_index == 0 else [])
            if PANEL_D_STYLE.get("dmf_first_at_top", False):
                ax.invert_yaxis()
            ax.tick_params(axis="y", labelsize=SIZE_TICK_LABEL, length=0, pad=1.5); ax.tick_params(axis="x", labelsize=SIZE_TICK_LABEL, length=1.4, pad=1.0)
            for tick in ax.get_yticklabels():
                if tick.get_text() == "DMF-Gen": tick.set_fontweight("bold")
            finite = [v for m in methods for v in groups[(m["name"], condition, metric_pair)] if np.isfinite(v)]
            max_val = max(finite, default=1.0)
            # Violin KDE support ends at max_val; the log-coordinate solve
            # below reserves a dedicated empty column to its right for μ.
            positive = [value for value in finite if value > 0]
            if not positive:
                positive = [max(max_val * 1.0e-3, 1.0e-12)]
            xmin = max(min(positive) * float(PANEL_D_STYLE["violin_log_lower_factor"]), 1.0e-12)
            # Solve the log-coordinate transform so max_val lands at a fixed
            # axes fraction, leaving a guaranteed-width annotation column.
            data_fraction = float(PANEL_D_STYLE["violin_data_right_fraction"])
            log_xlim = np.log(xmin) + (np.log(max_val) - np.log(xmin)) / data_fraction
            xlimit = max(
                float(np.exp(log_xlim)),
                max_val * float(PANEL_D_STYLE["mean_xlim_min_factor"]),
            )
            retained_ticks = [
                float(value) for value in PANEL_D_STYLE["retained_ticks"].get(pair, ())
                if xmin <= float(value) <= xlimit
            ]
            if not retained_ticks:
                retained_ticks = [float(np.sqrt(xmin * xlimit))]
            ax.set_xscale("log")
            ax.set_xlim(xmin, xlimit)
            ax.xaxis.set_major_locator(FixedLocator(retained_ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([f"{value:g}" for value in retained_ticks]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.grid(axis="x", lw=LW_GRID, alpha=.22)
            retained = getattr(fig, "_panel_d_retained_log_ticks", {})
            retained[f"{pair}|{condition}"] = retained_ticks
            fig._panel_d_retained_log_ticks = retained  # type: ignore[attr-defined]
            if PANEL_D_STYLE["annotate_means"]:
                for method_index, method in enumerate(methods):
                    values = [value for value in groups[(method["name"], condition, metric_pair)] if np.isfinite(value)]
                    if values:
                        ax.text(PANEL_D_STYLE["mean_x_axes"], method_index,
                                PANEL_D_STYLE["mean_prefix"] + _format_two_significant_figures(float(np.mean(values))),
                                ha="left", va="center", fontsize=PANEL_D_STYLE["mean_fontsize"],
                                color=PANEL_D_STYLE["mean_color"], clip_on=True,
                                transform=ax.get_yaxis_transform())
            show_xlabel = (not center_xlabel_only or col_index == len(subplot_specs) // 2)
            ax.set_xlabel(
                PANEL_D_STYLE["condition_xlabels"].get(
                    condition,
                    PANEL_D_STYLE["x_label_template"].format(
                        condition_label=cfg["conditions"][condition]["label"]
                    ),
                ) if show_xlabel else "",
                fontsize=PANEL_D_STYLE["x_label_fontsize"],
                labelpad=float(PANEL_D_STYLE["x_label_pad"]),
            )
    return axes


def draw_jsd_panel(fig, slot, cfg: dict, rows: list[dict], summary_rows: list[dict], *,
                   truth_ensemble: dict, precomputed_pdfs: dict | None = None) -> None:
    """Draw compact 0.8-height truth PDFs above expanded log-JSD violins."""
    selection = _panel_d_specification("main")
    subplot_specs = selection["subplots"]
    slots = _panel_d_grid_slots(fig, slot, len(subplot_specs))
    pdf_axes = _draw_ground_truth_joint_pdf_row(
        fig, cfg, truth_ensemble, subplot_specs,
        slot, slots["pdfs"], precomputed_pdfs,
    )
    violin_axes = draw_coupling_jsd_panel(
        fig, slot, cfg, rows, summary_rows, subplot_specs=subplot_specs,
        slots=slots["violins"], show_method_labels=True, show_titles=False,
        center_xlabel_only=True,
    )
    fig._panel_d_violin_axes = violin_axes  # type: ignore[attr-defined]
    # Render once and enforce the architectural invariants numerically.
    fig.canvas.draw()
    fig_width, fig_height = (float(value) for value in fig.get_size_inches())
    for pdf_ax, violin_ax in zip(pdf_axes, violin_axes):
        pdf_box, violin_box = pdf_ax.get_position(), violin_ax.get_position()
        pdf_width_in, pdf_height_in = pdf_box.width * fig_width, pdf_box.height * fig_height
        violin_width_in = violin_box.width * fig_width
        expected_pdf_height_in = pdf_width_in * float(PANEL_D_STYLE["pdf_height_scale"])
        if abs(pdf_height_in - expected_pdf_height_in) > 1.0e-3:
            raise ValueError("Panel-d ground-truth PDF box violates pdf_height_scale.")
        if abs(pdf_width_in - violin_width_in) > 1.0e-3:
            raise ValueError("Panel-d PDF and violin physical widths do not match.")


def _pdf_matrix(rows: list[dict]) -> tuple[np.ndarray, list[float]]:
    nx, ny = max(int(r["bin_x"]) for r in rows) + 1, max(int(r["bin_y"]) for r in rows) + 1
    matrix = np.full((nx, ny), np.nan)
    for row in rows: matrix[int(row["bin_x"]), int(row["bin_y"])] = _float(row["probability"])
    first, last = rows[0], rows[-1]
    return matrix, [_float(first["x_left"]), _float(last["x_right"]), _float(first["y_left"]), _float(last["y_right"])]


def _pooled_joint_pdf(cfg: dict, layout: dict, rows: list[dict]) -> tuple[dict, dict, list[int]]:
    """Pool objectively selected cached T--U1 fields with existing fixed bins."""
    j, pair = layout["joint_pdf"], layout["joint_pdf"].get("pair", "T-U1")
    template = [r for r in rows if r["pair"] == pair and r["condition"] == j["condition"] and r["source"] == "truth"]
    matrix, extent = _pdf_matrix(template)
    nx, ny = matrix.shape
    ex = np.linspace(extent[0], extent[1], nx + 1); ey = np.linspace(extent[2], extent[3], ny + 1)
    manifest = read_csv(RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{layout.get('_run_id', 'paper_full_20260711')}.csv")
    by_key = {(r["method"], int(r["snapshot"])): r for r in manifest if r.get("condition") == j["condition"] and r.get("status") == "ok"}
    snapshots = sorted({int(r["snapshot"]) for r in manifest if r.get("condition") == j["condition"] and r.get("status") == "ok"})
    rng = np.random.default_rng(int(j.get("snapshot_seed", 42)))
    selected = sorted(rng.choice(snapshots, min(int(j.get("snapshot_count", 5)), len(snapshots)), replace=False).tolist())
    out, metrics = {}, {}
    a, b = PAIR_FIELDS[pair]
    for method in ["Ground truth"] + j["fixed_columns"][1:]:
        cache_method = "DMF-Gen" if method == "Ground truth" else method
        truth, recon = [], []
        for snapshot in selected:
            entry = by_key.get((cache_method, snapshot))
            if entry is None: continue
            arrays, _ = load_cache(Path(entry["cache_path"]))
            truth.append(arrays["truth_phys"][:, [a, b]])
            if method != "Ground truth": recon.append(arrays["recon_phys"][:, [a, b]])
        if not truth: continue
        truth_hist = histogram(np.concatenate(truth)[:, 0], np.concatenate(truth)[:, 1], (ex, ey))
        out[(method, "truth")] = truth_hist
        if recon:
            rec_hist = histogram(np.concatenate(recon)[:, 0], np.concatenate(recon)[:, 1], (ex, ey))
            out[(method, "reconstruction")] = rec_hist
            metrics[method] = jsd_base2(truth_hist, rec_hist, cfg["defaults"]["pdf_pseudocount"])
    return out, metrics, selected


def draw_joint_pdf_panel(fig, slot, cfg: dict, layout: dict, rows: list[dict], metrics: list[dict]) -> None:
    """Draw two all-method fixed-slot representative joint-PDF rows in panel e."""
    j = layout["joint_pdf"]
    columns = j["fixed_columns"]
    pairs = [j.get("pair", "T-U1")]
    pooled, pooled_metrics, selected_snapshots = _pooled_joint_pdf(cfg, layout, rows)
    layout["_joint_selected_snapshots"] = selected_snapshots
    groups = defaultdict(list)
    for row in rows:
        if row["condition"] == j["condition"] and int(row["snapshot"]) == int(j.get("snapshot_index", 0)):
            groups[(row["method"], row["pair"], row["source"])].append(row)
    metric = {(row["method"], row["pair"]): row for row in metrics if row["condition"] == j["condition"] and int(row["snapshot"]) == int(j.get("snapshot_index", 0))}
    positive = [_float(row["probability"]) for row in rows if row["status"] == "ok" and _float(row["probability"]) > 0]
    norm = LogNorm(max(min(positive, default=1e-8), 1e-8), max(positive, default=1.0))
    grid = slot.subgridspec(1, len(columns) + 1, width_ratios=[1] * len(columns) + [.070], wspace=.018)
    axes = []
    for row_index, pair in enumerate(pairs):
        for col_index, label in enumerate(columns):
            ax = fig.add_subplot(grid[0, col_index]); axes.append(ax)
            if label == "Ground truth": key, source = ("truth", pair, "truth"), "truth"
            else: key, source = (label, pair, "reconstruction"), "reconstruction"
            cell_rows = groups.get(key, [])
            status = "ok" if label == "Ground truth" else metric.get((label, pair), {}).get("status", "missing cache")
            pooled_matrix = pooled.get((label, source))
            if (pooled_matrix is None and (not cell_rows or status != "ok")):
                mark_missing(ax, "Missing", cfg)
            else:
                matrix, extent = (pooled_matrix, _pdf_matrix(cell_rows)[1]) if pooled_matrix is not None else _pdf_matrix(cell_rows)
                # The fixed physical limits are retained in `extent`, while a
                # square visual box makes every joint-density slot comparable
                # on this dense full-page layout.  The old equal-data-aspect
                # call left a thin density ribbon inside a large square axes.
                ax.imshow(matrix.T, origin="lower", extent=extent, aspect="auto", cmap="magma", norm=norm, rasterized=True, interpolation="nearest")
                if label != "Ground truth":
                    value = pooled_metrics.get(label, _float(metric.get((label, pair), {}).get("jsd_base2")))
                    ax.text(.02, .94, f"JSD={value:.3f}", transform=ax.transAxes,
                            ha="left", va="top", fontsize=SIZE_ANNOTATION, color="white", bbox={"facecolor":COLOR_GROUND_TRUTH, "edgecolor":"none", "alpha":.55, "pad":.8})
            ax.set_title(label, fontsize=SIZE_SUBPLOT_TITLE, pad=1, fontweight="bold" if label == "DMF-Gen" else "normal")
            if col_index == 0:
                ax.set_ylabel(pair, fontsize=SIZE_AXIS_LABEL)
            ax.set_box_aspect(1)
            ax.set_anchor("S")
            _clean_axis(ax)
    cbar_slot = grid[0, -1].subgridspec(3, 1, height_ratios=[.20, .60, .20])
    cax = fig.add_subplot(cbar_slot[1, 0]); scalar = plt.cm.ScalarMappable(norm=norm, cmap="magma")
    cb = fig.colorbar(scalar, cax=cax)
    cb.ax.tick_params(labelsize=SIZE_TICK_LABEL, length=1.2, pad=1)
    cb.set_label("probability", fontsize=SIZE_AXIS_LABEL, labelpad=2)


def _load_sources(rid_arg: str | None) -> dict[str, tuple[Path, list[dict]]]:
    specs = {
        "field_l2": (RESULTS_DIR / "FieldL2", "FieldL2_summary"),
        "energy": (RESULTS_DIR / "Spectral" / "EnergySpectra", "EnergySpectra_snapshot"),
        "lsd_per": (RESULTS_DIR / "Spectral" / "SpectralLSD", "SpectralLSD_per_snapshot"),
        "lsd_summary": (RESULTS_DIR / "Spectral" / "SpectralLSD", "SpectralLSD_summary"),
        "pdf": (RESULTS_DIR / "JointPDF", "JointPDF_snapshot"),
        "pdf_metrics": (RESULTS_DIR / "JointPDF", "JointPDF_snapshot_metrics"),
        "jsd": (RESULTS_DIR / "JointPDF_JSD", "CouplingJSD_per_snapshot"),
        "jsd_summary": (RESULTS_DIR / "JointPDF_JSD", "CouplingJSD_summary"),
        # U1--p is intentionally produced as a diagnostic CSV, then merged
        # here with the established coupling table for Panel-D/SI selection.
        "flow_jsd": (RESULTS_DIR / "JointPDF_JSD", "FlowConsistencyJSD_per_snapshot"),
        "flow_jsd_summary": (RESULTS_DIR / "JointPDF_JSD", "FlowConsistencyJSD_summary"),
    }
    return {name: (path := _csv_path(folder, prefix, rid_arg), read_csv(path)) for name, (folder, prefix) in specs.items()}


def _panel_d_csv_rows(sources: dict[str, tuple[Path, list[dict]]],
                      gpu_payload: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Combine finalized rows with transient GPU-only Panel-D metrics."""
    gpu_payload = gpu_payload or {}
    return (
        [*sources["jsd"][1], *sources["flow_jsd"][1], *gpu_payload.get("rows", [])],
        [*sources["jsd_summary"][1], *sources["flow_jsd_summary"][1],
         *gpu_payload.get("summary", [])],
    )


def _gpu_helper_python() -> Path:
    """Locate the existing project CUDA environment without installing packages."""
    candidates = []
    override = os.environ.get("PHYCOFLOW_GPU_PYTHON")
    if override:
        candidates.append(Path(override))
    # ``fig`` and ``phycoflow_env`` are sibling conda environments in the
    # project installation. Keep the environment override above for portable
    # deployments with a different layout.
    if len(Path(sys.executable).parents) >= 3:
        candidates.append(Path(sys.executable).parents[2] / "phycoflow_env" / "bin" / "python")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "No CUDA-enabled project Python was found. Set PHYCOFLOW_GPU_PYTHON; "
        "CPU fallback is intentionally disabled for on-the-fly Panel-d metrics."
    )


def _compute_panel_d_gpu_payload(cfg: dict, rid: str, proposed_model: str,
                                 subplot_specs: list[dict]) -> dict:
    """Compute missing Panel-d pairs through read-only transient CUDA work."""
    requested = []
    for subplot in subplot_specs:
        pair = subplot["pair"]
        key = (pair, subplot["condition"])
        if pair in PANEL_D_GPU_ON_THE_FLY_PAIRS and key not in requested:
            requested.append(key)
    if not requested:
        return {"rows": [], "summary": [], "pdfs": {}, "metadata": []}

    helper = SCRIPT_DIR / "45_gpu_pair_postprocess.py"
    if not helper.is_file():
        raise FileNotFoundError(f"Missing CUDA post-processing helper: {helper}")
    manifest = RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{rid}.csv"
    methods = [method["name"] for method in method_items(cfg, None)]
    payload = {"rows": [], "summary": [], "pdfs": {}, "metadata": []}
    for pair, condition in requested:
        x_index, y_index = PAIR_FIELDS[pair]
        command = [
            str(_gpu_helper_python()), str(helper),
            "--manifest", str(manifest), "--pair", pair, "--condition", condition,
            "--x-index", str(x_index), "--y-index", str(y_index),
            "--proposed-model", proposed_model,
            "--bins", str(PANEL_D_STYLE["pdf_bins"]),
            "--quantiles", *(str(value) for value in PANEL_D_STYLE["pdf_quantiles"]),
            "--pseudocount", str(cfg["defaults"]["pdf_pseudocount"]),
            "--pdf-frame-count", str(PANEL_D_STYLE["pdf_ensemble_frame_count"]),
            "--methods", *methods,
        ]
        completed = subprocess.run(
            command, cwd=SCRIPT_DIR, check=True, text=True,
            stdout=subprocess.PIPE,
        )
        computed = json.loads(completed.stdout)
        metadata = computed.get("metadata", {})
        if metadata.get("backend") != "torch.cuda":
            raise RuntimeError("Panel-d transient post-processing did not use CUDA.")
        if metadata.get("persistent_metric_artifact_written") is not False:
            raise RuntimeError("GPU helper violated the no-persistent-metric contract.")
        payload["rows"].extend(computed.get("rows", []))
        payload["summary"].extend(computed.get("summary", []))
        pdf = computed.get("pdf")
        if pdf:
            payload["pdfs"][(pair, condition)] = pdf
        payload["metadata"].append(metadata)
    return payload


def _save_figure(fig, base: Path, formats: list[str], dpi: int) -> list[Path]:
    """Export through the portable fixed-canvas style and bbox validator."""
    return save_publication_figure(
        fig, base, formats, dpi=dpi, fixed_canvas=True, validate_bbox=True,
    )


def _shared_right_validation_grid(fig, slot):
    """Split the right column into 45% Panel c and 55% Panel d master slots.

    The parent slot has the same top and bottom as full-height Panel b. A
    physical text-safe gap is removed between the rows, so C's top and D's
    bottom remain exactly aligned with the full-height Panel b.
    """
    box = slot.get_position(fig)
    total_height_in = box.height * float(fig.get_size_inches()[1])
    hspace = gridspec_space_from_inches(total_height_in, PANEL_C_D_GAP_IN, 2)
    return slot.subgridspec(
        2, 1, height_ratios=PANEL_C_D_HEIGHT_RATIOS, hspace=hspace,
    )


def _composite_canvas_size(layout: dict) -> tuple[float, float]:
    """Return strict width plus content-derived compact composite height."""
    lower_height = float(layout.get("lower_row_content_height_in", DEFAULT_LOWER_ROW_HEIGHT_IN))
    height = adaptive_composite_height(
        [float(PANEL_A_LAYOUT["content_height_in"]), lower_height],
        row_gaps_in=[float(PANEL_A_LAYOUT["master_lower_gap_in"])],
        top_margin_in=float(PANEL_A_LAYOUT["master_top_margin_in"]),
        bottom_margin_in=COMPOSITE_MARGIN_BOTTOM_IN,
    )
    layout["_computed_width_in"] = COMPOSITE_WIDTH_IN
    layout["_computed_height_in"] = height
    return COMPOSITE_WIDTH_IN, height


def _master_panel_slots(fig, layout: dict) -> dict[str, Any]:
    """Create non-overlapping master slots with benchmark-locked Panel a.

    Panel a is positioned by physical inches from the top of the declared
    canvas.  The lower row receives all remaining height below an explicit
    protected gap, so B--D tuning cannot distort or overlap Panel a.
    """
    width, height = (float(value) for value in fig.get_size_inches())
    a_left = COMPOSITE_MARGIN_LEFT_IN / width
    a_right = 1.0 - COMPOSITE_MARGIN_RIGHT_IN / width
    # The lower validation row has long heatmap method labels and right-edge
    # violin mean annotations.  Keep it slightly farther inside the fixed
    # canvas; this does not alter the benchmark-locked Panel-a width.
    lower_left = LOWER_ROW_MARGIN_LEFT_IN / width
    lower_right = 1.0 - LOWER_ROW_MARGIN_RIGHT_IN / width
    a_top = 1.0 - float(PANEL_A_LAYOUT["master_top_margin_in"]) / height
    a_bottom = a_top - float(PANEL_A_LAYOUT["content_height_in"]) / height
    lower_top = a_bottom - float(PANEL_A_LAYOUT["master_lower_gap_in"]) / height
    lower_bottom = COMPOSITE_MARGIN_BOTTOM_IN / height
    if not 0.0 < lower_bottom < lower_top < a_bottom < a_top < 1.0:
        raise ValueError("Invalid master vertical geometry; check PANEL_A_LAYOUT inch controls.")
    a_slot = fig.add_gridspec(1, 1, left=a_left, right=a_right, top=a_top, bottom=a_bottom)[0, 0]
    lower = fig.add_gridspec(1, 1, left=lower_left, right=lower_right, top=lower_top, bottom=lower_bottom)[0, 0]
    lower_width_in = width - LOWER_ROW_MARGIN_LEFT_IN - LOWER_ROW_MARGIN_RIGHT_IN
    major_wspace = gridspec_space_from_inches(lower_width_in, PANEL_WSPACE_IN, 2)
    middle = lower.subgridspec(
        1, 2, width_ratios=layout["outer_middle_width_ratios"], wspace=major_wspace,
    )
    b_wrap = middle[0, 0].subgridspec(1, 2, width_ratios=[.07, .93], wspace=.0)
    b_slot, right_slot = b_wrap[0, 1], middle[0, 1]
    right_grid = _shared_right_validation_grid(fig, right_slot)
    return {
        "a": a_slot,
        "b": b_slot,
        "c": right_grid[0, 0],
        "d": right_grid[1, 0],
        "right": right_slot,
        "right_grid": right_grid,
    }


def _composite_panel_boxes(layout: dict) -> dict[str, Any]:
    """Measure the four major panel boxes from the native composite GridSpec.

    This is the single geometry authority for both the full figure and every
    ``--panel`` export.  Returning a measured box rather than maintaining
    hand-tuned standalone heights prevents visual-size drift when the full
    publication layout changes.
    """
    width, height = _composite_canvas_size(layout)
    probe = plt.figure(figsize=(width, height), facecolor="white")
    try:
        slots = _master_panel_slots(probe, layout)
        return {name: slots[name].get_position(probe) for name in ("a", "b", "c", "d")}
    finally:
        plt.close(probe)


def _standalone_panel_size(layout: dict, panel: str) -> tuple[float, float]:
    """Return the exact width and height (inches) of a panel in the composite."""
    if panel in QUALITATIVE_SI_PANELS:
        return _qualitative_content_size(
            1, list(layout["qualitative"]["reconstruction_to_error_height"]),
        )
    box = _composite_panel_boxes(layout)[panel]
    width, height = _composite_canvas_size(layout)
    return width * box.width, height * box.height


def _standalone_panel_gutter(panel: str) -> dict[str, float]:
    """Return the label-safe outer gutter for a named standalone panel."""
    gutter_key = "a" if panel in QUALITATIVE_SI_PANELS else panel
    return {**STANDALONE_PANEL_GUTTER_IN, **STANDALONE_PANEL_GUTTER_OVERRIDES.get(gutter_key, {})}


def _standalone_panel_canvas(layout: dict, panel: str) -> tuple[float, float, tuple[float, float, float, float]]:
    """Return canvas size plus normalized slot bounds for one exact-size panel.

    The returned slot has the same physical dimensions as the corresponding
    full-composite box.  The enclosing canvas adds a fixed label-safe gutter,
    rather than applying margins *inside* the target panel box.
    """
    panel_width, panel_height = _standalone_panel_size(layout, panel)
    gutters = _standalone_panel_gutter(panel)
    left, right = gutters["left"], gutters["right"]
    bottom, top = gutters["bottom"], gutters["top"]
    canvas_width, canvas_height = panel_width + left + right, panel_height + bottom + top
    return (
        canvas_width,
        canvas_height,
        (left / canvas_width, 1.0 - right / canvas_width, bottom / canvas_height, 1.0 - top / canvas_height),
    )


def build_figure(cfg: dict, layout: dict, sources: dict, payloads: dict, selection: dict,
                 panel_d_truth_ensemble: dict,
                 panel_d_gpu_payload: dict,
                 value_limits: dict, error_limits: dict, panel: str = "all") -> plt.Figure:
    """Build the native canvas, preserving composite physical sizes in panel mode."""
    panel_d_rows, panel_d_summary = _panel_d_csv_rows(sources, panel_d_gpu_payload)
    width, height = _composite_canvas_size(layout)
    if panel != "all":
        # The content slot has exact composite dimensions; its outer canvas
        # includes only a label-safe gutter and therefore never rescales it.
        width, height, standalone_bounds = _standalone_panel_canvas(layout, panel)
    fig = plt.figure(figsize=(width, height), facecolor="white")
    if panel != "all":
        left, right, bottom, top = standalone_bounds
        slot = fig.add_gridspec(1, 1, left=left, right=right, top=top, bottom=bottom)[0, 0]
        if panel == "a" or panel in QUALITATIVE_SI_PANELS:
            qualitative_layout = layout
            qualitative_label = panel
            if panel in QUALITATIVE_SI_PANELS:
                qualitative_layout = deepcopy(layout)
                qualitative_layout["qualitative"]["fields"] = [QUALITATIVE_SI_PANELS[panel]["field"]]
                qualitative_label = QUALITATIVE_SI_PANELS[panel]["panel_label"]
            qualitative_header_y = draw_qualitative_panel(
                fig, slot, cfg, qualitative_layout, payloads, selection, value_limits, error_limits,
                header_offset_scale=height / float(fig.get_size_inches()[1]),
            )
        elif panel == "b": draw_l2_heatmap_panel(fig, slot, cfg, sources["field_l2"][1], layout)
        elif panel == "c": draw_spectral_panel(fig, slot, cfg, layout, sources["energy"][1], sources["lsd_summary"][1], sources["lsd_per"][1], selection)
        elif panel == "d": draw_jsd_panel(
            fig, slot, cfg, panel_d_rows, panel_d_summary,
            truth_ensemble=panel_d_truth_ensemble,
            precomputed_pdfs=panel_d_gpu_payload.get("pdfs", {}),
        )
        else: draw_joint_pdf_panel(fig, slot, cfg, layout, sources["pdf"][1], sources["pdf_metrics"][1])
        _panel_letter(
            fig, slot, qualitative_label if panel in QUALITATIVE_SI_PANELS else panel,
            y=qualitative_header_y if panel == "a" or panel in QUALITATIVE_SI_PANELS else None,
        )
        fig._publication_panel_canvas_inches = (width, height)  # type: ignore[attr-defined]
        fig._publication_panel_content_inches = _standalone_panel_size(layout, panel)  # type: ignore[attr-defined]
        return fig
    # One geometry authority creates fixed-size Panel a and an independently
    # allocated lower row.  This is the composition-side anti-distortion lock.
    slots = _master_panel_slots(fig, layout)
    a_slot = slots["a"]
    a_header_y = draw_qualitative_panel(
        fig, a_slot, cfg, layout, payloads, selection, value_limits, error_limits,
    )
    a_label = _panel_letter(fig, a_slot, "a", y=a_header_y)
    b_slot = slots["b"]
    draw_l2_heatmap_panel(fig, b_slot, cfg, sources["field_l2"][1], layout)
    _panel_letter(
        fig, b_slot, "b",
        x=a_label.get_position()[0] if PANEL_B_LAYOUT["align_panel_label_x_with_panel_a"] else None,
    )
    c_slot, d_slot = slots["c"], slots["d"]
    draw_spectral_panel(
        fig, c_slot, cfg, layout, sources["energy"][1], sources["lsd_summary"][1],
        sources["lsd_per"][1], selection,
    ); _panel_letter(fig, c_slot, "c")
    draw_jsd_panel(
        fig, d_slot, cfg, panel_d_rows, panel_d_summary,
        truth_ensemble=panel_d_truth_ensemble,
        precomputed_pdfs=panel_d_gpu_payload.get("pdfs", {}),
    ); _panel_letter(fig, d_slot, "d")
    # Explicitly validate the external-text corridor. Manual physical GridSpec
    # geometry is more deterministic here than mixing constrained_layout with
    # the fixed-canvas, nested publication composition.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    c_labels = getattr(fig, "_panel_c_bottom_xlabel_artists", [])
    d_titles = getattr(fig, "_panel_d_title_artists", [])
    if c_labels and d_titles:
        c_bottom_px = min(artist.get_window_extent(renderer).y0 for artist in c_labels)
        d_top_px = max(artist.get_window_extent(renderer).y1 for artist in d_titles)
        clearance_in = (c_bottom_px - d_top_px) / float(fig.dpi)
        if clearance_in < PANEL_C_D_TEXT_CLEARANCE_IN:
            raise ValueError(
                f"Panel c/d text clearance is {clearance_in:.3f} in; expected at least "
                f"{PANEL_C_D_TEXT_CLEARANCE_IN:.3f} in. Increase PANEL_C_D_GAP_IN."
            )
        fig._panel_c_d_text_clearance_in = clearance_in  # type: ignore[attr-defined]
    right_box = slots["right"].get_position(fig)
    b_box = b_slot.get_position(fig)
    right_axes = [
        ax for ax in fig.axes
        if (right_box.x0 <= (ax.get_position().x0 + ax.get_position().x1) / 2.0 <= right_box.x1
            and right_box.y0 <= (ax.get_position().y0 + ax.get_position().y1) / 2.0 <= right_box.y1)
    ]
    b_axes = [
        ax for ax in fig.axes
        if (b_box.x0 <= (ax.get_position().x0 + ax.get_position().x1) / 2.0 <= b_box.x1
            and b_box.y0 <= (ax.get_position().y0 + ax.get_position().y1) / 2.0 <= b_box.y1)
    ]
    panel_letters = getattr(fig, "_panel_letter_artists", {})
    b_figure_artists = [panel_letters["b"]] if "b" in panel_letters else []
    right_figure_artists = [
        *d_titles,
        *(panel_letters[label] for label in ("c", "d") if label in panel_letters),
    ]

    def translate_artists(axes: list, figure_artists: list, shift_in: float) -> None:
        """Rigidly translate artists right without changing any dimensions."""
        shift_fraction = float(shift_in) / float(fig.get_size_inches()[0])
        for ax in axes:
            box = ax.get_position()
            ax.set_position(
                [box.x0 + shift_fraction, box.y0, box.width, box.height], which="both",
            )
        for artist in figure_artists:
            artist.set_x(float(artist.get_position()[0]) + shift_fraction)

    def measured_right_canvas_clearance_in() -> float | None:
        right_extents = [ax.get_tightbbox(renderer).x1 for ax in right_axes]
        right_extents.extend(
            artist.get_window_extent(renderer).x1 for artist in right_figure_artists
        )
        if not right_extents:
            return None
        return (float(fig.bbox.width) - max(right_extents)) / float(fig.dpi)

    def measured_lower_left_clearance_in() -> float | None:
        left_extents = [ax.get_tightbbox(renderer).x0 for ax in b_axes]
        left_extents.extend(
            artist.get_window_extent(renderer).x0 for artist in b_figure_artists
        )
        if not left_extents:
            return None
        return min(left_extents) / float(fig.dpi)

    # Stage 1: if larger Panel-b method labels cross the page edge, translate
    # the complete lower row together. This preserves the designed b-to-c/d
    # gap and consumes only genuinely unused right margin.
    lower_left_clearance_in = measured_lower_left_clearance_in()
    right_canvas_clearance_in = measured_right_canvas_clearance_in()
    lower_shift_in = 0.0
    if (lower_left_clearance_in is not None
            and lower_left_clearance_in < LOWER_ROW_CANVAS_CLEARANCE_IN):
        required_shift_in = (
            LOWER_ROW_CANVAS_CLEARANCE_IN - lower_left_clearance_in
            + float(AUTO_REFLOW_SAFETY_PAD_IN)
        )
        available_shift_in = (
            (right_canvas_clearance_in or 0.0) - RIGHT_COLUMN_CANVAS_CLEARANCE_IN
        )
        if AUTO_REFLOW_RIGHT_COLUMN_FOR_FONTS and required_shift_in <= available_shift_in:
            lower_shift_in = required_shift_in
            translate_artists(
                [*b_axes, *right_axes], [*b_figure_artists, *right_figure_artists],
                lower_shift_in,
            )
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            lower_left_clearance_in = measured_lower_left_clearance_in()
            right_canvas_clearance_in = measured_right_canvas_clearance_in()
        else:
            raise ValueError(
                f"Lower-row left clearance is {lower_left_clearance_in:.3f} in; "
                f"{required_shift_in:.3f} in of reflow is needed but only "
                f"{max(available_shift_in, 0.0):.3f} in is available. Reduce the font "
                "or increase the fixed canvas/right margin."
            )
    if lower_left_clearance_in is not None:
        fig._lower_row_left_canvas_clearance_in = lower_left_clearance_in  # type: ignore[attr-defined]
    fig._lower_row_font_autoshift_in = lower_shift_in  # type: ignore[attr-defined]

    # Stage 2: wider Panel-d method labels may still need extra room relative
    # to Panel b. Move only c/d, retaining the just-corrected Panel-b position.
    violin_axes = getattr(fig, "_panel_d_violin_axes", [])
    method_labels = (
        [label for label in violin_axes[0].get_yticklabels() if label.get_text()]
        if violin_axes else []
    )

    def measured_horizontal_clearance_in() -> float | None:
        if not method_labels:
            return None
        label_left_px = min(label.get_window_extent(renderer).x0 for label in method_labels)
        b_right_px = (
            max(ax.get_position().x1 for ax in b_axes) * float(fig.bbox.width)
            if b_axes else b_box.x1 * float(fig.bbox.width)
        )
        return (label_left_px - b_right_px) / float(fig.dpi)

    # Apply the deliberate composition-level spacing refinement before any
    # font-driven reflow.  Only c/d move; Panel b remains benchmark-locked.
    right_canvas_clearance_in = measured_right_canvas_clearance_in()
    if RIGHT_COLUMN_MANUAL_SHIFT_IN < 0:
        raise ValueError("RIGHT_COLUMN_MANUAL_SHIFT_IN must be non-negative.")
    available_manual_shift_in = (
        (right_canvas_clearance_in or 0.0) - RIGHT_COLUMN_CANVAS_CLEARANCE_IN
    )
    if RIGHT_COLUMN_MANUAL_SHIFT_IN > available_manual_shift_in + 1.0e-9:
        raise ValueError(
            f"Requested {RIGHT_COLUMN_MANUAL_SHIFT_IN:.3f} in right-column shift, "
            f"but only {max(available_manual_shift_in, 0.0):.3f} in is available."
        )
    if RIGHT_COLUMN_MANUAL_SHIFT_IN:
        translate_artists(right_axes, right_figure_artists, RIGHT_COLUMN_MANUAL_SHIFT_IN)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
    horizontal_clearance_in = measured_horizontal_clearance_in()
    right_canvas_clearance_in = measured_right_canvas_clearance_in()
    right_shift_in = 0.0
    if (horizontal_clearance_in is not None
            and horizontal_clearance_in < PANEL_B_TO_RIGHT_TEXT_CLEARANCE_IN):
        required_shift_in = (
            PANEL_B_TO_RIGHT_TEXT_CLEARANCE_IN - horizontal_clearance_in
            + float(AUTO_REFLOW_SAFETY_PAD_IN)
        )
        available_shift_in = (
            (right_canvas_clearance_in or 0.0) - RIGHT_COLUMN_CANVAS_CLEARANCE_IN
        )
        if AUTO_REFLOW_RIGHT_COLUMN_FOR_FONTS and required_shift_in <= available_shift_in:
            right_shift_in = required_shift_in
            translate_artists(right_axes, right_figure_artists, right_shift_in)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            horizontal_clearance_in = measured_horizontal_clearance_in()
            right_canvas_clearance_in = measured_right_canvas_clearance_in()
        else:
            raise ValueError(
                f"Panel-b/Panel-d label clearance is {horizontal_clearance_in:.3f} in; "
                f"{required_shift_in:.3f} in of rightward reflow is needed but only "
                f"{max(available_shift_in, 0.0):.3f} in is available. Reduce the font, "
                "increase the fixed canvas/right margin, or enable autoreflow."
            )
    if horizontal_clearance_in is not None:
        if horizontal_clearance_in + 1.0e-6 < PANEL_B_TO_RIGHT_TEXT_CLEARANCE_IN:
            raise ValueError("Automatic right-column reflow did not achieve the required label clearance.")
        fig._panel_b_to_right_text_clearance_in = horizontal_clearance_in  # type: ignore[attr-defined]
    fig._right_column_font_autoshift_in = right_shift_in  # type: ignore[attr-defined]
    fig._right_column_manual_shift_in = RIGHT_COLUMN_MANUAL_SHIFT_IN  # type: ignore[attr-defined]
    if right_canvas_clearance_in is not None:
        if right_canvas_clearance_in < RIGHT_COLUMN_CANVAS_CLEARANCE_IN:
            raise ValueError(
                f"Right-column canvas clearance is {right_canvas_clearance_in:.3f} in; "
                f"expected at least {RIGHT_COLUMN_CANVAS_CLEARANCE_IN:.3f} in."
            )
        fig._right_column_canvas_clearance_in = right_canvas_clearance_in  # type: ignore[attr-defined]
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=SCRIPT_DIR / "publication_layout_coupled_field.yaml")
    parser.add_argument("--panel", choices=["all", "a", "b", "c", "d", *QUALITATIVE_SI_PANELS], default="all",
                        help="Render one major panel, an automatically assigned qualitative SI field, or the A--D composite.")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--output-id", help="Hourly publication-export suffix (default: YYYYMMDD_HHMM).")
    args = parser.parse_args()
    cfg, rid = load_config(args.config), run_id(args.run_id)
    layout = yaml.safe_load(args.layout.read_text(encoding="utf-8")) or {}
    layout["_run_id"] = rid
    _apply_revision_overrides(layout)
    apply_style(cfg)
    # Keep the archived baseline reproducible while allowing narrowly scoped,
    # declarative Panel-a revisions in dedicated layout files.
    panel_a_overrides = dict(layout.get("panel_a_layout_overrides", {}))
    allowed_panel_a_overrides = {
        "content_height_in", "prediction_error_hspace", "colorbar_extend",
        "colorbar_title_loc", "master_lower_gap_in", "grid_wspace",
        "sit_enabled_grid_wspace", "composite_right_pad_ratio",
    }
    unsupported_panel_a_overrides = set(panel_a_overrides) - allowed_panel_a_overrides
    if unsupported_panel_a_overrides:
        raise ValueError(
            "Unsupported panel_a_layout_overrides: "
            + ", ".join(sorted(unsupported_panel_a_overrides))
        )
    PANEL_A_LAYOUT.update(panel_a_overrides)
    if PANEL_A_LAYOUT["colorbar_extend"] not in {"neither", "both", "min", "max"}:
        raise ValueError("Panel-a colorbar_extend must be neither, both, min, or max.")
    if PANEL_A_LAYOUT["colorbar_title_loc"] not in {"left", "center", "right"}:
        raise ValueError("Panel-a colorbar_title_loc must be left, center, or right.")
    # PANEL_A_LAYOUT is the single field-allocation authority. Applying it
    # immediately after YAML loading prevents a stale layout file from
    # silently restoring an outdated Panel-a/SI field split.
    layout["qualitative"]["fields"] = list(PANEL_A_LAYOUT["main_fields"])
    sources = _load_sources(args.run_id)
    payloads, selection, cache_paths = _load_qualitative_payloads(cfg, rid, layout)
    panel_d_truth_ensemble: dict = {}
    panel_d_ensemble_metadata: dict = {}
    panel_d_gpu_payload: dict = {"rows": [], "summary": [], "pdfs": {}, "metadata": []}
    if args.panel in {"all", "d"}:
        panel_d_subplots = _panel_d_specification("main")["subplots"]
        panel_d_truth_ensemble, panel_d_ensemble_metadata = _load_panel_d_truth_ensemble(
            rid, selection["proposed"], panel_d_subplots,
        )
        panel_d_gpu_payload = _compute_panel_d_gpu_payload(
            cfg, rid, selection["proposed"], panel_d_subplots,
        )
    qualitative_field_keys = (
        [QUALITATIVE_SI_PANELS[args.panel]["field"]]
        if args.panel in QUALITATIVE_SI_PANELS else layout["qualitative"]["fields"]
    )
    fields = [_field_lookup(cfg)[key] for key in qualitative_field_keys]
    value_limits, error_limits = _qualitative_limits(
        payloads, fields, list(layout["qualitative"].get("value_percentiles", [1.0, 99.0])),
        float(layout["qualitative"]["robust_error_percentile"]),
    )
    fig = build_figure(
        cfg, layout, sources, payloads, selection, panel_d_truth_ensemble,
        panel_d_gpu_payload,
        value_limits, error_limits, args.panel,
    )
    formats = args.formats or layout["export"]["formats"]
    output_id = args.output_id or datetime.now().strftime("%Y%m%d_%H%M")
    # Keep the output tree tidy: each major panel and the composite receive a
    # dedicated directory below ``Assembled`` rather than accumulating all
    # formats/runs in one flat folder.
    panel_names = {"a": "Panel_a_Qualitative", "b": "Panel_b_FieldL2", "c": "Panel_c_Spectral", "d": "Panel_d_ChannelCouplingJSD"}
    output_directories = {"all": "Composite", **panel_names,
                          **{key: spec["output_directory"] for key, spec in QUALITATIVE_SI_PANELS.items()}}
    base_name = (QUALITATIVE_SI_PANELS[args.panel]["output_name"] if args.panel in QUALITATIVE_SI_PANELS
                 else panel_names[args.panel] if args.panel != "all" else layout["name"])
    base = FIGURES_DIR / "Assembled" / output_directories[args.panel] / f"{base_name}_{output_id}"
    outputs = _save_figure(fig, base, formats, args.dpi or int(layout["export"]["png_dpi"]))
    retained_panel_d_ticks = getattr(fig, "_panel_d_retained_log_ticks", {})
    panel_d_pdf_frame_counts = getattr(fig, "_panel_d_pdf_frame_counts", {})
    panel_c_d_text_clearance_in = getattr(fig, "_panel_c_d_text_clearance_in", None)
    panel_b_to_right_text_clearance_in = getattr(fig, "_panel_b_to_right_text_clearance_in", None)
    right_column_canvas_clearance_in = getattr(fig, "_right_column_canvas_clearance_in", None)
    right_column_font_autoshift_in = getattr(fig, "_right_column_font_autoshift_in", 0.0)
    right_column_manual_shift_in = getattr(fig, "_right_column_manual_shift_in", 0.0)
    lower_row_font_autoshift_in = getattr(fig, "_lower_row_font_autoshift_in", 0.0)
    lower_row_left_canvas_clearance_in = getattr(fig, "_lower_row_left_canvas_clearance_in", None)
    canvas_size_inches = [float(value) for value in fig.get_size_inches()]
    plt.close(fig)
    right_column_alignment = None
    if args.panel == "all":
        boxes = _composite_panel_boxes(layout)
        right_column_alignment = {
            "c_to_d_height_ratio": float(boxes["c"].height / boxes["d"].height),
            "b_top_minus_c_top": float(boxes["b"].y1 - boxes["c"].y1),
            "b_bottom_minus_d_bottom": float(boxes["b"].y0 - boxes["d"].y0),
        }
    manifest = {
        "run_id": rid, "generated_at_utc": datetime.now(timezone.utc).isoformat(), "layout": str(args.layout),
        "panel": args.panel, "csv_paths": {name: str(path) for name, (path, _) in sources.items()},
        "composition_panels": ["a", "b", "c", "d"],
        "qualitative_geometry_lock": {
            "panel_a_main_fields": list(PANEL_A_LAYOUT["main_fields"]),
            "qualitative_si_fields": {
                key: spec["field"] for key, spec in QUALITATIVE_SI_PANELS.items()
            },
            "panel_a_field_value_colormaps": dict(COLORMAP_CONFIG["panel_a"]["field_values"]),
            "panel_a_absolute_error_colormap": COLORMAP_CONFIG["panel_a"]["absolute_error"],
            "panel_a_actual_min_rounded_fields": list(PANEL_A_VALUE_LIMITS["actual_min_rounded_fields"]),
            "panel_a_actual_min_significant_digits": PANEL_A_VALUE_LIMITS["actual_min_significant_digits"],
            "content_width_in": PANEL_A_LAYOUT["content_width_in"],
            "content_height_in": PANEL_A_LAYOUT["content_height_in"],
            "subplot_width_to_height": PANEL_A_LAYOUT["subplot_width_to_height"],
            "master_top_margin_in": PANEL_A_LAYOUT["master_top_margin_in"],
            "master_lower_gap_in": PANEL_A_LAYOUT["master_lower_gap_in"],
            "column_header_gap_above_grid_in": PANEL_A_LAYOUT["column_header_gap_above_grid_in"],
            "group_to_column_header_gap_in": PANEL_A_LAYOUT["group_to_column_header_gap_in"],
            "group_header_baseline_above_slot_in": PANEL_A_LAYOUT["group_header_baseline_above_slot_in"],
            "top_text_safety_in": PANEL_A_LAYOUT["top_text_safety_in"],
            "prediction_error_hspace": PANEL_A_LAYOUT["prediction_error_hspace"],
            "panel_a_active_grid_wspace": (
                PANEL_A_LAYOUT["sit_enabled_grid_wspace"]
                if selection["sit_comparison_enabled"] else PANEL_A_LAYOUT["grid_wspace"]
            ),
            "panel_a_composite_right_pad_ratio": PANEL_A_LAYOUT["composite_right_pad_ratio"],
            "colorbar_extend": PANEL_A_LAYOUT["colorbar_extend"],
            "colorbar_title_loc": PANEL_A_LAYOUT["colorbar_title_loc"],
            "field_group_spacer_ratio": PANEL_A_LAYOUT["field_group_spacer_ratio"],
            "sensor_style": {
                "size": PANEL_A_LAYOUT["sensor_marker_size"],
                "facecolor": PANEL_A_LAYOUT["sensor_facecolor"],
                "edgecolor": PANEL_A_LAYOUT["sensor_edgecolor"],
                "linewidth": PANEL_A_LAYOUT["sensor_linewidth"],
                "alpha": PANEL_A_LAYOUT["sensor_alpha"],
            },
            "l2_annotation": "LaTeX mantissa-times-ten with rounded dark alpha box",
            "colorbar_multiplier": "anchored colorbar title",
            "automatic_header_band_compaction": True,
            "panel_c_d_height_ratios": list(PANEL_C_D_HEIGHT_RATIOS),
            "panel_c_d_gap_in": PANEL_C_D_GAP_IN,
            "panel_c_internal_height_ratios": list(PANEL_C_INTERNAL_HEIGHT_RATIOS),
            "panel_c_grid_hspace": PANEL_C_LAYOUT["grid_hspace"],
            "panel_c_legend_bbox_to_anchor": list(PANEL_C_LAYOUT["legend_bbox_to_anchor"]),
            "panel_c_legend_layout": {
                "mode": PANEL_C_LAYOUT["legend_mode"],
                "columns": PANEL_C_LAYOUT["legend_ncol"],
                "frame": False,
                "horizontal_justification": "full panel width",
            },
            "panel_c_d_measured_text_clearance_in": panel_c_d_text_clearance_in,
            "panel_b_to_right_text_clearance_in": panel_b_to_right_text_clearance_in,
            "right_column_canvas_clearance_in": right_column_canvas_clearance_in,
            "right_column_font_autoshift_in": right_column_font_autoshift_in,
            "right_column_manual_shift_in": right_column_manual_shift_in,
            "lower_row_font_autoshift_in": lower_row_font_autoshift_in,
            "lower_row_left_canvas_clearance_in": lower_row_left_canvas_clearance_in,
            "right_column_font_autoreflow_enabled": AUTO_REFLOW_RIGHT_COLUMN_FOR_FONTS,
            "panel_b_to_right_gap_in": PANEL_WSPACE_IN,
            "panel_c_field_selection": layout["spectral"].get("field_selection", "legacy_fields"),
            "panel_c_fields": _panel_c_field_keys(layout),
            "panel_c_field_options": layout["spectral"].get("field_options", {}),
            "panel_c_scatter_visual_trim": {
                "upper_percentile_exclusive": PANEL_C_LAYOUT["scatter_upper_percentile"],
                "applies_to": "scatter artists only",
                "bars_errorbars_and_statistics": "complete untrimmed data",
            },
            "panel_b_colormap": COLORMAP_CONFIG["panel_b"]["selected"],
            "panel_b_colormap_options": list(COLORMAP_CONFIG["panel_b"]["options"]),
            "panel_b_bottom_xtick_label_pad": PANEL_B_LAYOUT["bottom_xtick_label_pad"],
            "panel_b_condition_hspace": PANEL_B_LAYOUT["condition_hspace"],
            "panel_b_label_x_aligned_with_panel_a": PANEL_B_LAYOUT["align_panel_label_x_with_panel_a"],
            "panel_d_ground_truth_pdf_row": True,
            "panel_d_grid_rows": ["0.8-height ground-truth PDFs", "expanded log-JSD violins"],
            "panel_d_pdf_height_to_width": PANEL_D_STYLE["pdf_height_scale"],
            "panel_d_pdf_colormap": COLORMAP_CONFIG["panel_d"]["selected"],
            "panel_d_pdf_colormap_options": list(COLORMAP_CONFIG["panel_d"]["options"]),
            "panel_d_pdf_zero_density_color": COLORMAP_CONFIG["panel_d"]["zero_density_color"],
            "panel_d_pdf_positive_vmin_floor": COLORMAP_CONFIG["panel_d"]["positive_vmin_floor"],
            "panel_d_pdf_numeric_ticks": False,
            "panel_d_pdf_ensemble": panel_d_ensemble_metadata,
            "panel_d_pdf_frame_counts": panel_d_pdf_frame_counts,
            "panel_d_middle_pair_option": dict(PANEL_D_MIDDLE_PAIR_OPTION),
            "panel_d_gpu_on_the_fly": panel_d_gpu_payload.get("metadata", []),
            "panel_d_axis_flip": {
                "display_pair": "p-U1",
                "metric_pair": "U1-p",
                "jsd_recomputed": False,
            },
            "panel_d_title_policy": "math-only; baseline aligned with panel label d",
            "panel_d_violin_xscale": "log with retained positive linear levels",
            "panel_d_violin_scatter_underlay": False,
            "panel_d_mean_safe_zone_axes_fraction": {
                "violin_data_ends_by": PANEL_D_STYLE["violin_data_right_fraction"],
                "mean_text_starts_at": PANEL_D_STYLE["mean_x_axes"],
            },
            "panel_d_x_label_pad_pt": PANEL_D_STYLE["x_label_pad"],
            "panel_d_row_hspace": PANEL_D_STYLE["row_hspace"],
            "panel_d_dmf_first_at_top": PANEL_D_STYLE["dmf_first_at_top"],
            "panel_d_condition_xlabels": dict(PANEL_D_STYLE["condition_xlabels"]),
            "fixed_canvas_export": True,
        },
        "global_style": {
            "version": GLOBAL_STYLE_VERSION,
            "module": "global_style.py",
            "font_sizes_pt": FONT_SIZES,
            "line_widths_pt": LINE_WIDTHS,
            "model_colors": MODEL_COLORS,
            "model_alphas": MODEL_ALPHAS,
            "strict_composite_width_in": COMPOSITE_WIDTH_IN,
            "composite_bottom_margin_in": COMPOSITE_MARGIN_BOTTOM_IN,
            "composite_left_margin_in": COMPOSITE_MARGIN_LEFT_IN,
            "composite_right_margin_in": COMPOSITE_MARGIN_RIGHT_IN,
            "adaptive_composite_height": args.panel == "all",
        },
        "panel_e_policy": "excluded from the primary publication composition; retained in existing supplementary outputs",
        "canvas_size_inches": canvas_size_inches,
        "right_column_alignment": right_column_alignment,
        "panel_d_retained_log_ticks": retained_panel_d_ticks,
        "composite_panel_content_size_inches": (
            [float(value) for value in _standalone_panel_size(layout, args.panel)] if args.panel != "all" else None
        ),
        "standalone_gutter_inches": (_standalone_panel_gutter(args.panel) if args.panel != "all" else None),
        "cache_paths": cache_paths, "checkpoint_selection": cfg["defaults"]["checkpoint"],
        "snapshot_index": layout["qualitative"]["snapshot_index"], "conditions": layout["qualitative"]["conditions"],
        "representative_models": selection, "fields": qualitative_field_keys,
        "field_value_limits": value_limits, "robust_error_limits": error_limits,
        "qualitative_colourbar_policy": "two display ticks; 0.0 and rounded 0.5/integer upper level; scientific multiplier anchored as each bar title",
        "qualitative_value_contour_lines": {
            "enabled": PANEL_A_LAYOUT["show_value_contour_lines"],
            "levels": PANEL_A_LAYOUT["value_contour_line_levels"],
            "color": PANEL_A_LAYOUT["value_contour_line_color"],
            "linewidth": PANEL_A_LAYOUT["value_contour_line_width"],
            "alpha": PANEL_A_LAYOUT["value_contour_line_alpha"],
            "applies_to": ["truth", "reconstruction"],
        },
        "x_compression": layout["qualitative"]["x_compression"],
        "display_only_transform": f"x_plot=x_min+{layout['qualitative']['x_compression']}*(x-x_min)",
        "timestamp_format": "%Y%m%d_%H%M", "output_id": output_id,
        "joint_pdf_snapshot_mode": layout["joint_pdf"].get("snapshot_mode", "single_fixed"),
        "joint_pdf_snapshot_indices": layout.get("_joint_selected_snapshots", []),
        "sit_validation": str(RESULTS_DIR / "Validation" / f"SiT_metric_validation_{rid}.json"),
        "panel_d_coupling_validation": {
            "selection_key": "main",
            "subplots": _panel_d_specification("main")["subplots"],
            "pair_interpretations": {
                pair: label.replace("\n", "; ") for pair, label in COUPLING_PAIR_LABELS.items()
            },
            "metric": "base-2 Jensen–Shannon divergence of fixed-bin physical-unit joint PDFs",
            "data_policy": (
                "finalized T-U1 and U1-p values plus transient read-only CUDA CH4-U1; "
                "p-U1 visually reuses symmetric U1-p JSD"
            ),
        },
        # When formats are exported in separate low-memory passes, retain the
        # complete set of already-present companion artefacts in the manifest.
        "output_formats": [suffix[1:] for suffix in (".pdf", ".svg", ".png") if base.with_suffix(suffix).is_file()],
        "outputs": [str(base.with_suffix(suffix)) for suffix in (".pdf", ".svg", ".png") if base.with_suffix(suffix).is_file()],
    }
    (base.parent / f"{base.name}_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] native publication figure: {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

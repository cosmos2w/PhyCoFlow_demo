"""Portable manuscript-wide Matplotlib style contract.

Copy this single file beside plotting scripts in another chapter and import it
directly.  It deliberately has no PhyCoFlow/project imports, no data paths, and
no dependence on a YAML configuration.  Scientific choices (normalizations,
statistics, field colormaps, bold/italic emphasis, and context-specific text
colors) remain in each panel renderer; only reusable visual identity lives here.

Recommended use
---------------
    from global_style import apply_global_style, model_alpha, model_color

    apply_global_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)
    ax.plot(x, y, color=model_color("DMF-Gen"), alpha=model_alpha("DMF-Gen"),
            lw=LW_LINE_PLOT)

For dense nested GridSpecs, use explicit physical margins/gaps plus
``validate_text_within_canvas``.  Matplotlib's ``constrained_layout`` is useful
for ordinary standalone axes but should not be mixed with manually positioned
publication composites.  ``bbox_inches='tight'`` is appropriate for flexible
standalone exports; fixed-width composites must validate their bboxes and save
the declared canvas unchanged.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

GLOBAL_STYLE_VERSION = "1.6.0"

# =============================================================================
# MANUSCRIPT FIGURE USER TUNING API
# =============================================================================
# These are the manuscript-wide controls that should normally be adjusted
# before touching an individual drawer.  Dimensions are physical: changing a
# panel size changes both its standalone page and its composite allocation.
# The detailed a--f rectangles remain in publication_layout_unified_v2.yaml so
# they can be audited without importing Python.
COMPOSITE_CANVAS_WIDTH_MM = 183.0
DEFAULT_INTER_ROW_GAP_MM = 0.707550614
MINIMUM_SAFE_GAP_MM = 0.60
PANEL_LABEL_X = 0.005
PANEL_LABEL_Y = 0.995
PANEL_LABEL_HA = "left"
PANEL_LABEL_VA = "top"

# Shared semantic colormaps.  Model colors remain categorical and are defined
# below; these names govern physical fields and quantitative scalar maps.
CMAP_PHYSICAL_FIELD = "viridis"
CMAP_SIGNED_COMPONENT = "RdBu_r"
CMAP_SIGNED_RESIDUAL = "PuOr_r"
CMAP_CORRELATION = "cividis"
CMAP_SIGNED_BIAS = "RdBu_r"

# Layout safeguards.  Fixed-canvas exports are rejected rather than silently
# cropped, compressed, or stretched when an artist leaves its declared page.
PHYSICAL_SIZE_TOLERANCE_MM = 0.02
CANVAS_BOUNDARY_TOLERANCE_MM = 0.02
# =============================================================================
# END MANUSCRIPT FIGURE USER TUNING API
# =============================================================================

# ---------------------------------------------------------------------------
# TYPOGRAPHY
# ---------------------------------------------------------------------------
# Arial is the manuscript font.  Bold, italic, and non-default text colors are
# intentionally NOT global: individual panel functions apply them only where
# they carry semantic meaning.
FONT_FAMILY = "Arial"
FONT_FALLBACKS = ("Liberation Sans", "DejaVu Sans")

# Explicit hierarchy in points.  These values are legible on a 7.2-in
# double-column page while remaining compact enough for dense multi-panels.
SIZE_PANEL_LABEL = 8.5       # a/b/c/d tags; largest
SIZE_SUBPLOT_TITLE = 6.5     # individual chart or image-column titles
SIZE_AXIS_LABEL = 6.0        # x/y labels and error-bar/metric titles
SIZE_TICK_LABEL = 5.5        # numeric/category ticks
SIZE_LEGEND = 5.5            # shared or per-axis legends
SIZE_ANNOTATION = 5.5        # in-plot values, missing labels, L2, means

FONT_SIZES = {
    "size_panel_label": SIZE_PANEL_LABEL,
    "size_subplot_title": SIZE_SUBPLOT_TITLE,
    "size_axis_label": SIZE_AXIS_LABEL,
    "size_tick_label": SIZE_TICK_LABEL,
    "size_legend": SIZE_LEGEND,
    "size_annotation": SIZE_ANNOTATION,
}

# Every visible text artist in a publication figure is assigned exactly one of
# these semantic roles.  Panel drawers may position or weight text locally, but
# they must not invent a new point size or shrink standalone/composite output
# differently.  ``enforce_figure_typography`` applies and audits this contract
# after all nested axes, colorbars, legends, and annotations have been created.
FONT_ROLE_SIZES = {
    "panel_label": SIZE_PANEL_LABEL,
    "subplot_title": SIZE_SUBPLOT_TITLE,
    "axis_label": SIZE_AXIS_LABEL,
    "tick_label": SIZE_TICK_LABEL,
    "legend": SIZE_LEGEND,
    "annotation": SIZE_ANNOTATION,
}
FONT_ROLE_GID_PREFIX = "font-role:"


def tag_font_role(artist, role: str, *, size_pt: float | None = None):
    """Tag text with a semantic role and, optionally, a local resolved size.

    ``size_pt`` is the supported escape hatch for dense, explicitly laid-out
    panels.  The semantic role still supplies the manuscript default, while a
    positive local value survives :func:`enforce_figure_typography` and is
    recorded by its QA payload.  Fixed-canvas validation remains authoritative,
    so a local increase cannot silently clip the exported page.
    """
    if role not in FONT_ROLE_SIZES:
        raise ValueError(f"Unknown font role {role!r}; choose from {sorted(FONT_ROLE_SIZES)}")
    if size_pt is not None and float(size_pt) <= 0:
        raise ValueError(f"size_pt must be positive, got {size_pt!r}")
    setattr(artist, "_global_font_role", role)
    if size_pt is not None:
        setattr(artist, "_global_font_size_pt", float(size_pt))
    if not artist.get_gid():
        artist.set_gid(f"{FONT_ROLE_GID_PREFIX}{role}")
    return artist


def _figure_axes_recursive(fig):
    """Return every figure and nested inset axis exactly once."""
    axes = []
    seen = set()

    def collect(ax):
        if id(ax) in seen:
            return
        seen.add(id(ax))
        axes.append(ax)
        for child in getattr(ax, "child_axes", []):
            collect(child)

    for ax in fig.axes:
        collect(ax)
    return axes


def enforce_figure_typography(fig, *, tolerance_pt: float = 1.0e-6) -> dict:
    """Apply the global font hierarchy to every visible figure text artist.

    Matplotlib-generated titles, axis labels, tick/offset labels, and legends
    are classified structurally.  Free text defaults to ``annotation`` unless
    its drawer explicitly calls :func:`tag_font_role`.  The returned payload is
    suitable for the source manifest and proves that no ad-hoc point size
    survived the final render.
    """
    from matplotlib.legend import Legend

    roles = {}

    def assign(artist, role):
        if artist is not None:
            roles[id(artist)] = (artist, role)

    # Explicit semantic tags take precedence over structural defaults.
    for artist in fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text)):
        explicit_role = getattr(artist, "_global_font_role", None)
        if explicit_role is not None:
            assign(artist, explicit_role)
            continue
        gid = str(artist.get_gid() or "")
        if gid.startswith(FONT_ROLE_GID_PREFIX):
            assign(artist, gid[len(FONT_ROLE_GID_PREFIX):])

    for ax in _figure_axes_recursive(fig):
        for title in (ax.title, ax._left_title, ax._right_title):
            if id(title) not in roles:
                assign(title, "subplot_title")
        for label in (ax.xaxis.label, ax.yaxis.label):
            if id(label) not in roles:
                assign(label, "axis_label")
        for axis in (ax.xaxis, ax.yaxis):
            if id(axis.offsetText) not in roles:
                assign(axis.offsetText, "tick_label")
            for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                for label in (tick.label1, tick.label2):
                    if id(label) not in roles:
                        assign(label, "tick_label")

    for legend in fig.findobj(match=lambda item: isinstance(item, Legend)):
        for text in legend.get_texts():
            if id(text) not in roles:
                assign(text, "legend")
        if legend.get_title() is not None and id(legend.get_title()) not in roles:
            assign(legend.get_title(), "legend")

    for artist in fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text)):
        if id(artist) not in roles:
            assign(artist, "annotation")

    counts = {role: 0 for role in FONT_ROLE_SIZES}
    examples = {role: [] for role in FONT_ROLE_SIZES}
    violations = []
    for artist, role in roles.values():
        if role not in FONT_ROLE_SIZES:
            violations.append({"text": artist.get_text(), "role": role, "error": "unknown role"})
            continue
        expected = float(getattr(
            artist, "_global_font_size_pt", FONT_ROLE_SIZES[role],
        ))
        artist.set_fontsize(expected)
        artist.set_fontfamily("sans-serif")
        if artist.get_visible() and artist.get_text():
            counts[role] += 1
            if len(examples[role]) < 8:
                examples[role].append(artist.get_text())
            observed = float(artist.get_fontsize())
            if abs(observed - expected) > float(tolerance_pt):
                violations.append({
                    "text": artist.get_text(), "role": role,
                    "expected_pt": expected, "observed_pt": observed,
                })
    result = {
        "passed": not violations,
        "role_sizes_pt": dict(FONT_ROLE_SIZES),
        "local_size_overrides": [
            {
                "text": artist.get_text(),
                "role": role,
                "size_pt": float(artist._global_font_size_pt),
            }
            for artist, role in roles.values()
            if artist.get_visible() and artist.get_text()
            and hasattr(artist, "_global_font_size_pt")
        ],
        "visible_text_counts": counts,
        "examples": examples,
        "violations": violations,
    }
    if violations:
        raise ValueError(f"Global typography contract failed: {violations[:12]}")
    return result

# ---------------------------------------------------------------------------
# MODEL COLOR IDENTITY
# ---------------------------------------------------------------------------
# Paul-Tol/Okabe-Ito-inspired, colorblind-friendly categorical palette.  The
# main SciML method uses a dark saturated blue; baselines occupy distinct hue
# families.  Never reassign these colors panel-by-panel.

# Version-1
# MODEL_COLORS = {
#     "DMF-Gen": "#004488",        # primary/high-contrast manuscript method
#     "FFM-Perceiver": "#88CCEE",  # light blue
#     "FFM-FNO": "#EE7733",        # orange
#     "Latent FM": "#AA4499",      # purple
#     "SiT": "#CC6677",            # rose; transformer/diffusion baseline
#     "Senseiver": "#999999",      # neutral gray
#     "Geo-FNO": "#DDCC77",        # sand/yellow
#     "MLP-RBF": "#44AA99",        # teal
# }

# Verison-2: more saturated, more contrast, more modern
MODEL_COLORS = {
    # 主模型：极具冲击力和现代感的帝国红，在任何背景下都能第一时间抓住眼球
    "DMF-Gen": "#E63946",       
    # FFM 家族：采用冷静、理性的深浅蓝色系
    "FFM-Perceiver": "#457B9D", # 钢蓝色 (Steel Blue)
    "FFM-FNO": "#1D3557",       # 普鲁士深蓝 (Prussian Blue)
    # 隐空间与 Diffusion/Transformer 家族：采用具有未来感的紫色系
    "Latent FM": "#6A4C93",     # 深紫 (Deep Purple)
    # "SiT": "#9D4EDD",           # 亮紫 (Bright Violet)
    "SiT": "#A28BC4",           # 丁香紫 (Lilac)
    # 纯算子与注意力 Baseline：大地色与青色
    "Geo-FNO": "#F4A261",       # 沙橙色 (Sandy Orange)，比原来的黄色在白底上更清晰
    "Senseiver": "#8D99AE",     # 冷灰 (Cool Gray)，比纯中性灰更显高级且不易被忽视
    # 传统 ML / 局部网络
    "MLP-RBF": "#2A9D8F",       # 深青绿 (Teal/Green)
}

# Transparency hierarchy for artists that directly represent a model. Apply
# this to model-colored curves, bars, violins, and matching legend swatches.
# Do not apply it to physical-field maps, quantitative heatmaps, ground truth,
# error bars, sensor overlays, or grid lines: those retain their own encoding.
ALPHA_DMF_GEN = 0.85
ALPHA_BASELINE = 0.70
MODEL_ALPHAS = {
    "DMF-Gen": ALPHA_DMF_GEN,
    "baseline_default": ALPHA_BASELINE,
}

MODEL_ALIASES = {
    "dmfgen": "DMF-Gen",
    "dmf_gen": "DMF-Gen",
    "ffmperceiver": "FFM-Perceiver",
    "ffm_perceiver": "FFM-Perceiver",
    "ffmfno": "FFM-FNO",
    "ffm_fno": "FFM-FNO",
    "latentfm": "Latent FM",
    "latent_fm": "Latent FM",
    "dit": "DiT-based",
    "dit-based": "DiT-based",
    "mlprbf": "MLP-RBF",
    "mlp_rbf": "MLP-RBF",
    "geofno": "Geo-FNO",
}

COLOR_GROUND_TRUTH = "#202020"
COLOR_AXIS = "#202020"
COLOR_GRID = "#C8C8C8"
COLOR_DIVIDER = "#B8B8B8"
COLOR_MISSING_FACE = "#D8D8D8"
COLOR_MISSING_TEXT = "#606060"


def canonical_model_name(name: str) -> str:
    """Return the stable palette key for a display name/directory/alias."""
    raw = str(name).strip()
    if raw in MODEL_COLORS:
        return raw
    normalized = raw.lower().replace(" ", "_")
    return MODEL_ALIASES.get(normalized, MODEL_ALIASES.get(raw.lower(), raw))


def model_color(name: str, fallback: str = "#777777") -> str:
    """Return one manuscript-wide model color without project dependencies."""
    return MODEL_COLORS.get(canonical_model_name(name), fallback)


def model_alpha(name: str) -> float:
    """Return the manuscript alpha for DMF-Gen or a baseline model."""
    return ALPHA_DMF_GEN if canonical_model_name(name) == "DMF-Gen" else ALPHA_BASELINE


def standardize_model_colors(methods: Iterable[dict]) -> None:
    """Update configuration method dictionaries with global colors/alphas.

    This compatibility bridge lets legacy scripts continue reading
    ``method['color']`` while ensuring that every chapter uses this palette.
    """
    for method in methods:
        identity = next(
            (
                method.get(field)
                for field in ("name", "label", "key", "directory")
                if method.get(field)
            ),
            "",
        )
        method["color"] = model_color(identity, method.get("color", "#777777"))
        method["alpha"] = model_alpha(identity)


# ---------------------------------------------------------------------------
# LINE WEIGHTS (points)
# ---------------------------------------------------------------------------
LW_AXIS_SPINE = 1.0
LW_LINE_PLOT = 1.6
LW_LINE_SECONDARY = 0.9
LW_DIVIDER = 0.65
LW_GRID = 0.45
LW_ERRORBAR = 0.8

LINE_WIDTHS = {
    "lw_axis_spine": LW_AXIS_SPINE,
    "lw_line_plot": LW_LINE_PLOT,
    "lw_divider": LW_DIVIDER,
}

# ---------------------------------------------------------------------------
# COMPOSITE DIMENSIONS AND PHYSICAL SPACING (inches unless noted)
# ---------------------------------------------------------------------------
# 7.2 in = 182.88 mm, the standard journal double-column width.  This width is
# strict and must never be inferred from content or changed by tight cropping.
COMPOSITE_WIDTH_IN = 7.2
COMPOSITE_DEFAULT_HEIGHT_IN = 9.1
SINGLE_COLUMN_WIDTH_IN = 3.5

# Major canvas margins.  The left/right values reproduce the approved coupled-
# field baseline while keeping long lower-panel labels inside the page.
COMPOSITE_MARGIN_TOP_IN = 0.15
COMPOSITE_MARGIN_BOTTOM_IN = 0.32
COMPOSITE_MARGIN_LEFT_IN = 0.324
COMPOSITE_MARGIN_RIGHT_IN = 0.0936
LOWER_ROW_MARGIN_LEFT_IN = 0.396
LOWER_ROW_MARGIN_RIGHT_IN = 0.324

# Major panel gaps.
PANEL_WSPACE_IN = 0.44        # label-safe gap between b and the c/d column
PANEL_HSPACE_IN = 0.25        # vertical gap between a and the lower row

# Reusable internal gaps.  GridSpec's wspace/hspace are fractions of average
# axes size, so these are dimensionless by design and named accordingly.
INTRA_PANEL_WSPACE = 0.18
INTRA_PANEL_HSPACE = 0.10
# The right validation column gives the denser coupling evidence in Panel d
# 60% of the usable height and the spectral evidence in Panel c 40%. A physical
# gap keeps Panel-c's bottom xlabel clear of Panel-d's external math headers.
PANEL_C_D_HEIGHT_RATIOS = (0.40, 0.60)
PANEL_C_D_GAP_IN = 0.50
PANEL_C_D_TEXT_CLEARANCE_IN = 0.04
# Panel-c has a shallow shared legend row followed by strict 1:1 bar/spectrum
# evidence rows. The final two entries must remain equal.
PANEL_C_INTERNAL_HEIGHT_RATIOS = (0.13, 1.0, 1.0)
# Deprecated compatibility name; new assemblers use PANEL_C_D_GAP_IN.
PANEL_C_TO_D_SPACER_RATIO = 0.24

# Content-height contract.  Composite height is calculated from these physical
# rows, the major vertical gap, and top/bottom margins.  Changing/adding a row
# therefore changes height automatically while width remains exactly 7.2 in.
DEFAULT_PANEL_A_HEIGHT_IN = 2.7473449352967423
GENERIC_PANEL_ROW_HEIGHT_IN = 2.20
DEFAULT_LOWER_ROW_HEIGHT_IN = (
    COMPOSITE_DEFAULT_HEIGHT_IN
    - COMPOSITE_MARGIN_TOP_IN
    - COMPOSITE_MARGIN_BOTTOM_IN
    - PANEL_HSPACE_IN
    - DEFAULT_PANEL_A_HEIGHT_IN
)

# Flexible standalone exports may use tight bounding boxes with this controlled
# pad.  Fixed composites instead use bbox validation and no tight cropping.
EXPORT_TIGHT_PAD_IN = 0.04
TEXT_BBOX_SAFETY_IN = 0.002
# Minimum physical padding for constrained fixed-canvas composites.  This is
# deliberately larger than the bbox safety tolerance so tick labels and
# colorbar text remain inside the declared page without tight-crop expansion.
FIXED_CANVAS_LAYOUT_PAD_IN = 0.070
# Font discovery and MathText metrics vary slightly across otherwise compatible
# Matplotlib environments.  Constrained fixed-canvas figures may absorb only
# small metric drift; larger overflow remains a hard layout error.
FIXED_CANVAS_AUTOFIT_LIMIT_IN = 0.08
FIXED_CANVAS_AUTOFIT_GUARD_IN = 0.004
FIXED_CANVAS_AUTOFIT_PASSES = 3


def adaptive_composite_height(
    row_heights_in: Iterable[float],
    *,
    row_gaps_in: Iterable[float] | None = None,
    top_margin_in: float = COMPOSITE_MARGIN_TOP_IN,
    bottom_margin_in: float = COMPOSITE_MARGIN_BOTTOM_IN,
) -> float:
    """Return compact canvas height from physical content rows and gaps.

    ``row_heights_in`` contains one height per major horizontal band.  If
    ``row_gaps_in`` is omitted, ``PANEL_HSPACE_IN`` is inserted between every
    pair.  This is deterministic and transferable across chapters.
    """
    rows = [float(value) for value in row_heights_in]
    if not rows or any(value <= 0 for value in rows):
        raise ValueError("row_heights_in must contain positive physical heights.")
    gaps = ([PANEL_HSPACE_IN] * (len(rows) - 1) if row_gaps_in is None
            else [float(value) for value in row_gaps_in])
    if len(gaps) != len(rows) - 1 or any(value < 0 for value in gaps):
        raise ValueError("row_gaps_in must contain one non-negative gap between rows.")
    return float(top_margin_in) + sum(rows) + sum(gaps) + float(bottom_margin_in)


def gridspec_space_from_inches(total_in: float, gap_in: float, count: int) -> float:
    """Convert an exact physical gap to Matplotlib GridSpec wspace/hspace."""
    if count <= 1:
        return 0.0
    usable = float(total_in) - float(gap_in) * (count - 1)
    if usable <= 0:
        raise ValueError("Requested gaps leave no room for GridSpec axes.")
    average_axis = usable / count
    return float(gap_in) / average_axis


# ---------------------------------------------------------------------------
# APPLICATION, FONT CHECKS, AND BBOX-SAFE EXPORT
# ---------------------------------------------------------------------------
def arial_available() -> bool:
    """Return True only when Matplotlib can resolve an actual Arial font."""
    try:
        path = font_manager.findfont(FONT_FAMILY, fallback_to_default=False)
    except ValueError:
        return False
    return "arial" in Path(path).name.lower()


def apply_global_style(*, strict_arial: bool = False) -> None:
    """Apply the complete manuscript rcParams contract.

    When Arial is unavailable, the declared family remains Arial but a
    metrically compatible fallback is appended so batch rendering succeeds.
    Use ``strict_arial=True`` in final submission CI to fail fast instead.
    """
    has_arial = arial_available()
    if strict_arial and not has_arial:
        raise RuntimeError("Arial is required but not installed/resolvable by Matplotlib.")
    if not has_arial:
        warnings.warn(
            "Arial is unavailable; using Liberation Sans/DejaVu Sans fallback. "
            "Install Arial before final manuscript submission.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Matplotlib otherwise routes $...$ labels through its DejaVu math font,
    # even when all normal text is Arial.  Bind MathText to the same family so
    # species names, subscripts, exponents, and panel prose remain coherent.
    active_math_font = FONT_FAMILY if has_arial else FONT_FALLBACKS[0]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT_FAMILY, *FONT_FALLBACKS],
        "font.size": SIZE_TICK_LABEL,
        "axes.titlesize": SIZE_SUBPLOT_TITLE,
        "axes.labelsize": SIZE_AXIS_LABEL,
        "xtick.labelsize": SIZE_TICK_LABEL,
        "ytick.labelsize": SIZE_TICK_LABEL,
        "legend.fontsize": SIZE_LEGEND,
        "mathtext.fontset": "custom",
        "mathtext.rm": active_math_font,
        "mathtext.it": f"{active_math_font}:italic",
        "mathtext.bf": f"{active_math_font}:bold",
        "mathtext.sf": active_math_font,
        "mathtext.cal": active_math_font,
        "mathtext.tt": active_math_font,
        "axes.linewidth": LW_AXIS_SPINE,
        "axes.edgecolor": COLOR_AXIS,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": LW_AXIS_SPINE,
        "ytick.major.width": LW_AXIS_SPINE,
        "lines.linewidth": LW_LINE_PLOT,
        "grid.linewidth": LW_GRID,
        "grid.color": COLOR_GRID,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "svg.fonttype": "none",  # editable SVG text
        "pdf.fonttype": 42,      # editable embedded TrueType text
        "ps.fonttype": 42,
    })


def _visible_text_bboxes(fig) -> list[tuple[matplotlib.text.Text, object]]:
    """Collect visible Text artists and rendered bboxes, including titles."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for artist in fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text)):
        if not artist.get_visible() or not artist.get_text():
            continue
        # ``Axes.set_axis_off()`` suppresses its axis/tick artists at draw time,
        # although Matplotlib may keep those Text objects individually visible
        # and populate them after an intermediate canvas draw. They are not
        # rendered content and must not create false canvas-overflow failures.
        hidden_axis_tick = False
        for candidate in _figure_axes_recursive(fig):
            if candidate.axison:
                continue
            axis_text = [
                candidate.xaxis.label, candidate.yaxis.label,
                candidate.xaxis.offsetText, candidate.yaxis.offsetText,
                *candidate.get_xticklabels(minor=False),
                *candidate.get_xticklabels(minor=True),
                *candidate.get_yticklabels(minor=False),
                *candidate.get_yticklabels(minor=True),
            ]
            if any(item is artist for item in axis_text):
                hidden_axis_tick = True
                break
        if hidden_axis_tick:
            continue
        box = artist.get_window_extent(renderer)
        # Log locators can retain tick-label artists for off-range decades.
        # Their anchor can be outside the declared canvas even when the glyph
        # bbox grazes it.  A clipped artist anchored outside contributes no
        # intended boundary text and is therefore not a layout violation.
        canvas = fig.bbox
        anchor = artist.get_transform().transform(artist.get_position())
        anchor_outside = not (canvas.x0 <= anchor[0] <= canvas.x1 and canvas.y0 <= anchor[1] <= canvas.y1)
        if artist.get_clip_on() and anchor_outside:
            continue
        wholly_outside = box.x1 <= canvas.x0 or box.x0 >= canvas.x1 or box.y1 <= canvas.y0 or box.y0 >= canvas.y1
        if artist.get_clip_on() and wholly_outside:
            continue
        if box.width > 0 and box.height > 0:
            boxes.append((artist, box))
    return boxes


def validate_text_within_canvas(
    fig,
    *,
    safety_in: float = TEXT_BBOX_SAFETY_IN,
    raise_on_error: bool = True,
) -> Mapping[str, float]:
    """Validate all boundary text against the declared fixed canvas.

    This is the fixed-canvas alternative to ``bbox_inches='tight'``: it catches
    clipping without changing the strict 7.2-in composite width.  Returned
    overflow values are in inches and are zero on a passing figure.
    """
    dpi = float(fig.dpi)
    pad_px = float(safety_in) * dpi
    canvas = fig.bbox
    artist_boxes = _visible_text_bboxes(fig)
    if not artist_boxes:
        return {"left": 0.0, "right": 0.0, "bottom": 0.0, "top": 0.0}
    boxes = [box for _, box in artist_boxes]
    overflow = {
        "left": max(0.0, max(pad_px - box.x0 for box in boxes) / dpi),
        "right": max(0.0, max(box.x1 + pad_px - canvas.x1 for box in boxes) / dpi),
        "bottom": max(0.0, max(pad_px - box.y0 for box in boxes) / dpi),
        "top": max(0.0, max(box.y1 + pad_px - canvas.y1 for box in boxes) / dpi),
    }
    if raise_on_error and any(value > 0 for value in overflow.values()):
        offenders = []
        for artist, box in artist_boxes:
            edges = []
            if box.x0 < pad_px: edges.append("left")
            if box.x1 + pad_px > canvas.x1: edges.append("right")
            if box.y0 < pad_px: edges.append("bottom")
            if box.y1 + pad_px > canvas.y1: edges.append("top")
            if edges:
                axes = getattr(artist, "axes", None)
                if axes is None:
                    for candidate in _figure_axes_recursive(fig):
                        axis_text = [
                            candidate.xaxis.label, candidate.yaxis.label,
                            candidate.xaxis.offsetText, candidate.yaxis.offsetText,
                            *candidate.get_xticklabels(minor=False),
                            *candidate.get_xticklabels(minor=True),
                            *candidate.get_yticklabels(minor=False),
                            *candidate.get_yticklabels(minor=True),
                        ]
                        if any(item is artist for item in axis_text):
                            axes = candidate
                            break
                axes_box = tuple(round(value, 4) for value in axes.get_position().bounds) if axes is not None else None
                offenders.append(
                    f"{artist.get_text()!r} ({'/'.join(edges)}; position={artist.get_position()}; "
                    f"clip_on={artist.get_clip_on()}; axes={axes_box}; "
                    f"axes_gid={axes.get_gid() if axes is not None else None!r})"
                )
        raise ValueError(
            f"Text extends outside the declared canvas (inches): {overflow}; "
            f"offending artists: {', '.join(offenders[:8])}"
        )
    return overflow


def fit_text_within_canvas(
    fig,
    *,
    safety_in: float = TEXT_BBOX_SAFETY_IN,
    overflow_limit_in: float = FIXED_CANVAS_AUTOFIT_LIMIT_IN,
    guard_in: float = FIXED_CANVAS_AUTOFIT_GUARD_IN,
    max_passes: int = FIXED_CANVAS_AUTOFIT_PASSES,
) -> Mapping[str, object]:
    """Repair small fallback-font bbox drift without changing page size.

    Only figures using Matplotlib's constrained-layout engine are eligible.
    Each pass adds the measured horizontal/vertical overflow plus a small
    physical guard to the corresponding layout pad, redraws, and validates
    again.  A large overflow, an incompatible layout engine, or a residual
    overflow after ``max_passes`` still raises through the strict validator.

    This preserves the manuscript canvas and keeps real layout defects visible
    while making exports reproducible across minor Matplotlib/font revisions.
    """
    if max_passes < 0:
        raise ValueError("max_passes must be non-negative.")
    if overflow_limit_in < 0 or guard_in < 0:
        raise ValueError("overflow_limit_in and guard_in must be non-negative.")

    initial = {
        edge: float(value)
        for edge, value in validate_text_within_canvas(
            fig, safety_in=safety_in, raise_on_error=False,
        ).items()
    }
    overflow = dict(initial)
    if not any(overflow.values()):
        return {
            "adjusted": False,
            "passes": 0,
            "initial_overflow_in": initial,
            "final_overflow_in": overflow,
        }

    largest = max(overflow.values())
    engine = fig.get_layout_engine()
    is_constrained = (
        engine is not None
        and engine.__class__.__name__ == "ConstrainedLayoutEngine"
        and hasattr(engine, "get")
    )
    if largest > float(overflow_limit_in) or not is_constrained or max_passes == 0:
        validate_text_within_canvas(fig, safety_in=safety_in, raise_on_error=True)

    settings = engine.get()
    w_pad = float(settings.get("w_pad", FIXED_CANVAS_LAYOUT_PAD_IN))
    h_pad = float(settings.get("h_pad", FIXED_CANVAS_LAYOUT_PAD_IN))
    wspace = float(settings.get("wspace", 0.02))
    hspace = float(settings.get("hspace", 0.02))
    passes = 0
    for passes in range(1, int(max_passes) + 1):
        largest = max(overflow.values())
        if largest > float(overflow_limit_in):
            break
        horizontal_overflow = max(overflow["left"], overflow["right"])
        vertical_overflow = max(overflow["bottom"], overflow["top"])
        if horizontal_overflow > 0:
            w_pad += horizontal_overflow + float(guard_in)
        if vertical_overflow > 0:
            h_pad += vertical_overflow + float(guard_in)
        fig.set_constrained_layout_pads(
            w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace,
        )
        fig.canvas.draw()
        overflow = {
            edge: float(value)
            for edge, value in validate_text_within_canvas(
                fig, safety_in=safety_in, raise_on_error=False,
            ).items()
        }
        if not any(overflow.values()):
            warnings.warn(
                "Adjusted constrained-layout padding to absorb small "
                f"font-metric bbox drift (passes={passes}, "
                f"w_pad={w_pad:.4f} in, h_pad={h_pad:.4f} in).",
                RuntimeWarning,
                stacklevel=2,
            )
            return {
                "adjusted": True,
                "passes": passes,
                "initial_overflow_in": initial,
                "final_overflow_in": overflow,
                "w_pad_in": w_pad,
                "h_pad_in": h_pad,
            }

    validate_text_within_canvas(fig, safety_in=safety_in, raise_on_error=True)
    raise AssertionError("Unreachable after strict text-bbox validation.")


def save_publication_figure(
    fig,
    base: Path,
    formats: Iterable[str] = ("svg", "pdf", "png"),
    *,
    dpi: int = 600,
    fixed_canvas: bool = False,
    validate_bbox: bool = True,
) -> list[Path]:
    """Export editable vector files plus a high-resolution preview.

    Flexible standalone plots use tight bbox logic.  Fixed composites preserve
    exact width/height and validate text before saving, preventing silent page
    growth or clipped boundary annotations.
    """
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    if fixed_canvas and validate_bbox:
        fit_text_within_canvas(fig)
    outputs = []
    for extension in dict.fromkeys(str(value).lower().lstrip(".") for value in formats):
        path = base.with_suffix(f".{extension}")
        kwargs = {
            "dpi": dpi,
            "facecolor": fig.get_facecolor(),
            "transparent": False,
        }
        if not fixed_canvas:
            kwargs.update({"bbox_inches": "tight", "pad_inches": EXPORT_TIGHT_PAD_IN})
        fig.savefig(path, **kwargs)
        outputs.append(path)
    return outputs


__all__ = [name for name in globals() if name.isupper()] + [
    "adaptive_composite_height",
    "apply_global_style",
    "arial_available",
    "canonical_model_name",
    "fit_text_within_canvas",
    "gridspec_space_from_inches",
    "model_alpha",
    "model_color",
    "save_publication_figure",
    "standardize_model_colors",
    "validate_text_within_canvas",
]

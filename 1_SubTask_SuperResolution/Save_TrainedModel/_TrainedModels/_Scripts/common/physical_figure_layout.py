"""Exact physical layout contract shared by standalone and composite figures.

The central invariant is simple: a panel is drawn into the same millimetre
rectangle everywhere.  Standalone exports use that rectangle as their page;
the composite places the identical rectangle at a calculated position.  No
constrained-layout engine, tight crop, box-aspect rescaling, or raster panel
placement is allowed in this workflow.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import matplotlib
import matplotlib.pyplot as plt

import global_style as manuscript


# =============================================================================
# PHYSICAL LAYOUT USER TUNING API
# =============================================================================
# The YAML defines rows, not redundant panel rectangles. Each row supplies one
# height, one horizontal gap, and optional width fractions. Panel widths and the
# total canvas height are derived. Legacy explicit rectangles remain readable.
RENDER_PROFILE = "publication-identical"
SIZE_TOLERANCE_MM = manuscript.PHYSICAL_SIZE_TOLERANCE_MM
BOUNDARY_TOLERANCE_MM = manuscript.CANVAS_BOUNDARY_TOLERANCE_MM
PANEL_LABEL_POSITION = (manuscript.PANEL_LABEL_X, manuscript.PANEL_LABEL_Y)
# =============================================================================
# END PHYSICAL LAYOUT USER TUNING API
# =============================================================================


@dataclass(frozen=True)
class PanelRect:
    """One panel-container rectangle in millimetres from canvas lower-left."""

    left_mm: float
    bottom_mm: float
    width_mm: float
    height_mm: float

    @property
    def right_mm(self) -> float:
        return self.left_mm + self.width_mm

    @property
    def top_mm(self) -> float:
        return self.bottom_mm + self.height_mm

    def as_dict(self) -> dict[str, float]:
        return {
            "left_mm": self.left_mm,
            "right_mm": self.right_mm,
            "bottom_mm": self.bottom_mm,
            "top_mm": self.top_mm,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
        }


@dataclass(frozen=True)
class PhysicalLayout:
    width_mm: float
    height_mm: float
    rows: tuple[tuple[str, ...], ...]
    minimum_row_gap_mm: float
    row_gaps_mm: tuple[float, ...]
    column_gaps_mm: Mapping[str, float]
    panels: Mapping[str, PanelRect]
    content_bounds: Mapping[str, tuple[float, float, float, float]]

    def manifest(self) -> dict:
        row_width_fractions = {
            "_".join(row): [
                self.panels[label].width_mm
                / sum(self.panels[item].width_mm for item in row)
                for label in row
            ]
            for row in self.rows if len(row) > 1
        }
        return {
            "layout_engine": "explicit millimetre rectangles",
            "geometry_definition": "row height + gap + normalized width fractions",
            "render_profile": RENDER_PROFILE,
            "canvas_width_mm": self.width_mm,
            "canvas_height_mm": self.height_mm,
            "canvas_height_mode": "sum(panel row heights + explicit row gaps)",
            "rows_top_to_bottom": [list(row) for row in self.rows],
            "minimum_row_gap_mm": self.minimum_row_gap_mm,
            "row_gaps_mm": list(self.row_gaps_mm),
            "column_gaps_mm": dict(self.column_gaps_mm),
            "row_width_fractions_excluding_gaps": row_width_fractions,
            "bottom_row_width_ratios": row_width_fractions.get("e_f"),
            "panel_rectangles_mm": {
                label: rect.as_dict() for label, rect in self.panels.items()
            },
            "content_bounds_panel_fraction": {
                label: list(bounds) for label, bounds in self.content_bounds.items()
            },
            "standalone_composite_size_tolerance_mm": SIZE_TOLERANCE_MM,
            "implicit_scaling": False,
            "tight_crop": False,
        }


def _pair(value, *, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{label} must contain [width_mm, height_mm]")
    width, height = map(float, value)
    if width <= 0 or height <= 0:
        raise ValueError(f"{label} must contain positive dimensions")
    return width, height


def _normalized_fractions(values, count: int, *, label: str) -> tuple[float, ...]:
    """Return positive fractions normalized to one; ratios need not sum to one."""
    if values is None:
        return tuple(1.0 / count for _ in range(count))
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f"{label} must contain one positive value per panel")
    raw = tuple(float(value) for value in values)
    if any(value <= 0 for value in raw):
        raise ValueError(f"{label} values must all be positive")
    total = sum(raw)
    return tuple(value / total for value in raw)


def _resolve_derived_rows(spec: dict, width_mm: float):
    """Resolve the preferred easy-to-tune row schema."""
    row_specs = spec.get("rows")
    if not isinstance(row_specs, list) or not row_specs or not all(isinstance(row, dict) for row in row_specs):
        return None
    height_scale = float(spec.get("row_height_scale", 1.0))
    if height_scale <= 0:
        raise ValueError("physical_layout.row_height_scale must be positive")
    minimum_row_gap = float(spec.get("minimum_row_gap_mm", 0.0))
    if minimum_row_gap < 0:
        raise ValueError("physical_layout.minimum_row_gap_mm cannot be negative")
    default_row_gap = float(spec.get("row_gap_mm", manuscript.DEFAULT_INTER_ROW_GAP_MM))
    if default_row_gap < minimum_row_gap:
        raise ValueError(
            f"physical_layout.row_gap_mm must be >= minimum_row_gap_mm={minimum_row_gap:.3f} mm"
        )

    rows = []
    sizes = {}
    row_heights = []
    row_gaps = []
    column_gaps = {}
    for row_index, row_spec in enumerate(row_specs):
        row = tuple(str(label) for label in row_spec.get("panels", ()))
        if not row:
            raise ValueError(f"physical_layout.rows[{row_index}].panels cannot be empty")
        if len(row) != len(set(row)) or any(label in sizes for label in row):
            raise ValueError(f"Panel labels must occur once; duplicate found in row {row_index}: {row}")
        height = float(row_spec["height_mm"]) * height_scale
        if height <= 0:
            raise ValueError(f"physical_layout.rows[{row_index}].height_mm must be positive")
        horizontal_gap = float(row_spec.get("horizontal_gap_mm", 0.0))
        if horizontal_gap < 0:
            raise ValueError(f"physical_layout.rows[{row_index}].horizontal_gap_mm cannot be negative")
        usable_width = width_mm - horizontal_gap * (len(row) - 1)
        if usable_width <= 0:
            raise ValueError(
                f"Row {row} gap={horizontal_gap:g} mm leaves no usable width on a {width_mm:g} mm canvas"
            )
        fractions = _normalized_fractions(
            row_spec.get("width_fractions"), len(row),
            label=f"physical_layout.rows[{row_index}].width_fractions",
        )
        for label, fraction in zip(row, fractions):
            sizes[label] = (usable_width * fraction, height)
        rows.append(row)
        row_heights.append(height)
        column_gaps["_".join(row)] = horizontal_gap
        if row_index < len(row_specs) - 1:
            gap_after = float(row_spec.get("gap_after_mm", default_row_gap))
            if gap_after < minimum_row_gap:
                raise ValueError(
                    f"physical_layout.rows[{row_index}].gap_after_mm must be >= "
                    f"minimum_row_gap_mm={minimum_row_gap:.3f} mm"
                )
            row_gaps.append(gap_after)
    return tuple(rows), sizes, row_heights, minimum_row_gap, tuple(row_gaps), column_gaps


def _resolve_legacy_rows(v2: dict, spec: dict, width_mm: float):
    """Read the former explicit-size schema for older saved layout files."""
    rows = tuple(tuple(str(label) for label in row) for row in spec["rows"])
    if not rows or any(not row for row in rows):
        raise ValueError("physical_layout.rows must contain non-empty rows")
    labels = [label for row in rows for label in row]
    if len(labels) != len(set(labels)):
        raise ValueError("Each panel label must appear exactly once in physical_layout.rows")
    raw_sizes = spec["panel_sizes_mm"]
    sizes = {label: _pair(raw_sizes[label], label=f"panel_sizes_mm.{label}") for label in labels}
    compatibility_sizes = v2.get("figure", {}).get("standalone_sizes_mm", {})
    for label in labels:
        if label not in compatibility_sizes:
            continue
        mirrored = _pair(compatibility_sizes[label], label=f"figure.standalone_sizes_mm.{label}")
        if any(abs(left - right) > SIZE_TOLERANCE_MM for left, right in zip(sizes[label], mirrored)):
            raise ValueError(
                f"figure.standalone_sizes_mm.{label} must mirror physical_layout.panel_sizes_mm.{label}: "
                f"{mirrored} != {sizes[label]}"
            )
    minimum_row_gap = float(spec.get("minimum_row_gap_mm", manuscript.MINIMUM_SAFE_GAP_MM))
    row_gaps = tuple(float(value) for value in spec["row_gaps_mm"])
    if len(row_gaps) != len(rows) - 1 or any(value < minimum_row_gap for value in row_gaps):
        raise ValueError(
            "physical_layout.row_gaps_mm must provide one gap per adjacent row, "
            f"each >= {minimum_row_gap:.3f} mm"
        )
    column_gaps = {str(key): float(value) for key, value in spec.get("column_gaps_mm", {}).items()}
    row_heights = []
    for row in rows:
        heights = [sizes[label][1] for label in row]
        if max(heights) - min(heights) > SIZE_TOLERANCE_MM:
            raise ValueError(f"Panels sharing row {row} must have identical physical heights: {heights}")
        row_heights.append(max(heights))
        gap = column_gaps.get("_".join(row), 0.0)
        occupied = sum(sizes[label][0] for label in row) + gap * (len(row) - 1)
        if abs(occupied - width_mm) > SIZE_TOLERANCE_MM:
            raise ValueError(
                f"Legacy row {row} occupies {occupied:.6f} mm, not canvas width {width_mm:.6f} mm. "
                "Migrate to row mappings with height_mm, horizontal_gap_mm, and width_fractions."
            )
    return rows, sizes, row_heights, minimum_row_gap, row_gaps, column_gaps


def resolve_physical_layout(v2: dict, *, width_override_mm: float | None = None) -> PhysicalLayout:
    """Validate YAML geometry and resolve all top-to-bottom panel rectangles."""
    spec = v2.get("physical_layout")
    if not isinstance(spec, dict):
        raise ValueError("publication layout requires a physical_layout mapping")
    width_mm = float(width_override_mm or spec.get("canvas_width_mm", manuscript.COMPOSITE_CANVAS_WIDTH_MM))
    if width_mm <= 0:
        raise ValueError("physical_layout.canvas_width_mm must be positive")
    resolved = _resolve_derived_rows(spec, width_mm)
    if resolved is None:
        resolved = _resolve_legacy_rows(v2, spec, width_mm)
    rows, sizes, row_heights, minimum_row_gap, row_gaps, column_gaps = resolved
    labels = [label for row in rows for label in row]
    content_bounds = {}
    for label in labels:
        values = spec.get("content_bounds", {}).get(label, [0.0, 0.0, 1.0, 1.0])
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError(f"physical_layout.content_bounds.{label} must contain [left, bottom, width, height]")
        left, bottom, width, height = map(float, values)
        if left < 0 or bottom < 0 or width <= 0 or height <= 0 or left + width > 1 or bottom + height > 1:
            raise ValueError(f"Invalid normalized content bounds for panel {label}: {values}")
        content_bounds[label] = (left, bottom, width, height)

    height_mm = sum(row_heights) + sum(row_gaps)
    panels: dict[str, PanelRect] = {}
    top_mm = height_mm
    for row_index, row in enumerate(rows):
        row_height = row_heights[row_index]
        bottom_mm = top_mm - row_height
        x_mm = 0.0
        gap = column_gaps.get("_".join(row), 0.0)
        for label in row:
            panel_width, panel_height = sizes[label]
            panels[label] = PanelRect(x_mm, bottom_mm, panel_width, panel_height)
            x_mm += panel_width + gap
        top_mm = bottom_mm - (row_gaps[row_index] if row_index < len(row_gaps) else 0.0)
    if abs(top_mm) > SIZE_TOLERANCE_MM:
        raise AssertionError(f"Physical layout did not close at canvas bottom: {top_mm:.6g} mm")
    # Keep legacy panel metadata consumers functional without asking users to
    # duplicate width fractions elsewhere in the YAML.
    for row in rows:
        if len(row) <= 1:
            continue
        total_panel_width = sum(panels[label].width_mm for label in row)
        for label in row:
            panel_cfg = v2.get(f"panel_{label}")
            if isinstance(panel_cfg, dict):
                panel_cfg["width_fraction"] = panels[label].width_mm / total_panel_width
    return PhysicalLayout(
        width_mm, height_mm, rows, minimum_row_gap, row_gaps,
        column_gaps, panels, content_bounds,
    )


def create_composite_canvas(layout: PhysicalLayout):
    """Return fixed figure, content axes, and exact panel-container axes."""
    fig = plt.figure(figsize=(layout.width_mm / 25.4, layout.height_mm / 25.4), layout=None)
    axes, containers = {}, {}
    for label, rect in layout.panels.items():
        container = fig.add_axes([
            rect.left_mm / layout.width_mm,
            rect.bottom_mm / layout.height_mm,
            rect.width_mm / layout.width_mm,
            rect.height_mm / layout.height_mm,
        ], label=f"panel-{label}-container", frameon=False)
        container.set_axis_off()
        containers[label] = container
        axes[label] = container.inset_axes(
            layout.content_bounds[label], transform=container.transAxes,
        )
        axes[label].set_label(f"panel-{label}-content")
    return fig, axes, containers


def create_standalone_canvas(layout: PhysicalLayout, label: str):
    """Return the exact physical page used by the same panel in the composite."""
    rect = layout.panels[label]
    fig = plt.figure(figsize=(rect.width_mm / 25.4, rect.height_mm / 25.4), layout=None)
    container = fig.add_axes([0.0, 0.0, 1.0, 1.0], label=f"panel-{label}-container", frameon=False)
    container.set_axis_off()
    ax = container.inset_axes(layout.content_bounds[label], transform=container.transAxes)
    ax.set_label(f"panel-{label}-content")
    return fig, ax, container


def measure_axes_mm(fig, axes: Mapping[str, object]) -> dict[str, dict[str, float]]:
    """Measure panel parent axes in physical units after all artists are drawn."""
    fig.canvas.draw()
    measured = {}
    for label, ax in axes.items():
        bbox = ax.get_window_extent()
        measured[label] = PanelRect(
            bbox.x0 / fig.dpi * 25.4,
            bbox.y0 / fig.dpi * 25.4,
            bbox.width / fig.dpi * 25.4,
            bbox.height / fig.dpi * 25.4,
        ).as_dict()
    return measured


def validate_panel_text_boundaries(
    fig,
    containers: Mapping[str, object],
    *,
    tolerance_mm: float = BOUNDARY_TOLERANCE_MM,
) -> dict:
    """Ensure visible text stays inside its physical panel container.

    Once every panel's text is contained, non-negative row gaps cannot create
    cross-row text overlap. This lets users compact vertical gaps safely instead
    of relying on a conservative hard-coded minimum.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = float(fig.dpi)
    tolerance_px = float(tolerance_mm) / 25.4 * dpi
    text_type = matplotlib.text.Text
    owners: dict[int, str] = {}
    owned_text: dict[int, object] = {}

    # Axes.findobj recursively includes titles, tick labels, inset/colorbar
    # axes, and legends belonging to the panel container.
    for label, container in containers.items():
        for artist in container.findobj(match=lambda item: isinstance(item, text_type)):
            if not artist.get_visible() or not artist.get_text():
                continue
            owners.setdefault(id(artist), label)
            owned_text[id(artist)] = artist

    # Panel C uses figure-level text with the panel content transform. Recover
    # ownership by transform identity; the GID fallback keeps this explicit.
    axes_owner = {}
    transform_owner = {}
    for label, container in containers.items():
        pending = [container]
        while pending:
            ax = pending.pop()
            axes_owner[id(ax)] = label
            transform_owner[id(ax.transAxes)] = label
            transform_owner[id(ax.transData)] = label
            pending.extend(getattr(ax, "child_axes", []))
    for artist in fig.findobj(match=lambda item: isinstance(item, text_type)):
        if not artist.get_visible() or not artist.get_text() or id(artist) in owners:
            continue
        label = axes_owner.get(id(getattr(artist, "axes", None)))
        if label is None:
            label = transform_owner.get(id(artist.get_transform()))
        if label is None and str(artist.get_gid() or "").startswith("panel-c-"):
            label = "c"
        if label is not None:
            owners[id(artist)] = label
            owned_text[id(artist)] = artist

    violations = []
    text_boxes_by_panel: dict[str, list[tuple[object, object]]] = {
        label: [] for label in containers
    }
    for artist_id, artist in owned_text.items():
        label = owners[artist_id]
        hidden_axis_tick = False
        for candidate in containers[label].findobj(
            match=lambda item: isinstance(item, matplotlib.axes.Axes)
        ):
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
        if box.width <= 0 or box.height <= 0:
            continue
        panel_box = containers[label].get_window_extent(renderer)
        anchor = artist.get_transform().transform(artist.get_position())
        anchor_outside = not (
            panel_box.x0 <= anchor[0] <= panel_box.x1
            and panel_box.y0 <= anchor[1] <= panel_box.y1
        )
        wholly_outside = (
            box.x1 <= panel_box.x0 or box.x0 >= panel_box.x1
            or box.y1 <= panel_box.y0 or box.y0 >= panel_box.y1
        )
        # Log locators retain clipped labels for decades outside the displayed
        # range. They are not rendered panel content and must not constrain gaps.
        if artist.get_clip_on() and (anchor_outside or wholly_outside):
            continue
        edges = []
        if box.x0 < panel_box.x0 - tolerance_px: edges.append("left")
        if box.x1 > panel_box.x1 + tolerance_px: edges.append("right")
        if box.y0 < panel_box.y0 - tolerance_px: edges.append("bottom")
        if box.y1 > panel_box.y1 + tolerance_px: edges.append("top")
        if edges:
            violations.append({
                "panel": label,
                "text": artist.get_text(),
                "edges": edges,
                "overflow_mm": {
                    "left": max(0.0, panel_box.x0 - box.x0) / dpi * 25.4,
                    "right": max(0.0, box.x1 - panel_box.x1) / dpi * 25.4,
                    "bottom": max(0.0, panel_box.y0 - box.y0) / dpi * 25.4,
                    "top": max(0.0, box.y1 - panel_box.y1) / dpi * 25.4,
                },
            })
        text_boxes_by_panel[label].append((artist, box))

    cross_panel_overlaps = []
    labels = list(containers)
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1:]:
            for left_artist, left_box in text_boxes_by_panel[left_label]:
                for right_artist, right_box in text_boxes_by_panel[right_label]:
                    if left_box.overlaps(right_box):
                        cross_panel_overlaps.append({
                            "left_panel": left_label,
                            "left_text": left_artist.get_text(),
                            "right_panel": right_label,
                            "right_text": right_artist.get_text(),
                        })
    result = {
        "passed": not violations and not cross_panel_overlaps,
        "tolerance_mm": float(tolerance_mm),
        "owned_text_count": len(owned_text),
        "panel_boundary_violations": violations,
        "cross_panel_text_overlaps": cross_panel_overlaps,
    }
    if not result["passed"]:
        raise ValueError(f"Panel text clearance audit failed: {result}")
    return result


def validate_measured_geometry(
    layout: PhysicalLayout,
    measured: Mapping[str, Mapping[str, float]],
    *,
    labels: Iterable[str] | None = None,
) -> dict:
    """Reject any parent-axis compression, stretching, or displacement."""
    checked = list(labels or layout.panels)
    errors = []
    for label in checked:
        target = layout.panels[label].as_dict()
        observed = measured[label]
        for field in ("left_mm", "bottom_mm", "width_mm", "height_mm"):
            # A standalone page starts at (0, 0); only dimensions are shared.
            if len(checked) == 1 and field in {"left_mm", "bottom_mm"}:
                expected = 0.0
            else:
                expected = target[field]
            delta = abs(float(observed[field]) - expected)
            if delta > SIZE_TOLERANCE_MM:
                errors.append({
                    "panel": label, "field": field, "expected_mm": expected,
                    "observed_mm": float(observed[field]), "delta_mm": delta,
                })
    if errors:
        raise ValueError(f"Physical panel geometry changed during rendering: {errors}")
    return {"passed": True, "tolerance_mm": SIZE_TOLERANCE_MM, "errors": []}


def assert_standalone_matches_composite(
    layout: PhysicalLayout,
    standalone_sizes: Mapping[str, Mapping[str, float]],
) -> dict:
    """Check every standalone page against its composite rectangle size."""
    errors = []
    for label, rect in layout.panels.items():
        observed = standalone_sizes.get(label)
        if observed is None:
            errors.append({"panel": label, "reason": "missing standalone measurement"})
            continue
        for field, target in (("width_mm", rect.width_mm), ("height_mm", rect.height_mm)):
            delta = abs(float(observed[field]) - target)
            if delta > SIZE_TOLERANCE_MM:
                errors.append({
                    "panel": label, "field": field, "expected_mm": target,
                    "observed_mm": float(observed[field]), "delta_mm": delta,
                })
    return {"passed": not errors, "tolerance_mm": SIZE_TOLERANCE_MM, "errors": errors}

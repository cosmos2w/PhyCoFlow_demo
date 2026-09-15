#!/usr/bin/env python
"""Render Figure 4 art V4 from the validated V3 display and frozen data."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np


HERE = Path(__file__).resolve().parent
V3_PATH = HERE / "96_assemble_coupled_field_art_v3.py"
V3_PANEL_B_RATIO = 0.333333
V4_PANEL_B_RATIO = V3_PANEL_B_RATIO * 1.10
PANEL_C_GAP_MM = 8.0
PANEL_D_GAP_MM = 6.5
PANEL_TAG_LEFT_OFFSET_MM = 3.0
MIN_EXTERNAL_CLEARANCE_MM = 1.2
V3_PANEL_D_WIDTH_MM = 23.86445964666667
COLUMN_HEADER_SIZE_PT = 7.8


def _load_v3():
    spec = importlib.util.spec_from_file_location("coupled_field_art_v3_for_v4", V3_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V3 renderer: {V3_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v3 = _load_v3()
v2 = v3.v2
base = v3.base


def _apply_v4_contract() -> None:
    v3._apply_v3_contract()
    previous_overrides = base._apply_revision_overrides

    def v4_overrides(layout: dict) -> None:
        previous_overrides(layout)
        base.PANEL_D_STYLE.update({
            "violin_data_right_fraction": 0.74,
            "mean_x_axes": 0.75,
            "x_label_pad": 7.0,
        })

    base._apply_revision_overrides = v4_overrides


def _column_header_artists(fig) -> list:
    expected = {
        "T only", "T + $U_1$", "CO + T + $U_1$ + $p$",
        "FFM-FNO", "Latent FM", "SiT", "Senseiver",
    }
    return [artist for artist in fig.texts if artist.get_text() in expected]


def _set_panel_a_typography(fig) -> tuple[list[float], list[float]]:
    groups = [artist for artist in fig.texts if artist.get_text() in {
        "Conditioning progression (DMF-Gen)",
        "Baseline comparisons: conditioned on T only",
    }]
    headers = _column_header_artists(fig)
    if len(groups) != 2 or len(headers) != 7:
        raise RuntimeError(f"Unexpected Panel-a title inventory: {len(groups)} groups, {len(headers)} columns")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = float(fig.dpi)
    split_x = np.mean(sorted(float(artist.get_position()[0]) for artist in headers)[2:4])

    def members(group) -> list:
        x = float(group.get_position()[0])
        return [artist for artist in headers if (float(artist.get_position()[0]) < split_x) == (x < split_x)]

    before = []
    for group in groups:
        header_top = max(item.get_window_extent(renderer).y1 for item in members(group))
        before.append((group.get_window_extent(renderer).y0 - header_top) / dpi * 25.4)
    for artist in headers:
        artist.set_fontsize(COLUMN_HEADER_SIZE_PT)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for group, old_gap_mm in zip(groups, before):
        header_top = max(item.get_window_extent(renderer).y1 for item in members(group))
        group_box = group.get_window_extent(renderer)
        desired_y0 = header_top + (old_gap_mm * 0.5 / 25.4 * dpi)
        delta_fraction = (desired_y0 - group_box.y0) / float(fig.bbox.height)
        group.set_y(float(group.get_position()[1]) + delta_fraction)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    after = []
    for group in groups:
        header_top = max(item.get_window_extent(renderer).y1 for item in members(group))
        after.append((group.get_window_extent(renderer).y0 - header_top) / dpi * 25.4)
    return before, after


def _panel_axes(fig):
    violin_axes = list(getattr(fig, "_panel_d_violin_axes", []))
    spectra = list(getattr(fig, "_panel_c_spectrum_axes", []))
    bar_axes = [
        ax for ax in fig.axes
        if ax.axison and ax not in violin_axes and ax not in spectra
        and ax.get_title().endswith(")") and len(ax.patches) >= 1
    ]
    pdf_axes = [
        ax for ax in fig.axes if ax.axison and len(ax.images) == 1
        and np.asarray(ax.images[0].get_array()).shape == (64, 64)
    ]
    b_axes = list(getattr(fig, "_panel_b_matrix_axes", []))
    if not all(len(items) == 3 for items in (bar_axes, spectra, pdf_axes, violin_axes, b_axes)):
        raise RuntimeError(
            "Unexpected lower-panel inventory: "
            f"b={len(b_axes)}, bars={len(bar_axes)}, spectra={len(spectra)}, "
            f"pdfs={len(pdf_axes)}, violins={len(violin_axes)}"
        )
    return b_axes, bar_axes, spectra, pdf_axes, violin_axes


def _axis_left_text_extent_px(ax, renderer) -> float:
    spine_x = ax.get_window_extent(renderer).x0
    artists = [ax.yaxis.label, *ax.get_yticklabels()]
    left = min(
        (artist.get_window_extent(renderer).x0 for artist in artists
         if artist.get_visible() and artist.get_text()),
        default=spine_x,
    )
    return max(0.0, spine_x - left)


def _reflow_lower_panels(fig) -> dict:
    b_axes, bar_axes, spectra, pdf_axes, violin_axes = _panel_axes(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    dpi = float(fig.dpi)
    canvas_width_px = float(fig.bbox.width)
    width_in = float(fig.get_size_inches()[0])

    b_right_px = max(ax.get_window_extent(renderer).x1 for ax in b_axes)
    c_outward_px = max(_axis_left_text_extent_px(ax, renderer) for ax in (bar_axes[0], spectra[0]))
    d_outward_px = _axis_left_text_extent_px(violin_axes[0], renderer)
    required_left_px = b_right_px + max(c_outward_px, d_outward_px) + MIN_EXTERNAL_CLEARANCE_MM / 25.4 * dpi
    right = max(ax.get_position().x1 for ax in [*bar_axes, *spectra, *pdf_axes, *violin_axes])
    minimum_d_width_mm = V3_PANEL_D_WIDTH_MM + 0.25
    max_left = right - (3.0 * minimum_d_width_mm + 2.0 * PANEL_D_GAP_MM) / (width_in * 25.4)
    required_left = required_left_px / canvas_width_px
    if required_left > max_left:
        raise RuntimeError(
            f"The requested b expansion leaves only {(right-required_left)*width_in*25.4:.2f} mm for Panel d; "
            f"at least {(right-max_left)*width_in*25.4:.2f} mm is required."
        )
    left = required_left

    c_gap = PANEL_C_GAP_MM / (width_in * 25.4)
    v3._move_axis_group_horizontally(bar_axes, left, right, c_gap)
    v3._move_axis_group_horizontally(spectra, left, right, c_gap)
    for legend in (getattr(fig, "_panel_c_bar_legend", None), getattr(fig, "_panel_c_line_legend", None)):
        if legend is not None:
            ax = legend.axes
            box = ax.get_position()
            ax.set_position([left, box.y0, right - left, box.height], which="both")

    d_gap = PANEL_D_GAP_MM / (width_in * 25.4)
    v3._move_axis_group_horizontally(pdf_axes, left, right, d_gap)
    v3._move_axis_group_horizontally(violin_axes, left, right, d_gap)
    for ax in pdf_axes:
        ax.set_box_aspect(None)
        ax.set_aspect("auto")
    titles = list(getattr(fig, "_panel_d_title_artists", []))
    for title, ax in zip(titles, sorted(pdf_axes, key=lambda item: item.get_position().x0)):
        box = ax.get_position()
        title.set_x((box.x0 + box.x1) / 2.0)

    tag_x = left - PANEL_TAG_LEFT_OFFSET_MM / (width_in * 25.4)
    panel_letters = getattr(fig, "_panel_letter_artists", {})
    for label in ("c", "d"):
        if label in panel_letters:
            panel_letters[label].set_x(tag_x)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    b_right_px = max(ax.get_window_extent(renderer).x1 for ax in b_axes)
    nearest_left_px = min(
        min(ax.get_tightbbox(renderer).x0 for ax in (bar_axes[0], spectra[0])),
        min(label.get_window_extent(renderer).x0 for label in violin_axes[0].get_yticklabels() if label.get_text()),
    )
    external_clearance_mm = (nearest_left_px - b_right_px) / dpi * 25.4
    if external_clearance_mm < 1.0:
        raise RuntimeError(f"Panel b to c/d text clearance is only {external_clearance_mm:.3f} mm")

    d_widths_mm = [ax.get_position().width * width_in * 25.4 for ax in pdf_axes]
    if min(d_widths_mm) <= V3_PANEL_D_WIDTH_MM:
        raise RuntimeError(f"Panel-d subplots did not widen relative to V3: {d_widths_mm}")
    return {
        "aligned_left_spine_fraction": left,
        "panel_tag_x_fraction": tag_x,
        "panel_b_to_c_d_text_clearance_mm": external_clearance_mm,
        "panel_c_total_width_mm": (right - left) * width_in * 25.4,
        "panel_c_gap_mm": PANEL_C_GAP_MM,
        "panel_d_gap_mm": PANEL_D_GAP_MM,
        "panel_d_subplot_widths_mm": d_widths_mm,
    }


def _tick_overlap_count(axes, renderer) -> int:
    count = 0
    for ax in axes:
        boxes = [
            tick.get_window_extent(renderer) for tick in ax.get_xticklabels()
            if tick.get_visible() and tick.get_text()
        ]
        count += sum(left.overlaps(right) for index, left in enumerate(boxes) for right in boxes[index + 1:])
    return count


def _place_panel_d_xlabel(fig) -> dict:
    b_axes, _, _, pdf_axes, violin_axes = _panel_axes(fig)
    for ax in violin_axes:
        ax.tick_params(axis="x", labelbottom=True, labelsize=base.SIZE_TICK_LABEL)
        ticks = [tick for tick in ax.get_xticklabels() if tick.get_text()]
        for tick in ticks:
            tick.set_y(0.0)
            tick.set_rotation(0)
            tick.set_horizontalalignment("center")
        # Preserve one baseline while directing the outer glyph boxes away
        # from the crowded central interval; tick positions/values stay fixed.
        if len(ticks) == 3:
            ticks[0].set_horizontalalignment("right")
            ticks[2].set_horizontalalignment("left")
    center = violin_axes[1]
    xlabel = center.xaxis.label
    center.xaxis.labelpad = 7.0
    center.xaxis.set_label_coords(0.5, -0.20)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_overlaps = _tick_overlap_count(violin_axes, renderer)
    dynamic_tick_size = None
    if tick_overlaps:
        dynamic_tick_size = base.SIZE_ANNOTATION
        for ax in violin_axes:
            ax.tick_params(axis="x", labelsize=dynamic_tick_size)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        tick_overlaps = _tick_overlap_count(violin_axes, renderer)
    if tick_overlaps:
        raise RuntimeError(f"Panel-d tick labels still collide after safe font reduction: {tick_overlaps}")

    dpi = float(fig.dpi)
    b_bottom = min(ax.get_tightbbox(renderer).y0 for ax in b_axes)

    def align_xlabel_bottom() -> None:
        nonlocal renderer
        box = xlabel.get_window_extent(renderer)
        x, y = xlabel.get_position()
        xlabel.set_position((x, y + (b_bottom - box.y0) / center.get_window_extent(renderer).height))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    align_xlabel_bottom()
    tick_boxes = [
        tick.get_window_extent(renderer) for ax in violin_axes for tick in ax.get_xticklabels()
        if tick.get_visible() and tick.get_text()
    ]
    xlabel_box = xlabel.get_window_extent(renderer)
    clearance_px = min(box.y0 for box in tick_boxes) - xlabel_box.y1
    required_px = 1.0 / 25.4 * dpi
    violin_bottom_raise_mm = 0.0
    if clearance_px < required_px:
        delta_px = required_px - clearance_px
        delta_fraction = delta_px / float(fig.bbox.height)
        for ax in violin_axes:
            box = ax.get_position()
            if box.height <= delta_fraction + 0.02:
                raise RuntimeError("Panel-d violin row cannot yield enough bottom label clearance")
            ax.set_position([box.x0, box.y0 + delta_fraction, box.width, box.height - delta_fraction], which="both")
        violin_bottom_raise_mm = delta_px / dpi * 25.4
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        align_xlabel_bottom()
        tick_boxes = [
            tick.get_window_extent(renderer) for ax in violin_axes for tick in ax.get_xticklabels()
            if tick.get_visible() and tick.get_text()
        ]
        xlabel_box = xlabel.get_window_extent(renderer)
        clearance_px = min(box.y0 for box in tick_boxes) - xlabel_box.y1

    pdf_bottom_px = min(ax.get_window_extent(renderer).y0 for ax in pdf_axes)
    violin_top_px = max(ax.get_window_extent(renderer).y1 for ax in violin_axes)
    pdf_violin_gap_mm = (pdf_bottom_px - violin_top_px) / dpi * 25.4
    bottom_alignment_mm = abs(xlabel_box.y0 - b_bottom) / dpi * 25.4
    return {
        "xlabel_labelpad_pt": 7.0,
        "xlabel_to_ticks_clearance_mm": clearance_px / dpi * 25.4,
        "xlabel_bottom_to_panel_b_bottom_delta_mm": bottom_alignment_mm,
        "violin_bottom_raise_mm": violin_bottom_raise_mm,
        "pdf_to_violin_gap_mm": pdf_violin_gap_mm,
        "tick_overlap_count": _tick_overlap_count(violin_axes, renderer),
        "dynamic_tick_font_size_pt": dynamic_tick_size,
    }


def _collision_gate(fig) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_owner = {}
    for ax in fig.axes:
        for axis_name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                for label in (tick.label1, tick.label2):
                    tick_owner[id(label)] = (ax, axis_name, float(tick.get_loc()))
    visible = []
    for artist in fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text)):
        if not artist.get_visible() or not artist.get_text():
            continue
        owner = tick_owner.get(id(artist))
        if owner is not None:
            ax, axis_name, location = owner
            if not ax.axison:
                continue
            lower, upper = ax.get_xlim() if axis_name == "x" else ax.get_ylim()
            if not min(lower, upper) <= location <= max(lower, upper):
                continue
        box = artist.get_window_extent(renderer)
        if box.width > 0 and box.height > 0:
            visible.append((artist, box))
    text_text = []
    for index, (left_artist, left_box) in enumerate(visible):
        for right_artist, right_box in visible[index + 1:]:
            if left_artist.axes is not None and left_artist.axes is right_artist.axes:
                continue
            iw = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
            ih = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
            if iw > 1.5 and ih > 1.5:
                text_text.append([left_artist.get_text(), right_artist.get_text(), iw, ih])
    text_axes = []
    for artist, box in visible:
        for axis_index, ax in enumerate(fig.axes):
            if not ax.axison or artist.axes is ax:
                continue
            target = ax.get_window_extent(renderer)
            iw = min(box.x1, target.x1) - max(box.x0, target.x0)
            ih = min(box.y1, target.y1) - max(box.y0, target.y0)
            if iw > 1.5 and ih > 1.5:
                text_axes.append([artist.get_text(), axis_index, iw, ih])
    return {
        "text_text_overlap_count": len(text_text),
        "text_text_overlaps": text_text,
        "text_nonowned_axes_overlap_count": len(text_axes),
        "text_nonowned_axes_overlaps": text_axes,
    }


def _final_v4(fig) -> dict:
    v2._postprocess(fig)
    v3_qa = v3._final_geometry_and_qa(fig)
    gap_before, gap_after = _set_panel_a_typography(fig)
    lower = _reflow_lower_panels(fig)
    bottom = _place_panel_d_xlabel(fig)
    collision = _collision_gate(fig)
    b_axes, bar_axes, spectra, pdf_axes, violin_axes = _panel_axes(fig)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    mean_clips = 0
    for ax in violin_axes:
        axis_box = ax.get_window_extent(renderer)
        for artist in ax.texts:
            try:
                float(artist.get_text())
            except ValueError:
                continue
            box = artist.get_window_extent(renderer)
            mean_clips += int(box.x0 < axis_box.x0 or box.x1 > axis_box.x1)
    c_lefts = [bar_axes[0].get_position().x0, spectra[0].get_position().x0]
    d_lefts = [pdf_axes[0].get_position().x0, violin_axes[0].get_position().x0]
    return {
        "revision": "art_v4",
        "canvas_mm": [float(value) * 25.4 for value in fig.get_size_inches()],
        "panel_a_column_header_size_pt": COLUMN_HEADER_SIZE_PT,
        "panel_a_axis_title_reference_size_pt": float(base.SIZE_TICK_LABEL),
        "panel_a_group_to_column_gap_before_mm": gap_before,
        "panel_a_group_to_column_gap_after_mm": gap_after,
        "panel_a_gap_ratio": [after / before for before, after in zip(gap_before, gap_after)],
        "panel_b_grid_width_multiplier": V4_PANEL_B_RATIO / V3_PANEL_B_RATIO,
        "panel_b_axis_widths_mm": [ax.get_position().width * float(fig.get_size_inches()[0]) * 25.4 for ax in b_axes],
        "panel_c_left_spines": c_lefts,
        "panel_d_left_spines": d_lefts,
        "panel_c_d_max_left_spine_delta_mm": max(abs(left - right) for left in c_lefts for right in d_lefts) * float(fig.get_size_inches()[0]) * 25.4,
        "panel_c_total_width_mm": lower["panel_c_total_width_mm"],
        "panel_c_width_multiplier_vs_v3": lower["panel_c_total_width_mm"] / ((0.9793550055098136 - 0.4166876038717284) * 180.0),
        "panel_c_intercolumn_gap_mm": lower["panel_c_gap_mm"],
        "panel_d_intercolumn_gap_mm": lower["panel_d_gap_mm"],
        "panel_d_subplot_widths_mm": lower["panel_d_subplot_widths_mm"],
        "panel_b_to_c_d_text_clearance_mm": lower["panel_b_to_c_d_text_clearance_mm"],
        "panel_c_d_tag_x_alignment_delta_mm": 0.0,
        "panel_d_mean_clip_count": mean_clips,
        "panel_d_bottom": bottom,
        "v3_pre_reflow_collision_measurements": {
            "text_text_overlap_count": v3_qa["text_text_overlap_count"],
            "text_axes_overlap_count": v3_qa["text_axes_overlap_count"],
        },
        **collision,
    }


def _install_v4_save_hook() -> None:
    original_save = base._save_figure

    def save_v4(fig, output_base: Path, formats: list[str], dpi: int):
        qa = _final_v4(fig)
        outputs = original_save(fig, output_base, formats, dpi)
        (output_base.parent / f"{output_base.name}_art_qa.json").write_text(
            json.dumps(qa, indent=2, sort_keys=True,
                       default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n",
            encoding="utf-8",
        )
        return outputs

    base._save_figure = save_v4


def main() -> int:
    _apply_v4_contract()
    _install_v4_save_hook()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

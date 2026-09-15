#!/usr/bin/env python
"""Render Figure 4 art V3 from the frozen round-3 scientific artifacts.

This revision changes display inventory and geometry only.  The author-approved
omissions (Panel-a truth column, Panel-b Unobs. column, and Panel-d mean prefix)
are applied after the frozen source arrays have been loaded by the V2 producer.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
V2_PATH = HERE / "94_assemble_coupled_field_art_v2.py"
PANEL_A_COLUMNS = [
    "T only", "T + U1", "CO + T + U1 + p",
    "FFM-FNO", "Latent FM", "SiT", "Senseiver",
]
PANEL_A_FIELD_LABELS = [r"$Y_{CH_4}$", r"$p$", r"$U_1$"]
V2_FIELD_SPACER = 0.25
V3_FIELD_SPACER = V2_FIELD_SPACER * 0.5
V2_CONTENT_HEIGHT_IN = 2.8154385965
V3_CONTENT_HEIGHT_IN = 2.71869
PANEL_C_WSPACE = 0.30
PANEL_D_WSPACE = 0.24


def _load_v2():
    spec = importlib.util.spec_from_file_location("coupled_field_art_v2_for_v3", V2_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V2 renderer: {V2_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


v2 = _load_v2()
base = v2.base


def _cluster_centers(axes: list, *, decimals: int = 7) -> list[float]:
    return sorted({round((ax.get_position().x0 + ax.get_position().x1) / 2.0, decimals) for ax in axes})


def _panel_a_v3_reflow(fig, axes: list, texts: list, divider_artists: list) -> None:
    if len(axes) != 45:
        raise RuntimeError(f"Expected 45 V2 Panel-a map axes before pruning; got {len(axes)}")
    old_centers = _cluster_centers(axes)
    if len(old_centers) != 8:
        raise RuntimeError(f"Expected eight V2 Panel-a columns; got {len(old_centers)}")
    center_array = np.asarray(old_centers)
    by_column: list[list] = [[] for _ in old_centers]
    for ax in axes:
        center = (ax.get_position().x0 + ax.get_position().x1) / 2.0
        by_column[int(np.argmin(np.abs(center_array - center)))].append(ax)
    if [len(items) for items in by_column] != [3, 6, 6, 6, 6, 6, 6, 6]:
        raise RuntimeError(f"Unexpected Panel-a column inventory: {[len(items) for items in by_column]}")

    truth_axes = by_column[0]
    field_centers = sorted((ax.get_position().y0 + ax.get_position().y1) / 2.0 for ax in truth_axes)[::-1]
    for ax in truth_axes:
        ax.remove()

    for artist in list(texts):
        if artist.get_text() == "Ground truth":
            artist.remove()

    remaining = [ax for items in by_column[1:] for ax in items]
    old_remaining_centers = np.asarray(old_centers[1:])
    # Keep a real dual-label gutter, then use the complete remaining page width.
    left = max(0.066, min(ax.get_position().x0 for ax in axes))
    right = min(0.989, max(ax.get_position().x1 for ax in axes))
    gap = 1.45 / 180.0
    width = (right - left - 6.0 * gap) / 7.0
    if width <= 0:
        raise RuntimeError("Panel-a V3 reflow produced a non-positive map width")
    new_centers = np.asarray([left + width / 2.0 + index * (width + gap) for index in range(7)])
    for ax in remaining:
        box = ax.get_position()
        old_center = (box.x0 + box.x1) / 2.0
        index = int(np.argmin(np.abs(old_remaining_centers - old_center)))
        ax.set_position([new_centers[index] - width / 2.0, box.y0, width, box.height], which="both")
        ax.set_aspect("auto")

    header_lookup = {text: index for index, text in enumerate(PANEL_A_COLUMNS)}
    for artist in texts:
        text = artist.get_text()
        normalized = text.replace("$U_1$", "U1").replace("$p$", "p").replace("\n", " ")
        normalized = " ".join(normalized.split())
        if normalized in header_lookup:
            artist.set_text(PANEL_A_COLUMNS[header_lookup[normalized]].replace("U1", r"$U_1$").replace("+ p", r"+ $p$"))
            artist.set_x(float(new_centers[header_lookup[normalized]]))
            artist.set_linespacing(1.0)
        elif text == "Conditioning progression (DMF-Gen)":
            artist.set_x(float(np.mean(new_centers[:3])))
        elif text == "Baseline comparisons: conditioned on T only":
            artist.set_x(float(np.mean(new_centers[3:])))

    divider_x = float((new_centers[2] + new_centers[3]) / 2.0)
    for artist in divider_artists:
        if isinstance(artist, Line2D):
            artist.set_xdata([divider_x, divider_x])

    # Recreate the physical-field labels removed with the truth axes, plus the
    # requested reconstruction/error row-type column immediately to their right.
    row_centers = sorted({round((ax.get_position().y0 + ax.get_position().y1) / 2.0, 7) for ax in remaining}, reverse=True)
    if len(row_centers) != 6:
        raise RuntimeError(f"Expected six Panel-a row centers; got {row_centers}")
    field_x, row_type_x = left - 0.044, left - 0.017
    for field_index, field_label in enumerate(PANEL_A_FIELD_LABELS):
        rec_y, err_y = row_centers[field_index * 2:field_index * 2 + 2]
        fig.text(field_x, (rec_y + err_y) / 2.0, field_label, rotation=90,
                 ha="center", va="center", fontsize=base.SIZE_AXIS_LABEL)
        fig.text(row_type_x, rec_y, "Recon.", rotation=90, ha="center", va="center",
                 fontsize=base.SIZE_TICK_LABEL)
        fig.text(row_type_x, err_y, "Error", rotation=90, ha="center", va="center",
                 fontsize=base.SIZE_TICK_LABEL)

    fig._panel_a_map_count = len(remaining)  # type: ignore[attr-defined]
    fig._panel_a_column_count = 7  # type: ignore[attr-defined]
    fig._panel_a_map_width_mm = width * 180.0  # type: ignore[attr-defined]
    fig._panel_a_intercolumn_gap_mm = gap * 180.0  # type: ignore[attr-defined]
    fig._panel_a_horizontal_bounds = [left, right]  # type: ignore[attr-defined]
    fig._panel_a_dual_row_label_count = 6  # type: ignore[attr-defined]


def _prune_panel_b_axis(ax) -> None:
    if len(ax.images) != 1:
        raise RuntimeError("Panel-b matrix axis must contain exactly one image")
    image = ax.images[0]
    rgba = np.asarray(image.get_array())
    if rgba.shape != (8, 6, 4):
        raise RuntimeError(f"Unexpected Panel-b RGBA matrix shape: {rgba.shape}")
    image.set_data(rgba[:, :5, :])
    image.set_extent((-0.5, 4.5, 7.5, -0.5))
    ax.set_xlim(-0.5, 4.5)
    for artist in list(ax.texts):
        x, _ = artist.get_position()
        if np.isclose(float(x), 5.0):
            artist.remove()
    labels = [tick.get_text() for tick in ax.get_xticklabels()][:5]
    ax.set_xticks(range(5), labels)


def _panel_d_slots(fig, slot, column_count: int) -> dict[str, list]:
    if column_count != 3:
        raise RuntimeError(f"Figure 4 Panel d expects three columns; got {column_count}")
    box = slot.get_position(fig)
    fig_width, fig_height = (float(value) for value in fig.get_size_inches())
    total_width_in = box.width * fig_width
    total_height_in = box.height * fig_height
    column_width_in = total_width_in / (column_count + (column_count - 1) * PANEL_D_WSPACE)
    hspace = float(base.PANEL_D_STYLE["row_hspace"])
    row_height_total_in = total_height_in / (1.0 + hspace / 2.0)
    pdf_height_in = column_width_in * float(base.PANEL_D_STYLE["pdf_height_scale"])
    violin_height_in = row_height_total_in - pdf_height_in
    if violin_height_in <= 0.45:
        raise RuntimeError(f"Panel-d violin height is only {violin_height_in:.3f} in")
    grid = slot.subgridspec(
        2, column_count, height_ratios=[pdf_height_in, violin_height_in],
        hspace=hspace, wspace=PANEL_D_WSPACE,
    )
    return {
        "pdfs": [grid[0, col] for col in range(column_count)],
        "violins": [grid[1, col] for col in range(column_count)],
    }


def _apply_v3_contract() -> None:
    # Install the complete V2 typography/palette treatment, then layer only the
    # author-approved V3 display geometry on top.
    v2._apply_art_contract()
    base.INTRA_PANEL_WSPACE = PANEL_C_WSPACE
    base._panel_d_grid_slots = _panel_d_slots

    previous_overrides = base._apply_revision_overrides

    def v3_overrides(layout: dict) -> None:
        previous_overrides(layout)
        base.PANEL_A_LAYOUT.update({
            "content_height_in": V3_CONTENT_HEIGHT_IN,
            "field_group_spacer_ratio": V3_FIELD_SPACER,
        })
        base.PANEL_D_STYLE.update({
            "pdf_height_scale": 0.62,
            "violin_width": 0.66,
            "violin_data_right_fraction": 0.66,
            "mean_x_axes": 0.71,
            "mean_prefix": "",
            "x_label_pad": -2.0,
        })

    base._apply_revision_overrides = v3_overrides

    previous_qualitative = base.draw_qualitative_panel

    def qualitative_v3(fig, slot, *args, **kwargs):
        axes_before = set(fig.axes)
        texts_before = set(fig.texts)
        artists_before = set(fig.artists)
        result = previous_qualitative(fig, slot, *args, **kwargs)
        _panel_a_v3_reflow(
            fig,
            [ax for ax in fig.axes if ax not in axes_before],
            [artist for artist in fig.texts if artist not in texts_before],
            [artist for artist in fig.artists if artist not in artists_before],
        )
        return result

    base.draw_qualitative_panel = qualitative_v3

    previous_heatmap = base.draw_l2_heatmap_panel

    def heatmap_v3(fig, slot, *args, **kwargs):
        axes_before = set(fig.axes)
        result = previous_heatmap(fig, slot, *args, **kwargs)
        created = [ax for ax in fig.axes if ax not in axes_before]
        if len(created) != 3:
            raise RuntimeError(f"Expected three Panel-b matrices; got {len(created)}")
        for ax in created:
            _prune_panel_b_axis(ax)
        fig._panel_b_matrix_axes = created  # type: ignore[attr-defined]
        return result

    base.draw_l2_heatmap_panel = heatmap_v3


def _relative_luminance(rgb) -> float:
    values = np.asarray(rgb[:3], dtype=float)
    values = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    return float(values @ np.asarray([0.2126, 0.7152, 0.0722]))


def _move_axis_group_horizontally(axes: list, left: float, right: float, gap: float) -> None:
    columns = _cluster_centers(axes)
    if len(columns) != 3:
        raise RuntimeError(f"Expected three columns while reflowing axes; got {columns}")
    width = (right - left - 2.0 * gap) / 3.0
    if width <= 0:
        raise RuntimeError("Panel-c reflow produced a non-positive axes width")
    old = np.asarray(columns)
    centers = np.asarray([left + width / 2.0 + index * (width + gap) for index in range(3)])
    for ax in axes:
        box = ax.get_position()
        index = int(np.argmin(np.abs(old - (box.x0 + box.x1) / 2.0)))
        ax.set_position([centers[index] - width / 2.0, box.y0, width, box.height], which="both")


def _strip_mean_prefixes(fig) -> int:
    count = 0
    for ax in getattr(fig, "_panel_d_violin_axes", []):
        for artist in ax.texts:
            text = artist.get_text().replace(r"$\mu$=", "").replace("μ=", "")
            if text != artist.get_text() or text[:1] in {".", "0"}:
                if text.startswith("."):
                    text = "0" + text
                artist.set_text(text)
                artist.set_x(float(base.PANEL_D_STYLE["mean_x_axes"]))
                count += 1
    return count


def _final_geometry_and_qa(fig) -> dict:
    mean_count = _strip_mean_prefixes(fig)
    panel_letters = getattr(fig, "_panel_letter_artists", {})
    for label in ("a", "b"):
        if label in panel_letters:
            panel_letters[label].set_x(0.004)

    # Preserve each five-column matrix's physical width but use the removable
    # wrapper slack to translate Panel b left, opening a text-safe corridor for
    # Panel c's left ylabel and scientific-notation ticks.
    b_axes = list(getattr(fig, "_panel_b_matrix_axes", []))
    b_shift = 2.0 / 180.0
    for ax in b_axes:
        box = ax.get_position()
        ax.set_position([box.x0 - b_shift, box.y0, box.width, box.height], which="both")

    # Panel d has more width than V2 after the matrix pruning.  Reallocate part
    # of that gain to the left-label corridor, while retaining wider PDF and
    # violin boxes than V2 and a six-millimetre inter-column gutter.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    violin_axes = list(getattr(fig, "_panel_d_violin_axes", []))
    d_pdf_axes = [ax for ax in fig.axes if ax.axison and len(ax.images) == 1
                  and np.asarray(ax.images[0].get_array()).shape == (64, 64)]
    if len(d_pdf_axes) != 3 or len(violin_axes) != 3:
        raise RuntimeError(f"Unexpected Panel-d axes inventory: {len(d_pdf_axes)} PDFs, {len(violin_axes)} violins")
    d_old_left = min(ax.get_position().x0 for ax in violin_axes)
    d_right = max(ax.get_position().x1 for ax in violin_axes)
    d_left = d_old_left + 0.30 / float(fig.get_size_inches()[0])
    d_gap = 6.0 / 180.0
    _move_axis_group_horizontally(d_pdf_axes, d_left, d_right, d_gap)
    _move_axis_group_horizontally(violin_axes, d_left, d_right, d_gap)
    for ax in d_pdf_axes:
        ax.set_box_aspect(None)
        ax.set_aspect("auto")
    d_titles = list(getattr(fig, "_panel_d_title_artists", []))
    for title, ax in zip(d_titles, sorted(d_pdf_axes, key=lambda item: item.get_position().x0)):
        box = ax.get_position()
        title.set_x((box.x0 + box.x1) / 2.0)

    # Panel c starts at the rendered left edge of the Panel-d method names and
    # still ends at its original right edge.  This is the requested cross-panel
    # alignment rather than an inferred GridSpec approximation.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    method_labels = [label for label in violin_axes[0].get_yticklabels() if label.get_text()] if violin_axes else []
    ffm_label = next((label for label in method_labels if label.get_text() == "FFM-Perceiver"), None)
    if ffm_label is None:
        raise RuntimeError("Could not locate the Panel-d FFM-Perceiver method label")
    c_left = ffm_label.get_window_extent(renderer).x0 / float(fig.bbox.width)
    spectra = list(getattr(fig, "_panel_c_spectrum_axes", []))
    bar_axes = [ax for ax in fig.axes if ax.axison and ax not in violin_axes and ax not in spectra
                and ax.get_title().endswith(")") and len(ax.patches) >= 1]
    if len(spectra) != 3 or len(bar_axes) != 3:
        raise RuntimeError(f"Unexpected Panel-c axes inventory: {len(bar_axes)} bars, {len(spectra)} spectra")
    c_right = max(ax.get_position().x1 for ax in [*bar_axes, *spectra])
    c_gap = 8.0 / 180.0
    _move_axis_group_horizontally(bar_axes, c_left, c_right, c_gap)
    _move_axis_group_horizontally(spectra, c_left, c_right, c_gap)
    for legend in (getattr(fig, "_panel_c_bar_legend", None), getattr(fig, "_panel_c_line_legend", None)):
        if legend is not None:
            ax = legend.axes
            box = ax.get_position()
            ax.set_position([c_left, box.y0, c_right - c_left, box.height], which="both")

    if "c" in panel_letters:
        panel_letters["c"].set_x(c_left)
    if "d" in panel_letters:
        panel_letters["d"].set_x(c_left)

    # Keep violin ticks on one baseline.  If their final glyph boxes collide,
    # remove the complete label set rather than creating a staggered hierarchy.
    for ax in violin_axes:
        for tick in ax.get_xticklabels():
            tick.set_rotation(0)
            tick.set_horizontalalignment("center")
            tick.set_y(0.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_overlap_count = 0
    for ax in violin_axes:
        boxes = [tick.get_window_extent(renderer) for tick in ax.get_xticklabels() if tick.get_visible() and tick.get_text()]
        tick_overlap_count += sum(left.overlaps(right) for index, left in enumerate(boxes) for right in boxes[index + 1:])
    tick_layout = "single baseline"
    if tick_overlap_count:
        for ax in violin_axes:
            ax.tick_params(axis="x", labelbottom=False)
        tick_layout = "labels suppressed after measured collision"
        tick_overlap_count = 0

    # Raise the shared central xlabel until its rendered bottom aligns with the
    # lowest visible edge of Panel b.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlabel_entry = next(((ax, ax.xaxis.label) for ax in violin_axes if ax.xaxis.label.get_text()), None)
    if b_axes and xlabel_entry is not None:
        owner, xlabel_artist = xlabel_entry
        target_y = min(ax.get_tightbbox(renderer).y0 for ax in b_axes)
        current = xlabel_artist.get_window_extent(renderer)
        x, y = xlabel_artist.get_position()
        xlabel_artist.set_position((x, y + (target_y - current.y0) / owner.get_window_extent(renderer).height))

    # Dynamic black/white contrast for the five displayed matrix columns.
    matrix_value_count = white_count = black_count = 0
    min_contrast = float("inf")
    for ax in b_axes:
        rgba = np.asarray(ax.images[0].get_array())
        if rgba.shape != (8, 5, 4):
            raise RuntimeError(f"Panel-b displayed matrix has shape {rgba.shape}, expected (8, 5, 4)")
        for artist in ax.texts:
            try:
                float(artist.get_text())
            except ValueError:
                continue
            x, y = artist.get_position()
            row, column = int(round(float(y))), int(round(float(x)))
            if not (0 <= row < 8 and 0 <= column < 5):
                continue
            luminance = _relative_luminance(rgba[row, column])
            white_contrast = 1.05 / (luminance + 0.05)
            black_contrast = (luminance + 0.05) / 0.05
            use_white = white_contrast >= black_contrast
            artist.set_color("white" if use_white else "#202020")
            matrix_value_count += 1
            white_count += int(use_white)
            black_count += int(not use_white)
            min_contrast = min(min_contrast, white_contrast if use_white else black_contrast)

    # Generalized final-aspect collision gate: compare text with other text and
    # with unrelated axes data rectangles.  Parent/own-axis intersections are
    # intentionally excluded because ticks, labels, and annotations attach to
    # their own axes by design.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    visible_text = []
    tick_owner = {}
    for ax in fig.axes:
        for axis_name, axis in (("x", ax.xaxis), ("y", ax.yaxis)):
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                for label in (tick.label1, tick.label2):
                    tick_owner[id(label)] = (ax, axis_name, float(tick.get_loc()))
    for artist in fig.findobj(match=lambda item: isinstance(item, matplotlib.text.Text)):
        if not artist.get_visible() or not artist.get_text():
            continue
        owner = tick_owner.get(id(artist))
        if owner is not None:
            ax, axis_name, location = owner
            lower, upper = ax.get_xlim() if axis_name == "x" else ax.get_ylim()
            if not min(lower, upper) <= location <= max(lower, upper):
                continue
        box = artist.get_window_extent(renderer)
        if box.width > 0 and box.height > 0:
            visible_text.append((artist, box))
    text_overlaps = []
    for index, (left_artist, left_box) in enumerate(visible_text):
        for right_artist, right_box in visible_text[index + 1:]:
            if left_artist.axes is not None and left_artist.axes is right_artist.axes:
                continue
            iw = min(left_box.x1, right_box.x1) - max(left_box.x0, right_box.x0)
            ih = min(left_box.y1, right_box.y1) - max(left_box.y0, right_box.y0)
            if iw > 1.5 and ih > 1.5:
                text_overlaps.append([left_artist.get_text(), right_artist.get_text(), iw, ih])
    text_axes_overlaps = []
    data_axes = [ax for ax in fig.axes if ax.axison]
    for artist, box in visible_text:
        for ax in data_axes:
            if artist.axes is ax:
                continue
            target = ax.get_window_extent(renderer)
            iw = min(box.x1, target.x1) - max(box.x0, target.x0)
            ih = min(box.y1, target.y1) - max(box.y0, target.y0)
            if iw > 1.5 and ih > 1.5:
                text_axes_overlaps.append([artist.get_text(), fig.axes.index(ax), iw, ih])

    d_pdf_widths = [ax.get_position().width * float(fig.get_size_inches()[0]) * 25.4 for ax in d_pdf_axes]
    mean_clip_count = 0
    for ax in violin_axes:
        axis_box = ax.get_window_extent(renderer)
        for artist in ax.texts:
            try:
                float(artist.get_text())
            except ValueError:
                continue
            box = artist.get_window_extent(renderer)
            mean_clip_count += int(box.x0 < axis_box.x0 or box.x1 > axis_box.x1)
    return {
        "revision": "art_v3",
        "canvas_mm": [float(value) * 25.4 for value in fig.get_size_inches()],
        "panel_a_map_count": int(getattr(fig, "_panel_a_map_count", -1)),
        "panel_a_column_count": int(getattr(fig, "_panel_a_column_count", -1)),
        "panel_a_map_width_mm": float(getattr(fig, "_panel_a_map_width_mm", 0.0)),
        "panel_a_intercolumn_gap_mm": float(getattr(fig, "_panel_a_intercolumn_gap_mm", 0.0)),
        "panel_a_horizontal_bounds": getattr(fig, "_panel_a_horizontal_bounds", []),
        "panel_a_field_spacer_before_after": [V2_FIELD_SPACER, V3_FIELD_SPACER],
        "panel_a_dual_row_label_count": int(getattr(fig, "_panel_a_dual_row_label_count", -1)),
        "panel_b_display_shape": [8, 5],
        "panel_b_left_translation_mm": b_shift * 180.0,
        "panel_b_matrix_value_count": matrix_value_count,
        "panel_b_white_text_count": white_count,
        "panel_b_black_text_count": black_count,
        "panel_b_min_contrast_ratio": min_contrast,
        "panel_c_left_spine_fraction": c_left,
        "panel_c_target_ffm_label_left_fraction": c_left,
        "panel_c_intercolumn_gap_mm": c_gap * 180.0,
        "panel_d_pdf_widths_mm": d_pdf_widths,
        "panel_d_intercolumn_gap_mm": d_gap * 180.0,
        "panel_d_wspace": PANEL_D_WSPACE,
        "panel_d_violin_width": float(base.PANEL_D_STYLE["violin_width"]),
        "panel_d_mean_count": mean_count,
        "panel_d_mean_clip_count": mean_clip_count,
        "panel_d_mean_prefix_removed": mean_count == 24,
        "panel_d_tick_layout": tick_layout,
        "panel_d_tick_overlap_count": tick_overlap_count,
        "text_text_overlap_count": len(text_overlaps),
        "text_text_overlaps": text_overlaps,
        "text_axes_overlap_count": len(text_axes_overlaps),
        "text_axes_overlaps": text_axes_overlaps,
    }


def _install_v3_save_hook() -> None:
    previous_save = base._save_figure

    def save_v3(fig, output_base: Path, formats: list[str], dpi: int):
        qa = _final_geometry_and_qa(fig)
        outputs = previous_save(fig, output_base, formats, dpi)
        (output_base.parent / f"{output_base.name}_art_qa.json").write_text(
            json.dumps(qa, indent=2, sort_keys=True,
                       default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n",
            encoding="utf-8",
        )
        return outputs

    base._save_figure = save_v3


def main() -> int:
    _apply_v3_contract()
    v2._install_save_hook()
    _install_v3_save_hook()
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())

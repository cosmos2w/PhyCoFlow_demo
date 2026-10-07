#!/usr/bin/env python
"""Restyle frozen V5.3 bars/sweeps without altering scientific artists or sources."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv, to_rgb, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, Polygon
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image, ImageOps
import fitz

import render_mixed_resolution_v5_3 as V53

ROOT, SCRIPTS = V53.ROOT, V53.SCRIPTS
LAYOUT = SCRIPTS / "publication_layout_unified_v5_4.yaml"
BASELINE = ROOT / "figures/generated/art_style_review/MixedResolution_unified_v5_3_20261001_2010"
BASELINE_PDF = V53.FIGURES_DIR / "Assembled/MixedResolution_unified_v5_3_20261001_2010.pdf"
DEFAULT_RUN_ID = "20261007_refined"
MODELS = ["DMFGen", "FFM_Perceiver", "Senseiver", "MLP_RBF"]
MARKERS = dict(zip(MODELS, ["o", "s", "^", "D"]))


def all_axes(fig):
    return list(dict.fromkeys(fig.findobj(Axes)))


def numerical_hash(axes, excluded_patch_ids=()):
    """Hash data coordinates, CI segments, bar dimensions, arrays/masks/norms.

Presentation-only span backgrounds and marker symbols are intentionally omitted.
The supplied axes list excludes the newly added top axis and legend strip.
"""
    digest = hashlib.sha256()
    def add(value):
        if isinstance(value, (np.ndarray, np.ma.MaskedArray)):
            array = np.ma.asarray(value)
            digest.update(str((array.shape, array.dtype)).encode())
            digest.update(np.ascontiguousarray(array.data).tobytes())
            digest.update(np.ascontiguousarray(np.ma.getmaskarray(array)).tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())
    for ax in axes:
        add([ax.get_label(), list(ax.get_xlim()), list(ax.get_ylim()), ax.get_xscale(), ax.get_yscale()])
        for line in ax.lines:
            add(np.asarray(line.get_xdata())); add(np.asarray(line.get_ydata()))
        for collection in ax.collections:
            add(np.asarray(collection.get_offsets()))
            if collection.get_array() is not None:
                add(collection.get_array())
                add([type(collection.norm).__name__, collection.norm.vmin, collection.norm.vmax, collection.cmap.name])
            if hasattr(collection, "get_segments"):
                for segment in collection.get_segments(): add(np.asarray(segment))
            for path in collection.get_paths():
                add(path.vertices)
                if path.codes is not None: add(path.codes)
        for patch in ax.patches:
            if id(patch) in excluded_patch_ids: continue
            gid = str(patch.get_gid() or "")
            if isinstance(patch, Rectangle):
                add([gid, patch.get_x(), patch.get_y(), patch.get_width(), patch.get_height()])
            elif isinstance(patch, Polygon):
                add([gid, patch.get_xy().tolist()])
    return digest.hexdigest()


def draw_v53(args, cfg, layout, baseline):
    """Use the unchanged V5.3 drawers and final physical colorbar placement."""
    ctx = V53.base.make_context(args, cfg, layout, args.run_id)
    fig, axes, containers, _, width, height, shared, cbar_parent = V53._create_canvas(layout)
    drawn = {}
    drawn["a"] = V53.draw_panel("a", axes["a"], ctx)
    drawn["_b_source"] = V53.draw_panel("b", axes["b"], ctx)
    drawn["d"] = V53.draw_panel("c", axes["c"], ctx, shared_axes=shared["c"], colorbar_parent=cbar_parent)
    V53.panel_label(axes["d"], "d")
    drawn["e"] = V53.draw_panel("d", axes["d"], ctx, shared_axes=shared["d"])
    drawn["f"] = V53.draw_panel("e", axes["e"], ctx)
    V53._reflow_a(axes["a"], layout["v5_geometry"], width, height)
    V53._reflow_b_c(axes["b"], layout["v5_geometry"], ctx, width, height)
    drawn["b"] = {**drawn["_b_source"], "plotted_rows": [r for r in drawn["_b_source"]["plotted_rows"] if r["role"] == "recipe_transfer_512"]}
    drawn["c"] = {**drawn["_b_source"], "plotted_rows": [r for r in drawn["_b_source"]["plotted_rows"] if r["role"] != "recipe_transfer_512"]}
    next(t for t in axes["d"].texts if t.get_text() == "d").set_text("e")
    V53._place_panel_e_tag(fig, axes["d"], layout["v5_geometry"], width)
    for visible, legacy in (("a", "a"), ("b", "b"), ("d", "c"), ("f", "e")):
        V53.panel_label(axes[legacy], visible)
    V53.panel_label(containers["new_c"], "c")
    V53.base.enforce_geometric_aspects(fig)
    fig.canvas.draw()
    colorbar_layout, colorbar_axes, colorbar_texts = V53._reflow_de_with_colorbars(
        fig, shared, drawn, width, height, layout["v5_geometry"])
    V53.center_panel_d_headers(axes["d"], shared["d"])
    V53.v47.v46.v4_5.activate_v4_5_font_roles(cfg)
    V53.apply_v4_7_typography(fig)
    V53.manuscript.enforce_figure_typography(fig, font_family=V53.manuscript.FONT_FAMILY)
    V53.center_panel_d_headers(axes["d"], shared["d"])
    V53.align_panel_cd_annotation_gutters(fig, shared, offset_mm=1.05, strict=True)
    V53.base.enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    panels = V53._publication_panel_metadata(drawn)
    scientific_keys = {
        "a": ["dimensions", "field_limits", "bar_segments", "exposure_values", "recipe_order"],
        "b": ["plotted_rows", "recipe_transfer_values", "recipe_transfer_y_range"],
        "c": ["plotted_rows", "sensor_counts", "sensor_density_percent", "sensor_sweep_values", "sensor_sweep_y_range"],
        "d": ["full_field_relative_l2", "local_relative_l2", "field_limits", "error_limits", "roi"],
        "e": ["component_color_limits", "residual_color_limits", "relative_l2_by_model_scale"],
        "f": ["correlation_values", "bias_values", "matrix_numeric_annotation_count"],
    }
    comparison = {f"{p}:{key}": panels[p][key] == baseline["panels"][p][key]
                  for p, keys in scientific_keys.items() for key in keys if key in baseline["panels"][p]}
    if not all(comparison.values()):
        raise RuntimeError(f"Rebuilt V5.3 scientific metadata changed: {comparison}")
    return fig, axes, containers, shared, panels, ctx, colorbar_axes, colorbar_texts, comparison


def restyle(fig, axes, ctx, layout, baseline):
    geo = layout["v5_4_geometry"]
    width, old_height = fig.get_size_inches() * 25.4
    height = geo["canvas_height_mm"]
    inherited_axes = all_axes(fig)
    module = __import__("common.publication_panels_unified_v4_6", fromlist=["_panel_b_axes"])
    bar, sweeps = module._panel_b_axes(axes["b"])
    background_ids = [id(p) for p in bar.patches if not str(p.get_gid() or "").startswith("model-bar:")]
    before = numerical_hash(inherited_axes, background_ids)
    physical_boxes = {ax: [ax.get_position().x0 * width, ax.get_position().y0 * old_height,
                          ax.get_position().width * width, ax.get_position().height * old_height]
                      for ax in inherited_axes}
    fig.set_size_inches(width / 25.4, height / 25.4)
    for ax, box in physical_boxes.items(): V53._place(ax, box, width, height)
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_transform() is fig.transFigure:
            x, y = text.get_position(); text.set_position((x, y * old_height / height))
    def descendants(ax):
        return [ax, *[a for child in ax.child_axes for a in descendants(child)]]
    a_axes = descendants(axes["a"])
    for ax in a_axes:
        box = physical_boxes[ax].copy(); box[1] += geo["panel_a_shift_up_mm"]
        V53._place(ax, box, width, height)
    c_container = next(ax for ax in inherited_axes if any(t.get_text() == "c" for t in ax.texts))
    c_box = physical_boxes[c_container].copy(); c_box[3] += height - old_height
    V53._place(c_container, c_box, width, height)
    V53._place(bar, geo["panel_b_bar_mm"], width, height)
    a_bar = next(ax for ax in axes["a"].child_axes if ax.get_ylabel() == "Training cases")
    for ax in [a_bar, bar]:
        ax.yaxis.set_label_coords((2.5 - 13.5) / 99.5, .5)
        ax.yaxis.label.set_fontsize(8.5); ax.yaxis.label.set_va("center")
        ax.tick_params(axis="both", labelsize=7.8)
    colors = {}
    for model in MODELS:
        color = next(p.get_facecolor()[:3] for p in bar.patches if p.get_gid() == f"model-bar:{model}")
        hsv = rgb_to_hsv(to_rgb(color))
        if model != "DMFGen": hsv[1] = min(1, hsv[1] * 1.18)
        colors[model] = tuple(hsv_to_rgb(hsv))
    removed_backgrounds = 0
    for patch in list(bar.patches):
        gid = str(patch.get_gid() or "")
        if not gid.startswith("model-bar:"):
            patch.remove(); removed_backgrounds += 1
            continue
        model = gid.split(":", 1)[1]
        patch.set_facecolor(colors[model]); patch.set_alpha(.95)
        patch.set_edgecolor(np.asarray(colors[model]) * .66)
        patch.set_linewidth(.5); patch.set_zorder(3)
    bar.set_facecolor("white"); bar.set_axisbelow(True)
    bar.grid(False, which="both")
    bar.grid(True, axis="y", which="major", color="lightgray", alpha=.3, linestyle="--", linewidth=.35, zorder=0)
    counts = baseline["panels"]["c"]["sensor_counts"]
    density = baseline["panels"]["c"]["sensor_density_percent"]
    titles = []
    for i, (ax, box) in enumerate(zip(sweeps, geo["panel_c_sweep_mm"])):
        V53._place(ax, box, width, height)
        title = ax.get_title(); ax.set_title("")
        titles.append(fig.text((box[0] + box[2] / 2) / width,
            geo["panel_c_title_bottom_mm"][i] / height, title, ha="center", va="bottom", fontsize=8.5))
        ax.set_facecolor("white"); ax.set_axisbelow(True)
        ax.grid(False, which="both")
        ax.grid(True, axis="y", which="major", color="lightgray", alpha=.3, linestyle="--", linewidth=.35, zorder=0)
        ax.set_xticks(counts, [str(n) for n in counts])
        ax.tick_params(axis="x", labelbottom=i == 2, labelsize=7.8, pad=1.5)
        ax.tick_params(axis="y", labelsize=7.8)
        ax.set_xlabel("Sensor count" if i == 2 else "", fontsize=8.5, labelpad=1.0)
        for line in ax.lines:
            gid = str(line.get_gid() or "")
            if not gid.startswith("model-line:"): continue
            model = gid.split(":", 1)[1]; hero = model == "DMFGen"
            line.set_color(colors[model]); line.set_alpha(1.0); line.set_linestyle("-")
            line.set_linewidth(geo["hero_linewidth_pt"] if hero else geo["baseline_linewidth_pt"])
            line.set_marker(MARKERS[model]); line.set_markersize(geo["hero_markersize_pt"] if hero else geo["baseline_markersize_pt"])
            line.set_markerfacecolor(colors[model] if hero else "none")
            line.set_markeredgecolor(colors[model]); line.set_markeredgewidth(.9)
            line.set_zorder(4 if hero else 3)
    for ax in [bar, *sweeps]:
        from matplotlib.container import ErrorbarContainer
        intervals = [c for c in ax.containers if isinstance(c, ErrorbarContainer)]
        for model, container in zip(MODELS, intervals):
            _, caps, collections = container.lines
            for artist in [*caps, *collections]:
                artist.set_color(colors[model]); artist.set_alpha(.85)
    old_legend = next(ax for ax in axes["b"].child_axes if ax.get_legend() is not None)
    old_legend.get_legend().remove(); old_legend.set_axis_off()
    line_handles = [Line2D([], [], label=ctx.model_label(m), color=colors[m], linestyle="-",
        linewidth=geo["hero_linewidth_pt"] if m == "DMFGen" else geo["baseline_linewidth_pt"],
        marker=MARKERS[m], markersize=geo["hero_markersize_pt"] if m == "DMFGen" else geo["baseline_markersize_pt"],
        markerfacecolor=colors[m] if m == "DMFGen" else "none", markeredgewidth=.9) for m in MODELS]
    c_legend = fig.legend(handles=line_handles, ncol=2, loc="upper center", frameon=False,
        bbox_to_anchor=(153.75 / width, geo["panel_c_legend_top_mm"] / height),
        borderaxespad=0, fontsize=7.8, handlelength=1.4, columnspacing=.8, handletextpad=.35, labelspacing=.25)
    patch_handles = [Patch(label=ctx.model_label(m), facecolor=colors[m],
        edgecolor=np.asarray(colors[m]) * .66, alpha=.95, linewidth=.5) for m in MODELS]
    b_legend = fig.legend(handles=patch_handles, ncol=4, loc="lower center", frameon=False,
        bbox_to_anchor=((13.5 + 99.5 / 2) / width, geo["panel_b_legend_bottom_mm"] / height),
        borderaxespad=0, fontsize=7.8, handlelength=1.15, handleheight=.8,
        columnspacing=1.0, handletextpad=.4, borderpad=.15)
    # One shared secondary axis serves the entire vertically aligned stack.
    top = sweeps[0].twiny()
    top.set_label("panel-c-H-grid-density")
    top.set_xlim(sweeps[0].get_xlim()); top.set_xticks(counts, [f"{d:.1f}" for d in density])
    top.tick_params(axis="x", labelsize=7.8, pad=1.5, length=2, width=.5, colors="#222222")
    top.set_xlabel("H-grid density (%)", fontsize=8.5, labelpad=1.0)
    top.xaxis.set_label_coords(.5, 1.28)
    top.xaxis.label.set_va("bottom")
    top.spines[["left", "right", "bottom"]].set_visible(False)
    top.spines["top"].set_visible(True)
    top.spines["top"].set_linewidth(.5); top.spines["top"].set_color("#444444")
    top.patch.set_visible(False); top.grid(False)
    # The inherited e tag grazes its multiline first heading. Move only
    # that tag left to reserve a measured 0.6-mm text gutter.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    e_tag = next(t for t in fig.texts if t.get_text() == "e")
    e_header = next(t for t in fig.findobj(matplotlib.text.Text) if t.get_text() == "Truth\ncomponent")
    dx = e_header.get_window_extent(renderer).x0 - .6 * fig.dpi / 25.4 - e_tag.get_window_extent(renderer).x1
    if dx < 0:
        x, y = e_tag.get_position(); e_tag.set_position((x + dx / fig.bbox.width, y))
    fig.canvas.draw()
    after = numerical_hash(inherited_axes, background_ids)
    if after != before: raise RuntimeError("V5.4 changed numerical artists, intervals, arrays, norms or view limits")
    final_boxes = {ax: np.asarray([ax.get_position().x0 * width, ax.get_position().y0 * height,
                   ax.get_position().width * width, ax.get_position().height * height]) for ax in inherited_axes}
    a_deltas = [(final_boxes[ax] - physical_boxes[ax]).tolist() for ax in a_axes]
    moved = {id(ax) for ax in [*a_axes, c_container, bar, *sweeps]}
    unchanged_offsets = [np.max(abs(final_boxes[ax] - physical_boxes[ax])) for ax in inherited_axes if id(ax) not in moved]
    return dict(bar=bar, sweeps=sweeps, top=top, c_container=c_container, b_legend=b_legend, c_legend=c_legend,
        titles=titles, colors=colors, removed_backgrounds=removed_backgrounds,
        numerical_before=before, numerical_after=after, inherited_axes=inherited_axes,
        counts=counts, density=density, panel_a_geometry_deltas_mm=a_deltas,
        unchanged_axis_geometry_max_delta_mm=float(max(unchanged_offsets)))


def refine_spacing(fig, axes, state, fraction):
    """Halve visible inter-row whitespace while preserving every plot size."""
    def descendants(ax):
        return [ax, *[a for child in ax.child_axes for a in descendants(child)]]
    upper_axes = set([*descendants(axes["a"]), *descendants(axes["b"]), state["top"], state["c_container"]])
    legends = [state["b_legend"], state["c_legend"]]
    upper_texts = set([*state["titles"], *[t for legend in legends for t in legend.get_texts()],
        *[t for ax in upper_axes for t in ax.findobj(matplotlib.text.Text)]])
    def row_boxes():
        fig.canvas.draw(); renderer = fig.canvas.get_renderer()
        upper, lower = [], []
        for ax in all_axes(fig):
            if ax.has_data():
                (upper if ax in upper_axes else lower).append(ax.get_window_extent(renderer))
        for text, box in V53.manuscript._visible_text_bboxes(fig):
            (upper if text.axes in upper_axes or text in upper_texts else lower).append(box)
        upper.extend(legend.get_window_extent(renderer) for legend in legends)
        return Bbox.union(upper), Bbox.union(lower)
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    a_tag = next(t for t in axes["a"].texts if t.get_text() == "a")
    c_tag = next(t for ax in upper_axes for t in ax.texts if t.get_text() == "c")
    delta_px = a_tag.get_window_extent(renderer).y1 - c_tag.get_window_extent(renderer).y1
    xy = c_tag.get_transform().transform(c_tag.get_position())
    c_tag.set_position(c_tag.get_transform().inverted().transform(xy + [0, delta_px]))
    before_upper, before_lower = row_boxes()
    width, old_height = fig.get_size_inches() * 25.4
    old_gap = (before_upper.y0 - before_lower.y1) * 25.4 / fig.dpi
    if old_gap <= 0: raise RuntimeError("Inter-row visible gap is not positive")
    reduction = old_gap * (1 - fraction)
    height = old_height - reduction
    inherited_axes = all_axes(fig)
    boxes = {ax: np.asarray([ax.get_position().x0 * width, ax.get_position().y0 * old_height,
        ax.get_position().width * width, ax.get_position().height * old_height]) for ax in inherited_axes}
    figure_texts = {t: (t.get_position()[0] * width, t.get_position()[1] * old_height)
        for t in fig.findobj(matplotlib.text.Text) if t.get_transform() is fig.transFigure}
    anchors = {legend: [legend.get_bbox_to_anchor().x0 * 25.4 / fig.dpi,
                       legend.get_bbox_to_anchor().y0 * 25.4 / fig.dpi] for legend in legends}
    science_before = numerical_hash(state["inherited_axes"])
    fig.set_size_inches(width / 25.4, height / 25.4)
    for ax, box in boxes.items():
        placed = box.copy()
        if ax in upper_axes: placed[1] -= reduction
        V53._place(ax, placed.tolist(), width, height)
    for text, (x, y) in figure_texts.items():
        if text.axes in upper_axes or text in upper_texts: y -= reduction
        text.set_position((x / width, y / height))
    for legend, (x, y) in anchors.items():
        legend.set_bbox_to_anchor((x / width, (y - reduction) / height), transform=fig.transFigure)
    after_upper, after_lower = row_boxes()
    new_gap = (after_upper.y0 - after_lower.y1) * 25.4 / fig.dpi
    renderer = fig.canvas.get_renderer()
    alignment = abs(a_tag.get_window_extent(renderer).y1 - c_tag.get_window_extent(renderer).y1) * 25.4 / fig.dpi
    max_geometry_error = 0.0
    for ax, box in boxes.items():
        actual = np.asarray([ax.get_position().x0 * width, ax.get_position().y0 * height,
            ax.get_position().width * width, ax.get_position().height * height])
        expected = box.copy()
        if ax in upper_axes: expected[1] -= reduction
        max_geometry_error = max(max_geometry_error, float(np.max(abs(actual - expected))))
    state["refinement_qa"] = {
        "visible_row_gap_before_mm": old_gap, "visible_row_gap_after_mm": new_gap,
        "requested_gap_fraction": fraction, "canvas_height_reduction_mm": reduction,
        "design_canvas_before_mm": [width, old_height], "design_canvas_after_mm": [width, height],
        "panel_a_c_label_top_alignment_error_mm": alignment,
        "plot_sizes_and_row_translation_max_error_mm": max_geometry_error,
        "numeric_hash_before_spacing": science_before,
        "numeric_hash_after_spacing": numerical_hash(state["inherited_axes"]),
        "upper_row_visible_boxes_before_after_mm": [V53._box_mm(before_upper, fig), V53._box_mm(after_upper, fig)],
        "lower_row_visible_boxes_before_after_mm": [V53._box_mm(before_lower, fig), V53._box_mm(after_lower, fig)],
    }


def layout_qa(fig, state, comparison):
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    texts = V53.manuscript._visible_text_bboxes(fig)
    overlaps = [[a.get_text(), b.get_text()] for i, (a, aa) in enumerate(texts)
                for b, bb in texts[i + 1:] if aa.overlaps(bb)]
    clipping = V53.manuscript.validate_text_within_canvas(fig, raise_on_error=False)
    bar, sweeps, top = state["bar"], state["sweeps"], state["top"]
    legends = [state["b_legend"], state["c_legend"]]
    guide_texts = [*state["titles"], bar.title, bar.xaxis.label, bar.yaxis.label,
        *top.get_xticklabels(), top.xaxis.label, *sweeps[-1].get_xticklabels(),
        sweeps[-1].xaxis.label, sweeps[1].yaxis.label,
        *[t for ax in [bar, *sweeps] for t in ax.get_yticklabels()]]
    text_data_hits = []
    for text in guide_texts:
        if not text.get_visible() or not text.get_text(): continue
        box = text.get_window_extent(renderer)
        for ax in [bar, *sweeps]:
            if box.overlaps(ax.get_window_extent(renderer)):
                text_data_hits.append([text.get_text(), ax.get_label()])
    for legend in legends:
        for ax in [bar, *sweeps]:
            if legend.get_window_extent(renderer).overlaps(ax.get_window_extent(renderer)):
                text_data_hits.append(["legend", ax.get_label()])
    width, height = fig.get_size_inches() * 25.4
    boxes = [[ax.get_position().x0 * width, ax.get_position().y0 * height,
              ax.get_position().x1 * width, ax.get_position().y1 * height] for ax in [bar, *sweeps]]
    bottom_tick_positions = sweeps[-1].transData.transform(np.column_stack([state["counts"], np.zeros(5)]))[:, 0]
    top_tick_positions = top.get_xaxis_transform().transform(np.column_stack([state["counts"], np.zeros(5)]))[:, 0]
    lines = [l for ax in sweeps for l in ax.lines if str(l.get_gid() or "").startswith("model-line:")]
    hero = [l for l in lines if l.get_gid() == "model-line:DMFGen"]
    baseline_lines = [l for l in lines if l not in hero]
    patches = [p for p in bar.patches if str(p.get_gid() or "").startswith("model-bar:")]
    checks = {
        "all_v5_3_scientific_metadata_exact": all(comparison.values()),
        "all_numerical_artists_arrays_intervals_and_limits_exact": state["numerical_before"] == state["numerical_after"],
        "before_gap_compression_panel_a_nested_geometry_preserved": np.allclose(state["panel_a_geometry_deltas_mm"], [0, 4, 0, 0], atol=.001),
        "before_gap_compression_other_inherited_geometry_exact": state["unchanged_axis_geometry_max_delta_mm"] < .001,
        "no_text_text_overlap": not overlaps,
        "no_clipped_text": not any(clipping.values()),
        "no_b_c_titles_ticks_or_legends_inside_data_or_on_spines": not text_data_hits,
        "bar_pure_white_and_background_spans_removed": bar.get_facecolor() == to_rgba("white") and len(bar.patches) == 20 and state["removed_backgrounds"] >= 1,
        "bar_twenty_thin_dark_borders": len(patches) == 20 and all(p.get_linewidth() == .5 for p in patches),
        "bar_lightgray_dashed_grid_below_bars": bar.get_axisbelow() and all(l.get_color() == "lightgray" and l.get_alpha() == .3 and l.get_linestyle() == "--" and l.get_zorder() == 0 for l in bar.get_ygridlines() if l.get_visible()),
        "dedicated_b_horizontal_four_method_patch_legend": len(state["b_legend"].legend_handles) == 4 and all(isinstance(p, Rectangle) for p in state["b_legend"].legend_handles),
        "b_and_c_bottom_spines_identical": abs(boxes[0][1] - boxes[-1][1]) < .001,
        "all_c_left_and_right_spines_identical": np.ptp(np.asarray(boxes[1:])[:, [0, 2]], axis=0).max() < .001,
        "all_c_sweep_heights_identical": np.ptp(np.asarray(boxes[1:])[:, 3] - np.asarray(boxes[1:])[:, 1]) < .001,
        "c_three_hero_lines_red_solid_2_2pt_large_solid_markers": len(hero) == 3 and all(l.get_linewidth() == 2.2 and l.get_linestyle() == "-" and l.get_markersize() == 5.2 and l.get_markerfacecolor() != "none" for l in hero),
        "c_nine_baselines_solid_1_2pt_hollow_geometric_markers": len(baseline_lines) == 9 and all(l.get_linewidth() == 1.2 and l.get_linestyle() == "-" and l.get_markerfacecolor() == "none" and l.get_marker() in {"s", "^", "D"} for l in baseline_lines),
        "c_bottom_axis_only_sensor_counts": [t.get_text() for t in sweeps[-1].get_xticklabels()] == [str(n) for n in state["counts"]],
        "c_top_axis_exact_density_mapping_and_symmetric_ticks": np.allclose(top_tick_positions, bottom_tick_positions, atol=.01) and np.allclose(state["density"], np.asarray(state["counts"]) * 100 / 16384),
    }
    return dict(revision="V5_4", canvas_mm=[width, height], checks=checks,
        all_gates_passed=all(checks.values()), text_overlaps=overlaps, clipped_text=clipping,
        text_or_legend_data_intersections=text_data_hits, b_c_spine_boxes_mm=boxes,
        sensor_counts=state["counts"], H_grid_density_percent_exact=state["density"],
        top_bottom_tick_alignment_error_px=float(np.max(abs(top_tick_positions - bottom_tick_positions))),
        panel_a_geometry_deltas_mm=state["panel_a_geometry_deltas_mm"],
        unchanged_axis_geometry_max_delta_mm=state["unchanged_axis_geometry_max_delta_mm"],
        b_c_text_boxes_mm=[{"text": t.get_text(), "box": V53._box_mm(box, fig)} for t, box in texts
            if t.get_text() in {"H-grid density (%)", "Mixed-HML", "1.6", "2.3", "Training cases", "Relative $L_2$"}],
        numerical_artist_hash_before=state["numerical_before"], numerical_artist_hash_after=state["numerical_after"])


def previews(pdf, output):
    with fitz.open(pdf) as doc:
        doc[0].get_pixmap(dpi=160, alpha=False).save(output / "preview.png")
        pixmap = doc[0].get_pixmap(dpi=270, alpha=False)
        with Image.open(io.BytesIO(pixmap.tobytes("png"))) as image:
            image.save(output / "preview_162mm.png", dpi=(300, 300))
    with Image.open(output / "preview_162mm.png") as source:
        rgb = source.convert("RGB")
        ImageOps.grayscale(rgb).save(output / "preview_grayscale.png")
        matrix = np.asarray([[.367, .861, -.228], [.280, .673, .047], [-.012, .043, .969]])
        values = np.asarray(rgb, dtype=float) / 255
        Image.fromarray(np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8)).save(output / "preview_deuteranopia.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    V53.add_common_args(parser, models=False)
    parser.set_defaults(run_id=DEFAULT_RUN_ID)
    parser.add_argument("--layout", type=Path, default=LAYOUT)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    parser.add_argument("--trial-dir", type=Path)
    args = parser.parse_args()
    baseline = json.loads((BASELINE / "source_manifest_v5_3.json").read_text())
    source_records = baseline["source_data_records"] + baseline["cache_source_records"]
    for record in source_records:
        if V53._sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"V5.3 frozen source changed: {record['path']}")
    baseline_hash = V53._sha256(BASELINE_PDF)
    results_before = V53.v4._tree_state(V53.RESULTS_DIR)
    layout = V53.base.load_layout(args.layout.resolve())
    cfg = V53.load_config(args.config)
    V53.v47.v46.v4_5.apply_v4_5_style_contract(cfg)
    output = args.trial_dir.resolve() if args.trial_dir else ROOT / f"figures/generated/art_style_review/MixedResolution_unified_v5_4_{args.run_id}"
    output.mkdir(parents=True, exist_ok=False)
    fig, axes, containers, shared, panels, ctx, cb_axes, cb_texts, comparison = draw_v53(args, cfg, layout, baseline)
    state = restyle(fig, axes, ctx, layout, baseline)
    if "inter_row_gap_fraction" in layout["v5_4_geometry"]:
        refine_spacing(fig, axes, state, layout["v5_4_geometry"]["inter_row_gap_fraction"])
    qa = layout_qa(fig, state, comparison)
    if "refinement_qa" in state:
        refinement = qa["refinement_qa"] = state["refinement_qa"]
        qa["checks"].update({
            "panel_a_and_c_letters_aligned": refinement["panel_a_c_label_top_alignment_error_mm"] < .001,
            "panel_c_secondary_top_axis_line_visible": state["top"].spines["top"].get_visible(),
            "visible_inter_row_gap_exactly_halved": abs(refinement["visible_row_gap_after_mm"] - refinement["visible_row_gap_before_mm"] * .5) < .001,
            "canvas_height_reduced_by_removed_gap": abs(refinement["design_canvas_before_mm"][1] - refinement["design_canvas_after_mm"][1] - refinement["canvas_height_reduction_mm"]) < .001,
            "plot_sizes_and_internal_row_geometry_preserved": refinement["plot_sizes_and_row_translation_max_error_mm"] < .001,
            "spacing_refinement_numerical_artists_exact": refinement["numeric_hash_before_spacing"] == refinement["numeric_hash_after_spacing"],
        })
    qa["physical_colorbars_qa"] = V53._colorbar_qa(fig, shared, cb_axes, cb_texts)
    qa["annotation_collision_qa"] = V53.measure_annotation_collision_gate(fig, minimum_clearance_mm=1.0, strict=False)
    qa["checks"]["physical_colorbars_preserved_and_clear"] = qa["physical_colorbars_qa"]["passed"]
    qa["checks"]["physical_annotations_clear"] = qa["annotation_collision_qa"]["passed"]
    qa["all_gates_passed"] = all(qa["checks"].values())
    V53._write_json(output / "LAYOUT_QA.json", qa)
    if not qa["all_gates_passed"]:
        fig.savefig(output / "failed_preview.png", dpi=160, bbox_inches=None)
        raise RuntimeError(f"V5.4 QA failed: {[k for k,v in qa['checks'].items() if not v]}")
    stem = output / f"MixedResolution_unified_v5_4_{args.run_id}"
    V53.save_figure(fig, stem, cfg, formats=("svg", "pdf", "png"), dpi=600, bbox_inches=None)
    V53.plt.close(fig)
    trim_qa = V53._trim_export_bottom(stem, 180, qa["canvas_mm"][1], layout["v5_geometry"]["bottom_export_trim_mm"])
    previews(stem.with_suffix(".pdf"), output)
    with fitz.open(stem.with_suffix(".pdf")) as document:
        page = document[0]
        text = page.get_text()
        qa["export_qa"] = {
            "pdf_page_count": len(document),
            "pdf_canvas_mm": [page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72],
            "vector_path_count": len(page.get_drawings()),
            "embedded_font_count": len(page.get_fonts()),
            "searchable_dual_axis_labels": all(label in text for label in ("Sensor count", "H-grid density (%)")),
        }
    qa["checks"]["pdf_canvas_matches_final_export"] = np.allclose(qa["export_qa"]["pdf_canvas_mm"], trim_qa["export_canvas_mm"], atol=.001)
    qa["checks"]["pdf_has_editable_vectors_fonts_and_dual_axis_labels"] = (qa["export_qa"]["vector_path_count"] > 100
        and qa["export_qa"]["embedded_font_count"] > 0 and qa["export_qa"]["searchable_dual_axis_labels"])
    qa["all_gates_passed"] = all(qa["checks"].values())
    if not qa["all_gates_passed"]:
        V53._write_json(output / "LAYOUT_QA.json", qa)
        raise RuntimeError("Final PDF export validation failed")
    if results_before != V53.v4._tree_state(V53.RESULTS_DIR): raise RuntimeError("Scientific results tree changed")
    if baseline_hash != V53._sha256(BASELINE_PDF): raise RuntimeError("V5.3 PDF changed")
    for record in source_records:
        if V53._sha256(Path(record["path"])) != record["sha256"]: raise RuntimeError("Frozen source changed during render")
    qa["bottom_trim_qa"] = trim_qa
    V53._write_json(output / "LAYOUT_QA.json", qa)
    for panel in panels.values():
        panel.update(figure_revision="V5_4", source_visual_revision="V5_3")
    panels["b"]["zero_h_region_shaded"] = False
    panels["c"]["dual_x_axis"] = {"bottom": "Sensor count", "top": "H-grid density (%)", "shared_by_all_three_sweeps": True}
    manifest = dict(revision="V5_4", run_id=args.run_id, backend="Python/Matplotlib in fig",
        source_visual_revision="V5_3", baseline_pdf=V53._record(BASELINE_PDF),
        source_data_records=baseline["source_data_records"], cache_source_records=baseline["cache_source_records"],
        panels=panels, scientific_comparison={"status": "PASS", "checks": comparison},
        numerical_artist_hash=state["numerical_after"], renderer=V53._record(Path(__file__)),
        layout_source=V53._record(args.layout), canvas_mm=trim_qa["export_canvas_mm"],
        model_training_or_inference=False, new_prediction_or_evaluation_rows_generated=False,
        release_status=baseline["release_status"])
    if "refinement_qa" in qa:
        manifest["spacing_refinement"] = qa["refinement_qa"]
    V53._write_json(output / "SCIENTIFIC_STATE_COMPARISON.json", {
        "status": "PASS", "metadata_checks": comparison, "artist_before": state["numerical_before"],
        "artist_after": state["numerical_after"], "source_file_count": len(source_records),
        "all_source_hashes_exact": True, "process_results_tree_unchanged": True})
    V53._write_json(output / "SOURCE_LOCK.json", {"baseline": V53._record(BASELINE_PDF), "sources": source_records})
    (output / "figure_contract.md").write_text("# MixedResolution V5.4 contract\n\nThe asymmetric mixed-modality figure retains the V5.3 evidence sequence: native resolution/training budgets; 512-sensor recipe transfer; sensor-count sweeps; fixed physical field/zoom/error example; multiscale components/residuals; complete-scale population matrices. All scientific source files, cache arrays, 20 bar estimates, 60 sweep estimates, confidence intervals, normalizations and axis limits are frozen. No model training or inference was performed. Existing A03/A04 author checks retain their prior status.\n\nPanel b has a pure white background, lightgray dashed grids below its bars, slightly stronger baseline chroma, 0.5-point darker bar edges and a dedicated horizontal patch legend. Panel c has solid 2.2-point red DMF-Gen traces with filled markers and solid 1.2-point baselines with distinct hollow geometric markers. Its shared bottom axis shows sensor counts; the shared top axis shows the original H-grid percentages. Titles occupy external bands. Panel a is translated upward by 4 mm to accommodate the new bar legend; its data/artists are unchanged. The design canvas grows from 210 to 214 mm. The three c plot windows are 17 mm tall, with equal 5-mm title gutters, while b retains its 21-mm height. b/c bottom spines align at 131 mm, b aligns with a's chart rails, and all c spines align.\n")
    (output / "STYLE_CHANGELOG.md").write_text("# MixedResolution V5.4 review\n\nThe requested bar and line styling was applied without changing the scientific inputs, intervals, arrays, model/recipe identities, color limits or axis scales/limits. Twelve source/cache hashes and all numerical-artist hashes match V5.3.\n\n![V5.4 full-page preview](preview.png)\n\nFigure 1. Rebuilt V5.4 layout with the V5.3 evidence preserved.\n\n![V5.4 vector preview at 162 mm](preview_162mm.png)\n\nFigure 2. Vector-derived review for manuscript insertion.\n\n![V5.4 grayscale review](preview_grayscale.png)\n\nFigure 3. Grayscale review of marker and line distinctions.\n\n![V5.4 deuteranopia review](preview_deuteranopia.png)\n\nFigure 4. Color-vision review of the same scientific figure.\n\nFigure index: Figure 1, full page; Figure 2, insertion review; Figure 3, grayscale; Figure 4, deuteranopia.\n")
    if "refinement_qa" in qa:
        refinement = qa["refinement_qa"]
        contract = output / "figure_contract.md"
        contract.write_text(contract.read_text().replace("b/c bottom spines align at 131 mm", "Before inter-row compaction, b/c bottom spines align at 131 mm")
            + f"\nThe final V5.4 refinement aligns the top edges of letters a and c and explicitly displays the secondary top-axis spine. The visible b/c-to-d/e whitespace is reduced from {refinement['visible_row_gap_before_mm']:.6f} to {refinement['visible_row_gap_after_mm']:.6f} mm, exactly half. All upper-row artists move downward together by {refinement['canvas_height_reduction_mm']:.6f} mm; lower panels and all plot dimensions remain fixed. The design height becomes {refinement['design_canvas_after_mm'][1]:.6f} mm and the final export height is {trim_qa['export_canvas_mm'][1]:.6f} mm. Numeric artist hashes remain exact after compaction.\n")
        changelog = output / "STYLE_CHANGELOG.md"
        changelog.write_text(changelog.read_text()
            + f"\nRefinement: a/c letter tops align, the density axis has a visible top spine, and the visible middle gap is halved to {refinement['visible_row_gap_after_mm']:.6f} mm with a matching reduction in canvas height. Every data window retains its physical dimensions.\n")
    manifest["outputs"] = {p.name: V53._record(p) for p in sorted(output.iterdir()) if p.is_file()}
    V53._write_json(output / "source_manifest_v5_4.json", manifest)
    if not args.trial_dir:
        assembled = V53.FIGURES_DIR / "Assembled"
        for ext in (".pdf", ".svg", ".png"):
            target = assembled / stem.with_suffix(ext).name
            if target.exists(): raise FileExistsError(target)
            shutil.copy2(stem.with_suffix(ext), target)
        target = assembled / f"FigureSourceManifest_unified_v5_4_{args.run_id}.json"
        if target.exists(): raise FileExistsError(target)
        shutil.copy2(output / "source_manifest_v5_4.json", target)
    print(json.dumps({"output": str(output), "checks": len(qa["checks"]), "all_passed": qa["all_gates_passed"], "canvas_mm": trim_qa["export_canvas_mm"]}))


if __name__ == "__main__":
    main()

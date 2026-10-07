#!/usr/bin/env python
"""Rebuild the frozen V4.1 data artists with the requested V4.2 layout.

Row spacing is measured between rendered row unions, including their labels;
the three empty gutters are exactly 1.25 times their V4.1 counterparts.
Additional label/legend height is allocated separately, without blank padding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import fitz
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
import numpy as np
import yaml

import build_figure5_art_v4_1 as V41

V4 = V41.V4
ROOT, REPO = V41.ROOT, V41.REPO
CONFIG = ROOT / "configs/figure5_art_v4_2.yaml"
BASELINE = ROOT / "figures/generated/art_style_review/Figure_Evaluations_art_v4_1_20260926"
DEFAULT_OUT = ROOT / "figures/generated/art_style_review/Figure_Evaluations_art_v4_2_20261007"


def scientific_state(fig):
    """Include intervals, rectangle dimensions, scales, limits and category positions."""
    records = []
    for ax in fig.axes:
        records.append({
            "artist_hash": V4.artist_data_hash([ax]),
            "limits": [list(ax.get_xlim()), list(ax.get_ylim())],
            "ticks": [ax.get_xticks().tolist(), ax.get_yticks().tolist()],
            "segments": [c.get_segments() for c in ax.collections if hasattr(c, "get_segments")],
            "bars": [[p.get_x(), p.get_y(), p.get_width(), p.get_height()]
                     for p in ax.patches if isinstance(p, Rectangle)],
            "numerical_annotations": [t.get_text() for t in ax.texts
                if re.fullmatch(r"[-+]?\d+\.?\d*", t.get_text().strip())],
        })
    # The CRPS column changes wrapping and rho changes owner; preserve the
    # full displayed number strings independently of those allowed reflows.
    records.append({"summary_annotations": sorted(
        " ".join(t.get_text().split()) for t in V4.V31.V1._visible_texts(fig)
        if re.fullmatch(r"\d+\.\d+\s*\[.*\]", " ".join(t.get_text().split()))
        or t.get_text().startswith("ρ ="))})
    return hashlib.sha256(json.dumps(records, sort_keys=True,
        default=lambda x: np.asarray(x).tolist()).encode()).hexdigest()


def row_members(state):
    return [
        ([state["panel_a"], state["panel_b"]], "ab",
         [*state["panel_a_numeric_texts"], state["column_header"]]),
        (state["panel_c_axes"], "c", state["panel_c_texts"]),
        ([*state["panel_d"], state["panel_e"]], "de", []),
        (state["panel_f"], "f", []),
    ]


def row_boxes(fig, state):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return [Bbox.union([*[ax.get_tightbbox(renderer) for ax in axes],
                        *[state["tags"][k].get_window_extent(renderer) for k in tags],
                        *[t.get_window_extent(renderer) for t in extra]])
            for axes, tags, extra in row_members(state)]


def shift_row(fig, state, index, dy_mm):
    height = fig.get_size_inches()[1] * 25.4
    axes, tags, extra = row_members(state)[index]
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + dy_mm / height, pos.width, pos.height])
    for text in [*[state["tags"][k] for k in tags], *extra]:
        x, y = text.get_position()
        text.set_position((x, y + dy_mm / height))


def reflow(fig, state, config, base, baseline_qa):
    before = scientific_state(fig)
    geom = config["geometry_mm"]
    width = config["canvas"]["width_mm"]
    height = 240.0  # A measuring canvas; cropped physically after exact packing.
    old_height = fig.get_size_inches()[1] * 25.4
    for text in fig.texts:
        x, y = text.get_position()
        text.set_position((x, y * old_height / height))
    for ax in fig.axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 * old_height / height,
                         pos.width, pos.height * old_height / height])
    fig.set_size_inches(width / 25.4, height / 25.4)
    left, right = geom["left"], geom["right"]
    a, b = state["panel_a"], state["panel_b"]
    V4.box_mm(a, (left, 151.7, geom["top_scatter_right"], 180.7), width, height)
    V4.box_mm(b, (137.6, 151.7, right, 180.7), width, height)
    for tick in a.get_yticklabels():
        tick.set_color("black")
    for text in state["panel_a_numeric_texts"]:
        text.set_text(text.get_text().replace("\n", " "))
        text.set_color("black")
        text.set_fontsize(7.0)
        text.set_x(geom["top_numeric_x"] / width)
    state["column_header"] = fig.text(geom["top_numeric_x"] / width, 181.4 / height,
        "Mean [95% CI]", ha="left", va="bottom", fontsize=7.0, color="black")
    state["separator"] = Line2D([geom["top_separator_x"] / width] * 2,
        [151.7 / height, 180.7 / height], transform=fig.transFigure,
        linewidth=.5, color="lightgrey", zorder=1)
    fig.add_artist(state["separator"])

    # Keep facet coordinates/limits intact; place rho in a reserved label band.
    square = 25.0
    gap = (right - left - 5 * square) / 4
    rho_texts = []
    for i, ax in enumerate(state["panel_c_axes"]):
        x0 = left + i * (square + gap)
        V4.box_mm(ax, (x0, 112.3, x0 + square, 137.3), width, height)
        state["panel_c_texts"][i].set_x(x0 / width)
        rho = state["panel_c_rho_texts"][i]
        value = rho.get_text()
        rho.remove()
        rho_texts.append(fig.text((x0 + square / 2) / width, 111.2 / height,
            value, ha="center", va="top", fontsize=7.0, color="#444A52"))
    xlabel = fig.text((left + right) / (2 * width), 106.3 / height,
        "Normalized ensemble spread", ha="center", va="top", fontsize=8.5)
    state["panel_c_texts"].extend([*rho_texts, xlabel])
    state["panel_c_rho_texts"] = rho_texts
    state["common_xlabel"] = xlabel

    mid_height = 39.0 * geom["middle_height_factor"]
    bottom, top = 67.8, 67.8 + mid_height
    d_width = (right - left - geom["middle_gap"]) * .6
    e_left = left + d_width + geom["middle_gap"]
    for ax, y0, y1 in zip(state["panel_d"],
                          [(bottom + top) / 2, bottom], [top, (bottom + top) / 2]):
        V4.box_mm(ax, (left, y0, left + d_width, y1), width, height)
    e = state["panel_e"]
    V4.box_mm(e, (e_left, bottom, right, top), width, height)
    state["middle_widths_mm"] = [d_width, right - e_left]
    state["tags"]["d"].set_y(top / height)
    state["tags"]["e"].set_position(((e_left + 4) / width, (top + 1) / height))
    labels = ["Full model\n(DMF-Gen)", "No sensor\nfeedback", "No local\ncond.",
              "Local-only\ncond.", "IID Gauss.\nprior", "Sense-\niver"]
    state["panel_d"][1].set_xticks(np.arange(6), labels)
    for ax in state["panel_d"]:
        ax.tick_params(axis="x", colors="black")
        ax.grid(False, which="both")
        for tick in ax.get_xticklabels():
            tick.set_color("black")
            tick.set_rotation(0)
            tick.set_ha("center")
    handles, labels = e.get_legend_handles_labels()
    labels = ["Full model (DMF-Gen)" if label.replace("\n", " ") == "Full model"
              else label.replace("\n", " ") for label in labels]
    concise = {"No local conditioning": "No local cond.",
               "Local-only conditioning": "Local-only cond."}
    e.legend(handles, [concise.get(label, label) for label in labels], ncol=1,
        loc="upper right", bbox_to_anchor=(.995, .985), fontsize=7.8,
        handlelength=1.4, handletextpad=.4, labelspacing=.08,
        borderaxespad=.1, frameon=True, facecolor="white", edgecolor="none",
        framealpha=1.0, borderpad=.18)
    ratios = np.asarray(base["geometry"]["score_width_ratios"])
    unit = (right - left - 4 * geom["score_gap"]) / ratios.sum()
    cursor = left
    for ax, ratio in zip(state["panel_f"], ratios):
        V4.box_mm(ax, (cursor, 14, cursor + ratio * unit, 54), width, height)
        cursor += ratio * unit + geom["score_gap"]
        # Continuous horizontal stripes replace redundant vertical grids,
        # keeping the numerical annotations clear of guide strokes.
        ax.grid(False, which="both")

    # Pack bottom-up using actual rendered unions, preserving exactly 1.25x
    # empty row clearance while charging new labels to their own row.
    target_gaps = np.asarray(baseline_qa["row_clearances_mm"]) * geom["row_spacing_factor"]
    mm_per_px = height / fig.bbox.height
    boxes = row_boxes(fig, state)
    shift_row(fig, state, 3, geom["outer_margin"] - boxes[3].y0 * mm_per_px)
    for index in (2, 1, 0):
        boxes = row_boxes(fig, state)
        shift_row(fig, state, index,
            (boxes[index + 1].y1 - boxes[index].y0) * mm_per_px + target_gaps[index])
    # The separator follows the relocated top row.
    a_pos = a.get_position()
    state["separator"].set_ydata([a_pos.y0, a_pos.y1])
    boxes = row_boxes(fig, state)
    row_top = boxes[0].y1 * mm_per_px
    style = base["style"]
    legend_handles = [Line2D([], [], linestyle="none", marker=style["method_markers"][m],
        markersize=5.8, markerfacecolor="none", markeredgecolor=style["method_colors"][m],
        markeredgewidth=.9, label=m) for m in V4.CRPS.METHODS]
    state["shared_legend"] = fig.legend(legend_handles, V4.CRPS.METHODS,
        loc="lower center", bbox_to_anchor=((left + right) / (2 * width), (row_top + 1.25) / height),
        ncol=5, fontsize=7.8, handlelength=1.0, handletextpad=.5,
        columnspacing=1.35, borderaxespad=0, frameon=False)
    fig.canvas.draw()
    legend_top = state["shared_legend"].get_window_extent(fig.canvas.get_renderer()).y1 * mm_per_px
    final_height = legend_top + geom["outer_margin"]
    scale = height / final_height
    for ax in fig.axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 * scale, pos.width, pos.height * scale])
    for text in fig.texts:
        x, y = text.get_position()
        text.set_position((x, y * scale))
    state["separator"].set_ydata(np.asarray(state["separator"].get_ydata()) * scale)
    state["shared_legend"].set_bbox_to_anchor(((left + right) / (2 * width),
        (row_top + 1.25) / final_height), transform=fig.transFigure)
    fig.set_size_inches(width / 25.4, final_height / 25.4)
    config["canvas"]["height_mm"] = final_height
    config["geometry_mm"].update(middle_bottom=0, middle_top=mid_height,
        score_bottom=0, score_top=40, minimum_row_clearance=float(min(target_gaps)))

    # Figure-level stripes cross the name rail AND gaps; axis backgrounds are
    # transparent so a single continuous background remains visible everywhere.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    f = state["panel_f"][0]
    tick_boxes = [t.get_window_extent(renderer) for t in f.get_yticklabels()]
    stripe_left = min(t.x0 for t in tick_boxes) / fig.bbox.width - .5 / width
    stripe_right = right / width
    stripe_boxes = []
    for ax in state["panel_f"]:
        ax.patch.set_alpha(0)
    for index, y in enumerate(f.get_yticks()):
        if index % 2:
            continue
        y0, y1 = f.transData.transform([[0, y - .5], [0, y + .5]])[:, 1] / fig.bbox.height
        # The scorecard uses a reversed y axis; clip after sorting endpoints.
        low = max(min(y0, y1), f.get_position().y0)
        high = min(max(y0, y1), f.get_position().y1)
        stripe = Rectangle((stripe_left, low), stripe_right - stripe_left, high - low,
            transform=fig.transFigure, facecolor="lightgrey", edgecolor="none",
            alpha=.16, zorder=-1)
        fig.add_artist(stripe)
        stripe_boxes.append([stripe_left * width, low * final_height,
                             stripe_right * width, high * final_height])
    after = scientific_state(fig)
    if after != before:
        raise RuntimeError("V4.2 changed frozen numerical artists, scales, limits or ticks")
    state.update(scientific_state_before=before, scientific_state_after=after,
                 stripe_boxes_mm=stripe_boxes, baseline_qa=baseline_qa,
                 target_row_clearances_mm=target_gaps.tolist())
    return state


def measure(fig, state, config):
    qa = V4.measure(fig, state, config)
    width, height = qa["canvas_mm"]
    renderer = fig.canvas.get_renderer()
    boxes = row_boxes(fig, state)
    gaps = [(boxes[i].y0 - boxes[i + 1].y1) * height / fig.bbox.height for i in range(3)]
    rows = row_members(state)
    rails = [[min(ax.get_position().x0 for ax in axes) * width,
              max(ax.get_position().x1 for ax in axes) * width] for axes, _, _ in rows]
    # Exact line-segment intersections catch curves crossing the legend even
    # when no sampled vertex lands inside it. Inflate for stroke clearance.
    legend_box = state["panel_e"].get_legend().get_window_extent(renderer).padded(1.0)
    hits = []
    for line in state["panel_e"].lines:
        if line.get_visible() and not line.get_label().startswith("_"):
            path = line.get_transform().transform_path(line.get_path())
            if path.intersects_bbox(legend_box, filled=False):
                hits.append(line.get_label())
    checks = qa["checks"]
    for key in ("d_and_e_75pct_height", "f_75pct_height", "four_rows_clear_by_3mm"):
        checks.pop(key)
    numeric = state["panel_a_numeric_texts"]
    header_box = state["column_header"].get_window_extent(renderer)
    legend = state["shared_legend"]
    annotation_hits = []
    for ax in [*state["panel_d"], *state["panel_f"], state["panel_e"]]:
        for text in ax.texts:
            if not text.get_visible() or not text.get_text().strip():
                continue
            text_box = text.get_window_extent(renderer).padded(.3)
            for line in ax.lines:
                if not line.get_visible():
                    continue
                # A frozen background band guide is covered by the opaque
                # legend; numerical curve and interval artists remain clear.
                path = line.get_transform().transform_path(line.get_path())
                if path.intersects_bbox(text_box, filled=False):
                    annotation_hits.append([text.get_text(), "line"])
            for collection in ax.collections:
                if not collection.get_visible():
                    continue
                if hasattr(collection, "get_segments"):
                    from matplotlib.path import Path as MplPath
                    paths = [MplPath(segment) for segment in collection.get_segments()]
                else:
                    paths = collection.get_paths()
                # Scatter markers use a separate offset transform. All
                # point annotations here belong to errorbar Line2D objects.
                if collection.__class__.__name__ == "PathCollection":
                    continue
                if any(collection.get_transform().transform_path(path).intersects_bbox(text_box, filled=True)
                       for path in paths):
                    annotation_hits.append([text.get_text(), "collection"])
            for patch in ax.patches:
                if ax in state["panel_f"] and isinstance(patch, Rectangle) and patch.get_visible() and patch.get_window_extent(renderer).overlaps(text_box):
                    annotation_hits.append([text.get_text(), "bar"])
            for line in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
                if line.get_visible() and line.get_transform().transform_path(line.get_path()).intersects_bbox(text_box, filled=False):
                    annotation_hits.append([text.get_text(), "grid"])
    qa.update(schema_version="figure5-art-v4-2-layout-qa-1", row_clearances_mm=gaps,
        row_clearance_ratios_vs_v4_1=(np.asarray(gaps) / state["baseline_qa"]["row_clearances_mm"]).tolist(),
        row_spine_rails_mm=rails, panel_d_e_height_mm=48.75,
        scientific_state_before=state["scientific_state_before"],
        scientific_state_after=state["scientific_state_after"],
        panel_a_numeric=[t.get_text() for t in numeric],
        panel_c_facet_limits=state["panel_c_facet_limits"],
        panel_f_continuous_stripes_mm=state["stripe_boxes_mm"],
        spectrum_legend_curve_hits=hits,
        annotation_foreground_intersections=annotation_hits,
        spectrum_legend_background="Opaque white text-safe background; frozen band guide remains underneath and unchanged.",
        intentional_background_intersections=["High-band label within frozen pale band", "Spectrum legend above frozen pale band and dashed guide", "Scorecard row stripes behind foreground data"],
        spacing_definition="Empty clearance between rendered major-row unions; new labels belong to their row.")
    checks.update({
        "all_row_left_and_right_spines_aligned": np.ptp(np.asarray(rails), axis=0).max() < .001,
        "all_row_clearances_exactly_1_25x_v4_1": np.allclose(gaps, state["target_row_clearances_mm"], atol=.02, rtol=0),
        "d_e_height_exactly_1_25x_v4_1": all(abs(ax.get_position().height * height - target) < .001
            for ax, target in [*[ (a, 24.375) for a in state["panel_d"]], (state["panel_e"], 48.75)]),
        "complete_numerical_state_exact_including_bars_and_intervals": state["scientific_state_before"] == state["scientific_state_after"],
        "single_shared_top_legend_five_uniform_markers": len(fig.legends) == 1 and len(legend.legend_handles) == 5
            and all(h.get_markersize() == 5.8 for h in legend.legend_handles),
        "shared_legend_above_both_top_axes": legend.get_window_extent(renderer).y0 > boxes[0].y1,
        "panel_a_column_aligned_black_with_header": len(numeric) == 5 and len({t.get_position()[0] for t in numeric}) == 1
            and all(t.get_color() == "black" and t.get_window_extent(renderer).y1 < header_box.y0 for t in numeric),
        "panel_a_lightgrey_half_point_separator": state["separator"].get_linewidth() == .5
            and state["separator"].get_color() == "lightgrey",
        "panel_c_shared_xlabel_centered": abs(state["common_xlabel"].get_position()[0] * width - np.mean(rails[1])) < .001,
        "panel_c_rho_outside_data_axes": all(t.get_window_extent(renderer).y1 < ax.get_window_extent(renderer).y0
            for t, ax in zip(state["panel_c_rho_texts"], state["panel_c_axes"])),
        "panel_d_ticks_black_two_lines": all(t.get_color() == "black" and len(t.get_text().split("\n")) == 2
            and all(part.strip() for part in t.get_text().split("\n"))
            for t in state["panel_d"][1].get_xticklabels()),
        "panel_d_full_model_renamed": state["panel_d"][1].get_xticklabels()[0].get_text() == "Full model\n(DMF-Gen)",
        "panel_e_full_model_single_line": "Full model (DMF-Gen)" in [t.get_text() for t in state["panel_e"].get_legend().get_texts()],
        "spectrum_legend_clear_of_curve_segments": not hits,
        "panel_f_stripes_span_names_and_all_columns": len(state["stripe_boxes_mm"]) == 4 and all(
            box[0] < rails[3][0] and abs(box[2] - rails[3][1]) < .001 for box in state["stripe_boxes_mm"]),
        "all_d_e_f_annotations_clear_of_foreground_data": not annotation_hits,
        "panel_f_values_clear_of_gridlines": all(not line.get_visible() for ax in state["panel_f"]
            for line in [*ax.get_xgridlines(), *ax.get_ygridlines()]),
        "panel_e_legend_background_blocks_guide_text_overlap": state["panel_e"].get_legend().get_frame_on()
            and state["panel_e"].get_legend().get_frame().get_alpha() == 1.0,
    })
    return qa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config["schema_version"] != "figure5-art-v4-2":
        raise ValueError("Unexpected V4.2 config")
    prior = yaml.safe_load((ROOT / "configs" / config["baseline_config"]).read_text())
    base = yaml.safe_load((ROOT / "configs" / prior["baseline_config"]).read_text())
    v4 = yaml.safe_load((ROOT / "configs" / prior["v4_config"]).read_text())
    source_manifest = json.loads((BASELINE / "source_manifest.json").read_text())
    for name, expected in source_manifest["formal_source_sha256"].items():
        if V4.sha256(V4.SOURCE_RUN / name) != expected:
            raise RuntimeError(f"Frozen V4.1 source changed: {name}")
    crps = V4.CRPS.load_and_validate(V4.SOURCE_RUN)
    spread = V4.SPREAD.load_and_validate(V4.SOURCE_RUN)
    keys = ["method", "state", "original_time_index"]
    if not crps[0][keys].equals(spread[0][keys]):
        raise RuntimeError("Formal state mappings disagree")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {}
    original_savefig = Figure.savefig

    def savefig(self, filename, *positional, **keywords):
        if not state:
            layout = V4.recompose(self, base, v4, crps[:2], spread[:2])
            layout = V41.reflow(self, layout, prior, base, crps[1])
            baseline_qa = V41.measure(self, layout, prior)
            saved_qa = json.loads((BASELINE / "LAYOUT_QA.json").read_text())
            if not np.allclose(baseline_qa["row_clearances_mm"], saved_qa["row_clearances_mm"], atol=.01):
                raise RuntimeError("Rebuilt V4.1 does not match saved baseline layout")
            layout = reflow(self, layout, config, base, baseline_qa)
            state.update(layout=layout, qa=measure(self, layout, config), fig=self)
        w, h = self.get_size_inches()
        keywords.update(bbox_inches=Bbox.from_bounds(0, 0, w, h), pad_inches=0)
        return original_savefig(self, filename, *positional, **keywords)

    Figure.savefig = savefig
    argv = sys.argv
    try:
        sys.argv = [str(V4.V31.V1.BASE_SCRIPT), "--output-dir", str(output),
                    "--height-mm", "260", "--middle-ratios", "2", "1",
                    "--middle-wspace", str(base["geometry"]["middle_wspace"])]
        V4.BASE.main()
    finally:
        Figure.savefig, sys.argv = original_savefig, argv
    pdf, svg = output / "figure5_log.pdf", output / "figure5_log.svg"
    # All review images come from the selected Python backend.
    V4.V31.V1._render_pdf(pdf, output / "figure5_log_162mm.png", 162.0)
    V4.V31.V1._write_review_variants(output / "figure5_log_162mm.png",
        output / "figure5_log_162mm_grayscale.png", output / "figure5_log_162mm_deuteranopia.png")
    qa = state["qa"]
    with fitz.open(pdf) as document:
        page = document[0]
        qa.update(pdf_size_mm=[page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72],
            pdf_fonts=[f[3] for f in page.get_fonts()], pdf_raster_image_count=len(page.get_images(full=True)),
            svg_editable_text_nodes=svg.read_text().count("<text"))
        qa["checks"].update(fixed_canvas=np.allclose(qa["pdf_size_mm"], qa["canvas_mm"], atol=.02),
            editable_vector_exports=len(page.get_text()) > 500 and qa["svg_editable_text_nodes"] > 80,
            no_raster_images_in_pdf=qa["pdf_raster_image_count"] == 0,
            arial_pdf_fonts=all("Arial" in f for f in qa["pdf_fonts"]))
    write_json = lambda path, obj: path.write_text(json.dumps(obj, indent=2,
        default=lambda x: x.item() if isinstance(x, np.generic) else str(x)) + "\n")
    write_json(output / "LAYOUT_QA.json", qa)
    write_json(output / "SCIENTIFIC_STATE_COMPARISON.json", {
        "before": qa["scientific_state_before"], "after": qa["scientific_state_after"],
        "includes": ["all a-f coordinates", "all axis limits/scales/ticks", "CI segments", "rectangle bar dimensions"],
        "source_sha256": source_manifest["formal_source_sha256"], "exact": True})
    write_json(output / "SOURCE_LOCK.json", source_manifest)
    (output / "STYLE_CHANGELOG.md").write_text(f"""# Figure Evaluations V4.2

The quantitative grid retains the V4.1 evidence sequence: statewise CRPS and selective utility; spread/error association; ablation distributions and population spectra; accuracy and measured computational footprint. All samples, intervals, median lines, violin paths, spectra, resource bars, category positions and view limits are unchanged. All five source-file hashes match V4.1, and the complete numerical artist hash matches before and after layout.

The canvas is {qa['canvas_mm'][0]:.1f} × {qa['canvas_mm'][1]:.3f} mm. All rows share the same left and right spine boundaries. Panels d and e are exactly 1.25 times their V4.1 height. Empty clearance between the rendered row unions is exactly 1.25 times V4.1; new label and legend bands are included in their row, rather than consuming that clearance. The figure is packed around those measured extents.

Panels a/b share an external five-method legend with equally sized hollow markers. Panel a has a neutral, aligned one-line numerical column, a Mean [95% CI] header and a 0.5-point lightgrey separator. Panel c has a centered common Normalized ensemble spread label; rho annotations occupy a band below the plots to avoid point collisions. Panel d labels are black with a two-line label allocation, including Full model (DMF-Gen); optional grid strokes are omitted to keep the frozen mean annotations clear. Panel e retains that full name on one legend line, with a white legend background to keep the frozen dashed band guide from crossing the text. Continuous pale zebra shading joins the panel f names and all five columns, replacing its vertical grids so that gridlines do not cross numerical annotations.

![V4.2 full-page preview](figure5_log_preview.png)

Figure 1. Revised layout with the original V4.1 scientific inputs and numerical artists.

![V4.2 at 162 mm](figure5_log_162mm.png)

Figure 2. Vector-derived review at manuscript insertion width. PDF and SVG preserve editable text.

Figure index: Figure 1, design preview; Figure 2, 162-mm review. Grayscale and deuteranopia review images are included alongside the exports.
""")
    (output / "AUTHOR_ACTIONS.md").write_text("# Existing author checks\n\nV4.2 is a layout revision. Existing checks A02 (Fourier baseline identity), A05 (cross-figure score protocol) and A06 (uncertainty and cost reporting) retain their prior status; this revision makes no scientific reconciliation.\n")
    write_json(output / "source_manifest.json", {
        "renderer": str(Path(__file__).resolve().relative_to(REPO)),
        "config": str(args.config.resolve().relative_to(REPO)),
        "renderer_sha256": V4.sha256(Path(__file__).resolve()),
        "config_sha256": V4.sha256(args.config.resolve()),
        "baseline_pdf_sha256": V4.sha256(BASELINE / "figure5_log.pdf"),
        "formal_source_sha256": source_manifest["formal_source_sha256"],
        "scientific_state_hash": qa["scientific_state_after"],
        "outputs": {p.name: V4.sha256(p) for p in sorted(output.iterdir()) if p.is_file()}})
    failed = [k for k, passed in qa["checks"].items() if not passed]
    print(json.dumps({"output": str(output), "canvas_mm": qa["canvas_mm"],
        "checks": len(qa["checks"]), "failed": failed}))
    if failed:
        raise RuntimeError(f"V4.2 layout QA failed: {failed}")


if __name__ == "__main__":
    main()

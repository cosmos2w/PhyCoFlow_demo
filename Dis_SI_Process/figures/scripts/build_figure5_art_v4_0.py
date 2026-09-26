#!/usr/bin/env python
"""Recompose Figure 5 art V4.0 from the frozen V3.1 and formal state tables.

The inherited selective-risk, ablation, spectrum, and scorecard artists are
kept intact. The two statewise displays are drawn from validated formal
reductions; formal bootstrap values are checked against the saved summaries.
No model inference is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import fitz
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator, FormatStrFormatter, MaxNLocator
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
REPO = ROOT.parent
CONFIG = ROOT / "configs/figure5_art_v4_0.yaml"
SOURCE_RUN = ROOT / "results/ValidationV3/UQCompare/uq_compare_formal_20260830_v3r6"
DEFAULT_OUT = ROOT / "figures/generated/art_style_review/Figure_Evaluations_art_v4_0_20260926_1143"
ACCEPTED_SOURCE_HASH = "8248fd8165ed1749d42bfc2672ea79aff3a7201e6aad5152b8a9f412c651da32"


def load_module(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V31 = load_module("figure5_art_v3_1_for_v4", SCRIPT_DIR / "build_figure5_art_v3_1.py")
CRPS = load_module("figure5a_statewise_for_v4", SCRIPT_DIR / "build_figure5a_statewise_crps.py")
SPREAD = load_module("figure5b_statewise_for_v4", SCRIPT_DIR / "build_figure5b_statewise_spread_error.py")
BASE = V31.BASE


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def box_mm(ax, box: tuple[float, float, float, float], width: float, height: float) -> None:
    x0, y0, x1, y1 = box
    ax.set_position([x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height])


def artist_data_hash(axes: list) -> str:
    """Hash plot coordinates and axis scales, excluding layout and appearance."""
    payload = []
    for ax in axes:
        payload.append({
            "lines": [[np.asarray(line.get_xdata()).tolist(), np.asarray(line.get_ydata()).tolist()] for line in ax.lines],
            "collections": [{
                "offsets": np.asarray(collection.get_offsets()).tolist(),
                "paths": [path.vertices.tolist() for path in collection.get_paths()],
            } for collection in ax.collections],
            "patches": [patch.get_path().vertices.tolist() for patch in ax.patches],
            "xscale": ax.get_xscale(),
            "yscale": ax.get_yscale(),
        })
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def add_panel_a(fig: Figure, pairs: pd.DataFrame, summary: pd.DataFrame, base: dict,
                box: tuple[float, float, float, float], width: float, height: float):
    typ, style = base["typography_pt"], base["style"]
    ax = fig.add_axes([0, 0, 1, 1])
    box_mm(ax, box, width, height)
    ax.set_xscale("log")
    ax.set_xlim(0.04, 1.0)
    ax.set_ylim(-0.48, 4.48)
    ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.4, 0.8]))
    ax.xaxis.set_major_formatter(FixedFormatter(["0.05", "0.1", "0.2", "0.4", "0.8"]))
    ax.minorticks_off()
    ax.set_yticks([4, 3, 2, 1, 0], CRPS.METHODS)
    ax.tick_params(axis="x", labelsize=typ["tick_label"], length=2, pad=2,
                   width=style["axis_linewidth_pt"], colors="#343A40")
    ax.tick_params(axis="y", labelsize=typ["tick_label"], length=0, pad=3,
                   colors="#343A40")
    ax.grid(axis="x", color="#E6E8EC", lw=style["grid_linewidth_pt"], zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#52565C")
    ax.spines["bottom"].set_linewidth(style["axis_linewidth_pt"])
    ax.set_xlabel("Normalized CRPS (log scale)", fontsize=typ["axis_title"], labelpad=3)
    summaries = summary.set_index("method")
    for index, method in enumerate(CRPS.METHODS):
        y = 4 - index
        color, marker = style["method_colors"][method], style["method_markers"][method]
        values = pairs.loc[pairs.method.eq(method), "macro_normalized_crps"].to_numpy(dtype=float)
        if len(values) != 200:
            raise ValueError(f"Unexpected state count for {method}")
        rng = np.random.default_rng(CRPS.stable_seed(20260925, "figure5a", method))
        jitter = rng.uniform(-0.30, 0.00, size=len(values))
        ax.scatter(values, y + jitter, s=5.0, color=color, alpha=0.43,
                   linewidths=0, rasterized=False, zorder=2)
        mean = float(summaries.loc[method, "mean_normalized_crps"])
        low = float(summaries.loc[method, "crps_ci_low"])
        high = float(summaries.loc[method, "crps_ci_high"])
        ax.errorbar(mean, y + 0.17, xerr=np.array([[mean - low], [high - mean]]),
                    fmt=marker, markersize=style["comparison_marker_size_pt"],
                    markerfacecolor="none", markeredgecolor=color,
                    markeredgewidth=style["comparison_marker_edge_width_pt"],
                    ecolor=color, elinewidth=1.15, capsize=2.3, capthick=1.0, zorder=4)
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold" if tick.get_text() == "DMF-Gen" else "normal")
    return ax


def add_panel_c(fig: Figure, pairs: pd.DataFrame, summary: pd.DataFrame, base: dict,
                geom: dict, width: float, height: float) -> tuple[list, list]:
    typ, style = base["typography_pt"], base["style"]
    left, right, gap = [float(geom[key]) for key in ("spread_left", "spread_right", "spread_gap")]
    bottom, top = float(geom["spread_bottom"]), float(geom["spread_top"])
    panel_width = (right - left - 4 * gap) / 5
    summaries = summary.set_index("method")
    axes, title_texts = [], []
    for index, method in enumerate(SPREAD.METHODS):
        x0 = left + index * (panel_width + gap)
        ax = fig.add_axes([0, 0, 1, 1])
        box_mm(ax, (x0, bottom, x0 + panel_width, top), width, height)
        group = pairs.loc[pairs.method.eq(method)]
        spread = group.macro_normalized_spread.to_numpy(dtype=float)
        error = group.macro_ensemble_mean_relative_l2.to_numpy(dtype=float)
        if len(spread) != 200:
            raise ValueError(f"Unexpected state count for {method}")
        color = style["method_colors"][method]
        ax.scatter(spread, error, s=5.5, color=color, alpha=0.34,
                   linewidths=0, rasterized=False, zorder=2)
        x_med, y_med = SPREAD.quintile_medians(spread, error)
        ax.plot(x_med, y_med, color=color, lw=1.05, marker="o", markersize=2.0,
                markeredgewidth=0, zorder=3)
        x_span = float(spread.max() - spread.min())
        y_span = float(error.max() - error.min())
        ax.set_xlim(max(0.0, float(spread.min()) - 0.055 * x_span), float(spread.max()) + 0.055 * x_span)
        ax.set_ylim(max(0.0, float(error.min()) - 0.065 * y_span), float(error.max()) + 0.065 * y_span)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(axis="both", labelsize=typ["tick_label"], length=2, pad=2,
                       width=style["axis_linewidth_pt"], colors="#343A40")
        ax.grid(axis="y", color="#E6E8EC", lw=style["grid_linewidth_pt"], zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#52565C")
        ax.spines[["left", "bottom"]].set_linewidth(style["axis_linewidth_pt"])
        title_texts.append(fig.text(x0 / width, 191.5 / height, method,
                                    ha="left", va="bottom", fontsize=typ["axis_title"],
                                    color=color, fontweight="bold" if method == "DMF-Gen" else "normal"))
        title_texts.append(fig.text(x0 / width, 186.0 / height,
                                    f"ρ = {summaries.loc[method, 'spearman_rho']:.3f}",
                                    ha="left", va="bottom", fontsize=typ["in_plot_annotation"],
                                    color="#444A52"))
        axes.append(ax)
    title_texts.append(fig.text(5.5 / width, (bottom + top) / (2 * height),
                                "Ensemble-mean\nrelative $L_2$ error", rotation=90,
                                ha="center", va="center", fontsize=typ["axis_title"], color="#252525"))
    title_texts.append(fig.text(0.52, 142.0 / height, "Normalized ensemble spread",
                                ha="center", va="center", fontsize=typ["axis_title"], color="#252525"))
    return axes, title_texts


def recompose(fig: Figure, base: dict, config: dict, crps_data: tuple, spread_data: tuple) -> dict:
    inherited = V31._style_and_reflow(fig, base)
    if inherited["scientific_hash_before"] != ACCEPTED_SOURCE_HASH:
        raise RuntimeError("Inherited Figure 5 source differs from accepted V3.1 baseline")
    if inherited["scientific_data_hash_before"] != inherited["scientific_data_hash_after"]:
        raise RuntimeError("V3.1 styling changed source artist coordinates")
    old_a, old_b, selective = fig.axes[:3]
    score = fig.axes[3:8]
    distribution_top, distribution_bottom, spectrum = fig.axes[8:11]
    retained = [selective, distribution_top, distribution_bottom, spectrum, *score]
    inherited_data_hash = artist_data_hash(retained)
    for ax in (old_a, old_b):
        fig.delaxes(ax)
    for label in list(fig.texts):
        if label.get_text() in set("abcdef"):
            label.remove()

    width = float(config["canvas"]["width_mm"])
    height = float(config["canvas"]["height_mm"])
    geom = config["geometry_mm"]
    fig.set_size_inches(width / 25.4, height / 25.4, forward=True)
    box_mm(selective, tuple(geom["selective_axis"]), width, height)
    left = float(geom["middle_left"])
    right = float(geom["middle_right"])
    middle_gap = float(geom["middle_gap"])
    available = right - left - middle_gap
    d_width = available * 3 / 5
    e_width = available * 2 / 5
    bottom = float(geom["middle_bottom"])
    top = float(geom["middle_top"])
    mid = (bottom + top) / 2
    box_mm(distribution_top, (left, mid, left + d_width, top), width, height)
    box_mm(distribution_bottom, (left, bottom, left + d_width, mid), width, height)
    box_mm(spectrum, (left + d_width + middle_gap, bottom, right, top), width, height)
    lower_ylim_before = distribution_bottom.get_ylim()
    distribution_bottom.set_ylim(lower_ylim_before[0], lower_ylim_before[1] *
                                 float(geom["lower_violin_headroom_factor"]))
    distribution_bottom.set_xticks(np.arange(6), [
        "Full model", "No sensor\nfeedback", "No local\ncond.",
        "Local-only\ncond.", "IID Gauss.\nprior", "Senseiver",
    ])
    for tick in distribution_bottom.get_xticklabels():
        tick.set_rotation(0)
        tick.set_ha("center")
    handles, labels = spectrum.get_legend_handles_labels()
    concise = {
        "No sensor feedback": "No sensor feedback",
        "No local conditioning": "No local cond.",
        "Local-only conditioning": "Local-only cond.",
        "IID Gaussian prior": "IID Gaussian prior",
    }
    spectrum.legend(handles, [concise.get(label.replace("\n", " "), label.replace("\n", " ")) for label in labels],
                    ncol=1, loc="upper right", bbox_to_anchor=(.99, 1.01),
                    fontsize=base["typography_pt"]["standard_legend"],
                    handlelength=1.4, handletextpad=.4, labelspacing=.08,
                    borderaxespad=.1, frameon=False)
    score_left, score_right = float(geom["score_left"]), float(geom["score_right"])
    score_gap = float(geom["score_gap"])
    ratios = np.asarray(base["geometry"]["score_width_ratios"], dtype=float)
    unit = (score_right - score_left - 4 * score_gap) / ratios.sum()
    cursor = score_left
    for ax, ratio in zip(score, ratios):
        axis_width = float(ratio * unit)
        box_mm(ax, (cursor, float(geom["score_bottom"]), cursor + axis_width,
                    float(geom["score_top"])), width, height)
        cursor += axis_width + score_gap

    crps_pairs, crps_summary = crps_data
    spread_pairs, spread_summary = spread_data
    panel_a = add_panel_a(fig, crps_pairs, crps_summary, base,
                          tuple(geom["top_axes"]), width, height)
    panel_c_axes, panel_c_texts = add_panel_c(fig, spread_pairs, spread_summary,
                                             base, geom, width, height)
    typ = base["typography_pt"]
    tags = {}
    tag_positions = {
        "a": (3.0, 246.0), "b": (125.0, 246.0), "c": (3.0, 192.0),
        "d": (3.0, top + 4.1), "e": (left + d_width + middle_gap, top + 4.1),
        "f": (3.0, float(geom["score_top"])),
    }
    for name, (x, y) in tag_positions.items():
        tags[name] = fig.text(x / width, y / height, name, ha="left", va="baseline",
                              fontsize=typ["panel_label"], fontweight="bold", color="#202124")
    if artist_data_hash(retained) != inherited_data_hash:
        raise RuntimeError("Inherited b/d/e/f artist coordinates changed during V4 composition")
    return {
        "inherited_v3_1_scientific_data_hash": inherited["scientific_data_hash_after"],
        "retained_artist_data_hash_before": inherited_data_hash,
        "retained_artist_data_hash_after": artist_data_hash(retained),
        "panel_a": panel_a, "panel_b": selective, "panel_c_axes": panel_c_axes,
        "panel_c_texts": panel_c_texts, "panel_d": [distribution_top, distribution_bottom],
        "panel_e": spectrum, "panel_f": score, "tags": tags,
        "middle_widths_mm": [d_width, e_width],
        "lower_violin_view_limit_before": list(lower_ylim_before),
        "lower_violin_view_limit_after": list(distribution_bottom.get_ylim()),
    }


def measure(fig: Figure, state: dict, config: dict) -> dict:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = [float(config["canvas"][key]) for key in ("width_mm", "height_mm")]
    sx, sy = width / fig.bbox.width, height / fig.bbox.height
    text_artists = V31.V1._visible_texts(fig)
    boxes = [(artist, artist.get_window_extent(renderer)) for artist in text_artists]
    canvas = Bbox.from_bounds(0, 0, fig.bbox.width, fig.bbox.height)
    clipped = [artist.get_text() for artist, box in boxes
               if not canvas.contains(*box.get_points()[0]) or not canvas.contains(*box.get_points()[1])]
    text_overlap = []
    for i, (a, first) in enumerate(boxes):
        for b, second in boxes[i + 1:]:
            if Bbox.overlaps(first, second):
                text_overlap.append([a.get_text(), b.get_text()])
    tick_overlaps = []
    for axis_index, ax in enumerate(fig.axes):
        for direction, ticks in (("x", ax.get_xticklabels()), ("y", ax.get_yticklabels())):
            visible = [tick.get_window_extent(renderer) for tick in ticks if tick.get_visible() and tick.get_text().strip()]
            for i, first in enumerate(visible):
                for j, second in enumerate(visible[i + 1:], start=i + 1):
                    if Bbox.overlaps(first, second):
                        tick_overlaps.append([axis_index, direction, i, j])
    nonowned_axes = []
    for artist, box in boxes:
        owner = V31.V2._text_owner(fig, artist)["axis_index"]
        for axis_index, ax in enumerate(fig.axes):
            if axis_index != owner and Bbox.overlaps(box, ax.get_window_extent(renderer)):
                nonowned_axes.append([artist.get_text(), owner, axis_index])

    rows = [
        [state["panel_a"], state["panel_b"]],
        state["panel_c_axes"],
        [*state["panel_d"], state["panel_e"]],
        state["panel_f"],
    ]
    row_labels = ["ab", "c", "de", "f"]
    row_extra = [[], state["panel_c_texts"], [], []]
    row_boxes = []
    for axes, labels, extra in zip(rows, row_labels, row_extra):
        members = [ax.get_tightbbox(renderer) for ax in axes]
        members.extend(state["tags"][letter].get_window_extent(renderer) for letter in labels)
        members.extend(text.get_window_extent(renderer) for text in extra)
        row_boxes.append(Bbox.union(members))
    clearances = [(row_boxes[i].y0 - row_boxes[i + 1].y1) * sy for i in range(3)]
    spectrum = state["panel_e"]
    legend_box = spectrum.get_legend().get_window_extent(renderer)
    legend_curve_hits = []
    for line in spectrum.lines:
        if not line.get_visible() or line.get_label().startswith("_"):
            continue
        points = line.get_transform().transform(line.get_path().vertices)
        if any(legend_box.contains(float(x), float(y)) for x, y in points):
            legend_curve_hits.append(line.get_label())
    tag_axes_overlap = [
        [name, fig.axes.index(ax)] for name, tag in state["tags"].items()
        for ax in fig.axes if Bbox.overlaps(tag.get_window_extent(renderer), ax.get_window_extent(renderer))
    ]
    axis_boxes = {name: [[round(ax.get_position().x0 * width, 4), round(ax.get_position().y0 * height, 4),
                           round(ax.get_position().x1 * width, 4), round(ax.get_position().y1 * height, 4)]
                          for ax in axes]
                  for name, axes in zip(row_labels, rows)}
    actual_tick_fonts = [tick.get_fontsize() * 162 / width for ax in fig.axes
                         for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]
                         if tick.get_visible() and tick.get_text().strip()]
    result = {
        "canvas_mm": [width, height],
        "axis_boxes_mm": axis_boxes,
        "middle_widths_mm": state["middle_widths_mm"],
        "top_column_widths_mm": [float(config["geometry_mm"]["top_column_split"]),
                                 width - float(config["geometry_mm"]["top_column_split"])],
        "lower_violin_view_limit_before": state["lower_violin_view_limit_before"],
        "lower_violin_view_limit_after": state["lower_violin_view_limit_after"],
        "middle_width_ratio": state["middle_widths_mm"][0] / state["middle_widths_mm"][1],
        "middle_height_ratio_vs_v3_1": (float(config["geometry_mm"]["middle_top"]) - float(config["geometry_mm"]["middle_bottom"])) / 65.6,
        "score_height_ratio_vs_v3_1": (float(config["geometry_mm"]["score_top"]) - float(config["geometry_mm"]["score_bottom"])) / 66.0,
        "row_clearances_mm": clearances,
        "row_right_edges_mm": [box.x1 * sx for box in row_boxes],
        "outer_artist_margins_mm": [height - row_boxes[0].y1 * sy, row_boxes[-1].y0 * sy,
                                    min(box.x0 for box in row_boxes) * sx,
                                    width - max(box.x1 for box in row_boxes) * sx],
        "clipped_text": clipped,
        "text_text_overlaps": text_overlap,
        "tick_label_overlaps": tick_overlaps,
        "text_nonowned_axes_overlaps": nonowned_axes,
        "panel_tag_axes_overlaps": tag_axes_overlap,
        "spectrum_legend_curve_hits": sorted(set(legend_curve_hits)),
        "inherited_v3_1_scientific_data_hash": state["inherited_v3_1_scientific_data_hash"],
        "retained_artist_data_hash_before": state["retained_artist_data_hash_before"],
        "retained_artist_data_hash_after": state["retained_artist_data_hash_after"],
        "font_pt_at_162mm": {
            "configured_tick": float(yaml.safe_load((ROOT / "configs/figure5_art_v3_1.yaml").read_text())["typography_pt"]["tick_label"]) * 162 / width,
            "minimum_actual_tick": min(actual_tick_fonts),
        },
    }
    result["checks"] = {
        "retained_artist_coordinates_exact": result["retained_artist_data_hash_before"] == result["retained_artist_data_hash_after"],
        "d_to_e_width_exact_3_to_2": abs(result["middle_width_ratio"] - 1.5) < 1e-12,
        "top_columns_two_thirds_one_third": abs(result["top_column_widths_mm"][0] / width - 2 / 3) < .002,
        "d_and_e_75pct_height": abs(result["middle_height_ratio_vs_v3_1"] - .75) < 1e-12,
        "f_75pct_height": abs(result["score_height_ratio_vs_v3_1"] - .75) < 1e-12,
        "four_rows_clear_by_3mm": min(clearances) >= float(config["geometry_mm"]["minimum_row_clearance"]),
        "outer_artist_margins_ge_1mm": min(result["outer_artist_margins_mm"]) >= 1.0,
        "no_clipped_text": not clipped,
        "no_text_text_overlap": not text_overlap,
        "no_tick_label_overlap": not tick_overlaps,
        "no_text_nonowned_axes_overlap": not nonowned_axes,
        "no_panel_tag_axes_overlap": not tag_axes_overlap,
        "spectrum_legend_clear_of_curves": not legend_curve_hits,
        "tick_font_ge_7pt_at_162mm": result["font_pt_at_162mm"]["minimum_actual_tick"] >= 7.0,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema_version") != "figure5-art-v4-0":
        raise ValueError("Unexpected V4.0 config schema")
    base = yaml.safe_load((ROOT / "configs" / config["baseline_config"]).read_text())
    crps_pairs, crps_summary, crps_manifest, crps_qa = CRPS.load_and_validate(SOURCE_RUN)
    spread_pairs, spread_summary, spread_manifest, spread_qa = SPREAD.load_and_validate(SOURCE_RUN)
    mapping = ["method", "state", "original_time_index"]
    if not crps_pairs[mapping].equals(spread_pairs[mapping]):
        raise RuntimeError("The CRPS and spread displays have different state orderings")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state: dict = {"styled": False}
    original_savefig = Figure.savefig

    def savefig_v4(self: Figure, filename, *positional, **keywords):
        if not state["styled"]:
            state["layout"] = recompose(self, base, config, (crps_pairs, crps_summary),
                                        (spread_pairs, spread_summary))
            state["qa"] = measure(self, state["layout"], config)
            state["styled"] = True
        width_in, height_in = self.get_size_inches()
        keywords["bbox_inches"] = Bbox.from_bounds(0, 0, width_in, height_in)
        keywords["pad_inches"] = 0
        return original_savefig(self, filename, *positional, **keywords)

    Figure.savefig = savefig_v4
    previous_argv = sys.argv
    try:
        sys.argv = [str(V31.V1.BASE_SCRIPT), "--output-dir", str(output),
                    "--height-mm", "260", "--middle-ratios", "2", "1",
                    "--middle-wspace", str(base["geometry"]["middle_wspace"])]
        BASE.main()
    finally:
        sys.argv = previous_argv
        Figure.savefig = original_savefig

    pdf, svg = output / "figure5_log.pdf", output / "figure5_log.svg"
    V31.V1._render_pdf(pdf, output / "figure5_log_162mm.png",
                       float(config["canvas"]["manuscript_width_mm"]))
    V31.V1._write_review_variants(output / "figure5_log_162mm.png",
                                  output / "figure5_log_162mm_grayscale.png",
                                  output / "figure5_log_162mm_deuteranopia.png")
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise RuntimeError("Expected a single-page Figure 5 PDF")
        page = document[0]
        qa = state["qa"]
        qa["pdf_size_mm"] = [page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72]
        qa["pdf_fonts"] = [font[3] for font in page.get_fonts()]
        qa["pdf_text_characters"] = len(page.get_text())
        qa["pdf_raster_image_count"] = len(page.get_images(full=True))
        qa["svg_editable_text_nodes"] = svg.read_text().count("<text")
        qa["checks"].update({
            "fixed_canvas": bool(np.allclose(qa["pdf_size_mm"], qa["canvas_mm"], atol=.02, rtol=0)),
            "pdf_and_svg_editable_text": qa["pdf_text_characters"] > 500 and qa["svg_editable_text_nodes"] > 80,
            "no_raster_images_in_pdf": qa["pdf_raster_image_count"] == 0,
            "arial_pdf_fonts": all("Arial" in name for name in qa["pdf_fonts"]),
        })
    (output / "LAYOUT_QA.json").write_text(json.dumps(qa, indent=2,
        default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n")
    (output / "figure_contract.md").write_text(f"""# Figure 5 art V4.0 contract

- Core conclusion: compare probabilistic quality, selective reconstruction, spread/error association, ablation fidelity, frequency behavior, and accuracy/cost on their recorded units.
- Archetype: four-row quantitative grid; panel order a = statewise normalized CRPS, b = selective risk, c = statewise spread/error, d = two ablation distributions, e = velocity spectrum, f = accuracy/cost scorecard.
- Source: inherited b/d/e/f artists are rebuilt with the V3.1 renderer and retain exact coordinate hash `{qa['retained_artist_data_hash_after']}`. Panels a/c use `{SOURCE_RUN.relative_to(REPO)}/per_state_method.csv` and frozen formal summaries; no inference was run.
- Statistics: panels a/c show 200 paired states per method with 64 draws per state. Panel a hollow symbols and whiskers are arithmetic means and formal 95% temporal moving-block bootstrap intervals (block 25, 2,000 replicates). Panel c's line joins medians of five equal-count spread bins (40 states each); its Spearman rho values are descriptive associations. Each facet has independent linear limits, so slopes are not directly comparable. The formal rho intervals remain in the saved source summary.
- Metric scope: macro quantities give equal weight to Y_CH4, Y_CO, U1, and p; temperature is conditioned on. The two visual changes do not establish calibration or physical causality.
- Layout: {config['canvas']['width_mm']} × {config['canvas']['height_mm']} mm; 162 mm review. Panels d/e are 49.2 mm high, f is 49.5 mm, and d:e data-axis widths are 3:2. The candidate's explanatory headers and duplicated numeric column were moved here from the standalone previews.
- Space-saving labels: `cond.` means `conditioning`, and `Gauss.` means `Gaussian`; the full configuration names are retained in the source tables and V3.1 baseline. No categories or values are removed.
- View-only headroom: the lower violin axis upper limit changes from `{qa['lower_violin_view_limit_before'][1]}` to `{qa['lower_violin_view_limit_after'][1]}` so its highest mean label stays within panel d after height compression. Violin samples, means, category order, and scale type are unchanged.
- Export: editable Arial text in PDF/SVG, vector marks, PNG review plus grayscale and deuteranopia previews. No accepted Figure 5 release is overwritten.
""")
    (output / "source_manifest.json").write_text(json.dumps({
        "schema_version": "figure5-art-v4-0-source-manifest-1",
        "renderer": str(Path(__file__).resolve().relative_to(REPO)),
        "config": str(args.config.resolve().relative_to(REPO)),
        "v3_1_renderer": str((SCRIPT_DIR / "build_figure5_art_v3_1.py").relative_to(REPO)),
        "v3_1_pdf_sha256": sha256(ROOT / "figures/generated/art_style_review/Figure_Evaluations_art_v3_1_20260915_0945/figure5_log.pdf"),
        "formal_source": str(SOURCE_RUN.relative_to(REPO)),
        "formal_source_sha256": {name: sha256(SOURCE_RUN / name) for name in
                                 ("per_state_method.csv", "crps_summary.csv", "spread_error_summary.csv", "manifest.json", "qa.json")},
        "formal_qa": [crps_qa["status"], spread_qa["status"]],
        "draws_per_state": crps_manifest["draws_per_state"],
        "methods": list(CRPS.METHODS),
        "retained_artist_data_hash": qa["retained_artist_data_hash_after"],
        "outputs": {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    }, indent=2) + "\n")
    failed = [name for name, passed in qa["checks"].items() if not passed]
    print(f"Figure 5 art V4.0: {output}")
    for name, passed in qa["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if failed:
        raise RuntimeError(f"Figure 5 V4.0 QA failed: {failed}")


if __name__ == "__main__":
    main()

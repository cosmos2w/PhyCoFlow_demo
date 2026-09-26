#!/usr/bin/env python
"""Build the compact Figure 5 V4.1 candidate from the validated V4.0 artists."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import fitz
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
REPO = ROOT.parent
CONFIG = ROOT / "configs/figure5_art_v4_1.yaml"
DEFAULT_OUT = ROOT / "figures/generated/art_style_review/Figure_Evaluations_art_v4_1_20260926"
V4 = __import__("build_figure5_art_v4_0")


def reflow(fig: Figure, state: dict, config: dict, base: dict, crps_summary) -> dict:
    geom = config["geometry_mm"]
    width = float(config["canvas"]["width_mm"])
    height = float(config["canvas"]["height_mm"])
    retained = [state["panel_b"], *state["panel_d"], state["panel_e"], *state["panel_f"]]
    retained_hash = V4.artist_data_hash(retained)
    fig.set_size_inches(width / 25.4, height / 25.4, forward=True)

    # The two printed model rails now start at the same physical x coordinate.
    V4.box_mm(state["panel_a"], tuple(geom["top_axes"]), width, height)
    V4.box_mm(state["panel_b"], tuple(geom["selective_axis"]), width, height)
    a_box = geom["top_axes"]
    summaries = crps_summary.set_index("method")
    numeric_texts = []
    for index, method in enumerate(V4.CRPS.METHODS):
        y = 4 - index
        mean = float(summaries.loc[method, "mean_normalized_crps"])
        low = float(summaries.loc[method, "crps_ci_low"])
        high = float(summaries.loc[method, "crps_ci_high"])
        y_mm = float(a_box[1]) + (float(a_box[3]) - float(a_box[1])) * (y + 0.48) / 4.96
        numeric_texts.append(fig.text(float(geom["top_numeric_x"]) / width, y_mm / height,
                                      f"{mean:.3f}\n[{low:.3f}, {high:.3f}]",
                                      ha="left", va="center", linespacing=.88,
                                      fontsize=6.8, color=base["style"]["method_colors"][method]))
    state["panel_a_numeric_texts"] = numeric_texts

    # The five independent spread/error views use square data axes. Numeric
    # limits stay in the QA and source table after their tick labels are hidden.
    square = float(geom["spread_square"])
    spread_left = float(geom["spread_left"])
    spread_right = float(geom["spread_right"])
    spread_bottom = float(geom["spread_bottom"])
    gap = (spread_right - spread_left - 5 * square) / 4
    old_texts = state["panel_c_texts"]
    if len(old_texts) != 12:
        raise RuntimeError("Unexpected V4.0 panel-c title inventory")
    model_titles = []
    rho_texts = []
    for index, ax in enumerate(state["panel_c_axes"]):
        x0 = spread_left + index * (square + gap)
        V4.box_mm(ax, (x0, spread_bottom, x0 + square, spread_bottom + square), width, height)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        ax.grid(False)
        title = old_texts[2 * index]
        title.set_position((x0 / width, (spread_bottom + square + .75) / height))
        title.set_va("bottom")
        model_titles.append(title)
        rho_display = old_texts[2 * index + 1].get_text()
        old_texts[2 * index + 1].remove()
        rho_texts.append(ax.text(.96, .04, rho_display, transform=ax.transAxes,
                                 ha="right", va="bottom", fontsize=base["typography_pt"]["in_plot_annotation"],
                                 color="#444A52", zorder=8))
    old_texts[-1].remove()  # The common spread x label belongs in the caption.
    y_title = old_texts[-2]
    y_title.set_position((5.5 / width, (spread_bottom + square / 2) / height))
    state["panel_c_texts"] = [*model_titles, y_title]
    state["panel_c_rho_texts"] = rho_texts
    state["panel_c_facet_limits"] = [
        {"method": method, "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim())}
        for method, ax in zip(V4.SPREAD.METHODS, state["panel_c_axes"])
    ]

    # All bottom three rows have one physical left axis rail.
    left, right = float(geom["middle_left"]), float(geom["middle_right"])
    middle_gap = float(geom["middle_gap"])
    available = right - left - middle_gap
    d_width, e_width = available * .6, available * .4
    bottom, top = float(geom["middle_bottom"]), float(geom["middle_top"])
    mid = (bottom + top) / 2
    distribution_top, distribution_bottom = state["panel_d"]
    V4.box_mm(distribution_top, (left, mid, left + d_width, top), width, height)
    V4.box_mm(distribution_bottom, (left, bottom, left + d_width, mid), width, height)
    V4.box_mm(state["panel_e"], (left + d_width + middle_gap, bottom, right, top), width, height)
    distribution_top.set_ylabel(r"Unobserved $L_2$")
    distribution_bottom.set_ylabel(r"$U_1$ High-band $L_2$")
    for ax in (distribution_top, distribution_bottom):
        ax.yaxis.label.set_fontsize(base["typography_pt"]["tick_label"])
    distribution_top.yaxis.set_label_coords(-.12, .55)
    distribution_bottom.yaxis.set_label_coords(-.12, .45)
    upper_before = distribution_top.get_ylim()
    lower_before = distribution_bottom.get_ylim()
    distribution_top.set_ylim(upper_before[0], upper_before[1] * float(geom["upper_violin_headroom_factor"]))
    distribution_bottom.set_ylim(lower_before[0], lower_before[1] * float(geom["lower_violin_headroom_factor"]))
    state["panel_d_view_limits_v4_1"] = {
        "upper_before": list(upper_before), "upper_after": list(distribution_top.get_ylim()),
        "lower_before": list(lower_before), "lower_after": list(distribution_bottom.get_ylim()),
    }
    state["middle_widths_mm"] = [d_width, e_width]
    state["lower_violin_view_limit_before"] = list(lower_before)
    state["lower_violin_view_limit_after"] = list(distribution_bottom.get_ylim())

    ratios = np.asarray(base["geometry"]["score_width_ratios"], dtype=float)
    score_left, score_right = float(geom["score_left"]), float(geom["score_right"])
    score_gap = float(geom["score_gap"])
    unit = (score_right - score_left - 4 * score_gap) / ratios.sum()
    cursor = score_left
    for ax, ratio in zip(state["panel_f"], ratios):
        axis_width = float(ratio * unit)
        V4.box_mm(ax, (cursor, float(geom["score_bottom"]), cursor + axis_width,
                            float(geom["score_top"])), width, height)
        cursor += axis_width + score_gap

    positions = {
        "a": (3.0, float(a_box[3])), "b": (121.0, float(a_box[3])),
        "c": (3.0, spread_bottom + square + .75),
        "d": (3.0, top), "e": (left + d_width + middle_gap + 4.0, top + 1.0),
        "f": (3.0, float(geom["score_top"])),
    }
    for name, (x, y) in positions.items():
        state["tags"][name].set_position((x / width, y / height))
    if V4.artist_data_hash(retained) != retained_hash:
        raise RuntimeError("The inherited b/d/e/f plot coordinates changed")
    state["retained_artist_data_hash_before"] = retained_hash
    state["retained_artist_data_hash_after"] = V4.artist_data_hash(retained)
    return state


def measure(fig: Figure, state: dict, config: dict) -> dict:
    qa = V4.measure(fig, state, config)
    geom = config["geometry_mm"]
    width = float(config["canvas"]["width_mm"])
    height = float(config["canvas"]["height_mm"])
    renderer = fig.canvas.get_renderer()
    a_right = float(geom["top_column_split"]) / width * fig.bbox.width
    numeric_in_column = all(text.get_window_extent(renderer).x1 <= a_right
                            for text in state["panel_a_numeric_texts"])
    rho_inside = all(ax.get_window_extent(renderer).contains(*text.get_window_extent(renderer).get_points()[0])
                     and ax.get_window_extent(renderer).contains(*text.get_window_extent(renderer).get_points()[1])
                     for ax, text in zip(state["panel_c_axes"], state["panel_c_rho_texts"]))
    aligned = [state["panel_a"], state["panel_c_axes"][0], state["panel_d"][0], state["panel_f"][0]]
    left_positions = [ax.get_position().x0 * width for ax in aligned]
    squares = [abs(ax.get_position().width * width - ax.get_position().height * height)
               for ax in state["panel_c_axes"]]
    no_c_ticks = all(not [tick for tick in [*ax.get_xticklabels(), *ax.get_yticklabels()]
                              if tick.get_visible() and tick.get_text().strip()]
                     for ax in state["panel_c_axes"])
    prior_clearances = [6.586, 4.136, 6.233]
    gap_ratios = [current / old for current, old in zip(qa["row_clearances_mm"], prior_clearances)]
    qa.update({
        "schema_version": "figure5-art-v4-1-layout-qa-1",
        "height_ratio_vs_v4_0": height / 250.0,
        "left_axis_positions_mm": left_positions,
        "panel_c_square_errors_mm": squares,
        "panel_c_facet_limits": state["panel_c_facet_limits"],
        "row_clearance_ratios_vs_v4_0": gap_ratios,
        "panel_d_view_limits_v4_1": state["panel_d_view_limits_v4_1"],
        "panel_a_numeric": [text.get_text() for text in state["panel_a_numeric_texts"]],
    })
    checks = qa["checks"]
    for old_name in ("d_and_e_75pct_height", "f_75pct_height", "four_rows_clear_by_3mm"):
        checks.pop(old_name)
    checks.update({
        "height_reduced_approximately_25pct": .73 <= height / 250.0 <= .77,
        "a_c_d_f_left_axes_aligned": max(left_positions) - min(left_positions) <= .01,
        "panel_a_five_numeric_summaries_in_column": len(state["panel_a_numeric_texts"]) == 5 and numeric_in_column,
        "panel_c_five_square_axes": len(squares) == 5 and max(squares) <= .01,
        "panel_c_numeric_ticks_removed": no_c_ticks,
        "panel_c_rho_inside_axes": rho_inside,
        "row_clearance_at_least_1_5mm": min(qa["row_clearances_mm"]) >= float(geom["minimum_row_clearance"]),
        "row_gaps_approximately_30_to_50pct_v4_0": all(.30 <= ratio <= .50 for ratio in gap_ratios),
        "panel_d_short_y_titles": [ax.get_ylabel() for ax in state["panel_d"]] ==
                                  [r"Unobserved $L_2$", r"$U_1$ High-band $L_2$"],
    })
    return qa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema_version") != "figure5-art-v4-1":
        raise ValueError("Unexpected Figure 5 V4.1 config schema")
    base = yaml.safe_load((ROOT / "configs" / config["baseline_config"]).read_text())
    v4 = yaml.safe_load((ROOT / "configs" / config["v4_config"]).read_text())
    crps_pairs, crps_summary, crps_manifest, crps_qa = V4.CRPS.load_and_validate(V4.SOURCE_RUN)
    spread_pairs, spread_summary, spread_manifest, spread_qa = V4.SPREAD.load_and_validate(V4.SOURCE_RUN)
    mapping = ["method", "state", "original_time_index"]
    if not crps_pairs[mapping].equals(spread_pairs[mapping]):
        raise RuntimeError("The two statewise panels have different state/time mappings")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state: dict = {"styled": False}
    original_savefig = Figure.savefig

    def savefig_v41(self: Figure, filename, *positional, **keywords):
        if not state["styled"]:
            composition = V4.recompose(self, base, v4, (crps_pairs, crps_summary),
                                       (spread_pairs, spread_summary))
            state["layout"] = reflow(self, composition, config, base, crps_summary)
            state["qa"] = measure(self, state["layout"], config)
            state["styled"] = True
        width_in, height_in = self.get_size_inches()
        keywords["bbox_inches"] = Bbox.from_bounds(0, 0, width_in, height_in)
        keywords["pad_inches"] = 0
        return original_savefig(self, filename, *positional, **keywords)

    Figure.savefig = savefig_v41
    previous_argv = sys.argv
    try:
        sys.argv = [str(V4.V31.V1.BASE_SCRIPT), "--output-dir", str(output),
                    "--height-mm", "260", "--middle-ratios", "2", "1",
                    "--middle-wspace", str(base["geometry"]["middle_wspace"])]
        V4.BASE.main()
    finally:
        sys.argv = previous_argv
        Figure.savefig = original_savefig

    pdf, svg = output / "figure5_log.pdf", output / "figure5_log.svg"
    V4.V31.V1._render_pdf(pdf, output / "figure5_log_162mm.png",
                          float(config["canvas"]["manuscript_width_mm"]))
    V4.V31.V1._write_review_variants(output / "figure5_log_162mm.png",
                                     output / "figure5_log_162mm_grayscale.png",
                                     output / "figure5_log_162mm_deuteranopia.png")
    with fitz.open(pdf) as document:
        if document.page_count != 1:
            raise RuntimeError("Expected a single-page PDF")
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
    (output / "figure_contract.md").write_text(f"""# Figure 5 art V4.1 contract

- Core conclusion and source cohort: unchanged from V4.0. The five formal methods each use the same 200 states and 64 draws per state. Inherited b/d/e/f artist coordinates retain hash `{qa['retained_artist_data_hash_after']}`; no inference or training was run.
- Layout: {config['canvas']['width_mm']} × {config['canvas']['height_mm']} mm, a {100 * (1 - qa['height_ratio_vs_v4_0']):.1f}% height reduction from V4.0 at the same width. Axes a/c/d/f share x = {qa['left_axis_positions_mm'][0]:.1f} mm. The five panel-c axes are square, and visible row clearances are {', '.join(f'{value:.2f}' for value in qa['row_clearances_mm'])} mm.
- Panel a: 200 statewise normalized-CRPS points per method on a log x axis. Hollow marker and whiskers show mean and formal 95% temporal moving-block CI; the adjacent two-line numbers repeat those values as `mean` then `[CI low, CI high]`.
- Panel c: each dot is a paired state and each line joins medians of five 40-state spread bins. Rho is Spearman association, not a causal or calibration estimate. The five independent linear numeric axis limits are saved in `LAYOUT_QA.json` and the source CSV; tick numbers and the common x title are intentionally omitted for this compact main-figure version. Facet slopes cannot be compared as effect sizes.
- Panel d: only the two y-axis titles and view-only upper limits changed for the shorter row. The violin samples, means, category order, and log scale are unchanged. `cond.` means `conditioning`; `Gauss.` means `Gaussian`.
- Export: Python/matplotlib, vector PDF/SVG with editable Arial text, 162-mm PNG plus grayscale and deuteranopia previews. V4.0 and V3.1 remain separate comparison baselines.
""")
    (output / "source_manifest.json").write_text(json.dumps({
        "schema_version": "figure5-art-v4-1-source-manifest-1",
        "renderer": str(Path(__file__).resolve().relative_to(REPO)),
        "config": str(args.config.resolve().relative_to(REPO)),
        "v4_0_pdf_sha256": V4.sha256(ROOT / "figures/generated/art_style_review/Figure_Evaluations_art_v4_0_20260926_1143/figure5_log.pdf"),
        "formal_source": str(V4.SOURCE_RUN.relative_to(REPO)),
        "formal_source_sha256": {name: V4.sha256(V4.SOURCE_RUN / name) for name in
                                 ("per_state_method.csv", "crps_summary.csv", "spread_error_summary.csv", "manifest.json", "qa.json")},
        "formal_qa": [crps_qa["status"], spread_qa["status"]],
        "draws_per_state": crps_manifest["draws_per_state"],
        "retained_artist_data_hash": qa["retained_artist_data_hash_after"],
        "outputs": {path.name: V4.sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    }, indent=2) + "\n")
    failed = [name for name, passed in qa["checks"].items() if not passed]
    print(f"Figure 5 art V4.1: {output}")
    for name, passed in qa["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if failed:
        raise RuntimeError(f"Figure 5 V4.1 QA failed: {failed}")


if __name__ == "__main__":
    main()

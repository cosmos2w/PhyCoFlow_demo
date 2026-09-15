#!/usr/bin/env python
"""Build Figure 5 art V3.1 with a compact middle-to-scorecard stack.

The accepted renderer remains the sole source of numerical arrays and
statistics. This wrapper changes only physical layout and artist appearance.
All resource-scorecard uncertainty artists remain stored in the figure object
but are hidden under the author-approved visual override.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import fitz
import matplotlib

matplotlib.use("Agg")
from matplotlib.collections import PolyCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
REPO = ROOT.parent
V3_SCRIPT = SCRIPT_DIR / "build_figure5_art_v3.py"
DEFAULT_CONFIG = ROOT / "configs" / "figure5_art_v3_1.yaml"
DEFAULT_OUT = ROOT / "figures/generated/art_style_review/Figure_Evaluations_art_v3_1_20260915_0945"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V3 = _load_module("figure5_art_v3_helpers", V3_SCRIPT)
V2 = V3.V2
V1 = V3.V1
BASE = V3.BASE


def _errorbar_artists(container: ErrorbarContainer) -> list:
    data_line, caplines, barlinecols = container.lines
    return ([data_line] if data_line is not None else []) + list(caplines) + list(barlinecols)


def _hide_resource_errorbars(score_axes: list) -> dict:
    records = []
    for local_index, ax in enumerate(score_axes[1:], start=1):
        containers = [container for container in ax.containers if isinstance(container, ErrorbarContainer)]
        hidden = 0
        for container in containers:
            for artist in _errorbar_artists(container):
                artist.set_visible(False)
                hidden += 1
        records.append(
            {
                "score_axis_index": local_index,
                "container_count": len(containers),
                "hidden_artist_count": hidden,
                "visible_artist_count": sum(
                    artist.get_visible()
                    for container in containers
                    for artist in _errorbar_artists(container)
                ),
            }
        )
    return {"axes": records, "total_hidden_artists": sum(record["hidden_artist_count"] for record in records)}


def _soften_violins(distribution_axes: tuple, alpha: float) -> list[dict]:
    records = []
    for local_index, ax in enumerate(distribution_axes):
        bodies = [collection for collection in ax.collections if isinstance(collection, PolyCollection) and collection.get_paths()]
        for body in bodies:
            body.set_alpha(alpha)
        records.append({"distribution_axis_index": local_index, "body_count": len(bodies), "alphas": [float(body.get_alpha()) for body in bodies]})
    return records


def _soften_resource_bars(score_axes: list, alpha: float) -> dict:
    filled_records = []
    hollow_records = []
    for local_index, ax in enumerate(score_axes[1:], start=1):
        for patch_index, patch in enumerate(ax.patches):
            face = np.asarray(patch.get_facecolor(), dtype=float)
            is_hollow = face.size < 4 or face[3] == 0 or np.allclose(face[:3], [1.0, 1.0, 1.0], atol=0.02)
            if is_hollow:
                patch.set_alpha(None)
                patch.set_facecolor("none")
                hollow_records.append({"score_axis_index": local_index, "patch_index": patch_index})
            else:
                patch.set_alpha(None)
                patch.set_facecolor((*face[:3], alpha))
                filled_records.append(
                    {"score_axis_index": local_index, "patch_index": patch_index, "face_alpha": float(patch.get_facecolor()[3])}
                )
    return {"filled": filled_records, "hollow": hollow_records}


def _mean_state(fig: Figure) -> list[dict]:
    records = []
    for semantic_panel, ax in (("panel_d_upper", fig.axes[8]), ("panel_d_lower", fig.axes[9])):
        annotations = [text for text in ax.texts if hasattr(text, "xy") and text.get_text().strip()]
        mean_lines = []
        for line in ax.lines:
            xdata = np.asarray(line.get_xdata(), dtype=float)
            ydata = np.asarray(line.get_ydata(), dtype=float)
            if xdata.size == 2 and ydata.size == 2 and np.isfinite(ydata).all() and np.allclose(ydata, ydata[0]):
                mean_lines.append((float(np.mean(xdata)), float(ydata[0])))
        if len(annotations) != 6 or len(mean_lines) != 6:
            raise RuntimeError(f"Could not reconstruct six mean annotations for {semantic_panel}")
        for annotation in sorted(annotations, key=lambda text: float(text.xy[0])):
            center, tip = map(float, annotation.xy)
            mean = min(mean_lines, key=lambda record: abs(record[0] - center))[1]
            records.append(
                {
                    "semantic_id": f"{semantic_panel}_mean_{int(round(center))}",
                    "center": center,
                    "tip": tip,
                    "mean": mean,
                    "display": annotation.get_text(),
                    "text_artist": annotation,
                }
            )
    return records


def _center_distribution_ylabels(fig: Figure, axes: tuple, width_mm: float) -> list[dict]:
    fig.canvas.draw()
    for ax in axes:
        renderer = fig.canvas.get_renderer()
        label = ax.yaxis.label
        box = label.get_window_extent(renderer)
        target_center_px = ax.bbox.x0 / 2.0
        xcoord, ycoord = label.get_position()
        label.set_position((xcoord + (target_center_px - (box.x0 + box.x1) / 2.0) / ax.bbox.width, ycoord))
        fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    px_to_mm_x = width_mm / fig.bbox.width
    records = []
    for ax in axes:
        box = ax.yaxis.label.get_window_extent(renderer)
        actual = (box.x0 + box.x1) / 2.0
        target = ax.bbox.x0 / 2.0
        records.append(
            {
                "axis_index": fig.axes.index(ax),
                "target_center_mm": float(target * px_to_mm_x),
                "actual_center_mm": float(actual * px_to_mm_x),
                "center_error_mm": float(abs(actual - target) * px_to_mm_x),
            }
        )
    return records


def _final_measure(fig: Figure, config: dict, prior: dict, modifications: dict) -> dict:
    panel_labels = {text.get_text(): text for text in fig.texts if text.get_visible() and text.get_text() in set("abcdef")}
    state = {
        "scientific_hash_before": prior["scientific_hash_before"],
        "scientific_data_hash_before": prior["scientific_data_hash_before"],
        "axis_limits_before": prior["axis_limits_before"],
        "authorized_view_limit_override": prior["authorized_view_limit_override"],
        "panel_labels": panel_labels,
        "mean_records": _mean_state(fig),
        "panel_a_marker_records": prior["panel_a_marker_records"],
        "panel_b_marker_records": prior["panel_b_marker_records"],
        "panel_c_marker_records": prior["panel_c_marker_records"],
        "panel_a_cloud_collection_count": prior["panel_a_cloud_collection_count"],
        "inference_memory_override": prior["inference_memory_override"],
        "bold_nonpanel_records": prior["bold_nonpanel_records"],
    }
    qa = V3._measure(fig, config, state)
    qa.update(modifications)
    return qa


def _style_and_reflow(fig: Figure, config: dict) -> dict:
    prior = V3._style_and_reflow(fig, config)
    width_mm = float(config["canvas"]["width_mm"])
    height_mm = float(config["canvas"]["height_mm"])
    geometry = config["geometry"]
    style = config["style"]
    score = fig.axes[3:8]
    distribution_top, distribution_bottom, spectrum = fig.axes[8:11]

    # Wrapped category labels fit on a clean horizontal baseline once the
    # middle row is compressed; the category order and wording are unchanged.
    for ax in (distribution_top, distribution_bottom):
        for label in ax.get_xticklabels():
            label.set_rotation(0)
            label.set_ha("center")

    violin_records = _soften_violins((distribution_top, distribution_bottom), float(style["violin_fill_alpha"]))
    bar_records = _soften_resource_bars(score, float(style["bar_fill_alpha"]))
    resource_errorbars = _hide_resource_errorbars(score)

    high_band_text = next(text for text in spectrum.texts if text.get_text() == "High-band")
    band_arrows = [text for text in spectrum.texts if getattr(text, "arrow_patch", None) is not None]
    if len(band_arrows) != 1:
        raise RuntimeError(f"Expected one Panel-e high-band arrow; found {len(band_arrows)}")
    band_arrow = band_arrows[0]
    band_left, band_right = sorted((float(band_arrow.get_position()[0]), float(band_arrow.xy[0])))
    band_center = (band_left + band_right) / 2.0
    band_arrow.arrow_patch.set_visible(False)
    band_arrow.set_visible(False)
    high_band_text.set_position((band_center, float(geometry["high_band_text_y_axes"])))
    high_band_text.set_ha("center")
    high_band_text.set_va("bottom")
    high_band_text.set_fontsize(float(config["typography_pt"]["in_plot_annotation"]))

    panel_labels = {text.get_text(): text for text in fig.texts if text.get_visible() and text.get_text() in set("abcdef")}
    panel_labels["f"].set_y(score[0].get_position().y1)
    panel_labels["f"].set_va(str(geometry["panel_f_top_tag_vertical_alignment"]))
    ylabel_records = _center_distribution_ylabels(fig, (distribution_top, distribution_bottom), width_mm)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    px_to_mm_y = height_mm / fig.bbox.height
    high_band_box = high_band_text.get_window_extent(renderer)
    legend_box = spectrum.get_legend().get_window_extent(renderer)
    curve_y_pixels = []
    for line in spectrum.lines:
        if not line.get_visible() or line.get_label().startswith("_"):
            continue
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        mask = np.isfinite(xdata) & np.isfinite(ydata) & (xdata >= band_left) & (xdata <= band_right)
        if mask.any():
            curve_y_pixels.extend(line.get_transform().transform(np.column_stack([xdata[mask], ydata[mask]]))[:, 1].tolist())
    if not curve_y_pixels:
        raise RuntimeError("No spectrum curve samples found inside the high-band region")
    high_band_placement = {
        "band_left": band_left,
        "band_right": band_right,
        "band_center": band_center,
        "label_x": float(high_band_text.get_position()[0]),
        "label_y_axes": float(high_band_text.get_position()[1]),
        "arrow_artist_count": len(band_arrows),
        "visible_arrow_artist_count": int(band_arrow.get_visible()) + int(band_arrow.arrow_patch.get_visible()),
        "label_below_legend_clearance_mm": float((legend_box.y0 - high_band_box.y1) * px_to_mm_y),
        "label_above_curves_clearance_mm": float((high_band_box.y0 - max(curve_y_pixels)) * px_to_mm_y),
    }

    middle_height = float(geometry["middle_axis_top_mm"]) - float(geometry["middle_axis_bottom_mm"])
    nominal_gap = float(geometry["middle_axis_bottom_mm"]) - float(geometry["score_axis_top_mm"])
    previous_middle_height = 82.0
    previous_gap = 22.6
    previous_canvas_height = 244.0
    expected_canvas_reduction = (previous_middle_height - middle_height) + (previous_gap - nominal_gap)
    modifications = {
        "v3_1_geometry": {
            "middle_height_mm": middle_height,
            "middle_height_ratio_vs_v3": middle_height / previous_middle_height,
            "nominal_middle_bottom_gap_mm": nominal_gap,
            "nominal_gap_ratio_vs_v3": nominal_gap / previous_gap,
            "canvas_height_reduction_mm": previous_canvas_height - height_mm,
            "expected_canvas_reduction_mm": expected_canvas_reduction,
        },
        "distribution_ylabel_centering": ylabel_records,
        "violin_fill_records": violin_records,
        "resource_bar_fill_records": bar_records,
        "resource_errorbar_override": resource_errorbars,
        "accuracy_errorbars_visible": any(
            artist.get_visible()
            for container in score[0].containers
            if isinstance(container, ErrorbarContainer)
            for artist in _errorbar_artists(container)
        ),
        "high_band_placement": high_band_placement,
        "panel_d_category_tick_rotations_deg": [float(label.get_rotation()) for ax in (distribution_top, distribution_bottom) for label in ax.get_xticklabels()],
    }
    return _final_measure(fig, config, prior, modifications)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema_version") != "figure5-art-v3-1":
        raise ValueError("Unexpected Figure 5 art V3.1 configuration schema")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline_pdf = ROOT / config["baseline"]["pdf"]
    art_v3_pdf = ROOT / config["baseline"]["art_v3_pdf"]

    original_savefig = Figure.savefig
    state: dict = {"styled": False}

    def savefig_fixed(self: Figure, filename, *positional, **keywords):
        if not state["styled"]:
            state["qa"] = _style_and_reflow(self, config)
            if state["qa"]["scientific_data_hash_before"] != state["qa"]["scientific_data_hash_after"]:
                raise RuntimeError("Scientific artist data changed during the V3.1 art pass")
            state["styled"] = True
        width_in, height_in = self.get_size_inches()
        keywords["bbox_inches"] = Bbox.from_bounds(0, 0, width_in, height_in)
        keywords["pad_inches"] = 0
        return original_savefig(self, filename, *positional, **keywords)

    Figure.savefig = savefig_fixed
    previous_argv = sys.argv
    try:
        sys.argv = [
            str(V1.BASE_SCRIPT),
            "--output-dir",
            str(output),
            "--height-mm",
            "260",
            "--middle-ratios",
            "2",
            "1",
            "--middle-wspace",
            str(config["geometry"]["middle_wspace"]),
        ]
        BASE.main()
    finally:
        sys.argv = previous_argv
        Figure.savefig = original_savefig

    pdf = output / "figure5_log.pdf"
    svg = output / "figure5_log.svg"
    V1._render_pdf(pdf, output / "figure5_log_162mm.png", float(config["canvas"]["manuscript_width_mm"]))
    V1._render_pdf(art_v3_pdf, output / "art_v3_162mm.png", float(config["canvas"]["manuscript_width_mm"]))
    V1._write_review_variants(
        output / "figure5_log_162mm.png",
        output / "figure5_log_162mm_grayscale.png",
        output / "figure5_log_162mm_deuteranopia.png",
    )

    document = fitz.open(pdf)
    page = document[0]
    qa = state["qa"]
    qa["schema_version"] = "figure5-art-v3-1-layout-qa-1"
    qa["pdf_size_mm"] = [page.rect.width * 25.4 / 72.0, page.rect.height * 25.4 / 72.0]
    qa["single_page"] = document.page_count == 1
    qa["pdf_font_count"] = len(page.get_fonts())
    qa["pdf_text_characters"] = len(page.get_text())
    qa["svg_editable_text_nodes"] = svg.read_text().count("<text")
    baseline_hash = "8248fd8165ed1749d42bfc2672ea79aff3a7201e6aad5152b8a9f412c651da32"
    limit_changes = [
        axis_index
        for axis_index, (before_limits, after_limits) in enumerate(zip(qa["axis_limits_before"], qa["axis_limits_after"]))
        if before_limits != after_limits
    ]
    authorized_axis = int(qa["authorized_view_limit_override"]["axis_index"])
    geometry_qa = qa["v3_1_geometry"]
    violin_alphas = [alpha for record in qa["violin_fill_records"] for alpha in record["alphas"]]
    filled_bar_alphas = [record["face_alpha"] for record in qa["resource_bar_fill_records"]["filled"]]
    errorbar_axes = qa["resource_errorbar_override"]["axes"]
    expected_methods = set(config["style"]["method_colors"]) & {"DMF-Gen", "FFM-FNO", "FFM-Perceiver", "Latent FM", "SiT"}
    qa["checks"] = {
        "accepted_source_state_confirmed": qa["scientific_hash_before"] == baseline_hash,
        "scientific_artist_data_exact": qa["scientific_data_hash_before"] == qa["scientific_data_hash_after"],
        "only_authorized_lower_violin_view_limit_changed": limit_changes == [authorized_axis],
        "fixed_declared_canvas": bool(np.allclose(qa["pdf_size_mm"], [float(config["canvas"]["width_mm"]), float(config["canvas"]["height_mm"])], atol=0.02, rtol=0)),
        "single_page": qa["single_page"],
        "pdf_text_present": qa["pdf_text_characters"] > 500,
        "svg_text_editable": qa["svg_editable_text_nodes"] > 80,
        "no_clipped_text": not qa["clipped_text"],
        "no_tick_label_overlap": not qa["tick_label_overlaps"],
        "no_text_text_overlap": not qa["text_text_overlaps"],
        "no_text_nonowned_axes_overlap": not qa["text_nonowned_axes_overlaps"],
        "no_panel_tag_axes_overlap": not qa["panel_tag_axes_overlaps"],
        "row_clearances_ge_3mm": min(qa["row_union_clearance_mm"].values()) >= float(config["geometry"]["minimum_row_union_clearance_mm"]),
        "middle_height_exactly_80pct_v3": abs(geometry_qa["middle_height_ratio_vs_v3"] - 0.8) <= 1e-10,
        "both_violin_axes_exactly_80pct_v3": bool(np.allclose(qa["panel_d_axis_heights_mm"], [32.8, 32.8], atol=0.01, rtol=0)),
        "nominal_middle_bottom_gap_reduced_50_to_66pct": 1.0 / 3.0 <= geometry_qa["nominal_gap_ratio_vs_v3"] <= 0.5,
        "canvas_trim_equals_saved_vertical_space": abs(geometry_qa["canvas_height_reduction_mm"] - geometry_qa["expected_canvas_reduction_mm"]) <= 1e-10,
        "distribution_ylabels_centered_in_left_whitespace": max(record["center_error_mm"] for record in qa["distribution_ylabel_centering"]) <= 0.05,
        "twelve_violin_bodies_softened": len(violin_alphas) == 12 and bool(np.allclose(violin_alphas, float(config["style"]["violin_fill_alpha"]), atol=1e-10, rtol=0)),
        "all_filled_resource_bars_softened": len(filled_bar_alphas) > 0 and bool(np.allclose(filled_bar_alphas, float(config["style"]["bar_fill_alpha"]), atol=1e-10, rtol=0)),
        "all_resource_errorbars_hidden": qa["resource_errorbar_override"]["total_hidden_artists"] > 0 and all(record["container_count"] > 0 and record["visible_artist_count"] == 0 for record in errorbar_axes),
        "accuracy_errorbars_remain_visible": qa["accuracy_errorbars_visible"],
        "panel_d_category_labels_horizontal": all(abs(value) <= 1e-10 for value in qa["panel_d_category_tick_rotations_deg"]),
        "twelve_two_sig_digit_means": len(qa["mean_annotations"]) == 12 and all(record["display"] == format(record["mean"], ".2g") for record in qa["mean_annotations"]),
        "mean_tip_clearance_ge_1mm": qa["minimum_mean_to_violin_clearance_mm"] >= float(config["geometry"]["minimum_mean_to_violin_clearance_mm"]),
        "legend_clear_of_curve_points": not qa["spectrum_legend_curve_point_intersections"],
        "high_band_arrow_removed": qa["high_band_placement"]["arrow_artist_count"] == 1 and qa["high_band_placement"]["visible_arrow_artist_count"] == 0,
        "high_band_label_centered_in_shaded_region": abs(qa["high_band_placement"]["label_x"] - qa["high_band_placement"]["band_center"]) <= 1e-10,
        "high_band_label_below_legend": qa["high_band_placement"]["label_below_legend_clearance_mm"] >= 1.0,
        "high_band_label_above_curves": qa["high_band_placement"]["label_above_curves_clearance_mm"] >= 1.0,
        "panel_a_palette_and_open_markers_exact": set(qa["panel_a_marker_records"]) == expected_methods,
        "panel_b_palette_and_open_markers_exact": set(qa["panel_b_marker_records"]) == expected_methods,
        "panel_c_marker_geometry_matches_a": len(qa["panel_c_marker_records"]) == 5,
        "panel_f_row_spacing_matches_a": 0.95 <= qa["panel_f_to_a_row_spacing_ratio"] <= 1.05,
        "dmf_gen_bold_in_a_and_f": qa["bold_nonpanel_records"] == [
            {"axis_index": 0, "text": "DMF-Gen", "weight": "bold"},
            {"axis_index": 3, "text": "DMF-Gen", "weight": "bold"},
        ],
        "tick_font_ge_7pt_at_162mm": bool(qa["tick_label_font_pt"] * qa["insertion_scale"] >= 7.0),
        "legend_font_ge_7pt_at_162mm": bool(qa["legend_font_pt"] * qa["insertion_scale"] >= 7.0),
    }
    (output / "LAYOUT_QA.json").write_text(json.dumps(qa, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)) + "\n")
    failed = [name for name, passed in qa["checks"].items() if not passed]
    if failed:
        raise RuntimeError(f"Figure 5 art V3.1 QA failed: {failed}")

    (output / "figure_contract.md").write_text(f"""# Figure 5 art V3.1 contract

- Core conclusion: unchanged ensemble-quality, ablation, frequency, and accuracy-cost evidence.
- Backend/output: Python/matplotlib; fixed-size editable PDF/SVG plus vector-derived PNG reviews.
- Final size: {config['canvas']['width_mm']} x {config['canvas']['height_mm']} mm; manuscript review at {config['canvas']['manuscript_width_mm']} mm width.
- Data integrity: exact before/after artist-data hash `{qa['scientific_data_hash_after']}`; no values, coordinates, category orders, or scale types changed.
- Visual overrides: both Panel-d axes and Panel e occupy an 80-percent-height middle row; Panel-d violin fills and Panel-f resource-bar fills use alpha {config['style']['violin_fill_alpha']}. Resource errorbar artists are retained but hidden; accuracy intervals remain visible.
- Panel e: the high-band arrow is hidden; `High-band` is centered in the shaded region, beneath the legend and above every sampled curve in that region.
- Collision contract: all visible text, axis ownership, panel tags, and rendered row unions pass the mandatory no-overlap and 3 mm minimum-clearance gates.
""")
    (output / "STYLE_CHANGELOG.md").write_text(f"""# Figure 5 art V3.1 style changelog

- Retained comparison baseline: `Figure_Evaluations_art_v3_20260915_0900`.
- Middle row: reduced from 82.0 to {geometry_qa['middle_height_mm']:.1f} mm (80 percent), with two {qa['panel_d_axis_heights_mm'][0]:.1f} mm violin axes and a matching Panel-e height.
- Row gap: nominal Panel d/e-to-f gap reduced from 22.6 to {geometry_qa['nominal_middle_bottom_gap_mm']:.1f} mm ({(1.0 - geometry_qa['nominal_gap_ratio_vs_v3']) * 100:.1f} percent reduction); renderer-derived row-union clearance remains {qa['row_union_clearance_mm']['middle_to_bottom_mm']:.3f} mm.
- Canvas: reduced from 244.0 to {config['canvas']['height_mm']} mm by exactly the sum of middle-row and nominal-gap savings.
- Panel d: y-axis title boxes are centered in the left-page whitespace; violin fills use alpha {config['style']['violin_fill_alpha']}.
- Panel f: filled resource bars use alpha {config['style']['bar_fill_alpha']}; all uncertainty artists in the four resource axes are hidden but retained in the figure object, while accuracy intervals remain visible.
- Panel e: removed the high-band arrow and centered its label within the shaded region, with {qa['high_band_placement']['label_below_legend_clearance_mm']:.3f} mm legend clearance and {qa['high_band_placement']['label_above_curves_clearance_mm']:.3f} mm curve clearance.
- Typography/palette: unchanged from V3; “DMF-Gen” remains the sole bold model-name exception in panels a and f.
""")
    manifest = {
        "schema_version": "figure5-art-v3-1-source-manifest-1",
        "baseline": str(baseline_pdf.relative_to(REPO)),
        "baseline_sha256": V1._sha256(baseline_pdf),
        "art_v3_comparison": str(art_v3_pdf.relative_to(REPO)),
        "art_v3_sha256": V1._sha256(art_v3_pdf),
        "renderer": str(Path(__file__).resolve().relative_to(REPO)),
        "config": str(args.config.resolve().relative_to(REPO)),
        "accepted_source_hash": qa["scientific_hash_before"],
        "scientific_data_hash": qa["scientific_data_hash_after"],
        "outputs": {path.name: V1._sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[PASS] Figure 5 art V3.1: {output}")
    for name, passed in qa["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build Figure 5 art V2 with row-wide alignment and readable violins.

The accepted renderer remains the sole source of numerical arrays and
statistics.  This wrapper changes only physical axes geometry, typography,
and marker rendering, then records renderer-derived release checks.
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
from matplotlib.figure import Figure
from matplotlib.container import ErrorbarContainer
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
REPO = ROOT.parent
V1_SCRIPT = SCRIPT_DIR / "build_figure5_art_v1.py"
DEFAULT_CONFIG = ROOT / "configs" / "figure5_art_v2.yaml"
DEFAULT_OUT = ROOT / "figures/generated/art_style_review/Figure_Evaluations_art_v2_20260915_0804"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V1 = _load_module("figure5_art_v1_helpers", V1_SCRIPT)
BASE = V1.BASE


def _set_position_mm(ax, box_mm: tuple[float, float, float, float], config: dict) -> None:
    width_mm = float(config["canvas"]["width_mm"])
    height_mm = float(config["canvas"]["height_mm"])
    x0, y0, x1, y1 = box_mm
    ax.set_position([x0 / width_mm, y0 / height_mm, (x1 - x0) / width_mm, (y1 - y0) / height_mm])


def _method_name(text: str) -> str:
    return text.replace("$", "").replace("\\mathrm{", "").replace("}", "").strip()


def _harmonize_comparison_markers(axes: list, config: dict) -> dict:
    size = float(config["style"]["comparison_marker_size_pt"])
    edge = float(config["style"]["comparison_marker_edge_width_pt"])
    marker_map = config["style"]["method_markers"]
    records: dict[str, dict] = {}
    for axis_index, ax in enumerate(axes):
        labels = [tick.get_text() for tick in ax.get_yticklabels()]
        ticks = np.asarray(ax.get_yticks(), dtype=float)
        containers = [container for container in ax.containers if isinstance(container, ErrorbarContainer)]
        candidates = [container.lines[0] for container in containers if container.lines[0] is not None]
        if not candidates:
            candidates = [line for line in ax.lines if line.get_marker() in set(marker_map.values())]
        for line in candidates:
            marker = line.get_marker()
            xdata = np.asarray(line.get_xdata())
            ydata = np.asarray(line.get_ydata())
            if marker in (None, "None", "", " ") or xdata.size != 1 or ydata.size != 1:
                continue
            line.set_markersize(size)
            line.set_markeredgewidth(edge)
            line.set_markerfacecolor("none")
            line.set_markeredgecolor(line.get_color())
            if labels and ticks.size:
                nearest = int(np.argmin(np.abs(ticks - float(ydata[0]))))
                if nearest < len(labels):
                    name = _method_name(labels[nearest])
                    if name in marker_map:
                        line.set_marker(marker_map[name])
                    records[f"{axis_index}:{name}"] = {
                        "marker": line.get_marker(),
                        "markersize_pt": float(line.get_markersize()),
                        "markerfacecolor": str(line.get_markerfacecolor()),
                        "markeredgewidth_pt": float(line.get_markeredgewidth()),
                        "markeredgecolor": str(line.get_markeredgecolor()),
                        "color": str(line.get_color()),
                    }
    return records


def _text_owner(fig: Figure, text: Text) -> dict:
    for axis_index, ax in enumerate(fig.axes):
        if text in ax.get_xticklabels():
            return {"axis_index": axis_index, "role": "xtick"}
        if text in ax.get_yticklabels():
            return {"axis_index": axis_index, "role": "ytick"}
        if text is ax.xaxis.label:
            return {"axis_index": axis_index, "role": "xlabel"}
        if text is ax.yaxis.label:
            return {"axis_index": axis_index, "role": "ylabel"}
        if text in ax.texts:
            return {"axis_index": axis_index, "role": "annotation"}
        legend = ax.get_legend()
        if legend is not None and text in legend.get_texts():
            return {"axis_index": axis_index, "role": "legend"}
    return {"axis_index": None, "role": "figure_text"}


def _style_and_reflow(fig: Figure, config: dict) -> dict:
    before = V1._scientific_hash(fig)
    # Apply the accepted final-size typography and line hierarchy first.
    V1._style_figure(fig, config)
    width_mm = float(config["canvas"]["width_mm"])
    height_mm = float(config["canvas"]["height_mm"])
    geometry = config["geometry"]
    fig.set_size_inches(width_mm / 25.4, height_mm / 25.4, forward=True)
    top = fig.axes[:3]
    score = fig.axes[3:8]
    distribution_top, distribution_bottom, spectrum = fig.axes[8:11]

    left = float(geometry["left_axis_mm"])
    right = float(geometry["right_axis_mm"])
    top_available = right - left - float(geometry["top_gap_ab_mm"]) - float(geometry["top_gap_bc_mm"])
    top_width = top_available / 3.0
    top_x = [left, left + top_width + float(geometry["top_gap_ab_mm"])]
    top_x.append(top_x[1] + top_width + float(geometry["top_gap_bc_mm"]))
    for ax, x0 in zip(top, top_x):
        _set_position_mm(ax, (x0, float(geometry["top_axis_bottom_mm"]), x0 + top_width, float(geometry["top_axis_top_mm"])), config)
    top[0].tick_params(axis="y", pad=3.0)

    middle_gap = float(geometry["middle_panel_gap_mm"])
    middle_available = right - left - middle_gap
    panel_e_width = middle_available / 3.0
    panel_d_width = middle_available * 2.0 / 3.0
    _set_position_mm(distribution_top, (left, float(geometry["violin_upper_bottom_mm"]), left + panel_d_width, float(geometry["violin_upper_top_mm"])), config)
    _set_position_mm(distribution_bottom, (left, float(geometry["violin_lower_bottom_mm"]), left + panel_d_width, float(geometry["violin_lower_top_mm"])), config)
    _set_position_mm(spectrum, (left + panel_d_width + middle_gap, float(geometry["middle_axis_bottom_mm"]), right, float(geometry["middle_axis_top_mm"])), config)

    score_ratios = np.asarray(geometry["score_width_ratios"], dtype=float)
    score_gap = float(geometry["score_gap_mm"])
    score_unit = (right - left - score_gap * 4.0) / float(score_ratios.sum())
    cursor = left
    for ax, ratio in zip(score, score_ratios):
        axis_width = score_unit * float(ratio)
        _set_position_mm(ax, (cursor, float(geometry["score_axis_bottom_mm"]), cursor + axis_width, float(geometry["score_axis_top_mm"])), config)
        cursor += axis_width + score_gap

    distribution_top.set_ylabel("Unobserved-field\nrelative L$_2$")
    distribution_bottom.set_ylabel("High-band relative\nL$_2$ ($U_1$)")
    distribution_top.yaxis.set_label_coords(-0.12, float(geometry["upper_ylabel_y"]))
    distribution_bottom.yaxis.set_label_coords(-0.12, float(geometry["lower_ylabel_y"]))

    # Large open symbols form one visual language in panels a, b, and f.
    marker_records = _harmonize_comparison_markers([top[0], top[1], score[0]], config)

    # The emphasized DMF-Gen row is a deliberate, author-approved exception.
    bold_records = []
    for ax in (top[0], score[0]):
        for tick in ax.get_yticklabels():
            is_dmf = _method_name(tick.get_text()) == "DMF-Gen"
            tick.set_fontweight("bold" if is_dmf else "normal")
            if is_dmf:
                bold_records.append({"axis_index": fig.axes.index(ax), "text": tick.get_text(), "weight": "bold"})

    panel_labels = {text.get_text(): text for text in fig.texts if text.get_visible() and text.get_text() in set("abcdef")}
    if sorted(panel_labels) != list("abcdef"):
        raise ValueError(f"Panel-label inventory differs from a-f: {sorted(panel_labels)}")
    tag_offset = float(geometry["panel_tag_offset_mm"])
    panel_labels["a"].set_y(float(geometry["top_axis_top_mm"]) / height_mm)
    panel_labels["a"].set_va("bottom")
    panel_labels["d"].set_y((float(geometry["middle_axis_top_mm"]) + tag_offset) / height_mm)
    panel_labels["d"].set_va("bottom")
    panel_labels["f"].set_y(float(geometry["score_axis_top_mm"]) / height_mm)
    panel_labels["f"].set_va("bottom")
    for key, ax in zip(("b", "c", "e"), (top[1], top[2], spectrum)):
        panel_labels[key].set_x(ax.get_position().x0)

    # First draw: align both rotated d-axis titles to the exact same visible
    # left rail as the method names in a/f, then align a/d/f tags to that rail.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    method_boxes = [tick.get_window_extent(renderer) for ax in (top[0], score[0]) for tick in ax.get_yticklabels() if tick.get_visible() and tick.get_text().strip()]
    left_rail_px = min(box.x0 for box in method_boxes)
    for ax in (distribution_top, distribution_bottom):
        label = ax.yaxis.label
        box = label.get_window_extent(renderer)
        xcoord, ycoord = label.get_position()
        label.set_position((xcoord + (left_rail_px - box.x0) / ax.bbox.width, ycoord))
    for key in ("a", "d", "f"):
        panel_labels[key].set_x(left_rail_px / fig.bbox.width)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = V1._visible_texts(fig)
    text_boxes = [(text, text.get_window_extent(renderer)) for text in texts]
    canvas = Bbox.from_bounds(0, 0, fig.bbox.width, fig.bbox.height)
    clipped = [
        {"text": text.get_text(), "bbox_px": list(map(float, box.extents)), **_text_owner(fig, text)}
        for text, box in text_boxes
        if not canvas.contains(*box.get_points()[0]) or not canvas.contains(*box.get_points()[1])
    ]
    text_text_overlaps = []
    for first in range(len(text_boxes)):
        first_text, first_box = text_boxes[first]
        for second in range(first + 1, len(text_boxes)):
            second_text, second_box = text_boxes[second]
            if Bbox.overlaps(first_box, second_box):
                text_text_overlaps.append({"first": first_text.get_text(), "second": second_text.get_text(), "first_owner": _text_owner(fig, first_text), "second_owner": _text_owner(fig, second_text)})
    nonowned = []
    for text, box in text_boxes:
        owner = _text_owner(fig, text)["axis_index"]
        for axis_index, ax in enumerate(fig.axes):
            if axis_index != owner and Bbox.overlaps(box, ax.get_window_extent(renderer)):
                nonowned.append({"text": text.get_text(), "owner_axis": owner, "intersected_axis": axis_index})
    tick_overlaps = []
    for axis_index, ax in enumerate(fig.axes):
        for direction, labels in (("x", ax.get_xticklabels()), ("y", ax.get_yticklabels())):
            boxes = [label.get_window_extent(renderer) for label in labels if label.get_visible() and label.get_text().strip()]
            for first in range(len(boxes)):
                for second in range(first + 1, len(boxes)):
                    if Bbox.overlaps(boxes[first], boxes[second]):
                        tick_overlaps.append([axis_index, direction, first, second])

    tags = {key: text.get_window_extent(renderer) for key, text in panel_labels.items()}
    top_union = V1._union([ax.get_tightbbox(renderer) for ax in top] + [tags[key] for key in "abc"])
    middle_union = V1._union([ax.get_tightbbox(renderer) for ax in (distribution_top, distribution_bottom, spectrum)] + [tags[key] for key in "de"])
    bottom_union = V1._union([ax.get_tightbbox(renderer) for ax in score] + [tags["f"]])
    px_to_mm_x = width_mm / fig.bbox.width
    px_to_mm_y = height_mm / fig.bbox.height
    ylabel_boxes = [distribution_top.yaxis.label.get_window_extent(renderer), distribution_bottom.yaxis.label.get_window_extent(renderer)]
    ylabel_boxes.sort(key=lambda box: box.y0)
    d_ylabel_gap = (ylabel_boxes[1].y0 - ylabel_boxes[0].y1) * px_to_mm_y
    left_items = {
        "panel_a_method_text": min(box.x0 for box in [tick.get_window_extent(renderer) for tick in top[0].get_yticklabels() if tick.get_visible() and tick.get_text().strip()]) * px_to_mm_x,
        "panel_d_upper_ylabel": distribution_top.yaxis.label.get_window_extent(renderer).x0 * px_to_mm_x,
        "panel_d_lower_ylabel": distribution_bottom.yaxis.label.get_window_extent(renderer).x0 * px_to_mm_x,
        "panel_f_method_text": min(box.x0 for box in [tick.get_window_extent(renderer) for tick in score[0].get_yticklabels() if tick.get_visible() and tick.get_text().strip()]) * px_to_mm_x,
        "tag_a": tags["a"].x0 * px_to_mm_x,
        "tag_d": tags["d"].x0 * px_to_mm_x,
        "tag_f": tags["f"].x0 * px_to_mm_x,
    }
    score_boxes = [ax.get_window_extent(renderer) for ax in score]
    score_gaps = [(score_boxes[index + 1].x0 - score_boxes[index].x1) * px_to_mm_x for index in range(4)]
    right_items = {
        "row_1_panel_c": top[2].get_window_extent(renderer).x1 * px_to_mm_x,
        "row_2_panel_e": spectrum.get_window_extent(renderer).x1 * px_to_mm_x,
        "row_3_last_score_axis": score[-1].get_window_extent(renderer).x1 * px_to_mm_x,
    }
    tag_axis_overlaps = [[key, index] for key, box in tags.items() for index, ax in enumerate(fig.axes) if Bbox.overlaps(box, ax.get_window_extent(renderer))]
    return {
        "canvas_mm": [width_mm, height_mm],
        "scientific_hash_before": before,
        "scientific_hash_after": V1._scientific_hash(fig),
        "visible_text_count": len(text_boxes),
        "clipped_text": clipped,
        "tick_label_overlaps": tick_overlaps,
        "text_text_overlaps": text_text_overlaps,
        "text_nonowned_axes_overlaps": nonowned,
        "panel_tag_axes_overlaps": tag_axis_overlaps,
        "row_union_clearance_mm": {
            "top_to_middle_mm": float((top_union.y0 - middle_union.y1) * px_to_mm_y),
            "middle_to_bottom_mm": float((middle_union.y0 - bottom_union.y1) * px_to_mm_y),
        },
        "row_unions_mm": {"top": V1._bbox_record(top_union, fig), "middle": V1._bbox_record(middle_union, fig), "bottom": V1._bbox_record(bottom_union, fig)},
        "axis_geometry_mm": {
            "panel_d_width": panel_d_width,
            "panel_e_width": panel_e_width,
            "panel_d_to_e_ratio": panel_d_width / panel_e_width,
            "panel_d_upper_height": float(geometry["violin_upper_top_mm"]) - float(geometry["violin_upper_bottom_mm"]),
            "panel_d_lower_height": float(geometry["violin_lower_top_mm"]) - float(geometry["violin_lower_bottom_mm"]),
            "common_left_axis_mm": left,
            "common_right_axis_mm": right,
            "score_gaps_mm": score_gaps,
        },
        "left_alignment_mm": left_items,
        "right_alignment_mm": right_items,
        "panel_d_ylabel_gap_mm": float(d_ylabel_gap),
        "panel_d_ylabel_max_height_mm": float(max(box.height for box in ylabel_boxes) * px_to_mm_y),
        "marker_records": marker_records,
        "bold_nonpanel_records": bold_records,
        "panel_label_font_pt": float(config["typography_pt"]["panel_label"]),
        "tick_label_font_pt": float(config["typography_pt"]["tick_label"]),
        "legend_font_pt": float(config["typography_pt"]["standard_legend"]),
        "insertion_scale": float(config["canvas"]["manuscript_width_mm"]) / width_mm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema_version") != "figure5-art-v2-1":
        raise ValueError("Unexpected Figure 5 art V2 configuration schema")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline_pdf = ROOT / config["baseline"]["pdf"]

    original_savefig = Figure.savefig
    state: dict = {"styled": False}

    def savefig_fixed(self: Figure, filename, *positional, **keywords):
        if not state["styled"]:
            state["qa"] = _style_and_reflow(self, config)
            if state["qa"]["scientific_hash_before"] != state["qa"]["scientific_hash_after"]:
                raise RuntimeError("Scientific artist coordinates changed during the V2 art pass")
            state["styled"] = True
        width_in, height_in = self.get_size_inches()
        keywords["bbox_inches"] = Bbox.from_bounds(0, 0, width_in, height_in)
        keywords["pad_inches"] = 0
        return original_savefig(self, filename, *positional, **keywords)

    Figure.savefig = savefig_fixed
    previous_argv = sys.argv
    try:
        sys.argv = [str(V1.BASE_SCRIPT), "--output-dir", str(output), "--height-mm", "260", "--middle-ratios", "2", "1", "--middle-wspace", str(config["geometry"]["middle_wspace"])]
        BASE.main()
    finally:
        sys.argv = previous_argv
        Figure.savefig = original_savefig

    pdf = output / "figure5_log.pdf"
    svg = output / "figure5_log.svg"
    V1._render_pdf(pdf, output / "figure5_log_162mm.png", float(config["canvas"]["manuscript_width_mm"]))
    V1._render_pdf(baseline_pdf, output / "baseline_162mm.png", float(config["canvas"]["manuscript_width_mm"]))
    V1._write_review_variants(output / "figure5_log_162mm.png", output / "figure5_log_162mm_grayscale.png", output / "figure5_log_162mm_deuteranopia.png")

    document = fitz.open(pdf)
    page = document[0]
    qa = state["qa"]
    qa["schema_version"] = "figure5-art-v2-layout-qa-1"
    qa["pdf_size_mm"] = [page.rect.width * 25.4 / 72.0, page.rect.height * 25.4 / 72.0]
    qa["single_page"] = document.page_count == 1
    qa["pdf_font_count"] = len(page.get_fonts())
    qa["pdf_text_characters"] = len(page.get_text())
    qa["svg_editable_text_nodes"] = svg.read_text().count("<text")
    left_values = list(qa["left_alignment_mm"].values())
    right_values = list(qa["right_alignment_mm"].values())
    marker_records = qa["marker_records"]
    marker_open = marker_records and all(record["markerfacecolor"] in {"none", "None"} for record in marker_records.values())
    qa["checks"] = {
        "scientific_state_exact": qa["scientific_hash_before"] == qa["scientific_hash_after"],
        "fixed_canvas_180x220_mm": bool(np.allclose(qa["pdf_size_mm"], [180.0, 220.0], atol=0.02, rtol=0)),
        "single_page": qa["single_page"],
        "pdf_text_present": qa["pdf_text_characters"] > 500,
        "svg_text_editable": qa["svg_editable_text_nodes"] > 80,
        "no_clipped_text": not qa["clipped_text"],
        "no_tick_label_overlap": not qa["tick_label_overlaps"],
        "no_text_text_overlap": not qa["text_text_overlaps"],
        "no_text_nonowned_axes_overlap": not qa["text_nonowned_axes_overlaps"],
        "no_panel_tag_axes_overlap": not qa["panel_tag_axes_overlaps"],
        "top_middle_clearance_ge_3mm": qa["row_union_clearance_mm"]["top_to_middle_mm"] >= float(config["geometry"]["minimum_row_union_clearance_mm"]),
        "middle_bottom_clearance_ge_3mm": qa["row_union_clearance_mm"]["middle_to_bottom_mm"] >= float(config["geometry"]["minimum_row_union_clearance_mm"]),
        "panel_d_e_width_ratio_exact_2_to_1": abs(qa["axis_geometry_mm"]["panel_d_to_e_ratio"] - 2.0) <= 1e-10,
        "violin_axes_at_least_33mm_high": min(qa["axis_geometry_mm"]["panel_d_upper_height"], qa["axis_geometry_mm"]["panel_d_lower_height"]) >= 33.0,
        "panel_d_ylabel_clearance_ge_3mm": qa["panel_d_ylabel_gap_mm"] >= 3.0,
        "left_alignment_spread_le_0_05mm": max(left_values) - min(left_values) <= 0.05,
        "right_alignment_spread_le_0_05mm": max(right_values) - min(right_values) <= 0.05,
        "combined_violin_height_exceeds_ylabel": qa["axis_geometry_mm"]["panel_d_upper_height"] + qa["axis_geometry_mm"]["panel_d_lower_height"] > qa["panel_d_ylabel_max_height_mm"],
        "scorecard_equal_gap_spread_le_0_05mm": max(qa["axis_geometry_mm"]["score_gaps_mm"]) - min(qa["axis_geometry_mm"]["score_gaps_mm"]) <= 0.05,
        "comparison_markers_open": bool(marker_open),
        "dmf_gen_bold_in_a_and_f": len(qa["bold_nonpanel_records"]) == 2,
        "tick_font_ge_7pt_at_162mm": bool(qa["tick_label_font_pt"] * qa["insertion_scale"] >= 7.0),
        "legend_font_ge_7pt_at_162mm": bool(qa["legend_font_pt"] * qa["insertion_scale"] >= 7.0),
    }
    (output / "LAYOUT_QA.json").write_text(
        json.dumps(
            qa,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
        + "\n"
    )
    failed = [name for name, passed in qa["checks"].items() if not passed]
    if failed:
        raise RuntimeError(f"Figure 5 art V2 QA failed: {failed}")

    (output / "figure_contract.md").write_text("""# Figure 5 art V2 contract

- Core conclusion and scientific inventory are inherited unchanged from the accepted precision-log renderer.
- Figure archetype: asymmetric quantitative composite for a Nature-family double-column placement.
- Final size: fixed 180 x 220 mm MediaBox; vector-derived review raster at 162 mm insertion width.
- Geometry: rows share one visible left rail and right boundary; Panel d:e axes widths are exactly 2:1 with a 14 mm gutter; both violin axes are 33.5 mm high; Panel f uses four equal 4 mm gutters.
- Styling: comparison markers in a, b, and f are large open symbols; marker identities and colours remain method-specific. “DMF-Gen” alone is bold in the two visible model-label rails by author instruction.
- Data integrity: plotted coordinates, scales, limits, category order, intervals, violin paths, spectra, bars, and numerical annotations are protected by an exact scientific-state hash.
- Review risks: author checks A02, A05, and A06 remain scientific-release gates and are not altered here.
""")
    (output / "STYLE_CHANGELOG.md").write_text(f"""# Figure 5 art V2 style changelog

- Baseline art comparison retained: `Figure_Evaluations_art_v1_20260915_0019`.
- Canvas: increased from 180 x 200 mm to 180 x 220 mm to expand both violin axes without compressing the other evidence rows.
- Alignment: panels a, d, and f tags plus the visible row labels share one measured left rail; all three rows terminate at 178.2 mm.
- Row 2: Panel d:e plotting widths are exactly 2:1 with a 14 mm gutter; each violin axis is 33.5 mm high and their long y-axis titles have independently positioned vertical centres.
- Row 3: five scorecard axes use four equal 4 mm physical gaps; the accuracy markers now match panels a/b as 5.8 pt open symbols.
- Typography: retained the V1 hierarchy. “DMF-Gen” is deliberately bold in panels a and f; all other model labels remain regular.
- Scientific state: exact before/after artist hash `{qa['scientific_hash_before']}`.
- Review: fixed-canvas vector outputs, 162-mm colour/grayscale/deuteranopia previews, and explicit alignment/collision/marker gates are included.
""")
    manifest = {
        "schema_version": "figure5-art-v2-source-manifest-1",
        "baseline": str(baseline_pdf.relative_to(REPO)),
        "baseline_sha256": V1._sha256(baseline_pdf),
        "renderer": str(Path(__file__).resolve().relative_to(REPO)),
        "config": str(args.config.resolve().relative_to(REPO)),
        "scientific_hash": qa["scientific_hash_after"],
        "outputs": {path.name: V1._sha256(path) for path in sorted(output.iterdir()) if path.is_file()},
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[PASS] Figure 5 art V2: {output}")
    for name, passed in qa["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")


if __name__ == "__main__":
    main()

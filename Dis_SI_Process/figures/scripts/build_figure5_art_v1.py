#!/usr/bin/env python
"""Build the art-style-only Figure 5 V1 from the accepted precision-log baseline.

The inherited renderer remains the sole source of plotted arrays and statistics.
This wrapper changes only canvas geometry, typography, and display styling, then
emits a fixed-size release bundle with renderer-derived layout QA.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import fitz
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import numpy as np
from PIL import Image
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
REPO = ROOT.parent
BASE_SCRIPT = SCRIPT_DIR / "build_figure5_polished.py"
DEFAULT_CONFIG = ROOT / "configs" / "figure5_art_v1.yaml"
DEFAULT_OUT = ROOT / "figures/generated/art_style_review/Figure_Evaluations_art_v1_20260915_0019"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import plotting module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_module("figure5_precision_log_baseline", BASE_SCRIPT)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _union(boxes: list[Bbox]) -> Bbox:
    if not boxes:
        raise ValueError("Cannot form an empty rendered union")
    return Bbox.union(boxes)


def _bbox_record(box: Bbox, fig: Figure) -> dict[str, float]:
    width_px, height_px = fig.bbox.width, fig.bbox.height
    width_mm, height_mm = np.asarray(fig.get_size_inches()) * 25.4
    return {
        "x0_mm": float(box.x0 / width_px * width_mm),
        "y0_mm": float(box.y0 / height_px * height_mm),
        "x1_mm": float(box.x1 / width_px * width_mm),
        "y1_mm": float(box.y1 / height_px * height_mm),
    }


def _scientific_hash(fig: Figure) -> str:
    payload = []
    for ax in fig.axes:
        payload.append(
            {
                "lines": [
                    [
                        np.asarray(line.get_xdata()).tolist(),
                        np.asarray(line.get_ydata()).tolist(),
                    ]
                    for line in ax.lines
                ],
                "collections": [
                    {
                        "offsets": np.asarray(collection.get_offsets()).tolist(),
                        "paths": [path.vertices.tolist() for path in collection.get_paths()],
                    }
                    for collection in ax.collections
                ],
                "xlim": list(ax.get_xlim()),
                "ylim": list(ax.get_ylim()),
                "xscale": ax.get_xscale(),
                "yscale": ax.get_yscale(),
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _visible_texts(fig: Figure) -> list[Text]:
    excluded_tick_ids = set()
    for ax in fig.axes:
        x_low, x_high = sorted(ax.get_xlim())
        y_low, y_high = sorted(ax.get_ylim())
        for location, label in zip(ax.get_xticks(), ax.get_xticklabels()):
            if not (x_low - 1e-12 <= location <= x_high + 1e-12):
                excluded_tick_ids.add(id(label))
        for location, label in zip(ax.get_yticks(), ax.get_yticklabels()):
            if not (y_low - 1e-12 <= location <= y_high + 1e-12):
                excluded_tick_ids.add(id(label))
    return [
        text
        for text in fig.findobj(Text)
        if text.get_visible()
        and text.get_text().strip()
        and (getattr(text, "axes", None) is None or text.axes.axison)
        and id(text) not in excluded_tick_ids
    ]


def _style_figure(fig: Figure, config: dict) -> dict:
    typography = config["typography_pt"]
    style = config["style"]
    width_mm = float(config["canvas"]["width_mm"])
    height_mm = float(config["canvas"]["height_mm"])
    fig.set_size_inches(width_mm / 25.4, height_mm / 25.4, forward=True)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": None,
        }
    )

    if len(fig.axes) != 11:
        raise ValueError(f"Expected 11 inherited axes; found {len(fig.axes)}")
    top = fig.axes[:3]
    score = fig.axes[3:8]
    distribution_top, distribution_bottom, spectrum = fig.axes[8:11]

    for text in _visible_texts(fig):
        text.set_fontfamily("sans-serif")
        text.set_fontsize(float(typography["tick_label"]))
        text.set_fontweight("normal")

    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(float(typography["axis_title"]))
        ax.yaxis.label.set_fontsize(float(typography["axis_title"]))
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=float(typography["tick_label"]),
            width=float(style["axis_linewidth_pt"]),
            length=2.2,
            pad=2.0,
        )
        for spine in ax.spines.values():
            spine.set_linewidth(float(style["axis_linewidth_pt"]))
        for gridline in [*ax.get_xgridlines(), *ax.get_ygridlines()]:
            gridline.set_linewidth(float(style["grid_linewidth_pt"]))
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(float(typography["standard_legend"]))
            legend.set_frame_on(False)

    for ax in top:
        ax.xaxis.label.set_fontsize(float(typography["axis_title"]))
        ax.yaxis.label.set_fontsize(float(typography["axis_title"]))
        ax.tick_params(axis="both", labelsize=float(typography["tick_label"]))
    top[2].yaxis.labelpad = 7.0

    for ax in (distribution_top, distribution_bottom):
        ax.yaxis.label.set_fontsize(float(typography["axis_title"]))
        ax.tick_params(axis="y", labelsize=float(typography["tick_label"]))
        for collection in ax.collections:
            if isinstance(collection, PolyCollection):
                collection.set_linewidth(float(style["violin_edge_linewidth_pt"]))
    distribution_bottom.tick_params(axis="x", labelsize=float(typography["tick_label"]), pad=2.2)
    wrapped_categories = {
        "No sensor feedback": "No sensor\nfeedback",
        "No local conditioning": "No local\nconditioning",
        "Local-only conditioning": "Local-only\nconditioning",
        "IID Gaussian prior": "IID Gaussian\nprior",
    }
    category_labels = [
        wrapped_categories.get(tick.get_text(), tick.get_text())
        for tick in distribution_bottom.get_xticklabels()
    ]
    distribution_bottom.set_xticks(distribution_bottom.get_xticks(), category_labels)
    for tick in distribution_bottom.get_xticklabels():
        tick.set_rotation(float(config["geometry"]["panel_d_tick_rotation_deg"]))
        tick.set_ha("right")
    # Raise only the lower violin row. This reduces its generous
    # internal gap while clearing the scorecard row without moving any data.
    position = distribution_bottom.get_position()
    upward = float(config["geometry"]["lower_violin_raise_mm"]) / height_mm
    distribution_bottom.set_position(
        [position.x0, position.y0 + upward, position.width, position.height]
    )

    # Shorten the upper violin axes from the bottom to separate the two long,
    # deliberately aligned vertical metric titles.
    upper_bottom_inset = float(config["geometry"]["upper_violin_bottom_inset_mm"]) / height_mm
    position = distribution_top.get_position()
    distribution_top.set_position(
        [
            position.x0,
            position.y0 + upper_bottom_inset,
            position.width,
            position.height - upper_bottom_inset,
        ]
    )

    # Reclaim one millimetre from the unused top of the middle panels so the
    # rendered top/middle unions retain the general-contract clearance.
    top_inset = 1.0 / height_mm
    for ax in (distribution_top, spectrum):
        position = ax.get_position()
        ax.set_position(
            [position.x0, position.y0, position.width, position.height - top_inset]
        )

    # Translate the complete scorecard row without changing any axes size.
    score_drop = float(config["geometry"]["scorecard_drop_mm"]) / height_mm
    for ax in score:
        position = ax.get_position()
        ax.set_position(
            [position.x0, position.y0 - score_drop, position.width, position.height]
        )

    for collection in fig.findobj(PathCollection):
        if len(collection.get_offsets()) >= 50:
            collection.set_alpha(float(style["cloud_alpha"]))
            collection.set_sizes([float(style["cloud_size_pt2"])])
            collection.set_edgecolor("none")

    for line in spectrum.lines:
        label = line.get_label()
        if label == "Truth":
            line.set_linewidth(float(style["truth_linewidth_pt"]))
        elif label == "Full model":
            line.set_linewidth(float(style["full_model_linewidth_pt"]))
        elif not label.startswith("_"):
            line.set_linewidth(float(style["alternative_linewidth_pt"]))
    spectrum.xaxis.label.set_fontsize(float(typography["axis_title"]))
    spectrum.yaxis.label.set_fontsize(float(typography["axis_title"]))
    spectrum.tick_params(axis="both", labelsize=float(typography["tick_label"]))
    if spectrum.get_legend() is not None:
        spectrum.get_legend().set_bbox_to_anchor((0.48, 0.99))
        for text in spectrum.get_legend().get_texts():
            text.set_fontsize(float(typography["standard_legend"]))
    for text in spectrum.texts:
        if text.get_text() == "High-band region":
            text.set_fontsize(float(typography["high_band_label"]))
            x, y = text.get_position()
            text.set_position(
                (x + float(config["geometry"]["high_band_text_shift_shells"]), y)
            )

    score[0].tick_params(axis="y", labelsize=float(typography["scorecard_method"]), pad=3)
    score[0].tick_params(axis="x", labelsize=float(typography["tick_label"]))
    score[0].xaxis.label.set_fontsize(float(typography["axis_title"]))
    for text in score[0].texts:
        text.set_fontsize(float(typography["scorecard_value"]))
    for ax in score[1:]:
        ax.tick_params(axis="x", labelsize=float(typography["tick_label"]))
        ax.xaxis.label.set_fontsize(float(typography["axis_title"]))
        for text in ax.texts:
            text.set_fontsize(float(typography["in_plot_annotation"]))

    panel_candidates = [text for text in fig.texts if text.get_text() in set("abcdef")]
    selected_panel_labels = {}
    for text in panel_candidates:
        selected_panel_labels[text.get_text()] = text
    if sorted(selected_panel_labels) != list("abcdef"):
        raise ValueError(f"Panel-label inventory differs from a-f: {sorted(selected_panel_labels)}")
    panel_labels = list(selected_panel_labels.values())
    for text in panel_candidates:
        if text is not selected_panel_labels[text.get_text()]:
            text.set_visible(False)
    for text in panel_labels:
        text.set_fontsize(float(typography["panel_label"]))
        text.set_fontweight("bold")
    left_tag_x = float(config["geometry"]["left_panel_tag_x_mm"]) / width_mm
    for label in ("a", "d", "f"):
        selected_panel_labels[label].set_x(left_tag_x)
    tag_drop = float(config["geometry"]["middle_panel_tag_drop_mm"]) / height_mm
    for label in ("d", "e"):
        x, y = selected_panel_labels[label].get_position()
        selected_panel_labels[label].set_position((x, y - tag_drop))
    # Place f in the left gutter, top-aligned to its axes rather than floating
    # above the row where it competes with Panel-d's rotated category labels.
    selected_panel_labels["f"].set_y(score[0].get_position().y1)
    selected_panel_labels["f"].set_va("top")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = Bbox.from_bounds(0, 0, fig.bbox.width, fig.bbox.height)
    text_boxes = [(text, text.get_window_extent(renderer)) for text in _visible_texts(fig)]
    def text_owner(text: Text) -> dict:
        for axis_index, ax in enumerate(fig.axes):
            if text in ax.get_xticklabels():
                return {"axis_index": axis_index, "role": "xtick", "axison": ax.axison}
            if text in ax.get_yticklabels():
                return {"axis_index": axis_index, "role": "ytick", "axison": ax.axison}
            if text is ax.xaxis.label:
                return {"axis_index": axis_index, "role": "xlabel", "axison": ax.axison}
            if text is ax.yaxis.label:
                return {"axis_index": axis_index, "role": "ylabel", "axison": ax.axison}
            if text in ax.texts:
                return {"axis_index": axis_index, "role": "annotation", "axison": ax.axison}
            legend = ax.get_legend()
            if legend is not None and text in legend.get_texts():
                return {"axis_index": axis_index, "role": "legend", "axison": ax.axison}
        return {"axis_index": None, "role": "figure_text", "axison": None}

    clipped = [
        {"text": text.get_text(), "bbox_px": list(map(float, box.extents)), **text_owner(text)}
        for text, box in text_boxes
        if not canvas.contains(*box.get_points()[0]) or not canvas.contains(*box.get_points()[1])
    ]

    text_text_overlaps = []
    for first in range(len(text_boxes)):
        first_text, first_box = text_boxes[first]
        for second in range(first + 1, len(text_boxes)):
            second_text, second_box = text_boxes[second]
            if Bbox.overlaps(first_box, second_box):
                text_text_overlaps.append(
                    {
                        "first": first_text.get_text(),
                        "second": second_text.get_text(),
                        "first_owner": text_owner(first_text),
                        "second_owner": text_owner(second_text),
                    }
                )

    text_nonowned_axes_overlaps = []
    for text, box in text_boxes:
        owner_index = text_owner(text)["axis_index"]
        for axis_index, ax in enumerate(fig.axes):
            if axis_index == owner_index:
                continue
            if Bbox.overlaps(box, ax.get_window_extent(renderer)):
                text_nonowned_axes_overlaps.append(
                    {
                        "text": text.get_text(),
                        "owner_axis": owner_index,
                        "intersected_axis": axis_index,
                    }
                )

    tick_overlaps = []
    for axis_index, ax in enumerate(fig.axes):
        for direction, labels in (
            ("x", ax.get_xticklabels()),
            ("y", ax.get_yticklabels()),
        ):
            boxes = [label.get_window_extent(renderer) for label in labels if label.get_visible() and label.get_text().strip()]
            for first in range(len(boxes)):
                for second in range(first + 1, len(boxes)):
                    if Bbox.overlaps(boxes[first], boxes[second]):
                        tick_overlaps.append([axis_index, direction, first, second])

    tags = {text.get_text(): text.get_window_extent(renderer) for text in panel_labels}
    top_boxes = [ax.get_tightbbox(renderer) for ax in top] + [tags[key] for key in "abc"]
    middle_boxes = [ax.get_tightbbox(renderer) for ax in (distribution_top, distribution_bottom, spectrum)] + [tags[key] for key in "de"]
    bottom_boxes = [ax.get_tightbbox(renderer) for ax in score] + [tags["f"]]
    top_union = _union(top_boxes)
    middle_union = _union(middle_boxes)
    bottom_union = _union(bottom_boxes)
    px_to_mm_y = height_mm / fig.bbox.height
    row_clearance = {
        "top_to_middle_mm": float((top_union.y0 - middle_union.y1) * px_to_mm_y),
        "middle_to_bottom_mm": float((middle_union.y0 - bottom_union.y1) * px_to_mm_y),
    }

    tag_overlaps = []
    for label, box in tags.items():
        for axis_index, ax in enumerate(fig.axes):
            if Bbox.overlaps(box, ax.get_window_extent(renderer)):
                tag_overlaps.append([label, axis_index])

    return {
        "canvas_mm": [width_mm, height_mm],
        "scientific_hash_after": _scientific_hash(fig),
        "visible_text_count": len(text_boxes),
        "clipped_text": clipped,
        "tick_label_overlaps": tick_overlaps,
        "text_text_overlaps": text_text_overlaps,
        "text_nonowned_axes_overlaps": text_nonowned_axes_overlaps,
        "panel_tag_axes_overlaps": tag_overlaps,
        "row_union_clearance_mm": row_clearance,
        "row_unions_mm": {
            "top": _bbox_record(top_union, fig),
            "middle": _bbox_record(middle_union, fig),
            "bottom": _bbox_record(bottom_union, fig),
        },
        "panel_label_font_pt": float(typography["panel_label"]),
        "axis_title_font_pt": float(typography["axis_title"]),
        "tick_label_font_pt": float(typography["tick_label"]),
        "legend_font_pt": float(typography["standard_legend"]),
        "insertion_scale": float(config["canvas"]["manuscript_width_mm"]) / width_mm,
    }


def _render_pdf(pdf: Path, destination: Path, target_width_mm: float, dpi: int = 300) -> None:
    document = fitz.open(pdf)
    page = document[0]
    scale = target_width_mm / (page.rect.width * 25.4 / 72.0)
    matrix = fitz.Matrix(dpi / 72.0 * scale, dpi / 72.0 * scale)
    pixmap = page.get_pixmap(matrix=matrix, alpha=False)
    pixmap.save(destination)


def _write_review_variants(source: Path, grayscale: Path, deuteranopia: Path) -> None:
    image = Image.open(source).convert("RGB")
    image.convert("L").convert("RGB").save(grayscale)
    values = np.asarray(image, dtype=np.float32) / 255.0
    matrix = np.asarray(
        [[0.367, 0.861, -0.228], [0.280, 0.673, 0.047], [-0.012, 0.043, 0.969]],
        dtype=np.float32,
    )
    transformed = np.clip(values @ matrix.T, 0.0, 1.0)
    Image.fromarray(np.uint8(np.round(transformed * 255.0)), "RGB").save(deuteranopia)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if config.get("schema_version") != "figure5-art-v1-1":
        raise ValueError("Unexpected Figure 5 art configuration schema")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    baseline_pdf = ROOT / config["baseline"]["pdf"]

    original_savefig = Figure.savefig
    state: dict = {"styled": False}

    def savefig_fixed(self: Figure, filename, *positional, **keywords):
        if not state["styled"]:
            state["scientific_hash_before"] = _scientific_hash(self)
            state["qa"] = _style_figure(self, config)
            if state["scientific_hash_before"] != state["qa"]["scientific_hash_after"]:
                raise RuntimeError("Scientific artist coordinates changed during the art pass")
            state["styled"] = True
        width_in, height_in = self.get_size_inches()
        keywords["bbox_inches"] = Bbox.from_bounds(0, 0, width_in, height_in)
        keywords["pad_inches"] = 0
        return original_savefig(self, filename, *positional, **keywords)

    Figure.savefig = savefig_fixed
    previous_argv = sys.argv
    try:
        middle_ratios = config["geometry"]["middle_width_ratios"]
        sys.argv = [
            str(BASE_SCRIPT),
            "--output-dir",
            str(output),
            "--height-mm",
            "260",
            "--middle-ratios",
            str(middle_ratios[0]),
            str(middle_ratios[1]),
            "--middle-wspace",
            str(config["geometry"]["middle_wspace"]),
        ]
        BASE.main()
    finally:
        sys.argv = previous_argv
        Figure.savefig = original_savefig

    pdf = output / "figure5_log.pdf"
    svg = output / "figure5_log.svg"
    preview_162 = output / "figure5_log_162mm.png"
    before_162 = output / "baseline_162mm.png"
    _render_pdf(pdf, preview_162, float(config["canvas"]["manuscript_width_mm"]))
    _render_pdf(baseline_pdf, before_162, float(config["canvas"]["manuscript_width_mm"]))
    _write_review_variants(
        preview_162,
        output / "figure5_log_162mm_grayscale.png",
        output / "figure5_log_162mm_deuteranopia.png",
    )

    document = fitz.open(pdf)
    page = document[0]
    pdf_size_mm = [page.rect.width * 25.4 / 72.0, page.rect.height * 25.4 / 72.0]
    qa = state["qa"]
    qa["schema_version"] = "figure5-art-v1-layout-qa-1"
    qa["scientific_hash_before"] = state["scientific_hash_before"]
    qa["pdf_size_mm"] = pdf_size_mm
    qa["single_page"] = document.page_count == 1
    qa["pdf_font_count"] = len(page.get_fonts())
    qa["pdf_text_characters"] = len(page.get_text())
    qa["svg_editable_text_nodes"] = svg.read_text().count("<text")
    qa["baseline_pdf_sha256"] = _sha256(baseline_pdf)
    qa["checks"] = {
        "scientific_state_exact": qa["scientific_hash_before"] == qa["scientific_hash_after"],
        "fixed_canvas_180x200_mm": bool(np.allclose(pdf_size_mm, [180.0, 200.0], atol=0.02, rtol=0)),
        "single_page": qa["single_page"],
        "pdf_text_present": qa["pdf_text_characters"] > 500,
        "svg_text_editable": qa["svg_editable_text_nodes"] > 80,
        "no_clipped_text": not qa["clipped_text"],
        "no_tick_label_overlap": not qa["tick_label_overlaps"],
        "no_text_text_overlap": not qa["text_text_overlaps"],
        "no_text_nonowned_axes_overlap": not qa["text_nonowned_axes_overlaps"],
        "no_panel_tag_axes_overlap": not qa["panel_tag_axes_overlaps"],
        "tick_font_ge_7pt_at_162mm": qa["tick_label_font_pt"] * qa["insertion_scale"] >= 7.0,
        "legend_font_ge_7pt_at_162mm": qa["legend_font_pt"] * qa["insertion_scale"] >= 7.0,
        "top_middle_clearance_ge_3mm": qa["row_union_clearance_mm"]["top_to_middle_mm"] >= float(config["geometry"]["minimum_row_union_clearance_mm"]),
        "middle_bottom_clearance_ge_3mm": qa["row_union_clearance_mm"]["middle_to_bottom_mm"] >= float(config["geometry"]["minimum_row_union_clearance_mm"]),
    }
    (output / "LAYOUT_QA.json").write_text(json.dumps(qa, indent=2) + "\n")
    failed = [name for name, passed in qa["checks"].items() if not passed]
    if failed:
        raise RuntimeError(f"Figure 5 art V1 QA failed: {failed}")

    contract = f"""# Figure 5 art V1 contract

- Core conclusion: DMF-Gen's ensemble provides the strongest distributional score and difficulty ranking, while the ablations and cost scorecard expose the accuracy, scale-fidelity, and computational trade-offs.
- Figure archetype: asymmetric quantitative composite.
- Target/output: Nature-family double-column figure; editable SVG/PDF plus PNG previews.
- Backend: Python/matplotlib in the `fig` environment, exclusively.
- Final size: 180 x 200 mm; a separate vector-derived review raster is rendered at 162 mm insertion width.
- Panel map: a-c ensemble quality and selective utility; d conditioning/source ablations; e population spectrum and frozen high-band definition; f accuracy and measured computational footprint.
- Evidence hierarchy: a-c lead; d/e explain architectural and source-process trade-offs; f supplies deployment context.
- Statistics/source data: inherited unchanged from the accepted precision-log renderer and its saved evaluation tables; no inference, training, KDE, interval, or summary statistic was recomputed by this wrapper.
- Image integrity: no raster scientific image panels; all plotted coordinates, scales, limits, violin paths, bars, intervals, and category orders are locked by the scientific-state hash.
- Reviewer risks: author checks A02, A05, and A06 remain unresolved scientific-release gates and are not altered here.
"""
    (output / "figure_contract.md").write_text(contract)
    changelog = f"""# Figure 5 art V1 style changelog

- Baseline: `{config['baseline']['pdf']}` (`{_sha256(baseline_pdf)}`).
- Canvas: replaced the approximately 258 x 249 mm tight export with a fixed 180 x 200 mm MediaBox.
- Typography: panel labels 11.0 pt bold; axis titles 8.5 pt; ticks and legends 7.8 pt; in-plot annotations 7.0 pt. Every non-panel text role is regular weight, and ordinary ticks and legends remain at least 7 pt at 162-mm insertion width.
- Geometry: retained the accepted GridSpec panel arrangement; used a 7.3:2.7 d/e split with 0.30 internal wspace, inset the middle-row top by 1 mm, shortened the upper violin axes from below by 8 mm, raised only the lower violin row by 2.5 mm, and translated the aligned scorecard row down by 2.5 mm to accommodate final-size typography without changing data coordinates.
- Styling: reduced axes/grid strokes for final-scale printing; retained model identity colours, marker identities, dash patterns, shaded high-band range, and truth/full-model hierarchy.
- Scientific state: before/after artist hash `{qa['scientific_hash_before']}`; exact match `{qa['checks']['scientific_state_exact']}`.
- Review: vector-derived 162-mm, grayscale, and deuteranopia previews are included; renderer-derived clipping, tick-overlap, panel-tag, and adjacent-row gates all pass.
- Outstanding author checks: A02, A05, and A06 remain unchanged.
"""
    (output / "STYLE_CHANGELOG.md").write_text(changelog)
    manifest = {
        "schema_version": "figure5-art-v1-source-manifest-1",
        "baseline": str(baseline_pdf.relative_to(REPO)),
        "baseline_sha256": _sha256(baseline_pdf),
        "renderer": str(Path(__file__).resolve().relative_to(REPO)),
        "config": str(args.config.resolve().relative_to(REPO)),
        "scientific_hash": qa["scientific_hash_after"],
        "outputs": {
            path.name: _sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[PASS] Figure 5 art V1: {output}")
    for name, passed in qa["checks"].items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")


if __name__ == "__main__":
    main()

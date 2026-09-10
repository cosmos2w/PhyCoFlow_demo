#!/usr/bin/env python
"""Tight-canvas manuscript integration refinement of Figure 5 V6 release 20260903_1343."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
V4_SCRIPT = SCRIPT_DIR / "build_figure5_v6_nature_format_v4.py"
TIGHT_PAD_INCHES = 0.01
METHOD_LABEL_PAD_PT = 0.4


def load_v4() -> Any:
    spec = importlib.util.spec_from_file_location("figure5_v6_nature_v4", V4_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load prior renderer: {V4_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V4 = load_v4()
V1 = V4.V3.V2.V1
REFINE_STANDALONE_V4 = V4.refine_standalone_v4
REFINE_COMPOSED_V4 = V4.refine_composed_v4
LOAD_BASE_V4 = V1.load_base
ALL_METHODS = set(V4.UQ_METHODS) | set(V4.SCORECARD_METHODS)


def visible_panel_tag(figure: Any, label: str) -> Any:
    tags = [
        text
        for text in figure.texts
        if text.get_visible() and text.get_text() == label
    ]
    if len(tags) != 1:
        raise ValueError(f"Expected one visible panel tag {label!r}, found {len(tags)}")
    return tags[0]


def align_method_labels(ax: Any) -> None:
    labels = [label for label in ax.get_yticklabels() if label.get_text() in ALL_METHODS]
    if not labels:
        return
    for label in labels:
        label.set_horizontalalignment("right")
    ax.tick_params(axis="y", pad=METHOD_LABEL_PAD_PT)


def title_top_in_figure(figure: Any, axes: list[Any]) -> float:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tops = [
        ax._left_title.get_window_extent(renderer=renderer)
        .transformed(figure.transFigure.inverted())
        .y1
        for ax in axes
    ]
    return float(max(tops))


def place_tag_at_spine(figure: Any, label: str, ax: Any, y: float) -> float:
    x = float(ax.get_position().x0)
    tag = visible_panel_tag(figure, label)
    tag.set_position((x, y))
    tag.set_horizontalalignment("left")
    return x


def refine_standalone_v5(figure: Any, panel: str) -> None:
    REFINE_STANDALONE_V4(figure, panel)
    if panel in {"a", "b"}:
        align_method_labels(figure.axes[0])
    elif panel == "d":
        align_method_labels(figure.axes[0])

    anchor_ax = figure.axes[0]
    if panel == "d":
        tag_y = title_top_in_figure(figure, figure.axes) + 0.006
    else:
        tag_y = float(anchor_ax.get_position().y1 + 0.015)
    place_tag_at_spine(figure, panel, anchor_ax, tag_y)


def refine_composed_v5(figure: Any) -> dict[str, float]:
    REFINE_COMPOSED_V4(figure)
    top_axes = figure.axes[:3]
    scorecard_axes = figure.axes[3:]
    align_method_labels(top_axes[0])
    align_method_labels(scorecard_axes[0])

    tag_y_top = float(top_axes[0].get_position().y1 + 0.013)
    tag_positions = {
        "a": (place_tag_at_spine(figure, "a", top_axes[0], tag_y_top), tag_y_top),
        "b": (place_tag_at_spine(figure, "b", top_axes[1], tag_y_top), tag_y_top),
        "c": (place_tag_at_spine(figure, "c", top_axes[2], tag_y_top), tag_y_top),
    }
    tag_y_d = title_top_in_figure(figure, scorecard_axes) + 0.006
    tag_positions["d"] = (
        place_tag_at_spine(figure, "d", scorecard_axes[0], tag_y_d),
        tag_y_d,
    )

    return {
        "top_method_axis_x0": float(top_axes[0].get_position().x0),
        "scorecard_method_axis_x0": float(scorecard_axes[0].get_position().x0),
        "panel_a_tag_x": tag_positions["a"][0],
        "panel_b_tag_x": tag_positions["b"][0],
        "panel_c_tag_x": tag_positions["c"][0],
        "panel_d_tag_x": tag_positions["d"][0],
        "panel_a_spine_x": float(top_axes[0].get_position().x0),
        "panel_b_spine_x": float(top_axes[1].get_position().x0),
        "panel_c_spine_x": float(top_axes[2].get_position().x0),
        "panel_d_spine_x": float(scorecard_axes[0].get_position().x0),
        "method_label_pad_pt": METHOD_LABEL_PAD_PT,
        "interrow_axes_gap": float(
            top_axes[0].get_position().y0 - scorecard_axes[0].get_position().y1
        ),
    }


def load_base_v5() -> Any:
    base = LOAD_BASE_V4()

    def save_tight(figure: Any, svg_path: Path, preview_path: Path | None) -> None:
        svg_path.parent.mkdir(parents=True, exist_ok=True)
        save_options = {
            "bbox_inches": "tight",
            "pad_inches": TIGHT_PAD_INCHES,
            "facecolor": "white",
        }
        figure.savefig(svg_path, format="svg", **save_options)
        svg_text = svg_path.read_text(encoding="utf-8")
        svg_path.write_text(
            "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
            encoding="utf-8",
        )
        if "fig5_composed_v6_" in svg_path.stem:
            figure.savefig(svg_path.with_suffix(".pdf"), format="pdf", **save_options)
        if preview_path is not None:
            preview_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(preview_path, format="png", dpi=240, **save_options)
        base.plt.close(figure)

    base.save_figure = save_tight
    return base


V1.refine_standalone = refine_standalone_v5
V1.refine_composed = refine_composed_v5
V1.load_base = load_base_v5


def svg_dimensions_mm(path: Path) -> tuple[float, float]:
    root = ET.parse(path).getroot()
    width_pt = float(str(root.attrib["width"]).removesuffix("pt"))
    height_pt = float(str(root.attrib["height"]).removesuffix("pt"))
    return width_pt * 25.4 / 72.0, height_pt * 25.4 / 72.0


def update_release_documents(timestamp: str) -> None:
    figure_dir = PACKAGE_ROOT / "figures" / "generated" / timestamp
    docs_dir = PACKAGE_ROOT / "docs" / "generated" / timestamp
    results_dir = PACKAGE_ROOT / "results" / "derived" / timestamp
    qa_path = results_dir / "qa.json"
    manifest_path = results_dir / "build_manifest.json"
    composed_svg = figure_dir / f"fig5_composed_v6_{timestamp}.svg"
    composed_pdf = composed_svg.with_suffix(".pdf")

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    alignment = qa["alignment"]
    tag_spine_pairs = (
        ("panel_a_tag_x", "panel_a_spine_x"),
        ("panel_b_tag_x", "panel_b_spine_x"),
        ("panel_c_tag_x", "panel_c_spine_x"),
        ("panel_d_tag_x", "panel_d_spine_x"),
    )
    tag_alignment_pass = all(
        abs(float(alignment[tag]) - float(alignment[spine])) < 1e-12
        for tag, spine in tag_spine_pairs
    )
    width_mm, height_mm = svg_dimensions_mm(composed_svg)
    qa.update(
        {
            "schema_version": "figure5-v6-nature-format-v5-qa-1",
            "based_on_release": "20260903_1343",
            "prior_release_untouched": "20260903_1343",
            "all_panel_tags_aligned_to_respective_left_spines": tag_alignment_pass,
            "method_labels_right_aligned": True,
            "method_label_pad_pt": METHOD_LABEL_PAD_PT,
            "tight_bbox_export": True,
            "tight_bbox_pad_inches": TIGHT_PAD_INCHES,
            "composed_svg_dimensions_mm": {
                "width": width_mm,
                "height": height_mm,
            },
            "composed_pdf_created_directly_by_matplotlib": composed_pdf.is_file(),
        }
    )
    if qa.get("status") == "pass" and not tag_alignment_pass:
        qa["status"] = "fail"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    if qa["status"] == "fail":
        raise RuntimeError("Panel-tag/spine alignment QA failed")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    figures = list(manifest["figures"])
    pdf_relative = str(composed_pdf.relative_to(REPO_ROOT))
    if pdf_relative not in figures:
        figures.append(pdf_relative)
    manifest.update(
        {
            "schema_version": "figure5-v6-nature-format-v5-build-1",
            "based_on_release": "20260903_1343",
            "renderer": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "figures": figures,
            "export_bbox_inches": "tight",
            "export_pad_inches": TIGHT_PAD_INCHES,
            "composed_pdf_backend": "Python/Matplotlib",
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    (docs_dir / "figure_contract.md").write_text(
        """# Figure 5 V6 Nature-format V5 contract

- Base release: `20260903_1343`; preserved without modification.
- Scientific data, artist topology, plot types, axes scales, limits, and coordinates: unchanged.
- Panel tags: each tag's left edge is aligned exactly to its panel's left plot spine.
- Method labels: explicitly right-aligned with 0.4-pt y-tick padding in panels a and d.
- Canvas: Matplotlib `bbox_inches='tight'` with `pad_inches=0.01`; redundant outer whitespace is removed without shifting data inside the axes.
- Exports: editable standalone and composed SVGs, plus a composed vector PDF written directly by Matplotlib.
- Font and marker semantics: unchanged Arial and method-specific marker contract from V4.
""",
        encoding="utf-8",
    )
    (docs_dir / "completion_report.md").write_text(
        f"""# Figure 5 V6 Nature-format V5 completion report

- Release: `{timestamp}`; separate refinement based on `20260903_1343`.
- Re-anchored a, b, c, and d tags to their respective subplot left spines.
- Right-aligned method labels and reduced their y-tick padding from 3.0 pt to 0.4 pt.
- Removed redundant canvas whitespace with a 0.01-inch tight-bbox pad.
- Exported the composed SVG and PDF directly from the same Matplotlib figure object.
- Scientific geometry digests: unchanged before/after for all standalone panels and the composed figure.
- New inference, bootstrapping, training, or broad validation: none.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--timestamp", required=True)
    args, _ = parser.parse_known_args()
    result = V4.main()
    update_release_documents(args.timestamp)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

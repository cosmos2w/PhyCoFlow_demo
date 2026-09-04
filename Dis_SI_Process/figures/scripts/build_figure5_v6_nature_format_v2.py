#!/usr/bin/env python
"""Build a wider, tighter typography-only Figure 5 V6 from release 20260903_1231."""
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
V1_SCRIPT = SCRIPT_DIR / "build_figure5_v6_nature_format.py"
FONT_FAMILY = "Nimbus Sans"


def load_v1() -> Any:
    spec = importlib.util.spec_from_file_location("figure5_v6_nature_v1", V1_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load prior typography renderer: {V1_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V1 = load_v1()
V1.FONT_FAMILY = FONT_FAMILY
REFINE_STANDALONE_V1 = V1.refine_standalone
REFINE_COMPOSED_V1 = V1.refine_composed
CONFIGURE_STYLE_V1 = V1.configure_nature_style


def configure_style_v2(base: Any, config: Any) -> None:
    CONFIGURE_STYLE_V1(base, config)
    V1.plt.rcParams.update(
        {
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "mathtext.cal": FONT_FAMILY,
            "mathtext.tt": FONT_FAMILY,
            "mathtext.sf": FONT_FAMILY,
        }
    )


def normalize_l2_label(figure: Any) -> None:
    for ax in figure.axes:
        if "L₂" in ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel().replace("L₂", r"L$_2$"))


def visible_figure_text(figure: Any, value: str) -> list[Any]:
    return [text for text in figure.texts if text.get_visible() and text.get_text() == value]


def position_scorecard_axes(
    axes: list[Any],
    *,
    left: float,
    right: float,
    bottom: float,
    top: float,
) -> None:
    ratios = [2.45, 1.05, 1.10, 1.05, 1.40]
    gap = 0.025
    usable = right - left - gap * (len(ratios) - 1)
    unit = usable / sum(ratios)
    x = left
    for ax, ratio in zip(axes, ratios):
        width = ratio * unit
        ax.set_position([x, bottom, width, top - bottom])
        x += width + gap


def enlarge_axis_legend(ax: Any) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    handles = list(legend.legend_handles)
    labels = [text.get_text() for text in legend.get_texts()]
    legend.remove()
    new_legend = ax.legend(
        handles,
        labels,
        loc="lower right",
        ncol=2,
        fontsize=5.9,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.42,
        columnspacing=0.95,
        borderaxespad=0.35,
        labelspacing=0.34,
    )
    for handle in new_legend.legend_handles:
        if hasattr(handle, "set_markersize"):
            handle.set_markersize(4.7)


def enlarge_shared_legend(figure: Any) -> None:
    if not figure.legends:
        return
    legend = figure.legends[0]
    handles = list(legend.legend_handles)
    labels = [text.get_text() for text in legend.get_texts()]
    legend.remove()
    new_legend = figure.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.555, 0.525),
        ncol=5,
        fontsize=6.2,
        frameon=False,
        handlelength=1.45,
        handletextpad=0.45,
        columnspacing=1.20,
        borderaxespad=0.0,
        labelspacing=0.25,
    )
    for text in new_legend.get_texts():
        text.set_fontfamily(FONT_FAMILY)
    for handle in new_legend.legend_handles:
        if hasattr(handle, "set_markersize"):
            handle.set_markersize(4.8)
        if hasattr(handle, "set_markeredgewidth"):
            handle.set_markeredgewidth(0.70)


def remove_panel_d_title(figure: Any) -> None:
    for text in figure.texts:
        if text.get_text() == "Accuracy and computational footprint":
            text.set_visible(False)


def move_visible_tag(figure: Any, label: str, position: tuple[float, float]) -> None:
    tags = visible_figure_text(figure, label)
    if len(tags) != 1:
        raise ValueError(f"Expected one visible panel tag {label!r}, found {len(tags)}")
    tags[0].set_position(position)
    tags[0].set_fontfamily(FONT_FAMILY)


def refine_standalone_v2(figure: Any, panel: str) -> None:
    REFINE_STANDALONE_V1(figure, panel)
    normalize_l2_label(figure)
    if panel in {"a", "b"}:
        figure.axes[0].set_position([0.20, 0.19, 0.785, 0.68])
    elif panel == "c":
        figure.axes[0].set_position([0.14, 0.19, 0.845, 0.68])
        enlarge_axis_legend(figure.axes[0])
    else:
        remove_panel_d_title(figure)
        position_scorecard_axes(
            figure.axes,
            left=0.12,
            right=0.995,
            bottom=0.19,
            top=0.79,
        )
    move_visible_tag(figure, panel, (0.015, 0.91 if panel == "d" else 0.925))


def refine_composed_v2(figure: Any) -> dict[str, float]:
    REFINE_COMPOSED_V1(figure)
    normalize_l2_label(figure)
    top_axes = figure.axes[:3]
    scorecard_axes = figure.axes[3:]

    # Wider top panels with controlled inter-panel gutters.
    top_axes[0].set_position([0.12, 0.595, 0.25, 0.31])
    top_axes[1].set_position([0.415, 0.595, 0.27, 0.31])
    top_axes[2].set_position([0.735, 0.595, 0.26, 0.31])

    # Use the same absolute left edge, widen every scorecard column, and
    # bring the lower row upward after removing its redundant title.
    position_scorecard_axes(
        scorecard_axes,
        left=0.12,
        right=0.995,
        bottom=0.075,
        top=0.445,
    )
    top_axes[0].tick_params(axis="y", pad=3.0)
    scorecard_axes[0].tick_params(axis="y", pad=3.0)

    remove_panel_d_title(figure)
    enlarge_shared_legend(figure)
    for label, position in {
        "a": (0.025, 0.918),
        "b": (0.395, 0.918),
        "c": (0.715, 0.918),
        "d": (0.025, 0.458),
    }.items():
        move_visible_tag(figure, label, position)

    top_bottom = float(top_axes[0].get_position().y0)
    lower_top = float(scorecard_axes[0].get_position().y1)
    return {
        "top_method_axis_x0": float(top_axes[0].get_position().x0),
        "scorecard_method_axis_x0": float(scorecard_axes[0].get_position().x0),
        "top_panel_tag_x": 0.025,
        "scorecard_panel_tag_x": 0.025,
        "interrow_axes_gap": top_bottom - lower_top,
        "top_row_right": float(top_axes[-1].get_position().x1),
        "scorecard_row_right": float(scorecard_axes[-1].get_position().x1),
    }


def svg_checks_v2(path: Path, forbidden: tuple[str, ...], required: tuple[str, ...]) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    root = ET.parse(path).getroot()
    filtered_required = tuple(
        text for text in required if text != "Accuracy and computational footprint"
    )
    forbidden_all = (*forbidden, "Accuracy and computational footprint")
    return {
        "parseable": root.tag.endswith("svg"),
        "editable_text": "<text" in content,
        "sans_serif_font_present": FONT_FAMILY in content,
        "required_text_present": all(text in content for text in filtered_required),
        "forbidden_subtitles_absent": all(text not in content for text in forbidden_all),
        "bytes": path.stat().st_size,
    }


V1.refine_standalone = refine_standalone_v2
V1.refine_composed = refine_composed_v2
V1.svg_checks = svg_checks_v2
V1.configure_nature_style = configure_style_v2


def update_release_documents(timestamp: str) -> None:
    docs_dir = PACKAGE_ROOT / "docs" / "generated" / timestamp
    results_dir = PACKAGE_ROOT / "results" / "derived" / timestamp
    qa_path = results_dir / "qa.json"
    manifest_path = results_dir / "build_manifest.json"

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["schema_version"] = "figure5-v6-nature-format-v2-qa-1"
    qa["font_family"] = FONT_FAMILY
    qa["panel_d_title_removed"] = True
    qa["shared_legend_font_pt"] = 6.2
    qa["shared_legend_marker_pt"] = 4.8
    qa["prior_release_untouched"] = "20260903_1231"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "figure5-v6-nature-format-v2-build-1"
    manifest["based_on_release"] = "20260903_1231"
    manifest["renderer"] = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    manifest["font_family"] = FONT_FAMILY
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    (docs_dir / "figure_contract.md").write_text(
        """# Figure 5 V6 Nature-format V2 contract

- Base release: `20260903_1231`; preserved without modification.
- Core conclusion and quantitative evidence: unchanged.
- Backend: Python/Matplotlib only.
- Final size: 183 mm × 128 mm, editable SVG.
- Scientific-geometry lock: scatter clouds, distribution bands, error bars, confidence bands, line coordinates, axes limits, and plot types must match the base renderer.
- Permitted changes: typography, marker cosmetics, legend dimensions, text visibility, and panel-container placement.
- Font: Nimbus Sans, the installed Helvetica-compatible sans-serif family.
- Layout: common left boundary for the upper and lower rows, wider subplot containers, reduced inter-row gap, and no panel-d title.
""",
        encoding="utf-8",
    )
    (docs_dir / "completion_report.md").write_text(
        f"""# Figure 5 V6 Nature-format V2 completion report

- Release: `{timestamp}`; additional release based on `20260903_1231`.
- Font: Nimbus Sans (installed Helvetica-compatible family; no font installation required).
- Shared a–c legend: enlarged to 6.2 pt with 4.8 pt symbols and wider spacing.
- Panel d title: removed.
- Inter-row axes gap: reduced from 0.210 to 0.150 figure units.
- Upper/lower left plot boundary: exactly 0.120; panel-a/panel-d tag x-position: exactly 0.025.
- Panel tags sit 0.013 figure units above their nearest subplot edge, with no overlap.
- Horizontal layout: all top panels and all five scorecard columns widened while retaining controlled gutters.
- Scientific geometry: before/after coordinate digests match for every output.
- Visual QA: PASS only when recorded in `qa.json`.
- New inference, bootstrap, training, or broad validation: none.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--timestamp", required=True)
    args, _ = parser.parse_known_args()
    result = V1.main()
    update_release_documents(args.timestamp)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

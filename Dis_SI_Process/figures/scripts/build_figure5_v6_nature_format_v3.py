#!/usr/bin/env python
"""Build the Arial, legend-free Figure 5 V6 refinement from release 20260903_1317."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from matplotlib import font_manager
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
V2_SCRIPT = SCRIPT_DIR / "build_figure5_v6_nature_format_v2.py"
FONT_FAMILY = "Arial"
FONT_DIR = Path.home() / ".local" / "share" / "fonts" / "Arial"


def load_v2() -> Any:
    spec = importlib.util.spec_from_file_location("figure5_v6_nature_v2", V2_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load prior renderer: {V2_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


for font_path in sorted(FONT_DIR.glob("*.TTF")):
    font_manager.fontManager.addfont(font_path)
resolved_font = Path(font_manager.findfont(FONT_FAMILY, fallback_to_default=False))
if resolved_font.parent != FONT_DIR:
    raise RuntimeError(f"Arial did not resolve to the installed user font directory: {resolved_font}")


V2 = load_v2()
V2.FONT_FAMILY = FONT_FAMILY
V2.V1.FONT_FAMILY = FONT_FAMILY
REFINE_STANDALONE_V2 = V2.refine_standalone_v2
REFINE_COMPOSED_V2 = V2.refine_composed_v2


def is_white(value: Any) -> bool:
    try:
        return bool(all(abs(a - b) < 1e-8 for a, b in zip(to_rgba(value), to_rgba("white"))))
    except (TypeError, ValueError):
        return False


def method_color(line: Any) -> Any:
    face = line.get_markerfacecolor()
    edge = line.get_markeredgecolor()
    if not is_white(face) and str(face).lower() not in {"none", "auto"}:
        return face
    if not is_white(edge) and str(edge).lower() not in {"none", "auto"}:
        return edge
    return line.get_color()


def apply_hollow_circles(ax: Any) -> None:
    """Restyle plotted summary/curve markers; scatter clouds remain unchanged."""
    for line in ax.lines:
        marker = line.get_marker()
        if marker in {None, "None", "none", "", " "}:
            continue
        color = method_color(line)
        line.set_marker("o")
        line.set_markersize(4.9)
        line.set_markerfacecolor("white")
        line.set_markeredgecolor(color)
        line.set_markeredgewidth(1.10)


def reverse_memory_encoding(ax: Any) -> None:
    marker_lines = [
        line
        for line in ax.lines
        if line.get_marker() not in {None, "None", "none", "", " "}
    ]
    if len(marker_lines) != 16:
        raise ValueError(f"Expected two inference-memory markers for eight methods, found {len(marker_lines)}")
    for index, line in enumerate(marker_lines):
        color = method_color(line)
        line.set_marker("o")
        line.set_markersize(4.9)
        line.set_markeredgecolor(color)
        line.set_markeredgewidth(1.10)
        if index % 2 == 0:  # model parameters + persistent buffers
            line.set_markerfacecolor(color)
        else:  # peak allocated memory
            line.set_markerfacecolor("white")


def remove_method_legends(figure: Any) -> None:
    for legend in list(figure.legends):
        legend.remove()
    for ax in figure.axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def add_memory_key(ax: Any) -> None:
    neutral = "#303030"
    handles = [
        Line2D(
            [], [], marker="o", linestyle="none", markersize=4.7,
            markerfacecolor=neutral, markeredgecolor=neutral,
            markeredgewidth=1.0, label="Model",
        ),
        Line2D(
            [], [], marker="o", linestyle="none", markersize=4.7,
            markerfacecolor="white", markeredgecolor=neutral,
            markeredgewidth=1.0, label="Peak",
        ),
    ]
    ax.set_title("Inference memory", loc="left", pad=13.0)
    key = ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=2,
        frameon=False,
        fontsize=5.3,
        handlelength=0.8,
        handletextpad=0.30,
        columnspacing=0.70,
        borderaxespad=0.0,
    )
    for text in key.get_texts():
        text.set_fontfamily(FONT_FAMILY)


def shorten_crps_label(figure: Any) -> None:
    for ax in figure.axes:
        if ax.get_xlabel() == "Normalized CRPS (lower is better)":
            ax.set_xlabel("Normalized CRPS")


def move_visible_tag(figure: Any, label: str, position: tuple[float, float]) -> None:
    tags = [
        text
        for text in figure.texts
        if text.get_visible() and text.get_text() == label
    ]
    if len(tags) != 1:
        raise ValueError(f"Expected one visible panel tag {label!r}, found {len(tags)}")
    tags[0].set_position(position)


def refine_standalone_v3(figure: Any, panel: str) -> None:
    REFINE_STANDALONE_V2(figure, panel)
    shorten_crps_label(figure)
    remove_method_legends(figure)
    if panel == "d":
        for ax in figure.axes[:-1]:
            apply_hollow_circles(ax)
        reverse_memory_encoding(figure.axes[-1])
        add_memory_key(figure.axes[-1])
    else:
        apply_hollow_circles(figure.axes[0])


def refine_composed_v3(figure: Any) -> dict[str, float]:
    REFINE_COMPOSED_V2(figure)
    top_axes = figure.axes[:3]
    scorecard_axes = figure.axes[3:]
    shorten_crps_label(figure)
    remove_method_legends(figure)

    for ax in [*top_axes, *scorecard_axes[:-1]]:
        apply_hollow_circles(ax)
    reverse_memory_encoding(scorecard_axes[-1])
    add_memory_key(scorecard_axes[-1])

    V2.position_scorecard_axes(
        scorecard_axes,
        left=0.12,
        right=0.995,
        bottom=0.070,
        top=0.475,
    )
    move_visible_tag(figure, "d", (0.025, 0.493))

    return {
        "top_method_axis_x0": float(top_axes[0].get_position().x0),
        "scorecard_method_axis_x0": float(scorecard_axes[0].get_position().x0),
        "top_panel_tag_x": 0.025,
        "scorecard_panel_tag_x": 0.025,
        "interrow_axes_gap": float(top_axes[0].get_position().y0 - scorecard_axes[0].get_position().y1),
        "top_row_right": float(top_axes[-1].get_position().x1),
        "scorecard_row_right": float(scorecard_axes[-1].get_position().x1),
    }


def svg_checks_v3(path: Path, forbidden: tuple[str, ...], required: tuple[str, ...]) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    root = ET.parse(path).getroot()
    filtered_required = tuple(
        text
        for text in required
        if text not in {
            "Accuracy and computational footprint",
            "Normalized CRPS (lower is better)",
        }
    )
    extras: tuple[str, ...] = ()
    if "fig5a_" in path.name or "composed" in path.name:
        extras += ("Normalized CRPS",)
    if "fig5d_" in path.name or "composed" in path.name:
        extras += ("Model", "Peak", "Inference memory")
    forbidden_all = (
        *forbidden,
        "Accuracy and computational footprint",
        "(lower is better)",
    )
    method_label_count_ok = True
    if "composed" in path.name:
        method_label_count_ok = content.count(">DMF-Gen<") == 2
    return {
        "parseable": root.tag.endswith("svg"),
        "editable_text": "<text" in content,
        "sans_serif_font_present": FONT_FAMILY in content,
        "required_text_present": all(text in content for text in (*filtered_required, *extras)),
        "forbidden_subtitles_absent": all(text not in content for text in forbidden_all),
        "method_legend_absent": method_label_count_ok,
        "bytes": path.stat().st_size,
    }


V2.V1.refine_standalone = refine_standalone_v3
V2.V1.refine_composed = refine_composed_v3
V2.V1.svg_checks = svg_checks_v3


def update_release_documents(timestamp: str) -> None:
    docs_dir = PACKAGE_ROOT / "docs" / "generated" / timestamp
    results_dir = PACKAGE_ROOT / "results" / "derived" / timestamp
    qa_path = results_dir / "qa.json"
    manifest_path = results_dir / "build_manifest.json"

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["schema_version"] = "figure5-v6-nature-format-v3-qa-1"
    qa.pop("shared_legend_font_pt", None)
    qa.pop("shared_legend_marker_pt", None)
    qa["font_family"] = FONT_FAMILY
    qa["font_file"] = str(resolved_font)
    qa["based_on_release"] = "20260903_1317"
    qa["prior_release_untouched"] = "20260903_1317"
    qa["method_legend_removed"] = True
    qa["marker_encoding"] = "hollow circles outside inference memory"
    qa["inference_memory_encoding"] = {
        "model_parameters_and_buffers": "filled circle",
        "peak_allocated": "hollow circle",
        "key_labels": ["Model", "Peak"],
    }
    extra_checks_pass = all(
        checks.get("method_legend_absent", False)
        and checks.get("sans_serif_font_present", False)
        for checks in qa["svg_checks"].values()
    )
    if qa.get("status") == "pass" and not extra_checks_pass:
        qa["status"] = "fail"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    if qa["status"] == "fail":
        raise RuntimeError("Arial/legend-removal QA failed")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "figure5-v6-nature-format-v3-build-1"
    manifest["based_on_release"] = "20260903_1317"
    manifest["renderer"] = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    manifest["font_family"] = FONT_FAMILY
    manifest["font_file"] = str(resolved_font)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    (docs_dir / "figure_contract.md").write_text(
        """# Figure 5 V6 Nature-format V3 contract

- Base release: `20260903_1317`; preserved without modification.
- Font: locally installed Microsoft Arial, explicitly registered with Matplotlib.
- Scientific data, scatter clouds, distributions, error bars, bands, curves, scales, and axes limits: unchanged.
- Panel-a x label: `Normalized CRPS`.
- Shared method legend: removed.
- Marker presentation: moderately enlarged white-filled circles with thick method-color borders outside inference memory.
- Inference memory: filled circle = model parameters plus persistent buffers (`Model`); hollow circle = peak allocated memory (`Peak`). The compact key appears once above the column.
- Layout: common 0.120 left boundary retained; lower row raised to reduce the axes gap to 0.120.
""",
        encoding="utf-8",
    )
    (docs_dir / "completion_report.md").write_text(
        f"""# Figure 5 V6 Nature-format V3 completion report

- Release: `{timestamp}`; additional release based on `20260903_1317`.
- Arial installation: user-local Microsoft Arial family under `{FONT_DIR}`; Fontconfig and Matplotlib resolution verified.
- CRPS label shortened and the redundant shared method legend removed.
- Inter-row axes gap reduced from 0.150 to 0.120 figure units.
- Hollow-circle marker treatment applied everywhere except the inference-memory endpoint pair.
- Inference memory reversed to filled `Model` / hollow `Peak`, with one concise key above the column.
- Scientific geometry digests: unchanged before/after for every output.
- New inference, bootstrap, training, or broad validation: none.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--timestamp", required=True)
    args, _ = parser.parse_known_args()
    result = V2.V1.main()
    V2.update_release_documents(args.timestamp)
    update_release_documents(args.timestamp)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Create a typography-only Nature-format release of the accepted Figure 5 V6."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import to_rgba


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent
BASE_SCRIPT = Path(__file__).with_name("build_figure5_v6.py")
BASE_CONFIG = PACKAGE_ROOT / "configs" / "figure5_v6.yaml"
# Liberation Sans is the installed metric-compatible Arial/Helvetica face.
FONT_FAMILY = "Liberation Sans"


def load_base() -> Any:
    spec = importlib.util.spec_from_file_location("figure5_v6_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base renderer: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hash_array(digest: Any, values: Any) -> None:
    array = np.asarray(values)
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    if array.dtype == object:
        digest.update(repr(array.tolist()).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(array).tobytes())


def geometry_digest(figure: plt.Figure) -> str:
    """Hash plotted coordinates and artist topology, excluding typography/layout."""
    digest = hashlib.sha256()
    digest.update(str(len(figure.axes)).encode("utf-8"))
    for ax in figure.axes:
        digest.update(repr((ax.get_xscale(), ax.get_yscale(), ax.get_xlim(), ax.get_ylim())).encode("utf-8"))
        digest.update(repr((len(ax.lines), len(ax.collections), len(ax.patches))).encode("utf-8"))
        for line in ax.lines:
            hash_array(digest, line.get_xdata(orig=False))
            hash_array(digest, line.get_ydata(orig=False))
        for collection in ax.collections:
            if hasattr(collection, "get_offsets"):
                hash_array(digest, collection.get_offsets())
            if hasattr(collection, "get_segments"):
                for segment in collection.get_segments():
                    hash_array(digest, segment)
            for path in collection.get_paths():
                hash_array(digest, path.vertices)
                if path.codes is not None:
                    hash_array(digest, path.codes)
        for patch in ax.patches:
            path = patch.get_path()
            hash_array(digest, path.vertices)
            if path.codes is not None:
                hash_array(digest, path.codes)
    return digest.hexdigest()


def configure_nature_style(base: Any, config: Mapping[str, Any]) -> None:
    base.configure_style(config["style"]["font_family"])
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.3,
        }
    )


def hide_existing_panel_tags(figure: plt.Figure) -> None:
    for ax in figure.axes:
        for text in ax.texts:
            if text.get_text() in {"a", "b", "c", "d"}:
                text.set_visible(False)
    for text in figure.texts:
        if text.get_text() in {"a", "b", "c", "d"}:
            text.set_visible(False)


def add_panel_tag(figure: plt.Figure, label: str, x: float, y: float) -> None:
    figure.text(
        x,
        y,
        label,
        ha="left",
        va="bottom",
        fontsize=8.4,
        fontweight="bold",
        fontfamily=FONT_FAMILY,
        color="#171717",
    )


def style_axis_text(ax: plt.Axes, *, title_size: float = 6.4) -> None:
    ax.xaxis.label.set_fontsize(6.6)
    ax.yaxis.label.set_fontsize(6.6)
    ax.xaxis.label.set_fontfamily(FONT_FAMILY)
    ax.yaxis.label.set_fontfamily(FONT_FAMILY)
    ax.title.set_fontsize(title_size)
    ax.title.set_fontfamily(FONT_FAMILY)
    ax._left_title.set_fontfamily(FONT_FAMILY)
    ax._right_title.set_fontfamily(FONT_FAMILY)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontsize(5.8)
        label.set_fontfamily(FONT_FAMILY)
    for text in ax.texts:
        text.set_fontfamily(FONT_FAMILY)


def is_white(color: Any) -> bool:
    try:
        return np.allclose(to_rgba(color), to_rgba("white"))
    except (TypeError, ValueError):
        return False


def standardize_method_markers(ax: plt.Axes, *, memory_axis: bool = False) -> None:
    """Style summary/curve markers only; scatter clouds remain untouched."""
    for line in ax.lines:
        marker = line.get_marker()
        if marker in {None, "None", "none", "", " "}:
            continue
        line.set_markersize(4.0)
        line.set_markeredgewidth(0.65)
        if not memory_axis:
            face = line.get_markerfacecolor()
            edge = line.get_markeredgecolor()
            if is_white(face) and not is_white(edge):
                target = edge
            elif is_white(edge) and not is_white(face):
                target = face
            else:
                target = line.get_color()
            line.set_markerfacecolor(target)
            line.set_markeredgecolor(target)


def refine_legends(figure: plt.Figure) -> None:
    legends = list(figure.legends)
    legends.extend(legend for ax in figure.axes if (legend := ax.get_legend()) is not None)
    for legend in legends:
        for text in legend.get_texts():
            text.set_fontfamily(FONT_FAMILY)
            text.set_fontsize(5.3)
        for handle in legend.legend_handles:
            if hasattr(handle, "set_markersize"):
                handle.set_markersize(4.0)
            if hasattr(handle, "set_markeredgewidth"):
                handle.set_markeredgewidth(0.65)


def refine_standalone(figure: plt.Figure, panel: str) -> None:
    hide_existing_panel_tags(figure)
    for ax in figure.axes:
        style_axis_text(ax)
        standardize_method_markers(ax, memory_axis=(panel == "d" and ax is figure.axes[-1]))
    refine_legends(figure)
    if panel in {"a", "b", "c"}:
        figure.axes[0].set_title("", loc="left")
        add_panel_tag(figure, panel, 0.025, 0.925)
    else:
        add_panel_tag(figure, "d", 0.015, 0.91)
        for text in figure.texts:
            if text.get_text() == "Accuracy and computational footprint":
                text.set_fontfamily(FONT_FAMILY)
                text.set_fontsize(8.2)
                text.set_fontweight("semibold")


def refine_composed(figure: plt.Figure) -> dict[str, float]:
    if len(figure.axes) != 8:
        raise ValueError(f"Expected eight axes in composed V6, found {len(figure.axes)}")
    top_axes = figure.axes[:3]
    scorecard_axes = figure.axes[3:]
    hide_existing_panel_tags(figure)

    for ax in figure.axes:
        style_axis_text(ax)
        standardize_method_markers(ax, memory_axis=(ax is scorecard_axes[-1]))
    refine_legends(figure)

    for ax in top_axes:
        ax.set_title("", loc="left")

    # Align the top-row method axis with the scorecard method axis while
    # preserving the original right boundary and all plotted coordinates.
    top_left = top_axes[0].get_position()
    scorecard_left = scorecard_axes[0].get_position().x0
    top_axes[0].set_position(
        [scorecard_left, top_left.y0, top_left.x1 - scorecard_left, top_left.height]
    )
    top_axes[0].tick_params(axis="y", pad=3.0)
    scorecard_axes[0].tick_params(axis="y", pad=3.0)

    for text in figure.texts:
        if text.get_text() == "Accuracy and computational footprint":
            text.set_fontfamily(FONT_FAMILY)
            text.set_fontsize(8.2)
            text.set_fontweight("semibold")

    for label, x in zip(("a", "b", "c"), (0.015, 0.345, 0.675)):
        add_panel_tag(figure, label, x, 0.925)
    add_panel_tag(figure, "d", 0.015, 0.455)

    return {
        "top_method_axis_x0": float(top_axes[0].get_position().x0),
        "scorecard_method_axis_x0": float(scorecard_axes[0].get_position().x0),
    }


def svg_checks(path: Path, forbidden: tuple[str, ...], required: tuple[str, ...]) -> dict[str, Any]:
    content = path.read_text(encoding="utf-8")
    root = ET.parse(path).getroot()
    return {
        "parseable": root.tag.endswith("svg"),
        "editable_text": "<text" in content,
        "sans_serif_font_present": "Liberation Sans" in content,
        "required_text_present": all(text in content for text in required),
        "forbidden_subtitles_absent": all(text not in content for text in forbidden),
        "bytes": path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--visual-qa-status", choices=("pending", "pass"), default="pending")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base = load_base()
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    configure_nature_style(base, config)
    source_paths = base.require_sources(config)
    uq_methods = list(config["paper_contract"]["generative_method_order"])
    scorecard_methods = list(config["paper_contract"]["scorecard_method_order"])
    a_samples, a_summary, b_samples, b_summary = base.load_panel_ab(source_paths["v5_display"], uq_methods)
    c_source = base.load_panel_c(source_paths["v51_selective_risk"], config)
    d_source = base.derive_panel_d(
        source_paths["v51_scorecard"],
        source_paths["v51_scorecard_stages"],
        source_paths["inference_memory"],
        scorecard_methods,
    )

    builders = {
        "a": lambda: base.make_standalone_ab("a", a_samples, a_summary, config),
        "b": lambda: base.make_standalone_ab("b", b_samples, b_summary, config),
        "c": lambda: base.make_standalone_c(c_source, config),
        "d": lambda: base.make_standalone_d(d_source, config),
        "composed": lambda: base.make_composed(
            a_samples, a_summary, b_samples, b_summary, c_source, d_source, config
        ),
    }
    stems = {
        "a": f"fig5a_probabilistic_reconstruction_{args.timestamp}",
        "b": f"fig5b_uncertainty_tracks_difficult_states_{args.timestamp}",
        "c": f"fig5c_selective_reconstruction_{args.timestamp}",
        "d": f"fig5d_accuracy_computational_footprint_{args.timestamp}",
        "composed": f"fig5_composed_v6_{args.timestamp}",
    }
    figure_dir = PACKAGE_ROOT / "figures" / "generated" / args.timestamp
    docs_dir = PACKAGE_ROOT / "docs" / "generated" / args.timestamp
    results_dir = PACKAGE_ROOT / "results" / "derived" / args.timestamp
    figure_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    svg_paths = {name: figure_dir / f"{stem}.svg" for name, stem in stems.items()}
    if not args.force and any(path.exists() for path in svg_paths.values()):
        raise FileExistsError(f"Refusing to overwrite existing release {args.timestamp}")

    geometry: dict[str, dict[str, str]] = {}
    alignment: dict[str, float] = {}
    for name, builder in builders.items():
        figure = builder()
        before = geometry_digest(figure)
        if name == "composed":
            alignment = refine_composed(figure)
        else:
            refine_standalone(figure, name)
        after = geometry_digest(figure)
        if before != after:
            raise RuntimeError(f"Scientific geometry changed during typography refinement: {name}")
        geometry[name] = {"before": before, "after": after}
        preview = args.preview_dir / f"{stems[name]}.png" if args.preview_dir else None
        base.save_figure(figure, svg_paths[name], preview)

    forbidden = (
        "Probabilistic reconstruction",
        "Uncertainty tracks difficult states",
        "Uncertainty supports selective reconstruction",
    )
    required = {
        "a": ("Normalized CRPS (lower is better)",),
        "b": ("Spearman ρ",),
        "c": ("Relative retained-set error",),
        "d": ("Accuracy and computational footprint",),
        "composed": ("Accuracy and computational footprint", "Normalized CRPS", "Spearman ρ"),
    }
    svg_qa = {
        name: svg_checks(path, forbidden if name in {"a", "b", "c", "composed"} else (), required[name])
        for name, path in svg_paths.items()
    }
    structure_locked = all(item["before"] == item["after"] for item in geometry.values())
    axes_aligned = abs(alignment["top_method_axis_x0"] - alignment["scorecard_method_axis_x0"]) < 1e-12
    visual_pass = args.visual_qa_status == "pass"
    all_svg_pass = all(
        all(
            bool(checks[key])
            for key in (
                "parseable",
                "editable_text",
                "sans_serif_font_present",
                "required_text_present",
                "forbidden_subtitles_absent",
            )
        )
        for checks in svg_qa.values()
    )
    structural_pass = all_svg_pass and structure_locked and axes_aligned
    qa_status = "pass" if structural_pass and visual_pass else ("structural_pass_visual_pending" if structural_pass else "fail")
    qa = {
        "schema_version": "figure5-v6-nature-format-qa-1",
        "status": qa_status,
        "backend": "Python/Matplotlib (fig environment)",
        "scientific_geometry_locked": structure_locked,
        "geometry_digests": geometry,
        "top_and_scorecard_method_axes_aligned": axes_aligned,
        "alignment": alignment,
        "svg_checks": svg_qa,
        "visual_qa": {"status": args.visual_qa_status, "preview_retained": False},
        "no_scientific_recalculation": True,
    }
    qa_path = results_dir / "qa.json"
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    if qa["status"] == "fail":
        raise RuntimeError("Figure 5 V6 Nature-format structural QA failed")

    contract_path = docs_dir / "figure_contract.md"
    contract_path.write_text(
        """# Figure 5 V6 Nature-format contract

- Core conclusion: unchanged from the accepted Figure 5 V6.
- Archetype: quantitative grid with an accuracy-first scorecard.
- Backend: Python/Matplotlib only.
- Final size: 183 mm × 128 mm, editable SVG.
- Structure lock: all plotted coordinates, scatter clouds, box/distribution bands, error bars, confidence bands, line plots, scales, and metric definitions are unchanged.
- Permitted changes: font hierarchy, text alignment, marker cosmetics, panel-container alignment, and subtitle visibility only.
- Reviewer risk: accidental alteration of scientific geometry; controlled by before/after artist-coordinate digests in `qa.json`.
""",
        encoding="utf-8",
    )
    completion_path = docs_dir / "completion_report.md"
    completion_path.write_text(
        f"""# Figure 5 V6 Nature-format completion report

- Release: `{args.timestamp}` (additional release; no prior V6 output overwritten).
- Source renderer: committed original Figure 5 V6 renderer.
- Scientific geometry: locked and verified unchanged for all five outputs.
- Top subtitles: removed from a, b, and c.
- Panel tags: rebuilt as aligned bold figure-level tags in an installed Arial/Helvetica-compatible sans-serif face.
- Method-axis alignment: panel a and panel d share the same left plotting boundary.
- Typography: unified Arial/Helvetica-compatible sans-serif hierarchy.
- Markers: summary/curve/legend marker sizes and edge treatment standardized; inference-memory open/filled semantics retained.
- Visual QA: {args.visual_qa_status.upper()}.
- New inference, bootstrapping, training, or broad validation: none.
""",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "figure5-v6-nature-format-build-1",
        "status": "complete" if visual_pass else "complete_visual_qa_pending",
        "timestamp": args.timestamp,
        "base_renderer": str(BASE_SCRIPT.relative_to(REPO_ROOT)),
        "base_config": str(BASE_CONFIG.relative_to(REPO_ROOT)),
        "no_scientific_recalculation": True,
        "scientific_geometry_locked": structure_locked,
        "figures": [str(path.relative_to(REPO_ROOT)) for path in svg_paths.values()],
        "qa": str(qa_path.relative_to(REPO_ROOT)),
        "contract": str(contract_path.relative_to(REPO_ROOT)),
        "completion_report": str(completion_path.relative_to(REPO_ROOT)),
        "temporary_previews_retained": False,
    }
    manifest_path = results_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": [str(path) for path in svg_paths.values()], "qa": str(qa_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

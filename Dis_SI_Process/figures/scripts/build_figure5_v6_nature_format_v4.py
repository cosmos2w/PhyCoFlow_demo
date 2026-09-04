#!/usr/bin/env python
"""Restore method shapes and simplify the memory labels from Figure 5 V6 release 20260903_1332."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import yaml
from matplotlib.lines import Line2D


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
V3_SCRIPT = SCRIPT_DIR / "build_figure5_v6_nature_format_v3.py"
CONFIG_PATH = PACKAGE_ROOT / "configs" / "figure5_v6.yaml"


def load_v3() -> Any:
    spec = importlib.util.spec_from_file_location("figure5_v6_nature_v3", V3_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load prior renderer: {V3_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V3 = load_v3()
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
UQ_METHODS = list(CONFIG["paper_contract"]["generative_method_order"])
SCORECARD_METHODS = list(CONFIG["paper_contract"]["scorecard_method_order"])
METHOD_MARKERS = dict(CONFIG["style"]["method_markers"])
REFINE_STANDALONE_V3 = V3.refine_standalone_v3
REFINE_COMPOSED_V3 = V3.refine_composed_v3


def marker_lines(ax: Any) -> list[Any]:
    return [
        line
        for line in ax.lines
        if line.get_marker() not in {None, "None", "none", "", " "}
    ]


def restore_hollow_method_shapes(ax: Any, methods: list[str]) -> None:
    lines = marker_lines(ax)
    if len(lines) != len(methods):
        raise ValueError(f"Expected {len(methods)} method markers, found {len(lines)}")
    for line, method in zip(lines, methods):
        line.set_marker(METHOD_MARKERS[method])
        line.set_markerfacecolor("white")
        line.set_markersize(5.0)
        line.set_markeredgewidth(1.10)


def restore_memory_method_shapes(ax: Any) -> list[Any]:
    lines = marker_lines(ax)
    if len(lines) != 2 * len(SCORECARD_METHODS):
        raise ValueError(f"Expected sixteen inference-memory endpoints, found {len(lines)}")
    for method_index, method in enumerate(SCORECARD_METHODS):
        model_line = lines[2 * method_index]
        peak_line = lines[2 * method_index + 1]
        model_line.set_marker(METHOD_MARKERS[method])
        peak_line.set_marker(METHOD_MARKERS[method])
        model_line.set_markersize(5.0)
        peak_line.set_markersize(5.0)
        model_line.set_markeredgewidth(1.10)
        peak_line.set_markeredgewidth(1.10)
        # V3 already established the requested filled-model / hollow-peak encoding.
    return lines


def remove_memory_key_and_match_title(ax: Any, reference_ax: Any) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    reference_title = reference_ax._left_title
    ax.set_title(
        "Inference memory",
        loc="left",
        pad=4.2,
        fontsize=reference_title.get_fontsize(),
        fontweight=reference_title.get_fontweight(),
        fontfamily="Arial",
        color=reference_title.get_color(),
    )


def add_top_row_memory_labels(figure: Any, ax: Any, lines: list[Any]) -> None:
    """Label the DMF endpoint pair with figure-level text and thin leaders."""
    figure.canvas.draw()
    endpoint_lines = (lines[0], lines[1])
    labels = ("Model", "Peak")
    for line, label in zip(endpoint_lines, labels):
        x = float(line.get_xdata(orig=False)[0])
        y = float(line.get_ydata(orig=False)[0])
        display_xy = ax.transData.transform((x, y))
        figure_xy = figure.transFigure.inverted().transform(display_xy)
        text_y = float(figure_xy[1] + 0.017)
        figure.lines.append(
            Line2D(
                [figure_xy[0], figure_xy[0]],
                [figure_xy[1] + 0.004, text_y - 0.003],
                transform=figure.transFigure,
                color="#555555",
                linewidth=0.50,
                solid_capstyle="round",
                clip_on=False,
            )
        )
        figure.text(
            float(figure_xy[0]),
            text_y,
            label,
            ha="center",
            va="bottom",
            fontsize=5.2,
            fontfamily="Arial",
            color="#303030",
        )


def refine_standalone_v4(figure: Any, panel: str) -> None:
    REFINE_STANDALONE_V3(figure, panel)
    if panel in {"a", "b", "c"}:
        restore_hollow_method_shapes(figure.axes[0], UQ_METHODS)
        return
    for ax in figure.axes[:-1]:
        restore_hollow_method_shapes(ax, SCORECARD_METHODS)
    memory_lines = restore_memory_method_shapes(figure.axes[-1])
    remove_memory_key_and_match_title(figure.axes[-1], figure.axes[-2])
    add_top_row_memory_labels(figure, figure.axes[-1], memory_lines)


def refine_composed_v4(figure: Any) -> dict[str, float]:
    alignment = REFINE_COMPOSED_V3(figure)
    top_axes = figure.axes[:3]
    scorecard_axes = figure.axes[3:]
    for ax in top_axes:
        restore_hollow_method_shapes(ax, UQ_METHODS)
    for ax in scorecard_axes[:-1]:
        restore_hollow_method_shapes(ax, SCORECARD_METHODS)
    memory_lines = restore_memory_method_shapes(scorecard_axes[-1])
    remove_memory_key_and_match_title(scorecard_axes[-1], scorecard_axes[-2])
    add_top_row_memory_labels(figure, scorecard_axes[-1], memory_lines)
    return alignment


V3.V2.V1.refine_standalone = refine_standalone_v4
V3.V2.V1.refine_composed = refine_composed_v4


def update_release_documents(timestamp: str) -> None:
    docs_dir = PACKAGE_ROOT / "docs" / "generated" / timestamp
    results_dir = PACKAGE_ROOT / "results" / "derived" / timestamp
    qa_path = results_dir / "qa.json"
    manifest_path = results_dir / "build_manifest.json"

    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["schema_version"] = "figure5-v6-nature-format-v4-qa-1"
    qa["based_on_release"] = "20260903_1332"
    qa["prior_release_untouched"] = "20260903_1332"
    qa["marker_encoding"] = "original method shapes, white-filled with thick method-color borders"
    qa["inference_memory_encoding"] = {
        "method_shape": "preserved for both endpoints",
        "model_parameters_and_buffers": "filled shape",
        "peak_allocated": "hollow shape",
        "label_strategy": "Model and Peak labels with thin leaders on the top method row only",
        "boxed_key": False,
    }
    qa["inference_memory_title_matches_other_scorecard_titles"] = True
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "figure5-v6-nature-format-v4-build-1"
    manifest["based_on_release"] = "20260903_1332"
    manifest["renderer"] = str(Path(__file__).resolve().relative_to(REPO_ROOT))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    (docs_dir / "figure_contract.md").write_text(
        """# Figure 5 V6 Nature-format V4 contract

- Base release: `20260903_1332`; preserved without modification.
- Font: installed Microsoft Arial.
- Scientific coordinates, scatter clouds, distributions, error bars, bands, curves, scales, and axes limits: unchanged.
- Method markers: original method-specific shapes restored; each shape is white-filled with a thick method-color border outside inference memory.
- Inference memory: original method shape retained; filled endpoint = model parameters plus persistent buffers, hollow endpoint = peak allocated memory.
- Memory labels: concise `Model` and `Peak` labels with thin leaders identify the two endpoints once on the top method row; no compact key remains.
- Inference-memory title: identical typography and padding to the adjacent scorecard titles.
""",
        encoding="utf-8",
    )
    (docs_dir / "completion_report.md").write_text(
        f"""# Figure 5 V6 Nature-format V4 completion report

- Release: `{timestamp}`; additional release based on `20260903_1332`.
- Restored all original method-specific marker shapes while retaining the enlarged hollow treatment.
- Retained filled-model / hollow-peak inference-memory endpoints in each method's own shape.
- Removed the compact memory key and labeled the top-row endpoint pair with thin leaders.
- Matched the inference-memory title to all other scorecard titles.
- Scientific geometry digests: unchanged before/after for every output.
- New inference, bootstrap, training, or broad validation: none.
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--timestamp", required=True)
    args, _ = parser.parse_known_args()
    result = V3.main()
    update_release_documents(args.timestamp)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

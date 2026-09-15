#!/usr/bin/env python
"""Assemble corrected mixed-resolution Figure V4_6 from frozen V4_5 state."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import global_style as manuscript
import matplotlib.pyplot as plt

from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config
from common.figure_style import save_figure
from common.io_utils import write_json
from common.physical_figure_layout import validate_panel_text_boundaries
from common.publication_panels_unified_v4_6 import (
    align_panel_b_ylabels,
    apply_v4_6_typography,
    center_panel_d_headers,
    draw_panel,
    measure_adjacent_major_row_unions,
    measure_major_content_gaps,
    measure_panel_b_legend_clearance,
    measure_panel_d_header_alignment,
    panel_label,
    record_panel_e_v4_3_tick_settings,
)

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4_5 = _load("mixed_resolution_v4_5_base_for_v4_6", HERE / "133_assemble_mixed_resolution_unified_v4_5.py")
v4 = v4_5.v4
base = v4_5.base
V4_5_RELEASE = (
    HERE.parents[2] / "figures" / "generated" / "art_style_review"
    / "MixedResolution_unified_v4_5_20260914_2136"
)


def _v4_5_baseline_anchor() -> dict:
    paths = {
        "release_pdf": V4_5_RELEASE / "MixedResolution_unified_v4_5_20260914_2136.pdf",
        "release_svg": V4_5_RELEASE / "MixedResolution_unified_v4_5_20260914_2136.svg",
        "release_png": V4_5_RELEASE / "MixedResolution_unified_v4_5_20260914_2136.png",
        "release_manifest": V4_5_RELEASE / "source_manifest_v4_5.json",
        "release_qa": V4_5_RELEASE / "qa_v4_5.json",
        "renderer": HERE / "133_assemble_mixed_resolution_unified_v4_5.py",
        "exporter": HERE / "134_export_unified_v4_5_panels.py",
        "audit": HERE / "135_audit_mixed_resolution_v4_5.py",
        "panel_renderer": HERE / "common" / "publication_panels_unified_v4_5.py",
        "layout": HERE / "publication_layout_unified_v4_5.yaml",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V4_5 baseline anchor is incomplete: {missing}")
    return {key: v4._record(path) for key, path in paths.items()}


def _geometry_v4_6(layout):
    width, height, rects = v4_5._geometry_v4_5(layout)
    rects = {label: list(values) for label, values in rects.items()}
    cfg = layout["v4_6_geometry"]
    height += float(cfg["canvas_height_increase_mm"])
    rects["b"][1] += float(cfg["panel_b_bottom_upshift_mm"])
    rects["b"][3] += float(cfg["panel_b_height_increase_mm"])
    rects["a"][1] += float(cfg["panel_a_bottom_upshift_mm"])
    if abs(rects["a"][1] + rects["a"][3] - height) > 1e-8:
        raise ValueError("V4_6 geometry does not close at the top canvas edge")
    return width, height, rects


def _create_canvas_v4_6(layout):
    previous = v4._geometry
    v4._geometry = _geometry_v4_6
    try:
        return v4.create_canvas(layout)
    finally:
        v4._geometry = previous


def _create_standalone_canvas_v4_6(layout, label):
    previous = v4._geometry
    v4._geometry = _geometry_v4_6
    try:
        return v4.create_standalone_canvas(layout, label)
    finally:
        v4._geometry = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=HERE / "publication_layout_unified_v4_6.yaml")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    rid = base.publication_timestamp(args.run_id)
    layout_path = Path(args.layout).resolve()
    cfg = load_config(args.config)
    style_record = v4_5.apply_v4_5_style_contract(cfg)
    style_record["scope"] = "V4_6 approved geometry/text refinement; V4_5 scientific state preserved"
    ensure_output_dirs()
    layout = base.load_layout(layout_path)
    ctx = base.make_context(args, cfg, layout, rid)
    v3_anchor = v4._v3_7_baseline_anchor()
    v4_5_anchor = _v4_5_baseline_anchor()
    v4_5_manifest = json.loads((V4_5_RELEASE / "source_manifest_v4_5.json").read_text(encoding="utf-8"))
    results_before = v4._tree_state(RESULTS_DIR)
    out = FIGURES_DIR / "Assembled" / f"MixedResolution_unified_v4_6_{rid}"
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v4_6_{rid}.json"
    targets = [out.with_suffix(ext) for ext in (".svg", ".pdf", ".png")] + [manifest_path]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"Refusing to overwrite existing V4_6 artifacts for {rid}")

    fig, axes, containers, rects, width, height, shared_axes, cbar_parent = _create_canvas_v4_6(layout)
    panel_meta = {}
    for label in ("a", "b"):
        panel_label(axes[label], label)
        panel_meta[label] = draw_panel(label, axes[label], ctx)
    panel_label(axes["c"], "c")
    panel_meta["c"] = draw_panel("c", axes["c"], ctx, shared_axes=shared_axes["c"], colorbar_parent=cbar_parent)
    panel_label(axes["d"], "d")
    panel_meta["d"] = draw_panel("d", axes["d"], ctx, shared_axes=shared_axes["d"])
    panel_label(axes["e"], "e")
    panel_meta["e"] = draw_panel("e", axes["e"], ctx)

    for axis in fig.axes:
        for item in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            if item.get_visible() and item.get_text():
                item._global_font_role = "tick_label"
    geometric_axis_count = base.enforce_geometric_aspects(fig)
    fig.canvas.draw()
    center_panel_d_headers(axes["d"], shared_axes["d"])
    v4_5.activate_v4_5_font_roles(cfg)
    role_qa = apply_v4_6_typography(fig)
    typography_qa = manuscript.enforce_figure_typography(fig, font_family=manuscript.FONT_FAMILY)
    center_panel_d_headers(axes["d"], shared_axes["d"])
    panel_meta["b"].update(align_panel_b_ylabels(axes["b"], strict=True))
    panel_meta["e"].update(record_panel_e_v4_3_tick_settings(axes["e"], strict=True))
    panel_meta["b"].update(measure_panel_b_legend_clearance(axes["b"], strict=True))
    panel_meta["d"].update(measure_panel_d_header_alignment(axes["d"], shared_axes["d"], strict=True))
    panel_meta["e"].update({
        "v4_6_matrix_annotation_count": role_qa["matrix_annotation_count"],
        "v4_6_matrix_min_contrast_ratio": role_qa["matrix_annotation_min_contrast_ratio"],
        "v4_6_matrix_annotation_colors": role_qa["matrix_annotation_colors"],
    })
    baseline_bar_height = float(v4_5_manifest["panels"]["b"]["grouped_bar_axis_height_mm"])
    top_axis, _sweeps = __import__(
        "common.publication_panels_unified_v4_6", fromlist=["_panel_b_axes"]
    )._panel_b_axes(axes["b"])
    bar_height = float(top_axis.get_position().height * fig.get_size_inches()[1] * 25.4)
    bar_ratio = bar_height / baseline_bar_height
    panel_meta["b"].update({
        "grouped_bar_axis_height_mm": bar_height,
        "grouped_bar_axis_height_v4_5_mm": baseline_bar_height,
        "grouped_bar_axis_height_ratio_to_v4_5": bar_ratio,
        "grouped_bar_axis_height_exactly_1_25x_v4_5": abs(bar_ratio - 1.25) <= 1e-8,
    })
    if not panel_meta["b"]["grouped_bar_axis_height_exactly_1_25x_v4_5"]:
        raise ValueError(f"V4_6 Panel-b height ratio failed: {bar_ratio}")
    union_qa = measure_adjacent_major_row_unions(
        fig, axes, shared_axes, cbar_parent,
        minimum_at_162_mm=float(layout["v4_6_geometry"]["minimum_adjacent_row_union_gap_mm_at_162mm"]),
        strict=True,
    )
    legacy_spacing = measure_major_content_gaps(fig, axes, shared_axes, cbar_parent, strict=False)
    frame_qa = base.enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    model_artist_qa = v4._model_artist_qa(fig, cfg)
    geometry_qa = v4._geometry_qa(fig, containers, rects)
    alignment_qa = v4._panel_cd_alignment_qa(fig, shared_axes, cfg["figure_style"]["paper_dpi"])
    if not model_artist_qa["passed"] or not geometry_qa["passed"] or not alignment_qa["passed"]:
        raise ValueError("V4_6 model, geometry, or c/d alignment contract failed")
    if not role_qa["only_panel_labels_bold"] or role_qa["local_size_override_count"]:
        raise ValueError(f"V4_6 typography role contract failed: {role_qa}")
    qa = {
        "geometric_axis_count": geometric_axis_count,
        "typography_qa": typography_qa,
        "typography_role_qa": role_qa,
        "frame_lineweight_qa": frame_qa,
        "model_artist_qa": model_artist_qa,
        "geometry_qa": geometry_qa,
        "adjacent_major_row_union_qa": union_qa,
        "legacy_major_content_spacing_diagnostic": legacy_spacing,
        "text_overflow_in": dict(manuscript.validate_text_within_canvas(fig)),
        "panel_text_clearance_qa": validate_panel_text_boundaries(fig, containers),
    }
    outputs = save_figure(fig, out, cfg, formats=("svg", "pdf", "png"),
                          dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None)
    plt.close(fig)
    results_after = v4._tree_state(RESULTS_DIR)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during V4_6 rendering")

    manifest = v4.build_manifest(
        ctx, cfg, layout, layout_path, outputs, panel_meta, rects, width, height,
        qa, alignment_qa, style_record, args, rid, v3_anchor, results_before, results_after,
    )
    manifest.update({
        "workflow_label": "mixed_resolution_unified_v4_6",
        "schema_version": "4.6", "revision": "V4_6",
        "source_visual_revision": "V4_5",
        "v4_5_baseline_anchor": v4_5_anchor,
        "renderer": v4._record(Path(__file__).resolve()),
        "panel_renderer": v4._record(HERE / "common" / "publication_panels_unified_v4_6.py"),
        "exporter": v4._record(HERE / "137_export_unified_v4_6_panels.py"),
        "author_authorized_deltas": {
            "underlying_source_arrays_changed": False,
            "v4_6": {
                "raw_data_delta": False, "data_mapping_delta": False,
                "bar_axis_height_ratio_to_v4_5": bar_ratio,
                "panel_b_zero_h_text_removed": True,
                "panel_c_error_title_replacement": ["Zoom-region absolute error", "Absolute error"],
                "panel_cd_row_gap_mm": float(layout["v4_geometry"]["shared_cd_grid"]["row_gap_mm"]),
                "panel_c_colorbar_strip_mm": float(layout["v4_geometry"]["shared_cd_grid"]["bottom_colorbar_strip_mm"]),
            },
        },
    })
    manifest["layout"].update({
        "v4_6_canvas_height_increase_mm": float(layout["v4_6_geometry"]["canvas_height_increase_mm"]),
        "v4_6_panel_b_height_increase_mm": float(layout["v4_6_geometry"]["panel_b_height_increase_mm"]),
        "shared_cd_row_gap_mm": float(layout["v4_geometry"]["shared_cd_grid"]["row_gap_mm"]),
        "panel_c_colorbar_strip_mm": float(layout["v4_geometry"]["shared_cd_grid"]["bottom_colorbar_strip_mm"]),
        "adjacent_major_row_union_qa": union_qa,
        "major_vertical_gaps_mm": {
            f"{item['upper_row']}_{item['lower_row']}": item["vertical_gap_mm"] for item in union_qa["adjacent_major_row_checks"]
        },
        "major_vertical_gap_mm": min(item["vertical_gap_mm"] for item in union_qa["adjacent_major_row_checks"]),
    })
    manifest["figure_contract"].update({
        "revision_scope": "V4_6 approved layout, typography, and wording corrections only; V4_5 data/mapping preserved",
        "underlying_source_arrays_unchanged": True,
        "release_status": "art reviewed, scientific release pending",
    })
    manifest["selection_contract"]["panel_b"]["sweep_recipes"] = list(layout["panel_b_v4"]["sweep_recipes"])
    manifest["selection_contract"]["panel_b"]["sweep_count"] = 3
    manifest["selection_contract"]["panel_d"]["semantic_colormaps"] = {
        "truth_component": layout["panel_d_v4"]["truth_component_cmap"],
        "residuals": layout["panel_d_v4"]["residual_cmap"],
    }
    manifest["style_contract"]["semantic_colormaps"] = {
        "panel_c_physical_field": cfg["rendering"]["cmap"],
        "panel_c_absolute_error": cfg["rendering"]["error_cmap"],
        "panel_d_truth_component": layout["panel_d_v4"]["truth_component_cmap"],
        "panel_d_residuals": layout["panel_d_v4"]["residual_cmap"],
    }
    write_json(manifest_path, manifest)
    print(f"[OK] {out}.pdf")
    print(f"[OK] {manifest_path}")


if __name__ == "__main__":
    main()

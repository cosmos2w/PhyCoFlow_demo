#!/usr/bin/env python
"""Assemble visual-only mixed-resolution Figure V4_7 from frozen V4_6."""
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
from common.publication_panels_unified_v4_7 import (
    align_panel_b_ylabels,
    align_panel_cd_annotation_gutters,
    apply_v4_7_typography,
    center_panel_d_headers,
    draw_panel,
    measure_adjacent_major_row_unions,
    measure_annotation_collision_gate,
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


v46 = _load("mixed_resolution_v4_6_base_for_v4_7", HERE / "136_assemble_mixed_resolution_unified_v4_6.py")
v4 = v46.v4
base = v46.base
V4_6_RELEASE = (
    HERE.parents[2] / "figures" / "generated" / "art_style_review"
    / "MixedResolution_unified_v4_6_20260914_2227"
)


def _v4_6_baseline_anchor() -> dict:
    paths = {
        "release_pdf": V4_6_RELEASE / "MixedResolution_unified_v4_6_20260914_2227.pdf",
        "release_svg": V4_6_RELEASE / "MixedResolution_unified_v4_6_20260914_2227.svg",
        "release_png": V4_6_RELEASE / "MixedResolution_unified_v4_6_20260914_2227.png",
        "release_manifest": V4_6_RELEASE / "source_manifest_v4_6.json",
        "release_qa": V4_6_RELEASE / "qa_v4_6.json",
        "renderer": HERE / "136_assemble_mixed_resolution_unified_v4_6.py",
        "exporter": HERE / "137_export_unified_v4_6_panels.py",
        "audit": HERE / "138_audit_mixed_resolution_v4_6.py",
        "panel_renderer": HERE / "common" / "publication_panels_unified_v4_6.py",
        "layout": HERE / "publication_layout_unified_v4_6.yaml",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V4_6 baseline anchor is incomplete: {missing}")
    return {key: v4._record(path) for key, path in paths.items()}


def _geometry_v4_7(layout):
    width, height, rects = v46._geometry_v4_6(layout)
    rects = {label: list(values) for label, values in rects.items()}
    cfg = layout["v4_7_geometry"]
    old_b_top = rects["b"][1] + rects["b"][3]
    rects["b"][1] -= float(cfg["panel_b_bottom_downshift_mm"])
    rects["b"][3] += float(cfg["panel_b_height_increase_mm"])
    for label in ("c", "d"):
        rects[label][1] += float(cfg["panel_cd_bottom_upshift_mm"])
        rects[label][3] -= float(cfg["panel_cd_height_decrease_mm"])
    rects["e"][1] += float(cfg["panel_e_bottom_upshift_mm"])
    if abs(rects["b"][1] + rects["b"][3] - old_b_top) > 1e-9:
        raise ValueError("V4_7 Panel-b top edge must remain fixed")
    if rects["c"] != rects["d"][:1] + rects["c"][1:]:
        # Widths differ, while bottom and height must stay shared.
        if abs(rects["c"][1] - rects["d"][1]) > 1e-9 or abs(rects["c"][3] - rects["d"][3]) > 1e-9:
            raise ValueError("V4_7 c/d shared vertical geometry diverged")
    return width, height, rects


def _create_canvas_v4_7(layout):
    previous = v4._geometry
    v4._geometry = _geometry_v4_7
    try:
        return v4.create_canvas(layout)
    finally:
        v4._geometry = previous


def _create_standalone_canvas_v4_7(layout, label):
    previous = v4._geometry
    v4._geometry = _geometry_v4_7
    try:
        return v4.create_standalone_canvas(layout, label)
    finally:
        v4._geometry = previous


def _axis_height_mm(axis, fig) -> float:
    return float(axis.get_position().height * fig.get_size_inches()[1] * 25.4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=HERE / "publication_layout_unified_v4_7.yaml")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    rid = base.publication_timestamp(args.run_id)
    layout_path = Path(args.layout).resolve()
    cfg = load_config(args.config)
    style_record = v46.v4_5.apply_v4_5_style_contract(cfg)
    style_record["scope"] = "V4_7 approved text-gutter, colourbar-removal, and sweep-height refinement"
    ensure_output_dirs()
    layout = base.load_layout(layout_path)
    ctx = base.make_context(args, cfg, layout, rid)
    v3_anchor = v4._v3_7_baseline_anchor()
    v4_6_anchor = _v4_6_baseline_anchor()
    v4_6_manifest = json.loads((V4_6_RELEASE / "source_manifest_v4_6.json").read_text(encoding="utf-8"))
    results_before = v4._tree_state(RESULTS_DIR)
    out = FIGURES_DIR / "Assembled" / f"MixedResolution_unified_v4_7_{rid}"
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v4_7_{rid}.json"
    targets = [out.with_suffix(ext) for ext in (".svg", ".pdf", ".png")] + [manifest_path]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"Refusing to overwrite existing V4_7 artifacts for {rid}")

    fig, axes, containers, rects, width, height, shared_axes, cbar_parent = _create_canvas_v4_7(layout)
    panel_meta = {}
    for label in ("a", "b"):
        panel_label(axes[label], label)
        panel_meta[label] = draw_panel(label, axes[label], ctx)
    panel_label(axes["c"], "c")
    panel_meta["c"] = draw_panel(
        "c", axes["c"], ctx, shared_axes=shared_axes["c"], colorbar_parent=cbar_parent,
    )
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
    v46.v4_5.activate_v4_5_font_roles(cfg)
    role_qa = apply_v4_7_typography(fig)
    typography_qa = manuscript.enforce_figure_typography(fig, font_family=manuscript.FONT_FAMILY)
    center_panel_d_headers(axes["d"], shared_axes["d"])
    annotation_alignment = align_panel_cd_annotation_gutters(
        fig, shared_axes,
        offset_mm=float(layout["panel_c_v4"]["bottom_label_offset_mm"]), strict=True,
    )
    panel_meta["c"].update(annotation_alignment)
    panel_meta["d"].update(annotation_alignment)
    panel_meta["b"].update(align_panel_b_ylabels(axes["b"], strict=True))
    panel_meta["e"].update(record_panel_e_v4_3_tick_settings(axes["e"], strict=True))
    panel_meta["b"].update(measure_panel_b_legend_clearance(axes["b"], strict=True))
    panel_meta["d"].update(measure_panel_d_header_alignment(axes["d"], shared_axes["d"], strict=True))
    panel_meta["e"].update({
        "v4_7_matrix_annotation_count": role_qa["matrix_annotation_count"],
        "v4_7_matrix_min_contrast_ratio": role_qa["matrix_annotation_min_contrast_ratio"],
        "v4_7_matrix_annotation_colors": role_qa["matrix_annotation_colors"],
    })
    top_axis, sweeps = __import__(
        "common.publication_panels_unified_v4_6", fromlist=["_panel_b_axes"]
    )._panel_b_axes(axes["b"])
    bar_height = _axis_height_mm(top_axis, fig)
    sweep_heights = [_axis_height_mm(axis, fig) for axis in sweeps]
    prior_bar_height = float(v4_6_manifest["panels"]["b"]["grouped_bar_axis_height_mm"])
    prior_sweep_height = float(v4_6_manifest["panels"]["b"]["sweep_axis_heights_mm"][0])
    panel_meta["b"].update({
        "grouped_bar_axis_height_mm": bar_height,
        "grouped_bar_axis_height_v4_6_mm": prior_bar_height,
        "grouped_bar_height_preserved_from_v4_6": abs(bar_height - prior_bar_height) <= 0.02,
        "sweep_axis_heights_mm": sweep_heights,
        "sweep_axis_height_v4_6_mm": prior_sweep_height,
        "minimum_sweep_height_ratio_to_v4_6": min(sweep_heights) / prior_sweep_height,
    })
    if not panel_meta["b"]["grouped_bar_height_preserved_from_v4_6"]:
        raise ValueError(f"V4_7 changed Panel-b grouped-bar height: {bar_height} vs {prior_bar_height}")
    if panel_meta["b"]["minimum_sweep_height_ratio_to_v4_6"] < 1.10:
        raise ValueError(f"V4_7 Panel-b sweep height did not increase enough: {panel_meta['b']}")

    annotation_gate = measure_annotation_collision_gate(
        fig,
        minimum_clearance_mm=float(layout["v4_7_annotation_gate"]["minimum_data_window_clearance_mm"]),
        strict=True,
    )
    union_qa = measure_adjacent_major_row_unions(
        fig, axes, shared_axes, cbar_parent,
        minimum_at_162_mm=float(layout["v4_7_geometry"]["minimum_adjacent_row_union_gap_mm_at_162mm"]),
        strict=True,
    )
    legacy_spacing = measure_major_content_gaps(fig, axes, shared_axes, cbar_parent, strict=False)
    frame_qa = base.enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    model_artist_qa = v4._model_artist_qa(fig, cfg)
    geometry_qa = v4._geometry_qa(fig, containers, rects)
    alignment_qa = v4._panel_cd_alignment_qa(fig, shared_axes, cfg["figure_style"]["paper_dpi"])
    if not model_artist_qa["passed"] or not geometry_qa["passed"] or not alignment_qa["passed"]:
        raise ValueError("V4_7 model, geometry, or c/d alignment contract failed")
    qa = {
        "geometric_axis_count": geometric_axis_count,
        "typography_qa": typography_qa,
        "typography_role_qa": role_qa,
        "frame_lineweight_qa": frame_qa,
        "model_artist_qa": model_artist_qa,
        "geometry_qa": geometry_qa,
        "annotation_alignment_qa": annotation_alignment,
        "annotation_collision_qa": annotation_gate,
        "adjacent_major_row_union_qa": union_qa,
        "legacy_major_content_spacing_diagnostic": legacy_spacing,
        "text_overflow_in": dict(manuscript.validate_text_within_canvas(fig)),
        "panel_text_clearance_qa": validate_panel_text_boundaries(fig, containers),
    }
    outputs = save_figure(
        fig, out, cfg, formats=("svg", ".pdf".lstrip("."), "png"),
        dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None,
    )
    plt.close(fig)
    results_after = v4._tree_state(RESULTS_DIR)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during V4_7 rendering")

    manifest = v4.build_manifest(
        ctx, cfg, layout, layout_path, outputs, panel_meta, rects, width, height,
        qa, alignment_qa, style_record, args, rid, v3_anchor, results_before, results_after,
    )
    manifest.update({
        "workflow_label": "mixed_resolution_unified_v4_7",
        "schema_version": "4.7", "revision": "V4_7",
        "source_visual_revision": "V4_6",
        "v4_6_baseline_anchor": v4_6_anchor,
        "renderer": v4._record(Path(__file__).resolve()),
        "panel_renderer": v4._record(HERE / "common" / "publication_panels_unified_v4_7.py"),
        "author_authorized_deltas": {
            "underlying_source_arrays_changed": False,
            "v4_7": {
                "raw_data_delta": False, "data_mapping_delta": False,
                "panel_c_colorbar_axes_removed": 2,
                "panel_c_bottom_labels_moved_after_final_aspect": True,
                "panel_d_metrics_moved_after_final_aspect": True,
                "panel_e_bottom_upshift_mm": float(layout["v4_7_geometry"]["panel_e_bottom_upshift_mm"]),
                "panel_b_minimum_sweep_height_ratio_to_v4_6": panel_meta["b"]["minimum_sweep_height_ratio_to_v4_6"],
            },
        },
    })
    manifest["layout"].update({
        "shared_cd_row_gap_mm": float(layout["v4_geometry"]["shared_cd_grid"]["row_gap_mm"]),
        "panel_c_annotation_gutter_mm": float(layout["v4_geometry"]["shared_cd_grid"]["bottom_colorbar_strip_mm"]),
        "panel_c_colorbar_count": 0,
        "annotation_alignment_qa": annotation_alignment,
        "annotation_collision_qa": annotation_gate,
        "adjacent_major_row_union_qa": union_qa,
        "major_vertical_gap_mm": min(item["vertical_gap_mm"] for item in union_qa["adjacent_major_row_checks"]),
    })
    manifest["figure_contract"].update({
        "revision_scope": "V4_7 visual-only label gutters, Panel-c colourbar removal, Panel-e upshift, and taller Panel-b sweeps",
        "underlying_source_arrays_unchanged": True,
        "release_status": "art reviewed, scientific release pending",
    })
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

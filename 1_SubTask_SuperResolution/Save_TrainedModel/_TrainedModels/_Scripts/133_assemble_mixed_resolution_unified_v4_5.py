#!/usr/bin/env python
"""Assemble mixed-resolution Figure V4_5 from frozen V4_4 state."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import global_style as manuscript
import matplotlib.pyplot as plt

from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config
from common.figure_style import save_figure
from common.io_utils import write_json
from common.physical_figure_layout import validate_panel_text_boundaries
from common.publication_panels_unified_v4_5 import (
    apply_v4_5_typography,
    center_panel_d_headers,
    draw_panel,
    measure_major_content_gaps,
    measure_panel_b_legend_clearance,
    measure_panel_d_header_alignment,
    panel_label,
    record_panel_e_v4_3_tick_settings,
)

HERE = Path(__file__).resolve().parent
V4_5_FONT_ROLES = {
    "panel_label": 11.0,
    "major_title": 9.5,
    "subplot_title": 8.5,
    "axis_label": 8.5,
    "tick_label": 7.8,
    "legend": 7.8,
    "annotation": 7.0,
}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4_4 = _load("mixed_resolution_v4_4_base_for_v4_5", HERE / "130_assemble_mixed_resolution_unified_v4_4.py")
v4 = v4_4.v4
base = v4_4.base
V4_4_RELEASE = (
    HERE.parents[2] / "figures" / "generated" / "art_style_review"
    / "MixedResolution_unified_v4_4_20260915_0001"
)


def _v4_4_baseline_anchor() -> dict:
    paths = {
        "release_pdf": V4_4_RELEASE / "MixedResolution_unified_v4_4_20260915_0001.pdf",
        "release_svg": V4_4_RELEASE / "MixedResolution_unified_v4_4_20260915_0001.svg",
        "release_png": V4_4_RELEASE / "MixedResolution_unified_v4_4_20260915_0001.png",
        "release_manifest": V4_4_RELEASE / "source_manifest_v4_4.json",
        "release_qa": V4_4_RELEASE / "qa_v4_4.json",
        "renderer": HERE / "130_assemble_mixed_resolution_unified_v4_4.py",
        "exporter": HERE / "131_export_unified_v4_4_panels.py",
        "audit": HERE / "132_audit_mixed_resolution_v4_4.py",
        "panel_renderer": HERE / "common" / "publication_panels_unified_v4_4.py",
        "layout": HERE / "publication_layout_unified_v4_4.yaml",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V4_4 baseline anchor is incomplete: {missing}")
    return {key: v4._record(path) for key, path in paths.items()}


def apply_v4_5_style_contract(cfg: dict) -> dict:
    record = v4.apply_v4_style_contract(cfg)

    # Draw with V4_4's exact geometry contract first. Several inherited panel
    # builders perform intermediate bbox assertions while constructing nested
    # axes. The V4_5 hierarchy is activated after every artist exists, which
    # preserves those geometry decisions while making the final export and its
    # renderer-based QA authoritative.
    record.update({
        "font_role_sizes_pt": dict(V4_5_FONT_ROLES),
        "hierarchy": "panel label > major title > axis/subplot title > tick/legend > annotation",
        "only_panel_labels_bold": True,
        "scope": "V4_5 process-local typography hierarchy; V4_4 data, mapping, and panel geometry preserved",
    })
    return record


def activate_v4_5_font_roles(cfg: dict) -> None:
    manuscript.FONT_ROLE_SIZES = dict(V4_5_FONT_ROLES)
    manuscript.FONT_SIZES = {
        "size_panel_label": 11.0, "size_subplot_title": 8.5,
        "size_axis_label": 8.5, "size_tick_label": 7.8,
        "size_legend": 7.8, "size_annotation": 7.0,
    }
    constants = {
        "SIZE_PANEL_LABEL": 11.0, "SIZE_SUBPLOT_TITLE": 8.5,
        "SIZE_AXIS_LABEL": 8.5, "SIZE_TICK_LABEL": 7.8,
        "SIZE_LEGEND": 7.8, "SIZE_ANNOTATION": 7.0,
    }
    for module in (manuscript, v4.project_style):
        for name, value in constants.items():
            setattr(module, name, value)
        module.FONT_ROLE_SIZES = manuscript.FONT_ROLE_SIZES
        module.FONT_SIZES = manuscript.FONT_SIZES
    cfg.setdefault("figure_style", {})["font_sizes"] = {
        "body": 7.8, "title": 8.5, "axis": 8.5, "tick": 7.8,
        "legend": 7.8, "panel": 11.0, "block": 9.5, "annotation": 7.0,
    }
    v4.apply_style(cfg)


def _geometry_v4_5(layout):
    width, height, rects = v4_4._geometry_v4_4(layout)
    rects = {label: list(values) for label, values in rects.items()}
    trim = float(layout["v4_5_geometry"]["canvas_height_trim_mm"])
    compaction = float(layout["v4_5_geometry"]["b_to_cd_compaction_mm"])
    if abs(trim - compaction) > 1e-9:
        raise ValueError("V4_5 geometry requires equal b-to-c/d compaction and canvas trim")
    height -= trim
    for label in ("a", "b"):
        rects[label][1] -= compaction
    return width, height, rects


def _create_canvas_v4_5(layout):
    previous = v4._geometry
    v4._geometry = _geometry_v4_5
    try:
        return v4.create_canvas(layout)
    finally:
        v4._geometry = previous


def _create_standalone_canvas_v4_5(layout, label):
    previous = v4._geometry
    v4._geometry = _geometry_v4_5
    try:
        return v4.create_standalone_canvas(layout, label)
    finally:
        v4._geometry = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=HERE / "publication_layout_unified_v4_5.yaml")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    rid = base.publication_timestamp(args.run_id)
    layout_path = Path(args.layout).resolve()
    cfg = load_config(args.config)
    style_record = apply_v4_5_style_contract(cfg)
    ensure_output_dirs()
    layout = base.load_layout(layout_path)
    ctx = base.make_context(args, cfg, layout, rid)
    v3_anchor = v4._v3_7_baseline_anchor()
    v4_4_anchor = _v4_4_baseline_anchor()
    results_before = v4._tree_state(RESULTS_DIR)
    out = FIGURES_DIR / "Assembled" / f"MixedResolution_unified_v4_5_{rid}"
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v4_5_{rid}.json"
    targets = [out.with_suffix(ext) for ext in (".svg", ".pdf", ".png")] + [manifest_path]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"Refusing to overwrite existing V4_5 artifacts for {rid}")

    fig, axes, containers, rects, width, height, shared_axes, cbar_parent = _create_canvas_v4_5(layout)
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
    activate_v4_5_font_roles(cfg)
    role_qa = apply_v4_5_typography(fig)
    typography_qa = manuscript.enforce_figure_typography(fig, font_family=manuscript.FONT_FAMILY)
    center_panel_d_headers(axes["d"], shared_axes["d"])
    panel_meta["e"].update(record_panel_e_v4_3_tick_settings(axes["e"], strict=True))
    panel_meta["b"].update(measure_panel_b_legend_clearance(axes["b"], strict=True))
    panel_meta["d"].update(measure_panel_d_header_alignment(axes["d"], shared_axes["d"], strict=True))
    panel_meta["e"].update({
        "v4_5_matrix_annotation_count": role_qa["matrix_annotation_count"],
        "v4_5_matrix_min_contrast_ratio": role_qa["matrix_annotation_min_contrast_ratio"],
        "v4_5_matrix_annotation_colors": role_qa["matrix_annotation_colors"],
    })
    spacing_qa = measure_major_content_gaps(fig, axes, shared_axes, cbar_parent, strict=True)
    frame_qa = base.enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    model_artist_qa = v4._model_artist_qa(fig, cfg)
    geometry_qa = v4._geometry_qa(fig, containers, rects)
    alignment_qa = v4._panel_cd_alignment_qa(fig, shared_axes, cfg["figure_style"]["paper_dpi"])
    if not model_artist_qa["passed"] or not geometry_qa["passed"] or not alignment_qa["passed"]:
        raise ValueError("V4_5 model, geometry, or c/d alignment contract failed")
    if not role_qa["only_panel_labels_bold"] or role_qa["local_size_override_count"]:
        raise ValueError(f"V4_5 typography role contract failed: {role_qa}")
    qa = {
        "geometric_axis_count": geometric_axis_count,
        "typography_qa": typography_qa,
        "typography_role_qa": role_qa,
        "frame_lineweight_qa": frame_qa,
        "model_artist_qa": model_artist_qa,
        "geometry_qa": geometry_qa,
        "major_content_spacing_qa": spacing_qa,
        "text_overflow_in": dict(manuscript.validate_text_within_canvas(fig)),
        "panel_text_clearance_qa": validate_panel_text_boundaries(fig, containers),
    }
    outputs = save_figure(fig, out, cfg, formats=("svg", "pdf", "png"), dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None)
    plt.close(fig)
    results_after = v4._tree_state(RESULTS_DIR)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during V4_5 rendering")

    manifest = v4.build_manifest(
        ctx, cfg, layout, layout_path, outputs, panel_meta, rects, width, height,
        qa, alignment_qa, style_record, args, rid, v3_anchor, results_before, results_after,
    )
    manifest.update({
        "workflow_label": "mixed_resolution_unified_v4_5",
        "schema_version": "4.5", "revision": "V4_5",
        "source_visual_revision": "V4_4",
        "v4_4_baseline_anchor": v4_4_anchor,
        "renderer": v4._record(Path(__file__).resolve()),
        "panel_renderer": v4._record(HERE / "common" / "publication_panels_unified_v4_5.py"),
        "exporter": v4._record(HERE / "134_export_unified_v4_5_panels.py"),
        "author_authorized_deltas": {
            "underlying_source_arrays_changed": False,
            "v4_5": {
                "scientific_inventory_delta": False, "raw_data_delta": False,
                "data_mapping_delta": False, "font_hierarchy_only": True,
                "b_to_cd_compaction_mm": float(layout["v4_5_geometry"]["b_to_cd_compaction_mm"]),
            },
        },
    })
    manifest["layout"].update({
        "b_to_cd_compaction_mm": float(layout["v4_5_geometry"]["b_to_cd_compaction_mm"]),
        "v4_5_canvas_height_trim_mm": float(layout["v4_5_geometry"]["canvas_height_trim_mm"]),
        "major_vertical_gaps_mm": spacing_qa["visible_content_gaps_mm"],
        "major_vertical_gap_mm": min(spacing_qa["visible_content_gaps_mm"].values()),
    })
    manifest["figure_contract"].update({
        "revision_scope": "V4_5 typography hierarchy and excess b-to-c/d whitespace compaction only; V4_4 display and scientific state unchanged",
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

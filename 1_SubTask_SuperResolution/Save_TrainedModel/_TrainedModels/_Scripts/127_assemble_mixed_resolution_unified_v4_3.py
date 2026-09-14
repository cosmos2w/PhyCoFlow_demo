#!/usr/bin/env python
"""Assemble the frozen-data mixed-resolution Figure V4_3 refinement."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import global_style as manuscript
import matplotlib.pyplot as plt
from common.config import (
    FIGURES_DIR,
    RESULTS_DIR,
    add_common_args,
    ensure_output_dirs,
    load_config,
)
from common.figure_style import save_figure
from common.io_utils import write_json
from common.physical_figure_layout import validate_panel_text_boundaries
from common.publication_panels_unified_v4_3 import (
    center_panel_d_headers,
    draw_panel,
    measure_panel_b_legend_clearance,
    measure_panel_d_header_alignment,
    panel_label,
)

HERE = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4_1 = _load("mixed_resolution_v4_1_base_for_v4_3", HERE / "121_assemble_mixed_resolution_unified_v4_1.py")
v4 = v4_1.v4
base = v4_1.base
V4_2_BASELINE = (
    HERE.parents[2] / "figures" / "generated" / "art_style_review"
    / "MixedResolution_unified_v4_2_20260914_2200"
)


def _v4_2_baseline_anchor() -> dict:
    paths = {
        "release_pdf": V4_2_BASELINE / "MixedResolution_unified_v4_2_20260914_2200.pdf",
        "release_svg": V4_2_BASELINE / "MixedResolution_unified_v4_2_20260914_2200.svg",
        "release_png": V4_2_BASELINE / "MixedResolution_unified_v4_2_20260914_2200.png",
        "release_manifest": V4_2_BASELINE / "source_manifest_v4_2.json",
        "release_qa": V4_2_BASELINE / "qa_v4_2.json",
        "renderer": HERE / "124_assemble_mixed_resolution_unified_v4_2.py",
        "exporter": HERE / "125_export_unified_v4_2_panels.py",
        "audit": HERE / "126_audit_mixed_resolution_v4_2.py",
        "panel_renderer": HERE / "common" / "publication_panels_unified_v4_2.py",
        "layout": HERE / "publication_layout_unified_v4_2.yaml",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V4_2 baseline anchor is incomplete: {missing}")
    return {key: v4._record(path) for key, path in paths.items()}


def _geometry_v4_3(layout):
    """Use uniform contract-minimum major gaps and an expanded panel-d gutter."""
    cfg = layout["v4_geometry"]
    width = float(cfg["canvas_width_mm"])
    height = float(cfg["canvas_height_mm"])
    heights = {key: float(value) for key, value in cfg["row_heights_mm"].items()}
    gaps = {key: float(value) for key, value in cfg["vertical_gaps_mm"].items()}
    c_width = float(cfg["cd_widths_mm"]["c"])
    spacer = float(cfg["cd_widths_mm"]["spacer"])
    d_width = float(cfg["cd_widths_mm"]["d"])
    if abs(c_width + spacer + d_width - width) > 1e-9:
        raise ValueError("V4_3 c/spacer/d widths do not close to the canvas width")
    total = sum(heights.values()) + sum(gaps.values())
    if abs(total - height) > 1e-9:
        raise ValueError(f"V4_3 row heights and gaps close to {total}, not {height} mm")
    e_bottom = 0.0
    cd_bottom = heights["e"] + gaps["cd_e"]
    b_bottom = cd_bottom + heights["cd"] + gaps["b_cd"]
    a_bottom = b_bottom + heights["b"] + gaps["a_b"]
    return width, height, {
        "a": [0.0, a_bottom, width, heights["a"]],
        "b": [0.0, b_bottom, width, heights["b"]],
        "c": [0.0, cd_bottom, c_width, heights["cd"]],
        "d": [c_width + spacer, cd_bottom, d_width, heights["cd"]],
        "e": [0.0, e_bottom, width, heights["e"]],
    }


def _create_canvas_v4_3(layout):
    previous = v4._geometry
    v4._geometry = _geometry_v4_3
    try:
        return v4.create_canvas(layout)
    finally:
        v4._geometry = previous


def _create_standalone_canvas_v4_3(layout, label):
    previous = v4._geometry
    v4._geometry = _geometry_v4_3
    try:
        return v4.create_standalone_canvas(layout, label)
    finally:
        v4._geometry = previous


def _author_delta_contract(panel_meta: dict) -> dict:
    return {
        "authorization": "explicit user instructions in V4_1, V4_2, and V4_3 revision tasks",
        "underlying_source_arrays_changed": False,
        "panel_a": {
            "change": "same-size shared ROI translated right; boxes, connectors, and insets remain consistent",
            "before_shift_parent_fraction": [-0.150, 0.180],
            "after_shift_parent_fraction": [-0.050, 0.180],
            "same_native_field": True,
        },
        "panel_b": {
            "change": "five redundant DMF-Gen labels hidden and saved Mixed-HML sweep restored",
            "before_sweep_recipes": ["4_ZeroH_Balanced", "5_ZeroH_MRich"],
            "after_sweep_recipes": ["3_Mixed_HML", "4_ZeroH_Balanced", "5_ZeroH_MRich"],
            "saved_rows_only": True,
            "bar_values_and_confidence_intervals_unchanged": True,
        },
        "v4_3": {
            "scientific_inventory_delta": False,
            "raw_data_delta": False,
            "panel_d_display_derivation": "absolute value of the unchanged signed prediction-minus-truth component residual",
            "panel_d_residual_norm": "Normalize(vmin=0, vmax=unchanged V4_2 positive clipping limit)",
            "panel_a_identity_labels": ["L", "M", "H"],
            "scope": "authorized derived display mapping, annotations, and geometry only",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=HERE / "publication_layout_unified_v4_3.yaml")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    rid = base.publication_timestamp(args.run_id)
    layout_path = Path(args.layout).resolve()
    cfg = load_config(args.config)
    style_record = v4.apply_v4_style_contract(cfg)
    style_record["scope"] = "V4_3 process-local override; V4_2 and scientific sources untouched"
    ensure_output_dirs()
    layout = base.load_layout(layout_path)
    ctx = base.make_context(args, cfg, layout, rid)
    v3_anchor = v4._v3_7_baseline_anchor()
    v4_2_anchor = _v4_2_baseline_anchor()
    results_before = v4._tree_state(RESULTS_DIR)
    out = FIGURES_DIR / "Assembled" / f"MixedResolution_unified_v4_3_{rid}"
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v4_3_{rid}.json"
    targets = [out.with_suffix(ext) for ext in (".svg", ".pdf", ".png")] + [manifest_path]
    if any(path.exists() for path in targets):
        raise FileExistsError(f"Refusing to overwrite existing V4_3 artifacts for {rid}")

    fig, axes, containers, rects, width, height, shared_axes, cbar_parent = _create_canvas_v4_3(layout)
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
                item.set_gid("font-role:tick_label")
    geometric_axis_count = base.enforce_geometric_aspects(fig)
    fig.canvas.draw()
    center_panel_d_headers(axes["d"], shared_axes["d"])
    typography_qa = manuscript.enforce_figure_typography(fig, font_family=manuscript.FONT_FAMILY)
    center_panel_d_headers(axes["d"], shared_axes["d"])
    panel_meta["b"].update(measure_panel_b_legend_clearance(axes["b"], strict=True))
    panel_meta["d"].update(
        measure_panel_d_header_alignment(axes["d"], shared_axes["d"], strict=True)
    )
    frame_qa = base.enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    model_artist_qa = v4._model_artist_qa(fig, cfg)
    geometry_qa = v4._geometry_qa(fig, containers, rects)
    alignment_qa = v4._panel_cd_alignment_qa(fig, shared_axes, cfg["figure_style"]["paper_dpi"])
    if not model_artist_qa["passed"]:
        raise ValueError(f"V4_3 model artist contract failed: {model_artist_qa}")
    if not geometry_qa["passed"]:
        raise ValueError(f"V4_3 geometry failed: {geometry_qa}")
    if not alignment_qa["passed"]:
        raise ValueError(f"V4_3 shared c+d alignment failed: {alignment_qa}")
    qa = {
        "geometric_axis_count": geometric_axis_count,
        "typography_qa": typography_qa,
        "frame_lineweight_qa": frame_qa,
        "model_artist_qa": model_artist_qa,
        "geometry_qa": geometry_qa,
        "text_overflow_in": dict(manuscript.validate_text_within_canvas(fig)),
        "panel_text_clearance_qa": validate_panel_text_boundaries(fig, containers),
    }
    outputs = save_figure(
        fig, out, cfg, formats=("svg", "pdf", "png"),
        dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None,
    )
    plt.close(fig)
    results_after = v4._tree_state(RESULTS_DIR)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during V4_3 rendering")

    manifest = v4.build_manifest(
        ctx, cfg, layout, layout_path, outputs, panel_meta, rects, width, height,
        qa, alignment_qa, style_record, args, rid, v3_anchor, results_before, results_after,
    )
    manifest.update({
        "workflow_label": "mixed_resolution_unified_v4_3",
        "schema_version": "4.3",
        "revision": "V4_3",
        "source_visual_revision": "V4_2",
        "v4_2_baseline_anchor": v4_2_anchor,
        "author_authorized_deltas": _author_delta_contract(panel_meta),
        "renderer": v4._record(Path(__file__).resolve()),
        "panel_renderer": v4._record(HERE / "common" / "publication_panels_unified_v4_3.py"),
        "exporter": v4._record(HERE / "128_export_unified_v4_3_panels.py"),
    })
    manifest["layout"]["major_vertical_gaps_mm"] = {
        key: float(value) for key, value in layout["v4_geometry"]["vertical_gaps_mm"].items()
    }
    manifest["layout"]["major_vertical_gap_mm"] = min(
        manifest["layout"]["major_vertical_gaps_mm"].values()
    )
    manifest["figure_contract"].update({
        "core_conclusion": (
            "The saved results continue to show resolution-transfer behavior and DMF-Gen's "
            "relative fidelity; V4_3 gives the multi-scale component and error evidence distinct semantics."
        ),
        "revision_scope": (
            "V4_3 display correction using unchanged saved fields and metrics; V4_2 and its "
            "carried ROI, visible-annotation, and sweep-inventory exceptions remain recorded."
        ),
        "underlying_source_arrays_unchanged": True,
    })
    manifest["selection_contract"]["panel_b"]["sweep_recipes"] = list(layout["panel_b_v4"]["sweep_recipes"])
    manifest["selection_contract"]["panel_b"]["sweep_count"] = 3
    manifest["selection_contract"]["panel_d"]["semantic_colormaps"] = {
        "truth_component": layout["panel_d_v4"]["truth_component_cmap"],
        "residuals": layout["panel_d_v4"]["residual_cmap"],
    }
    manifest["selection_contract"]["panel_d"].update({
        "display_residual_definition": panel_meta["d"]["residual_display_definition"],
        "display_residual_vmin": 0.0,
        "display_residual_positive_limits_preserved_from_v4_2": True,
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

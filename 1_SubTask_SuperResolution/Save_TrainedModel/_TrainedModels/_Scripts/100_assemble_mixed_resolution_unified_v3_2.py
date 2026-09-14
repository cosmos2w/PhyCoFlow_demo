#!/usr/bin/env python
"""Assemble the additive four-panel mixed-resolution V3-2 hybrid figure."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import global_style as manuscript
from common.config import FIGURES_DIR, add_common_args, ensure_output_dirs, load_config
from common.figure_style import apply_style, finalize_colorbar_multiplier_alignment, save_figure, style_manifest
from common.io_utils import write_json
from common.physical_figure_layout import (
    create_composite_canvas,
    measure_axes_mm,
    resolve_physical_layout,
    validate_measured_geometry,
    validate_panel_text_boundaries,
)
from common.publication_panels_unified_v3_2 import draw_panel, panel_label


HERE = Path(__file__).resolve().parent
BASE_ASSEMBLER = HERE / "97_assemble_mixed_resolution_unified_v3.py"
_spec = importlib.util.spec_from_file_location("unified_v3_base_assembler", BASE_ASSEMBLER)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load V3 base assembler: {BASE_ASSEMBLER}")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)


def build_manifest(ctx, cfg, layout, layout_path, outputs, panel_meta, physical_layout, qa, args, rid):
    source_paths = sorted({Path(path) for meta in panel_meta.values() for path in meta.get("sources", [])})
    cache_paths = sorted({Path(path) for meta in panel_meta.values() for path in meta.get("cache_sources", [])})
    v3_baseline = (
        FIGURES_DIR / "Assembled" / "FigureSourceManifest_unified_v3_20260913_1618.json"
    )
    return {
        "workflow_label": "mixed_resolution_unified_v3_2_hybrid",
        "schema_version": "3.2",
        "run_id": rid,
        "source_data_run_id": args.data_run_id,
        "multiscale_run_id": args.multiscale_run_id,
        "base_data_run_id": args.base_data_run_id,
        "figure_contract": {
            "core_conclusion": (
                "DMF-Gen remains strongest as H-resolution training information is removed, and in the "
                "decisive Zero-H-M-rich state its global advantage persists in local H-resolution structure "
                "and error, consistent with stronger intermediate/fine multiscale fidelity."
            ),
            "archetype": "asymmetric mixed-modality figure with image-led spatial proof",
            "backend": "Python/Matplotlib",
            "final_size_mm": [physical_layout.width_mm, physical_layout.height_mm],
            "panel_sequence": ["a", "b", "c", "d"],
            "hero_evidence": ["b", "c"],
            "image_integrity": (
                "Validated cache fields only; one globally defined ROI; common field/error normalization; "
                "no smoothing, sharpening, training, or inference."
            ),
        },
        "selection_contract": {
            "panel_a": "validated design protocol and exposure values",
            "panel_b_recipe_transfer": {
                "sensor_count": 512,
                "recipes": layout["panel_b_streamlined"]["recipes"],
            },
            "panel_b_sensor_sweeps": {
                "sensor_counts": layout["panel_b_streamlined"]["sensor_counts"],
                "recipes": layout["panel_b_streamlined"]["sweep_recipes"],
            },
            "panel_c": {
                "recipe": layout["panel_c_hybrid"]["recipe"],
                "sensor_count": layout["panel_c_hybrid"]["sensor_count"],
                "models": layout["panel_c_hybrid"]["models"],
                "rows": ["full_field", "zoomed_field", "local_absolute_error"],
            },
            "panel_d": {
                "qualitative_scales": ["intermediate", "fine"],
                "quantitative_scale": "fine",
                "recipes": layout["panel_d_streamlined"]["quantitative_recipes"],
            },
        },
        "layout": {**physical_layout.manifest(), **qa},
        "bottom_row_contract": {
            "panels": ["c", "d"],
            "width_fractions_excluding_gap": [0.60, 0.40],
            "horizontal_gap_mm": 1.5,
        },
        "style_contract": style_manifest(cfg),
        "configuration": base.record(Path(args.config)),
        "layout_configuration": base.record(layout_path),
        "renderer": base.record(Path(__file__).resolve()),
        "panel_renderer": base.record(HERE / "common" / "publication_panels_unified_v3_2.py"),
        "unified_v3_baseline_manifest": base.record(v3_baseline),
        "cache_manifest": base.record(ctx.cache_manifest_path),
        "representative_index": base.record(ctx.representatives_path),
        "csv_sources": [base.record(path) for path in source_paths],
        "cache_sources": [base.record(path) for path in cache_paths],
        "panels": panel_meta,
        "si_contract": {
            "Sx1": "complete five-recipe sensor sweeps",
            "Sx2": "expanded validated multi-recipe qualitative gallery",
            "Sx3": "complete large/intermediate/fine qualitative decomposition and residual maps",
            "Sx4": "complete three-scale pattern-correlation and variance-bias matrices",
            "tables": [
                "accuracy_512", "sensor_sweeps_64_512",
                "pattern_correlations_all_scales", "variance_allocation_bias_all_scales",
            ],
        },
        "outputs": [base.record(path) for path in outputs],
        "model_inference_performed": False,
        "validated_sources_modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=HERE / "publication_layout_unified_v3_2.yaml")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manuscript.register_local_arial()
    apply_style(cfg)
    ensure_output_dirs()
    layout = base.load_layout(args.layout)
    physical_layout = resolve_physical_layout(layout)
    rid = base.publication_timestamp(args.run_id)
    ctx = base.make_context(args, cfg, layout, rid)

    fig, axes, containers = create_composite_canvas(physical_layout)
    panel_meta = {}
    for label in layout["figure"]["panel_letters"]:
        panel_label(axes[label], label)
        panel_meta[label] = draw_panel(label, axes[label], ctx)
    geometric_axis_count = base.enforce_geometric_aspects(fig)
    typography_qa = manuscript.enforce_figure_typography(fig)
    finalize_colorbar_multiplier_alignment(fig)
    frame_qa = base.enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    model_line_qa = base.validate_v3_model_lines(fig, cfg, layout)
    if not model_line_qa["passed"]:
        raise ValueError(f"V3-2 model-line contract failed: {model_line_qa}")
    measured = measure_axes_mm(fig, containers)
    geometry_qa = validate_measured_geometry(physical_layout, measured)
    text_overflow = manuscript.validate_text_within_canvas(fig)
    panel_text_qa = validate_panel_text_boundaries(fig, containers)
    qa = {
        "geometric_axis_count": geometric_axis_count,
        "typography_qa": typography_qa,
        "frame_lineweight_qa": frame_qa,
        "model_line_qa": model_line_qa,
        "geometry_qa": geometry_qa,
        "text_overflow_in": dict(text_overflow),
        "panel_text_clearance_qa": panel_text_qa,
    }

    out = FIGURES_DIR / "Assembled" / f"MixedResolution_unified_v3_2_hybrid_{rid}"
    outputs = save_figure(
        fig, out, cfg, formats=("svg", "pdf", "png"),
        dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None,
    )
    plt.close(fig)
    manifest = build_manifest(
        ctx, cfg, layout, args.layout.resolve(), outputs, panel_meta,
        physical_layout, qa, args, rid,
    )
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v3_2_{rid}.json"
    write_json(manifest_path, manifest)
    print(f"[OK] {out}.pdf")
    print(f"[OK] {manifest_path}")


if __name__ == "__main__":
    main()

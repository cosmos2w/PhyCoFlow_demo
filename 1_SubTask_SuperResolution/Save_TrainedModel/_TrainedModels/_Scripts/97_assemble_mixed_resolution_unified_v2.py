#!/usr/bin/env python
"""Assemble the native, full-page a--f mixed-resolution unified-v2 figure."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import global_style as manuscript

from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config
from common.figure_style import (
    apply_style,
    condition_colors,
    finalize_colorbar_multiplier_alignment,
    model_colors,
    save_figure,
    style_manifest,
    validate_model_line_contract,
)
from common.io_utils import matching_or_latest, read_csv, write_json
from common.panel_c_tuning import apply_panel_c_tuning
from common.physical_figure_layout import (
    create_composite_canvas,
    measure_axes_mm,
    resolve_physical_layout,
    validate_measured_geometry,
    validate_panel_text_boundaries,
)
from common.publication_panels import PublicationContext
from common.publication_panels_unified_v2 import draw_panel, panel_label


def _load_layout(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _publication_timestamp(value=None):
    """Return the publication filename timestamp in YYYYMMDD_HHMM format."""
    if value is None:
        return datetime.now().strftime("%Y%m%d_%H%M")
    for pattern in ("%Y-%m-%d_%H-%M", "%Y%m%d_%H%M"):
        try:
            return datetime.strptime(value, pattern).strftime("%Y%m%d_%H%M")
        except ValueError:
            pass
    raise ValueError("--run-id must use YYYYMMDD_HHMM or YYYY-MM-DD_HH-MM")


def _enforce_geometric_aspects(fig):
    """Apply the final anti-distortion guard to every tagged field axis."""
    axes_by_id = {}

    def collect_axis(ax):
        axes_by_id[id(ax)] = ax
        for child in getattr(ax, "child_axes", []):
            collect_axis(child)

    def collect(container):
        for ax in getattr(container, "axes", []):
            collect_axis(ax)
        for subfigure in getattr(container, "subfigs", []):
            collect(subfigure)

    collect(fig)
    geometric_axes = [ax for ax in axes_by_id.values() if ax.get_gid() == "geometric-field"]
    for ax in geometric_axes:
        ax.set_aspect("equal", adjustable="box")
    return len(geometric_axes)


def _panel_c_overlap_audit(fig):
    """Detect panel-c header collisions in final display coordinates."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    header_gids = {"panel-c-model-header", "panel-c-recipe-header"}
    headers = []
    image_axes = []
    seen = set()

    def collect_axis(ax):
        if id(ax) in seen:
            return
        seen.add(id(ax))
        if ax.get_gid() == "panel-c-image-cell" and ax.get_visible():
            image_axes.append(ax)
        for child in getattr(ax, "child_axes", []):
            collect_axis(child)

    def collect(container):
        headers.extend(
            text for text in getattr(container, "texts", [])
            if text.get_gid() in header_gids and text.get_visible()
        )
        for ax in getattr(container, "axes", []):
            collect_axis(ax)
        for subfigure in getattr(container, "subfigs", []):
            collect(subfigure)

    collect(fig)
    text_boxes = [(text, text.get_window_extent(renderer)) for text in headers]
    axis_boxes = [(ax, ax.get_window_extent(renderer)) for ax in image_axes]
    text_text = []
    for index, (left, left_box) in enumerate(text_boxes):
        for right, right_box in text_boxes[index + 1:]:
            if left_box.overlaps(right_box):
                text_text.append({
                    "left": left.get_text(), "left_role": left.get_gid(),
                    "right": right.get_text(), "right_role": right.get_gid(),
                })
    text_figure = []
    for text, text_box in text_boxes:
        for axis_index, (_, axis_box) in enumerate(axis_boxes):
            if text_box.overlaps(axis_box):
                text_figure.append({
                    "text": text.get_text(), "role": text.get_gid(),
                    "image_axis_index": axis_index,
                })
    return {
        "passed": not text_text and not text_figure,
        "header_count": len(headers),
        "image_axis_count": len(image_axes),
        "text_text_collisions": text_text,
        "text_figure_collisions": text_figure,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path, checksum=True):
    path = Path(path)
    return {
        "path": str(path.resolve()), "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256(path) if checksum and path.exists() else None,
    }


def _context(args, cfg, v2, rid):
    manifest = args.cache_manifest or RESULTS_DIR / "ReconstructionCache" / "ReconstructionCache_manifest_formal_20260712.csv"
    representatives = args.representatives or matching_or_latest(
        RESULTS_DIR / "CanonicalTestIndex", "RepresentativeSnapshots", args.base_data_run_id, "csv"
    )
    source_run_ids = {
        "ResolutionProtocol_fields": args.base_data_run_id,
        "ResolutionProtocol_budgets": args.base_data_run_id,
        "ResolutionProtocol_sensors": args.base_data_run_id,
        "FrequencyError_summary": args.base_data_run_id,
        "MultiscaleWavelet_summary": args.multiscale_run_id or args.data_run_id or rid,
        "MultiscaleWavelet_metadata": args.multiscale_run_id or args.data_run_id or rid,
    }
    ctx = PublicationContext(cfg, args.data_run_id or rid, Path(manifest), Path(representatives), source_run_ids)
    ctx.v2 = v2
    return ctx


def _source_manifest(ctx, panel_meta, outputs, cfg, v2, args, rid, canvas_plan, panel_geometry):
    data_manifest = RESULTS_DIR / "UnifiedPublicationV2" / f"UnifiedV2DataManifest_{ctx.run_id}.json"
    data_payload = json.loads(data_manifest.read_text(encoding="utf-8")) if data_manifest.exists() else {}
    source_paths = sorted({path for meta in panel_meta.values() for path in meta.get("sources", [])})
    cache_paths = sorted({path for meta in panel_meta.values() for path in meta.get("cache_sources", [])})
    canonical = RESULTS_DIR / "CanonicalTestIndex" / f"CanonicalTestIndex_{args.canonical_run_id}.csv"
    sensor_plan = RESULTS_DIR / "SensorPlans" / f"SensorPlan_{args.canonical_run_id}.csv"
    selected = panel_meta.get("c", {})
    protocol_budget_path = matching_or_latest(
        RESULTS_DIR / "ResolutionProtocol", "ResolutionProtocol_budgets", args.base_data_run_id, "csv"
    )
    protocol_budget = {row["recipe"]: row for row in read_csv(protocol_budget_path)}
    standalone_outputs = sorted({
        str(path.resolve())
        for root in (FIGURES_DIR / "PublicationPanels", FIGURES_DIR / "MultiscaleWavelet")
        for path in root.rglob(f"*_{rid}.*")
        if path.suffix in {".pdf", ".svg", ".png"}
    })
    return {
        "workflow_label": v2["workflow_label"],
        "run_id": rid,
        "data_run_id": ctx.run_id,
        "multiscale_run_id": args.multiscale_run_id or ctx.run_id,
        "base_data_run_id": args.base_data_run_id,
        "figure_contract": {
            "core_conclusion": "DMF-Gen retains scale-specific spatial structure with smaller signed residuals while preserving multiscale correlation and variance allocation across training recipes.",
            "archetype": "asymmetric mixed-modality full-page figure",
            "backend": "Python/Matplotlib native one-canvas composition with explicit millimetre rectangles",
            "final_size_mm": [canvas_plan["width_mm"], canvas_plan["height_mm"]],
            "image_integrity": "Audited physical cache fields only; no smoothing, sharpening, or model-specific normalization.",
        },
        "layout": {
            **canvas_plan["physical_layout"],
            "panel_sequence": v2["figure"]["panel_letters"],
            "panel_geometry": panel_geometry,
            "panel_e_width_fraction": float(v2["panel_e"]["width_fraction"]),
            "panel_f_width_fraction": float(v2["panel_f"]["width_fraction"]),
            "anti_distortion_policy": "identical physical parent rectangle in standalone and composite; equal-aspect field axes",
            "geometric_axis_count": canvas_plan["geometric_axis_count"],
            "geometry_qa": canvas_plan["geometry_qa"],
            "text_overflow_in": canvas_plan["text_overflow_in"],
            "panel_text_clearance_qa": canvas_plan["panel_text_clearance_qa"],
            "typography_qa": canvas_plan["typography_qa"],
            "model_line_qa": canvas_plan["model_line_qa"],
            "colorbar_multiplier_qa": canvas_plan["colorbar_multiplier_qa"],
            "panel_c_overlap_qa": canvas_plan["panel_c_overlap_qa"],
        },
        "recipe_registry": {
            key: {
                "label": cfg["recipes"][key]["label"],
                "resolved_publication_label": v2["recipes"]["short_labels"][key],
                "expected_ratio": cfg["recipes"][key]["expected_ratio"],
                "expected_case_fraction": cfg["recipes"][key]["expected_case_fraction"],
                "actual_ratio": protocol_budget.get(key, {}).get("actual_ratio"),
                "actual_case_fraction": protocol_budget.get(key, {}).get("actual_case_fraction"),
                "active_train_cases": protocol_budget.get(key, {}).get("active_train_cases"),
                "active_train_trajectories": protocol_budget.get(key, {}).get("active_train_trajectories"),
                "train_snapshots": protocol_budget.get(key, {}).get("train_snapshots"),
                "validation_status": protocol_budget.get(key, {}).get("status"),
            }
            for key in v2["recipes"]["order"]
        },
        "selected_models": [item["key"] for item in cfg["models"]],
        "selected_field": selected.get("field", panel_meta.get("a", {}).get("field")),
        "canonical_selection": {
            "snapshot_index": selected.get("snapshot"), "case_id": selected.get("case_id"),
            "time_index": selected.get("time_index"), "sensor_count": selected.get("sensor_count"),
            "roi": selected.get("roi"),
        },
        "sensor_plan": _record(sensor_plan),
        "canonical_test_index": _record(canonical),
        "cache_manifest": _record(ctx.cache_manifest_path),
        "cache_paths": [_record(path, checksum=False) for path in cache_paths],
        "csv_sources": [_record(path) for path in source_paths],
        "data_preparation_manifest": _record(data_manifest),
        "incremental_cache_fill_used": bool(data_payload.get("incremental_cache_fill_used", False)),
        "supplemental_cache_manifest": data_payload.get("supplemental_cache_manifest"),
        "model_palette": model_colors(cfg),
        "recipe_palette": condition_colors(cfg),
        "style_contract": style_manifest(cfg),
        "panels": panel_meta,
        "panels_cdef_contract": {
            "panel_c": {
                "models": panel_meta.get("c", {}).get("models"),
                "recipes": panel_meta.get("c", {}).get("recipes"),
                "roi": panel_meta.get("c", {}).get("roi"),
                "field_limits": panel_meta.get("c", {}).get("field_limits"),
                "error_limits": panel_meta.get("c", {}).get("error_limits"),
                "error_map_definition": panel_meta.get("c", {}).get("error_map_definition"),
                "relative_l2_annotation_format": panel_meta.get("c", {}).get("relative_l2_annotation_format"),
            },
            "panel_d": {
                "role": "sensor_efficiency",
                "sensor_counts": panel_meta.get("d", {}).get("sensor_counts"),
                "formal_setting": panel_meta.get("d", {}).get("formal_setting"),
                "statistic": panel_meta.get("d", {}).get("statistic"),
            },
            "panel_e": {
                "role": "scale_specific_truth_structure_and_signed_residuals",
                "representative_snapshot": panel_meta.get("e", {}).get("representative_snapshot"),
                "representative_baseline": panel_meta.get("e", {}).get("representative_baseline"),
                "recipe": panel_meta.get("e", {}).get("recipe"),
                "displayed_models": panel_meta.get("e", {}).get("displayed_models"),
                "scale_groups": panel_meta.get("e", {}).get("scale_groups"),
                "scale_group_labels": panel_meta.get("e", {}).get("scale_group_labels"),
                "gt_rms": panel_meta.get("e", {}).get("gt_rms"),
                "component_definition": panel_meta.get("e", {}).get("component_definition"),
                "residual_definition": panel_meta.get("e", {}).get("residual_definition"),
                "display_units": panel_meta.get("e", {}).get("display_units"),
                "residual_sign": panel_meta.get("e", {}).get("residual_sign"),
                "component_color_limits": panel_meta.get("e", {}).get("component_color_limits"),
                "residual_color_limits": panel_meta.get("e", {}).get("residual_color_limits"),
                "residual_norm": panel_meta.get("e", {}).get("residual_norm"),
                "colorbar_count": panel_meta.get("e", {}).get("colorbar_count"),
                "residual_colorbar_height_fraction": panel_meta.get("e", {}).get("residual_colorbar_height_fraction"),
                "residual_colorbar_bounds_parent": panel_meta.get("e", {}).get("residual_colorbar_bounds_parent"),
                "component_cmap": panel_meta.get("e", {}).get("component_cmap"),
                "residual_cmap": panel_meta.get("e", {}).get("residual_cmap"),
                "clipping_percentiles": {
                    "component": panel_meta.get("e", {}).get("component_percentile"),
                    "residual": panel_meta.get("e", {}).get("residual_percentile"),
                },
                "wavelet": panel_meta.get("e", {}).get("wavelet"),
                "boundary_mode": panel_meta.get("e", {}).get("boundary_mode"),
                "colorbar_tick_formatter": panel_meta.get("e", {}).get("colorbar_tick_formatter"),
                "width_fraction": panel_meta.get("e", {}).get("width_fraction"),
            },
            "panel_f": {
                "role": "quantitative_scale_resolved_fidelity_heatmaps",
                "recipes": panel_meta.get("f", {}).get("recipes"),
                "models": panel_meta.get("f", {}).get("models"),
                "metrics": panel_meta.get("f", {}).get("metrics"),
                "statistic": panel_meta.get("f", {}).get("statistic"),
                "heatmap_source_csv": panel_meta.get("f", {}).get("heatmap_source_csv"),
                "correlation_color_limits": panel_meta.get("f", {}).get("correlation_color_limits"),
                "variance_bias_color_limits": panel_meta.get("f", {}).get("variance_bias_color_limits"),
                "standalone_heatmap_titles": panel_meta.get("f", {}).get("standalone_heatmap_titles"),
                "colorbar_titles": panel_meta.get("f", {}).get("colorbar_titles"),
                "colorbar_tick_formatter": panel_meta.get("f", {}).get("colorbar_tick_formatter"),
                "width_fraction": panel_meta.get("f", {}).get("width_fraction"),
            },
            "new_source_artifacts": sorted(
                set(panel_meta.get("e", {}).get("sources", []))
                | set(panel_meta.get("f", {}).get("sources", []))
            ),
            "output_paths": [str(Path(path).resolve()) for path in outputs],
            "standalone_output_paths": standalone_outputs,
        },
        "missing_entries": {label: meta.get("missing", []) for label, meta in panel_meta.items()},
        "rendering_modes": {
            "a": panel_meta.get("a", {}).get("rendering_mode"),
            "c": "nearest native H cells with vector contours; rasterized dense fields",
            "e": "nearest native H raw truth components and raw signed residuals with row-specific linear contrast and five vector contours",
            "f": "median summary values in two annotated heatmaps",
        },
        "axis_scales": {
            "b": panel_meta.get("b", {}).get("axis_scale"),
            "d": panel_meta.get("d", {}).get("axis_scale"),
            "f": {
                "pattern_correlation": "fixed heatmap normalization [0, 1]",
                "variance_fraction_bias_pp": "symmetric zero-centered robust heatmap normalization",
            },
        },
        "color_limits": {
            "c_field": panel_meta.get("c", {}).get("field_limits"),
            "c_error": panel_meta.get("c", {}).get("error_limits"),
            "e_component": panel_meta.get("e", {}).get("component_color_limits"),
            "e_residual": panel_meta.get("e", {}).get("residual_color_limits"),
            "f_correlation": panel_meta.get("f", {}).get("correlation_color_limits"),
            "f_variance_bias": panel_meta.get("f", {}).get("variance_bias_color_limits"),
        },
        "configuration": _record(cfg["_config_path"]),
        "layout_configuration": _record(args.layout),
        "outputs": [str(Path(path).resolve()) for path in outputs],
        "standalone_outputs": standalone_outputs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", default=str(Path(__file__).with_name("publication_layout_unified_v2.yaml")))
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id")
    parser.add_argument("--multiscale-run-id")
    parser.add_argument("--base-data-run-id", default="2026-07-13_14-55")
    parser.add_argument("--canonical-run-id", default="formal_20260712")
    parser.add_argument("--qualitative-version", type=int, choices=[1, 2])
    parser.add_argument("--width-mm", type=float, help="Override the configured master-canvas width.")
    parser.add_argument("--height-mm", type=float, help="Override automatic master-canvas height planning.")
    args = parser.parse_args()
    cfg = load_config(args.config); apply_style(cfg); ensure_output_dirs()
    # Shared Panel C tuning API: edit common/panel_c_tuning.py, not this runner.
    v2 = apply_panel_c_tuning(_load_layout(args.layout))
    rid = _publication_timestamp(args.run_id)
    ctx = _context(args, cfg, v2, rid)
    figure_cfg = v2["figure"]
    physical_layout = resolve_physical_layout(v2, width_override_mm=args.width_mm)
    if args.height_mm is not None and abs(float(args.height_mm) - physical_layout.height_mm) > 0.02:
        raise ValueError(
            "Composite height is derived from exact panel heights and gutters; "
            f"resolved={physical_layout.height_mm:.6f} mm, requested={args.height_mm:.6f} mm."
        )
    width_mm, height_mm = physical_layout.width_mm, physical_layout.height_mm
    fig, axes, containers = create_composite_canvas(physical_layout)
    meta = {}
    panel_label_y_offsets = figure_cfg.get("composite_panel_label_y_offsets", {})
    for label, ax in axes.items():
        panel_label(
            containers[label], label, cfg,
            x=manuscript.PANEL_LABEL_X,
            y=manuscript.PANEL_LABEL_Y + float(panel_label_y_offsets.get(label, 0.0)),
            ha=manuscript.PANEL_LABEL_HA, va=manuscript.PANEL_LABEL_VA,
        )
        version = int(args.qualitative_version or v2["panel_c"]["default_version"]) if label == "c" else None
        meta[label] = draw_panel(label, ax, ctx, standalone=False, show_legend=True, version=version)
    typography_qa = manuscript.enforce_figure_typography(fig)
    model_line_qa = validate_model_line_contract(fig, cfg)
    geometric_axis_count = _enforce_geometric_aspects(fig)
    colorbar_multiplier_qa = finalize_colorbar_multiplier_alignment(fig)
    measured_geometry = measure_axes_mm(fig, containers)
    geometry_qa = validate_measured_geometry(physical_layout, measured_geometry)
    text_overflow = manuscript.validate_text_within_canvas(fig)
    panel_text_clearance_qa = validate_panel_text_boundaries(fig, containers)
    panel_c_overlap_qa = _panel_c_overlap_audit(fig)
    if not panel_c_overlap_qa["passed"]:
        raise ValueError(f"Panel C text overlap detected: {panel_c_overlap_qa}")
    panel_geometry = measured_geometry
    canvas_plan = {
        "width_mm": width_mm, "height_mm": height_mm,
        "physical_layout": physical_layout.manifest(),
        "geometric_axis_count": geometric_axis_count,
        "geometry_qa": geometry_qa,
        "text_overflow_in": dict(text_overflow),
        "panel_text_clearance_qa": panel_text_clearance_qa,
        "typography_qa": typography_qa,
        "model_line_qa": model_line_qa,
        "colorbar_multiplier_qa": colorbar_multiplier_qa,
        "panel_c_overlap_qa": panel_c_overlap_qa,
    }
    out = FIGURES_DIR / "Assembled" / f"{figure_cfg['output_name']}_{rid}"
    outputs = save_figure(
        fig, out, cfg, formats=cfg["figure_style"]["paper_formats"],
        dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None,
    )
    plt.close(fig)
    manifest = _source_manifest(ctx, meta, outputs, cfg, v2, args, rid, canvas_plan, panel_geometry)
    source_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v2_{rid}.json"
    write_json(source_path, manifest)
    print(f"[OK] {out}")
    print(f"[OK] {source_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Assemble the additive five-panel mixed-resolution V3-6 figure."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import global_style as manuscript
import matplotlib.pyplot as plt
from common.config import FIGURES_DIR, add_common_args, ensure_output_dirs, load_config
from common.figure_style import apply_style, model_colors, save_figure, style_manifest
from common.io_utils import write_json
from common.physical_figure_layout import (
    measure_axes_mm,
    validate_panel_text_boundaries,
)
from common.publication_panels_unified_v3_6 import draw_panel, panel_label
from matplotlib.colors import to_hex

HERE = Path(__file__).resolve().parent
BASE_ASSEMBLER = HERE / "109_assemble_mixed_resolution_unified_v3_5.py"
_spec = importlib.util.spec_from_file_location("unified_v3_5_base_assembler", BASE_ASSEMBLER)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load V3-5 base assembler: {BASE_ASSEMBLER}")
base35 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base35)
base = base35.base


def _geometry(layout):
    cfg = layout["v3_6_geometry"]
    width = float(cfg["canvas_width_mm"]); height = float(cfg["canvas_height_mm"])
    vgap = float(cfg["vertical_gap_mm"])
    heights = {key: float(value) for key, value in cfg["row_heights_mm"].items()}
    c_width = float(cfg["cd_widths_mm"]["c"])
    spacer = float(cfg["cd_widths_mm"]["spacer"])
    d_width = float(cfg["cd_widths_mm"]["d"])
    if abs(c_width + spacer + d_width - width) > 1e-9:
        raise ValueError("V3-6 c/spacer/d widths do not close to the canvas width")
    if abs(heights["a"] + vgap + heights["b"] + vgap + heights["cd"] + vgap + heights["e"] - height) > 1e-9:
        raise ValueError("V3-6 row heights do not close to the canvas height")
    e_bottom = 0.0
    cd_bottom = heights["e"] + vgap
    b_bottom = cd_bottom + heights["cd"] + vgap
    a_bottom = b_bottom + heights["b"] + vgap
    return width, height, {
        "a": [0.0, a_bottom, width, heights["a"]],
        "b": [0.0, b_bottom, width, heights["b"]],
        "c": [0.0, cd_bottom, c_width, heights["cd"]],
        "d": [c_width + spacer, cd_bottom, d_width, heights["cd"]],
        "e": [0.0, e_bottom, width, heights["e"]],
    }


def create_canvas(layout):
    width, height, rects = _geometry(layout)
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), layout=None)
    containers, axes = {}, {}
    content = layout["v3_6_geometry"]["content_bounds"]
    for label, (left, bottom, panel_width, panel_height) in rects.items():
        container = fig.add_axes(
            [left / width, bottom / height, panel_width / width, panel_height / height],
            label=f"panel-{label}-container", frameon=False,
        )
        container.set_axis_off(); containers[label] = container
        ax = container.inset_axes(list(map(float, content[label])), transform=container.transAxes)
        ax.set_label(f"panel-{label}-content"); axes[label] = ax
    shared_cfg = layout["v3_6_geometry"]["shared_cd_grid"]
    cd_bottom = rects["c"][1]
    grid_left = float(shared_cfg["left_mm"]) / width
    grid_right = float(shared_cfg["right_mm"]) / width
    grid_bottom_mm = cd_bottom + float(shared_cfg["bottom_colorbar_strip_mm"])
    grid_top_mm = cd_bottom + rects["c"][3] - float(shared_cfg["top_margin_mm"])
    visual_height = grid_top_mm - grid_bottom_mm
    row_gap = float(shared_cfg["row_gap_mm"])
    row_height = (visual_height - 2 * row_gap) / 3
    if row_height <= 0:
        raise ValueError("V3-6 shared c+d visual rows have non-positive height")
    shared_grid = fig.add_gridspec(
        3, 9, left=grid_left, right=grid_right,
        bottom=grid_bottom_mm / height, top=grid_top_mm / height,
        height_ratios=[1, 1, 1],
        width_ratios=[1, 1, 1, 1, 1, float(shared_cfg["spacer_ratio"]),
                      *([float(shared_cfg["d_column_ratio"])] * 3)],
        hspace=row_gap / row_height,
        wspace=float(shared_cfg["column_gap_fraction"]),
    )
    shared_axes = {
        "c": [[fig.add_subplot(shared_grid[row, column], label=f"panel-c-r{row}-c{column}")
               for column in range(5)] for row in range(3)],
        "d": [[fig.add_subplot(shared_grid[row, column], label=f"panel-d-r{row}-c{column - 6}")
               for column in range(6, 9)] for row in range(3)],
    }
    cbar_parent = containers["c"].inset_axes(
        [0.0, 0.0, 1.0, float(shared_cfg["bottom_colorbar_strip_mm"]) / rects["c"][3]],
        transform=containers["c"].transAxes,
    )
    cbar_parent.set_axis_off(); cbar_parent.set_label("panel-c-colorbar-strip")
    return fig, axes, containers, rects, width, height, shared_axes, cbar_parent


def create_standalone_canvas(layout, label):
    _width, _height, rects = _geometry(layout)
    panel_width, panel_height = rects[label][2], rects[label][3]
    fig = plt.figure(figsize=(panel_width / 25.4, panel_height / 25.4), layout=None)
    container = fig.add_axes([0, 0, 1, 1], label=f"panel-{label}-container", frameon=False)
    container.set_axis_off()
    ax = container.inset_axes(
        list(map(float, layout["v3_6_geometry"]["content_bounds"][label])),
        transform=container.transAxes,
    )
    ax.set_label(f"panel-{label}-content")
    return fig, ax, container


def _geometry_qa(fig, containers, expected):
    measured = measure_axes_mm(fig, containers); errors = []
    for label, (left, bottom, width, height) in expected.items():
        observed = measured[label]
        target = {"left_mm": left, "bottom_mm": bottom, "width_mm": width, "height_mm": height,
                  "right_mm": left + width, "top_mm": bottom + height}
        for key, value in target.items():
            if abs(float(observed[key]) - value) > .02:
                errors.append({"panel": label, "key": key, "expected": value, "observed": observed[key]})
    return {"passed": not errors, "tolerance_mm": .02, "errors": errors, "measured": measured}


def _panel_cd_alignment_qa(fig, shared_axes):
    """Measure exact rendered c/d row boundaries from the one shared GridSpec."""
    fig.canvas.draw()
    height_mm = float(fig.get_size_inches()[1] * 25.4)
    rows = []
    deltas = []
    for index, (c_row, d_row) in enumerate(zip(shared_axes["c"], shared_axes["d"])):
        c_box = c_row[0].get_position(); d_box = d_row[0].get_position()
        record = {
            "row": index,
            "c_bottom_mm": float(c_box.y0 * height_mm), "c_top_mm": float(c_box.y1 * height_mm),
            "d_bottom_mm": float(d_box.y0 * height_mm), "d_top_mm": float(d_box.y1 * height_mm),
            "c_height_mm": float(c_box.height * height_mm), "d_height_mm": float(d_box.height * height_mm),
        }
        record["bottom_delta_mm"] = abs(record["c_bottom_mm"] - record["d_bottom_mm"])
        record["top_delta_mm"] = abs(record["c_top_mm"] - record["d_top_mm"])
        record["height_delta_mm"] = abs(record["c_height_mm"] - record["d_height_mm"])
        deltas.extend([record["bottom_delta_mm"], record["top_delta_mm"], record["height_delta_mm"]])
        rows.append(record)
    exact_shared_boundaries = max(deltas) <= 1e-9
    c_boundaries = [[row["c_bottom_mm"], row["c_top_mm"]] for row in rows]
    d_boundaries = [[row["d_bottom_mm"], row["d_top_mm"]] for row in rows]
    return {
        "passed": exact_shared_boundaries,
        "exact_shared_boundaries": exact_shared_boundaries,
        "implementation": "one matplotlib GridSpec(3, 9) spanning c+d",
        "shared_parent_gridspec": True, "shared_cd_grid": True,
        "row_geometry_source": "shared_parent_gridspec",
        "row_order": ["full/large", "zoom/intermediate", "local-error/fine"],
        "rows": rows, "max_boundary_delta_mm": max(deltas),
        "v3_6_row_boundaries_mm": {"c": c_boundaries, "d": d_boundaries},
    }


def _model_artist_qa(fig, cfg):
    colors = model_colors(cfg)
    checked, violations = [], []
    for line in fig.findobj(match=lambda item: isinstance(item, matplotlib.lines.Line2D)):
        gid = str(line.get_gid() or "")
        if not gid.startswith("model-line:"):
            continue
        model = gid.split(":", 1)[1]; observed = to_hex(line.get_color()).lower()
        expected = colors[model].lower(); checked.append({"model": model, "color": observed})
        if observed != expected:
            violations.append({"model": model, "expected": expected, "observed": observed})
    for patch in fig.findobj(match=lambda item: isinstance(item, matplotlib.patches.Rectangle)):
        gid = str(patch.get_gid() or "")
        if not gid.startswith("model-bar:"):
            continue
        model = gid.split(":", 1)[1]; observed = to_hex(patch.get_facecolor()).lower()
        expected = colors[model].lower(); checked.append({"model": model, "color": observed, "kind": "bar"})
        if observed != expected:
            violations.append({"model": model, "expected": expected, "observed": observed})
    return {"passed": bool(checked) and not violations, "checked_count": len(checked),
            "checked": checked, "violations": violations}


def _v3_5_anchor():
    """Verify the exact local V3-5 stack before any V3-6 build is attempted."""
    manifest_path = FIGURES_DIR / "Assembled" / "FigureSourceManifest_unified_v3_5_20260914_1100.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    main_base = FIGURES_DIR / "Assembled" / "MixedResolution_unified_v3_5_20260914_1100"
    wrapper = HERE.parents[2] / "figures" / "scripts" / "render_mixed_resolution_v3_5.py"
    records = {
        "manifest": base.record(manifest_path),
        "main_pdf": base.record(main_base.with_suffix(".pdf")),
        "main_svg": base.record(main_base.with_suffix(".svg")),
        "main_png": base.record(main_base.with_suffix(".png")),
        "assembler": base.record(HERE / "109_assemble_mixed_resolution_unified_v3_5.py"),
        "exporter": base.record(HERE / "110_export_unified_v3_5_panels.py"),
        "audit": base.record(HERE / "111_audit_unified_v3_5.py"),
        "wrapper": base.record(wrapper),
        "panel_renderer": base.record(HERE / "common" / "publication_panels_unified_v3_5.py"),
        "layout": base.record(HERE / "publication_layout_unified_v3_5.yaml"),
    }
    orchestration = manifest["orchestration"]
    expected_pdf = next(item for item in manifest["outputs"] if item["path"].endswith(".pdf"))
    expected_svg = next(item for item in manifest["outputs"] if item["path"].endswith(".svg"))
    expected_png = next(item for item in manifest["outputs"] if item["path"].endswith(".png"))
    matches = (
        records["assembler"]["sha256"] == manifest["renderer"]["sha256"]
        and records["panel_renderer"]["sha256"] == manifest["panel_renderer"]["sha256"]
        and records["layout"]["sha256"] == manifest["layout_configuration"]["sha256"]
        and records["exporter"]["sha256"] == orchestration["panel_exporter"]["sha256"]
        and records["audit"]["sha256"] == orchestration["audit"]["sha256"]
        and records["wrapper"]["sha256"] == orchestration["wrapper"]["sha256"]
        and records["main_pdf"]["sha256"] == expected_pdf["sha256"]
        and records["main_svg"]["sha256"] == expected_svg["sha256"]
        and records["main_png"]["sha256"] == expected_png["sha256"]
    )
    if not matches:
        raise RuntimeError("Local V3-5 renderer stack or baseline no longer matches the 20260914_1100 manifest")
    return records


def build_manifest(ctx, cfg, layout, layout_path, outputs, panel_meta, rects, width, height,
                   qa, alignment_qa, args, rid):
    source_paths = sorted({Path(path) for meta in panel_meta.values() for path in meta.get("sources", [])})
    cache_paths = sorted({Path(path) for meta in panel_meta.values() for path in meta.get("cache_sources", [])})
    v2_pdf = FIGURES_DIR / "Assembled" / "MixedResolution_unified_v2_phase2_20260903_2213.pdf"
    v2_manifest = FIGURES_DIR / "Assembled" / "FigureSourceManifest_unified_v2_20260903_2213.json"
    v3_3_manifest = FIGURES_DIR / "Assembled" / "FigureSourceManifest_unified_v3_3_20260913_2352.json"
    v3_3_base = FIGURES_DIR / "Assembled" / "MixedResolution_unified_v3_3_20260913_2352"
    v3_4_manifest = FIGURES_DIR / "Assembled" / "FigureSourceManifest_unified_v3_4_20260914_1040.json"
    v3_4_base = FIGURES_DIR / "Assembled" / "MixedResolution_unified_v3_4_20260914_1040"
    v3_5_manifest = FIGURES_DIR / "Assembled" / "FigureSourceManifest_unified_v3_5_20260914_1100.json"
    v3_5_base = FIGURES_DIR / "Assembled" / "MixedResolution_unified_v3_5_20260914_1100"
    v3_5_release = HERE.parents[2] / "figures" / "generated" / "MixedResolution_unified_v3_5_20260914_1100"
    requested_style_reference = Path(
        "/mnt/data/ghostwriter_images/context/b6f81bed-6b96-5470-8555-f947734f6ea7.png"
    )
    retained_style_proxy = (
        FIGURES_DIR / "PublicationPanels" / "Panel_A"
        / "Panel_a_ResolutionProtocol_20260806_1124.png"
    )
    return {
        "workflow_label": "mixed_resolution_unified_v3_6",
        "schema_version": "3.6", "run_id": rid,
        "additive_release": True, "release_mode": "additive",
        "source_data_run_id": args.data_run_id, "multiscale_run_id": args.multiscale_run_id,
        "base_data_run_id": args.base_data_run_id,
        "figure_contract": {
            "core_conclusion": (
                "V3-6 preserves the validated mixed-resolution scientific conclusion: L/M/H encode distinct "
                "discretizations and DMF-Gen retains the strongest H-resolution fidelity as H fields are removed."
            ),
            "archetype": "asymmetric mixed-modality figure with image-led physical proof",
            "backend": "Python/Matplotlib", "final_size_mm": [width, height],
            "panel_sequence": ["a", "b", "c", "d", "e"],
            "hero_evidence": ["b", "c", "e"],
            "revision_scope": "micro-adjustments to aspect ratios, whitespace, and marker styling only; validated scientific content and selection are unchanged",
            "removed_redundant_main_figure_evidence": [
                "fine-scale pattern-correlation line chart",
                "fine-scale variance-allocation-bias line chart",
            ],
            "image_integrity": (
                "Validated cache fields only; shared state/ROI/color limits; no smoothing, sharpening, "
                "training, inference, or metric recomputation."
            ),
        },
        "selection_contract": {
            "panel_a": "validated L/M/H protocol fields and recipe budgets",
            "panel_b": {"sensor_count": 512, "recipes": layout["panel_b_v3_6"]["recipes"],
                        "sweep_recipes": layout["panel_b_v3_6"]["sweep_recipes"]},
            "panel_c": {"recipe": layout["panel_c_v3_6"]["recipe"], "sensor_count": 512,
                        "models": layout["panel_c_v3_6"]["models"],
                        "rows": ["full_field", "zoomed_field", "local_absolute_error"]},
            "panel_d": {"scales": ["large", "intermediate", "fine"],
                        "qualitative_only": True, "quantitative_evidence_panel": "e"},
            "panel_e": {"metrics": ["pattern_correlation", "variance_fraction_bias_pp"],
                        "matrix_shape": [4, 9],
                        "matrix_arrangement": "side_by_side_metric_blocks",
                        "colorbar_arrangement": "dual_horizontal_bottom_band"},
        },
        "cross_panel_alignment_qa": alignment_qa,
        "panel_cd_rows_aligned": alignment_qa["passed"],
        "shared_cd_parent_gridspec": True, "shared_cd_grid": True,
        "exact_cd_row_boundaries": alignment_qa["passed"],
        "c_d_rows_exactly_aligned": alignment_qa["passed"],
        "layout": {
            "canvas_width_mm": width, "canvas_height_mm": height,
            "panel_rectangles_mm": {
                label: {"left_mm": values[0], "bottom_mm": values[1], "width_mm": values[2],
                        "height_mm": values[3], "right_mm": values[0] + values[2],
                        "top_mm": values[1] + values[3]} for label, values in rects.items()
            },
            "row_spanning_panel": "e", "left_lower_stack": ["c", "d"], **qa,
        },
        "style_contract": style_manifest(cfg),
        "configuration": base.record(Path(args.config)), "layout_configuration": base.record(layout_path),
        "renderer": base.record(Path(__file__).resolve()),
        "panel_renderer": base.record(HERE / "common" / "publication_panels_unified_v3_6.py"),
        "v3_5_starting_renderer_anchor": _v3_5_anchor(),
        "v3_5_baseline": {
            "manifest": base.record(v3_5_manifest),
            "main_pdf": base.record(v3_5_base.with_suffix(".pdf")),
            "main_svg": base.record(v3_5_base.with_suffix(".svg")),
            "main_png": base.record(v3_5_base.with_suffix(".png")),
        },
        "v3_5_hash_anchor": [
            base.record(v3_5_release / "MixedResolution_unified_v3_5_20260914_1100.pdf"),
            base.record(v3_5_release / "MixedResolution_unified_v3_5_20260914_1100.svg"),
            base.record(v3_5_release / "MixedResolution_unified_v3_5_20260914_1100.png"),
            base.record(v3_5_release / "source_manifest_v3_5.json"),
            base.record(v3_5_release / "qa_v3_5.json"),
        ],
        "v3_4_baseline": {
            "manifest": base.record(v3_4_manifest),
            "main_pdf": base.record(v3_4_base.with_suffix(".pdf")),
            "main_svg": base.record(v3_4_base.with_suffix(".svg")),
            "main_png": base.record(v3_4_base.with_suffix(".png")),
        },
        "v3_4_baseline_manifest": base.record(v3_4_manifest),
        "v3_4_baseline_pdf": base.record(v3_4_base.with_suffix(".pdf")),
        "v3_4_baseline_svg": base.record(v3_4_base.with_suffix(".svg")),
        "v3_4_baseline_png": base.record(v3_4_base.with_suffix(".png")),
        "v3_3_baseline": {
            "manifest": base.record(v3_3_manifest),
            "main_pdf": base.record(v3_3_base.with_suffix(".pdf")),
            "main_svg": base.record(v3_3_base.with_suffix(".svg")),
            "main_png": base.record(v3_3_base.with_suffix(".png")),
        },
        "old_v2_baseline_pdf": base.record(v2_pdf), "old_v2_baseline_manifest": base.record(v2_manifest),
        "panel_a_style_reference": {
            "requested_attachment": {
                "path": str(requested_style_reference),
                "available_during_build": requested_style_reference.exists(),
            },
            "retained_old_style_proxy": base.record(retained_style_proxy),
            "implementation_basis": (
                "The V3-5 nested-inset treatment is retained as the starting visual language; V3-6 changes "
                "geometry and spacing only while preserving the validated field data."
            ),
        },
        "cache_manifest": base.record(ctx.cache_manifest_path),
        "representative_index": base.record(ctx.representatives_path),
        "csv_sources": [base.record(path) for path in source_paths],
        "cache_sources": [base.record(path) for path in cache_paths],
        "panels": panel_meta,
        "si_contract": {
            "Sx1": "complete five-recipe sensor-count sweeps",
            "Sx2": "validated multi-recipe qualitative gallery",
            "Sx3": "complete three-scale qualitative evidence",
            "Sx4": "extended three-scale quantitative matrices/distributions",
            "tables": ["accuracy_512", "sensor_sweeps_64_512",
                       "pattern_correlations_all_scales", "variance_allocation_bias_all_scales"],
        },
        "outputs": [base.record(path) for path in outputs],
        "model_inference_performed": False, "validated_sources_modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=HERE / "publication_layout_unified_v3_6.yaml")
    parser.add_argument("--cache-manifest", type=Path); parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    cfg = load_config(args.config); manuscript.register_local_arial(); apply_style(cfg); ensure_output_dirs()
    layout = base.load_layout(args.layout); rid = base.publication_timestamp(args.run_id)
    ctx = base.make_context(args, cfg, layout, rid)
    _v3_5_anchor()
    fig, axes, containers, rects, width, height, shared_axes, cbar_parent = create_canvas(layout)
    panel_meta = {}
    for label in ("a", "b"):
        panel_label(axes[label], label); panel_meta[label] = draw_panel(label, axes[label], ctx)
    panel_label(axes["c"], "c")
    panel_meta["c"] = draw_panel("c", axes["c"], ctx,
                                 shared_axes=shared_axes["c"], colorbar_parent=cbar_parent)
    panel_label(axes["d"], "d")
    panel_meta["d"] = draw_panel("d", axes["d"], ctx, shared_axes=shared_axes["d"])
    panel_label(axes["e"], "e"); panel_meta["e"] = draw_panel("e", axes["e"], ctx)
    geometric_axis_count = base.enforce_geometric_aspects(fig)
    typography_qa = manuscript.enforce_figure_typography(fig)
    frame_qa = base.enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    model_artist_qa = _model_artist_qa(fig, cfg)
    if not model_artist_qa["passed"]:
        raise ValueError(f"V3-6 model artist contract failed: {model_artist_qa}")
    geometry_qa = _geometry_qa(fig, containers, rects)
    if not geometry_qa["passed"]:
        raise ValueError(f"V3-6 geometry failed: {geometry_qa}")
    alignment_qa = _panel_cd_alignment_qa(fig, shared_axes)
    if not alignment_qa["passed"]:
        raise ValueError(f"V3-6 shared c+d row alignment failed: {alignment_qa}")
    text_overflow = manuscript.validate_text_within_canvas(fig)
    panel_text_qa = validate_panel_text_boundaries(fig, containers)
    qa = {"geometric_axis_count": geometric_axis_count, "typography_qa": typography_qa,
          "frame_lineweight_qa": frame_qa, "model_artist_qa": model_artist_qa,
          "geometry_qa": geometry_qa, "text_overflow_in": dict(text_overflow),
          "panel_text_clearance_qa": panel_text_qa}
    out = FIGURES_DIR / "Assembled" / f"MixedResolution_unified_v3_6_{rid}"
    outputs = save_figure(fig, out, cfg, formats=("svg", "pdf", "png"),
                          dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None)
    plt.close(fig)
    manifest = build_manifest(ctx, cfg, layout, args.layout.resolve(), outputs, panel_meta,
                              rects, width, height, qa, alignment_qa, args, rid)
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v3_6_{rid}.json"
    write_json(manifest_path, manifest)
    print(f"[OK] {out}.pdf"); print(f"[OK] {manifest_path}")


if __name__ == "__main__":
    main()

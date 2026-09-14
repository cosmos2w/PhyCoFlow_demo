#!/usr/bin/env python
"""Assemble the additive five-panel mixed-resolution V3-3 figure."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt

import global_style as manuscript
from common.config import FIGURES_DIR, add_common_args, ensure_output_dirs, load_config
from common.figure_style import apply_style, model_colors, save_figure, style_manifest
from common.io_utils import write_json
from common.physical_figure_layout import measure_axes_mm, validate_panel_text_boundaries
from common.publication_panels_unified_v3_3 import draw_panel, panel_label


HERE = Path(__file__).resolve().parent
BASE_ASSEMBLER = HERE / "100_assemble_mixed_resolution_unified_v3_2.py"
_spec = importlib.util.spec_from_file_location("unified_v3_2_base_assembler", BASE_ASSEMBLER)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot load V3-2 base assembler: {BASE_ASSEMBLER}")
base32 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base32)
base = base32.base


def _geometry(layout):
    cfg = layout["v3_3_geometry"]
    width = float(cfg["canvas_width_mm"]); height = float(cfg["canvas_height_mm"])
    hgap = float(cfg["horizontal_gap_mm"]); vgap = float(cfg["vertical_gap_mm"])
    heights = {key: float(value) for key, value in cfg["row_heights_mm"].items()}
    left = float(cfg["lower_widths_mm"]["left"]); right = float(cfg["lower_widths_mm"]["right"])
    if abs(left + hgap + right - width) > 1e-9:
        raise ValueError("V3-3 lower widths do not close to the canvas width")
    if abs(heights["a"] + vgap + heights["b"] + vgap + heights["c"] + vgap + heights["e"] - height) > 1e-9:
        raise ValueError("V3-3 row heights do not close to the canvas height")
    e_bottom = 0.0
    c_bottom = heights["e"] + vgap
    b_bottom = c_bottom + heights["c"] + vgap
    a_bottom = b_bottom + heights["b"] + vgap
    return width, height, {
        "a": [0.0, a_bottom, width, heights["a"]],
        "b": [0.0, b_bottom, width, heights["b"]],
        "c": [0.0, c_bottom, left, heights["c"]],
        "d": [left + hgap, e_bottom, right, heights["c"] + vgap + heights["e"]],
        "e": [0.0, e_bottom, left, heights["e"]],
    }


def create_canvas(layout):
    width, height, rects = _geometry(layout)
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), layout=None)
    containers, axes = {}, {}
    content = layout["v3_3_geometry"]["content_bounds"]
    for label, (left, bottom, panel_width, panel_height) in rects.items():
        container = fig.add_axes(
            [left / width, bottom / height, panel_width / width, panel_height / height],
            label=f"panel-{label}-container", frameon=False,
        )
        container.set_axis_off(); containers[label] = container
        ax = container.inset_axes(list(map(float, content[label])), transform=container.transAxes)
        ax.set_label(f"panel-{label}-content"); axes[label] = ax
    return fig, axes, containers, rects, width, height


def create_standalone_canvas(layout, label):
    _width, _height, rects = _geometry(layout)
    panel_width, panel_height = rects[label][2], rects[label][3]
    fig = plt.figure(figsize=(panel_width / 25.4, panel_height / 25.4), layout=None)
    container = fig.add_axes([0, 0, 1, 1], label=f"panel-{label}-container", frameon=False)
    container.set_axis_off()
    ax = container.inset_axes(
        list(map(float, layout["v3_3_geometry"]["content_bounds"][label])),
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


def _v3_2_anchor():
    manifest_path = FIGURES_DIR / "Assembled" / "FigureSourceManifest_unified_v3_2_20260913_1739.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = {
        "manifest": base.record(manifest_path),
        "main_pdf": base.record(FIGURES_DIR / "Assembled" / "MixedResolution_unified_v3_2_hybrid_20260913_1739.pdf"),
        "renderer": base.record(HERE / "100_assemble_mixed_resolution_unified_v3_2.py"),
        "panel_renderer": base.record(HERE / "common" / "publication_panels_unified_v3_2.py"),
        "layout": base.record(HERE / "publication_layout_unified_v3_2.yaml"),
    }
    matches = (
        records["renderer"]["sha256"] == manifest["renderer"]["sha256"]
        and records["panel_renderer"]["sha256"] == manifest["panel_renderer"]["sha256"]
        and records["layout"]["sha256"] == manifest["layout_configuration"]["sha256"]
    )
    if not matches:
        raise RuntimeError("Local V3-2 renderer stack no longer matches the 20260913_1739 source manifest")
    return records


def build_manifest(ctx, cfg, layout, layout_path, outputs, panel_meta, rects, width, height, qa, args, rid):
    source_paths = sorted({Path(path) for meta in panel_meta.values() for path in meta.get("sources", [])})
    cache_paths = sorted({Path(path) for meta in panel_meta.values() for path in meta.get("cache_sources", [])})
    v2_pdf = FIGURES_DIR / "Assembled" / "MixedResolution_unified_v2_phase2_20260903_2213.pdf"
    v2_manifest = FIGURES_DIR / "Assembled" / "FigureSourceManifest_unified_v2_20260903_2213.json"
    return {
        "workflow_label": "mixed_resolution_unified_v3_3",
        "schema_version": "3.3", "run_id": rid,
        "source_data_run_id": args.data_run_id, "multiscale_run_id": args.multiscale_run_id,
        "base_data_run_id": args.base_data_run_id,
        "figure_contract": {
            "core_conclusion": (
                "L/M/H encode distinct discretizations and, as H-resolution training fields are removed, "
                "DMF-Gen retains the strongest aggregate, local, and scale-resolved H-resolution fidelity."
            ),
            "archetype": "asymmetric mixed-modality figure with image-led physical proof",
            "backend": "Python/Matplotlib", "final_size_mm": [width, height],
            "panel_sequence": ["a", "b", "c", "d", "e"],
            "hero_evidence": ["b", "c"],
            "image_integrity": (
                "Validated cache fields only; shared state/ROI/color limits; no smoothing, sharpening, "
                "training, inference, or metric recomputation."
            ),
        },
        "selection_contract": {
            "panel_a": "validated L/M/H protocol fields and recipe budgets",
            "panel_b": {"sensor_count": 512, "recipes": layout["panel_b_v3_3"]["recipes"],
                        "sweep_recipes": layout["panel_b_v3_3"]["sweep_recipes"]},
            "panel_c": {"recipe": layout["panel_c_v3_3"]["recipe"], "sensor_count": 512,
                        "models": layout["panel_c_v3_3"]["models"],
                        "rows": ["full_field", "zoomed_field", "local_absolute_error"]},
            "panel_d": {"scales": ["large", "intermediate", "fine"], "quantitative_scale": "fine"},
            "panel_e": {"metrics": ["pattern_correlation", "variance_fraction_bias_pp"],
                        "matrix_shape": [4, 9]},
        },
        "layout": {
            "canvas_width_mm": width, "canvas_height_mm": height,
            "panel_rectangles_mm": {
                label: {"left_mm": values[0], "bottom_mm": values[1], "width_mm": values[2],
                        "height_mm": values[3], "right_mm": values[0] + values[2],
                        "top_mm": values[1] + values[3]} for label, values in rects.items()
            },
            "row_spanning_panel": "d", "left_lower_stack": ["c", "e"], **qa,
        },
        "style_contract": style_manifest(cfg),
        "configuration": base.record(Path(args.config)), "layout_configuration": base.record(layout_path),
        "renderer": base.record(Path(__file__).resolve()),
        "panel_renderer": base.record(HERE / "common" / "publication_panels_unified_v3_3.py"),
        "v3_2_starting_renderer_anchor": _v3_2_anchor(),
        "old_v2_baseline_pdf": base.record(v2_pdf), "old_v2_baseline_manifest": base.record(v2_manifest),
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
    parser.add_argument("--layout", type=Path, default=HERE / "publication_layout_unified_v3_3.yaml")
    parser.add_argument("--cache-manifest", type=Path); parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    cfg = load_config(args.config); manuscript.register_local_arial(); apply_style(cfg); ensure_output_dirs()
    layout = base.load_layout(args.layout); rid = base.publication_timestamp(args.run_id)
    ctx = base.make_context(args, cfg, layout, rid)
    _v3_2_anchor()
    fig, axes, containers, rects, width, height = create_canvas(layout)
    panel_meta = {}
    for label in layout["figure"]["panel_letters"]:
        panel_label(axes[label], label); panel_meta[label] = draw_panel(label, axes[label], ctx)
    geometric_axis_count = base.enforce_geometric_aspects(fig)
    typography_qa = manuscript.enforce_figure_typography(fig)
    frame_qa = base.enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    model_artist_qa = _model_artist_qa(fig, cfg)
    if not model_artist_qa["passed"]:
        raise ValueError(f"V3-3 model artist contract failed: {model_artist_qa}")
    geometry_qa = _geometry_qa(fig, containers, rects)
    if not geometry_qa["passed"]:
        raise ValueError(f"V3-3 geometry failed: {geometry_qa}")
    text_overflow = manuscript.validate_text_within_canvas(fig)
    panel_text_qa = validate_panel_text_boundaries(fig, containers)
    qa = {"geometric_axis_count": geometric_axis_count, "typography_qa": typography_qa,
          "frame_lineweight_qa": frame_qa, "model_artist_qa": model_artist_qa,
          "geometry_qa": geometry_qa, "text_overflow_in": dict(text_overflow),
          "panel_text_clearance_qa": panel_text_qa}
    out = FIGURES_DIR / "Assembled" / f"MixedResolution_unified_v3_3_{rid}"
    outputs = save_figure(fig, out, cfg, formats=("svg", "pdf", "png"),
                          dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None)
    plt.close(fig)
    manifest = build_manifest(ctx, cfg, layout, args.layout.resolve(), outputs, panel_meta,
                              rects, width, height, qa, args, rid)
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v3_3_{rid}.json"
    write_json(manifest_path, manifest)
    print(f"[OK] {out}.pdf"); print(f"[OK] {manifest_path}")


if __name__ == "__main__":
    main()

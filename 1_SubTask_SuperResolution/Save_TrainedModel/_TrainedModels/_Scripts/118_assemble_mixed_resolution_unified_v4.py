#!/usr/bin/env python
"""Assemble the additive mixed-resolution Figure V4 art-style release.

V4 is deliberately isolated from the V3-7 implementation.  It imports the
validated V3-7 context/utility layer, uses the V4-local layout and panel
wrapper, and writes new files only.  No training, inference, source-data
rewrite, metric recomputation, or V3-7 overwrite is performed.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import global_style as manuscript
from common import figure_style as project_style
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config
from common.figure_style import apply_style, model_colors, save_figure, style_manifest
from common.io_utils import write_json
from common.physical_figure_layout import measure_axes_mm, validate_panel_text_boundaries
from common.publication_panels_unified_v4 import draw_panel, panel_label

HERE = Path(__file__).resolve().parent
BASE_ASSEMBLER = HERE / "115_assemble_mixed_resolution_unified_v3_7.py"
_SPEC = importlib.util.spec_from_file_location("unified_v3_7_base_assembler", BASE_ASSEMBLER)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Cannot load V3-7 base assembler: {BASE_ASSEMBLER}")
base37 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base37)
base = base37.base


V4_MODEL_COLORS = {
    "DMF-Gen": "#C94053",
    "FFM-Perceiver": "#4C86A6",
    "FFM-FNO": "#425B76",
    "SiT": "#9B83C1",
    "Latent FM": "#725591",
    "Geo-FNO": "#DA9A66",
    "Senseiver": "#8D9BAD",
    "MLP-RBF": "#4C9E91",
    "Truth": "#252525",
}
V4_FONT_ROLES = {
    "panel_label": 10.0,
    "subplot_title": 8.5,
    "axis_label": 8.5,
    "tick_label": 8.0,
    "legend": 8.0,
    "annotation": 7.8,
}


def apply_v4_style_contract(cfg: dict) -> dict:
    """Apply the scoped V4 palette and type hierarchy to this process only."""
    # Register the repository user's local Arial faces explicitly.  A fresh
    # Matplotlib process can otherwise retain a stale font cache and silently
    # fall back, changing text metrics during the fixed-canvas audit.
    registered_fonts = manuscript.register_local_arial()
    manuscript.MODEL_COLORS = dict(V4_MODEL_COLORS)
    manuscript.FONT_ROLE_SIZES = dict(V4_FONT_ROLES)
    manuscript.FONT_SIZES = {
        "size_panel_label": V4_FONT_ROLES["panel_label"],
        "size_subplot_title": V4_FONT_ROLES["subplot_title"],
        "size_axis_label": V4_FONT_ROLES["axis_label"],
        "size_tick_label": V4_FONT_ROLES["tick_label"],
        "size_legend": V4_FONT_ROLES["legend"],
        "size_annotation": V4_FONT_ROLES["annotation"],
    }
    # figure_style re-exports these constants at import time.  Keep its
    # compatibility API synchronized without editing the shared V3 module.
    for name, value in (
        ("SIZE_PANEL_LABEL", 10.0),
        ("SIZE_SUBPLOT_TITLE", 8.5),
        ("SIZE_AXIS_LABEL", 8.5),
        ("SIZE_TICK_LABEL", 8.0),
        ("SIZE_LEGEND", 8.0),
        ("SIZE_ANNOTATION", 7.8),
    ):
        setattr(project_style, name, value)
    project_style.FONT_ROLE_SIZES = manuscript.FONT_ROLE_SIZES
    for model in cfg.get("models", []):
        label = str(model.get("label") or model.get("key") or "")
        if label in V4_MODEL_COLORS:
            model["color"] = V4_MODEL_COLORS[label]
    style = cfg.setdefault("figure_style", {})
    style["font_sizes"] = {
        "body": 8.0,
        "title": 8.5,
        "axis": 8.5,
        "tick": 8.0,
        "legend": 8.0,
        "panel": 10.0,
        "block": 9.0,
        "annotation": 7.8,
    }
    # This calls manuscript.apply_global_style(), which now sees the V4-local
    # constants.  The mutation is process-scoped and never written to YAML.
    apply_style(cfg)
    return {
        "font_family_declared": manuscript.FONT_FAMILY,
        "registered_font_files": registered_fonts,
        "arial_resolved": manuscript.arial_available(),
        "font_role_sizes_pt": dict(V4_FONT_ROLES),
        "model_palette": dict(V4_MODEL_COLORS),
        "model_alpha": {
            "DMF-Gen": manuscript.ALPHA_DMF_GEN,
            "baseline_default": manuscript.ALPHA_BASELINE,
        },
        "marker_order": ["o", "D", ">", "+"],
        "scope": "V4 process-local override; V3-7 global source untouched",
    }


def _geometry(layout):
    cfg = layout["v4_geometry"]
    width = float(cfg["canvas_width_mm"])
    height = float(cfg["canvas_height_mm"])
    vgap = float(cfg["vertical_gap_mm"])
    heights = {key: float(value) for key, value in cfg["row_heights_mm"].items()}
    c_width = float(cfg["cd_widths_mm"]["c"])
    spacer = float(cfg["cd_widths_mm"]["spacer"])
    d_width = float(cfg["cd_widths_mm"]["d"])
    if abs(c_width + spacer + d_width - width) > 1e-9:
        raise ValueError("V4 c/spacer/d widths do not close to the canvas width")
    if abs(heights["a"] + vgap + heights["b"] + vgap + heights["cd"]
           + vgap + heights["e"] - height) > 1e-9:
        raise ValueError("V4 row heights do not close to the canvas height")
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
    content = layout["v4_geometry"]["content_bounds"]
    for label, (left, bottom, panel_width, panel_height) in rects.items():
        container = fig.add_axes(
            [left / width, bottom / height, panel_width / width, panel_height / height],
            label=f"panel-{label}-container", frameon=False,
        )
        container.set_axis_off()
        containers[label] = container
        ax = container.inset_axes(list(map(float, content[label])), transform=container.transAxes)
        ax.set_label(f"panel-{label}-content")
        axes[label] = ax

    shared_cfg = layout["v4_geometry"]["shared_cd_grid"]
    cd_bottom = rects["c"][1]
    grid_left = float(shared_cfg["left_mm"]) / width
    grid_right = float(shared_cfg["right_mm"]) / width
    grid_bottom_mm = cd_bottom + float(shared_cfg["bottom_colorbar_strip_mm"])
    grid_top_mm = cd_bottom + rects["c"][3] - float(shared_cfg["top_margin_mm"])
    visual_height = grid_top_mm - grid_bottom_mm
    row_gap = float(shared_cfg["row_gap_mm"])
    row_height = (visual_height - 2 * row_gap) / 3
    if row_height <= 0:
        raise ValueError("V4 shared c+d visual rows have non-positive height")
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
    cbar_parent.set_axis_off()
    cbar_parent.set_label("panel-c-colorbar-strip")
    return fig, axes, containers, rects, width, height, shared_axes, cbar_parent


def create_standalone_canvas(layout, label):
    _width, _height, rects = _geometry(layout)
    panel_width, panel_height = rects[label][2], rects[label][3]
    fig = plt.figure(figsize=(panel_width / 25.4, panel_height / 25.4), layout=None)
    container = fig.add_axes([0, 0, 1, 1], label=f"panel-{label}-container", frameon=False)
    container.set_axis_off()
    standalone = layout["v4_geometry"].get("standalone_content_bounds", {})
    bounds = standalone.get(label, layout["v4_geometry"]["content_bounds"][label])
    ax = container.inset_axes(
        list(map(float, bounds)),
        transform=container.transAxes,
    )
    ax.set_label(f"panel-{label}-content")
    return fig, ax, container


def _geometry_qa(fig, containers, expected):
    measured = measure_axes_mm(fig, containers)
    errors = []
    for label, (left, bottom, width, height) in expected.items():
        observed = measured[label]
        target = {
            "left_mm": left, "bottom_mm": bottom, "width_mm": width,
            "height_mm": height, "right_mm": left + width, "top_mm": bottom + height,
        }
        for key, value in target.items():
            if abs(float(observed[key]) - value) > 0.02:
                errors.append({"panel": label, "key": key,
                               "expected": value, "observed": observed[key]})
    return {"passed": not errors, "tolerance_mm": 0.02,
            "errors": errors, "measured": measured}


def _panel_cd_alignment_qa(fig, shared_axes, export_dpi):
    fig.canvas.draw()
    height_mm = float(fig.get_size_inches()[1] * 25.4)
    export_height_px = round(float(fig.get_size_inches()[1]) * float(export_dpi))
    rows, deltas = [], []
    for index, (c_row, d_row) in enumerate(zip(shared_axes["c"], shared_axes["d"])):
        c_box, d_box = c_row[0].get_position(), d_row[0].get_position()
        record = {
            "row": index,
            "c_bottom_mm": float(c_box.y0 * height_mm),
            "c_top_mm": float(c_box.y1 * height_mm),
            "d_bottom_mm": float(d_box.y0 * height_mm),
            "d_top_mm": float(d_box.y1 * height_mm),
            "c_height_mm": float(c_box.height * height_mm),
            "d_height_mm": float(d_box.height * height_mm),
        }
        record["bottom_delta_mm"] = abs(record["c_bottom_mm"] - record["d_bottom_mm"])
        record["top_delta_mm"] = abs(record["c_top_mm"] - record["d_top_mm"])
        record["height_delta_mm"] = abs(record["c_height_mm"] - record["d_height_mm"])
        record["c_bottom_export_pixel"] = round(c_box.y0 * export_height_px)
        record["d_bottom_export_pixel"] = round(d_box.y0 * export_height_px)
        record["bottom_export_pixel_delta"] = abs(
            record["c_bottom_export_pixel"] - record["d_bottom_export_pixel"]
        )
        deltas.extend([record["bottom_delta_mm"], record["top_delta_mm"], record["height_delta_mm"]])
        rows.append(record)
    return {
        "passed": max(deltas) <= 1e-9 and all(item["bottom_export_pixel_delta"] == 0 for item in rows),
        "exact_shared_boundaries": max(deltas) <= 1e-9,
        "shared_parent_gridspec": True,
        "implementation": "one matplotlib GridSpec(3, 9) spanning c+d",
        "row_geometry_source": "shared_parent_gridspec",
        "row_order": ["full/large", "zoom/intermediate", "local-error/fine"],
        "rows": rows,
        "max_boundary_delta_mm": max(deltas),
        "export_dpi": int(export_dpi),
        "export_height_px": export_height_px,
    }


def _model_artist_qa(fig, cfg):
    colors = model_colors(cfg)
    checked, violations = [], []
    for line in fig.findobj(match=lambda item: isinstance(item, matplotlib.lines.Line2D)):
        gid = str(line.get_gid() or "")
        if not gid.startswith("model-line:"):
            continue
        model = gid.split(":", 1)[1]
        observed = to_hex(line.get_color()).lower()
        expected = colors[model].lower()
        checked.append({"model": model, "color": observed, "marker": line.get_marker()})
        if observed != expected:
            violations.append({"model": model, "expected": expected, "observed": observed})
    for patch in fig.findobj(match=lambda item: isinstance(item, matplotlib.patches.Rectangle)):
        gid = str(patch.get_gid() or "")
        if not gid.startswith("model-bar:"):
            continue
        model = gid.split(":", 1)[1]
        observed = to_hex(patch.get_facecolor()).lower()
        expected = colors[model].lower()
        checked.append({"model": model, "color": observed, "kind": "bar"})
        if observed != expected:
            violations.append({"model": model, "expected": expected, "observed": observed})
    return {"passed": bool(checked) and not violations,
            "checked_count": len(checked), "checked": checked, "violations": violations,
            "expected_palette": colors}


def _record(path: Path) -> dict:
    return base.record(Path(path))


def _tree_state(root: Path) -> dict:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _state_digest(state: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()


def _v3_7_baseline_anchor() -> dict:
    root = HERE.parents[2] / "figures" / "generated" / "MixedResolution_unified_v3_7_20260914_1158"
    assembled = FIGURES_DIR / "Assembled"
    paths = {
        "release_pdf": root / "MixedResolution_unified_v3_7_20260914_1158.pdf",
        "release_svg": root / "MixedResolution_unified_v3_7_20260914_1158.svg",
        "release_png": root / "MixedResolution_unified_v3_7_20260914_1158.png",
        "release_source_manifest": root / "source_manifest_v3_7.json",
        "release_qa": root / "qa_v3_7.json",
        "assembled_pdf": assembled / "MixedResolution_unified_v3_7_20260914_1158.pdf",
        "assembled_svg": assembled / "MixedResolution_unified_v3_7_20260914_1158.svg",
        "assembled_png": assembled / "MixedResolution_unified_v3_7_20260914_1158.png",
        "assembled_manifest": assembled / "FigureSourceManifest_unified_v3_7_20260914_1158.json",
        "renderer": HERE / "115_assemble_mixed_resolution_unified_v3_7.py",
        "exporter": HERE / "116_export_unified_v3_7_panels.py",
        "panel_renderer": HERE / "common" / "publication_panels_unified_v3_7.py",
        "layout": HERE / "publication_layout_unified_v3_7.yaml",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V3-7 baseline anchor is incomplete: {missing}")
    return {key: _record(path) for key, path in paths.items()}


def build_manifest(ctx, cfg, layout, layout_path, outputs, panel_meta,
                   rects, width, height, qa, alignment_qa, style_record,
                   args, rid, baseline, results_before, results_after):
    source_paths = sorted({Path(path) for meta in panel_meta.values()
                           for path in meta.get("sources", []) if Path(path).exists()})
    cache_paths = sorted({Path(path) for meta in panel_meta.values()
                          for path in meta.get("cache_sources", []) if Path(path).exists()})
    return {
        "workflow_label": "mixed_resolution_unified_v4",
        "schema_version": "4.0",
        "revision": "V4",
        "run_id": rid,
        "additive_release": True,
        "release_mode": "art_style_review",
        "source_data_run_id": args.data_run_id,
        "multiscale_run_id": args.multiscale_run_id,
        "base_data_run_id": args.base_data_run_id,
        "figure_contract": {
            "core_conclusion": (
                "The V3-7 scientific conclusion is unchanged: L/M/H are distinct spatial "
                "discretizations and DMF-Gen retains the strongest H-resolution fidelity as H fields are removed."
            ),
            "archetype": "asymmetric mixed-modality figure with image-led physical proof",
            "backend": "Python/Matplotlib",
            "final_size_mm": [width, height],
            "review_size_mm": {"design": [180.0, height], "manuscript_insertion_width": 162.0},
            "panel_sequence": ["a", "b", "c", "d", "e"],
            "hero_evidence": ["b", "c", "e"],
            "revision_scope": "art-style-only V4; no scientific story, source arrays, axes, norms, inventory, or annotations changed",
            "release_status": "art reviewed, scientific release pending",
            "pending_author_checks": ["A03", "A04"],
            "image_integrity": (
                "V3-7 validated caches and summary tables; no smoothing, sharpening, training, "
                "inference, metric recomputation, or scientific source substitution."
            ),
        },
        "selection_contract": {
            "panel_a": "V3-7 native 32x32, 64x64, 128x128 protocol fields and five recipe budgets",
            "panel_b": {"sensor_count": 512, "recipes": layout["panel_b_v4"]["recipes"],
                        "sweep_recipes": layout["panel_b_v4"]["sweep_recipes"]},
            "panel_c": {"recipe": layout["panel_c_v4"]["recipe"], "sensor_count": 512,
                        "models": layout["panel_c_v4"]["models"],
                        "rows": ["full_field", "zoomed_field", "local_absolute_error"]},
            "panel_d": {"scales": ["large", "intermediate", "fine"],
                        "qualitative_only": True, "quantitative_evidence_panel": "e"},
            "panel_e": {"metrics": ["pattern_correlation", "variance_fraction_bias_pp"],
                        "matrix_shape": [4, 9], "matrix_arrangement": "side_by_side_metric_blocks",
                        "colorbar_arrangement": "none"},
        },
        "layout": {
            "canvas_width_mm": width,
            "canvas_height_mm": height,
            "panel_rectangles_mm": {
                label: {"left_mm": values[0], "bottom_mm": values[1],
                        "width_mm": values[2], "height_mm": values[3],
                        "right_mm": values[0] + values[2],
                        "top_mm": values[1] + values[3]}
                for label, values in rects.items()
            },
            "major_vertical_gap_mm": float(layout["v4_geometry"]["vertical_gap_mm"]),
            "shared_cd_row_gap_mm": float(layout["v4_geometry"]["shared_cd_grid"]["row_gap_mm"]),
            "row_spanning_panel": "e",
            "left_lower_stack": ["c", "d"],
            **qa,
        },
        "cross_panel_alignment_qa": alignment_qa,
        "panel_cd_rows_aligned": alignment_qa["passed"],
        "shared_cd_parent_gridspec": True,
        "style_contract": {
            **style_manifest(cfg),
            "v4_override": style_record,
            "resolved_palette": dict(V4_MODEL_COLORS),
            "resolved_markers": {"DMF-Gen": "o", "FFM-Perceiver": "D",
                                 "Senseiver": ">", "MLP-RBF": "+"},
        },
        "configuration": _record(Path(args.config)),
        "layout_configuration": _record(layout_path),
        "renderer": _record(Path(__file__).resolve()),
        "panel_renderer": _record(HERE / "common" / "publication_panels_unified_v4.py"),
        "exporter": _record(HERE / "119_export_unified_v4_panels.py"),
        "v3_7_baseline_anchor": baseline,
        "source_data_records": [_record(path) for path in source_paths],
        "cache_source_records": [_record(path) for path in cache_paths],
        "panels": panel_meta,
        "outputs": [_record(path) for path in outputs],
        "results_tree_immutability": {
            "unchanged": results_before == results_after,
            "state_digest_before": _state_digest(results_before),
            "state_digest_after": _state_digest(results_after),
            "file_count_before": len(results_before),
            "file_count_after": len(results_after),
        },
        "model_inference_performed": False,
        "validated_sources_modified": False,
        "art_review_bundle": {
            "status": "art reviewed, scientific release pending",
            "required_external_qa": "LAYOUT_QA.json at 180 mm and 162 mm",
            "author_checks": ["A03", "A04"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=HERE / "publication_layout_unified_v4.yaml")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    rid = base.publication_timestamp(args.run_id)
    layout_path = Path(args.layout).resolve()
    cfg = load_config(args.config)
    style_record = apply_v4_style_contract(cfg)
    ensure_output_dirs()
    layout = base.load_layout(layout_path)
    ctx = base.make_context(args, cfg, layout, rid)
    baseline = _v3_7_baseline_anchor()
    results_before = _tree_state(RESULTS_DIR)
    out = FIGURES_DIR / "Assembled" / f"MixedResolution_unified_v4_{rid}"
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v4_{rid}.json"
    if any(path.exists() for path in [out.with_suffix(ext) for ext in (".svg", ".pdf", ".png")] + [manifest_path]):
        raise FileExistsError(f"Refusing to overwrite existing V4 artifacts for {rid}")

    fig, axes, containers, rects, width, height, shared_axes, cbar_parent = create_canvas(layout)
    panel_meta = {}
    for label in ("a", "b"):
        panel_label(axes[label], label)
        panel_meta[label] = draw_panel(label, axes[label], ctx)
    panel_label(axes["c"], "c")
    panel_meta["c"] = draw_panel("c", axes["c"], ctx,
                                 shared_axes=shared_axes["c"], colorbar_parent=cbar_parent)
    panel_label(axes["d"], "d")
    panel_meta["d"] = draw_panel("d", axes["d"], ctx, shared_axes=shared_axes["d"])
    panel_label(axes["e"], "e")
    panel_meta["e"] = draw_panel("e", axes["e"], ctx)

    # Assign vector-readable semantic roles after every panel has been drawn.
    # Identical strings occur as titles, axis labels, and categorical ticks;
    # the role tag keeps the typography audit deterministic without changing
    # any visible text or artist geometry.
    for axis in fig.axes:
        for item in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            if item.get_visible() and item.get_text():
                item.set_gid("font-role:tick_label")

    geometric_axis_count = base.enforce_geometric_aspects(fig)
    typography_qa = manuscript.enforce_figure_typography(fig, font_family=manuscript.FONT_FAMILY)
    frame_qa = base.enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    model_artist_qa = _model_artist_qa(fig, cfg)
    if not model_artist_qa["passed"]:
        raise ValueError(f"V4 model artist contract failed: {model_artist_qa}")
    geometry_qa = _geometry_qa(fig, containers, rects)
    if not geometry_qa["passed"]:
        raise ValueError(f"V4 geometry failed: {geometry_qa}")
    alignment_qa = _panel_cd_alignment_qa(fig, shared_axes, cfg["figure_style"]["paper_dpi"])
    if not alignment_qa["passed"]:
        raise ValueError(f"V4 shared c+d row alignment failed: {alignment_qa}")
    text_overflow = manuscript.validate_text_within_canvas(fig)
    panel_text_qa = validate_panel_text_boundaries(fig, containers)
    qa = {
        "geometric_axis_count": geometric_axis_count,
        "typography_qa": typography_qa,
        "frame_lineweight_qa": frame_qa,
        "model_artist_qa": model_artist_qa,
        "geometry_qa": geometry_qa,
        "text_overflow_in": dict(text_overflow),
        "panel_text_clearance_qa": panel_text_qa,
    }
    outputs = save_figure(fig, out, cfg, formats=("svg", "pdf", "png"),
                          dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None)
    plt.close(fig)
    results_after = _tree_state(RESULTS_DIR)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during the visualization-only V4 build")
    manifest = build_manifest(ctx, cfg, layout, layout_path, outputs, panel_meta,
                              rects, width, height, qa, alignment_qa, style_record,
                              args, rid, baseline, results_before, results_after)
    write_json(manifest_path, manifest)
    print(f"[OK] {out}.pdf")
    print(f"[OK] {manifest_path}")


if __name__ == "__main__":
    main()

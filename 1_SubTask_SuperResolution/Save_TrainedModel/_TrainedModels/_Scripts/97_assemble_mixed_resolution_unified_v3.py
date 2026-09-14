#!/usr/bin/env python
"""Assemble the additive four-panel mixed-resolution unified-v3 main figure."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
import yaml

import global_style as manuscript
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config
from common.figure_style import apply_style, finalize_colorbar_multiplier_alignment, save_figure, style_manifest
from common.io_utils import matching_or_latest, write_json
from common.physical_figure_layout import (
    create_composite_canvas,
    measure_axes_mm,
    resolve_physical_layout,
    validate_measured_geometry,
    validate_panel_text_boundaries,
)
from common.publication_panels import PublicationContext
from common.publication_panels_unified_v3 import draw_panel, panel_label


def _deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    merged = dict(base)
    for key, value in override.items():
        merged[key] = _deep_merge(merged[key], value) if key in merged else value
    return merged


def load_layout(path: Path):
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    parent = payload.pop("extends", None)
    if parent is None:
        return payload
    return _deep_merge(load_layout((path.resolve().parent / parent).resolve()), payload)


def publication_timestamp(value=None):
    if value is None:
        return datetime.now().strftime("%Y%m%d_%H%M")
    for pattern in ("%Y%m%d_%H%M", "%Y-%m-%d_%H-%M"):
        try:
            return datetime.strptime(value, pattern).strftime("%Y%m%d_%H%M")
        except ValueError:
            pass
    raise ValueError("--run-id must use YYYYMMDD_HHMM or YYYY-MM-DD_HH-MM")


def sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path, *, checksum=True):
    path = Path(path)
    return {
        "path": str(path.resolve()), "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
        "sha256": sha256(path) if checksum and path.exists() else None,
    }


def make_context(args, cfg, layout, rid):
    cache_manifest = args.cache_manifest or (
        RESULTS_DIR / "ReconstructionCache" / "ReconstructionCache_manifest_formal_20260712.csv"
    )
    representatives = args.representatives or matching_or_latest(
        RESULTS_DIR / "CanonicalTestIndex", "RepresentativeSnapshots", args.base_data_run_id, "csv",
    )
    source_run_ids = {
        "ResolutionProtocol_fields": args.base_data_run_id,
        "ResolutionProtocol_budgets": args.base_data_run_id,
        "ResolutionProtocol_sensors": args.base_data_run_id,
        "SensorSweepAllRecipes_summary": args.data_run_id,
        "MultiscaleWavelet_summary": args.multiscale_run_id,
        "MultiscaleWavelet_metadata": args.multiscale_run_id,
    }
    ctx = PublicationContext(
        cfg, args.data_run_id or rid, Path(cache_manifest), Path(representatives), source_run_ids,
    )
    ctx.v2 = layout
    return ctx


def enforce_geometric_aspects(fig):
    axes, seen = [], set()
    pending = list(fig.axes)
    while pending:
        ax = pending.pop()
        if id(ax) in seen:
            continue
        seen.add(id(ax)); axes.append(ax); pending.extend(getattr(ax, "child_axes", []))
    selected = [ax for ax in axes if ax.get_gid() == "geometric-field"]
    for ax in selected:
        ax.set_aspect("equal", adjustable="box")
    return len(selected)


def enforce_frame_lineweights(fig, linewidth_pt):
    target = float(linewidth_pt)
    axes, seen = [], set()
    pending = list(fig.axes)
    while pending:
        ax = pending.pop()
        if id(ax) in seen:
            continue
        seen.add(id(ax)); axes.append(ax); pending.extend(getattr(ax, "child_axes", []))
    spine_count = tick_count = patch_count = 0
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(target)
            spine_count += int(spine.get_visible())
        ax.tick_params(axis="both", which="both", width=target)
        for axis in (ax.xaxis, ax.yaxis):
            for tick in (*axis.get_major_ticks(), *axis.get_minor_ticks()):
                for line in (tick.tick1line, tick.tick2line):
                    line.set_markeredgewidth(target)
                    tick_count += int(line.get_visible())
    for patch in fig.findobj(match=lambda item: isinstance(item, Patch)):
        if not patch.get_visible() or float(patch.get_linewidth()) <= 0:
            continue
        if to_rgba(patch.get_edgecolor())[3] <= 0:
            continue
        patch.set_linewidth(target); patch_count += 1
    return {
        "passed": True, "linewidth_pt": target, "visible_spine_count": spine_count,
        "visible_tick_mark_count": tick_count, "visible_frame_patch_count": patch_count,
    }


def validate_v3_model_lines(fig, cfg, layout):
    expected_width = {
        "DMFGen": float(layout["panel_b_streamlined"]["dmfg_linewidth_pt"]),
        **{
            item["key"]: float(layout["panel_b_streamlined"]["baseline_linewidth_pt"])
            for item in cfg["models"] if item["key"] != "DMFGen"
        },
    }
    checked, violations = [], []
    expected_colors = {item["key"]: item["color"].lower() for item in cfg["models"]}
    for line in fig.findobj(match=lambda item: isinstance(item, matplotlib.lines.Line2D)):
        gid = str(line.get_gid() or "")
        if not gid.startswith("model-line:"):
            continue
        model = gid.split(":", 1)[1]
        observed = {
            "model": model, "linewidth_pt": float(line.get_linewidth()),
            "color": str(line.get_color()).lower(), "alpha": float(line.get_alpha() or 1.0),
        }
        checked.append(observed)
        if not np_isclose(observed["linewidth_pt"], expected_width[model]):
            violations.append({**observed, "expected_linewidth_pt": expected_width[model]})
        if observed["color"] != expected_colors[model]:
            violations.append({**observed, "expected_color": expected_colors[model]})
    return {"passed": bool(checked) and not violations, "checked_count": len(checked),
            "expected_linewidth_pt": expected_width, "checked": checked, "violations": violations}


def np_isclose(left, right, atol=1e-9):
    return abs(float(left) - float(right)) <= float(atol)


def build_manifest(ctx, cfg, layout, layout_path, script_path, outputs, panel_meta,
                   physical_layout, qa, args, rid):
    source_paths = sorted({Path(path) for meta in panel_meta.values() for path in meta.get("sources", [])})
    cache_paths = sorted({Path(path) for meta in panel_meta.values() for path in meta.get("cache_sources", [])})
    baseline = FIGURES_DIR / "Assembled" / "FigureSourceManifest_unified_v2_20260903_2213.json"
    return {
        "workflow_label": "mixed_resolution_unified_v3_streamlined",
        "run_id": rid, "source_data_run_id": args.data_run_id,
        "multiscale_run_id": args.multiscale_run_id, "base_data_run_id": args.base_data_run_id,
        "figure_contract": {
            "core_conclusion": (
                "DMF-Gen remains strongest without H-resolution training fields across sensor budgets, "
                "and the gain reflects preserved intermediate/fine spatial structure."
            ),
            "archetype": "asymmetric mixed-modality figure",
            "backend": "Python/Matplotlib",
            "final_size_mm": [physical_layout.width_mm, physical_layout.height_mm],
            "panel_sequence": ["a", "b", "c", "d"],
            "hero_panel": "b",
            "image_integrity": "Validated cache fields only; no smoothing, sharpening, or model-specific normalization.",
        },
        "selection_contract": {
            "panel_a": "validated design protocol and exposure values",
            "panel_b_recipe_transfer": {"sensor_count": 512, "recipes": layout["panel_b_streamlined"]["recipes"]},
            "panel_b_sensor_sweeps": {"sensor_counts": layout["panel_b_streamlined"]["sensor_counts"],
                                            "recipes": layout["panel_b_streamlined"]["sweep_recipes"]},
            "panel_c": {"recipe": layout["panel_c_streamlined"]["recipe"], "sensor_count": 512,
                        "models": layout["panel_c_streamlined"]["models"]},
            "panel_d": {"qualitative_scales": ["intermediate", "fine"],
                        "quantitative_scale": "fine",
                        "recipes": layout["panel_d_streamlined"]["quantitative_recipes"]},
        },
        "layout": {**physical_layout.manifest(), **qa},
        "style_contract": style_manifest(cfg),
        "configuration": record(Path(args.config)),
        "layout_configuration": record(layout_path),
        "renderer": record(script_path),
        "unified_v2_baseline_manifest": record(baseline),
        "cache_manifest": record(ctx.cache_manifest_path),
        "representative_index": record(ctx.representatives_path),
        "csv_sources": [record(path) for path in source_paths],
        "cache_sources": [record(path) for path in cache_paths],
        "panels": panel_meta,
        "si_contract": {
            "Sx1": "complete five-recipe sensor sweeps",
            "Sx2": "complete current multi-recipe qualitative gallery",
            "Sx3": "complete three-scale qualitative and quantitative wavelet analysis",
            "tables": ["accuracy_512", "sensor_sweeps_64_512", "pattern_correlations_all_scales",
                       "variance_allocation_bias_all_scales"],
        },
        "outputs": [record(path) for path in outputs],
        "model_inference_performed": False,
        "validated_sources_modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=Path(__file__).with_name("publication_layout_unified_v3.yaml"))
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    cfg = load_config(args.config); manuscript.register_local_arial(); apply_style(cfg); ensure_output_dirs()
    layout = load_layout(args.layout)
    physical_layout = resolve_physical_layout(layout)
    rid = publication_timestamp(args.run_id)
    ctx = make_context(args, cfg, layout, rid)
    fig, axes, containers = create_composite_canvas(physical_layout)
    panel_meta = {}
    for label in layout["figure"]["panel_letters"]:
        panel_label(axes[label], label)
        panel_meta[label] = draw_panel(label, axes[label], ctx)
    geometric_axis_count = enforce_geometric_aspects(fig)
    typography_qa = manuscript.enforce_figure_typography(fig)
    finalize_colorbar_multiplier_alignment(fig)
    frame_qa = enforce_frame_lineweights(fig, layout["figure"]["uniform_frame_linewidth_pt"])
    model_line_qa = validate_v3_model_lines(fig, cfg, layout)
    if not model_line_qa["passed"]:
        raise ValueError(f"Model-line contract failed: {model_line_qa}")
    measured = measure_axes_mm(fig, containers)
    geometry_qa = validate_measured_geometry(physical_layout, measured)
    text_overflow = manuscript.validate_text_within_canvas(fig)
    panel_text_qa = validate_panel_text_boundaries(fig, containers)
    qa = {
        "geometric_axis_count": geometric_axis_count, "typography_qa": typography_qa,
        "frame_lineweight_qa": frame_qa, "model_line_qa": model_line_qa,
        "geometry_qa": geometry_qa, "text_overflow_in": dict(text_overflow),
        "panel_text_clearance_qa": panel_text_qa,
    }
    out = FIGURES_DIR / "Assembled" / f"MixedResolution_unified_v3_streamlined_{rid}"
    outputs = save_figure(
        fig, out, cfg, formats=("svg", "pdf", "png"),
        dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None,
    )
    plt.close(fig)
    manifest = build_manifest(
        ctx, cfg, layout, args.layout.resolve(), Path(__file__).resolve(), outputs,
        panel_meta, physical_layout, qa, args, rid,
    )
    manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v3_{rid}.json"
    write_json(manifest_path, manifest)
    print(f"[OK] {out}.pdf")
    print(f"[OK] {manifest_path}")


if __name__ == "__main__":
    main()

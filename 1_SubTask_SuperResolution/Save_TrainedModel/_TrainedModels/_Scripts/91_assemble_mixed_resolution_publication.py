#!/usr/bin/env python
"""Assemble the native, data-driven a--h publication figure.

No standalone panel images are imported.  The composite and standalone files
share the drawing functions in ``common.publication_panels``.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, run_id
from common.figure_style import apply_style, model_colors, save_figure, style_manifest
from common.io_utils import matching_or_latest, write_json
from common.publication_panels import (
    PANEL_OUTPUT_NAMES, PublicationContext, draw_panel, model_legend_handles, panel_label,
)


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, checksum=True):
    path = Path(path)
    return {
        "path": str(path), "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256(path) if checksum and path.exists() else None,
    }


def _source_manifest(ctx, panel_meta, outputs, cfg, layout, rid):
    source_paths = {
        str(path)
        for meta in panel_meta.values()
        for path in meta.get("sources", [])
    }
    cache_sources = sorted({
        str(path)
        for meta in panel_meta.values()
        for path in meta.get("cache_sources", [])
    })
    qa_snapshot = ctx.representatives.get("questionA", int(cfg["canonical_test"]["representative_snapshot"]))
    qa_count = int(cfg["sensor_plan"]["default_count"])
    for model in cfg["questionA"].get("qualitative_models_standalone", []):
        for recipe in cfg["questionA"]["recipes"]:
            row = ctx.cache_index.get((model, recipe, qa_snapshot, qa_count))
            if row and row.get("cache_path"):
                cache_sources.append(row["cache_path"])
    cache_sources = sorted(set(cache_sources))
    data_rid = ctx.run_id
    sensor_plan = RESULTS_DIR / "SensorPlans" / f"SensorPlan_{data_rid}.csv"
    canonical = RESULTS_DIR / "CanonicalTestIndex" / f"CanonicalTestIndex_{data_rid}.csv"
    sensor_plan_paths = {str(sensor_plan)}
    sensor_plan_paths.update(
        row.get("sensor_plan_path", "") for row in ctx.manifest_rows
        if row.get("status") == "ok" and row.get("sensor_plan_path")
    )
    standalone_outputs = sorted(
        str(path) for prefix in PANEL_OUTPUT_NAMES.values()
        for path in (FIGURES_DIR / "PublicationPanels").glob(f"{prefix}_{rid}.*")
        if path.suffix in {".pdf", ".svg", ".png"}
    )
    return {
        "run_id": rid,
        "data_run_id": data_rid,
        "timestamp_format": "YYYY-MM-DD_HH-MM" if "-" in rid else "YYYYMMDD_HHMM",
        "figure_contract": {
            "core_conclusion": "Mixed-resolution training improves H-resolution reconstruction and fine-structure recovery, while zero-H training retains measurable transfer across models and sensor budgets.",
            "archetype": "asymmetric image-plate plus quantitative composite",
            "backend": "Python/Matplotlib",
            "final_size_mm": [layout["width_mm"], layout["height_mm"]],
            "statistics": {"aggregate_center": "mean or median as labelled", "intervals": "bootstrap 95% CI or IQR as labelled", "n": 300},
            "image_integrity": "Cached physical fields only; no smoothing, sharpening, or model-dependent normalization.",
        },
        "layout": {
            "panel_width_ratios": layout["panel_width_ratios"],
            "row_height_ratios": layout["row_height_ratios"],
            "compact_gutter": layout["compact_gutter"],
            "aligned_width_groups": [["a", "c"], ["b", "d"], ["g", "h"]],
        },
        "rendering": cfg["rendering"],
        "style_contract": style_manifest(cfg),
        "selected_field": panel_meta.get("c", {}).get(
            "field", panel_meta.get("a", {}).get("field")
        ),
        "selected_models": {
            "questionA_qualitative": panel_meta.get("c", {}).get("models", []),
            "zeroH_qualitative": panel_meta.get("f", {}).get("models", []),
            "frequency_error": panel_meta.get("g", {}).get("models", []),
            "sensor_sweep": panel_meta.get("h", {}).get("models", []),
        },
        "model_palette": model_colors(cfg),
        "condition_palette": cfg["figure_style"]["condition_colors"],
        "panels": panel_meta,
        "sources": [_record(Path(path)) for path in sorted(source_paths)],
        "cache_manifest": _record(ctx.cache_manifest_path),
        "cache_sources": [_record(Path(path), checksum=False) for path in cache_sources],
        "canonical_test_index": _record(canonical),
        "sensor_plan": _record(sensor_plan),
        "sensor_plans": [_record(Path(path)) for path in sorted(sensor_plan_paths)],
        "representative_snapshot_table": _record(ctx.representatives_path),
        "configuration": _record(Path(cfg["_config_path"])),
        "outputs": [str(path) for path in outputs],
        "standalone_outputs": standalone_outputs,
        "standalone_configuration": {
            "panel_c_models": cfg["questionA"].get("qualitative_models_standalone", []),
            "panel_f_models": cfg["questionB"].get("qualitative_models", []),
            "sizes_mm": layout["standalone_sizes_mm"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", help="Source-data run ID when it differs from the output timestamp.")
    parser.add_argument("--base-data-run-id", help="Finalized base run for panels a, b, d, and e.")
    parser.add_argument("--paper", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_style(cfg)
    ensure_output_dirs()
    (FIGURES_DIR / "PublicationPanels").mkdir(parents=True, exist_ok=True)
    rid = run_id(args.run_id)
    layout = cfg["assembly"]["publication"]
    manifest = args.cache_manifest or matching_or_latest(
        RESULTS_DIR / "ReconstructionCache", "ReconstructionCache_manifest", rid, "csv"
    )
    representatives = args.representatives or matching_or_latest(
        RESULTS_DIR / "CanonicalTestIndex", "RepresentativeSnapshots", rid, "csv"
    )
    source_run_ids = {}
    if args.base_data_run_id:
        for prefix in ("ResolutionProtocol_fields", "ResolutionProtocol_budgets", "ResolutionProtocol_sensors",
                       "QuestionA_summary", "CoarseDetail_summary", "QuestionB_summary"):
            source_run_ids[prefix] = args.base_data_run_id
    ctx = PublicationContext(cfg, args.data_run_id or rid, Path(manifest), Path(representatives), source_run_ids)

    fig = plt.figure(figsize=(layout["width_mm"] / 25.4, layout["height_mm"] / 25.4))
    row_heights = layout["row_height_ratios"]
    outer = fig.add_gridspec(
        7, 1,
        height_ratios=[.13, row_heights[0], row_heights[1], .14, row_heights[2], .16, row_heights[3]],
        left=.045, right=.99, bottom=.035, top=.975, hspace=.22,
    )
    header_i = fig.add_subplot(outer[0]); header_i.set_axis_off()
    header_i.text(0, .55, layout["block_titles"][0], ha="left", va="center",
                  fontsize=cfg["figure_style"]["font_sizes"]["block"], fontweight="bold")
    header_ii = fig.add_subplot(outer[3]); header_ii.set_axis_off()
    header_ii.axhline(.92, color="#D8D8D8", lw=.8, clip_on=False)
    header_ii.text(0, .25, layout["block_titles"][1], ha="left", va="center",
                   fontsize=cfg["figure_style"]["font_sizes"]["block"], fontweight="bold")

    gutter = float(layout.get("compact_gutter", .10))
    gs_ab = outer[1].subgridspec(1, 2, width_ratios=layout["panel_width_ratios"]["top"], wspace=gutter)
    gs_cd = outer[2].subgridspec(1, 2, width_ratios=layout["panel_width_ratios"]["middle"], wspace=gutter)
    gs_ef = outer[4].subgridspec(1, 2, width_ratios=layout["panel_width_ratios"]["transfer"], wspace=gutter)
    legend_ax = fig.add_subplot(outer[5]); legend_ax.set_axis_off()
    legend_ax.legend(handles=model_legend_handles(cfg), ncol=4, loc="lower center",
                     columnspacing=1.0, handletextpad=.35, borderaxespad=0)
    gs_gh = outer[6].subgridspec(1, 2, width_ratios=layout["panel_width_ratios"]["bottom"], wspace=gutter)
    axes = {
        "a": fig.add_subplot(gs_ab[0]), "b": fig.add_subplot(gs_ab[1]),
        "c": fig.add_subplot(gs_cd[0]), "d": fig.add_subplot(gs_cd[1]),
        "e": fig.add_subplot(gs_ef[0]), "f": fig.add_subplot(gs_ef[1]),
        "g": fig.add_subplot(gs_gh[0]), "h": fig.add_subplot(gs_gh[1]),
    }
    panel_meta = {}
    for label, ax in axes.items():
        panel_label(ax, label, cfg)
        panel_meta[label] = draw_panel(label, ax, ctx, standalone=False, show_legend=False if label in {"g", "h"} else True)

    formats = cfg["figure_style"]["paper_formats"] if args.paper else cfg["figure_style"]["default_formats"]
    out = FIGURES_DIR / "Assembled" / f"{layout['output_name']}_{rid}"
    outputs = save_figure(
        fig, out, cfg, formats=formats,
        dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None,
    )
    plt.close(fig)
    source_payload = _source_manifest(ctx, panel_meta, outputs, cfg, layout, rid)
    source_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_{rid}.json"
    write_json(source_path, source_payload)
    print(f"[OK] {out}")
    print(f"[OK] {source_path}")


if __name__ == "__main__":
    main()

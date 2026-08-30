#!/usr/bin/env python
"""Export standalone mixed-resolution unified-v2 panels from shared drawers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import global_style as manuscript

from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, run_id
from common.figure_style import (
    apply_style, finalize_colorbar_multiplier_alignment, save_figure, style_manifest,
    validate_model_line_contract,
)
from common.io_utils import matching_or_latest, write_json
from common.panel_c_tuning import apply_panel_c_tuning
from common.physical_figure_layout import (
    create_standalone_canvas,
    measure_axes_mm,
    resolve_physical_layout,
    validate_measured_geometry,
    validate_panel_text_boundaries,
)
from common.publication_panels import PublicationContext
from common.publication_panels_unified_v2 import (
    PANEL_OUTPUT_NAMES,
    draw_panel,
    draw_panel_d_detail_rel_l2_diagnostic,
    draw_panel_e_shellwise_diagnostic,
    draw_panel_e_weighted_band_lsd,
    panel_label,
)


def _load_layout(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


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


def _save_one(label, version, rid, cfg, v2, ctx, physical_layout, alternate=False):
    """Render one exact panel page using the composite's publication profile."""
    fig, ax, container = create_standalone_canvas(physical_layout, label)
    panel_label(
        container, label, cfg, x=manuscript.PANEL_LABEL_X, y=manuscript.PANEL_LABEL_Y,
        ha=manuscript.PANEL_LABEL_HA, va=manuscript.PANEL_LABEL_VA,
    )
    # Deliberately use the same mode as the composite.  Standalone is an export
    # destination, not a second visual design.
    meta = draw_panel(label, ax, ctx, standalone=False, show_legend=True, version=version)
    typography_qa = manuscript.enforce_figure_typography(fig)
    finalize_colorbar_multiplier_alignment(fig)
    model_line_qa = validate_model_line_contract(fig, cfg)
    measured = measure_axes_mm(fig, {label: container})
    geometry_qa = validate_measured_geometry(physical_layout, measured, labels=[label])
    text_overflow = manuscript.validate_text_within_canvas(fig)
    panel_text_clearance_qa = validate_panel_text_boundaries(fig, {label: container})
    suffix = f"_Version{version}" if alternate else ""
    panel_dir = FIGURES_DIR / "PublicationPanels" / f"Panel_{label.upper()}"
    os.makedirs(panel_dir, exist_ok=True)
    base = panel_dir / f"{PANEL_OUTPUT_NAMES[label]}{suffix}_{rid}"
    outputs = save_figure(
        fig, base, cfg, formats=cfg["figure_style"]["paper_formats"],
        dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None,
    )
    plt.close(fig)
    print(f"[OK] {base}")
    meta["physical_layout"] = {
        "render_profile": "publication-identical",
        "panel_page_mm": measured[label],
        "composite_target_mm": physical_layout.panels[label].as_dict(),
        "geometry_qa": geometry_qa,
        "text_overflow_in": dict(text_overflow),
        "panel_text_clearance_qa": panel_text_clearance_qa,
        "typography_qa": typography_qa,
        "model_line_qa": model_line_qa,
        "implicit_scaling": False,
    }
    return meta, [str(path) for path in outputs]


def _save_diagnostic(name, drawer, label, rid, cfg, v2, ctx, physical_layout):
    rect = physical_layout.panels[label]
    width_mm, height_mm = rect.width_mm, rect.height_mm
    fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
    fig.subplots_adjust(left=.035, right=.99, bottom=.05, top=.95)
    meta = drawer(ax, ctx, standalone=True, show_legend=True)
    panel_dir = FIGURES_DIR / "PublicationPanels" / f"Panel_{label.upper()}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    base = panel_dir / f"{name}_{rid}"
    outputs = save_figure(fig, base, cfg, formats=("pdf",), dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None)
    plt.close(fig)
    print(f"[OK] {base}.pdf")
    return meta, [str(path) for path in outputs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", default=str(Path(__file__).with_name("publication_layout_unified_v2.yaml")))
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id")
    parser.add_argument("--multiscale-run-id")
    parser.add_argument("--base-data-run-id", default="2026-07-13_14-55")
    parser.add_argument("--panels", nargs="+", choices=list("abcdef"), default=list("abcdef"))
    parser.add_argument("--qualitative-version", type=int, choices=[1, 2])
    parser.add_argument("--export-qualitative-alternate", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    cfg = load_config(args.config); apply_style(cfg); ensure_output_dirs()
    # Shared Panel C tuning API: edit common/panel_c_tuning.py, not this runner.
    v2 = apply_panel_c_tuning(_load_layout(args.layout))
    physical_layout = resolve_physical_layout(v2)
    rid = run_id(args.run_id)
    ctx = _context(args, cfg, v2, rid)
    default_version = int(args.qualitative_version or v2["panel_c"]["default_version"])
    metadata, outputs = {}, []
    for label in args.panels:
        version = default_version if label == "c" else None
        metadata[label], paths = _save_one(label, version, rid, cfg, v2, ctx, physical_layout)
        outputs.extend(paths)
        if label == "c" and args.export_qualitative_alternate:
            alternate = 1 if default_version == 2 else 2
            alt_meta, alt_paths = _save_one(
                label, alternate, rid, cfg, v2, ctx, physical_layout, alternate=True,
            )
            metadata[f"c_version_{alternate}"] = alt_meta
            outputs.extend(alt_paths)
        if label == "d" and bool(v2["panel_d"].get("show_detail_rel_l2_diagnostic", False)):
            diag_meta, diag_paths = _save_diagnostic(
                "Panel_d_detail_rel_l2_diagnostic", draw_panel_d_detail_rel_l2_diagnostic,
                "d", rid, cfg, v2, ctx, physical_layout,
            )
            metadata["d_detail_rel_l2_diagnostic"] = diag_meta; outputs.extend(diag_paths)
        if label == "e" and bool(v2["panel_e"].get("export_weighted_band_lsd", False)):
            diag_meta, diag_paths = _save_diagnostic(
                "Panel_e_weighted_band_lsd", draw_panel_e_weighted_band_lsd,
                "e", rid, cfg, v2, ctx, physical_layout,
            )
            metadata["e_weighted_band_lsd"] = diag_meta; outputs.extend(diag_paths)
        if label == "e" and bool(v2["panel_e"].get("export_shellwise_diagnostic", False)):
            diag_meta, diag_paths = _save_diagnostic(
                "Panel_e_shellwise_absolute_mismatch", draw_panel_e_shellwise_diagnostic,
                "e", rid, cfg, v2, ctx, physical_layout,
            )
            metadata["e_shellwise_absolute_mismatch"] = diag_meta; outputs.extend(diag_paths)
    source_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v2_{rid}.json"
    payload = json.loads(source_path.read_text(encoding="utf-8")) if source_path.exists() else {
        "workflow_label": v2["workflow_label"], "run_id": rid,
    }
    payload["standalone_outputs"] = sorted(set(payload.get("standalone_outputs", [])) | set(outputs))
    payload["standalone_panel_metadata"] = metadata
    payload["physical_layout"] = physical_layout.manifest()
    payload["style_contract"] = style_manifest(cfg)
    write_json(source_path, payload)
    print(f"[OK] {source_path}")


if __name__ == "__main__":
    main()

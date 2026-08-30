#!/usr/bin/env python
"""Plot main fidelity heatmaps and the preserved line-interval SI diagnostic."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
import global_style as manuscript

from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, run_id
from common.figure_style import apply_style, finalize_colorbar_multiplier_alignment, save_figure
from common.io_utils import matching_or_latest, write_json
from common.multiscale_wavelet_panels import (
    draw_multiscale_fidelity,
    draw_multiscale_fidelity_intervals,
)
from common.physical_figure_layout import (
    create_standalone_canvas,
    measure_axes_mm,
    resolve_physical_layout,
    validate_measured_geometry,
    validate_panel_text_boundaries,
)
from common.publication_panels import PublicationContext
from common.publication_panels_unified_v2 import panel_label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=Path(__file__).with_name("publication_layout_unified_v2.yaml"))
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--data-run-id", required=True)
    parser.add_argument("--multiscale-run-id", required=True)
    parser.add_argument("--base-data-run-id", default="2026-07-22_10-22")
    args = parser.parse_args()
    cfg = load_config(args.config); apply_style(cfg)
    with args.layout.open("r", encoding="utf-8") as handle:
        layout = yaml.safe_load(handle) or {}
    representatives = matching_or_latest(
        RESULTS_DIR / "CanonicalTestIndex", "RepresentativeSnapshots", args.base_data_run_id, "csv"
    )
    ctx = PublicationContext(cfg, args.data_run_id, args.cache_manifest, representatives, {
        "MultiscaleWavelet_metadata": args.multiscale_run_id,
        "MultiscaleWavelet_summary": args.multiscale_run_id,
    })
    ctx.v2 = layout
    rid = run_id(args.run_id)
    physical_layout = resolve_physical_layout(layout)
    fig, ax, container = create_standalone_canvas(physical_layout, "f")
    panel_label(
        container, "f", cfg, x=manuscript.PANEL_LABEL_X, y=manuscript.PANEL_LABEL_Y,
        ha=manuscript.PANEL_LABEL_HA, va=manuscript.PANEL_LABEL_VA,
    )
    metadata = draw_multiscale_fidelity(ax, ctx, standalone=False, show_legend=False)
    typography_qa = manuscript.enforce_figure_typography(fig)
    finalize_colorbar_multiplier_alignment(fig)
    measured = measure_axes_mm(fig, {"f": container})
    metadata["physical_layout"] = {
        "panel_page_mm": measured["f"],
        "composite_target_mm": physical_layout.panels["f"].as_dict(),
        "geometry_qa": validate_measured_geometry(physical_layout, measured, labels=["f"]),
        "text_overflow_in": dict(manuscript.validate_text_within_canvas(fig)),
        "panel_text_clearance_qa": validate_panel_text_boundaries(fig, {"f": container}),
        "typography_qa": typography_qa,
        "render_profile": "publication-identical",
    }
    out = FIGURES_DIR / "MultiscaleWavelet" / f"Panel_f_MultiscaleFidelityHeatmaps_{rid}"
    outputs = save_figure(fig, out, cfg, formats=cfg["figure_style"]["paper_formats"],
                          dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None)
    plt.close(fig)
    metadata["outputs"] = [str(Path(path).resolve()) for path in outputs]
    write_json(out.with_name(out.name + "_metadata.json"), metadata)
    print(f"[OK] {out}")

    si_width_mm, si_height_mm = layout["panel_f"]["si_size_mm"]
    si_fig, si_ax = plt.subplots(figsize=(si_width_mm / 25.4, si_height_mm / 25.4))
    si_fig.subplots_adjust(left=.02, right=.995, bottom=.02, top=.985)
    si_metadata = draw_multiscale_fidelity_intervals(
        si_ax, ctx, standalone=True, show_legend=True,
    )
    si_metadata["typography_qa"] = manuscript.enforce_figure_typography(si_fig)
    si_metadata["text_overflow_in"] = dict(manuscript.validate_text_within_canvas(si_fig))
    si_out = FIGURES_DIR / "MultiscaleWavelet" / f"SI_MultiscaleFidelityIntervals_{rid}"
    si_outputs = save_figure(
        si_fig, si_out, cfg, formats=cfg["figure_style"]["paper_formats"],
        dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None,
    )
    plt.close(si_fig)
    si_metadata["outputs"] = [str(Path(path).resolve()) for path in si_outputs]
    write_json(si_out.with_name(si_out.name + "_metadata.json"), si_metadata)
    print(f"[OK] {si_out}")


if __name__ == "__main__":
    main()

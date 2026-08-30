#!/usr/bin/env python
"""Export standalone a--h panels using the composite's drawing functions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, run_id
from common.figure_style import apply_style, save_figure, style_manifest
from common.io_utils import matching_or_latest, write_json
from common.publication_panels import PublicationContext, draw_panel, panel_label, standalone_output_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--representatives", type=Path)
    parser.add_argument("--data-run-id", help="Source-data run ID when it differs from the output timestamp.")
    parser.add_argument("--base-data-run-id", help="Finalized base run for panels a, b, d, and e.")
    parser.add_argument("--panels", nargs="+", choices=list("abcdefgh"), default=list("abcdefgh"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    apply_style(cfg)
    ensure_output_dirs()
    (FIGURES_DIR / "PublicationPanels").mkdir(parents=True, exist_ok=True)
    rid = run_id(args.run_id)
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
    sizes = cfg["assembly"]["publication"]["standalone_sizes_mm"]
    outputs = []
    panel_meta = {}
    for label in args.panels:
        width_mm, height_mm = sizes[label]
        fig, ax = plt.subplots(figsize=(width_mm / 25.4, height_mm / 25.4))
        if label in {"b", "e"}:
            fig.subplots_adjust(left=.16, right=.985, bottom=.20, top=.88)
        else:
            fig.subplots_adjust(left=.09, right=.985, bottom=.10, top=.93)
        panel_label(ax, label, cfg, x=.005, y=.985, ha="left", va="top")
        panel_meta[label] = draw_panel(label, ax, ctx, standalone=True, show_legend=True)
        base = standalone_output_base(label, rid)
        paths = save_figure(
            fig, base, cfg, formats=cfg["figure_style"]["paper_formats"],
            dpi=cfg["figure_style"]["paper_dpi"], bbox_inches=None,
        )
        plt.close(fig)
        outputs.extend(str(path) for path in paths)
        print(f"[OK] {base}")

    source_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_{rid}.json"
    if source_path.exists():
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    else:
        payload = {"run_id": rid}
    payload["standalone_outputs"] = sorted(set(payload.get("standalone_outputs", [])) | set(outputs))
    payload["standalone_panel_metadata"] = panel_meta
    payload["style_contract"] = style_manifest(cfg)
    write_json(source_path, payload)
    print(f"[OK] {source_path}")


if __name__ == "__main__":
    main()

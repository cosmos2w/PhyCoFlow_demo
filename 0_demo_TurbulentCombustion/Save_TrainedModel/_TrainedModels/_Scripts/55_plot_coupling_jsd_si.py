#!/usr/bin/env python
"""Render header-selected supplementary channel-coupling JSD panels from CSV.

The editable selection API lives at the top of the native publication
assembler.  No model, checkpoint, reconstruction cache, or metric is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from common.config import FIGURES_DIR, RESULTS_DIR, SCRIPT_DIR, add_common_args, load_config
from common.figure_style import apply_style
from common.io_utils import latest, read_csv


def _load_assembler_module():
    """Load the native assembler without renaming its established script file."""
    path = SCRIPT_DIR / "91_assemble_coupled_field_publication.py"
    spec = importlib.util.spec_from_file_location("coupled_field_publication_assembler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import reusable Panel-D renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _input_path(args) -> Path:
    exact = RESULTS_DIR / "JointPDF_JSD" / f"CouplingJSD_per_snapshot_{args.run_id}.csv" if args.run_id else None
    return args.input or (exact if exact and exact.exists() else latest(RESULTS_DIR / "JointPDF_JSD", "CouplingJSD_per_snapshot", "csv"))


def _summary_path(args) -> Path:
    exact = RESULTS_DIR / "JointPDF_JSD" / f"CouplingJSD_summary_{args.run_id}.csv" if args.run_id else None
    return args.summary or (exact if exact and exact.exists() else latest(RESULTS_DIR / "JointPDF_JSD", "CouplingJSD_summary", "csv"))


def _flow_input_path(args) -> Path:
    exact = RESULTS_DIR / "JointPDF_JSD" / f"FlowConsistencyJSD_per_snapshot_{args.run_id}.csv" if args.run_id else None
    return args.flow_input or (exact if exact and exact.exists() else latest(RESULTS_DIR / "JointPDF_JSD", "FlowConsistencyJSD_per_snapshot", "csv"))


def _flow_summary_path(args) -> Path:
    exact = RESULTS_DIR / "JointPDF_JSD" / f"FlowConsistencyJSD_summary_{args.run_id}.csv" if args.run_id else None
    return args.flow_summary or (exact if exact and exact.exists() else latest(RESULTS_DIR / "JointPDF_JSD", "FlowConsistencyJSD_summary", "csv"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=SCRIPT_DIR / "publication_layout_coupled_field.yaml")
    parser.add_argument("--input", type=Path, help="CouplingJSD_per_snapshot CSV.")
    parser.add_argument("--summary", type=Path, help="CouplingJSD_summary CSV.")
    parser.add_argument("--flow-input", type=Path, help="FlowConsistencyJSD_per_snapshot CSV for U1-p.")
    parser.add_argument("--flow-summary", type=Path, help="FlowConsistencyJSD_summary CSV for U1-p.")
    parser.add_argument("--figures", nargs="+", default=["si_1", "si_2"],
                        help="Header-API selection keys to export (default: si_1 si_2).")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--output-id", help="Output suffix; defaults to YYYYMMDD_HHMM.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    apply_style(cfg)
    layout = yaml.safe_load(args.layout.read_text(encoding="utf-8")) or {}
    renderer = _load_assembler_module()
    coupling_input, coupling_summary = _input_path(args), _summary_path(args)
    flow_input, flow_summary = _flow_input_path(args), _flow_summary_path(args)
    rows = [*read_csv(coupling_input), *read_csv(flow_input)]
    summary_rows = [*read_csv(coupling_summary), *read_csv(flow_summary)]
    output_id = args.output_id or datetime.now().strftime("%Y%m%d_%H%M")
    formats = args.formats or layout["export"]["formats"]
    dpi = args.dpi or int(layout["export"]["png_dpi"])
    outputs = []
    for figure_key in args.figures:
        selection = renderer._panel_d_specification(figure_key)
        if figure_key == "main":
            raise ValueError("Use 91_assemble_coupled_field_publication.py --panel d for the main manuscript Panel D.")
        width, height, bounds = renderer._standalone_panel_canvas(layout, "d")
        fig = plt.figure(figsize=(width, height), facecolor="white")
        left, right, bottom, top = bounds
        slot = fig.add_gridspec(1, 1, left=left, right=right, bottom=bottom, top=top)[0, 0]
        renderer.draw_coupling_jsd_panel(
            fig, slot, cfg, rows, summary_rows, subplot_specs=selection["subplots"], show_method_labels=True,
        )
        renderer._panel_letter(fig, slot, selection["panel_letter"])
        base = FIGURES_DIR / "JointPDF_JSD" / "Supplementary" / f"SI_CouplingJSD_{selection['output_tag']}_{output_id}"
        outputs.extend(renderer._save_figure(
            fig, base, formats, dpi,
        ))
        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(), "selection_key": figure_key,
            "panel_letter": selection["panel_letter"], "subplots": selection["subplots"],
            "pair_interpretations": {
                pair: label.replace("\n", "; ") for pair, label in renderer.COUPLING_PAIR_LABELS.items()
            },
            "per_snapshot_csv": [str(coupling_input), str(flow_input)],
            "summary_csv": [str(coupling_summary), str(flow_summary)],
            "source_policy": "CSV only; cache-only CO-U1 and U1-p exports plus reused finalized T-U1/T-CO JSD rows",
            "outputs": [str(path) for path in outputs if path.stem.startswith(base.name)],
        }
        (base.parent / f"{base.name}_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        plt.close(fig)
        print(f"[OK] supplementary coupling panel: {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

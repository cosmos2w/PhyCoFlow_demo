#!/usr/bin/env python
"""Preview U1--p flow-field-consistency JSD across the three conditioning cases.

Reads diagnostic CSVs only.  It reuses the current Panel-D renderer so method
palette, missing-data slots, mean annotations, and x-axis conventions match.
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


PAIR = "U1-p"
TITLE = "$U_1$–$p$\nFlow field consistency"


def _renderer():
    path = SCRIPT_DIR / "91_assemble_coupled_field_publication.py"
    spec = importlib.util.spec_from_file_location("coupled_field_publication_assembler", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Panel-D renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _path(args, prefix: str) -> Path:
    exact = RESULTS_DIR / "JointPDF_JSD" / f"{prefix}_{args.run_id}.csv" if args.run_id else None
    return (args.input if prefix.endswith("per_snapshot") else args.summary) or (
        exact if exact and exact.exists() else latest(RESULTS_DIR / "JointPDF_JSD", prefix, "csv")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser, models=False)
    parser.add_argument("--layout", type=Path, default=SCRIPT_DIR / "publication_layout_coupled_field.yaml")
    parser.add_argument("--input", type=Path, help="FlowConsistencyJSD_per_snapshot CSV.")
    parser.add_argument("--summary", type=Path, help="FlowConsistencyJSD_summary CSV.")
    parser.add_argument("--conditions", nargs="+", default=["Cond_T", "Cond_TU1", "Cond_COTU1P"],
                        help="Columns, in order, for the diagnostic preview.")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--output-id", help="Output suffix; defaults to YYYYMMDD_HHMM.")
    args = parser.parse_args()
    if not 1 <= len(args.conditions) <= 3:
        raise ValueError("Specify one to three conditions for this three-column diagnostic preview.")
    cfg = load_config(args.config); apply_style(cfg)
    renderer = _renderer()
    for condition in args.conditions:
        if condition not in cfg["conditions"]:
            raise ValueError(f"Unknown condition {condition!r}")
    layout = yaml.safe_load(args.layout.read_text(encoding="utf-8")) or {}
    rows = read_csv(_path(args, "FlowConsistencyJSD_per_snapshot"))
    summary = read_csv(_path(args, "FlowConsistencyJSD_summary"))
    width, height, bounds = renderer._standalone_panel_canvas(layout, "d")
    fig = plt.figure(figsize=(width, height), facecolor="white")
    left, right, bottom, top = bounds
    slot = fig.add_gridspec(1, 1, left=left, right=right, bottom=bottom, top=top)[0, 0]
    specifications = [{"pair": PAIR, "condition": condition, "title": TITLE} for condition in args.conditions]
    renderer.draw_coupling_jsd_panel(fig, slot, cfg, rows, summary, subplot_specs=specifications, show_method_labels=True)
    renderer._panel_letter(fig, slot, "D")
    output_id = args.output_id or datetime.now().strftime("%Y%m%d_%H%M")
    base = FIGURES_DIR / "JointPDF_JSD" / "Diagnostic" / f"FlowConsistency_U1_p_{output_id}"
    outputs = renderer._save_figure(fig, base, args.formats or layout["export"]["formats"],
                                    args.dpi or int(layout["export"]["png_dpi"]))
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "purpose": "diagnostic preview only",
        "pair": PAIR, "interpretation": "Flow field consistency", "conditions": args.conditions,
        "per_snapshot_csv": str(_path(args, "FlowConsistencyJSD_per_snapshot")),
        "summary_csv": str(_path(args, "FlowConsistencyJSD_summary")),
        "source_policy": "cache-derived physical-unit U1-p JSD; no model inference or cache regeneration",
        "outputs": [str(path) for path in outputs],
    }
    (base.parent / f"{base.name}_source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plt.close(fig)
    print(f"[OK] U1-p flow-consistency diagnostic preview: {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

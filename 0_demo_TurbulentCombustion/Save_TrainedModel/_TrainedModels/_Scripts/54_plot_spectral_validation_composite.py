#!/usr/bin/env python
"""Build a reference-style LSD-plus-spectrum composite from CSV exports only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import string

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, method_items, run_id
from common.figure_style import add_panel_label, apply_style, figure_size, mark_missing, method_line_style, save_figure
from common.io_utils import latest, read_csv
from global_style import (
    COLOR_AXIS, COLOR_DIVIDER, COLOR_GROUND_TRUTH, COLOR_MISSING_TEXT,
    LW_DIVIDER, LW_ERRORBAR, LW_GRID, LW_LINE_PLOT, LW_LINE_SECONDARY,
    SIZE_ANNOTATION, model_alpha,
)


def _paths(args):
    energy_dir, lsd_dir = RESULTS_DIR / "Spectral" / "EnergySpectra", RESULTS_DIR / "Spectral" / "SpectralLSD"
    exact_energy = energy_dir / f"EnergySpectra_snapshot_{args.run_id}.csv" if args.run_id else None
    exact_per = lsd_dir / f"SpectralLSD_per_snapshot_{args.run_id}.csv" if args.run_id else None
    exact_summary = lsd_dir / f"SpectralLSD_summary_{args.run_id}.csv" if args.run_id else None
    return (
        args.energy or (exact_energy if exact_energy and exact_energy.exists() else latest(energy_dir, "EnergySpectra_snapshot", "csv")),
        args.lsd_per_snapshot or (exact_per if exact_per and exact_per.exists() else latest(lsd_dir, "SpectralLSD_per_snapshot", "csv")),
        args.lsd_summary or (exact_summary if exact_summary and exact_summary.exists() else latest(lsd_dir, "SpectralLSD_summary", "csv")),
    )


def _float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def _selected_fields(cfg, names):
    normalize = lambda value: "".join(char for char in str(value).lower() if char.isalnum())
    wanted = {normalize(name) for name in names}
    fields = [field for field in cfg["fields"] if normalize(field["key"]) in wanted]
    if not fields:
        raise ValueError(f"No configured fields match --channels {names}.")
    return fields


def _bar_panel(ax, field, methods, condition, summary, per_values, cfg, rng):
    missing, any_valid = [], False
    for index, method in enumerate(methods):
        row = summary.get((method["directory"], condition, field["key"]))
        mean = _float(row["mean_lsd_db"]) if row and row["status"] == "ok" else np.nan
        if not np.isfinite(mean):
            missing.append(method["name"])
            ax.text(index, .03, "Missing", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=SIZE_ANNOTATION, color=COLOR_MISSING_TEXT)
            continue
        low, high = _float(row["ci95_low_lsd_db"]), _float(row["ci95_high_lsd_db"])
        color = cfg["spectral"]["plotting"]["dmf_gen_accent"] if method["name"] == "DMF-Gen" else method["color"]
        ax.bar(index, mean, color=color, alpha=model_alpha(method["name"]),
               width=.72, edgecolor=COLOR_AXIS, linewidth=LW_DIVIDER)
        ax.errorbar(index, mean, yerr=[[max(mean-low, 0)], [max(high-mean, 0)]], color=COLOR_AXIS, linewidth=LW_ERRORBAR, capsize=2)
        values = [value for value in per_values.get((method["directory"], condition, field["key"]), []) if np.isfinite(value)]
        ax.scatter(index + rng.uniform(-.15, .15, len(values)), values, color=COLOR_DIVIDER, s=6, alpha=.7, linewidths=0, zorder=4)
        any_valid = True
    observed = field["index"] in cfg["conditions"][condition]["cond_fields"]
    ax.set_title(f"{field['label']} ({'Observed' if observed else 'Unobserved'})")
    ax.set_xticks(range(len(methods)), [method["name"] for method in methods], rotation=45, ha="right")
    ax.set_ylabel("LSD (dB)")
    ax.grid(axis="y", linewidth=LW_GRID, alpha=.3)
    if not any_valid:
        mark_missing(ax, "Missing", cfg)
    elif missing:
        ax.text(.02, .03, f"Missing: {len(missing)} model(s)", transform=ax.transAxes, fontsize=SIZE_ANNOTATION, color=COLOR_MISSING_TEXT)


def _spectrum_panel(ax, field, methods, condition, snapshot, curves, cfg):
    truth = curves.get(("truth", field["key"], "truth"), [])
    def valid(rows):
        return [( _float(row["wavenumber"]), _float(row["spectral_energy"]) ) for row in rows if row["status"] == "ok" and _float(row["wavenumber"]) > 0 and _float(row["spectral_energy"]) > 0]
    true = valid(truth)
    if true:
        ax.plot(*np.array(true).T, color=COLOR_GROUND_TRUTH, linewidth=LW_LINE_PLOT, label="Ground truth")
    missing, any_valid = [], bool(true)
    for index, method in enumerate(methods):
        data = valid(curves.get((method["directory"], field["key"], "reconstruction"), []))
        if not data:
            missing.append(method["name"]); continue
        color = cfg["spectral"]["plotting"]["dmf_gen_accent"] if method["name"] == "DMF-Gen" else method["color"]
        ax.plot(*np.array(data).T, color=color, alpha=model_alpha(method["name"]),
                linewidth=LW_LINE_PLOT if method["name"] == "DMF-Gen" else LW_LINE_SECONDARY,
                linestyle=method_line_style(index))
        any_valid = True
    if not any_valid:
        mark_missing(ax, "Missing", cfg)
    else:
        ax.set_xscale(cfg["spectral"]["plotting"]["xscale"]); ax.set_yscale(cfg["spectral"]["plotting"]["yscale"])
        ax.set_xlabel("Wavenumber"); ax.set_ylabel("Spectral energy"); ax.grid(which="both", linewidth=LW_GRID, alpha=.25)
        if missing: ax.text(.02, .03, f"Missing: {len(missing)} model(s)", transform=ax.transAxes, fontsize=SIZE_ANNOTATION, color=COLOR_MISSING_TEXT)
    return true


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    parser.set_defaults(models=["DMFGen", "LatentFM", "SiT", "Senseiver", "GeoFNO"])
    parser.add_argument("--energy", type=Path)
    parser.add_argument("--lsd-per-snapshot", type=Path)
    parser.add_argument("--lsd-summary", type=Path)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--snapshot-index", type=int, default=None)
    parser.add_argument("--channels", nargs="+", default=["CH4", "CO", "U1"])
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int)
    args = parser.parse_args()
    cfg, rid = load_config(args.config), run_id(args.run_id)
    apply_style(cfg)
    energy_path, per_path, summary_path = _paths(args)
    energy_rows, per_rows, summary_rows = read_csv(energy_path), read_csv(per_path), read_csv(summary_path)
    condition = args.condition or cfg["spectral"]["representative_condition"]
    snapshot = int(args.snapshot_index if args.snapshot_index is not None else cfg["spectral"]["representative_snapshot"])
    fields, methods = _selected_fields(cfg, args.channels), list(method_items(cfg, args.models))
    curves = defaultdict(list)
    for row in energy_rows:
        if row["condition"] == condition and int(row["snapshot_index"]) == snapshot:
            curves[(row["model_key"], row["field_name"], row["source"])].append(row)
    summary = {(row["model_key"], row["condition"], row["field_name"]): row for row in summary_rows}
    per_values = defaultdict(list)
    for row in per_rows:
        if row["status"] == "ok":
            per_values[(row["model_key"], row["condition"], row["field_name"])].append(_float(row["lsd_db"]))

    fig, axes = plt.subplots(2, len(fields), figsize=figure_size(cfg, "double", 96), squeeze=False, constrained_layout=True)
    rng = np.random.default_rng(cfg["defaults"]["seed"])
    truth_present = False
    for col, field in enumerate(fields):
        _bar_panel(axes[0, col], field, methods, condition, summary, per_values, cfg, rng)
        truth_present = bool(_spectrum_panel(axes[1, col], field, methods, condition, snapshot, curves, cfg)) or truth_present
        add_panel_label(axes[0, col], string.ascii_lowercase[col], cfg, x=-.14, y=1.03)
        add_panel_label(axes[1, col], string.ascii_lowercase[col + len(fields)], cfg, x=-.14, y=1.03)
    model_handles = [Patch(
        facecolor=(cfg["spectral"]["plotting"]["dmf_gen_accent"] if method["name"] == "DMF-Gen" else method["color"]),
        alpha=model_alpha(method["name"]), label=method["name"],
    ) for method in methods]
    if model_handles:
        fig.legend(model_handles, [handle.get_label() for handle in model_handles], loc="upper center", ncol=len(model_handles), bbox_to_anchor=(.5, 1.08), title="Models")
    if truth_present:
        axes[1, 0].legend([Line2D([], [], color=COLOR_GROUND_TRUTH, linewidth=LW_LINE_PLOT)], ["Ground truth"], loc="best")
    base = FIGURES_DIR / "Spectral" / "Composite" / f"SpectralValidationComposite_{condition}_s{snapshot:04d}_{rid}"
    save_figure(fig, base, cfg, args.formats, args.dpi)
    plt.close(fig)
    print(f"[OK] {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

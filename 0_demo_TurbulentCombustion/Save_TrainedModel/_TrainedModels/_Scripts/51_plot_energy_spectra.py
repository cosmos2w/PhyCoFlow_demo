#!/usr/bin/env python
"""Plot representative channel-wise spectral-energy curves from CSV only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, method_items, run_id
from common.figure_style import apply_style, figure_size, mark_missing, method_line_style, save_figure
from common.io_utils import latest, read_csv
from global_style import (
    COLOR_GROUND_TRUTH, COLOR_MISSING_TEXT, LW_GRID, LW_LINE_PLOT,
    LW_LINE_SECONDARY, SIZE_ANNOTATION, model_alpha,
)


def _input_path(args) -> Path:
    exact = RESULTS_DIR / "Spectral" / "EnergySpectra" / f"EnergySpectra_snapshot_{args.run_id}.csv" if args.run_id else None
    return args.input or (exact if exact and exact.exists() else latest(RESULTS_DIR / "Spectral" / "EnergySpectra", "EnergySpectra_snapshot", "csv"))


def _field_map(cfg: dict) -> dict[str, dict]:
    return {field["key"]: field for field in cfg["fields"]}


def _curve(rows: list[dict]) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        valid = [row for row in rows if row["status"] == "ok" and float(row["wavenumber"]) > 0 and float(row["spectral_energy"]) > 0]
        if not valid:
            return None
        return np.array([float(row["wavenumber"]) for row in valid]), np.array([float(row["spectral_energy"]) for row in valid])
    except (KeyError, ValueError):
        return None


def _plot_field(ax, grouped, field: dict, methods: list[dict], condition: str, snapshot: int, cfg: dict):
    truth = _curve(grouped.get(("truth", field["key"], "truth"), []))
    if truth is not None:
        ax.plot(*truth, color=COLOR_GROUND_TRUTH, linewidth=LW_LINE_PLOT, linestyle="-", label="Ground truth", zorder=4)
    missing, handles = [], []
    for index, method in enumerate(methods):
        curve = _curve(grouped.get((method["directory"], field["key"], "reconstruction"), []))
        if curve is None:
            missing.append(method["name"])
            continue
        color = cfg["spectral"]["plotting"]["dmf_gen_accent"] if method["name"] == "DMF-Gen" else method["color"]
        line, = ax.plot(*curve, color=color, alpha=model_alpha(method["name"]),
                        linewidth=LW_LINE_PLOT if method["name"] == "DMF-Gen" else LW_LINE_SECONDARY,
                        linestyle=method_line_style(index), label=method["name"])
        handles.append(line)
    observed = field["index"] in cfg["conditions"][condition]["cond_fields"]
    ax.set_title(f"{field['label']} ({'Observed' if observed else 'Unobserved'})")
    if truth is None and not handles:
        mark_missing(ax, "Missing", cfg)
    else:
        ax.set_xscale(cfg["spectral"]["plotting"]["xscale"])
        ax.set_yscale(cfg["spectral"]["plotting"]["yscale"])
        ax.set_xlabel("Wavenumber")
        ax.set_ylabel("Channel-wise spectral energy")
        ax.grid(which="both", linewidth=LW_GRID, alpha=.25)
        if missing:
            ax.text(.02, .03, f"Missing: {len(missing)} model(s)", transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=SIZE_ANNOTATION, color=COLOR_MISSING_TEXT)
    return handles


def _save_layout(rows, fields, methods, condition, snapshot, cfg, base: Path, formats, dpi):
    grouped = defaultdict(list)
    for row in rows:
        if row["condition"] == condition and int(row["snapshot_index"]) == snapshot:
            grouped[(row["model_key"], row["field_name"], row["source"])].append(row)
    fig, axes = plt.subplots(2, 3, figsize=figure_size(cfg, "double", 112), constrained_layout=True)
    legend_handles = []
    for ax, field in zip(axes.flat, fields):
        legend_handles.extend(_plot_field(ax, grouped, field, methods, condition, snapshot, cfg))
    for ax in axes.flat[len(fields):]:
        ax.set_axis_off()
    # Deduplicate the common legend while retaining truth only once.
    by_label = {handle.get_label(): handle for handle in legend_handles}
    truth_handle = next((line for ax in axes.flat for line in ax.get_lines() if line.get_label() == "Ground truth"), None)
    if truth_handle is not None:
        by_label = {"Ground truth": truth_handle, **by_label}
    if by_label:
        fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=min(5, len(by_label)), bbox_to_anchor=(.5, 1.02))
    save_figure(fig, base, cfg, formats, dpi)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--snapshot-index", type=int, default=None)
    parser.add_argument("--channels", nargs="+")
    parser.add_argument("--reduced-models", nargs="+", default=["DMFGen", "SiT", "GeoFNO", "Senseiver"])
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int)
    args = parser.parse_args()
    cfg, rid = load_config(args.config), run_id(args.run_id)
    apply_style(cfg)
    rows = read_csv(_input_path(args))
    condition = args.condition or cfg["spectral"]["representative_condition"]
    snapshot = int(args.snapshot_index if args.snapshot_index is not None else cfg["spectral"]["representative_snapshot"])
    normalize = lambda value: "".join(char for char in str(value).lower() if char.isalnum())
    wanted = {normalize(value) for value in args.channels} if args.channels else None
    fields = [field for field in cfg["fields"] if field["index"] in cfg["spectral"]["channels"] and (wanted is None or normalize(field["key"]) in wanted)]
    methods = list(method_items(cfg, args.models))
    out = FIGURES_DIR / "Spectral" / "EnergySpectra"
    _save_layout(rows, fields, methods, condition, snapshot, cfg, out / f"EnergySpectra_five_channel_{condition}_s{snapshot:04d}_{rid}", args.formats, args.dpi)
    reduced = list(method_items(cfg, args.reduced_models))
    _save_layout(rows, fields, reduced, condition, snapshot, cfg, out / f"EnergySpectra_reduced_{condition}_s{snapshot:04d}_{rid}", args.formats, args.dpi)
    for field in fields:
        fig, ax = plt.subplots(figsize=figure_size(cfg, "single", 54), constrained_layout=True)
        grouped = defaultdict(list)
        for row in rows:
            if row["condition"] == condition and int(row["snapshot_index"]) == snapshot:
                grouped[(row["model_key"], row["field_name"], row["source"])].append(row)
        handles = _plot_field(ax, grouped, field, methods, condition, snapshot, cfg)
        truth = next((line for line in ax.get_lines() if line.get_label() == "Ground truth"), None)
        legend = ([truth] if truth else []) + handles
        if legend:
            unique = {line.get_label(): line for line in legend}; ax.legend(unique.values(), unique.keys(), loc="best")
        save_figure(fig, out / f"EnergySpectra_{field['key']}_{condition}_s{snapshot:04d}_{rid}", cfg, args.formats, args.dpi)
        plt.close(fig)
    print(f"[OK] spectra figures under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

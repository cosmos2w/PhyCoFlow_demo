#!/usr/bin/env python
"""Plot cache-exported channel-wise dB spectral-LSD statistics from CSV only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, method_items, run_id
from common.figure_style import apply_style, figure_size, mark_missing, save_figure
from common.io_utils import latest, read_csv
from global_style import (
    COLOR_AXIS, COLOR_DIVIDER, COLOR_MISSING_TEXT, LW_DIVIDER, LW_ERRORBAR,
    LW_GRID, SIZE_ANNOTATION, model_alpha,
)


def _input_paths(args):
    folder = RESULTS_DIR / "Spectral" / "SpectralLSD"
    exact_per = folder / f"SpectralLSD_per_snapshot_{args.run_id}.csv" if args.run_id else None
    exact_summary = folder / f"SpectralLSD_summary_{args.run_id}.csv" if args.run_id else None
    per = args.per_snapshot or (exact_per if exact_per and exact_per.exists() else latest(folder, "SpectralLSD_per_snapshot", "csv"))
    summary = args.summary or (exact_summary if exact_summary and exact_summary.exists() else latest(folder, "SpectralLSD_summary", "csv"))
    return per, summary


def _as_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _plot_field(ax, field, methods, condition, summary_lookup, per_lookup, cfg, log_y, rng):
    x = np.arange(len(methods)); valid_any = False
    missing = []
    for i, method in enumerate(methods):
        summary = summary_lookup.get((method["directory"], condition, field["key"]))
        mean = _as_float(summary["mean_lsd_db"]) if summary and summary["status"] == "ok" else np.nan
        if not np.isfinite(mean):
            missing.append(method["name"])
            ax.text(i, .03, "Missing", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=SIZE_ANNOTATION, color=COLOR_MISSING_TEXT)
            continue
        low, high = _as_float(summary["ci95_low_lsd_db"]), _as_float(summary["ci95_high_lsd_db"])
        errors = np.array([[max(mean - low, 0)], [max(high - mean, 0)]])
        color = cfg["spectral"]["plotting"]["dmf_gen_accent"] if method["name"] == "DMF-Gen" else method["color"]
        ax.bar(i, mean, color=color, alpha=model_alpha(method["name"]), width=.72,
               edgecolor=COLOR_AXIS, linewidth=LW_DIVIDER, zorder=2)
        ax.errorbar(i, mean, yerr=errors, color=COLOR_AXIS, capsize=2, linewidth=LW_ERRORBAR, zorder=3)
        values = per_lookup.get((method["directory"], condition, field["key"]), [])
        if cfg["spectral"]["plotting"]["show_snapshot_points"]:
            values = [value for value in values if np.isfinite(value)]
            ax.scatter(i + rng.uniform(-.16, .16, len(values)), values, s=7, color=COLOR_DIVIDER, alpha=.7, zorder=4, linewidths=0)
        valid_any = True
    observed = field["index"] in cfg["conditions"][condition]["cond_fields"]
    ax.set_title(f"{field['label']} ({'Observed' if observed else 'Unobserved'})")
    ax.set_xticks(x, [method["name"] for method in methods], rotation=45, ha="right")
    ax.set_ylabel("LSD (dB)")
    ax.grid(axis="y", linewidth=LW_GRID, alpha=.3, zorder=0)
    if log_y and valid_any:
        ax.set_yscale("log")
    if not valid_any:
        mark_missing(ax, "Missing", cfg)
    elif missing:
        ax.text(.02, .03, f"Missing: {len(missing)} model(s)", transform=ax.transAxes, fontsize=SIZE_ANNOTATION, color=COLOR_MISSING_TEXT)


def _plot_condition(fields, methods, condition, summary_rows, per_rows, cfg, base, formats, dpi, log_y):
    summary_lookup = {(row["model_key"], row["condition"], row["field_name"]): row for row in summary_rows}
    per_lookup = defaultdict(list)
    for row in per_rows:
        if row["status"] == "ok":
            per_lookup[(row["model_key"], row["condition"], row["field_name"])].append(_as_float(row["lsd_db"]))
    fig, axes = plt.subplots(2, 3, figsize=figure_size(cfg, "double", 95), constrained_layout=True)
    rng = np.random.default_rng(cfg["defaults"]["seed"])
    for ax, field in zip(axes.flat, fields):
        _plot_field(ax, field, methods, condition, summary_lookup, per_lookup, cfg, log_y, rng)
    for ax in axes.flat[len(fields):]:
        ax.set_axis_off()
    save_figure(fig, base, cfg, formats, dpi)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--per-snapshot", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--condition", default="all", help="One condition or 'all' for separate figures.")
    parser.add_argument("--channels", nargs="+")
    parser.add_argument("--log-y", action="store_true")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"])
    parser.add_argument("--dpi", type=int)
    args = parser.parse_args()
    cfg, rid = load_config(args.config), run_id(args.run_id)
    apply_style(cfg)
    per_path, summary_path = _input_paths(args)
    per_rows, summary_rows = read_csv(per_path), read_csv(summary_path)
    normalize = lambda value: "".join(char for char in str(value).lower() if char.isalnum())
    wanted = {normalize(value) for value in args.channels} if args.channels else None
    fields = [field for field in cfg["fields"] if field["index"] in cfg["spectral"]["channels"] and (wanted is None or normalize(field["key"]) in wanted)]
    methods = list(method_items(cfg, args.models))
    conditions = list(cfg["conditions"]) if args.condition == "all" else [args.condition]
    out = FIGURES_DIR / "Spectral" / "SpectralLSD"
    for condition in conditions:
        _plot_condition(fields, methods, condition, summary_rows, per_rows, cfg,
                        out / f"SpectralLSD_{condition}_{rid}", args.formats, args.dpi, args.log_y)
    print(f"[OK] spectral LSD figures under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Plot representative joint PDFs using only exported CSV source data."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, method_items, run_id
from common.figure_style import apply_style, mark_missing, save_figure
from common.io_utils import latest, read_csv



def matrix(rows):
    n = max(int(r["bin_x"]) for r in rows) + 1; out = np.full((n, n), np.nan)
    for r in rows: out[int(r["bin_x"]), int(r["bin_y"])] = float(r["probability"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p); p.add_argument("--input", type=Path); p.add_argument("--metrics", type=Path)
    p.add_argument("--subset", nargs="+", help="Reduced model set for a compact contact sheet.")
    p.add_argument("--tag", default="", help="Optional contact-sheet filename tag (for example 'all' or 'main').")
    p.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"]); p.add_argument("--dpi", type=int); args = p.parse_args()
    cfg = load_config(args.config); rid = run_id(args.run_id); apply_style(cfg)
    exact = RESULTS_DIR / "JointPDF" / f"JointPDF_snapshot_{args.run_id}.csv" if args.run_id else None
    exact_metrics = RESULTS_DIR / "JointPDF" / f"JointPDF_snapshot_metrics_{args.run_id}.csv" if args.run_id else None
    path = args.input or (exact if exact and exact.exists() else latest(RESULTS_DIR / "JointPDF", "JointPDF_snapshot", "csv")); metric_path = args.metrics or (exact_metrics if exact_metrics and exact_metrics.exists() else latest(RESULTS_DIR / "JointPDF", "JointPDF_snapshot_metrics", "csv"))
    rows, metrics = read_csv(path), read_csv(metric_path); metric = {(r["method"], r["pair"]): r for r in metrics}; groups = defaultdict(list)
    for r in rows: groups[(r["method"], r["pair"], r["source"])].append(r)
    methods = [m["name"] for m in method_items(cfg, args.subset or args.models)]; pairs = list(dict.fromkeys(r["pair"] for r in rows))
    positive = [float(r["probability"]) for r in rows if r["status"] == "ok" and float(r["probability"]) > 0]; norm = LogNorm(max(min(positive, default=1e-8), 1e-8), max(positive, default=1))
    # One shared truth column keeps the visual reference available without
    # duplicating the truth distribution for every reconstruction method.
    fig, axes = plt.subplots(len(pairs), len(methods) + 1, figsize=(max(3.5, 1.35*(len(methods) + 1)), 1.5*len(pairs)), squeeze=False, constrained_layout=True)
    for i, pair in enumerate(pairs):
        truth_ax = axes[i, 0]
        truth_key = ("truth", pair, "truth")
        if truth_key not in groups:
            mark_missing(truth_ax, "Missing truth", cfg)
        else:
            rs = groups[truth_key]; m = matrix(rs); first = rs[0]
            extent = [float(first["x_left"]), float(rs[-1]["x_right"]), float(first["y_left"]), float(rs[-1]["y_right"])]
            truth_ax.imshow(m.T, origin="lower", extent=extent, aspect="auto", cmap="magma", norm=norm)
        truth_ax.set_title("Ground truth")
        truth_ax.set_ylabel(pair.split("-")[1])
        if i == len(pairs) - 1:
            truth_ax.set_xlabel(pair.split("-")[0])
        for j, method in enumerate(methods, start=1):
            ax = axes[i, j]; key = (method, pair, "reconstruction")
            title = method
            method_metric = metric.get((method, pair), {"status": "missing cache"})
            if key not in groups or method_metric["status"] != "ok": mark_missing(ax, "Missing", cfg); ax.set_title(title)
            else:
                m = matrix(groups[key]); first = groups[key][0]; extent=[float(first["x_left"]), float(groups[key][-1]["x_right"]), float(first["y_left"]), float(groups[key][-1]["y_right"])]
                ax.imshow(m.T, origin="lower", extent=extent, aspect="auto", cmap="magma", norm=norm)
                ax.set_title(f"{title}\nJSD={float(method_metric['jsd_base2']):.3f}")
            if i == len(pairs)-1: ax.set_xlabel(pair.split("-")[0])
    tag = f"_{args.tag.strip()}" if args.tag.strip() else ""
    base = FIGURES_DIR / "JointPDF" / f"JointPDF_contact_sheet{tag}_{rid}"; save_figure(fig, base, cfg, args.formats, args.dpi); plt.close(fig)
    # Individual model/pair panels use the same source CSV and normalization.
    for method in methods:
        for pair in pairs:
            fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.55), constrained_layout=True)
            for ax, source in zip(axes, ("truth", "reconstruction")):
                key=("truth", pair, "truth") if source == "truth" else (method, pair, "reconstruction")
                if key not in groups or metric[(method,pair)]["status"] != "ok": mark_missing(ax, "Missing", cfg); continue
                rs=groups[key]; m=matrix(rs); ex=[float(r["x_left"]) for r in rs]+[float(rs[-1]["x_right"])]; ey=[float(r["y_left"]) for r in rs]+[float(rs[-1]["y_right"])]
                ax.imshow(m.T, origin="lower", extent=[min(ex),max(ex),min(ey),max(ey)], aspect="auto", cmap="magma", norm=norm); ax.set_title("Ground truth" if source=="truth" else f"Reconstruction (JSD={float(metric[(method,pair)]['jsd_base2']):.3f})")
            save_figure(fig, FIGURES_DIR / "JointPDF" / f"JointPDF_{pair}_{method.replace(' ','_')}_{rid}", cfg, args.formats, args.dpi); plt.close(fig)
    print(f"[OK] {base}"); return 0


if __name__ == "__main__": raise SystemExit(main())

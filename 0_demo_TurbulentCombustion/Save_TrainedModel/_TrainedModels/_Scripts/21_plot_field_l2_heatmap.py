#!/usr/bin/env python
"""Plot three aligned condition heatmaps from FieldL2 summary CSV only."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, method_items, run_id
from common.figure_style import apply_style, figure_size, missing_cmap, save_figure
from common.io_utils import latest, read_csv
from global_style import COLOR_AXIS, SIZE_ANNOTATION


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p); p.add_argument("--input", type=Path); p.add_argument("--scale", choices=["linear", "log"], default="linear")
    p.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"]); p.add_argument("--dpi", type=int); args = p.parse_args()
    cfg = load_config(args.config); rid = run_id(args.run_id); apply_style(cfg)
    exact = RESULTS_DIR / "FieldL2" / f"FieldL2_summary_{args.run_id}.csv" if args.run_id else None
    path = args.input or (exact if exact and exact.exists() else latest(RESULTS_DIR / "FieldL2", "FieldL2_summary", "csv")); rows = read_csv(path)
    methods = [m["name"] for m in method_items(cfg, args.models)]; fields = [f["key"] for f in cfg["fields"]] + ["Unobserved_mean"]
    lookup = {(r["method"], r["condition"], r["field"]): float(r["mean"]) if r["status"] == "ok" else np.nan for r in rows}
    matrices = [np.array([[lookup.get((m, c, f), np.nan) for f in fields] for m in methods]) for c in cfg["conditions"]]
    finite = np.concatenate([x[np.isfinite(x)] for x in matrices]); vmin, vmax = (finite.min(), finite.max()) if finite.size else (0, 1)
    norm = LogNorm(max(vmin, 1e-6), vmax) if args.scale == "log" else Normalize(vmin, vmax)
    fig, axes = plt.subplots(1, 3, figsize=figure_size(cfg, "double", 80), sharey=True, constrained_layout=True)
    cmap = missing_cmap("magma", cfg["style"]["missing"]["facecolor"]); image = None
    for ax, (condition, spec), matrix in zip(axes, cfg["conditions"].items(), matrices):
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(spec["label"]); ax.set_xticks(range(len(fields)), [next((x["label"] for x in cfg["fields"] if x["key"] == f), "Unobs. mean") for f in fields], rotation=45, ha="right")
        ax.set_yticks(range(len(methods)), methods); ax.tick_params(length=0)
        for i in range(len(methods)):
            for j, field in enumerate(fields):
                value = matrix[i, j]; text = "Missing" if not np.isfinite(value) else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=SIZE_ANNOTATION,
                        color="white" if np.isfinite(value) and norm(value) > .55 else COLOR_AXIS)
                if field != "Unobserved_mean" and cfg["fields"][j]["index"] in spec["cond_fields"]:
                    ax.plot(j, i - .42, marker="_", color="#22D7E6", ms=6, mew=1)
    fig.colorbar(image, ax=axes, fraction=.025, pad=.02, label="Mean physical relative L2")
    base = FIGURES_DIR / "FieldL2" / f"FieldL2_heatmap_{rid}"; save_figure(fig, base, cfg, args.formats, args.dpi); plt.close(fig)
    print(f"[OK] {base}"); return 0


if __name__ == "__main__": raise SystemExit(main())

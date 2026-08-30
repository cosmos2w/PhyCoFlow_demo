#!/usr/bin/env python
"""Export one primary-size GT/reconstruction/error contour panel per field."""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from common.cache import load_cache
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, method_items, run_id
from common.figure_style import apply_style, mark_missing, save_figure
from common.io_utils import latest, read_csv, write_json
from common.statistics import relative_l2
from global_style import SIZE_TICK_LABEL



def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(p); p.set_defaults(models=["DMFGen"]); p.add_argument("--cache-manifest", type=Path)
    p.add_argument("--conditions", nargs="+", default=["all"]); p.add_argument("--snapshot", type=int, default=0)
    p.add_argument("--y-compression", type=float, default=1.0); p.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"])
    p.add_argument("--dpi", type=int); p.add_argument("--contour-levels", type=int, default=50,
        help="Number of filled contour boundaries for every physical-field panel.")
    args = p.parse_args(); cfg = load_config(args.config); rid = run_id(args.run_id); apply_style(cfg)
    exact = RESULTS_DIR / "ReconstructionCache" / f"ReconstructionCache_manifest_{args.run_id}.csv" if args.run_id else None
    manifest_path = args.cache_manifest or (exact if exact and exact.exists() else latest(RESULTS_DIR / "ReconstructionCache", "ReconstructionCache_manifest", "csv"))
    manifest = read_csv(manifest_path); methods = list(method_items(cfg, args.models)); conditions = list(cfg["conditions"]) if "all" in args.conditions else args.conditions
    selected = [r for r in manifest if r["method"] in {m["name"] for m in methods} and r["condition"] in conditions and int(r["snapshot"]) == args.snapshot]
    loaded = {}
    for row in selected:
        if row.get("status") == "ok" and row.get("cache_path"):
            try: loaded[(row["method"], row["condition"])] = load_cache(Path(row["cache_path"]))
            except Exception: pass
    truth_reference = {condition: next((payload for (method_name, cond), payload in loaded.items() if cond == condition), None) for condition in conditions}
    limits = {}; error_limits = {}
    for field in cfg["fields"]:
        c = field["index"]; vals = []; errs = []
        for arrays, _ in loaded.values(): vals.extend([arrays["truth_phys"][:, c], arrays["recon_phys"][:, c]]); errs.append(np.abs(arrays["recon_phys"][:, c] - arrays["truth_phys"][:, c]))
        pool = np.concatenate(vals) if vals else np.array([0., 1.]); limits[field["key"]] = [float(np.nanmin(pool)), float(np.nanmax(pool))]
        epool = np.concatenate(errs) if errs else np.array([1.]); error_limits[field["key"]] = [0., float(np.nanquantile(epool, cfg["defaults"]["error_quantile"]))]
    for method in methods:
        for condition in conditions:
            payload = loaded.get((method["name"], condition)); out_dir = FIGURES_DIR / "_Contours" / method["directory"] / condition; out_dir.mkdir(parents=True, exist_ok=True)
            for field in cfg["fields"]:
                for kind in ("GT", "Rec", "Err"):
                    active_payload = truth_reference.get(condition) if kind == "GT" else payload
                    # Single-column width: the physical map gets nearly all width;
                    # a narrow, same-height colorbar sits in an adjacent GridSpec cell.
                    fig = plt.figure(figsize=(3.50, 1.18), constrained_layout=True)
                    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 0.045), wspace=0.03)
                    ax = fig.add_subplot(grid[0, 0]); cax = fig.add_subplot(grid[0, 1])
                    if active_payload is None:
                        mark_missing(ax, "Missing", cfg); cax.set_visible(False)
                        ax.set_title(field["label"])
                    else:
                        arrays, meta = active_payload; c = field["index"]
                        xy = arrays["coords_phys"][:, :2].copy(); xy[:, 1] *= args.y_compression
                        tri = mtri.Triangulation(xy[:, 0], xy[:, 1])
                        truth = arrays["truth_phys"][:, c]; recon = arrays["recon_phys"][:, c]
                        values = truth if kind == "GT" else recon if kind == "Rec" else np.abs(recon - truth)
                        lo, hi = error_limits[field["key"]] if kind == "Err" else limits[field["key"]]
                        upper = hi if hi > lo else lo + 1.0
                        levels = np.linspace(lo, upper, args.contour_levels)
                        contour = ax.tricontourf(
                            tri, values, levels=levels,
                            cmap="magma" if kind == "Err" else field["cmap"], extend="max",
                        )
                        observed = c in cfg["conditions"][condition]["cond_fields"]
                        if observed:
                            idx = arrays["obs_indices"][arrays["obs_field_ids"] == c]
                            ax.scatter(xy[idx, 0], xy[idx, 1], s=.65, c="white", edgecolors="black", linewidths=.12)
                        title = f"{field['label']} ({'obs.' if observed else 'unobs.'})"
                        if kind == "Err":
                            title += f"\nrelative L2 = {relative_l2(truth, recon):.2e}"
                        ax.set_title(title)
                        error_ticks = np.linspace(lo, upper, 4) if kind == "Err" else None
                        colorbar = fig.colorbar(contour, cax=cax, ticks=error_ticks)
                        ticks = colorbar.get_ticks()
                        # Use fixed labels rather than a formatter: colorbar's
                        # update routine otherwise resets scientific notation.
                        colorbar.set_ticks(ticks)
                        colorbar.set_ticklabels([f"{tick:.2e}" for tick in ticks])
                        colorbar.ax.tick_params(labelsize=SIZE_TICK_LABEL, length=2)
                        ax.set_aspect("equal")
                        ax.set_xticks([]); ax.set_yticks([])
                    base = out_dir / f"Fig_{kind}_{field['key']}_s{args.snapshot:04d}_{method['directory']}_{condition}_{rid}"
                    save_figure(fig, base, cfg, args.formats, args.dpi); plt.close(fig)
    write_json(FIGURES_DIR / "_Contours" / f"ContourColorLimits_{rid}.json", {"physical_limits": limits, "robust_error_limits": error_limits,
        "error_quantile": cfg["defaults"]["error_quantile"], "cache_manifest": str(manifest_path),
        "y_compression": args.y_compression, "contour_levels": args.contour_levels,
        "error_colorbar_ticks": 4, "numeric_format": "%.2e"})
    print(f"[OK] contour plates under {FIGURES_DIR / '_Contours'}"); return 0


if __name__ == "__main__": raise SystemExit(main())

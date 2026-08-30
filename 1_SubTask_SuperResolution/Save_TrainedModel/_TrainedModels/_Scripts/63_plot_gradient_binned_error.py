#!/usr/bin/env python
"""Plot gradient-quantile reconstruction error from summary CSV only."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, run_id
from common.figure_style import (
    LW_LINE_PLOT, apply_style, method_line_style, model_alphas,
    model_colors, save_figure,
)
from common.io_utils import matching_or_latest, read_csv

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p,models=False); p.add_argument("--summary",type=Path); args=p.parse_args(); cfg=load_config(args.config); apply_style(cfg); rid=run_id(args.run_id); rows=read_csv(args.summary or matching_or_latest(RESULTS_DIR/"GradientError","GradientError_summary",args.run_id,"csv")); fig,ax=plt.subplots(figsize=(89/25.4,68/25.4)); combos=sorted({(r["model"],r["model_label"],r["recipe"],r["recipe_label"]) for r in rows}); colors=model_colors(cfg); alphas=model_alphas(cfg)
    for i,(model,mlabel,recipe,rlabel) in enumerate(combos): s=sorted([r for r in rows if r["model"]==model and r["recipe"]==recipe],key=lambda r:int(r["bin_index"])); x=[.5*(float(r["quantile_low"])+float(r["quantile_high"])) for r in s]; y=[float(r["normalized_rmse"]) for r in s]; ax.plot(x,y,marker="o",ms=3,lw=LW_LINE_PLOT,color=colors.get(model),alpha=alphas.get(model),ls=method_line_style(i),label=f"{mlabel}, {rlabel}")
    ax.set_xlabel("Ground-truth gradient quantile"); ax.set_ylabel("Normalized RMSE"); ax.legend(fontsize=5); out=FIGURES_DIR/"GradientError"/f"GradientError_{rid}"; save_figure(fig,out,cfg); plt.close(fig); print(f"[OK] {out}")
if __name__=="__main__": main()

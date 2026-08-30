#!/usr/bin/env python
"""Plot the 2x3 horizontal JSD violin grid from per-snapshot CSV only."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, method_items, run_id
from common.figure_style import apply_style, figure_size, save_figure
from common.io_utils import latest, read_csv
from global_style import COLOR_AXIS, COLOR_MISSING_TEXT, LW_DIVIDER, LW_GRID, LW_LINE_PLOT, SIZE_ANNOTATION, model_alpha


def main()->int:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter); add_common_args(p); p.add_argument("--input",type=Path); p.add_argument("--summary",type=Path); p.add_argument("--jitter",action="store_true"); p.add_argument("--formats",nargs="+",choices=["png","pdf","svg"]); p.add_argument("--dpi",type=int); args=p.parse_args()
    cfg=load_config(args.config); rid=run_id(args.run_id); apply_style(cfg); exact=RESULTS_DIR/"JointPDF_JSD"/f"JointPDF_JSD_per_snapshot_{args.run_id}.csv" if args.run_id else None; path=args.input or (exact if exact and exact.exists() else latest(RESULTS_DIR/"JointPDF_JSD","JointPDF_JSD_per_snapshot","csv")); rows=read_csv(path)
    exact_summary=RESULTS_DIR/"JointPDF_JSD"/f"JointPDF_JSD_summary_{args.run_id}.csv" if args.run_id else None
    summary_path=args.summary or (exact_summary if exact_summary and exact_summary.exists() else latest(RESULTS_DIR/"JointPDF_JSD","JointPDF_JSD_summary","csv"))
    complete={(r["method"],r["condition"],r["pair"]):r.get("status")=="ok" for r in read_csv(summary_path)}
    methods=[m for m in method_items(cfg,args.models)]; pairs=["T-CO","T-U1"]; conditions=list(cfg["conditions"]); groups=defaultdict(list)
    for r in rows:
        if r["status"]=="ok" and complete.get((r["method"],r["condition"],r["pair"]),False):
            try: groups[(r["method"],r["condition"],r["pair"])].append(float(r["jsd_base2"]))
            except ValueError: pass
    fig,axes=plt.subplots(2,3,figsize=figure_size(cfg,"double",105),sharex=True,sharey=True,constrained_layout=True); rng=np.random.default_rng(cfg["defaults"]["seed"])
    for i,pair in enumerate(pairs):
        for j,condition in enumerate(conditions):
            ax=axes[i,j]; datasets=[]; positions=[]; colors=[]; alphas=[]
            for k,m in enumerate(methods):
                values=groups.get((m["name"],condition,pair),[])
                if values: datasets.append(values); positions.append(k); colors.append(m["color"]); alphas.append(model_alpha(m["name"]))
                else: ax.text(.98,k,"Missing",transform=ax.get_yaxis_transform(),ha="right",va="center",fontsize=SIZE_ANNOTATION,color=COLOR_MISSING_TEXT)
            if datasets:
                vp=ax.violinplot(datasets,positions=positions,vert=False,widths=.75,showextrema=False,showmedians=False)
                for body,color,alpha in zip(vp["bodies"],colors,alphas): body.set_facecolor(color); body.set_edgecolor(COLOR_AXIS); body.set_alpha(alpha); body.set_linewidth(LW_DIVIDER)
                for vals,pos in zip(datasets,positions):
                    q=np.quantile(vals,[.25,.5,.75]); ax.plot([q[0],q[2]],[pos,pos],color=COLOR_AXIS,lw=LW_LINE_PLOT); ax.plot(q[1],pos,"|",color="white",ms=5,mew=LW_DIVIDER)
                    if args.jitter: ax.scatter(vals,pos+rng.uniform(-.10,.10,len(vals)),s=3,c=COLOR_AXIS,alpha=.35,zorder=3)
            ax.set_title(cfg["conditions"][condition]["label"]); ax.set_yticks(range(len(methods)),[m["name"] for m in methods]); ax.grid(axis="x",lw=LW_GRID,alpha=.3)
            if j==0: ax.set_ylabel(pair)
            if i==1: ax.set_xlabel("JSD (base 2)")
    for ax in axes.flat: ax.set_xlim(0,1)
    base=FIGURES_DIR/"JointPDF_JSD"/f"JointPDF_JSD_violin_{rid}"; save_figure(fig,base,cfg,args.formats,args.dpi); plt.close(fig); print(f"[OK] {base}"); return 0
if __name__=="__main__": raise SystemExit(main())

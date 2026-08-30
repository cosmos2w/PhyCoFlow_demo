#!/usr/bin/env python
"""Plot nested sensor-count error curves from summary CSV only."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, run_id
from common.figure_style import (
    LW_ERRORBAR, LW_LINE_PLOT, LW_LINE_SECONDARY, NEUTRAL_MID,
    apply_style, model_alphas, model_colors, save_figure,
)
from common.io_utils import matching_or_latest, read_csv

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p,models=False); p.add_argument("--summary",type=Path); p.add_argument("--sensor-excluded",action="store_true"); p.add_argument("--grid-points",type=int); args=p.parse_args(); cfg=load_config(args.config); apply_style(cfg); rid=run_id(args.run_id); rows=read_csv(args.summary or matching_or_latest(RESULTS_DIR/"SensorSweep","SensorSweep_summary",args.run_id,"csv")); metric="physical_rel_l2_sensor_excluded" if args.sensor_excluded else "physical_rel_l2"; rows=[r for r in rows if r["metric"]==metric]
    recipes=cfg["sensor_sweep"]["recipes"]; counts=[int(v) for v in cfg["sensor_sweep"]["counts"]]; models=[m["key"] for m in cfg["models"] if m["key"] in cfg["sensor_sweep"]["main_models"]]; colors=model_colors(cfg); alphas=model_alphas(cfg); grid_points=args.grid_points or max((int(float(r.get("evaluation_grid_points",0))) for r in rows),default=0)
    if grid_points<=0: raise ValueError("Sensor density requires evaluation_grid_points in the summary or --grid-points")
    fig,axes=plt.subplots(1,len(recipes),figsize=(122/25.4,72/25.4),sharey=True); axes=np.atleast_1d(axes); xpos=np.arange(len(counts)); lows=[]; highs=[]
    for r in rows:
        if r.get("model") in models and r.get("recipe") in recipes: lows.append(float(r["ci95_low"])); highs.append(float(r["ci95_high"]))
    ymin=max(0,min(lows)*.86); ymax=max(highs)*1.08
    labels=[f"{n}\n{100*n/grid_points:.1f}" for n in counts]
    for j,(ax,recipe) in enumerate(zip(axes,recipes)):
        for model in models:
            subset=sorted([r for r in rows if r["model"]==model and r["recipe"]==recipe],key=lambda r:int(r["sensor_count"])); subset=[r for r in subset if int(r["sensor_count"]) in counts]
            x=np.array([counts.index(int(r["sensor_count"])) for r in subset]); y=np.array([float(r["mean"]) for r in subset]); lo=np.array([float(r["ci95_low"]) for r in subset]); hi=np.array([float(r["ci95_high"]) for r in subset]); ax.plot(x,y,marker="o",ms=3,color=colors[model],alpha=alphas[model],lw=LW_LINE_PLOT,label=next(m["label"] for m in cfg["models"] if m["key"]==model)); ax.errorbar(x,y,yerr=[y-lo,hi-y],fmt="none",ecolor=colors[model],alpha=alphas[model],elinewidth=LW_ERRORBAR,capsize=1.4)
        formal=counts.index(int(cfg["sensor_plan"]["default_count"])); ax.axvline(formal,color=NEUTRAL_MID,ls=":",lw=LW_LINE_SECONDARY); ax.text(formal+.12,.97,"formal setting",transform=ax.get_xaxis_transform(),rotation=90,ha="left",va="top",fontsize=cfg["figure_style"]["font_sizes"]["tick"]); ax.set_xticks(xpos,labels); ax.tick_params(axis="x",labelsize=cfg["figure_style"]["font_sizes"]["tick"]); ax.set_xlim(-.35,len(counts)-.65); ax.set_ylim(ymin,ymax); ax.set_title(cfg["recipes"][recipe]["label"]); ax.set_xlabel("Count / density (%)")
        if j==0: ax.set_ylabel("Sensor-excluded relative L2" if args.sensor_excluded else "Physical relative L2")
    handles,legend_labels=axes[0].get_legend_handles_labels(); fig.legend(handles,legend_labels,ncol=4,loc="upper center",bbox_to_anchor=(.5,.995)); fig.subplots_adjust(left=.11,right=.985,bottom=.20,top=.78,wspace=.14); tag="_sensor_excluded" if args.sensor_excluded else ""; out=FIGURES_DIR/"SensorSweep"/f"SensorSweep{tag}_{rid}"; save_figure(fig,out,cfg); plt.close(fig); print(f"[OK] {out}")
if __name__=="__main__": main()

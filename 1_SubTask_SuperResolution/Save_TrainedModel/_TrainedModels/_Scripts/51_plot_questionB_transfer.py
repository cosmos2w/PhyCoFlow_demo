#!/usr/bin/env python
"""Plot separate three-estimate zero-H transfer groups from CSV only."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, run_id
from common.figure_style import apply_style, save_figure
from common.io_utils import matching_or_latest, read_csv

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p,models=False); p.add_argument("--summary",type=Path); args=p.parse_args(); cfg=load_config(args.config); apply_style(cfg); rid=run_id(args.run_id); rows=read_csv(args.summary or matching_or_latest(RESULTS_DIR/"QuestionB_ZeroH","QuestionB_summary",args.run_id,"csv")); fig,ax=plt.subplots(figsize=(89/25.4,72/25.4)); recipes=cfg["questionB"]["recipes"]; colors=[cfg["figure_style"]["condition_colors"][r] for r in recipes]
    labeled=set()
    for i,model in enumerate(cfg["models"]):
        vals=[]
        for j,(recipe,color) in enumerate(zip(recipes,colors)):
            f=next((r for r in rows if r["model"]==model["key"] and r["recipe"]==recipe),None)
            if f:
                y=float(f["mean"]); lo=float(f["ci95_low"]); hi=float(f["ci95_high"]); x=i+(j-1)*.16; label=f.get("recipe_label",recipe) if recipe not in labeled else None; labeled.add(recipe); ax.errorbar(x,y,yerr=[[y-lo],[hi-y]],fmt="o",color=color,capsize=2,label=label); vals.append((x,y))
        if len(vals)>1: ax.plot([x for x,_ in vals],[y for _,y in vals],color="#A8A8A8",lw=.8,zorder=0)
    ax.set_xticks(range(len(cfg["models"])),[m["label"] for m in cfg["models"]],rotation=15,ha="right"); ax.set_ylabel("Physical relative L2"); ax.legend(); out=FIGURES_DIR/"QuestionB_ZeroH"/f"QuestionB_ZeroH_{rid}"; save_figure(fig,out,cfg); plt.close(fig); print(f"[OK] {out}")
if __name__=="__main__": main()

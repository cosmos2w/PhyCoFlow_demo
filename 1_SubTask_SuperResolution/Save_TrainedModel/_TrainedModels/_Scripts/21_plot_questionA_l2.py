#!/usr/bin/env python
"""Plot Question A paired model estimates from summary CSV only."""
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
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p,models=False); p.add_argument("--summary",type=Path); p.add_argument("--paired",type=Path); p.add_argument("--paper",action="store_true"); args=p.parse_args(); cfg=load_config(args.config); apply_style(cfg); rid=run_id(args.run_id)
    rows=read_csv(args.summary or matching_or_latest(RESULTS_DIR/"QuestionA_DataBenefit","QuestionA_summary",args.run_id,"csv")); models=[m["key"] for m in cfg["models"]]; labels={m["key"]:m["label"] for m in cfg["models"]}; colors={r:cfg["figure_style"]["condition_colors"][r] for r in cfg["questionA"]["recipes"]}
    fig,ax=plt.subplots(figsize=(89/25.4,72/25.4)); x=np.arange(len(models)); offsets=[-.12,.12]
    for off,recipe in zip(offsets,cfg["questionA"]["recipes"]):
        for i,model in enumerate(models):
            found=next((r for r in rows if r["model"]==model and r["recipe"]==recipe),None)
            if found:
                y=float(found["mean"]); lo=float(found["ci95_low"]); hi=float(found["ci95_high"]); ax.errorbar(i+off,y,yerr=[[y-lo],[hi-y]],fmt="o",color=colors[recipe],capsize=2,label=found.get("recipe_label",recipe) if i==0 else None)
        # connect the two estimates within each model below
    ref_labeled=False
    for i,model in enumerate(models):
        vals=[]
        for off,recipe in zip(offsets,cfg["questionA"]["recipes"]):
            f=next((r for r in rows if r["model"]==model and r["recipe"]==recipe),None)
            if f: vals.append((i+off,float(f["mean"])))
        if len(vals)==2: ax.plot([v[0] for v in vals],[v[1] for v in vals],color="#A8A8A8",lw=.8,zorder=0)
        ref=next((r for r in rows if r["model"]==model and r["recipe"]==cfg["questionA"]["reference_recipe"]),None)
        if ref: ax.plot(i,float(ref["mean"]),marker="D",ms=3,mfc="white",mec="#606060",ls="none",label="H-only reference" if not ref_labeled else None); ref_labeled=True
    ax.set_xticks(x,[labels[m] for m in models],rotation=15,ha="right"); ax.set_ylabel("Physical relative L2"); ax.legend(); out=FIGURES_DIR/"QuestionA_DataBenefit"/f"QuestionA_DataBenefit_{rid}"; save_figure(fig,out,cfg,formats=cfg["figure_style"]["paper_formats"] if args.paper else None); plt.close(fig); print(f"[OK] {out}")
    try: paired=read_csv(args.paired or matching_or_latest(RESULTS_DIR/"QuestionA_DataBenefit","QuestionA_paired",args.run_id,"csv"))
    except FileNotFoundError: paired=[]
    if paired:
        fig,ax=plt.subplots(figsize=(89/25.4,60/25.4)); rng=np.random.default_rng(42)
        for i,model in enumerate(models):
            vals=np.array([float(r["relative_improvement_percent"]) for r in paired if r["model"]==model]);
            if vals.size: ax.scatter(i+rng.normal(0,.035,vals.size),vals,s=7,color="#7884B4",alpha=.45); ax.plot([i-.12,i+.12],[np.median(vals)]*2,color="#484878",lw=1.4)
        ax.axhline(0,color="#767676",lw=.8,ls="--"); ax.set_xticks(range(len(models)),[labels[m] for m in models],rotation=15,ha="right"); ax.set_ylabel("Paired improvement (%)"); out_si=FIGURES_DIR/"QuestionA_DataBenefit"/f"QuestionA_paired_improvement_SI_{rid}"; save_figure(fig,out_si,cfg); plt.close(fig); print(f"[OK] {out_si}")
if __name__=="__main__": main()

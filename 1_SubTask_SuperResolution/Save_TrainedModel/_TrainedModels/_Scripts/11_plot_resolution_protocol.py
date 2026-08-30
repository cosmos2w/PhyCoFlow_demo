#!/usr/bin/env python
"""Plot native-resolution protocol fields and recipe composition from CSV only."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, run_id
from common.figure_style import apply_style, save_figure
from common.io_utils import matching_or_latest, read_csv
from common.rendering import render_field

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p,models=False); p.add_argument("--fields",type=Path); p.add_argument("--budgets",type=Path); p.add_argument("--sensors",type=Path); p.add_argument("--paper",action="store_true"); p.add_argument("--proportional-panels",action="store_true"); args=p.parse_args()
    cfg=load_config(args.config); apply_style(cfg); rid=run_id(args.run_id); fp=args.fields or matching_or_latest(RESULTS_DIR/"ResolutionProtocol","ResolutionProtocol_fields",args.run_id,"csv"); bp=args.budgets or matching_or_latest(RESULTS_DIR/"ResolutionProtocol","ResolutionProtocol_budgets",args.run_id,"csv"); sp=args.sensors or matching_or_latest(RESULTS_DIR/"ResolutionProtocol","ResolutionProtocol_sensors",args.run_id,"csv")
    fields=read_csv(fp); budgets=read_csv(bp); sensors=read_csv(sp); fig=plt.figure(figsize=(183/25.4,92/25.4)); widths=[32,64,128] if args.proportional_panels else [1,1,1]; gs=fig.add_gridspec(2,3,height_ratios=[1,.8],width_ratios=widths,hspace=.5,wspace=.35)
    values=np.array([float(r["field_value"]) for r in fields]); vmin,vmax=np.nanquantile(values,[.01,.99]); artist=None
    for j,tag in enumerate("LMH"):
        rows=[r for r in fields if r["resolution"]==tag]; sr=[r for r in sensors if r["resolution"]==tag]; coords=np.array([[float(r["x_phys"]),float(r["y_phys"])] for r in rows]); z=np.array([float(r["field_value"]) for r in rows]); ax=fig.add_subplot(gs[0,j]); artist=render_field(ax,coords,z,mode="native_cells",vmin=vmin,vmax=vmax,cmap=cfg["rendering"]["cmap"]); neff=sr[0]["effective_unique_sensor_count"] if sr else "?"; ax.set_title(f"{tag}: {rows[0]['num_x']}×{rows[0]['num_y']} (n_eff={neff})")
    cax=fig.add_axes([.92,.56,.012,.3]); fig.colorbar(artist,cax=cax,label=fields[0]["field_name"])
    ax=fig.add_subplot(gs[1,:]); labels=[r["recipe_label"] for r in budgets]; x=np.arange(len(labels)); bottom=np.zeros(len(labels)); colors={"L":"#D8D8D8","M":"#7884B4","H":"#484878"}
    for tag in "LMH":
        vals=np.array([float(r.get(f"train_cases_{tag}",0)) for r in budgets]); ax.bar(x,vals,bottom=bottom,label=tag,color=colors[tag]); bottom+=vals
    for i,r in enumerate(budgets): ax.text(i,bottom[i]+.25,f"DOF {float(r['spatial_dof_budget_normalized_H_only']):.2f}×",ha="center",fontsize=5.5)
    ax.set_xticks(x,labels,rotation=15,ha="right"); ax.set_ylabel("Active training cases"); ax.legend(ncol=3,title="Native resolution")
    formats=cfg["figure_style"]["paper_formats"] if args.paper else None; out=FIGURES_DIR/"ResolutionProtocol"/f"ResolutionProtocol_{rid}"; save_figure(fig,out,cfg,formats=formats,dpi=cfg["figure_style"]["paper_dpi"] if args.paper else None); plt.close(fig); print(f"[OK] {out}")
if __name__=="__main__": main()

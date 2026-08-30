#!/usr/bin/env python
"""Render cache-backed Question A/B contour comparisons and auditable ROI metadata."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.cache import load_cache
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, run_id
from common.figure_style import apply_style, mark_missing, save_figure
from common.io_utils import artifact_name, write_csv
from common.rendering import automatic_gradient_roi, render_field
from common.statistics import relative_l2
from common.workflow import cache_entries

def render_all_models(cfg,args,rid,recipes,count):
    models=cfg["models"]; entries=cache_entries(args.cache_manifest,models=["all"],recipes=recipes,sensor_count=count,snapshots=[args.snapshot]); indexed={(e["model"],e["recipe"]):e for e in entries}; fig,axes=plt.subplots(len(models),1+len(recipes),figsize=(183/25.4,max(55,36*len(models))/25.4),squeeze=False)
    roi_rows=[]
    for i,model in enumerate(models):
        available=[]
        for recipe in recipes:
            e=indexed.get((model["key"],recipe))
            if e:
                try: available.append((recipe,*load_cache(Path(e["cache_path"]))))
                except Exception: pass
        if not available:
            for ax in axes[i]: mark_missing(ax,cfg=cfg)
            axes[i,0].set_ylabel(model["label"]); continue
        truth=available[0][1]["truth_phys"].reshape(-1); coords=available[0][1]["coords_phys"]; roi=args.manual_roi or automatic_gradient_roi(coords,truth); allv=[truth,*[a["recon_phys"].reshape(-1) for _,a,_ in available]]; vmin,vmax=np.nanquantile(np.concatenate(allv),[.01,.99]); mode=args.render_mode or cfg["rendering"]["default_comparison_mode"]
        render_field(axes[i,0],coords,truth,mode=mode,cmap=cfg["rendering"]["cmap"],vmin=vmin,vmax=vmax,roi=roi); axes[i,0].set_title("Ground truth" if i==0 else ""); axes[i,0].set_ylabel(model["label"])
        by={r:(a,m) for r,a,m in available}
        for j,recipe in enumerate(recipes,1):
            if recipe not in by: mark_missing(axes[i,j],cfg=cfg); continue
            a,m=by[recipe]; pred=a["recon_phys"].reshape(-1); render_field(axes[i,j],a["coords_phys"],pred,mode=mode,cmap=cfg["rendering"]["cmap"],vmin=vmin,vmax=vmax,roi=roi); axes[i,j].set_title(m.get("recipe_label",recipe) if i==0 else f"L2={relative_l2(truth,pred):.3f}")
        roi_rows.append({"mode":args.mode,"model":model["key"],"snapshot_index":args.snapshot,"sensor_count":count,"roi_xmin":roi[0],"roi_xmax":roi[1],"roi_ymin":roi[2],"roi_ymax":roi[3],"roi_source":"manual" if args.manual_roi else "ground_truth_gradient_only"})
    out=FIGURES_DIR/"_Contours"/f"all_models_{args.mode}_{args.render_mode or 'default'}_{rid}"; save_figure(fig,out,cfg); plt.close(fig); folder="QuestionA_DataBenefit" if args.mode=="questionA" else "QuestionB_ZeroH"; rp=RESULTS_DIR/folder/artifact_name(f"ContourROI_all_models_{args.mode}",rid,"csv"); write_csv(rp,roi_rows); print(f"[OK] {out}\n[OK] {rp}")

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p,models=False); p.add_argument("--mode",choices=["questionA","questionB"],default="questionA"); p.add_argument("--model",default="DMFGen"); p.add_argument("--snapshot",type=int,default=0); p.add_argument("--sensor-count",type=int); p.add_argument("--render-mode",choices=["native_cells","scatter","smooth","hybrid"]); p.add_argument("--manual-roi",nargs=4,type=float); p.add_argument("--cache-manifest",type=Path); p.add_argument("--show-sensors",action=argparse.BooleanOptionalAction,default=True); args=p.parse_args(); cfg=load_config(args.config); apply_style(cfg); rid=run_id(args.run_id); ensure_output_dirs(); recipes=cfg["questionA"]["recipes"] if args.mode=="questionA" else cfg["questionB"]["recipes"]; count=args.sensor_count or cfg["sensor_plan"]["default_count"]
    if args.model.lower()=="all": render_all_models(cfg,args,rid,recipes,count); return
    entries=cache_entries(args.cache_manifest,models=[args.model],recipes=recipes,sensor_count=count,snapshots=[args.snapshot]); loaded={}
    for e in entries:
        try: loaded[e["recipe"]]=load_cache(Path(e["cache_path"]))
        except Exception: pass
    first=next(iter(loaded.values()),None); ncols=5 if args.mode=="questionA" else 4; fig,axes=plt.subplots(1,ncols,figsize=(183/25.4,48/25.4),squeeze=False); axes=axes[0]
    if first is None:
        for ax in axes: mark_missing(ax,cfg=cfg)
        roi=[np.nan]*4
    else:
        arrays0,meta0=first; truth=arrays0["truth_phys"].reshape(-1); coords=arrays0["coords_phys"]; roi=args.manual_roi or automatic_gradient_roi(coords,truth); all_fields=[truth]
        for recipe in recipes:
            if recipe in loaded: all_fields.append(loaded[recipe][0]["recon_phys"].reshape(-1))
        vmin,vmax=np.nanquantile(np.concatenate(all_fields),[.01,.99]); errors=[np.abs(v-truth) for v in all_fields[1:]]; emax=np.nanquantile(np.concatenate(errors),.99) if errors else 1
        mode=args.render_mode or (cfg["rendering"]["default_zero_h_mode"] if args.mode=="questionB" else cfg["rendering"]["default_comparison_mode"])
        render_field(axes[0],coords,truth,mode=mode,cmap=cfg["rendering"]["cmap"],vmin=vmin,vmax=vmax,roi=roi); axes[0].set_title("Ground truth")
        if args.mode=="questionA":
            pos=1
            for recipe in recipes:
                if recipe not in loaded: mark_missing(axes[pos],cfg=cfg); mark_missing(axes[pos+1],cfg=cfg)
                else:
                    a,m=loaded[recipe]; pred=a["recon_phys"].reshape(-1); render_field(axes[pos],a["coords_phys"],pred,mode=mode,cmap=cfg["rendering"]["cmap"],vmin=vmin,vmax=vmax,roi=roi); axes[pos].set_title(f"{m.get('recipe_label',recipe)}\nL2={relative_l2(truth,pred):.3f}"); render_field(axes[pos+1],a["coords_phys"],np.abs(pred-truth),mode=mode,cmap=cfg["rendering"]["error_cmap"],vmin=0,vmax=emax,roi=roi); axes[pos+1].set_title("Absolute error")
                pos+=2
        else:
            for pos,recipe in enumerate(recipes,1):
                if recipe not in loaded: mark_missing(axes[pos],cfg=cfg); continue
                a,m=loaded[recipe]; pred=a["recon_phys"].reshape(-1); render_field(axes[pos],a["coords_phys"],pred,mode=mode,cmap=cfg["rendering"]["cmap"],vmin=vmin,vmax=vmax,roi=roi); axes[pos].set_title(f"{m.get('recipe_label',recipe)}\nL2={relative_l2(truth,pred):.3f}")
        if args.show_sensors:
            obs=arrays0["obs_indices"].astype(int)
            for ax in axes: ax.scatter(coords[obs,0],coords[obs,1],s=2,facecolors="none",edgecolors="white",linewidths=.25)
    result_folder="QuestionA_DataBenefit" if args.mode=="questionA" else "QuestionB_ZeroH"; roi_path=RESULTS_DIR/result_folder/artifact_name(f"ContourROI_{args.mode}",rid,"csv"); write_csv(roi_path,[{"mode":args.mode,"model":args.model,"snapshot_index":args.snapshot,"sensor_count":count,"roi_xmin":roi[0],"roi_xmax":roi[1],"roi_ymin":roi[2],"roi_ymax":roi[3],"roi_source":"manual" if args.manual_roi else "ground_truth_gradient_only","render_mode":args.render_mode or "default"}]); out=FIGURES_DIR/"_Contours"/f"{args.mode}_{args.model}_{args.render_mode or 'default'}_{rid}"; save_figure(fig,out,cfg); plt.close(fig); print(f"[OK] {out}\n[OK] {roi_path}")
if __name__=="__main__": main()

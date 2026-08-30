#!/usr/bin/env python
"""Plot observed/unobserved channels and exact sensor totals from YAML."""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib.pyplot as plt
from common.config import FIGURES_DIR, add_common_args, load_config, run_id
from common.figure_style import apply_style, figure_size, save_figure
from global_style import COLOR_MISSING_FACE, COLOR_MISSING_TEXT, SIZE_ANNOTATION, model_color


def main()->int:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter); add_common_args(p,models=False); p.add_argument("--formats",nargs="+",choices=["png","pdf","svg"]); p.add_argument("--dpi",type=int); args=p.parse_args(); cfg=load_config(args.config); rid=run_id(args.run_id); apply_style(cfg)
    conditions=list(cfg["conditions"]); mat=np.array([[f["index"] in cfg["conditions"][c]["cond_fields"] for f in cfg["fields"]] for c in conditions])
    fig,ax=plt.subplots(figsize=figure_size(cfg,"single",42),constrained_layout=True); ax.imshow(mat,cmap=plt.matplotlib.colors.ListedColormap([COLOR_MISSING_FACE,model_color("DMF-Gen")]),vmin=0,vmax=1,aspect="auto")
    ax.set_xticks(range(5),[f["label"] for f in cfg["fields"]]); labels=[f"{cfg['conditions'][c]['label']}  (N={sum(cfg['conditions'][c]['n_obs'])})" for c in conditions]; ax.set_yticks(range(3),labels)
    for i,c in enumerate(conditions):
        spec=cfg["conditions"][c]
        for j,f in enumerate(cfg["fields"]):
            count=spec["n_obs"][spec["cond_fields"].index(f["index"])] if f["index"] in spec["cond_fields"] else 0; ax.text(j,i,str(count) if count else "—",ha="center",va="center",color="white" if count else COLOR_MISSING_TEXT,fontsize=SIZE_ANNOTATION)
    ax.set_xlabel("Measurement channel (numbers are sensors)"); ax.tick_params(length=0); base=FIGURES_DIR/"ConditionMatrix"/f"ConditionMatrix_{rid}"; save_figure(fig,base,cfg,args.formats,args.dpi); plt.close(fig); print(f"[OK] {base}"); return 0
if __name__=="__main__": raise SystemExit(main())

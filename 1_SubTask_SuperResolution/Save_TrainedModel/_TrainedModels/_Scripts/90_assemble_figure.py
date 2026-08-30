#!/usr/bin/env python
"""Assemble existing panel images using YAML layout; never recompute data."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.config import FIGURES_DIR, add_common_args, ensure_output_dirs, load_config, run_id
from common.figure_style import apply_style, mark_missing, save_figure

def find_panel(source):
    root=FIGURES_DIR/source
    candidates=sorted(root.glob("*.png")) if root.is_dir() else sorted(FIGURES_DIR.glob(f"{source}*.png"))
    return candidates[-1] if candidates else None

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p,models=False); p.add_argument("--layout",choices=["main","si"],default="main"); args=p.parse_args(); cfg=load_config(args.config); apply_style(cfg); ensure_output_dirs(); rid=run_id(args.run_id); layout=cfg["assembly"][args.layout]; panels=layout["panels"]; ncols=2 if args.layout=="si" else 2; nrows=(len(panels)+ncols-1)//ncols; fig,axes=plt.subplots(nrows,ncols,figsize=(layout["width_mm"]/25.4,layout["height_mm"]/25.4),squeeze=False)
    for ax,panel in zip(axes.flat,panels):
        path=find_panel(panel["source"])
        if path is None: mark_missing(ax,cfg=cfg)
        else:
            image=plt.imread(path); ax.imshow(image); ax.set_axis_off()
        ax.text(.005,.995,panel["label"],transform=ax.transAxes,fontweight="bold",fontsize=cfg["figure_style"]["font_sizes"]["panel"],ha="left",va="top")
    for ax in list(axes.flat)[len(panels):]: ax.set_axis_off()
    fig.subplots_adjust(left=.02,right=.99,bottom=.02,top=.99,wspace=.05,hspace=.08); out=FIGURES_DIR/"Assembled"/f"{layout['output_name']}_{rid}"; save_figure(fig,out,cfg,formats=cfg["figure_style"]["paper_formats"],dpi=cfg["figure_style"]["paper_dpi"],bbox_inches=None); plt.close(fig); print(f"[OK] {out}")
if __name__=="__main__": main()

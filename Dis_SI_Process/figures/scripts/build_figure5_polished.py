#!/usr/bin/env python
"""Additive GridSpec redesign of the validated Figure 5 main composition.

No evaluation, SI generation or edits to prior figure releases are performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, PolyCollection
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator, LogLocator, NullLocator, FuncFormatter
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd
import yaml
import fitz
from PIL import Image
import build_figure5_v7r2 as R2
import build_figure5_polish_style as STYLE

ROOT=R2.ROOT
REPO=R2.REPO
SOURCE_STAMP="20260910_1707"
DEFAULT_OUT=ROOT/"figures/generated/figure5_precision_log_revision_20260910"
TRAINING_REPEATS=ROOT/"results/ValidationV51/TrainingFootprint/training_footprint_common_b32_v51/benchmark_repeats.csv"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scientific_coordinates(axes):
    data=[]
    for ax in axes:
        data.append({"lines":[[np.asarray(line.get_xdata()).tolist(),np.asarray(line.get_ydata()).tolist()] for line in ax.lines],
            "collections":[{"offsets":np.asarray(col.get_offsets()).tolist(),"paths":[p.vertices.tolist() for p in col.get_paths()]} for col in ax.collections]})
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def soften_clouds(axes):
    records=[]
    for ax in axes:
        for col in ax.collections:
            if isinstance(col,PathCollection) and len(col.get_offsets())>=50:
                records.append({"n":len(col.get_offsets()),"old_alpha":col.get_alpha(),"old_sizes":col.get_sizes().tolist()})
                col.set_sizes([.55]);col.set_alpha(.045);col.set_edgecolor("none")
    return records


def place(ax,spec,fig):
    ax.set_subplotspec(spec)
    ax.set_position(spec.get_position(fig))


def tag(fig,ax,label):
    pos=ax.get_position()
    fig.text(pos.x0,pos.y1+.01,label,fontsize=8.5,fontweight="bold",va="bottom")


def resource_bars(axes,table,colors,memory,repeats):
    records=[]
    specs=[("training_update_ms","training_update_q25_ms","training_update_q75_ms","Training update time\n(ms / update)",1.,[0,300,600]),
        ("training_peak_allocated_mib",None,None,"Training memory\n(GiB)",1024.,[0,8,16]),
        ("inference_latency_ms","inference_latency_q25_ms","inference_latency_q75_ms","Inference time\n(ms)",1.,[0,10,20])]
    for ax,(column,low,high,label,scale,ticks) in zip(axes[:3],specs):
        ax.clear()
        for y,row in enumerate(table.itertuples(index=False)):
            value=float(getattr(row,column))/scale;color=colors[row.method]
            ax.barh(y,value,height=.48,color=color,alpha=.8,zorder=2)
            interval=None
            if low:
                lo=float(getattr(row,low))/scale;hi=float(getattr(row,high))/scale
            else:
                selected=repeats[repeats.method.eq(row.method)&repeats.stage_ordinal.eq(int(row.training_memory_stage))]
                if len(selected)!=100:raise ValueError('Training-memory repeat coverage differs from accepted protocol')
                lo=float(selected.peak_allocated_mib.min())/scale;hi=float(selected.peak_allocated_mib.max())/scale
                if not lo<=value<=hi:raise ValueError('Training-memory bar differs from measured repeat range')
            ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt="none",ecolor=color,capsize=1.6,elinewidth=.8,capthick=.8,zorder=4)
            interval=[lo,hi]
            records.append({"method":row.method,"metric":column,"value":value,"interval":interval,"interval_kind":"timing IQR" if low else "100 measured updates: min-max (no repeat dispersion)","color":color})
        ax.set_xlim(0,float(table[column].max())/scale*1.11)
        ax.set_xticks(ticks);ax.set_xlabel(label,fontsize=5.9,labelpad=4)
    ax=axes[-1];ax.clear()
    for y,row in enumerate(table.itertuples(index=False)):
        color=colors[row.method];m=memory.loc[row.method]
        model=float(row.model_state_mib);peak=float(row.inference_peak_allocated_mib)
        lo=float(m.inference_peak_allocated_min_mib);hi=float(m.inference_peak_allocated_max_mib)
        if not np.isclose(float(m.inference_peak_allocated_mib),peak,rtol=0,atol=1e-10):raise ValueError("Inference memory source mismatch")
        ax.barh(y-.14,model,height=.24,color=color,zorder=3)
        ax.errorbar(model,y-.14,xerr=0,fmt="none",ecolor=color,capsize=1.1,elinewidth=.8,capthick=.8,zorder=4)
        ax.barh(y+.14,peak,height=.24,facecolor="white",edgecolor=color,linewidth=.8,zorder=2)
        ax.errorbar(peak,y+.14,xerr=[[peak-lo],[hi-peak]],fmt="none",ecolor=color,capsize=1.5,elinewidth=.8,capthick=.8,zorder=4)
        records.append({"method":row.method,"metric":"model_state_mib","value":model,"interval":[model,model],"interval_kind":"fixed tensor inventory; zero-width range","color":color})
        records.append({"method":row.method,"metric":"inference_peak_allocated_mib","value":peak,"interval":[lo,hi],"interval_kind":"measured repeat min-max","color":color})
    ax.set_xlim(0,table.inference_peak_allocated_mib.max()*1.1)
    ax.set_xticks([0,250,500]);ax.set_xlabel("Inference memory\n(MiB)",fontsize=5.9,labelpad=4)
    first=table.iloc[0]
    ax.annotate("Model",(first.model_state_mib,-.14),xytext=(0,8),textcoords="offset points",ha="left",fontsize=5.5,color="#444444")
    ax.annotate("Peak",(first.inference_peak_allocated_mib,.14),xytext=(0,8),textcoords="offset points",ha="right",fontsize=5.5,color="#444444")
    for ax in axes:
        ax.set_ylim(len(table)-.5,-.85);ax.set_yticks(range(len(table)));ax.tick_params(axis="y",labelleft=False,length=0)
        ax.tick_params(axis="x",labelsize=5.5,length=2,pad=2)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_linewidth(.6)
        ax.set_axisbelow(True);ax.grid(axis="x",color="#E8EAED",lw=.4)
    return records


def update_accuracy(ax, f, summary, colors, markers):
    full=summary[summary.method.eq('A0')&summary.field.eq('Unobserved_mean')]
    if len(full)!=1:raise ValueError('Missing or ambiguous A0 accuracy source')
    full=full.iloc[0];updated=f.copy()
    for target,source in [('mean_unobserved_relative_l2','mean'),('error_ci_low','block20_ci95_low'),('error_ci_high','block20_ci95_high')]:
        updated.loc[updated.method.eq('DMF-Gen'),target]=float(full[source])
    ax.clear()
    for y,row in enumerate(updated.itertuples(index=False)):
        color=colors[row.method]
        ax.errorbar(row.mean_unobserved_relative_l2,y,
            xerr=[[row.mean_unobserved_relative_l2-row.error_ci_low],[row.error_ci_high-row.mean_unobserved_relative_l2]],
            fmt=markers[row.method],ms=4.9 if row.method=='DMF-Gen' else 4.2,color=color,
            mec='white',mew=.55,elinewidth=1.15,capsize=1.5,zorder=4)
        ax.text(row.error_ci_high+.008,y-.16,f'{row.mean_unobserved_relative_l2:.3f}',color=color,fontsize=4.8,va='bottom')
    ax.set_yticks(range(len(updated)),updated.method)
    for tick,method in zip(ax.get_yticklabels(),updated.method):
        tick.set_color(colors[method]);tick.set_fontweight('bold' if method=='DMF-Gen' else 'normal')
    ax.set_xlim(.08,.49);ax.set_xticks([.1,.2,.3,.4])
    ax.spines[['left','right','top']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.set_axisbelow(True);ax.grid(axis='x',color='#E8EAED',lw=.4)
    return updated


# The previous middle row duplicated long rotated model labels on both
# distributions. Its explicit GridSpec weights did not strictly constrain
# the three scientific rows to compact proportions, and a larger canvas
# amplified the excess height and width. Share category labels and enforce
# [1, 2.2, 1.2]; narrower panel-d allocation reduces physical violin spacing.

def set_distribution_scale(axes, tables, scale):
    for ax, name, field in zip(axes, ['reconstruction_states','highband_states'], ['Unobserved_mean','U1']):
        ax.set_yscale(scale)
        ax.yaxis.set_minor_locator(NullLocator())
        if scale == 'log':
            values=tables[name].loc[tables[name].field.eq(field)&tables[name].method.isin(R2.MAIN_METHODS),'value'].to_numpy(float)
            if not np.isfinite(values).all() or np.any(values<=0):
                raise ValueError(f'Log distribution requires positive finite source errors: {field}')
            # Preserve every observation; suppress Gaussian support at/below zero.
            ax.set_ylim(float(values.min())*.8, ax.dataLim.ymax*1.05)
            ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5),numticks=6))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
        else:
            ax.set_ylim(0,ax.dataLim.ymax*1.05)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--height-mm",type=float,default=260.)
    parser.add_argument("--row-ratios",type=float,nargs=3,default=[1.25,2.0,1.5])
    parser.add_argument("--middle-ratios",type=float,nargs=2,default=[6.,4.])
    parser.add_argument("--middle-wspace",type=float,default=.17)
    args=parser.parse_args();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    # Preflight both variants so an existing export cannot cause a partial run.
    if any((out/('figure5_'+scale+ext)).exists() for scale in ['log'] for ext in ['.svg','.png','.pdf','_preview.png']):
        raise FileExistsError('Choose a fresh output directory; existing exports are never overwritten')
    tables=R2.load_tables(SOURCE_STAMP)
    accepted=R2.V7.accepted_renderer();v1=accepted.V1;base=v1.load_base()
    cfg=yaml.safe_load((ROOT/"configs/figure5_v6.yaml").read_text())
    v1.configure_nature_style(base,cfg);STYLE.configure_style()
    plt.rcParams.update({'svg.hashsalt':'figure5-layout-redesign','pdf.fonttype':42,'pdf.compression':6})
    paths=base.require_sources(cfg);uq=cfg['paper_contract']['generative_method_order']
    a,sa,b,sb=base.load_panel_ab(paths['v5_display'],uq)
    c=base.load_panel_c(paths['v51_selective_risk'],cfg)
    f=base.derive_panel_d(paths['v51_scorecard'],paths['v51_scorecard_stages'],paths['inference_memory'],cfg['paper_contract']['scorecard_method_order'])
    memory=pd.read_csv(paths['inference_memory']).set_index('method')
    fig=base.make_composed(a,sa,b,sb,c,f,cfg);v1.refine_composed(fig)
    archived=json.loads((ROOT/'results/derived/20260904_1200/qa.json').read_text())['geometry_digests']['composed']['after']
    if v1.geometry_digest(fig)!=archived:raise ValueError('Inherited figure evidence differs from accepted geometry')
    top=fig.axes[:3];score=fig.axes[3:];preserved_before=scientific_coordinates(top)
    R2.V7.clear_figure_labels(fig);R2.V7.polish_inherited(fig)
    for ax in fig.axes:
        ax.set_title('');ax.set_title('',loc='left');ax.set_title('',loc='right')
        if ax.get_legend() is not None:ax.get_legend().remove()
    fig.set_size_inches(280/25.4,args.height_mm/25.4)
    # Matched spacer rows make the row-1/row-2 and row-2/row-3 gaps identical.
    # The shorter canvas removes the former oversized lower gutter while
    # preserving the physical scale of the three scientific rows.
    row_gap=.55
    gs=GridSpec(5,1,figure=fig,left=.16,right=.985,bottom=.085,top=.955,height_ratios=[args.row_ratios[0],row_gap,args.row_ratios[1],row_gap,args.row_ratios[2]],hspace=0)
    topgs=gs[0].subgridspec(1,5,width_ratios=[1,.12,1,.32,1],wspace=0)
    for ax,spec in zip(top,[topgs[0],topgs[2],topgs[4]]):place(ax,spec,fig)
    # The c ordinate needs a narrow offset; a/b have a common method-label rail.
    top[2].set_ylabel('Relative retained-set error',fontsize=5.8,labelpad=0)
    top[2].set_xlabel('Retained least-uncertain states',fontsize=6,labelpad=3)
    for ax in top:ax.tick_params(axis='both',labelsize=5.8,pad=2)
    middlegs=gs[2].subgridspec(1,2,width_ratios=args.middle_ratios,wspace=args.middle_wspace)
    leftgs=middlegs[0].subgridspec(2,1,height_ratios=[1,1],hspace=.55)
    d=fig.add_subplot(leftgs[0]);h=fig.add_subplot(leftgs[1],sharex=d);e=fig.add_subplot(middlegs[1])
    distribution_records=[];spectrum_records=[]
    STYLE.distribution(d,tables['reconstruction_states'],tables['reconstruction_summary'],R2.MAIN_METHODS,'Unobserved_mean',r'Unobserved-field relative L$_2$',fontsize=5.9,record=distribution_records,wrap_labels=False)
    STYLE.distribution(h,tables['highband_states'],tables['highband_summary'],R2.MAIN_METHODS,'U1',r'High-band relative L$_2$ ($U_1$)',fontsize=5.9,record=distribution_records,wrap_labels=False)
    # Hide only the upper labels; clearing a shared formatter would also
    # erase the lower labels. Narrowing the panel makes category gaps compact.
    d.tick_params(axis='x',which='both',labelbottom=False,bottom=False)
    d.set_xlabel('')
    d.set_xlim(-.48,len(R2.MAIN_METHODS)-.52)
    for tick in h.get_xticklabels():tick.set_rotation(30);tick.set_ha('right')
    STYLE.spectra(e,tables['spectra_population'],'U1',fontsize=6.2,legend=True,record=spectrum_records)
    high_shells=tables['spectra_population'].loc[
        tables['spectra_population'].field.eq('U1') &
        tables['spectra_population'].high_band.astype(str).str.lower().isin(['true','1','1.0']), 'shell_index']
    e.axvline(float(high_shells.min())-.5,color='#718096',lw=.55,ls=(0,(3,3)),alpha=.65,zorder=1)
    # Legend occupies the empty upper middle; band annotation is separate.
    e.legend(ncol=1,loc='upper center',bbox_to_anchor=(.46,.99),
             labelspacing=.5,handlelength=2.6,fontsize=9)
    for text in list(e.texts):
        if text.get_text()=='High-band':text.remove()
    band_left=float(high_shells.min())-.5
    band_right=float(high_shells.max())+.5
    e.set_xlim(right=band_right)  # Show the entire final shaded shell edge.
    band_text=e.text((band_left+band_right)/2,.88,'High-band region',
                    transform=e.get_xaxis_transform(),ha='center',va='bottom',
                    fontsize=10,fontweight='normal',color='#46556A')
    e.annotate('',xy=(band_right,.86),xytext=(band_left,.86),
               xycoords=e.get_xaxis_transform(),textcoords=e.get_xaxis_transform(),
               arrowprops=dict(arrowstyle='<->',color='#46556A',lw=1.1,shrinkA=0,shrinkB=0))
    for ax in [d,h]:
        for annotation in ax.texts:annotation.set_position((3,3));annotation.set_fontsize(5.5)
        ax.yaxis.set_major_locator(MaxNLocator(4))
    cloud_changes=soften_clouds([*top,d,h])
    fgs=gs[4].subgridspec(1,5,width_ratios=[1.65,1,1,1,1.15],wspace=.28)
    for ax,spec in zip(score,fgs):place(ax,spec,fig)
    resource_records=resource_bars(score[1:],f,cfg['style']['method_colors'],memory,pd.read_csv(TRAINING_REPEATS))
    accuracy=update_accuracy(score[0],f,tables['reconstruction_summary'],cfg['style']['method_colors'],cfg['style']['method_markers'])
    score[0].set_ylim(len(f)-.5,-.85)
    score[0].set_xlabel('Unobserved-field\n'+r'(relative L$_2$)',fontsize=5.9,labelpad=4)
    score[0].tick_params(axis='x',labelsize=5.5,pad=2)
    score[0].tick_params(axis='y',labelsize=5.8,pad=3)
    STYLE.standardize(fig)
    fig.canvas.draw()
    # Anchor a/d/f at the leftmost visible figure text, rather than the axes.
    renderer=fig.canvas.get_renderer()
    left_edge=min(text.get_window_extent(renderer).x0 for ax in fig.axes for text in [ax.yaxis.label,*ax.get_yticklabels()] if text.get_visible() and text.get_text())/fig.bbox.width
    for label,ax in zip('abcdef',[*top,d,e,score[0]]):
        x=left_edge if label in 'adf' else ax.get_position().x0
        fig.text(x,ax.get_position().y1+.01,label,fontsize=12,fontweight='bold',va='bottom')
    if scientific_coordinates(top)!=preserved_before:
        raise ValueError('Inherited a/b/c plotting coordinates changed')
    if float(accuracy.iloc[0].mean_unobserved_relative_l2)!=distribution_records[0]['mean']:
        raise ValueError('DMF-Gen mean differs between panels d and f')
    # Both exports reuse the same artists and saved evidence; only d's axis
    # transform changes. No derived tables, reports, or QA files are written.
    for scale in ['log']:
        set_distribution_scale([d,h],tables,scale)
        # Match the high-band label to the standard 10 pt axis-label size
        # after the shared standardizer has touched all text artists.
        STYLE.standardize(fig)
        band_text.set_fontsize(10)
        band_text.set_fontweight('normal')
        for ext in ['pdf','svg','png']:
            fig.savefig(out/('figure5_'+scale+'.'+ext),dpi=600,bbox_inches='tight',pad_inches=.05,facecolor='white')
        fig.savefig(out/('figure5_'+scale+'_preview.png'),dpi=150,bbox_inches='tight',pad_inches=.05,facecolor='white')
    plt.close(fig)
    print(f'Figure exports written to {out}: figure5_log (PDF/PNG/SVG).')

if __name__=='__main__':main()

#!/usr/bin/env python
"""Figure 5 V7R2: inherited UQ, saved-field distributions and spectra."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
import numpy as np
import pandas as pd
import yaml
import build_figure5_v7_ablation as V7

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT.parent
FIELDS=['CH4','CO','T','U1','p']
FIELD_EXPORT={'CH4':'Y_CH4','CO':'Y_CO','T':'T','U1':'U1','p':'p'}
FIELD_LABEL={'CH4':r'$Y_{\mathrm{CH}_4}$','CO':r'$Y_{\mathrm{CO}}$','T':r'$T$','U1':r'$U_1$','p':r'$p$'}
MAIN_METHODS=['A0','A2','A3','A5','A4','Senseiver']
STYLES=yaml.safe_load((ROOT/'configs/figure5_v7_ablation.yaml').read_text())['styles']
STYLES['Senseiver']={'label':'Senseiver','color':'#8D99AE','marker':'h'}
LINE_STYLES={'A0':'-','A2':'--','A3':':','A5':'-.','A4':'-','A1':'--','Senseiver':(0,(5,1,1,1))}

def configure_style():
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':6.6,'axes.labelsize':6.6,'axes.titlesize':6.6,'xtick.labelsize':6,'ytick.labelsize':6,
        'axes.linewidth':.65,'svg.fonttype':'none','svg.hashsalt':'figure5-v7r2','legend.frameon':False,
        'savefig.transparent':False})

def load_tables(timestamp):
    d=ROOT/'results/derived'/timestamp
    qa=json.loads((d/'source_qa.json').read_text())
    if qa['status']!='pass':raise RuntimeError('V7R2 source gate has not passed; no formal rendering')
    return {name:pd.read_csv(d/(name+'.csv'),float_precision='round_trip') for name in
        ['reconstruction_states','reconstruction_summary','highband_states','highband_summary','spectra_population']}

def draw_distribution(ax,states,summary,methods,field,xlabel,fontsize=6,record=None,wrap_labels=True):
    """All state points, quartile box, 1.5-IQR whiskers and source mean/CI."""
    data=states[states.field.eq(field)]
    positions=np.arange(len(methods))[::-1]
    labels=[];records=[];all_values=[]
    for method,y in zip(methods,positions):
        selected=data[data.method.eq(method)].sort_values('snapshot')
        values=selected.value.to_numpy(float)
        if len(values)!=1000 or not np.isfinite(values).all():raise ValueError(f'Incomplete distribution {field}/{method}')
        r=summary[summary.field.eq(field)&summary.method.eq(method)]
        if len(r)!=1:raise ValueError(f'Ambiguous summary {field}/{method}')
        r=r.iloc[0];style=STYLES[method];color=style['color']
        seed=int(hashlib.sha256((method+field).encode()).hexdigest()[:8],16)
        jitter=np.random.default_rng(seed).uniform(-.22,.22,len(values))
        ax.scatter(values,y+jitter,s=1.4,alpha=.12,color=color,edgecolors='none',zorder=1)
        ax.boxplot([values],positions=[y],orientation='horizontal',widths=.38,showfliers=False,whis=1.5,
            patch_artist=True,manage_ticks=False,boxprops={'facecolor':color,'alpha':.18,'edgecolor':color,'linewidth':.7},
            medianprops={'color':color,'linewidth':1},whiskerprops={'color':color,'linewidth':.7},capprops={'color':color,'linewidth':.7})
        mean=float(r['mean']);lo=float(r['block20_ci95_low']);hi=float(r['block20_ci95_high'])
        ax.errorbar(mean,y,xerr=[[mean-lo],[hi-mean]],fmt=style['marker'],color=color,
            mfc='white' if method in ['A0','Senseiver'] else color,ms=4.1,mew=.85,lw=.75,capsize=1.7,zorder=5)
        annotation=f'{mean:.4f}' if field=='Unobserved_mean' else f'{mean:.2f}'
        ax.annotate(annotation,(mean,y),xytext=(3,5.5),textcoords='offset points',fontsize=fontsize,ha='left',va='bottom',color=color)
        label=style['label']
        if wrap_labels:label=label.replace('No sensor feedback','No sensor\nfeedback').replace('No local conditioning','No local\nconditioning').replace('Local-only conditioning','Local-only\nconditioning').replace('IID Gaussian prior','IID Gaussian\nprior')
        labels.append(label);all_values.extend(values)
        records.append({'method':method,'field':field,'n':len(values),'mean':mean,'ci_low':lo,'ci_high':hi,
            'q25':float(np.quantile(values,.25)),'median':float(np.median(values)),'q75':float(np.quantile(values,.75)),
            'annotation':annotation,'y':int(y),'jitter_seed':seed,'state_source_rows':selected.index.tolist()})
    ax.set_yticks(positions,labels)
    for tick,m in zip(ax.get_yticklabels(),methods):tick.set_color(STYLES[m]['color'])
    ax.set_ylim(-.55,len(methods)-.25)
    vmax=max(all_values);vmin=min(all_values)
    # Keep all rare extremes visible; only the axis transform changes.
    if vmax/max(float(np.median(all_values)),1e-12)>30 and vmin>0:
        ax.set_xscale('log');ax.set_xlim(vmin*.8,vmax*1.15)
    else:ax.set_xlim(0,vmax*1.10)
    ax.set_xlabel(xlabel,fontsize=fontsize+.3,labelpad=3)
    V7.base_axis(ax);ax.tick_params(labelsize=fontsize)
    if record is not None:record.extend(records)
    return records

def draw_spectra(ax,pop,field,methods=MAIN_METHODS,fontsize=6,legend=False,record=None):
    records=[]
    for method in ['Truth',*methods]:
        rows=pop[pop.field.eq(field)&pop.method.eq(method)].sort_values('shell_index')
        if len(rows)!=198 or not np.isfinite(rows['median']).all() or not (rows['median']>0).all():raise ValueError(f'Incomplete positive spectrum {field}/{method}')
        color='#222222' if method=='Truth' else STYLES[method]['color']
        label='Truth' if method=='Truth' else STYLES[method]['label']
        marker_options={} if method=='Truth' else {'marker':STYLES[method]['marker'],'markersize':2.7,
            'markerfacecolor':'white' if method in ['A0','Senseiver'] else color,'markeredgewidth':.55,
            'markevery':(5+5*list(methods).index(method),42)}
        ax.plot(rows.shell_index,rows['median'],color=color,lw=1.05 if method=='Truth' else .8,
            ls='-' if method=='Truth' else LINE_STYLES[method],label=label,zorder=5 if method=='Truth' else 3,**marker_options)
        records.append({'method':method,'field':field,'shell_index':rows.shell_index.tolist(),'median_power':rows['median'].tolist(),
            'statistic':'median of absolute per-state shell-mean energy; no IQR bands in main panel'})
    mask=rows.high_band.astype(str).str.lower().isin(['true','1','1.0'])
    ax.axvspan(float(rows.loc[mask,'shell_index'].min())-.5,float(rows.shell_index.max())+.5,color='#64748B',alpha=.08,lw=0)
    ax.set_yscale('log');ax.set_xlim(rows.shell_index.min(),rows.shell_index.max())
    ax.set_xlabel('Frequency shell',fontsize=fontsize+.3,labelpad=2)
    ax.set_ylabel('Spectral power '+FIELD_LABEL[field],fontsize=fontsize+.3,labelpad=2)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=fontsize,length=2,width=.6,pad=1.5)
    if legend:ax.legend(fontsize=fontsize,ncol=2,loc='best',handlelength=2,columnspacing=.7,labelspacing=.2)
    else:
        # A compact legend identifies truth; method colors agree with the rows below.
        ax.legend(handles=[Line2D([],[],color='#222222',lw=1,label='Truth')],fontsize=fontsize,loc='upper right',handlelength=1.4,borderpad=.15)
    if record is not None:record.extend(records)
    return records

def text_bounds(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=fig.bbox;skipped=set();outside=[]
    for ax in fig.axes:
        for axis,limits in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            low,high=sorted(limits)
            for tick in [*axis.get_major_ticks(),*axis.get_minor_ticks()]:
                if not low<=tick.get_loc()<=high:skipped.update([id(tick.label1),id(tick.label2)])
    for txt in fig.findobj(Text):
        if id(txt) in skipped or not txt.get_visible() or not txt.get_text() or (txt.axes is not None and not txt.axes.get_visible()):continue
        box=txt.get_window_extent(renderer)
        if box.width and box.height and (box.x0<bounds.x0-1 or box.x1>bounds.x1+1 or box.y0<bounds.y0-1 or box.y1>bounds.y1+1):outside.append(txt.get_text())
    return outside

def save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    outside=text_bounds(fig)
    fig.savefig(path,facecolor='white',metadata={'Date':None})
    fig.savefig(path.with_suffix('.png'),dpi=600,facecolor='white')
    if 'composed' in path.name:fig.savefig(path.with_name(path.stem+'_print_proof.png'),dpi=150,facecolor='white')
    entry={'path':str(path.relative_to(REPO)),'off_canvas_text':outside,'size_mm':(fig.get_size_inches()*25.4).tolist()}
    plt.close(fig);return entry

def memory_labels(ax):
    legend=ax.get_legend()
    if legend is not None:legend.remove()
    lines=[line for line in ax.lines if len(line.get_xdata())==1 and line.get_marker() not in [None,'None','','|','_']]
    if len(lines)<2:raise ValueError('Cannot identify inherited memory endpoints')
    for label,line,align in zip(['Model','Peak'],lines[:2],['center','right']):
        ax.annotate(label,(float(line.get_xdata()[0]),float(line.get_ydata()[0])),xytext=(0,7),textcoords='offset points',
            ha=align,va='bottom',fontsize=5.8,color='#555555')

def letter(fig,label,x,y):fig.text(x,y,label,fontsize=8.5,fontweight='bold',va='bottom')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--timestamp',required=True);parser.add_argument('--strict-formal',action='store_true')
    args=parser.parse_args();tables=load_tables(args.timestamp);configure_style()
    out=ROOT/'figures/generated'/args.timestamp;derived=ROOT/'results/derived'/args.timestamp
    accepted=V7.accepted_renderer();v1=accepted.V1;base=v1.load_base();old=yaml.safe_load((ROOT/'configs/figure5_v6.yaml').read_text())
    v1.configure_nature_style(base,old);configure_style();paths=base.require_sources(old)
    uq=old['paper_contract']['generative_method_order'];a,sa,b,sb=base.load_panel_ab(paths['v5_display'],uq)
    c=base.load_panel_c(paths['v51_selective_risk'],old)
    f=base.derive_panel_d(paths['v51_scorecard'],paths['v51_scorecard_stages'],paths['inference_memory'],old['paper_contract']['scorecard_method_order'])
    archived=json.loads((ROOT/'results/derived/20260904_1200/qa.json').read_text())['geometry_digests']
    result={'status':'pending','figures':[],'inherited_geometry':{},'panel_map':{'a':'normalized CRPS','b':'spread-error Spearman association','c':'selective reconstruction','d':'ablation error distributions','e':'population spectra and high-band residual distributions','f':'accuracy and computational footprint'},'selected_field':'U1','new_checkpoint_policy':'last','plot_coordinates':[]}
    def inherited(fig,name,original):
        before=v1.geometry_digest(fig)
        if before!=archived[original]['after']:raise ValueError(f'Inherited geometry differs: {name}')
        V7.clear_figure_labels(fig);V7.polish_inherited(fig)
        for ax in fig.axes:
            if ax.get_legend() is not None:ax.get_legend().remove()
        result['inherited_geometry'][name]={'archived':before,'after':v1.geometry_digest(fig)}
    builders={'a':lambda:base.make_standalone_ab('a',a,sa,old),'b':lambda:base.make_standalone_ab('b',b,sb,old),'c':lambda:base.make_standalone_c(c,old),'f':lambda:base.make_standalone_d(f,old)}
    for name,builder in builders.items():
        original='d' if name=='f' else name;fig=builder();v1.refine_standalone(fig,original);inherited(fig,name,original)
        if name=='f':
            fig.set_size_inches(183/25.4,72/25.4);accepted.V4.V3.V2.position_scorecard_axes(fig.axes,left=.12,right=.98,bottom=.16,top=.78);memory_labels(fig.axes[-1]);letter(fig,'f',.12,.92)
        else:
            fig.set_size_inches(89/25.4,72/25.4);fig.axes[0].set_position([.29,.18,.68,.72]);letter(fig,name,.06,.92)
            if name=='b':fig.axes[0].tick_params(axis='y',labelleft=True);fig.axes[0].set_yticklabels(uq)
            if name=='c':
                fig.set_size_inches(89/25.4,88/25.4)
                fig.axes[0].set_position([.20,.31,.77,.59])
                curves=[line for line in fig.axes[0].lines if len(line.get_xdata())==len(c.coverage_fraction.unique())]
                selected=list(zip(curves,uq))
                if len(selected)!=len(uq):raise ValueError('Standalone selective-risk legend is incomplete')
                fig.legend([h for h,label in selected],[label for h,label in selected],loc='lower center',
                    bbox_to_anchor=(.57,.025),ncol=2,fontsize=5.8,handlelength=1.8,columnspacing=1)
        result['figures'].append(save(fig,out/f'fig5{name}_v7r2_{args.timestamp}.svg'))
    fig=base.make_composed(a,sa,b,sb,c,f,old);v1.refine_composed(fig);inherited(fig,'composed','composed')
    fig.set_size_inches(183/25.4,205/25.4)
    fig.axes[0].set_position([.12,.78,.215,.17]);fig.axes[1].set_position([.405,.78,.21,.17]);fig.axes[2].set_position([.715,.78,.27,.17])
    accepted.V4.V3.V2.position_scorecard_axes(fig.axes[3:],left=.12,right=.98,bottom=.06,top=.265);memory_labels(fig.axes[-1])
    draw_distribution(fig.add_axes([.12,.35,.495,.31]),tables['reconstruction_states'],tables['reconstruction_summary'],MAIN_METHODS[:-1],'Unobserved_mean',r'Unobserved-field relative L$_2$',record=result['plot_coordinates'])
    draw_spectra(fig.add_axes([.765,.535,.22,.135]),tables['spectra_population'],'U1',record=result['plot_coordinates'])
    draw_distribution(fig.add_axes([.765,.35,.22,.135]),tables['highband_states'],tables['highband_summary'],MAIN_METHODS,'U1',r'High-band relative L$_2$',fontsize=5.6,record=result['plot_coordinates'])
    for label,x,y in [('a',.12,.962),('b',.405,.962),('c',.715,.962),('d',.12,.69),('e',.69,.69),('f',.12,.305)]:letter(fig,label,x,y)
    result['layout']={'canvas_mm':[183,205],'top_b_c_axis_gap_fraction':.10,'middle_d_e_axis_gap_fraction':.15,'prior_v7_top_b_c_gap_fraction':.20,'prior_v7_middle_d_e_gap_fraction':.20}
    (derived/'main_artist_coordinates.json').write_text(json.dumps(V7.artist_coordinates(fig),indent=2)+'\n')
    result['figures'].append(save(fig,out/f'fig5_composed_v7r2_{args.timestamp}.svg'))
    fig=plt.figure(figsize=(140/25.4,82/25.4));draw_distribution(fig.add_axes([.24,.17,.73,.74]),tables['reconstruction_states'],tables['reconstruction_summary'],MAIN_METHODS[:-1],'Unobserved_mean',r'Unobserved-field relative L$_2$',wrap_labels=False);letter(fig,'d',.04,.94)
    result['figures'].append(save(fig,out/f'fig5d_v7r2_{args.timestamp}.svg'))
    for field in FIELDS:
        fig=plt.figure(figsize=(89/25.4,112/25.4));draw_spectra(fig.add_axes([.22,.61,.75,.33]),tables['spectra_population'],field,legend=True)
        draw_distribution(fig.add_axes([.31,.12,.66,.34]),tables['highband_states'],tables['highband_summary'],MAIN_METHODS,field,r'High-band relative L$_2$',wrap_labels=False);letter(fig,'e',.05,.96)
        if field=='U1':
            # Default main e is also exported with its ordinary panel name.
            fig.savefig(out/f'fig5e_v7r2_{args.timestamp}.svg',facecolor='white',metadata={'Date':None})
            fig.savefig(out/f'fig5e_v7r2_{args.timestamp}.png',dpi=600,facecolor='white')
            result['figures'].append({'path':str((out/f'fig5e_v7r2_{args.timestamp}.svg').relative_to(REPO)),'off_canvas_text':text_bounds(fig),'size_mm':[89,112]})
        result['figures'].append(save(fig,out/f'fig5e_{FIELD_EXPORT[field]}_v7r2_{args.timestamp}.svg'))
    result['status']='pass' if all(not item['off_canvas_text'] for item in result['figures']) else 'layout_review_required'
    (derived/'main_plot_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'figures':len(result['figures'])}))
    if args.strict_formal and result['status']!='pass':raise SystemExit(1)

if __name__=='__main__':main()

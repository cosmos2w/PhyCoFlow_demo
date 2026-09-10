#!/usr/bin/env python
"""Recompose the accepted benchmark artists with frozen stochastic controls."""
from __future__ import annotations

import argparse
import importlib.util
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

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
CONFIG = ROOT / 'configs/figure5_v7_ablation.yaml'
STEMS = {'a': 'crps', 'b': 'spread_error', 'c': 'architecture_prior',
         'd': 'selective_risk', 'e': 'highband_velocity', 'f': 'accuracy_footprint'}


def accepted_renderer():
    manifest = json.loads((ROOT / 'results/derived/20260904_1200/build_manifest.json').read_text())
    spec = importlib.util.spec_from_file_location('accepted_figure5_renderer', REPO / manifest['renderer'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one(table, **filters):
    rows = table
    for key, value in filters.items():
        rows = rows.loc[rows[key].eq(value)]
    if len(rows) != 1:
        raise ValueError(f'Expected one source row: {filters}, got {len(rows)}')
    row = rows.iloc[0]
    values = row[['mean', 'block20_ci95_low', 'block20_ci95_high']].to_numpy(float)
    if not np.isfinite(values).all() or not values[1] <= values[0] <= values[2]:
        raise ValueError(f'Invalid mean interval: {filters}')
    return row


def base_axis(ax):
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.spines['bottom'].set_linewidth(.65)
    ax.tick_params(axis='both', labelsize=6, width=.6, length=2)
    ax.tick_params(axis='y', length=0, pad=3)
    ax.grid(axis='x', color='#E6E8EB', lw=.45)
    ax.set_axisbelow(True)


def point(ax, row, y, style):
    x, lo, hi = map(float, row[['mean', 'block20_ci95_low', 'block20_ci95_high']])
    ax.errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt=style['marker'], color=style['color'],
                mfc='white' if style['marker']=='o' else style['color'], mec=style['color'],
                ms=5.5 if style['marker']=='o' else 4.6, mew=1.05, lw=.9, capsize=2, zorder=4)


def draw_c(ax, primary, cfg):
    table = primary.loc[primary.policy.eq('last')]
    full = one(table, method='A0', metric='physical_relative_l2', target='Unobserved_mean')
    ax.axhspan(3.62, 4.38, color=cfg['styles']['A0']['color'], alpha=.06)
    ax.axvline(full['mean'], color=cfg['styles']['A0']['color'], lw=.65, alpha=.55)
    positions = [4, 3, 2, 1, -.25]
    labels = ['Full model', 'No sensor\nfeedback', 'No local\nconditioning', 'Local-only\nconditioning', 'IID Gaussian\nprior']
    records = []
    for method, y in zip(cfg['stochastic_order'], positions):
        row = one(table, method=method, metric='physical_relative_l2', target='Unobserved_mean')
        style = cfg['styles'][method]
        point(ax, row, y, style)
        annotation = f"{row['mean']:.4f}" if method=='A0' else f"{100*(row['mean']/full['mean']-1):+.1f}%"
        # Largest effect is labelled left of its point to keep the full interval visible.
        dx, ha = (-7, 'right') if method=='A5' else (7, 'left')
        ax.annotate(annotation, (row['mean'], y), xytext=(dx, 0), textcoords='offset points',
                    ha=ha, va='center', fontsize=6.2, color=style['color'])
        records.append(dict(row.to_dict(), panel='c', plotted_y=y, annotation=annotation))
    ax.axhline(.35, color='#DDE1E5', lw=.55)
    ax.set_yticks(positions, labels)
    ax.set_ylim(-.83, 4.7)
    ax.set_xlim(.08, .395)
    ax.set_xticks([.1, .2, .3])
    ax.set_xlabel(r'Unobserved-field relative L$_2$', fontsize=6.6, labelpad=4)
    base_axis(ax)
    assert all(ax.get_xlim()[0] <= r['block20_ci95_low'] <= r['block20_ci95_high'] <= ax.get_xlim()[1] for r in records)
    return records


def draw_e(ax, hf, cfg, metric, upper):
    records = []
    if upper:
        ax.axvline(1, color='#6E747B', ls=(0,(3,2)), lw=.7)
        ax.text(1.05, 1.38, 'truth = 1', color='#666666', fontsize=5.8)
        ax.set_xlim(0, 3.85)
        ax.set_xticks([0,1,2,3])
        xlabel = 'High-band power / truth'
    else:
        ax.axvline(0, color='#6E747B', ls=(0,(3,2)), lw=.7)
        ax.text(.03, 1.38, 'ideal = 0', color='#666666', fontsize=5.8)
        ax.set_xlim(0, 2.15)
        ax.set_xticks([0,.5,1,1.5,2])
        xlabel = r'High-band relative L$_2$'
    for method, y in [('A0',1), ('A4',0)]:
        row = one(hf, policy='last', method=method, field='U1', metric=metric)
        point(ax, row, y, cfg['styles'][method])
        annotation=f"{row['mean']:.2f}"
        ax.annotate(annotation, (row['mean'],y), xytext=(0,7), textcoords='offset points',
                    ha='center', fontsize=6.1, color=cfg['styles'][method]['color'])
        records.append(dict(row.to_dict(), panel='e_power' if upper else 'e_residual', plotted_y=y, annotation=annotation))
    ax.set_yticks([1,0], ['RFF (full)','IID'])
    ax.set_ylim(-.6,1.75)
    ax.set_xlabel(xlabel, fontsize=6.6, labelpad=4)
    base_axis(ax)
    assert all(ax.get_xlim()[0] <= r['block20_ci95_low'] <= r['block20_ci95_high'] <= ax.get_xlim()[1] for r in records)
    return records


def clear_figure_labels(fig):
    for item in list(fig.texts) + list(fig.lines) + list(fig.legends):
        item.set_visible(False)


def polish_inherited(fig):
    for ax in fig.axes:
        for txt in ax.texts:
            if 'DMF AURC' in txt.get_text():
                txt.set_visible(False)
        if ax._left_title.get_text()=='Training time':
            ax.set_title('Training update\ntime', loc='left', fontsize=6.6, pad=4.2, fontweight='semibold')
        for obj in ax.findobj(Text):
            if obj.get_fontsize() < 5.8:
                obj.set_fontsize(5.8)


def tags(fig, specs):
    for letter, x, y, title in specs:
        fig.text(x,y,letter, fontsize=8.5, fontweight='bold', va='bottom')
        if title:
            fig.text(x+4.5/(fig.get_figwidth()*25.4),y,title,fontsize=7, fontweight='semibold',va='bottom')


def memory_key(ax):
    # The accepted renderer has already assigned filled Model and hollow Peak.
    handles = [Line2D([],[],marker='o',color='#555555',mfc=face,ls='none',ms=4,label=label)
               for face,label in [('#555555','Model'),('white','Peak')]]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,1.17), ncol=2,
              fontsize=5.8, frameon=False, columnspacing=.6, handletextpad=.2, borderaxespad=0)


def text_bounds(fig):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    bounds=fig.bbox
    outside=[]
    for txt in fig.findobj(Text):
        if not txt.get_visible() or not txt.get_text() or (txt.axes is not None and not txt.axes.get_visible()):
            continue
        # Tick objects outside the view interval are intentionally not rendered.
        if txt in [t.label1 for ax in fig.axes for t in ax.xaxis.get_major_ticks() if t.get_loc()<ax.get_xlim()[0] or t.get_loc()>ax.get_xlim()[1]]:
            continue
        box=txt.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < bounds.x0-1 or box.x1>bounds.x1+1 or box.y0<bounds.y0-1 or box.y1>bounds.y1+1):
            outside.append(txt.get_text())
    return outside


def artist_coordinates(fig):
    """Export the actual rendered coordinates, including inherited display jitter."""
    result=[]
    for ax in fig.axes:
        item={'xlabel':ax.get_xlabel(),'ylabel':ax.get_ylabel(),'xscale':ax.get_xscale(),
              'yscale':ax.get_yscale(),'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),
              'position_fraction':list(ax.get_position().bounds),'lines':[],'collections':[],
              'annotations':[{'text':t.get_text(),'position':list(t.get_position())} for t in ax.texts if t.get_visible()]}
        for line in ax.lines:
            item['lines'].append({'x':np.asarray(line.get_xdata(orig=False)).tolist(),
                                  'y':np.asarray(line.get_ydata(orig=False)).tolist(),
                                  'marker':line.get_marker(),'color':str(line.get_color())})
        for collection in ax.collections:
            rec={'paths':[p.vertices.tolist() for p in collection.get_paths()]}
            if hasattr(collection,'get_offsets'):rec['offsets']=np.asarray(collection.get_offsets()).tolist()
            if hasattr(collection,'get_segments'):rec['segments']=[p.tolist() for p in collection.get_segments()]
            item['collections'].append(rec)
        result.append(item)
    return result


def save(fig, path, dpi):
    path.parent.mkdir(parents=True,exist_ok=True)
    outside=text_bounds(fig)
    fig.savefig(path, facecolor='white', metadata={'Date':None})
    fig.savefig(path.with_suffix('.png'),dpi=dpi,facecolor='white')
    # A true 183-mm, 150-dpi proof is separately supplied for final-size review.
    if 'composed' in path.name:
        fig.savefig(path.with_name(path.stem+'_print_proof.png'),dpi=150,facecolor='white')
    plt.close(fig)
    return {'path':str(path.relative_to(REPO)), 'off_canvas_text':outside}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timestamp',required=True)
    parser.add_argument('--layout',choices=['third-column'],default='third-column')
    parser.add_argument('--strict-formal',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(CONFIG.read_text())
    derived=ROOT/'results/derived'/args.timestamp
    out=ROOT/'figures/generated'/args.timestamp
    derived.mkdir(parents=True,exist_ok=True)
    if args.strict_formal:
        qa=json.loads((derived/'source_qa.json').read_text())
        if qa['status']!='pass': raise RuntimeError('Source validation has not passed')
    primary=pd.read_csv(derived/'ablation_primary_summary.csv')
    hf=pd.read_csv(derived/'ablation_highband_summary.csv')
    accepted=accepted_renderer()
    v1=accepted.V1
    base=v1.load_base()
    old=yaml.safe_load((ROOT/'configs/figure5_v6.yaml').read_text())
    v1.configure_nature_style(base,old)
    plt.rcParams.update({'font.size':6.6,'axes.labelsize':6.6,'xtick.labelsize':6,'ytick.labelsize':6,'svg.hashsalt':'figure5-v7-ablation'})
    paths=base.require_sources(old)
    uq=old['paper_contract']['generative_method_order']
    a,sa,b,sb=base.load_panel_ab(paths['v5_display'],uq)
    d=base.load_panel_c(paths['v51_selective_risk'],old)
    f=base.derive_panel_d(paths['v51_scorecard'],paths['v51_scorecard_stages'],paths['inference_memory'],old['paper_contract']['scorecard_method_order'])
    # The collector writes the same accepted numbers with row-level evidence
    # identities.  Do not replace these enriched audit tables with bare rows.
    oldqa=json.loads((ROOT/'results/derived/20260904_1200/qa.json').read_text())
    manifest={'backend':'Python/Matplotlib','size_mm':[183,cfg['figure']['height_mm']], 'geometry':{},'figures':[], 'comparison_scope':cfg['comparison_scope']}
    builders={'a':lambda:base.make_standalone_ab('a',a,sa,old), 'b':lambda:base.make_standalone_ab('b',b,sb,old),
              'd':lambda:base.make_standalone_c(d,old), 'f':lambda:base.make_standalone_d(f,old)}
    oldmap={'a':'a','b':'b','d':'c','f':'d'}
    for name,builder in builders.items():
        fig=builder(); orig=oldmap[name]
        v1.refine_standalone(fig,orig)
        before=v1.geometry_digest(fig)
        if before!=oldqa['geometry_digests'][orig]['after']: raise RuntimeError(f'Archived scientific digest mismatch: {name}')
        clear_figure_labels(fig); polish_inherited(fig)
        if name=='b':
            fig.axes[0].tick_params(axis='y',labelleft=True)
            fig.axes[0].set_yticklabels(uq)
            for tick,method in zip(fig.axes[0].get_yticklabels(),uq):
                tick.set_color(old['style']['method_colors'][method])
        if name=='d':
            fig.axes[0].set_position([.17,.27,.80,.58])
            handles=[Line2D([],[],color=old['style']['method_colors'][m], marker=old['style']['method_markers'][m],mfc='white',ms=4,lw=.8,label=m) for m in uq]
            fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.015),ncol=3,frameon=False,fontsize=5.8,handlelength=1.2,columnspacing=.8)
        if name=='f':
            fig.set_size_inches(183/25.4,73/25.4)
            accepted.V4.V3.V2.position_scorecard_axes(fig.axes,left=.12,right=.98,bottom=.15,top=.72)
            tags(fig,[('f',.12,.92,'Accuracy and computational footprint')]); memory_key(fig.axes[-1])
        else:
            tags(fig,[(name,.19,.91,{'a':'Normalized CRPS','b':'Spread–error association','d':'Selective reconstruction'}[name])])
        after=v1.geometry_digest(fig)
        assert before==after
        manifest['geometry'][name]={'archived':before,'recomposed':after}
        manifest['figures'].append(save(fig,out/f'fig5{name}_{STEMS[name]}_{args.timestamp}.svg',cfg['figure']['png_dpi']))
    records=[]
    fig=base.make_composed(a,sa,b,sb,d,f,old)
    v1.refine_composed(fig)
    before=v1.geometry_digest(fig)
    assert before==oldqa['geometry_digests']['composed']['after']
    clear_figure_labels(fig); polish_inherited(fig)
    fig.set_size_inches(183/25.4,cfg['figure']['height_mm']/25.4)
    fig.axes[0].set_position([.12,.735,.205,.205])
    fig.axes[1].set_position([.385,.735,.205,.205])
    fig.axes[2].set_position([.12,.43,.47,.215])
    accepted.V4.V3.V2.position_scorecard_axes(fig.axes[3:],left=.12,right=.98,bottom=.065,top=.29)
    memory_key(fig.axes[-1])
    after=v1.geometry_digest(fig)
    assert before==after
    manifest['geometry']['composed_inherited']={'archived':before,'recomposed':after}
    tags(fig,[('a',.12,.96,''),('b',.385,.96,''),('c',.68,.96,'Conditioning and\nsource variants'),
              ('d',.12,.674,'Selective reconstruction'),('e',.68,.674,'Fine-scale velocity fidelity'),
              ('f',.12,.352,'Accuracy and computational footprint')])
    records+=draw_c(fig.add_axes([.79,.735,.195,.205]),primary,cfg)
    records+=draw_e(fig.add_axes([.79,.574,.195,.082]),hf,cfg,'canonical_shellmean_high_energy_ratio',True)
    records+=draw_e(fig.add_axes([.79,.43,.195,.082]),hf,cfg,'highband_error_relative_l2',False)
    (derived/'main_artist_coordinates.json').write_text(json.dumps(artist_coordinates(fig),indent=2)+'\n')
    manifest['figures'].append(save(fig,out/f'fig5_composed_v7_{args.timestamp}.svg',cfg['figure']['png_dpi']))
    fig=plt.figure(figsize=(89/25.4,74/25.4)); draw_c(fig.add_axes([.38,.17,.59,.67]),primary,cfg)
    tags(fig,[('c',.06,.92,'Conditioning and source variants')])
    manifest['figures'].append(save(fig,out/f'fig5c_{STEMS["c"]}_{args.timestamp}.svg',cfg['figure']['png_dpi']))
    fig=plt.figure(figsize=(89/25.4,95/25.4))
    draw_e(fig.add_axes([.27,.60,.69,.24]),hf,cfg,'canonical_shellmean_high_energy_ratio',True)
    draw_e(fig.add_axes([.27,.16,.69,.24]),hf,cfg,'highband_error_relative_l2',False)
    tags(fig,[('e',.07,.93,'Fine-scale velocity fidelity')])
    manifest['figures'].append(save(fig,out/f'fig5e_{STEMS["e"]}_{args.timestamp}.svg',cfg['figure']['png_dpi']))
    for suffix,metric,upper in [('power','canonical_shellmean_high_energy_ratio',True),('residual','highband_error_relative_l2',False)]:
        fig=plt.figure(figsize=(89/25.4,48/25.4)); draw_e(fig.add_axes([.27,.27,.69,.52]),hf,cfg,metric,upper)
        manifest['figures'].append(save(fig,out/f'fig5e_{suffix}_{args.timestamp}.svg',cfg['figure']['png_dpi']))
    pd.DataFrame(records).to_csv(derived/'main_ablation_plot_coordinates.csv',index=False)
    manifest['new_coordinates']='main_ablation_plot_coordinates.csv'
    manifest['scientific_geometry_preserved']=True
    manifest['status']='pass' if all(not f['off_canvas_text'] for f in manifest['figures']) else 'layout_review_required'
    (derived/'main_plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'figures':len(manifest['figures'])}))


if __name__=='__main__':
    main()

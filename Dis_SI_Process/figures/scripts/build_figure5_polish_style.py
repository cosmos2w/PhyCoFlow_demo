"""Shared Figure 5 / SI presentation functions; saved evaluation sources only."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, LogLocator, NullLocator
from matplotlib.text import Text
import numpy as np
from scipy.stats import gaussian_kde
import build_figure5_v7r2 as R2

FONTS={'axes.titlesize':12,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'legend.fontsize':9,'font.size':9}
def configure_style():
    R2.configure_style()
    plt.rcParams.update({**FONTS,'svg.fonttype':'none','pdf.fonttype':42,'font.family':'sans-serif','savefig.bbox':'tight','savefig.pad_inches':.05})

def standardize(fig):
    for text in fig.findobj(Text):text.set_fontsize(8)
    for ax in fig.axes:
        ax.xaxis.label.set_fontsize(10);ax.yaxis.label.set_fontsize(10)
        ax.title.set_fontsize(12)
        ax._left_title.set_fontsize(12);ax._right_title.set_fontsize(12)
        ax.tick_params(axis='both',which='both',labelsize=9)
        if ax.get_legend():
            for text in ax.get_legend().get_texts():text.set_fontsize(9)
    for text in fig.texts:text.set_fontsize(12)

def distribution(ax,states,summary,methods,field,xlabel,record=None,**kwargs):
    records=[]
    for i,method in enumerate(methods):
        rows=states[states.field.eq(field)&states.method.eq(method)].sort_values('snapshot')
        values=rows.value.to_numpy(float)
        selected=summary[summary.field.eq(field)&summary.method.eq(method)]
        if len(values)!=1000 or not np.isfinite(values).all() or len(selected)!=1:raise ValueError(f'Missing/ambiguous {method}/{field}')
        r=selected.iloc[0];mean=float(r['mean'])
        if not np.isclose(mean,values.mean(),atol=1e-12,rtol=0):raise ValueError('Summary/state mean mismatch')
        kde=gaussian_kde(values,bw_method='scott');bw=float(np.sqrt(kde.covariance[0,0]))
        grid=np.linspace(max(0,values.min()-3*bw),values.max()+3*bw,600)
        dens=kde(grid);width=.4*dens/dens.max();color=R2.STYLES[method]['color']
        ax.fill_betweenx(grid,i-width,i+width,facecolor=color,edgecolor=color,alpha=.65,lw=.8,zorder=3)
        w=float(.4*kde([mean])[0]/dens.max())
        ax.plot([i-w,i+w],[mean,mean],color='#222222',ls=(0,(2,1.5)),lw=1.2,zorder=4)
        records.append({'method':method,'field':field,'n':1000,'mean':mean,'median':float(r['median']),'ci_low':float(r['block20_ci95_low']),'ci_high':float(r['block20_ci95_high']),'bandwidth':bw,'grid':grid.tolist(),'density':dens.tolist()})
    ax.set_xticks(range(len(methods)),[R2.STYLES[m]['label'] for m in methods],rotation=30,ha='right')
    for t,m in zip(ax.get_xticklabels(),methods):t.set_color(R2.STYLES[m]['color'])
    ax.set_xlim(-.65,len(methods)-.35);ax.set_ylabel(xlabel,fontsize=10)
    ax.set_ylim(bottom=0);ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',length=0)
    ax.set_axisbelow(True);ax.grid(axis='y',color='#DCE0E5',lw=.5)
    if record is not None:record.extend(records)
    return records

def spectra(ax,pop,field,methods=None,record=None,**kwargs):
    methods=R2.MAIN_METHODS if methods is None else methods;records=[]
    for index,method in enumerate(['Truth',*methods]):
        rows=pop[pop.field.eq(field)&pop.method.eq(method)].sort_values('shell_index')
        if len(rows)!=198 or not np.isfinite(rows['median']).all() or not (rows['median']>0).all():raise ValueError(f'Missing spectrum {method}/{field}')
        primary=method in ['Truth','A0'];color='#222222' if method=='Truth' else R2.STYLES[method]['color']
        label='Truth' if method=='Truth' else R2.STYLES[method]['label']
        ax.plot(rows.shell_index,rows['median'],color=color,lw=2 if primary else 1,ls='-' if primary else ['--',':','-.'][index%3],alpha=1 if primary else .7,label=label,zorder=5 if primary else 3)
        records.append({'method':method,'field':field,'shell_index':rows.shell_index.tolist(),'median_power':rows['median'].tolist()})
    high=rows.high_band.astype(str).str.lower().isin(['true','1','1.0']);boundary=float(rows.loc[high,'shell_index'].min())-.5
    ax.axvspan(boundary,float(rows.shell_index.max())+.5,color='#64748B',alpha=.08,lw=0,zorder=0)
    ax.text(.98,.025,'High-band',transform=ax.transAxes,ha='right',va='bottom',color='#46556A',fontsize=9)
    ax.set_yscale('log');ax.set_xlim(float(rows.shell_index.min()),float(rows.shell_index.max()))
    ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(LogLocator(base=10,numticks=6));ax.yaxis.set_minor_locator(NullLocator());ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel('Frequency shell');ax.set_ylabel('Spectral power '+R2.FIELD_LABEL[field])
    ax.spines[['top','right']].set_visible(False)
    ax.legend(ncol=1,loc='upper right',labelspacing=.5,handlelength=2.6,fontsize=9)
    if record is not None:record.extend(records)
    return records

def save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if any(path.with_suffix(ext).exists() for ext in ['.svg','.pdf','.png']):raise FileExistsError(path)
    standardize(fig)
    for ext in ['svg','pdf','png']:
        fig.savefig(path.with_suffix('.'+ext),dpi=600,bbox_inches='tight',pad_inches=.05,facecolor='white')
    fig.savefig(path.with_name(path.stem+'_preview.png'),dpi=150,bbox_inches='tight',pad_inches=.05,facecolor='white')

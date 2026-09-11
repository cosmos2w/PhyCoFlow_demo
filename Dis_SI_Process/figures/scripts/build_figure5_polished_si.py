#!/usr/bin/env python
"""Additive SI figures and manuscript text for the revised Figure 5 release."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import fitz
import build_figure5_polish_style as S
R=S.R2
OUT=R.ROOT/'figures/generated/figure5_precision_log_revision_20260910/SI'
FIELDS=['CH4','CO','T','U1','p']
ABLATIONS=['A0','A2','A3','A5','A4','A1']
CONTROL=['A0','A1','Senseiver']


def markdown_table(headers,rows):
    return ['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|',
            *['| '+' | '.join(str(value) for value in row)+' |' for row in rows]]


def main_caption():
    return r'''\begin{figure}[t]
\centering
\includegraphics[width=\textwidth]{figure5_log.pdf}
\caption{Uncertainty, ablation performance and computational cost.
\textbf{a}, Normalized CRPS distributions and bootstrap mean intervals for five generative models.
\textbf{b}, Across-state predictive-spread/error Spearman correlations and bootstrap uncertainty.
\textbf{c}, Relative reconstruction error among the least-uncertain retained states.
\textbf{d}, Unobserved-field relative $L_2$ (top) and $U_1$ high-band relative $L_2$ (bottom) for six model variants. Vertical violins use Gaussian KDE with Scott bandwidth, independently normalized widths, and 1,000 states per model; dark dashed segments indicate arithmetic means.
\textbf{e}, Median absolute $U_1$ shell-mean spectral power over the same evaluation population. Thick solid lines identify truth and the full model; thinner dashed or dotted lines identify alternatives. Shading marks high-band shells 134--201. All 198 evaluated shells (4--201) are retained.
\textbf{f}, Mean unobserved-field relative $L_2$ with 95\% confidence intervals, followed by training update time, training memory, inference time and inference memory. Timing whiskers show interquartile ranges; memory whiskers show measured repeat extrema and have zero width where repeats agree. Filled and outlined inference-memory bars denote model storage and peak allocation. DMF-Gen accuracy uses the same A0 \texttt{last.pt} evaluation as \textbf{d}; resource measurements retain their historical benchmark sources. Saved checkpoint epochs differ across ablations.}
\label{fig:figure5-precision-log-revision}
\end{figure}
'''


def main_quantitative_report(summary_a,summary_b,selective,score,records):
    parts=['# Figure 5 precision-log revision: quantitative results and plotting logic','',
        'These are direct saved-evaluation summaries, not new training results. All new ablation sources use `last.pt`. Unequal checkpoint epochs and the shared holdout remain inherited limitations.','']
    for title,description,frame in [
        ('Panel a — Normalized CRPS','Lower values indicate better probabilistic accuracy. Inherited statewise evidence and bootstrap mean intervals are unchanged.',summary_a),
        ('Panel b — Across-state uncertainty/error correlation','Higher positive Spearman correlation indicates better ranking of uncertain states. Inherited bootstrap intervals are unchanged.',summary_b)]:
        parts.extend(['## '+title,'',description,''])
        parts.extend(markdown_table(['Method','Value','95% bootstrap CI'],[
            [row.method,f'{row.metric_value:.9f}',f'[{row.ci_low:.9f}, {row.ci_high:.9f}]'] for row in frame.itertuples(index=False)]))
        parts.append('')
    parts.extend(['## Panel c — Selective reconstruction','',
        'Relative retained-set error against the fraction of least-uncertain states retained. The comparison reflects uncertainty-ranking quality, not raw absolute reconstruction error.',''])
    parts.extend(markdown_table(['Method','Retained fraction','Relative error','95% bootstrap CI','States'],[
        [row.method,f'{row.coverage_fraction:.1f}',f'{row.risk:.9f}',f'[{row.ci_low:.9f}, {row.ci_high:.9f}]',row.selected_state_count]
        for row in selective.itertuples(index=False)]))
    # Recover the six main-panel reconstruction records from the S1/S3 SI
    # computations, and pair them with the U1 records used in panel-e_U1.
    reconstruction={r['method']:r for name in ['S1_ablation_error','S3_deterministic_control'] for r in records[name]['distributions']}
    highband={r['method']:r for r in records['panel_e_U1']['distributions']}
    main_methods=R.MAIN_METHODS
    parts.extend(['','## Panel d — Vertical error distributions','',
        'Upper: macro unobserved-field relative L2. Lower: U1 high-band relative L2. Labels are right-aligned at 30 degrees. Each violin contains 1,000 states, uses a Scott Gaussian KDE with independently normalized width, and marks the arithmetic mean with a dark dashed segment.',''])
    parts.extend(markdown_table(['Field','Method','n','Mean','Median','95% block CI','KDE bandwidth'],[
        [field,method,r['n'],f"{r['mean']:.9f}",f"{r['median']:.9f}",f"[{r['ci_low']:.9f}, {r['ci_high']:.9f}]",f"{r['bandwidth']:.9f}"]
        for field,source in [('Unobserved_mean',reconstruction),('U1 high-band',highband)] for method in main_methods for r in [source[method]]]))
    spectra=records['panel_e_U1']['spectra']
    parts.extend(['','## Panel e — Population spectra','',
        'Absolute median U1 shell-mean power for all 198 retained shells. No smoothing, normalization, or marker subsampling is applied; the shaded high band is shells 134–201.',''])
    parts.extend(markdown_table(['Method','Shell 4','Shell 50','Shell 100','Shell 150','Shell 201'],[
        [r['method'],*[f"{r['median_power'][r['shell_index'].index(shell)]:.9g}" for shell in [4,50,100,150,201]]] for r in spectra]))
    parts.extend(['','## Panel f — Accuracy and resource cost','',
        'The first subplot reports mean unobserved-field error and block-bootstrap intervals. Timing intervals are IQRs; memory intervals are measured repeat extrema. Model storage is a fixed tensor inventory.',''])
    cols=['method','mean_unobserved_relative_l2','error_ci_low','error_ci_high','training_update_ms','training_peak_allocated_mib','inference_latency_ms','model_state_mib','inference_peak_allocated_mib']
    parts.extend(markdown_table(['Method','Mean error','CI low','CI high','Train ms/update','Train MiB','Inference ms','Model MiB','Peak MiB'],[
        [getattr(row,col) if col=='method' else f'{float(getattr(row,col)):.9f}' for col in cols] for row in score.itertuples(index=False)]))
    full=reconstruction['A0']['mean'];no_sensor=reconstruction['A2']['mean'];no_local=reconstruction['A3']['mean'];local_only=reconstruction['A5']['mean'];iid=reconstruction['A4']['mean'];senseiver=reconstruction['Senseiver']['mean']
    hb_full=highband['A0']['mean'];hb_iid=highband['A4']['mean'];hb_senseiver=highband['Senseiver']['mean']
    parts.extend(['','## Direct comparisons for manuscript writing','',
        f'- Full-model macro error is {full:.6f}. Removing sensor feedback raises it by {(no_sensor/full-1)*100:.2f}%; removing local conditioning raises it by {(no_local/full-1)*100:.2f}%; local-only conditioning raises it by {(local_only/full-1)*100:.2f}%.',
        f'- IID-prior macro error is {(1-iid/full)*100:.2f}% lower than Full model, while U1 high-band error is {(hb_iid/hb_full-1)*100:.2f}% higher ({hb_iid:.6f} versus {hb_full:.6f}).',
        f'- Full-model U1 high-band error is {(1-hb_full/hb_senseiver)*100:.2f}% below Senseiver; its macro error is {(1-full/senseiver)*100:.2f}% below Senseiver.','',
        'These are descriptive ratios of saved-checkpoint means, not paired significance tests.','',
        '## Revision-specific layout and export QA','',
        'The two inter-row spacer weights are both 0.55, panels d/e use a 6:4 width ratio, panel-d category labels are right-aligned at 30 degrees, and the panel-e high-band label is normal-weight 10 pt. The tightly cropped PDF measures 258.05 × 248.71 mm and uses embedded editable TrueType fonts. The full-resolution PNG is 6,096 × 5,872 pixels. No evaluation, training, confidence interval, spectrum, or resource measurement was recomputed.',''])
    return '\n'.join(parts)

def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output_dir.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'si_manifest.json').exists():raise FileExistsError('SI release exists')
    tables=R.load_tables('20260910_1707');S.configure_style()
    accepted=R.V7.accepted_renderer();base=accepted.V1.load_base()
    import yaml
    cfg=yaml.safe_load((R.ROOT/'configs/figure5_v6.yaml').read_text())
    paths=base.require_sources(cfg);uq=cfg['paper_contract']['generative_method_order']
    _,summary_a,_,summary_b=base.load_panel_ab(paths['v5_display'],uq)
    selective=base.load_panel_c(paths['v51_selective_risk'],cfg)
    score=base.derive_panel_d(paths['v51_scorecard'],paths['v51_scorecard_stages'],paths['inference_memory'],cfg['paper_contract']['scorecard_method_order'])
    full=tables['reconstruction_summary'].loc[lambda x:x.method.eq('A0')&x.field.eq('Unobserved_mean')]
    if len(full)!=1:raise ValueError('Missing or ambiguous A0 accuracy source')
    for target,source in [('mean_unobserved_relative_l2','mean'),('error_ci_low','block20_ci95_low'),('error_ci_high','block20_ci95_high')]:
        score.loc[score.method.eq('DMF-Gen'),target]=float(full.iloc[0][source])
    for name in ['reconstruction_states','highband_states']:
        if not tables[name].policy.eq('last').all():raise ValueError('SI source policy is not last')
    manifests=[];records={};checks={};captions=[]
    def export(fig,name,description,dist,spec):
        path=OUT/(name+'.svg');S.save(fig,path)
        fig.canvas.draw();renderer=fig.canvas.get_renderer()
        labels=[]
        for ax in fig.axes:
            if len(ax.collections)==len(ax.get_xticklabels()) and len(ax.collections)>0:
                ticks=[t for t in ax.get_xticklabels() if t.get_visible()]
                # All category labels are parallel at 30 degrees. Their perpendicular
                # separation is independent of text length; axis-aligned boxes
                # falsely flag the empty corners of rotated long labels.
                anchors=[ax.transData.transform((tick,0))[0] for tick in ax.get_xticks()]
                required=max(t.get_fontsize() for t in ticks)*fig.dpi/72*1.2
                labels.extend([abs(b-a)*np.sin(np.pi/6)>required for a,b in zip(anchors,anchors[1:])])
        with fitz.open(path.with_suffix('.pdf')) as doc:
            doc[0].get_pixmap(matrix=fitz.Matrix(.9,.9)).save(OUT/(name+'_pdf_review.png'))
            embedded=all(len(doc.extract_font(f[0])[-1])>0 for f in doc[0].get_fonts())
            dimensions=[doc[0].rect.width*25.4/72,doc[0].rect.height*25.4/72]
        item={'name':name,'description':description,'pdf_mm':dimensions,'font_embedding':embedded,'distribution_labels_nonoverlapping':all(labels),'distribution_count':len(dist),'spectrum_count':len(spec)}
        manifests.append(item);records[name]={'distributions':dist,'spectra':spec}
        checks[name+'_font_embedding']=embedded;checks[name+'_rotated_labels']=all(labels)
        checks[name+'_state_coverage']=all(r['n']==1000 for r in dist)
        checks[name+'_shell_coverage']=all(len(r['shell_index'])==198 and r['shell_index'][0]==4 and r['shell_index'][-1]==201 for r in spec)
        captions.append('\\paragraph{'+name.replace('_',r'\_')+'} '+description)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.7));dist=S.distribution(ax,tables['reconstruction_states'],tables['reconstruction_summary'],ABLATIONS,'Unobserved_mean',r'Unobserved-field relative $L_2$');ax.set_title('S1',loc='left');fig.subplots_adjust(bottom=.34,left=.12,right=.98,top=.9)
    export(fig,'S1_ablation_error','Saved-checkpoint ablation macro reconstruction errors, including the deterministic-objective control A1. Violin widths are independently normalized Scott Gaussian KDEs of 1,000 states; dashed segments denote arithmetic means.',dist,[])
    fig,axes=plt.subplots(5,2,figsize=(13,24),gridspec_kw={'width_ratios':[1.1,1]});dist=[];spec=[]
    for (left,right),field in zip(axes,FIELDS):
        S.spectra(left,tables['spectra_population'],field,record=spec)
        S.distribution(right,tables['highband_states'],tables['highband_summary'],R.MAIN_METHODS,field,r'High-band relative $L_2$',record=dist)
        left.set_title('S2 '+R.FIELD_LABEL[field],loc='left');right.set_title(R.FIELD_LABEL[field],loc='left')
    fig.subplots_adjust(left=.08,right=.99,bottom=.065,top=.98,wspace=.3,hspace=.78)
    export(fig,'S2_all_fields','Five candidate physical fields: absolute median shell-mean spectra at left and high-band error violins at right. Thick solid Truth and Full model lines stand above thin dashed alternatives. Shading denotes saved high-band shells 134--201; all 198 evaluated shells are retained. Dashed violin segments mark means.',dist,spec)
    for field in FIELDS:
        fig,(left,right)=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[1.1,1]});spec=S.spectra(left,tables['spectra_population'],field);dist=S.distribution(right,tables['highband_states'],tables['highband_summary'],R.MAIN_METHODS,field,r'High-band relative $L_2$')
        left.set_title('e '+R.FIELD_LABEL[field],loc='left');fig.subplots_adjust(left=.08,right=.99,bottom=.32,top=.92,wspace=.3)
        export(fig,'panel_e_'+R.FIELD_EXPORT[field],f'Panel-e alternative for {field}: median absolute spectrum and corresponding high-band error distribution, with the same methods, styles and KDE statistics as S2.',dist,spec)
    fig,(left,right)=plt.subplots(1,2,figsize=(12,5.5),gridspec_kw={'width_ratios':[1,1.2]});dist=S.distribution(left,tables['reconstruction_states'],tables['reconstruction_summary'],CONTROL,'Unobserved_mean',r'Unobserved-field relative $L_2$')
    summary=tables['reconstruction_summary']
    controls=[]
    for j,method in enumerate(CONTROL):
        rows=summary[summary.method.eq(method)].set_index('field').loc[FIELDS]
        mean=rows['mean'].to_numpy();lo=rows.block20_ci95_low.to_numpy();hi=rows.block20_ci95_high.to_numpy()
        right.errorbar(np.arange(5)+(j-1)*.18,mean,yerr=[mean-lo,hi-mean],fmt=R.STYLES[method]['marker'],color=R.STYLES[method]['color'],label=R.STYLES[method]['label'],capsize=3,lw=1,ms=5)
        for field,(_,row) in zip(FIELDS,rows.iterrows()):controls.append({'field':field,'method':method,'mean':float(row['mean']),'ci_low':float(row.block20_ci95_low),'ci_high':float(row.block20_ci95_high)})
    right.set_xticks(range(5),[R.FIELD_LABEL[f] for f in FIELDS]);right.set_ylabel(r'Fieldwise relative $L_2$');right.spines[['top','right']].set_visible(False);right.legend(loc='upper left',bbox_to_anchor=(1.02,1),ncol=1);right.grid(axis='y',color='#DCE0E5',lw=.5);right.set_axisbelow(True)
    left.set_title('S3',loc='left');fig.subplots_adjust(left=.08,right=.99,bottom=.32,top=.92,wspace=.35)
    export(fig,'S3_deterministic_control','Full model, deterministic-objective control A1 and Senseiver. Left: macro reconstruction violins. Right: fieldwise mean relative L2 and saved block-bootstrap 95 percent confidence intervals. These are unequal-epoch saved checkpoints, not equal-budget retraining.',dist,[])
    records['S3_deterministic_control']['fieldwise_controls']=controls
    report=['# Supplementary figure quantitative results and plotting logic','','All new SI sources use last.pt. No evaluation was rerun. Each violin uses all 1,000 saved state errors, Scott Gaussian KDE in linear error units, tails extended by three bandwidths and clipped at zero, independent maximum width 0.8, alpha 0.65, and a dark dashed arithmetic-mean segment. Widths are not comparable absolute densities. Labels rotate 30 degrees; axes10, ticks9, legends9, titles12 pt. All exports use tight bounding boxes and 0.05-inch padding. Spectral curves retain their absolute median energy values; no markers or smoothing are applied.','']
    for item in manifests:
        name=item['name'];report.extend(['## '+name,'',item['description'],'','| Field | Method | n | Mean | Median | 95% block CI | Bandwidth |','|---|---|---:|---:|---:|---|---:|'])
        for r in records[name]['distributions']:report.append(f"| {r['field']} | {r['method']} | {r['n']} | {r['mean']:.9f} | {r['median']:.9f} | [{r['ci_low']:.9f}, {r['ci_high']:.9f}] | {r['bandwidth']:.9f} |")
        if records[name]['spectra']:
            report.extend(['','| Field | Method | Shell 4 | Shell 100 | Shell 201 |','|---|---|---:|---:|---:|'])
            for r in records[name]['spectra']:report.append('| '+r['field']+' | '+r['method']+' | '+' | '.join(f"{r['median_power'][r['shell_index'].index(k)]:.9g}" for k in [4,100,201])+' |')
        report.append('')
    report.extend(['## S3 fieldwise values','','| Field | Method | Mean | 95% block CI |','|---|---|---:|---|'])
    for r in controls:report.append(f"| {r['field']} | {r['method']} | {r['mean']:.9f} | [{r['ci_low']:.9f}, {r['ci_high']:.9f}] |")
    report.extend(['','## Limitations and QA','','Saved checkpoint epochs differ; single saved runs and the shared validation/test holdout remain inherited limitations. Descriptive mean differences are not significance tests. All source numerical values and KDE grids are retained in si_coordinates.json.','',f"Automated checks: {sum(checks.values())}/{len(checks)} pass. Failed checks: {[k for k,v in checks.items() if not v]}. PDF visual review is recorded in the release review manifest."])
    (OUT/'quantitative_report.md').write_text('\n'.join(report)+'\n');(OUT/'captions.tex').write_text('\n\n'.join(captions)+'\n')
    (OUT/'si_coordinates.json').write_text(json.dumps(records,indent=2)+'\n')
    source=R.ROOT/'results/derived/20260910_1707';manifest={'status':'pass' if all(checks.values()) else 'review_required','policy':'last.pt','checks':checks,'figures':manifests,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.name in [n+'.csv' for n in tables]},'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'si_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    # Rebuild copy-ready manuscript text directly from the same validated
    # frames used by the plots; no ignored prior release is required.
    main_out=OUT.parent
    (main_out/'caption.tex').write_text(main_caption())
    (main_out/'quantitative_report.md').write_text(main_quantitative_report(summary_a,summary_b,selective,score,records))
    readme=(
        '# Figure 5 precision-log revision\n\n'
        'Main exports are `figure5_log.pdf`, `.svg`, `.png`, and preview files. '
        'The `SI/` directory contains S1, S2, S3, and five field-specific panel-e alternatives in PDF/SVG/PNG, '
        'with PDF review rasters, copy-ready LaTeX captions, a detailed quantitative report, exact coordinates, and a QA manifest.\n'
    )
    (main_out/'README.md').write_text(readme)
    bundle_files=[p for p in main_out.rglob('*') if p.is_file() and p.name!='bundle_manifest.json']
    bundle={
        'status':manifest['status'],
        'main_caption':'caption.tex',
        'main_quantitative_report':'quantitative_report.md',
        'si_caption':'SI/captions.tex',
        'si_quantitative_report':'SI/quantitative_report.md',
        'si_figure_count':len(manifests),
        'checks':checks,
        'files':{str(p.relative_to(main_out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(bundle_files)},
    }
    (main_out/'bundle_manifest.json').write_text(json.dumps(bundle,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'figures':len(manifests),'checks':checks,'main_output':str(main_out)}))
if __name__=='__main__':main()

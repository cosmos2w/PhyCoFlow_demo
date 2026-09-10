#!/usr/bin/env python
"""Generate manuscript inserts and quantitative V7 companions from saved tables."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
SI=[('si_ablation_reconstruction','reconstruction','Complete reconstruction and conditioning-route effects'),
    ('si_ablation_highband_all_fields','highband','All-field scale-resolved source-prior comparison'),
    ('si_ablation_u1_spectral_audit','spectral-audit','Velocity spectra, prevalence and leakage sensitivity'),
    ('si_ablation_coupling_range_audit','coupling','Coupled distributions and histogram-range sensitivity'),
    ('si_ablation_checkpoint_sensitivity','checkpoint','Checkpoint sensitivity and recorded training histories'),
    ('si_deterministic_objective_control','deterministic-control','Deterministic objective control')]


def mdtable(frame):
    def val(v):
        if isinstance(v,float):return format(v,'.17g')
        return str(v).replace('|','/').replace('\n',' ')
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join('| '+' | '.join(val(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--timestamp',required=True);p.add_argument('--strict-formal',action='store_true');a=p.parse_args()
    d=ROOT/'results/derived'/a.timestamp;docs=ROOT/'docs/generated'/a.timestamp;tex=docs/'latex';tex.mkdir(parents=True,exist_ok=True)
    load=lambda name:pd.read_csv(d/(name+'.csv'))
    manifest=json.loads((d/'source_manifest.json').read_text());sourceqa=json.loads((d/'source_qa.json').read_text())
    if a.strict_formal and sourceqa['status']!='pass':raise RuntimeError('Source validation must pass before generating formal prose')
    cfg=yaml.safe_load((ROOT/'configs/figure5_v7_ablation.yaml').read_text())
    summ=load('ablation_all_summary');hf=load('ablation_highband_summary');prov=load('ablation_provenance')
    def value(policy,method,metric,target):return float(summ.loc[summ.policy.eq(policy)&summ.method.eq(method)&summ.metric.eq(metric)&summ.target.eq(target),'mean'].iloc[0])
    def high(policy,method,metric,field='U1'):return float(hf.loc[hf.policy.eq(policy)&hf.method.eq(method)&hf.metric.eq(metric)&hf.field.eq(field),'mean'].iloc[0])
    def write(name,text): (tex/name).write_text(text.strip()+'\n')
    full=value('last','A0','physical_relative_l2','Unobserved_mean');iid=value('last','A4','physical_relative_l2','Unobserved_mean')
    paragraph=rf'''The saved stochastic configurations distinguish conditioning-route effects from the source-prior tradeoff (Fig.~\ref{{fig:uq_cost}}c,e). The new full reference achieved mean unobserved-field relative-$L_2$ of {full:.4f}. Removing global-to-sensor feedback or local query conditioning increased this error to {value('last','A2','physical_relative_l2','Unobserved_mean'):.4f} and {value('last','A3','physical_relative_l2','Unobserved_mean'):.4f}, respectively; local-only conditioning reached {value('last','A5','physical_relative_l2','Unobserved_mean'):.4f}. The IID-prior run reduced whole-field error to {iid:.4f}, while high-band velocity power reached {high('last','A4','canonical_shellmean_high_energy_ratio'):.2f} times truth and high-band relative-$L_2$ rose to {high('last','A4','highband_error_relative_l2'):.2f}, compared with {high('last','A0','canonical_shellmean_high_energy_ratio'):.2f} and {high('last','A0','highband_error_relative_l2'):.2f} for the full RFF model. The full model attenuated high-band power, whereas IID combined a small bulk-error improvement with larger fine-scale residuals. The high-band residual increase occurred in all five fields under both checkpoint policies (Supplementary Fig.~\ref{{fig:si-ablation-highband}}). These comparisons describe available trained configurations with unequal endpoints and effective optimized capacity; the separate deterministic objective control is reported in the Supplementary Information.'''
    caption=r'''\caption{\textbf{Conditional ensembles, architectural controls and scale-resolved reconstruction fidelity.}
All evaluations use temperature-only conditioning with 256 measurements and 40,300 output points.
\textbf{a}, Normalized empirical CRPS for the benchmark generators.
\textbf{b}, Association between ensemble spread and ensemble-mean error.
\textbf{c}, Mean unobserved-field relative-$L_2$ for separately trained stochastic configurations; annotations give changes relative to the new full ablation reference. Intervals summarize 1,000 temporally related evaluation states, not training repeats.
\textbf{d}, Selective reconstruction using ensemble spread, normalized by each method's full-cohort error. Panels a, b and d use the existing 200-state, 64-draw benchmark and 95\% temporal moving-block intervals (block length 25, 2,000 replicates).
\textbf{e}, High-band $U_1$ power relative to truth and phase-sensitive high-band relative-$L_2$ for RFF and IID. Ideal values are one and zero. The strict upper-third band contains 68 retained index-space shells and 17,704 Fourier modes. Power uses trapezoidal integration of shell-mean spectra; the residual uses individual complex Fourier coefficients. The nonuniform physical grid precludes interpreting these as physical-wavenumber turbulence spectra.
\textbf{f}, Original 1,000-state benchmark accuracy and separately measured footprints. Training uses common batch size 32 with method-native target workloads; update time is milliseconds per update, not total training cost. Inference uses batch size one. Filled and hollow memory symbols denote model state and peak allocation.
Panels c and e use last checkpoints, one draw, two Euler steps, an imposed observed-entry clamp and no EMA substitution, with 95\% circular moving-block intervals of length 20 (2,000 replicates). Their full checkpoint differs from the benchmark checkpoint in a, b, d and f. Available training endpoints and effective trained capacity differ. Best-checkpoint sensitivity, the shared validation/test holdout, and the separate deterministic objective control are reported in SI.}
\label{fig:uq_cost}'''
    write('figure5_v7_caption.tex',caption);write('figure5_v7_ablation_paragraph.tex',paragraph)
    write('figure5_v7_panel_reference_updates.tex',r'''% Apply these reference updates manually; the manuscript was not edited.
% Previous Figure 5a -> Figure 5a (normalized CRPS).
% Previous Figure 5b -> Figure 5b (spread--error association).
% Previous Figure 5c -> Figure 5d (selective reconstruction).
% Previous Figure 5d -> Figure 5f (accuracy and footprint).
% New architectural/prior comparison: Fig.~\ref{fig:uq_cost}c.
% New fine-scale velocity comparison: Fig.~\ref{fig:uq_cost}e.''')
    mainpng=f'Dis_SI_Process/figures/generated/{a.timestamp}/fig5_composed_v7_{a.timestamp}.png'
    write('figure5_v7_figure.tex',rf'''\begin{{figure}}[p]
\centering
\includegraphics[width=183mm]{{{mainpng}}}
\input{{Dis_SI_Process/docs/generated/{a.timestamp}/latex/figure5_v7_caption.tex}}
\end{{figure}}''')
    methods=r'''\paragraph{Saved-checkpoint evaluation and scope.}
We retain two separate evidence packages. The archived benchmark provides the 200-state, 64-draw uncertainty results and the original 1,000-state accuracy/resource scorecard. Architectural comparisons use six saved implementations on 1,000 paired held-out snapshots, with five stochastic configurations and a separately reported deterministic objective. Every state has 256 temperature observations and 40,300 queries. The stochastic sampler uses one draw and two Euler steps with a hard observed-entry clamp, live weights and no EMA substitution. The full reference is frozen at epoch 7520 (last) and 7095 (best). The validation and test loaders share the same holdout; these are temporally related snapshots from one simulation. Endpoints and parameter elements with optimizer state differ across configurations; these element counts do not denote FLOPs or measured memory. Equal seed strings do not verify matched backbone initialization. Best-versus-last sensitivity does not make training budgets equal.
\paragraph{Reconstruction and uncertainty.}
For state $i$ and field $c$, $E_{ic}=\|\widehat u_{ic}-u_{ic}\|_2/(\|u_{ic}\|_2+10^{-12})$ in physical units. The primary macro error is $E_i=\frac14\sum_{c\in\{\mathrm{CH}_4,\mathrm{CO},U_1,p\}} E_{ic}$; temperature is excluded. States are equally weighted. Standardized-field and truth-fluctuation-normalized errors use distinct denominators and remain separate. The latter uses $\|u_{ic}-\overline u_{ic}\|_2$. The clamp imposes sensor consistency; sensor-excluded temperature error evaluates the remaining points. Percent effects are $100(\overline E_{\rm variant}/\overline E_{\rm full}-1)$, a ratio of cohort means. Paired effects resample differences after pairing exact snapshot and original time identities. Accepted intervals use 2,000 circular moving-block resamples of temporally sorted held-out states, block 20, with paired sensitivities at 5 and 50. Reconstruction uses recorded seed 20260906; high-frequency summaries use 20260910. Blocks count sorted held-out states, not consecutive simulation steps. These are conditional mean intervals, not training-seed variability or state dispersion. No new resampling was necessary for the plotted means.
\paragraph{Scale-resolved metrics.}
Fields on the $403\times100$ grid are centered spatially. The retained index-space radial shells exclude DC, shells with fewer than four modes, and modes beyond the isotropic cutoff. The strict high band is $k>(2/3)k_{\max}$, containing 68 shells and 17,704 Fourier modes. Canonical power integrates shell-mean spectra with the existing trapezoidal weights. The mode-sum ratio instead uses $R_H=\sum_{q\in H}|P_q|^2/\sum_{q\in H}|T_q|^2$. Phase-sensitive high-band error is $E_H=(\sum_{q\in H}|P_q-T_q|^2/\sum_{q\in H}|T_q|^2)^{1/2}$. Ratios are calculated per state before population averaging. Power has ideal one and attenuation below one; residual error has ideal zero. Whole-spectrum LSD is the root mean square of $10\log_{10}[(E_P(k)+\epsilon)/(E_T(k)+\epsilon)]$ over retained shells, with the original truth-relative numerical floor. Population spectra display median per-state shell ratios and IQR, which are state dispersion. Integrating that median curve does not recover a mean integrated power ratio. Full truth fluctuation budgets include all centered Fourier modes, including modes outside retained radial shells. Hann-window diagnostics apply the recorded separable taper and mean-square correction with unchanged band membership; they remain a separate estimator. No fields or spectra were smoothed or clipped.
\paragraph{Coupling and admissibility.}
Paper JSD is base-2 Jensen--Shannon divergence, not its square root, of fixed $64\times64$ joint histograms for T--$U_1$, CH$_4$--$U_1$ and p--$U_1$. Common archived truth-derived 0.5--99.5\% edges and the original $10^{-12}$ probability pseudocount are reused across all methods/policies. Paper JSD normalizes retained mass; the overflow extension uses $66\times66$ bins including all finite underflow/overflow mass. Predicted and truth retention accompany every coupling interpretation. Pearson correlation error is a coarse scalar complement. These distributions do not verify spatial alignment. Admissibility uses CH$_4$,CO $<0$ and T,p $\leq0$ before clipping, with exact counts out of 40,300,000 points per model/policy, minima and affected time IDs. None of these diagnostics establishes PDE satisfaction, conservation or thermodynamic closure.'''
    write('si_ablation_methods.tex',methods)
    detlast=value('last','A1','physical_relative_l2','Unobserved_mean');detbest=value('best','A1','physical_relative_l2','Unobserved_mean');fullbest=value('best','A0','physical_relative_l2','Unobserved_mean')
    results=paragraph+'\n\n'+rf'''\paragraph{{Distributional and objective controls.}}
Full-model CH$_4$--$U_1$ predicted retention is {100*value('last','A0','joint_pdf_reconstruction_retained_fraction','CH4-U1'):.3f}\% last and {100*value('best','A0','joint_pdf_reconstruction_retained_fraction','CH4-U1'):.3f}\% best, versus {100*value('last','A0','joint_pdf_truth_retained_fraction','CH4-U1'):.3f}\% truth; overflow JSD is {value('last','A0','joint_pdf_jsd_with_overflow_base2','CH4-U1'):.6f} and {value('best','A0','joint_pdf_jsd_with_overflow_base2','CH4-U1'):.6f}. Removing local conditioning yields lower CH$_4$--$U_1$ overflow JSD despite larger spatial error. IID coupling comparisons depend on policy and histogram definition. Deterministic regression achieves macro errors {detlast:.6f} and {detbest:.6f}, versus full {full:.6f} and {fullbest:.6f}, advantages of {100*(1-detlast/full):.1f}\% and {100*(1-detbest/fullbest):.1f}\%. Full has lower pressure error under both policies. Full last has lower truncated JSD for all three pairs, while best-policy and overflow comparisons alter parts of that result. This is deterministic forward prediction versus one sampled reconstruction, not an ensemble-mean test or evidence about conditional diversity.
\paragraph{{Fine-scale and admissibility limits.}}
The mean truth high-band velocity energy fraction is {100*high('last','A0','truth_high_energy_fraction_total'):.4f}\% of full centered fluctuation energy. IID high-band residual-budget fractions are {100*high('last','A4','highband_error_energy_over_truth_total_fluctuation_energy'):.4f}\% and {100*high('best','A4','highband_error_energy_over_truth_total_fluctuation_energy'):.4f}\%, versus full {100*high('last','A0','highband_error_energy_over_truth_total_fluctuation_energy'):.4f}\% and {100*high('best','A0','highband_error_energy_over_truth_total_fluctuation_energy'):.4f}\%. These explain a weak-band/global-error separation without forming an exact decomposition of primary physical relative-$L_2$. IID mode-sum power exceeds truth in {100*float(hf.loc[hf.policy.eq('last')&hf.method.eq('A4')&hf.metric.eq('reconstruction_to_truth_high_energy_ratio')&hf.field.eq('U1'),'fraction_gt_1'].iloc[0]):.1f}\% / {100*float(hf.loc[hf.policy.eq('best')&hf.method.eq('A4')&hf.metric.eq('reconstruction_to_truth_high_energy_ratio')&hf.field.eq('U1'),'fraction_gt_1'].iloc[0]):.1f}\% of states. Hann tapering retains IID excess and stronger RFF attenuation. The exact admissibility table preserves three nonpositive-temperature points for IID under each policy and one for local-only last, despite percentages rounding to zero. These results remain conditional on the saved checkpoints, sensor plan and two-step protocol.'''
    prevalence=[]
    for policy in ['last','best']:
        row=hf.loc[hf.policy.eq(policy)&hf.method.eq('A4')&hf.metric.eq('reconstruction_to_truth_high_energy_ratio')&hf.field.eq('U1')].iloc[0]
        prevalence.append(100*float(row['fraction_gt_2']))
    results+=rf' IID mode-sum high-band velocity power exceeds twice truth in {prevalence[0]:.1f}\% / {prevalence[1]:.1f}\% of states (last/best). All ten IID-minus-full high-band residual comparisons (five fields, two policies) have positive mean differences and positive interval lower bounds for circular blocks 5, 20 and 50. These are paired checkpoint-conditional contrasts, not training-seed evidence.'
    write('si_ablation_results.tex',results)
    sicaps=[
      r'Physical relative-$L_2$ and fieldwise effects under both policies. Temperature is observed. Effects use log$_2$ ratios of cohort means with a nonsaturating color range; raw means, intervals, paired effects, pressure-normalization checks and the additional local-only versus no-feedback contrast are supplied in tables and standalone diagnostics.',
      r'Full RFF versus IID phase-sensitive high-band residuals and canonical shell-integrated high-band power in all five fields, separately for last and best. Intervals are mean block-20 confidence intervals; power and residual ideal values are one and zero. All stochastic variants remain in the tables.',
      r'Median shell-power ratios with IQR state-dispersion shading; empirical distributions of mode-sum high-band power and phase-sensitive residuals; separate Hann-window sensitivity. Both policies are shown. These index-space spectra do not identify a physical-wavenumber inertial range. Canonical, mode-sum and tapered quantities are distinct estimators.',
      r'Fixed-range paper JSD, overflow-retaining JSD, and predicted/truth retention for three coupled field pairs under both policies. All stochastic configurations use the same archived edges. Low conditional JSD does not establish fidelity of discarded mass or spatial alignment.',
      r'Paired best-minus-last sensitivity and temporal block-length diagnostics alongside available raw training/validation loss histories. Longer histories are retained at their actual recorded endpoints with selected checkpoint markers. The comparison is not budget matched; validation and test share a holdout. Direct-field and flow-velocity training objectives are not pooled.',
      r'Separate deterministic objective control against the full stochastic reference: macro/fieldwise physical error and coupling diagnostics under both policies. Pressure fluctuation error and complete retention/overflow companions are tabulated. Lower deterministic point error does not quantify diversity or calibration.'
    ]
    figure_inserts=[]
    for (stem,label,title),cap in zip(SI,sicaps):
        png=f'Dis_SI_Process/figures/generated/{a.timestamp}/si/{stem}_{a.timestamp}.png'
        figure_inserts.append(rf'''\begin{{figure}}[p]
\centering
\includegraphics[width=183mm]{{{png}}}
\caption{{\textbf{{{title}.}} {cap}}}
\label{{fig:si-ablation-{label}}}
\end{{figure}}
\clearpage''')
    write('si_ablation_figures.tex','\n\n'.join(figure_inserts))
    write('si_ablation_package.tex','\n'.join(rf'\input{{Dis_SI_Process/docs/generated/{a.timestamp}/latex/{name}.tex}}' for name in ['si_ablation_methods','si_ablation_results','si_ablation_figures','si_ablation_tables']))
    table_script=ROOT/'scripts/write_figure5_v7_tables.py'
    if table_script.exists():
        subprocess.run([sys.executable,str(table_script),'--timestamp',a.timestamp]+(['--strict-formal'] if a.strict_formal else []),cwd=REPO,check=True)
    commonA='Package A: 1,000 paired temporally related states, all five fields, 256 temperature observations, 40,300 queries; last primary and best sensitivity. Stochastic configurations use one draw, two Euler steps and an imposed clamp; deterministic regression uses a direct forward pass without a draw or flow integration. Mean intervals: circular blocks 20 / 2,000 replicates; paired sensitivities 5/20/50; reconstruction seed 20260906 and high-frequency seed 20260910. Physical macro excludes T. Confidence intervals condition on these saved checkpoints, not training repeats.'
    commonB='Package B: archived benchmark checkpoint identities; a/b/d use200 paired states and64 draws with moving-block25/2,000 intervals; a/b bootstrap seed20260830, d method-specific recorded bootstrap_seed in benchmark_main_d.csv. f uses original1,000-state accuracy and separately measured resource protocols. These results were not remeasured for the new ablation reference.'
    panel_specs={
      'a':('Does the archived empirical conditional ensemble assign useful marginal predictive distributions?','Normalized empirical CRPS = mean over points and four unobserved fields of [mean_k|X_k-y| - (1/(2K²))sum_kl|X_k-X_l|] / frozen training-field standard deviation. Ideal0; marginal, not phase-sensitive. Per-state values then equal-weight cohort mean. Boxes show state dispersion; mean intervals are separate.','benchmark_main_a_summary','DMF-Gen has the lowest accepted normalized CRPS. This is not a demonstration of perfect calibration; inherited reliability remains underdispersed.'),
      'b':('Does larger ensemble spread track harder states?','Spearman rank association between macro normalized spread and macro ensemble-mean relative-L2 across states; range[-1,1], positive is informative, zero reference. The cloud contains accepted bootstrap estimates, not independent models.','benchmark_main_b_summary','Positive DMF association supports uncertainty informativeness. Association does not establish calibration, prospective prediction or causality.'),
      'c':('How do saved stochastic route/prior configurations compare in physical reconstruction?','Per-state physical field relative-L2 followed by equal-weight four-field macro and cohort mean; ideal0. Raw mean95%block20 intervals. Annotation100*(variant cohort mean/full cohort mean-1), not a mean of statewise ratios or percent-effect CI.','ablation_primary_summary','Route-removal implementations have larger error. IID improves whole-field error. Local-only differs in optimized capacity and multiple routes, so this is not an isolated equal-capacity causal estimate.'),
      'd':('Does uncertainty support selecting states with lower relative reconstruction error?','Retain least-uncertain fraction q=0.2,...,1.0, divide retained-set mean error by same method full-cohort mean; endpoint1. Rank is recomputed in each accepted resample. AURC is integral over0.2–1.0, not divided by0.8.','benchmark_main_d','DMF retained-set relative error is lower; the metric is within-method normalized and must be read with absolute accuracy. No new metric fills the wider layout.'),
      'e':('What does IID’s small global-error improvement conceal in fine-scale velocity?','Two distinct per-state high-band quantities: canonical trapezoidal shell-mean power ratio (ideal1) and complex-mode residual relative-L2 (ideal0, phase-sensitive). Mean block20 intervals; full/IID only, last policy.','main_ablation_plot_coordinates','IID has excess high-band velocity power and roughly twice the high-band residual. Full power is attenuated and is not ideal. This is an index-space, one-draw, two-step diagnostic.'),
      'f':('What accuracy and measured update/inference footprints accompany the archived benchmark methods?','Five columns: mean physical macro error with stateCI; median synchronized training-update ms with IQR; peak allocated trainingGiB; median warm inference ms with IQR; Model stateMiB and Peak allocatedMiB. TrainingB32 retains native workloads; inferenceB1.','benchmark_main_f','The archived accuracy/resource comparison is preserved. IQR timing variation is not a confidence interval; per-update time is not total training time; stage maxima are not additive concurrent costs.')}
    sections=[]
    for panel,(question,definition,table_name,boundary) in panel_specs.items():
        t=load(table_name)
        if panel in ['a','b']:cols=['method','metric_value','ci_low','ci_high']
        elif panel=='c':
            t=t[t.metric.eq('physical_relative_l2')&t.target.eq('Unobserved_mean')&t.policy.eq('last')&t.method.ne('A1')].copy()
            t=t.set_index('method').loc[['A0','A2','A3','A5','A4']].reset_index()
            plotted=load('main_ablation_plot_coordinates').query("panel == 'c'").set_index('method')
            t['annotation']=t.method.map(plotted.annotation)
            cols=['display_label','policy','mean','block20_ci95_low','block20_ci95_high','annotation']
        elif panel=='d':cols=['method','coverage_fraction','risk','ci_low','ci_high','risk_auc']
        elif panel=='e':t=t[t.panel.str.startswith('e')];cols=['display_label','metric','mean','block20_ci95_low','block20_ci95_high','annotation']
        else:cols=['method','mean_unobserved_relative_l2','error_ci_low','error_ci_high','training_update_ms','training_peak_allocated_mib','inference_latency_ms','model_state_mib','inference_peak_allocated_mib']
        protocol=commonB if panel in 'abdf' else commonA
        text=f'# Figure5{panel} quantitative companion\n\n{question}\n\n{protocol}\n\nMetric and aggregation: {definition}\n\nResult and boundary: {boundary}\n\nCoordinates at source precision:\n\n'+mdtable(t[cols])+f'\nSource rows: `{logical(d/(table_name+".csv"))}`. Exact input paths/hashes and schema filters are in source_manifest.json; full plotted artist coordinates and layout in main_artist_coordinates.json.\n'
        (docs/f'fig5{panel}_companion.md').write_text(text)
        sections.append(text.replace('# Figure5', '## Figure5',1))
    si_manifest_path=ROOT/'figures/generated'/a.timestamp/'si/si_plot_manifest.json';si_manifest=json.loads(si_manifest_path.read_text()) if si_manifest_path.exists() else {}
    si_dir=docs/'si_companions';si_dir.mkdir(exist_ok=True)
    for n,((stem,label,title),cap) in enumerate(zip(SI,sicaps),1):
        plots=[p for p in si_manifest.get('plots',[]) if str(p.get('id','')).startswith(f'si{n:02}')]
        detail=f'# SI-{n:02}: {title}\n\n{commonA}\n\nPurpose and interpretation: {cap}\n\nEach standalone diagnostic retains full metric/policy context. The renderer manifest provides source-key to exact-file/hash mappings, all numerical coordinates, interval endpoints, transformations, labels and visual encodings.\n\n'
        for plot in plots:
            panel_id=plot.get('id','')
            detail+='## '+panel_id+'\n\n```json\n'+json.dumps({k:v for k,v in plot.items() if k!='coordinates'},ensure_ascii=False,indent=2)+'\n```\n\n'
            if plot.get('coordinates'):
                source_records={key:si_manifest.get('sources',{}).get(key,{}) for key in plot.get('source_keys',[])}
                companion=f'# {panel_id}: quantitative companion\n\n{commonA}\n\n{cap}\n\nComplete plotted coordinates and display definitions (source precision):\n\n```json\n'+json.dumps(plot,ensure_ascii=False,indent=2)+'\n```\n\nExact source identities:\n\n```json\n'+json.dumps(source_records,ensure_ascii=False,indent=2)+'\n```\n'
                (si_dir/(panel_id+'_companion.md')).write_text(companion)
                detail+=f'Individual panel companion with every plotted coordinate: [{panel_id}]({panel_id}_companion.md).\n\n'
        detail+=f'Full coordinates: `{logical(si_manifest_path)}`. Raw means and intervals: ablation_all_summary.csv and ablation_highband_summary.csv; paired estimates: ablation_all_paired.csv and ablation_highband_paired.csv. Both policies retained. QA: source validation {sourceqa["status"]}; final artwork status is recorded in qa.json.\n'
        (si_dir/(stem+'_companion.md')).write_text(detail)
        report_detail=detail.replace('# SI-','## SI-',1)
        for plot in plots:
            panel_id=plot.get('id','')
            report_detail=report_detail.replace(f']({panel_id}_companion.md)',f'](si_companions/{panel_id}_companion.md)')
        sections.append(report_detail)
    checkpoint_map=prov[['display_label','policy','checkpoint_epoch','global_step','parameter_elements_with_optimizer_state','scheduler_t_max','checkpoint_sha256']]
    checkpoint_text='# Checkpoints and comparison scope\n\n'+commonA+'\n\n'+commonB+'\n\n'+mdtable(checkpoint_map)+'\n```yaml\n'+yaml.safe_dump(cfg['comparison_scope'])+'```\n'
    (docs/'checkpoint_and_scope_notes.md').write_text(checkpoint_text)
    contract=f'''# Figure contract\n\nConclusion: the available full stochastic model reconstructs more accurately than its conditioning-route variants; IID slightly improves whole-field error while producing excess fine-scale velocity power and larger phase-sensitive residuals.\n\nArchetype: quantitative grid; dedicated right-hand architecture/prior column. Python/Matplotlib in fig only. Main canvas183×205mm; editable SVG plus600dpiPNG, no new PDF delivery. Panel map aCRPS, bspread/error, cstochastic controls, dselective risk across two columns, etwo high-band axes, ffull-width five-column scorecard.\n\nAll old benchmark coordinates and method identities are preserved by accepted-renderer scientific geometry digests. Layout changes: larger canvas; repositioned axes; standalone bmethod labels and dlegend; minimum inherited annotations5.8pt; Training time wording becomes Training update time; isolated AURC annotation removed; filled Model/hollow Peak retained. Full red; route-removal slate; IID muted amber. New axes use6pt row/tick text and6.6pt labels. No smoothing/clipping/inference.\n\nEvidence packages and source identity are defined in source_manifest.json. {commonA} {commonB}\n'''
    (docs/'figure_contract.md').write_text(contract)
    trade=[]
    for pol in ['last','best']:
        for m in ['A0','A4']:
            trade.append({'policy':pol,'model':cfg['styles'][m]['label'],'macro_L2':value(pol,m,'physical_relative_l2','Unobserved_mean'),'U1_canonical_power':high(pol,m,'canonical_shellmean_high_energy_ratio'),'U1_highband_L2':high(pol,m,'highband_error_relative_l2'),'U1_whole_LSD_dB':high(pol,m,'canonical_spectral_lsd_db'),'U1_residual_full_fluctuation_percent':100*high(pol,m,'highband_error_energy_over_truth_total_fluctuation_energy')})
    finalqa=json.loads((d/'qa.json').read_text()) if (d/'qa.json').exists() else {'status':'pending artwork/package audit'}
    report='# Quantitative figure-making report: Figure5 V7\n\n'+contract.split('\n\n',1)[1]+'\n\n## Prior-source tradeoff\n\n'+mdtable(pd.DataFrame(trade))+'\n## Checkpoint map\n\nArchived benchmark DMF SHA256: '+str(load('benchmark_main_f').iloc[0].checkpoint_sha256)+'; its mean error remains0.1170851006737255 (display0.117). Full ablation hashes below are distinct; no archived64-draw UQ or cost is reassigned to them.\n\n'+mdtable(checkpoint_map)+'\n## Exact command and environment record\n\n```json\n'+json.dumps(manifest['preflight'],indent=2)+'\n```\n\n```bash\nrtk proxy conda run -n fig python Dis_SI_Process/scripts/build_figure5_v7_bundle.py --timestamp '+a.timestamp+' --strict-formal\nrtk proxy conda run -n fig python -m unittest Dis_SI_Process.tests.test_figure5_v7_ablation\n```\n\nNo training, inference, GPU timing, new solver settings or fresh bootstrap was run for this release. Raw historical curves are descriptive recordings, not dense reconstruction-error trajectories or independent repeats. All rows available at 2026-09-10 20:38:23 UTC were frozen in the SI history snapshot; the full-model history contains8026 rows, while its evaluated last checkpoint is epoch7520. The snapshot manifest records each captured source hash and compact CSV hash; reruns use these frozen histories.\n\n## Panel-reference map\n\n'+mdtable(load('panel_reference_map'))+'\n## All five benchmark selective-risk areas\n\n'+mdtable(load('benchmark_aurc')[['method','risk_auc']])+'\nThese areas integrate the normalized curve over0.2–1.0 and are not divided by0.8. At80%retention the DMF within-method normalized risk is '+format(float(load('benchmark_main_d').query("method == 'DMF-Gen' and coverage_fraction == 0.8").iloc[0].risk),'.17g')+'.\n\n'+'\n\n'.join(sections)
    report+='\n## Admissibility event ledger\n\n'+mdtable(load('ablation_admissibility_events')[['display_label','policy','target','snapshot','time_index','event_count','minimum']])+'\nDenominator40,300,000 points per configuration/policy. Tables retain species negative frequencies, below−0.0001 frequencies, negative-part norms/magnitudes, exact T/p counts, global minima and their state/time identities. Rounded zero percentages are not evidence of zero events.\n'
    effects=load('ablation_highband_paired')
    effects=effects.loc[effects.policy.isin(['last','best'])&effects.method.eq('A4')&effects.baseline.eq('A0')&effects.metric.eq('highband_error_relative_l2')]
    if len(effects)!=10 or not effects[['block5_ci95_low','block20_ci95_low','block50_ci95_low']].gt(0).all().all():raise ValueError('All-field paired residual statement does not match saved intervals')
    report+='\n## All-field paired fine-scale residual sensitivity\n\nIID minus full; all ten saved comparisons retain positive lower bounds at every block length.\n\n'+mdtable(effects[['policy','field','method_mean_minus_baseline','block5_ci95_low','block5_ci95_high','block20_ci95_low','block20_ci95_high','block50_ci95_low','block50_ci95_high']])
    report+='\n## Scope and measurement boundaries\n\n'+methods.replace('\\paragraph','Paragraph ')+'\n\n'+results+'\n\n## Validation status\n\nSource QA: '+sourceqa['status']+'; '+str(len(sourceqa['checks']))+' source checks. Final package QA at document generation: '+str(finalqa['status'])+'. The live final status is results/derived/'+a.timestamp+'/qa.json. SVG visual review is bound to output hashes; source integrity does not establish equal-budget causal identification. The optional cached-field gallery is not needed for these metric figures.\n\n## Source ledger\n\n'
    report+=mdtable(pd.DataFrame([{'role':r['role'],'path':r['path'],'sha256':r['sha256']} for r in manifest['sources'].values()]))
    report+='\n## Complete precision and display audit\n\n`figure5_v7_display_source.csv` carries the evidence-package/checkpoint/cohort/metric/source identities. `main_artist_coordinates.json` records every main artist line, scatter offset, confidence-band path, annotation position and axis transform. `main_ablation_plot_coordinates.csv` records new point/interval values and formatted annotations. `si_plot_manifest.json` records every standalone SI plotted coordinate, scientific definition, source hash/filter, interval versus dispersion, and encoding; composites use the same complete scientific subpanels. Full summary and paired-effect tables preserve source precision. Histogram context stays attached in ablation_coupling_audit.csv; no policies are pooled.\n'
    (docs/'quantitative_figure_making_report.md').write_text(report)
    index=f'# Figure 5 V7 release {a.timestamp}\n\n[Completion and QA](completion_report.md) · [Quantitative report](quantitative_figure_making_report.md) · [Checkpoint scope](checkpoint_and_scope_notes.md)\n\n'
    index+=f'[Main SVG](../../../figures/generated/{a.timestamp}/fig5_composed_v7_{a.timestamp}.svg) · [600-dpi PNG](../../../figures/generated/{a.timestamp}/fig5_composed_v7_{a.timestamp}.png) · [Main LaTeX environment](latex/figure5_v7_figure.tex)\n\n'
    index+='[Complete SI LaTeX entry point](latex/si_ablation_package.tex). Compile from the repository root with graphicx, booktabs, array, longtable and amsmath available. SVG is the editable artwork; LaTeX uses the delivered PNGs.\n\n'
    for stem,label,title in SI:
        index+=f'- {title}: [SVG](../../../figures/generated/{a.timestamp}/si/{stem}_{a.timestamp}.svg), [PNG](../../../figures/generated/{a.timestamp}/si/{stem}_{a.timestamp}.png), [companion](si_companions/{stem}_companion.md).\n'
    index+='\nAll individual main and SI subpanels are beside their composites; individual SI companions contain full plotted coordinates and source identities. Numerical audit files are under `results/derived/'+a.timestamp+'`.\n'
    (docs/'README.md').write_text(index)
    print(json.dumps({'source_status':sourceqa['status'],'report':str(docs/'quantitative_figure_making_report.md'),'table_generator_available':table_script.exists()}))


def logical(path):return str(path.relative_to(REPO))


if __name__=='__main__':main()

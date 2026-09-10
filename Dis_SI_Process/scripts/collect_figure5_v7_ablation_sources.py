#!/usr/bin/env python
"""Validate immutable Figure 5 evidence and export compact source tables only."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
sys.path.insert(0,str(REPO))
from Dis_SI_Process.utils.figure5_v7_ablation_data import (
    DISPLAY_INFO,PRIMARY_METRICS,COUPLING_METRICS,ADMISSIBILITY_METRICS,
    decorate_row,sha256,as_jsonable,
)
KEYS={'A0':'ablation_full','A2':'ablation_no_sensor_feedback','A3':'ablation_no_local_conditioning',
      'A5':'ablation_local_only_conditioning','A4':'ablation_iid_prior','A1':'deterministic_objective_control'}
ORDER=['A0','A2','A3','A5','A4','A1']
REPORT_HASH='84d5c9fc708988f51f95d4668ef84786822367570e19bd20f71c466e632a1138'


def logical(path):
    try:return str(Path(path).relative_to(REPO))
    except ValueError:return str(path)


def validate_state_table(table,field='target',policies=False):
    grouping=(['policy'] if policies else [])+['method','metric',field]
    identity=grouping+['snapshot','time_index']
    if table.duplicated(identity).any():raise ValueError('Duplicate state/metric identities')
    if not table.groupby(grouping,dropna=False).size().eq(1000).all():raise ValueError('Expected exactly 1000 states per group')
    if not np.isfinite(table.value.to_numpy(float)).all():raise ValueError('Nonfinite state values')
    if not table.groupby(grouping).snapshot.nunique().eq(1000).all():raise ValueError('Repeated state index')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--timestamp',required=True);p.add_argument('--strict-formal',action='store_true')
    p.add_argument('--evaluation-dir',type=Path)
    p.add_argument('--benchmark-release',type=Path)
    args=p.parse_args()
    cfg=yaml.safe_load((ROOT/'configs/figure5_v7_ablation.yaml').read_text())
    ev=REPO/cfg['evaluation_root'];out=ROOT/'results/derived'/args.timestamp;out.mkdir(parents=True,exist_ok=True)
    if args.evaluation_dir is not None and args.evaluation_dir.resolve()!=ev.resolve():
        raise ValueError('This frozen release requires the configured evaluation_20260910 source; no older-reference substitution is allowed')
    if args.benchmark_release is not None and args.benchmark_release.resolve()!=(ROOT/'figures/generated/20260904_1200').resolve():
        raise ValueError('This release requires the accepted benchmark release 20260904_1200')
    registry={};checks={};errors=[];tables={}
    def check(name,condition,detail=None):
        checks[name]={'pass':bool(condition),'detail':as_jsonable(detail)}
        if not condition:errors.append(name)
    def reg(path,role='saved evidence',protect=True):
        path=Path(path);key=logical(path)
        if key not in registry:
            if not path.is_file():raise FileNotFoundError(path)
            registry[key]={'path':key,'resolved_path':str(path.resolve()),'sha256':sha256(path),'size_bytes':path.stat().st_size,'role':role,'protected':protect}
        return registry[key]
    def js(path):reg(path);return json.loads(Path(path).read_text())
    def csv(path,**kwargs):
        entry=reg(path);t=pd.read_csv(path,**kwargs);entry.update({'columns':list(t.columns),'rows':len(t)});return t
    def write(name,t):
        t.to_csv(out/(name+'.csv'),index=False);tables[name]={'path':logical(out/(name+'.csv')),'rows':len(t),'columns':list(t.columns)}
        return t
    try:
        # Every previous release artifact is protected, not just its manifest.
        for folder in ['figures/generated','results/derived','docs/generated']:
            for path in (ROOT/folder/'20260904_1200').rglob('*'):
                if path.is_file():reg(path,'protected accepted release')
        old_manifest=js(ROOT/'results/derived/20260904_1200/build_manifest.json')
        check('accepted_renderer',old_manifest['renderer'].endswith('build_figure5_v6_nature_format_v5.py'))
        oldcfg=yaml.safe_load((ROOT/'configs/figure5_v6.yaml').read_text());reg(ROOT/'configs/figure5_v6.yaml')
        for value in oldcfg['sources'].values():reg(REPO/value,'Package B accepted lineage')
        uq_lineage=js(REPO/oldcfg['sources']['v5_manifest'])
        uq_folder=Path(next(r['source'] for r in uq_lineage['reuse_matrix'] if r['panel']=='a')).parent
        for name in ['manifest.json','qa.json','method_draw_audit.csv','crps_summary.csv','spread_error_summary.csv','per_state_method.csv']:
            reg(uq_folder/name,'Package B formal 200-state 64-draw UQ provenance')
        formal_uq=js(uq_folder/'manifest.json')
        check('formal_uq_identity',formal_uq['formal'] and formal_uq['status']=='complete' and len(formal_uq['states'])==200 and formal_uq['draws_per_state']==64 and all(r['pass'] for r in formal_uq['identity_checks']))
        for path in (ROOT/'figures/scripts').glob('build_figure5_v6*.py'):reg(path,'protected inherited renderer')
        report=ev/'Ablation_CondT_Evaluation_20260910.md';check('frozen_report_hash',reg(report)['sha256']==REPORT_HASH)
        reg(ev.parent/'Ablation_CondT_Evaluation_20260906.md','identical report alias')
        snapshot=js(ev/'checkpoint_inputs/A0/snapshot_manifest.json')
        audit=js(ev/'checkpoint_audit.json');runs={r['id']:r for r in audit['runs']}
        env=js(ev/'evaluation_environment.json');validation=js(ev/'validation_summary.json')
        for name in ['A0_training_progress.csv','checkpoint_sensitivity_vs_last.csv','source_hashes.json','coverage_audit.json']:
            reg(ev/name)
        metainfo={};provs={};prov_rows=[];cache_frames=[];manifests=[]
        for policy in ['last','best']:
            meta=js(ev/policy/'metrics/analysis_metadata.json')
            check(policy+'_metadata_complete',not meta['partial'] and meta['truth_sensor_temporal_identity_matches'])
            for method in ORDER:
                paths=list((ev/policy).glob(f'provenance_{method}.json'))
                if len(paths)!=1:raise ValueError(f'Ambiguous provenance {policy}/{method}')
                path=paths[0];pr=js(path);provs[policy,method]=pr
                cp=runs[method]['checkpoints'][policy+'.pt']
                check(policy+'_'+method+'_checkpoint_identity',pr['checkpoint_sha256']==cp['sha256'] and pr['checkpoint_epoch']==cp['epoch'])
                check(policy+'_'+method+'_strict_load',pr['strict_load'] and pr['completed_snapshots']==1000)
                check(policy+'_'+method+'_generation',pr['n_steps']==2 and pr['ode_solver']=='euler' and pr['obs_consistency_applied']=='default_hard')
                if method=='A0':
                    frozen=snapshot['checkpoints'][policy+'.pt']
                    check(policy+'_frozen_A0',cp['epoch']=={'last':7520,'best':7095}[policy] and pr['checkpoint_path']==frozen['frozen_path'] and sha256(Path(frozen['frozen_path']))==frozen['sha256'])
                sensor=Path(pr['sensor_plan']);check(policy+'_'+method+'_sensor_plan',reg(sensor)['sha256']==pr['sensor_plan_hash'])
                cpath=Path(pr['config_path']);config=yaml.safe_load(cpath.read_text());reg(cpath)
                expected_time=pr['test_time_indices']
                cohort_hash=hashlib.sha256(json.dumps(expected_time,separators=(',',':')).encode()).hexdigest()
                m=dict(pr,policy=policy,cohort_id='ablation_condT_1000_'+cohort_hash[:12],cohort_identity=cohort_hash,
                       sensor_plan_path=str(sensor),sensor_plan_sha256=pr['sensor_plan_hash'])
                metainfo[policy,method]=m
                base={'method':method,'internal_key':KEYS[method],'display_label':DISPLAY_INFO[method]['label'],
                      'expanded_definition':DISPLAY_INFO[method]['definition'],'global_step':cp.get('global_step','not recorded'),
                      'optimizer_state_parameter_elements':cp.get('parameter_elements_with_optimizer_state','not recorded'),
                      'parameter_elements_with_optimizer_state':cp.get('parameter_elements_with_optimizer_state','not recorded'),
                      'scheduler_t_max':config.get('scheduler_t_max','not recorded'),'configured_epochs':config.get('epochs','not recorded'),
                      'learning_rate':config.get('lr','not recorded'),'training_batch_size':config.get('batch_size','not recorded'),
                      'training_query_count':config.get('n_query_points','not recorded'),
                      'selection_objective':'validation direct-field MSE' if method=='A1' else 'validation flow-velocity MSE',
                      'train_loss':cp.get('train_loss'),'val_loss':cp.get('val_loss'),
                      'run_directory':snapshot['source_run_directory'] if method=='A0' else pr['run_directory'],
                      'generative_draws':0 if method=='A1' else 1,'comparison_role':'deterministic objective control' if method=='A1' else 'stochastic architecture/prior comparison'}
                prov_rows.append(decorate_row(base,meta=m,source_path=path,source_hash=reg(path)['sha256'],repo_root=REPO))
                mfpath=ev/policy/f'manifest_{method}.csv';mf=csv(mfpath)
                check(policy+'_'+method+'_manifest_complete',len(mf)==1000 and mf.snapshot.nunique()==1000 and set(mf.status)=={'ok'})
                check(policy+'_'+method+'_manifest_checkpoint',mf.checkpoint_sha256.eq(cp['sha256']).all() and mf.checkpoint_epoch.eq(cp['epoch']).all())
                check(policy+'_'+method+'_time_indices',mf.sort_values('snapshot').time_index.tolist()==expected_time)
                manifests.append(mf.assign(policy=policy))
            cache_frames.append(csv(ev/policy/'metrics/cache_audit.csv').assign(policy=policy))
        provenance=write('ablation_provenance',pd.DataFrame(prov_rows))
        def enrich(t,path,policy=None,hf=False):
            result=[]
            for row in t.to_dict('records'):
                pol=policy or row.get('policy','last');basepol='best' if pol in ['best_vs_last','best_minus_last'] else pol
                method=row['method'];m=metainfo[basepol,method]
                rec=decorate_row(row,meta=m,source_path=path,source_hash=reg(path)['sha256'],repo_root=REPO,high_frequency=hf,policy_override=pol)
                rec.update(internal_key=KEYS[method],display_label=DISPLAY_INFO[method]['label'])
                if row.get('baseline') in KEYS:
                    reference=metainfo[basepol,row['baseline']];rec.update(baseline_checkpoint_sha256=reference['checkpoint_sha256'],baseline_internal_key=KEYS[row['baseline']])
                if pol in ['best_vs_last','best_minus_last']:
                    rec.update(baseline_checkpoint_sha256=metainfo['last',method]['checkpoint_sha256'],baseline_checkpoint_policy='last',checkpoint_policy='best versus last')
                result.append(rec)
            return pd.DataFrame(result)
        summaries=[];paireds=[]
        for pol in ['last','best']:
            path=ev/pol/'metrics/summary_metrics.csv';summaries.append(enrich(csv(path),path,pol))
            path=ev/pol/'metrics/paired_differences.csv';paireds.append(enrich(csv(path),path,pol))
        summary=pd.concat(summaries,ignore_index=True);paired=pd.concat(paireds,ignore_index=True)
        write('ablation_primary_summary',summary[summary.metric.isin(PRIMARY_METRICS)])
        write('ablation_primary_paired',paired[paired.metric.isin(PRIMARY_METRICS)])
        write('ablation_all_summary',summary);write('ablation_all_paired',paired)
        path=ev/'high_frequency/summary_high_frequency.csv';hf=write('ablation_highband_summary',enrich(csv(path),path,hf=True))
        path=ev/'high_frequency/paired_high_frequency.csv';hfp=write('ablation_highband_paired',enrich(csv(path),path,hf=True))
        write('ablation_spectral_summary',summary[summary.metric.str.startswith('spectral_')])
        coupling=summary[summary.metric.isin(COUPLING_METRICS)].copy()
        coupling['histogram_bins']='64x64 paper;66x66 overflow';coupling['histogram_edges']='common archived truth 0.5–99.5% edges; unchanged across methods/policies'
        idx=summary.set_index(['policy','method','metric','target'])
        coupling['predicted_retained_fraction']=[idx.loc[(r.policy,r.method,'joint_pdf_reconstruction_retained_fraction',r.target),'mean'] for r in coupling.itertuples()]
        coupling['truth_retained_fraction']=[idx.loc[(r.policy,r.method,'joint_pdf_truth_retained_fraction',r.target),'mean'] for r in coupling.itertuples()]
        write('ablation_coupling_audit',coupling);write('ablation_coupling_paired',paired[paired.metric.isin(COUPLING_METRICS)])
        write('deterministic_control_summary',summary[summary.method.isin(['A0','A1'])])
        write('deterministic_control_paired',paired[paired.method.eq('A1')&paired.baseline.eq('A0')])
        path=ev/'checkpoint_sensitivity_vs_last.csv';sensitivity=enrich(csv(path),path,'best_minus_last');write('checkpoint_sensitivity',sensitivity)
        path=ev/'A0_training_progress.csv';progress=enrich(csv(path).assign(method='A0'),path);write('A0_training_progress',progress)
        # Full schemas and summary tables are available before slower per-state audits.
        name_map=pd.DataFrame([dict(internal_run=m,method=m,internal_key=KEYS[m],display_label=DISPLAY_INFO[m]['label'],expanded_definition=DISPLAY_INFO[m]['definition'],main_column=m!='A1',scientific_order=i) for i,m in enumerate(ORDER)])
        write('ablation_name_map',name_map)
        write('panel_reference_map',pd.DataFrame([{'old_panel':old,'new_panel':new,'role':role} for old,new,role in [('a','a','Normalized CRPS'),('b','b','Spread–error association'),('c','d','Selective reconstruction'),('d','f','Accuracy and footprint'),('new','c','Conditioning and source variants'),('new','e','Fine-scale velocity fidelity')]]))
        caches=pd.concat(cache_frames);allmf=pd.concat(manifests)
        check('cache_audit_complete',len(caches)==12000 and not caches.duplicated(['policy','method','snapshot']).any())
        identities=caches.groupby('snapshot')[['truth_sha256','sensor_sha256','time_index','generation_seed']].nunique()
        check('paired_truth_sensor_time_seed_identity',len(identities)==1000 and identities.eq(1).all().all())
        check('paired_measurement_manifest_identity',allmf.groupby('snapshot')[['sensor_seed','sensor_plan_hash','time_index','generation_seed']].nunique().eq(1).all().all())
        cohort=caches[caches.method.eq('A0')&caches.policy.eq('last')][['snapshot','time_index','truth_sha256','sensor_sha256','generation_seed']].sort_values('snapshot')
        write('evaluation_state_identity',cohort)
        event_records=[];count_records=[];per_state_macro=[]
        for pol in ['last','best']:
            path=ev/pol/'metrics/per_state_metrics.csv';state=csv(path);validate_state_table(state)
            group=state.groupby(['method','metric','target']).value.mean()
            sm=summary[summary.policy.eq(pol)].set_index(['method','metric','target'])['mean']
            check(pol+'_summary_means_match_saved_states',np.allclose(group.sort_index(),sm.sort_index(),rtol=1e-12,atol=1e-14),{'metric_groups':len(group),'rows':len(state)})
            keyed=state.merge(cohort[['snapshot','time_index']],on=['snapshot','time_index'],how='left',indicator=True)
            check(pol+'_state_time_pairing',keyed['_merge'].eq('both').all())
            physical=state[state.metric.eq('physical_relative_l2')]
            pivot=physical.pivot(index=['method','snapshot','time_index'],columns='target',values='value')
            check(pol+'_primary_macro_definition',np.allclose(pivot['Unobserved_mean'],pivot[['CH4','CO','U1','p']].mean(axis=1),rtol=1e-12,atol=1e-14))
            macro=physical[physical.target.eq('Unobserved_mean')].copy();macro['policy']=pol;per_state_macro.append(macro)
            for method in ORDER:
                for field in ['CH4','CO','T','p']:
                    r=state[state.method.eq(method)&state.target.eq(field)&state.metric.eq('reconstruction_nonphysical_fraction')]
                    counts=r.value.to_numpy()*40300
                    check(f'{pol}_{method}_{field}_integral_event_counts',np.max(np.abs(counts-np.rint(counts)))<1e-6)
                    total=int(np.rint(counts).sum())
                    minima=state[state.method.eq(method)&state.target.eq(field)&state.metric.eq('reconstruction_minimum')]
                    minimum=minima.value.min();minids=minima[minima.value.eq(minimum)]
                    count_records.append(dict(method=method,policy=pol,metric='exact_nonphysical_count',target=field,event_count=total,total_points=40300000,global_minimum=minimum,minimum_state_ids=';'.join(minids.snapshot.astype(str)),minimum_time_indices=';'.join(minids.time_index.astype(str))))
                    if field in ['T','p']:
                        bad=r.loc[np.rint(counts)>0].merge(minima[['snapshot','time_index','value']],on=['snapshot','time_index'],suffixes=('_fraction','_minimum'))
                        for event in bad.itertuples():event_records.append(dict(method=method,policy=pol,metric='nonpositive_event',target=field,snapshot=event.snapshot,time_index=event.time_index,event_count=int(round(event.value_fraction*40300)),minimum=event.value_minimum,total_points_per_state=40300))
            # Pair by both original identities; never CSV order.
            v=macro.pivot(index=['snapshot','time_index'],columns='method',values='value')
            check(pol+'_IID_fraction_lower',int((v.A4<v.A0).sum())=={'last':573,'best':626}[pol])
        count_table=pd.DataFrame(count_records)
        events=pd.DataFrame(event_records)
        write('ablation_admissibility_events',pd.concat([enrich(events[events.policy.eq(pol)],ev/pol/'metrics/per_state_metrics.csv',pol) for pol in ['last','best']],ignore_index=True))
        counts_enriched=pd.concat([enrich(count_table[count_table.policy.eq(pol)],ev/pol/'metrics/per_state_metrics.csv',pol) for pol in ['last','best']],ignore_index=True)
        write('ablation_admissibility_counts',counts_enriched)
        admissibility=summary[summary.metric.isin(ADMISSIBILITY_METRICS)|summary.metric.str.endswith('excluding_sensors')].copy()
        write('ablation_admissibility',pd.concat([admissibility,counts_enriched],ignore_index=True))
        write('ablation_macro_per_state',pd.concat([enrich(t,ev/t.policy.iloc[0]/'metrics/per_state_metrics.csv',t.policy.iloc[0]) for t in per_state_macro],ignore_index=True))
        hmeta=js(ev/'high_frequency/analysis_metadata.json')
        check('canonical_highband_definition',hmeta['grid']['high_shell_count']==68 and hmeta['grid']['high_mode_count']==17704 and hmeta['grid']['min_shell_count']==4 and hmeta['grid']['isotropic_cutoff'])
        check('highband_original_seed',hmeta['statistics']['seed']==20260910 and hmeta['statistics']['bootstrap_resamples']==2000)
        hp=csv(ev/'high_frequency/per_state_high_frequency.csv');validate_state_table(hp,'field',True)
        check('high_frequency_source_complete',len(hp)==1440000)
        hg=hp.groupby(['policy','method','metric','field']).value.mean().sort_index();hs=hf.set_index(['policy','method','metric','field'])['mean'].sort_index()
        check('highband_summary_means_match_saved_states',np.allclose(hg,hs,rtol=1e-12,atol=1e-14))
        for name in ['population_U1_spectra.csv','U1_hann_robustness_summary.csv']:reg(ev/'high_frequency'/name)
        hfprior=hfp[hfp.comparison.eq('ablation_vs_A0')&hfp.method.eq('A4')&hfp.metric.eq('highband_error_relative_l2')]
        check('all_ten_prior_highband_effects_positive',len(hfprior)==10 and hfprior[['block5_ci95_low','block20_ci95_low','block50_ci95_low']].gt(0).all().all())
        # Regression anchors are validation-only and never plotting coordinates.
        expected={'last':['0.106321','0.126974','0.144703','0.354951','0.104294','0.078705'], 'best':['0.107893','0.129288','0.146284','0.356403','0.104967','0.078561']}
        for pol in ['last','best']:
            for method,value in zip(ORDER,expected[pol]):check(f'anchor_{pol}_{method}',f"{idx.loc[(pol,method,'physical_relative_l2','Unobserved_mean'),'mean']:.6f}"==value)
        def anchor(name,value,expected):
            digits=len(expected.split('.')[-1]) if '.' in expected else 0
            check('anchor_'+name,format(float(value),f'.{digits}f')==expected,{'actual':float(value),'expected_at_report_precision':expected})
        intervals={'last':[('0.104495','0.108229'),('0.124868','0.129248'),('0.142757','0.146664'),('0.351422','0.358696'),('0.102412','0.106302'),('0.077359','0.080195')],
                   'best':[('0.106044','0.109840'),('0.127076','0.131656'),('0.144486','0.148103'),('0.352764','0.360126'),('0.103069','0.106949'),('0.077237','0.079971')]}
        for pol,values in intervals.items():
            for method,(lo,hi) in zip(ORDER,values):
                rr=idx.loc[(pol,method,'physical_relative_l2','Unobserved_mean')]
                anchor(pol+'_'+method+'_CI_low',rr.block20_ci95_low,lo);anchor(pol+'_'+method+'_CI_high',rr.block20_ci95_high,hi)
        for pol,values in {'last':['-0.002027','-0.002634','-0.001416'],'best':['-0.002926','-0.003527','-0.002308']}.items():
            rr=paired[paired.policy.eq(pol)&paired.method.eq('A4')&paired.baseline.eq('A0')&paired.metric.eq('physical_relative_l2')&paired.target.eq('Unobserved_mean')].iloc[0]
            for column,value in zip(['mean_difference','block20_ci95_low','block20_ci95_high'],values):anchor(pol+'_IID_paired_'+column,rr[column],value)
        hi=hf.set_index(['policy','method','metric','field'])
        u1={('last','A0'):['0.4285','0.873','4.5602','0.0312'],('last','A4'):['3.1199','1.825','3.7883','0.1317'],
            ('best','A0'):['0.4227','0.874','4.5819','0.0313'],('best','A4'):['3.2379','1.857','3.8503','0.1363']}
        for (pol,method),values in u1.items():
            for metric,value in zip(['canonical_shellmean_high_energy_ratio','highband_error_relative_l2','canonical_spectral_lsd_db','highband_error_energy_over_truth_total_fluctuation_energy'],values):
                factor=100 if 'total_fluctuation' in metric else 1
                anchor(pol+'_'+method+'_'+metric,hi.loc[(pol,method,metric,'U1'),'mean']*factor,value)
        anchor('truth_U1_high_energy_percent',hi.loc[('last','A0','truth_high_energy_fraction_total','U1'),'mean']*100,'0.0459')
        field_expected={('last','A0'):['0.275','0.833','0.544','0.873','0.545'],('last','A4'):['0.408','1.115','0.801','1.825','0.611'],('best','A0'):['0.277','0.837','0.548','0.874','0.547'],('best','A4'):['0.407','1.115','0.801','1.857','0.608']}
        for (pol,method),values in field_expected.items():
            for field,value in zip(['CH4','CO','T','U1','p'],values):anchor(pol+'_'+method+'_highband_'+field,hi.loc[(pol,method,'highband_error_relative_l2',field),'mean'],value)
        for pol in ['last','best']:
            for file,key in [('summary_metrics.csv','summary_sha256'),('per_state_metrics.csv','per_state_metrics_sha256'),('paired_differences.csv','paired_differences_sha256'),('cache_audit.csv','cache_audit_sha256')]:
                check(pol+'_'+file+'_recorded_audit_hash',reg(ev/pol/'metrics'/file)['sha256']==audit['cache_coverage'][pol][key])
        exp_counts={('last','A4'):3,('best','A4'):3,('last','A5'):1}
        for row in count_table[count_table.target.eq('T')].itertuples():check('temperature_count_'+row.policy+'_'+row.method,row.event_count==exp_counts.get((row.policy,row.method),0))
        check('pressure_nonpositive_counts_zero',count_table[count_table.target.eq('p')].event_count.eq(0).all())
        # Record available histories through exact run paths, never a model-name glob.
        histories=[]
        for method in ORDER:
            run=Path(snapshot['source_run_directory'] if method=='A0' else provs['last',method]['run_directory'])
            path=run/'loss_history.csv'
            if path.exists():
                entry=reg(path,'available recorded history; live file may continue',protect=False)
                t=pd.read_csv(path);histories.append({'method':method,'display_label':DISPLAY_INFO[method]['label'],'path':str(path),'sha256':entry['sha256'],'nrows':len(t),'first_epoch':t.epoch.min(),'last_recorded_epoch':t.epoch.max(),'evaluation_last_epoch':metainfo['last',method]['checkpoint_epoch']})
        write('ablation_history_provenance',pd.DataFrame(histories))
        history_snapshot=ROOT/'figures/generated'/args.timestamp/'si/history_snapshot_manifest.json'
        if history_snapshot.exists():
            captured=js(history_snapshot)
            for method,item in captured['histories'].items():
                entry=reg(REPO/item['snapshot_path'],'timestamp-frozen recorded history used by SI; all captured rows retained')
                check('history_snapshot_'+method,entry['sha256']==item['snapshot_sha256'] and item['source_unchanged_during_capture'])
                entry['raw_source_path']=item['raw_source_path'];entry['raw_sha256_at_capture']=item['raw_sha256_at_capture'];entry['rows']=item['captured_rows'];entry['columns']=item['captured_columns']
        # Package B tables come from the same accepted loader as the renderer.
        import importlib.util
        spec=importlib.util.spec_from_file_location('v7_accepted',REPO/old_manifest['renderer']);renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
        base=renderer.V1.load_base();sp=base.require_sources(oldcfg);uq=oldcfg['paper_contract']['generative_method_order']
        a,sa,b,sb=base.load_panel_ab(sp['v5_display'],uq);d=base.load_panel_c(sp['v51_selective_risk'],oldcfg)
        f=base.derive_panel_d(sp['v51_scorecard'],sp['v51_scorecard_stages'],sp['inference_memory'],oldcfg['paper_contract']['scorecard_method_order'])
        bframes=[]
        for name,t,source in [('a_samples',a,sp['v5_display']),('a_summary',sa,sp['v5_display']),('b_samples',b,sp['v5_display']),('b_summary',sb,sp['v5_display']),('d',d,sp['v51_selective_risk']),('f',f,sp['v51_scorecard'])]:
            t=t.copy();t['evidence_package']='Package B';t['checkpoint_policy']='archived accepted benchmark selection';t['source_path']=logical(source);t['source_sha256']=reg(source)['sha256'];t['internal_key']=t.method.map(lambda m:'benchmark_dmf' if m=='DMF-Gen' else 'benchmark_'+m)
            t['metric_definition']=t['metric_name'] if 'metric_name' in t else ('risk / full-cohort risk' if name=='d' else 'accepted per-column error/update time/memory/latency; see companion')
            t['cohort_identity']='benchmark 200 states / 64 draws' if name!='f' else 'benchmark 1000-state accuracy; separate measured costs'
            if 'checkpoint_sha256' not in t:t['checkpoint_sha256']=t.method.map(f.set_index('method').checkpoint_sha256)
            t['v7_panel']=name[0];bframes.append(t);write('benchmark_main_'+name,t)
        check('benchmark_identity_separate',f.loc[f.method.eq('DMF-Gen'),'checkpoint_sha256'].iloc[0].startswith('857a505ff96c') and f"{f.loc[f.method.eq('DMF-Gen'),'mean_unobserved_relative_l2'].iloc[0]:.3f}"=='0.117')
        write('figure5_v7_display_source',pd.concat(bframes+[summary,hf],ignore_index=True))
        aurc=bframes[4].groupby('method',sort=False).first().reset_index()[['method','risk_auc','evidence_package','checkpoint_policy','checkpoint_sha256','cohort_identity','source_path','source_sha256','internal_key']]
        aurc['metric_definition']='trapezoidal area under normalized selective-risk curve over retention 0.2 to 1.0; not divided by 0.8'
        write('benchmark_aurc',aurc)
        # Re-hash protected compact artifacts after the build; no weights are copied.
        changed=[]
        for entry in registry.values():
            if entry['protected']:
                entry['sha256_after']=sha256(REPO/entry['path']);
                if entry['sha256_after']!=entry['sha256']:changed.append(entry['path'])
        check('protected_sources_unchanged',not changed,changed)
    except Exception as e:
        errors.append(type(e).__name__+': '+str(e))
    preflight={'repository_root':str(REPO),'branch':'paper/postprocessing-multifield-superresolution','head':'9b1ef32978edd49cfddd55365ea54b4b7bcac57f','timestamp':'2026-09-10T15:36:42-04:00','git_status_short':'?? Dis_SI_Process/docs/Figure5_V7_Ablation_Codex_Instructions_20260910.md','environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'matplotlib':'3.10.9','conda_environment':'fig'}}
    manifest={'schema_version':'figure5-v7-sources-1','status':'pass' if not errors else 'blocked','preflight':preflight,'collected_at':datetime.now().astimezone().isoformat(),'sources':registry,'derived_tables':tables,'comparison_scope':cfg['comparison_scope'],'checkpoint_record_policy':'frozen A0 weight hashes verified; all checkpoint identities matched to saved checkpoint audit, cache audit, manifest and provenance; no weights copied','report_sha256_expected':REPORT_HASH,'schema_mapping':{'reconstruction_mean':'summary_metrics.mean','mean_CI':'block20_ci95_low/high','physical_macro':'physical_relative_l2/Unobserved_mean','canonical_highband_power':'canonical_shellmean_high_energy_ratio','mode_sum_power':'reconstruction_to_truth_high_energy_ratio','phase_sensitive_residual':'highband_error_relative_l2','population_spectra':'median,q25,q75 describe state dispersion','primary_seed':20260906,'highfrequency_seed':20260910}}
    (out/'source_manifest.json').write_text(json.dumps(as_jsonable(manifest),indent=2)+'\n')
    (out/'source_qa.json').write_text(json.dumps(as_jsonable({'status':'pass' if not errors else 'blocked','errors':errors,'checks':checks}),indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'errors':errors,'tables':len(tables),'sources':len(registry)}))
    if args.strict_formal and errors:raise SystemExit(1)


if __name__=='__main__':main()

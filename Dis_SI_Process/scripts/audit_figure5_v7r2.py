#!/usr/bin/env python
"""Audit source, scientific, visual and delivery gates for Figure 5 V7R2."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text()) if p.exists() else {}

def main():
    p=argparse.ArgumentParser();p.add_argument('--timestamp',required=True);p.add_argument('--strict-formal',action='store_true');args=p.parse_args()
    d=ROOT/'results/derived'/args.timestamp;f=ROOT/'figures/generated'/args.timestamp;docs=ROOT/'docs/generated'/args.timestamp
    d.mkdir(parents=True,exist_ok=True);checks=[]
    def check(name,passed,detail=None):checks.append({'name':name,'status':'pass' if passed else 'fail','detail':detail})
    source=read(d/'source_qa.json');mainplot=read(d/'main_plot_manifest.json');siplot=read(d/'si_plot_manifest.json')
    check('source_integrity',source.get('status')=='pass',source)
    tests=read(d/'test_results.json')
    if tests:check('source_regression_tests',tests.get('status')=='pass',tests)
    check('main_artist_bounds',mainplot.get('status')=='pass',[(r['path'],r.get('off_canvas_text')) for r in mainplot.get('figures',[])])
    check('si_artist_bounds',siplot.get('status')=='pass',siplot.get('status','not rendered'))
    geometry=mainplot.get('inherited_geometry',{})
    check('inherited_scientific_geometry',len(geometry)==5 and all(r['archived']==r['after'] for r in geometry.values()),geometry)
    required=[f'fig5_composed_v7r2_{args.timestamp}',*[f'fig5{x}_v7r2_{args.timestamp}' for x in 'abcdef'],*[f'fig5e_{x}_v7r2_{args.timestamp}' for x in ['Y_CH4','Y_CO','T','U1','p']]]
    check('main_and_field_alternatives',all((f/(stem+'.svg')).exists() and (f/(stem+'.png')).exists() for stem in required),required)
    si_required=[f'si_figure_s1_ablation_error_distributions_{args.timestamp}',
        f'si_figure_s2_scale_resolved_all_fields_{args.timestamp}',f'si_figure_s3_deterministic_objective_control_{args.timestamp}',
        *[f'panel_e_{field}_{args.timestamp}' for field in ['Y_CH4','Y_CO','T','U1','p']]]
    check('required_si_exports',len(siplot.get('figures',[]))==8 and all((f/'si'/(stem+ext)).exists() for stem in si_required for ext in ['.svg','.png']),si_required)
    table_paths=[docs/sub/f'si_v7r2_table_s{i}.{ext}' for i in range(1,6) for sub,ext in [('tables','csv'),('latex/tables','tex')]]
    check('si_tables_machine_and_latex',all(path.exists() and path.stat().st_size>100 for path in table_paths),[str(p.relative_to(REPO)) for p in table_paths])
    companions=[docs/f'fig5{x}_companion.md' for x in 'abcdef']+[docs/'fig5_composed_v7r2_companion.md']+[docs/f'fig5e_{x}_companion.md' for x in ['Y_CH4','Y_CO','T','U1','p']]
    check('companions_and_field_selection',all(p.exists() for p in companions) and (docs/'field_selection_comparison.csv').exists() and (docs/'field_selection_comparison.md').exists())
    docqa=read(docs/'documentation_qa.json')
    check('documentation_qa',docqa.get('status')=='pass',docqa.get('issues',[]))
    svgfiles=sorted(f.rglob('*.svg'));exports=[]
    for path in svgfiles:
        problems=[]
        try:
            root=ET.parse(path).getroot();labels=[''.join(t.itertext()) for t in root.iter('{http://www.w3.org/2000/svg}text')]
            if not labels:problems.append('no editable text')
            if any(re.search(r'\bA[0-5]\b',t) for t in labels):problems.append('internal model codes displayed')
            with Image.open(path.with_suffix('.png')) as im:
                if min(im.info.get('dpi',(0,0)))<599:problems.append('PNG below600dpi')
        except Exception as exc:problems.append(str(exc))
        exports.append({'path':str(path.relative_to(REPO)),'problems':problems})
    check('editable_svg_and_600dpi_png',bool(exports) and all(not x['problems'] for x in exports),exports)
    composed=f/f'fig5_composed_v7r2_{args.timestamp}.svg';labels=[]
    if composed.exists():
        root=ET.parse(composed).getroot();labels=[''.join(t.itertext()) for t in root.iter('{http://www.w3.org/2000/svg}text')]
        check('nature_width',abs(float(root.attrib['width'].removesuffix('pt'))*25.4/72-183)<.01)
    check('panel_letters',all(labels.count(letter)==1 for letter in 'abcdef'),{letter:labels.count(letter) for letter in 'abcdef'})
    check('no_decorative_headings',bool(labels) and not any(any(term in t for term in ['Conditioning and source variants','Fine-scale velocity fidelity','Accuracy and computational footprint','Selective reconstruction']) for t in labels))
    check('panel_c_is_selective',mainplot.get('panel_map',{}).get('c')=='selective reconstruction')
    coords=mainplot.get('plot_coordinates',[]);dist=[x for x in coords if 'jitter_seed' in x]
    primary=[x for x in dist if x['field']=='Unobserved_mean'];high=[x for x in dist if x['field']=='U1'];spectra=[x for x in coords if 'median_power' in x]
    methods=['A0','A2','A3','A5','A4'];six=[*methods,'Senseiver']
    check('panel_d_all_state_distributions', [x['method'] for x in primary]==methods and all(x['n']==1000 for x in primary))
    anchors={'A0':.106321,'A2':.126974,'A3':.144703,'A5':.354951,'A4':.104294}
    check('panel_d_brief_numeric_anchors',len(primary)==5 and all(abs(x['mean']-anchors[x['method']])<.00000051 for x in primary),{x['method']:x['mean'] for x in primary})
    check('d_e_mean_only_annotations',len(dist)==11 and all(re.fullmatch(r'\d+\.\d+',x['annotation']) for x in dist))
    check('panel_e_required_population_spectra',[x['method'] for x in spectra]==['Truth',*six] and all(len(x['median_power'])==198 for x in spectra))
    check('panel_e_highband_distributions',[x['method'] for x in high]==six and all(x['n']==1000 for x in high))
    layout=mainplot.get('layout',{})
    check('reduced_gutters',bool(layout) and layout['top_b_c_axis_gap_fraction']<layout['prior_v7_top_b_c_gap_fraction'] and layout['middle_d_e_axis_gap_fraction']<layout['prior_v7_middle_d_e_gap_fraction'],layout)
    artists=read(d/'main_artist_coordinates.json')
    memory=[x for x in artists if x.get('xlabel')=='MiB'] if isinstance(artists,list) else []
    check('direct_memory_labels',len(memory)==1 and {'Model','Peak'}.issubset({x['text'] for x in memory[0]['annotations']}))
    check('benchmark_error_preserved',any(t=='0.117' for t in labels))
    si_text=' '.join(p.read_text() for p in (f/'si').glob('*.svg'))
    check('deterministic_only_si','A1' not in [x.get('method') for x in coords] and 'Deterministic' in si_text)
    policies={}
    for name in ['reconstruction_states','reconstruction_summary','highband_states','highband_summary','checkpoint_provenance']:
        path=d/(name+'.csv')
        if path.exists():
            table=pd.read_csv(path)
            policies[name]=sorted(table.policy.dropna().astype(str).unique().tolist()) if 'policy' in table else []
    check('last_policy_main_and_si',mainplot.get('new_checkpoint_policy')=='last' and siplot.get('policy')=='last.pt' and len(policies)==5 and all(v==['last'] for v in policies.values()),policies)
    report=(docs/'quantitative_figure_making_report.md').read_text() if (docs/'quantitative_figure_making_report.md').exists() else ''
    check('report_budget_caveat',bool(report) and 'unequal' in report.lower())
    check('package_separation_documented',all(term in report for term in ['0.117','0.106321']))
    broken=[]
    for path in docs.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' not in target and not target.startswith('#') and not (path.parent/target.split('#')[0]).exists():broken.append({'path':str(path.relative_to(REPO)),'target':target})
    check('delivery_links',bool(report) and not broken,broken)
    smoke=read(d/'latex_smoke.json');tex=list((docs/'latex').rglob('*.tex'))
    check('latex_compile_current',bool(tex) and smoke.get('status')=='pass' and all(smoke.get('tex_sha256',{}).get(str(p.relative_to(REPO)))==sha(p) for p in tex),{k:v for k,v in smoke.items() if k!='tex_sha256'})
    visual=read(d/'visual_review.json')
    check('visual_review_current',bool(svgfiles) and visual.get('status')=='pass' and all(visual.get('svg_sha256',{}).get(str(p.relative_to(REPO)))==sha(p) for p in svgfiles),visual.get('notes'))
    protected=read(d/'protected_prior_outputs.json');before=protected.get('sha256_before',{})
    changed=[path for path,digest in before.items() if not (REPO/path).exists() or sha(REPO/path)!=digest]
    check('prior_outputs_unchanged',bool(before) and not changed,changed)
    protected['after_check']={'status':'pass' if not changed else 'fail','changed':changed,'count':len(before)}
    (d/'protected_prior_outputs.json').write_text(json.dumps(protected,indent=2)+'\n')
    failed=[x['name'] for x in checks if x['status']!='pass'];qa={'status':'pass' if not failed else 'incomplete','failed_checks':failed,'checks':checks,'scope':'saved-checkpoint comparison; single ablation draw; unequal training endpoints; last policy only'}
    (d/'qa.json').write_text(json.dumps(qa,indent=2)+'\n')
    artifacts=[p for folder in [f,docs,d] for p in folder.rglob('*') if p.is_file() and p.name!='build_manifest.json']
    manifest={'release':'Figure5 V7R2','timestamp':args.timestamp,'status':'complete' if not failed else 'incomplete','qa':'qa.json','source_manifest':'source_manifest.json','artifacts':[{'path':str(p.relative_to(REPO)),'sha256':sha(p)} for p in sorted(artifacts)],'failed_checks':failed,'inherited_release':'20260904_1200','layout_reference_release':'20260910_1540'}
    implementation=[*sorted((ROOT/'scripts').glob('*v7r2*.py')),*sorted((ROOT/'figures/scripts').glob('*v7r2*.py')),*sorted((ROOT/'tests').glob('*v7r2*.py'))]
    manifest['implementation']=[{'path':str(p.relative_to(REPO)),'sha256':sha(p)} for p in implementation]
    brief=ROOT/'docs/Figure5_V7R2_Codex_Update_Instructions.md'
    manifest['brief']={'path':str(brief.relative_to(REPO)),'sha256':sha(brief),'user_override':'All new required SI figures and tables use last.pt only.'}
    manifest['git']={'head':subprocess.check_output(['rtk','proxy','git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),'branch':subprocess.check_output(['rtk','proxy','git','branch','--show-current'],cwd=REPO,text=True).strip()}
    (d/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':qa['status'],'failed_checks':failed,'svg_count':len(svgfiles),'latex_files':len(tex)}))
    if args.strict_formal and failed:raise SystemExit(1)

if __name__=='__main__':main()

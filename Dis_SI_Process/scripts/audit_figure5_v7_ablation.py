#!/usr/bin/env python
"""Audit source validation, artwork, inserts, and recorded visual review of V7."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import yaml
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def latex_check(path):
    text=path.read_text()
    text=re.sub(r'(?<!\\)%[^\n]*','',text)
    braces=0
    issues=[]
    for c in re.sub(r'\\[{}%]','',text):
        if c=='{':braces+=1
        elif c=='}':braces-=1
        if braces<0:issues.append('closing brace without opening brace');break
    if braces:issues.append('unbalanced braces')
    stack=[]
    for which,env in re.findall(r'\\(begin|end)\{([^}]+)\}',text):
        if which=='begin':stack.append(env)
        elif not stack or stack.pop()!=env:issues.append(f'unbalanced environment {env}')
    if stack:issues.append('unclosed environments: '+','.join(stack))
    # Paths in inserts are repo-relative, delivery-relative, or absolute.
    for name in re.findall(r'\\(?:includegraphics(?:\[[^]]*\])?|input)\{([^}]+)\}',text):
        if '\\' in name:continue
        options=[Path(name), REPO/name, path.parent/name]
        if not any(p.exists() or p.with_suffix('.tex').exists() for p in options):issues.append('unresolved path: '+name)
    return issues


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timestamp',required=True)
    parser.add_argument('--strict-formal',action='store_true')
    args=parser.parse_args()
    d=ROOT/'results/derived'/args.timestamp
    f=ROOT/'figures/generated'/args.timestamp
    docs=ROOT/'docs/generated'/args.timestamp
    checks={};blocks=[]
    def check(name,condition,detail=None):
        checks[name]={'pass':bool(condition),'detail':detail}
        if not condition:blocks.append(name)
    source=json.loads((d/'source_qa.json').read_text()) if (d/'source_qa.json').exists() else {}
    check('source_integrity',source.get('status')=='pass',source)
    if (d/'ablation_primary_summary.csv').exists() and (d/'ablation_highband_summary.csv').exists():
        primary=pd.read_csv(d/'ablation_primary_summary.csv')
        hf=pd.read_csv(d/'ablation_highband_summary.csv')
        check('primary_unique_keys',not primary.duplicated(['policy','method','metric','target']).any())
        check('highband_unique_keys',not hf.duplicated(['policy','method','metric','field']).any())
        expected={'last':{'A0':'0.106321','A2':'0.126974','A3':'0.144703','A5':'0.354951','A4':'0.104294','A1':'0.078705'},
                  'best':{'A0':'0.107893','A2':'0.129288','A3':'0.146284','A5':'0.356403','A4':'0.104967','A1':'0.078561'}}
        anchors={}
        for policy,methods in expected.items():
            for method,value in methods.items():
                rows=primary.loc[primary.policy.eq(policy)&primary.method.eq(method)&primary.metric.eq('physical_relative_l2')&primary.target.eq('Unobserved_mean')]
                anchors[policy+'/'+method]=len(rows)==1 and int(rows.iloc[0]['n'])==1000 and f"{rows.iloc[0]['mean']:.6f}"==value
        check('primary_frozen_anchors',all(anchors.values()),anchors)
        directions={}
        for policy in ['last','best']:
            for field in ['CH4','CO','T','U1','p']:
                rows=hf.loc[hf.policy.eq(policy)&hf.field.eq(field)&hf.metric.eq('highband_error_relative_l2')].set_index('method')
                directions[policy+'/'+field]=float(rows.loc['A4','mean'])>float(rows.loc['A0','mean'])
        check('allfield_highband_prior_directions',all(directions.values()),directions)
    else:
        check('quantitative_derived_inputs',False)
    manifest=json.loads((d/'main_plot_manifest.json').read_text()) if (d/'main_plot_manifest.json').exists() else {}
    check('inherited_scientific_geometry',bool(manifest.get('geometry')) and all(x['archived']==x['recomposed'] for x in manifest.get('geometry',{}).values()))
    check('main_text_inside_canvas',manifest.get('status')=='pass',[(x['path'],x['off_canvas_text']) for x in manifest.get('figures',[]) if x['off_canvas_text']])
    svg_checks={}
    for path in sorted(f.rglob('*.svg')):
        try:
            root=ET.parse(path).getroot()
            labels=[''.join(node.itertext()) for node in root.iter('{http://www.w3.org/2000/svg}text')]
            published=' '.join(labels)
            item={'parseable':True,'editable_text':len(labels),'internal_run_labels_absent':not bool(re.search(r'\bA[0-5]\b',published)), 'matching_png':path.with_suffix('.png').exists()}
        except ET.ParseError as error:item={'parseable':False,'error':str(error)}
        svg_checks[str(path.relative_to(REPO))]=item
    check('svg_structure',len(svg_checks)>=15 and all(x.get('parseable') and x.get('editable_text',0)>0 and x.get('internal_run_labels_absent') and x.get('matching_png') for x in svg_checks.values()),svg_checks)
    dpi={}
    for path in sorted(f.rglob('*.svg')):
        if path.with_suffix('.png').exists():
            with Image.open(path.with_suffix('.png')) as im:
                dpi[str(path.relative_to(REPO))]=list(im.info.get('dpi',[]))
    check('png_600dpi',len(dpi)==len(svg_checks) and all(len(v)==2 and all(abs(x-600)<.1 for x in v) for v in dpi.values()),dpi)
    required=[f'fig5{x}_{stem}_{args.timestamp}.svg' for x,stem in {'a':'crps','b':'spread_error','c':'architecture_prior','d':'selective_risk','e':'highband_velocity','f':'accuracy_footprint'}.items()]
    required += [f'fig5_composed_v7_{args.timestamp}.svg',f'fig5e_power_{args.timestamp}.svg',f'fig5e_residual_{args.timestamp}.svg']
    check('all_main_artwork',all((f/p).exists() for p in required),required)
    si_stems=['si_ablation_reconstruction','si_ablation_highband_all_fields','si_ablation_u1_spectral_audit','si_ablation_coupling_range_audit','si_ablation_checkpoint_sensitivity','si_deterministic_objective_control']
    check('six_si_composites',all((f/'si'/f'{stem}_{args.timestamp}.svg').exists() for stem in si_stems))
    si_manifest=json.loads((f/'si/si_plot_manifest.json').read_text()) if (f/'si/si_plot_manifest.json').exists() else {}
    check('si_source_and_display_qa',si_manifest.get('qa',{}).get('status')=='pass' and si_manifest.get('qa',{}).get('visual_summary',{}).get('status')=='pass',si_manifest.get('qa',{}).get('visual_summary',{}))
    si_panels=[p['id'] for p in si_manifest.get('plots',[]) if p.get('coordinates')]
    check('individual_si_companions',len(si_panels)>=18 and all((docs/'si_companions'/f'{panel}_companion.md').exists() for panel in si_panels),si_panels)
    check('no_new_pdf_delivery',not any(f.rglob('*.pdf')) and not any(docs.rglob('*.pdf')))
    if (f/required[6]).exists():
        root=ET.parse(f/required[6]).getroot()
        width=float(root.attrib['width'].removesuffix('pt'))*25.4/72
        check('composed_width_183mm',abs(width-183)<.001,width)
        labels=' '.join(''.join(t.itertext()) for t in root.iter('{http://www.w3.org/2000/svg}text'))
        check('main_semantics',all(v in labels for v in ['Model','Peak','0.117','0.1063','+19.4%','+36.1%','+233.8%','-1.9%','0.43','3.12','0.87','1.83']),labels)
        check('deterministic_separate_from_main','Deterministic' not in labels)
    tex_files=list(docs.rglob('*.tex'))
    # The project may keep inserts in a timestamped latex/ directory.
    for rootpath in [ROOT/'latex'/args.timestamp,ROOT/'latex/generated'/args.timestamp]:
        if rootpath.exists():tex_files.extend(rootpath.rglob('*.tex'))
    tex_issues={str(p.relative_to(REPO)):latex_check(p) for p in tex_files}
    check('latex_structure_and_paths',len(tex_files)>=8 and all(not v for v in tex_issues.values()),tex_issues)
    names={p.name for p in tex_files}
    check('required_latex_inserts',all(name in names for name in ['figure5_v7_caption.tex','figure5_v7_ablation_paragraph.tex','figure5_v7_panel_reference_updates.tex','si_ablation_methods.tex','si_ablation_results.tex','si_ablation_figures.tex','si_ablation_tables.tex']))
    smoke=json.loads((d/'latex_smoke.json').read_text()) if (d/'latex_smoke.json').exists() else {}
    check('latex_compile_smoke_current',smoke.get('status')=='pass' and all(smoke.get('tex_sha256',{}).get(str(p.relative_to(REPO)))==sha(p) for p in tex_files),{k:v for k,v in smoke.items() if k!='tex_sha256'})
    check('quantitative_report',(docs/'quantitative_figure_making_report.md').exists())
    broken_links=[]
    for markdown in docs.rglob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)',markdown.read_text()):
            if '://' not in target and not target.startswith('#'):
                if not (markdown.parent/target.split('#')[0]).exists():
                    broken_links.append({'file':str(markdown.relative_to(REPO)),'target':target})
    check('markdown_delivery_links',not broken_links,broken_links)
    visual=json.loads((d/'visual_review.json').read_text()) if (d/'visual_review.json').exists() else {}
    reviewed=visual.get('svg_sha256',{})
    check('visual_review_current',visual.get('status')=='pass' and bool(svg_checks) and all(reviewed.get(str(p.relative_to(REPO)))==sha(p) for p in f.rglob('*.svg')),visual.get('notes','Review not yet recorded'))
    cfg=yaml.safe_load((ROOT/'configs/figure5_v7_ablation.yaml').read_text())
    result={'schema_version':'figure5-v7-qa-1','status':'pass' if not blocks else 'blocked','blocks':blocks,'checks':checks,'comparison_scope':cfg['comparison_scope']}
    (d/'qa.json').write_text(json.dumps(result,indent=2)+'\n')
    build={'schema_version':'figure5-v7-build-1','timestamp':args.timestamp,'status':'complete' if not blocks else 'blocked','source_manifest':str((d/'source_manifest.json').relative_to(REPO)),'qa':str((d/'qa.json').relative_to(REPO)), 'figures':[{ 'path':str(p.relative_to(REPO)),'sha256':sha(p)} for p in sorted(f.rglob('*')) if p.is_file()], 'comparison_scope':cfg['comparison_scope']}
    implementation=[ROOT/'configs/figure5_v7_ablation.yaml',ROOT/'utils/figure5_v7_ablation_data.py',ROOT/'tests/test_figure5_v7_ablation.py']
    implementation+=list((ROOT/'scripts').glob('*figure5_v7*.py'))+[ROOT/'figures/scripts/build_figure5_v7_ablation.py',ROOT/'figures/scripts/build_ablation_si_v7.py']
    build['implementation']=[{'path':str(p.relative_to(REPO)),'sha256':sha(p)} for p in sorted(set(implementation))]
    build['derived_tables']=[{'path':str(p.relative_to(REPO)),'sha256':sha(p)} for p in sorted(d.glob('*.csv'))]
    build['latex']=[{'path':str(p.relative_to(REPO)),'sha256':sha(p)} for p in sorted(tex_files)]
    (d/'build_manifest.json').write_text(json.dumps(build,indent=2)+'\n')
    completion=f'''# Figure 5 V7 completion report

Release: {args.timestamp}. Final package status: **{result['status']}**.

Produced: one 183 mm composed Figure 5 with dedicated right-hand panels c/e;
six main standalones plus separate power/residual exports for e; six SI
composites and {len(si_panels)} standalone SI diagnostics. There are
{len(svg_checks)} editable SVG artworks with matching 600-dpi PNGs,
{len(tex_files)} LaTeX inserts/table files, individual quantitative companions,
and the quantitative figure-making report.

Main artwork: `Dis_SI_Process/figures/generated/{args.timestamp}/fig5_composed_v7_{args.timestamp}.svg`.
SI entry point: `Dis_SI_Process/docs/generated/{args.timestamp}/latex/si_ablation_package.tex`.
Source ledger: `Dis_SI_Process/results/derived/{args.timestamp}/source_manifest.json`.
Report: `Dis_SI_Process/docs/generated/{args.timestamp}/quantitative_figure_making_report.md`.

Source validation, inherited-coordinate comparisons, editable SVG structure,
600-dpi raster metadata, current visual review, and LaTeX compile status are
recorded in `qa.json`. Tests are run with the command below. The LaTeX smoke
test uses a 183 mm text width and 300 mm text height to accommodate the main
caption at full artwork width; temporary PDFs/logs are removed.

Required-but-unavailable outputs: {', '.join(blocks) if blocks else 'none'}.
The optional cached-field gallery was not produced; it is not required for
the saved-metric evidence sequence. No training, model inference, GPU timing,
solver change, clipping, smoothing, or ensemble reevaluation was performed.
Heavy data and checkpoints remain in place. Prior release hashes are preserved.

The archived benchmark error remains 0.117. The new full reference is separate;
IID's lower bulk error, the full model's high-band attenuation, and the
deterministic aggregate advantage remain explicit. The available configurations
have unequal training endpoints/capacity, a shared validation/test holdout,
and one stochastic draw per evaluation state. Formal QA does not imply a
matched-budget causal comparison.

Reproduce from the repository root:

```bash
rtk proxy conda run -n fig python Dis_SI_Process/scripts/build_figure5_v7_bundle.py --timestamp {args.timestamp} --strict-formal
rtk proxy conda run -n fig python -m unittest Dis_SI_Process.tests.test_figure5_v7_ablation
```

Changed sources or artwork require renewed visual review; the saved review is
bound to exact SVG hashes.
'''
    (docs/'completion_report.md').write_text(completion)
    print(json.dumps({'status':result['status'],'blocks':blocks,'svg_count':len(svg_checks),'tex_count':len(tex_files)}))
    if args.strict_formal and blocks:raise SystemExit(1)


if __name__=='__main__':main()

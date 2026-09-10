#!/usr/bin/env python
"""Run the additive V7R2 saved-field post-processing and QA stages."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--timestamp',required=True);p.add_argument('--strict-formal',action='store_true')
    p.add_argument('--resume-from',choices=['sources','main','si','documents','latex','audit'],default='sources',help='Resume downstream work after a documented correction; earlier source/visual gates remain mandatory.')
    args=p.parse_args()
    derived=ROOT/'results/derived'/args.timestamp;derived.mkdir(parents=True,exist_ok=True)
    protected=derived/'protected_prior_outputs.json'
    if not protected.exists():
        files=[p for release in ['20260904_1200','20260910_1540'] for folder in ['figures/generated','results/derived','docs/generated'] for p in (ROOT/folder/release).rglob('*') if p.is_file()]
        protected.write_text(json.dumps({'captured_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256_before':{str(p.relative_to(ROOT.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}},indent=2)+'\n')
    record=derived/'execution_record.json'
    executions=json.loads(record.read_text()) if record.exists() else []
    stages=['scripts/collect_figure5_v7r2_sources.py','figures/scripts/build_figure5_v7r2.py',
                'figures/scripts/build_figure5_v7r2_si.py','scripts/write_figure5_v7r2_documents.py',
                'scripts/smoke_figure5_v7r2_latex.py','scripts/audit_figure5_v7r2.py']
    start=['sources','main','si','documents','latex','audit'].index(args.resume_from)
    for rel in stages[start:]:
        command=[sys.executable,str(ROOT/rel),'--timestamp',args.timestamp]
        if args.strict_formal:command.append('--strict-formal')
        print(f'Running {rel}',flush=True)
        run=subprocess.run(command,cwd=ROOT.parent)
        executions.append({'command':command,'returncode':run.returncode})
        (derived/'execution_record.json').write_text(json.dumps(executions,indent=2)+'\n')
        if run.returncode:
            subprocess.run([sys.executable,str(ROOT/'scripts/audit_figure5_v7r2.py'),'--timestamp',args.timestamp],cwd=ROOT.parent)
            raise SystemExit(run.returncode)
    command=[sys.executable,'-m','unittest','Dis_SI_Process.tests.test_figure5_v7r2_sources']
    run=subprocess.run(command,cwd=ROOT.parent,capture_output=True,text=True)
    print(run.stdout+run.stderr,flush=True)
    executions.append({'command':command,'returncode':run.returncode})
    (derived/'test_results.json').write_text(json.dumps({'status':'pass' if run.returncode==0 else 'fail','command':command,'returncode':run.returncode,'output':run.stdout+run.stderr},indent=2)+'\n')
    (derived/'execution_record.json').write_text(json.dumps(executions,indent=2)+'\n')
    final=[sys.executable,str(ROOT/'scripts/audit_figure5_v7r2.py'),'--timestamp',args.timestamp]
    if args.strict_formal:final.append('--strict-formal')
    audit=subprocess.run(final,cwd=ROOT.parent)
    if run.returncode or audit.returncode:raise SystemExit(run.returncode or audit.returncode)

if __name__=='__main__':main()

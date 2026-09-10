#!/usr/bin/env python
"""Run the saved-data-only Figure 5 V7 publication bundle pipeline."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timestamp',required=True)
    parser.add_argument('--strict-formal',action='store_true')
    args=parser.parse_args()
    stages=[ROOT/'scripts/collect_figure5_v7_ablation_sources.py',
            ROOT/'figures/scripts/build_figure5_v7_ablation.py',
            ROOT/'figures/scripts/build_ablation_si_v7.py',
            ROOT/'scripts/write_figure5_v7_documents.py',
            ROOT/'scripts/smoke_figure5_v7_latex.py',
            ROOT/'scripts/audit_figure5_v7_ablation.py']
    for stage in stages:
        if not stage.is_file(): raise FileNotFoundError(stage)
    for stage in stages:
        command=[sys.executable,str(stage),'--timestamp',args.timestamp]
        if args.strict_formal:command.append('--strict-formal')
        print(f'Running {stage.name}',flush=True)
        subprocess.run(command,cwd=ROOT.parent,check=True)
    subprocess.run([sys.executable,'-m','unittest','Dis_SI_Process.tests.test_figure5_v7_ablation'],cwd=ROOT.parent,check=True)


if __name__=='__main__':main()

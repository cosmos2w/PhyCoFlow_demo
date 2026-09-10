#!/usr/bin/env python
"""Compile delivered V7 inserts without retaining a PDF in the release."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--timestamp',required=True)
    p.add_argument('--strict-formal',action='store_true')
    args=p.parse_args()
    folder=ROOT/'docs/generated'/args.timestamp/'latex'
    entry=folder/'si_ablation_package.tex'
    mainfig=folder/'figure5_v7_figure.tex'
    preamble=r'''\documentclass[10pt]{article}
\usepackage[paperwidth=213mm,paperheight=330mm,left=15mm,right=15mm,top=15mm,bottom=15mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{graphicx,booktabs,longtable,array,amsmath,amssymb}
\begin{document}
'''
    # A 300 mm text height accommodates the long main caption at its full
    # artwork width.  The width gate remains the requested 183 mm.
    body='\\input{'+str(mainfig)+'}\n\\clearpage\n\\input{'+str(entry)+'}\n\\end{document}\n'
    with tempfile.TemporaryDirectory(prefix='figure5_v7_tex_') as tmp:
        tex=Path(tmp)/'smoke.tex';tex.write_text(preamble+body)
        outputs=[]
        for _ in range(2):
            r=subprocess.run([shutil.which('pdflatex') or 'pdflatex','-interaction=nonstopmode','-halt-on-error','-output-directory',tmp,str(tex)],cwd=ROOT.parent,capture_output=True,text=True)
            outputs.append(r.stdout)
            if r.returncode:break
        log=(Path(tmp)/'smoke.log').read_text(errors='replace') if (Path(tmp)/'smoke.log').exists() else outputs[-1]
        overflow=re.findall(r'Overfull \\[hv]box[^\n]*',log)
        errors=[line for line in log.splitlines() if line.startswith('!')]
        warnings=re.findall(r'LaTeX Warning: [^\n]*',log)
        files={str(f.relative_to(ROOT.parent)):hashlib.sha256(f.read_bytes()).hexdigest() for f in folder.rglob('*.tex')}
        large_floats=[w for w in warnings if 'Float too large' in w or 'multiply' in w or 'undefined' in w]
        result={'status':'pass' if r.returncode==0 and not overflow and not large_floats else 'blocked','returncode':r.returncode,'overfull_boxes':overflow,'errors':errors,'warnings':warnings,'text_width_mm':183,'text_height_mm':300,'runs':len(outputs),'tex_sha256':files,'temporary_products_retained':False}
    path=ROOT/'results/derived'/args.timestamp/'latex_smoke.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='tex_sha256'}))
    if args.strict_formal and result['status']!='pass':raise SystemExit(1)

if __name__=='__main__':main()

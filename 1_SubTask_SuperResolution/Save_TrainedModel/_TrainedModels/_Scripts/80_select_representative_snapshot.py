#!/usr/bin/env python
"""Select auditable median-like qualitative snapshots from quantitative CSVs."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np

from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, run_id
from common.io_utils import artifact_name, matching_or_latest, read_csv, write_csv


def robust_z(values):
    x=np.asarray(values,float); med=np.nanmedian(x); scale=np.nanmedian(np.abs(x-med))*1.4826
    return np.abs(x-med)/max(float(scale),1e-12)


def select(rows, recipes, preferred_models):
    by=defaultdict(dict)
    for row in rows:
        if row.get("status")!="ok" or row.get("recipe") not in recipes: continue
        by[(row["model"],int(row["snapshot_index"]))][row["recipe"]]=row
    for model in preferred_models:
        candidates=[]
        for (m,snapshot),payload in by.items():
            if m==model and all(r in payload for r in recipes):
                vals=[float(payload[r]["physical_rel_l2"]) for r in recipes]
                candidates.append((snapshot,payload,vals))
        if not candidates: continue
        matrix=np.asarray([c[2] for c in candidates]); scores=np.zeros(len(candidates))
        for j in range(matrix.shape[1]): scores+=robust_z(matrix[:,j])
        if len(recipes)>=2: scores+=robust_z(matrix[:,0]-matrix[:,1])
        best=int(np.nanargmin(scores)); snapshot,payload,vals=candidates[best]; first=payload[recipes[0]]
        return {"model":model,"snapshot_index":snapshot,"case_id":first.get("case_id"),"time_index":first.get("time_index"),
                "selection_score":float(scores[best]),"selection_method":"minimum robust distance to median recipe errors and paired difference",
                **{f"error_{recipe}":value for recipe,value in zip(recipes,vals)}}
    return {"model":"","snapshot_index":"","case_id":"","time_index":"","selection_score":np.nan,"selection_method":"missing_quantitative_pairs"}


def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p,models=False)
    p.add_argument("--questionA",type=Path); p.add_argument("--questionB",type=Path); args=p.parse_args()
    cfg=load_config(args.config); rid=run_id(args.run_id); ensure_output_dirs()
    qa=read_csv(args.questionA or matching_or_latest(RESULTS_DIR/"QuestionA_DataBenefit","QuestionA_per_snapshot",args.run_id,"csv"))
    qb=read_csv(args.questionB or matching_or_latest(RESULTS_DIR/"QuestionB_ZeroH","QuestionB_per_snapshot",args.run_id,"csv"))
    model_order=[cfg["questionA"]["qualitative_model"],*[m["key"] for m in cfg["models"]]]
    a=select(qa,cfg["questionA"]["recipes"],model_order); a["block"]="questionA"
    b=select(qb,cfg["questionB"]["recipes"],cfg["questionB"]["qualitative_models"]); b["block"]="questionB"
    path=RESULTS_DIR/"CanonicalTestIndex"/artifact_name("RepresentativeSnapshots",rid,"csv"); write_csv(path,[a,b]); print(f"[OK] {path}")

if __name__=="__main__": main()

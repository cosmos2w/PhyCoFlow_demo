#!/usr/bin/env python
"""Export native L/M/H fields and actual recipe spatial-DOF budgets."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from common.config import ARCHIVE_DIR, RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, recipe_items, run_id
from common.dataset_loader import build_run_dataset, physical_native_field
from common.io_utils import artifact_name, latest, read_csv, write_csv
from common.recipe_registry import resolve_recipe_dir, validate_recipe
from helpers import project_physical_sensor_coords_to_indices


def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p,models=False); p.add_argument("--snapshot",type=int,default=0); p.add_argument("--sensor-plan",type=Path); p.add_argument("--sensor-count",type=int); args=p.parse_args()
    cfg=load_config(args.config); rid=run_id(args.run_id); ensure_output_dirs(); model=next(method_items(cfg))
    # Use the first real run only to recover the canonical dataset; field identity comes from its manifest.
    chosen=None
    for key,spec in recipe_items(cfg):
        rd=resolve_recipe_dir(ARCHIVE_DIR/model["directory"],key,spec)
        if (rd/"run_config.yaml").exists(): chosen=(key,spec,rd); break
    key,spec,run_dir=chosen; ds,_,manifest,_=build_run_dataset(run_dir,model["key"],key,split="test",eval_resolution="H")
    plan_path=args.sensor_plan or latest(RESULTS_DIR/"SensorPlans","SensorPlan","csv"); formal_count=args.sensor_count or cfg["sensor_plan"]["default_count"]
    planned_all=[r for r in read_csv(plan_path) if int(r["snapshot_index"])==args.snapshot]
    if not planned_all: raise KeyError(f"Snapshot {args.snapshot} is absent from sensor plan {plan_path}")
    first=planned_all[0]; identity={k:first[k] for k in ("snapshot_index","dataset_index","case_id","time_index","physical_time","eval_resolution","selection_strategy","selection_seed") if k in first}; identity={k:(int(v) if k in {"snapshot_index","dataset_index","case_id","time_index","selection_seed"} else float(v) if k=="physical_time" else v) for k,v in identity.items()}
    fields=[]
    for tag in "LMH":
        coords,values=physical_native_field(manifest,identity["case_id"],identity["time_index"],tag)
        for i,(coord,value) in enumerate(zip(coords,values)):
            fields.append({**identity,"resolution":tag,"point_index":i,"x_phys":coord[0],"y_phys":coord[1],
                           "field_value":value,"num_x":manifest["resolutions"][tag]["Num_x"],"num_y":manifest["resolutions"][tag]["Num_y"],
                           "field_name":manifest["selected_field_name"]})
    field_path=RESULTS_DIR/"ResolutionProtocol"/artifact_name("ResolutionProtocol_fields",rid,"csv"); write_csv(field_path,fields)
    planned=[r for r in planned_all if int(r["sensor_order"])<formal_count]
    sensor_phys=torch.tensor([[float(r["x_phys"]),float(r["y_phys"])] for r in planned],dtype=torch.float32); sensor_rows=[]
    for tag in "LMH":
        coords,_=physical_native_field(manifest,identity["case_id"],identity["time_index"],tag); idx=project_physical_sensor_coords_to_indices(torch.tensor(coords[:,:2]),sensor_phys).cpu().numpy(); unique=np.unique(idx)
        for point_index in unique:
            sensor_rows.append({**identity,"resolution":tag,"formal_sensor_count":formal_count,"effective_unique_sensor_count":len(unique),"point_index":int(point_index),"x_phys":coords[point_index,0],"y_phys":coords[point_index,1]})
    sensor_path=RESULTS_DIR/"ResolutionProtocol"/artifact_name("ResolutionProtocol_sensors",rid,"csv"); write_csv(sensor_path,sensor_rows)
    budgets=[]
    npts={tag:int(manifest["resolutions"][tag]["num_points"]) for tag in "LMH"}; h_only=None
    for recipe,rspec in recipe_items(cfg):
        budget_model = next((candidate for candidate in method_items(cfg)
                             if (resolve_recipe_dir(ARCHIVE_DIR/candidate["directory"],recipe,rspec)/"run_config.yaml").exists()), None)
        rd=resolve_recipe_dir(ARCHIVE_DIR/budget_model["directory"],recipe,rspec) if budget_model else ARCHIVE_DIR/"__missing__"
        try:
            ds2,_,man2,_=build_run_dataset(rd,budget_model["key"],recipe,split="test",eval_resolution="H")
            from common.dataset_loader import read_run_config
            _,flat=read_run_config(rd); v=validate_recipe(recipe,rspec,flat,man2); status=v["status"]
            dof=sum(v[f"train_cases_{tag}"]*npts[tag] for tag in "LMH")
        except Exception: v={f"train_cases_{tag}":0 for tag in "LMH"}; dof=float("nan"); status="missing"
        if recipe=="1_H_only": h_only=dof
        budgets.append({"recipe":recipe,"recipe_label":rspec["label"],**v,"spatial_dof_budget":dof,"status":status})
    denom=h_only if h_only and np.isfinite(h_only) else np.nan
    for row in budgets: row["spatial_dof_budget_normalized_H_only"]=row["spatial_dof_budget"]/denom
    budget_path=RESULTS_DIR/"ResolutionProtocol"/artifact_name("ResolutionProtocol_budgets",rid,"csv"); write_csv(budget_path,budgets)
    print(f"[OK] {field_path}\n[OK] {sensor_path}\n[OK] {budget_path}")
if __name__=="__main__": main()

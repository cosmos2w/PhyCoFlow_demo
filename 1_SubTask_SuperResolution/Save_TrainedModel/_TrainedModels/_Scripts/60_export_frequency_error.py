#!/usr/bin/env python
"""Export per-snapshot and median/IQR radial frequency-error curves."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
import yaml
from common.cache import load_cache
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, run_id
from common.io_utils import artifact_name, read_csv, write_csv
from common.panels_de_data import export_spectral_bands, normalize_recipe_keys
from common.spectral import compare_channel_spectra
from common.workflow import cache_entries, cache_manifest_path

def spacing(path):
    with h5py.File(path,"r") as h5: c=h5["coordinates"][:,0,0,:2]
    x=np.unique(c[:,0]); y=np.unique(c[:,1]); return float(np.median(np.diff(x))),float(np.median(np.diff(y)))

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p); p.add_argument("--cache-manifest",type=Path); p.add_argument("--recipes",nargs="+"); p.add_argument("--snapshots",nargs="+",type=int); p.add_argument("--sensor-count",type=int); p.add_argument("--layout",default=str(Path(__file__).with_name("publication_layout_unified_v2.yaml"))); args=p.parse_args(); cfg=load_config(args.config); rid=run_id(args.run_id); ensure_output_dirs(); recipes=args.recipes or cfg["frequency_error"]["recipes"]; models=cfg["frequency_error"]["models"] if args.models==["all"] else args.models; count=args.sensor_count or cfg["sensor_plan"]["default_count"]; rows=[]; native_nyquist={}
    for e in cache_entries(args.cache_manifest,models=models,recipes=recipes,sensor_count=count,snapshots=args.snapshots):
        try:
            arrays,meta=load_cache(Path(e["cache_path"])); comp=compare_channel_spectra(arrays["truth_phys"].reshape(-1),arrays["recon_phys"].reshape(-1),arrays["coords_phys"][:,:2],num_x=int(meta["num_x"]),num_y=int(meta["num_y"]),coordinate_mode="physical",relative_epsilon=cfg["frequency_error"]["epsilon"]); k=np.asarray(comp["truth"]["wavenumber"]); et=np.asarray(comp["truth"]["spectral_energy"]); ep=np.asarray(comp["reconstruction"]["spectral_energy"]); eps=max(1e-30,cfg["frequency_error"]["epsilon"]*float(np.max(et))); d=np.abs(10*np.log10((ep+eps)/(et+eps))); kn_h=min(np.pi/comp["grid"]["physical_dx"],np.pi/comp["grid"]["physical_dy"]); manifest_key=str(meta["manifest_path"])
            if manifest_key not in native_nyquist:
                manifest=json.loads(Path(manifest_key).read_text()); native_nyquist[manifest_key]={}
                for tag in "LM":
                    dx,dy=spacing(manifest["paths"][tag]); native_nyquist[manifest_key][tag]=min(np.pi/dx,np.pi/dy)
            bounds={tag:native_nyquist[manifest_key][tag]/kn_h for tag in "LM"}
            for i,(kv,dv) in enumerate(zip(k/kn_h,d)): rows.append({**{q:meta.get(q) for q in ("model","model_label","recipe","recipe_label","case_id","time_index","snapshot_index","sensor_count")},"shell_index":i,"k_normalized_H_nyquist":kv,"frequency_error_db_abs":dv,"L_nyquist_normalized":bounds["L"],"M_nyquist_normalized":bounds["M"],"status":"ok"})
        except Exception as exc: rows.append({"model":e.get("model"),"recipe":e.get("recipe"),"snapshot_index":e.get("snapshot_index"),"status":f"error:{exc}"})
    per=RESULTS_DIR/"FrequencyError"/artifact_name("FrequencyError_per_snapshot",rid,"csv"); write_csv(per,rows)
    groups=defaultdict(list)
    for r in rows:
        if r.get("status")=="ok": groups[(r["model"],r["model_label"],r["recipe"],r["recipe_label"],int(r["shell_index"]))].append(r)
    summary=[]
    for key,vals in groups.items():
        d=np.array([float(v["frequency_error_db_abs"]) for v in vals]); summary.append({"model":key[0],"model_label":key[1],"recipe":key[2],"recipe_label":key[3],"shell_index":key[4],"k_normalized_H_nyquist":np.median([float(v["k_normalized_H_nyquist"]) for v in vals]),"median":np.median(d),"q25":np.quantile(d,.25),"q75":np.quantile(d,.75),"valid_n":len(d),"L_nyquist_normalized":vals[0]["L_nyquist_normalized"],"M_nyquist_normalized":vals[0]["M_nyquist_normalized"]})
    sp=RESULTS_DIR/"FrequencyError"/artifact_name("FrequencyError_summary",rid,"csv"); write_csv(sp,summary); print(f"[OK] {per}\n[OK] {sp}")
    with Path(args.layout).open("r",encoding="utf-8") as handle: panel_cfg=(yaml.safe_load(handle) or {}).get("panel_e",{})
    manifest_path=cache_manifest_path(args.cache_manifest)
    band_outputs=export_spectral_bands(cfg,read_csv(manifest_path),RESULTS_DIR/"UnifiedPublicationV2",rid,sensor_count=count,bootstrap_samples=int(cfg["frequency_error"].get("bootstrap_samples",2000)),relative_epsilon=float(cfg["frequency_error"].get("epsilon",1e-12)),robust_ylim_percentile=float(panel_cfg.get("robust_ylim_percentile",99.0)),main_recipes=normalize_recipe_keys(panel_cfg.get("recipes",recipes)))[:3]
    for output in band_outputs: print(f"[OK] {output}")
if __name__=="__main__": main()

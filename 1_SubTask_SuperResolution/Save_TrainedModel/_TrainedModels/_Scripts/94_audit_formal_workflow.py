#!/usr/bin/env python
"""Requirement-by-requirement audit for a completed formal publication run."""
from __future__ import annotations
import argparse
import csv
import hashlib
import re
import subprocess
from collections import Counter,defaultdict
from pathlib import Path

from common.cache import cache_manifest
from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config, method_items, recipe_items, run_id
from common.io_utils import matching_or_latest, read_csv, write_json

def main():
    p=argparse.ArgumentParser(description=__doc__); add_common_args(p,models=False)
    p.add_argument("--cache-manifest",type=Path); p.add_argument("--canonical-index",type=Path); p.add_argument("--sensor-plan",type=Path)
    args=p.parse_args(); cfg=load_config(args.config); rid=run_id(args.run_id); failures=[]; evidence={}
    def require(condition,message):
        if not condition: failures.append(message)
    require(RESULTS_DIR.is_symlink(),"_Process_Results is not a symlink"); evidence["results_target"]=str(RESULTS_DIR.resolve())
    index_path=args.canonical_index or matching_or_latest(RESULTS_DIR/"CanonicalTestIndex","CanonicalTestIndex",rid,"csv"); index=read_csv(index_path); require(len(index)==300,"canonical index does not contain 300 rows"); require(len({r["case_id"] for r in index})==300,"canonical screen is not 300 distinct cases"); canonical={int(r["snapshot_index"]):(r["case_id"],r["time_index"]) for r in index}
    plan_path=args.sensor_plan or matching_or_latest(RESULTS_DIR/"SensorPlans","SensorPlan",rid,"csv"); plan=read_csv(plan_path); by_snapshot=defaultdict(list)
    for row in plan: by_snapshot[int(row["snapshot_index"])].append(row)
    require(len(by_snapshot)==300,"sensor plan does not cover 300 samples")
    for snapshot,rows in by_snapshot.items():
        rows.sort(key=lambda r:int(r["sensor_order"])); ids=[int(r["point_index"]) for r in rows]; counts=[int(v) for v in cfg["sensor_plan"]["counts"]]; maximum=max(counts); require(len(ids)==maximum and len(set(ids))==maximum,f"snapshot {snapshot} does not have {maximum} unique H sensors"); require(all(set(ids[:a])<set(ids[:b]) for a,b in zip(counts[:-1],counts[1:])),f"snapshot {snapshot} sensor sets are not strictly nested")
    manifest_path=args.cache_manifest or cache_manifest(rid); manifest=read_csv(manifest_path); keys=[(r.get("model"),r.get("recipe"),int(r.get("snapshot_index") or -1),int(r.get("sensor_count") or -1)) for r in manifest]; require(len(keys)==len(set(keys)),"cache manifest contains duplicate keys")
    grouped=Counter((r.get("model"),r.get("recipe"),int(r.get("sensor_count") or -1),r.get("status")) for r in manifest); missing_allowed={("DMFGen","1_H_only"),("FFM_Perceiver","1_H_only")}
    for model in method_items(cfg):
        for recipe,_ in recipe_items(cfg):
            ok=grouped[(model["key"],recipe,256,"ok")]; missing=sum(v for (m,r,c,s),v in grouped.items() if (m,r,c)==(model["key"],recipe,256) and str(s).startswith("missing"));
            if (model["key"],recipe) in missing_allowed: require(missing==300,f"{model['key']}/{recipe} missing placeholders != 300")
            else: require(ok==300,f"{model['key']}/{recipe} successful n=256 caches != 300 (got {ok})")
    for model in method_items(cfg):
        for recipe in cfg["sensor_sweep"]["recipes"]:
            for count in cfg["sensor_sweep"]["counts"]: require(grouped[(model["key"],recipe,int(count),"ok")]==300,f"sweep {model['key']}/{recipe}/n{count} != 300")
    error_rows=[r for r in manifest if r.get("status") not in {"ok","missing_config","missing_checkpoint","missing_directory"}]; require(not error_rows,f"manifest contains {len(error_rows)} error rows")
    ok_rows=[r for r in manifest if r.get("status")=="ok"]; require(all(r.get("storage_mode")=="compact_shared_v1" for r in ok_rows),"not all successful caches use compact_shared_v1"); require(all(Path(r["cache_path"]).is_file() for r in ok_rows),"one or more cache files are missing")
    require(all((r.get("case_id"),r.get("time_index"))==canonical.get(int(r["snapshot_index"])) for r in ok_rows),"one or more caches are not aligned to the canonical case/time identity")
    plan_hash=hashlib.sha256(plan_path.read_bytes()).hexdigest(); plan_hashes={plan_hash:plan_path}
    for candidate in (RESULTS_DIR/"SensorPlans").glob("SensorPlan_*.csv"):
        plan_hashes.setdefault(hashlib.sha256(candidate.read_bytes()).hexdigest(),candidate)
    require(all(r.get("sensor_plan_hash") in plan_hashes for r in ok_rows),"one or more caches record an unknown sensor-plan hash")
    seeds=defaultdict(set)
    for r in ok_rows: seeds[(r["case_id"],r["time_index"])].add(r["generation_seed"])
    require(all(len(v)==1 for v in seeds.values()),"generation seeds are not paired across methods/recipes/counts")
    require(len(list((RESULTS_DIR/"ReconstructionCache"/"Shared"/"Truth").glob("*.npz")))==300,"shared truth store does not contain exactly 300 samples")
    referenced={str(Path(r["cache_path"]).resolve()) for r in ok_rows}; cache_run_ids={r.get("run_id") for r in manifest if r.get("run_id")}; roots=[RESULTS_DIR/"ReconstructionCache"/value for value in cache_run_ids]; orphans=[p for root in roots if root.exists() for p in root.rglob("*.npz") if str(p.resolve()) not in referenced]; require(not orphans,f"{len(orphans)} orphan cache files remain")
    for folder,prefix in (("ResolutionProtocol","ResolutionProtocol_fields"),("QuestionA_DataBenefit","QuestionA_summary"),("CoarseDetail","CoarseDetail_summary"),("QuestionB_ZeroH","QuestionB_summary"),("FrequencyError","FrequencyError_summary"),("GradientError","GradientError_summary"),("SensorSweep","SensorSweep_summary")):
        try: matching_or_latest(RESULTS_DIR/folder,prefix,rid,"csv")
        except FileNotFoundError: require(False,f"missing source CSV: {folder}/{prefix}")
    base=FIGURES_DIR/"Assembled"/f"{cfg['assembly']['publication']['output_name']}_{rid}"; pdf=base.with_suffix(".pdf"); svg=base.with_suffix(".svg"); png=base.with_suffix(".png")
    require(pdf.is_file() and svg.is_file() and png.is_file(),"assembled SVG/PDF/PNG bundle is incomplete")
    if pdf.exists():
        info=subprocess.run(["pdfinfo",str(pdf)],capture_output=True,text=True,check=True).stdout; match=re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts",info)
        require(bool(match),"could not read PDF page size")
        if match:
            width_mm=float(match.group(1))/72*25.4; height_mm=float(match.group(2))/72*25.4; expected=cfg["assembly"]["publication"]; require(abs(width_mm-float(expected["width_mm"]))<.2 and abs(height_mm-float(expected["height_mm"]))<.2,f"PDF dimensions are {width_mm:.2f} x {height_mm:.2f} mm")
        fonts=subprocess.run(["pdffonts",str(pdf)],capture_output=True,text=True,check=True).stdout; require("TrueType" in fonts and " yes " in fonts,"PDF fonts are not embedded TrueType")
        require(pdf.stat().st_size<50*1024**2,"PDF exceeds Nature 50 MB recommendation")
    if svg.exists(): require("<text" in svg.read_text(encoding="utf-8",errors="ignore"),"SVG text is not editable")
    source_manifest=FIGURES_DIR/"Assembled"/f"FigureSourceManifest_{rid}.json"; require(source_manifest.is_file(),"figure source manifest is missing")
    standalone_names={"a":"ResolutionProtocol","b":"DataBenefitL2","c":"DataBenefitQualitative","d":"CoarseDetail","e":"ZeroHTransfer","f":"ZeroHQualitative","g":"FrequencyError","h":"SensorSweep"}
    for label,name in standalone_names.items():
        for suffix in ("svg","pdf","png"):
            require((FIGURES_DIR/"PublicationPanels"/f"Panel_{label}_{name}_{rid}.{suffix}").is_file(),f"missing standalone panel {label} {suffix}")
    evidence.update({"canonical_cases":len(index),"sensor_plan_rows":len(plan),"manifest_rows":len(manifest),"successful_caches":len(ok_rows),"missing_placeholders":len(manifest)-len(ok_rows),"orphan_files":len(orphans),"failures":failures})
    out=RESULTS_DIR/"ModelInventory"/f"FormalAudit_{rid}.json"; write_json(out,evidence)
    if failures:
        for failure in failures: print(f"[FAIL] {failure}")
        raise SystemExit(1)
    print(f"[OK] formal audit passed: {out}")

if __name__=="__main__": main()

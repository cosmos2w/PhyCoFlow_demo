#!/usr/bin/env python
"""Resumable formal 300-case workflow for the eight-panel publication figure."""
from __future__ import annotations
import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE=Path(__file__).resolve().parent


def run(cmd):
    print("[RUN]"," ".join(map(str,cmd)),flush=True); subprocess.run([str(x) for x in cmd],check=True)


def py(script,*args): return [sys.executable,HERE/script,*args]
def fig(script,*args): return ["conda","run","-n","fig","python",HERE/script,*args]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=["preflight","plan","cache_main","cache_sweep","export","plot","assemble","all"],default="preflight")
    p.add_argument("--run-id",default=None); p.add_argument("--config",default=str(HERE/"postprocess_config.yaml")); args=p.parse_args()
    rid=args.run_id or datetime.now().strftime("formal_%Y%m%d_%H%M%S"); common=["--config",args.config,"--run-id",rid]
    phases=["plan","cache_main","cache_sweep","export","plot","assemble"] if args.phase=="all" else [args.phase]
    if "preflight" in phases:
        run(py("02_build_reconstruction_cache.py",*common,"--preflight-only")); return
    if "plan" in phases:
        run(py("00_inventory_models.py",*common)); run(py("01_build_sensor_plan.py",*common,"--max-snapshots","300"))
    plan_path=HERE.parent/"_Process_Results"/"SensorPlans"/f"SensorPlan_{rid}.csv"
    manifest=HERE.parent/"_Process_Results"/"ReconstructionCache"/f"ReconstructionCache_manifest_{rid}.csv"
    if "cache_main" in phases:
        run(py("02_build_reconstruction_cache.py",*common,"--sensor-plan",str(plan_path),"--sensor-counts","256"))
    if "cache_sweep" in phases:
        run(py("02_build_reconstruction_cache.py",*common,"--sensor-plan",str(plan_path),"--recipes","3_Mixed_HML","4_ZeroH_Balanced","5_ZeroH_MRich","--sensor-counts","64","128","384","512","768","1024"))
    if "export" in phases:
        run(py("10_export_resolution_protocol.py",*common,"--sensor-plan",str(plan_path),"--snapshot","0"))
        for script in ("20_export_questionA_l2.py","40_export_coarse_detail.py","50_export_questionB_transfer.py","60_export_frequency_error.py","62_export_gradient_binned_error.py","70_export_sensor_sweep.py"):
            run(py(script,*common,"--cache-manifest",str(manifest)))
        run(py("80_select_representative_snapshot.py",*common))
    if "plot" in phases:
        for script in ("11_plot_resolution_protocol.py","21_plot_questionA_l2.py","41_plot_coarse_detail.py","51_plot_questionB_transfer.py","61_plot_frequency_error.py","63_plot_gradient_binned_error.py","71_plot_sensor_sweep.py"):
            run(fig(script,*common,"--paper") if script in {"11_plot_resolution_protocol.py","21_plot_questionA_l2.py"} else fig(script,*common))
    if "assemble" in phases:
        reps=HERE.parent/"_Process_Results"/"CanonicalTestIndex"/f"RepresentativeSnapshots_{rid}.csv"
        run(fig("93_export_publication_panels.py",*common,"--cache-manifest",str(manifest),"--representatives",str(reps)))
        run(fig("91_assemble_mixed_resolution_publication.py",*common,"--cache-manifest",str(manifest),"--representatives",str(reps)))
    print(f"[OK] formal workflow phase(s) complete | run_id={rid}")

if __name__=="__main__": main()

#!/usr/bin/env python
"""Orchestrate a safe smoke workflow or an explicitly requested full export."""
from __future__ import annotations
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from common.config import SCRIPT_DIR


def run(name,*args):
    cmd=[sys.executable,str(SCRIPT_DIR/name),*map(str,args)]; print("[RUN] "+" ".join(cmd),flush=True); subprocess.run(cmd,check=True,cwd=SCRIPT_DIR)


def main()->int:
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter); p.add_argument("--mode",choices=["smoke","full"],default="smoke"); p.add_argument("--run-id",default=datetime.now().strftime("%Y%m%d_%H%M%S")); p.add_argument("--device"); p.add_argument("--checkpoint",choices=["last","best"],default="last"); p.add_argument("--allow-checkpoint-fallback",action="store_true"); p.add_argument("--force-regenerate",action="store_true",help="Reconstruct even when a compatible cache already exists for this run ID."); args=p.parse_args(); common=["--run-id",args.run_id,"--checkpoint",args.checkpoint]
    if args.allow_checkpoint_fallback: common.append("--allow-checkpoint-fallback")
    run("00_inventory_models.py",*common)
    sensor=["--run-id",args.run_id]
    if args.mode=="smoke": sensor += ["--max-snapshots","2"]
    run("01_build_sensor_plan.py",*sensor)
    cache=common+["--sensor-plan",str(SCRIPT_DIR.parent/"_Process_Results"/"SensorPlans"/f"SensorPlan_{args.run_id}.csv")]
    if args.device: cache += ["--device",args.device]
    if args.force_regenerate: cache += ["--force-regenerate"]
    run("02_build_reconstruction_cache.py",*cache)
    manifest=str(SCRIPT_DIR.parent/"_Process_Results"/"ReconstructionCache"/f"ReconstructionCache_manifest_{args.run_id}.csv")
    run("10_export_contours.py","--run-id",args.run_id,"--cache-manifest",manifest,"--models","DMFGen","--conditions","Cond_T")
    run("20_export_field_l2.py","--run-id",args.run_id,"--cache-manifest",manifest); run("21_plot_field_l2_heatmap.py","--run-id",args.run_id)
    run("30_export_joint_pdf_snapshot.py","--run-id",args.run_id,"--cache-manifest",manifest); run("31_plot_joint_pdf_snapshot.py","--run-id",args.run_id)
    run("40_export_joint_pdf_jsd.py","--run-id",args.run_id,"--cache-manifest",manifest); run("41_plot_joint_pdf_jsd_violin.py","--run-id",args.run_id)
    # Cache-only publication coupling validation: T--U1, CO--U1, and CO--T.
    # This never triggers model inference; it reuses finalized JSD rows where
    # possible and reads caches solely for the new CO--U1 distribution.
    run("42_export_coupling_jsd.py", "--run-id", args.run_id, "--cache-manifest", manifest)
    run("55_plot_coupling_jsd_si.py", "--run-id", args.run_id)
    run("50_export_energy_spectra.py", "--run-id", args.run_id, "--cache-manifest", manifest)
    run("51_plot_energy_spectra.py", "--run-id", args.run_id)
    lsd_args = ["--run-id", args.run_id, "--cache-manifest", manifest]
    if args.device: lsd_args += ["--device", args.device]
    if args.mode == "smoke": lsd_args += ["--max-snapshots", "2"]
    run("52_export_spectral_lsd.py", *lsd_args)
    run("53_plot_spectral_lsd.py", "--run-id", args.run_id)
    run("54_plot_spectral_validation_composite.py", "--run-id", args.run_id)
    run("plot_condition_matrix.py","--run-id",args.run_id); run("90_assemble_figure.py","--run-id",args.run_id)
    print(f"[DONE] run-id={args.run_id}"); return 0
if __name__=="__main__": raise SystemExit(main())

#!/usr/bin/env python
"""Run selected post-processing stages; full-test inference requires --full."""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
STAGES={
    "inventory":"00_inventory_models.py", "sensors":"01_build_sensor_plan.py", "cache":"02_build_reconstruction_cache.py",
    "resolution_export":"10_export_resolution_protocol.py", "resolution_plot":"11_plot_resolution_protocol.py",
    "questionA_export":"20_export_questionA_l2.py", "questionA_plot":"21_plot_questionA_l2.py", "contours":"30_export_comparison_contours.py",
    "coarse_export":"40_export_coarse_detail.py", "coarse_plot":"41_plot_coarse_detail.py",
    "questionB_export":"50_export_questionB_transfer.py", "questionB_plot":"51_plot_questionB_transfer.py",
    "frequency_export":"60_export_frequency_error.py", "frequency_plot":"61_plot_frequency_error.py",
    "gradient_export":"62_export_gradient_binned_error.py", "gradient_plot":"63_plot_gradient_binned_error.py",
    "sweep_export":"70_export_sensor_sweep.py", "sweep_plot":"71_plot_sensor_sweep.py", "assemble":"90_assemble_figure.py",
}

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--stages",nargs="+",choices=[*STAGES,"all"],default=["inventory"]); p.add_argument("--run-id"); p.add_argument("--config",default=str(HERE/"postprocess_config.yaml")); p.add_argument("--full",action="store_true",help="Permit default 200-snapshot cache/inference processing."); p.add_argument("--max-snapshots",type=int,default=2,help="Safety limit used without --full."); args,extra=p.parse_known_args(); stages=list(STAGES) if "all" in args.stages else args.stages
    for stage in stages:
        cmd=[sys.executable,str(HERE/STAGES[stage]),"--config",args.config]
        if args.run_id: cmd += ["--run-id",args.run_id]
        if not args.full:
            if stage=="sensors": cmd += ["--max-snapshots",str(args.max_snapshots)]
            if stage=="cache": cmd += ["--snapshots",*map(str,range(args.max_snapshots))]
        cmd += extra; print("[RUN]"," ".join(cmd),flush=True); subprocess.run(cmd,check=True)

if __name__=="__main__": main()

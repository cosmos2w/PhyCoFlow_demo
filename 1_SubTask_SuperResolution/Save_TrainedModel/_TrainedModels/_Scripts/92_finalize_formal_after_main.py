#!/usr/bin/env python
"""Wait for the main cache, then run sweep, exports, plots, and final assembly."""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent


def run(*args):
    cmd=[str(x) for x in args]; print("[RUN]"," ".join(cmd),flush=True); subprocess.run(cmd,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run-id",required=True); p.add_argument("--poll-seconds",type=int,default=30); p.add_argument("--timeout-hours",type=float,default=12); args=p.parse_args()
    results=HERE.parent/"_Process_Results"; log=results/"Logs"/f"cache_main_{args.run_id}.log"; deadline=time.time()+args.timeout_hours*3600
    while time.time()<deadline:
        text=log.read_text(encoding="utf-8",errors="replace") if log.exists() else ""
        if "[OK] formal workflow phase(s) complete" in text: break
        if "Traceback (most recent call last)" in text or "cache_run_already_active" in text: raise RuntimeError(f"main cache failed; inspect {log}")
        print(f"[WAIT] main cache still active | log={log}",flush=True); time.sleep(max(5,args.poll_seconds))
    else: raise TimeoutError(f"main cache did not finish within {args.timeout_hours} h")
    py=sys.executable; common=["--run-id",args.run_id]
    run(py,HERE/"03_rebuild_cache_manifest.py",*common)
    run(py,HERE/"run_formal_postprocess.py","--phase","cache_sweep",*common)
    run(py,HERE/"03_rebuild_cache_manifest.py",*common)
    plan=results/"SensorPlans"/f"SensorPlan_{args.run_id}.csv"
    run(py,HERE/"02_build_reconstruction_cache.py",*common,"--sensor-plan",plan,"--models","FFM_Perceiver",
        "--recipes","4_ZeroH_Balanced","5_ZeroH_MRich","--sensor-counts","256")
    run(py,HERE/"03_rebuild_cache_manifest.py",*common)
    run(py,HERE/"06_reconcile_sensor_metadata.py",*common,"--sensor-plan",plan)
    run(py,HERE/"03_rebuild_cache_manifest.py",*common)
    run(py,HERE/"04_prune_orphaned_cache.py",*common,"--apply")
    run(py,HERE/"run_formal_postprocess.py","--phase","export",*common)
    run(py,HERE/"run_formal_postprocess.py","--phase","plot",*common)
    run(py,HERE/"run_formal_postprocess.py","--phase","assemble",*common)
    run(py,HERE/"05_cache_status.py",*common)
    run(py,HERE/"94_audit_formal_workflow.py",*common)
    print(f"[OK] full formal workflow finalized | run_id={args.run_id}",flush=True)

if __name__=="__main__": main()

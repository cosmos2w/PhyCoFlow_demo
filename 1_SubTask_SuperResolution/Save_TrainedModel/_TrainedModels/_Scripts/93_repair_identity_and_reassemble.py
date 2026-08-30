#!/usr/bin/env python
"""Wait for an older finalizer, repair identity-sensitive caches, and reassemble."""
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
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run-id",required=True); args=p.parse_args(); py=sys.executable; results=HERE.parent/"_Process_Results"; log=results/"Logs"/f"finalize_{args.run_id}.log"
    while True:
        text=log.read_text(encoding="utf-8",errors="replace") if log.exists() else ""
        if "[OK] full formal workflow finalized" in text: break
        if "Traceback (most recent call last)" in text: raise RuntimeError(f"upstream finalizer failed; inspect {log}")
        print("[WAIT] upstream sweep/finalizer still active",flush=True); time.sleep(30)
    common=["--run-id",args.run_id]; plan=results/"SensorPlans"/f"SensorPlan_{args.run_id}.csv"
    run(py,HERE/"02_build_reconstruction_cache.py",*common,"--sensor-plan",plan,"--models","FFM_Perceiver","--recipes","4_ZeroH_Balanced","5_ZeroH_MRich","--sensor-counts","256")
    run(py,HERE/"03_rebuild_cache_manifest.py",*common)
    run(py,HERE/"run_formal_postprocess.py","--phase","export",*common)
    run(py,HERE/"run_formal_postprocess.py","--phase","plot",*common)
    run(py,HERE/"run_formal_postprocess.py","--phase","assemble",*common)
    run(py,HERE/"05_cache_status.py",*common)
    print(f"[OK] identity repair and final assembly complete | run_id={args.run_id}",flush=True)

if __name__=="__main__": main()

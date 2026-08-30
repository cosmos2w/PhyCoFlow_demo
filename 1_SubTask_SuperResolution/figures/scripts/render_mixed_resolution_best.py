#!/usr/bin/env python
"""Re-render and audit unified-v2 panels from an existing validated cache/results run."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "Save_TrainedModel" / "_TrainedModels" / "_Scripts"


def run(script: str, *arguments: str) -> None:
    subprocess.run([sys.executable, str(SCRIPTS / script), *map(str, arguments)], check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-data-run-id", required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--representatives", type=Path, required=True)
    args = parser.parse_args()
    common = [
        "--run-id", args.run_id,
        "--data-run-id", args.run_id,
        "--multiscale-run-id", args.run_id,
        "--base-data-run-id", args.base_data_run_id,
        "--cache-manifest", args.cache_manifest,
        "--representatives", args.representatives,
        "--qualitative-version", "2",
    ]
    run("96_export_unified_v2_panels.py", *common)
    run("97_assemble_mixed_resolution_unified_v2.py", *common)
    run(
        "98_audit_unified_v2.py",
        "--run-id", args.run_id,
        "--data-run-id", args.run_id,
        "--source-data-run-id", args.run_id,
        "--multiscale-run-id", args.run_id,
        "--base-data-run-id", args.base_data_run_id,
        "--allow-updated-cache-artifacts",
        "--refreshed-models", "DMFGen", "FFM_Perceiver", "Senseiver", "MLP_RBF",
    )


if __name__ == "__main__":
    main()

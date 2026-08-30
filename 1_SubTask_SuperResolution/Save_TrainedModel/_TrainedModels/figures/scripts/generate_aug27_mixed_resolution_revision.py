#!/usr/bin/env python
"""Rebuild and package the Aug. 27 mixed-resolution figure revision."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


SOURCE_DATA_RUN_ID = "20260806_1124"
MULTISCALE_RUN_ID = "20260802_1250"
BASE_DATA_RUN_ID = "2026-08-06_11-24"


def run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def cvd_preview(source: Path, destination: Path) -> None:
    """Write an approximate deuteranopia screening preview from the PNG export."""
    rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.float32) / 255.0
    matrix = np.asarray([
        [0.367, 0.861, -0.228],
        [0.280, 0.673, 0.047],
        [-0.012, 0.043, 0.969],
    ], dtype=np.float32)
    simulated = np.clip(rgb @ matrix.T, 0.0, 1.0)
    Image.fromarray(np.round(simulated * 255.0).astype(np.uint8)).save(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    args = parser.parse_args()

    models_root = Path(__file__).resolve().parents[2]
    subtask_root = models_root.parents[1]
    scripts = models_root / "_Scripts"
    assembled = models_root / "_Process_Figures" / "Assembled"
    delivery = models_root / "figures" / "generated" / "mixed_resolution_aug27_revision_v2"
    delivery.mkdir(parents=True, exist_ok=True)

    common = [
        "--data-run-id", SOURCE_DATA_RUN_ID,
        "--multiscale-run-id", MULTISCALE_RUN_ID,
        "--base-data-run-id", BASE_DATA_RUN_ID,
    ]
    run([
        sys.executable, str(scripts / "97_assemble_mixed_resolution_unified_v2.py"),
        "--run-id", args.run_id, *common,
    ], cwd=subtask_root)
    run([
        sys.executable, str(scripts / "98_audit_unified_v2.py"),
        "--data-run-id", args.run_id,
        "--source-data-run-id", SOURCE_DATA_RUN_ID,
        "--multiscale-run-id", MULTISCALE_RUN_ID,
        "--base-data-run-id", BASE_DATA_RUN_ID,
        "--composite-only", "--refreshed-models", "FFM_Perceiver",
        "--allow-updated-cache-artifacts",
    ], cwd=subtask_root)

    stem = f"MixedResolution_unified_v2_{args.run_id}"
    for extension in (".svg", ".pdf", ".png"):
        shutil.copy2(assembled / f"{stem}{extension}", delivery / f"{stem}{extension}")
    shutil.copy2(
        assembled / f"FigureSourceManifest_unified_v2_{args.run_id}.json",
        delivery / f"FigureSourceManifest_unified_v2_{args.run_id}.json",
    )
    shutil.copy2(
        models_root / "_Process_Results" / "UnifiedPublicationV2"
        / f"UnifiedV2Audit_{args.run_id}.json",
        delivery / f"UnifiedV2Audit_{args.run_id}.json",
    )

    png = delivery / f"{stem}.png"
    Image.open(png).convert("RGB").save(
        delivery / f"{stem}.tiff", dpi=(600, 600), compression="tiff_lzw",
    )
    Image.open(png).convert("L").save(delivery / f"{stem}_grayscale_QA.png")
    cvd_preview(png, delivery / f"{stem}_deuteranopia_QA.png")


if __name__ == "__main__":
    main()

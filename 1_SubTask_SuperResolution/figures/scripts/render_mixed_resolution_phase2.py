#!/usr/bin/env python
"""Render and verify the visualization-only Nature Phase 2 revision."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

from render_mixed_resolution_phase1 import sha256, scientific_signature, tree_state


ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = ROOT / "Save_TrainedModel" / "_TrainedModels"
SCRIPTS = MODELS_ROOT / "_Scripts"
ASSEMBLED = MODELS_ROOT / "_Process_Figures" / "Assembled"
RESULTS = MODELS_ROOT / "_Process_Results"
DELIVERY = ROOT / "figures" / "generated" / "MixedResolution_unified_v2_phase2"
LAYOUT = SCRIPTS / "publication_layout_unified_v2_phase2.yaml"
BASELINE_DIR = ROOT / "figures" / "generated" / "MixedResolution_unified_v2_phase1"
BASELINE_STEM = "MixedResolution_unified_v2_phase1_20260903_2114"
BASELINE_MANIFEST = BASELINE_DIR / "FigureSourceManifest_unified_v2_20260903_2114.json"

SOURCE_DATA_RUN_ID = "20260806_1124"
MULTISCALE_RUN_ID = "20260802_1250"
BASE_DATA_RUN_ID = "2026-08-06_11-24"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    args = parser.parse_args()

    output_stem = f"MixedResolution_unified_v2_phase2_{args.run_id}"
    assembled_outputs = [ASSEMBLED / f"{output_stem}.{suffix}" for suffix in ("svg", "pdf", "png")]
    assembled_manifest = ASSEMBLED / f"FigureSourceManifest_unified_v2_{args.run_id}.json"
    delivery_outputs = [DELIVERY / path.name for path in assembled_outputs]
    delivery_manifest = DELIVERY / assembled_manifest.name
    tiff_path = DELIVERY / f"{output_stem}.tiff"
    qa_path = DELIVERY / f"Phase2_visual_QA_{args.run_id}.json"
    targets = [*assembled_outputs, assembled_manifest, *delivery_outputs, delivery_manifest, tiff_path, qa_path]
    collisions = [str(path) for path in targets if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite existing Phase 2 outputs: {collisions}")

    baseline_paths = [BASELINE_DIR / f"{BASELINE_STEM}.{suffix}" for suffix in ("svg", "pdf", "png")]
    missing = [str(path) for path in [*baseline_paths, BASELINE_MANIFEST] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Phase 1 baseline artifact(s): {missing}")
    baseline_hashes_before = {path.name: sha256(path) for path in baseline_paths}
    result_state_before = tree_state(RESULTS)

    command = [
        sys.executable,
        str(SCRIPTS / "97_assemble_mixed_resolution_unified_v2.py"),
        "--layout", str(LAYOUT),
        "--run-id", args.run_id,
        "--data-run-id", SOURCE_DATA_RUN_ID,
        "--multiscale-run-id", MULTISCALE_RUN_ID,
        "--base-data-run-id", BASE_DATA_RUN_ID,
        "--qualitative-version", "2",
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    DELIVERY.mkdir(parents=True, exist_ok=True)
    for source, destination in zip(assembled_outputs, delivery_outputs):
        shutil.copy2(source, destination)
    shutil.copy2(assembled_manifest, delivery_manifest)
    Image.open(delivery_outputs[2]).convert("RGB").save(
        tiff_path, dpi=(600, 600), compression="tiff_lzw"
    )

    svg_text = delivery_outputs[0].read_text(encoding="utf-8")
    svg_uses_arial_only = (
        "font-family: 'Arial'" in svg_text
        and "Liberation Sans" not in svg_text
        and "DejaVu Sans" not in svg_text
    )
    pdffonts_output = ""
    if shutil.which("pdffonts") is not None:
        pdffonts_output = subprocess.run(
            ["pdffonts", str(delivery_outputs[1])],
            check=True, capture_output=True, text=True,
        ).stdout
    pdf_embeds_arial_only = bool(pdffonts_output) and all(
        "Arial" in line for line in pdffonts_output.splitlines()[2:] if line.strip()
    )

    baseline_manifest = load_json(BASELINE_MANIFEST)
    phase2_manifest = load_json(delivery_manifest)
    baseline_hashes_after = {path.name: sha256(path) for path in baseline_paths}
    result_state_after = tree_state(RESULTS)
    layout_qa = phase2_manifest["layout"]
    panel_c_layout = phase2_manifest["panels"]["c"]["layout"]
    panel_c_style = panel_c_layout["relative_l2_annotation_style"]
    panel_e_style = phase2_manifest["panels"]["e"]["relative_l2_annotation_style"]

    checks = {
        "phase1_files_unchanged": baseline_hashes_before == baseline_hashes_after,
        "result_files_unchanged": result_state_before == result_state_after,
        "scientific_signature_identical": (
            scientific_signature(baseline_manifest) == scientific_signature(phase2_manifest)
        ),
        "panel_b_top_right_spines_removed": layout_qa["panel_b_open_axis_qa"]["passed"],
        "qualitative_annotations_pure_white": layout_qa["relative_l2_contrast_qa"]["passed"],
        "panel_c_annotation_box_tight": panel_c_style["box_pad"] <= 0.07,
        "panel_c_extra_third_row_title_removed": (
            panel_c_layout["bottom_row_headers"]["models"] is None
        ),
        "panel_e_annotation_box_tight": panel_e_style["box_pad"] <= 0.07,
        "panel_f_values_centered": layout_qa["panel_f_cell_alignment_qa"]["passed"],
        "uniform_frame_lineweight_0_75_pt": (
            layout_qa["frame_lineweight_qa"]["passed"]
            and layout_qa["frame_lineweight_qa"]["linewidth_pt"] == 0.75
        ),
        "panel_c_header_overlap_passed": layout_qa["panel_c_overlap_qa"]["passed"],
        "typography_contract_passed": layout_qa["typography_qa"]["passed"],
        "model_line_contract_passed": layout_qa["model_line_qa"]["passed"],
        "svg_uses_arial_only": svg_uses_arial_only,
        "pdf_embeds_arial_only": pdf_embeds_arial_only,
    }
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": args.run_id,
        "baseline_stem": BASELINE_STEM,
        "checks": checks,
        "phase1_sha256_before": baseline_hashes_before,
        "phase1_sha256_after": baseline_hashes_after,
        "frame_lineweight_qa": layout_qa["frame_lineweight_qa"],
        "panel_b_open_axis_qa": layout_qa["panel_b_open_axis_qa"],
        "relative_l2_contrast_qa": layout_qa["relative_l2_contrast_qa"],
        "panel_f_cell_alignment_qa": layout_qa["panel_f_cell_alignment_qa"],
        "panel_a_resolution_legend_y": phase2_manifest["panels"]["a"]["image_layout"]["resolution_legend_y"],
        "result_file_count": len(result_state_before),
        "pdf_font_listing": pdffonts_output.splitlines()[2:],
        "outputs": [str(path.resolve()) for path in [*delivery_outputs, tiff_path, delivery_manifest]],
    }
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if qa["status"] != "pass":
        raise RuntimeError(f"Phase 2 QA failed; inspect {qa_path}")
    print(f"[OK] {delivery_outputs[1]}")
    print(f"[OK] {qa_path}")


if __name__ == "__main__":
    main()

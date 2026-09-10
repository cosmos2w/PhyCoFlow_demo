#!/usr/bin/env python
"""Render and verify the visualization-only Nature Phase 1 revision."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = ROOT / "Save_TrainedModel" / "_TrainedModels"
SCRIPTS = MODELS_ROOT / "_Scripts"
ASSEMBLED = MODELS_ROOT / "_Process_Figures" / "Assembled"
RESULTS = MODELS_ROOT / "_Process_Results"
DELIVERY = ROOT / "figures" / "generated" / "MixedResolution_unified_v2_phase1"
LAYOUT = SCRIPTS / "publication_layout_unified_v2_phase1.yaml"
BASELINE_STEM = "MixedResolution_unified_v2_20260827_1537"
BASELINE_DIR = ASSEMBLED
BASELINE_MANIFEST = (
    MODELS_ROOT / "figures" / "generated" / "mixed_resolution_aug27_revision_v2"
    / "FigureSourceManifest_unified_v2_20260827_1537.json"
)

SOURCE_DATA_RUN_ID = "20260806_1124"
MULTISCALE_RUN_ID = "20260802_1250"
BASE_DATA_RUN_ID = "2026-08-06_11-24"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_state(root: Path) -> dict[str, list[int]]:
    """Capture non-content metadata to prove the result tree stayed untouched."""
    return {
        str(path.relative_to(root)): [path.stat().st_size, path.stat().st_mtime_ns]
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def scientific_signature(manifest: dict) -> dict:
    """Remove only presentation metadata before exact baseline comparison."""
    panels = json.loads(json.dumps(manifest["panels"]))
    for key in ("image_layout", "grid_linewidth", "removed_elements"):
        panels["a"].pop(key, None)
    panels["c"].pop("layout", None)
    for key in ("shared_x_label", "secondary_tick_layout"):
        panels["d"].pop(key, None)
    panels["b"].pop("visual_style", None)
    for key in (
        "colorbar_count", "colorbar_ticks", "colorbar_tick_formatter",
        "residual_colorbar_bounds_parent", "residual_colorbar_height_fraction",
        "residual_row_height_parent", "columns",
        "relative_l2_annotation_style",
    ):
        panels["e"].pop(key, None)
    for key in (
        "colorbar_ticks", "colorbar_tick_formatter", "colorbar_titles",
        "annotation_text_color",
        "cell_text_alignment",
    ):
        panels["f"].pop(key, None)
    return {
        "data_run_id": manifest["data_run_id"],
        "multiscale_run_id": manifest["multiscale_run_id"],
        "base_data_run_id": manifest["base_data_run_id"],
        "selected_models": manifest["selected_models"],
        "selected_field": manifest["selected_field"],
        "canonical_selection": manifest["canonical_selection"],
        "csv_sources": sorted(
            (entry["path"], entry["sha256"]) for entry in manifest["csv_sources"]
        ),
        "cache_paths": sorted(entry["path"] for entry in manifest["cache_paths"]),
        "panels": panels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    args = parser.parse_args()

    output_stem = f"MixedResolution_unified_v2_phase1_{args.run_id}"
    assembled_outputs = [ASSEMBLED / f"{output_stem}.{suffix}" for suffix in ("svg", "pdf", "png")]
    assembled_manifest = ASSEMBLED / f"FigureSourceManifest_unified_v2_{args.run_id}.json"
    delivery_outputs = [DELIVERY / path.name for path in assembled_outputs]
    delivery_manifest = DELIVERY / assembled_manifest.name
    tiff_path = DELIVERY / f"{output_stem}.tiff"
    qa_path = DELIVERY / f"Phase1_visual_QA_{args.run_id}.json"
    all_targets = [*assembled_outputs, assembled_manifest, *delivery_outputs, delivery_manifest, tiff_path, qa_path]
    collisions = [str(path) for path in all_targets if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite existing Phase 1 outputs: {collisions}")

    baseline_paths = [BASELINE_DIR / f"{BASELINE_STEM}.{suffix}" for suffix in ("svg", "pdf", "png")]
    missing = [str(path) for path in [*baseline_paths, BASELINE_MANIFEST] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing baseline artifact(s): {missing}")

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
        "Arial" in line
        for line in pdffonts_output.splitlines()[2:]
        if line.strip()
    )

    baseline_manifest = load_json(BASELINE_MANIFEST)
    phase1_manifest = load_json(delivery_manifest)
    baseline_signature = scientific_signature(baseline_manifest)
    phase1_signature = scientific_signature(phase1_manifest)
    baseline_hashes_after = {path.name: sha256(path) for path in baseline_paths}
    result_state_after = tree_state(RESULTS)

    panel_geometry = phase1_manifest["layout"]["panel_geometry"]
    content_bounds = phase1_manifest["layout"]["content_bounds_panel_fraction"]
    c_parent_width = panel_geometry["c"]["width_mm"] * content_bounds["c"][2]
    e_parent_width = panel_geometry["e"]["width_mm"] * content_bounds["e"][2]
    f_parent_width = panel_geometry["f"]["width_mm"] * content_bounds["f"][2]
    c_layout = phase1_manifest["panels"]["c"]["layout"]
    e_bounds = phase1_manifest["panels"]["e"]["residual_colorbar_bounds_parent"]
    colorbar_widths_mm = {
        "c": c_parent_width * c_layout["colorbar_width_parent_fraction"],
        "e": e_parent_width * e_bounds["large"][2],
        "f": f_parent_width * 0.012396,
    }
    colorbar_gaps_mm = {
        "c": c_parent_width * c_layout["colorbar_gap_parent_fraction"],
        "e": e_parent_width * (0.824253 - 0.800000),
        "f": f_parent_width * (0.865660 - 0.845000),
    }
    width_spread = max(colorbar_widths_mm.values()) - min(colorbar_widths_mm.values())
    gap_spread = max(colorbar_gaps_mm.values()) - min(colorbar_gaps_mm.values())

    checks = {
        "baseline_files_unchanged": baseline_hashes_before == baseline_hashes_after,
        "result_files_unchanged": result_state_before == result_state_after,
        "scientific_signature_identical": baseline_signature == phase1_signature,
        "panel_c_header_overlap_passed": phase1_manifest["layout"]["panel_c_overlap_qa"]["passed"],
        "typography_contract_passed": phase1_manifest["layout"]["typography_qa"]["passed"],
        "svg_uses_arial_only": svg_uses_arial_only,
        "pdf_embeds_arial_only": pdf_embeds_arial_only,
        "model_line_contract_passed": phase1_manifest["layout"]["model_line_qa"]["passed"],
        "colorbar_width_spread_le_0_01_mm": width_spread <= 0.01,
        "colorbar_gap_spread_le_0_01_mm": gap_spread <= 0.01,
        "panel_d_secondary_ticks_separated": (
            phase1_manifest["panels"]["d"].get("secondary_tick_layout")
            == "separate aligned row"
        ),
    }
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "run_id": args.run_id,
        "baseline_stem": BASELINE_STEM,
        "checks": checks,
        "baseline_sha256_before": baseline_hashes_before,
        "baseline_sha256_after": baseline_hashes_after,
        "colorbar_widths_mm": colorbar_widths_mm,
        "colorbar_width_spread_mm": width_spread,
        "colorbar_gaps_mm": colorbar_gaps_mm,
        "colorbar_gap_spread_mm": gap_spread,
        "typography_role_sizes_pt": phase1_manifest["layout"]["typography_qa"]["role_sizes_pt"],
        "pdf_font_listing": pdffonts_output.splitlines()[2:],
        "result_file_count": len(result_state_before),
        "outputs": [str(path.resolve()) for path in [*delivery_outputs, tiff_path, delivery_manifest]],
    }
    qa_path.write_text(json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if qa["status"] != "pass":
        raise RuntimeError(f"Phase 1 QA failed; inspect {qa_path}")
    print(f"[OK] {delivery_outputs[1]}")
    print(f"[OK] {qa_path}")


if __name__ == "__main__":
    main()

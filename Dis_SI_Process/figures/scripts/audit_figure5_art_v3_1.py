#!/usr/bin/env python
"""Independent release audit for the Figure 5 art-style V3.1 bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
from PIL import Image


REQUIRED = {
    "figure5_log.pdf",
    "figure5_log.svg",
    "figure5_log.png",
    "figure5_log_preview.png",
    "figure5_log_162mm.png",
    "figure5_log_162mm_grayscale.png",
    "figure5_log_162mm_deuteranopia.png",
    "art_v3_162mm.png",
    "LAYOUT_QA.json",
    "figure_contract.md",
    "STYLE_CHANGELOG.md",
    "source_manifest.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    release = args.release_dir.resolve()
    missing = sorted(name for name in REQUIRED if not (release / name).is_file())
    qa = json.loads((release / "LAYOUT_QA.json").read_text()) if not missing else {}
    manifest = json.loads((release / "source_manifest.json").read_text()) if not missing else {}
    pdf = fitz.open(release / "figure5_log.pdf") if not missing else None
    size_mm = [pdf[0].rect.width * 25.4 / 72.0, pdf[0].rect.height * 25.4 / 72.0] if pdf is not None else [0.0, 0.0]
    pdf_text = pdf[0].get_text() if pdf is not None else ""
    svg_text = (release / "figure5_log.svg").read_text() if not missing else ""
    stored_hashes = manifest.get("outputs", {})
    hash_mismatches = sorted(
        name for name, expected in stored_hashes.items() if not (release / name).is_file() or _sha256(release / name) != expected
    )
    raster_sizes = {
        name: Image.open(release / name).size
        for name in ("figure5_log_162mm.png", "figure5_log_162mm_grayscale.png", "figure5_log_162mm_deuteranopia.png")
        if (release / name).is_file()
    }
    geometry = qa.get("v3_1_geometry", {})
    violin_alphas = [alpha for record in qa.get("violin_fill_records", []) for alpha in record.get("alphas", [])]
    bar_records = qa.get("resource_bar_fill_records", {})
    filled_alphas = [record.get("face_alpha") for record in bar_records.get("filled", [])]
    error_axes = qa.get("resource_errorbar_override", {}).get("axes", [])
    required_text = [
        "Normalized CRPS",
        "Spearman",
        "Retained least-uncertain states",
        "Unobserved-field",
        "High-band relative",
        "High-band",
        "Frequency shell",
        "Training update time",
        "Inference memory",
        "DMF-Gen",
        "Senseiver",
    ]
    checks = {
        "required_artifacts": not missing,
        "all_renderer_checks_pass": bool(qa.get("checks")) and all(qa["checks"].values()),
        "accepted_source_hash": qa.get("scientific_hash_before") == "8248fd8165ed1749d42bfc2672ea79aff3a7201e6aad5152b8a9f412c651da32",
        "scientific_artist_data_exact": qa.get("scientific_data_hash_before") == qa.get("scientific_data_hash_after") == "ee2c40b7cc252cde3ee1e2589ab5e487ea53cdd161b9a247a5bbdba3b003b3ab",
        "single_authorized_view_limit_override": qa.get("authorized_view_limit_override", {}).get("axis_index") == 9 and qa.get("authorized_view_limit_override", {}).get("data_artists_changed") is False,
        "fixed_canvas_179_8x216_mm": bool(np.allclose(size_mm, [179.8, 216.0], atol=0.02, rtol=0)),
        "single_page": pdf is not None and pdf.page_count == 1,
        "fonts_embedded": pdf is not None and len(pdf[0].get_fonts()) >= 2,
        "pdf_text_inventory": all(text in pdf_text for text in required_text) and "High-band region" not in pdf_text,
        "svg_text_editable": svg_text.count("<text") > 100,
        "source_manifest_hashes": not hash_mismatches and len(stored_hashes) >= 10,
        "review_raster_dimensions_match": len(set(raster_sizes.values())) == 1 and len(raster_sizes) == 3,
        "no_collision_or_clipping_records": not qa.get("clipped_text") and not qa.get("tick_label_overlaps") and not qa.get("text_text_overlaps") and not qa.get("text_nonowned_axes_overlaps") and not qa.get("panel_tag_axes_overlaps"),
        "rendered_row_clearances_ge_3mm": min(qa.get("row_union_clearance_mm", {}).values(), default=-1) >= 3.0,
        "middle_row_exactly_80pct_v3": abs(float(geometry.get("middle_height_ratio_vs_v3", 0)) - 0.8) <= 1e-10 and bool(np.allclose(qa.get("panel_d_axis_heights_mm", []), [32.8, 32.8], atol=0.01, rtol=0)),
        "nominal_gap_cut_50_to_66pct": 1.0 / 3.0 <= float(geometry.get("nominal_gap_ratio_vs_v3", 9)) <= 0.5,
        "canvas_trim_matches_saved_space": abs(float(geometry.get("canvas_height_reduction_mm", -1)) - float(geometry.get("expected_canvas_reduction_mm", -2))) <= 1e-10,
        "distribution_ylabels_centered": len(qa.get("distribution_ylabel_centering", [])) == 2 and max(record.get("center_error_mm", 99) for record in qa.get("distribution_ylabel_centering", [])) <= 0.05,
        "twelve_violin_fills_alpha_078": len(violin_alphas) == 12 and bool(np.allclose(violin_alphas, 0.78, atol=1e-10, rtol=0)),
        "all_filled_resource_bars_alpha_078": len(filled_alphas) == 32 and bool(np.allclose(filled_alphas, 0.78, atol=1e-10, rtol=0)),
        "hollow_inference_bars_preserved": len(bar_records.get("hollow", [])) == 8,
        "all_resource_errorbars_hidden": len(error_axes) == 4 and qa.get("resource_errorbar_override", {}).get("total_hidden_artists") == 120 and all(record.get("container_count", 0) > 0 and record.get("visible_artist_count") == 0 for record in error_axes),
        "accuracy_errorbars_visible": qa.get("accuracy_errorbars_visible") is True,
        "twelve_two_sig_digit_means": len(qa.get("mean_annotations", [])) == 12 and all(record.get("display") == format(record.get("mean"), ".2g") for record in qa.get("mean_annotations", [])),
        "legend_clear_of_curve_points": not qa.get("spectrum_legend_curve_point_intersections"),
        "panel_f_spacing_matches_a": 0.95 <= float(qa.get("panel_f_to_a_row_spacing_ratio", 0)) <= 1.05,
        "dmf_gen_only_bold_model_label": qa.get("bold_nonpanel_records") == [{"axis_index": 0, "text": "DMF-Gen", "weight": "bold"}, {"axis_index": 3, "text": "DMF-Gen", "weight": "bold"}],
        "font_floor_at_162mm": qa.get("tick_label_font_pt", 0) * qa.get("insertion_scale", 0) >= 7.0 and qa.get("legend_font_pt", 0) * qa.get("insertion_scale", 0) >= 7.0,
    }
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if missing:
        print(f"Missing: {missing}")
    if hash_mismatches:
        print(f"Hash mismatches: {hash_mismatches}")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Figure 5 art V3.1 release audit failed: {failed}")
    print(f"[PASS] {release}")


if __name__ == "__main__":
    main()

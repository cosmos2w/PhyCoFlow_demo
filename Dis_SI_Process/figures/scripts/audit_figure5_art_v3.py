#!/usr/bin/env python
"""Independent release audit for the Figure 5 art-style V3 bundle."""
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
    "art_v2_162mm.png",
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
    hash_mismatches = sorted(name for name, expected in stored_hashes.items() if not (release / name).is_file() or _sha256(release / name) != expected)
    raster_sizes = {
        name: Image.open(release / name).size
        for name in ("figure5_log_162mm.png", "figure5_log_162mm_grayscale.png", "figure5_log_162mm_deuteranopia.png")
        if (release / name).is_file()
    }
    markers = qa.get("panel_a_marker_records", {})
    panel_b_markers = qa.get("panel_b_marker_records", {})
    panel_c_markers = qa.get("panel_c_marker_records", [])
    expected_palette = {
        "DMF-Gen": "#E63946",
        "FFM-FNO": "#1D3557",
        "FFM-Perceiver": "#457B9D",
        "Latent FM": "#6A4C93",
        "SiT": "#A28BC4",
    }
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
        "scientific_artist_data_exact": qa.get("scientific_data_hash_before") == qa.get("scientific_data_hash_after"),
        "single_authorized_view_limit_override": qa.get("authorized_view_limit_override", {}).get("axis_index") == 9 and qa.get("authorized_view_limit_override", {}).get("data_artists_changed") is False,
        "fixed_canvas_179_8x244_mm": bool(np.allclose(size_mm, [179.8, 244.0], atol=0.02, rtol=0)),
        "single_page": pdf is not None and pdf.page_count == 1,
        "fonts_embedded": pdf is not None and len(pdf[0].get_fonts()) >= 2,
        "pdf_text_inventory": all(text in pdf_text for text in required_text) and "High-band region" not in pdf_text,
        "svg_text_editable": svg_text.count("<text") > 100,
        "source_manifest_hashes": not hash_mismatches and len(stored_hashes) >= 10,
        "review_raster_dimensions_match": len(set(raster_sizes.values())) == 1 and len(raster_sizes) == 3,
        "outer_top_bottom_right_margins_le_1_5mm": max(qa.get("outer_artist_margins_mm", {"missing": 99}).values()) <= 1.5,
        "row_clearances_ge_3mm": min(qa.get("row_union_clearance_mm", {}).values(), default=-1) >= 3.0,
        "no_collision_or_clipping_records": not qa.get("clipped_text") and not qa.get("tick_label_overlaps") and not qa.get("text_text_overlaps") and not qa.get("text_nonowned_axes_overlaps") and not qa.get("panel_tag_axes_overlaps"),
        "top_tag_baselines_exact": max(qa.get("top_tag_baselines_mm", {"missing": 99}).values()) - min(qa.get("top_tag_baselines_mm", {"missing": 0}).values()) <= 1e-10,
        "top_subplot_gaps_increased": bool(np.allclose(qa.get("top_axis_gaps_mm", []), [9.0, 16.0], atol=0.01, rtol=0)),
        "row_tags_on_axes_top_baselines": qa.get("row_tag_baselines_mm", {}).get("d") == qa.get("middle_axes_top_mm") and qa.get("row_tag_baselines_mm", {}).get("e") == qa.get("middle_axes_top_mm") and qa.get("row_tag_baselines_mm", {}).get("f") == qa.get("score_axes_top_mm"),
        "panel_d_zero_hspace_and_stretched": abs(float(qa.get("panel_d_hspace_mm", 99))) <= 1e-10 and min(qa.get("panel_d_axis_heights_mm", [0])) >= 40.0,
        "twelve_two_sig_digit_means": len(qa.get("mean_annotations", [])) == 12 and all(record.get("display") == format(record.get("mean"), ".2g") for record in qa.get("mean_annotations", [])),
        "mean_labels_clear_violin_tips": float(qa.get("minimum_mean_to_violin_clearance_mm", -1)) >= 1.0,
        "legend_clear_of_curve_points": not qa.get("spectrum_legend_curve_point_intersections"),
        "panel_a_palette_exact": {method: record.get("color") for method, record in markers.items()} == expected_palette,
        "panel_a_markers_large_and_open": len(markers) == 5 and all(record.get("markersize_pt") == 5.8 and record.get("markeredgewidth_pt") == 0.9 and record.get("markerfacecolor") == "none" for record in markers.values()),
        "panel_b_palette_and_markers_match_a": panel_b_markers == markers,
        "panel_c_marker_geometry_matches_a": len(panel_c_markers) == 5 and all(record.get("markersize_pt") == 5.8 and record.get("markeredgewidth_pt") == 0.9 and record.get("markerfacecolor") == "none" for record in panel_c_markers),
        "panel_a_clouds_enlarged": qa.get("panel_a_cloud_collection_count") == 5,
        "inference_memory_errorbars_hidden": qa.get("inference_memory_override", {}).get("hidden_errorbar_artists", 0) > 0,
        "inference_memory_hollow_edges": qa.get("inference_memory_override", {}).get("hollow_peak_bars") == 8 and qa.get("inference_memory_override", {}).get("hollow_edge_width_pt") == 0.9,
        "panel_f_spacing_matches_a": 0.95 <= float(qa.get("panel_f_to_a_row_spacing_ratio", 0)) <= 1.05,
        "middle_bottom_gap_at_contract_floor": 3.0 <= qa.get("row_union_clearance_mm", {}).get("middle_to_bottom_mm", 99) <= 3.15,
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
        raise SystemExit(f"Figure 5 art V3 release audit failed: {failed}")
    print(f"[PASS] {release}")


if __name__ == "__main__":
    main()

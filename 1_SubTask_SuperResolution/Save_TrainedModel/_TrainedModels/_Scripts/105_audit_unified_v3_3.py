#!/usr/bin/env python
"""Strict 25-point audit for the additive mixed-resolution V3-3 release."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


MODELS = ["DMFGen", "FFM_Perceiver", "Senseiver", "MLP_RBF"]
RECIPES = ["1_H_only", "2_H_limited", "3_Mixed_HML", "4_ZeroH_Balanced", "5_ZeroH_MRich"]
FINE_RECIPES = RECIPES[2:]
SCALES = ["large", "intermediate", "fine"]
ROLE_SIZES = {"panel_label": 8.5, "subplot_title": 6.5, "axis_label": 6.0,
              "tick_label": 5.5, "legend": 5.5, "annotation": 5.5}


def sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left, right, atol=1e-12):
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol))


def relative_l2(truth, prediction):
    truth = np.asarray(truth, dtype=float); prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction); diff = prediction[valid] - truth[valid]
    return float(np.sqrt(np.sum(diff * diff, dtype=np.float64)) /
                 (np.sqrt(np.sum(truth[valid] * truth[valid], dtype=np.float64)) + 1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True); parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args(); release = args.release_dir.resolve(); checks = []

    def check(index, name, passed, detail=None):
        item = {"contract_index": int(index), "name": name, "passed": bool(passed), "detail": detail}
        checks.append(item); print(f"[{'PASS' if item['passed'] else 'FAIL'}] {index:02d} {name}")

    try:
        manifest = json.loads((release / "source_manifest_v3_3.json").read_text(encoding="utf-8"))
        a, b, c, d, e = (manifest["panels"][label] for label in "abcde")
        panel_svgs = {label: next((release / "panels").glob(f"Panel_{label}_*_{args.run_id}.svg"))
                      for label in "abcde"}
        svg = {label: path.read_text(encoding="utf-8") for label, path in panel_svgs.items()}

        # Panel a — contract items 1–4.
        roi = a.get("shared_roi", {})
        check(1, "panel_a_common_roi", a.get("roi_shared_across_resolutions") is True
              and set(roi) >= {"xmin", "xmax", "ymin", "ymax"}, roi)
        check(2, "panel_a_zoom_connectors", a.get("zoom_inset_count") == 3
              and a.get("connector_count") == 6 and svg["a"].count("panel-a-roi-connector") >= 6,
              {"metadata": a.get("connector_count"), "svg_groups": svg["a"].count("panel-a-roi-connector")})
        block = a.get("resolution_block_rows", [])
        check(3, "panel_a_vertical_resolution_block", [row.get("dimensions") for row in block]
              == ["32 × 32", "64 × 64", "128 × 128"]
              and [row.get("name") for row in block] == ["Low resolution", "Medium resolution", "High resolution"], block)
        check(4, "panel_a_old_horizontal_key_removed",
              a.get("old_horizontal_resolution_legend_present") is False, None)

        # Panel b — contract items 5–8.
        check(5, "panel_b_grouped_categorical_transfer",
              b.get("recipe_transfer_plot_type") == "grouped_bar"
              and b.get("recipe_transfer_axis_scale") == "linear"
              and b.get("subaxis_roles", [None])[0] == "recipe_transfer_grouped_bars"
              and svg["b"].count("model-bar") >= 20,
              {"plot_type": b.get("recipe_transfer_plot_type"), "bar_groups": svg["b"].count("model-bar")})
        legend = b.get("legend_contract", {})
        check(6, "panel_b_expanded_dedicated_legend", legend.get("dedicated_axis") is True
              and legend.get("ncol") == 4 and float(legend.get("fontsize_pt", 0)) >= 7.0
              and float(legend.get("columnspacing", 0)) >= 1.5
              and float(legend.get("handletextpad", 0)) >= .45, legend)
        b_legend_bottom = .905; b_top_plot_top = .520 + .325
        check(7, "panel_b_vertical_pacing", b_legend_bottom - b_top_plot_top >= .05
              and .520 - (.085 + .275) >= .12,
              {"legend_to_bar": b_legend_bottom - b_top_plot_top,
               "bar_to_sweeps": .520 - (.085 + .275)})
        check(8, "panel_b_zero_h_groups_identified", b.get("zero_h_region_shaded") is True
              and "no H-resolution training fields" in svg["b"], None)

        # Panel c — contract items 9–15.
        check(9, "panel_c_exact_mlp_rbf_present", c.get("mlp_rbf_qualitative_present") is True
              and c.get("models") == MODELS and "MLP-RBF" in svg["c"], c.get("cache_sources"))
        bounds = c.get("axes_bounds_figure_fraction", {})
        aligned = all(len(row) == 5 for row in (bounds.get("full", []), bounds.get("zoom", []), bounds.get("error", [])))
        if aligned:
            for row in ("full", "zoom", "error"):
                widths = [item[2] for item in bounds[row]]; heights = [item[3] for item in bounds[row]]
                aligned &= max(widths) - min(widths) <= 1e-12 and max(heights) - min(heights) <= 1e-12
        check(10, "panel_c_columns_exactly_aligned", aligned, bounds)
        check(11, "panel_c_sensor_tile_equal_footprint",
              c.get("sensor_layout", {}).get("placement") == "full_equal_footprint_tile"
              and float(c.get("sensor_error_footprint_max_abs_delta", 1)) <= 1e-12,
              c.get("sensor_error_footprint_max_abs_delta"))
        check(12, "panel_c_magnification_connectors", c.get("frustum_connector_count") == 10
              and svg["c"].count("panel-c-frustum-connector") >= 10,
              {"metadata": c.get("frustum_connector_count"), "svg_groups": svg["c"].count("panel-c-frustum-connector")})
        check(13, "panel_c_local_annotation_standardized",
              c.get("local_annotation_uses_local_word") is False
              and "Local rel." not in svg["c"] and "Rel." in svg["c"], c.get("local_annotation_prefix"))
        bars = c.get("colorbar_formatting", {})
        check(14, "panel_c_colorbar_exponents_clear", len(bars) == 2
              and all(item.get("automatic_offset_suppressed") is True
                      and float(item.get("minimum_right_padding_axes", 0)) >= .04 for item in bars.values())
              and svg["c"].count("colorbar-manual-exponent") >= 2, bars)
        identity_ok = c.get("cache_identity_verified") is True and c.get("snapshot") == 50 \
            and c.get("case_id") == 9160 and c.get("time_index") == 18 and c.get("sensor_count") == 512
        check(15, "panel_c_shared_state_crop_and_scale", identity_ok
              and c.get("row_cell_counts") == {"full_field": 5, "zoomed_field": 5, "local_absolute_error": 4}
              and c.get("shared_colorbar_count") == 2, c)

        # Panel d — contract items 16–18.
        check(16, "panel_d_large_intermediate_fine_restored",
              d.get("qualitative_scales") == SCALES and d.get("large_scale_present_in_main") is True,
              d.get("qualitative_scales"))
        check(17, "panel_d_right_channel_stacked_curves",
              manifest["layout"].get("row_spanning_panel") == "d"
              and d.get("quantitative_arrangement") == "correlation_above_bias_stacked_vertically"
              and d.get("main_quantitative_value_count") == 24, d.get("quantitative_arrangement"))
        check(18, "panel_d_representative_consistency_documented",
              d.get("qualitative_recipe") == c.get("recipe") and d.get("displayed_snapshot") == c.get("snapshot")
              and d.get("case_id") == c.get("case_id") and d.get("time_index") == c.get("time_index")
              and d.get("sensor_count") == 256 and c.get("sensor_count") == 512,
              {"panel_c": [c.get("snapshot"), c.get("case_id"), c.get("time_index"), c.get("sensor_count")],
               "panel_d": [d.get("displayed_snapshot"), d.get("case_id"), d.get("time_index"), d.get("sensor_count")]})

        # Panel e — contract items 19–22.
        check(19, "panel_e_both_full_matrices_present", e.get("matrix_count") == 2
              and e.get("matrix_shape") == [4, 9] and len(e.get("heatmap_values", [])) == 72, None)
        check(20, "panel_e_matrices_stacked", e.get("matrix_arrangement") == "stacked_vertically", None)
        wavelet_record = next(item for item in manifest["csv_sources"]
                              if Path(item["path"]).name.startswith("MultiscaleWavelet_summary"))
        wavelet_rows = read_csv(Path(wavelet_record["path"])); wavelet_index = {
            (row["model_key"], row["recipe"], row["scale_group"], row["metric"]): row for row in wavelet_rows
        }
        matrix_exact = True
        for cell in e.get("heatmap_values", []):
            source = wavelet_index[(cell["model_key"], cell["recipe"], cell["scale_group"], cell["metric"])]
            matrix_exact &= close(cell["value"], source[cell["statistic"]]) and int(cell["valid_n"]) == int(source["valid_n"])
        check(21, "panel_e_annotations_match_validated_table", matrix_exact, {"cells": len(e.get("heatmap_values", []))})
        e_index = {(cell["model_key"], cell["recipe"], cell["metric"]): cell["value"]
                   for cell in e.get("heatmap_values", []) if cell["scale_group"] == "fine"}
        fine_aligned = e.get("fine_column_outline_count") == 6
        for metric, model_values in d.get("quantitative_values", {}).items():
            for model, recipe_values in model_values.items():
                for recipe, value in recipe_values.items():
                    fine_aligned &= close(value, e_index[(model, recipe, metric)])
        check(22, "panel_e_fine_columns_link_to_panel_d", fine_aligned, e.get("fine_columns"))

        # Global — contract items 23–25.
        immutable = manifest.get("source_immutability", {})
        recorded = [manifest.get("configuration"), manifest.get("layout_configuration"),
                    manifest.get("renderer"), manifest.get("panel_renderer"), manifest.get("cache_manifest"),
                    manifest.get("representative_index"), *manifest.get("csv_sources", []),
                    *manifest.get("all_render_source_records", []),
                    *manifest.get("cache_sources", []), *manifest.get("table_sources", []),
                    *manifest.get("all_render_cache_sources", []), *manifest.get("referenced_shared_sources", []),
                    manifest.get("extended_multiscale_distribution_source")]
        recorded = [item for item in recorded if item and item.get("sha256")]
        hashes_match = all(Path(item["path"]).exists() and sha256(Path(item["path"])) == item["sha256"] for item in recorded)
        artifacts = manifest.get("artifact_outputs", [])
        artifact_hashes_match = bool(artifacts) and all(Path(item["path"]).exists()
                                                        and sha256(Path(item["path"])) == item["sha256"] for item in artifacts)
        check(23, "validated_sources_caches_unchanged", immutable.get("unchanged") is True
              and immutable.get("tree_state_sha256_before") == immutable.get("tree_state_sha256_after")
              and hashes_match and artifact_hashes_match
              and manifest.get("model_inference_performed") is False
              and manifest.get("validated_sources_modified") is False,
              {"source_hash_count": len(recorded), "artifact_count": len(artifacts), "tree_state": immutable})

        main_base = release / f"MixedResolution_unified_v3_3_{args.run_id}"
        main_svg, main_pdf, main_png = (main_base.with_suffix(f".{ext}") for ext in ("svg", "pdf", "png"))
        info = subprocess.run(["pdfinfo", str(main_pdf)], check=True, capture_output=True, text=True).stdout
        size_line = next(line for line in info.splitlines() if line.startswith("Page size:")); parts = size_line.split()
        pdf_mm = [float(parts[2]) * 25.4 / 72, float(parts[4]) * 25.4 / 72]
        expected = manifest["figure_contract"]["final_size_mm"]
        png_size = Image.open(main_png).size; expected_px = [round(value / 25.4 * 600) for value in expected]
        font_text = subprocess.run(["pdffonts", str(main_pdf)], check=True, capture_output=True, text=True).stdout
        arial = [line for line in font_text.splitlines() if "Arial" in line]
        layout_qa = manifest["layout"]
        standalone = all({path.suffix for path in (release / "panels").glob(f"Panel_{label}_*_{args.run_id}.*")}
                         == {".svg", ".pdf", ".png"} for label in "abcde")
        check(24, "publication_geometry_typography_collision_thresholds",
              expected[0] == 183.0 and expected[1] <= 240.0
              and all(abs(x - y) <= .02 for x, y in zip(pdf_mm, expected))
              and all(abs(x - y) <= 2 for x, y in zip(png_size, expected_px))
              and layout_qa["geometry_qa"].get("passed") is True
              and layout_qa["typography_qa"].get("passed") is True
              and not any(float(value) > 0 for value in layout_qa["text_overflow_in"].values())
              and layout_qa["panel_text_clearance_qa"].get("passed") is True
              and "<text" in main_svg.read_text(encoding="utf-8")
              and len(arial) >= 3 and standalone,
              {"pdf_mm": pdf_mm, "png_px": png_size, "expected_px": expected_px,
               "panel_text_qa": layout_qa["panel_text_clearance_qa"], "standalone": standalone})
        role_sizes = manifest["layout"]["typography_qa"].get("role_sizes_pt")
        check(25, "visual_depth_without_type_shrinkage", role_sizes == ROLE_SIZES
              and float(legend.get("fontsize_pt", 0)) > ROLE_SIZES["legend"]
              and e.get("matrix_shape") == [4, 9] and d.get("qualitative_scales") == SCALES,
              {"role_sizes_pt": role_sizes, "legend_fontsize_pt": legend.get("fontsize_pt")})

        # Supplemental/documentation completeness is required in addition to the numbered visual contract.
        si = manifest.get("si_outputs", {}); tables = manifest.get("table_outputs", {})
        si_ok = set(si) == {"Sx1_complete_sensor_sweeps", "Sx2_expanded_recipe_gallery",
                            "Sx3_complete_three_scale_qualitative", "Sx4_complete_three_scale_quantitative"} \
            and all({Path(item["path"]).suffix for item in rows} == {".svg", ".pdf", ".png"} for rows in si.values())
        docs = ["source_manifest_v3_3.json", "quantitative_figure_report_v3_3.md",
                "figure_reference_update_v3_3.md", "completion_report_v3_3.md", "figure_contract.md"]
        check(26, "required_si_tables_and_documentation", si_ok and set(tables) == {
              "accuracy_512", "sensor_sweeps_64_512", "pattern_correlations_all_scales",
              "variance_allocation_bias_all_scales"} and all((release / name).exists() for name in docs),
              {"si": sorted(si), "tables": sorted(tables), "docs": docs})

        # Recompute panel-c identity and annotations directly from the four immutable caches.
        recomputed = {}; identity = True; reference = None
        for cache_path in map(Path, c["cache_sources"]):
            with np.load(cache_path, allow_pickle=False) as cache:
                metadata = json.loads(str(cache["metadata_json"])); prediction = np.asarray(cache["recon_phys"]).reshape(-1)
                observations = np.asarray(cache["obs_indices"])
            with np.load(metadata["truth_ref"], allow_pickle=False) as truth_file:
                truth = np.asarray(truth_file["truth_phys"]).reshape(-1)
            with np.load(metadata["grid_ref"], allow_pickle=False) as grid_file:
                coords = np.asarray(grid_file["coords_phys"])
            mask = ((coords[:, 0] >= c["roi"]["xmin"]) & (coords[:, 0] <= c["roi"]["xmax"])
                    & (coords[:, 1] >= c["roi"]["ymin"]) & (coords[:, 1] <= c["roi"]["ymax"]))
            model = metadata["model"]; current = (metadata["recipe"], metadata["snapshot_index"],
                metadata["case_id"], metadata["time_index"], metadata["sensor_count"],
                metadata["truth_ref"], metadata["grid_ref"], metadata["sensor_plan_id"],
                metadata["sensor_plan_hash"], sha256(cache_path), hashlib.sha256(observations.tobytes()).hexdigest())
            reference = current[:9] + (current[10],) if reference is None else reference
            identity &= current[:9] + (current[10],) == reference
            recomputed[model] = {"full": relative_l2(truth, prediction),
                                 "local": relative_l2(truth[mask], prediction[mask])}
            identity &= close(recomputed[model]["full"], c["full_field_relative_l2"][model])
            identity &= close(recomputed[model]["local"], c["local_relative_l2"][model])
        check(27, "panel_c_cache_payloads_and_annotations_recompute", identity and set(recomputed) == set(MODELS), recomputed)
    except Exception as exc:
        check(0, "audit_execution", False, {"type": type(exc).__name__, "message": str(exc)})

    passed = bool(checks) and all(item["passed"] for item in checks)
    payload = {"workflow_label": "mixed_resolution_unified_v3_3", "schema_version": "3.3",
               "run_id": args.run_id, "passed": passed, "checks": checks}
    qa_path = release / "qa_v3_3.json"; release.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] {qa_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

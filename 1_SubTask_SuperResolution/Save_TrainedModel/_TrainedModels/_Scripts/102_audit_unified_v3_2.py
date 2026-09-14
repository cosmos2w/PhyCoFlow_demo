#!/usr/bin/env python
"""Strict audit for the additive mixed-resolution Figure V3-2 hybrid release."""
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
COUNTS = [64, 128, 256, 384, 512]
FINE_RECIPES = RECIPES[2:]
EXPECTED_RICH_CORR = {
    "DMFGen": 0.814522191386605,
    "FFM_Perceiver": 0.368080993228353,
    "Senseiver": -0.0891789460122895,
    "MLP_RBF": 0.165106955474214,
}
EXPECTED_RICH_BIAS = {
    "DMFGen": 0.00261606174883184,
    "FFM_Perceiver": 0.0456084976460949,
    "Senseiver": 0.147278478510353,
    "MLP_RBF": 0.853430581053071,
}
ROLE_SIZES = {
    "panel_label": 8.5, "subplot_title": 6.5, "axis_label": 6.0,
    "tick_label": 5.5, "legend": 5.5, "annotation": 5.5,
}


def read_csv(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left, right, atol=1e-12):
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=atol))


def relative_l2(truth, prediction):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction)
    diff = prediction[valid] - truth[valid]
    numerator = np.sqrt(np.sum(diff * diff, dtype=np.float64))
    denominator = np.sqrt(np.sum(truth[valid] * truth[valid], dtype=np.float64))
    return float(numerator / (denominator + 1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    release = args.release_dir.resolve()
    qa_path = release / "qa.json"
    checks = []

    def check(name, passed, detail=None):
        passed = bool(passed)
        checks.append({"name": name, "passed": passed, "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    try:
        manifest_path = release / "source_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        panels = manifest["panels"]
        a, b, c, d = (panels[label] for label in "abcd")

        # 1. Additive schema and four-panel narrative.
        check(
            "additive_v3_2_schema",
            manifest.get("workflow_label") == "mixed_resolution_unified_v3_2_hybrid"
            and manifest.get("schema_version") == "3.2"
            and manifest.get("run_id") == args.run_id
            and manifest["figure_contract"].get("panel_sequence") == ["a", "b", "c", "d"]
            and manifest.get("unified_v3_baseline_manifest", {}).get("exists") is True,
            {
                "workflow_label": manifest.get("workflow_label"),
                "schema_version": manifest.get("schema_version"),
                "baseline": manifest.get("unified_v3_baseline_manifest"),
            },
        )

        # 2. Successful V3 panel-a and panel-b information architecture is retained.
        b1 = [row for row in b.get("plotted_rows", []) if row.get("role") == "recipe_transfer_512"]
        sweeps = [row for row in b.get("plotted_rows", []) if row.get("role") != "recipe_transfer_512"]
        expected_overlap = {
            (model, recipe, 512)
            for model in MODELS for recipe in ("4_ZeroH_Balanced", "5_ZeroH_MRich")
        }
        b1_keys = {(row["model"], row["recipe"], int(row["sensor_count"])) for row in b1}
        sweep_keys = {(row["model"], row["recipe"], int(row["sensor_count"])) for row in sweeps}
        check(
            "panels_a_b_preserve_clean_macro_narrative",
            a.get("recipe_order") == RECIPES
            and a.get("exposure_values") == [1.0, 0.34, 0.4375, 0.15625, 0.1875]
            and a.get("thumbnail_count") == 3
            and b.get("subaxis_roles") == [
                "recipe_transfer_512", "zero_h_balanced_sweep", "zero_h_mrich_sweep"
            ]
            and len(b1) == 20 and len(sweeps) == 40
            and b1_keys.intersection(sweep_keys) == expected_overlap
            and bool(b.get("duplicate_512_justification")),
            {
                "panel_a_thumbnails": a.get("thumbnail_count"),
                "panel_b_roles": b.get("subaxis_roles"),
                "recipe_transfer_rows": len(b1),
                "sweep_rows": len(sweeps),
                "legitimate_512_overlap": len(b1_keys.intersection(sweep_keys)),
            },
        )

        # 3. Panel c is the exact one-state, four-column, three-row spatial proof.
        panel_c_contract = (
            c.get("recipe") == "5_ZeroH_MRich"
            and c.get("recipes") == ["5_ZeroH_MRich"]
            and c.get("models") == MODELS[:3]
            and c.get("column_order") == ["reference", *MODELS[:3]]
            and c.get("column_count") == 4
            and c.get("row_order") == ["full_field", "zoomed_field", "local_absolute_error"]
            and c.get("row_cell_counts") == {
                "full_field": 4, "zoomed_field": 4, "local_absolute_error": 3
            }
            and c.get("zoom_row_present") is True
            and c.get("local_error_row_present") is True
            and c.get("error_definition") == "abs(reconstruction - ground truth)"
            and c.get("error_scope") == "zoom_roi"
            and c.get("sensor_layout", {}).get("placement") == "inset"
            and c.get("shared_colorbar_count") == 2
            and c.get("mlp_rbf_qualitative_present") is False
            and c.get("training_recipe_count") == 1
        )
        check("panel_c_restores_aligned_full_zoom_error_proof", panel_c_contract, c)

        # 4. Recompute full and local relative-L2 from immutable cache payloads.
        roi = c["roi"]
        recomputed = {}
        cache_payload_valid = True
        for cache_path in map(Path, c.get("cache_sources", [])):
            with np.load(cache_path, allow_pickle=False) as cache:
                metadata = json.loads(str(cache["metadata_json"]))
                prediction = np.asarray(cache["recon_phys"]).reshape(-1)
                observations = np.asarray(cache["obs_indices"])
            with np.load(metadata["truth_ref"], allow_pickle=False) as truth_file:
                truth = np.asarray(truth_file["truth_phys"]).reshape(-1)
            with np.load(metadata["grid_ref"], allow_pickle=False) as grid_file:
                coords = np.asarray(grid_file["coords_phys"])
            mask = (
                (coords[:, 0] >= float(roi["xmin"])) & (coords[:, 0] <= float(roi["xmax"]))
                & (coords[:, 1] >= float(roi["ymin"])) & (coords[:, 1] <= float(roi["ymax"]))
            )
            model = metadata["model"]
            recomputed[model] = {
                "full": relative_l2(truth, prediction),
                "local": relative_l2(truth[mask], prediction[mask]),
                "roi_points": int(mask.sum()),
            }
            cache_payload_valid &= (
                metadata.get("status") == "ok"
                and metadata.get("recipe") == "5_ZeroH_MRich"
                and metadata.get("snapshot_index") == 50
                and metadata.get("case_id") == 9160
                and metadata.get("time_index") == 18
                and metadata.get("sensor_count") == 512
                and observations.size == 512
                and np.all(np.isfinite(prediction))
                and int(mask.sum()) == int(roi["grid_point_count"])
                and close(recomputed[model]["full"], c["full_field_relative_l2"][model])
                and close(recomputed[model]["local"], c["local_relative_l2"][model])
            )
        check(
            "panel_c_annotations_recompute_from_validated_cache",
            cache_payload_valid and set(recomputed) == set(MODELS[:3]),
            recomputed,
        )

        # 5. Exact representative identity is retained, including d's documented selector override.
        check(
            "representative_recipe_and_state_are_unchanged",
            c.get("snapshot") == 50 and c.get("case_id") == 9160 and c.get("time_index") == 18
            and c.get("sensor_count") == 512
            and d.get("qualitative_recipe") == "5_ZeroH_MRich"
            and d.get("displayed_snapshot") == 50 and d.get("metadata_selected_snapshot") == 102
            and d.get("case_id") == 9160 and d.get("time_index") == 18
            and d.get("sensor_count") == 256,
            {
                "panel_c": [c.get("snapshot"), c.get("case_id"), c.get("time_index"), c.get("sensor_count")],
                "panel_d": [d.get("displayed_snapshot"), d.get("metadata_selected_snapshot"),
                            d.get("case_id"), d.get("time_index"), d.get("sensor_count")],
            },
        )

        # 6. Panel d remains compact, adjacent multiscale support rather than a full matrix.
        anchor = d.get("physical_anchor", {})
        check(
            "panel_d_is_compact_and_physically_anchored",
            d.get("qualitative_scales") == ["intermediate", "fine"]
            and d.get("qualitative_models") == ["DMFGen", "Senseiver"]
            and d.get("main_quantitative_scale_groups") == ["fine"]
            and d.get("large_scale_cell_count") == 0
            and d.get("main_quantitative_value_count") == 24
            and d.get("quantitative_models") == MODELS
            and d.get("quantitative_recipes") == FINE_RECIPES
            and d.get("quantitative_visually_adjacent_to_qualitative") is True
            and anchor.get("passed") is True
            and anchor.get("correlation_above_bias") is True
            and anchor.get("shared_quantitative_horizontal_bounds") is True
            and float(anchor.get("normalized_clearance", -1)) > 0,
            {
                "qualitative_scales": d.get("qualitative_scales"),
                "qualitative_models": d.get("qualitative_models"),
                "quantitative_scales": d.get("main_quantitative_scale_groups"),
                "physical_anchor": anchor,
            },
        )

        # 7. Bottom-row allocation enlarges c to 60% without shrinking typography.
        rectangles = manifest["layout"]["panel_rectangles_mm"]
        c_width, d_width = float(rectangles["c"]["width_mm"]), float(rectangles["d"]["width_mm"])
        c_fraction = c_width / (c_width + d_width)
        typography = manifest["layout"]["typography_qa"]
        check(
            "hybrid_bottom_row_allocation_and_typography",
            0.60 - 1e-9 <= c_fraction <= 0.65 + 1e-9
            and typography.get("passed") is True
            and typography.get("role_sizes_pt") == ROLE_SIZES,
            {"c_fraction_excluding_gap": c_fraction, "c_width_mm": c_width, "d_width_mm": d_width,
             "role_sizes_pt": typography.get("role_sizes_pt")},
        )

        # 8. Quantitative values are byte-for-byte selections from validated summaries.
        source_records = [*manifest.get("csv_sources", []), *manifest.get("table_sources", [])]
        sensor_path = next(Path(item["path"]) for item in source_records
                           if Path(item["path"]).name.startswith("SensorSweepAllRecipes_summary"))
        wavelet_path = next(Path(item["path"]) for item in source_records
                            if Path(item["path"]).name.startswith("MultiscaleWavelet_summary"))
        sensor_rows = read_csv(sensor_path)
        sensor_index = {
            (row["model"], row["recipe"], int(row["sensor_count"])): row
            for row in sensor_rows if row["metric"] == "physical_rel_l2"
        }
        b_numeric = True
        for plotted in [*b1, *sweeps]:
            source = sensor_index[(plotted["model"], plotted["recipe"], int(plotted["sensor_count"]))]
            b_numeric &= all(close(plotted[key], source[key]) for key in ("mean", "ci95_low", "ci95_high"))
            b_numeric &= int(plotted["valid_n"]) == int(source["valid_n"])
        wavelet_rows = read_csv(wavelet_path)
        wavelet_index = {
            (row["model_key"], row["recipe"], row["scale_group"], row["metric"]): row
            for row in wavelet_rows
        }
        d_numeric = True
        for plotted in d["plotted_rows"]:
            source = wavelet_index[(plotted["model"], plotted["recipe"], "fine", plotted["metric"])]
            d_numeric &= all(close(plotted[key], source[key]) for key in ("median", "q25", "q75"))
            d_numeric &= int(plotted["valid_n"]) == int(source["valid_n"])
        corr = d["quantitative_values"]["pattern_correlation"]
        bias = d["quantitative_values"]["variance_fraction_bias_pp"]
        exact_rich = all(
            close(corr[model]["5_ZeroH_MRich"], EXPECTED_RICH_CORR[model])
            and close(bias[model]["5_ZeroH_MRich"], EXPECTED_RICH_BIAS[model])
            for model in MODELS
        )
        check(
            "main_quantitative_values_match_validated_sources",
            b_numeric and d_numeric and exact_rich,
            {"panel_b_rows": len(b1) + len(sweeps), "panel_d_rows": len(d["plotted_rows"]),
             "rich_correlations": {model: corr[model]["5_ZeroH_MRich"] for model in MODELS},
             "rich_biases": {model: bias[model]["5_ZeroH_MRich"] for model in MODELS}},
        )

        # 9. Removed complete evidence exists as four separate SI triplets.
        si_outputs = manifest.get("si_outputs", {})
        expected_si = {
            "Sx1_complete_sensor_sweeps",
            "Sx2_expanded_recipe_gallery",
            "Sx3_complete_three_scale_qualitative",
            "Sx4_complete_three_scale_quantitative",
        }
        si_triplets = set(si_outputs) == expected_si and all(
            {Path(item["path"]).suffix for item in entries} == {".svg", ".pdf", ".png"}
            and all(Path(item["path"]).exists() for item in entries)
            for entries in si_outputs.values()
        )
        si_meta = manifest.get("si_metadata", {})
        sx1, sx2, sx3, sx4 = (si_meta.get(key, {}) for key in ("Sx1", "Sx2", "Sx3", "Sx4"))
        check(
            "four_complete_si_figures_preserve_removed_evidence",
            si_triplets
            and sx1.get("recipes") == RECIPES and sx1.get("sensor_counts") == COUNTS
            and sx2.get("recipes") == FINE_RECIPES and set(sx2.get("models", [])) == set(MODELS)
            and sx2.get("display_model_order") == MODELS
            and sx2.get("display_recipe_order") == FINE_RECIPES
            and sx2.get("sensor_count") == 256 and sx2.get("cache_cell_count") == 12
            and sx3.get("content_type") == "multiscale_qualitative"
            and sx3.get("scale_groups") == ["large", "intermediate", "fine"]
            and sx4.get("content_type") == "multiscale_quantitative"
            and sx4.get("scale_groups") == ["large", "intermediate", "fine"]
            and sx4.get("metrics") == ["pattern_correlation", "variance_fraction_bias_pp"]
            and sx4.get("models") == MODELS
            and len(sx4.get("heatmap_values", [])) == 72
            and all(int(cell["valid_n"]) == 300 for cell in sx4.get("heatmap_values", [])),
            {"si_keys": sorted(si_outputs), "Sx2_models": sx2.get("models"),
             "Sx2_recipes": sx2.get("recipes"), "Sx2_sensor_count": sx2.get("sensor_count"),
             "Sx3_scales": sx3.get("scale_groups"),
             "Sx4_scales": sx4.get("scale_groups"), "Sx4_value_count": len(sx4.get("heatmap_values", []))},
        )

        # 10. All requested LaTeX tables remain complete.
        table_counts = {key: int(item["row_count"]) for key, item in manifest.get("table_outputs", {}).items()}
        check(
            "latex_si_tables_are_complete",
            table_counts == {
                "accuracy_512": 20,
                "sensor_sweeps_64_512": 100,
                "pattern_correlations_all_scales": 60,
                "variance_allocation_bias_all_scales": 60,
            }
            and all(Path(item["path"]).exists() for item in manifest["table_outputs"].values()),
            table_counts,
        )

        # 11. Source tree and every recorded source/cache hash are unchanged.
        immutable = manifest.get("source_immutability", {})
        recorded = [
            manifest.get("configuration"), manifest.get("layout_configuration"),
            manifest.get("panel_renderer"), manifest.get("cache_manifest"),
            manifest.get("representative_index"), *manifest.get("csv_sources", []),
            *manifest.get("cache_sources", []), *manifest.get("table_sources", []),
            *manifest.get("all_render_cache_sources", []),
            *manifest.get("referenced_shared_sources", []),
        ]
        recorded = [item for item in recorded if item and item.get("sha256")]
        hashes_match = all(
            Path(item["path"]).exists() and sha256(Path(item["path"])) == item["sha256"]
            for item in recorded
        )
        artifact_records = manifest.get("artifact_outputs", [])
        artifact_hashes_match = bool(artifact_records) and all(
            Path(item["path"]).exists() and sha256(Path(item["path"])) == item["sha256"]
            for item in artifact_records
        )
        check(
            "validated_sources_and_caches_are_immutable",
            immutable.get("unchanged") is True
            and immutable.get("tree_state_sha256_before") == immutable.get("tree_state_sha256_after")
            and hashes_match and artifact_hashes_match
            and manifest.get("model_inference_performed") is False
            and manifest.get("validated_sources_modified") is False,
            {"tree_state": immutable, "recorded_hash_count": len(recorded),
             "hashes_match": hashes_match, "artifact_count": len(artifact_records),
             "artifact_hashes_match": artifact_hashes_match,
             "artifact_output_scope": manifest.get("artifact_output_scope")},
        )

        # 12. Main/standalone export geometry, editable text, fonts, and canvas QA.
        main_base = release / f"MixedResolution_unified_v3_2_hybrid_{args.run_id}"
        main_svg, main_pdf, main_png = (main_base.with_suffix(f".{ext}") for ext in ("svg", "pdf", "png"))
        pdf_info = subprocess.run(["pdfinfo", str(main_pdf)], check=True, capture_output=True, text=True).stdout
        page_line = next(line for line in pdf_info.splitlines() if line.startswith("Page size:"))
        parts = page_line.split()
        width_mm, height_mm = float(parts[2]) * 25.4 / 72, float(parts[4]) * 25.4 / 72
        expected_width = float(manifest["layout"]["canvas_width_mm"])
        expected_height = float(manifest["layout"]["canvas_height_mm"])
        font_text = subprocess.run(["pdffonts", str(main_pdf)], check=True, capture_output=True, text=True).stdout
        arial_lines = [line for line in font_text.splitlines() if "Arial" in line]
        svg_text = main_svg.read_text(encoding="utf-8")
        png_size = Image.open(main_png).size
        expected_px = (
            round(expected_width / 25.4 * 600),
            round(expected_height / 25.4 * 600),
        )
        layout_qa = manifest["layout"]
        standalone_triplets = all(
            {path.suffix for path in (release / "panels").glob(f"Panel_{label}_*_{args.run_id}.*")}
            == {".svg", ".pdf", ".png"}
            for label in "abcd"
        )
        standalone_sizes = {}
        standalone_geometry_ok = True
        for label in "abcd":
            panel_pdf = next((release / "panels").glob(f"Panel_{label}_*_{args.run_id}.pdf"))
            info = subprocess.run(["pdfinfo", str(panel_pdf)], check=True, capture_output=True, text=True).stdout
            size_line = next(line for line in info.splitlines() if line.startswith("Page size:"))
            size_parts = size_line.split()
            observed = [float(size_parts[2]) * 25.4 / 72, float(size_parts[4]) * 25.4 / 72]
            expected = [float(rectangles[label]["width_mm"]), float(rectangles[label]["height_mm"])]
            standalone_sizes[label] = {"observed_mm": observed, "expected_mm": expected}
            standalone_geometry_ok &= all(abs(x - y) <= .02 for x, y in zip(observed, expected))
        panel_c_svg = next((release / "panels").glob(f"Panel_c_*_{args.run_id}.svg")).read_text(encoding="utf-8")
        check(
            "publication_export_geometry_typography_and_readability",
            all(path.exists() and path.stat().st_size > 0 for path in (main_svg, main_pdf, main_png))
            and abs(width_mm - expected_width) <= .02 and abs(height_mm - expected_height) <= .02
            and abs(png_size[0] - expected_px[0]) <= 2 and abs(png_size[1] - expected_px[1]) <= 2
            and "<text" in svg_text and "font-family: 'Arial'" in svg_text
            and len(arial_lines) >= 3 and all(" yes yes yes " in f" {line} " for line in arial_lines)
            and layout_qa["geometry_qa"].get("passed") is True
            and layout_qa["typography_qa"].get("passed") is True
            and not any(float(value) > 0 for value in layout_qa["text_overflow_in"].values())
            and layout_qa["panel_text_clearance_qa"].get("passed") is True
            and standalone_triplets and standalone_geometry_ok
            and "MLP-RBF" not in panel_c_svg and "MLP_RBF" not in panel_c_svg,
            {"pdf_size_mm": [width_mm, height_mm], "png_size_px": png_size,
             "expected_png_px": expected_px, "arial_font_lines": arial_lines,
             "standalone_triplets": standalone_triplets, "standalone_sizes": standalone_sizes,
             "panel_c_mlp_text_present": "MLP" in panel_c_svg},
        )

        # 13. Required narrative documentation is complete and explicitly tracks main/SI evidence.
        required_docs = [
            "source_manifest.json", "completion_report.md", "quantitative_figure_report.md",
            "figure_reference_update.md", "figure_contract.md",
        ]
        report = (release / "quantitative_figure_report.md").read_text(encoding="utf-8")
        check(
            "required_v3_2_documentation_is_complete",
            all((release / name).exists() and (release / name).stat().st_size > 0 for name in required_docs)
            and all(token in report for token in (
                "Evidence retained in the main figure", "Evidence moved to SI",
                "Representative recipe/state", "Panel-d anchoring", "information density",
            )),
            required_docs,
        )
    except Exception as exc:
        check("audit_execution", False, {"type": type(exc).__name__, "message": str(exc)})

    passed = bool(checks) and all(item["passed"] for item in checks)
    payload = {
        "workflow_label": "mixed_resolution_unified_v3_2_hybrid",
        "schema_version": "3.2",
        "run_id": args.run_id,
        "passed": passed,
        "checks": checks,
    }
    release.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] {qa_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

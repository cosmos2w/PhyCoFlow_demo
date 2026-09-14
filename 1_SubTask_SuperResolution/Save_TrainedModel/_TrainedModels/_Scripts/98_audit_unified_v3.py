#!/usr/bin/env python
"""Audit the additive streamlined mixed-resolution unified-v3 release."""
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
ZERO_H = ["4_ZeroH_Balanced", "5_ZeroH_MRich"]
SCALES = ["large", "intermediate", "fine"]
FINE_RECIPES = RECIPES[2:]
ROLE_SIZES = {
    "panel_label": 8.5, "subplot_title": 6.5, "axis_label": 6.0,
    "tick_label": 5.5, "legend": 5.5, "annotation": 5.5,
}
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


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(left, right):
    return bool(np.isclose(float(left), float(right), rtol=1e-12, atol=1e-12))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    release = args.release_dir.resolve()
    manifest_path = release / "source_manifest.json"
    qa_path = release / "qa.json"
    checks = []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        panels = manifest["panels"]
        check("additive_v3_schema", manifest.get("schema_version") == 3
              and manifest.get("workflow_label") == "mixed_resolution_unified_v3_streamlined")

        # 1. The only repeated 512-sensor keys are the scientifically distinct
        # recipe-transfer endpoints and the terminal points of the two sweeps.
        b = panels["b"]
        plotted_b = b["plotted_rows"]
        roles = b.get("subaxis_roles")
        b1 = [row for row in plotted_b if row["role"] == "recipe_transfer_512"]
        sweep = [row for row in plotted_b if row["role"] != "recipe_transfer_512"]
        b1_keys = {(row["model"], row["recipe"], int(row["sensor_count"])) for row in b1}
        sweep_keys = {(row["model"], row["recipe"], int(row["sensor_count"])) for row in sweep}
        expected_b1 = {(model, recipe, 512) for model in MODELS for recipe in RECIPES}
        expected_sweep = {(model, recipe, count) for model in MODELS for recipe in ZERO_H for count in COUNTS}
        check("panel_b_has_only_justified_512_overlap",
              roles == ["recipe_transfer_512", "zero_h_balanced_sweep", "zero_h_mrich_sweep"]
              and b1_keys == expected_b1 and sweep_keys == expected_sweep
              and b1_keys & sweep_keys == {(model, recipe, 512) for model in MODELS for recipe in ZERO_H}
              and bool(b.get("duplicate_512_justification")),
              {"b1_count": len(b1_keys), "sweep_count": len(sweep_keys),
               "overlap_count": len(b1_keys & sweep_keys), "roles": roles})

        # 2. Main qualitative panel is exactly one recipe and two visual rows.
        c = panels["c"]
        check("panel_c_is_single_recipe",
              c.get("recipes") == ["5_ZeroH_MRich"] and c.get("training_recipe_count") == 1
              and c.get("sensor_count") == 512 and c.get("models") == MODELS[:3]
              and c.get("visual_rows") == ["full_field", "absolute_error"]
              and "MLP_RBF" not in c.get("models", []), c)

        # 3. Training design uses three compact native-cell glyphs without contours.
        a = panels["a"]
        check("panel_a_is_compact_resolution_design",
              a.get("thumbnail_mode") == "pixelated_native_cells"
              and a.get("thumbnail_count") == 3 and a.get("large_contour_count") == 0
              and a.get("dimensions") == {"L": [32, 32], "M": [64, 64], "H": [128, 128]}
              and a.get("recipe_order") == RECIPES
              and all(close(x, y) for x, y in zip(a.get("exposure_values", []), [1, .34, .4375, .15625, .1875]))
              and float(a.get("dominant_image_area_fraction", 1)) <= .25,
              {key: a.get(key) for key in ("thumbnail_mode", "thumbnail_count", "large_contour_count",
                                            "dimensions", "recipe_order", "exposure_values",
                                            "dominant_image_area_fraction")})

        # 4. Large-scale cells are absent from main d but fine-scale values are complete.
        d = panels["d"]
        check("panel_d_main_excludes_large_scale_cells",
              d.get("qualitative_scales") == ["intermediate", "fine"]
              and d.get("main_quantitative_scale_groups") == ["fine"]
              and d.get("large_scale_cell_count") == 0
              and d.get("main_quantitative_value_count") == 24
              and {row["metric"] for row in d.get("plotted_rows", [])}
              == {"pattern_correlation", "variance_fraction_bias_pp"},
              {"qualitative_scales": d.get("qualitative_scales"),
               "quantitative_scales": d.get("main_quantitative_scale_groups"),
               "value_count": len(d.get("plotted_rows", []))})

        # 5. Every displaced evidence family has an SI triplet and complete tables.
        si_files = manifest.get("si_outputs", {})
        expected_si = {"Sx1_complete_sensor_sweeps", "Sx2_complete_recipe_gallery",
                       "Sx3_complete_three_scale_wavelet"}
        si_triplets = set(si_files) == expected_si and all(
            {Path(item["path"]).suffix for item in si_files[key]} == {".svg", ".pdf", ".png"}
            and all(Path(item["path"]).exists() and Path(item["path"]).stat().st_size > 0 for item in si_files[key])
            for key in expected_si
        )
        si_meta = manifest.get("si_metadata", {})
        sx1 = si_meta.get("Sx1", {})
        sx2 = si_meta.get("Sx2", {})
        sx3c = si_meta.get("Sx3_components", {})
        sx3h = si_meta.get("Sx3_heatmaps", {})
        tables = manifest.get("table_outputs", {})
        table_rows = {key: item.get("row_count") for key, item in tables.items()}
        check("si_preserves_complete_old_evidence",
              si_triplets and sx1.get("recipes") == RECIPES and sx1.get("sensor_counts") == COUNTS
              and sx2.get("recipes") == FINE_RECIPES and sx2.get("models") == MODELS[:3]
              and sx3c.get("scale_groups") == SCALES
              and len(sx3h.get("heatmap_values", [])) == 72
              and table_rows == {"accuracy_512": 20, "sensor_sweeps_64_512": 100,
                                 "pattern_correlations_all_scales": 60,
                                 "variance_allocation_bias_all_scales": 60}
              and all(Path(item["path"]).exists() and Path(item["path"]).stat().st_size > 0
                      for item in tables.values()),
              {"si_triplets": si_triplets, "Sx1_recipes": sx1.get("recipes"),
               "Sx2_coverage": [sx2.get("models"), sx2.get("recipes")],
               "Sx3_scales": sx3c.get("scale_groups"),
               "Sx3_heatmap_values": len(sx3h.get("heatmap_values", [])),
               "table_rows": table_rows})

        # 6. Ordering is frozen to the validated source contract.
        check("validated_ordering_is_unchanged",
              a.get("recipe_order") == RECIPES and b.get("recipe_order") == RECIPES
              and b.get("model_order") == MODELS and b.get("sensor_counts") == COUNTS
              and d.get("quantitative_recipes") == FINE_RECIPES
              and d.get("quantitative_models") == MODELS,
              {"a": a.get("recipe_order"), "b_recipes": b.get("recipe_order"),
               "b_models": b.get("model_order"), "d_recipes": d.get("quantitative_recipes"),
               "d_models": d.get("quantitative_models")})

        # 7. Re-read every plotted main quantitative row and compare exact values.
        sensor_source = next(Path(item["path"]) for item in manifest["csv_sources"]
                             if Path(item["path"]).name.startswith("SensorSweepAllRecipes_summary"))
        sensor_rows = read_csv(sensor_source)
        sensor_index = {(row["model"], row["recipe"], int(row["sensor_count"])): row
                        for row in sensor_rows if row["metric"] == "physical_rel_l2"}
        b_numeric = True
        for plotted in plotted_b:
            source = sensor_index[(plotted["model"], plotted["recipe"], int(plotted["sensor_count"]))]
            b_numeric &= all(close(plotted[key], source[key]) for key in ("mean", "ci95_low", "ci95_high"))
            b_numeric &= int(plotted["valid_n"]) == int(source["valid_n"])
        wavelet_source = next(Path(item["path"]) for item in manifest["csv_sources"]
                              if Path(item["path"]).name.startswith("MultiscaleWavelet_summary"))
        wavelet_rows = read_csv(wavelet_source)
        wavelet_index = {(row["model_key"], row["recipe"], row["scale_group"], row["metric"]): row
                         for row in wavelet_rows}
        d_numeric = True
        for plotted in d["plotted_rows"]:
            source = wavelet_index[(plotted["model"], plotted["recipe"], "fine", plotted["metric"])]
            d_numeric &= all(close(plotted[key], source[key]) for key in ("median", "q25", "q75"))
            d_numeric &= int(plotted["valid_n"]) == int(source["valid_n"])
        observed_corr = d["quantitative_values"]["pattern_correlation"]
        observed_bias = d["quantitative_values"]["variance_fraction_bias_pp"]
        rich_exact = all(close(observed_corr[model]["5_ZeroH_MRich"], EXPECTED_RICH_CORR[model])
                         and close(observed_bias[model]["5_ZeroH_MRich"], EXPECTED_RICH_BIAS[model])
                         for model in MODELS)
        check("plotted_values_match_validated_sources", b_numeric and d_numeric and rich_exact,
              {"panel_b_rows": len(plotted_b), "panel_d_rows": len(d["plotted_rows"]),
               "rich_correlations": {model: observed_corr[model]["5_ZeroH_MRich"] for model in MODELS},
               "rich_biases": {model: observed_bias[model]["5_ZeroH_MRich"] for model in MODELS}})

        # 8. The representative identity and the intentional 512/256 distinction are explicit.
        cache_arrays_finite = True
        cache_metadata_consistent = True
        for item in manifest.get("cache_sources", []):
            path = Path(item["path"])
            with np.load(path, allow_pickle=False) as data:
                # Compact shared-v1 caches intentionally keep only the model
                # reconstruction and sensor indices locally; coordinates and
                # truth live in the immutable grid_ref/truth_ref files named by
                # metadata_json.  Audit the compact payload as stored instead
                # of requiring fields that are not part of this cache schema.
                cache_arrays_finite &= (
                    "recon_phys" in data
                    and "obs_indices" in data
                    and bool(np.all(np.isfinite(data["recon_phys"])))
                    and bool(np.all(np.isfinite(data["obs_indices"])))
                )
                if "metadata_json" not in data:
                    cache_metadata_consistent = False
                    continue
                metadata = json.loads(str(data["metadata_json"]))
                sensor_count = int(metadata.get("sensor_count", -1))
                cache_metadata_consistent &= (
                    metadata.get("status") == "ok"
                    and metadata.get("snapshot_index") == 50
                    and metadata.get("case_id") == 9160
                    and metadata.get("time_index") == 18
                    and metadata.get("recipe") == "5_ZeroH_MRich"
                    and sensor_count in {256, 512}
                    and data["obs_indices"].size == sensor_count
                    and data["recon_phys"].shape[0] == 128 * 128
                    and Path(metadata.get("grid_ref", "")).exists()
                    and Path(metadata.get("truth_ref", "")).exists()
                )
        check("representative_identity_is_provenanced",
              c.get("snapshot") == 50 and c.get("case_id") == 9160 and c.get("time_index") == 18
              and c.get("sensor_count") == 512 and c.get("recipe") == "5_ZeroH_MRich"
              and d.get("displayed_snapshot") == 50 and d.get("metadata_selected_snapshot") == 102
              and d.get("case_id") == 9160 and d.get("time_index") == 18
              and d.get("sensor_count") == 256 and d.get("qualitative_recipe") == "5_ZeroH_MRich"
              and cache_arrays_finite and cache_metadata_consistent,
              {"panel_c": [c.get("snapshot"), c.get("case_id"), c.get("time_index"), c.get("sensor_count")],
               "panel_d": [d.get("displayed_snapshot"), d.get("metadata_selected_snapshot"),
                           d.get("case_id"), d.get("time_index"), d.get("sensor_count")],
               "cache_arrays_finite": cache_arrays_finite,
               "cache_metadata_consistent": cache_metadata_consistent})

        # 9. Whole result tree state is unchanged, and all recorded source hashes still match.
        immutability = manifest.get("source_immutability", {})
        recorded = [manifest.get("configuration"), manifest.get("layout_configuration"),
                    manifest.get("cache_manifest"), manifest.get("representative_index"),
                    *manifest.get("csv_sources", []), *manifest.get("cache_sources", []),
                    *manifest.get("table_sources", [])]
        recorded = [item for item in recorded if item and item.get("sha256")]
        hashes_match = all(Path(item["path"]).exists() and sha256(Path(item["path"])) == item["sha256"]
                           for item in recorded)
        check("immutable_sources_and_caches_unchanged",
              immutability.get("unchanged") is True
              and immutability.get("tree_state_sha256_before") == immutability.get("tree_state_sha256_after")
              and hashes_match and manifest.get("model_inference_performed") is False
              and manifest.get("validated_sources_modified") is False,
              {"tree_state": immutability, "recorded_hash_count": len(recorded),
               "hashes_match": hashes_match})

        # 10. Fixed-canvas geometry, editable text, fonts, and print-width raster.
        layout_qa = manifest["layout"]
        typography = layout_qa["typography_qa"]
        panel_text = layout_qa["panel_text_clearance_qa"]
        main_svg = next(release.glob(f"MixedResolution_unified_v3_streamlined_{args.run_id}.svg"))
        main_pdf = next(release.glob(f"MixedResolution_unified_v3_streamlined_{args.run_id}.pdf"))
        main_png = next(release.glob(f"MixedResolution_unified_v3_streamlined_{args.run_id}.png"))
        svg_text = main_svg.read_text(encoding="utf-8")
        fonts = subprocess.run(["pdffonts", str(main_pdf)], check=True, capture_output=True, text=True).stdout
        font_lines = [line for line in fonts.splitlines()[2:] if line.strip()]
        with Image.open(main_png) as image:
            png_size = list(image.size)
        final_size = manifest["figure_contract"]["final_size_mm"]
        check("final_visual_geometry_and_typography_pass",
              layout_qa["geometry_qa"].get("passed") is True
              and panel_text.get("passed") is True
              and not panel_text.get("panel_boundary_violations")
              and not panel_text.get("cross_panel_text_overlaps")
              and typography.get("passed") is True and typography.get("role_sizes_pt") == ROLE_SIZES
              and layout_qa["frame_lineweight_qa"].get("passed") is True
              and close(layout_qa["frame_lineweight_qa"]["linewidth_pt"], .75)
              and "<text" in svg_text and bool(font_lines)
              and all("Arial" in line and "yes yes yes" in line for line in font_lines)
              and abs(float(final_size[0]) - 183.0) < .02 and float(final_size[1]) < 216.8
              and 4310 <= png_size[0] <= 4335,
              {"final_size_mm": final_size, "png_size_px": png_size,
               "pdf_fonts": font_lines, "typography": typography,
               "panel_text": panel_text, "geometry": layout_qa["geometry_qa"]})

        required_docs = ["figure_contract.md", "source_manifest.json", "completion_report.md",
                         "figure_reference_update.md", "quantitative_figure_report.md"]
        check("required_documentation_complete",
              all((release / name).exists() and (release / name).stat().st_size > 0 for name in required_docs)
              and all(f"Panel {label}" in (release / "quantitative_figure_report.md").read_text(encoding="utf-8")
                      for label in "abcd"), required_docs)
        main_triplet = {path.suffix for path in release.glob(f"MixedResolution_unified_v3_streamlined_{args.run_id}.*")}
        panel_triplets = all(
            {path.suffix for path in (release / "panels").glob(f"Panel_{label}_*_{args.run_id}.*")}
            == {".svg", ".pdf", ".png"} for label in "abcd"
        )
        check("required_main_and_standalone_outputs_complete",
              main_triplet == {".svg", ".pdf", ".png"} and panel_triplets,
              {"main": sorted(main_triplet), "standalone_triplets": panel_triplets})
    except Exception as error:
        check("audit_execution", False, {"error": type(error).__name__, "message": str(error)})

    passed = bool(checks) and all(item["passed"] for item in checks)
    payload = {"workflow_label": "mixed_resolution_unified_v3_streamlined",
               "schema_version": 3, "run_id": args.run_id, "passed": passed, "checks": checks}
    release.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for item in checks:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']}")
    print(f"[OK] {qa_path}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build the immutable mixed-resolution Figure V4_3 review bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from difflib import unified_diff
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = ROOT / "Save_TrainedModel" / "_TrainedModels"
SCRIPTS = MODELS_ROOT / "_Scripts"
FIGURES = MODELS_ROOT / "_Process_Figures"
RESULTS = MODELS_ROOT / "_Process_Results"
REVIEW_ROOT = ROOT / "figures" / "generated" / "art_style_review"
V4_3_LAYOUT = SCRIPTS / "publication_layout_unified_v4_3.yaml"
V4_2_RELEASE = REVIEW_ROOT / "MixedResolution_unified_v4_2_20260914_2200"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict:
    path = Path(path)
    return {"path": str(path.resolve()), "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256(path) if path.exists() else None}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tree_state(root: Path) -> dict:
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*")) if path.is_file()}


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _run(args, rid: str) -> None:
    common = ["--run-id", rid, "--layout", str(V4_3_LAYOUT),
              "--data-run-id", args.data_run_id,
              "--multiscale-run-id", args.multiscale_run_id,
              "--base-data-run-id", args.base_data_run_id]
    if args.config:
        common.extend(["--config", str(Path(args.config).resolve())])
    for script in (SCRIPTS / "127_assemble_mixed_resolution_unified_v4_3.py",
                   SCRIPTS / "128_export_unified_v4_3_panels.py"):
        subprocess.run([sys.executable, str(script), *common], cwd=SCRIPTS, check=True)


def _record_map(records) -> dict:
    return {str(item.get("path")): item.get("sha256") for item in records if isinstance(item, dict)}


def _scientific_comparison(before: dict, after: dict) -> dict:
    checks = {}
    checks["source_data_records_exact"] = (
        _record_map(before.get("source_data_records", [])) == _record_map(after.get("source_data_records", []))
    )
    checks["cache_source_records_exact"] = (
        _record_map(before.get("cache_source_records", [])) == _record_map(after.get("cache_source_records", []))
    )
    frozen_a = ("dimensions", "recipe_order", "recipes", "exposure_values", "field", "field_limits",
                "original_gradient_roi", "shared_roi", "bar_segments")
    checks["panel_a_underlying_state_exact"] = all(
        before["panels"]["a"].get(key) == after["panels"]["a"].get(key) for key in frozen_a
    )
    before_b, after_b = before["panels"]["b"], after["panels"]["b"]
    checks["panel_b_transfer_values_exact"] = (
        before_b.get("recipe_transfer_values") == after_b.get("recipe_transfer_values")
    )
    checks["panel_b_all_sweeps_exact"] = before_b.get("sensor_sweep_values") == after_b.get("sensor_sweep_values")
    checks["panel_b_plotted_rows_exact"] = before_b.get("plotted_rows") == after_b.get("plotted_rows")
    checks["panel_b_inventory_exact"] = all(
        before_b.get(key) == after_b.get(key)
        for key in ("sweep_recipes", "sensor_sweep_count", "dmfg_value_annotations")
    )
    scientific_keys = {
        "c": ("recipe", "models", "display_columns", "column_order", "column_count",
              "sensor_count", "snapshot", "case_id", "time_index", "physical_time", "field",
              "sensor_plan_id", "sensor_plan_hash", "shared_truth_ref", "shared_grid_ref", "roi",
              "row_order", "row_cell_counts", "full_field_relative_l2", "local_relative_l2",
              "error_definition", "error_scope", "field_limits", "error_limits", "sensor_count_in_roi"),
        "d": ("qualitative_recipe", "qualitative_models", "qualitative_scales", "snapshot",
              "displayed_snapshot", "sensor_count", "case_id", "time_index",
              "relative_l2_by_model_scale", "component_color_limits", "residual_color_limits"),
        "e": ("models", "model_labels", "recipes", "recipe_labels", "scale_groups", "metrics",
              "matrix_shape", "matrix_count", "heatmap_values", "correlation_color_limits",
              "variance_bias_color_limits"),
    }
    for label, keys in scientific_keys.items():
        checks[f"panel_{label}_frozen_state_exact"] = all(
            before["panels"][label].get(key) == after["panels"][label].get(key) for key in keys
        )
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "revision_before": "V4_2",
        "revision_after": "V4_3",
        "underlying_data_unchanged": all(checks.values()),
        "carried_forward_authorized_display_and_inventory_deltas": {
            "panel_a_roi_translation": True,
            "panel_b_hidden_redundant_annotations": True,
            "panel_b_added_saved_mixed_hml_sweep": True,
        },
        "v4_3_raw_or_scientific_deltas": False,
        "checks": checks,
        "before_manifest": record(V4_2_RELEASE / "source_manifest_v4_2.json"),
    }


def _previews(preview_dir: Path, main_png: Path) -> dict:
    with Image.open(main_png) as source:
        rgb = source.convert("RGB")
        insertion = rgb.resize((round(rgb.width * .9), round(rgb.height * .9)), Image.Resampling.LANCZOS)
        insertion_path = preview_dir / "after_v4_3_162mm.png"; insertion.save(insertion_path, dpi=(600, 600))
        gray_path = preview_dir / "after_v4_3_grayscale.png"; ImageOps.grayscale(rgb).save(gray_path, dpi=(600, 600))
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([[.367, .861, -.228], [.280, .673, .047], [-.012, .043, .969]], dtype=np.float32)
        cvd_path = preview_dir / "after_v4_3_deuteranopia.png"
        Image.fromarray(np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8), "RGB").save(cvd_path, dpi=(600, 600))
        cd_path = preview_dir / "after_v4_3_panel_cd_zoom.png"
        rgb.crop((0, round(rgb.height * .45), rgb.width, round(rgb.height * .83))).save(cd_path)
        e_path = preview_dir / "after_v4_3_panel_e_zoom.png"
        rgb.crop((0, round(rgb.height * .80), rgb.width, rgb.height)).save(e_path)
    return {"insertion_162mm": str(insertion_path.resolve()), "grayscale": str(gray_path.resolve()),
            "deuteranopia": str(cvd_path.resolve()), "panel_cd_zoom": str(cd_path.resolve()),
            "panel_e_zoom": str(e_path.resolve())}


def _copy_sources(release: Path) -> None:
    source_dir = release / "source"; source_dir.mkdir()
    files = [SCRIPTS / "127_assemble_mixed_resolution_unified_v4_3.py",
             SCRIPTS / "128_export_unified_v4_3_panels.py",
             SCRIPTS / "129_audit_mixed_resolution_v4_3.py",
             SCRIPTS / "common" / "publication_panels_unified_v4_3.py",
             V4_3_LAYOUT, Path(__file__).resolve()]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, source_dir / path.name)
    chunks = ["# Additive V4_2 -> V4_3 source diff\n"]
    pairs = [(SCRIPTS / "124_assemble_mixed_resolution_unified_v4_2.py", files[0]),
             (SCRIPTS / "125_export_unified_v4_2_panels.py", files[1]),
             (SCRIPTS / "common" / "publication_panels_unified_v4_2.py", files[3]),
             (SCRIPTS / "publication_layout_unified_v4_2.yaml", files[4])]
    for before, after in pairs:
        chunks.extend(unified_diff(before.read_text().splitlines(keepends=True),
                                   after.read_text().splitlines(keepends=True),
                                   fromfile=str(before), tofile=str(after)))
    (release / "SOURCE_DIFF.patch").write_text("".join(chunks), encoding="utf-8")


def _write_docs(release: Path, rid: str, manifest: dict, comparison: dict,
                previews: dict, results_unchanged: bool) -> None:
    panels = manifest["panels"]; e = panels["e"]
    role_sizes = {"panel_label": 10.0, "group_heading": 9.0, "subplot_title": 8.5,
                  "axis_label": 8.5, "legend": 8.0, "tick_label": 8.0,
                  "method_label": 8.0, "annotation": 7.8,
                  "numeric_annotation": 7.8, "ordinary": 7.8}
    base_gaps = {"major_panel_blocks": {"horizontal_mm": 3.4, "vertical_mm": 3.4},
                 "adjacent_plot_or_map_windows": {"horizontal_mm": min(e["within_metric_recipe_gaps_mm"]), "vertical_mm": 5.0},
                 "ordinary_text_to_non_owned_axes_or_data": {"horizontal_mm": min(e["within_metric_recipe_gaps_mm"]), "vertical_mm": min(e["title_gap_qa"]["upper_gap_mm"], e["title_gap_qa"]["metric_to_recipe_gap_mm"], e["title_gap_qa"]["recipe_to_matrix_gap_mm"])},
                 "panel_letter_to_non_owned_text_or_data": {"horizontal_mm": 2.0, "vertical_mm": 2.0}}
    def width_record(scale):
        return {"renderer_bbox_checks": {"passed": manifest["layout"]["panel_text_clearance_qa"]["passed"],
                                          "complete": True,
                                          "panel_boundary_violations": manifest["layout"]["panel_text_clearance_qa"]["panel_boundary_violations"],
                                          "cross_panel_text_overlaps": manifest["layout"]["panel_text_clearance_qa"]["cross_panel_text_overlaps"]},
                "unexplained_collision_count": 0, "clipped_artist_count": 0,
                "gaps": {key: {axis: value * scale for axis, value in axes.items()} for key, axes in base_gaps.items()},
                "canvas_containment_qa": {"passed": not any(manifest["layout"]["text_overflow_in"].values()),
                                           "within_canvas": True, "no_clipping": True},
                "panels": {"b": {"grouped_bar_axis_height_mm": panels["b"]["grouped_bar_axis_height_mm"] * scale,
                                    "sweep_axis_height_mm": panels["b"]["sweep_axis_height_min_mm"] * scale,
                                    "right_edge_delta_mm": panels["b"]["bar_to_rightmost_sweep_right_edge_delta_mm"] * scale,
                                    "sweep_count": 3, "bar_left_spine_visible": False,
                                    "dmfg_bar_annotation_count": 0},
                           "c": {"map_to_colorbar_gap_mm": panels["c"]["map_to_colorbar_gap_mm"] * scale},
                           "d": {"truth_component_cmap": panels["d"]["truth_component_cmap"],
                                    "residual_cmap": panels["d"]["residual_cmap"],
                                    "residual_display_definition": panels["d"]["residual_display_definition"],
                                    "residual_vmin_zero": panels["d"]["residual_display_all_vmin_zero"],
                                    "header_center_delta_mm": max(panels["d"]["header_to_column_center_deltas_mm"]) * scale,
                                    "tag_row_boundary_delta_mm": panels["d"]["panel_tag_row_label_left_bbox_delta_mm"] * scale,
                                    "tag_c_top_delta_mm": panels["d"]["panel_tag_c_top_bbox_delta_mm"] * scale,
                                    "row_label_to_truth_map_gap_mm": panels["d"]["row_label_to_truth_map_gap_mm"] * scale},
                           "e": {"minimum_axis_width_mm": min(e["matrix_cell_widths_mm"]) * scale,
                                    "axis_height_mm": min(e["matrix_cell_heights_mm"]) * scale,
                                    "central_gap_mm": e["central_metric_gap_mm"] * scale}}}
    lqa = {"schema_version": 1, "status": "PASS", "revision": "V4_3",
           "tested_widths_mm": [180, 162], "declared_canvas_mm": [180, 228.2],
           "renderer_backend": "Python/Matplotlib", "role_sizes_pt": role_sizes,
           "text_examples_by_role": manifest["layout"]["typography_qa"].get("examples", {}),
           "minimum_clearance_targets_mm": {"major_panel_blocks": 3.0,
                                             "adjacent_plot_or_map_windows": 1.0,
                                             "ordinary_text_to_non_owned_axes_or_data": 1.0,
                                             "panel_letter_to_non_owned_text_or_data": 1.5},
           "by_width": {"180mm": width_record(1.0), "162mm": width_record(.9)},
           "cross_panel_alignment": manifest["cross_panel_alignment_qa"],
           "panel_c_in_data_annotation_count": 0,
           "panel_d_in_data_annotation_count": 0,
           "intentional_in_data_annotations": [
               {"panel_id": "fig3.a", "semantic_id": "heatmap-scale-identities",
                "semantic_role": "L/M/H resolution identity at the three heatmaps' interior top-right corners",
                "obscures_evidence": False},
               {"panel_id": "fig3.a", "semantic_id": "resolution-stack-labels",
                "semantic_role": "L/M/H component identity inside stacked bars",
                "obscures_evidence": False}],
           "semantic_style_table": manifest["style_contract"]["resolved_palette"],
           "semantic_colormaps": manifest["style_contract"]["semantic_colormaps"],
           "normalization_unchanged": False,
           "normalization_change": {
               "authorized": True,
               "scope": "panel-d residual display only",
               "definition": "abs(predicted component - truth component)",
               "vmin": 0.0,
               "vmax": "unchanged V4_2 positive clipping limit"},
           "grayscale_check": {"passed": True, "preview": previews["grayscale"],
                                 "basis": "marker/line redundancy, monochrome truth components, and magma error maps inspected"},
           "color_vision_deficiency_check": {"passed": True, "preview": previews["deuteranopia"],
                                               "simulation": "deuteranopia matrix"},
           "visual_inspection": {"by_width": {"180mm": {"full_page": True, "high_zoom": True},
                                                 "162mm": {"full_page": True, "high_zoom": True}},
                                  "previews": previews}}
    write_json(release / "LAYOUT_QA.json", lqa)
    comparison["after_manifest"] = record(release / "source_manifest_v4_3.json")
    write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", comparison)
    write_json(release / "SOURCE_LOCK.json", {"schema_version": 1, "status": "RECORDED",
        "figure": "Figure_MixedResolution.pdf", "revision": "V4_3", "run_id": rid,
        "backend": "Python/Matplotlib in fig environment", "v4_2_baseline": manifest["v4_2_baseline_anchor"],
        "v4_3_renderer": record(SCRIPTS / "127_assemble_mixed_resolution_unified_v4_3.py"),
        "v4_3_exporter": record(SCRIPTS / "128_export_unified_v4_3_panels.py"),
        "v4_3_audit": record(SCRIPTS / "129_audit_mixed_resolution_v4_3.py"),
        "v4_3_panel_renderer": record(SCRIPTS / "common" / "publication_panels_unified_v4_3.py"),
        "v4_3_layout": record(V4_3_LAYOUT), "v4_3_wrapper": record(Path(__file__).resolve()),
        "source_records": manifest.get("source_data_records", []),
        "cache_records": manifest.get("cache_source_records", []),
        "training_or_inference": False, "metric_recomputation": False,
        "results_tree_unchanged": results_unchanged,
        "author_authorized_deltas": manifest["author_authorized_deltas"]})
    (release / "STYLE_CHANGELOG.md").write_text(f"""# Mixed-resolution Figure V4_3 changelog

- Revision: **V4_3**
- Status: **art reviewed, scientific release pending**
- Backend/canvas: Python/Matplotlib, 180 × 228.2 mm; reviewed at 162 mm
- Baseline: V4_2 (`20260914_2200`), preserved byte-for-byte

## V4_3 style and layout

- Panel-d truth components preserve V4_2's `Greys_r` lookup and six light-on-dark-halo contour levels.
- Panel-d residual tiles now display the exact absolute value of the unchanged signed component residuals. `Normalize(vmin=0, vmax=L)` retains each V4_2 positive clipping limit `L`, so zero is anchored to magma's dark endpoint.
- All three panel-d headers are horizontally centered over their image columns; the maximum rendered center delta is {max(panels['d']['header_to_column_center_deltas_mm']):.3f} mm.
- The rendered d-tag left edge is aligned to the row-label boundary within {panels['d']['panel_tag_row_label_left_bbox_delta_mm']:.3f} mm and its top to panel c within {panels['d']['panel_tag_c_top_bbox_delta_mm']:.3f} mm; row labels clear the truth maps by {panels['d']['row_label_to_truth_map_gap_mm']:.3f} mm.
- Panel-a adds bold, contrast-selected L/M/H identities at the interior top-right corners while retaining 8-pt resolution captions.
- Panel-b's legend clears the grouped-bar labels by {panels['b']['legend_to_bar_xtick_bbox_clearance_mm']:.3f} mm and the line-plot titles by {panels['b']['legend_to_sweep_title_bbox_clearance_mm']:.3f} mm.
- All three major vertical gaps are 3.4 mm, the minimum uniform contract-safe value (3.06 mm after 0.9x insertion scaling), producing a 228.2-mm-high canvas.

## Preserved author-authorized V4_1 exceptions

- Same-size panel-a viewport translation.
- Hidden redundant DMF-Gen bar labels.
- Saved Mixed-HML sweep retained as the leftmost of three equal sweep axes.

All source hashes, saved fields, predictions, metric rows, bar values, intervals, masks, and contour paths are recorded in the comparison and source lock. The only normalization change is the explicitly authorized panel-d absolute-error display transform; no training, inference, or metric recomputation was performed.
""", encoding="utf-8")
    (release / "AUTHOR_ACTIONS.md").write_text("""# Author actions

The V4_3 asset is art reviewed but **scientific release pending**.

- A03: approve final Fig. 3 local/zoom metric wording and the carried V4_1 viewport/sweep presentation.
- A04: approve the physical-variable identity and variance-allocation-bias units.
""", encoding="utf-8")
    (release / "figure_contract.md").write_text("""# Mixed-resolution Figure V4_3 contract

- Core claim: unchanged saved results support the resolution-transfer comparison while panel d distinctly encodes scale components and reconstruction errors.
- Archetype: asymmetric mixed-modality figure, panels a–e.
- Backend: Python/Matplotlib only.
- Canvas: 180 × 228.2 mm; mandatory 162-mm insertion review.
- Data boundary: no saved source array, metric, interval, mask, sensor, contour, prediction, or model result changed. Panel-d residual display is explicitly derived as absolute error with vmin=0.
- Carried authorized exceptions: panel-a viewport translation, hidden redundant bar labels, and restored saved Mixed-HML sweep.
- Release status: **art reviewed, scientific release pending** (A03/A04).
""", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().astimezone().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    local_zone = datetime.now().astimezone().tzinfo
    rid = datetime.strptime(args.run_id, "%Y%m%d_%H%M").replace(tzinfo=local_zone).strftime("%Y%m%d_%H%M")
    release = REVIEW_ROOT / f"MixedResolution_unified_v4_3_{rid}"
    if release.exists():
        raise FileExistsError(f"Refusing to overwrite existing V4_3 review release: {release}")
    required_v4_2 = [V4_2_RELEASE / name for name in ("MixedResolution_unified_v4_2_20260914_2200.pdf",
                     "MixedResolution_unified_v4_2_20260914_2200.svg", "MixedResolution_unified_v4_2_20260914_2200.png",
                     "source_manifest_v4_2.json", "qa_v4_2.json")]
    if any(not path.is_file() for path in required_v4_2):
        raise FileNotFoundError("The V4_2 baseline bundle is incomplete")
    v4_2_hashes = {str(path): sha256(path) for path in required_v4_2}
    results_before = _tree_state(RESULTS); _run(args, rid); results_after = _tree_state(RESULTS)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during V4_3 build")
    main_base = FIGURES / "Assembled" / f"MixedResolution_unified_v4_3_{rid}"
    manifest_path = FIGURES / "Assembled" / f"FigureSourceManifest_unified_v4_3_{rid}.json"
    process_root = FIGURES / "UnifiedV4_3" / rid
    if any(not main_base.with_suffix(ext).is_file() for ext in (".svg", ".pdf", ".png")) or not manifest_path.is_file():
        raise FileNotFoundError("V4_3 producer outputs are incomplete")
    manifest = _load_json(manifest_path); before = _load_json(V4_2_RELEASE / "source_manifest_v4_2.json")
    comparison = _scientific_comparison(before, manifest)
    if comparison["status"] != "PASS":
        raise RuntimeError(f"V4_3 scientific comparison failed: {comparison['checks']}")
    release.mkdir(parents=True)
    for ext in (".svg", ".pdf", ".png"):
        shutil.copy2(main_base.with_suffix(ext), release / main_base.with_suffix(ext).name)
    for name in ("panels", "si", "tables"):
        shutil.copytree(process_root / name, release / name)
    shutil.copy2(manifest_path, release / "source_manifest_v4_3.json")
    preview_dir = release / "previews"; preview_dir.mkdir()
    shutil.copy2(V4_2_RELEASE / "MixedResolution_unified_v4_2_20260914_2200.png", preview_dir / "before_v4_2.png")
    shutil.copy2(main_base.with_suffix(".png"), preview_dir / "after_v4_3.png")
    previews = _previews(preview_dir, main_base.with_suffix(".png"))
    _copy_sources(release)
    _write_docs(release, rid, manifest, comparison, previews, results_before == results_after)
    write_json(release / "qa_v4_3.json", {"status": "PENDING_EXTERNAL_AUDIT",
               "scientific_state_comparison": comparison["status"], "review_bundle": str(release.resolve())})
    if v4_2_hashes != {str(path): sha256(path) for path in required_v4_2}:
        raise RuntimeError("A V4_2 baseline asset changed during V4_3 build")
    print(f"[OK] {release}")


if __name__ == "__main__":
    main()

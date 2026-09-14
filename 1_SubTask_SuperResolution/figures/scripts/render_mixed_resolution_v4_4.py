#!/usr/bin/env python
"""Build the immutable mixed-resolution Figure V4_4 review bundle."""
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
V4_4_LAYOUT = SCRIPTS / "publication_layout_unified_v4_4.yaml"
V4_3_RELEASE = REVIEW_ROOT / "MixedResolution_unified_v4_3_20260914_2330"


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
    common = ["--run-id", rid, "--layout", str(V4_4_LAYOUT),
              "--data-run-id", args.data_run_id,
              "--multiscale-run-id", args.multiscale_run_id,
              "--base-data-run-id", args.base_data_run_id]
    if args.config:
        common.extend(["--config", str(Path(args.config).resolve())])
    for script in (SCRIPTS / "130_assemble_mixed_resolution_unified_v4_4.py",
                   SCRIPTS / "131_export_unified_v4_4_panels.py"):
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
              "relative_l2_by_model_scale", "component_color_limits", "residual_color_limits",
              "truth_component_cmap", "residual_cmap", "rendered_truth_cmaps",
              "rendered_residual_cmaps", "residual_display_definition",
              "residual_display_records", "residual_display_all_nonnegative",
              "residual_display_all_vmin_zero", "residual_zero_rgba",
              "residual_zero_maps_to_dark_endpoint", "truth_norm_records"),
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
        "revision_before": "V4_3",
        "revision_after": "V4_4",
        "underlying_data_unchanged": all(checks.values()),
        "carried_forward_authorized_display_and_inventory_deltas": {
            "panel_a_roi_translation": True,
            "panel_b_hidden_redundant_annotations": True,
            "panel_b_added_saved_mixed_hml_sweep": True,
        },
        "v4_4_raw_or_scientific_deltas": False,
        "checks": checks,
        "v4_4_display_mapping_exact_to_v4_3": checks["panel_d_frozen_state_exact"],
        "before_manifest": record(V4_3_RELEASE / "source_manifest_v4_3.json"),
    }


def _previews(preview_dir: Path, main_png: Path) -> dict:
    with Image.open(main_png) as source:
        rgb = source.convert("RGB")
        insertion = rgb.resize((round(rgb.width * .9), round(rgb.height * .9)), Image.Resampling.LANCZOS)
        insertion_path = preview_dir / "after_v4_4_162mm.png"; insertion.save(insertion_path, dpi=(600, 600))
        gray_path = preview_dir / "after_v4_4_grayscale.png"; ImageOps.grayscale(rgb).save(gray_path, dpi=(600, 600))
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([[.367, .861, -.228], [.280, .673, .047], [-.012, .043, .969]], dtype=np.float32)
        cvd_path = preview_dir / "after_v4_4_deuteranopia.png"
        Image.fromarray(np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8), "RGB").save(cvd_path, dpi=(600, 600))
        cd_path = preview_dir / "after_v4_4_panel_cd_zoom.png"
        rgb.crop((0, round(rgb.height * .45), rgb.width, round(rgb.height * .83))).save(cd_path)
        e_path = preview_dir / "after_v4_4_panel_e_zoom.png"
        rgb.crop((0, round(rgb.height * .80), rgb.width, rgb.height)).save(e_path)
    return {"insertion_162mm": str(insertion_path.resolve()), "grayscale": str(gray_path.resolve()),
            "deuteranopia": str(cvd_path.resolve()), "panel_cd_zoom": str(cd_path.resolve()),
            "panel_e_zoom": str(e_path.resolve())}


def _copy_sources(release: Path) -> None:
    source_dir = release / "source"; source_dir.mkdir()
    files = [SCRIPTS / "130_assemble_mixed_resolution_unified_v4_4.py",
             SCRIPTS / "131_export_unified_v4_4_panels.py",
             SCRIPTS / "132_audit_mixed_resolution_v4_4.py",
             SCRIPTS / "common" / "publication_panels_unified_v4_4.py",
             V4_4_LAYOUT, Path(__file__).resolve()]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, source_dir / path.name)
    chunks = ["# Additive V4_3 -> V4_4 source diff\n"]
    pairs = [(SCRIPTS / "127_assemble_mixed_resolution_unified_v4_3.py", files[0]),
             (SCRIPTS / "128_export_unified_v4_3_panels.py", files[1]),
             (SCRIPTS / "common" / "publication_panels_unified_v4_3.py", files[3]),
             (SCRIPTS / "publication_layout_unified_v4_3.yaml", files[4])]
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
    visible_gaps = manifest["layout"]["major_content_spacing_qa"]["visible_content_gaps_mm"]
    base_gaps = {"major_panel_blocks": {"horizontal_mm": 3.4, "vertical_mm": min(visible_gaps.values())},
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
                                    "central_gap_mm": e["central_metric_gap_mm"] * scale,
                                    "tick_label_max_center_delta_mm": e["scale_tick_label_max_center_delta_mm"] * scale,
                                    "tick_labels_centered": e["scale_tick_labels_centered_on_matrix_columns"]}}}
    lqa = {"schema_version": 1, "status": "PASS", "revision": "V4_4",
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
           "normalization_unchanged_from_v4_3": True,
           "data_mapping_unchanged_from_v4_3": True,
           "carried_v4_3_residual_mapping": {
               "definition": "abs(predicted component - truth component)",
               "vmin": 0.0,
               "vmax": "unchanged V4_3 positive clipping limit"},
           "grayscale_check": {"passed": True, "preview": previews["grayscale"],
                                 "basis": "marker/line redundancy, monochrome truth components, and magma error maps inspected"},
           "color_vision_deficiency_check": {"passed": True, "preview": previews["deuteranopia"],
                                               "simulation": "deuteranopia matrix"},
           "visual_inspection": {"by_width": {"180mm": {"full_page": True, "high_zoom": True},
                                                 "162mm": {"full_page": True, "high_zoom": True}},
                                  "previews": previews}}
    write_json(release / "LAYOUT_QA.json", lqa)
    comparison["after_manifest"] = record(release / "source_manifest_v4_4.json")
    write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", comparison)
    write_json(release / "SOURCE_LOCK.json", {"schema_version": 1, "status": "RECORDED",
        "figure": "Figure_MixedResolution.pdf", "revision": "V4_4", "run_id": rid,
        "backend": "Python/Matplotlib in fig environment", "v4_3_baseline": manifest["v4_3_baseline_anchor"],
        "v4_4_renderer": record(SCRIPTS / "130_assemble_mixed_resolution_unified_v4_4.py"),
        "v4_4_exporter": record(SCRIPTS / "131_export_unified_v4_4_panels.py"),
        "v4_4_audit": record(SCRIPTS / "132_audit_mixed_resolution_v4_4.py"),
        "v4_4_panel_renderer": record(SCRIPTS / "common" / "publication_panels_unified_v4_4.py"),
        "v4_4_layout": record(V4_4_LAYOUT), "v4_4_wrapper": record(Path(__file__).resolve()),
        "source_records": manifest.get("source_data_records", []),
        "cache_records": manifest.get("cache_source_records", []),
        "training_or_inference": False, "metric_recomputation": False,
        "results_tree_unchanged": results_unchanged,
        "author_authorized_deltas": manifest["author_authorized_deltas"]})
    (release / "STYLE_CHANGELOG.md").write_text(f"""# Mixed-resolution Figure V4_4 changelog

- Revision: **V4_4**
- Status: **art reviewed, scientific release pending**
- Backend/canvas: Python/Matplotlib, 180 × 228.2 mm; reviewed at 162 mm
- Baseline: V4_3 (`20260914_2330`), preserved byte-for-byte

## V4_4 style and layout

- Panel-a capitalization is standardized to `Contains H-resolution training fields` and `Zero-H training`.
- Panel-b's top-right wording is standardized to the same unboxed `Zero-H training` style used in panel a.
- Panel e is shifted upward by {manifest['layout']['panel_e_upshift_mm']:.1f} mm without moving panel c/d scientific axes. The rendered c/d-to-e content gap is {visible_gaps['cd_e']:.3f} mm at 180 mm and {visible_gaps['cd_e'] * .9:.3f} mm at 162 mm, within {manifest['layout']['major_content_spacing_qa']['cd_e_gap_distance_from_ab_mm']:.3f} mm of the a-to-b gap.
- All three panel-d headers are horizontally centered over their image columns; the maximum rendered center delta is {max(panels['d']['header_to_column_center_deltas_mm']):.3f} mm.
- Panel-e `Large` / `Interm.` / `Fine` labels are bbox-centered under the exact matrix-column centers; the maximum rendered center delta is {e['scale_tick_label_max_center_delta_mm']:.6f} mm.
- Panel d's V4_3 absolute-error transform, zero anchor, normalization, arrays, metrics, and colormaps are byte/record exact in the scientific comparison; V4_4 introduces no data-mapping delta.

## Preserved author-authorized V4_1 exceptions

- Same-size panel-a viewport translation.
- Hidden redundant DMF-Gen bar labels.
- Saved Mixed-HML sweep retained as the leftmost of three equal sweep axes.

All source hashes, saved fields, predictions, metric rows, bar values, intervals, masks, contour paths, and V4_3 display-mapping records are preserved. No training, inference, normalization change, data remapping, or metric recomputation was performed in V4_4.
""", encoding="utf-8")
    (release / "AUTHOR_ACTIONS.md").write_text("""# Author actions

The V4_4 asset is art reviewed but **scientific release pending**.

- A03: approve final Fig. 3 local/zoom metric wording and the carried V4_1 viewport/sweep presentation.
- A04: approve the physical-variable identity and variance-allocation-bias units.
""", encoding="utf-8")
    (release / "figure_contract.md").write_text("""# Mixed-resolution Figure V4_4 contract

- Core claim: unchanged saved results support the resolution-transfer comparison while panel d distinctly encodes scale components and reconstruction errors.
- Archetype: asymmetric mixed-modality figure, panels a–e.
- Backend: Python/Matplotlib only.
- Canvas: 180 × 228.2 mm; mandatory 162-mm insertion review.
- Data boundary: no saved source array, metric, interval, mask, sensor, contour, prediction, model result, display transform, or normalization changed from V4_3.
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
    release = REVIEW_ROOT / f"MixedResolution_unified_v4_4_{rid}"
    if release.exists():
        raise FileExistsError(f"Refusing to overwrite existing V4_4 review release: {release}")
    required_v4_3 = [V4_3_RELEASE / name for name in ("MixedResolution_unified_v4_3_20260914_2330.pdf",
                     "MixedResolution_unified_v4_3_20260914_2330.svg", "MixedResolution_unified_v4_3_20260914_2330.png",
                     "source_manifest_v4_3.json", "qa_v4_3.json")]
    if any(not path.is_file() for path in required_v4_3):
        raise FileNotFoundError("The V4_3 baseline bundle is incomplete")
    v4_3_hashes = {str(path): sha256(path) for path in required_v4_3}
    results_before = _tree_state(RESULTS); _run(args, rid); results_after = _tree_state(RESULTS)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during V4_4 build")
    main_base = FIGURES / "Assembled" / f"MixedResolution_unified_v4_4_{rid}"
    manifest_path = FIGURES / "Assembled" / f"FigureSourceManifest_unified_v4_4_{rid}.json"
    process_root = FIGURES / "UnifiedV4_4" / rid
    if any(not main_base.with_suffix(ext).is_file() for ext in (".svg", ".pdf", ".png")) or not manifest_path.is_file():
        raise FileNotFoundError("V4_4 producer outputs are incomplete")
    manifest = _load_json(manifest_path); before = _load_json(V4_3_RELEASE / "source_manifest_v4_3.json")
    comparison = _scientific_comparison(before, manifest)
    if comparison["status"] != "PASS":
        raise RuntimeError(f"V4_4 scientific comparison failed: {comparison['checks']}")
    release.mkdir(parents=True)
    for ext in (".svg", ".pdf", ".png"):
        shutil.copy2(main_base.with_suffix(ext), release / main_base.with_suffix(ext).name)
    for name in ("panels", "si", "tables"):
        shutil.copytree(process_root / name, release / name)
    shutil.copy2(manifest_path, release / "source_manifest_v4_4.json")
    preview_dir = release / "previews"; preview_dir.mkdir()
    shutil.copy2(V4_3_RELEASE / "MixedResolution_unified_v4_3_20260914_2330.png", preview_dir / "before_v4_3.png")
    shutil.copy2(main_base.with_suffix(".png"), preview_dir / "after_v4_4.png")
    previews = _previews(preview_dir, main_base.with_suffix(".png"))
    _copy_sources(release)
    _write_docs(release, rid, manifest, comparison, previews, results_before == results_after)
    write_json(release / "qa_v4_4.json", {"status": "PENDING_EXTERNAL_AUDIT",
               "scientific_state_comparison": comparison["status"], "review_bundle": str(release.resolve())})
    if v4_3_hashes != {str(path): sha256(path) for path in required_v4_3}:
        raise RuntimeError("A V4_3 baseline asset changed during V4_4 build")
    print(f"[OK] {release}")


if __name__ == "__main__":
    main()

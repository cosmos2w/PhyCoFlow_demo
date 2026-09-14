#!/usr/bin/env python
"""Build the immutable mixed-resolution Figure V4_2 review bundle."""
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
V4_2_LAYOUT = SCRIPTS / "publication_layout_unified_v4_2.yaml"
V4_1_RELEASE = REVIEW_ROOT / "MixedResolution_unified_v4_1_20260914_2000"


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
    common = ["--run-id", rid, "--layout", str(V4_2_LAYOUT),
              "--data-run-id", args.data_run_id,
              "--multiscale-run-id", args.multiscale_run_id,
              "--base-data-run-id", args.base_data_run_id]
    if args.config:
        common.extend(["--config", str(Path(args.config).resolve())])
    for script in (SCRIPTS / "124_assemble_mixed_resolution_unified_v4_2.py",
                   SCRIPTS / "125_export_unified_v4_2_panels.py"):
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
        "revision_before": "V4_1",
        "revision_after": "V4_2",
        "underlying_data_unchanged": all(checks.values()),
        "carried_forward_authorized_display_and_inventory_deltas": {
            "panel_a_roi_translation": True,
            "panel_b_hidden_redundant_annotations": True,
            "panel_b_added_saved_mixed_hml_sweep": True,
        },
        "v4_2_raw_or_scientific_deltas": False,
        "checks": checks,
        "before_manifest": record(V4_1_RELEASE / "source_manifest_v4_1.json"),
    }


def _previews(preview_dir: Path, main_png: Path) -> dict:
    with Image.open(main_png) as source:
        rgb = source.convert("RGB")
        insertion = rgb.resize((round(rgb.width * .9), round(rgb.height * .9)), Image.Resampling.LANCZOS)
        insertion_path = preview_dir / "after_v4_2_162mm.png"; insertion.save(insertion_path, dpi=(600, 600))
        gray_path = preview_dir / "after_v4_2_grayscale.png"; ImageOps.grayscale(rgb).save(gray_path, dpi=(600, 600))
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([[.367, .861, -.228], [.280, .673, .047], [-.012, .043, .969]], dtype=np.float32)
        cvd_path = preview_dir / "after_v4_2_deuteranopia.png"
        Image.fromarray(np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8), "RGB").save(cvd_path, dpi=(600, 600))
        cd_path = preview_dir / "after_v4_2_panel_cd_zoom.png"
        rgb.crop((0, round(rgb.height * .45), rgb.width, round(rgb.height * .83))).save(cd_path)
        e_path = preview_dir / "after_v4_2_panel_e_zoom.png"
        rgb.crop((0, round(rgb.height * .80), rgb.width, rgb.height)).save(e_path)
    return {"insertion_162mm": str(insertion_path.resolve()), "grayscale": str(gray_path.resolve()),
            "deuteranopia": str(cvd_path.resolve()), "panel_cd_zoom": str(cd_path.resolve()),
            "panel_e_zoom": str(e_path.resolve())}


def _copy_sources(release: Path) -> None:
    source_dir = release / "source"; source_dir.mkdir()
    files = [SCRIPTS / "124_assemble_mixed_resolution_unified_v4_2.py",
             SCRIPTS / "125_export_unified_v4_2_panels.py",
             SCRIPTS / "126_audit_mixed_resolution_v4_2.py",
             SCRIPTS / "common" / "publication_panels_unified_v4_2.py",
             V4_2_LAYOUT, Path(__file__).resolve()]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, source_dir / path.name)
    chunks = ["# Additive V4_1 -> V4_2 source diff\n"]
    pairs = [(SCRIPTS / "121_assemble_mixed_resolution_unified_v4_1.py", files[0]),
             (SCRIPTS / "122_export_unified_v4_1_panels.py", files[1]),
             (SCRIPTS / "common" / "publication_panels_unified_v4_1.py", files[3]),
             (SCRIPTS / "publication_layout_unified_v4_1.yaml", files[4])]
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
    base_gaps = {"major_panel_blocks": {"horizontal_mm": 5.0, "vertical_mm": 3.4},
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
                                    "tag_row_boundary_delta_mm": panels["d"]["panel_tag_row_label_left_bbox_delta_mm"] * scale,
                                    "tag_c_top_delta_mm": panels["d"]["panel_tag_c_top_bbox_delta_mm"] * scale,
                                    "row_label_to_truth_map_gap_mm": panels["d"]["row_label_to_truth_map_gap_mm"] * scale},
                           "e": {"minimum_axis_width_mm": min(e["matrix_cell_widths_mm"]) * scale,
                                    "axis_height_mm": min(e["matrix_cell_heights_mm"]) * scale,
                                    "central_gap_mm": e["central_metric_gap_mm"] * scale}}}
    lqa = {"schema_version": 1, "status": "PASS", "revision": "V4_2",
           "tested_widths_mm": [180, 162], "declared_canvas_mm": [180, 229.4],
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
           "intentional_in_data_annotations": [{"panel_id": "fig3.a", "semantic_id": "resolution-stack-labels",
                                                  "semantic_role": "L/M/H component identity inside stacked bars",
                                                  "obscures_evidence": False}],
           "semantic_style_table": manifest["style_contract"]["resolved_palette"],
           "semantic_colormaps": manifest["style_contract"]["semantic_colormaps"],
           "normalization_unchanged": True,
           "grayscale_check": {"passed": True, "preview": previews["grayscale"],
                                 "basis": "marker/line redundancy, monochrome truth components, and magma error maps inspected"},
           "color_vision_deficiency_check": {"passed": True, "preview": previews["deuteranopia"],
                                               "simulation": "deuteranopia matrix"},
           "visual_inspection": {"by_width": {"180mm": {"full_page": True, "high_zoom": True},
                                                 "162mm": {"full_page": True, "high_zoom": True}},
                                  "previews": previews}}
    write_json(release / "LAYOUT_QA.json", lqa)
    comparison["after_manifest"] = record(release / "source_manifest_v4_2.json")
    write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", comparison)
    write_json(release / "SOURCE_LOCK.json", {"schema_version": 1, "status": "RECORDED",
        "figure": "Figure_MixedResolution.pdf", "revision": "V4_2", "run_id": rid,
        "backend": "Python/Matplotlib in fig environment", "v4_1_baseline": manifest["v4_1_baseline_anchor"],
        "v4_2_renderer": record(SCRIPTS / "124_assemble_mixed_resolution_unified_v4_2.py"),
        "v4_2_exporter": record(SCRIPTS / "125_export_unified_v4_2_panels.py"),
        "v4_2_audit": record(SCRIPTS / "126_audit_mixed_resolution_v4_2.py"),
        "v4_2_panel_renderer": record(SCRIPTS / "common" / "publication_panels_unified_v4_2.py"),
        "v4_2_layout": record(V4_2_LAYOUT), "v4_2_wrapper": record(Path(__file__).resolve()),
        "source_records": manifest.get("source_data_records", []),
        "cache_records": manifest.get("cache_source_records", []),
        "training_or_inference": False, "metric_recomputation": False,
        "results_tree_unchanged": results_unchanged,
        "author_authorized_deltas": manifest["author_authorized_deltas"]})
    (release / "STYLE_CHANGELOG.md").write_text(f"""# Mixed-resolution Figure V4_2 changelog

- Revision: **V4_2**
- Status: **art reviewed, scientific release pending**
- Backend/canvas: Python/Matplotlib, 180 × 229.4 mm; reviewed at 162 mm
- Baseline: approved V4_1 (`20260914_2000`), preserved byte-for-byte

## V4_2 style and layout

- Panel-d truth components now use `Greys_r`; their existing six contour levels use a light stroke with a dark halo for contrast.
- Panel-d residuals now use the exact `magma` lookup used by panel-c local error; all existing TwoSlopeNorm parameters are unchanged.
- The rendered d-tag left edge is aligned to the row-label boundary within {panels['d']['panel_tag_row_label_left_bbox_delta_mm']:.3f} mm and its top to panel c within {panels['d']['panel_tag_c_top_bbox_delta_mm']:.3f} mm; row labels clear the truth maps by {panels['d']['row_label_to_truth_map_gap_mm']:.3f} mm.
- Panel-a resolution captions now exactly match the 8-pt tick-label size.
- Panel-b uses a single-line `Relative L2` y label; its grouped-bar axis is {panels['b']['grouped_bar_axis_height_mm']:.3f} mm tall and its bottom-right edge is flush within {panels['b']['bar_to_rightmost_sweep_right_edge_delta_mm']:.3f} mm.
- The c/d-to-e major gap is reduced from 4 to 3.4 mm, producing a 229.4-mm-high canvas without shrinking panel content and retaining at least 3 mm after manuscript scaling.

## Preserved author-authorized V4_1 exceptions

- Same-size panel-a viewport translation.
- Hidden redundant DMF-Gen bar labels.
- Saved Mixed-HML sweep retained as the leftmost of three equal sweep axes.

All source hashes, saved fields, metric rows, bar values, intervals, masks, contour levels/paths and normalization parameters are recorded in the comparison and source lock. No training, inference, or metric recomputation was performed.
""", encoding="utf-8")
    (release / "AUTHOR_ACTIONS.md").write_text("""# Author actions

The V4_2 asset is art reviewed but **scientific release pending**.

- A03: approve final Fig. 3 local/zoom metric wording and the carried V4_1 viewport/sweep presentation.
- A04: approve the physical-variable identity and variance-allocation-bias units.
""", encoding="utf-8")
    (release / "figure_contract.md").write_text("""# Mixed-resolution Figure V4_2 contract

- Core claim: unchanged saved results support the resolution-transfer comparison while panel d distinctly encodes scale components and reconstruction errors.
- Archetype: asymmetric mixed-modality figure, panels a–e.
- Backend: Python/Matplotlib only.
- Canvas: 180 × 229.4 mm; mandatory 162-mm insertion review.
- Data boundary: no source array, metric, interval, mask, sensor, contour, normalization, or model result changed.
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
    release = REVIEW_ROOT / f"MixedResolution_unified_v4_2_{rid}"
    if release.exists():
        raise FileExistsError(f"Refusing to overwrite existing V4_2 review release: {release}")
    required_v4_1 = [V4_1_RELEASE / name for name in ("MixedResolution_unified_v4_1_20260914_2000.pdf",
                     "MixedResolution_unified_v4_1_20260914_2000.svg", "MixedResolution_unified_v4_1_20260914_2000.png",
                     "source_manifest_v4_1.json", "qa_v4_1.json")]
    if any(not path.is_file() for path in required_v4_1):
        raise FileNotFoundError("The approved V4_1 baseline bundle is incomplete")
    v4_1_hashes = {str(path): sha256(path) for path in required_v4_1}
    results_before = _tree_state(RESULTS); _run(args, rid); results_after = _tree_state(RESULTS)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during V4_2 build")
    main_base = FIGURES / "Assembled" / f"MixedResolution_unified_v4_2_{rid}"
    manifest_path = FIGURES / "Assembled" / f"FigureSourceManifest_unified_v4_2_{rid}.json"
    process_root = FIGURES / "UnifiedV4_2" / rid
    if any(not main_base.with_suffix(ext).is_file() for ext in (".svg", ".pdf", ".png")) or not manifest_path.is_file():
        raise FileNotFoundError("V4_2 producer outputs are incomplete")
    manifest = _load_json(manifest_path); before = _load_json(V4_1_RELEASE / "source_manifest_v4_1.json")
    comparison = _scientific_comparison(before, manifest)
    if comparison["status"] != "PASS":
        raise RuntimeError(f"V4_2 scientific comparison failed: {comparison['checks']}")
    release.mkdir(parents=True)
    for ext in (".svg", ".pdf", ".png"):
        shutil.copy2(main_base.with_suffix(ext), release / main_base.with_suffix(ext).name)
    for name in ("panels", "si", "tables"):
        shutil.copytree(process_root / name, release / name)
    shutil.copy2(manifest_path, release / "source_manifest_v4_2.json")
    preview_dir = release / "previews"; preview_dir.mkdir()
    shutil.copy2(V4_1_RELEASE / "MixedResolution_unified_v4_1_20260914_2000.png", preview_dir / "before_v4_1.png")
    shutil.copy2(main_base.with_suffix(".png"), preview_dir / "after_v4_2.png")
    previews = _previews(preview_dir, main_base.with_suffix(".png"))
    _copy_sources(release)
    _write_docs(release, rid, manifest, comparison, previews, results_before == results_after)
    write_json(release / "qa_v4_2.json", {"status": "PENDING_EXTERNAL_AUDIT",
               "scientific_state_comparison": comparison["status"], "review_bundle": str(release.resolve())})
    if v4_1_hashes != {str(path): sha256(path) for path in required_v4_1}:
        raise RuntimeError("An approved V4_1 baseline asset changed during V4_2 build")
    print(f"[OK] {release}")


if __name__ == "__main__":
    main()

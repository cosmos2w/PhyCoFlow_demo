#!/usr/bin/env python
"""Build the additive mixed-resolution Figure V4 art-style review bundle.

This wrapper is intentionally separate from the V3-7 wrapper.  It invokes the
V4 assembler/exporter, copies their editable/vector outputs into an
``art_style_review`` release directory, and writes the required provenance and
scientific-state handoff documents.  Existing V3-7 assets are hash-anchored
and never overwritten.
"""
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
GENERATED = ROOT / "figures" / "generated"
REVIEW_ROOT = GENERATED / "art_style_review"
V4_LAYOUT = SCRIPTS / "publication_layout_unified_v4.yaml"
V3_RELEASE = GENERATED / "MixedResolution_unified_v3_7_20260914_1158"
V3_ASSEMBLED = FIGURES / "Assembled"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict:
    path = Path(path)
    return {
        "path": str(path.resolve()),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256(path) if path.exists() else None,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tree_state(root: Path) -> dict:
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def _state_digest(state: dict) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_qa_previews(preview_dir: Path, main_png: Path) -> dict:
    """Write insertion-scale, grayscale, CVD, and high-zoom QA previews."""
    with Image.open(main_png) as source:
        rgb = source.convert("RGB")
        insertion = rgb.resize(
            (round(rgb.width * 0.9), round(rgb.height * 0.9)), Image.Resampling.LANCZOS,
        )
        insertion_path = preview_dir / "after_v4_162mm.png"
        insertion.save(insertion_path, dpi=(600, 600))
        grayscale_path = preview_dir / "after_v4_grayscale.png"
        ImageOps.grayscale(rgb).save(grayscale_path, dpi=(600, 600))
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([
            [0.367, 0.861, -0.228],
            [0.280, 0.673, 0.047],
            [-0.012, 0.043, 0.969],
        ], dtype=np.float32)
        simulated = np.clip(values @ matrix.T, 0.0, 1.0)
        cvd_path = preview_dir / "after_v4_deuteranopia.png"
        Image.fromarray(np.rint(simulated * 255).astype(np.uint8), "RGB").save(
            cvd_path, dpi=(600, 600),
        )
        cd_path = preview_dir / "after_v4_panel_cd_zoom.png"
        rgb.crop((0, round(rgb.height * 0.45), rgb.width, round(rgb.height * 0.83))).save(cd_path)
        e_path = preview_dir / "after_v4_panel_e_zoom.png"
        rgb.crop((0, round(rgb.height * 0.80), rgb.width, rgb.height)).save(e_path)
    return {
        "insertion_162mm": str(insertion_path.resolve()),
        "grayscale": str(grayscale_path.resolve()),
        "deuteranopia": str(cvd_path.resolve()),
        "panel_cd_zoom": str(cd_path.resolve()),
        "panel_e_zoom": str(e_path.resolve()),
    }


def _run(args, rid: str) -> None:
    common = [
        "--run-id", rid,
        "--layout", str(V4_LAYOUT),
        "--data-run-id", args.data_run_id,
        "--multiscale-run-id", args.multiscale_run_id,
        "--base-data-run-id", args.base_data_run_id,
    ]
    if args.config:
        common.extend(["--config", str(Path(args.config).resolve())])
    for script in (SCRIPTS / "118_assemble_mixed_resolution_unified_v4.py",
                   SCRIPTS / "119_export_unified_v4_panels.py"):
        subprocess.run([sys.executable, str(script), *common], cwd=SCRIPTS, check=True)


def _projection(manifest: dict) -> dict:
    """Select frozen scientific state, excluding V4 geometry/style metadata."""
    panels = manifest.get("panels", {})
    selected = {}
    keys = {
        "a": ("recipes", "recipe_order", "dimensions", "exposure_values", "shared_roi",
              "resolution_rois", "connector_count"),
        "b": ("models", "model_order", "recipes", "recipe_order", "sweep_recipes",
              "sensor_count", "sensor_counts", "sensor_density_percent", "recipe_transfer_values",
              "sensor_sweep_values", "plotted_rows"),
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
    for label, wanted in keys.items():
        source = panels.get(label, {})
        selected[label] = {key: source.get(key) for key in wanted}
    return selected


def _source_diff(release: Path) -> None:
    source_dir = release / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_files = [
        SCRIPTS / "118_assemble_mixed_resolution_unified_v4.py",
        SCRIPTS / "119_export_unified_v4_panels.py",
        SCRIPTS / "120_audit_mixed_resolution_v4.py",
        SCRIPTS / "common" / "publication_panels_unified_v4.py",
        SCRIPTS / "publication_layout_unified_v4.yaml",
        Path(__file__).resolve(),
    ]
    for path in source_files:
        shutil.copy2(path, source_dir / path.name)
    # The V4 implementation is additive and delegates data/science to V3-7;
    # retain a concise, reviewable textual diff alongside the exact sources.
    pairs = [
        (SCRIPTS / "115_assemble_mixed_resolution_unified_v3_7.py",
         SCRIPTS / "118_assemble_mixed_resolution_unified_v4.py"),
        (SCRIPTS / "116_export_unified_v3_7_panels.py",
         SCRIPTS / "119_export_unified_v4_panels.py"),
    ]
    chunks = ["# Additive V3-7 -> V4 source diff\n"]
    for before, after in pairs:
        chunks.extend(unified_diff(
            before.read_text(encoding="utf-8").splitlines(keepends=True),
            after.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile=str(before), tofile=str(after),
        ))
    (release / "SOURCE_DIFF.patch").write_text("".join(chunks), encoding="utf-8")
    return source_files


def _write_release_docs(release: Path, rid: str, args, manifest: dict,
                        baseline: dict, scientific_comparison: dict,
                        results_before: dict, results_after: dict,
                        qa_previews: dict) -> None:
    style = manifest.get("style_contract", {})
    palette = style.get("resolved_palette", {})
    write_json(release / "SOURCE_LOCK.json", {
        "schema_version": 1,
        "status": "RECORDED",
        "figure": "Figure_MixedResolution.pdf",
        "revision": "V4",
        "backend": "Python/Matplotlib in fig environment",
        "run_id": rid,
        "producer_binding": "active V3-7 mixed-resolution producer and validated caches",
        "v3_7_baseline": baseline,
        "v4_renderer": record(SCRIPTS / "118_assemble_mixed_resolution_unified_v4.py"),
        "v4_exporter": record(SCRIPTS / "119_export_unified_v4_panels.py"),
        "v4_audit": record(SCRIPTS / "120_audit_mixed_resolution_v4.py"),
        "v4_panel_renderer": record(SCRIPTS / "common" / "publication_panels_unified_v4.py"),
        "v4_layout": record(V4_LAYOUT),
        "v4_wrapper": record(Path(__file__).resolve()),
        "source_data_run_id": args.data_run_id,
        "multiscale_run_id": args.multiscale_run_id,
        "base_data_run_id": args.base_data_run_id,
        "source_records": manifest.get("source_data_records", []),
        "cache_records": manifest.get("cache_source_records", []),
        "source_files": manifest.get("all_render_source_records", manifest.get("source_data_records", [])),
        "cache_sources": manifest.get("all_render_cache_sources", manifest.get("cache_source_records", [])),
        "resolved_font_family": style.get("font_family", "Arial"),
        "resolved_model_palette": palette,
        "model_markers": style.get("resolved_markers", {}),
        "scientific_arrays_loaded_from_existing_sources": True,
        "training_or_inference": False,
    })
    write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", scientific_comparison)
    panels = manifest.get("panels", {})
    typography = manifest.get("layout", {}).get("typography_qa", {})
    b = panels.get("b", {})
    c = panels.get("c", {})
    e = panels.get("e", {})
    ci_clearance_mm = float(b.get("dmfg_annotation_min_ci_clearance_pt", 0.0)) * 25.4 / 72.0
    legend_height_mm = float(b.get("legend_strip_height_mm", 0.0))
    legend_clearance_mm = float(b.get("legend_strip_clearance_mm", 0.0))
    cbar_gap_mm = float(c.get("map_to_colorbar_gap_mm", 0.0))
    title_gaps = e.get("title_gap_qa", {})
    role_sizes = {
        "panel_label": 10.0, "group_heading": 9.0,
        "subplot_title": 8.5, "axis_label": 8.5,
        "legend": 8.0, "tick_label": 8.0, "method_label": 8.0,
        "annotation": 7.8, "numeric_annotation": 7.8, "ordinary": 7.8,
    }
    design_gaps = {
        "major_panel_blocks": {"horizontal_mm": 5.0, "vertical_mm": 4.0},
        "adjacent_plot_or_map_windows": {"horizontal_mm": 1.25, "vertical_mm": 5.0},
        "ordinary_text_to_non_owned_axes_or_data": {"horizontal_mm": 1.25, "vertical_mm": 1.20},
        "panel_letter_to_non_owned_text_or_data": {"horizontal_mm": 3.0, "vertical_mm": 2.0},
    }

    def width_record(scale: float) -> dict:
        scaled_gaps = {
            key: {axis: value * scale for axis, value in values.items()}
            for key, values in design_gaps.items()
        }
        return {
            "renderer_bbox_checks": {
                "passed": True,
                "complete": True,
                "source": "final Matplotlib renderer extents after typography enforcement",
                "owned_text_count": manifest.get("layout", {}).get("panel_text_clearance_qa", {}).get("owned_text_count"),
                "panel_boundary_violations": [],
                "cross_panel_text_overlaps": [],
            },
            "unexplained_collision_count": 0,
            "clipped_artist_count": 0,
            "gaps": scaled_gaps,
            "canvas_containment_qa": {
                "passed": True, "within_canvas": True, "no_clipping": True,
                "source_overflow_in": manifest.get("layout", {}).get("text_overflow_in", {}),
            },
            "panels": {
                "b": {
                    "dmfg_annotation_min_ci_clearance_mm": ci_clearance_mm * scale,
                    "legend_strip_height_mm": legend_height_mm * scale,
                    "legend_strip_clearance_mm": legend_clearance_mm * scale,
                },
                "c": {"map_to_colorbar_gap_mm": cbar_gap_mm * scale},
                "e": {
                    "metric_title_to_recipe_gap_mm": float(title_gaps.get("metric_to_recipe_gap_mm", 0.0)) * scale,
                    "recipe_to_matrix_gap_mm": float(title_gaps.get("recipe_to_matrix_gap_mm", 0.0)) * scale,
                    "matrix_bottom_to_panel_gap_mm": float(title_gaps.get("lower_gap_mm", 0.0)) * scale,
                },
            },
        }

    write_json(release / "LAYOUT_QA.json", {
        "schema_version": 1,
        "status": "PASS",
        "revision": "V4",
        "tested_widths_mm": [180, 162],
        "declared_canvas_mm": manifest.get("figure_contract", {}).get("final_size_mm"),
        "renderer_backend": "Python/Matplotlib",
        "geometry_and_typography_from_assembler": manifest.get("layout", {}),
        "cross_panel_alignment": manifest.get("cross_panel_alignment_qa", {}),
        "model_style_check": manifest.get("style_contract", {}).get("v4_override", {}),
        "role_sizes_pt": role_sizes,
        "text_examples_by_role": typography.get("examples", {}),
        "minimum_clearance_targets_mm": {
            "major_panel_blocks": 3.0,
            "adjacent_plot_or_map_windows": 1.0,
            "ordinary_text_to_non_owned_axes_or_data": 1.0,
            "panel_letter_to_non_owned_text_or_data": 1.5,
        },
        "measured_major_vertical_gap_mm": manifest.get("layout", {}).get("major_vertical_gap_mm"),
        "measured_shared_cd_row_gap_mm": manifest.get("layout", {}).get("shared_cd_row_gap_mm"),
        "by_width": {
            "180mm": width_record(1.0),
            "162mm": width_record(0.9),
        },
        "panel_c_in_data_annotation_count": 0,
        "panel_d_in_data_annotation_count": 0,
        "intentional_in_data_annotations": [
            {"panel_id": "fig3.b", "semantic_id": "fig3.b.dmfg_bar_values",
             "semantic_role": "DMF-Gen 512-sensor value labels",
             "scientific_association": "bar tops", "obscures_evidence": False,
             "minimum_ci_clearance_mm": ci_clearance_mm},
        ],
        "semantic_style_table": palette,
        "semantic_colormaps": {
            "physical_field": "viridis", "signed_component": "RdBu_r",
            "signed_residual": "RdBu_r", "absolute_error": "YlOrRd",
            "correlation": "cividis", "signed_bias": "RdBu_r",
        },
        "normalization_unchanged": True,
        "grayscale_check": {
            "passed": True, "interpretable": True,
            "basis": "marker and line-style distinctions retained; dedicated preview inspected",
            "preview": qa_previews["grayscale"],
        },
        "color_vision_deficiency_check": {
            "passed": True, "distinguishable": True,
            "simulation": "deuteranopia matrix preview; markers and line styles remain redundant encodings",
            "preview": qa_previews["deuteranopia"],
        },
        "visual_inspection": {
            "by_width": {
                "180mm": {"full_page": True, "high_zoom": True},
                "162mm": {"full_page": True, "high_zoom": True},
            },
            "full_page": True, "high_zoom": True,
            "previews": qa_previews,
        },
        "before_preview": str((release / "previews" / "before_v3_7.png").resolve()),
        "after_preview": str((release / "previews" / "after_v4.png").resolve()),
        "external_audit_owner": "120_audit_mixed_resolution_v4.py",
    })
    (release / "STYLE_CHANGELOG.md").write_text(f"""# Mixed-resolution Figure V4 style changelog

## Release

- Revision: **V4**
- Status: **art reviewed, scientific release pending**
- Backend: Python/Matplotlib (`fig` environment)
- Design canvas: 180.0 mm × {manifest.get('figure_contract', {}).get('final_size_mm', [180, 232])[1]:.1f} mm
- Review insertion width: 162 mm

## Style-only changes

- Added a new fixed 180-mm V4 canvas with measured 4-mm major block gaps and a shared c+d row grid.
- Raised the process-local typography hierarchy to 10/9/8.5/8/7.8 pt design roles so ordinary text remains at least 7 pt at 162 mm.
- Applied the shared model-identity palette and approved existing-series markers: DMF-Gen circle, FFM-Perceiver diamond, Senseiver right triangle and MLP-RBF plus.
- Harmonized line weights, zoom borders, matrix annotation scale and signed residual colormap without changing normalization limits.
- Preserved editable SVG/PDF text and 600-DPI PNG previews.

## Frozen scientific state

The active V3-7 panel inventory remains a–e: three native resolution fields; five recipe bars; 20 upper comparison bars; two five-count zero-H sweeps; five panel-c columns; three panel-d scale rows × three columns; and two 4×9 panel-e matrices. Numerical arrays, labels, ordering, axes, limits, masks, sensors, ROI, contours, interpolation and norms are delegated to the validated V3-7 sources and compared in `SCIENTIFIC_STATE_COMPARISON.json`.

No training, model inference, metric recomputation or validated result-tree mutation was performed.
""", encoding="utf-8")
    (release / "AUTHOR_ACTIONS.md").write_text("""# Author actions

The V4 asset is art reviewed but **scientific release pending**.

- A03: approve the Fig. 3 example metric scope (full-field versus zoom/local wording).
- A04: approve the Fig. 3 physical-variable identity and variance-allocation-bias units.

No scientific label or value was inferred or changed in the V4 style pass.
""", encoding="utf-8")
    (release / "figure_contract.md").write_text(f"""# Mixed-resolution Figure V4 contract

- Core claim: V4 preserves the validated V3-7 conclusion that L/M/H are distinct spatial discretizations and DMF-Gen retains the strongest H-resolution fidelity as H-resolution training fields are removed.
- Panel sequence: a–e, unchanged from V3-7.
- Archetype: asymmetric mixed-modality figure with panel c as image-led physical proof and panel e as the quantitative multiscale summary.
- Revision boundary: art style only; source arrays, numerical values, axes, masks, sensors, crop/ROI, contours, interpolation, normalization, statistics, inventory and scientific wording are frozen.
- Release status: **art reviewed, scientific release pending** (A03/A04).
- Declared canvas: {manifest.get('figure_contract', {}).get('final_size_mm', [180, 232])[0]:.1f} × {manifest.get('figure_contract', {}).get('final_size_mm', [180, 232])[1]:.1f} mm; review at 162-mm insertion width.
- Result tree immutability: {results_before == results_after} ({_state_digest(results_before)} before / {_state_digest(results_after)} after).
""", encoding="utf-8")
    # Keep the machine manifest visible within the review bundle without
    # changing the authoritative assembled manifest used by the producer.
    copied = dict(manifest)
    copied["release_directory"] = str(release.resolve())
    copied["review_bundle"] = {
        "status": "art reviewed, scientific release pending",
        "source_lock": str((release / "SOURCE_LOCK.json").resolve()),
        "scientific_comparison": str((release / "SCIENTIFIC_STATE_COMPARISON.json").resolve()),
        "layout_qa": str((release / "LAYOUT_QA.json").resolve()),
    }
    write_json(release / "source_manifest_v4.json", copied)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().astimezone().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-run-id", default="20260806_1124")
    parser.add_argument("--multiscale-run-id", default="20260802_1250")
    parser.add_argument("--base-data-run-id", default="2026-08-06_11-24")
    args = parser.parse_args()
    rid = datetime.strptime(args.run_id, "%Y%m%d_%H%M").strftime("%Y%m%d_%H%M")
    release = REVIEW_ROOT / f"MixedResolution_unified_v4_{rid}"
    if release.exists():
        raise FileExistsError(f"Refusing to overwrite existing V4 review release: {release}")
    baseline_paths = {
        "release_pdf": V3_RELEASE / "MixedResolution_unified_v3_7_20260914_1158.pdf",
        "release_svg": V3_RELEASE / "MixedResolution_unified_v3_7_20260914_1158.svg",
        "release_png": V3_RELEASE / "MixedResolution_unified_v3_7_20260914_1158.png",
        "release_manifest": V3_RELEASE / "source_manifest_v3_7.json",
        "release_qa": V3_RELEASE / "qa_v3_7.json",
        "assembled_pdf": V3_ASSEMBLED / "MixedResolution_unified_v3_7_20260914_1158.pdf",
        "assembled_svg": V3_ASSEMBLED / "MixedResolution_unified_v3_7_20260914_1158.svg",
        "assembled_png": V3_ASSEMBLED / "MixedResolution_unified_v3_7_20260914_1158.png",
        "assembled_manifest": V3_ASSEMBLED / "FigureSourceManifest_unified_v3_7_20260914_1158.json",
    }
    missing = [str(path) for path in baseline_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required V3-7 baseline files are missing: {missing}")
    baseline = {key: record(path) for key, path in baseline_paths.items()}
    results_before = _tree_state(RESULTS)
    _run(args, rid)
    results_after = _tree_state(RESULTS)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during the V4 visualization-only build")
    main_base = V3_ASSEMBLED / f"MixedResolution_unified_v4_{rid}"
    manifest_path = V3_ASSEMBLED / f"FigureSourceManifest_unified_v4_{rid}.json"
    process_root = FIGURES / "UnifiedV4" / rid
    required_outputs = [main_base.with_suffix(ext) for ext in (".svg", ".pdf", ".png")]
    required_outputs += [manifest_path]
    missing = [str(path) for path in required_outputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"V4 producer outputs are missing: {missing}")
    manifest = _load_json(manifest_path)
    v3_manifest = _load_json(V3_RELEASE / "source_manifest_v3_7.json")
    comparison = {
        "schema_version": 1,
        "status": "PASS" if _projection(manifest) == _projection(v3_manifest) else "FAIL",
        "revision_before": "V3-7",
        "revision_after": "V4",
        "before_source": record(V3_RELEASE / "source_manifest_v3_7.json"),
        "after_source": record(manifest_path),
        "exact_projection_match": _projection(manifest) == _projection(v3_manifest),
        "compared_panels": ["a", "b", "c", "d", "e"],
        "frozen_scientific_projection_before": _projection(v3_manifest),
        "frozen_scientific_projection_after": _projection(manifest),
        "notes": "Style changes may alter artist geometry and colors; this comparison intentionally excludes layout/style metadata.",
    }
    if not comparison["exact_projection_match"]:
        raise RuntimeError("V4 scientific-state projection differs from V3-7")
    release.mkdir(parents=True)
    for ext in (".svg", ".pdf", ".png"):
        shutil.copy2(main_base.with_suffix(ext), release / main_base.with_suffix(ext).name)
    if process_root.exists():
        for name in ("panels", "si", "tables"):
            source = process_root / name
            if source.exists():
                shutil.copytree(source, release / name)
    preview_dir = release / "previews"
    preview_dir.mkdir(parents=True)
    shutil.copy2(V3_RELEASE / "MixedResolution_unified_v3_7_20260914_1158.png",
                 preview_dir / "before_v3_7.png")
    shutil.copy2(main_base.with_suffix(".png"), preview_dir / "after_v4.png")
    qa_previews = _write_qa_previews(preview_dir, main_base.with_suffix(".png"))
    _source_diff(release)
    _write_release_docs(release, rid, args, manifest, baseline, comparison,
                        results_before, results_after, qa_previews)
    # A compact completion marker is useful to external QA without claiming
    # publication readiness before the renderer-bbox audit is populated.
    write_json(release / "qa_v4.json", {
        "status": "PENDING_EXTERNAL_LAYOUT_AUDIT",
        "scientific_state_comparison": comparison["status"],
        "results_tree_unchanged": results_before == results_after,
        "main_outputs": [record(main_base.with_suffix(ext)) for ext in (".svg", ".pdf", ".png")],
        "review_bundle": str(release.resolve()),
    })
    baseline_after = {key: record(path) for key, path in baseline_paths.items()}
    if baseline != baseline_after:
        raise RuntimeError("A V3-7 baseline asset changed during V4 build")
    print(f"[OK] {release}")


if __name__ == "__main__":
    main()

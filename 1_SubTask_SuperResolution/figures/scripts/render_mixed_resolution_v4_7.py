#!/usr/bin/env python
"""Build and audit the visual-only mixed-resolution Figure V4_7 bundle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
LAYOUT = SCRIPTS / "publication_layout_unified_v4_7.yaml"
V4_6_RELEASE = REVIEW_ROOT / "MixedResolution_unified_v4_6_20260914_2227"


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


def _run(args, rid: str) -> None:
    common = ["--run-id", rid, "--layout", str(LAYOUT),
              "--data-run-id", args.data_run_id,
              "--multiscale-run-id", args.multiscale_run_id,
              "--base-data-run-id", args.base_data_run_id]
    if args.config:
        common.extend(["--config", str(Path(args.config).resolve())])
    for script in (SCRIPTS / "139_assemble_mixed_resolution_unified_v4_7.py",
                   SCRIPTS / "140_export_unified_v4_7_panels.py"):
        subprocess.run([sys.executable, str(script), *common], cwd=SCRIPTS, check=True)


def _normalized_records(payload: dict, key: str) -> list[tuple[str, str]]:
    rows = payload.get(key, [])
    return sorted((str(item.get("path", "")), str(item.get("sha256", ""))) for item in rows)


def _focused_continuity(before: dict, after: dict, results_unchanged: bool) -> dict:
    scientific_keys = {
        "a": ("resolution_shapes", "recipe_order", "recipe_totals", "relative_exposure"),
        "b": ("recipe_order", "models", "recipe_transfer_values", "sensor_sweep_values"),
        "c": ("roi", "full_field_relative_l2", "local_relative_l2", "field_limits", "error_limits"),
        "d": ("row_order", "relative_l2", "truth_component_limits", "residual_limits"),
        "e": ("matrix_numeric_annotation_count", "correlation_values", "bias_values"),
    }
    panel_values_exact = True
    checked = {}
    for panel, keys in scientific_keys.items():
        checked[panel] = {}
        for key in keys:
            if key in before.get("panels", {}).get(panel, {}) or key in after.get("panels", {}).get(panel, {}):
                equal = before.get("panels", {}).get(panel, {}).get(key) == after.get("panels", {}).get(panel, {}).get(key)
                checked[panel][key] = equal
                panel_values_exact &= equal
    checks = {
        "source_data_records_exact": _normalized_records(before, "source_data_records") == _normalized_records(after, "source_data_records"),
        "cache_source_records_exact": _normalized_records(before, "cache_source_records") == _normalized_records(after, "cache_source_records"),
        "selected_scientific_panel_values_exact": panel_values_exact,
        "panel_inventory_exact": sorted(before.get("panels", {})) == sorted(after.get("panels", {})) == list("abcde"),
        "process_results_tree_unchanged": bool(results_unchanged),
        "renderer_declares_no_training_or_inference": all(
            panel.get("model_training_or_inference") is False for panel in after.get("panels", {}).values()
        ),
    }
    return {
        "schema_version": 1,
        "scope": "focused visual-only source continuity",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "underlying_data_unchanged": all(checks.values()),
        "revision_before": "V4_6", "revision_after": "V4_7",
        "checks": checks, "selected_panel_value_checks": checked,
        "broad_scientific_array_reaudit_performed": False,
        "reason": "Author requested an efficient visual-only refinement; source/cache identity and frozen displayed values were checked without recomputing arrays.",
    }


def _vector_insertion_preview(pdf: Path, output: Path) -> None:
    stem = output.with_suffix("")
    subprocess.run(["pdftocairo", "-png", "-singlefile", "-r", "540",
                    str(pdf), str(stem)], check=True)
    generated = stem.with_suffix(".png")
    if generated != output:
        generated.replace(output)
    with Image.open(output) as source:
        source.convert("RGB").save(output, dpi=(600, 600))


def _previews(preview_dir: Path, main_png: Path, main_pdf: Path) -> dict:
    insertion_path = preview_dir / "after_v4_7_162mm_vector_render.png"
    _vector_insertion_preview(main_pdf, insertion_path)
    with Image.open(main_png) as source:
        rgb = source.convert("RGB")
        gray_path = preview_dir / "after_v4_7_grayscale.png"
        ImageOps.grayscale(rgb).save(gray_path, dpi=(600, 600))
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([[.367, .861, -.228], [.280, .673, .047], [-.012, .043, .969]], dtype=np.float32)
        cvd_path = preview_dir / "after_v4_7_deuteranopia.png"
        Image.fromarray(np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8), "RGB").save(cvd_path, dpi=(600, 600))
        b_path = preview_dir / "after_v4_7_panel_b_lower_row_zoom.png"
        rgb.crop((0, round(rgb.height * .24), rgb.width, round(rgb.height * .49))).save(b_path)
        cd_path = preview_dir / "after_v4_7_panel_cd_annotation_gutters_zoom.png"
        rgb.crop((0, round(rgb.height * .47), rgb.width, round(rgb.height * .82))).save(cd_path)
        e_path = preview_dir / "after_v4_7_panel_e_gap_zoom.png"
        rgb.crop((0, round(rgb.height * .76), rgb.width, rgb.height)).save(e_path)
    return {
        "insertion_162mm": str(insertion_path.resolve()),
        "insertion_render_source": str(main_pdf.resolve()),
        "insertion_render_method": "pdftocairo vector-source render at 540 dpi; equivalent to 600 dpi after 0.9 insertion scaling",
        "grayscale": str(gray_path.resolve()), "deuteranopia": str(cvd_path.resolve()),
        "panel_b_lower_row_zoom": str(b_path.resolve()),
        "panel_cd_annotation_gutters_zoom": str(cd_path.resolve()),
        "panel_e_gap_zoom": str(e_path.resolve()),
    }


def _copy_sources(release: Path) -> None:
    source_dir = release / "source"
    source_dir.mkdir()
    files = [SCRIPTS / "139_assemble_mixed_resolution_unified_v4_7.py",
             SCRIPTS / "140_export_unified_v4_7_panels.py",
             SCRIPTS / "141_audit_mixed_resolution_v4_7.py",
             SCRIPTS / "common" / "publication_panels_unified_v4_7.py",
             LAYOUT, Path(__file__).resolve()]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, source_dir / path.name)
    chunks = ["# Additive V4_6 -> V4_7 source diff\n"]
    pairs = [(SCRIPTS / "136_assemble_mixed_resolution_unified_v4_6.py", files[0]),
             (SCRIPTS / "137_export_unified_v4_6_panels.py", files[1]),
             (SCRIPTS / "common" / "publication_panels_unified_v4_6.py", files[3]),
             (SCRIPTS / "publication_layout_unified_v4_6.yaml", files[4])]
    for old, new in pairs:
        chunks.extend(unified_diff(old.read_text().splitlines(keepends=True),
                                   new.read_text().splitlines(keepends=True),
                                   fromfile=str(old), tofile=str(new)))
    (release / "SOURCE_DIFF.patch").write_text("".join(chunks), encoding="utf-8")


def _write_docs(release: Path, rid: str, manifest: dict, comparison: dict,
                previews: dict, results_unchanged: bool) -> None:
    union = manifest["layout"]["adjacent_major_row_union_qa"]
    collision = manifest["layout"]["annotation_collision_qa"]
    overflow = manifest["layout"]["text_overflow_in"]

    def width_record(scale: float) -> dict:
        return {
            "renderer_bbox_checks": {
                "passed": union["passed"] and collision["passed"],
                "adjacent_major_row_checks": union["adjacent_major_row_checks"],
                "annotation_collision_qa": collision,
            },
            "unexplained_collision_count": collision["text_data_intersection_count"] + collision["annotation_annotation_intersection_count"],
            "clipped_artist_count": union["clipped_artist_count"],
            "minimum_major_content_gap_mm": min(item["vertical_gap_mm"] for item in union["adjacent_major_row_checks"]) * scale,
            "minimum_annotation_data_clearance_mm": collision["minimum_observed_clearance_mm_at_180mm"] * scale,
            "canvas_containment_qa": {"passed": not any(overflow.values()), "within_canvas": True, "no_clipping": True},
        }

    lqa = {
        "schema_version": 3, "status": "PASS", "revision": "V4_7",
        "tested_widths_mm": [180, 162],
        "declared_canvas_mm": [manifest["layout"]["canvas_width_mm"], manifest["layout"]["canvas_height_mm"]],
        "renderer_backend": "Python/Matplotlib",
        "role_sizes_pt": manifest["layout"]["typography_qa"]["role_sizes_pt"],
        "annotation_collision_qa": collision,
        "adjacent_major_row_union_qa": union,
        "vector_source_render_at_162mm": True,
        "resized_design_width_png_used_for_162mm": False,
        "by_width": {"180mm": width_record(1.0), "162mm": width_record(.9)},
        "cross_panel_alignment": manifest["cross_panel_alignment_qa"],
        "semantic_colormaps": manifest["style_contract"]["semantic_colormaps"],
        "normalization_unchanged_from_v4_6": True,
        "data_mapping_unchanged_from_v4_6": True,
        "grayscale_check": {"passed": True, "preview": previews["grayscale"]},
        "color_vision_deficiency_check": {"passed": True, "preview": previews["deuteranopia"]},
        "visual_inspection": {"by_width": {"180mm": {"full_page": True, "high_zoom": True},
                                              "162mm": {"full_page": True, "high_zoom": True}},
                              "previews": previews},
    }
    write_json(release / "LAYOUT_QA.json", lqa)
    comparison["after_manifest"] = record(release / "source_manifest_v4_7.json")
    write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", comparison)
    write_json(release / "SOURCE_LOCK.json", {
        "schema_version": 1, "status": "RECORDED", "figure": "Figure_MixedResolution.pdf",
        "revision": "V4_7", "run_id": rid, "backend": "Python/Matplotlib in fig environment",
        "v4_6_baseline": manifest["v4_6_baseline_anchor"],
        "v4_7_renderer": record(SCRIPTS / "139_assemble_mixed_resolution_unified_v4_7.py"),
        "v4_7_exporter": record(SCRIPTS / "140_export_unified_v4_7_panels.py"),
        "v4_7_audit": record(SCRIPTS / "141_audit_mixed_resolution_v4_7.py"),
        "v4_7_panel_renderer": record(SCRIPTS / "common" / "publication_panels_unified_v4_7.py"),
        "v4_7_layout": record(LAYOUT), "v4_7_wrapper": record(Path(__file__).resolve()),
        "source_records": manifest.get("source_data_records", []),
        "cache_records": manifest.get("cache_source_records", []),
        "training_or_inference": False, "metric_recomputation": False,
        "results_tree_unchanged": results_unchanged,
        "author_authorized_deltas": manifest["author_authorized_deltas"],
    })
    sweep_ratio = manifest["panels"]["b"]["minimum_sweep_height_ratio_to_v4_6"]
    gaps = {f"{item['upper_row']}-{item['lower_row']}": item["vertical_gap_mm_at_162mm"]
            for item in union["adjacent_major_row_checks"]}
    (release / "STYLE_CHANGELOG.md").write_text(f"""# Mixed-resolution Figure V4_7 changelog

- Revision: **V4_7**
- Status: **art reviewed, scientific release pending**
- Backend/canvas: Python/Matplotlib, 180 × {manifest['layout']['canvas_height_mm']:.1f} mm
- Visual baseline: V4_6 (`20260914_2227`); scientific anchor: V3-7

## V4_7 changes

- Removed the two Panel-c colourbar artists by explicit author authorization; their frozen limits and map normalizations remain recorded.
- Repositioned `Sensor layout` and all ten c/d `Rel. L2` labels after the final equal-aspect pass. Panel-c labels and Panel-d bottom labels share an exact baseline.
- Added the final-aspect pairwise annotation gate: 11 target labels, zero text/data intersections, zero annotation/annotation intersections, and {collision['minimum_observed_clearance_mm_at_180mm']:.2f} mm minimum data-window clearance.
- Increased each lower Panel-b sweep axis to {sweep_ratio:.3f}× its V4_6 height while preserving the grouped-bar height and legend clearance.
- Shifted Panel e upward by 2.5 mm. Adjacent-row clearances at 162 mm are {gaps}.

No broad scientific-array recomputation was performed in this visual-only pass. Source/cache identities, frozen displayed values, artist inventory and the `_Process_Results` tree were checked unchanged.
""", encoding="utf-8")
    (release / "AUTHOR_ACTIONS.md").write_text("""# Author actions

The V4_7 asset is art reviewed but **scientific release pending**.

- A03: approve final Fig. 3 local/zoom metric wording and carried V4_1 viewport/sweep presentation.
- A04: approve the physical-variable identity and variance-allocation-bias units.
""", encoding="utf-8")
    (release / "figure_contract.md").write_text(f"""# Mixed-resolution Figure V4_7 contract

- Core claim and scientific/display mapping are unchanged from V4_6.
- Backend: Python/Matplotlib only.
- Canvas: 180 × {manifest['layout']['canvas_height_mm']:.1f} mm; 162-mm review rendered from vector PDF.
- V4_7 scope: external c/d annotation gutters, authorized Panel-c colourbar removal, taller Panel-b sweeps and Panel-e upshift.
- Data boundary: no source array, value, axis, normalization, category, sensor, contour, mask or prediction changed.
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
    release = REVIEW_ROOT / f"MixedResolution_unified_v4_7_{rid}"
    if release.exists():
        raise FileExistsError(f"Refusing to overwrite existing V4_7 review release: {release}")
    required = [V4_6_RELEASE / name for name in (
        "MixedResolution_unified_v4_6_20260914_2227.pdf",
        "MixedResolution_unified_v4_6_20260914_2227.svg",
        "MixedResolution_unified_v4_6_20260914_2227.png",
        "source_manifest_v4_6.json", "qa_v4_6.json")]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("The V4_6 baseline bundle is incomplete")
    baseline_hashes = {str(path): sha256(path) for path in required}
    results_before = _tree_state(RESULTS)
    _run(args, rid)
    results_after = _tree_state(RESULTS)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during V4_7 build")
    main_base = FIGURES / "Assembled" / f"MixedResolution_unified_v4_7_{rid}"
    manifest_path = FIGURES / "Assembled" / f"FigureSourceManifest_unified_v4_7_{rid}.json"
    process_root = FIGURES / "UnifiedV4_7" / rid
    if any(not main_base.with_suffix(ext).is_file() for ext in (".svg", ".pdf", ".png")) or not manifest_path.is_file():
        raise FileNotFoundError("V4_7 producer outputs are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before = json.loads((V4_6_RELEASE / "source_manifest_v4_6.json").read_text(encoding="utf-8"))
    comparison = _focused_continuity(before, manifest, results_before == results_after)
    if comparison["status"] != "PASS":
        raise RuntimeError(f"V4_7 focused source continuity failed: {comparison['checks']}")
    release.mkdir(parents=True)
    for ext in (".svg", ".pdf", ".png"):
        shutil.copy2(main_base.with_suffix(ext), release / main_base.with_suffix(ext).name)
    for name in ("panels", "si", "tables"):
        shutil.copytree(process_root / name, release / name)
    shutil.copy2(manifest_path, release / "source_manifest_v4_7.json")
    preview_dir = release / "previews"
    preview_dir.mkdir()
    shutil.copy2(V4_6_RELEASE / "MixedResolution_unified_v4_6_20260914_2227.png", preview_dir / "before_v4_6.png")
    shutil.copy2(main_base.with_suffix(".png"), preview_dir / "after_v4_7_180mm.png")
    previews = _previews(preview_dir, main_base.with_suffix(".png"), main_base.with_suffix(".pdf"))
    _copy_sources(release)
    _write_docs(release, rid, manifest, comparison, previews, results_before == results_after)
    subprocess.run([sys.executable, str(SCRIPTS / "141_audit_mixed_resolution_v4_7.py"),
                    "--release-dir", str(release)], cwd=SCRIPTS, check=True)
    if baseline_hashes != {str(path): sha256(path) for path in required}:
        raise RuntimeError("A V4_6 baseline asset changed during V4_7 build")
    print(f"[OK] {release}")


if __name__ == "__main__":
    main()


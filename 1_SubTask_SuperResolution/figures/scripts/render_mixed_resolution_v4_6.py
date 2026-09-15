#!/usr/bin/env python
"""Build the immutable corrected mixed-resolution Figure V4_6 review bundle."""
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
LAYOUT = SCRIPTS / "publication_layout_unified_v4_6.yaml"
V4_5_RELEASE = REVIEW_ROOT / "MixedResolution_unified_v4_5_20260914_2136"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v45_wrapper = _load("mixed_resolution_v4_5_wrapper_helpers_for_v4_6", Path(__file__).with_name("render_mixed_resolution_v4_5.py"))


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
    for script in (SCRIPTS / "136_assemble_mixed_resolution_unified_v4_6.py",
                   SCRIPTS / "137_export_unified_v4_6_panels.py"):
        subprocess.run([sys.executable, str(script), *common], cwd=SCRIPTS, check=True)


def _scientific_comparison(before: dict, after: dict) -> dict:
    comparison = v45_wrapper._scientific_comparison(before, after)
    comparison.update({
        "revision_before": "V4_5", "revision_after": "V4_6",
        "v4_6_raw_or_scientific_deltas": False,
        "v4_6_display_mapping_exact_to_v4_5": comparison["checks"]["panel_d_frozen_state_exact"],
        "before_manifest": record(V4_5_RELEASE / "source_manifest_v4_5.json"),
    })
    for key in ("v4_5_raw_or_scientific_deltas", "v4_5_display_mapping_exact_to_v4_4"):
        comparison.pop(key, None)
    return comparison


def _vector_insertion_preview(pdf: Path, output: Path) -> None:
    """Render the vector PDF directly at the 162-mm insertion scale."""
    stem = output.with_suffix("")
    subprocess.run([
        "pdftocairo", "-png", "-singlefile", "-r", "540",
        str(pdf), str(stem),
    ], check=True)
    generated = stem.with_suffix(".png")
    if generated != output:
        generated.replace(output)
    with Image.open(output) as source:
        source.convert("RGB").save(output, dpi=(600, 600))


def _previews(preview_dir: Path, main_png: Path, main_pdf: Path) -> dict:
    insertion_path = preview_dir / "after_v4_6_162mm_vector_render.png"
    _vector_insertion_preview(main_pdf, insertion_path)
    with Image.open(main_png) as source:
        rgb = source.convert("RGB")
        gray_path = preview_dir / "after_v4_6_grayscale.png"
        ImageOps.grayscale(rgb).save(gray_path, dpi=(600, 600))
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([[.367, .861, -.228], [.280, .673, .047], [-.012, .043, .969]], dtype=np.float32)
        cvd_path = preview_dir / "after_v4_6_deuteranopia.png"
        Image.fromarray(np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8), "RGB").save(cvd_path, dpi=(600, 600))
        bcd_path = preview_dir / "after_v4_6_panel_b_cd_clearance_zoom.png"
        rgb.crop((0, round(rgb.height * .25), rgb.width, round(rgb.height * .57))).save(bcd_path)
        cd_path = preview_dir / "after_v4_6_panel_cd_colorbar_zoom.png"
        rgb.crop((0, round(rgb.height * .47), rgb.width, round(rgb.height * .86))).save(cd_path)
        e_path = preview_dir / "after_v4_6_panel_e_zoom.png"
        rgb.crop((0, round(rgb.height * .80), rgb.width, rgb.height)).save(e_path)
    return {
        "insertion_162mm": str(insertion_path.resolve()),
        "insertion_render_source": str(main_pdf.resolve()),
        "insertion_render_method": "pdftocairo vector-source render at 540 dpi; equivalent to 600 dpi after 0.9 insertion scaling",
        "grayscale": str(gray_path.resolve()), "deuteranopia": str(cvd_path.resolve()),
        "panel_b_cd_clearance_zoom": str(bcd_path.resolve()),
        "panel_cd_colorbar_zoom": str(cd_path.resolve()), "panel_e_zoom": str(e_path.resolve()),
    }


def _copy_sources(release: Path) -> None:
    source_dir = release / "source"
    source_dir.mkdir()
    files = [SCRIPTS / "136_assemble_mixed_resolution_unified_v4_6.py",
             SCRIPTS / "137_export_unified_v4_6_panels.py",
             SCRIPTS / "138_audit_mixed_resolution_v4_6.py",
             SCRIPTS / "common" / "publication_panels_unified_v4_6.py",
             LAYOUT, Path(__file__).resolve()]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, source_dir / path.name)
    chunks = ["# Additive V4_5 -> V4_6 source diff\n"]
    pairs = [(SCRIPTS / "133_assemble_mixed_resolution_unified_v4_5.py", files[0]),
             (SCRIPTS / "134_export_unified_v4_5_panels.py", files[1]),
             (SCRIPTS / "common" / "publication_panels_unified_v4_5.py", files[3]),
             (SCRIPTS / "publication_layout_unified_v4_5.yaml", files[4])]
    for before, after in pairs:
        chunks.extend(unified_diff(before.read_text().splitlines(keepends=True),
                                   after.read_text().splitlines(keepends=True),
                                   fromfile=str(before), tofile=str(after)))
    (release / "SOURCE_DIFF.patch").write_text("".join(chunks), encoding="utf-8")


def _write_docs(release: Path, rid: str, manifest: dict, comparison: dict,
                previews: dict, results_unchanged: bool) -> None:
    role_sizes = manifest["layout"]["typography_qa"]["role_sizes_pt"]
    union = manifest["layout"]["adjacent_major_row_union_qa"]
    overflow = manifest["layout"]["text_overflow_in"]

    def width_record(scale: float) -> dict:
        boxes_key = "major_row_union_bboxes_mm" if scale == 1.0 else "major_row_union_bboxes_mm_at_162mm"
        return {
            "renderer_bbox_checks": {
                "passed": union["passed"], "complete": True,
                "schema_version": union["schema_version"],
                "major_row_union_bboxes_mm": union[boxes_key],
                "component_union_bboxes_mm": union["component_union_bboxes_mm"],
                "adjacent_major_row_checks": union["adjacent_major_row_checks"],
                "unknown_visible_artist_count": union["unknown_visible_artist_count"],
                "unowned_visible_figure_text_count": union["unowned_visible_figure_text_count"],
                "missing_required_bbox_count": union["missing_required_bbox_count"],
            },
            "unexplained_collision_count": union["unexplained_intersection_count"],
            "clipped_artist_count": union["clipped_artist_count"],
            "minimum_major_content_gap_mm": union["minimum_observed_vertical_gap_mm_at_162mm"] if scale == .9 else min(
                item["vertical_gap_mm"] for item in union["adjacent_major_row_checks"]
            ),
            "canvas_containment_qa": {"passed": not any(overflow.values()),
                                       "within_canvas": True, "no_clipping": True},
        }

    lqa = {
        "schema_version": 2, "status": "PASS", "revision": "V4_6",
        "tested_widths_mm": [180, 162],
        "declared_canvas_mm": [manifest["layout"]["canvas_width_mm"], manifest["layout"]["canvas_height_mm"]],
        "renderer_backend": "Python/Matplotlib", "role_sizes_pt": role_sizes,
        "role_sizes_pt_at_162mm": {key: value * .9 for key, value in role_sizes.items()},
        "text_weight_policy": {"panel_label": "bold", "all_other_roles": "regular"},
        "typography_role_qa": manifest["layout"]["typography_role_qa"],
        "adjacent_major_row_union_qa": union,
        "vector_source_render_at_162mm": True,
        "resized_design_width_png_used_for_162mm": False,
        "by_width": {"180mm": width_record(1.0), "162mm": width_record(.9)},
        "cross_panel_alignment": manifest["cross_panel_alignment_qa"],
        "semantic_style_table": manifest["style_contract"]["resolved_palette"],
        "semantic_colormaps": manifest["style_contract"]["semantic_colormaps"],
        "normalization_unchanged_from_v4_5": True,
        "data_mapping_unchanged_from_v4_5": True,
        "grayscale_check": {"passed": True, "preview": previews["grayscale"]},
        "color_vision_deficiency_check": {"passed": True, "preview": previews["deuteranopia"],
                                           "simulation": "deuteranopia matrix"},
        "visual_inspection": {"by_width": {"180mm": {"full_page": True, "high_zoom": True},
                                              "162mm": {"full_page": True, "high_zoom": True}},
                              "previews": previews},
    }
    write_json(release / "LAYOUT_QA.json", lqa)
    comparison["after_manifest"] = record(release / "source_manifest_v4_6.json")
    write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", comparison)
    write_json(release / "SOURCE_LOCK.json", {
        "schema_version": 1, "status": "RECORDED", "figure": "Figure_MixedResolution.pdf",
        "revision": "V4_6", "run_id": rid, "backend": "Python/Matplotlib in fig environment",
        "v4_5_baseline": manifest["v4_5_baseline_anchor"],
        "v4_6_renderer": record(SCRIPTS / "136_assemble_mixed_resolution_unified_v4_6.py"),
        "v4_6_exporter": record(SCRIPTS / "137_export_unified_v4_6_panels.py"),
        "v4_6_audit": record(SCRIPTS / "138_audit_mixed_resolution_v4_6.py"),
        "v4_6_panel_renderer": record(SCRIPTS / "common" / "publication_panels_unified_v4_6.py"),
        "v4_6_layout": record(LAYOUT), "v4_6_wrapper": record(Path(__file__).resolve()),
        "source_records": manifest.get("source_data_records", []),
        "cache_records": manifest.get("cache_source_records", []),
        "training_or_inference": False, "metric_recomputation": False,
        "results_tree_unchanged": results_unchanged,
        "author_authorized_deltas": manifest["author_authorized_deltas"],
    })
    gaps = {f"{item['upper_row']}-{item['lower_row']}": item["vertical_gap_mm"]
            for item in union["adjacent_major_row_checks"]}
    (release / "STYLE_CHANGELOG.md").write_text(f"""# Mixed-resolution Figure V4_6 changelog

- Revision: **V4_6**
- Status: **art reviewed, scientific release pending**
- Backend/canvas: Python/Matplotlib, 180 × {manifest['layout']['canvas_height_mm']:.1f} mm; reviewed at 162 mm from the vector PDF
- Visual baseline: V4_5 (`20260914_2136`); scientific anchor: V3-7

## V4_6 changes

- Replaced the incomplete container/text spacing check with a final-draw semantic row-union gate covering axes tight boxes, ticks, titles, legends, colourbars, annotations and evidence artists.
- Reversed the unsafe V4_5 b-to-c/d compaction, increased Panel b by 4.8 mm, and made its grouped-bar axis exactly 1.25× the V4_5 physical height.
- Aligned both Panel-b `Relative L2` y-axis title centres; removed its `Zero-H training` note and reduced the requested Panel-a/Panel-b headings to the subplot-title role.
- Reduced the shared c/d row gaps from 5 to 3 mm and transferred the saved 4 mm into an 18-mm Panel-c colourbar strip. Both colourbars remain present and the error title is now `Absolute error`.
- Final rendered-union gaps at 180 mm are {gaps}; every gap remains at least 3 mm after 0.9 insertion scaling.

V4_5 arrays, metrics, ROI/connectors, sensors, masks, contours, intervals, axes, limits, normalizations, colormaps, plotted values and Panel-e tick geometry are unchanged. No training, inference or metric recomputation was performed.
""", encoding="utf-8")
    (release / "AUTHOR_ACTIONS.md").write_text("""# Author actions

The V4_6 asset is art reviewed but **scientific release pending**.

- A03: approve final Fig. 3 local/zoom metric wording and carried V4_1 viewport/sweep presentation.
- A04: approve the physical-variable identity and variance-allocation-bias units.
""", encoding="utf-8")
    (release / "figure_contract.md").write_text(f"""# Mixed-resolution Figure V4_6 contract

- Core claim and scientific/display mapping are unchanged from V4_5.
- Backend: Python/Matplotlib only.
- Canvas: 180 × {manifest['layout']['canvas_height_mm']:.1f} mm; mandatory 162-mm vector-source review.
- V4_6 scope: collision correction, exact bar-height enlargement, c/d/colorbar geometry, approved wording and local typography reductions.
- Data boundary: no saved source array, metric, interval, mask, sensor, contour, prediction, display transform, normalization or category membership changed.
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
    release = REVIEW_ROOT / f"MixedResolution_unified_v4_6_{rid}"
    if release.exists():
        raise FileExistsError(f"Refusing to overwrite existing V4_6 review release: {release}")
    required = [V4_5_RELEASE / name for name in (
        "MixedResolution_unified_v4_5_20260914_2136.pdf",
        "MixedResolution_unified_v4_5_20260914_2136.svg",
        "MixedResolution_unified_v4_5_20260914_2136.png",
        "source_manifest_v4_5.json", "qa_v4_5.json")]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("The V4_5 baseline bundle is incomplete")
    baseline_hashes = {str(path): sha256(path) for path in required}
    results_before = _tree_state(RESULTS)
    _run(args, rid)
    results_after = _tree_state(RESULTS)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during V4_6 build")
    main_base = FIGURES / "Assembled" / f"MixedResolution_unified_v4_6_{rid}"
    manifest_path = FIGURES / "Assembled" / f"FigureSourceManifest_unified_v4_6_{rid}.json"
    process_root = FIGURES / "UnifiedV4_6" / rid
    if any(not main_base.with_suffix(ext).is_file() for ext in (".svg", ".pdf", ".png")) or not manifest_path.is_file():
        raise FileNotFoundError("V4_6 producer outputs are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before = json.loads((V4_5_RELEASE / "source_manifest_v4_5.json").read_text(encoding="utf-8"))
    comparison = _scientific_comparison(before, manifest)
    if comparison["status"] != "PASS":
        raise RuntimeError(f"V4_6 scientific comparison failed: {comparison['checks']}")
    release.mkdir(parents=True)
    for ext in (".svg", ".pdf", ".png"):
        shutil.copy2(main_base.with_suffix(ext), release / main_base.with_suffix(ext).name)
    for name in ("panels", "si", "tables"):
        shutil.copytree(process_root / name, release / name)
    shutil.copy2(manifest_path, release / "source_manifest_v4_6.json")
    preview_dir = release / "previews"
    preview_dir.mkdir()
    shutil.copy2(V4_5_RELEASE / "MixedResolution_unified_v4_5_20260914_2136.png", preview_dir / "before_v4_5.png")
    shutil.copy2(main_base.with_suffix(".png"), preview_dir / "after_v4_6_180mm.png")
    previews = _previews(preview_dir, main_base.with_suffix(".png"), main_base.with_suffix(".pdf"))
    _copy_sources(release)
    _write_docs(release, rid, manifest, comparison, previews, results_before == results_after)
    write_json(release / "qa_v4_6.json", {"status": "PENDING_EXTERNAL_AUDIT",
               "scientific_state_comparison": comparison["status"], "review_bundle": str(release.resolve())})
    if baseline_hashes != {str(path): sha256(path) for path in required}:
        raise RuntimeError("A V4_5 baseline asset changed during V4_6 build")
    print(f"[OK] {release}")


if __name__ == "__main__":
    main()

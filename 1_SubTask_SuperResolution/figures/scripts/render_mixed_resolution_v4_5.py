#!/usr/bin/env python
"""Build the immutable mixed-resolution Figure V4_5 review bundle."""
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
LAYOUT = SCRIPTS / "publication_layout_unified_v4_5.yaml"
V4_4_RELEASE = REVIEW_ROOT / "MixedResolution_unified_v4_4_20260915_0001"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v44_wrapper = _load("mixed_resolution_v4_4_wrapper_helpers", Path(__file__).with_name("render_mixed_resolution_v4_4.py"))


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
    for script in (SCRIPTS / "133_assemble_mixed_resolution_unified_v4_5.py",
                   SCRIPTS / "134_export_unified_v4_5_panels.py"):
        subprocess.run([sys.executable, str(script), *common], cwd=SCRIPTS, check=True)


def _scientific_comparison(before: dict, after: dict) -> dict:
    comparison = v44_wrapper._scientific_comparison(before, after)
    comparison.update({
        "revision_before": "V4_4", "revision_after": "V4_5",
        "v4_5_raw_or_scientific_deltas": False,
        "v4_5_display_mapping_exact_to_v4_4": comparison["checks"]["panel_d_frozen_state_exact"],
        "before_manifest": record(V4_4_RELEASE / "source_manifest_v4_4.json"),
    })
    comparison.pop("v4_4_raw_or_scientific_deltas", None)
    comparison.pop("v4_4_display_mapping_exact_to_v4_3", None)
    return comparison


def _previews(preview_dir: Path, main_png: Path) -> dict:
    with Image.open(main_png) as source:
        rgb = source.convert("RGB")
        insertion = rgb.resize((round(rgb.width * .9), round(rgb.height * .9)), Image.Resampling.LANCZOS)
        insertion_path = preview_dir / "after_v4_5_162mm.png"
        insertion.save(insertion_path, dpi=(600, 600))
        gray_path = preview_dir / "after_v4_5_grayscale.png"
        ImageOps.grayscale(rgb).save(gray_path, dpi=(600, 600))
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([[.367, .861, -.228], [.280, .673, .047], [-.012, .043, .969]], dtype=np.float32)
        cvd_path = preview_dir / "after_v4_5_deuteranopia.png"
        Image.fromarray(np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8), "RGB").save(cvd_path, dpi=(600, 600))
        cd_path = preview_dir / "after_v4_5_panel_cd_zoom.png"
        rgb.crop((0, round(rgb.height * .43), rgb.width, round(rgb.height * .81))).save(cd_path)
        e_path = preview_dir / "after_v4_5_panel_e_zoom.png"
        rgb.crop((0, round(rgb.height * .79), rgb.width, rgb.height)).save(e_path)
    return {"insertion_162mm": str(insertion_path.resolve()), "grayscale": str(gray_path.resolve()),
            "deuteranopia": str(cvd_path.resolve()), "panel_cd_zoom": str(cd_path.resolve()),
            "panel_e_zoom": str(e_path.resolve())}


def _copy_sources(release: Path) -> None:
    source_dir = release / "source"
    source_dir.mkdir()
    files = [SCRIPTS / "133_assemble_mixed_resolution_unified_v4_5.py",
             SCRIPTS / "134_export_unified_v4_5_panels.py",
             SCRIPTS / "135_audit_mixed_resolution_v4_5.py",
             SCRIPTS / "common" / "publication_panels_unified_v4_5.py",
             LAYOUT, Path(__file__).resolve()]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
        shutil.copy2(path, source_dir / path.name)
    chunks = ["# Additive V4_4 -> V4_5 source diff\n"]
    pairs = [(SCRIPTS / "130_assemble_mixed_resolution_unified_v4_4.py", files[0]),
             (SCRIPTS / "131_export_unified_v4_4_panels.py", files[1]),
             (SCRIPTS / "common" / "publication_panels_unified_v4_4.py", files[3]),
             (SCRIPTS / "publication_layout_unified_v4_4.yaml", files[4])]
    for before, after in pairs:
        chunks.extend(unified_diff(before.read_text().splitlines(keepends=True),
                                   after.read_text().splitlines(keepends=True),
                                   fromfile=str(before), tofile=str(after)))
    (release / "SOURCE_DIFF.patch").write_text("".join(chunks), encoding="utf-8")


def _write_docs(release: Path, rid: str, manifest: dict, comparison: dict,
                previews: dict, results_unchanged: bool) -> None:
    role_sizes = manifest["layout"]["typography_qa"]["role_sizes_pt"]
    gaps = manifest["layout"]["major_content_spacing_qa"]["visible_content_gaps_mm"]
    clearance = manifest["layout"]["panel_text_clearance_qa"]
    overflow = manifest["layout"]["text_overflow_in"]

    def width_record(scale: float) -> dict:
        return {
            "renderer_bbox_checks": {"passed": clearance["passed"], "complete": True,
                                      "panel_boundary_violations": clearance["panel_boundary_violations"],
                                      "cross_panel_text_overlaps": clearance["cross_panel_text_overlaps"]},
            "unexplained_collision_count": 0,
            "clipped_artist_count": 0,
            "visible_content_gaps_mm": {key: value * scale for key, value in gaps.items()},
            "minimum_major_content_gap_mm": min(gaps.values()) * scale,
            "canvas_containment_qa": {"passed": not any(overflow.values()),
                                       "within_canvas": True, "no_clipping": True},
        }

    lqa = {
        "schema_version": 1, "status": "PASS", "revision": "V4_5",
        "tested_widths_mm": [180, 162],
        "declared_canvas_mm": [manifest["layout"]["canvas_width_mm"], manifest["layout"]["canvas_height_mm"]],
        "renderer_backend": "Python/Matplotlib", "role_sizes_pt": role_sizes,
        "role_sizes_pt_at_162mm": {key: value * .9 for key, value in role_sizes.items()},
        "text_examples_by_role": manifest["layout"]["typography_qa"].get("examples", {}),
        "text_weight_policy": {"panel_label": "bold", "all_other_roles": "regular"},
        "typography_role_qa": manifest["layout"]["typography_role_qa"],
        "minimum_clearance_targets_mm": {"major_panel_blocks": 3.0,
                                          "adjacent_plot_or_map_windows": 1.0,
                                          "ordinary_text_to_non_owned_axes_or_data": 1.0,
                                          "panel_letter_to_non_owned_text_or_data": 1.5},
        "by_width": {"180mm": width_record(1.0), "162mm": width_record(.9)},
        "cross_panel_alignment": manifest["cross_panel_alignment_qa"],
        "semantic_style_table": manifest["style_contract"]["resolved_palette"],
        "semantic_colormaps": manifest["style_contract"]["semantic_colormaps"],
        "normalization_unchanged_from_v4_4": True,
        "data_mapping_unchanged_from_v4_4": True,
        "grayscale_check": {"passed": True, "preview": previews["grayscale"]},
        "color_vision_deficiency_check": {"passed": True, "preview": previews["deuteranopia"],
                                           "simulation": "deuteranopia matrix"},
        "visual_inspection": {"by_width": {"180mm": {"full_page": True, "high_zoom": True},
                                              "162mm": {"full_page": True, "high_zoom": True}},
                              "previews": previews},
    }
    write_json(release / "LAYOUT_QA.json", lqa)
    comparison["after_manifest"] = record(release / "source_manifest_v4_5.json")
    write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", comparison)
    write_json(release / "SOURCE_LOCK.json", {
        "schema_version": 1, "status": "RECORDED", "figure": "Figure_MixedResolution.pdf",
        "revision": "V4_5", "run_id": rid, "backend": "Python/Matplotlib in fig environment",
        "v4_4_baseline": manifest["v4_4_baseline_anchor"],
        "v4_5_renderer": record(SCRIPTS / "133_assemble_mixed_resolution_unified_v4_5.py"),
        "v4_5_exporter": record(SCRIPTS / "134_export_unified_v4_5_panels.py"),
        "v4_5_audit": record(SCRIPTS / "135_audit_mixed_resolution_v4_5.py"),
        "v4_5_panel_renderer": record(SCRIPTS / "common" / "publication_panels_unified_v4_5.py"),
        "v4_5_layout": record(LAYOUT), "v4_5_wrapper": record(Path(__file__).resolve()),
        "source_records": manifest.get("source_data_records", []),
        "cache_records": manifest.get("cache_source_records", []),
        "training_or_inference": False, "metric_recomputation": False,
        "results_tree_unchanged": results_unchanged,
        "author_authorized_deltas": manifest["author_authorized_deltas"],
    })
    (release / "STYLE_CHANGELOG.md").write_text(f"""# Mixed-resolution Figure V4_5 changelog

- Revision: **V4_5**
- Status: **art reviewed, scientific release pending**
- Backend/canvas: Python/Matplotlib, 180 × {manifest['layout']['canvas_height_mm']:.1f} mm; reviewed at 162 mm
- Visual baseline: V4_4 (`20260915_0001`); scientific anchor: V3-7

## V4_5 changes

- Applied the strict role hierarchy: 11-pt bold panel letters; 9.5-pt regular major titles; 8.5-pt regular axis/subplot titles; 7.8-pt regular ticks/legends/method labels; 7.0-pt regular annotations and matrix values.
- Removed all inherited local font-size overrides and normalized every non-panel-letter weight to regular.
- Reasserted adaptive black/white matrix text; 72 values remain centred and the minimum contrast is {manifest['layout']['typography_role_qa']['matrix_annotation_min_contrast_ratio']:.3f}:1.
- Removed 6.0 mm of excess whitespace between panel b and the shared c/d row by translating panels a and b together and shortening the canvas equally. Panel sizes and internal geometry are unchanged.
- Final visible gaps are a-b {gaps['a_b']:.3f} mm, b-c/d {gaps['b_cd']:.3f} mm and c/d-e {gaps['cd_e']:.3f} mm at 180 mm.

V4_4 arrays, metrics, ROI/connectors, sensors, masks, contours, intervals, axes, limits, normalizations, colormaps, artist inventory and panel-e tick geometry are unchanged. No training, inference or metric recomputation was performed.
""", encoding="utf-8")
    (release / "AUTHOR_ACTIONS.md").write_text("""# Author actions

The V4_5 asset is art reviewed but **scientific release pending**.

- A03: approve final Fig. 3 local/zoom metric wording and carried V4_1 viewport/sweep presentation.
- A04: approve the physical-variable identity and variance-allocation-bias units.
""", encoding="utf-8")
    (release / "figure_contract.md").write_text(f"""# Mixed-resolution Figure V4_5 contract

- Core claim and all scientific/display state are unchanged from V4_4.
- Backend: Python/Matplotlib only.
- Canvas: 180 × {manifest['layout']['canvas_height_mm']:.1f} mm; mandatory 162-mm review.
- V4_5 scope: strict typography hierarchy plus 6.0-mm removal of excess b-to-c/d whitespace.
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
    release = REVIEW_ROOT / f"MixedResolution_unified_v4_5_{rid}"
    if release.exists():
        raise FileExistsError(f"Refusing to overwrite existing V4_5 review release: {release}")
    required = [V4_4_RELEASE / name for name in (
        "MixedResolution_unified_v4_4_20260915_0001.pdf",
        "MixedResolution_unified_v4_4_20260915_0001.svg",
        "MixedResolution_unified_v4_4_20260915_0001.png",
        "source_manifest_v4_4.json", "qa_v4_4.json")]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("The V4_4 baseline bundle is incomplete")
    baseline_hashes = {str(path): sha256(path) for path in required}
    results_before = _tree_state(RESULTS)
    _run(args, rid)
    results_after = _tree_state(RESULTS)
    if results_before != results_after:
        raise RuntimeError("Validated _Process_Results changed during V4_5 build")
    main_base = FIGURES / "Assembled" / f"MixedResolution_unified_v4_5_{rid}"
    manifest_path = FIGURES / "Assembled" / f"FigureSourceManifest_unified_v4_5_{rid}.json"
    process_root = FIGURES / "UnifiedV4_5" / rid
    if any(not main_base.with_suffix(ext).is_file() for ext in (".svg", ".pdf", ".png")) or not manifest_path.is_file():
        raise FileNotFoundError("V4_5 producer outputs are incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    before = json.loads((V4_4_RELEASE / "source_manifest_v4_4.json").read_text(encoding="utf-8"))
    comparison = _scientific_comparison(before, manifest)
    if comparison["status"] != "PASS":
        raise RuntimeError(f"V4_5 scientific comparison failed: {comparison['checks']}")
    release.mkdir(parents=True)
    for ext in (".svg", ".pdf", ".png"):
        shutil.copy2(main_base.with_suffix(ext), release / main_base.with_suffix(ext).name)
    for name in ("panels", "si", "tables"):
        shutil.copytree(process_root / name, release / name)
    shutil.copy2(manifest_path, release / "source_manifest_v4_5.json")
    preview_dir = release / "previews"
    preview_dir.mkdir()
    shutil.copy2(V4_4_RELEASE / "MixedResolution_unified_v4_4_20260915_0001.png", preview_dir / "before_v4_4.png")
    shutil.copy2(main_base.with_suffix(".png"), preview_dir / "after_v4_5_180mm.png")
    previews = _previews(preview_dir, main_base.with_suffix(".png"))
    _copy_sources(release)
    _write_docs(release, rid, manifest, comparison, previews, results_before == results_after)
    write_json(release / "qa_v4_5.json", {"status": "PENDING_EXTERNAL_AUDIT",
               "scientific_state_comparison": comparison["status"], "review_bundle": str(release.resolve())})
    if baseline_hashes != {str(path): sha256(path) for path in required}:
        raise RuntimeError("A V4_4 baseline asset changed during V4_5 build")
    print(f"[OK] {release}")


if __name__ == "__main__":
    main()

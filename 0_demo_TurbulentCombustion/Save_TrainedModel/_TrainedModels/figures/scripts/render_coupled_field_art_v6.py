#!/usr/bin/env python
"""Package and independently audit the Figure 4 art-V6 revision."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from difflib import unified_diff
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "_Scripts"
ASSEMBLED = ROOT / "_Process_Figures" / "Assembled" / "Composite"
REVIEW_ROOT = ROOT / "figures" / "generated" / "art_style_review"
BASELINE_PDF = ASSEMBLED / "CoupledFieldReconstruction_round3_spacing_20260828_0810.pdf"
BASELINE_MANIFEST = ASSEMBLED / "CoupledFieldReconstruction_round3_spacing_20260828_0810_source_manifest.json"
V5_PDF = ASSEMBLED / "CoupledFieldReconstruction_art_v5_20260915_1035.pdf"
V5_RENDERER = SCRIPTS / "100_assemble_coupled_field_art_v5.py"
V5_LAYOUT = SCRIPTS / "publication_layout_coupled_field_art_v5.yaml"
V5_WRAPPER = ROOT / "figures" / "scripts" / "render_coupled_field_art_v5.py"
V4_WRAPPER = ROOT / "figures" / "scripts" / "render_coupled_field_art_v4.py"
RENDERER = SCRIPTS / "102_assemble_coupled_field_art_v6.py"
AUDITOR = SCRIPTS / "103_audit_coupled_field_art_v6.py"
LAYOUT = SCRIPTS / "publication_layout_coupled_field_art_v6.yaml"
RUN_ID = "paper_full_20260711"


def _load_v4_wrapper():
    spec = importlib.util.spec_from_file_location("coupled_field_v4_bundle_utils_for_v6", V4_WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V4 bundle utilities: {V4_WRAPPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = _load_v4_wrapper().utils


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_previews(release: Path, pdf: Path, png: Path) -> dict:
    preview = release / "previews"
    preview.mkdir()
    before = preview / "before_art_v5.png"
    utils.rasterize_pdf(V5_PDF, before, 1.0, dpi=300)
    after_180 = preview / "after_art_v6_180mm.png"
    shutil.copy2(png, after_180)
    after_162 = preview / "after_art_v6_162mm_vector_render.png"
    utils.rasterize_pdf(pdf, after_162, 0.9, dpi=600)
    with Image.open(after_180) as image:
        rgb = image.convert("RGB")
        gray = preview / "after_art_v6_grayscale.png"
        ImageOps.grayscale(rgb).save(gray, dpi=(600, 600))
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([
            [.367, .861, -.228], [.280, .673, .047], [-.012, .043, .969],
        ], dtype=np.float32)
        cvd = preview / "after_art_v6_deuteranopia.png"
        Image.fromarray(
            np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8), "RGB",
        ).save(cvd, dpi=(600, 600))
    return {key: str(path.resolve()) for key, path in {
        "before": before,
        "after_180mm": after_180,
        "after_162mm": after_162,
        "grayscale": gray,
        "deuteranopia": cvd,
    }.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    stem = ASSEMBLED / f"CoupledFieldReconstruction_{args.output_id}"
    generated = {suffix: stem.with_suffix(f".{suffix}") for suffix in ("pdf", "svg", "png")}
    generated["manifest"] = ASSEMBLED / f"{stem.name}_source_manifest.json"
    generated["art_qa"] = ASSEMBLED / f"{stem.name}_art_qa.json"
    if not args.reuse_existing:
        subprocess.run([
            sys.executable, str(RENDERER), "--run-id", RUN_ID,
            "--layout", str(LAYOUT), "--output-id", args.output_id,
            "--formats", "pdf", "svg", "png",
        ], cwd=ROOT, check=True)
    for path in generated.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    release = REVIEW_ROOT / f"Figure_MultiFieldReconstruction_{args.output_id}"
    if release.exists():
        raise FileExistsError(f"Refusing to overwrite review bundle: {release}")
    release.mkdir(parents=True)
    destinations = {
        "pdf": release / "Figure_MultiFieldReconstruction.pdf",
        "svg": release / "Figure_MultiFieldReconstruction.svg",
        "png": release / "Figure_MultiFieldReconstruction.png",
        "manifest": release / "source_manifest_art_v6.json",
        "art_qa": release / "renderer_art_qa.json",
    }
    for key, destination in destinations.items():
        shutil.copy2(generated[key], destination)
    shutil.copy2(V5_PDF, release / "comparison_art_v5.pdf")

    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    current = json.loads(destinations["manifest"].read_text(encoding="utf-8"))
    current["art_v6_display_overrides"] = {
        "major_row_order": ["a", "b", "c", "d"],
        "panel_a_geometry": "preserved_from_art_v5",
        "panel_b_condition_layout": "three_horizontal_tables",
        "panel_b_model_label_rail_count": 1,
        "panel_b_field_header_rail_count": 3,
        "panel_c_consolidated_legend_entries": 9,
        "panel_d_consolidated_legend_entries": 8,
        "canvas_mm": [180.0, 271.0],
    }
    write_json(destinations["manifest"], current)
    states_equal = utils.scientific_state(baseline) == utils.scientific_state(current)
    hashes_equal = utils.source_hashes(baseline) == utils.source_hashes(current)
    scientific_exact = states_equal and hashes_equal
    write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", {
        "schema_version": 1,
        "status": "PASS" if scientific_exact else "FAIL",
        "underlying_data_unchanged": scientific_exact,
        "frozen_state_exact": states_equal,
        "source_records_exact": hashes_equal,
        "baseline_pdf": utils.record(BASELINE_PDF),
        "baseline_manifest": utils.record(BASELINE_MANIFEST),
        "before": utils.scientific_state(baseline),
        "after": utils.scientific_state(current),
    })

    qa = json.loads(destinations["art_qa"].read_text(encoding="utf-8"))
    checks = {
        "scientific_state_exact": scientific_exact,
        "four_sequential_rows": qa["layout_topology"] == "four_full_width_rows_a_b_c_d",
        "panel_a_preserved": qa["panel_a_preserved_from_v5"] and qa["panel_a_axes_count"] == 42,
        "panel_b_horizontal": qa["panel_b_horizontal_conditions"]["model_label_axes_count"] == 1,
        "panel_b_field_headers_repeated": qa["panel_b_horizontal_conditions"]["field_header_axes_count"] == 3,
        "panel_c_single_line_legend": qa["panel_c_full_width"]["legend_entry_count"] == 9 and qa["panel_c_full_width"]["legend_ncols"] == 9,
        "panel_d_single_line_legend": qa["panel_d_full_width"]["legend_entry_count"] == 8 and qa["panel_d_full_width"]["legend_ncols"] == 8,
        "minimum_row_union_clearance_3mm": min(qa[key] for key in (
            "a_to_b_clearance_mm", "b_to_c_clearance_mm", "c_to_d_clearance_mm",
        )) >= 3.0,
        "panel_tags_aligned": qa["panel_tag_x_spread_mm"] <= 0.01,
        "lower_right_edges_aligned": qa["lower_rows_right_edge_spread_mm"] <= 0.01,
        "zero_final_collisions": qa["text_text_overlap_count"] == 0 and qa["text_nonowned_axes_overlap_count"] == 0,
        "zero_clipped_text": qa["clipped_text_count"] == 0,
        "canvas_180_by_271mm": bool(np.allclose(qa["canvas_mm"], [180.0, 271.0], atol=0.02, rtol=0)),
    }
    previews = make_previews(release, destinations["pdf"], destinations["png"])
    write_json(release / "LAYOUT_QA.json", {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "tested_widths_mm": [180, 162],
        "canvas_mm": qa["canvas_mm"],
        "hard_checks": checks,
        "renderer_measurements": qa,
        "insertion_preview_policy": "vector PDF rasterized directly at 0.9 physical scale",
        "previews": previews,
    })

    source_dir = release / "source"
    source_dir.mkdir()
    for path in (RENDERER, AUDITOR, LAYOUT, Path(__file__).resolve()):
        shutil.copy2(path, source_dir / path.name)
    write_json(release / "SOURCE_LOCK.json", {
        "schema_version": 1,
        "status": "RECORDED",
        "revision": "art_v6",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "Python/Matplotlib in fig environment",
        "baseline": utils.record(BASELINE_PDF),
        "baseline_manifest": utils.record(BASELINE_MANIFEST),
        "renderer": utils.record(RENDERER),
        "layout": utils.record(LAYOUT),
        "auditor": utils.record(AUDITOR),
        "source_records": utils.source_hashes(current),
        "training": False,
        "inference": False,
        "persistent_metric_artifact_written": False,
    })
    diffs = []
    for before, after in ((V5_RENDERER, RENDERER), (V5_LAYOUT, LAYOUT), (V5_WRAPPER, Path(__file__).resolve())):
        diffs.extend(unified_diff(
            before.read_text(encoding="utf-8").splitlines(keepends=True),
            after.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile=str(before), tofile=str(after),
        ))
    (release / "SOURCE_DIFF.patch").write_text("".join(diffs), encoding="utf-8")
    (release / "STYLE_CHANGELOG.md").write_text(
        "# Figure 4 art V6\n\n"
        "- Preserved Panel a at its V5 physical size and position.\n"
        "- Rebuilt Panels b, c and d as sequential full-width rows on a 180 x 271 mm canvas.\n"
        "- Arranged the three Panel-b conditions horizontally; model labels appear once and all five field headers repeat above each table.\n"
        "- Consolidated Panel c into one nine-entry top legend and Panel d into one eight-entry top legend.\n"
        "- Widened and flattened the Panel-c charts and Panel-d PDFs/violins without changing data, scales, limits, distributions, normalizations or ordering.\n"
        "- Retained V5 alpha, typography, palette, and simplified labels.\n",
        encoding="utf-8",
    )
    (release / "figure_contract.md").write_text(
        "# Figure 4 art-V6 contract\n\n"
        "The evidence order remains a through d. Panel a is V5-locked. Panels b through d change only page geometry and legend placement. All plotted arrays, field memberships, model ordering, limits, scales, normalizations, statistics and numerical annotations remain frozen.\n",
        encoding="utf-8",
    )
    (release / "AUTHOR_ACTIONS.md").write_text(
        "# Author actions\n\nArt and geometry review pass. Scientific release remains pending A02, A05 and A07.\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(AUDITOR), "--release-dir", str(release)], cwd=ROOT, check=True)
    print(f"[OK] review bundle: {release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

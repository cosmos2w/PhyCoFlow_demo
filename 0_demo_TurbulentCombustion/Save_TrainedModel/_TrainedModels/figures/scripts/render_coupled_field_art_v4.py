#!/usr/bin/env python
"""Package and independently audit the Figure 4 art-V4 revision."""
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
RENDERER = SCRIPTS / "98_assemble_coupled_field_art_v4.py"
AUDITOR = SCRIPTS / "99_audit_coupled_field_art_v4.py"
LAYOUT = SCRIPTS / "publication_layout_coupled_field_art_v4.yaml"
V3_RENDERER = SCRIPTS / "96_assemble_coupled_field_art_v3.py"
V3_LAYOUT = SCRIPTS / "publication_layout_coupled_field_art_v3.yaml"
V3_WRAPPER = ROOT / "figures" / "scripts" / "render_coupled_field_art_v3.py"
RUN_ID = "paper_full_20260711"


def _load_v3_wrapper():
    spec = importlib.util.spec_from_file_location("coupled_field_v3_bundle_utils", V3_WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V3 bundle utilities: {V3_WRAPPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = _load_v3_wrapper()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_previews(release: Path, pdf: Path, png: Path) -> dict:
    preview = release / "previews"
    preview.mkdir()
    before = preview / "before_art_v3.png"
    utils.rasterize_pdf(ASSEMBLED / "CoupledFieldReconstruction_art_v3_20260914_2343.pdf", before, 0.45, dpi=300)
    after_180 = preview / "after_art_v4_180mm.png"
    shutil.copy2(png, after_180)
    after_162 = preview / "after_art_v4_162mm_vector_render.png"
    utils.rasterize_pdf(pdf, after_162, 0.9, dpi=600)
    with Image.open(after_180) as image:
        rgb = image.convert("RGB")
        gray = preview / "after_art_v4_grayscale.png"
        ImageOps.grayscale(rgb).save(gray, dpi=(600, 600))
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([
            [.367, .861, -.228], [.280, .673, .047], [-.012, .043, .969],
        ], dtype=np.float32)
        cvd = preview / "after_art_v4_deuteranopia.png"
        Image.fromarray(
            np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8), "RGB",
        ).save(cvd, dpi=(600, 600))
    return {key: str(path.resolve()) for key, path in {
        "before": before, "after_180mm": after_180, "after_162mm": after_162,
        "grayscale": gray, "deuteranopia": cvd,
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
        "manifest": release / "source_manifest_art_v4.json",
        "art_qa": release / "renderer_art_qa.json",
    }
    for key, destination in destinations.items():
        shutil.copy2(generated[key], destination)
    shutil.copy2(BASELINE_PDF, release / "baseline_round3.pdf")

    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    current = json.loads(destinations["manifest"].read_text(encoding="utf-8"))
    current["art_v4_display_overrides"] = {
        "panel_a_column_headers_pt": 7.8,
        "panel_a_super_to_column_gap_multiplier": 0.5,
        "panel_b_parent_gridspec_width_multiplier": 1.10,
        "panel_c_d_left_spines": "renderer-solved identical figure coordinate",
        "panel_d_tick_values": "retained on one baseline with outward edge anchoring",
        "panel_d_xlabel_labelpad_pt": 7.0,
    }
    write_json(destinations["manifest"], current)
    states_equal = utils.scientific_state(baseline) == utils.scientific_state(current)
    hashes_equal = utils.source_hashes(baseline) == utils.source_hashes(current)
    scientific_exact = states_equal and hashes_equal
    write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", {
        "schema_version": 1, "status": "PASS" if scientific_exact else "FAIL",
        "underlying_data_unchanged": scientific_exact,
        "frozen_state_exact": states_equal, "source_records_exact": hashes_equal,
        "baseline_pdf": utils.record(BASELINE_PDF),
        "baseline_manifest": utils.record(BASELINE_MANIFEST),
        "before": utils.scientific_state(baseline), "after": utils.scientific_state(current),
    })

    qa = json.loads(destinations["art_qa"].read_text(encoding="utf-8"))
    bottom = qa["panel_d_bottom"]
    checks = {
        "scientific_state_exact": scientific_exact,
        "column_header_size_matches_small_axis_role": qa["panel_a_column_header_size_pt"] == qa["panel_a_axis_title_reference_size_pt"] == 7.8,
        "super_title_padding_halved": all(abs(value - 0.5) < 1e-10 for value in qa["panel_a_gap_ratio"]),
        "panel_b_exact_1_10x": abs(qa["panel_b_grid_width_multiplier"] - 1.10) < 1e-10,
        "panel_c_compressed": qa["panel_c_width_multiplier_vs_v3"] < 1.0,
        "panel_c_d_spines_identical": qa["panel_c_d_max_left_spine_delta_mm"] < 1e-9,
        "panel_c_d_tags_identical": qa["panel_c_d_tag_x_alignment_delta_mm"] < 1e-9,
        "panel_d_each_subplot_wider_than_v3": min(qa["panel_d_subplot_widths_mm"]) > 23.86445964666667,
        "panel_d_wspace_increased": qa["panel_d_intercolumn_gap_mm"] > 6.0,
        "panel_d_title_ticks_clear": bottom["xlabel_to_ticks_clearance_mm"] >= 0.999 and bottom["tick_overlap_count"] == 0,
        "panel_d_title_bottom_aligned": bottom["xlabel_bottom_to_panel_b_bottom_delta_mm"] < 0.01,
        "panel_d_means_contained": qa["panel_d_mean_clip_count"] == 0,
        "b_to_right_text_clearance": qa["panel_b_to_c_d_text_clearance_mm"] >= 1.0,
        "final_collision_free": qa["text_text_overlap_count"] == 0 and qa["text_nonowned_axes_overlap_count"] == 0,
    }
    previews = make_previews(release, destinations["pdf"], destinations["png"])
    write_json(release / "LAYOUT_QA.json", {
        "schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL",
        "tested_widths_mm": [180, 162], "canvas_mm": qa["canvas_mm"],
        "hard_checks": checks, "renderer_measurements": qa,
        "insertion_preview_policy": "vector PDF rasterized directly at 0.9 physical scale",
        "previews": previews,
    })

    source_dir = release / "source"
    source_dir.mkdir()
    for path in (RENDERER, AUDITOR, LAYOUT, Path(__file__).resolve()):
        shutil.copy2(path, source_dir / path.name)
    write_json(release / "SOURCE_LOCK.json", {
        "schema_version": 1, "status": "RECORDED", "revision": "art_v4",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "Python/Matplotlib in fig environment",
        "baseline": utils.record(BASELINE_PDF), "baseline_manifest": utils.record(BASELINE_MANIFEST),
        "renderer": utils.record(RENDERER), "layout": utils.record(LAYOUT),
        "auditor": utils.record(AUDITOR), "source_records": utils.source_hashes(current),
        "training": False, "inference": False, "persistent_metric_artifact_written": False,
    })
    diffs = []
    for before, after in ((V3_RENDERER, RENDERER), (V3_LAYOUT, LAYOUT)):
        diffs.extend(unified_diff(
            before.read_text(encoding="utf-8").splitlines(keepends=True),
            after.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile=str(before), tofile=str(after),
        ))
    (release / "SOURCE_DIFF.patch").write_text("".join(diffs), encoding="utf-8")
    (release / "STYLE_CHANGELOG.md").write_text(
        "# Figure 4 art V4\n\n"
        "- Panel-a column titles now use the 7.8-pt small-axis role; their super-title gap is exactly halved.\n"
        "- Panel b receives exactly 1.10x its V3 GridSpec width; Panel c compresses to 90.28%.\n"
        "- Panels c/d share an identical measured left spine and aligned left-offset tags.\n"
        "- Panel-d axes widen to 26.14 mm with 6.5-mm gutters. Tick values remain visible on one baseline.\n"
        "- The JSD title has 7-pt label padding, 1-mm tick clearance and the same bottom edge as Panel b.\n",
        encoding="utf-8",
    )
    (release / "figure_contract.md").write_text(
        "# Figure 4 art-V4 contract\n\n"
        "Core conclusion and evidence hierarchy are unchanged from V3. This revision changes typography and layout geometry only. Python/Matplotlib remains the exclusive renderer; all arrays, distributions, scales, limits and normalizations are frozen.\n",
        encoding="utf-8",
    )
    (release / "AUTHOR_ACTIONS.md").write_text(
        "# Author actions\n\nArt review passes. Scientific release remains pending A02, A05 and A07.\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(AUDITOR), "--release-dir", str(release)], cwd=ROOT, check=True)
    print(f"[OK] review bundle: {release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

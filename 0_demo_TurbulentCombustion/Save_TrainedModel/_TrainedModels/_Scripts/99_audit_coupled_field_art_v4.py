#!/usr/bin/env python
"""Independent release audit for Figure 4 art V4."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import fitz


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(release: Path) -> dict:
    release = release.resolve()
    layout = read_json(release / "LAYOUT_QA.json")
    scientific = read_json(release / "SCIENTIFIC_STATE_COMPARISON.json")
    lock = read_json(release / "SOURCE_LOCK.json")
    renderer = read_json(release / "renderer_art_qa.json")
    pdf = release / "Figure_MultiFieldReconstruction.pdf"
    svg = release / "Figure_MultiFieldReconstruction.svg"
    png = release / "Figure_MultiFieldReconstruction.png"
    required = [
        pdf, svg, png, release / "STYLE_CHANGELOG.md", release / "AUTHOR_ACTIONS.md",
        release / "figure_contract.md", release / "SOURCE_DIFF.patch",
        release / "source_manifest_art_v4.json", release / "renderer_art_qa.json",
        release / "previews" / "after_art_v4_162mm_vector_render.png",
    ]
    document = fitz.open(pdf)
    page = document[0]
    width_mm = page.rect.width / 72.0 * 25.4
    height_mm = page.rect.height / 72.0 * 25.4
    text = "\n".join(block[4] for block in page.get_text("blocks"))
    page_count = document.page_count
    document.close()
    fonts = subprocess.run(["pdffonts", str(pdf)], check=True, text=True, capture_output=True).stdout
    svg_text = svg.read_text(encoding="utf-8", errors="replace")
    bottom = renderer["panel_d_bottom"]
    checks = {
        "required_artifacts": all(path.is_file() for path in required),
        "scientific_state_exact": scientific.get("status") == "PASS",
        "layout_hard_checks": layout.get("status") == "PASS",
        "fixed_width_180mm": abs(width_mm - 180.0) <= 0.02,
        "single_page": page_count == 1,
        "pdf_text_editable": len(text.strip()) > 100,
        "svg_text_editable": "<text" in svg_text,
        "fonts_embedded": "yes yes" in fonts,
        "source_lock_recorded": lock.get("status") == "RECORDED",
        "panel_inventory": all(label in text for label in ("a", "b", "c", "d")),
        "column_headers_7_8pt": renderer["panel_a_column_header_size_pt"] == 7.8,
        "super_title_gap_halved": all(abs(value - 0.5) < 1e-10 for value in renderer["panel_a_gap_ratio"]),
        "panel_b_width_exact_1_10x": abs(renderer["panel_b_grid_width_multiplier"] - 1.10) < 1e-10,
        "panel_c_compressed": renderer["panel_c_width_multiplier_vs_v3"] < 1.0,
        "panel_c_d_spines_exact": renderer["panel_c_d_max_left_spine_delta_mm"] < 1e-9,
        "panel_c_d_tags_exact": renderer["panel_c_d_tag_x_alignment_delta_mm"] < 1e-9,
        "panel_d_subplots_wider": min(renderer["panel_d_subplot_widths_mm"]) > 23.86445964666667,
        "panel_d_wspace_increased": renderer["panel_d_intercolumn_gap_mm"] > 6.0,
        "panel_d_means_unclipped": renderer["panel_d_mean_clip_count"] == 0,
        "panel_d_ticks_noncolliding": bottom["tick_overlap_count"] == 0,
        "panel_d_title_tick_clearance_ge_1mm": bottom["xlabel_to_ticks_clearance_mm"] >= 0.999,
        "panel_d_bottom_aligned_to_b": bottom["xlabel_bottom_to_panel_b_bottom_delta_mm"] < 0.01,
        "panel_b_to_right_clearance_ge_1mm": renderer["panel_b_to_c_d_text_clearance_mm"] >= 1.0,
        "final_aspect_collision_free": renderer["text_text_overlap_count"] == 0 and renderer["text_nonowned_axes_overlap_count"] == 0,
    }
    passed = all(checks.values())
    result = {
        "schema_version": 1, "revision": "art_v4",
        "status": "PASS" if passed else "FAIL", "passed": passed,
        "release_dir": str(release), "canvas_mm": [width_mm, height_mm],
        "checks": checks,
    }
    destination = release / "qa_art_v4.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, value in checks.items():
        print(f"[{'PASS' if value else 'FAIL'}] {name}")
    print(f"[{'PASS' if passed else 'FAIL'}] {destination}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    if not audit(args.release_dir)["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

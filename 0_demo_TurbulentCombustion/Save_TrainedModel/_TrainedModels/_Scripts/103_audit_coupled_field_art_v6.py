#!/usr/bin/env python
"""Independent release audit for Figure 4 art V6."""
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
        pdf, svg, png,
        release / "STYLE_CHANGELOG.md",
        release / "AUTHOR_ACTIONS.md",
        release / "figure_contract.md",
        release / "SOURCE_DIFF.patch",
        release / "source_manifest_art_v6.json",
        release / "renderer_art_qa.json",
        release / "previews" / "after_art_v6_162mm_vector_render.png",
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
    checks = {
        "required_artifacts": all(path.is_file() for path in required),
        "scientific_state_exact": scientific.get("status") == "PASS",
        "layout_hard_checks": layout.get("status") == "PASS",
        "fixed_canvas_180_by_271mm": abs(width_mm - 180.0) <= 0.02 and abs(height_mm - 271.0) <= 0.02,
        "single_page": page_count == 1,
        "pdf_text_editable": len(text.strip()) > 100,
        "svg_text_editable": "<text" in svg_text,
        "fonts_embedded": "yes yes" in fonts,
        "source_lock_recorded": lock.get("status") == "RECORDED",
        "panel_inventory": all(label in text for label in ("a", "b", "c", "d")),
        "panel_a_42_maps": renderer.get("panel_a_axes_count") == 42,
        "four_sequential_rows": renderer.get("layout_topology") == "four_full_width_rows_a_b_c_d",
        "panel_b_three_conditions": text.count("Conditioned on") >= 3,
        "panel_b_one_model_rail": renderer["panel_b_horizontal_conditions"]["model_label_axes_count"] == 1,
        "panel_b_three_field_header_rails": renderer["panel_b_horizontal_conditions"]["field_header_axes_count"] == 3,
        "panel_c_one_line_legend": renderer["panel_c_full_width"]["legend_entry_count"] == 9 and renderer["panel_c_full_width"]["legend_ncols"] == 9,
        "panel_d_one_line_legend": renderer["panel_d_full_width"]["legend_entry_count"] == 8 and renderer["panel_d_full_width"]["legend_ncols"] == 8,
        "row_clearances_at_least_3mm": min(renderer[key] for key in (
            "a_to_b_clearance_mm", "b_to_c_clearance_mm", "c_to_d_clearance_mm",
        )) >= 3.0,
        "panel_tags_aligned": renderer.get("panel_tag_x_spread_mm", 1.0) <= 0.01,
        "lower_right_edges_aligned": renderer.get("lower_rows_right_edge_spread_mm", 1.0) <= 0.01,
        "no_text_collisions": renderer.get("text_text_overlap_count") == 0 and renderer.get("text_nonowned_axes_overlap_count") == 0,
        "no_clipped_text": renderer.get("clipped_text_count") == 0,
        "v5_alpha_retained": renderer.get("face_alpha") == 0.75,
    }
    passed = all(checks.values())
    result = {
        "schema_version": 1,
        "revision": "art_v6",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "release_dir": str(release),
        "canvas_mm": [width_mm, height_mm],
        "checks": checks,
    }
    destination = release / "qa_art_v6.json"
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

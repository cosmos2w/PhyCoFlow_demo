#!/usr/bin/env python
"""Independent release audit for Figure 4 art V5."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import fitz
import numpy as np


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
        pdf,
        svg,
        png,
        release / "STYLE_CHANGELOG.md",
        release / "AUTHOR_ACTIONS.md",
        release / "figure_contract.md",
        release / "SOURCE_DIFF.patch",
        release / "source_manifest_art_v5.json",
        release / "renderer_art_qa.json",
        release / "previews" / "after_art_v5_162mm_vector_render.png",
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
    heatmap_alphas = [record["alpha"] for record in renderer["panel_b_heatmap_alpha_records"]]
    bar_face_alphas = [record["face_alpha"] for record in renderer["panel_c_bar_alpha_records"]]
    bar_edge_alphas = [record["edge_alpha"] for record in renderer["panel_c_bar_alpha_records"]]
    violin_face_alphas = [alpha for record in renderer["panel_d_violin_alpha_records"] for alpha in record["face_alphas"]]
    violin_edge_alphas = [alpha for record in renderer["panel_d_violin_alpha_records"] for alpha in record["edge_alphas"]]
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
        "panel_c_titles_simplified": renderer.get("panel_c_title_after") == ["$Y_{CH_4}$", "$p$", "$U_1$"] and "(unobs.)" not in text,
        "panel_c_xlabel_shortened": renderer.get("panel_c_spectrum_xlabels") == ["Wavenumber"] and "Wavenumber; conditioned" not in text,
        "panel_d_xlabel_shortened": renderer.get("panel_d_violin_xlabels") == ["JSD of joint PDF"] and "JSD of joint PDF; conditioned" not in text,
        "three_panel_b_heatmaps_alpha_075": len(heatmap_alphas) == 3 and bool(np.allclose(heatmap_alphas, 0.75, atol=1e-10, rtol=0)),
        "twenty_four_panel_c_bars_alpha_075": len(bar_face_alphas) == 24 and bool(np.allclose(bar_face_alphas, 0.75, atol=1e-10, rtol=0)),
        "panel_c_bar_edges_opaque": len(bar_edge_alphas) == 24 and bool(np.allclose(bar_edge_alphas, 1.0, atol=1e-10, rtol=0)),
        "twenty_four_panel_d_violins_alpha_075": len(violin_face_alphas) == 24 and bool(np.allclose(violin_face_alphas, 0.75, atol=1e-10, rtol=0)),
        "panel_d_violin_edges_opaque": len(violin_edge_alphas) == 24 and bool(np.allclose(violin_edge_alphas, 1.0, atol=1e-10, rtol=0)),
        "inherited_v4_geometry": renderer.get("panel_b_grid_width_multiplier") == 1.10 and renderer.get("panel_c_d_max_left_spine_delta_mm", 1) < 1e-9,
        "no_text_collisions": renderer.get("text_text_overlap_count") == 0 and renderer.get("text_nonowned_axes_overlap_count") == 0,
    }
    passed = all(checks.values())
    result = {
        "schema_version": 1,
        "revision": "art_v5",
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "release_dir": str(release),
        "canvas_mm": [width_mm, height_mm],
        "checks": checks,
    }
    destination = release / "qa_art_v5.json"
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

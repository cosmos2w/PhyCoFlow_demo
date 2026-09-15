#!/usr/bin/env python
"""Independent release audit for the Figure 5 art-style V1 bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
from PIL import Image


REQUIRED = {
    "figure5_log.pdf",
    "figure5_log.svg",
    "figure5_log.png",
    "figure5_log_preview.png",
    "figure5_log_162mm.png",
    "figure5_log_162mm_grayscale.png",
    "figure5_log_162mm_deuteranopia.png",
    "baseline_162mm.png",
    "LAYOUT_QA.json",
    "figure_contract.md",
    "STYLE_CHANGELOG.md",
    "source_manifest.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    release = args.release_dir.resolve()
    missing = sorted(name for name in REQUIRED if not (release / name).is_file())

    qa = json.loads((release / "LAYOUT_QA.json").read_text()) if not missing else {}
    manifest = json.loads((release / "source_manifest.json").read_text()) if not missing else {}
    pdf = fitz.open(release / "figure5_log.pdf") if not missing else None
    size_mm = (
        [pdf[0].rect.width * 25.4 / 72.0, pdf[0].rect.height * 25.4 / 72.0]
        if pdf is not None
        else [0.0, 0.0]
    )
    pdf_text = pdf[0].get_text() if pdf is not None else ""
    svg_text = (release / "figure5_log.svg").read_text() if not missing else ""
    stored_hashes = manifest.get("outputs", {})
    hash_mismatches = sorted(
        name
        for name, expected in stored_hashes.items()
        if not (release / name).is_file() or sha256(release / name) != expected
    )
    raster_sizes = {
        name: Image.open(release / name).size
        for name in (
            "figure5_log_162mm.png",
            "figure5_log_162mm_grayscale.png",
            "figure5_log_162mm_deuteranopia.png",
        )
        if (release / name).is_file()
    }

    required_text = [
        "Normalized CRPS",
        "Spearman",
        "Retained least-uncertain states",
        "Unobserved-field relative",
        "High-band relative",
        "Frequency shell",
        "Training update time",
        "Inference memory",
        "DMF-Gen",
        "Senseiver",
    ]
    checks = {
        "required_artifacts": not missing,
        "all_renderer_checks_pass": bool(qa.get("checks")) and all(qa["checks"].values()),
        "scientific_state_exact": qa.get("scientific_hash_before") == qa.get("scientific_hash_after"),
        "fixed_canvas_180x200_mm": bool(np.allclose(size_mm, [180.0, 200.0], atol=0.02, rtol=0)),
        "single_page": pdf is not None and pdf.page_count == 1,
        "fonts_embedded": pdf is not None and len(pdf[0].get_fonts()) >= 2,
        "pdf_text_inventory": all(text in pdf_text for text in required_text),
        "svg_text_editable": svg_text.count("<text") > 80,
        "source_manifest_hashes": not hash_mismatches and len(stored_hashes) >= 10,
        "review_raster_dimensions_match": len(set(raster_sizes.values())) == 1 and len(raster_sizes) == 3,
        "row_clearances_ge_3mm": min(qa.get("row_union_clearance_mm", {}).values(), default=-1) >= 3.0,
        "no_collision_or_clipping_records": not qa.get("clipped_text") and not qa.get("tick_label_overlaps") and not qa.get("text_text_overlaps") and not qa.get("text_nonowned_axes_overlaps") and not qa.get("panel_tag_axes_overlaps"),
        "font_floor_at_162mm": qa.get("tick_label_font_pt", 0) * qa.get("insertion_scale", 0) >= 7.0 and qa.get("legend_font_pt", 0) * qa.get("insertion_scale", 0) >= 7.0,
    }
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if missing:
        print(f"Missing: {missing}")
    if hash_mismatches:
        print(f"Hash mismatches: {hash_mismatches}")
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"Figure 5 art V1 release audit failed: {failed}")
    print(f"[PASS] {release}")


if __name__ == "__main__":
    main()

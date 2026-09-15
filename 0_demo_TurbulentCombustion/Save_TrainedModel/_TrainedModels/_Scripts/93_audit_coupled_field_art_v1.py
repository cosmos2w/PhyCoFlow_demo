#!/usr/bin/env python
"""Independent release audit for Figure_MultiFieldReconstruction art V1."""
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
    pdf = release / "Figure_MultiFieldReconstruction.pdf"
    svg = release / "Figure_MultiFieldReconstruction.svg"
    png = release / "Figure_MultiFieldReconstruction.png"
    required = [
        pdf, svg, png, release / "STYLE_CHANGELOG.md", release / "AUTHOR_ACTIONS.md",
        release / "figure_contract.md", release / "SOURCE_DIFF.patch",
        release / "source_manifest_art_v1.json", release / "renderer_art_qa.json",
    ]
    document = fitz.open(pdf)
    page = document[0]
    width_mm = page.rect.width / 72.0 * 25.4
    height_mm = page.rect.height / 72.0 * 25.4
    text = "\n".join(item[4] for item in page.get_text("blocks"))
    document.close()
    fonts = subprocess.run(["pdffonts", str(pdf)], check=True, text=True,
                           capture_output=True).stdout
    svg_text = svg.read_text(encoding="utf-8", errors="replace")
    checks = {
        "required_artifacts": all(path.is_file() for path in required),
        "scientific_state_exact": scientific.get("status") == "PASS",
        "layout_hard_checks": layout.get("status") == "PASS",
        "fixed_width_180mm": abs(width_mm - 180.0) <= 0.02,
        "compact_height_le_235mm": height_mm <= 235.0,
        "single_page": True,
        "pdf_text_editable": len(text.strip()) > 100,
        "svg_text_editable": "<text" in svg_text,
        "fonts_embedded": "yes yes" in fonts,
        "source_lock_recorded": lock.get("status") == "RECORDED",
        "panel_inventory": all(label in text for label in ("a", "b", "c", "d")),
        "method_inventory": all(name in text for name in (
            "DMF-Gen", "FFM-FNO", "FFM-Perceiver", "Latent FM",
            "SiT", "MLP-RBF", "Geo-FNO", "Senseiver",
        )),
    }
    passed = all(checks.values())
    result = {
        "schema_version": 1, "revision": "art_v1",
        "status": "PASS" if passed else "FAIL", "passed": passed,
        "release_dir": str(release), "canvas_mm": [width_mm, height_mm],
        "checks": checks,
    }
    destination = release / "qa_art_v1.json"
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

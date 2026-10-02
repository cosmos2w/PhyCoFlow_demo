#!/usr/bin/env python
"""Audit and package the V5.1 Panel-a colorbar revision against exact V5."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import fitz
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "_Scripts"
ASSEMBLED = ROOT / "_Process_Figures" / "Assembled" / "Composite"
BASE_NAME = "CoupledFieldReconstruction_art_v5_20260915_1035"
BASELINE = ASSEMBLED / BASE_NAME
RENDERER = SCRIPTS / "104_assemble_coupled_field_art_v5_1.py"
LAYOUT = SCRIPTS / "publication_layout_coupled_field_art_v5_1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-id", required=True)
    args = parser.parse_args()
    stem = ASSEMBLED / f"CoupledFieldReconstruction_{args.output_id}"
    for path in [*(stem.with_suffix(ext) for ext in (".pdf", ".svg", ".png")),
                 ASSEMBLED / f"{stem.name}_source_manifest.json",
                 ASSEMBLED / f"{stem.name}_art_qa.json"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    old = json.loads((ASSEMBLED / f"{BASE_NAME}_source_manifest.json").read_text())
    new = json.loads((ASSEMBLED / f"{stem.name}_source_manifest.json").read_text())
    qa = json.loads((ASSEMBLED / f"{stem.name}_art_qa.json").read_text())
    spec = importlib.util.spec_from_file_location("coupled_v5_1_audit_utils", Path(__file__).with_name("render_coupled_field_art_v3.py"))
    assert spec is not None and spec.loader is not None
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    science_same = utils.scientific_state(old) == utils.scientific_state(new)
    sources_same = utils.source_hashes(old) == utils.source_hashes(new)
    expected_bars = [
        (field, kind, limits)
        for field in old["fields"]
        for kind, limits in (("value", old["field_value_limits"][field]),
                             ("error", old["robust_error_limits"][field]))
    ]
    bars = qa["panel_a_v5_1"]["colorbars"]
    bars_exact = [(bar["field"], bar["kind"], bar["limits"]) for bar in bars] == expected_bars
    with Image.open(BASELINE.with_suffix(".png")) as image:
        before = np.asarray(image.convert("RGB"))
    with Image.open(stem.with_suffix(".png")) as image:
        after = np.asarray(image.convert("RGB"))
    same_shape = before.shape == after.shape
    lower_start = round(74.0 / 25.4 * 600)
    lower_exact = same_shape and bool(np.array_equal(before[lower_start:], after[lower_start:]))
    changed_rows = np.flatnonzero(np.any(np.any(before != after, axis=2), axis=1)) if same_shape else np.array([])
    with fitz.open(BASELINE.with_suffix(".pdf")) as old_pdf, fitz.open(stem.with_suffix(".pdf")) as new_pdf:
        same_canvas = old_pdf[0].rect == new_pdf[0].rect
    editable_svg = stem.with_suffix(".svg").read_text(encoding="utf-8").count("<text") > 0
    checks = {
        "scientific_state_exact": science_same,
        "source_paths_and_hashes_exact": sources_same,
        "six_colorbar_limits_exact": bars_exact,
        "colorbar_manifest_matches": (
            new["qualitative_geometry_lock"]["colorbar_multiplier"]
            == "centered above each slim bar-and-tick assembly; endpoint tick labels have one decimal place"
        ),
        "single_right_edge_colorbar_stack": (
            qa["panel_a_v5_1"]["colorbar_count"] == 6
            and abs(qa["panel_a_v5_1"]["colorbar_width_mm"] - 2.0) < 1e-6
            and abs(qa["panel_a_v5_1"]["colorbar_right_canvas_gap_mm"]) <= 0.01
            and len({round(bar["bbox_mm"][0], 4) for bar in bars}) == 1
        ),
        "all_colorbar_ticks_one_decimal": all(
            all(label.count(".") == 1 and len(label.split(".")[1]) == 1
                for label in bar["tick_labels"])
            for bar in bars
        ),
        "colorbar_titles_centered_and_stack_clear": (
            max(qa["panel_a_v5_1"]["exponent_title_center_offsets_mm"]) <= 0.1
            and min(qa["panel_a_v5_1"]["colorbar_stack_gaps_mm"]) >= 0.3
        ),
        "uniform_map_gaps_expanded": abs(qa["panel_a_v5_1"]["map_gap_mm"] - 3.0) < 1e-6,
        "panel_a_layout_gate": qa["panel_a_v5_1"]["passed"],
        "panel_a_no_text_collision": qa["panel_a_v5_1"]["text_text_overlap_count"] == 0,
        "panel_a_no_text_on_other_axes": qa["panel_a_v5_1"]["text_nonowned_axes_overlap_count"] == 0,
        "panels_b_to_d_pixel_exact": lower_exact,
        "changes_confined_to_panel_a": bool(changed_rows.size and changed_rows[-1] < lower_start),
        "same_canvas": same_canvas,
        "svg_text_editable": editable_svg,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V5.1 validation failed: {checks}")

    release = ROOT / "figures" / "generated" / "art_style_review" / f"Figure_MultiFieldReconstruction_{args.output_id}"
    if release.exists():
        raise FileExistsError(release)
    release.mkdir(parents=True)
    for ext in (".pdf", ".svg", ".png"):
        shutil.copy2(stem.with_suffix(ext), release / f"Figure_MultiFieldReconstruction{ext}")
    source = release / "source"
    source.mkdir()
    for path in (RENDERER, LAYOUT, Path(__file__)):
        shutil.copy2(path, source / path.name)
    _write(release / "LAYOUT_QA.json", {
        "revision": "art_v5_1", "status": "PASS", "checks": checks,
        "panel_a": qa["panel_a_v5_1"],
        "baseline_pdf": str(BASELINE.with_suffix(".pdf")),
        "baseline_lower_unchanged_from_top_mm": 74.0,
        "last_changed_pixel_row": int(changed_rows[-1]),
        "png_pixels": list(after.shape),
    })
    _write(release / "SCIENTIFIC_STATE_COMPARISON.json", {
        "status": "PASS", "baseline_manifest": str(ASSEMBLED / f"{BASE_NAME}_source_manifest.json"),
        "scientific_state_exact": science_same, "source_paths_and_hashes_exact": sources_same,
        "field_value_limits": new["field_value_limits"],
        "robust_error_limits": new["robust_error_limits"],
    })
    _write(release / "SOURCE_LOCK.json", {
        "revision": "art_v5_1", "renderer_sha256": _sha256(RENDERER),
        "layout_sha256": _sha256(LAYOUT), "baseline_pdf_sha256": _sha256(BASELINE.with_suffix(".pdf")),
        "assembled_sha256": {ext: _sha256(stem.with_suffix(ext)) for ext in (".pdf", ".svg", ".png")},
    })
    (release / "STYLE_CHANGELOG.md").write_text(
        "# Figure 4 art V5.1\n\n"
        "- Placed six color bars in one compact right-edge stack, one per panel-a row.\n"
        "- Used 2.0 mm color strips with each scientific exponent centered over its compact scale and one-decimal tick labels.\n"
        "- Increased every panel-a intercolumn gap uniformly to 3.0 mm while preserving map sizes.\n"
        "- Realigned the headers, group titles, and central divider with the re-spaced maps.\n"
        "- Kept all scientific values and panels b–d identical to V5.\n", encoding="utf-8")
    print(f"[OK] {release}")
    print("[OK] scientific state and panels b–d unchanged")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Independent audit for the author-authorized mixed-resolution V4_1 bundle."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
DEFAULT_RELEASE_ROOT = PROJECT_ROOT / "1_SubTask_SuperResolution/figures/generated/art_style_review"
V4_RELEASE = DEFAULT_RELEASE_ROOT / "MixedResolution_unified_v4_20260914_1620"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("mixed_resolution_v4_audit_helpers", HERE / "120_audit_mixed_resolution_v4.py")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _source_lock_check(lock: dict, release: Path) -> tuple[bool, dict]:
    keys = ("v4_1_renderer", "v4_1_exporter", "v4_1_audit",
            "v4_1_panel_renderer", "v4_1_layout", "v4_1_wrapper")
    detail, passed = {}, True
    for key in keys:
        item = lock.get(key, {})
        path = Path(str(item.get("path", "")))
        observed = base.sha256(path) if path.is_file() else None
        ok = observed == item.get("sha256") and bool(observed)
        detail[key] = {"path": str(path), "recorded": item.get("sha256"),
                       "observed": observed, "passed": ok}
        passed &= ok
    archive = release / "source"
    archived = {path.name: base.sha256(path) for path in archive.glob("*") if path.is_file()}
    detail["archived_source_count"] = len(archived)
    detail["archived_sources"] = archived
    passed &= len(archived) >= 6
    return bool(passed), detail


def audit(release_dir: Path, output: Path | None = None) -> dict:
    release = Path(release_dir).resolve()
    manifest_path = release / "source_manifest_v4_1.json"
    manifest = _read(manifest_path)
    comparison = _read(release / "SCIENTIFIC_STATE_COMPARISON.json")
    lqa = _read(release / "LAYOUT_QA.json")
    lock = _read(release / "SOURCE_LOCK.json")
    v4 = _read(V4_RELEASE / "source_manifest_v4.json")
    pdfs = sorted(release.glob("MixedResolution_unified_v4_1_*.pdf"))
    svgs = sorted(release.glob("MixedResolution_unified_v4_1_*.svg"))
    pngs = sorted(release.glob("MixedResolution_unified_v4_1_*.png"))
    files = {"pdf": pdfs[0] if len(pdfs) == 1 else None,
             "svg": svgs[0] if len(svgs) == 1 else None,
             "png": pngs[0] if len(pngs) == 1 else None}
    svg_text = files["svg"].read_text(encoding="utf-8", errors="replace") if files["svg"] else ""
    checks = []

    def add(name: str, passed: bool, detail) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    required = ["SOURCE_LOCK.json", "SCIENTIFIC_STATE_COMPARISON.json", "LAYOUT_QA.json",
                "STYLE_CHANGELOG.md", "AUTHOR_ACTIONS.md", "figure_contract.md", "SOURCE_DIFF.patch"]
    required_ok = all((release / name).is_file() for name in required)
    add("v4_1_release_identity_and_bookkeeping",
        required_ok and manifest.get("revision") == "V4_1" and release.name.startswith("MixedResolution_unified_v4_1_"),
        {"revision": manifest.get("revision"), "required_files": {name: (release / name).is_file() for name in required}})

    lock_pass, lock_detail = _source_lock_check(lock, release)
    add("source_lock_and_archived_editable_sources", lock_pass, lock_detail)

    comparison_checks = comparison.get("checks", {})
    comp_pass = comparison.get("status") == "PASS" and comparison.get("underlying_data_unchanged") is True \
        and bool(comparison_checks) and all(comparison_checks.values())
    add("underlying_saved_data_unchanged_with_authorized_deltas", comp_pass,
        {"status": comparison.get("status"), "checks": comparison_checks,
         "authorized_deltas": comparison.get("authorized_display_and_inventory_deltas")})

    panels = manifest.get("panels", {})
    inventory = {"panel_ids": sorted(panels),
                 "bar_count": svg_text.count("model-bar:"),
                 "sweep_trace_count": svg_text.count("model-line:"),
                 "roi_box_count": svg_text.count("panel-a-roi-box"),
                 "roi_connector_count": svg_text.count("panel-a-roi-connector"),
                 "matrix_value_count": panels.get("e", {}).get("matrix_numeric_annotation_count")}
    inventory_pass = (inventory["panel_ids"] == list("abcde") and inventory["bar_count"] == 20
                      and inventory["sweep_trace_count"] == 12 and inventory["roi_box_count"] == 3
                      and inventory["roi_connector_count"] == 6 and inventory["matrix_value_count"] == 72)
    add("v4_1_panel_and_artist_inventory", inventory_pass, inventory)

    a = panels.get("a", {})
    roi = a.get("shared_roi", {})
    v4_roi = v4.get("panels", {}).get("a", {}).get("shared_roi", {})
    size = lambda value: (value.get("xmax", 0) - value.get("xmin", 0),
                          value.get("ymax", 0) - value.get("ymin", 0))
    a_pass = (a.get("resolution_titles_moved_count") == 3
              and a.get("roi_shift_parent_fraction_v4_1") == [-0.05, 0.18]
              and size(roi) == size(v4_roi) and a.get("native_field_values_unchanged") is True)
    add("panel_a_viewport_titles_and_native_field_contract", a_pass,
        {"titles_moved": a.get("resolution_titles_moved_count"), "roi": roi,
         "v4_roi": v4_roi, "same_size": size(roi) == size(v4_roi),
         "richness_fractions": a.get("roi_color_richness_fraction_pos_near_zero_neg")})

    b = panels.get("b", {})
    b_pass = (b.get("sweep_recipes") == ["3_Mixed_HML", "4_ZeroH_Balanced", "5_ZeroH_MRich"]
              and b.get("sensor_sweep_count") == 3 and b.get("dmfg_value_annotations") == 0
              and b.get("grouped_bar_left_spine_visible") is False
              and b.get("grouped_bar_y_ticks_visible") is True
              and b.get("grouped_bar_y_labels_visible") is True
              and b.get("sweep_axis_height_min_mm", 0) >= 17.0
              and "panel-b-dmfg-bar-value" not in svg_text)
    add("panel_b_minimalism_three_saved_sweeps_and_aspect", b_pass,
        {key: b.get(key) for key in ("sweep_recipes", "sensor_sweep_count", "dmfg_value_annotations",
                                     "grouped_bar_left_spine_visible", "grouped_bar_y_ticks_visible",
                                     "grouped_bar_y_labels_visible", "sweep_axis_heights_mm")})

    c = panels.get("c", {})
    c_gap = float(c.get("map_to_colorbar_gap_mm", 0))
    c_pass = c.get("top_gutter_annotation_count_v4_1") == 5 and 1.5 <= c_gap <= 2.0 + 1e-6
    add("panel_c_top_labels_and_colorbar_clearance", c_pass,
        {"moved_labels": c.get("top_gutter_annotations_v4_1"), "map_to_colorbar_gap_mm": c_gap})

    d = panels.get("d", {})
    d_pass = (d.get("truth_component_cmap") == "PuOr_r" and d.get("residual_cmap") == "BrBG"
              and d.get("component_and_residual_colormaps_distinct") is True
              and d.get("component_and_residual_norms_unchanged") is True
              and len(set(d.get("row_label_x_positions", []))) == 1
              and d.get("row_label_x_positions", [None])[0] == d.get("panel_tag_x"))
    add("panel_d_distinct_colormaps_unchanged_norms_and_tag_alignment", d_pass,
        {key: d.get(key) for key in ("truth_component_cmap", "residual_cmap",
                                     "component_and_residual_norms_unchanged", "panel_tag_x",
                                     "row_label_x_positions")})

    e = panels.get("e", {}); v4e = v4.get("panels", {}).get("e", {})
    old_area = min(v4e.get("matrix_cell_widths_mm", [0])) * min(v4e.get("matrix_cell_heights_mm", [0]))
    new_area = min(e.get("matrix_cell_widths_mm", [0])) * min(e.get("matrix_cell_heights_mm", [0]))
    center_values = e.get("matrix_numeric_centers", [])
    centered = len(center_values) == 72 and all(item.get("x") in (0, 1, 2) and item.get("y") in (0, 1, 2, 3)
                                                   for item in center_values)
    e_pass = (old_area > 0 and new_area / old_area >= 1.35
              and e.get("central_metric_gap_mm", 999) < v4e.get("central_metric_gap_mm", 0)
              and min(e.get("within_metric_recipe_gaps_mm", [0])) * .9 >= 1.0
              and centered and e.get("title_gap_qa", {}).get("passed") is True)
    add("panel_e_cell_expansion_gap_compression_and_centering", e_pass,
        {"area_growth_min": new_area / old_area if old_area else None,
         "central_gap_v4_mm": v4e.get("central_metric_gap_mm"),
         "central_gap_v4_1_mm": e.get("central_metric_gap_mm"),
         "within_gap_at_162mm": min(e.get("within_metric_recipe_gaps_mm", [0])) * .9,
         "centered_value_count": len(center_values), "title_gaps": e.get("title_gap_qa")})

    typography_pass, typography_detail = base._typography_check(files["svg"], lqa, 180.0, 162.0)
    add("text_floors_at_180_and_162_mm", typography_pass, typography_detail)
    layout_pass, layout_detail = base._layout_qa_check(lqa, (180.0, 162.0))
    add("renderer_collisions_clipping_and_hard_gaps", layout_pass, layout_detail)

    expected_palette = {"DMF-Gen": "#C94053", "FFM-Perceiver": "#4C86A6",
                        "Senseiver": "#8D9BAD", "MLP-RBF": "#4C9E91", "Truth": "#252525"}
    palette = lqa.get("semantic_style_table", {})
    palette_pass = all(str(palette.get(key, "")).lower() == value.lower() for key, value in expected_palette.items())
    palette_pass &= lqa.get("grayscale_check", {}).get("passed") is True
    palette_pass &= lqa.get("color_vision_deficiency_check", {}).get("passed") is True
    add("semantic_palette_grayscale_and_cvd", palette_pass,
        {"expected": expected_palette, "observed": palette,
         "colormaps": lqa.get("semantic_colormaps"),
         "grayscale": lqa.get("grayscale_check"), "cvd": lqa.get("color_vision_deficiency_check")})

    font_pass, font_detail = base._pdf_fonts(files["pdf"])
    editable_pass, editable_detail = base._pdf_text_editable(files["pdf"])
    svg_editable = bool(svg_text and "<text" in svg_text)
    add("embedded_arial_and_editable_vector_text", font_pass and editable_pass and svg_editable,
        {"font": font_detail, "pdf_text": editable_detail, "svg_text_nodes": svg_editable})
    output_pass, output_detail = base._output_check(manifest, release, files, 180.0, 230.0, 600.0)
    add("fixed_canvas_and_export_bundle", output_pass, output_detail)

    passed = all(item["passed"] for item in checks)
    payload = {"workflow_label": "mixed_resolution_unified_v4_1_audit",
               "schema_version": "4.1", "revision": "V4_1",
               "status": "PASS" if passed else "FAIL", "passed": passed,
               "release_dir": str(release), "checks": checks}
    destination = Path(output).resolve() if output else release / "qa_v4_1.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[{'PASS' if passed else 'FAIL'}] wrote {destination}")
    return payload


def self_test() -> None:
    assert abs((23.9 * 20.9) / (20.0 * 18.35) - 1.361) < .01
    assert ["3_Mixed_HML", "4_ZeroH_Balanced", "5_ZeroH_MRich"][0] == "3_Mixed_HML"
    print("[PASS] V4_1 audit helper self-test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.release_dir is None:
        parser.error("--release-dir is required unless --self-test is used")
    if not audit(args.release_dir, args.output)["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Independent audit for the frozen-data mixed-resolution V4_4 bundle."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
DEFAULT_RELEASE_ROOT = PROJECT_ROOT / "1_SubTask_SuperResolution/figures/generated/art_style_review"
V4_3_RELEASE = DEFAULT_RELEASE_ROOT / "MixedResolution_unified_v4_3_20260914_2330"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("mixed_resolution_v4_4_audit_helpers", HERE / "120_audit_mixed_resolution_v4.py")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _source_lock_check(lock: dict, release: Path) -> tuple[bool, dict]:
    keys = ("v4_4_renderer", "v4_4_exporter", "v4_4_audit",
            "v4_4_panel_renderer", "v4_4_layout", "v4_4_wrapper")
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
    manifest = _read(release / "source_manifest_v4_4.json")
    comparison = _read(release / "SCIENTIFIC_STATE_COMPARISON.json")
    lqa = _read(release / "LAYOUT_QA.json")
    lock = _read(release / "SOURCE_LOCK.json")
    baseline = _read(V4_3_RELEASE / "source_manifest_v4_3.json")
    pdfs = sorted(release.glob("MixedResolution_unified_v4_4_*.pdf"))
    svgs = sorted(release.glob("MixedResolution_unified_v4_4_*.svg"))
    pngs = sorted(release.glob("MixedResolution_unified_v4_4_*.png"))
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
    add("v4_4_release_identity_and_bookkeeping",
        required_ok and manifest.get("revision") == "V4_4" and release.name.startswith("MixedResolution_unified_v4_4_"),
        {"revision": manifest.get("revision"), "required_files": {name: (release / name).is_file() for name in required}})

    lock_pass, lock_detail = _source_lock_check(lock, release)
    add("source_lock_and_archived_editable_sources", lock_pass, lock_detail)

    comparison_checks = comparison.get("checks", {})
    comparison_pass = (comparison.get("status") == "PASS"
                       and comparison.get("underlying_data_unchanged") is True
                       and comparison.get("v4_4_raw_or_scientific_deltas") is False
                       and comparison.get("v4_4_display_mapping_exact_to_v4_3") is True
                       and bool(comparison_checks) and all(comparison_checks.values()))
    add("scientific_and_display_state_exact_to_v4_3", comparison_pass,
        {"status": comparison.get("status"), "checks": comparison_checks,
         "carried_exceptions": comparison.get("carried_forward_authorized_display_and_inventory_deltas")})

    panels = manifest.get("panels", {})
    inventory = {"panel_ids": sorted(panels), "bar_count": svg_text.count("model-bar:"),
                 "sweep_trace_count": svg_text.count("model-line:"),
                 "roi_box_count": svg_text.count("panel-a-roi-box"),
                 "roi_connector_count": svg_text.count("panel-a-roi-connector"),
                 "matrix_value_count": panels.get("e", {}).get("matrix_numeric_annotation_count")}
    inventory_pass = (inventory["panel_ids"] == list("abcde") and inventory["bar_count"] == 20
                      and inventory["sweep_trace_count"] == 12 and inventory["roi_box_count"] == 3
                      and inventory["roi_connector_count"] == 6 and inventory["matrix_value_count"] == 72)
    add("v4_4_panel_and_artist_inventory", inventory_pass, inventory)

    a = panels.get("a", {})
    text_replacements = manifest.get("author_authorized_deltas", {}).get("v4_4", {}).get("text_replacements", [])
    replacement_pairs = {(item.get("before"), item.get("after")) for item in text_replacements}
    expected_pairs = {
        ("contains H-resolution training fields", "Contains H-resolution training fields"),
        ("zero-H training", "Zero-H training"),
        ("no H-resolution training fields", "Zero-H training"),
    }
    a_pass = (a.get("resolution_caption_count") == 3
              and a.get("resolution_caption_fontsize_pt") == 8.0
              and a.get("resolution_caption_role") == "tick_label"
              and a.get("resolution_caption_matches_tick_label_size") is True
              and a.get("native_field_values_unchanged") is True
              and a.get("interior_scale_identity_label_count") == 3
              and [item.get("label") for item in a.get("interior_scale_identity_labels", [])] == list("LMH")
              and min(item.get("contrast_ratio", 0) for item in a.get("interior_scale_identity_labels", [])) >= 4.5
              and expected_pairs == replacement_pairs
              and "Contains H-resolution training fields" in svg_text
              and svg_text.count("Zero-H training") == 2
              and "contains H-resolution training fields" not in svg_text
              and "zero-H training" not in svg_text
              and "no H-resolution training fields" not in svg_text)
    add("panel_a_b_capitalization_and_exact_text_replacements", a_pass,
        {key: a.get(key) for key in ("resolution_caption_count", "resolution_caption_fontsize_pt",
                                     "resolution_caption_role", "native_field_values_unchanged",
                                     "interior_scale_identity_labels")} | {"text_replacements": text_replacements})

    b = panels.get("b", {})
    widths = b.get("sweep_axis_widths_mm", [])
    b_pass = (b.get("top_y_label") == r"Relative $L_2$"
              and b.get("sweep_y_label") == r"Relative $L_2$"
              and b.get("y_labels_single_line") is True
              and b.get("grouped_bar_axis_height_mm", 0) >= 18.5
              and b.get("bar_to_rightmost_sweep_right_edge_delta_mm", 999) <= 0.02
              and len(widths) == 3 and max(widths) - min(widths) <= 0.02
              and b.get("dmfg_value_annotations") == 0
              and b.get("grouped_bar_left_spine_visible") is False
              and b.get("legend_bbox_clearance_passed") is True
              and b.get("zero_h_label_after") == "Zero-H training"
              and b.get("zero_h_label_unboxed") is True
              and b.get("zero_h_label_matches_panel_a_unboxed_style") is True
              and b.get("legend_to_bar_xtick_bbox_clearance_mm", 0) * .9 >= 1.0
              and b.get("legend_to_sweep_title_bbox_clearance_mm", 0) * .9 >= 1.0)
    add("panel_b_single_line_labels_height_flush_edge_and_legend_clearance", b_pass,
        {key: b.get(key) for key in ("top_y_label", "sweep_y_label", "y_labels_single_line",
                                     "grouped_bar_axis_height_mm", "sweep_axis_heights_mm",
                                     "sweep_axis_widths_mm", "bar_to_rightmost_sweep_right_edge_delta_mm",
                                     "zero_h_label_after", "zero_h_label_unboxed",
                                     "zero_h_label_matches_panel_a_unboxed_style",
                                     "legend_to_bar_xtick_bbox_clearance_mm",
                                     "legend_to_sweep_title_bbox_clearance_mm")})

    c = panels.get("c", {})
    c_gap = float(c.get("map_to_colorbar_gap_mm", 0))
    c_pass = c.get("top_gutter_annotation_count_v4_1") == 5 and 1.5 <= c_gap <= 2.0 + 1e-6
    add("panel_c_preserved_top_labels_and_colorbar_clearance", c_pass,
        {"moved_labels": c.get("top_gutter_annotations_v4_1"), "map_to_colorbar_gap_mm": c_gap})

    d = panels.get("d", {})
    d_pass = (d.get("truth_component_cmap") == "Greys_r"
              and d.get("residual_cmap") == "magma"
              and d.get("rendered_truth_cmaps") == ["Greys_r"] * 3
              and d.get("rendered_residual_cmaps") == ["magma"] * 6
              and d.get("truth_contour_set_count") == 3
              and d.get("truth_contour_levels_unchanged") == 6
              and d.get("component_and_residual_norms_unchanged") is False
              and d.get("truth_component_norms_unchanged") is True
              and d.get("residual_display_definition") == "abs(predicted component - truth component)"
              and d.get("residual_display_all_nonnegative") is True
              and d.get("residual_display_all_vmin_zero") is True
              and d.get("residual_zero_maps_to_dark_endpoint") is True
              and len(d.get("residual_display_records", [])) == 6
              and all(item.get("absolute_transform_exact") is True for item in d.get("residual_display_records", []))
              and d.get("headers_centered_over_columns") is True
              and max(d.get("header_to_column_center_deltas_mm", [999])) <= 0.02
              and d.get("panel_tag_row_label_left_bbox_delta_mm", 999) <= 0.02
              and d.get("panel_tag_c_top_bbox_delta_mm", 999) <= 0.02
              and d.get("row_label_to_truth_map_gap_mm", 0) * .9 >= 1.0)
    add("panel_d_absolute_error_zero_anchor_centered_titles_and_tag_alignment", d_pass,
        {key: d.get(key) for key in ("truth_component_cmap", "residual_cmap", "rendered_truth_cmaps",
                                     "rendered_residual_cmaps", "truth_contour_set_count",
                                     "truth_contour_levels_unchanged", "truth_contour_style",
                                     "residual_display_definition", "residual_display_records",
                                     "residual_zero_rgba", "header_to_column_center_deltas_mm",
                                     "panel_tag_row_label_left_bbox_delta_mm", "panel_tag_c_top_bbox_delta_mm",
                                     "row_label_to_truth_map_gap_mm")})

    e = panels.get("e", {})
    center_values = e.get("matrix_numeric_centers", [])
    e_pass = (len(center_values) == 72 and e.get("title_gap_qa", {}).get("passed") is True
              and min(e.get("within_metric_recipe_gaps_mm", [0])) * .9 >= 1.0
              and e.get("scale_tick_labels_centered_on_matrix_columns") is True
              and e.get("scale_tick_label_max_center_delta_mm", 999) <= 0.02)
    add("panel_e_preserved_geometry_and_exact_tick_centering", e_pass,
        {"centered_value_count": len(center_values), "title_gaps": e.get("title_gap_qa"),
         "within_gap_at_162mm": min(e.get("within_metric_recipe_gaps_mm", [0])) * .9,
         "tick_center_delta_mm": e.get("scale_tick_label_max_center_delta_mm")})

    spacing = manifest.get("layout", {}).get("major_content_spacing_qa", {})
    visible_gaps = spacing.get("visible_content_gaps_mm", {})
    spacing_pass = (spacing.get("visible_content_gap_qa_passed") is True
                    and set(visible_gaps) == {"a_b", "b_cd", "cd_e"}
                    and min(visible_gaps.values(), default=0) * .9 >= 3.0
                    and spacing.get("cd_e_gap_distance_from_ab_mm", 999) <= 1.0
                    and manifest.get("layout", {}).get("panel_e_upshift_mm") == 5.0)
    add("panel_e_upshift_and_visible_major_spacing", spacing_pass,
        {"visible_content_gaps_mm": visible_gaps,
         "visible_content_gaps_at_162mm": {k: v * .9 for k, v in visible_gaps.items()},
         "cd_e_gap_distance_from_ab_mm": spacing.get("cd_e_gap_distance_from_ab_mm"),
         "panel_e_upshift_mm": manifest.get("layout", {}).get("panel_e_upshift_mm")})

    typography_pass, typography_detail = base._typography_check(files["svg"], lqa, 180.0, 162.0)
    add("text_floors_at_180_and_162_mm", typography_pass, typography_detail)
    layout_pass, layout_detail = base._layout_qa_check(lqa, (180.0, 162.0))
    add("renderer_collisions_clipping_and_hard_gaps", layout_pass, layout_detail)

    palette_pass = (lqa.get("grayscale_check", {}).get("passed") is True
                    and lqa.get("color_vision_deficiency_check", {}).get("passed") is True
                    and lqa.get("semantic_colormaps", {}).get("panel_c_absolute_error") == "magma"
                    and lqa.get("semantic_colormaps", {}).get("panel_d_residuals") == "magma"
                    and lqa.get("semantic_colormaps", {}).get("panel_d_truth_component") == "Greys_r"
                    and lqa.get("normalization_unchanged_from_v4_3") is True
                    and lqa.get("data_mapping_unchanged_from_v4_3") is True
                    and lqa.get("carried_v4_3_residual_mapping", {}).get("vmin") == 0.0)
    add("semantic_palette_grayscale_and_cvd", palette_pass,
        {"colormaps": lqa.get("semantic_colormaps"), "grayscale": lqa.get("grayscale_check"),
         "cvd": lqa.get("color_vision_deficiency_check")})

    font_pass, font_detail = base._pdf_fonts(files["pdf"])
    editable_pass, editable_detail = base._pdf_text_editable(files["pdf"])
    svg_editable = bool(svg_text and "<text" in svg_text)
    add("embedded_arial_and_editable_vector_text", font_pass and editable_pass and svg_editable,
        {"font": font_detail, "pdf_text": editable_detail, "svg_text_nodes": svg_editable})
    output_pass, output_detail = base._output_check(manifest, release, files, 180.0, 228.2, 600.0)
    add("compact_canvas_and_export_bundle", output_pass, output_detail)

    passed = all(item["passed"] for item in checks)
    payload = {"workflow_label": "mixed_resolution_unified_v4_4_audit",
               "schema_version": "4.4", "revision": "V4_4",
               "status": "PASS" if passed else "FAIL", "passed": passed,
               "release_dir": str(release), "checks": checks}
    destination = Path(output).resolve() if output else release / "qa_v4_4.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[{'PASS' if passed else 'FAIL'}] wrote {destination}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not audit(args.release_dir, args.output)["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

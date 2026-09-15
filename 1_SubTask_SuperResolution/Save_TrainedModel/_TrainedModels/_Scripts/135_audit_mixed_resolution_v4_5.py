#!/usr/bin/env python
"""Independent audit for the frozen-data mixed-resolution V4_5 bundle."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
DEFAULT_RELEASE_ROOT = PROJECT_ROOT / "1_SubTask_SuperResolution/figures/generated/art_style_review"
V4_4_RELEASE = DEFAULT_RELEASE_ROOT / "MixedResolution_unified_v4_4_20260915_0001"
EXPECTED_ROLES = {"panel_label": 11.0, "major_title": 9.5, "subplot_title": 8.5,
                  "axis_label": 8.5, "tick_label": 7.8, "legend": 7.8,
                  "annotation": 7.0}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load("mixed_resolution_v4_5_audit_helpers", HERE / "120_audit_mixed_resolution_v4.py")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _source_lock_check(lock: dict, release: Path) -> tuple[bool, dict]:
    keys = ("v4_5_renderer", "v4_5_exporter", "v4_5_audit",
            "v4_5_panel_renderer", "v4_5_layout", "v4_5_wrapper")
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
    manifest = _read(release / "source_manifest_v4_5.json")
    comparison = _read(release / "SCIENTIFIC_STATE_COMPARISON.json")
    lqa = _read(release / "LAYOUT_QA.json")
    lock = _read(release / "SOURCE_LOCK.json")
    pdfs = sorted(release.glob("MixedResolution_unified_v4_5_*.pdf"))
    svgs = sorted(release.glob("MixedResolution_unified_v4_5_*.svg"))
    pngs = sorted(release.glob("MixedResolution_unified_v4_5_*.png"))
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
    add("v4_5_release_identity_and_bookkeeping",
        all((release / name).is_file() for name in required)
        and manifest.get("revision") == "V4_5"
        and release.name.startswith("MixedResolution_unified_v4_5_"),
        {"revision": manifest.get("revision"),
         "required_files": {name: (release / name).is_file() for name in required}})

    lock_pass, lock_detail = _source_lock_check(lock, release)
    add("source_lock_and_archived_editable_sources", lock_pass, lock_detail)

    comparison_checks = comparison.get("checks", {})
    comparison_pass = (comparison.get("status") == "PASS"
                       and comparison.get("underlying_data_unchanged") is True
                       and comparison.get("v4_5_raw_or_scientific_deltas") is False
                       and comparison.get("v4_5_display_mapping_exact_to_v4_4") is True
                       and bool(comparison_checks) and all(comparison_checks.values()))
    add("scientific_and_display_state_exact_to_v4_4", comparison_pass,
        {"status": comparison.get("status"), "checks": comparison_checks})

    panels = manifest.get("panels", {})
    inventory = {"panel_ids": sorted(panels), "bar_count": svg_text.count("model-bar:"),
                 "sweep_trace_count": svg_text.count("model-line:"),
                 "roi_box_count": svg_text.count("panel-a-roi-box"),
                 "roi_connector_count": svg_text.count("panel-a-roi-connector"),
                 "matrix_value_count": panels.get("e", {}).get("matrix_numeric_annotation_count")}
    inventory_pass = (inventory["panel_ids"] == list("abcde") and inventory["bar_count"] == 20
                      and inventory["sweep_trace_count"] == 12 and inventory["roi_box_count"] == 3
                      and inventory["roi_connector_count"] == 6 and inventory["matrix_value_count"] == 72)
    add("v4_5_panel_and_artist_inventory", inventory_pass, inventory)

    typography = manifest.get("layout", {}).get("typography_qa", {})
    role_qa = manifest.get("layout", {}).get("typography_role_qa", {})
    observed_roles = typography.get("role_sizes_pt", {})
    role_counts = typography.get("visible_text_counts", {})
    typography_pass = (
        observed_roles == EXPECTED_ROLES
        and typography.get("local_size_overrides") == []
        and role_qa.get("local_size_override_count") == 0
        and role_qa.get("only_panel_labels_bold") is True
        and role_qa.get("unexpected_bold_text") == []
        and role_qa.get("major_title_count") == 6
        and role_counts.get("panel_label") == 5
        and role_counts.get("major_title") == 6
        and role_qa.get("matrix_annotation_count") == 72
        and role_qa.get("matrix_annotation_min_contrast_ratio", 0) >= 4.5
        and set(role_qa.get("matrix_annotation_colors", [])) == {"#111111", "#FFFFFF"}
    )
    add("strict_hierarchy_weights_and_matrix_contrast", typography_pass,
        {"role_sizes_pt": observed_roles, "role_counts": role_counts,
         "local_size_overrides": typography.get("local_size_overrides"),
         "major_titles": role_qa.get("major_title_text"),
         "only_panel_labels_bold": role_qa.get("only_panel_labels_bold"),
         "matrix_annotation_count": role_qa.get("matrix_annotation_count"),
         "matrix_min_contrast": role_qa.get("matrix_annotation_min_contrast_ratio")})

    expected_162 = {key: value * .9 for key, value in EXPECTED_ROLES.items()}
    lqa_typography_pass = (lqa.get("role_sizes_pt") == EXPECTED_ROLES
                           and lqa.get("role_sizes_pt_at_162mm") == expected_162
                           and lqa.get("text_weight_policy") == {
                               "panel_label": "bold", "all_other_roles": "regular"})
    add("typography_recorded_at_180_and_162_mm", lqa_typography_pass,
        {"design": lqa.get("role_sizes_pt"), "insertion": lqa.get("role_sizes_pt_at_162mm"),
         "weights": lqa.get("text_weight_policy")})

    layout = manifest.get("layout", {})
    spacing = layout.get("major_content_spacing_qa", {})
    visible_gaps = spacing.get("visible_content_gaps_mm", {})
    by_width = lqa.get("by_width", {})
    layout_pass = (
        abs(float(layout.get("canvas_width_mm", 0)) - 180.0) <= .02
        and abs(float(layout.get("canvas_height_mm", 0)) - 218.2) <= .02
        and layout.get("b_to_cd_compaction_mm") == 6.0
        and spacing.get("visible_content_gap_qa_passed") is True
        and set(visible_gaps) == {"a_b", "b_cd", "cd_e"}
        and min(visible_gaps.values(), default=0) * .9 >= 3.0
        and layout.get("panel_text_clearance_qa", {}).get("passed") is True
        and not any(layout.get("text_overflow_in", {}).values())
        and all(by_width.get(width, {}).get("unexplained_collision_count") == 0
                and by_width.get(width, {}).get("clipped_artist_count") == 0
                and by_width.get(width, {}).get("renderer_bbox_checks", {}).get("passed") is True
                for width in ("180mm", "162mm"))
    )
    add("compact_canvas_no_collisions_clipping_and_hard_gaps", layout_pass,
        {"canvas_mm": [layout.get("canvas_width_mm"), layout.get("canvas_height_mm")],
         "compaction_mm": layout.get("b_to_cd_compaction_mm"),
         "visible_gaps_mm": visible_gaps, "by_width": by_width})

    b, d, e = panels.get("b", {}), panels.get("d", {}), panels.get("e", {})
    panel_pass = (
        b.get("legend_bbox_clearance_passed") is True
        and b.get("sensor_sweep_count") == 3
        and d.get("truth_component_cmap") == "Greys_r"
        and d.get("residual_cmap") == "magma"
        and d.get("residual_display_all_vmin_zero") is True
        and d.get("headers_centered_over_columns") is True
        and max(d.get("header_to_column_center_deltas_mm", [999])) <= .02
        and e.get("scale_tick_label_settings_match_v4_3") is True
        and e.get("v4_4_tick_centering_override_applied") is False
        and e.get("v4_5_matrix_annotation_count") == 72
    )
    add("v4_4_panel_specific_visual_state_preserved", panel_pass,
        {"panel_b_sweeps": b.get("sweep_recipes"),
         "panel_b_legend_clearance": b.get("legend_bbox_clearance_passed"),
         "panel_d_colormaps": [d.get("truth_component_cmap"), d.get("residual_cmap")],
         "panel_d_headers": d.get("header_to_column_center_deltas_mm"),
         "panel_e_tick_source": e.get("scale_tick_label_settings_source")})

    palette_pass = (lqa.get("grayscale_check", {}).get("passed") is True
                    and lqa.get("color_vision_deficiency_check", {}).get("passed") is True
                    and lqa.get("semantic_colormaps", {}).get("panel_c_absolute_error") == "magma"
                    and lqa.get("semantic_colormaps", {}).get("panel_d_residuals") == "magma"
                    and lqa.get("semantic_colormaps", {}).get("panel_d_truth_component") == "Greys_r"
                    and lqa.get("normalization_unchanged_from_v4_4") is True
                    and lqa.get("data_mapping_unchanged_from_v4_4") is True)
    add("semantic_palette_grayscale_and_cvd", palette_pass,
        {"colormaps": lqa.get("semantic_colormaps"),
         "grayscale": lqa.get("grayscale_check"),
         "cvd": lqa.get("color_vision_deficiency_check")})

    font_pass, font_detail = base._pdf_fonts(files["pdf"])
    editable_pass, editable_detail = base._pdf_text_editable(files["pdf"])
    svg_editable = bool(svg_text and "<text" in svg_text)
    add("embedded_arial_and_editable_vector_text", font_pass and editable_pass and svg_editable,
        {"font": font_detail, "pdf_text": editable_detail, "svg_text_nodes": svg_editable})

    output_pass, output_detail = base._output_check(manifest, release, files, 180.0, 218.2, 600.0)
    add("compact_canvas_and_export_bundle", output_pass, output_detail)

    passed = all(item["passed"] for item in checks)
    payload = {"workflow_label": "mixed_resolution_unified_v4_5_audit",
               "schema_version": "4.5", "revision": "V4_5",
               "status": "PASS" if passed else "FAIL", "passed": passed,
               "release_dir": str(release), "checks": checks}
    destination = Path(output).resolve() if output else release / "qa_v4_5.json"
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

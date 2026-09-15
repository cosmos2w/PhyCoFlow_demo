#!/usr/bin/env python
"""Independent audit for the corrected mixed-resolution V4_6 bundle."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[3]
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


base = _load("mixed_resolution_v4_6_audit_helpers", HERE / "120_audit_mixed_resolution_v4.py")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _source_lock_check(lock: dict, release: Path) -> tuple[bool, dict]:
    keys = ("v4_6_renderer", "v4_6_exporter", "v4_6_audit",
            "v4_6_panel_renderer", "v4_6_layout", "v4_6_wrapper")
    detail, passed = {}, True
    for key in keys:
        item = lock.get(key, {})
        path = Path(str(item.get("path", "")))
        observed = base.sha256(path) if path.is_file() else None
        ok = observed == item.get("sha256") and bool(observed)
        detail[key] = {"path": str(path), "recorded": item.get("sha256"),
                       "observed": observed, "passed": ok}
        passed &= ok
    archived = [path for path in (release / "source").glob("*") if path.is_file()]
    detail["archived_source_count"] = len(archived)
    passed &= len(archived) >= 6
    return bool(passed), detail


def audit(release_dir: Path, output: Path | None = None) -> dict:
    release = Path(release_dir).resolve()
    manifest = _read(release / "source_manifest_v4_6.json")
    comparison = _read(release / "SCIENTIFIC_STATE_COMPARISON.json")
    lqa = _read(release / "LAYOUT_QA.json")
    lock = _read(release / "SOURCE_LOCK.json")
    pdfs = sorted(release.glob("MixedResolution_unified_v4_6_*.pdf"))
    svgs = sorted(release.glob("MixedResolution_unified_v4_6_*.svg"))
    pngs = sorted(release.glob("MixedResolution_unified_v4_6_*.png"))
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
    add("v4_6_release_identity_and_bookkeeping",
        all((release / name).is_file() for name in required)
        and manifest.get("revision") == "V4_6" and release.name.startswith("MixedResolution_unified_v4_6_"),
        {name: (release / name).is_file() for name in required})

    lock_pass, lock_detail = _source_lock_check(lock, release)
    add("source_lock_and_archived_editable_sources", lock_pass, lock_detail)

    comparison_checks = comparison.get("checks", {})
    comparison_pass = (comparison.get("status") == "PASS"
                       and comparison.get("underlying_data_unchanged") is True
                       and comparison.get("v4_6_raw_or_scientific_deltas") is False
                       and comparison.get("v4_6_display_mapping_exact_to_v4_5") is True
                       and bool(comparison_checks) and all(comparison_checks.values()))
    add("scientific_and_display_state_exact_to_v4_5", comparison_pass,
        {"status": comparison.get("status"), "checks": comparison_checks})

    panels = manifest.get("panels", {})
    inventory = {"panel_ids": sorted(panels), "bar_count": svg_text.count("model-bar:"),
                 "sweep_trace_count": svg_text.count("model-line:"),
                 "roi_box_count": svg_text.count("panel-a-roi-box"),
                 "roi_connector_count": svg_text.count("panel-a-roi-connector"),
                 "matrix_value_count": panels.get("e", {}).get("matrix_numeric_annotation_count")}
    add("v4_6_panel_and_artist_inventory",
        inventory == {"panel_ids": list("abcde"), "bar_count": 20, "sweep_trace_count": 12,
                      "roi_box_count": 3, "roi_connector_count": 6, "matrix_value_count": 72}, inventory)

    typography = manifest.get("layout", {}).get("typography_qa", {})
    role_qa = manifest.get("layout", {}).get("typography_role_qa", {})
    observed_roles = typography.get("role_sizes_pt", {})
    typography_pass = (
        observed_roles == EXPECTED_ROLES and typography.get("local_size_overrides") == []
        and role_qa.get("local_size_override_count") == 0
        and role_qa.get("only_panel_labels_bold") is True
        and role_qa.get("unexpected_bold_text") == []
        and role_qa.get("major_title_count") == 2
        and role_qa.get("locally_reduced_text_count") == 3
        and role_qa.get("panel_b_zero_h_text_present") is False
        and role_qa.get("matrix_annotation_count") == 72
        and role_qa.get("matrix_annotation_min_contrast_ratio", 0) >= 4.5
    )
    add("strict_hierarchy_and_requested_local_reductions", typography_pass,
        {"role_sizes_pt": observed_roles, "role_qa": role_qa})

    layout = manifest.get("layout", {})
    union = layout.get("adjacent_major_row_union_qa", {})
    adjacent = union.get("adjacent_major_row_checks", [])
    layout_pass = (
        abs(float(layout.get("canvas_width_mm", 0)) - 180.0) <= .02
        and abs(float(layout.get("canvas_height_mm", 0)) - 229.5) <= .02
        and union.get("passed") is True and len(adjacent) == 3
        and all(item.get("passed") is True and item.get("forbidden_intersection_count") == 0
                and item.get("vertical_gap_mm_at_162mm", 0) >= 3.0 for item in adjacent)
        and union.get("unknown_visible_artist_count") == 0
        and union.get("unowned_visible_figure_text_count") == 0
        and union.get("missing_required_bbox_count") == 0
        and union.get("clipped_artist_count") == 0
        and lqa.get("vector_source_render_at_162mm") is True
        and lqa.get("by_width", {}).get("162mm", {}).get("renderer_bbox_checks", {}).get("passed") is True
    )
    add("rendered_union_gate_canvas_and_true_insertion_render", layout_pass,
        {"canvas_mm": [layout.get("canvas_width_mm"), layout.get("canvas_height_mm")],
         "union": union, "vector_source_render_at_162mm": lqa.get("vector_source_render_at_162mm")})

    b, c, d, e = (panels.get(key, {}) for key in "bcde")
    panel_pass = (
        b.get("grouped_bar_axis_height_exactly_1_25x_v4_5") is True
        and abs(float(b.get("grouped_bar_axis_height_ratio_to_v4_5", 0)) - 1.25) <= 1e-8
        and b.get("legend_bbox_clearance_passed") is True
        and b.get("y_axis_titles_aligned") is True
        and b.get("y_axis_title_center_delta_mm", 999) <= .02
        and b.get("zero_h_label_removed_v4_6") is True
        and c.get("error_colorbar_title_after") == "Absolute error"
        and c.get("shared_colorbar_count") == 2
        and layout.get("shared_cd_row_gap_mm") == 3.0
        and layout.get("panel_c_colorbar_strip_mm") == 18.0
        and d.get("headers_centered_over_columns") is True
        and max(d.get("header_to_column_center_deltas_mm", [999])) <= .02
        and e.get("scale_tick_label_settings_match_v4_3") is True
    )
    add("requested_panel_geometry_text_and_alignment", panel_pass,
        {"panel_b": b, "panel_c": {key: c.get(key) for key in ("error_colorbar_title_after", "shared_colorbar_count")},
         "panel_d_header_delta_mm": d.get("header_to_column_center_deltas_mm"),
         "row_gap_mm": layout.get("shared_cd_row_gap_mm"),
         "colorbar_strip_mm": layout.get("panel_c_colorbar_strip_mm")})

    palette_pass = (lqa.get("semantic_colormaps", {}).get("panel_c_absolute_error") == "magma"
                    and lqa.get("semantic_colormaps", {}).get("panel_d_residuals") == "magma"
                    and lqa.get("semantic_colormaps", {}).get("panel_d_truth_component") == "Greys_r"
                    and lqa.get("normalization_unchanged_from_v4_5") is True
                    and lqa.get("data_mapping_unchanged_from_v4_5") is True)
    add("semantic_palette_and_mapping_freeze", palette_pass, lqa.get("semantic_colormaps"))

    font_pass, font_detail = base._pdf_fonts(files["pdf"])
    editable_pass, editable_detail = base._pdf_text_editable(files["pdf"])
    add("embedded_arial_and_editable_vector_text", font_pass and editable_pass and "<text" in svg_text,
        {"font": font_detail, "pdf_text": editable_detail, "svg_text_nodes": "<text" in svg_text})

    output_pass, output_detail = base._output_check(manifest, release, files, 180.0, 229.5, 600.0)
    add("canvas_and_export_bundle", output_pass, output_detail)

    passed = all(item["passed"] for item in checks)
    payload = {"workflow_label": "mixed_resolution_unified_v4_6_audit",
               "schema_version": "4.6", "revision": "V4_6",
               "status": "PASS" if passed else "FAIL", "passed": passed,
               "release_dir": str(release), "checks": checks}
    destination = Path(output).resolve() if output else release / "qa_v4_6.json"
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

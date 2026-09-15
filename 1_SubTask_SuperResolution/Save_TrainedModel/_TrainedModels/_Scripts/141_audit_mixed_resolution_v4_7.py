#!/usr/bin/env python
"""Focused visual/layout audit for mixed-resolution Figure V4_7."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
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


base = _load("mixed_resolution_v4_7_audit_helpers", HERE / "120_audit_mixed_resolution_v4.py")


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _source_lock_check(lock: dict, release: Path) -> tuple[bool, dict]:
    keys = ("v4_7_renderer", "v4_7_exporter", "v4_7_audit",
            "v4_7_panel_renderer", "v4_7_layout", "v4_7_wrapper")
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
    manifest = _read(release / "source_manifest_v4_7.json")
    comparison = _read(release / "SCIENTIFIC_STATE_COMPARISON.json")
    lqa = _read(release / "LAYOUT_QA.json")
    lock = _read(release / "SOURCE_LOCK.json")
    pdfs = sorted(release.glob("MixedResolution_unified_v4_7_*.pdf"))
    svgs = sorted(release.glob("MixedResolution_unified_v4_7_*.svg"))
    pngs = sorted(release.glob("MixedResolution_unified_v4_7_*.png"))
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
    add("v4_7_release_identity_and_bookkeeping",
        all((release / name).is_file() for name in required)
        and manifest.get("revision") == "V4_7" and release.name.startswith("MixedResolution_unified_v4_7_"),
        {name: (release / name).is_file() for name in required})

    lock_pass, lock_detail = _source_lock_check(lock, release)
    add("source_lock_and_archived_editable_sources", lock_pass, lock_detail)

    continuity = comparison.get("checks", {})
    continuity_pass = (
        comparison.get("status") == "PASS"
        and comparison.get("underlying_data_unchanged") is True
        and comparison.get("scope") == "focused visual-only source continuity"
        and continuity and all(continuity.values())
    )
    add("focused_visual_only_source_continuity", continuity_pass, comparison)

    panels = manifest.get("panels", {})
    inventory = {"panel_ids": sorted(panels), "bar_count": svg_text.count("model-bar:"),
                 "sweep_trace_count": svg_text.count("model-line:"),
                 "roi_box_count": svg_text.count("panel-a-roi-box"),
                 "roi_connector_count": svg_text.count("panel-a-roi-connector"),
                 "matrix_value_count": panels.get("e", {}).get("matrix_numeric_annotation_count")}
    add("panel_and_artist_inventory_unchanged",
        inventory == {"panel_ids": list("abcde"), "bar_count": 20, "sweep_trace_count": 12,
                      "roi_box_count": 3, "roi_connector_count": 6, "matrix_value_count": 72}, inventory)

    typography = manifest.get("layout", {}).get("typography_qa", {})
    role_qa = manifest.get("layout", {}).get("typography_role_qa", {})
    add("strict_typography_hierarchy",
        typography.get("role_sizes_pt") == EXPECTED_ROLES
        and typography.get("local_size_overrides") == []
        and role_qa.get("only_panel_labels_bold") is True
        and role_qa.get("matrix_annotation_count") == 72,
        {"role_sizes_pt": typography.get("role_sizes_pt"), "role_qa": role_qa})

    layout = manifest.get("layout", {})
    gate = layout.get("annotation_collision_qa", {})
    union = layout.get("adjacent_major_row_union_qa", {})
    adjacent = union.get("adjacent_major_row_checks", [])
    layout_pass = (
        abs(float(layout.get("canvas_width_mm", 0)) - 180.0) <= .02
        and abs(float(layout.get("canvas_height_mm", 0)) - 229.5) <= .02
        and gate.get("passed") is True
        and gate.get("run_after_final_aspect_and_axes_positioning") is True
        and gate.get("target_annotation_count") == 11
        and gate.get("text_data_intersection_count") == 0
        and gate.get("annotation_annotation_intersection_count") == 0
        and gate.get("minimum_observed_clearance_mm_at_180mm", 0) >= 1.0
        and union.get("passed") is True and len(adjacent) == 3
        and all(item.get("vertical_gap_mm_at_162mm", 0) >= 3.0 for item in adjacent)
        and lqa.get("vector_source_render_at_162mm") is True
    )
    add("final_aspect_collision_and_major_row_gates", layout_pass,
        {"annotation_gate": gate, "adjacent": adjacent,
         "vector_source_render_at_162mm": lqa.get("vector_source_render_at_162mm")})

    b, c, d, e = (panels.get(key, {}) for key in "bcde")
    panel_pass = (
        b.get("grouped_bar_height_preserved_from_v4_6") is True
        and b.get("minimum_sweep_height_ratio_to_v4_6", 0) >= 1.10
        and b.get("legend_bbox_clearance_passed") is True
        and b.get("y_axis_titles_aligned") is True
        and c.get("shared_colorbar_count") == 0
        and c.get("colorbar_axes_removed_v4_7") == 2
        and c.get("panel_c_panel_d_bottom_labels_aligned") is True
        and c.get("panel_c_panel_d_bottom_alignment_delta_mm", 999) <= .02
        and d.get("headers_centered_over_columns") is True
        and e.get("scale_tick_label_settings_match_v4_3") is True
    )
    add("requested_panel_geometry_and_label_changes", panel_pass,
        {"panel_b": b, "panel_c": c, "panel_d": d, "panel_e": e})

    palette = lqa.get("semantic_colormaps", {})
    add("semantic_palette_and_mapping_freeze",
        palette.get("panel_c_absolute_error") == "magma"
        and palette.get("panel_d_residuals") == "magma"
        and palette.get("panel_d_truth_component") == "Greys_r"
        and lqa.get("data_mapping_unchanged_from_v4_6") is True,
        palette)

    font_pass, font_detail = base._pdf_fonts(files["pdf"])
    editable_pass, editable_detail = base._pdf_text_editable(files["pdf"])
    add("embedded_arial_and_editable_vector_text", font_pass and editable_pass and "<text" in svg_text,
        {"font": font_detail, "pdf_text": editable_detail, "svg_text_nodes": "<text" in svg_text})

    output_pass, output_detail = base._output_check(manifest, release, files, 180.0, 229.5, 600.0)
    add("canvas_and_export_bundle", output_pass, output_detail)

    passed = all(item["passed"] for item in checks)
    payload = {"workflow_label": "mixed_resolution_unified_v4_7_audit",
               "schema_version": "4.7", "revision": "V4_7",
               "status": "PASS" if passed else "FAIL", "passed": passed,
               "release_dir": str(release), "checks": checks}
    destination = Path(output).resolve() if output else release / "qa_v4_7.json"
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


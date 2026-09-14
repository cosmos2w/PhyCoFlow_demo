#!/usr/bin/env python
"""Strict additive 25-point audit for the mixed-resolution V3-4 figure.

The V3-4 contract is a publication-polish revision of the validated V3-3
figure.  This audit therefore checks both the explicit V3-4 layout contract
and the scientific/provenance invariants carried forward from V3-3.  It reads
only release metadata, exported artifacts, validated CSV summaries, and
validated reconstruction caches; it never renders, trains, infers, or writes
source/result data.  Running the audit writes only ``qa_v3_4.json`` in the
requested additive release directory.

V3-4 renderers should expose the geometry/style decisions in panel metadata.
The accepted aliases below intentionally cover the V3-3 field names plus the
more descriptive V3-4 names (for example ``inset_layout`` and
``inset_style_contract``).  A missing required decision is a failed check,
not an inferred pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

MODELS = ["DMFGen", "FFM_Perceiver", "Senseiver", "MLP_RBF"]
RECIPES = [
    "1_H_only",
    "2_H_limited",
    "3_Mixed_HML",
    "4_ZeroH_Balanced",
    "5_ZeroH_MRich",
]
ZERO_H = RECIPES[3:]
FINE_RECIPES = RECIPES[2:]
COUNTS = [64, 128, 256, 384, 512]
SCALES = ["large", "intermediate", "fine"]
METRICS = ["pattern_correlation", "variance_fraction_bias_pp"]
ROLE_SIZES = {
    "panel_label": 8.5,
    "subplot_title": 6.5,
    "axis_label": 6.0,
    "tick_label": 5.5,
    "legend": 5.5,
    "annotation": 5.5,
}

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ASSEMBLED_DIR = (
    PROJECT_ROOT
    / "1_SubTask_SuperResolution/Save_TrainedModel/_TrainedModels/_Process_Figures/Assembled"
)
V3_2_BASELINE_PDF = ASSEMBLED_DIR / "MixedResolution_unified_v3_2_hybrid_20260913_1739.pdf"
V3_2_BASELINE_MANIFEST = ASSEMBLED_DIR / "FigureSourceManifest_unified_v3_2_20260913_1739.json"
V2_BASELINE_PDF = ASSEMBLED_DIR / "MixedResolution_unified_v2_phase2_20260903_2213.pdf"
V2_BASELINE_MANIFEST = ASSEMBLED_DIR / "FigureSourceManifest_unified_v2_20260903_2213.json"
KNOWN_BASELINE_SHA256 = {
    str(V3_2_BASELINE_PDF): "178abe243847baf876eeefa951b6f5ba2c5dd26f13fe81d29b3b29737a715939",
    str(V3_2_BASELINE_MANIFEST): "6787146e8e13149958241d4806fc6e682affac0ff36459b48456b4dfaabfac53",
    str(V2_BASELINE_PDF): "60a613800df6dac8aac43720f3b081b0927474e9c9070e1ab5bf026ef896cefb",
    str(V2_BASELINE_MANIFEST): "48e4860c2fbbe25826adf4d28799aa4cc86fc9f000fc0125a2ebb168a42a361f",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def close(left, right, atol=1e-12, rtol=0.0) -> bool:
    try:
        return bool(np.isclose(float(left), float(right), atol=atol, rtol=rtol))
    except (TypeError, ValueError):
        return False


def relative_l2(truth, prediction) -> float:
    truth = np.asarray(truth, dtype=float).reshape(-1)
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if truth.size != prediction.size:
        raise ValueError(f"relative L2 shape mismatch: {truth.shape} vs {prediction.shape}")
    valid = np.isfinite(truth) & np.isfinite(prediction)
    if not np.any(valid):
        raise ValueError("relative L2 has no finite paired values")
    diff = prediction[valid] - truth[valid]
    numerator = np.sqrt(np.sum(diff * diff, dtype=np.float64))
    denominator = np.sqrt(np.sum(truth[valid] * truth[valid], dtype=np.float64))
    return float(numerator / (denominator + 1e-12))


def _normalise_text(value) -> str:
    return str(value).replace("\\n", " ").replace("\n", " ").replace("$", "")


def _contains_text(text: str, phrase: str) -> bool:
    return phrase in _normalise_text(text)


def _first(mapping, *keys, default=None):
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _flag(mapping, *keys) -> bool:
    return any(isinstance(mapping, dict) and mapping.get(key) is True for key in keys)


def _qa_passed(mapping, *keys) -> bool:
    if not isinstance(mapping, dict):
        return False
    if mapping.get("passed") is True:
        return True
    for key in keys:
        value = mapping.get(key)
        if value is True:
            return True
        if isinstance(value, dict) and value.get("passed") is True:
            return True
    return False


def _resolve_path(value, release: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    for candidate in (release / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return release / path


def _record_path(record, release: Path):
    if not isinstance(record, dict) or not record.get("path"):
        return None
    return _resolve_path(record["path"], release)


def _records_from(value):
    """Yield hash-bearing path records from nested manifest values."""
    if isinstance(value, dict):
        if value.get("path") and value.get("sha256"):
            yield value
        for child in value.values():
            yield from _records_from(child)
    elif isinstance(value, list):
        for child in value:
            yield from _records_from(child)


def _unique_records(records):
    unique = {}
    for record in records:
        key = (str(record.get("path")), str(record.get("sha256")))
        unique[key] = record
    return list(unique.values())


def _source_records(manifest):
    fields = (
        "configuration",
        "layout_configuration",
        "renderer",
        "panel_renderer",
        "cache_manifest",
        "representative_index",
        "v3_3_starting_renderer_anchor",
        "v3_4_starting_renderer_anchor",
        "csv_sources",
        "all_render_source_records",
        "cache_sources",
        "all_render_cache_sources",
        "referenced_shared_sources",
        "table_sources",
        "extended_multiscale_distribution_source",
        "per_snapshot_distribution_source",
    )
    records = []
    for field in fields:
        records.extend(_records_from(manifest.get(field)))
    return _unique_records(records)


def _artifact_records(manifest):
    records = list(_records_from(manifest.get("artifact_outputs", [])))
    if records:
        # The packaged release is the canonical artifact scope.  The explicit
        # assembler/exporter lists below point to the pre-copy process tree and
        # are provenance records, not release destinations.
        return _unique_records(records)
    # Some V3-4 exporters retain the explicit output lists in addition to the
    # canonical artifact_outputs list.  Include them when present, while
    # deduplicating by path/hash.
    records.extend(_records_from(manifest.get("outputs", [])))
    records.extend(_records_from(manifest.get("standalone_outputs", [])))
    records.extend(_records_from(manifest.get("si_outputs", {})))
    records.extend(_records_from(manifest.get("table_outputs", {})))
    return _unique_records(records)


def _hash_audit(records, release: Path):
    details = []
    passed = True
    for record in records:
        path = _record_path(record, release)
        exists = bool(path and path.exists() and path.is_file())
        observed = sha256(path) if exists else None
        item_passed = exists and observed == record.get("sha256")
        passed &= item_passed
        details.append({
            "path": str(path) if path else record.get("path"),
            "exists": exists,
            "expected_sha256": record.get("sha256"),
            "observed_sha256": observed,
            "passed": item_passed,
        })
    return bool(passed and records), details


def _find_panel_svg(release: Path, label: str, run_id: str):
    panel_dir = release / "panels"
    candidates = sorted(panel_dir.glob(f"Panel_{label}_*_{run_id}.svg"))
    if not candidates:
        candidates = sorted(panel_dir.glob(f"Panel_{label}_*.svg"))
    return candidates[0] if candidates else None


def _find_main_file(release: Path, run_id: str, suffix: str):
    exact = release / f"MixedResolution_unified_v3_4_{run_id}.{suffix}"
    if exact.exists():
        return exact
    candidates = sorted(release.glob(f"MixedResolution_unified_v3_4_*_{run_id}.{suffix}"))
    if not candidates:
        candidates = sorted(release.glob(f"MixedResolution_unified_v3_4_*.{suffix}"))
    return candidates[0] if candidates else exact


def _source_path(manifest, release: Path, prefix: str):
    records = _source_records(manifest)
    for record in records:
        path = _record_path(record, release)
        if path and path.name.startswith(prefix):
            return path
    raise FileNotFoundError(f"No manifest source starts with {prefix!r}")


def _read_metadata_json(value):
    if isinstance(value, np.ndarray):
        value = value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, list):
        value = value[0]
    return json.loads(str(value))


def _recorded_baseline_paths(manifest, release: Path):
    records = []
    for key in (
        "old_v2_baseline_pdf",
        "old_v2_baseline_manifest",
        "baseline_v2_pdf",
        "baseline_v2_manifest",
        "baseline_v3_2_pdf",
        "baseline_v3_2_manifest",
        "v3_3_baseline_pdf",
        "v3_3_baseline_manifest",
        "baseline_immutability",
        "v3_3_starting_renderer_anchor",
        "v3_4_starting_renderer_anchor",
    ):
        records.extend(_records_from(manifest.get(key)))
    # ``baseline_immutability`` in V3-3/V3-4 stores path -> hash maps rather
    # than record objects.  Materialize those maps so a changed baseline is
    # caught instead of merely checking that it still exists.
    baseline_immutability = manifest.get("baseline_immutability", {})
    if isinstance(baseline_immutability, dict):
        for field in ("sha256_before", "sha256_after"):
            hashes = baseline_immutability.get(field, {})
            if isinstance(hashes, dict):
                records.extend({"path": path, "sha256": digest}
                               for path, digest in hashes.items())
    # The fixed V2/V3-2 baselines are part of the additive contract.  If the
    # manifest already records an alias, the hash audit deduplicates it.
    for path in (V2_BASELINE_PDF, V2_BASELINE_MANIFEST, V3_2_BASELINE_PDF, V3_2_BASELINE_MANIFEST):
        if path.exists():
            records.append({"path": str(path), "sha256": KNOWN_BASELINE_SHA256[str(path)]})
    return _unique_records(records)


def _pdf_size_mm(path: Path):
    output = subprocess.run(
        ["pdfinfo", str(path)], check=True, capture_output=True, text=True,
    ).stdout
    line = next(line for line in output.splitlines() if line.startswith("Page size:"))
    parts = line.split()
    return [float(parts[2]) * 25.4 / 72.0, float(parts[4]) * 25.4 / 72.0]


def _pdf_font_lines(path: Path):
    output = subprocess.run(
        ["pdffonts", str(path)], check=True, capture_output=True, text=True,
    ).stdout
    return [line for line in output.splitlines()[2:] if line.strip()]


def _geometry_bounds(meta, *keys):
    for key in keys:
        value = meta.get(key) if isinstance(meta, dict) else None
        if isinstance(value, list) and value:
            return value
        if isinstance(value, dict) and value:
            return list(value.values())
    return []


def _panel_text_count(svg: str, marker: str) -> int:
    return svg.count(marker)


def _all_expected_matrix_keys():
    return {
        (model, recipe, scale, metric)
        for model in MODELS
        for recipe in FINE_RECIPES
        for scale in SCALES
        for metric in METRICS
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        help="Optional source_manifest_v3_4.json path; defaults to release-dir/source_manifest_v3_4.json",
    )
    args = parser.parse_args()
    release = args.release_dir.resolve()
    manifest_path = (args.manifest or (release / "source_manifest_v3_4.json")).resolve()
    qa_path = release / "qa_v3_4.json"
    checks = []

    def check(index, name, function):
        try:
            result = function()
            if isinstance(result, tuple):
                passed, detail = result
            else:
                passed, detail = result, None
            item = {
                "contract_index": int(index),
                "name": name,
                "passed": bool(passed),
                "detail": detail,
            }
        except Exception as exc:  # A missing source is a failed audit item.  # noqa: BLE001
            item = {
                "contract_index": int(index),
                "name": name,
                "passed": False,
                "detail": {"type": type(exc).__name__, "message": str(exc)},
            }
        checks.append(item)
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {index:02d} {name}")
        return item

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        manifest = {}
        error = {"path": str(manifest_path), "type": type(exc).__name__, "message": str(exc)}
        check(0, "audit_manifest_load", lambda error=error: (False, error))

    panels = manifest.get("panels", {}) if isinstance(manifest, dict) else {}
    a, b, c, d, e = (panels.get(label, {}) for label in "abcde")
    panel_svgs = {
        label: _find_panel_svg(release, label, args.run_id) for label in "abcde"
    }
    svg = {
        label: path.read_text(encoding="utf-8") if path and path.exists() else ""
        for label, path in panel_svgs.items()
    }

    # ------------------------------------------------------------------
    # V3-4 contract items 1--7: panel a.
    # ------------------------------------------------------------------
    check(1, "panel_a_insets_nested_bottom_right", lambda: (
        int(_first(a, "zoom_inset_count", "inset_count", default=0)) == 3
        and _flag(a, "insets_nested_bottom_right", "nested_insets")
        and str(_first(a, "inset_placement", "inset_layout", default="")).lower()
        in {"nested_bottom_right", "bottom_right_nested"},
        {
            "inset_count": _first(a, "zoom_inset_count", "inset_count"),
            "placement": _first(a, "inset_placement", "inset_layout"),
        },
    ))

    def panel_a_style():
        style = _first(a, "inset_style_contract", "inset_style", default={})
        style = style if isinstance(style, dict) else {}
        border = str(_first(style, "border_color", "box_color", default="")).lower()
        connector = str(_first(style, "connector_color", "line_color", default="")).lower()
        fill = _first(style, "preserve_parent_colormap", "same_colormap", default=False)
        crisp = _first(style, "crisp", "pixelated", "native_cells", default=False)
        passed = (
            _flag(a, "inset_style_matches_reference")
            or _qa_passed(a, "reference_style_qa")
            or (
                border in {"black", "#000", "#000000", "near-black", "#111111", "#222222"}
                and connector in {"black", "#000", "#000000", "near-black", "#111111", "#222222"}
                and fill is True and crisp is True
            )
        )
        return passed, {"style": style, "svg_inset_boxes": _panel_text_count(svg["a"], "panel-a-inset")}

    check(2, "panel_a_inset_style_approximates_reference", panel_a_style)
    check(3, "panel_a_side_resolution_tags_removed", lambda: (
        _first(a, "side_resolution_tags_present", "free_floating_resolution_tags_present",
               default=True) is False
        and _flag(a, "side_resolution_tags_removed", "resolution_side_tags_removed")
        and _panel_text_count(svg["a"], "panel-a-side-resolution-tag") == 0,
        {"side_tags_present": _first(a, "side_resolution_tags_present",
                                      "free_floating_resolution_tags_present")},
    ))

    def panel_a_spacing():
        spacing = _first(a, "heatmap_spacing_qa", "thumbnail_spacing_qa", default={})
        spacing = spacing if isinstance(spacing, dict) else {}
        bounds = _geometry_bounds(a, "heatmap_axes_bounds", "thumbnail_bounds", "heatmap_bounds")
        geometric = False
        if len(bounds) == 3:
            try:
                widths = [float(item[2]) for item in bounds]
                gaps = [float(bounds[i + 1][0]) - float(bounds[i][0]) - widths[i]
                        for i in range(2)]
                geometric = (
                    max(widths) - min(widths) <= 1e-10
                    and max(gaps) - min(gaps) <= 1e-10
                    and min(gaps) >= 0.0
                )
            except (IndexError, TypeError, ValueError):
                geometric = False
        return (
            _qa_passed(spacing, "qa", "passed")
            or _flag(a, "heatmaps_evenly_spaced", "thumbnail_row_evenly_spaced")
            or geometric,
            {"spacing": spacing, "bounds": bounds},
        )

    check(4, "panel_a_heatmaps_evenly_spaced", panel_a_spacing)
    def panel_a_labels():
        rows = _first(a, "resolution_label_rows", "resolution_block_rows", default=[])
        normalised = [
            {
                "name": _first(row, "name", "line1", "label", default=""),
                "dimensions": _first(row, "dimensions", "line2", "grid_size", default=""),
            }
            for row in rows if isinstance(row, dict)
        ]
        expected = [
            {"name": "Low resolution", "dimensions": "32 × 32"},
            {"name": "Medium resolution", "dimensions": "64 × 64"},
            {"name": "High resolution", "dimensions": "128 × 128"},
        ]
        dimensions_ok = [
            (_normalise_text(item["name"]), _normalise_text(item["dimensions"]))
            for item in normalised
        ] == [
            (_normalise_text(item["name"]), _normalise_text(item["dimensions"]))
            for item in expected
        ]
        return (
            _flag(a, "two_line_resolution_labels", "resolution_labels_two_lines")
            and dimensions_ok
            and _contains_text(svg["a"], "Low resolution")
            and _contains_text(svg["a"], "Medium resolution")
            and _contains_text(svg["a"], "High resolution")
            and all(token in svg["a"] for token in ("32", "64", "128")),
            {"labels": rows, "normalised": normalised},
        )

    check(5, "panel_a_two_line_resolution_labels", panel_a_labels)
    check(6, "panel_a_exposure_explanatory_label_removed", lambda: (
        _first(a, "exposure_explanatory_label_present", default=True) is False
        and not _contains_text(svg["a"], "Values above bars: relative spatial-field exposure"),
        None,
    ))
    check(7, "panel_a_orphaned_vertical_gap_removed", lambda: (
        _first(a, "orphaned_vertical_gap_present", "orphan_gap_present", default=True) is False
        and (_flag(a, "compact_heatmap_bar_spacing")
             or _qa_passed(a, "vertical_gap_qa")
             or _qa_passed(_first(a, "heatmap_bar_spacing_qa", default={}), "passed")),
        {"heatmap_bar_spacing_qa": _first(a, "heatmap_bar_spacing_qa")},
    ))

    # ------------------------------------------------------------------
    # V3-4 contract items 8--13: panel b.
    # ------------------------------------------------------------------
    legend = b.get("legend_contract", {})
    legend = legend if isinstance(legend, dict) else {}
    check(8, "panel_b_legend_between_plot_blocks", lambda: (
        str(_first(legend, "placement", "location", "anchor_region", default="")).lower()
        in {"between_blocks", "between_plot_blocks", "horizontal_gap", "inter_block_gap"}
        and _first(legend, "top_right", "top_right_corner", default=False) is False
        and (_flag(b, "legend_between_plot_blocks", "legend_anchored_between_blocks")
             or _first(legend, "dedicated_axis", default=False) is True),
        legend,
    ))
    check(9, "panel_b_legend_visual_dominance_reduced", lambda: (
        5.5 <= float(_first(legend, "fontsize_pt", "font_size_pt", default=0)) < 7.2
        and float(_first(legend, "columnspacing", default=0)) >= 1.5
        and float(_first(legend, "handletextpad", default=0)) >= .45
        and (_flag(legend, "font_size_reduced", "lighter_than_v3_3")
             or _qa_passed(legend, "dominance_qa")
             or _qa_passed(legend.get("dominance", {}), "passed")),
        legend,
    ))
    check(10, "panel_b_top_title_renamed", lambda: (
        _first(b, "top_axis_title", "recipe_transfer_title", "title_text", default="")
        == "High resolution reconstruction (512 sensors)"
        and _contains_text(svg["b"], "High resolution reconstruction (512 sensors)"),
        {"metadata_title": _first(b, "top_axis_title", "recipe_transfer_title", "title_text")},
    ))
    check(11, "panel_b_top_title_inside_axis", lambda: (
        _flag(b, "top_title_inside_axis", "title_inside_top_axis")
        and str(_first(b, "top_title_placement", "title_placement", default="")).lower()
        in {"inside_top_axis", "top_left_inside", "inside_negative_space"},
        {"placement": _first(b, "top_title_placement", "title_placement")},
    ))
    check(12, "panel_b_zero_h_annotation_nonoverlap", lambda: (
        _flag(b, "zero_h_annotation_nonoverlap", "no_h_annotation_nonoverlap")
        and _first(b, "zero_h_annotation_overlap", "no_h_annotation_overlap", default=True) is False
        and (_qa_passed(_first(b, "annotation_collision_qa", default={}), "passed")
             or _qa_passed(_first(b, "zero_h_annotation_qa", default={}), "passed")),
        {"annotation_qa": _first(b, "annotation_collision_qa", "zero_h_annotation_qa")},
    ))
    check(13, "panel_b_vertical_pacing_increased", lambda: (
        _flag(b, "vertical_pacing_increased", "hspace_increased_vs_v3_3")
        and (_qa_passed(_first(b, "vertical_pacing_qa", "subplot_spacing_qa", default={}), "passed")
             or float(_first(b, "inter_block_gap", "hspace", default=0)) >= .05),
        {"pacing": _first(b, "vertical_pacing_qa", "subplot_spacing_qa"),
         "hspace": _first(b, "hspace", "inter_block_gap")},
    ))

    # ------------------------------------------------------------------
    # V3-4 contract items 14--18: panel c.
    # ------------------------------------------------------------------
    check(14, "panel_c_first_row_relative_l2_removed", lambda: (
        _first(c, "full_field_relative_l2_annotations", "first_row_relative_l2_count", default=1) == 0
        and _first(c, "full_row_metric_labels_present", default=True) is False
        and _panel_text_count(svg["c"], "qualitative-full-relative-l2") == 0,
        {"full_row_metric_count": _first(c, "full_field_relative_l2_annotations",
                                           "first_row_relative_l2_count")},
    ))
    check(15, "panel_c_frustum_connectors_dashed_black", lambda: (
        int(c.get("frustum_connector_count", 0)) == 10
        and _panel_text_count(svg["c"], "panel-c-frustum-connector") >= 10
        and str(_first(c, "frustum_connector_linestyle", "frustum_line_style", default="")).lower()
        in {"--", "dashed", "dash"}
        and str(_first(c, "frustum_connector_color", "frustum_line_color", default="")).lower()
        in {"black", "#000", "#000000", "near-black", "#111111", "#222222"}
        and "stroke-dasharray" in svg["c"],
        {"connector_count": c.get("frustum_connector_count"),
         "linestyle": _first(c, "frustum_connector_linestyle", "frustum_line_style"),
         "color": _first(c, "frustum_connector_color", "frustum_line_color")},
    ))
    check(16, "panel_c_zoom_tiles_black_bordered", lambda: (
        int(_first(c, "zoom_tile_black_border_count", "zoom_black_border_count", default=0)) == 5
        and _flag(c, "zoom_tiles_have_black_borders", "zoom_tile_borders_qa")
        and _panel_text_count(svg["c"], "panel-c-zoom-border") >= 5,
        {"border_count": _first(c, "zoom_tile_black_border_count", "zoom_black_border_count"),
         "svg_borders": _panel_text_count(svg["c"], "panel-c-zoom-border")},
    ))
    check(17, "panel_c_sensor_layout_footprint_matches", lambda: (
        c.get("sensor_layout", {}).get("placement") == "full_equal_footprint_tile"
        and float(c.get("sensor_error_footprint_max_abs_delta", 1.0)) <= 1e-10,
        {"sensor_layout": c.get("sensor_layout"),
         "footprint_delta": c.get("sensor_error_footprint_max_abs_delta")},
    ))
    check(18, "panel_c_colorbar_exponents_unclipped", lambda: (
        len(c.get("colorbar_formatting", {})) == 2
        and all(
            item.get("automatic_offset_suppressed") is True
            and float(item.get("minimum_right_padding_axes", 0)) >= .04
            for item in c.get("colorbar_formatting", {}).values()
        )
        and (_flag(c, "colorbar_exponents_clear")
             or _qa_passed(c, "colorbar_exponent_clipping_qa")
             or _qa_passed(c.get("colorbar_exponent_qa", {}), "passed"))
        and _panel_text_count(svg["c"], "colorbar-manual-exponent") >= 2,
        c.get("colorbar_formatting"),
    ))

    # ------------------------------------------------------------------
    # V3-4 contract items 19--21: panel d.
    # ------------------------------------------------------------------
    check(19, "panel_d_truth_component_visually_distinct", lambda: (
        _flag(d, "truth_component_visually_distinct", "truth_component_distinct_from_panel_c")
        and str(_first(d, "truth_component_visual_encoding", "truth_component_encoding", default="")).lower()
        in {"distinct_colormap", "contour_overlay", "grayscale", "desaturated", "spectral"}
        and (_panel_text_count(svg["d"], "panel-d-truth-distinct") >= 1
             or _flag(d, "truth_component_distinct_qa")
             or _qa_passed(d, "truth_component_distinct_qa")),
        {"encoding": _first(d, "truth_component_visual_encoding", "truth_component_encoding"),
         "marker_count": _panel_text_count(svg["d"], "panel-d-truth-distinct")},
    ))
    check(20, "panel_d_fine_row_connector_present", lambda: (
        _flag(d, "fine_row_connector_present", "fine_scale_connector_present")
        and _panel_text_count(svg["d"], "panel-d-fine-row-connector") >= 1
        and str(_first(d, "fine_row_connector_style", "fine_connector_style", default="")).lower()
        in {"subtle", "light_arrow", "funnel", "bracket_arrow", "minimal"},
        {"style": _first(d, "fine_row_connector_style", "fine_connector_style"),
         "marker_count": _panel_text_count(svg["d"], "panel-d-fine-row-connector")},
    ))
    check(21, "panel_d_vertical_pacing_increased", lambda: (
        _flag(d, "vertical_pacing_increased", "hspace_increased_vs_v3_3")
        and (_qa_passed(_first(d, "vertical_pacing_qa", "qualitative_spacing_qa", default={}), "passed")
             or float(_first(d, "qualitative_row_gap", "hspace", default=0)) >= .01),
        {"pacing": _first(d, "vertical_pacing_qa", "qualitative_spacing_qa"),
         "row_gap": _first(d, "qualitative_row_gap", "hspace")},
    ))

    # ------------------------------------------------------------------
    # V3-4 contract items 22--25: cross-panel and panel e.
    # ------------------------------------------------------------------
    def alignment_qa():
        qa = _first(manifest, "cross_panel_alignment_qa", "panel_cd_alignment_qa", default={})
        qa = qa if isinstance(qa, dict) else {}
        return (
            _qa_passed(qa, "passed")
            or _flag(manifest, "panel_cd_rows_aligned", "cross_panel_rows_aligned")
            or _flag(c, "panel_d_row_alignment_passed"),
            qa,
        )

    check(22, "panels_c_d_row_alignment_improved", alignment_qa)
    check(23, "panel_e_recipe_blocks_separated", lambda: (
        int(_first(e, "recipe_group_separator_count", "recipe_block_separator_count", default=0)) >= 4
        and _flag(e, "recipe_blocks_separated", "recipe_group_separation_visible")
        and (_qa_passed(_first(e, "recipe_group_separation_qa", "recipe_block_separation_qa", default={}), "passed")
             or _panel_text_count(svg["e"], "panel-e-recipe-separator") >= 4),
        {"separator_count": _first(e, "recipe_group_separator_count", "recipe_block_separator_count"),
         "separation_qa": _first(e, "recipe_group_separation_qa", "recipe_block_separation_qa")},
    ))
    check(24, "panel_e_enlarged_matrix_readability", lambda: (
        e.get("matrix_shape") == [4, 9]
        and e.get("matrix_count") == 2
        and e.get("matrix_arrangement") == "stacked_vertically"
        and _flag(e, "matrix_cell_area_increased_vs_v3_3", "matrix_enlarged_for_readability")
        and (_qa_passed(_first(e, "matrix_readability_qa", "annotation_readability_qa", default={}), "passed")
             or _flag(e, "matrix_annotations_readable")),
        {"shape": e.get("matrix_shape"), "count": e.get("matrix_count"),
         "readability_qa": _first(e, "matrix_readability_qa", "annotation_readability_qa")},
    ))
    check(25, "global_polish_without_text_shrinkage", lambda: (
        manifest.get("layout", {}).get("typography_qa", {}).get("role_sizes_pt") == ROLE_SIZES
        and manifest.get("layout", {}).get("typography_qa", {}).get("passed") is True
        and (_flag(manifest, "text_shrinkage_qa_passed", "layout_polish_without_text_shrinkage")
             or _qa_passed(manifest.get("layout", {}).get("typography_qa", {}), "passed"))
    ))

    # ------------------------------------------------------------------
    # Supplemental scientific/provenance checks carried forward from V3-3.
    # ------------------------------------------------------------------
    check(26, "additive_v3_4_schema_and_five_panel_narrative", lambda: (
        manifest.get("workflow_label") == "mixed_resolution_unified_v3_4"
        and str(manifest.get("schema_version")) == "3.4"
        and manifest.get("run_id") == args.run_id
        and manifest.get("figure_contract", {}).get("panel_sequence") == ["a", "b", "c", "d", "e"]
        and (_flag(manifest, "additive_release", "additive")
             or str(manifest.get("release_mode", "")).lower() in {"additive", "new_release"})
        and bool(manifest.get("v3_3_starting_renderer_anchor")
                 or manifest.get("baseline_v3_3_manifest")
                 or manifest.get("v3_3_baseline_manifest")),
        {"workflow": manifest.get("workflow_label"), "schema": manifest.get("schema_version"),
         "run_id": manifest.get("run_id"), "panel_sequence": manifest.get("figure_contract", {}).get("panel_sequence")},
    ))

    check(27, "validated_scientific_ordering_and_panel_schema", lambda: (
        a.get("recipe_order") == RECIPES
        and a.get("dimensions") == {"L": [32, 32], "M": [64, 64], "H": [128, 128]}
        and all(close(left, right) for left, right in zip(
            a.get("exposure_values", []), [1.0, 0.34, 0.4375, 0.15625, 0.1875]
        ))
        and a.get("roi_shared_across_resolutions") is True
        and set(a.get("shared_roi", {})) >= {"xmin", "xmax", "ymin", "ymax"}
        and b.get("recipe_order") == RECIPES
        and b.get("model_order") == MODELS
        and b.get("sensor_counts") == COUNTS
        and b.get("subaxis_roles") == [
            "recipe_transfer_grouped_bars", "zero_h_balanced_sweep", "zero_h_mrich_sweep"
        ]
        and c.get("models") == MODELS
        and c.get("column_order") == ["reference", *MODELS]
        and c.get("column_count") == 5
        and c.get("row_order") == ["full_field", "zoomed_field", "local_absolute_error"]
        and c.get("row_cell_counts") == {"full_field": 5, "zoomed_field": 5, "local_absolute_error": 4}
        and d.get("qualitative_scales") == SCALES
        and d.get("quantitative_models") == MODELS
        and d.get("quantitative_recipes") == FINE_RECIPES
        and e.get("matrix_shape") == [4, 9]
        and e.get("matrix_count") == 2,
        {"panel_a_recipes": a.get("recipe_order"), "panel_b_roles": b.get("subaxis_roles"),
         "panel_c_columns": c.get("column_order"), "panel_d_scales": d.get("qualitative_scales"),
         "panel_e_shape": e.get("matrix_shape")},
    ))

    def panel_b_numeric():
        source = _source_path(manifest, release, "SensorSweepAllRecipes_summary")
        rows = read_csv(source)
        index = {
            (row["model"], row["recipe"], int(row["sensor_count"])): row
            for row in rows if row.get("metric") == "physical_rel_l2"
        }
        plotted = b.get("plotted_rows", [])
        expected = {
            (model, recipe, count)
            for model in MODELS for recipe in RECIPES for count in [512]
        } | {
            (model, recipe, count)
            for model in MODELS for recipe in ZERO_H for count in COUNTS
        }
        observed = {(row["model"], row["recipe"], int(row["sensor_count"])) for row in plotted}
        exact = observed == expected
        for row in plotted:
            source_row = index[(row["model"], row["recipe"], int(row["sensor_count"]))]
            exact &= all(close(row[key], source_row[key]) for key in ("mean", "ci95_low", "ci95_high"))
            exact &= int(row["valid_n"]) == int(source_row["valid_n"]) == 300
            exact &= float(row["ci95_low"]) <= float(row["mean"]) <= float(row["ci95_high"])
        return exact and len(plotted) == 60, {"source": str(source), "plotted_rows": len(plotted),
                                             "expected_rows": len(expected), "keys_exact": observed == expected}

    check(28, "panel_b_values_match_validated_sensor_summary", panel_b_numeric)

    def panel_c_cache_recompute():
        if c.get("models") != MODELS or len(c.get("cache_sources", [])) != 4:
            return False, {"models": c.get("models"), "cache_count": len(c.get("cache_sources", []))}
        roi = c["roi"]
        recomputed = {}
        identities = []
        cache_ok = True
        for cache_value in c["cache_sources"]:
            if isinstance(cache_value, dict):
                cache_value = cache_value.get("path")
            cache_path = _resolve_path(cache_value, release)
            with np.load(cache_path, allow_pickle=False) as cache:
                metadata = _read_metadata_json(cache["metadata_json"])
                prediction = np.asarray(cache["recon_phys"]).reshape(-1)
                observations = np.asarray(cache["obs_indices"])
            truth_ref = _resolve_path(metadata["truth_ref"], release)
            grid_ref = _resolve_path(metadata["grid_ref"], release)
            with np.load(truth_ref, allow_pickle=False) as truth_file:
                truth = np.asarray(truth_file["truth_phys"]).reshape(-1)
            with np.load(grid_ref, allow_pickle=False) as grid_file:
                coords = np.asarray(grid_file["coords_phys"])
            mask = (
                (coords[:, 0] >= float(roi["xmin"])) & (coords[:, 0] <= float(roi["xmax"]))
                & (coords[:, 1] >= float(roi["ymin"])) & (coords[:, 1] <= float(roi["ymax"]))
            )
            model = metadata["model"]
            identities.append({
                "model": model,
                "recipe": metadata.get("recipe"),
                "snapshot": metadata.get("snapshot_index"),
                "case_id": metadata.get("case_id"),
                "time_index": metadata.get("time_index"),
                "sensor_count": metadata.get("sensor_count"),
                "cache_sha256": sha256(cache_path),
                "sensor_plan_hash": metadata.get("sensor_plan_hash"),
            })
            recomputed[model] = {
                "full": relative_l2(truth, prediction),
                "local": relative_l2(truth[mask], prediction[mask]),
                "roi_points": int(mask.sum()),
            }
            cache_ok &= (
                metadata.get("status") == "ok"
                and metadata.get("recipe") == c.get("recipe") == "5_ZeroH_MRich"
                and int(metadata.get("snapshot_index", -1)) == int(c.get("snapshot", -2)) == 50
                and int(metadata.get("case_id", -1)) == int(c.get("case_id", -2)) == 9160
                and int(metadata.get("time_index", -1)) == int(c.get("time_index", -2)) == 18
                and int(metadata.get("sensor_count", -1)) == int(c.get("sensor_count", -2)) == 512
                and observations.size == 512
                and np.all(np.isfinite(prediction))
                and int(mask.sum()) == int(roi["grid_point_count"])
                and close(recomputed[model]["full"], c["full_field_relative_l2"][model])
                and close(recomputed[model]["local"], c["local_relative_l2"][model])
            )
        identity_keys = {(item["recipe"], item["snapshot"], item["case_id"], item["time_index"],
                          item["sensor_count"], item["sensor_plan_hash"]) for item in identities}
        return cache_ok and set(recomputed) == set(MODELS) and len(identity_keys) == 1, {
            "recomputed": recomputed, "identities": identities,
        }

    check(29, "panel_c_cache_payloads_and_l2_recompute", panel_c_cache_recompute)

    def panel_d_numeric():
        source = _source_path(manifest, release, "MultiscaleWavelet_summary")
        rows = read_csv(source)
        index = {(row["model_key"], row["recipe"], row["scale_group"], row["metric"]): row for row in rows}
        plotted = d.get("plotted_rows", [])
        exact = True
        for row in plotted:
            source_row = index[(row["model"], row["recipe"], row["scale_group"], row["metric"])]
            exact &= all(close(row[key], source_row[key]) for key in ("median", "q25", "q75"))
            exact &= int(row["valid_n"]) == int(source_row["valid_n"]) == 300
        identity = (
            d.get("qualitative_recipe") == "5_ZeroH_MRich"
            and d.get("displayed_snapshot") == 50
            and d.get("metadata_selected_snapshot") == 102
            and d.get("case_id") == 9160 and d.get("time_index") == 18
            and d.get("sensor_count") == 256
            and d.get("quantitative_arrangement") == "correlation_above_bias_stacked_vertically"
            and d.get("main_quantitative_value_count") == 24
        )
        return exact and identity, {"plotted_rows": len(plotted), "identity": identity}

    check(30, "panel_d_values_and_representative_identity_validated", panel_d_numeric)

    def panel_e_numeric():
        source = _source_path(manifest, release, "MultiscaleWavelet_summary")
        rows = read_csv(source)
        index = {(row["model_key"], row["recipe"], row["scale_group"], row["metric"]): row for row in rows}
        cells = e.get("heatmap_values", [])
        observed_keys = set()
        exact = len(cells) == 72
        for cell in cells:
            key = (cell["model_key"], cell["recipe"], cell["scale_group"], cell["metric"])
            observed_keys.add(key)
            source_row = index[key]
            statistic = cell.get("statistic", "median")
            expected = source_row.get(statistic, source_row["median"])
            exact &= close(cell["value"], expected)
            exact &= int(cell["valid_n"]) == int(source_row["valid_n"]) == 300
        exact &= observed_keys == _all_expected_matrix_keys()
        fine_index = {(cell["model_key"], cell["recipe"], cell["metric"]): cell["value"]
                      for cell in cells if cell.get("scale_group") == "fine"}
        links = True
        for metric, model_values in d.get("quantitative_values", {}).items():
            for model, recipe_values in model_values.items():
                for recipe, value in recipe_values.items():
                    links &= close(value, fine_index[(model, recipe, metric)])
        return exact and links, {"cell_count": len(cells), "keys_exact": observed_keys == _all_expected_matrix_keys(),
                                "fine_links": links}

    check(31, "panel_e_matrix_values_and_fine_links_validated", panel_e_numeric)

    def si_and_tables():
        si = manifest.get("si_outputs", {})
        expected_si = {
            "Sx1_complete_sensor_sweeps",
            "Sx2_expanded_recipe_gallery",
            "Sx3_complete_three_scale_qualitative",
            "Sx4_complete_three_scale_quantitative",
        }
        si_ok = set(si) == expected_si and all(
            {Path(item["path"]).suffix for item in entries} == {".svg", ".pdf", ".png"}
            and all(_record_path(item, release) and _record_path(item, release).exists() for item in entries)
            for entries in si.values()
        )
        si_meta = manifest.get("si_metadata", {})
        sx1, sx2, sx3, sx4 = (si_meta.get(key, {}) for key in ("Sx1", "Sx2", "Sx3", "Sx4"))
        si_ok &= (
            sx1.get("recipes") == RECIPES and sx1.get("sensor_counts") == COUNTS
            and sx2.get("recipes") == FINE_RECIPES and set(sx2.get("models", [])) == set(MODELS)
            and sx2.get("cache_cell_count") == 12
            and sx3.get("scale_groups") == SCALES
            and sx4.get("scale_groups") == SCALES
            and sx4.get("metrics") == METRICS
            and sx4.get("models") == MODELS
            and len(sx4.get("heatmap_values", [])) == 72
            and all(int(cell["valid_n"]) == 300 for cell in sx4.get("heatmap_values", []))
        )
        tables = manifest.get("table_outputs", {})
        expected_counts = {
            "accuracy_512": 20,
            "sensor_sweeps_64_512": 100,
            "pattern_correlations_all_scales": 60,
            "variance_allocation_bias_all_scales": 60,
        }
        table_ok = set(tables) == set(expected_counts) and all(
            int(item.get("row_count", -1)) == expected_counts[key]
            and _record_path(item, release) and _record_path(item, release).exists()
            for key, item in tables.items()
        )
        extended = _first(
            manifest,
            "extended_multiscale_distribution_source",
            "per_snapshot_distribution_source",
            default={},
        )
        extended_path = _record_path(extended, release)
        extended_ok = bool(extended_path and extended_path.exists() and extended.get("sha256")
                           and sha256(extended_path) == extended["sha256"])
        return si_ok and table_ok and extended_ok, {
            "si_keys": sorted(si), "table_counts": {key: item.get("row_count") for key, item in tables.items()},
            "per_snapshot_distribution": str(extended_path) if extended_path else None,
        }

    check(32, "complete_si_tables_and_per_snapshot_distribution", si_and_tables)

    def source_immutability():
        immutable = manifest.get("source_immutability", {})
        recorded = _source_records(manifest)
        source_hashes_match, source_details = _hash_audit(recorded, release)
        result_count_before = _first(immutable, "result_file_count_before", "file_count_before", default=None)
        result_count_after = _first(immutable, "result_file_count_after", "file_count_after", default=None)
        tree_before = _first(immutable, "tree_state_sha256_before", "sha256_before", default=None)
        tree_after = _first(immutable, "tree_state_sha256_after", "sha256_after", default=None)
        tree_ok = (
            immutable.get("unchanged") is True
            and tree_before is not None and tree_before == tree_after
            and (result_count_before is None or result_count_before == result_count_after)
        )
        return (
            tree_ok and source_hashes_match
            and manifest.get("model_inference_performed") is False
            and manifest.get("validated_sources_modified") is False
            and all(meta.get("model_inference_performed") is False
                    for meta in panels.values() if isinstance(meta, dict))
        ), {
            "source_hash_count": len(recorded), "source_hashes_match": source_hashes_match,
            "tree_state": immutable, "source_failures": [item for item in source_details if not item["passed"]],
        }

    check(33, "validated_sources_and_result_tree_immutable", source_immutability)

    def output_and_collision_qa():
        layout = manifest.get("layout", {})
        typography = layout.get("typography_qa", {})
        panel_text = layout.get("panel_text_clearance_qa", {})
        geometry = layout.get("geometry_qa", {})
        frame = layout.get("frame_lineweight_qa", {})
        main_svg = _find_main_file(release, args.run_id, "svg")
        main_pdf = _find_main_file(release, args.run_id, "pdf")
        main_png = _find_main_file(release, args.run_id, "png")
        final_size = manifest.get("figure_contract", {}).get("final_size_mm", [])
        files_ok = all(path.exists() and path.stat().st_size > 0 for path in (main_svg, main_pdf, main_png))
        pdf_size = _pdf_size_mm(main_pdf) if files_ok else [0.0, 0.0]
        png_size = list(Image.open(main_png).size) if files_ok else [0, 0]
        expected = [float(value) for value in final_size]
        expected_px = [round(value / 25.4 * 600) for value in expected]
        fonts = _pdf_font_lines(main_pdf) if files_ok else []
        embedded_fonts = bool(fonts) and all(
            all(token in line for token in ("yes", "yes", "yes")) for line in fonts
        )
        standalone = {}
        standalone_ok = True
        rectangles = layout.get("panel_rectangles_mm", {})
        for label in "abcde":
            paths = list((release / "panels").glob(f"Panel_{label}_*_{args.run_id}.*"))
            suffixes = {path.suffix for path in paths}
            triplet = suffixes == {".svg", ".pdf", ".png"} and len(paths) >= 3
            standalone[label] = {"suffixes": sorted(suffixes), "triplet": triplet}
            standalone_ok &= triplet
            panel_pdf = next((path for path in paths if path.suffix == ".pdf"), None)
            if panel_pdf and label in rectangles:
                observed = _pdf_size_mm(panel_pdf)
                target = [float(rectangles[label][key]) for key in ("width_mm", "height_mm")]
                standalone[label]["observed_mm"] = observed
                standalone[label]["expected_mm"] = target
                standalone_ok &= all(abs(left - right) <= .02 for left, right in zip(observed, target))
        passed = (
            files_ok and expected and abs(expected[0] - 183.0) <= .02 and 225.0 <= expected[1] <= 240.0
            and all(abs(left - right) <= .02 for left, right in zip(pdf_size, expected))
            and all(abs(left - right) <= 2 for left, right in zip(png_size, expected_px))
            and geometry.get("passed") is True
            and typography.get("passed") is True
            and panel_text.get("passed") is True
            and not any(float(value) > 0 for value in layout.get("text_overflow_in", {}).values())
            and not panel_text.get("cross_panel_text_overlaps")
            and frame.get("passed") is True
            and "<text" in main_svg.read_text(encoding="utf-8")
            and embedded_fonts
            and standalone_ok
        )
        return passed, {
            "pdf_size_mm": pdf_size, "png_size_px": png_size, "expected_px": expected_px,
            "standalone": standalone, "fonts": fonts, "embedded_fonts": embedded_fonts,
            "geometry": geometry, "typography": typography, "panel_text": panel_text,
        }

    check(34, "publication_exports_geometry_typography_and_collisions", output_and_collision_qa)

    def artifact_and_additivity():
        artifacts = _artifact_records(manifest)
        artifact_hashes_match, artifact_details = _hash_audit(artifacts, release)
        baseline_records = _recorded_baseline_paths(manifest, release)
        baseline_hashes_match, baseline_details = _hash_audit(baseline_records, release)
        output_paths = [str(_record_path(item, release).resolve()) for item in artifacts
                        if _record_path(item, release)]
        unique_outputs = len(output_paths) == len(set(output_paths))
        release_path = str(release.resolve())
        baseline_output_collision = any(
            path == str(V3_2_BASELINE_PDF.resolve())
            or path == str(V2_BASELINE_PDF.resolve())
            or path == str(V3_2_BASELINE_MANIFEST.resolve())
            or path == str(V2_BASELINE_MANIFEST.resolve())
            for path in output_paths
        )
        outside_release = [
            path for path in output_paths
            if not Path(path).is_relative_to(release)
        ]
        baseline_immutability = manifest.get("baseline_immutability", {})
        baseline_qa = baseline_immutability.get("unchanged") is True
        additive = (not baseline_output_collision) and not outside_release and release_path not in {
            str(V3_2_BASELINE_PDF.parent.resolve()), str(V2_BASELINE_PDF.parent.resolve()),
        }
        return bool(artifacts and artifact_hashes_match and baseline_hashes_match and baseline_qa
                    and unique_outputs and additive), {
            "artifact_count": len(artifacts), "artifact_hashes_match": artifact_hashes_match,
            "artifact_failures": [item for item in artifact_details if not item["passed"]],
            "baseline_count": len(baseline_records), "baseline_hashes_match": baseline_hashes_match,
            "baseline_failures": [item for item in baseline_details if not item["passed"]],
            "baseline_immutability": baseline_immutability,
            "unique_output_paths": unique_outputs, "outside_release": outside_release,
            "baseline_output_collision": baseline_output_collision,
        }

    check(35, "additive_outputs_and_baselines_not_overwritten", artifact_and_additivity)

    def documentation_complete():
        required = (
            "quantitative_figure_report_v3_4.md",
            "figure_reference_update_v3_4.md",
            "completion_report_v3_4.md",
        )
        present = {
            name: (release / name).exists() and (release / name).stat().st_size > 0
            for name in required
        }
        report = (release / "quantitative_figure_report_v3_4.md").read_text(encoding="utf-8") \
            if present["quantitative_figure_report_v3_4.md"] else ""
        reference_map = (release / "figure_reference_update_v3_4.md").read_text(encoding="utf-8") \
            if present["figure_reference_update_v3_4.md"] else ""
        completion = (release / "completion_report_v3_4.md").read_text(encoding="utf-8") \
            if present["completion_report_v3_4.md"] else ""
        no_science_change = all(token in report.lower() for token in ("v3-4", "layout", "typograph")) \
            and any(token in report.lower() for token in ("no new evidence", "scientific content", "no metric"))
        panel_map = all(label in reference_map for label in "abcde")
        completion_mentions = all(token in completion.lower() for token in ("v3-4", "qa"))
        return all(present.values()) and no_science_change and panel_map and completion_mentions, {
            "required": present, "no_science_change_statement": no_science_change,
            "panel_map": panel_map, "completion_mentions_qa": completion_mentions,
        }

    check(36, "required_v3_4_documentation_complete", documentation_complete)

    passed = bool(checks) and all(item["passed"] for item in checks)
    payload = {
        "workflow_label": "mixed_resolution_unified_v3_4",
        "schema_version": "3.4",
        "run_id": args.run_id,
        "manifest_path": str(manifest_path),
        "passed": passed,
        "checks": checks,
    }
    release.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[OK] {qa_path}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

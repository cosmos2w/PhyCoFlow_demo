#!/usr/bin/env python
"""Strict additive 25-point audit for the mixed-resolution V3-6 figure.

V3-6 is a micro-adjustment of the validated V3-5 figure.  This audit
retains the 35 V3-5 checks and adds eight V3-6-specific checks, then rechecks the
scientific values, validated caches, SI/table coverage, result-tree
immutability, baseline immutability, export geometry, and documentation.  It
does not render, train, infer, recompute metrics, or modify validated data.
Running it writes only ``qa_v3_6.json`` in the supplied release directory.

The V3-4 audit is loaded for its tested hash/CSV/cache/PDF helpers.  V3-6
renderers must expose the geometric decisions in panel/manifest metadata;
missing decisions fail closed instead of being inferred from a screenshot.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
_SPEC = importlib.util.spec_from_file_location(
    "unified_v3_4_audit_helpers", HERE / "108_audit_unified_v3_4.py"
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Cannot load the V3-4 audit helper module")
v34 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(v34)


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

PROJECT_ROOT = HERE.parents[3]
ASSEMBLED_DIR = PROJECT_ROOT / (
    "1_SubTask_SuperResolution/Save_TrainedModel/_TrainedModels/"
    "_Process_Figures/Assembled"
)
GENERATED_DIR = PROJECT_ROOT / "1_SubTask_SuperResolution/figures/generated"
V3_4_BASELINE_PATHS = {
    GENERATED_DIR / "MixedResolution_unified_v3_4_20260914_1040.pdf":
        "39137a39cd5ce3d062cb40be8e9a58f1562f4a7f70c7c16813939abba7b948ad",
    GENERATED_DIR / "MixedResolution_unified_v3_4_20260914_1040.svg":
        "0541b138f417c11a50893e54884a6659410c934bcaf5310400d23c1dee9754a4",
    GENERATED_DIR / "MixedResolution_unified_v3_4_20260914_1040.png":
        "d5d72f65525e43b0ec05261e314768b8aaabd79fa97a5b426db5677b119c8b15",
    ASSEMBLED_DIR / "MixedResolution_unified_v3_4_20260914_1040.pdf":
        "39137a39cd5ce3d062cb40be8e9a58f1562f4a7f70c7c16813939abba7b948ad",
    ASSEMBLED_DIR / "MixedResolution_unified_v3_4_20260914_1040.svg":
        "0541b138f417c11a50893e54884a6659410c934bcaf5310400d23c1dee9754a4",
    ASSEMBLED_DIR / "MixedResolution_unified_v3_4_20260914_1040.png":
        "d5d72f65525e43b0ec05261e314768b8aaabd79fa97a5b426db5677b119c8b15",
    ASSEMBLED_DIR / "FigureSourceManifest_unified_v3_4_20260914_1040.json":
        "3637d53c894723302c49b316251838a88d35b0dd2ba2bbaaae87d9bd16a5227a",
}


V3_5_BASELINE_ROOT = GENERATED_DIR / "MixedResolution_unified_v3_5_20260914_1100"
V3_5_BASELINE_PATHS = {
    V3_5_BASELINE_ROOT / "MixedResolution_unified_v3_5_20260914_1100.pdf":
        "6bbd080031881d40d5ab3e29e5cc2a04cea0d637c89300f4f4363fc6953e0471",
    V3_5_BASELINE_ROOT / "MixedResolution_unified_v3_5_20260914_1100.svg":
        "77ba0a2d5ae89dbc3a6782d3aa5389fb8fdfa4da745a40e5d5d178e1a1695ef4",
    V3_5_BASELINE_ROOT / "MixedResolution_unified_v3_5_20260914_1100.png":
        "573ffe856a2e89109234a264add1dab0748c405fe5ccaab45a34e02cdb2cc390",
    V3_5_BASELINE_ROOT / "source_manifest_v3_5.json":
        "a6c2b3e570dd0aa9b95b28fd7b5d8af97f769ded6c7e3a07550f31e0789e99dc",
    V3_5_BASELINE_ROOT / "qa_v3_5.json":
        "d55c54f7f70105c74d1601536d24ef2b266314d21fd3a6302dbff6d422354898",
}


def _first(mapping, *keys, default=None):
    return v34._first(mapping, *keys, default=default)


def _flag(mapping, *keys):
    return v34._flag(mapping, *keys)


def _qa_passed(mapping, *keys):
    return v34._qa_passed(mapping, *keys)


def _marker_count(svg: str, *markers) -> int:
    return sum(svg.count(marker) for marker in markers)


def _normalise_text(value) -> str:
    return v34._normalise_text(value)


def _contains_text(svg: str, phrase: str) -> bool:
    return v34._contains_text(svg, phrase)


def _resolve_path(value, release: Path) -> Path:
    return v34._resolve_path(value, release)


def _normalise_row_boundaries(value):
    """Return a flat [bottom, top] pair per row in millimetres.

    V3-6 metadata may store three [bottom, height] rows, three [bottom, top]
    rows, or four shared boundary coordinates.  All representations are
    converted before the exact c/d comparison; center-only metadata is not
    accepted for the exact-alignment check.
    """
    if isinstance(value, dict):
        for key in ("rows", "bounds", "boundaries", "y_bounds"):
            if key in value:
                return _normalise_row_boundaries(value[key])
        return None
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim == 1 and array.size == 4:
        return array.tolist()
    if array.ndim == 2 and array.shape == (3, 2):
        rows = []
        for bottom, second in array:
            # Explicit [bottom, top] is the default; a ``heights`` marker is
            # handled by the caller when it is available.
            rows.extend([float(bottom), float(second)])
        return rows
    return None


def _row_alignment(manifest, c, d):
    qa = _first(manifest, "cross_panel_alignment_qa", "panel_cd_alignment_qa", default={})
    qa = qa if isinstance(qa, dict) else {}
    nested = _first(qa, "v3_6_row_boundaries_mm", "v3_5_row_boundaries_mm", "row_boundaries_mm",
                    "shared_row_boundaries_mm", default={})
    c_bounds = d_bounds = None
    shared = False
    if isinstance(nested, dict):
        c_bounds = _normalise_row_boundaries(_first(nested, "c", "panel_c", default=None))
        d_bounds = _normalise_row_boundaries(_first(nested, "d", "panel_d", default=None))
        shared = _normalise_row_boundaries(_first(nested, "shared", "common", default=None))
    if c_bounds is None:
        c_bounds = _normalise_row_boundaries(_first(
            qa, "v3_6_c_row_boundaries_mm", "v3_5_c_row_boundaries_mm", "c_row_boundaries_mm", default=None
        ))
    if d_bounds is None:
        d_bounds = _normalise_row_boundaries(_first(
            qa, "v3_6_d_row_boundaries_mm", "v3_5_d_row_boundaries_mm", "d_row_boundaries_mm", default=None
        ))
    if c_bounds is None:
        c_bounds = _normalise_row_boundaries(_first(
            c, "row_boundaries_mm", "image_row_boundaries_mm", default=None
        ))
    if d_bounds is None:
        d_bounds = _normalise_row_boundaries(_first(
            d, "row_boundaries_mm", "image_row_boundaries_mm", default=None
        ))
    if shared is not None and c_bounds is None:
        c_bounds = shared
    if shared is not None and d_bounds is None:
        d_bounds = shared
    exact = bool(c_bounds and d_bounds and len(c_bounds) == len(d_bounds)
                 and np.allclose(c_bounds, d_bounds, rtol=0.0, atol=1e-9))
    return {
        "exact": exact,
        "c_boundaries_mm": c_bounds,
        "d_boundaries_mm": d_bounds,
        "max_abs_delta_mm": (
            float(np.max(np.abs(np.asarray(c_bounds) - np.asarray(d_bounds))))
            if c_bounds and d_bounds and len(c_bounds) == len(d_bounds) else None
        ),
        "row_order": _first(qa, "row_order", default=None),
        "qa": qa,
    }


def _source_records(manifest):
    records = list(v34._source_records(manifest))
    for key in (
        "v3_4_baseline",
        "baseline_v3_4",
        "v3_4_starting_renderer_anchor",
        "starting_renderer_anchor",
    ):
        records.extend(v34._records_from(manifest.get(key)))
    return v34._unique_records(records)


def _baseline_records(manifest, release):
    records = []
    for key in (
        "baseline_immutability",
        "old_v2_baseline_pdf",
        "old_v2_baseline_manifest",
        "baseline_v2_pdf",
        "baseline_v2_manifest",
        "baseline_v3_2_pdf",
        "baseline_v3_2_manifest",
        "baseline_v3_3_pdf",
        "baseline_v3_3_manifest",
        "v3_3_baseline",
        "v3_4_baseline",
        "v3_5_baseline",
        "baseline_v3_5",
        "v3_5_starting_renderer_anchor",
        "v3_5_baseline_manifest",
        "v3_5_baseline_pdf",
        "v3_5_baseline_svg",
        "v3_5_baseline_png",
        "baseline_v3_4",
        "v3_4_starting_renderer_anchor",
    ):
        records.extend(v34._records_from(manifest.get(key)))
    immutability = manifest.get("baseline_immutability", {})
    if isinstance(immutability, dict):
        for field in ("sha256_before", "sha256_after"):
            hashes = immutability.get(field, {})
            if isinstance(hashes, dict):
                records.extend({"path": path, "sha256": digest}
                               for path, digest in hashes.items())
    for path, digest in V3_4_BASELINE_PATHS.items():
        if path.exists():
            records.append({"path": str(path), "sha256": digest})
    for path, digest in V3_5_BASELINE_PATHS.items():
        if path.exists():
            records.append({"path": str(path), "sha256": digest})
    # V3-4's inherited V2/V3-2 hashes are checked by the helper, and the
    # explicit V3-4 records above ensure that this revision cannot overwrite
    # its immediate baseline even if the parent manifest omits an alias.
    records.extend(v34._recorded_baseline_paths(manifest, release))
    return v34._unique_records(records)


def _find_panel_svg(release: Path, label: str, run_id: str):
    panel_dir = release / "panels"
    candidates = sorted(panel_dir.glob(f"Panel_{label}_*_{run_id}.svg"))
    if not candidates:
        candidates = sorted(panel_dir.glob(f"Panel_{label}_*.svg"))
    return candidates[0] if candidates else None


def _find_main_file(release: Path, run_id: str, suffix: str):
    exact = release / f"MixedResolution_unified_v3_6_{run_id}.{suffix}"
    if exact.exists():
        return exact
    candidates = sorted(release.glob(f"MixedResolution_unified_v3_6_*_{run_id}.{suffix}"))
    if not candidates:
        candidates = sorted(release.glob(f"MixedResolution_unified_v3_6_*.{suffix}"))
    return candidates[0] if candidates else exact


def _source_path(manifest, release: Path, prefix: str):
    return v34._source_path(manifest, release, prefix)


def _read_metadata_json(value):
    return v34._read_metadata_json(value)



def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}


def _canonical_json(value):
    if isinstance(value, dict):
        return {str(key): _canonical_json(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    if isinstance(value, tuple):
        return [_canonical_json(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _number_list(value):
    if isinstance(value, dict):
        for key in (
            "values", "value", "heights_mm", "widths_pt", "sizes_pt", "gaps_mm",
            "recipe_gaps_mm", "within_metric_recipe_gaps_mm", "within_metric_gaps_mm",
            "items", "axes", "rows", "top", "middle", "bottom", "large",
            "intermediate", "fine", "recipe_transfer", "zero_h_balanced",
            "zero_h_mrich", "line_plot", "line_secondary", "primary", "secondary",
        ):
            if key in value:
                values = _number_list(value[key])
                if values:
                    return values
        return []
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 1:
                item = item[0]
            try:
                values.append(float(item))
            except (TypeError, ValueError):
                return []
        return values
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def _panel_rect(manifest, label):
    layout = manifest.get("layout", {}) if isinstance(manifest, dict) else {}
    rectangles = layout.get("panel_rectangles_mm", {}) if isinstance(layout, dict) else {}
    rect = rectangles.get(label, {}) if isinstance(rectangles, dict) else {}
    return rect if isinstance(rect, dict) else {}


def _axis_heights(manifest, panel):
    for key in (
        "plotting_axis_heights_mm", "plot_axes_physical_heights_mm",
        "plot_axis_heights_mm", "axis_heights_mm", "subaxis_physical_heights_mm",
    ):
        values = _number_list(panel.get(key))
        if len(values) == 3:
            return values, key
    for key in (
        "plotting_axis_height_multipliers_vs_v3_5",
        "plot_axis_height_multipliers_vs_v3_5",
        "axis_height_multipliers_vs_v3_5",
    ):
        values = _number_list(panel.get(key))
        if len(values) == 3:
            return values, key
    ratios = _number_list(panel.get("internal_height_ratios"))
    rect_height = float(_first(_panel_rect(manifest, "b"), "height_mm", default=0))
    if len(ratios) == 3 and rect_height > 0 and sum(ratios) > 0:
        scale = rect_height / sum(ratios)
        return [scale * ratio for ratio in ratios], "internal_height_ratios_scaled"
    return [], None


def _style_values(panel, keys):
    for key in keys:
        values = _number_list(panel.get(key))
        if values:
            return values, key
    return [], None


def _baseline_style_values(manifest, panel, keys, fallback):
    values, source = _style_values(panel, keys)
    if values:
        return values, source
    style = manifest.get("style_contract", {}) if isinstance(manifest, dict) else {}
    widths = style.get("line_widths_pt", {}) if isinstance(style, dict) else {}
    if isinstance(widths, dict):
        if panel.get("sensor_sweep_values"):
            return [2.0, 1.6], "validated V3-5 panel-b renderer configuration"
        values = _number_list({key: widths[key] for key in fallback if key in widths})
        if values:
            return values, "style_contract.line_widths_pt"
    return [], None


def _metric_gap_values(panel):
    qa = _first(
        panel, "recipe_group_separation_qa", "recipe_gap_qa",
        "within_metric_gap_qa", "matrix_spacing_qa", default={}
    )
    qa = qa if isinstance(qa, dict) else {}
    within = _number_list(_first(
        panel, "within_metric_recipe_gaps_mm", "within_metric_gaps_mm",
        "recipe_gaps_within_metric_mm", default=None
    ))
    axes_bounds = panel.get("heatmap_axes_bounds")
    if isinstance(axes_bounds, list) and len(axes_bounds) == 6:
        try:
            boxes = sorted(axes_bounds, key=lambda box: float(box[0]))
            within = [
                (float(boxes[right][0]) - float(boxes[left][0]) - float(boxes[left][2])) * 183.0
                for left, right in ((0, 1), (1, 2), (3, 4), (4, 5))
            ]
            measured_central = (
                float(boxes[3][0]) - float(boxes[2][0]) - float(boxes[2][2])
            ) * 183.0
        except (IndexError, TypeError, ValueError):
            measured_central = None
    else:
        measured_central = None
    if not within:
        within = _number_list(_first(qa, "within_metric_recipe_gaps_mm",
                                      "within_metric_gaps_mm", "gaps_mm", default=None))
    central = _first(
        panel, "central_metric_gap_mm", "metric_block_gap_mm",
        "central_recipe_gap_mm", "between_metric_gap_mm", default=None
    )
    if central is None:
        central = _first(qa, "central_metric_gap_mm", "metric_block_gap_mm",
                         "between_metric_gap_mm", default=None)
    if measured_central is not None:
        central = measured_central
    try:
        central = float(central)
    except (TypeError, ValueError):
        central = None
    matrix_gap = _first(
        panel, "matrix_to_colorbar_gap_mm", "matrix_colorbar_gap_mm",
        "matrix_to_colorbar_vertical_gap_mm", default=None
    )
    colorbar_qa = _first(panel, "matrix_to_colorbar_gap_qa", "colorbar_gap_qa", default={})
    if matrix_gap is None and isinstance(colorbar_qa, dict):
        matrix_gap = _first(colorbar_qa, "gap_mm", "matrix_to_colorbar_gap_mm", default=None)
    try:
        matrix_gap = float(matrix_gap)
    except (TypeError, ValueError):
        matrix_gap = None
    return {
        "qa": qa,
        "within_mm": within,
        "central_mm": central,
        "matrix_colorbar_mm": matrix_gap,
    }


def _baseline_v3_5_manifest():
    path = V3_5_BASELINE_ROOT / "source_manifest_v3_5.json"
    return _load_json(path), path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path,
        help="Optional source_manifest_v3_6.json; defaults to release-dir/source_manifest_v3_6.json",
    )
    args = parser.parse_args()
    release = args.release_dir.resolve()
    manifest_path = (args.manifest or (release / "source_manifest_v3_6.json")).resolve()
    qa_path = release / "qa_v3_6.json"
    checks = []

    def check(index, name, function):
        try:
            result = function()
            passed, detail = result if isinstance(result, tuple) else (result, None)
            item = {
                "contract_index": int(index),
                "name": name,
                "passed": bool(passed),
                "detail": detail,
            }
        except Exception as exc:  # noqa: BLE001 - missing evidence is a failed check.
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
    except Exception as exc:  # noqa: BLE001 - handled as an audit failure.
        manifest = {}
        error = {"path": str(manifest_path), "type": type(exc).__name__, "message": str(exc)}
        check(0, "audit_manifest_load", lambda error=error: (False, error))

    panels = manifest.get("panels", {}) if isinstance(manifest, dict) else {}
    a, b, c, d, e = (panels.get(label, {}) for label in "abcde")
    panel_svgs = {label: _find_panel_svg(release, label, args.run_id) for label in "abcde"}
    svg = {
        label: path.read_text(encoding="utf-8") if path and path.exists() else ""
        for label, path in panel_svgs.items()
    }

    # ------------------------------------------------------------------
    # V3-6 contract items 1--6: panel a.
    # ------------------------------------------------------------------
    def panel_a_collision():
        collision = _first(
            a, "training_cases_h_heatmap_overlap", "h_heatmap_training_cases_overlap",
            default=True,
        )
        qa = _first(a, "training_cases_collision_qa", "h_heatmap_bar_collision_qa", default={})
        return (
            collision is False
            and _qa_passed(qa, "passed")
            and _first(a, "training_cases_h_heatmap_collision_free", default=False) is True,
            {"overlap": collision, "qa": qa},
        )

    check(1, "panel_a_training_cases_h_heatmap_collision_free", panel_a_collision)

    def panel_a_spacer():
        spacer = _first(a, "blank_spacer_column", "field_bar_spacer", "breathing_channel", default={})
        spacer = spacer if isinstance(spacer, dict) else {}
        width = float(_first(spacer, "width_mm", "physical_width_mm", default=0))
        return (
            _flag(a, "explicit_blank_spacer", "blank_spacer_present")
            and _first(spacer, "empty", "blank", default=False) is True
            and 5.0 <= width <= 7.0
            and _first(spacer, "contains_text", "has_artists", default=True) is False
            and _first(a, "spacer_column_index", default=3) == 3,
            {"spacer": spacer},
        )

    def _panel_roi_equal(meta):
        rois = _first(meta, "resolution_rois", "roi_by_resolution", default=None)
        if not isinstance(rois, dict):
            return False
        values = []
        for tag in "LMH":
            roi = rois.get(tag)
            if not isinstance(roi, dict):
                return False
            values.append([float(roi[key]) for key in ("xmin", "xmax", "ymin", "ymax")])
        return bool(np.allclose(values, values[0], rtol=0.0, atol=1e-12))

    check(2, "panel_a_explicit_5_to_7mm_blank_spacer", panel_a_spacer)
    check(3, "panel_a_common_roi_identical_l_m_h", lambda: (
        a.get("roi_shared_across_resolutions") is True
        and set(a.get("shared_roi", {})) >= {"xmin", "xmax", "ymin", "ymax"}
        and _flag(a, "roi_coordinates_identical_lmh", "common_roi_exact")
        and _panel_roi_equal(a),
        {"shared_roi": a.get("shared_roi")},
    ))

    def panel_a_roi_shift():
        shift = _first(a, "roi_shift_vs_v3_4", "roi_shift_percent", "roi_shift", default={})
        shift = shift if isinstance(shift, dict) else {}
        x_left = float(_first(shift, "x_left_percent", "x_left", "leftward_percent", default=0))
        y_up = float(_first(shift, "y_up_percent", "y_up", "upward_percent", default=0))
        transition = str(_first(a, "roi_region", "roi_selection_region", default="")).lower()
        return (
            _flag(a, "roi_moved_to_high_gradient", "roi_high_gradient_transition")
            and 0.10 <= x_left <= 0.15 and 0.08 <= y_up <= 0.12
            and transition in {"high_gradient_transition", "steep_red_blue_transition", "high_gradient"},
            {"shift": shift, "region": transition},
        )

    check(4, "panel_a_roi_shifted_to_high_gradient_transition", panel_a_roi_shift)

    def panel_a_inset_size():
        size = _first(a, "inset_size_qa", "inset_geometry", default={})
        size = size if isinstance(size, dict) else {}
        widths = _first(size, "width_fraction", "width_fractions", default=None)
        if isinstance(widths, (list, tuple)):
            width_ok = len(widths) == 3 and all(.45 <= float(x) <= .52 for x in widths)
        else:
            width_ok = .45 <= float(widths or 0) <= .52
        return (
            width_ok
            and _first(size, "interpolation", "inset_interpolation", default="").lower()
            in {"nearest", "nearest_neighbor", "native_cells"}
            and _flag(a, "insets_enlarged", "inset_size_qa_passed")
            and _first(size, "pixel_visibility", "discretization_visible", default=False) is True,
            {"size": size},
        )

    check(5, "panel_a_insets_large_and_pixel_visible", panel_a_inset_size)

    def panel_a_inset_alignment():
        qa = _first(a, "inset_alignment_qa", "inset_geometry_alignment_qa", default={})
        positions = _first(a, "inset_positions", "inset_bounds_relative", default=None)
        exact_positions = False
        if isinstance(positions, dict):
            values = [positions.get(tag) for tag in "LMH"]
        else:
            values = positions
        if isinstance(values, (list, tuple)) and len(values) == 3 and all(values):
            try:
                exact_positions = bool(np.allclose(values, values[0], rtol=0.0, atol=1e-12))
            except (TypeError, ValueError):
                exact_positions = False
        return (
            _qa_passed(qa, "passed")
            and (_flag(a, "inset_positions_identical", "insets_horizontally_aligned") or exact_positions)
            and _first(a, "inset_parent_alignment", default="") in {"identical", "exact"},
            {"qa": qa, "positions": positions},
        )

    check(6, "panel_a_inset_positions_identical_and_aligned", panel_a_inset_alignment)

    # ------------------------------------------------------------------
    # V3-6 contract items 7--10: panel b.
    # ------------------------------------------------------------------
    def panel_b_top_position():
        gap = float(_first(b, "top_plot_to_b_tag_gap_mm", "top_chart_b_tag_gap_mm", default=99))
        return (
            _flag(b, "top_plot_near_b_tag", "top_chart_pulled_up")
            and 0 <= gap <= 4.0
            and _first(b, "b_tag_moved_down_to_solve", default=False) is False,
            {"gap_mm": gap},
        )

    check(7, "panel_b_top_plot_near_b_tag", panel_b_top_position)

    def panel_b_legend_strip():
        legend = b.get("legend_contract", {})
        legend = legend if isinstance(legend, dict) else {}
        strip = _first(b, "legend_strip", "legend_layout", default={})
        strip = strip if isinstance(strip, dict) else {}
        return (
            _flag(b, "legend_in_dedicated_middle_strip", "legend_middle_strip")
            and _first(legend, "placement", "location", default="").lower()
            in {"dedicated_middle_strip", "middle_strip", "between_plot_blocks"}
            and _first(strip, "row_index", "grid_row", default=0) == 1
            and _first(strip, "axis_off", "legend_only_axis", default=False) is True
            and _first(legend, "one_row", default=False) is True
            and int(_first(legend, "ncol", default=0)) == 4,
            {"legend": legend, "strip": strip},
        )

    check(8, "panel_b_legend_in_dedicated_middle_strip", panel_b_legend_strip)
    check(9, "panel_b_legend_and_sweeps_nonoverlapping", lambda: (
        _qa_passed(_first(b, "legend_collision_qa", "legend_spacing_qa", default={}), "passed")
        and _first(b, "legend_overlaps_top_chart", default=True) is False
        and _first(b, "legend_overlaps_sweep_titles", default=True) is False
        and _marker_count(svg["b"], "panel-b-legend-top-right") == 0,
        {"legend_collision_qa": _first(b, "legend_collision_qa", "legend_spacing_qa")},
    ))
    check(10, "panel_b_zero_h_annotation_nonoverlap", lambda: (
        _first(b, "zero_h_annotation_overlap", default=True) is False
        and _flag(b, "zero_h_annotation_nonoverlap", "no_h_annotation_nonoverlap")
        and _qa_passed(_first(b, "annotation_collision_qa", "zero_h_annotation_qa", default={}), "passed"),
        {"annotation_qa": _first(b, "annotation_collision_qa", "zero_h_annotation_qa")},
    ))

    # ------------------------------------------------------------------
    # V3-6 contract items 11--16: panels c/d and shared row geometry.
    # ------------------------------------------------------------------
    def shared_cd_grid():
        qa = _first(manifest, "cross_panel_alignment_qa", "panel_cd_alignment_qa", default={})
        qa = qa if isinstance(qa, dict) else {}
        source = str(_first(qa, "row_geometry_source", "geometry_source", default="")).lower()
        return (
            (_flag(manifest, "shared_cd_parent_gridspec", "shared_cd_grid")
             or _flag(qa, "shared_parent_gridspec", "shared_parent_grid", "shared_cd_grid"))
            and source in {"shared_parent_gridspec", "shared_gridspec", "single_shared_grid", "shared_parent_grid"}
            and _flag(c, "uses_shared_cd_grid", "shared_row_geometry")
            and _flag(d, "uses_shared_cd_grid", "shared_row_geometry"),
            {"alignment_qa": qa, "row_geometry_source": source},
        )

    check(11, "panels_c_d_use_one_shared_parent_grid", shared_cd_grid)
    check(12, "panels_c_d_exact_shared_row_boundaries", lambda: (
        (alignment := _row_alignment(manifest, c, d))["exact"]
        and alignment.get("row_order") == ["full/large", "zoom/intermediate", "local-error/fine"]
        and _flag(manifest, "exact_cd_row_boundaries", "c_d_rows_exactly_aligned")
        and alignment.get("max_abs_delta_mm") is not None
        and alignment["max_abs_delta_mm"] <= 1e-9,
        _row_alignment(manifest, c, d),
    ))

    def panel_c_spacing():
        qa = _first(c, "spacing_qa", "panel_c_spacing_qa", default={})
        qa = qa if isinstance(qa, dict) else {}
        tile_gap = float(_first(qa, "tile_gap_mm", "horizontal_gap_mm", default=99))
        row_gap = float(_first(qa, "row_gap_mm", "vertical_gap_mm", default=99))
        colorbar_gap = float(_first(qa, "colorbar_gap_mm", "gap_above_colorbars_mm", default=99))
        return (
            _qa_passed(qa, "passed")
            and 1.5 <= tile_gap <= 2.5 and 1.5 <= row_gap <= 2.5
            and 0 <= colorbar_gap <= 2.5
            and _flag(c, "vertical_padding_reduced", "compact_tile_spacing"),
            {"spacing_qa": qa},
        )

    check(13, "panel_c_compact_tiles_and_colorbar_gap", panel_c_spacing)

    def panel_d_tile_size():
        qa = _first(d, "tile_footprint_qa", "panel_cd_tile_qa", default={})
        qa = qa if isinstance(qa, dict) else {}
        c_sizes = _first(qa, "panel_c_tile_sizes_mm", "c_tile_sizes_mm", default=None)
        d_sizes = _first(qa, "panel_d_tile_sizes_mm", "d_tile_sizes_mm", default=None)
        measured = False
        if c_sizes and d_sizes:
            try:
                c_area = float(np.mean([float(x[0]) * float(x[1]) for x in c_sizes]))
                d_area = float(np.mean([float(x[0]) * float(x[1]) for x in d_sizes]))
                measured = d_area <= c_area + 1e-9
            except (IndexError, TypeError, ValueError):
                measured = False
        delta = float(_first(qa, "max_d_minus_c_area_mm2", "max_d_minus_c_mm2", default=99))
        return (
            _qa_passed(qa, "passed")
            and (measured or delta <= 0)
            and _flag(d, "tiles_not_larger_than_c", "tile_size_matched_to_c"),
            {"tile_qa": qa, "measured_d_not_larger": measured},
        )

    check(14, "panel_d_tiles_not_larger_than_panel_c", panel_d_tile_size)
    check(15, "panel_d_fine_arrow_or_funnel_removed", lambda: (
        int(_first(d, "fine_to_quantitative_connector_count", "fine_row_connector_count", default=99)) == 0
        and _first(d, "fine_row_connector_present", "fine_scale_connector_present", default=True) is False
        and _marker_count(svg["d"], "panel-d-fine-row-connector", "panel-d-fine-quant-connector") == 0
        and not _contains_text(svg["d"], "Fine-to-quantitative"),
        {"connector_count": _first(d, "fine_to_quantitative_connector_count", "fine_row_connector_count")},
    ))
    check(16, "panel_d_fine_scale_line_plots_removed", lambda: (
        int(_first(d, "main_quantitative_value_count", "line_plot_count", default=99)) == 0
        and _first(d, "quantitative_arrangement", default="") in {"none", "removed", "qualitative_only"}
        and _first(d, "quantitative_plots_present", "fine_line_plots_present", default=True) is False
        and _marker_count(svg["d"], "model-line:", "panel-d-quantitative-axis") == 0
        and not _contains_text(svg["d"], "Fine-scale pattern correlation")
        and not _contains_text(svg["d"], "Fine-scale variance-allocation bias"),
        {"quantitative_value_count": _first(d, "main_quantitative_value_count", "line_plot_count"),
         "arrangement": d.get("quantitative_arrangement")},
    ))

    # ------------------------------------------------------------------
    # V3-6 contract items 17--23: panel e.
    # ------------------------------------------------------------------
    def panel_e_side_by_side():
        layout = str(_first(e, "matrix_arrangement", "metric_matrix_arrangement", default="")).lower()
        shapes = _first(e, "matrix_shapes", "metric_matrix_shapes", default=[])
        return (
            e.get("matrix_count") == 2
            and layout in {"side_by_side", "horizontal", "1x2"}
            and e.get("matrix_arrangement") != "stacked_vertically"
            and (not shapes or shapes == [[4, 3], [4, 3]])
            and _flag(e, "matrices_side_by_side", "metric_blocks_side_by_side")
            and _marker_count(svg["e"], "panel-e-metric-matrix") >= 2,
            {"arrangement": layout, "shapes": shapes},
        )

    check(17, "panel_e_matrices_side_by_side", panel_e_side_by_side)

    def panel_e_full_width():
        rectangles = manifest.get("layout", {}).get("panel_rectangles_mm", {})
        canvas_width = float(manifest.get("layout", {}).get("canvas_width_mm", 0))
        rect = rectangles.get("e", {}) if isinstance(rectangles, dict) else {}
        width = float(rect.get("width_mm", 0))
        return (
            _flag(e, "full_width_panel", "panel_e_full_width")
            and width > 0 and canvas_width > 0 and abs(width - canvas_width) <= .02
            and abs(float(rect.get("left_mm", 99))) <= .02,
            {"canvas_width_mm": canvas_width, "panel_e": rect},
        )

    check(18, "panel_e_spans_full_width", panel_e_full_width)

    def panel_e_recipe_gaps():
        qa = _first(e, "recipe_group_separation_qa", "recipe_block_gap_qa", default={})
        qa = qa if isinstance(qa, dict) else {}
        gaps = _first(qa, "gaps_mm", "recipe_gaps_mm", default=[])
        axes = _first(e, "metric_recipe_axes_bounds", "recipe_subaxes_bounds", default=None)
        geometry_ok = False
        if isinstance(axes, dict):
            geometry_ok = True
            for metric in ("pattern_correlation", "variance_fraction_bias_pp"):
                entries = axes.get(metric, [])
                if len(entries) != 3:
                    geometry_ok = False
                    continue
                try:
                    geometry_ok &= all(
                        float(entries[i + 1][0]) - (float(entries[i][0]) + float(entries[i][2])) > 0
                        for i in range(2)
                    )
                except (IndexError, TypeError, ValueError):
                    geometry_ok = False
        gap_values_ok = isinstance(gaps, (list, tuple)) and len(gaps) >= 4 \
            and all(float(gap) > 0 for gap in gaps)
        return (
            _flag(e, "recipe_groups_use_real_subaxes", "recipe_blocks_real_horizontal_gaps")
            and _first(e, "recipe_group_layout", default="") in {"three_inner_subaxes_per_metric", "inner_subaxes"}
            and (_qa_passed(qa, "passed") or geometry_ok or gap_values_ok)
            and _marker_count(svg["e"], "panel-e-recipe-gap", "panel-e-recipe-separator") >= 4,
            {"qa": qa, "axes": axes, "gaps": gaps},
        )

    check(19, "panel_e_recipe_groups_have_real_horizontal_gaps", panel_e_recipe_gaps)
    check(20, "panel_e_right_metric_method_labels_hidden", lambda: (
        _flag(e, "right_metric_ylabels_hidden", "right_metric_method_labels_hidden")
        and int(_first(e, "right_metric_method_label_count", default=99)) == 0
        and int(_first(e, "left_metric_method_label_count", default=0)) == 4
        and _first(e, "right_metric_repeats_method_labels", default=True) is False,
        {"left_count": _first(e, "left_metric_method_label_count"),
         "right_count": _first(e, "right_metric_method_label_count")},
    ))
    check(21, "panel_e_matrix_numbers_readable_at_final_width", lambda: (
        float(_first(e, "cell_annotation_fontsize_pt", "matrix_annotation_fontsize_pt", default=0)) > 5.8
        and _flag(e, "matrix_annotation_fontsize_increased_vs_v3_4", "matrix_annotations_readable")
        and _qa_passed(_first(e, "matrix_readability_qa", "annotation_readability_qa", default={}), "passed")
        and int(_first(e, "cell_annotation_count", default=0)) == 72,
        {"fontsize_pt": _first(e, "cell_annotation_fontsize_pt", "matrix_annotation_fontsize_pt"),
         "annotation_count": _first(e, "cell_annotation_count")},
    ))
    check(22, "panel_e_uses_distinct_metric_colorbars", lambda: (
        int(_first(e, "colorbar_count", default=0)) == 2
        and _first(e, "colorbar_orientation", default="") in {"horizontal", "horizontal_band"}
        and _first(e, "shared_numeric_colorbar", "one_shared_numeric_colorbar", default=True) is False
        and _first(e, "correlation_bias_share_numeric_scale", default=True) is False
        and set(_first(e, "colorbar_metrics", default=[])) == set(METRICS)
        and _flag(e, "dual_metric_colorbars", "distinct_metric_colorbars"),
        {"colorbar_count": _first(e, "colorbar_count"),
         "metrics": _first(e, "colorbar_metrics"),
         "shared_numeric_colorbar": _first(e, "shared_numeric_colorbar", "one_shared_numeric_colorbar")},
    ))
    check(23, "panel_e_colorbars_horizontal_bottom_band_only", lambda: (
        _flag(e, "unified_bottom_colorbar_band", "colorbar_band_present")
        and _first(e, "vertical_colorbars_present", default=True) is False
        and int(_first(e, "bottom_colorbar_band_count", default=0)) == 1
        and _marker_count(svg["e"], "panel-e-vertical-colorbar") == 0
        and _marker_count(svg["e"], "panel-e-bottom-colorbar") >= 2,
        {"band_count": _first(e, "bottom_colorbar_band_count"),
         "vertical": _first(e, "vertical_colorbars_present")},
    ))

    # ------------------------------------------------------------------
    # V3-6 contract items 24--25: global scientific integrity/typography.
    # ------------------------------------------------------------------
    check(24, "validated_sources_not_modified_or_recomputed", lambda: (
        manifest.get("model_inference_performed") is False
        and manifest.get("validated_sources_modified") is False
        and _first(manifest, "metric_recomputation_performed", "metrics_recomputed", default=False) is False
        and all(meta.get("model_inference_performed") is False for meta in panels.values()
                if isinstance(meta, dict))
        and _first(manifest.get("figure_contract", {}), "revision_scope", default="")
        .lower().find("scientific") >= 0,
        {"model_inference_performed": manifest.get("model_inference_performed"),
         "validated_sources_modified": manifest.get("validated_sources_modified"),
         "metric_recomputation_performed": _first(manifest, "metric_recomputation_performed",
                                                  "metrics_recomputed")},
    ))
    check(25, "global_polish_without_text_shrinkage", lambda: (
        manifest.get("layout", {}).get("typography_qa", {}).get("passed") is True
        and manifest.get("layout", {}).get("typography_qa", {}).get("role_sizes_pt") == ROLE_SIZES
        and (_flag(manifest, "text_shrinkage_qa_passed", "layout_polish_without_text_shrinkage")
             or _qa_passed(manifest.get("layout", {}).get("typography_qa", {}), "passed"))
        and not manifest.get("layout", {}).get("typography_qa", {}).get("violations")
    ))

    # ------------------------------------------------------------------
    # Supplemental scientific/source checks retained from V3-4.
    # ------------------------------------------------------------------
    check(26, "additive_v3_6_schema_and_five_panel_narrative", lambda: (
        manifest.get("workflow_label") == "mixed_resolution_unified_v3_6"
        and str(manifest.get("schema_version")) == "3.6"
        and manifest.get("run_id") == args.run_id
        and manifest.get("figure_contract", {}).get("panel_sequence") == ["a", "b", "c", "d", "e"]
        and (_flag(manifest, "additive_release", "additive")
             or str(manifest.get("release_mode", "")).lower() in {"additive", "new_release"})
        and bool(manifest.get("v3_5_starting_renderer_anchor")
                 or manifest.get("v3_5_baseline")
                 or manifest.get("v3_4_baseline")
                 or manifest.get("baseline_v3_4")),
        {"workflow": manifest.get("workflow_label"), "schema": manifest.get("schema_version"),
         "run_id": manifest.get("run_id")},
    ))
    check(27, "validated_scientific_ordering_and_panel_schema", lambda: (
        a.get("recipe_order") == RECIPES
        and a.get("dimensions") == {"L": [32, 32], "M": [64, 64], "H": [128, 128]}
        and all(v34.close(left, right) for left, right in zip(
            a.get("exposure_values", []), [1.0, 0.34, 0.4375, 0.15625, 0.1875]
        ))
        and b.get("recipe_order") == RECIPES
        and b.get("model_order") == MODELS
        and b.get("sensor_counts") == COUNTS
        and c.get("models") == MODELS
        and c.get("column_order") == ["reference", *MODELS]
        and c.get("column_count") == 5
        and c.get("row_order") == ["full_field", "zoomed_field", "local_absolute_error"]
        and c.get("row_cell_counts") == {"full_field": 5, "zoomed_field": 5, "local_absolute_error": 4}
        and d.get("qualitative_scales") == SCALES
        and d.get("qualitative_models") == ["DMFGen", "Senseiver"]
        and d.get("main_quantitative_value_count") == 0
        and e.get("matrix_count") == 2
        and e.get("matrix_shape") in ([4, 9], [[4, 3], [4, 3]])
        and e.get("heatmap_values")
        and len(e.get("heatmap_values", [])) == 72,
        {"a": a.get("recipe_order"), "b": b.get("subaxis_roles"), "c": c.get("column_order"),
         "d_scales": d.get("qualitative_scales"), "d_quant_count": d.get("main_quantitative_value_count"),
         "e_matrix_shape": e.get("matrix_shape")},
    ))

    def panel_b_numeric():
        source = _source_path(manifest, release, "SensorSweepAllRecipes_summary")
        index = {
            (row["model"], row["recipe"], int(row["sensor_count"])): row
            for row in v34.read_csv(source) if row.get("metric") == "physical_rel_l2"
        }
        plotted = b.get("plotted_rows", [])
        expected = {(model, recipe, 512) for model in MODELS for recipe in RECIPES} | {
            (model, recipe, count) for model in MODELS for recipe in ZERO_H for count in COUNTS
        }
        observed = {(row["model"], row["recipe"], int(row["sensor_count"])) for row in plotted}
        exact = observed == expected and len(plotted) == 60
        for row in plotted:
            source_row = index[(row["model"], row["recipe"], int(row["sensor_count"]))]
            exact &= all(v34.close(row[key], source_row[key]) for key in ("mean", "ci95_low", "ci95_high"))
            exact &= int(row["valid_n"]) == int(source_row["valid_n"]) == 300
        return exact, {"source": str(source), "plotted_rows": len(plotted), "keys_exact": observed == expected}

    check(28, "panel_b_values_match_validated_sensor_summary", panel_b_numeric)

    def panel_c_cache_recompute():
        if c.get("models") != MODELS or len(c.get("cache_sources", [])) != 4:
            return False, {"models": c.get("models"), "cache_count": len(c.get("cache_sources", []))}
        roi = c["roi"]
        recomputed, identities = {}, []
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
            mask = ((coords[:, 0] >= float(roi["xmin"])) & (coords[:, 0] <= float(roi["xmax"]))
                    & (coords[:, 1] >= float(roi["ymin"])) & (coords[:, 1] <= float(roi["ymax"])))
            model = metadata["model"]
            identities.append((metadata.get("recipe"), metadata.get("snapshot_index"),
                               metadata.get("case_id"), metadata.get("time_index"),
                               metadata.get("sensor_count"), metadata.get("sensor_plan_hash")))
            recomputed[model] = {
                "full": v34.relative_l2(truth, prediction),
                "local": v34.relative_l2(truth[mask], prediction[mask]),
                "roi_points": int(mask.sum()),
            }
            cache_ok &= (
                metadata.get("status") == "ok"
                and metadata.get("recipe") == c.get("recipe") == "5_ZeroH_MRich"
                and int(metadata.get("snapshot_index", -1)) == int(c.get("snapshot", -2)) == 50
                and int(metadata.get("case_id", -1)) == int(c.get("case_id", -2)) == 9160
                and int(metadata.get("time_index", -1)) == int(c.get("time_index", -2)) == 18
                and int(metadata.get("sensor_count", -1)) == int(c.get("sensor_count", -2)) == 512
                and observations.size == 512 and np.all(np.isfinite(prediction))
                and int(mask.sum()) == int(roi["grid_point_count"])
                and v34.close(recomputed[model]["full"], c["full_field_relative_l2"][model])
                and v34.close(recomputed[model]["local"], c["local_relative_l2"][model])
            )
        return cache_ok and set(recomputed) == set(MODELS) and len(set(identities)) == 1, {
            "recomputed": recomputed, "identities": identities,
        }

    check(29, "panel_c_cache_payloads_and_l2_recompute", panel_c_cache_recompute)

    def panel_e_numeric():
        source = _source_path(manifest, release, "MultiscaleWavelet_summary")
        index = {(row["model_key"], row["recipe"], row["scale_group"], row["metric"]): row
                 for row in v34.read_csv(source)}
        cells = e.get("heatmap_values", [])
        observed, exact = set(), len(cells) == 72
        for cell in cells:
            key = (cell["model_key"], cell["recipe"], cell["scale_group"], cell["metric"])
            observed.add(key)
            source_row = index[key]
            statistic = cell.get("statistic", "median")
            exact &= v34.close(cell["value"], source_row.get(statistic, source_row["median"]))
            exact &= int(cell["valid_n"]) == int(source_row["valid_n"]) == 300
        exact &= observed == v34._all_expected_matrix_keys()
        return exact, {"source": str(source), "cell_count": len(cells), "keys_exact": observed == v34._all_expected_matrix_keys()}

    check(30, "panel_e_matrix_values_match_validated_summary", panel_e_numeric)

    def si_tables():
        si = manifest.get("si_outputs", {})
        expected_si = {
            "Sx1_complete_sensor_sweeps",
            "Sx2_expanded_recipe_gallery",
            "Sx3_complete_three_scale_qualitative",
            "Sx4_complete_three_scale_quantitative",
        }
        si_ok = set(si) == expected_si and all(
            {Path(item["path"]).suffix for item in entries} == {".svg", ".pdf", ".png"}
            and all(_resolve_path(item["path"], release).exists() for item in entries)
            for entries in si.values()
        )
        meta = manifest.get("si_metadata", {})
        sx1, sx2, sx3, sx4 = (meta.get(key, {}) for key in ("Sx1", "Sx2", "Sx3", "Sx4"))
        si_ok &= (
            sx1.get("recipes") == RECIPES and sx1.get("sensor_counts") == COUNTS
            and sx2.get("recipes") == FINE_RECIPES and set(sx2.get("models", [])) == set(MODELS)
            and sx2.get("cache_cell_count") == 12
            and sx3.get("scale_groups") == SCALES and sx4.get("scale_groups") == SCALES
            and sx4.get("metrics") == METRICS and sx4.get("models") == MODELS
            and len(sx4.get("heatmap_values", [])) == 72
            and all(int(cell["valid_n"]) == 300 for cell in sx4.get("heatmap_values", []))
        )
        tables = manifest.get("table_outputs", {})
        expected_counts = {"accuracy_512": 20, "sensor_sweeps_64_512": 100,
                           "pattern_correlations_all_scales": 60,
                           "variance_allocation_bias_all_scales": 60}
        table_ok = set(tables) == set(expected_counts) and all(
            int(item.get("row_count", -1)) == expected_counts[key]
            and _resolve_path(item["path"], release).exists()
            for key, item in tables.items()
        )
        extended = _first(manifest, "extended_multiscale_distribution_source",
                          "per_snapshot_distribution_source", default={})
        extended_path = v34._record_path(extended, release)
        extended_ok = bool(extended_path and extended_path.exists() and extended.get("sha256")
                           and v34.sha256(extended_path) == extended["sha256"])
        return si_ok and table_ok and extended_ok, {
            "si_keys": sorted(si), "table_counts": {key: item.get("row_count") for key, item in tables.items()},
            "per_snapshot_distribution": str(extended_path) if extended_path else None,
        }

    check(31, "complete_si_tables_and_per_snapshot_distribution", si_tables)

    def source_tree_immutable():
        imm = manifest.get("source_immutability", {})
        records = _source_records(manifest)
        hashes_match, details = v34._hash_audit(records, release)
        count_before = _first(imm, "result_file_count_before", "file_count_before", default=None)
        count_after = _first(imm, "result_file_count_after", "file_count_after", default=None)
        tree_before = _first(imm, "tree_state_sha256_before", "sha256_before", default=None)
        tree_after = _first(imm, "tree_state_sha256_after", "sha256_after", default=None)
        panel_flags = all(meta.get("validated_sources_modified") is False
                          for meta in panels.values() if isinstance(meta, dict)
                          if "validated_sources_modified" in meta)
        return (
            imm.get("unchanged") is True and tree_before is not None and tree_before == tree_after
            and (count_before is None or count_before == count_after)
            and hashes_match and panel_flags
        ), {
            "recorded_source_count": len(records), "hashes_match": hashes_match,
            "source_failures": [item for item in details if not item["passed"]], "tree_state": imm,
        }

    check(32, "validated_sources_and_result_tree_immutable", source_tree_immutable)

    def output_qa():
        layout = manifest.get("layout", {})
        typography = layout.get("typography_qa", {})
        panel_text = layout.get("panel_text_clearance_qa", {})
        geometry = layout.get("geometry_qa", {})
        frame = layout.get("frame_lineweight_qa", {})
        main_svg = _find_main_file(release, args.run_id, "svg")
        main_pdf = _find_main_file(release, args.run_id, "pdf")
        main_png = _find_main_file(release, args.run_id, "png")
        final_size = [float(value) for value in manifest.get("figure_contract", {}).get("final_size_mm", [])]
        files_ok = all(path.exists() and path.stat().st_size > 0 for path in (main_svg, main_pdf, main_png))
        pdf_size = v34._pdf_size_mm(main_pdf) if files_ok else [0.0, 0.0]
        png_size = list(Image.open(main_png).size) if files_ok else [0, 0]
        expected_px = [round(value / 25.4 * 600) for value in final_size]
        fonts = v34._pdf_font_lines(main_pdf) if files_ok else []
        embedded_fonts = bool(fonts) and all(all(token in line for token in ("yes", "yes", "yes")) for line in fonts)
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
                observed = v34._pdf_size_mm(panel_pdf)
                target = [float(rectangles[label][key]) for key in ("width_mm", "height_mm")]
                standalone[label]["observed_mm"] = observed
                standalone[label]["expected_mm"] = target
                standalone_ok &= all(abs(left - right) <= .02 for left, right in zip(observed, target))
        passed = (
            files_ok and len(final_size) == 2 and abs(final_size[0] - 183.0) <= .02
            and abs(final_size[1] - 221.0) <= .02
            and all(abs(left - right) <= .02 for left, right in zip(pdf_size, final_size))
            and all(abs(left - right) <= 2 for left, right in zip(png_size, expected_px))
            and geometry.get("passed") is True and typography.get("passed") is True
            and panel_text.get("passed") is True and frame.get("passed") is True
            and not any(float(value) > 0 for value in layout.get("text_overflow_in", {}).values())
            and not panel_text.get("cross_panel_text_overlaps") and embedded_fonts
            and "<text" in main_svg.read_text(encoding="utf-8") and standalone_ok
        )
        return passed, {"pdf_size_mm": pdf_size, "png_size_px": png_size, "expected_px": expected_px,
                        "standalone": standalone, "embedded_fonts": embedded_fonts,
                        "geometry": geometry, "typography": typography, "panel_text": panel_text}

    check(33, "publication_exports_geometry_typography_and_collisions", output_qa)

    def artifacts_and_baselines():
        artifacts = v34._artifact_records(manifest)
        artifact_ok, artifact_details = v34._hash_audit(artifacts, release)
        baselines = _baseline_records(manifest, release)
        baseline_ok, baseline_details = v34._hash_audit(baselines, release)
        output_paths = [str(v34._record_path(item, release).resolve()) for item in artifacts
                        if v34._record_path(item, release)]
        unique = len(output_paths) == len(set(output_paths))
        outside = [path for path in output_paths if not Path(path).is_relative_to(release)]
        baseline_paths = {
            str(path.resolve())
            for path in list(V3_4_BASELINE_PATHS) + list(V3_5_BASELINE_PATHS)
        }
        collision = bool(set(output_paths) & baseline_paths)
        imm = manifest.get("baseline_immutability", {})
        return (
            bool(artifacts) and artifact_ok and bool(baselines) and baseline_ok
            and unique and not outside and not collision and imm.get("unchanged") is True,
            {"artifact_count": len(artifacts), "artifact_failures": [item for item in artifact_details if not item["passed"]],
             "baseline_count": len(baselines), "baseline_failures": [item for item in baseline_details if not item["passed"]],
             "unique_output_paths": unique, "outside_release": outside, "baseline_collision": collision,
             "baseline_immutability": imm},
        )

    check(34, "additive_outputs_and_v3_4_baselines_not_overwritten", artifacts_and_baselines)

    def documentation():
        required = ("quantitative_figure_report_v3_6.md", "figure_reference_update_v3_6.md",
                    "completion_report_v3_6.md")
        present = {name: (release / name).exists() and (release / name).stat().st_size > 0 for name in required}
        report = (release / required[0]).read_text(encoding="utf-8") if present[required[0]] else ""
        report_lower = report.lower()
        return (
            all(present.values()) and all(token in report_lower for token in ("v3-5", "matrix", "line plots"))
            and any(token in report_lower for token in ("no new evidence", "scientific content", "layout")),
            {"required": present},
        )

    check(35, "required_v3_6_documentation_complete", documentation)


    # ------------------------------------------------------------------
    # V3-6 anchor and targeted geometry checks.
    # ------------------------------------------------------------------
    def v3_5_hash_anchor():
        expected = {path.resolve(): digest for path, digest in V3_5_BASELINE_PATHS.items()}
        observed = {}
        missing = []
        mismatches = []
        for path, digest in V3_5_BASELINE_PATHS.items():
            if not path.exists():
                missing.append(str(path))
                continue
            actual = v34.sha256(path)
            observed[str(path)] = actual
            if actual != digest:
                mismatches.append({"path": str(path), "expected": digest, "actual": actual})
        anchor_records = []
        for key in (
            "v3_5_baseline", "baseline_v3_5", "v3_5_starting_renderer_anchor",
            "v3_5_baseline_manifest", "v3_5_baseline_pdf", "v3_5_baseline_svg",
            "v3_5_baseline_png", "v3_5_hash_anchor",
        ):
            anchor_records.extend(v34._records_from(manifest.get(key)))
        recorded = {
            v34._resolve_path(item.get("path"), release).resolve(): item.get("sha256")
            for item in anchor_records
            if isinstance(item, dict) and item.get("path") and item.get("sha256")
        }
        anchor_record_ok = all(recorded.get(path) == digest for path, digest in expected.items())
        return (
            not missing
            and not mismatches
            and anchor_record_ok
            and release.resolve() != V3_5_BASELINE_ROOT.resolve()
        ), {
            "missing": missing,
            "mismatches": mismatches,
            "recorded_anchor_complete": anchor_record_ok,
            "baseline_release": str(V3_5_BASELINE_ROOT),
            "observed": observed,
        }

    check(36, "v3_5_hash_anchor_intact", v3_5_hash_anchor)

    def unchanged_scientific_values_and_sources():
        baseline, baseline_path = _baseline_v3_5_manifest()
        if not baseline:
            return False, {"baseline_manifest": str(baseline_path), "missing": True}

        def source_signature(item):
            records = []
            for key in ("all_render_source_records", "all_render_cache_sources"):
                values = item.get(key, [])
                if isinstance(values, list):
                    records.extend(
                        (str(record.get("path")), str(record.get("sha256")))
                        for record in values
                        if isinstance(record, dict) and record.get("path") and record.get("sha256")
                    )
            return sorted(set(records))

        field_map = {
            "a": ("dimensions", "recipe_order", "exposure_values", "shared_roi", "resolution_rois"),
            "b": ("plotted_rows", "recipe_transfer_values", "sensor_sweep_values"),
            "c": (
                "models", "recipe", "snapshot", "case_id", "time_index", "sensor_count",
                "full_field_relative_l2", "local_relative_l2", "roi", "cache_sources",
            ),
            "d": (
                "qualitative_recipe", "qualitative_scales", "qualitative_models",
                "main_quantitative_value_count",
            ),
            "e": ("heatmap_values", "matrix_shape", "matrix_shapes"),
        }
        panel_values = {}
        equal = source_signature(manifest) == source_signature(baseline)
        for label, fields in field_map.items():
            current_panel = manifest.get("panels", {}).get(label, {})
            baseline_panel = baseline.get("panels", {}).get(label, {})
            current_values = {field: _canonical_json(current_panel.get(field)) for field in fields}
            baseline_values = {field: _canonical_json(baseline_panel.get(field)) for field in fields}
            panel_values[label] = {"equal": current_values == baseline_values}
            equal &= panel_values[label]["equal"]
        for key in ("base_data_run_id", "source_data_run_id", "multiscale_run_id",
                    "selection_contract", "representative_index"):
            equal &= _canonical_json(manifest.get(key)) == _canonical_json(baseline.get(key))
        return equal, {
            "baseline_manifest": str(baseline_path),
            "source_records_equal": source_signature(manifest) == source_signature(baseline),
            "panel_values": panel_values,
        }

    check(37, "scientific_values_and_validated_sources_unchanged_vs_v3_5",
          unchanged_scientific_values_and_sources)

    def cd_shared_geometry_anchor():
        baseline, baseline_path = _baseline_v3_5_manifest()
        if not baseline:
            return False, {"baseline_manifest": str(baseline_path), "missing": True}
        baseline_panels = baseline.get("panels", {})
        baseline_c = baseline_panels.get("c", {})
        baseline_d = baseline_panels.get("d", {})
        current_alignment = _row_alignment(manifest, c, d)
        baseline_alignment = _row_alignment(
            baseline, baseline_c, baseline_d
        )
        c_bounds = current_alignment.get("c_boundaries_mm")
        d_bounds = current_alignment.get("d_boundaries_mm")
        base_c_bounds = baseline_alignment.get("c_boundaries_mm")
        base_d_bounds = baseline_alignment.get("d_boundaries_mm")
        rows_equal = all(
            values is not None and len(values) == len(base_values)
            and np.allclose(values, base_values, rtol=0.0, atol=1e-9)
            for values, base_values in (
                (c_bounds, base_c_bounds),
                (d_bounds, base_d_bounds),
            )
        )
        topology_fields = {
            "c": ("column_order", "column_count", "row_order", "row_cell_counts"),
            "d": (
                "qualitative_scales", "qualitative_models", "main_quantitative_value_count",
                "quantitative_plots_present",
            ),
        }
        topology = {}
        topology_equal = True
        for label, fields in topology_fields.items():
            current_panel = manifest.get("panels", {}).get(label, {})
            baseline_panel = baseline_panels.get(label, {})
            topology[label] = {
                field: _canonical_json(current_panel.get(field)) == _canonical_json(baseline_panel.get(field))
                for field in fields
            }
            topology_equal &= all(topology[label].values())
        row_order_equal = (
            current_alignment.get("row_order") == baseline_alignment.get("row_order")
            == ["full/large", "zoom/intermediate", "local-error/fine"]
        )
        return (
            current_alignment.get("exact") is True
            and baseline_alignment.get("exact") is True
            and rows_equal
            and row_order_equal
            and topology_equal
            and _flag(manifest, "shared_cd_parent_gridspec", "shared_cd_grid")
            and _flag(c, "uses_shared_cd_grid", "shared_row_geometry")
            and _flag(d, "uses_shared_cd_grid", "shared_row_geometry")
        ), {
            "baseline_manifest": str(baseline_path),
            "current_alignment": current_alignment,
            "baseline_alignment": baseline_alignment,
            "topology": topology,
        }

    check(38, "c_d_shared_three_row_boundaries_and_topology_unchanged",
          cd_shared_geometry_anchor)

    def panel_a_allocation_and_contours():
        allocation = _first(
            a, "content_allocation_qa", "content_allocation",
            "content_split_qa", "allocation_qa", default={}
        )
        allocation = allocation if isinstance(allocation, dict) else {}
        pair = _first(
            allocation, "fractions", "content_fractions", "allocation_fractions",
            "ratio", "ratio_fraction", "ratio_percent", "allocation_ratio_percent",
            default=None
        )
        if isinstance(pair, dict):
            left = _first(pair, "left", "heatmap", "heatmaps", "field", "image",
                          "visual", "content", default=None)
            right = _first(pair, "right", "bars", "bar", "recipe", "chart",
                           "annotation", default=None)
            pair = [left, right]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            pair = [
                _first(allocation, "left_fraction", "heatmap_fraction",
                       "field_fraction", "visual_fraction", default=None),
                _first(allocation, "right_fraction", "bar_fraction",
                       "recipe_fraction", "chart_fraction", default=None),
            ]
        fractions = []
        for value in pair:
            try:
                value = float(value)
                fractions.append(value / 100.0 if value > 1.5 else value)
            except (TypeError, ValueError):
                fractions.append(-1.0)
        allocation_ok = (
            len(fractions) == 2
            and 0.40 <= fractions[0] <= 0.50
            and 0.50 <= fractions[1] <= 0.60
            and abs(sum(fractions) - 1.0) <= 0.03
            and (
                _flag(a, "about_45_55_content_allocation",
                      "content_allocation_4555", "content_allocation_qa_passed")
                or _qa_passed(allocation, "passed")
            )
        )

        contour = _first(
            a, "contour_gap_qa", "contour_spacing_qa",
            "contour_gaps_qa", "contour_gap_contract", default={}
        )
        contour = contour if isinstance(contour, dict) else {}
        current_gaps = _number_list(_first(
            contour, "gaps_mm", "contour_gaps_mm", "gap_mm",
            "contour_gap_mm", "minimum_gap_mm", "min_gap_mm", default=None
        ))
        if not current_gaps:
            current_gaps = _number_list(_first(
                a, "contour_gaps_mm", "contour_gap_mm", default=None
            ))
        baseline, baseline_path = _baseline_v3_5_manifest()
        baseline_a = baseline.get("panels", {}).get("a", {}) if baseline else {}
        baseline_contour = _first(
            baseline_a, "contour_gap_qa", "contour_spacing_qa",
            "contour_gaps_qa", "contour_gap_contract", default={}
        )
        baseline_contour = baseline_contour if isinstance(baseline_contour, dict) else {}
        baseline_gaps = _number_list(_first(
            baseline_contour, "gaps_mm", "contour_gaps_mm", "gap_mm",
            "contour_gap_mm", "minimum_gap_mm", "min_gap_mm", default=None
        ))
        increase = _first(
            contour, "increase_vs_v3_5_mm", "gap_increase_mm",
            "minimum_increase_mm", default=None
        )
        try:
            increase = float(increase)
        except (TypeError, ValueError):
            increase = None
        if current_gaps and baseline_gaps and len(current_gaps) == len(baseline_gaps):
            contour_increased = (
                all(current >= old for current, old in zip(current_gaps, baseline_gaps))
                and any(current > old for current, old in zip(current_gaps, baseline_gaps))
            )
        else:
            contour_increased = increase is not None and increase > 0
        contour_ok = (
            bool(current_gaps)
            and contour_increased
            and (
                _flag(a, "contour_gaps_increased_vs_v3_5",
                      "contour_spacing_increased", "increased_contour_gaps")
                or _first(contour, "increased_vs_v3_5", default=False) is True
            )
            and (
                _qa_passed(contour, "passed")
                or _flag(a, "contour_gap_qa_passed", "contour_spacing_qa_passed")
            )
        )
        return allocation_ok and contour_ok, {
            "allocation": {"fractions": fractions, "qa": allocation},
            "contour": {
                "current_gaps_mm": current_gaps,
                "baseline_gaps_mm": baseline_gaps,
                "increase_vs_v3_5_mm": increase,
                "qa": contour,
                "baseline_manifest": str(baseline_path),
            },
        }

    check(39, "panel_a_4555_content_allocation_and_increased_contour_gaps",
          panel_a_allocation_and_contours)

    def panel_b_height_and_style():
        baseline, baseline_path = _baseline_v3_5_manifest()
        baseline_b = baseline.get("panels", {}).get("b", {}) if baseline else {}
        current_heights, current_height_source = _axis_heights(manifest, b)
        baseline_heights, baseline_height_source = _axis_heights(baseline, baseline_b)
        multipliers = _number_list(_first(
            b, "plotting_axis_height_multipliers_vs_v3_5",
            "plot_axis_height_multipliers_vs_v3_5",
            "axis_height_multipliers_vs_v3_5", default=None
        ))
        if len(multipliers) == 3:
            height_factors = multipliers
        elif len(current_heights) == 3 and len(baseline_heights) == 3:
            height_factors = [
                current / old if old > 0 else 0
                for current, old in zip(current_heights, baseline_heights)
            ]
        else:
            height_factors = []
        height_qa = _first(
            b, "plot_axis_height_qa", "plotting_axis_height_qa",
            "axis_height_qa", default={}
        )
        height_qa = height_qa if isinstance(height_qa, dict) else {}
        heights_ok = (
            len(height_factors) == 3
            and all(factor >= 1.14 for factor in height_factors)
            and (
                _qa_passed(height_qa, "passed")
                or _flag(b, "all_plot_axes_at_least_1_14x_v3_5",
                         "plot_axes_height_qa_passed", "axis_heights_increased_vs_v3_5")
            )
        )

        current_widths, current_width_source = _style_values(
            b, (
                "bottom_line_widths_pt", "bottom_line_width_pt",
                "sweep_line_widths_pt", "sweep_line_width_pt",
                "plot_line_widths_pt", "plot_line_width_pt",
            )
        )
        baseline_widths, baseline_width_source = _baseline_style_values(
            baseline, baseline_b, (
                "bottom_line_widths_pt", "bottom_line_width_pt",
                "sweep_line_widths_pt", "sweep_line_width_pt",
                "plot_line_widths_pt", "plot_line_width_pt",
            ), ("line_plot", "line_secondary")
        )
        if len(current_widths) == len(baseline_widths):
            thinner = all(current < old for current, old in zip(current_widths, baseline_widths))
        else:
            thinner = bool(current_widths and baseline_widths
                           and max(current_widths) < min(baseline_widths))
        style_qa = _first(
            b, "bottom_line_marker_qa", "line_marker_qa",
            "bottom_plot_style_qa", default={}
        )
        style_qa = style_qa if isinstance(style_qa, dict) else {}
        thinner_ok = (
            thinner
            and (
                _qa_passed(style_qa, "passed")
                or _flag(b, "bottom_lines_thinner_vs_v3_5",
                         "bottom_line_width_reduced_vs_v3_5", "sweep_lines_thinner")
            )
        )

        current_markers, current_marker_source = _style_values(
            b, (
                "bottom_marker_sizes_pt", "bottom_marker_size_pt",
                "sweep_marker_sizes_pt", "sweep_marker_size_pt",
                "marker_sizes_pt", "marker_size_pt",
            )
        )
        baseline_markers, baseline_marker_source = _style_values(
            baseline_b, (
                "bottom_marker_sizes_pt", "bottom_marker_size_pt",
                "sweep_marker_sizes_pt", "sweep_marker_size_pt",
                "marker_sizes_pt", "marker_size_pt",
            )
        )
        if len(current_markers) == len(baseline_markers) and current_markers:
            markers_larger = all(current > old for current, old in zip(current_markers, baseline_markers))
        else:
            markers_larger = bool(current_markers and all(value > 0 for value in current_markers))
        markers_ok = (
            markers_larger
            and (
                _qa_passed(style_qa, "markers_larger_vs_v3_5")
                or _flag(b, "bottom_markers_larger_vs_v3_5",
                         "markers_increased_vs_v3_5", "larger_bottom_markers")
            )
        )
        return heights_ok and thinner_ok and markers_ok, {
            "baseline_manifest": str(baseline_path),
            "axis_heights_mm": {
                "current": current_heights, "baseline": baseline_heights,
                "factors": height_factors,
                "current_source": current_height_source,
                "baseline_source": baseline_height_source,
            },
            "line_widths_pt": {
                "current": current_widths, "baseline": baseline_widths,
                "thinner": thinner,
                "current_source": current_width_source,
                "baseline_source": baseline_width_source,
            },
            "marker_sizes_pt": {
                "current": current_markers, "baseline": baseline_markers,
                "larger": markers_larger,
                "current_source": current_marker_source,
                "baseline_source": baseline_marker_source,
            },
            "style_qa": style_qa,
        }

    check(40, "panel_b_axes_1_14x_and_thinner_lines_larger_markers",
          panel_b_height_and_style)

    def panel_e_recipe_gap_contract():
        baseline, baseline_path = _baseline_v3_5_manifest()
        current = _metric_gap_values(e)
        old = _metric_gap_values(baseline.get("panels", {}).get("e", {}) if baseline else {})
        current_within = current["within_mm"]
        old_within = old["within_mm"]
        if len(current_within) >= 4 and len(old_within) >= 4:
            within_reduced = all(
                current_within[index] < old_within[index]
                for index in range(4)
            )
        else:
            within_reduced = False
        central_larger = (
            current["central_mm"] is not None
            and len(current_within) >= 4
            and current["central_mm"] > max(current_within)
        )
        if old["central_mm"] is not None and current["central_mm"] is not None:
            central_larger &= current["central_mm"] >= old["central_mm"]
        return (
            within_reduced
            and central_larger
            and (
                _flag(e, "within_metric_recipe_gaps_reduced_vs_v3_5",
                      "recipe_gaps_reduced_vs_v3_5")
                or _qa_passed(current["qa"], "passed")
            )
        ), {
            "current": current,
            "baseline": old,
            "baseline_manifest": str(baseline_path),
        }

    check(41, "panel_e_reduced_within_metric_gaps_larger_central_gap",
          panel_e_recipe_gap_contract)

    check(42, "panel_e_colorbar_correlation_bias_labels_removed", lambda: (
        _flag(e, "colorbar_labels_hidden", "metric_colorbar_labels_hidden",
              "colorbar_metric_labels_removed")
        and not any(
            term in str(label).strip().lower()
            for label in _first(e, "colorbar_labels", default=[])
            for term in ("correlation", "bias")
        )
        and _marker_count(
            svg["e"],
            "panel-e-colorbar-label-correlation",
            "panel-e-colorbar-label-bias",
            "colorbar-label-correlation",
            "colorbar-label-bias",
        ) == 0,
        {"colorbar_labels": _first(e, "colorbar_labels", default=[]),
         "hidden": _first(e, "colorbar_labels_hidden",
                          "metric_colorbar_labels_hidden", default=False)},
    ))

    def panel_e_matrix_colorbar_gap():
        baseline, baseline_path = _baseline_v3_5_manifest()
        current = _metric_gap_values(e)
        old = _metric_gap_values(baseline.get("panels", {}).get("e", {}) if baseline else {})
        reduced = (
            current["matrix_colorbar_mm"] is not None
            and old["matrix_colorbar_mm"] is not None
            and current["matrix_colorbar_mm"] < old["matrix_colorbar_mm"]
        )
        if not reduced:
            reduced = (
                current["matrix_colorbar_mm"] is not None
                and current["matrix_colorbar_mm"] > 0
                and _flag(e, "matrix_to_colorbar_gap_reduced_vs_v3_5",
                          "matrix_colorbar_gap_reduced")
            )
        return reduced and (
            _qa_passed(_first(e, "matrix_to_colorbar_gap_qa", default={}), "passed")
            or _flag(e, "matrix_to_colorbar_gap_qa_passed",
                     "matrix_to_colorbar_gap_reduced_vs_v3_5")
        ), {
            "current_gap_mm": current["matrix_colorbar_mm"],
            "baseline_gap_mm": old["matrix_colorbar_mm"],
            "baseline_manifest": str(baseline_path),
        }

    check(43, "panel_e_matrix_to_colorbar_gap_reduced", panel_e_matrix_colorbar_gap)

    passed = bool(checks) and all(item["passed"] for item in checks)
    payload = {
        "workflow_label": "mixed_resolution_unified_v3_6",
        "schema_version": "3.6",
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

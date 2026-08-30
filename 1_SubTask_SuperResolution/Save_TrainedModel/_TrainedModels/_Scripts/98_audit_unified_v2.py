#!/usr/bin/env python
"""Audit unified-v2 figure provenance, numerical agreement, and exports."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml
import global_style as manuscript

from common.config import FIGURES_DIR, RESULTS_DIR, add_common_args, load_config
from common.io_utils import matching_or_latest, read_csv, write_json
from common.panel_c_tuning import apply_panel_c_tuning
from common.physical_figure_layout import resolve_physical_layout


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _index(rows, keys):
    return {tuple(row.get(key) for key in keys): row for row in rows}


def _close_rows(left, right, fields, rtol=1e-10, atol=1e-12):
    return all(np.isclose(_float(left.get(field)), _float(right.get(field)), rtol=rtol, atol=atol, equal_nan=True) for field in fields)


def _pdf_size_mm(path):
    try:
        from pypdf import PdfReader
        page = PdfReader(str(path)).pages[0]
        width_pt = float(page.mediabox.width); height_pt = float(page.mediabox.height)
    except ImportError:
        match = re.search(
            rb"/MediaBox\s*\[\s*[-+0-9.]+\s+[-+0-9.]+\s+([-+0-9.]+)\s+([-+0-9.]+)\s*\]",
            Path(path).read_bytes(),
        )
        if match is None:
            raise RuntimeError(f"Could not locate PDF MediaBox in {path}")
        width_pt, height_pt = map(float, match.groups())
    return [width_pt * 25.4 / 72.0, height_pt * 25.4 / 72.0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, models=False)
    parser.add_argument("--data-run-id", required=True)
    parser.add_argument(
        "--source-data-run-id",
        help="Run ID of immutable/derived metric CSVs when figures use a newer plotting-only timestamp.",
    )
    parser.add_argument(
        "--multiscale-run-id",
        help="Run ID of the cache-only wavelet CSV/metadata bundle (defaults to the figure run ID).",
    )
    parser.add_argument("--base-data-run-id", default="2026-07-13_14-55")
    parser.add_argument("--layout", type=Path, default=Path(__file__).with_name("publication_layout_unified_v2.yaml"))
    parser.add_argument("--backup", type=Path, default=Path(__file__).parent / "_Backup" / "visualization_pre_unified_20260713_1553")
    parser.add_argument("--composite-only", action="store_true", help="Audit the master figure without requiring same-timestamp standalone exports.")
    parser.add_argument(
        "--allow-updated-cache-artifacts", action="store_true",
        help="Allow the formal cache manifest/status hashes to change during an authorized checkpoint refresh.",
    )
    parser.add_argument(
        "--require-qualitative-alternate", action="store_true",
        help="Also require the optional Version 1 qualitative standalone triplet.",
    )
    parser.add_argument(
        "--expect-incremental-cache-fill", action="store_true",
        help="Require the source manifest to record an explicitly authorized supplemental cache fill.",
    )
    parser.add_argument(
        "--refreshed-models", nargs="+", default=[],
        help="Models explicitly recomputed from refreshed checkpoints.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    rid = args.data_run_id
    source_rid = args.source_data_run_id or rid
    multiscale_rid = args.multiscale_run_id or rid
    checks = []

    def check(name, passed, detail=None):
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    backup_hashes = args.backup / "validated_artifact_hashes_before.json"
    baseline = json.loads(backup_hashes.read_text(encoding="utf-8"))["sha256"]
    changed = []
    missing_originals = []
    for raw, digest in baseline.items():
        path = Path(raw)
        if not path.exists():
            missing_originals.append(raw)
        elif _sha256(path) != digest:
            changed.append(raw)
    allowed_cache_suffixes = {
        "ReconstructionCache/CacheStatus_formal_20260712.csv",
        "ReconstructionCache/ReconstructionCache_manifest_formal_20260712.csv",
    } if args.allow_updated_cache_artifacts else set()
    unexpected_changes = [
        path for path in changed
        if not any(path.endswith(suffix) for suffix in allowed_cache_suffixes)
    ]
    check("immutable_validated_artifacts", not unexpected_changes and not missing_originals,
          {"hashed": len(baseline), "changed": changed, "allowed_change_suffixes": sorted(allowed_cache_suffixes),
           "unexpected_changes": unexpected_changes, "missing": missing_originals})

    result_dir = RESULTS_DIR / "UnifiedPublicationV2"
    accuracy = read_csv(result_dir / f"AllRecipeAccuracy_summary_{source_rid}.csv")
    qa = read_csv(RESULTS_DIR / "QuestionA_DataBenefit" / f"QuestionA_summary_{args.base_data_run_id}.csv")
    qb = read_csv(RESULTS_DIR / "QuestionB_ZeroH" / f"QuestionB_summary_{args.base_data_run_id}.csv")
    source_accuracy = _index(
        [row for row in qa + qb if row.get("metric") == "physical_rel_l2"], ["model", "recipe"],
    )
    discrepancies = []
    for row in accuracy:
        source = source_accuracy.get((row.get("model"), row.get("recipe")))
        if row.get("status") == "ok" and (source is None or not _close_rows(row, source, ["mean", "ci95_low", "ci95_high", "valid_n"])):
            discrepancies.append({"row": row, "source": source})
    check("panel_b_matches_finalized_summaries", not discrepancies, discrepancies[:4])

    sweep_new = read_csv(result_dir / f"SensorSweepAllRecipes_summary_{source_rid}.csv")
    sweep_old = read_csv(RESULTS_DIR / "SensorSweep" / f"SensorSweep_summary_{args.base_data_run_id}.csv")
    sweep_index = _index(sweep_new, ["model", "recipe", "sensor_count", "metric"])
    discrepancies = []
    for row in sweep_old:
        if row.get("metric") != "physical_rel_l2" or int(float(row.get("sensor_count", -1))) not in {64, 128, 256, 384, 512}:
            continue
        other = sweep_index.get((row.get("model"), row.get("recipe"), row.get("sensor_count"), row.get("metric")))
        if other is None or not _close_rows(row, other, ["mean", "ci95_low", "ci95_high", "valid_n"]):
            discrepancies.append({"source": row, "unified": other})
    check("panel_d_matches_finalized_sweep", not discrepancies, discrepancies[:4])

    wavelet_dir = RESULTS_DIR / "MultiscaleWavelet"
    wavelet_summary_path = wavelet_dir / f"MultiscaleWavelet_summary_{multiscale_rid}.csv"
    wavelet_metadata_path = wavelet_dir / f"MultiscaleWavelet_metadata_{multiscale_rid}.json"
    wavelet_per_snapshot_path = wavelet_dir / f"MultiscaleWavelet_per_snapshot_{multiscale_rid}.csv"
    wavelet_summary = read_csv(wavelet_summary_path)
    wavelet_metadata = json.loads(wavelet_metadata_path.read_text(encoding="utf-8"))
    validation = wavelet_metadata.get("validation", {})
    check(
        "wavelet_reconstruction_identities",
        validation.get("truth_max_reconstruction_residual", np.inf) <= 1e-6
        and validation.get("prediction_max_reconstruction_residual", np.inf) <= 1e-6,
        validation,
    )
    check(
        "wavelet_truth_identity_metrics",
        validation.get("truth_vs_truth_max_pattern_correlation_error", np.inf) <= 1e-6
        and validation.get("truth_vs_truth_max_variance_fraction_bias_pp", np.inf) <= 1e-9
        and validation.get("truth_vs_truth_max_component_rel_l2", np.inf) <= 1e-12,
        validation,
    )
    check(
        "wavelet_variance_fractions_and_nonempty_groups",
        validation.get("truth_variance_fraction_max_sum_error", np.inf) <= 1e-12
        and validation.get("prediction_variance_fraction_max_sum_error", np.inf) <= 1e-12
        and validation.get("all_scale_groups_nonempty") is True
        and all(float(value) > 0.0 for value in validation.get("minimum_true_variance_fraction_by_scale", {}).values()),
        validation,
    )
    check(
        "canonical_300_alignment",
        wavelet_metadata.get("canonical_snapshot_count") == 300
        and validation.get("cache_entries_processed") == 6000
        and validation.get("per_snapshot_rows") == 18000
        and validation.get("all_summary_cells_valid_n_300") is True
        and all(int(float(row.get("valid_n", -1))) == 300 for row in wavelet_summary),
        validation,
    )
    check(
        "representative_snapshot_truth_only",
        wavelet_metadata.get("representative_snapshot", {}).get("selection_uses_model_performance") is False
        and wavelet_metadata.get("representative_snapshot", {}).get("selection_rule") == "median_fine_variance_fraction_true",
        wavelet_metadata.get("representative_snapshot"),
    )
    check(
        "representative_baseline_aggregate_only",
        wavelet_metadata.get("representative_baseline", {}).get("selection_rule") == "best_non_dmf_by_mean_physical_l2"
        and wavelet_metadata.get("representative_baseline", {}).get("valid_n") == 300,
        wavelet_metadata.get("representative_baseline"),
    )
    cache_fingerprint = wavelet_metadata.get("cache_inventory_fingerprint_before_after", [])
    cache_manifest_hash = wavelet_metadata.get("cache_manifest_sha256_before_after", [])
    check(
        "cache_only_no_inference_no_cache_changes",
        wavelet_metadata.get("metric_source") == "existing validated reconstruction caches only; no model inference"
        and wavelet_metadata.get("cache_files_modified") is False
        and len(cache_fingerprint) == 2 and cache_fingerprint[0] == cache_fingerprint[1]
        and len(cache_manifest_hash) == 2 and cache_manifest_hash[0] == cache_manifest_hash[1],
        {
            "metric_source": wavelet_metadata.get("metric_source"),
            "cache_files_modified": wavelet_metadata.get("cache_files_modified"),
            "cache_inventory": cache_fingerprint,
            "cache_manifest": cache_manifest_hash,
        },
    )
    check(
        "wavelet_export_triplet_exists",
        all(path.exists() and path.stat().st_size > 0 for path in (
            wavelet_per_snapshot_path, wavelet_summary_path, wavelet_metadata_path,
        )),
        [str(wavelet_per_snapshot_path), str(wavelet_summary_path), str(wavelet_metadata_path)],
    )

    assembled = FIGURES_DIR / "Assembled" / f"MixedResolution_unified_v2_{rid}"
    source_manifest_path = FIGURES_DIR / "Assembled" / f"FigureSourceManifest_unified_v2_{rid}.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    data_preparation_path = Path(source_manifest["data_preparation_manifest"]["path"])
    data_preparation = json.loads(data_preparation_path.read_text(encoding="utf-8"))
    check(
        "checkpoint_refresh_scope_recorded",
        sorted(set(data_preparation.get("refreshed_models", []))) == sorted(set(args.refreshed_models)),
        {
            "recorded": data_preparation.get("refreshed_models", []),
            "expected": args.refreshed_models,
        },
    )
    outputs = [assembled.with_suffix(ext) for ext in (".pdf", ".svg", ".png")]
    check("composite_triplet_exists", all(path.exists() and path.stat().st_size > 0 for path in outputs), [str(path) for path in outputs])
    pdf_size = _pdf_size_mm(assembled.with_suffix(".pdf"))
    # Audit the one physical contract used by both figure entry points.
    layout = apply_panel_c_tuning(yaml.safe_load(args.layout.read_text(encoding="utf-8")))
    physical_layout = resolve_physical_layout(layout)
    expected_size = source_manifest.get("figure_contract", {}).get(
        "final_size_mm", [physical_layout.width_mm, physical_layout.height_mm],
    )
    check("composite_size_matches_layout", np.allclose(pdf_size, expected_size, atol=.15), {
        "actual_mm": pdf_size, "expected_mm": expected_size,
    })
    if not args.composite_only:
        panel_c_pdf = FIGURES_DIR / "PublicationPanels" / "Panel_C" / f"Panel_c_SelectedQualitative_{rid}.pdf"
        panel_c_size = _pdf_size_mm(panel_c_pdf)
        expected_panel_c_size = [
            physical_layout.panels["c"].width_mm,
            physical_layout.panels["c"].height_mm,
        ]
        check("panel_c_size_matches_compact_layout", np.allclose(panel_c_size, expected_panel_c_size, atol=.15), {
            "actual_mm": panel_c_size, "expected_mm": expected_panel_c_size,
        })
    svg_text = assembled.with_suffix(".svg").read_text(encoding="utf-8")
    check("unified_story_has_no_block_headers", "Value of lower-resolution training data" not in svg_text and "Zero-H transfer" not in svg_text)
    check(
        "requested_panel_subtitles_removed",
        "one matched held-out case" not in svg_text
        and "Zero-H-M-rich training" not in svg_text,
    )
    check("svg_keeps_editable_text", "<text" in svg_text, {"text_elements": svg_text.count("<text")})

    required_panel_bases = []
    if not args.composite_only:
        required_panel_bases = [
            "Panel_a_ResolutionProtocol", "Panel_b_AllRecipeAccuracy", "Panel_c_SelectedQualitative",
            "Panel_d_SensorSweep",
        ]
    missing_outputs = []
    for base in required_panel_bases:
        panel_label = base.split("_", 2)[1].upper()
        for ext in ("pdf", "svg", "png"):
            path = FIGURES_DIR / "PublicationPanels" / f"Panel_{panel_label}" / f"{base}_{rid}.{ext}"
            if not path.exists() or path.stat().st_size == 0:
                missing_outputs.append(str(path))
    if not args.composite_only:
        for base in (
            "Panel_e_ScaleSpecificResiduals",
            "Panel_f_MultiscaleFidelityHeatmaps",
            "SI_MultiscaleFidelityIntervals",
        ):
            for ext in ("pdf", "svg", "png"):
                path = FIGURES_DIR / "MultiscaleWavelet" / f"{base}_{rid}.{ext}"
                if not path.exists() or path.stat().st_size == 0:
                    missing_outputs.append(str(path))
    if not args.composite_only:
        if args.require_qualitative_alternate:
            for ext in ("pdf", "svg", "png"):
                path = FIGURES_DIR / "PublicationPanels" / "Panel_C" / f"Panel_c_SelectedQualitative_Version1_{rid}.{ext}"
                if not path.exists() or path.stat().st_size == 0:
                    missing_outputs.append(str(path))
    check("required_standalone_panel_triplets_exist", not missing_outputs, missing_outputs)
    if not args.composite_only:
        panel_pdf_paths = {
            "a": FIGURES_DIR / "PublicationPanels" / "Panel_A" / f"Panel_a_ResolutionProtocol_{rid}.pdf",
            "b": FIGURES_DIR / "PublicationPanels" / "Panel_B" / f"Panel_b_AllRecipeAccuracy_{rid}.pdf",
            "c": FIGURES_DIR / "PublicationPanels" / "Panel_C" / f"Panel_c_SelectedQualitative_{rid}.pdf",
            "d": FIGURES_DIR / "PublicationPanels" / "Panel_D" / f"Panel_d_SensorSweep_{rid}.pdf",
            "e": FIGURES_DIR / "MultiscaleWavelet" / f"Panel_e_ScaleSpecificResiduals_{rid}.pdf",
            "f": FIGURES_DIR / "MultiscaleWavelet" / f"Panel_f_MultiscaleFidelityHeatmaps_{rid}.pdf",
        }
        size_errors = {}
        for label, path in panel_pdf_paths.items():
            actual = _pdf_size_mm(path)
            target = [physical_layout.panels[label].width_mm, physical_layout.panels[label].height_mm]
            size_errors[label] = {
                "actual_mm": actual, "target_mm": target,
                "absolute_error_mm": list(np.abs(np.asarray(actual) - np.asarray(target))),
            }
        check(
            "standalone_pages_equal_composite_panel_rectangles",
            all(max(row["absolute_error_mm"]) <= .02 for row in size_errors.values()),
            size_errors,
        )
    check("panel_sequence_is_a_to_f", source_manifest.get("layout", {}).get("panel_sequence") == list("abcdef"))
    check("panel_b_and_d_log_axes", source_manifest.get("axis_scales", {}).get("b") == "log" and source_manifest.get("axis_scales", {}).get("d") == "log")
    check(
        "incremental_fill_policy",
        bool(source_manifest.get("incremental_cache_fill_used")) is bool(args.expect_incremental_cache_fill),
        {
            "recorded": bool(source_manifest.get("incremental_cache_fill_used")),
            "expected": bool(args.expect_incremental_cache_fill),
            "supplemental_manifest": source_manifest.get("supplemental_cache_manifest"),
        },
    )
    panel_b_meta = source_manifest.get("panels", {}).get("b", {})
    check(
        "panel_b_uses_configured_sensor_count",
        int(panel_b_meta.get("sensor_count", -1)) == int(layout["panel_b"].get("sensor_count", cfg["sensor_plan"]["default_count"])),
        panel_b_meta,
    )
    roi_selection = source_manifest.get("canonical_selection", {}).get("roi", {}).get("selection")
    check("qualitative_roi_truth_only", roi_selection in {
        "maximum integrated ground-truth gradient magnitude",
        "manual ground-truth-coordinate square",
    }, roi_selection)
    check("qualitative_shared_limits_recorded", bool(source_manifest.get("color_limits", {}).get("c_field")) and bool(source_manifest.get("color_limits", {}).get("c_error")))
    panel_c_meta = source_manifest.get("panels", {}).get("c", {})
    compact = panel_c_meta.get("layout", {})
    panel_c_overlap_qa = source_manifest.get("layout", {}).get("panel_c_overlap_qa", {})
    check(
        "panel_c_has_no_text_collisions",
        panel_c_overlap_qa.get("passed") is True
        and not panel_c_overlap_qa.get("text_text_collisions")
        and not panel_c_overlap_qa.get("text_figure_collisions"),
        panel_c_overlap_qa,
    )
    panel_c_colorbars = compact.get("colorbars", {})
    check("qualitative_compact_structure_recorded", all((
        compact.get("row_labels") == ["Full H-resolution field", "Zoomed-in region"],
        compact.get("bottom_row_headers") == {"reference": "Sensor layout", "models": "Absolute error"},
        np.isclose(float(panel_c_colorbars.get("field_length_ratio", np.nan)), 1.0)
        and np.isclose(float(panel_c_colorbars.get("error_length_ratio", np.nan)), 1.0),
        panel_c_colorbars.get("field") == "vertical, centered on rows 1 and 2; exactly 4 ticks; Field value label",
        panel_c_colorbars.get("error") == "vertical, bottom-aligned on row 3 with a top multiplier cap; exactly 4 ticks; Absolute error label",
        len(compact.get("field_colorbar_ticks", [])) == 4,
        len(compact.get("error_colorbar_ticks", [])) == 4,
        compact.get("colorbar_labels") == {"field": "Field value", "error": "Absolute error"},
        compact.get("colorbar_tick_format", {}).get("mantissa_decimals") == 1,
        compact.get("colorbar_tick_format", {}).get("common_exponent_at_top") is True,
        all(
            abs(float((compact.get("colorbar_tick_format", {}).get(key) or {}).get(
                "alignment_error_px"
            ) or 0.0)) <= 0.1
            for key in ("field", "error")
        ),
        compact.get("subplot_borders") == "solid black on the middle zoomed-in row and row 3 column 1",
        panel_c_meta.get("sensors_on_reference_zoom") is False,
        len(compact.get("vertical_divider_x_parent", compact.get("vertical_divider_x_figure", []))) == 3,
    )), panel_c_meta)
    check(
        "panel_c_error_semantics",
        panel_c_meta.get("error_map_definition") == "pointwise absolute error = abs(reconstruction - ground truth)"
        and panel_c_meta.get("relative_l2_annotation_format") == "Rel. L_2 = {value:.3f}",
        panel_c_meta,
    )
    panel_d_meta = source_manifest.get("panels", {}).get("d", {})
    check(
        "panel_d_visual_contract",
        panel_d_meta.get("sensor_counts") == [64, 128, 256, 384, 512]
        and panel_d_meta.get("formal_setting") == 256
        and panel_d_meta.get("axis_scale") == "log"
        and panel_d_meta.get("statistic") == "mean with bootstrap 95% CI"
        and len(panel_d_meta.get("recipes", [])) == 5,
        panel_d_meta,
    )
    panel_e_meta = source_manifest.get("panels", {}).get("e", {})
    configured_colorbar_fraction = float(
        layout["panel_e"]["layout"]["residual_colorbar_height_fraction"]
    )
    residual_colorbar_bounds = panel_e_meta.get("residual_colorbar_bounds_parent", {})
    residual_row_height = float(panel_e_meta.get("residual_row_height_parent", np.nan))
    check(
        "panel_e_raw_truth_residual_contract",
        panel_e_meta.get("scale_groups") == ["large", "intermediate", "fine"]
        and panel_e_meta.get("wavelet") == wavelet_metadata.get("actual_wavelet")
        and panel_e_meta.get("boundary_mode") == wavelet_metadata.get("boundary_mode")
        and panel_e_meta.get("sensor_overlays") is False
        and panel_e_meta.get("model_specific_adjustment") is False
        and panel_e_meta.get("smoothing") is False
        and panel_e_meta.get("component_definition") == "component_truth_s"
        and panel_e_meta.get("residual_definition") == "component_pred_model_s - component_truth_s"
        and panel_e_meta.get("display_units") == "density"
        and panel_e_meta.get("residual_sign") == "prediction minus truth"
        and panel_e_meta.get("truth_residual_max_abs") == 0.0
        and set(panel_e_meta.get("gt_rms", {})) == {"large", "intermediate", "fine"}
        and all(float(value) > 0.0 for value in panel_e_meta.get("gt_rms", {}).values())
        and panel_e_meta.get("component_percentile") == 99.5
        and panel_e_meta.get("residual_percentile") == 99.5
        and panel_e_meta.get("colorbar_count") == 3
        and panel_e_meta.get("residual_colorbar_height_fraction") == configured_colorbar_fraction
        and set(residual_colorbar_bounds) == {"large", "intermediate", "fine"}
        and np.isfinite(residual_row_height) and residual_row_height > 0.0
        and all(
            np.isclose(bounds[3] / residual_row_height, configured_colorbar_fraction)
            for bounds in residual_colorbar_bounds.values()
        )
        and panel_e_meta.get("component_cmap") == "RdBu_r"
        and panel_e_meta.get("residual_cmap") == layout["panel_e"]["residual_cmap"]
        and set(panel_e_meta.get("residual_norm", {})) == {"large", "intermediate", "fine"}
        and all(
            norm.get("name") == "TwoSlopeNorm"
            and norm.get("vcenter") == 0.0
            and np.isclose(norm.get("vmin", np.nan), -norm.get("vmax", np.nan))
            for norm in panel_e_meta.get("residual_norm", {}).values()
        )
        and panel_e_meta.get("contour_levels") == {"truth": 5, "residual": 0}
        and panel_e_meta.get("model_inference_performed") is False
        and panel_e_meta.get("representative_snapshot", {}).get("snapshot_index")
            == wavelet_metadata.get("representative_snapshot", {}).get("snapshot_index")
        and panel_e_meta.get("representative_baseline", {}).get("model_key")
            == wavelet_metadata.get("representative_baseline", {}).get("model_key"),
        panel_e_meta,
    )
    panel_f_meta = source_manifest.get("panels", {}).get("f", {})
    check(
        "panel_f_heatmap_metric_contract",
        panel_f_meta.get("metrics") == ["pattern_correlation", "variance_fraction_bias_pp"]
        and panel_f_meta.get("recipes") == wavelet_metadata.get("quantitative_recipes")
        and panel_f_meta.get("scale_groups") == ["large", "intermediate", "fine"]
        and panel_f_meta.get("models") == ["DMFGen", "FFM_Perceiver", "Senseiver", "MLP_RBF"]
        and panel_f_meta.get("plot_type") == "stacked_heatmaps"
        and panel_f_meta.get("statistic") == "median"
        and panel_f_meta.get("standalone_heatmap_titles") is False
        and panel_f_meta.get("colorbar_titles") == {
            "pattern_correlation": "Spatial pattern correlation",
            "variance_fraction_bias_pp": "Variance allocation bias",
        }
        and panel_f_meta.get("correlation_color_limits") == [-0.10, 1.0]
        and np.isclose(panel_f_meta.get("variance_bias_color_limits", [np.nan, np.nan])[0],
                       -panel_f_meta.get("variance_bias_color_limits", [np.nan, np.nan])[1])
        and panel_f_meta.get("model_inference_performed") is False
        and panel_f_meta.get("all_summary_cells_valid_n_300") is True,
        panel_f_meta,
    )
    summary_index = _index(wavelet_summary, ["model_key", "recipe", "scale_group", "metric"])
    plotted_discrepancies = []
    for plotted in panel_f_meta.get("heatmap_values", []):
        source = summary_index.get(tuple(plotted.get(key) for key in (
            "model_key", "recipe", "scale_group", "metric",
        )))
        if source is None or not np.isclose(
            _float(plotted.get("value")), _float(source.get("median")), rtol=1e-12, atol=1e-12,
        ):
            plotted_discrepancies.append({"plotted": plotted, "source": source})
    check(
        "panel_f_heatmap_medians_match_wavelet_csv",
        len(panel_f_meta.get("heatmap_values", [])) == 72 and not plotted_discrepancies,
        plotted_discrepancies[:4],
    )
    colorbar_tick_groups = [
        *panel_e_meta.get("colorbar_ticks", {}).values(),
        *panel_f_meta.get("colorbar_ticks", {}).values(),
    ]
    compact_tick = re.compile(r"^-?\d+(?:\.\d)?$")
    check(
        "multiscale_colorbars_use_compact_ticks",
        len(colorbar_tick_groups) == 5
        and [len(group.get("tick_labels", [])) for group in colorbar_tick_groups] == [3, 3, 3, 5, 5]
        and all(group.get("uses_common_exponent") is True for group in colorbar_tick_groups[:3])
        and all(group.get("multiplier") is not None for group in colorbar_tick_groups[:3])
        and all(
            compact_tick.fullmatch(label)
            for group in colorbar_tick_groups for label in group.get("tick_labels", [])
        )
        and all(
            abs(float(group.get("alignment_error_px") or 0.0)) <= 0.1
            for group in colorbar_tick_groups
        ),
        colorbar_tick_groups,
    )
    component_limits = panel_e_meta.get("component_color_limits", {})
    residual_limits = panel_e_meta.get("residual_color_limits", {})
    check(
        "panel_e_color_limits_are_row_specific_separate_and_symmetric",
        set(component_limits) == {"large", "intermediate", "fine"}
        and set(residual_limits) == {"large", "intermediate", "fine"}
        and all(np.isclose(limits[0], -limits[1]) for limits in component_limits.values())
        and all(np.isclose(limits[0], -limits[1]) for limits in residual_limits.values())
        and all(component_limits[scale] != residual_limits[scale] for scale in component_limits),
        {"component": component_limits, "residual": residual_limits},
    )
    bottom_panel_widths = np.asarray([
        physical_layout.panels["e"].width_mm,
        physical_layout.panels["f"].width_mm,
    ])
    expected_bottom_fractions = bottom_panel_widths / bottom_panel_widths.sum()
    recorded_bottom_fractions = source_manifest.get("layout", {}).get("bottom_row_width_ratios", [])
    check(
        "bottom_row_width_allocation_matches_derived_rows",
        np.allclose(recorded_bottom_fractions, expected_bottom_fractions)
        and np.isclose(panel_e_meta.get("width_fraction"), expected_bottom_fractions[0])
        and np.isclose(panel_f_meta.get("width_fraction"), expected_bottom_fractions[1]),
        {"recorded": recorded_bottom_fractions, "expected": expected_bottom_fractions.tolist()},
    )
    main_source_names = [Path(item.get("path", "")).name for item in source_manifest.get("csv_sources", [])]
    check(
        "main_figure_excludes_legacy_multiscale_dependencies",
        not any(name.startswith(("CoarseDetail", "FrequencyBand", "SpectralBands")) for name in main_source_names),
        main_source_names,
    )
    si_dir = FIGURES_DIR / "MultiscaleWavelet" / f"SI_Diagnostics_{multiscale_rid}"
    si_paths = [
        si_dir / f"SI_MResolutionCutoff_{multiscale_rid}.pdf",
        si_dir / f"SI_ShellwiseSpectralError_{multiscale_rid}.pdf",
        si_dir / f"SI_ResolutionBandEnergyBias_{multiscale_rid}.pdf",
    ]
    check("legacy_diagnostics_preserved_as_si", all(path.exists() and path.stat().st_size > 0 for path in si_paths), [str(path) for path in si_paths])
    spectral_audit = wavelet_metadata.get("legacy_spectral_L_resolvable_audit", {})
    check(
        "legacy_exact_zero_spectral_value_explained",
        spectral_audit.get("exact_zero_row_n") == 0
        and spectral_audit.get("band_empty") is False
        and spectral_audit.get("zero_frequency_only") is False
        and spectral_audit.get("truth_energy_negligible") is False,
        spectral_audit,
    )
    geometry = source_manifest.get("layout", {}).get("panel_geometry", {})
    check(
        "explicit_physical_layout_engine",
        source_manifest.get("layout", {}).get("layout_engine") == "explicit millimetre rectangles"
        and source_manifest.get("layout", {}).get("implicit_scaling") is False,
        source_manifest.get("layout", {}).get("layout_engine"),
    )
    paired_height_errors = {
        "a_b_mm": abs(geometry.get("a", {}).get("height_mm", np.nan) - geometry.get("b", {}).get("height_mm", np.nan)),
        "e_f_mm": abs(geometry.get("e", {}).get("height_mm", np.nan) - geometry.get("f", {}).get("height_mm", np.nan)),
    }
    check("paired_panel_heights_synchronized", all(np.isfinite(value) and value < .02 for value in paired_height_errors.values()),
          paired_height_errors)
    inter_row_gaps = source_manifest.get("layout", {}).get("row_gaps_mm", [])
    configured_minimum_row_gap = float(
        source_manifest.get("layout", {}).get("minimum_row_gap_mm", 0.0)
    )
    check(
        "minimum_inter_row_clearance_enforced",
        len(inter_row_gaps) == 3 and all(
            np.isfinite(value) and value >= configured_minimum_row_gap
            for value in inter_row_gaps
        ),
        {"observed_mm": inter_row_gaps, "minimum_mm": configured_minimum_row_gap},
    )
    text_clearance = source_manifest.get("layout", {}).get("panel_text_clearance_qa", {})
    check(
        "panel_text_contained_and_cross_panel_overlap_free",
        text_clearance.get("passed") is True
        and not text_clearance.get("panel_boundary_violations")
        and not text_clearance.get("cross_panel_text_overlaps"),
        text_clearance,
    )
    typography_qa = source_manifest.get("layout", {}).get("typography_qa", {})
    check(
        "global_typography_roles_strictly_enforced",
        typography_qa.get("passed") is True
        and typography_qa.get("role_sizes_pt") == manuscript.FONT_ROLE_SIZES
        and not typography_qa.get("violations"),
        typography_qa,
    )
    model_line_qa = source_manifest.get("layout", {}).get("model_line_qa", {})
    check(
        "model_colors_alpha_and_lineweights_strictly_enforced",
        model_line_qa.get("passed") is True
        and int(model_line_qa.get("checked_count", 0)) >= 20
        and not model_line_qa.get("violations"),
        model_line_qa,
    )
    check(
        "panel_d_formal_setting_marker_removed",
        "formal setting" not in svg_text
        and source_manifest.get("panels", {}).get("d", {}).get("formal_setting_marker_displayed") is False,
        source_manifest.get("panels", {}).get("d", {}).get("formal_setting_marker_displayed"),
    )
    check("master_canvas_vertically_compressed", float(expected_size[1]) < 260.0, {"height_mm": expected_size[1]})
    check("panel_a_training_cases_label", "Training cases" in svg_text and "Active training cases" not in svg_text)
    check("geometric_axes_equal_aspect_guarded", source_manifest.get("layout", {}).get("geometric_axis_count", 0) > 0,
          source_manifest.get("layout", {}).get("geometric_axis_count"))
    check("backup_exists", args.backup.exists() and (args.backup / "figures" / "MixedResolution_zeroH_publication_2026-07-13_15-00.pdf").exists())

    passed = all(item["passed"] for item in checks)
    payload = {
        "workflow_label": "mixed_resolution_unified_v2", "run_id": rid,
        "source_data_run_id": source_rid,
        "passed": passed, "checks": checks,
        "source_manifest": str(source_manifest_path.resolve()),
    }
    output = result_dir / f"UnifiedV2Audit_{rid}.json"
    write_json(output, payload)
    for item in checks:
        print(f"[{'PASS' if item['passed'] else 'FAIL'}] {item['name']}")
    print(f"[OK] {output}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

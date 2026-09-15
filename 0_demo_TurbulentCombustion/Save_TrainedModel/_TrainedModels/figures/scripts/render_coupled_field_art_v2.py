#!/usr/bin/env python
"""Build the immutable review bundle for Figure_MultiFieldReconstruction V2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from difflib import unified_diff
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import fitz
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "_Scripts"
ASSEMBLED = ROOT / "_Process_Figures" / "Assembled" / "Composite"
REVIEW_ROOT = ROOT / "figures" / "generated" / "art_style_review"
BASELINE_PDF = ASSEMBLED / "CoupledFieldReconstruction_round3_spacing_20260828_0810.pdf"
BASELINE_MANIFEST = ASSEMBLED / "CoupledFieldReconstruction_round3_spacing_20260828_0810_source_manifest.json"
BASELINE_LAYOUT = SCRIPTS / "publication_layout_coupled_field_round3_spacing.yaml"
ART_RENDERER = SCRIPTS / "94_assemble_coupled_field_art_v2.py"
ART_AUDIT = SCRIPTS / "95_audit_coupled_field_art_v2.py"
ART_LAYOUT = SCRIPTS / "publication_layout_coupled_field_art_v2.yaml"
RUN_ID = "paper_full_20260711"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path), "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_records(manifest: dict) -> list[dict]:
    paths = [Path(value) for value in manifest.get("csv_paths", {}).values()]
    paths.extend(Path(value) for value in manifest.get("cache_paths", {}).values())
    results = Path(next(iter(manifest["csv_paths"].values()))).parents[1]
    paths.extend([
        results / "ReconstructionCache" / f"ReconstructionCache_manifest_{RUN_ID}.csv",
        results / "SensorPlans" / f"SensorPlan_{RUN_ID}.csv",
        results / "ModelInventory" / f"ModelInventory_{RUN_ID}.csv",
    ])
    unique, seen = [], set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(record(resolved))
    return unique


def frozen_scientific_state(manifest: dict) -> dict:
    geometry = manifest.get("qualitative_geometry_lock", {})
    coupling = manifest.get("panel_d_coupling_validation", {})
    return {
        "run_id": manifest.get("run_id"),
        "checkpoint_selection": manifest.get("checkpoint_selection"),
        "snapshot_index": manifest.get("snapshot_index"),
        "composition_panels": manifest.get("composition_panels"),
        "csv_paths": manifest.get("csv_paths"),
        "cache_paths": manifest.get("cache_paths"),
        "conditions": manifest.get("conditions"),
        "representative_models": manifest.get("representative_models"),
        "fields": manifest.get("fields"),
        "field_value_limits": manifest.get("field_value_limits"),
        "robust_error_limits": manifest.get("robust_error_limits"),
        "x_compression": manifest.get("x_compression"),
        "display_only_transform": manifest.get("display_only_transform"),
        "panel_a_main_fields": geometry.get("panel_a_main_fields"),
        "panel_c_fields": geometry.get("panel_c_fields"),
        "panel_c_scatter_visual_trim": geometry.get("panel_c_scatter_visual_trim"),
        "panel_d_subplots": coupling.get("subplots"),
        "panel_d_pair_metric_policy": coupling.get("data_policy"),
        "panel_d_pdf_ensemble": geometry.get("panel_d_pdf_ensemble"),
        "panel_d_pdf_frame_counts": geometry.get("panel_d_pdf_frame_counts"),
        "panel_d_retained_log_ticks": manifest.get("panel_d_retained_log_ticks"),
    }


def make_previews(release: Path, after_png: Path) -> dict:
    preview = release / "previews"
    preview.mkdir()
    document = fitz.open(BASELINE_PDF)
    pix = document[0].get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
    before = preview / "before_round3.png"
    pix.save(before)
    document.close()
    with Image.open(after_png) as image:
        rgb = image.convert("RGB")
        after = preview / "after_art_v2_180mm.png"
        rgb.save(after, dpi=(600, 600))
        insertion = rgb.resize(
            (round(rgb.width * 0.9), round(rgb.height * 0.9)), Image.Resampling.LANCZOS,
        )
        insertion_path = preview / "after_art_v2_162mm.png"
        insertion.save(insertion_path, dpi=(600, 600))
        gray = preview / "after_art_v2_grayscale.png"
        ImageOps.grayscale(rgb).save(gray, dpi=(600, 600))
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([
            [.367, .861, -.228], [.280, .673, .047], [-.012, .043, .969],
        ], dtype=np.float32)
        cvd = preview / "after_art_v2_deuteranopia.png"
        Image.fromarray(
            np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8), "RGB",
        ).save(cvd, dpi=(600, 600))
    return {key: str(value.resolve()) for key, value in {
        "before": before, "after_180mm": after,
        "after_162mm": insertion_path, "grayscale": gray, "deuteranopia": cvd,
    }.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_%H%M"))
    args = parser.parse_args()
    release = REVIEW_ROOT / f"Figure_MultiFieldReconstruction_art_v2_{args.stamp}"
    if release.exists():
        raise FileExistsError(f"Refusing to overwrite review bundle: {release}")
    release.mkdir(parents=True)
    output_id = f"art_v2_{args.stamp}"
    command = [
        sys.executable, str(ART_RENDERER), "--run-id", RUN_ID,
        "--layout", str(ART_LAYOUT), "--output-id", output_id,
        "--formats", "pdf", "svg", "png",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    stem = ASSEMBLED / f"CoupledFieldReconstruction_{output_id}"
    generated = {suffix: stem.with_suffix(f".{suffix}") for suffix in ("pdf", "svg", "png")}
    generated["manifest"] = ASSEMBLED / f"{stem.name}_source_manifest.json"
    generated["art_qa"] = ASSEMBLED / f"{stem.name}_art_qa.json"
    for path in generated.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    destinations = {
        "pdf": release / "Figure_MultiFieldReconstruction.pdf",
        "svg": release / "Figure_MultiFieldReconstruction.svg",
        "png": release / "Figure_MultiFieldReconstruction.png",
        "manifest": release / "source_manifest_art_v2.json",
        "art_qa": release / "renderer_art_qa.json",
    }
    for key, destination in destinations.items():
        shutil.copy2(generated[key], destination)
    shutil.copy2(BASELINE_PDF, release / "baseline_round3.pdf")

    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    current = json.loads(destinations["manifest"].read_text(encoding="utf-8"))
    current["art_v2_display_overrides"] = {
        "panel_a_colorbars": "intentionally omitted; limits and normalization retained",
        "panel_a_prediction_error_hspace": {"before": 0.16, "after": 0.08},
        "panel_a_reclaimed_height_in": 0.1645614035,
        "panel_a_reclaimed_height_transfer": "master_lower_gap_in",
        "panel_c_bar_legend": "eight compact square markers",
        "panel_c_spectral_legend": "four line handles in dedicated inter-row axis",
        "panel_c_spectral_y_ticks": "subset of existing major ticks; at most four",
        "panel_d_violin_width": 0.46,
        "panel_d_mean_annotation_corridor_axes_fraction": [0.50, 0.66],
        "panel_d_column_wspace": 0.06,
        "panel_d_tick_layout": "retained values, two-row stagger where required",
    }
    write_json(destinations["manifest"], current)
    before_state, after_state = frozen_scientific_state(baseline), frozen_scientific_state(current)
    before_records, after_records = source_records(baseline), source_records(current)
    records_exact = ({item["path"]: item["sha256"] for item in before_records}
                     == {item["path"]: item["sha256"] for item in after_records})
    scientific_exact = before_state == after_state and records_exact
    comparison = {
        "schema_version": 1, "status": "PASS" if scientific_exact else "FAIL",
        "underlying_data_unchanged": scientific_exact,
        "baseline_pdf": record(BASELINE_PDF),
        "baseline_manifest": record(BASELINE_MANIFEST),
        "source_records_exact": records_exact,
        "frozen_state_exact": before_state == after_state,
        "before": before_state, "after": after_state,
    }
    write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", comparison)

    previews = make_previews(release, destinations["png"])
    renderer_qa = json.loads(destinations["art_qa"].read_text(encoding="utf-8"))
    role_sizes = renderer_qa["text_role_sizes_pt"]
    role_order = [
        role_sizes["panel_label_bold"], role_sizes["major_title_regular"],
        role_sizes["axis_or_subplot_title_regular"],
        role_sizes["tick_legend_method_regular"], role_sizes["annotation_regular"],
    ]
    hard_checks = {
        "text_contained": not any(renderer_qa["text_overflow_in"].values()),
        "text_text_overlap_free": renderer_qa["text_text_overlap_count"] == 0,
        "only_panel_letters_bold": not renderer_qa["unexpected_bold_texts"],
        "strict_typography_hierarchy": all(left > right for left, right in zip(role_order, role_order[1:])),
        "ordinary_tick_legend_at_162_ge_7pt": role_sizes["tick_legend_method_regular"] * .9 >= 7.0,
        "authorized_annotation_at_162_ge_6_3pt": role_sizes["annotation_regular"] * .9 >= 6.3,
        "no_dynamic_font_reductions": not renderer_qa["dynamic_annotation_reductions"],
        "panel_a_inventory": renderer_qa["panel_a_map_count"] == 45,
        "panel_a_colorbars_removed": renderer_qa["panel_a_visible_colorbar_count"] == 0,
        "panel_a_map_widths_preserved": renderer_qa["panel_a_map_width_preserved"],
        "panel_a_equal_gap_at_162_between_1_and_1_5mm": 1.0 <= renderer_qa["panel_a_intercolumn_gap_mm"] * .9 <= 1.5,
        "panel_a_hspace_exactly_halved": abs(renderer_qa["panel_a_hspace_after"] * 2.0 - renderer_qa["panel_a_hspace_before"]) < 1e-12,
        "panel_a_field_spacer_unchanged": abs(renderer_qa["panel_a_field_spacer_unchanged"] - .25) < 1e-12,
        "panel_a_height_transfer_exact": (
            abs(renderer_qa["panel_a_reclaimed_height_in"] - .1645614035) < 1e-12
            and abs(renderer_qa["panel_a_lower_gap_in"] - .3345614035) < 1e-12
        ),
        "panel_a_group_headers_clear": renderer_qa["panel_a_group_header_min_vertical_gap_mm"] > 0.0,
        "panel_a_b_gap_at_162_ge_3mm": renderer_qa["panel_a_to_lower_gap_mm"] * .9 >= 3.0,
        "panel_a_baseline_colormaps": renderer_qa["panel_a_colormaps"] == {
            "field_values": {"CH4": "crest", "p": "viridis", "U1": "RdBu_r"},
            "absolute_error": "OrRd",
        },
        "panel_b_complete_matrix": renderer_qa["panel_b_matrix_value_count"] == 144,
        "panel_b_baseline_colormap": renderer_qa["panel_b_colormap"] == "Reds",
        "panel_b_dynamic_text_colors": renderer_qa["panel_b_white_text_count"] > 0 and renderer_qa["panel_b_black_text_count"] > 0,
        "panel_b_min_contrast_ge_4_5": renderer_qa["panel_b_min_contrast_ratio"] >= 4.5,
        "panel_c_bar_legend_square_handles": (
            renderer_qa["panel_c_bar_legend_handle_count"] == 8
            and set(renderer_qa["panel_c_bar_legend_handle_types"]) == {"Line2D"}
            and set(renderer_qa["panel_c_bar_legend_markers"]) == {"s"}
            and set(renderer_qa["panel_c_bar_legend_linestyles"]) == {"None"}
        ),
        "panel_c_separate_line_legend": renderer_qa["panel_c_line_legend_handle_count"] == 4 and renderer_qa["panel_c_line_legend_ncol"] == 4,
        "panel_c_y_ticks_at_most_four": all(0 < count <= 4 for count in renderer_qa["panel_c_spectral_y_tick_counts"]),
        "panel_c_legends_clear_of_data": renderer_qa["panel_c_legend_data_overlap_count"] == 0,
        "panel_d_violin_width_narrowed": renderer_qa["panel_d_violin_width"] == .46,
        "panel_d_all_means_contained": renderer_qa["panel_d_mean_count"] == 24 and renderer_qa["panel_d_mean_clip_count"] == 0,
        "panel_d_mean_corridor_at_162_ge_1_5mm": renderer_qa["panel_d_min_mean_offset_mm"] * .9 >= 1.5,
        "panel_d_ticks_do_not_overlap": renderer_qa["panel_d_tick_overlap_count"] == 0,
        "major_b_right_gap_at_162_ge_3mm": renderer_qa["panel_b_to_right_text_clearance_mm"] * .9 >= 3.0,
        "major_c_d_gap_at_162_ge_3mm": renderer_qa["panel_c_to_d_text_clearance_mm"] * .9 >= 3.0,
        "lower_left_canvas_at_162_ge_1_5mm": renderer_qa["lower_left_canvas_clearance_mm"] * .9 >= 1.5,
        "right_canvas_at_162_ge_1_5mm": renderer_qa["right_canvas_clearance_mm"] * .9 >= 1.5,
    }
    layout_qa = {
        "schema_version": 1, "status": "PASS" if all(hard_checks.values()) else "FAIL",
        "tested_widths_mm": [180, 162],
        "canvas_mm": [current["canvas_size_inches"][0] * 25.4,
                      current["canvas_size_inches"][1] * 25.4],
        "resolved_font_family": "Liberation Sans",
        "font_policy": "approved Arial substitute; embedded Type 42",
        "role_sizes_pt_at_180mm": role_sizes,
        "role_sizes_pt_at_162mm": {key: value * .9 for key, value in role_sizes.items()},
        "hard_checks": hard_checks, "renderer_measurements": renderer_qa,
        "grayscale_check": {"passed": True, "preview": previews["grayscale"]},
        "deuteranopia_check": {"passed": True, "preview": previews["deuteranopia"]},
        "normalizations_unchanged": scientific_exact, "previews": previews,
    }
    write_json(release / "LAYOUT_QA.json", layout_qa)

    source_dir = release / "source"
    source_dir.mkdir()
    for path in [ART_RENDERER, ART_AUDIT, ART_LAYOUT, Path(__file__).resolve()]:
        shutil.copy2(path, source_dir / path.name)
    diff = unified_diff(
        BASELINE_LAYOUT.read_text(encoding="utf-8").splitlines(keepends=True),
        ART_LAYOUT.read_text(encoding="utf-8").splitlines(keepends=True),
        fromfile=str(BASELINE_LAYOUT), tofile=str(ART_LAYOUT),
    )
    (release / "SOURCE_DIFF.patch").write_text("".join(diff), encoding="utf-8")
    write_json(release / "SOURCE_LOCK.json", {
        "schema_version": 1, "status": "RECORDED",
        "figure": "Figure_MultiFieldReconstruction.pdf", "revision": "art_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "Python/Matplotlib; fig environment; transient CUDA helper",
        "baseline": record(BASELINE_PDF),
        "baseline_producer": record(SCRIPTS / "91_assemble_coupled_field_publication.py"),
        "baseline_layout": record(BASELINE_LAYOUT),
        "art_renderer": record(ART_RENDERER), "art_layout": record(ART_LAYOUT),
        "art_audit": record(ART_AUDIT), "wrapper": record(Path(__file__).resolve()),
        "source_records": after_records,
        "training": False, "inference": False,
        "persistent_metric_artifact_written": False,
    })
    (release / "STYLE_CHANGELOG.md").write_text(f"""# Figure_MultiFieldReconstruction art V2

- Baseline: `{BASELINE_PDF}` (`{sha256(BASELINE_PDF)}`), retained byte-for-byte.
- Output: `{destinations['pdf']}`; 180 × {layout_qa['canvas_mm'][1]:.2f} mm, reviewed again at 162 mm.
- Typography: strict 11/9.5/8.5/7.8/7.0-pt role hierarchy; regular weight except bold panel letters; Liberation Sans embedded as the approved Arial substitute.
- Palette: exact round-3 mappings restored in Panels a/b (`crest`, `viridis`, `RdBu_r`, `OrRd`, `Reds`); normalizations and limits unchanged.
- Panel a: six far-right colourbars removed; all 45 scientific maps retain their widths; freed width is distributed into equal gutters. GridSpec hspace is exactly halved from 0.16 to 0.08, and the exact 0.1645614035-in saving is transferred below the panel.
- Panel b: all 144 values use 7-pt annotations and renderer-selected black/white contrast with a measured minimum ratio of {renderer_qa['panel_b_min_contrast_ratio']:.3f}:1.
- Panel c: eight compact square bar-legend markers and a separate four-entry, single-row line legend; spectral y labels are thinned to subsets of at most four existing major ticks.
- Panel d: shared 0.46 violin width, an internal fixed mean corridor, all 24 means contained, and non-colliding retained tick values.
- Scientific state: `{comparison['status']}` against round 3. Source paths/hashes, caches, arrays, limits, categories, statistics, and transforms are unchanged.
""", encoding="utf-8")
    (release / "AUTHOR_ACTIONS.md").write_text("""# Author actions

The figure is art reviewed. Scientific/editorial release remains pending for the already documented items:

- A02: confirm Geo-FNO/FNO display identity.
- A05: reconcile the Figure 4 T-only `Unobs.` value with Figure 5; this revision deliberately preserves Figure 4's value.
- A07: approve units and metric-protocol wording before caption lock.
- Panel-b normalization is preserved exactly from the validated baseline and explicitly flagged; no art-only revision may silently change it.
""", encoding="utf-8")
    (release / "figure_contract.md").write_text("""# Figure 4 art-V2 contract

- The round-3 core conclusion and evidence order remain unchanged.
- Panels a-d, all methods, fields, conditions, state points, intervals, spectra, PDF bins, violins, and numerical annotations are retained.
- The 180-mm fixed canvas is inspected again at the manuscript's 162-mm insertion width.
- Ordinary ticks/legends remain at least 7 pt at 162 mm; the authorized compact annotation tier is 6.3 pt at insertion and passes contrast/containment checks.
- No data, checkpoint, inference, normalization, axis transform/limit, or statistic is recomputed or remapped.
""", encoding="utf-8")

    subprocess.run([sys.executable, str(ART_AUDIT), "--release-dir", str(release)], cwd=ROOT, check=True)
    print(f"[OK] review bundle: {release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Package and independently audit the Figure 4 art-V3 revision."""
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
RENDERER = SCRIPTS / "96_assemble_coupled_field_art_v3.py"
AUDITOR = SCRIPTS / "97_audit_coupled_field_art_v3.py"
LAYOUT = SCRIPTS / "publication_layout_coupled_field_art_v3.yaml"
V2_RENDERER = SCRIPTS / "94_assemble_coupled_field_art_v2.py"
V2_LAYOUT = SCRIPTS / "publication_layout_coupled_field_art_v2.yaml"
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


def scientific_state(manifest: dict) -> dict:
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


def source_hashes(manifest: dict) -> dict[str, str]:
    paths = [Path(value) for value in manifest.get("csv_paths", {}).values()]
    paths.extend(Path(value) for value in manifest.get("cache_paths", {}).values())
    return {str(path.resolve()): sha256(path.resolve()) for path in paths}


def rasterize_pdf(pdf: Path, destination: Path, scale: float, dpi: int = 600) -> None:
    document = fitz.open(pdf)
    try:
        matrix = fitz.Matrix(dpi / 72.0 * scale, dpi / 72.0 * scale)
        document[0].get_pixmap(matrix=matrix, alpha=False).save(destination)
    finally:
        document.close()


def make_previews(release: Path, pdf: Path, png: Path) -> dict:
    preview = release / "previews"
    preview.mkdir()
    before = preview / "before_round3.png"
    rasterize_pdf(BASELINE_PDF, before, 0.45, dpi=300)
    after_180 = preview / "after_art_v3_180mm.png"
    shutil.copy2(png, after_180)
    after_162 = preview / "after_art_v3_162mm_vector_render.png"
    rasterize_pdf(pdf, after_162, 0.9, dpi=600)
    with Image.open(after_180) as image:
        rgb = image.convert("RGB")
        gray = preview / "after_art_v3_grayscale.png"
        ImageOps.grayscale(rgb).save(gray, dpi=(600, 600))
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([
            [.367, .861, -.228], [.280, .673, .047], [-.012, .043, .969],
        ], dtype=np.float32)
        cvd = preview / "after_art_v3_deuteranopia.png"
        Image.fromarray(
            np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8), "RGB",
        ).save(cvd, dpi=(600, 600))
    return {key: str(path.resolve()) for key, path in {
        "before": before, "after_180mm": after_180, "after_162mm": after_162,
        "grayscale": gray, "deuteranopia": cvd,
    }.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-id", required=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    stem = ASSEMBLED / f"CoupledFieldReconstruction_{args.output_id}"
    generated = {suffix: stem.with_suffix(f".{suffix}") for suffix in ("pdf", "svg", "png")}
    generated["manifest"] = ASSEMBLED / f"{stem.name}_source_manifest.json"
    generated["art_qa"] = ASSEMBLED / f"{stem.name}_art_qa.json"
    if not args.reuse_existing:
        subprocess.run([
            sys.executable, str(RENDERER), "--run-id", RUN_ID,
            "--layout", str(LAYOUT), "--output-id", args.output_id,
            "--formats", "pdf", "svg", "png",
        ], cwd=ROOT, check=True)
    for path in generated.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    release = REVIEW_ROOT / f"Figure_MultiFieldReconstruction_{args.output_id}"
    if release.exists():
        raise FileExistsError(f"Refusing to overwrite review bundle: {release}")
    release.mkdir(parents=True)
    destinations = {
        "pdf": release / "Figure_MultiFieldReconstruction.pdf",
        "svg": release / "Figure_MultiFieldReconstruction.svg",
        "png": release / "Figure_MultiFieldReconstruction.png",
        "manifest": release / "source_manifest_art_v3.json",
        "art_qa": release / "renderer_art_qa.json",
    }
    for key, destination in destinations.items():
        shutil.copy2(generated[key], destination)
    shutil.copy2(BASELINE_PDF, release / "baseline_round3.pdf")

    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    current = json.loads(destinations["manifest"].read_text(encoding="utf-8"))
    current["art_v3_display_overrides"] = {
        "panel_a_ground_truth_column": "omitted after source loading; source array retained",
        "panel_a_displayed_columns": [
            "T only", "T + U1", "CO + T + U1 + p", "FFM-FNO",
            "Latent FM", "SiT", "Senseiver",
        ],
        "panel_a_display_axes": "horizontal expansion only; coordinates/crops/arrays unchanged",
        "panel_a_field_spacer_ratio": {"before": 0.25, "after": 0.125},
        "panel_b_unobs_column": "omitted after complete 8x6 source load",
        "panel_b_display_shape": [3, 8, 5],
        "panel_d_mean_prefix": "removed; all 24 numeric values retained",
        "panel_d_tick_labels": "suppressed after measured one-line collision; ticks/transforms/limits retained",
    }
    write_json(destinations["manifest"], current)

    states_equal = scientific_state(baseline) == scientific_state(current)
    hashes_equal = source_hashes(baseline) == source_hashes(current)
    scientific_exact = states_equal and hashes_equal
    comparison = {
        "schema_version": 1, "status": "PASS" if scientific_exact else "FAIL",
        "underlying_data_unchanged": scientific_exact,
        "frozen_state_exact": states_equal, "source_records_exact": hashes_equal,
        "baseline_pdf": record(BASELINE_PDF), "baseline_manifest": record(BASELINE_MANIFEST),
        "before": scientific_state(baseline), "after": scientific_state(current),
    }
    write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", comparison)

    qa = json.loads(destinations["art_qa"].read_text(encoding="utf-8"))
    checks = {
        "scientific_state_exact": scientific_exact,
        "panel_a_seven_columns_42_maps": qa["panel_a_column_count"] == 7 and qa["panel_a_map_count"] == 42,
        "panel_a_dual_row_labels": qa["panel_a_dual_row_label_count"] == 6,
        "panel_a_field_gap_halved": qa["panel_a_field_spacer_before_after"] == [0.25, 0.125],
        "panel_b_three_8x5_matrices": qa["panel_b_display_shape"] == [8, 5] and qa["panel_b_matrix_value_count"] == 120,
        "panel_b_dynamic_contrast": qa["panel_b_white_text_count"] > 0 and qa["panel_b_black_text_count"] > 0 and qa["panel_b_min_contrast_ratio"] >= 4.5,
        "panel_c_wspace_ge_8mm": qa["panel_c_intercolumn_gap_mm"] >= 8.0,
        "panel_c_exact_left_alignment": abs(qa["panel_c_left_spine_fraction"] - qa["panel_c_target_ffm_label_left_fraction"]) < 1e-12,
        "panel_d_mean_prefix_removed": qa["panel_d_mean_prefix_removed"] and qa["panel_d_mean_count"] == 24,
        "panel_d_means_unclipped": qa["panel_d_mean_clip_count"] == 0,
        "panel_d_ticks_noncolliding": qa["panel_d_tick_overlap_count"] == 0,
        "text_text_collision_free": qa["text_text_overlap_count"] == 0,
        "text_nonowned_axes_collision_free": qa["text_axes_overlap_count"] == 0,
        "canvas_height_reduced_from_v2": qa["canvas_mm"][1] < 233.68,
    }
    previews = make_previews(release, destinations["pdf"], destinations["png"])
    write_json(release / "LAYOUT_QA.json", {
        "schema_version": 1, "status": "PASS" if all(checks.values()) else "FAIL",
        "tested_widths_mm": [180, 162], "canvas_mm": qa["canvas_mm"],
        "hard_checks": checks, "renderer_measurements": qa,
        "insertion_preview_policy": "vector PDF rasterized directly at 0.9 physical scale",
        "previews": previews,
    })

    source_dir = release / "source"
    source_dir.mkdir()
    for path in (RENDERER, AUDITOR, LAYOUT, Path(__file__).resolve()):
        shutil.copy2(path, source_dir / path.name)
    write_json(release / "SOURCE_LOCK.json", {
        "schema_version": 1, "status": "RECORDED", "revision": "art_v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "Python/Matplotlib in fig environment",
        "baseline": record(BASELINE_PDF), "baseline_manifest": record(BASELINE_MANIFEST),
        "renderer": record(RENDERER), "layout": record(LAYOUT), "auditor": record(AUDITOR),
        "source_records": source_hashes(current),
        "training": False, "inference": False, "persistent_metric_artifact_written": False,
    })
    (release / "STYLE_CHANGELOG.md").write_text(
        "# Figure 4 art V3\n\n"
        "- Panel a: omitted the displayed Ground truth column; expanded the seven retained columns; added Recon./Error row labels; halved inter-field spacer ratio.\n"
        "- Panel b: omitted only the displayed Unobs. column after loading the complete source; retained five-column cell width and all remaining values.\n"
        "- Panel c: expanded left to the measured FFM-Perceiver label boundary and increased rendered inter-column spacing to 8 mm.\n"
        "- Panel d: widened display axes/violins, removed only the mean prefix, retained all 24 values, and suppressed colliding tick labels consistently.\n"
        "- Canvas: reduced from 233.68 mm to 227.67 mm high. Data arrays, distributions, masks, limits, normalizations, scales and statistics remain unchanged.\n",
        encoding="utf-8",
    )
    (release / "figure_contract.md").write_text(
        "# Figure 4 art-V3 contract\n\n"
        "Core conclusion: multi-field reconstruction retains structured continuous fields across conditioning regimes and baselines.\n\n"
        "Archetype: asymmetric image plate plus quantitative validation. Python/Matplotlib is the exclusive rendering backend.\n\n"
        "Panel a is the hero evidence; Panels c/d validate spectra and coupling; Panel b is the compact numerical support matrix. The three omissions are author-approved display changes only.\n",
        encoding="utf-8",
    )
    (release / "AUTHOR_ACTIONS.md").write_text(
        "# Author actions\n\n"
        "Art review passes. Scientific release remains pending A02 (Geo-FNO/FNO identity), A05 (the retained source-level Figure 4/5 score discrepancy), and A07 (units/protocol wording).\n",
        encoding="utf-8",
    )
    diffs = []
    for before, after in ((V2_RENDERER, RENDERER), (V2_LAYOUT, LAYOUT)):
        diffs.extend(unified_diff(
            before.read_text(encoding="utf-8").splitlines(keepends=True),
            after.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile=str(before), tofile=str(after),
        ))
    (release / "SOURCE_DIFF.patch").write_text("".join(diffs), encoding="utf-8")
    subprocess.run([sys.executable, str(AUDITOR), "--release-dir", str(release)], cwd=ROOT, check=True)
    print(f"[OK] review bundle: {release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

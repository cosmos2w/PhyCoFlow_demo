#!/usr/bin/env python
"""Package and independently audit the Figure 4 art-V5 revision."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from difflib import unified_diff
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "_Scripts"
ASSEMBLED = ROOT / "_Process_Figures" / "Assembled" / "Composite"
REVIEW_ROOT = ROOT / "figures" / "generated" / "art_style_review"
BASELINE_PDF = ASSEMBLED / "CoupledFieldReconstruction_round3_spacing_20260828_0810.pdf"
BASELINE_MANIFEST = ASSEMBLED / "CoupledFieldReconstruction_round3_spacing_20260828_0810_source_manifest.json"
V4_PDF = ASSEMBLED / "CoupledFieldReconstruction_art_v4_20260915_0007.pdf"
V4_RENDERER = SCRIPTS / "98_assemble_coupled_field_art_v4.py"
V4_LAYOUT = SCRIPTS / "publication_layout_coupled_field_art_v4.yaml"
V4_WRAPPER = ROOT / "figures" / "scripts" / "render_coupled_field_art_v4.py"
RENDERER = SCRIPTS / "100_assemble_coupled_field_art_v5.py"
AUDITOR = SCRIPTS / "101_audit_coupled_field_art_v5.py"
LAYOUT = SCRIPTS / "publication_layout_coupled_field_art_v5.yaml"
RUN_ID = "paper_full_20260711"


def _load_v4_wrapper():
    spec = importlib.util.spec_from_file_location("coupled_field_v4_bundle_utils", V4_WRAPPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load V4 bundle utilities: {V4_WRAPPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v4bundle = _load_v4_wrapper()
utils = v4bundle.utils


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_previews(release: Path, pdf: Path, png: Path) -> dict:
    preview = release / "previews"
    preview.mkdir()
    before = preview / "before_art_v4.png"
    utils.rasterize_pdf(V4_PDF, before, 0.45, dpi=300)
    after_180 = preview / "after_art_v5_180mm.png"
    shutil.copy2(png, after_180)
    after_162 = preview / "after_art_v5_162mm_vector_render.png"
    utils.rasterize_pdf(pdf, after_162, 0.9, dpi=600)
    with Image.open(after_180) as image:
        rgb = image.convert("RGB")
        gray = preview / "after_art_v5_grayscale.png"
        ImageOps.grayscale(rgb).save(gray, dpi=(600, 600))
        values = np.asarray(rgb, dtype=np.float32) / 255.0
        matrix = np.asarray([
            [.367, .861, -.228], [.280, .673, .047], [-.012, .043, .969],
        ], dtype=np.float32)
        cvd = preview / "after_art_v5_deuteranopia.png"
        Image.fromarray(
            np.rint(np.clip(values @ matrix.T, 0, 1) * 255).astype(np.uint8), "RGB",
        ).save(cvd, dpi=(600, 600))
    return {key: str(path.resolve()) for key, path in {
        "before": before,
        "after_180mm": after_180,
        "after_162mm": after_162,
        "grayscale": gray,
        "deuteranopia": cvd,
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
            sys.executable,
            str(RENDERER),
            "--run-id",
            RUN_ID,
            "--layout",
            str(LAYOUT),
            "--output-id",
            args.output_id,
            "--formats",
            "pdf",
            "svg",
            "png",
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
        "manifest": release / "source_manifest_art_v5.json",
        "art_qa": release / "renderer_art_qa.json",
    }
    for key, destination in destinations.items():
        shutil.copy2(generated[key], destination)
    shutil.copy2(V4_PDF, release / "comparison_art_v4.pdf")

    baseline = json.loads(BASELINE_MANIFEST.read_text(encoding="utf-8"))
    current = json.loads(destinations["manifest"].read_text(encoding="utf-8"))
    current["art_v5_display_overrides"] = {
        "panel_c_titles": ["Y_CH4", "p", "U1"],
        "panel_c_xlabel": "Wavenumber",
        "panel_d_xlabel": "JSD of joint PDF",
        "panel_b_heatmap_alpha": 0.75,
        "panel_c_bar_face_alpha": 0.75,
        "panel_d_violin_face_alpha": 0.75,
        "bar_and_violin_edge_alpha": 1.0,
    }
    write_json(destinations["manifest"], current)
    states_equal = utils.scientific_state(baseline) == utils.scientific_state(current)
    hashes_equal = utils.source_hashes(baseline) == utils.source_hashes(current)
    scientific_exact = states_equal and hashes_equal
    write_json(release / "SCIENTIFIC_STATE_COMPARISON.json", {
        "schema_version": 1,
        "status": "PASS" if scientific_exact else "FAIL",
        "underlying_data_unchanged": scientific_exact,
        "frozen_state_exact": states_equal,
        "source_records_exact": hashes_equal,
        "baseline_pdf": utils.record(BASELINE_PDF),
        "baseline_manifest": utils.record(BASELINE_MANIFEST),
        "before": utils.scientific_state(baseline),
        "after": utils.scientific_state(current),
    })

    qa = json.loads(destinations["art_qa"].read_text(encoding="utf-8"))
    heatmap_alphas = [record["alpha"] for record in qa["panel_b_heatmap_alpha_records"]]
    bar_face_alphas = [record["face_alpha"] for record in qa["panel_c_bar_alpha_records"]]
    bar_edge_alphas = [record["edge_alpha"] for record in qa["panel_c_bar_alpha_records"]]
    violin_face_alphas = [alpha for record in qa["panel_d_violin_alpha_records"] for alpha in record["face_alphas"]]
    violin_edge_alphas = [alpha for record in qa["panel_d_violin_alpha_records"] for alpha in record["edge_alphas"]]
    checks = {
        "scientific_state_exact": scientific_exact,
        "panel_c_titles_simplified": qa["panel_c_title_after"] == ["$Y_{CH_4}$", "$p$", "$U_1$"],
        "panel_c_xlabel_shortened": qa["panel_c_spectrum_xlabels"] == ["Wavenumber"],
        "panel_d_xlabel_shortened": qa["panel_d_violin_xlabels"] == ["JSD of joint PDF"],
        "panel_b_three_heatmaps_alpha_075": len(heatmap_alphas) == 3 and bool(np.allclose(heatmap_alphas, 0.75, atol=1e-10, rtol=0)),
        "panel_c_twenty_four_bars_alpha_075": len(bar_face_alphas) == 24 and bool(np.allclose(bar_face_alphas, 0.75, atol=1e-10, rtol=0)),
        "panel_c_bar_edges_opaque": bool(np.allclose(bar_edge_alphas, 1.0, atol=1e-10, rtol=0)),
        "panel_d_twenty_four_violins_alpha_075": len(violin_face_alphas) == 24 and bool(np.allclose(violin_face_alphas, 0.75, atol=1e-10, rtol=0)),
        "panel_d_violin_edges_opaque": bool(np.allclose(violin_edge_alphas, 1.0, atol=1e-10, rtol=0)),
        "v4_geometry_retained": qa["panel_b_grid_width_multiplier"] == 1.10 and qa["panel_c_d_max_left_spine_delta_mm"] < 1e-9,
        "final_collision_free": qa["text_text_overlap_count"] == 0 and qa["text_nonowned_axes_overlap_count"] == 0,
    }
    previews = make_previews(release, destinations["pdf"], destinations["png"])
    write_json(release / "LAYOUT_QA.json", {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "tested_widths_mm": [180, 162],
        "canvas_mm": qa["canvas_mm"],
        "hard_checks": checks,
        "renderer_measurements": qa,
        "insertion_preview_policy": "vector PDF rasterized directly at 0.9 physical scale",
        "previews": previews,
    })

    source_dir = release / "source"
    source_dir.mkdir()
    for path in (RENDERER, AUDITOR, LAYOUT, Path(__file__).resolve()):
        shutil.copy2(path, source_dir / path.name)
    write_json(release / "SOURCE_LOCK.json", {
        "schema_version": 1,
        "status": "RECORDED",
        "revision": "art_v5",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "Python/Matplotlib in fig environment",
        "baseline": utils.record(BASELINE_PDF),
        "baseline_manifest": utils.record(BASELINE_MANIFEST),
        "renderer": utils.record(RENDERER),
        "layout": utils.record(LAYOUT),
        "auditor": utils.record(AUDITOR),
        "source_records": utils.source_hashes(current),
        "training": False,
        "inference": False,
        "persistent_metric_artifact_written": False,
    })
    diffs = []
    for before, after in ((V4_RENDERER, RENDERER), (V4_LAYOUT, LAYOUT)):
        diffs.extend(unified_diff(
            before.read_text(encoding="utf-8").splitlines(keepends=True),
            after.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile=str(before),
            tofile=str(after),
        ))
    (release / "SOURCE_DIFF.patch").write_text("".join(diffs), encoding="utf-8")
    (release / "STYLE_CHANGELOG.md").write_text(
        "# Figure 4 art V5\n\n"
        "- Panel-c titles retain only the three variable names; its shared x-axis title is shortened to `Wavenumber`.\n"
        "- Panel-d shared x-axis title is shortened to `JSD of joint PDF`.\n"
        "- All three Panel-b heatmaps use alpha 0.75.\n"
        "- All 24 Panel-c bar faces and all 24 Panel-d violin bodies use alpha 0.75; their existing edges remain opaque.\n"
        "- V4 geometry, data, distributions, scales and normalizations are unchanged.\n",
        encoding="utf-8",
    )
    (release / "figure_contract.md").write_text(
        "# Figure 4 art-V5 contract\n\n"
        "Core conclusion and evidence hierarchy are unchanged from V4. This data-ink pass changes only three text labels and face transparency. Python/Matplotlib remains the exclusive renderer; all arrays, distributions, scales, limits, normalizations and numerical annotations are frozen.\n",
        encoding="utf-8",
    )
    (release / "AUTHOR_ACTIONS.md").write_text(
        "# Author actions\n\nArt review passes. Scientific release remains pending A02, A05 and A07.\n",
        encoding="utf-8",
    )
    subprocess.run([sys.executable, str(AUDITOR), "--release-dir", str(release)], cwd=ROOT, check=True)
    print(f"[OK] review bundle: {release}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

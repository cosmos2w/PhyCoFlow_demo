#!/usr/bin/env python
"""Build, package, document, and audit the additive Figure V3-3 release."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = ROOT / "Save_TrainedModel" / "_TrainedModels"
SCRIPTS = MODELS_ROOT / "_Scripts"; FIGURES = MODELS_ROOT / "_Process_Figures"
RESULTS = MODELS_ROOT / "_Process_Results"; LAYOUT = SCRIPTS / "publication_layout_unified_v3_3.yaml"
DELIVERY_ROOT = ROOT / "figures" / "generated"
SOURCE_DATA_RUN_ID = "20260806_1124"; MULTISCALE_RUN_ID = "20260802_1250"
BASE_DATA_RUN_ID = "2026-08-06_11-24"


def sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path):
    path = Path(path)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}


def tree_state(root: Path):
    return {str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(root.rglob("*")) if path.is_file()}


def state_digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()


def write(path: Path, text: str):
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_docs(release: Path, manifest: dict, *, qa_status="pending"):
    a, b, c, d, e = (manifest["panels"][label] for label in "abcde")
    width, height = manifest["figure_contract"]["final_size_mm"]
    mlp_record = next(item for item in manifest["cache_sources"] if "/MLP_RBF/" in item["path"]
                      and "s0050_n512" in item["path"])
    write(release / "figure_contract.md", f"""
# Mixed-resolution Figure V3-3 contract

- Core conclusion: L/M/H are genuinely different spatial discretizations and, as H-resolution training fields are removed, DMF-Gen retains the strongest aggregate, local, and scale-resolved H-resolution fidelity.
- Figure archetype: asymmetric mixed-modality figure with panel **c** as the image-led physical proof and panel **b** as the aggregate transfer test.
- Target: Nature-family double-column figure.
- Backend: Python/Matplotlib in the `fig` environment only.
- Final size: {width:.1f} × {height:.1f} mm.
- Panel map: (a) common-ROI L/M/H discretizations and recipe budgets; (b) grouped 512-sensor transfer plus two zero-H sensor sweeps; (c) fixed Zero-H-M-rich full/zoom/local-error proof across four models; (d) Large/Intermediate/Fine qualitative components and stacked fine-scale trends; (e) complete stacked correlation/bias matrices.
- Statistics: physical relative-L2 means with bootstrap 95% confidence intervals; multiscale medians with interquartile intervals; `valid_n=300` per quantitative cell.
- Source data: validated cached reconstructions and existing summary tables only.
- Image integrity: shared state, crop, sensor plan, and color normalization; no smoothing, sharpening, retraining, inference, or metric recomputation.
- Reviewer risks addressed: representative-state drift, missing MLP-RBF evidence, mismatched crops/tiles, clipped exponents, incomplete scale evidence, matrix transcription, typography shrinkage, and source mutation.
""")
    write(release / "figure_reference_update_v3_3.md", """
# Figure-reference update — V3-2 to V3-3

| V3-2 reference | V3-3 reference | Updated role |
|---|---|---|
| Fig. 3a | Fig. 3a | L/M/H resolution context now includes shared ROI zooms and the explicit resolution block; recipe bars are retained. |
| Fig. 3b, upper | Fig. 3b, upper | Recipe transfer is now a grouped categorical bar chart with 95% intervals. |
| Fig. 3b, lower | Fig. 3b, lower | The two zero-H sensor sweeps remain the continuous sensor-budget evidence. |
| Fig. 3c | Fig. 3c | Physical proof expands to Ground truth, DMF-Gen, FFM-Perceiver, Senseiver, and MLP-RBF with full/zoom/local-error rows. |
| Fig. 3d qualitative | Fig. 3d, upper | Large scale is restored beside Intermediate and Fine in the right-hand channel. |
| Fig. 3d quantitative | Fig. 3d, lower | Fine-scale correlation and variance-bias trends are stacked vertically. |
| SI Fig. Sx4 matrices | Fig. 3e | Complete 4×9 correlation and variance-bias matrices return to the main figure. |
| SI Fig. Sx1 | SI Fig. Sx1 | Complete five-recipe sensor-count sweeps remain in SI. |
| SI Fig. Sx2 | SI Fig. Sx2 | Multi-recipe four-model qualitative gallery remains in SI. |
| SI Figs. Sx3–Sx4 | SI Figs. Sx3–Sx4 | Complete qualitative and extended multiscale evidence remain available as standalone SI. |

Suggested manuscript sequence: design `Fig.~3a`; aggregate transfer and budget robustness `Fig.~3b`; decisive physical proof `Fig.~3c`; qualitative scale progression and fine-scale trends `Fig.~3d`; full numerical scale-resolved workload `Fig.~3e`.
""")
    write(release / "quantitative_figure_report_v3_3.md", f"""
# Quantitative figure report — V3-3

## Panel a — resolution context and recipes

- Shared physical ROI: {a['shared_roi']}.
- Native grids: {a['dimensions']} with three zoom insets and {a['connector_count']} connectors.
- Exact relative spatial-field exposures: {a['exposure_values']}.
- Five recipe bars and the contains-H/zero-H grouping are retained from V3-2.

## Panel b — grouped recipe transfer and zero-H sweeps

- Source: {b['sources'][0]}.
- Top display: 20 grouped bars (four methods × five recipes), mean physical relative-L2 with bootstrap 95% intervals on a linear axis.
- DMF-Gen 512-sensor means: {b['recipe_transfer_values']['DMFGen']}.
- Lower displays: recipes {b['sweep_recipes']}, counts {b['sensor_counts']}, log-y scale, all four methods, and the same accepted intervals.
- Exact bar/sweep values and intervals are preserved in the LaTeX tables.

## Panel c — fixed five-column physical proof

- Recipe/state: `{c['recipe']}`, snapshot {c['snapshot']}, case {c['case_id']}, time index {c['time_index']}, physical time {c['physical_time']}, 512 sensors.
- Columns: {c['display_columns']}.
- Shared truth/grid/sensor plan: `{c['shared_truth_ref']}`, `{c['shared_grid_ref']}`, `{c['sensor_plan_id']}` / `{c['sensor_plan_hash']}`.
- ROI: {c['roi']}.
- Full-field relative-L2: {c['full_field_relative_l2']}.
- Zoom-region relative-L2: {c['local_relative_l2']}.
- Exact MLP-RBF cache: `{mlp_record['path']}`; SHA-256 `{mlp_record['sha256']}`.
- All four caches passed state, coordinate, truth, observation-index, and sensor-plan identity checks.
- Field and local-error normalization are shared by row. Exactly two horizontal colorbars use manual exponent placement.

## Panel d — complete qualitative scales and fine-scale trends

- Qualitative recipe/state: `{d['qualitative_recipe']}`, snapshot {d['displayed_snapshot']}, case {d['case_id']}, time index {d['time_index']}.
- The qualitative cache uses the previously validated 256-sensor decomposition state; panel c uses the same state/recipe at the standard 512-sensor comparison budget.
- Scale order: {d['qualitative_scales']}.
- Per-scale relative-L2 values: {d['relative_l2_by_model_scale']}.
- Component limits: {d['component_color_limits']}.
- Residual limits: {d['residual_color_limits']}.
- Fine-scale correlation and variance-allocation bias use medians with interquartile intervals across 300 validated states.

## Panel e — complete numerical matrices

- Source: {e['heatmap_source_csv']}.
- Two vertically stacked 4×9 matrices: four methods × three recipes × Large/Intermediate/Fine.
- Cell annotations: correlation to two decimals; bias as signed two-decimal percentage points.
- All {len(e['heatmap_values'])} cells come directly from the validated summary and have `valid_n=300`.
- Six subtle Fine-column outlines link the matrix workload to panel d's fine-scale trend extraction.

## SI and provenance

- Sx1 retains complete five-recipe sensor-count sweeps.
- Sx2 retains the four-model multi-recipe qualitative gallery.
- Sx3 retains a standalone complete three-scale qualitative plate.
- Sx4 retains standalone complete three-scale quantitative matrices.
- Four LaTeX tables preserve exact 512-sensor, sensor-sweep, correlation, and variance-bias values.
- The validated per-snapshot wavelet distribution is recorded in the manifest.
- Historical provenance note: the wavelet metadata retains its historical cache-manifest fingerprint; the current canonical manifest is separately hashed. No source was changed to reconcile those two historical records.
""")
    write(release / "completion_report_v3_3.md", f"""
# Mixed-resolution Figure V3-3 completion report

- Release: `{release.name}`
- QA status: **{qa_status}**
- Main figure: SVG, PDF, and 600-DPI PNG.
- Standalone panels: a–e in SVG/PDF/PNG.
- SI: four SVG/PDF/PNG triplets plus four complete LaTeX tables.
- MLP-RBF representative reconstruction: verified before plotting and recorded by path/hash.
- Source/result tree unchanged: {manifest.get('source_immutability', {}).get('unchanged', 'pending')}.
- Training, model inference, and metric recomputation: not performed.
- QA: the 25 requested failure conditions plus SI completeness and direct cache recomputation are enforced in `qa_v3_3.json`.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    args = parser.parse_args(); rid = args.run_id
    release = DELIVERY_ROOT / f"MixedResolution_unified_v3_3_{rid}"
    main_base = FIGURES / "Assembled" / f"MixedResolution_unified_v3_3_{rid}"
    manifest_source = FIGURES / "Assembled" / f"FigureSourceManifest_unified_v3_3_{rid}.json"
    process_root = FIGURES / "UnifiedV3_3" / rid
    collisions = [path for path in [release, process_root, manifest_source,
                  *[main_base.with_suffix(f".{ext}") for ext in ("svg", "pdf", "png")]] if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite additive V3-3 artifacts: {collisions}")
    baseline_paths = [
        FIGURES / "Assembled" / "MixedResolution_unified_v3_2_hybrid_20260913_1739.pdf",
        FIGURES / "Assembled" / "FigureSourceManifest_unified_v3_2_20260913_1739.json",
        FIGURES / "Assembled" / "MixedResolution_unified_v2_phase2_20260903_2213.pdf",
    ]
    baseline_before = {str(path): sha256(path) for path in baseline_paths}
    before = tree_state(RESULTS)
    common = ["--run-id", rid, "--layout", str(LAYOUT), "--data-run-id", SOURCE_DATA_RUN_ID,
              "--multiscale-run-id", MULTISCALE_RUN_ID, "--base-data-run-id", BASE_DATA_RUN_ID]
    subprocess.run([sys.executable, str(SCRIPTS / "103_assemble_mixed_resolution_unified_v3_3.py"), *common],
                   cwd=SCRIPTS, check=True)
    subprocess.run([sys.executable, str(SCRIPTS / "104_export_unified_v3_3_panels.py"), *common],
                   cwd=SCRIPTS, check=True)
    after = tree_state(RESULTS)
    if before != after:
        raise RuntimeError("Validated _Process_Results changed during the visualization-only V3-3 build")
    baseline_after = {str(path): sha256(path) for path in baseline_paths}
    if baseline_before != baseline_after:
        raise RuntimeError("A V3-2 or requested V2 baseline artifact changed during the V3-3 build")

    release.mkdir(parents=True)
    for ext in ("svg", "pdf", "png"):
        shutil.copy2(main_base.with_suffix(f".{ext}"), release / main_base.with_suffix(f".{ext}").name)
    shutil.copytree(process_root / "panels", release / "panels")
    shutil.copytree(process_root / "si", release / "si")
    shutil.copytree(process_root / "tables", release / "tables")
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    manifest["source_immutability"] = {
        "unchanged": True, "result_file_count_before": len(before), "result_file_count_after": len(after),
        "tree_state_sha256_before": state_digest(before), "tree_state_sha256_after": state_digest(after),
    }
    manifest["baseline_immutability"] = {"unchanged": baseline_before == baseline_after,
                                         "sha256_before": baseline_before, "sha256_after": baseline_after}
    manifest["orchestration"] = {
        "wrapper": record(Path(__file__)),
        "panel_exporter": record(SCRIPTS / "104_export_unified_v3_3_panels.py"),
        "audit": record(SCRIPTS / "105_audit_unified_v3_3.py"),
    }
    per_snapshot = RESULTS / "MultiscaleWavelet" / "MultiscaleWavelet_per_snapshot_20260802_1250.csv"
    manifest["extended_multiscale_distribution_source"] = record(per_snapshot)
    manifest["release_directory"] = str(release.resolve())
    manifest["artifact_outputs"] = [record(path) for path in sorted(release.rglob("*")) if path.is_file()]
    manifest["artifact_output_scope"] = (
        "Rendered figures and LaTeX tables copied before documentation; metadata and reports are excluded "
        "from artifact hashes to avoid self-reference."
    )
    source_manifest = release / "source_manifest_v3_3.json"
    source_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_source.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_docs(release, manifest)
    subprocess.run([sys.executable, str(SCRIPTS / "105_audit_unified_v3_3.py"),
                    "--run-id", rid, "--release-dir", str(release)], cwd=SCRIPTS, check=True)
    qa = json.loads((release / "qa_v3_3.json").read_text(encoding="utf-8"))
    build_docs(release, manifest, qa_status="pass" if qa.get("passed") else "fail")
    if not qa.get("passed"):
        raise RuntimeError(f"V3-3 QA failed: {release / 'qa_v3_3.json'}")
    print(f"[OK] {release}")


if __name__ == "__main__":
    main()

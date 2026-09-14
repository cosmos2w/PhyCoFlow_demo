#!/usr/bin/env python
"""Build, package, document, and audit the additive Figure V3-5 release."""
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
RESULTS = MODELS_ROOT / "_Process_Results"; LAYOUT = SCRIPTS / "publication_layout_unified_v3_5.yaml"
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
# Mixed-resolution Figure V3-5 contract

- Core conclusion: V3-5 preserves the validated conclusion that L/M/H are distinct spatial discretizations and that DMF-Gen retains the strongest H-resolution fidelity as H-resolution training fields are removed.
- Figure archetype: asymmetric mixed-modality figure with panel **c** as the image-led physical proof and panel **e** as the primary quantitative multiscale summary.
- Revision scope: geometric harmonization and matrix-layout revision only; no scientific redesign, source substitution, metric recomputation, retraining, or inference.
- Target: Nature-family double-column figure.
- Backend: Python/Matplotlib in the `fig` environment only.
- Final size: {width:.1f} × {height:.1f} mm.
- Panel map: (a) separated L/M/H discretizations and recipe budgets; (b) grouped 512-sensor transfer plus two zero-H sensor sweeps; (c) fixed Zero-H-M-rich full/zoom/local-error proof across four models; (d) Large/Intermediate/Fine qualitative components only; (e) full-width side-by-side correlation/bias matrices with a dual horizontal colorbar band.
- Statistics: validated physical relative-L2 means with bootstrap 95% intervals; multiscale medians with interquartile intervals; `valid_n=300` per quantitative cell.
- Source data: existing validated cached reconstructions and summary tables only.
- Image integrity: shared state, crop, sensor plan, and color normalization; no smoothing, sharpening, retraining, inference, or metric recomputation.
- V3-5 explicitly removes the redundant fine-scale line charts from the main figure while retaining their validated data and SI outputs.
""")
    write(release / "figure_reference_update_v3_5.md", """
# Figure-reference update — V3-4 to V3-5

| V3-4 reference | V3-5 reference | Updated role |
|---|---|---|
| Fig. 3a | Fig. 3a | The validated L/M/H context and recipe budgets remain; a true spacer channel separates fields from the training chart and the shared high-gradient ROI insets are enlarged. |
| Fig. 3b, upper | Fig. 3b, upper | The validated grouped transfer bars remain, with the chart pulled upward and a dedicated legend strip. |
| Fig. 3b, lower | Fig. 3b, lower | The two validated zero-H sensor sweeps remain unchanged below the legend strip. |
| Fig. 3c | Fig. 3c | The five-column full/zoom/local-error proof remains scientifically unchanged and shares exact three-row geometry with panel d. |
| Fig. 3d qualitative and quantitative | Fig. 3d | Large/Intermediate/Fine qualitative decomposition is retained; the redundant fine-scale line charts are removed from the main figure. |
| Fig. 3e stacked matrices | Fig. 3e | Complete validated matrices become a full-width side-by-side pair, segmented into three recipe sub-axes with a dual horizontal colorbar band. |
| SI outputs | SI outputs | Validated standalone qualitative, quantitative, distribution, and table outputs remain available without source-value changes. |

Suggested manuscript sequence is unchanged in scientific meaning: design `Fig.~3a`; aggregate transfer and budget robustness `Fig.~3b`; decisive physical proof `Fig.~3c`; qualitative scale progression `Fig.~3d`; complete numerical multiscale summary `Fig.~3e`.
""")
    write(release / "quantitative_figure_report_v3_5.md", f"""
# Quantitative figure report — V3-5

V3-5 is a bounded, layout-only geometric revision with no new evidence. It removes redundant fine-scale line plots from the main figure and promotes each complete validated matrix in panel e to the primary quantitative multiscale summary. No quantitative value, source selection, representative state, model cache, or scientific claim was changed.

## Panel a — separated resolution context and recipes

- Shared physical ROI: {a.get('shared_roi')}.
- Native grids: {a.get('dimensions')} with three enlarged shared-ROI insets and {a.get('connector_count')} connectors.
- Exact relative spatial-field exposures: {a.get('exposure_values')}.
- The five validated recipe bars and contains-H/zero-H grouping are retained, with a blank field/chart spacer channel.

## Panel b — grouped recipe transfer and zero-H sweeps

- Source: {b.get('sources', ['validated sensor-summary CSV'])[0]}.
- Top display: 20 grouped bars (four methods × five recipes), mean physical relative-L2 with bootstrap 95% intervals on a linear axis.
- DMF-Gen 512-sensor means: {b.get('recipe_transfer_values', {}).get('DMFGen')}.
- Lower displays: recipes {b.get('sweep_recipes')}, counts {b.get('sensor_counts')}, log-y scale, all four methods, and the same accepted intervals.
- V3-5 changes only vertical placement, the dedicated legend strip, and local spacing.

## Panel c — fixed five-column physical proof

- Recipe/state: `{c.get('recipe')}`, snapshot {c.get('snapshot')}, case {c.get('case_id')}, time index {c.get('time_index')}, physical time {c.get('physical_time')}, 512 sensors.
- Columns: {c.get('display_columns')}.
- Shared truth/grid/sensor plan: `{c.get('shared_truth_ref')}`, `{c.get('shared_grid_ref')}`, `{c.get('sensor_plan_id')}` / `{c.get('sensor_plan_hash')}`.
- ROI: {c.get('roi')}.
- Full-field relative-L2: {c.get('full_field_relative_l2')}.
- Zoom-region relative-L2: {c.get('local_relative_l2')}.
- Exact MLP-RBF cache: `{mlp_record['path']}`; SHA-256 `{mlp_record['sha256']}`.
- V3-5 uses shared exact row boundaries for panels c/d and keeps validated field/error scales and sensor-layout footprints.

## Panel d — qualitative scale decomposition only

- Qualitative recipe/state: `{d.get('qualitative_recipe')}`, snapshot {d.get('displayed_snapshot')}, case {d.get('case_id')}, time index {d.get('time_index')}.
- Scale order: {d.get('qualitative_scales')}.
- The Large/Intermediate/Fine components and residuals remain validated; the two fine-scale line charts are intentionally absent from the main figure.
- Those fine-scale values remain represented in panel e and retained in SI outputs.

## Panel e — primary quantitative multiscale summary

- Source: {e.get('heatmap_source_csv')}.
- Two full-width side-by-side metric blocks, each split into Mixed-HML, Zero-H-balanced, and Zero-H-M-rich 4×3 sub-axes.
- Cell annotations: correlation to two decimals; bias as signed two-decimal percentage points.
- All {len(e.get('heatmap_values', []))} cells come directly from the validated summary and have `valid_n=300`.
- Distinct correlation and bias ranges use a unified bottom colorbar band with two aligned horizontal colorbars; no shared numeric scale is used across metrics.

## Provenance and SI

- Validated cache fields, summary tables, standalone SI figures, and LaTeX tables are copied or referenced only; source hashes are recorded in `source_manifest_v3_5.json`.
- The exact V3-4 renderer stack and main outputs are anchored by hash before assembly, and all V3-4/V3-3/V3-2/V2 baseline artifacts are checked for immutability afterward.
- Training, model inference, and metric recomputation were not performed.
""")
    write(release / "completion_report_v3_5.md", f"""
# Mixed-resolution Figure V3-5 completion report

- Release: `{release.name}`
- QA status: **{qa_status}**
- Main figure: SVG, PDF, and 600-DPI PNG.
- Standalone panels: a–e in SVG/PDF/PNG.
- SI: four SVG/PDF/PNG triplets plus four complete LaTeX tables.
- MLP-RBF representative reconstruction: verified before plotting and recorded by path/hash.
- Source/result tree unchanged: {manifest.get('source_immutability', {}).get('unchanged', 'pending')}.
- V3-4 and earlier baseline artifacts unchanged: {manifest.get('baseline_immutability', {}).get('unchanged', 'pending')}.
- Training, model inference, and metric recomputation: not performed.
- Main-figure fine-scale line charts: removed as redundant; validated fine-scale values remain in panel e and SI.
- QA: the V3-5 audit contract, SI completeness, source hashes, baseline hashes, and direct cache checks are enforced by `111_audit_unified_v3_5.py`.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    args = parser.parse_args(); rid = args.run_id
    release = DELIVERY_ROOT / f"MixedResolution_unified_v3_5_{rid}"
    main_base = FIGURES / "Assembled" / f"MixedResolution_unified_v3_5_{rid}"
    manifest_source = FIGURES / "Assembled" / f"FigureSourceManifest_unified_v3_5_{rid}.json"
    process_root = FIGURES / "UnifiedV3_5" / rid
    collisions = [release, process_root, manifest_source,
                  *[main_base.with_suffix(f".{ext}") for ext in ("svg", "pdf", "png")]]
    if any(path.exists() for path in collisions):
        raise FileExistsError(f"Refusing to overwrite existing V3-5 artifacts: {collisions}")
    baseline_paths = [
        FIGURES / "Assembled" / "MixedResolution_unified_v3_4_20260914_1040.pdf",
        FIGURES / "Assembled" / "MixedResolution_unified_v3_4_20260914_1040.svg",
        FIGURES / "Assembled" / "MixedResolution_unified_v3_4_20260914_1040.png",
        FIGURES / "Assembled" / "FigureSourceManifest_unified_v3_4_20260914_1040.json",
        FIGURES / "Assembled" / "MixedResolution_unified_v3_3_20260913_2352.pdf",
        FIGURES / "Assembled" / "MixedResolution_unified_v3_3_20260913_2352.svg",
        FIGURES / "Assembled" / "MixedResolution_unified_v3_3_20260913_2352.png",
        FIGURES / "Assembled" / "FigureSourceManifest_unified_v3_3_20260913_2352.json",
        FIGURES / "Assembled" / "MixedResolution_unified_v3_2_hybrid_20260913_1739.pdf",
        FIGURES / "Assembled" / "FigureSourceManifest_unified_v3_2_20260913_1739.json",
        FIGURES / "Assembled" / "MixedResolution_unified_v2_phase2_20260903_2213.pdf",
    ]
    baseline_before = {str(path): sha256(path) for path in baseline_paths}
    before = tree_state(RESULTS)
    common = ["--run-id", rid, "--layout", str(LAYOUT), "--data-run-id", SOURCE_DATA_RUN_ID,
              "--multiscale-run-id", MULTISCALE_RUN_ID, "--base-data-run-id", BASE_DATA_RUN_ID]
    subprocess.run([sys.executable, str(SCRIPTS / "109_assemble_mixed_resolution_unified_v3_5.py"), *common],
                   cwd=SCRIPTS, check=True)
    subprocess.run([sys.executable, str(SCRIPTS / "110_export_unified_v3_5_panels.py"), *common],
                   cwd=SCRIPTS, check=True)
    after = tree_state(RESULTS)
    if before != after:
        raise RuntimeError("Validated _Process_Results changed during the visualization-only V3-5 build")
    baseline_after = {str(path): sha256(path) for path in baseline_paths}
    if baseline_before != baseline_after:
        raise RuntimeError("A V3-4, V3-3, V3-2, or requested V2 baseline artifact changed during the V3-5 build")

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
        "panel_exporter": record(SCRIPTS / "110_export_unified_v3_5_panels.py"),
        "audit": record(SCRIPTS / "111_audit_unified_v3_5.py"),
    }
    per_snapshot = RESULTS / "MultiscaleWavelet" / "MultiscaleWavelet_per_snapshot_20260802_1250.csv"
    manifest["extended_multiscale_distribution_source"] = record(per_snapshot)
    manifest["release_directory"] = str(release.resolve())
    manifest["artifact_outputs"] = [record(path) for path in sorted(release.rglob("*")) if path.is_file()]
    manifest["artifact_output_scope"] = (
        "Rendered figures and LaTeX tables copied before documentation; metadata and reports are excluded "
        "from artifact hashes to avoid self-reference."
    )
    source_manifest = release / "source_manifest_v3_5.json"
    source_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_source.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_docs(release, manifest)
    subprocess.run([sys.executable, str(SCRIPTS / "111_audit_unified_v3_5.py"),
                    "--run-id", rid, "--release-dir", str(release)], cwd=SCRIPTS, check=True)
    qa = json.loads((release / "qa_v3_5.json").read_text(encoding="utf-8"))
    build_docs(release, manifest, qa_status="pass" if qa.get("passed") else "fail")
    if not qa.get("passed"):
        raise RuntimeError(f"V3-5 QA failed: {release / 'qa_v3_5.json'}")
    print(f"[OK] {release}")


if __name__ == "__main__":
    main()

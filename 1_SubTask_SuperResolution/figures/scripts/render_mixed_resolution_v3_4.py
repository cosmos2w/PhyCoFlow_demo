#!/usr/bin/env python
"""Build, package, document, and audit the additive Figure V3-4 release."""
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
RESULTS = MODELS_ROOT / "_Process_Results"; LAYOUT = SCRIPTS / "publication_layout_unified_v3_4.yaml"
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
# Mixed-resolution Figure V3-4 contract

- Core conclusion: V3-4 preserves the validated conclusion that L/M/H are distinct spatial discretizations and that DMF-Gen retains the strongest H-resolution fidelity as H-resolution training fields are removed.
- Figure archetype: asymmetric mixed-modality figure with panel **c** as the image-led physical proof and panel **b** as the aggregate transfer test.
- Revision scope: publication polish only; no scientific redesign, source substitution, metric recomputation, retraining, or inference.
- Target: Nature-family double-column figure.
- Backend: Python/Matplotlib in the `fig` environment only.
- Final size: {width:.1f} × {height:.1f} mm.
- Panel map: (a) nested common-ROI L/M/H discretizations and recipe budgets; (b) grouped 512-sensor transfer plus two zero-H sensor sweeps; (c) fixed Zero-H-M-rich full/zoom/local-error proof across four models; (d) Large/Intermediate/Fine qualitative components and stacked fine-scale trends; (e) complete stacked correlation/bias matrices.
- Statistics: validated physical relative-L2 means with bootstrap 95% confidence intervals; multiscale medians with interquartile intervals; `valid_n=300` per quantitative cell.
- Source data: existing validated cached reconstructions and summary tables only.
- Image integrity: shared state, crop, sensor plan, and color normalization; no smoothing, sharpening, retraining, inference, or metric recomputation.
- V3-4 polish targets: nested inset geometry, typography de-collision, dashed frustum connectors, distinct component encoding, fine-row linkage, cross-panel pacing, and matrix block separation.
- Panel-a style provenance: the named attachment was unavailable in the mounted environment; the explicit revision geometry and the retained old nested-inset render were used and are recorded in the source manifest.
""")
    write(release / "figure_reference_update_v3_4.md", """
# Figure-reference update — V3-3 to V3-4

| V3-3 reference | V3-4 reference | Updated role |
|---|---|---|
| Fig. 3a | Fig. 3a | The validated L/M/H context and recipe budgets are retained; common-ROI insets are nested and resolution labels are cleaned up. |
| Fig. 3b, upper | Fig. 3b, upper | The validated grouped transfer bars are retained with a lighter, gap-anchored legend and de-collided title/annotation. |
| Fig. 3b, lower | Fig. 3b, lower | The two validated zero-H sensor sweeps remain unchanged, with improved vertical pacing. |
| Fig. 3c | Fig. 3c | The five-column physical proof is unchanged scientifically; row labels, dashed frustums, tile borders, and colorbar exponents are polished. |
| Fig. 3d qualitative | Fig. 3d, upper | Large/Intermediate/Fine qualitative evidence is retained, with a distinct truth-component encoding and an explicit Fine-row link. |
| Fig. 3d quantitative | Fig. 3d, lower | The stacked fine-scale correlation and variance-bias trends are retained and given clearer separation from the qualitative block. |
| Fig. 3e | Fig. 3e | The complete validated 4×9 matrices remain stacked, enlarged, and visibly segmented by recipe. |
| SI outputs | SI outputs | Validated standalone panels, distributions, and LaTeX tables remain available without changing their source values. |

Suggested manuscript sequence is unchanged: design `Fig.~3a`; aggregate transfer and budget robustness `Fig.~3b`; decisive physical proof `Fig.~3c`; qualitative scale progression and fine-scale trends `Fig.~3d`; full numerical scale-resolved workload `Fig.~3e`.
""")
    write(release / "quantitative_figure_report_v3_4.md", f"""
# Quantitative figure report — V3-4

V3-4 is a publication-polish revision. No scientific content or validated metric changes in this round. All quantitative values, validated source selections, representative states, and model caches are inherited from V3-3; this release changes layout, typography, connector styling, and matrix readability only.

## Panel a — resolution context and recipes

- Shared physical ROI: {a['shared_roi']}.
- Native grids: {a['dimensions']} with three nested zoom insets and {a['connector_count']} connectors.
- Exact relative spatial-field exposures: {a['exposure_values']}.
- The five validated recipe bars and contains-H/zero-H grouping are retained.

## Panel b — grouped recipe transfer and zero-H sweeps

- Source: {b['sources'][0]}.
- Top display: 20 grouped bars (four methods × five recipes), mean physical relative-L2 with bootstrap 95% intervals on a linear axis.
- DMF-Gen 512-sensor means: {b['recipe_transfer_values']['DMFGen']}.
- Lower displays: recipes {b['sweep_recipes']}, counts {b['sensor_counts']}, log-y scale, all four methods, and the same accepted intervals.
- V3-4 only changes title placement, annotation placement, legend anchoring, and subplot pacing.

## Panel c — fixed five-column physical proof

- Recipe/state: `{c['recipe']}`, snapshot {c['snapshot']}, case {c['case_id']}, time index {c['time_index']}, physical time {c['physical_time']}, 512 sensors.
- Columns: {c['display_columns']}.
- Shared truth/grid/sensor plan: `{c['shared_truth_ref']}`, `{c['shared_grid_ref']}`, `{c['sensor_plan_id']}` / `{c['sensor_plan_hash']}`.
- ROI: {c['roi']}.
- Full-field relative-L2: {c['full_field_relative_l2']}.
- Zoom-region relative-L2: {c['local_relative_l2']}.
- Exact MLP-RBF cache: `{mlp_record['path']}`; SHA-256 `{mlp_record['sha256']}`.
- V3-4 removes redundant first-row metric labels, uses dashed black frustums and black zoom borders, preserves equal sensor/error footprints, and keeps manual colorbar exponent placement.

## Panel d — complete qualitative scales and fine-scale trends

- Qualitative recipe/state: `{d['qualitative_recipe']}`, snapshot {d['displayed_snapshot']}, case {d['case_id']}, time index {d['time_index']}.
- The validated 256-sensor decomposition state and stacked fine-scale metrics are unchanged.
- Scale order: {d['qualitative_scales']}.
- Per-scale relative-L2 values: {d['relative_l2_by_model_scale']}.
- V3-4 distinguishes the truth component and visually links the Fine row to the two fine-scale charts.

## Panel e — complete numerical matrices

- Source: {e['heatmap_source_csv']}.
- Two vertically stacked 4×9 matrices: four methods × three recipes × Large/Intermediate/Fine.
- Cell annotations: correlation to two decimals; bias as signed two-decimal percentage points.
- All {len(e['heatmap_values'])} cells come directly from the validated summary and have `valid_n=300`.
- V3-4 enlarges the matrices and inserts visible recipe-block separation without changing values.

## Provenance and SI

- Validated cache fields, summary tables, standalone SI figures, and LaTeX tables are copied or referenced only; source hashes are recorded in `source_manifest_v3_4.json`.
- The exact V3-3 renderer stack and main outputs are anchored by hash before assembly, and all V3-3/V3-2/V2 baseline artifacts are checked for immutability after the build.
- Training, model inference, and metric recomputation were not performed.
""")
    write(release / "completion_report_v3_4.md", f"""
# Mixed-resolution Figure V3-4 completion report

- Release: `{release.name}`
- QA status: **{qa_status}**
- Main figure: SVG, PDF, and 600-DPI PNG.
- Standalone panels: a–e in SVG/PDF/PNG.
- SI: four SVG/PDF/PNG triplets plus four complete LaTeX tables.
- MLP-RBF representative reconstruction: verified before plotting and recorded by path/hash.
- Source/result tree unchanged: {manifest.get('source_immutability', {}).get('unchanged', 'pending')}.
- V3-3 baseline artifacts unchanged: {manifest.get('baseline_immutability', {}).get('unchanged', 'pending')}.
- Training, model inference, and metric recomputation: not performed.
- QA: the V3-4 audit contract, SI completeness, source hashes, baseline hashes, and direct cache checks are enforced by `108_audit_unified_v3_4.py`.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    args = parser.parse_args(); rid = args.run_id
    release = DELIVERY_ROOT / f"MixedResolution_unified_v3_4_{rid}"
    main_base = FIGURES / "Assembled" / f"MixedResolution_unified_v3_4_{rid}"
    manifest_source = FIGURES / "Assembled" / f"FigureSourceManifest_unified_v3_4_{rid}.json"
    process_root = FIGURES / "UnifiedV3_4" / rid
    collisions = [release, process_root, manifest_source,
                  *[main_base.with_suffix(f".{ext}") for ext in ("svg", "pdf", "png")]]
    if any(path.exists() for path in collisions):
        raise FileExistsError(f"Refusing to overwrite existing V3-4 artifacts: {collisions}")
    baseline_paths = [
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
    subprocess.run([sys.executable, str(SCRIPTS / "106_assemble_mixed_resolution_unified_v3_4.py"), *common],
                   cwd=SCRIPTS, check=True)
    subprocess.run([sys.executable, str(SCRIPTS / "107_export_unified_v3_4_panels.py"), *common],
                   cwd=SCRIPTS, check=True)
    after = tree_state(RESULTS)
    if before != after:
        raise RuntimeError("Validated _Process_Results changed during the visualization-only V3-4 build")
    baseline_after = {str(path): sha256(path) for path in baseline_paths}
    if baseline_before != baseline_after:
        raise RuntimeError("A V3-3, V3-2, or requested V2 baseline artifact changed during the V3-4 build")

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
        "panel_exporter": record(SCRIPTS / "107_export_unified_v3_4_panels.py"),
        "audit": record(SCRIPTS / "108_audit_unified_v3_4.py"),
    }
    per_snapshot = RESULTS / "MultiscaleWavelet" / "MultiscaleWavelet_per_snapshot_20260802_1250.csv"
    manifest["extended_multiscale_distribution_source"] = record(per_snapshot)
    manifest["release_directory"] = str(release.resolve())
    manifest["artifact_outputs"] = [record(path) for path in sorted(release.rglob("*")) if path.is_file()]
    manifest["artifact_output_scope"] = (
        "Rendered figures and LaTeX tables copied before documentation; metadata and reports are excluded "
        "from artifact hashes to avoid self-reference."
    )
    source_manifest = release / "source_manifest_v3_4.json"
    source_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_source.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_docs(release, manifest)
    subprocess.run([sys.executable, str(SCRIPTS / "108_audit_unified_v3_4.py"),
                    "--run-id", rid, "--release-dir", str(release)], cwd=SCRIPTS, check=True)
    qa = json.loads((release / "qa_v3_4.json").read_text(encoding="utf-8"))
    build_docs(release, manifest, qa_status="pass" if qa.get("passed") else "fail")
    if not qa.get("passed"):
        raise RuntimeError(f"V3-4 QA failed: {release / 'qa_v3_4.json'}")
    print(f"[OK] {release}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build and audit the additive mixed-resolution Figure V3-2 hybrid release."""
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
SCRIPTS = MODELS_ROOT / "_Scripts"
FIGURES = MODELS_ROOT / "_Process_Figures"
RESULTS = MODELS_ROOT / "_Process_Results"
LAYOUT = SCRIPTS / "publication_layout_unified_v3_2.yaml"
DELIVERY_ROOT = ROOT / "figures" / "generated"

SOURCE_DATA_RUN_ID = "20260806_1124"
MULTISCALE_RUN_ID = "20260802_1250"
BASE_DATA_RUN_ID = "2026-08-06_11-24"


def sha256(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_state(root: Path):
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def state_digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode("utf-8")).hexdigest()


def write(path: Path, text: str):
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def build_docs(release: Path, manifest: dict, *, qa_status="pending"):
    a, b, c, d = (manifest["panels"][label] for label in "abcd")
    width, height = manifest["figure_contract"]["final_size_mm"]
    write(release / "figure_contract.md", f"""
# Mixed-resolution Figure V3-2 hybrid contract

- Core conclusion: progressive removal of H-resolution training information does not dislodge DMF-Gen; in the decisive Zero-H-M-rich case, its advantage is visible globally, in the aligned local H-resolution crop, and in local error, while fine-scale metrics confirm the spatial interpretation.
- Figure archetype: asymmetric mixed-modality figure with panel **b** as the macro quantitative summary and panel **c** as the image-led spatial proof.
- Target: Nature-family double-column figure.
- Backend: Python/Matplotlib in the `fig` environment only.
- Final size: {width:.1f} × {height:.1f} mm.
- Panel map: (a) compact training-resolution design; (b) 512-sensor recipe transfer plus two zero-H sensor sweeps; (c) one Zero-H-M-rich state shown as full field, aligned zoom, and zoom-region error; (d) adjacent intermediate/fine residual evidence and fine-scale quantitative summaries.
- Statistics: performance means with bootstrap 95% confidence intervals; multiscale medians with interquartile intervals; `n=300` validated states per multiscale summary cell.
- Image integrity: validated cached physical fields only; one shared ROI; common row-level normalizations; no smoothing, sharpening, retraining, or inference.
- SI contract: Sx1 complete sensor sweeps; Sx2 expanded recipe gallery; Sx3 complete three-scale qualitative evidence; Sx4 complete three-scale quantitative matrices; four LaTeX tables.
- Main reviewer risks: qualitative recipe leakage, representative-state drift, inconsistent crops, local normalization bias, duplicated quantitative evidence, detached multiscale support, unreadable recovered detail, and source mutation.
""")
    write(release / "figure_reference_update.md", """
# Figure-reference update for V3-2

| Earlier evidence | V3-2 reference | Meaning |
|---|---|---|
| training-design panel | Fig. 3a | resolutions, recipes, and spatial-field exposure |
| recipe transfer + zero-H sweeps | Fig. 3b | cross-resolution and sensor-budget robustness |
| representative reconstruction | Fig. 3c | paired full-field, zoomed-field, and local-error spatial proof |
| compact multiscale support | Fig. 3d | intermediate/fine residuals and fine-scale quantitative fidelity |
| full five-recipe sweeps | SI Fig. Sx1 | complete sensor-budget evidence |
| multi-recipe reconstruction gallery | SI Fig. Sx2 | expanded qualitative evidence |
| complete three-scale fields/residuals | SI Fig. Sx3 | full qualitative wavelet decomposition |
| complete three-scale matrices | SI Fig. Sx4 | full multiscale quantitative summary |

Suggested manuscript logic: perturbation design `Fig.~3a`; aggregate performance `Fig.~3b`; decisive local spatial proof `Fig.~3c`; multiscale confirmation `Fig.~3d`; complete evidence matrices `Supplementary Figs.~Sx1--Sx4`.
""")
    write(release / "quantitative_figure_report.md", f"""
# Quantitative figure report — V3-2 hybrid

## Evidence retained in the main figure

### Panel a — training-resolution design

- Validated sources: {', '.join(a['sources'])}
- Evidence: native L/M/H glyphs, all five recipe budgets, and exact relative exposures {a['exposure_values']}.
- Information-density rationale: the compact strip establishes what information is removed without competing with the performance evidence.

### Panel b — cross-resolution performance

- Validated source: {b['sources'][0]}
- Recipe-transfer filter: `physical_rel_l2`, 512 sensors, four methods × five recipes.
- Zero-H sweep filters: recipes `4_ZeroH_Balanced` and `5_ZeroH_MRich`, sensor counts {b['sensor_counts']}.
- Statistic: {b['statistic']}; common log-y range {b['shared_y_range']}.
- Information-density rationale: retaining only the two zero-H sweeps avoids the redundant five-facet wall while directly testing the zero-shot claim.

### Panel c — restored spatial proof

- Validated caches: {', '.join(c['cache_sources'])}
- Representative recipe/state: `{c['recipe']}`, snapshot {c['snapshot']}, case {c['case_id']}, time index {c['time_index']}, sensor count {c['sensor_count']}.
- Selection rule: the established shared representative state is retained; the ROI is the same deterministic maximum-integrated-gradient region used by V3 and is recorded as `{c['roi_id']}`.
- Columns: {c['display_columns']}.
- Rows: {c['row_order']} with cell counts {c['row_cell_counts']}.
- Full-field relative-L2: {c['full_field_relative_l2']}.
- Zoom-region relative-L2: {c['local_relative_l2']}.
- Image handling: {c['image_adjustments']}.
- Information-density rationale: a dedicated aligned zoom row restores the irreplaceable micro-scale comparison; the paired local-error row turns visual sharpness into spatially explicit error evidence without restoring multiple recipes or MLP-RBF contours.

### Panel d — multiscale confirmation

- Validated sources: {', '.join(d['sources'] + d['cache_sources'])}
- Qualitative selection: `{d['qualitative_recipe']}`, displayed snapshot {d['displayed_snapshot']}, sensor count {d['sensor_count']}, scales {d['qualitative_scales']}, residual models {d['qualitative_models']}.
- Provenance: the wavelet metadata's optional automatic selector is snapshot {d['metadata_selected_snapshot']}; the established explicit shared layout displays snapshot {d['displayed_snapshot']} and remains unchanged.
- Quantitative filters: fine scale; recipes {d['quantitative_recipes']}; methods {d['quantitative_models']}.
- Statistic: {d['statistic']} with {d['dispersion']} across 300 validated states.
- Panel-d anchoring: {d['qualitative_quantitative_arrangement']}. The stacked plots numerically summarize the fine-scale behavior immediately beside the qualitative residual evidence.
- Information-density rationale: intermediate/fine residuals carry the discriminating spatial evidence; saturated large-scale evidence remains available in SI rather than consuming main-figure area.

## Evidence moved to SI

- SI Sx1: complete five-recipe sensor sweeps.
- SI Sx2: expanded validated Mixed-HML, Zero-H-balanced, and Zero-H-M-rich gallery for all four available models at 256 sensors.
- SI Sx3: complete large/intermediate/fine ground-truth components and residual maps.
- SI Sx4: complete three-scale pattern-correlation and variance-allocation-bias matrices.
- Tables: 512-sensor comparison, all 64–512 sensor sweeps, all-scale correlations, and all-scale variance-allocation biases.

No evidence was silently discarded or substituted. The main/SI split improves information density by reserving the main figure for the shortest defensible inspection sequence while keeping every complete comparison in the additive release.
""")
    write(release / "completion_report.md", f"""
# Mixed-resolution Figure V3-2 hybrid completion report

- Release: `{release.name}`
- QA status: **{qa_status}**
- Main figure: SVG, PDF, and 600-DPI PNG.
- Standalone panels: a–d in SVG/PDF/PNG.
- SI figures: four separate SVG/PDF/PNG triplets (Sx1–Sx4).
- SI tables: 512-sensor accuracy; 64–512 sensor sweeps; all-scale correlations; all-scale variance-allocation biases.
- Panel-c proof: one fixed Zero-H-M-rich state with full field, aligned zoom crop, and zoom-region absolute error.
- Provenance: `source_manifest.json` records sources, hashes, exact filters, representative identity, crop bounds, annotations, and output hashes.
- Validated sources/caches unchanged: {manifest.get('source_immutability', {}).get('unchanged', 'pending')}.
- Training/inference: not performed.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    args = parser.parse_args()
    rid = args.run_id
    release = DELIVERY_ROOT / f"MixedResolution_unified_v3_2_hybrid_{rid}"
    main_base = FIGURES / "Assembled" / f"MixedResolution_unified_v3_2_hybrid_{rid}"
    manifest_source = FIGURES / "Assembled" / f"FigureSourceManifest_unified_v3_2_{rid}.json"
    process_root = FIGURES / "UnifiedV3_2" / rid
    collisions = [
        path for path in [
            release, process_root, manifest_source,
            *[main_base.with_suffix(f".{ext}") for ext in ("svg", "pdf", "png")],
        ] if path.exists()
    ]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite additive V3-2 artifacts: {collisions}")

    before = tree_state(RESULTS)
    common = [
        "--run-id", rid, "--layout", str(LAYOUT),
        "--data-run-id", SOURCE_DATA_RUN_ID,
        "--multiscale-run-id", MULTISCALE_RUN_ID,
        "--base-data-run-id", BASE_DATA_RUN_ID,
    ]
    subprocess.run(
        [sys.executable, str(SCRIPTS / "100_assemble_mixed_resolution_unified_v3_2.py"), *common],
        cwd=SCRIPTS, check=True,
    )
    subprocess.run(
        [sys.executable, str(SCRIPTS / "101_export_unified_v3_2_panels.py"), *common],
        cwd=SCRIPTS, check=True,
    )
    after = tree_state(RESULTS)
    if before != after:
        raise RuntimeError("Validated _Process_Results changed during the visualization-only V3-2 build")

    release.mkdir(parents=True)
    for ext in ("svg", "pdf", "png"):
        shutil.copy2(main_base.with_suffix(f".{ext}"), release / main_base.with_suffix(f".{ext}").name)
    shutil.copytree(process_root / "panels", release / "panels")
    shutil.copytree(process_root / "si", release / "si")
    shutil.copytree(process_root / "tables", release / "tables")

    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    manifest["source_immutability"] = {
        "unchanged": True,
        "result_file_count_before": len(before),
        "result_file_count_after": len(after),
        "tree_state_sha256_before": state_digest(before),
        "tree_state_sha256_after": state_digest(after),
    }
    manifest["release_directory"] = str(release.resolve())
    manifest["artifact_outputs"] = [
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(release.rglob("*")) if path.is_file()
    ]
    manifest["artifact_output_scope"] = (
        "Rendered figures and LaTeX tables copied before metadata/document generation; "
        "source_manifest.json, qa.json, and Markdown reports are intentionally excluded "
        "to avoid self-referential hashes."
    )
    source_manifest = release / "source_manifest.json"
    source_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_source.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_docs(release, manifest)
    subprocess.run([
        sys.executable, str(SCRIPTS / "102_audit_unified_v3_2.py"),
        "--run-id", rid, "--release-dir", str(release),
    ], cwd=SCRIPTS, check=True)
    qa = json.loads((release / "qa.json").read_text(encoding="utf-8"))
    build_docs(release, manifest, qa_status="pass" if qa.get("passed") else "fail")
    if not qa.get("passed"):
        raise RuntimeError(f"V3-2 QA failed: {release / 'qa.json'}")
    print(f"[OK] {release}")


if __name__ == "__main__":
    main()

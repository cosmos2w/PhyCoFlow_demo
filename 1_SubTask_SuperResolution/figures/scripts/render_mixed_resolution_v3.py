#!/usr/bin/env python
"""Build and audit the additive streamlined mixed-resolution Figure 3 release."""
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
LAYOUT = SCRIPTS / "publication_layout_unified_v3.yaml"
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
    panels = manifest["panels"]
    a, b, c, d = (panels[label] for label in "abcd")
    write(release / "figure_contract.md", f"""
# Streamlined mixed-resolution Figure 3 contract

- Core conclusion: DMF-Gen remains strongest without H-resolution training fields across sensor budgets, and the gain reflects preserved intermediate/fine spatial structure.
- Figure archetype: asymmetric mixed-modality figure with panel **b** as the hero evidence.
- Target: Nature-family double-column figure.
- Backend: Python/Matplotlib in the `fig` environment only.
- Final size: {manifest['figure_contract']['final_size_mm'][0]:.1f} × {manifest['figure_contract']['final_size_mm'][1]:.1f} mm.
- Panel map: (a) compact training-resolution design; (b) recipe transfer at 512 sensors plus two zero-H sensor sweeps; (c) one Zero-H-M-rich reconstruction; (d) intermediate/fine residual structure plus fine-scale fidelity.
- Statistics: performance means with bootstrap 95% confidence intervals; multiscale medians with interquartile intervals; `n=300` validated held-out states per summary cell.
- Image integrity: validated cached physical fields only; common normalizations; no smoothing, sharpening, training, or inference.
- SI contract: all five sensor sweeps, the full prior recipe gallery, complete three-scale wavelet evidence, and four LaTeX tables.
- Main reviewer risks: duplicate evidence, recipe leakage into panel c, loss of displaced large-scale evidence, representative-state drift, source mutation, small text, and label overlap.
""")
    write(release / "figure_reference_update.md", """
# Figure-reference update

| Previous reference | Streamlined reference | Meaning |
|---|---|---|
| old panel a | new Fig. 3a | training resolutions, recipes, and exposure |
| old panel b | new Fig. 3b | 512-sensor recipe transfer |
| old panel c | new Fig. 3c | representative Zero-H-M-rich reconstruction |
| old panel d | new Fig. 3b | zero-H sensor-budget behavior |
| old panel e | new Fig. 3d | intermediate/fine qualitative wavelet evidence |
| old panel f | new Fig. 3d | fine-scale quantitative fidelity |

Suggested manuscript logic: recipes/exposure `Fig.~3a`; 512-sensor and sensor-budget results `Fig.~3b`; representative reconstruction `Fig.~3c`; multiscale evidence `Fig.~3d`.
""")
    write(release / "quantitative_figure_report.md", f"""
# Quantitative figure report

## Panel a — training-resolution design

- Sources: {', '.join(a['sources'])}
- Filters: the validated L/M/H protocol state and all five recipe-budget rows in the established order.
- Resolution glyphs: {a['dimensions']} using native cells without interpolation, contours, or zoom panels.
- Exposure values: {a['exposure_values']} (displayed as 1.00×, 0.34×, 0.44×, 0.16×, 0.19×).
- Statistic/interval: deterministic training-set composition; no sampling interval.
- Unified-v2 relationship: unchanged source values, reduced image footprint.
- Scientific role: define the perturbation before showing performance.

## Panel b — integrated reconstruction performance

- Source: {b['sources'][0]}
- b1 filter: metric `physical_rel_l2`, sensor count 512, four methods × five recipes.
- b2/b3 filters: the same metric, recipes `4_ZeroH_Balanced` and `5_ZeroH_MRich`, sensor counts {b['sensor_counts']}.
- Plotted statistic: {b['statistic']}; shared log-y range {b['shared_y_range']}.
- 512 overlap: the two zero-H endpoints recur only as the terminal points of the sensor-budget trends; this is the explicit distinct purpose recorded in the source manifest.
- Unified-v2 relationship: exact validated rows from the sensor-sweep summary; only grouping and subset selection changed.
- Scientific role: primary evidence for recipe adaptability and sparse-sensor robustness.

## Panel c — representative zero-H reconstruction

- Sources: {', '.join(c['cache_sources'])}
- Filters: recipe `{c['recipe']}`, sensor count {c['sensor_count']}, models {c['models']}.
- Representative identity: snapshot {c['snapshot']}, case {c['case_id']}, time index {c['time_index']}.
- Rows: full H-resolution field with a common zoom box; local absolute-error maps.
- Relative-L2 annotations: {c['relative_l2']}.
- Statistic/interval: one preselected validated state; no inferential interval.
- Unified-v2 relationship: identical state, caches, field/error definitions, and common normalization policy; recipes 3/4 were removed from the main qualitative plate.
- Scientific role: spatially localize the performance difference.

## Panel d — multiscale fidelity

- Sources: {', '.join(d['sources'] + d['cache_sources'])}
- Qualitative filters: recipe `{d['qualitative_recipe']}`, displayed snapshot {d['displayed_snapshot']}, sensor count {d['sensor_count']}, scales {d['qualitative_scales']}, models {d['qualitative_models']}.
- Provenance note: the wavelet metadata's optional selector is snapshot {d['metadata_selected_snapshot']}; unified-v2's explicit shared layout selected snapshot {d['displayed_snapshot']}, which is preserved here.
- Quantitative filters: fine scale only, recipes {d['quantitative_recipes']}, models {d['quantitative_models']}, metrics `pattern_correlation` and `variance_fraction_bias_pp`.
- Plotted statistic: {d['statistic']} with {d['dispersion']} across 300 validated states.
- Zero-H-M-rich correlations: {d['quantitative_values']['pattern_correlation']}.
- Unified-v2 relationship: exact validated medians; the large-scale row and complete heatmaps move to SI Sx3.
- Scientific role: prove that the reconstruction advantage extends below the dominant large scale.
""")
    write(release / "completion_report.md", f"""
# Streamlined mixed-resolution Figure 3 completion report

- Release: `{release.name}`
- QA status: **{qa_status}**
- Main figure: SVG, PDF, and 600-DPI PNG.
- Standalone panels: a–d in SVG/PDF/PNG.
- SI figures: Sx1 complete sensor sweeps; Sx2 full prior recipe gallery; Sx3 complete three-scale wavelet analysis.
- SI tables: 512-sensor accuracy; 64–512 sensor sweeps; all-scale correlations; all-scale variance-allocation biases.
- Provenance: `source_manifest.json` records paths, hashes, row selections, representative identities, and output hashes.
- Scientific evidence: unchanged from validated unified-v2 sources.
- Validated sources/caches unchanged: {manifest.get('source_immutability', {}).get('unchanged', 'pending')}.
- Training/inference: not performed.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    args = parser.parse_args()
    rid = args.run_id
    release = DELIVERY_ROOT / f"MixedResolution_unified_v3_streamlined_{rid}"
    main_base = FIGURES / "Assembled" / f"MixedResolution_unified_v3_streamlined_{rid}"
    manifest_source = FIGURES / "Assembled" / f"FigureSourceManifest_unified_v3_{rid}.json"
    process_root = FIGURES / "UnifiedV3" / rid
    collisions = [path for path in [release, process_root, manifest_source, *[main_base.with_suffix(f'.{ext}') for ext in ('svg','pdf','png')]] if path.exists()]
    if collisions:
        raise FileExistsError(f"Refusing to overwrite additive v3 artifacts: {collisions}")

    before = tree_state(RESULTS)
    common = [
        "--run-id", rid, "--layout", str(LAYOUT),
        "--data-run-id", SOURCE_DATA_RUN_ID,
        "--multiscale-run-id", MULTISCALE_RUN_ID,
        "--base-data-run-id", BASE_DATA_RUN_ID,
    ]
    subprocess.run([sys.executable, str(SCRIPTS / "97_assemble_mixed_resolution_unified_v3.py"), *common],
                   cwd=SCRIPTS, check=True)
    subprocess.run([sys.executable, str(SCRIPTS / "96_export_unified_v3_panels.py"), *common],
                   cwd=SCRIPTS, check=True)
    after = tree_state(RESULTS)
    if before != after:
        raise RuntimeError("Validated _Process_Results changed during the visualization-only v3 build")

    release.mkdir(parents=True)
    for ext in ("svg", "pdf", "png"):
        shutil.copy2(main_base.with_suffix(f".{ext}"), release / main_base.with_suffix(f".{ext}").name)
    shutil.copytree(process_root / "panels", release / "panels")
    shutil.copytree(process_root / "si", release / "si")
    shutil.copytree(process_root / "tables", release / "tables")
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    manifest["schema_version"] = 3
    manifest["source_immutability"] = {
        "unchanged": True,
        "result_file_count_before": len(before), "result_file_count_after": len(after),
        "tree_state_sha256_before": state_digest(before),
        "tree_state_sha256_after": state_digest(after),
    }
    manifest["release_directory"] = str(release.resolve())
    manifest["release_outputs"] = [
        {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(release.rglob("*")) if path.is_file()
    ]
    source_manifest = release / "source_manifest.json"
    source_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_source.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    build_docs(release, manifest)
    subprocess.run([
        sys.executable, str(SCRIPTS / "98_audit_unified_v3.py"),
        "--run-id", rid, "--release-dir", str(release),
    ], cwd=SCRIPTS, check=True)
    qa = json.loads((release / "qa.json").read_text(encoding="utf-8"))
    build_docs(release, manifest, qa_status="pass" if qa.get("passed") else "fail")
    if not qa.get("passed"):
        raise RuntimeError(f"Unified-v3 QA failed: {release / 'qa.json'}")
    print(f"[OK] {release}")


if __name__ == "__main__":
    main()

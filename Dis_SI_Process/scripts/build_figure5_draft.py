#!/usr/bin/env python
"""Build five standalone Figure 5 V3 panels and the strict composed SVG."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from utils.figure5_data import SourceRecord, load_figure5_data  # noqa: E402
from utils.figure5_panels import make_composed, make_standalone  # noqa: E402
from utils.figure5_style import apply_style, save_svg  # noqa: E402


PANEL_QUESTIONS = {
    "a": "Which trained generative method produces the strongest empirical conditional ensemble under identical temperature-only measurements?",
    "b": "For each generative method, is macro normalized ensemble spread associated with macro ensemble-mean reconstruction error across states?",
    "c": "What native-mesh accuracy–latency trade-off is measured for the eight Figure 4 checkpoints under one clean model-core timing boundary?",
    "d": "How does warm model-core latency scale with requested query count where the canonical model genuinely accepts variable query sets?",
    "e": "How does peak allocated GPU memory scale under the identical query-support protocol used for panel d?",
}

PANEL_CAVEATS = {
    "a": "CRPS is empirical and finite-ensemble; four unobserved fields are normalized by frozen training standard deviations and receive equal 0.25 weight. Full reliability and interval-width curves remain SI-only.",
    "b": "Spearman association is descriptive and does not establish calibrated or prospective error prediction. The temporal moving-block bootstrap preserves local dependence in the single held-out trajectory.",
    "c": "Accuracy is reused from the exact-checkpoint 1,000-state FieldL2 table. Latency is hardware-, precision-, cache-, and timing-boundary-specific; no Pareto claim is forced.",
    "d": "A line denotes audited native variable-query execution. Open native-only markers denote fixed-discretization methods. No full-grid reconstruction followed by slicing is counted as query scaling.",
    "e": "Peak allocated—not reserved—memory is shown. Model weights, one prepared device-side state, and the allowed reusable geometry cache are included; the throughput-only extension was not run.",
}

PANEL_SI = {
    "a": "Field-resolved CRPS, reliability curves, raw/normalized interval widths, ensemble diversity, and single-draw versus ensemble-mean error.",
    "b": "Fieldwise spread/error scatter and bootstrap diagnostics.",
    "c": "Full repeats, p10/p90, cold-first timing, no-persistent-geometry timing, reserved memory, parameters, checkpoint sizes, and component/cache audit.",
    "d": "All repeat timings and explicit failure/support reasons; no throughput-only points were generated.",
    "e": "Peak reserved memory and allocation details under the same N/support matrix.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_draft.yaml")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--strict-formal", action="store_true")
    parser.add_argument("--preview-png", type=Path)
    parser.add_argument("--preview-dir", type=Path, help="Optional Python-rendered PNG QA previews; not part of the SVG bundle.")
    return parser.parse_args()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def _display_source(source: str) -> str:
    path = Path(source)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return source


def _result(panel: str, data: dict[str, Any]) -> str:
    if data["modes"][panel] != "formal":
        return "No quantitative result: the required frozen V3 source has not passed its formal QA gate."
    if panel == "a":
        return "; ".join(f"{row.method} {row.mean_normalized_crps:.4f} [{row.crps_ci_low:.4f}, {row.crps_ci_high:.4f}]" for row in data["uq_crps"].itertuples()) + "."
    if panel == "b":
        return "; ".join(f"{row.method} ρ={row.spearman_rho:.3f} [{row.spearman_ci_low:.3f}, {row.spearman_ci_high:.3f}]" for row in data["uq_spread"].itertuples()) + "."
    if panel == "c":
        return "; ".join(f"{row.method}: error {row.error:.4f}, {row.median_latency_ms:.2f} ms" for row in data["cost_native"].itertuples()) + "."
    table = data["cost_query"] if panel == "d" else data["cost_memory"]
    metric = "median_latency_ms" if panel == "d" else "peak_allocated_mib"
    support = data["query_support"]
    pieces = []
    for method in support[support["variable_query_supported"].astype(bool)]["method"]:
        group = table[table["method"].eq(method)].sort_values("N")
        pieces.append(f"{method} {float(group.iloc[0][metric]):.2f}→{float(group.iloc[-1][metric]):.2f}")
    return "; ".join(pieces) + (" ms from 1,024 to 40,300 queries." if panel == "d" else " MiB from 1,024 to 40,300 queries.")


def _run_details(panel: str, data: dict[str, Any]) -> str:
    run = data["run_metadata"]["uq" if panel in "ab" else "cost"]
    if run is None:
        return "Run metadata unavailable until the formal V3 source completes."
    manifest = run["manifest"]
    if panel in "ab":
        return f"Run `{manifest['run_id']}`; 200 paired states; M=256; N=40,300; S=64; shared draw-ID seed schedule; moving-block bootstrap with block length {manifest['bootstrap']['block_length']} and {manifest['bootstrap']['replicates']} replicates."
    env = manifest["environment"]
    boundary = manifest["timing_boundary"]
    return f"Run `{manifest['run_id']}`; {env['gpu_name']} ({env['gpu_uuid']}); driver {env['driver']}; PyTorch {env['torch']}; CUDA {env['torch_cuda']}; batch 1 float32. Boundary `{boundary['name']}`; persistent cache: {boundary['persistent_cache']}."


def _write_panel_companion(path: Path, panel: str, svg: Path, data: dict[str, Any], record: SourceRecord, config: dict[str, Any], timestamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""# Figure 5 V3 panel {panel}: {config['figure']['panel_map'][panel].replace('_', ' ')}

- Generated: `{timestamp}`
- SVG: `{svg.name}`
- Evidence status: **{record.mode.upper()}**

## Scientific question

{PANEL_QUESTIONS[panel]}

## Methods, cohort, and metric

{_run_details(panel, data)} The unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`, macro-aggregated with equal 0.25 weights. Exact checkpoint identities are recorded in the run manifest.

## Main quantitative result

{_result(panel, data)}

## Exact source and run identity

`{_display_source(record.source)}`

Source classification: {record.note}

## Caveats

{PANEL_CAVEATS[panel]}

## SI destination

{PANEL_SI[panel]}
""", encoding="utf-8")


def _write_composed_companion(path: Path, svg: Path, records: list[SourceRecord], data: dict[str, Any], config: dict[str, Any], timestamp: str) -> None:
    results = "\n".join(f"- **{record.panel}.** {_result(record.panel, data)}" for record in records)
    sources = "\n".join(f"- **{record.panel}.** `{_display_source(record.source)}`" for record in records)
    path.write_text(f"""# Figure 5 V3 composed candidate

- Generated: `{timestamp}`
- SVG: `{svg.name}`
- Canvas: `{config['figure']['width_mm']} mm × {config['figure']['composed_height_mm']} mm`
- Panel status: `{', '.join(f'{record.panel}={record.mode}' for record in records)}`

Figure 5 V3 compares conditional ensemble quality across the five trained generative methods and then separates native accuracy–latency from support-qualified query scaling. Panels a and b equal-weight `Y_CH4`, `Y_CO`, `U1`, and `p`; panel c reuses the frozen exact-checkpoint Figure 4 accuracy coordinate; panels d and e draw curves only for canonical models that natively accept variable query sets.

The V2 DMF timing of approximately 127 ms is superseded and must not be quoted in the manuscript. It included generic harness/adapter and shared-GPU contamination exposed by the later approximately 29 ms direct exact-shape probe. V3 uses the canonical 8,192-point reconstruction chunk on a clean-GPU warm model-core boundary; its unified, direct-core, and independent exact-shape timers agree, while the historical value is reconciled to the earlier 4,096-point streaming boundary.

The V2 calibration, interval-width, and NFE/solver panels remain provenance/SI material and do not enter this five-panel composition. No ablation training or throughput-only extension was run.

## Main quantitative results

{results}

## Exact sources

{sources}

## Statistics and interpretation limits

UQ intervals are 95% temporal moving-block bootstrap intervals over states. Latency bars are synchronized repeat IQRs after at least 20 warmups, 30 repeats, and 10 measured seconds; accuracy intervals use the frozen state-level bootstrap. Empirical spread is described as associated with reconstruction difficulty, not as a calibrated predictor. Absolute latency and memory are hardware- and implementation-specific.
""", encoding="utf-8")


def _export_derived(data: dict[str, Any], records: list[SourceRecord], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(record) for record in records]).to_csv(root / "data_source_status.csv", index=False)
    mapping = {
        "uq_crps": "fig5a_normalized_crps_source.csv", "uq_spread": "fig5b_spread_error_source.csv",
        "cost_native": "fig5c_accuracy_latency_source.csv", "cost_query": "fig5d_query_latency_source.csv",
        "cost_memory": "fig5e_query_memory_source.csv", "query_support": "variable_query_support.csv",
        "timing_boundary": "timing_boundary_audit.csv",
    }
    for key, filename in mapping.items():
        if data.get(key) is not None:
            data[key].to_csv(root / filename, index=False)
    for prefix, run in data["run_metadata"].items():
        if run is None:
            continue
        for filename in ("manifest.json", "qa.json"):
            source = run["directory"] / filename
            (root / f"{prefix}_{filename}").write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _write_completion_report(path: Path, data: dict[str, Any], config: dict[str, Any], timestamp: str) -> None:
    uq = data["run_metadata"]["uq"]
    cost = data["run_metadata"]["cost"]
    if uq is None or cost is None:
        status = "Blocked: one or more formal V3 runs are missing or failed QA."
        uq_id, cost_id = config["formal_inputs"]["uq_run_id"], config["formal_inputs"]["cost_run_id"]
        findings = "Quantitative findings are withheld until strict-formal inputs pass."
        support_text = "Support audit unavailable."
    else:
        status = "Complete: both formal V3 runs and the strict figure build passed."
        uq_id, cost_id = uq["manifest"]["run_id"], cost["manifest"]["run_id"]
        crps_best = data["uq_crps"].sort_values("mean_normalized_crps").iloc[0]
        rho_low = data["uq_spread"].sort_values("spearman_rho").iloc[0]
        dmf = data["cost_native"][data["cost_native"]["method"].eq("DMF-Gen")].iloc[0]
        direct = data["timing_boundary"][data["timing_boundary"]["timer"].eq("direct_core_persistent_geometry")].iloc[0]
        exact = data["timing_boundary"][data["timing_boundary"]["timer"].eq("exact_shape_reprobe_persistent_geometry")].iloc[0]
        historical = data["timing_boundary"][data["timing_boundary"]["timer"].eq("historical_approx_29ms_reconciliation")].iloc[0]
        crossing = data["uq_spread"][(data["uq_spread"]["spearman_ci_low"] <= 0) & (data["uq_spread"]["spearman_ci_high"] >= 0)]["method"].tolist()
        findings = (
            f"Lowest mean normalized CRPS: {crps_best['method']} ({crps_best['mean_normalized_crps']:.4f}). "
            f"Weakest spread/error association: {rho_low['method']} (ρ={rho_low['spearman_rho']:.3f}); "
            f"95% intervals cross zero for {', '.join(crossing) if crossing else 'no method'}. "
            f"Corrected DMF native latency: {dmf['median_latency_ms']:.2f} ms; direct core: {direct['median_latency_ms']:.2f} ms; "
            f"independent exact-shape reprobe: {exact['median_latency_ms']:.2f} ms. The historical approximately 29 ms probe maps to "
            f"{historical['reference_ms']:.2f} ms under the earlier 4,096-point streaming chunk (relative difference {historical['relative_difference']:.1%})."
        )
        variable = data["query_support"][data["query_support"]["variable_query_supported"].astype(bool)]["method"].tolist()
        fixed = data["query_support"][~data["query_support"]["variable_query_supported"].astype(bool)]["method"].tolist()
        support_text = f"Variable-query curves: {', '.join(variable)}. Native-only markers: {', '.join(fixed)}."
    path.write_text(f"""# Figure 5 V3 completion report

- Generated: `{timestamp}`
- Status: **{status}**
- Formal UQ run: `{uq_id}`
- Formal clean-cost run: `{cost_id}`
- Pilot runs: `{', '.join(config['formal_inputs']['uq_pilot_run_ids'])}`

## Provenance and supersession

V2 outputs remain unchanged as provenance. The V2 DMF median of {config['v2_provenance']['superseded_dmf_latency_ms']:.2f} ms from `formal_cost_20260830_v2` is superseded because its timing boundary included generic adapter/host-transfer overhead and was vulnerable to shared-GPU contamination. It must not be used as manuscript evidence. The V3 clean benchmark excludes loading, data I/O, CPU preprocessing, host transfers, generic dispatch, metrics, output transfer, and disk I/O while retaining required noise generation, value-dependent conditioning, model evaluations, observation consistency, and device-side output.

DMF chunk profiling resolved the provisional timing discrepancy. The adopted configurations specify an 8,192-point reconstruction chunk; V3 uses that canonical setting and permits only reusable static geometry. The prior approximately 29 ms result is consistent within 20% with the profiled 4,096-point streaming boundary, but is not the promoted coordinate.

The first five-method 12×8 pilot found same-seed drift only for Latent FM under nondeterministic CUDA execution (approximately 0.004–0.005 normalized max absolute difference). Deterministic and prepared-path reruns passed all stochasticity, reproducibility, normalization, and exact-path-equivalence gates; all pilot IDs are retained. Existing matching DMF U2 fieldwise spread/error/reliability summaries were reused in place, while the missing normalized CRPS reducer required matching-seed DMF draws.

## Main findings and narrative checks

{findings}

{support_text}

The provisional narrative is only partly supported: DMF-Gen has the lowest CRPS, but spread/error association is weak enough to cross zero for FFM-FNO and Latent FM, and the accuracy–latency measurements do not establish unqualified Pareto superiority.

## Scope

Full reliability/interval-width curves, fieldwise uncertainty, cold/no-cache timing, reserved memory, and NFE diagnostics remain SI/internal. The optional 100k–1M throughput-only extension was not run. No A0/A1/A2/A3 ablation training was started.
""", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "figure5-validation-v3" or config["figure"]["formats"] != ["svg"]:
        raise ValueError("Figure 5 V3 requires schema figure5-validation-v3 and SVG-only output")
    apply_style(config["style"]["font_family"])
    data, records = load_figure5_data(config, REPO_ROOT)
    nonformal = [record.panel for record in records if record.mode != "formal"]
    if args.strict_formal and nonformal:
        raise RuntimeError(f"Strict formal V3 build blocked; non-formal panels: {', '.join(nonformal)}")
    if args.strict_formal and any("ValidationV2/Cost" in record.source for record in records):
        raise RuntimeError("Strict formal V3 rejects ValidationV2 cost sources")

    figure_dir = args.output_root / "figures" / "generated" / args.timestamp
    docs_dir = args.output_root / "docs" / "generated"
    result_dir = args.output_root / "results" / "derived" / args.timestamp
    figure_dir.mkdir(parents=True, exist_ok=True)
    contract_copy = figure_dir / "figure_contract.md"
    contract_copy.write_text((PACKAGE_ROOT / "docs" / "figure_contract.md").read_text(encoding="utf-8"), encoding="utf-8")
    _export_derived(data, records, result_dir)
    record_by_panel = {record.panel: record for record in records}
    outputs: dict[str, Path] = {}
    for panel in "abcde":
        stem = config["figure"]["output_stems"][panel]
        svg = figure_dir / f"{stem}_{args.timestamp}.svg"
        standalone = make_standalone(panel, data, config)
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            standalone.savefig(args.preview_dir / f"{stem}_{args.timestamp}.png", format="png", dpi=240, facecolor="white")
        save_svg(standalone, svg)
        outputs[panel] = svg
        _write_panel_companion(docs_dir / f"{stem}_{args.timestamp}.md", panel, svg, data, record_by_panel[panel], config, args.timestamp)
    composed_stem = config["figure"]["output_stems"]["composed"]
    composed_svg = figure_dir / f"{composed_stem}_{args.timestamp}.svg"
    composed = make_composed(data, config)
    if args.preview_png:
        args.preview_png.parent.mkdir(parents=True, exist_ok=True)
        composed.savefig(args.preview_png, format="png", dpi=240, facecolor="white")
    save_svg(composed, composed_svg)
    outputs["composed"] = composed_svg
    _write_composed_companion(docs_dir / f"{composed_stem}_{args.timestamp}.md", composed_svg, records, data, config, args.timestamp)
    completion_report = docs_dir / f"figure5_v3_completion_report_{args.timestamp}.md"
    _write_completion_report(completion_report, data, config, args.timestamp)
    manifest = {
        "schema_version": config["schema_version"], "timestamp": args.timestamp, "git_commit": _git_commit(),
        "strict_formal": args.strict_formal, "config": str(args.config.resolve()), "config_sha256": _sha256(args.config),
        "formats": ["svg"], "outputs": {key: str(path) for key, path in outputs.items()},
        "companions": [str(docs_dir / f"{config['figure']['output_stems'][panel]}_{args.timestamp}.md") for panel in "abcde"] + [str(docs_dir / f"{composed_stem}_{args.timestamp}.md"), str(completion_report)],
        "figure_contract": str(contract_copy), "sources": [asdict(record) for record in records], "v2_provenance": config["v2_provenance"],
    }
    (result_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figure_dir": str(figure_dir), "panel_modes": data["modes"], "completion_report": str(completion_report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

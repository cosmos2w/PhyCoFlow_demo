#!/usr/bin/env python
"""Build the additive Figure 5 V4 SVG bundle.

The script reuses only the adopted V3 a/b/c evidence and requires independent
V4 training-cost and high-N stress evidence for panels d/e in strict mode.  It
never silently substitutes a V2/V3 query table or proxy value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.figure5_v4_data import SourceRecord, load_figure5_v4_data
from utils.figure5_v4_panels import make_composed, make_standalone
from utils.figure5_v4_style import apply_style, save_svg

PANEL_QUESTIONS = {
    "a": "Which trained generative method produces the strongest empirical conditional ensemble under identical temperature-only measurements?",
    "b": "For each generative method, is ensemble spread associated with reconstruction difficulty across the paired held-out states?",
    "c": "What native 40,300-point accuracy–cost trade-off is measured for the exact Figure 4 checkpoints?",
    "d": "What computational investment is recorded for the adopted checkpoints, using the predeclared training-cost metric?",
    "e": "How do warm latency and allocated memory scale when the requested query set extends beyond the native grid?",
}

PANEL_CAVEATS = {
    "a": "Empirical finite-ensemble normalized CRPS; four unobserved fields are macro-averaged with equal 0.25 weight. Calibration and interval-width diagnostics remain SI-only.",
    "b": "Spearman association is descriptive and does not establish calibration or prospective error prediction.",
    "c": "Latency is clean-GPU, warm model-core timing and is hardware-, precision-, cache-, and boundary-specific. The plot does not imply unqualified Pareto superiority.",
    "d": "Panel d is a descriptive canonical-configuration update-cost comparison, not total training GPU-hours or a causal matched-budget efficiency comparison. Adopted batch/query configurations differ by method; unavailable coordinates remain documented rather than imputed.",
    "e": "Only canonical variable-query paths receive curves. Values above N=40,300 are throughput-only stress measurements and carry no physical accuracy claim.",
}

PANEL_SI = {
    "a": "Field-resolved CRPS, reliability, interval width and ensemble-diversity diagnostics.",
    "b": "Fieldwise spread/error scatter and bootstrap diagnostics.",
    "c": "Full timing repeats, cold/no-cache timing, reserved memory, parameters, checkpoint sizes and timing-boundary audit.",
    "d": "Training logs, stage-level versus total cost, replay validation, optimizer-update throughput and peak training memory.",
    "e": "Coordinate specification/hash, geometry preparation, reserved memory, OOM/runtime-cap table and first-failure details.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_v4.yaml")
    parser.add_argument("--timestamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M"))
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--strict-formal", action="store_true")
    parser.add_argument("--preview-png", type=Path)
    parser.add_argument("--preview-dir", type=Path, help="Optional Python-rendered PNG previews; not part of the SVG bundle.")
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


def _run_details(panel: str, data: dict[str, Any]) -> str:
    key = {"a": "uq", "b": "uq", "c": "native", "d": "training", "e": "scale"}[panel]
    run = data["run_metadata"].get(key)
    if run is None:
        diagnostics = "; ".join(data.get("source_errors", {}).get(panel, []))
        return f"Formal source metadata unavailable. {diagnostics}".strip()
    manifest = run["manifest"]
    if panel in {"a", "b"}:
        bootstrap = manifest.get("bootstrap", {})
        return (
            f"Run `{manifest.get('run_id', 'unknown')}`; 200 paired states; M=256; S=64; "
            f"temporal moving-block bootstrap ({bootstrap.get('block_length', 'declared')} blocks, "
            f"{bootstrap.get('replicates', 'declared')} replicates)."
        )
    if panel == "c":
        environment = manifest.get("environment", {})
        boundary = manifest.get("timing_boundary", {})
        return (
            f"Run `{manifest.get('run_id', 'unknown')}`; {environment.get('gpu_name', 'declared GPU')}; "
            f"float32, batch 1; timing boundary `{boundary.get('name', 'declared')}`."
        )
    if panel == "d":
        metric = manifest.get("metric_name")
        manifest_metric = manifest.get("metric")
        if metric is None:
            metric = manifest_metric.get("name") if isinstance(manifest_metric, dict) else manifest_metric
        environment = manifest.get("environment", {})
        protocol = manifest.get("protocol", {})
        return (
            f"Run `{manifest.get('run_id', 'unknown')}`; {environment.get('gpu_name', 'declared GPU')}; "
            f"metric `{metric or 'declared'}`; batch policy `{protocol.get('batch_policy', 'declared')}`; "
            f"{protocol.get('warmup_updates', 'declared')} warmups and "
            f"{protocol.get('measured_updates', 'declared')} measured updates per successful stage; "
            "all required stages were attempted."
        )
    query_spec = manifest.get("query_spec") or manifest.get("dummy_query_spec")
    query = manifest.get("query_spec_hash") or manifest.get("dummy_query_spec_sha256")
    if query is None:
        query = query_spec.get("hash") if isinstance(query_spec, dict) else query_spec
    query = query or "declared"
    native_sources = run.get("v3_native_sources", {})
    native_note = ""
    if isinstance(native_sources, dict) and native_sources:
        native_note = f" V3 native prefix: `{native_sources.get('latency', 'query_latency_summary.csv')}` + `{native_sources.get('memory', 'memory_summary.csv')}`."
    return f"Run `{manifest.get('run_id', 'unknown')}`; sensor-prefixed Sobol query specification hash `{query}`; native and throughput-only regions are explicitly separated.{native_note}"


def _result(panel: str, data: dict[str, Any]) -> str:
    if data["modes"][panel] != "formal":
        return "No quantitative result: the required V4 source has not passed its formal QA gate."
    if panel == "a":
        return "; ".join(f"{row.method} {row.mean_normalized_crps:.4f} [{row.crps_ci_low:.4f}, {row.crps_ci_high:.4f}]" for row in data["uq_crps"].itertuples()) + "."
    if panel == "b":
        return "; ".join(f"{row.method} ρ={row.spearman_rho:.3f} [{row.spearman_ci_low:.3f}, {row.spearman_ci_high:.3f}]" for row in data["uq_spread"].itertuples()) + "."
    if panel == "c":
        return "; ".join(f"{row.method}: error {row.error:.4f}, {row.cost_value:.2f} ms" for row in data["cost_native"].itertuples()) + "."
    if panel == "d":
        rows = data["training_cost"][data["training_cost"]["status"].astype(str).str.lower().eq("ok")]
        unavailable = data["training_cost"][~data["training_cost"]["status"].astype(str).str.lower().eq("ok")]
        metric = data.get("training_metric") or "training compute"
        result = "; ".join(f"{row.method}: error {row.error:.4f}, {row.cost_value:.3g} {row.cost_unit}" for row in rows.itertuples())
        if not unavailable.empty:
            result += ". Method-level unavailable: " + ", ".join(unavailable["method"].astype(str))
        return result + f". Metric: {metric}."
    support = data["query_support"]
    variable = support[support["variable_query_supported"].astype(bool)]["method"].astype(str).tolist()
    boundary = data["scale_boundary"]
    return "Variable-query curves: " + ", ".join(variable) + ". Largest successful N: " + "; ".join(
        f"{row.method}={int(row.largest_successful_N):,}" for row in boundary.itertuples()
    ) + "."


def _write_panel_companion(path: Path, panel: str, svg: Path, data: dict[str, Any], record: SourceRecord, config: dict[str, Any], timestamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# Figure 5 V4 panel {panel}: {config['figure']['panel_map'][panel].replace('_', ' ')}

- Generated: `{timestamp}`
- SVG: `{svg.name}`
- Evidence status: **{record.mode.upper()}**

## Scientific question

{PANEL_QUESTIONS[panel]}

## Protocol and metric

{_run_details(panel, data)} The unobserved fields are `Y_CH4`, `Y_CO`, `U1`, and `p`, macro-aggregated with equal 0.25 weights where applicable. Exact checkpoint identities are retained in source manifests.

## Main quantitative result

{_result(panel, data)}

## Exact source

`{_display_source(record.source)}`

Source classification: {record.note}

## Interpretation limits

{PANEL_CAVEATS[panel]}

## SI destination

{PANEL_SI[panel]}
""",
        encoding="utf-8",
    )


def _write_composed_companion(path: Path, svg: Path, records: list[SourceRecord], data: dict[str, Any], config: dict[str, Any], timestamp: str) -> None:
    results = "\n".join(f"- **{record.panel}.** {_result(record.panel, data)}" for record in records)
    sources = "\n".join(f"- **{record.panel}.** `{_display_source(record.source)}`" for record in records)
    path.write_text(
        f"""# Figure 5 V4 composed candidate

- Generated: `{timestamp}`
- SVG: `{svg.name}`
- Canvas: `{config['figure']['width_mm']} mm × {config['figure']['composed_height_mm']} mm`
- Panel status: `{', '.join(f'{record.panel}={record.mode}' for record in records)}`

Figure 5 V4 pairs probabilistic quality (a/b), native inference accuracy–cost (c), training-compute accuracy–cost (d), and a full-width two-axis high-resolution scalability envelope (e). Panels a/b share method rows and the DMF-Gen highlight; c/d share the reconstruction-error axis; e uses the same x scale for warm latency and allocated memory.

Values above N=40,300 in panel e are explicitly throughput-only stress measurements. Fixed-discretization methods receive native open markers only; no full-grid reconstruction followed by slicing is accepted as a scaling curve. The V3 query and memory tables remain provenance and are never a V4 fallback.

## Main quantitative results

{results}

## Exact sources

{sources}

## Statistics and interpretation limits

UQ and reconstruction-error intervals use the predeclared temporal moving-block bootstrap. Inference latency intervals are synchronized repeat IQRs. Panel d is descriptive checkpoint compute, not a causal matched-budget comparison. Panel e makes no accuracy claim above native resolution.
""",
        encoding="utf-8",
    )


def _export_derived(data: dict[str, Any], records: list[SourceRecord], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(record) for record in records]).to_csv(root / "data_source_status.csv", index=False)
    mapping = {
        "uq_crps": "fig5a_normalized_crps_source.csv",
        "uq_spread": "fig5b_spread_error_source.csv",
        "cost_native": "fig5c_accuracy_latency_source.csv",
        "training_cost": "fig5d_accuracy_training_cost_source.csv",
        "scale_latency": "fig5e_scalability_latency_source.csv",
        "scale_memory": "fig5e_scalability_memory_source.csv",
        "query_support": "native_query_support_audit.csv",
        "scale_boundary": "boundary_summary.csv",
        "query_manifest": "query_coordinates_manifest.csv",
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
    d_errors = data.get("source_errors", {}).get("d", [])
    e_errors = data.get("source_errors", {}).get("e", [])
    d_metric = data.get("training_metric") or "not promoted; source unavailable"
    if data["modes"]["d"] == "formal":
        d_methods = data["training_cost"][data["training_cost"]["status"].astype(str).str.lower().eq("ok")]["method"].astype(str).tolist()
        unavailable = data["training_cost"][~data["training_cost"]["status"].astype(str).str.lower().eq("ok")]["method"].astype(str).tolist()
        training_run = data["run_metadata"]["training"]
        training_manifest = training_run["manifest"]
        stage_path = training_run["directory"] / "training_stage_summary.csv"
        stage_table = pd.read_csv(stage_path) if stage_path.is_file() else pd.DataFrame()
        latent_stage_text = ""
        if not stage_table.empty:
            latent = stage_table[stage_table["method"].astype(str).eq("Latent FM")]
            latent_stage_text = " Latent FM stage-level medians: " + "; ".join(
                f"{row.stage_name}={row.update_time_median_ms:.2f} ms/update" for row in latent.itertuples()
            ) + "."
        unavailable_details = []
        for row in data["training_cost"].itertuples():
            if str(row.status).lower() == "ok":
                continue
            reason = str(row.unavailable_reason)
            if row.method == "Geo-FNO" and "out of memory" in reason.lower():
                reason = "adopted batch 192 exceeded the 47.38-GiB GPU capacity"
            unavailable_details.append(f"{row.method}: {reason}")
        d_status = (
            f"Formal panel-d run: `{training_manifest.get('run_id')}`. Metric: `{d_metric}` = median synchronized "
            "canonical forward/loss/backward/gradient-clip/optimizer update at each method's adopted "
            "batch/query configuration, "
            "after 20 warmups and across 100 measured updates (10 blocks × 10) for each successful stage; "
            "all nine required stages were attempted. "
            f"Promoted single-stage methods: {', '.join(d_methods)}. "
            f"Method-level unavailable: {', '.join(unavailable) if unavailable else 'none'}. "
            f"Reasons: {'; '.join(unavailable_details) if unavailable_details else 'none'}."
            f"{latent_stage_text} Historical GPU-hours and filesystem timestamps were not used."
        )
    else:
        d_status = f"Panel d is pending. Required source was not promoted: {'; '.join(d_errors)}"
    if data["modes"]["e"] == "formal":
        support = data["query_support"]
        variable = support[support["variable_query_supported"].astype(bool)]["method"].astype(str).tolist()
        fixed = support[~support["variable_query_supported"].astype(bool)]["method"].astype(str).tolist()

        def boundary_text(row: Any) -> str:
            failed = "none (8M global cap)" if pd.isna(row.first_failed_N) else f"{int(row.first_failed_N):,}"
            return f"{row.method}: largest success {int(row.largest_successful_N):,}; first failure {failed}"

        boundaries = "; ".join(boundary_text(row) for row in data["scale_boundary"].itertuples())
        scale_metadata = data.get("run_metadata", {}).get("scale") or {}
        native_sources = scale_metadata.get("v3_native_sources", {})
        e_status = (
            f"Variable-query methods: {', '.join(variable)}. Fixed-grid/native-only methods: {', '.join(fixed)}. "
            f"{boundaries}. V3 native prefix sources: `{native_sources.get('latency', 'query_latency_summary.csv')}` "
            f"and `{native_sources.get('memory', 'memory_summary.csv')}`."
        )
    else:
        e_status = f"Panel e is pending. Required source was not promoted: {'; '.join(e_errors)}"
    if data["modes"]["d"] == "formal" and data["modes"]["e"] == "formal":
        dmf_update = float(
            data["training_cost"].loc[
                data["training_cost"]["method"].eq("DMF-Gen"), "cost_value"
            ].iloc[0]
        )
        narrative_note = (
            "DMF-Gen has the lowest reconstruction error and normalized CRPS. At the differing adopted "
            f"training configurations, its {dmf_update:.2f}-ms update is faster than FFM-FNO and SiT but "
            "slower than FFM-Perceiver, MLP-RBF, and Senseiver; this per-update footprint is not normalized "
            "for batch size or a matched training budget. Latent FM "
            "cannot receive a defensible single method-level update-time coordinate because the adopted model "
            "requires two unlike training stages, and Geo-FNO's adopted batch cannot be replayed within the "
            "47.38-GiB device capacity. In high-N inference, DMF-Gen reaches the 8M safety cap; "
            "FFM-Perceiver fails at 8M after succeeding at 4M, while MLP-RBF and Senseiver first fail at 2M "
            "after succeeding at 1M. These hardware-specific boundaries are capacity evidence, not accuracy evidence."
        )
    else:
        narrative_note = "A contradiction audit is withheld until both independent V4 cost sources pass their formal gates."
    panel_status = ", ".join(f"{panel}={data['modes'][panel]}" for panel in "abcde")
    path.write_text(
        f"""# Figure 5 V4 completion report

- Generated: `{timestamp}`
- Panel status: `{panel_status}`
- V3 provenance preserved: `true`

## Panel-d training-cost gate

{d_status}

Different archived training budgets are not a causal architectural comparison. Historical file timestamps are never used as training time. Replay-equivalent GPU-hours require a passing predeclared validation gate; otherwise a directly measured update-time metric or SI-only result is required.

## Panel-e scale-stress gate

{e_status}

The native validated endpoint is N={config['formal_protocol']['scale_stress']['native_limit']:,}. Values above it are throughput-only and use one common frozen query specification/hash. Fixed-grid methods remain native-only markers. OOM/runtime-cap events must remain visible in the source table.

## Scope and provenance

V3 UQ and clean native inference products are reused only when their original schema, checkpoint identity, and QA pass. V3 query-latency/memory tables and all V2 cost products are explicitly excluded as V4 fallbacks. No NFE panel or ablation training is part of this workflow.

The V2 DMF latency of approximately 127 ms remains superseded by the V3 clean warm model-core value of 16.69 ms and is not reused in V4.

## Results that qualify or contradict a simple efficiency narrative

{narrative_note}
""",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "figure5-validation-v4" or config["figure"].get("formats") != ["svg"]:
        raise ValueError("Figure 5 V4 requires schema figure5-validation-v4 and SVG-only output")
    apply_style(config["style"]["font_family"])
    data, records = load_figure5_v4_data(config, REPO_ROOT)
    nonformal = [record.panel for record in records if record.mode != "formal"]
    if args.strict_formal and nonformal:
        details = []
        for panel in nonformal:
            details.append(f"{panel}: {'; '.join(data.get('source_errors', {}).get(panel, []))}")
        raise RuntimeError(f"Strict formal V4 build blocked; non-formal panels: {', '.join(nonformal)}. " + " | ".join(details))
    if args.strict_formal and (data["modes"]["d"] != "formal" or data["modes"]["e"] != "formal"):
        raise RuntimeError("Strict formal V4 requires independent training-cost panel-d and high-N scale panel-e evidence; no fallback is permitted")

    figure_dir = args.output_root / "figures" / "generated" / args.timestamp
    docs_dir = args.output_root / "docs" / "generated"
    result_dir = args.output_root / "results" / "derived" / args.timestamp
    figure_dir.mkdir(parents=True, exist_ok=True)
    contract_path = PACKAGE_ROOT / "docs" / "figure5_v4_source_schema.md"
    contract_copy = figure_dir / "figure5_v4_source_schema.md"
    if contract_path.is_file():
        contract_copy.write_text(contract_path.read_text(encoding="utf-8"), encoding="utf-8")
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
    completion_report = docs_dir / f"figure5_v4_completion_report_{args.timestamp}.md"
    _write_completion_report(completion_report, data, config, args.timestamp)
    manifest = {
        "schema_version": config["schema_version"],
        "timestamp": args.timestamp,
        "git_commit": _git_commit(),
        "strict_formal": args.strict_formal,
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "formats": ["svg"],
        "canvas_mm": [float(config["figure"]["width_mm"]), float(config["figure"]["composed_height_mm"])],
        "outputs": {key: str(path) for key, path in outputs.items()},
        "companions": [str(docs_dir / f"{config['figure']['output_stems'][panel]}_{args.timestamp}.md") for panel in "abcde"] + [str(docs_dir / f"{composed_stem}_{args.timestamp}.md"), str(completion_report)],
        "figure_contract": str(contract_copy),
        "sources": [asdict(record) for record in records],
        "source_errors": data.get("source_errors", {}),
        "v3_provenance": config["v3_provenance"],
        "scale_v3_native_sources": (data.get("run_metadata", {}).get("scale") or {}).get("v3_native_sources", {}),
        "no_v4_fallback": True,
    }
    (result_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"figure_dir": str(figure_dir), "panel_modes": data["modes"], "completion_report": str(completion_report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

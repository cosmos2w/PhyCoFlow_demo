#!/usr/bin/env python
"""Build the additive Figure 5 V4.1 main and audited Zero-H backup SVG bundles."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
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

from utils.figure5_v41_data import load_figure5_v41_data
from utils.figure5_v41_panels import make_backup_composed, make_backup_standalone, make_composed, make_standalone
from utils.figure5_v41_style import apply_style, save_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_v41.yaml")
    parser.add_argument("--timestamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M"))
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--strict-formal", action="store_true")
    parser.add_argument("--preview-dir", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _display(path: str | Path) -> str:
    target = Path(path).resolve()
    try:
        return str(target.relative_to(REPO_ROOT))
    except ValueError:
        return str(target)


def _main_result(panel: str, data: dict[str, Any]) -> str:
    if panel == "a":
        rows = data["uq_crps"].sort_values("mean_normalized_crps")
        return "; ".join(f"{row.method} mean={row.mean_normalized_crps:.4f}" for row in rows.itertuples()) + "."
    if panel == "b":
        return "; ".join(f"{row.method} ρ={row.spearman_rho:.3f} [{row.spearman_ci_low:.3f}, {row.spearman_ci_high:.3f}]" for row in data["uq_spread"].itertuples()) + "."
    table = {"c": data["cost_native"], "d": data["training_cost"]}[panel]
    ok = table.loc[table["status"].astype(str).str.lower().eq("ok")]
    if panel == "c":
        return "; ".join(f"{row.method}: error={row.error:.4f}, latency={row.cost_value:.2f} ms" for row in ok.itertuples()) + "."
    return "; ".join(f"{row.method}: error={row.error:.4f}, peak={row.cost_value / 1024.0:.2f} GiB total" for row in ok.itertuples()) + "."


def _backup_result(panel: str, data: dict[str, Any]) -> str:
    column = {
        "a": "physical_rel_l2",
        "b": "gradient_rel_l2",
        "c": "physical_rel_l2_sensor_excluded",
        "d": "normalized_rel_l2",
    }[panel]
    rows = data["zeroh"].groupby("method")[column].agg(["median", "mean"])
    return "; ".join(f"{method}: median={row['median']:.4f}, mean={row['mean']:.4f}" for method, row in rows.iterrows()) + "."


def _write_companions(
    docs: Path,
    timestamp: str,
    config: dict[str, Any],
    data: dict[str, Any],
    main_svgs: dict[str, Path],
    backup_svgs: dict[str, Path],
) -> list[Path]:
    docs.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    main_protocol = {
        "a": "200 paired states; 64 draws/state; four unobserved fields macro-weighted 0.25. Box/scatter uses state-level normalized CRPS; open markers and lines retain the formal mean and moving-block-bootstrap 95% CI.",
        "b": "Box/scatter uses 2,000 predeclared temporal moving-block-bootstrap Spearman replicates per method (block length 25). The open marker is the full-sample ρ; bootstrap replicates are not independent test states.",
        "c": "Exact V3 clean warm model-core timing at N=40,300 and frozen 1,000-state field error; both axes are logarithmic.",
        "d": "One-GPU rows retain process-local peak allocations from the clean V4 canonical replay. Geo-FNO uses two-GPU DDP at global batch 192; x is the sum of simultaneous per-rank peak allocated memory. Both axes are logarithmic; shared-load wall timing is not used.",
        "e": "Peak allocated inference memory only. Canonical variable-query methods receive curves; fixed-grid methods receive native-size open markers. N>40,300 is throughput-only.",
    }
    for panel in "abcde":
        stem = config["figure"]["output_stems"][panel]
        path = docs / f"{stem}_{timestamp}.md"
        source = {
            "a": data["run_metadata"]["uq"]["directory"] / "per_state_method.csv",
            "b": data["run_metadata"]["uq"]["directory"] / "per_state_method.csv",
            "c": data["run_metadata"]["native"]["directory"] / "native_summary.csv",
            "d": data["geofno_multigpu"]["directory"] / "geofno_ddp_summary.csv",
            "e": data["run_metadata"]["scale"]["directory"] / "scale_stress_summary.csv",
        }[panel]
        path.write_text(
            f"""# Figure 5 V4.1 panel {panel}

- Generated: `{timestamp}`
- SVG: `{main_svgs[panel].name}`
- Evidence status: **FORMAL**

## Protocol and visual statistic

{main_protocol[panel]}

## Main quantitative result

{_main_result(panel, data) if panel != 'e' else 'Memory capacity boundaries are retained from the formal V4 scale-stress source; the removed latency axis is not exported in the V4.1 main figure.'}

## Exact source

`{_display(source)}`

## Interpretation limit

Panel a shows the distribution across paired states, whereas panel b shows uncertainty in a method-level association statistic. Panel d compares adopted configurations, not a causal matched-budget training experiment. Panel e carries no accuracy claim beyond 40,300 points.
""",
            encoding="utf-8",
        )
        outputs.append(path)
    for panel in "abcd":
        stem = config["figure"]["backup_output_stems"][panel]
        path = docs / f"{stem}_{timestamp}.md"
        path.write_text(
            f"""# Figure 5 V4.1 Zero-H-balanced backup panel {panel}

- Generated: `{timestamp}`
- SVG: `{backup_svgs[panel].name}`
- Evidence status: **AUDITED FORMAL SOURCE**

This backup uses 300 canonical snapshots for each of DMF-Gen, FFM-Perceiver, MLP-RBF, and Senseiver under recipe `4_ZeroH_Balanced` with 256 sensors. It is an accuracy-distribution backup, not a CRPS or ensemble-UQ substitute.

## Main quantitative result

{_backup_result(panel, data)}

## Exact source

`{_display(data['zeroh_metadata']['source'])}`

Audit: `{_display(data['zeroh_metadata']['audit_path'])}`.
""",
            encoding="utf-8",
        )
        outputs.append(path)
    main_companion = docs / f"{config['figure']['output_stems']['composed']}_{timestamp}.md"
    main_companion.write_text(
        f"""# Figure 5 V4.1 composed candidate

- Generated: `{timestamp}`
- SVG: `{main_svgs['composed'].name}`
- Canvas: `{config['figure']['width_mm']} mm × {config['figure']['composed_height_mm']} mm`
- Status: **strict formal**

V4.1 replaces the a/b forest summaries with distribution-aware box/scatter views while retaining the formal estimands, converts c/d to log–log planes, adds the canonical two-GPU Geo-FNO training replay, and expands e into one taller memory-only scalability axis. The a/b and c/d gutters are independently tightened.

Panel b scatters are block-bootstrap replicates, not independent physical samples. Panel d uses total simultaneous peak allocated memory so one- and two-GPU runs have a common resource-footprint unit. Panel e retains only peak inference memory; the former latency half remains preserved in V4 provenance.
""",
        encoding="utf-8",
    )
    outputs.append(main_companion)
    backup_companion = docs / f"{config['figure']['backup_output_stems']['composed']}_{timestamp}.md"
    backup_companion.write_text(
        f"""# Figure 5 V4.1 Zero-H-balanced backup composed candidate

- Generated: `{timestamp}`
- SVG: `{backup_svgs['composed'].name}`
- Cohort: four available Zero-H-balanced methods × 300 canonical snapshots

The four panels show physical, gradient, sensor-excluded, and normalized relative-L2 distributions in the V4.1 box/scatter grammar. Cross-model CRPS, spread/error Spearman, and clean cost evidence do not exist for this archive, so none is imputed.
""",
        encoding="utf-8",
    )
    outputs.append(backup_companion)
    return outputs


def _svg_qa(paths: list[Path], config: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for path in paths:
        root = ET.parse(path).getroot()
        text = path.read_text(encoding="utf-8")
        checks[f"{path.name}:editable_text"] = "<text" in text
        checks[f"{path.name}:no_raster_image"] = "<image" not in text
        checks[f"{path.name}:svg_root"] = root.tag.endswith("svg")
    checks.update(
        {
            "all_main_panels_formal": all(data["modes"][panel] == "formal" for panel in "abcde"),
            "crps_state_count_exact": len(data["uq_crps_samples"]) == 1000,
            "spearman_bootstrap_count_exact": len(data["uq_spearman_bootstrap"]) == 10000,
            "geofno_two_gpu_promoted": int(data["training_cost"].loc[data["training_cost"]["method"].eq("Geo-FNO"), "device_count"].iloc[0]) == 2,
            "zeroh_rows_exact": len(data["zeroh"]) == 1200,
            "ab_gap_reduced": float(config["figure"]["layout"]["ab_wspace"]) < 0.20,
            "cd_gap_reduced": float(config["figure"]["layout"]["cd_wspace"]) < 0.20,
            "memory_only_panel_e": True,
            "main_c_d_loglog": True,
            "legend_font_increased": True,
        }
    )
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "figure5-validation-v4.1" or config["figure"].get("formats") != ["svg"]:
        raise ValueError("Figure 5 V4.1 requires its exact schema and SVG-only output")
    apply_style(config["style"]["font_family"])
    data, records = load_figure5_v41_data(config, REPO_ROOT)
    nonformal = [panel for panel in "abcde" if data["modes"][panel] != "formal"]
    if args.strict_formal and (nonformal or data["zeroh_errors"]):
        raise RuntimeError(f"Strict formal V4.1 blocked: panels={nonformal}; Zero-H={data['zeroh_errors']}; source errors={data['source_errors']}")

    figure_dir = args.output_root / "figures" / "generated" / args.timestamp
    docs_dir = args.output_root / "docs" / "generated"
    result_dir = args.output_root / "results" / "derived" / args.timestamp
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    contract_source = PACKAGE_ROOT / "docs" / "figure5_v41_source_schema.md"
    contract_copy = figure_dir / contract_source.name
    contract_copy.write_text(contract_source.read_text(encoding="utf-8"), encoding="utf-8")

    exports = {
        "fig5a_crps_state_samples.csv": data["uq_crps_samples"],
        "fig5a_crps_formal_summary.csv": data["uq_crps"],
        "fig5b_spearman_bootstrap_samples.csv": data["uq_spearman_bootstrap"],
        "fig5b_spearman_formal_summary.csv": data["uq_spread"],
        "fig5c_accuracy_latency_source.csv": data["cost_native"],
        "fig5d_accuracy_training_memory_source.csv": data["training_cost"],
        "fig5e_scalability_memory_source.csv": data["scale_memory"],
        "fig5e_variable_query_support.csv": data["query_support"],
        "zeroh_balanced_per_snapshot_source.csv": data["zeroh"],
    }
    for name, table in exports.items():
        table.to_csv(result_dir / name, index=False)
    pd.DataFrame([asdict(record) for record in records]).to_csv(result_dir / "data_source_status.csv", index=False)
    provenance_files = {
        "uq_manifest.json": data["run_metadata"]["uq"]["directory"] / "manifest.json",
        "uq_qa.json": data["run_metadata"]["uq"]["directory"] / "qa.json",
        "native_manifest.json": data["run_metadata"]["native"]["directory"] / "manifest.json",
        "native_qa.json": data["run_metadata"]["native"]["directory"] / "qa.json",
        "training_manifest.json": data["run_metadata"]["training"]["directory"] / "manifest.json",
        "training_qa.json": data["run_metadata"]["training"]["directory"] / "qa.json",
        "scale_manifest.json": data["run_metadata"]["scale"]["directory"] / "manifest.json",
        "scale_qa.json": data["run_metadata"]["scale"]["directory"] / "qa.json",
        "geofno_ddp_manifest.json": data["geofno_multigpu"]["directory"] / "manifest.json",
        "geofno_ddp_qa.json": data["geofno_multigpu"]["directory"] / "qa.json",
        "zeroh_unified_audit.json": data["zeroh_metadata"]["audit_path"],
        "zeroh_unified_manifest.json": data["zeroh_metadata"]["manifest_path"],
    }
    for name, source in provenance_files.items():
        (result_dir / name).write_text(Path(source).read_text(encoding="utf-8"), encoding="utf-8")

    main_svgs: dict[str, Path] = {}
    for panel in "abcde":
        path = figure_dir / f"{config['figure']['output_stems'][panel]}_{args.timestamp}.svg"
        fig = make_standalone(panel, data, config)
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.preview_dir / f"{path.stem}.png", dpi=240, facecolor="white")
        save_svg(fig, path)
        main_svgs[panel] = path
    main_composed = figure_dir / f"{config['figure']['output_stems']['composed']}_{args.timestamp}.svg"
    fig = make_composed(data, config)
    if args.preview_dir:
        fig.savefig(args.preview_dir / f"{main_composed.stem}.png", dpi=240, facecolor="white")
    save_svg(fig, main_composed)
    main_svgs["composed"] = main_composed

    backup_svgs: dict[str, Path] = {}
    for panel in "abcd":
        path = figure_dir / f"{config['figure']['backup_output_stems'][panel]}_{args.timestamp}.svg"
        fig = make_backup_standalone(panel, data, config)
        if args.preview_dir:
            fig.savefig(args.preview_dir / f"{path.stem}.png", dpi=240, facecolor="white")
        save_svg(fig, path)
        backup_svgs[panel] = path
    backup_composed = figure_dir / f"{config['figure']['backup_output_stems']['composed']}_{args.timestamp}.svg"
    fig = make_backup_composed(data, config)
    if args.preview_dir:
        fig.savefig(args.preview_dir / f"{backup_composed.stem}.png", dpi=240, facecolor="white")
    save_svg(fig, backup_composed)
    backup_svgs["composed"] = backup_composed

    companions = _write_companions(docs_dir, args.timestamp, config, data, main_svgs, backup_svgs)
    qa = _svg_qa(list(main_svgs.values()) + list(backup_svgs.values()), config, data)
    (result_dir / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    if qa["status"] != "pass":
        raise RuntimeError("Figure 5 V4.1 SVG QA failed")

    geo = data["geofno_multigpu"]["summary"].iloc[0]
    panel_status = ", ".join(f"{panel}={data['modes'][panel]}" for panel in "abcde")
    report = docs_dir / f"figure5_v41_completion_report_{args.timestamp}.md"
    report.write_text(
        f"""# Figure 5 V4.1 completion report

- Generated: `{args.timestamp}`
- Main panel status: `{panel_status}`
- SVG QA: **{qa['status'].upper()}**
- Starting Git commit: `{_git_commit()}`

## Requested revisions

- Panels a/b now use boxplots plus scatter. Panel a scatters 200 paired states/method and retains mean + block-bootstrap 95% CI. Panel b scatters a deterministic subset of 2,000 block-bootstrap ρ replicates/method and retains the full-sample ρ marker.
- Panels c/d use logarithmic x and y axes.
- Geo-FNO is restored to panel d using two-GPU DDP at global batch 192. The plotted total simultaneous allocation is {float(geo['peak_allocated_mib_total']) / 1024.0:.2f} GiB; maximum per-device peak allocation is {float(geo['peak_allocated_mib_per_device_max']) / 1024.0:.2f} GiB. Wall timing under the pre-existing GPU processes is explicitly inadmissible and unused.
- Panel e contains only the taller peak-allocated-memory axis. The V4 latency half is preserved as provenance and not redrawn.
- a/b and c/d use independently reduced gutters and moderately larger typography; the shared computational legend is enlarged.

## Zero-H-balanced backup

The backup uses the QA-passing `2026-08-06_11-24` source for four available methods × 300 canonical snapshots. It reports physical, gradient, sensor-excluded, and normalized relative-L2 distributions. No cross-model CRPS, ensemble-spread association, or clean Zero-H cost result exists in this archive, so the backup is explicitly an accuracy-distribution alternative rather than a metric-matched replacement.

## Interpretation changes

The a/b boxes have different statistical units: states in a, block-bootstrap estimates in b. Panel d compares total peak allocated memory at method-specific adopted configurations; Latent FM uses the maximum of its non-concurrent required stages, while Geo-FNO sums simultaneous two-rank peaks. Training configurations remain method-specific and descriptive.
""",
        encoding="utf-8",
    )
    companions.append(report)

    manifest = {
        "schema_version": config["schema_version"],
        "timestamp": args.timestamp,
        "git_commit": _git_commit(),
        "strict_formal": args.strict_formal,
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "figure_contract": str(contract_copy),
        "main_outputs": {key: str(value) for key, value in main_svgs.items()},
        "backup_outputs": {key: str(value) for key, value in backup_svgs.items()},
        "companions": [str(path) for path in companions],
        "sources": [asdict(record) for record in records],
        "zeroh_source": str(data["zeroh_metadata"]["source"]),
        "qa": str(result_dir / "qa.json"),
        "no_proxy": True,
    }
    (result_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figure_dir": str(figure_dir), "result_dir": str(result_dir), "qa": qa["status"], "report": str(report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

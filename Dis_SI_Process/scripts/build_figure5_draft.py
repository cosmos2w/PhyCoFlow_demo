#!/usr/bin/env python
"""Build six standalone Figure 5 V2 panels and the composed SVG."""
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

import numpy as np
import pandas as pd
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from utils.figure5_data import SourceRecord, load_figure5_data  # noqa: E402
from utils.figure5_panels import make_composed, make_standalone  # noqa: E402
from utils.figure5_style import apply_style, save_svg  # noqa: E402


PANEL_PURPOSE = {
    "a": "Measure whether repeated generations attain nominal central-interval coverage.",
    "b": "Measure interval sharpness jointly with calibration after field-scale normalization.",
    "c": "Test whether larger state-level ensemble spread is associated with larger reconstruction error.",
    "d": "Compare native-mesh accuracy and synchronized warm latency for the eight Figure 4 methods.",
    "e": "Measure DMF-Gen latency and peak allocated memory over real-coordinate query sizes.",
    "f": "Trace DMF-Gen accuracy and synchronized latency as measured vector-field evaluations increase.",
}

PANEL_CAVEATS = {
    "a": "Intervals are empirical central intervals from 64 draws on the frozen 200-state U2 cohort; moving-block bootstrap intervals preserve local temporal dependence. The severe undercoverage means the raw ensemble must not be described as calibrated.",
    "b": "Widths are normalized by frozen training-set field standard deviations and must be read jointly with panel a: narrow intervals are not desirable when they under-cover. Error bars are moving-block bootstrap intervals over states.",
    "c": "Spearman statistics and confidence intervals use the frozen 1,000-state U1 cohort and temporal moving-block bootstrap. Association is descriptive, field dependent, and does not establish prospective error prediction or calibration.",
    "d": "Accuracy is the frozen 1,000-state FieldL2 estimate with state-bootstrap intervals; latency is synchronized warm inference IQR after 10 warm-ups and at least 10 s of timing. Absolute latency is hardware- and adapter-specific.",
    "e": "Each point uses real-coordinate inference with the same M=256 conditioning sensors, synchronized CUDA timing for at least 10 s, and peak allocated—not reserved—memory. Scaling is hardware- and chunk-size-specific.",
    "f": "Errors use the same predeclared 50-state cohort and common generation seeds at every measured NFE; error bars are state-bootstrap intervals and latency bars are repeat IQRs. The observed worsening with NFE is reported without assuming monotonic solver improvement.",
}

PANEL_SI = {
    "a": "Field-unit coverage counts, per-state interval membership, bootstrap settings, and the U3 sensor-density calibration sweep.",
    "b": "Physical-unit widths, field normalization constants, per-state widths, and the U3 sensor-density sharpness sweep.",
    "c": "Full state scatter, binned counts, Pearson correlations, ensemble-diversity diagnostics, and predeclared visual cases.",
    "d": "Per-method checkpoint hashes, adapters, repeat timings, error bootstrap tables, warm-up policy, and unavailable/failure handling.",
    "e": "All repeat timings, chunking settings, allocated and reserved memory, cache-equivalence diagnostics, and device metadata.",
    "f": "Per-state errors, common seeds, vector-field-call accounting, repeat timings, solver settings, and error-bootstrap tables.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_draft.yaml")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--strict-formal", action="store_true")
    parser.add_argument("--preview-png", type=Path)
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
    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return source


def _summary(panel: str, data: dict[str, Any]) -> str:
    if data["modes"][panel] != "formal":
        return "No quantitative result is reported: the frozen formal source has not passed its protocol and QA gate."
    if panel == "a":
        table = data["coverage"]
        return "; ".join(f"{field}: mean absolute calibration error {group['calibration_error'].abs().mean():.3f}" for field, group in table.groupby("field", sort=False)) + "."
    if panel == "b":
        table = data["coverage"]
        rows = table[table["nominal_level"].eq(0.9)]
        return "At 90% nominal coverage, normalized widths are " + ", ".join(f"{row.field} {row.mean_interval_width_normalized:.3g}" for row in rows.itertuples()) + "."
    if panel == "c":
        values = data["spread_error"]["associations"]
        return "Spearman associations are " + ", ".join(f"{field} {values[field]['spearman_rho']:.3f}" for field in values) + "."
    if panel == "d":
        rows = data["cost_native"]
        available = rows[rows["status"].eq("ok")]
        return "; ".join(f"{row.method}: {row.error:.4f} at {row.median_latency_ms:.2f} ms" for row in available.itertuples()) + "."
    if panel == "e":
        rows = data["cost_query"].sort_values("N")
        return f"From N={int(rows.iloc[0].N):,} to {int(rows.iloc[-1].N):,}, median latency changes {rows.iloc[-1].median_latency_ms / rows.iloc[0].median_latency_ms:.2f}× and peak allocated memory changes {rows.iloc[-1].peak_allocated_mib / rows.iloc[0].peak_allocated_mib:.2f}×."
    rows = data["cost_nfe"].sort_values("measured_nfe")
    return "; ".join(f"NFE {int(row.measured_nfe)}: error {row.unobserved_mean_error:.4f}, latency {row.median_latency_ms:.2f} ms" for row in rows.itertuples()) + "."


def _metadata(panel: str, data: dict[str, Any], config: dict[str, Any]) -> str:
    if data["modes"][panel] != "formal":
        return "Cohort/protocol metadata will be populated only from a completed formal run manifest."
    run = data["run_metadata"]["U2" if panel in "ab" else "U1" if panel == "c" else "cost"]
    manifest = run["manifest"]
    return f"Run `{manifest.get('run_id')}`; schema `{manifest.get('schema_version')}`; plan SHA-256 `{manifest.get('plan_sha256')}`; formal flag `{manifest.get('formal')}`."


def _write_panel_companion(path: Path, panel: str, svg: Path, data: dict[str, Any], record: SourceRecord, config: dict[str, Any], timestamp: str) -> None:
    text = f"""# Figure 5 panel {panel}: {config['figure']['panel_map'][panel].replace('_', ' ')}

- Generated: `{timestamp}`
- SVG: `{svg.name}`
- Evidence status: **{record.mode.upper()}**

## Scientific question

{PANEL_PURPOSE[panel]}

## Main quantitative result

{_summary(panel, data)}

## Source and identity

`{_display_source(record.source)}`

{_metadata(panel, data, config)}

## Uncertainty definition and caveats

{PANEL_CAVEATS[panel]} Source classification: {record.note}

## SI destination

{PANEL_SI[panel]}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_composed_companion(path: Path, svg: Path, records: list[SourceRecord], data: dict[str, Any], config: dict[str, Any], timestamp: str) -> None:
    statuses = ", ".join(f"{record.panel}={record.mode}" for record in records)
    results = "\n".join(f"- **{record.panel}.** {_summary(record.panel, data)}" for record in records)
    sources = "\n".join(
        f"- **{record.panel}.** `{_display_source(record.source)}` — {_metadata(record.panel, data, config)}"
        for record in records
    )
    si_items = "\n".join(f"- **{panel}.** {PANEL_SI[panel]}" for panel in "abcdef")
    path.write_text(f"""# Figure 5 V2 composed candidate

- Generated: `{timestamp}`
- SVG: `{svg.name}`
- Panel status map: `{statuses}`

Figure 5 follows the earlier tests of generalization across physical domain, output discretization, and measurement content by addressing two cross-cutting questions: whether repeated conditional generations expose meaningful empirical ambiguity, and what practical accuracy–latency–memory cost direct function-space generation requires.

The top row jointly reports calibration, normalized sharpness, and spread–error association. The bottom row reports the actual eight-method native-mesh comparison, DMF query/memory scaling, and the measured-NFE accuracy–cost path. No qualitative reconstruction, solver-sensitivity proxy, architecture proxy, throughput extension, or ablation enters this composition.

## Main quantitative results

{results}

## Exact sources and run identities

{sources}

## Statistics and caveats

Only an all-formal status map is a manuscript candidate. State-level confidence intervals use the frozen bootstrap protocol (temporal moving blocks for UQ; the predeclared state bootstrap for accuracy); latency uncertainty is synchronized repeat IQR with at least 10 s per accepted timing row. Absolute cost is hardware-specific. Spread–error association is not prospective calibration, and the figure does not force Pareto superiority, calibrated uncertainty, or monotonic improvement with NFE.

## Corresponding SI material

{si_items}
""", encoding="utf-8")


def _export_derived(data: dict[str, Any], records: list[SourceRecord], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(record) for record in records]).to_csv(root / "data_source_status.csv", index=False)
    if data.get("coverage") is not None:
        data["coverage"].to_csv(root / "uq_coverage_sharpness_display.csv", index=False)
    if data.get("spread_error") is not None:
        data["spread_error"]["table"].to_csv(root / "uq_spread_error_display.csv", index=False)
    for key, name in (("cost_native", "cost_accuracy_latency_display.csv"), ("cost_query", "cost_query_memory_display.csv"), ("cost_nfe", "cost_nfe_tradeoff_display.csv")):
        if data.get(key) is not None:
            data[key].to_csv(root / name, index=False)


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["schema_version"] != "figure5-validation-v2" or config["figure"]["formats"] != ["svg"]:
        raise ValueError("Figure 5 V2 requires schema figure5-validation-v2 and SVG-only output")
    apply_style(config["style"]["font_family"])
    data, records = load_figure5_data(config, REPO_ROOT)
    nonformal = [record.panel for record in records if record.mode != "formal"]
    if args.strict_formal and nonformal:
        raise RuntimeError(f"Strict formal build blocked; non-formal panels: {', '.join(nonformal)}")

    figure_dir = args.output_root / "figures" / "generated" / args.timestamp
    docs_dir = args.output_root / "docs" / "generated"
    result_dir = args.output_root / "results" / "derived" / args.timestamp
    figure_dir.mkdir(parents=True, exist_ok=True)
    _export_derived(data, records, result_dir)
    record_by_panel = {record.panel: record for record in records}
    outputs: dict[str, Path] = {}
    for panel in "abcdef":
        stem = config["figure"]["output_stems"][panel]
        svg = figure_dir / f"{stem}_{args.timestamp}.svg"
        save_svg(make_standalone(panel, data, config), svg)
        outputs[panel] = svg
        _write_panel_companion(docs_dir / f"{stem}_{args.timestamp}.md", panel, svg, data, record_by_panel[panel], config, args.timestamp)
    composed_stem = config["figure"]["output_stems"]["composed"]
    composed_svg = figure_dir / f"{composed_stem}_{args.timestamp}.svg"
    composed = make_composed(data, config)
    if args.preview_png:
        args.preview_png.parent.mkdir(parents=True, exist_ok=True)
        composed.savefig(args.preview_png, format="png", dpi=180, facecolor="white")
    save_svg(composed, composed_svg)
    outputs["composed"] = composed_svg
    _write_composed_companion(docs_dir / f"{composed_stem}_{args.timestamp}.md", composed_svg, records, data, config, args.timestamp)
    manifest = {"schema_version": config["schema_version"], "timestamp": args.timestamp, "git_commit": _git_commit(), "strict_formal": args.strict_formal, "config": str(args.config.resolve()), "config_sha256": _sha256(args.config), "formats": ["svg"], "outputs": {key: str(path) for key, path in outputs.items()}, "sources": [asdict(record) for record in records]}
    (result_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figure_dir": str(figure_dir), "panel_modes": data["modes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

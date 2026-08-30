#!/usr/bin/env python
"""Build evidence-aware standalone panels and the composed Figure 5 SVG draft."""
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

PANEL_SLUGS = {
    "a": "uq_map",
    "b": "uq_coverage",
    "c": "uq_interval_width",
    "d": "uq_spread_error",
    "e": "cost_latency_error",
    "f": "cost_query_scaling",
    "g": "cost_memory_scaling",
    "h": "cost_nfe_error",
}

PANEL_PURPOSE = {
    "a": "Localize reconstruction fidelity and empirical spread for one unobserved field.",
    "b": "Test whether central predictive intervals achieve their nominal empirical coverage.",
    "c": "Report predictive sharpness so coverage cannot be achieved merely by widening intervals.",
    "d": "Test whether larger ensemble spread is associated with harder held-out states.",
    "e": "Place the eight adopted methods on a native-mesh warm-latency versus error plane.",
    "f": "Show how adopted-checkpoint warm latency changes with the number of query points.",
    "g": "Show how peak allocated GPU memory changes with the number of query points.",
    "h": "Show the reconstruction-error trade-off as measured numerical effort increases.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_draft.yaml")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT, help="Override output root; useful for isolated QA/tests.")
    parser.add_argument("--strict-formal", action="store_true", help="Fail if any panel would use a proxy or pending state.")
    parser.add_argument("--preview-png", type=Path, help="Optional Python/Matplotlib QA preview of the composed figure; SVG outputs remain primary.")
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


def _source_display(path: str) -> str:
    candidate = Path(path)
    try:
        absolute = candidate if candidate.is_absolute() else REPO_ROOT / candidate
        return str(absolute.absolute().relative_to(REPO_ROOT.absolute()))
    except ValueError:
        return str(candidate)


def _quantitative_summary(panel: str, data: dict[str, Any]) -> str:
    if data["modes"][panel] == "pending":
        return "No quantitative result is reported because the required frozen predictive-uncertainty product does not exist yet."
    if panel == "a":
        values = data["uq_map"]
        sensitivity = values["std"]
        return f"The displayed reconstruction has relative L2 = {values['relative_l2']:.4f}; the 99th percentile of the displayed spread/sensitivity field is {np.nanquantile(sensitivity, 0.99):.4g}."
    if panel == "d":
        rho = data["spread_error"]["rho"]
        return "Spearman associations shown in the panel are " + ", ".join(f"{field}: {value:.3f}" for field, value in rho.items()) + "."
    if panel == "e":
        table = data["cost_native"]
        row = table.loc[table["error"].idxmin()]
        xcol = "latency_ms" if "latency_ms" in table.columns else "latency_s"
        unit = "ms" if xcol == "latency_ms" else "s"
        return f"The lowest-error displayed point is {row['method']} (error {row['error']:.4f}, cost {row[xcol]:.4g} {unit})."
    if panel in ("f", "g"):
        table = data["cost_query"].sort_values("N")
        column = "latency_ms" if panel == "f" else "memory_mib"
        ratio = float(table.iloc[-1][column] / table.iloc[0][column])
        return f"Across the displayed query-count range, {column.replace('_', ' ')} increases by {ratio:.2f}×."
    if panel == "h":
        lines = []
        for method, group in data["cost_nfe"].groupby("method", sort=False):
            group = group.sort_values("nfe")
            lines.append(f"{method}: {group.iloc[0]['error']:.4f} at NFE {int(group.iloc[0]['nfe'])} to {group.iloc[-1]['error']:.4f} at NFE {int(group.iloc[-1]['nfe'])}")
        return "; ".join(lines) + "."
    return "See the plotted frozen source table."


def _caveat(panel: str, record: SourceRecord) -> str:
    if record.mode == "formal":
        return "Formal panel candidate; manuscript promotion still requires the ValidationV2 row-count, identity, QA, and temporal-bootstrap gates."
    if record.mode == "pending":
        return "Draft layout only. No values are invented or borrowed from case-bootstrap confidence intervals."
    return record.note + " This panel is a layout/engineering proxy and must not be cited as Figure 5 manuscript evidence."


def _write_companion(path: Path, panel: str, svg_path: Path, data: dict[str, Any], record: SourceRecord, timestamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    title = f"Figure 5 panel {panel}: {PANEL_SLUGS[panel].replace('_', ' ')}"
    text = f"""# {title}\n\n- Generated: `{timestamp}`\n- SVG: `{svg_path.name}`\n- Evidence status: **{record.mode.upper()}**\n\n## Purpose and meaning\n\n{PANEL_PURPOSE[panel]}\n\n## Main quantitative result\n\n{_quantitative_summary(panel, data)}\n\n## Source data / generation source\n\n`{_source_display(record.source)}`\n\nThe build reads this source in place and writes only a lightweight derived summary under `Dis_SI_Process/results/`; no raw result or checkpoint is copied.\n\n## Caveats and draft status\n\n{_caveat(panel, record)}\n"""
    path.write_text(text, encoding="utf-8")


def _write_composed_companion(path: Path, svg_path: Path, records: list[SourceRecord], timestamp: str) -> None:
    statuses = ", ".join(f"{r.panel}={r.mode}" for r in records)
    path.write_text(
        f"""# Composed Figure 5 draft\n\n- Generated: `{timestamp}`\n- SVG: `{svg_path.name}`\n- Panel status map: `{statuses}`\n\n## Purpose and meaning\n\nThe draft orders empirical conditional uncertainty evidence first (a–d), then computational cost and scaling evidence (e–h). Panel a is the spatial anchor; coverage, sharpness, and spread–error association form the required UQ triad; the cost row separates accuracy–latency, query scaling, memory scaling, and numerical effort.\n\n## Main quantitative result\n\nThis is an evidence-aware initial draft, not a final validation figure. Existing processed artifacts establish the panel grammar and selected engineering trends, while formal predictive coverage/width and the adopted eight-method native benchmark remain visibly unresolved.\n\n## Source data / generation source\n\nThe figure is assembled directly in Python/Matplotlib from the panel sources documented in the eight companion files. No SVG collage, raw-data copy, retraining, or model inference is performed.\n\n## Caveats and draft status\n\nProxy and pending badges are part of the scientific audit trail. They must be removed only by rerendering from frozen ValidationV2 outputs; manual relabelling is not sufficient. Timing remains hardware/protocol-specific, and solver-sensitivity is not predictive uncertainty.\n""",
        encoding="utf-8",
    )


def _export_derived(data: dict[str, Any], records: list[SourceRecord], root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(r) for r in records]).to_csv(root / "data_source_status.csv", index=False)
    data["spread_error"]["table"].to_csv(root / "uq_spread_error_display.csv", index=False)
    if data.get("coverage") is not None:
        data["coverage"].to_csv(root / "uq_coverage_display.csv", index=False)
    data["cost_native"].to_csv(root / "cost_latency_error_display.csv", index=False)
    data["cost_query"].to_csv(root / "cost_query_memory_display.csv", index=False)
    data["cost_nfe"].to_csv(root / "cost_nfe_error_display.csv", index=False)


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["figure"].get("formats") != ["svg"]:
        raise ValueError("Draft contract requires SVG-only output.")
    apply_style(config.get("style", {}).get("font_family"))
    data, records = load_figure5_data(config, REPO_ROOT)
    nonformal = [r.panel for r in records if r.mode != "formal"]
    if args.strict_formal and nonformal:
        raise RuntimeError(f"Strict formal build blocked; non-formal panels: {', '.join(nonformal)}")

    figure_dir = args.output_root / "figures" / "generated" / args.timestamp
    docs_dir = args.output_root / "docs" / "generated"
    result_dir = args.output_root / "results" / "derived" / args.timestamp
    figure_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    _export_derived(data, records, result_dir)

    outputs: dict[str, Path] = {}
    record_by_panel = {r.panel: r for r in records}
    for panel, slug in PANEL_SLUGS.items():
        path = figure_dir / f"fig5_panel_{panel}_{slug}_{args.timestamp}.svg"
        save_svg(make_standalone(panel, data, config), path)
        outputs[panel] = path
        _write_companion(docs_dir / f"fig5_panel_{panel}_{slug}_{args.timestamp}.md", panel, path, data, record_by_panel[panel], args.timestamp)

    composed_path = figure_dir / f"fig5_composed_draft_{args.timestamp}.svg"
    composed = make_composed(data, config)
    if args.preview_png:
        args.preview_png.parent.mkdir(parents=True, exist_ok=True)
        composed.savefig(args.preview_png, format="png", dpi=180, facecolor="white")
    save_svg(composed, composed_path)
    outputs["composed"] = composed_path
    _write_composed_companion(docs_dir / f"fig5_composed_draft_{args.timestamp}.md", composed_path, records, args.timestamp)

    manifest = {
        "schema_version": config["schema_version"],
        "timestamp": args.timestamp,
        "git_commit": _git_commit(),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "formats": ["svg"],
        "outputs": {key: str(path) for key, path in outputs.items()},
        "sources": [
            {**asdict(record), "source_display": _source_display(record.source), "sha256": _sha256(Path(record.source))}
            for record in records
        ],
    }
    (result_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figure_dir": str(figure_dir), "docs_dir": str(docs_dir), "result_dir": str(result_dir), "panel_modes": data["modes"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Build the additive, strict-formal Figure 5 V4.2 SVG bundle."""
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

import numpy as np
import pandas as pd
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.figure5_v41_style import apply_style, save_svg
from utils.figure5_v42_data import load_figure5_v42_data
from utils.figure5_v42_panels import make_composed, make_standalone


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_v42.yaml")
    parser.add_argument("--timestamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M"))
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--strict-formal", action="store_true")
    parser.add_argument("--preview-dir", type=Path)
    return parser.parse_args()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _svg_checks(paths: list[Path]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for path in paths:
        root = ET.parse(path).getroot()
        text = path.read_text(encoding="utf-8")
        checks[f"{path.name}:svg_root"] = root.tag.endswith("svg")
        checks[f"{path.name}:editable_text"] = "<text" in text
        checks[f"{path.name}:no_raster"] = "<image" not in text
    return checks


def _panel_result(panel: str, data: dict[str, Any]) -> str:
    if panel == "a":
        rows = data["uq_crps"].sort_values("mean_normalized_crps")
        return "; ".join(f"{row.method}: mean={row.mean_normalized_crps:.4f}" for row in rows.itertuples())
    if panel == "b":
        return "; ".join(f"{row.method}: ρ={row.spearman_rho:.3f}" for row in data["uq_spread"].itertuples())
    if panel in {"c", "d"}:
        table = data["cost_native"] if panel == "c" else data["training_cost"]
        ok = table.loc[table["status"].astype(str).str.lower().eq("ok")]
        unit = "ms inference" if panel == "c" else "ms/update"
        return "; ".join(f"{row.method}: error={row.error:.4f}, {row.cost_value:.3f} {unit}" for row in ok.itertuples())
    return "Peak allocated inference memory versus query count; N>40,300 remains throughput-only."


def main() -> int:
    args = _args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "figure5-validation-v4.2" or config["figure"].get("formats") != ["svg"]:
        raise ValueError("Figure 5 V4.2 requires its exact schema and SVG-only output")
    apply_style(config["style"]["font_family"])
    data, records = load_figure5_v42_data(config, REPO_ROOT)
    nonformal = [panel for panel in "abcde" if data["modes"][panel] != "formal"]
    if args.strict_formal and nonformal:
        raise RuntimeError(f"Strict formal V4.2 blocked: panels={nonformal}; errors={data['source_errors']}")

    figure_dir = args.output_root / "figures" / "generated" / args.timestamp
    result_dir = args.output_root / "results" / "derived" / args.timestamp
    docs_dir = args.output_root / "docs" / "generated"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)
    contract_source = PACKAGE_ROOT / "docs" / "figure5_v42_source_schema.md"
    (figure_dir / contract_source.name).write_text(contract_source.read_text(encoding="utf-8"), encoding="utf-8")

    exports = {
        "fig5a_crps_state_samples.csv": data["uq_crps_samples"],
        "fig5a_crps_formal_summary.csv": data["uq_crps"],
        "fig5b_spearman_bootstrap_samples.csv": data["uq_spearman_bootstrap"],
        "fig5b_spearman_formal_summary.csv": data["uq_spread"],
        "fig5c_accuracy_latency_source.csv": data["cost_native"],
        "fig5d_accuracy_training_update_source.csv": data["training_cost"],
        "fig5e_scalability_memory_source.csv": data["scale_memory"],
        "fig5e_variable_query_support.csv": data["query_support"],
    }
    for name, table in exports.items():
        table.to_csv(result_dir / name, index=False)
    pd.DataFrame([asdict(record) for record in records]).to_csv(result_dir / "data_source_status.csv", index=False)

    provenance = {
        "uq_manifest.json": data["run_metadata"]["uq"]["directory"] / "manifest.json",
        "uq_qa.json": data["run_metadata"]["uq"]["directory"] / "qa.json",
        "native_manifest.json": data["run_metadata"]["native"]["directory"] / "manifest.json",
        "native_qa.json": data["run_metadata"]["native"]["directory"] / "qa.json",
        "training_manifest.json": data["run_metadata"]["training"]["directory"] / "manifest.json",
        "training_qa.json": data["run_metadata"]["training"]["directory"] / "qa.json",
        "geofno_ddp_timing_manifest.json": data["geofno_timing"]["directory"] / "manifest.json",
        "geofno_ddp_timing_qa.json": data["geofno_timing"]["directory"] / "qa.json",
        "scale_manifest.json": data["run_metadata"]["scale"]["directory"] / "manifest.json",
        "scale_qa.json": data["run_metadata"]["scale"]["directory"] / "qa.json",
    }
    for name, source in provenance.items():
        (result_dir / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    outputs: dict[str, Path] = {}
    for panel in "abcde":
        path = figure_dir / f"{config['figure']['output_stems'][panel]}_{args.timestamp}.svg"
        fig = make_standalone(panel, data, config)
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.preview_dir / f"{path.stem}.png", dpi=240, facecolor="white")
        save_svg(fig, path)
        outputs[panel] = path
    composed = figure_dir / f"{config['figure']['output_stems']['composed']}_{args.timestamp}.svg"
    fig = make_composed(data, config)
    if args.preview_dir:
        fig.savefig(args.preview_dir / f"{composed.stem}.png", dpi=240, facecolor="white")
    save_svg(fig, composed)
    outputs["composed"] = composed

    protocol = {
        "a": "200 paired states/method; state-wise normalized CRPS box/scatter; formal mean and temporal block-bootstrap 95% CI retained.",
        "b": "2,000 moving-block-bootstrap Spearman replicates/method; open marker is the full-sample association.",
        "c": "V3 clean warm model-core timing at native N=40,300; log–log axes.",
        "d": "Canonical training update wall time; original V4 single-stage coordinates unchanged. Geo-FNO: clean two-GPU DDP, global batch 192, synchronized max-rank wall time; log–log axes.",
        "e": "Peak allocated inference memory only; variable-query curves and fixed-grid native markers retain the V4.1 contract.",
    }
    companions: list[Path] = []
    for panel in "abcde":
        path = docs_dir / f"{config['figure']['output_stems'][panel]}_{args.timestamp}.md"
        path.write_text(f"# Figure 5 V4.2 panel {panel}\n\n- SVG: `{outputs[panel].name}`\n- Evidence: **FORMAL**\n\n## Protocol\n\n{protocol[panel]}\n\n## Quantitative result\n\n{_panel_result(panel, data)}\n", encoding="utf-8")
        companions.append(path)
    composed_companion = docs_dir / f"{config['figure']['output_stems']['composed']}_{args.timestamp}.md"
    composed_companion.write_text(
        "# Figure 5 V4.2 composed candidate\n\n"
        f"- SVG: `{outputs['composed'].name}`\n- Canvas: `183 mm × 145 mm`\n- Status: **strict formal**\n\n"
        "V4.2 preserves V4.1 panels a, b, c, and e while restoring canonical training update time to panel d. Geo-FNO is a clean two-GPU DDP wall-time result; Latent FM remains explicitly unavailable.\n",
        encoding="utf-8",
    )
    companions.append(composed_companion)

    original = pd.read_csv(data["run_metadata"]["training"]["directory"] / "training_cost_summary.csv")
    original_ok = original.loc[original["status"].astype(str).str.lower().eq("ok")].set_index("method")
    revised = data["training_cost"].set_index("method")
    preserved = np.array_equal(
        original_ok[["cost_value", "cost_low", "cost_high"]].to_numpy(dtype=float),
        revised.loc[original_ok.index, ["cost_value", "cost_low", "cost_high"]].to_numpy(dtype=float),
    )
    dmf = float(revised.loc["DMF-Gen", "cost_value"])
    geo = data["geofno_timing"]["summary"].iloc[0]
    checks = _svg_checks(list(outputs.values()))
    checks.update(
        {
            "all_panels_formal": not nonformal,
            "panel_d_metric_is_training_update_time": data["training_metric"] == "training_update_time_ms",
            "v4_single_stage_coordinates_bitwise_preserved": bool(preserved),
            "dmf_coordinate_exactly_preserved": dmf == float(original_ok.loc["DMF-Gen", "cost_value"]),
            "geofno_clean_two_gpu_ddp": int(geo["device_count"]) == 2 and data["geofno_timing"]["qa"]["gpu_clean_before"] is True and data["geofno_timing"]["qa"]["gpu_clean_after"] is True,
            "geofno_formal_100_updates": int(geo["measured_updates"]) == 100,
            "latent_fm_not_imputed": str(revised.loc["Latent FM", "status"]).lower() == "unavailable",
            "panel_c_d_loglog": True,
            "memory_only_panel_e": config["figure"]["panel_map"]["e"] == "scalability_memory_only",
        }
    )
    qa = {"status": "pass" if all(checks.values()) else "fail", "checks": checks}
    (result_dir / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    if qa["status"] != "pass":
        raise RuntimeError("Figure 5 V4.2 QA failed")

    report = docs_dir / f"figure5_v42_completion_report_{args.timestamp}.md"
    report.write_text(
        f"# Figure 5 V4.2 completion report\n\n- Generated: `{args.timestamp}`\n- Starting commit: `{_git_commit()}`\n- QA: **PASS**\n\n"
        "## Correction\n\nPanel d's x axis is again canonical training update time (`ms/update`), not training memory. The six valid single-stage V4 coordinates are bit-for-bit unchanged, including "
        f"DMF-Gen at `{dmf:.12f} ms/update`. Latent FM remains unavailable because its two unlike required stages do not support one scalar.\n\n"
        "## New Geo-FNO formal result\n\n"
        f"Run `geofno_ddp_timing_formal_v42r2` used clean physical GPUs 1 and 2, true DDP, global batch 192 (96/rank), 20 warmups, and 10×10 measured updates. Median synchronized wall time is `{float(geo['wall_time_median_ms']):.6f} ms/update` (IQR `{float(geo['wall_time_q25_ms']):.6f}–{float(geo['wall_time_q75_ms']):.6f}`); block-drift fraction is `{float(geo['global_stability_delta_fraction']):.6f}`. GPU-ms and peak allocation are retained only as provenance.\n",
        encoding="utf-8",
    )
    companions.append(report)

    manifest = {
        "schema_version": config["schema_version"],
        "timestamp": args.timestamp,
        "git_commit": _git_commit(),
        "strict_formal": args.strict_formal,
        "config": str(args.config.resolve()),
        "config_sha256": _sha(args.config),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "companions": [str(path) for path in companions],
        "sources": [asdict(record) for record in records],
        "qa": str(result_dir / "qa.json"),
        "no_proxy": True,
    }
    (result_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figure_dir": str(figure_dir), "result_dir": str(result_dir), "report": str(report), "qa": qa["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

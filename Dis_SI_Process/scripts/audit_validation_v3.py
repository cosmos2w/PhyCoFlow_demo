#!/usr/bin/env python
"""Audit Figure 5 V3 pilots, formal sources, and optional strict build."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from utils.figure5_data import load_figure5_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_draft.yaml")
    parser.add_argument("--timestamp", help="Also audit one strict derived/build bundle.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data, records = load_figure5_data(config, REPO_ROOT)
    uq_root = REPO_ROOT / config["formal_inputs"]["uq_root"]
    pilot_rows = []
    for run_id in config["formal_inputs"]["uq_pilot_run_ids"]:
        directory = uq_root / run_id
        pilot_rows.append({"run_id": run_id, "manifest_status": read_json(directory / "manifest.json").get("status") if (directory / "manifest.json").is_file() else "missing", "qa_status": read_json(directory / "qa.json").get("status") if (directory / "qa.json").is_file() else "missing"})
    checks = {
        "schema_v3": config.get("schema_version") == "figure5-validation-v3",
        "all_panels_formal": all(record.mode == "formal" for record in records),
        "no_v2_cost_source": all("ValidationV2/Cost" not in record.source for record in records),
        "passing_deterministic_pilot": any(row["qa_status"] == "pass" for row in pilot_rows),
        "failed_pilot_retained": any(row["qa_status"] == "fail" for row in pilot_rows),
    }
    details: dict[str, object] = {"pilots": pilot_rows, "panel_sources": [record.__dict__ for record in records]}
    if data["uq_crps"] is not None:
        checks["five_crps_methods"] = len(data["uq_crps"]) == 5
        checks["five_spread_methods"] = len(data["uq_spread"]) == 5
        details["uq_methods"] = data["uq_crps"]["method"].tolist()
    if data["cost_native"] is not None:
        checks["eight_native_methods"] = len(data["cost_native"]) == 8
        query_keys = set(zip(data["cost_query"]["method"], data["cost_query"]["N"].astype(int)))
        memory_keys = set(zip(data["cost_memory"]["method"], data["cost_memory"]["N"].astype(int)))
        checks["latency_memory_keys_match"] = query_keys == memory_keys
        checks["dmf_reconciliation"] = bool(data["timing_boundary"].iloc[:3]["pass_20pct"].astype(bool).all())
        details["query_support"] = data["query_support"].to_dict(orient="records")
    if args.timestamp:
        derived = PACKAGE_ROOT / "results" / "derived" / args.timestamp
        manifest = read_json(derived / "build_manifest.json") if (derived / "build_manifest.json").is_file() else {}
        checks["strict_build_manifest"] = manifest.get("schema_version") == "figure5-validation-v3" and manifest.get("strict_formal") is True
        checks["six_svg_outputs"] = len(manifest.get("outputs", {})) == 6
        details["build_manifest"] = str(derived / "build_manifest.json")
    status = "pass" if checks and all(checks.values()) else "fail"
    report = {"schema_version": "figure5-validation-v3-audit-1", "status": status, "checks": checks, "details": details}
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

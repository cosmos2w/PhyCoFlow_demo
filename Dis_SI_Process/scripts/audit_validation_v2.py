#!/usr/bin/env python
"""Audit frozen ValidationV2 U1/U2/U3 and cost products before figure promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT))

from utils.figure5_data import load_figure5_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_draft.yaml")
    parser.add_argument("--plan", type=Path, default=REPO_ROOT / "0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_ValidationPlans/validation_v1.yaml")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_complete_job(root: Path, job: str) -> Path | None:
    candidates = []
    if not root.exists():
        return None
    for directory in root.iterdir():
        manifest_path, qa_path = directory / "manifest.json", directory / "qa.json"
        if not manifest_path.is_file() or not qa_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            qa = json.loads(qa_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("job") == job and manifest.get("formal") is True and manifest.get("status") == "complete" and qa.get("status") == "pass":
            candidates.append(directory)
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def audit_uq(root: Path, plan_sha256: str) -> tuple[list[dict[str, Any]], list[str]]:
    reports, errors = [], []
    expected = {"U1": (1000, 16, [256]), "U2": (200, 64, [256]), "U3": (200, 16, [192, 256, 384])}
    for job, (states, draws, sensors) in expected.items():
        directory = latest_complete_job(root, job)
        report: dict[str, Any] = {"job": job, "directory": str(directory) if directory else None, "status": "missing"}
        if directory is None:
            errors.append(f"{job}: no complete formal run")
            reports.append(report)
            continue
        table = pd.read_csv(directory / "per_state_field.csv")
        coverage = pd.read_csv(directory / "coverage_by_level.csv")
        expected_rows = states * len(sensors) * 5
        expected_coverage_rows = len(sensors) * 5 * 4
        key_columns = ["state", "field", "sensor_count"]
        coverage_key_columns = ["sensor_count", "field", "nominal_level"]
        report.update({"status": "pass", "rows": len(table), "expected_rows": expected_rows, "coverage_rows": len(coverage), "expected_coverage_rows": expected_coverage_rows, "state_count": int(table["state"].nunique()), "sensor_counts": sorted(table["sensor_count"].astype(int).unique().tolist()), "draw_counts": sorted(table["draw_count"].astype(int).unique().tolist()), "duplicate_keys": int(table.duplicated(key_columns).sum()), "coverage_duplicate_keys": int(coverage.duplicated(coverage_key_columns).sum()), "finite": bool(not table.select_dtypes(include=[np.number]).isna().any().any() and np.isfinite(table.select_dtypes(include=[np.number]).to_numpy()).all()), "coverage_finite": bool(not coverage.select_dtypes(include=[np.number]).isna().any().any() and np.isfinite(coverage.select_dtypes(include=[np.number]).to_numpy()).all())})
        checks = [len(table) == expected_rows, len(coverage) == expected_coverage_rows, report["state_count"] == states, report["sensor_counts"] == sensors, report["draw_counts"] == [draws], report["duplicate_keys"] == 0, report["coverage_duplicate_keys"] == 0, report["finite"], report["coverage_finite"], set(coverage["field"].astype(str)) == {"Y_CH4", "Y_CO", "T", "U1", "p"}, set(coverage["sensor_count"].astype(int)) == set(sensors), set(coverage["nominal_level"].astype(float)) == {0.5, 0.8, 0.9, 0.95}]
        if not all(checks):
            report["status"] = "fail"
            errors.append(f"{job}: row/key/finite gate failed")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("plan_sha256") != plan_sha256 or not manifest.get("identity", {}).get("pass", False):
            report["status"] = "fail"
            errors.append(f"{job}: plan/identity gate failed")
        reports.append(report)
    return reports, errors


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    uq_root = resolve(config["formal_inputs"]["uncertainty_root"])
    uq_reports, errors = audit_uq(uq_root, sha256(args.plan))
    data, records = load_figure5_data(config, REPO_ROOT)
    panel_modes = data["modes"]
    if any(mode != "formal" for mode in panel_modes.values()):
        errors.append("main figure does not resolve all six panels as formal")
    report = {"schema_version": "validation-v2-audit-1", "plan": str(args.plan.resolve()), "uq": uq_reports, "panel_modes": panel_modes, "panel_sources": [record.source for record in records], "errors": errors, "status": "pass" if not errors else "fail"}
    print(json.dumps(report, indent=2, default=str))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

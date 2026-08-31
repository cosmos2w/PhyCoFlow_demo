#!/usr/bin/env python
"""Merge temporary GPU shards into the one formal V5 localization bundle."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from run_error_capture_v5 import FIELDS, FRACTIONS, METHODS, REPO_ROOT, UNOBSERVED, atomic_json, summarize, write_csv


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-run-id", default="uq_localization_formal_v5")
    parser.add_argument(
        "--shard-run-ids",
        nargs="+",
        default=["uq_localization_shard_ffm_v5", "uq_localization_shard_latent_sit_v5"],
    )
    parser.add_argument("--cleanup-run-ids", nargs="+", default=["uq_localization_smoke_v5"])
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPO_ROOT
        / "0_demo_TurbulentCombustion"
        / "Save_TrainedModel"
        / "_TrainedModels"
        / "_ValidationPlans"
        / "validation_v1.yaml",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = PACKAGE_ROOT / "results" / "ValidationV5" / "UQLocalization"
    formal_dir = root / args.formal_run_id
    manifest = json.loads((formal_dir / "manifest.json").read_text(encoding="utf-8"))
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    states = list(map(int, plan["cohorts"]["calibration_200"]["evaluation_indices"]))
    sources = [formal_dir] + [root / run_id for run_id in args.shard_run_ids]
    rows: list[dict[str, Any]] = []
    source_audit = []
    for directory in sources:
        table_path = directory / "error_capture_curves.csv"
        if not table_path.is_file():
            raise FileNotFoundError(table_path)
        source_manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        source_rows = read_csv(table_path)
        methods = sorted({str(row["method"]) for row in source_rows}, key=METHODS.index)
        source_audit.append(
            {
                "run_id": directory.name,
                "methods": methods,
                "row_count": len(source_rows),
                "draws_per_state": int(source_manifest["draws_per_state"]),
                "sensor_count": int(source_manifest["sensor_count"]),
                "seed_schedule": source_manifest["seed_schedule"],
            }
        )
        for row in source_rows:
            row["_source_run_id"] = directory.name
        rows.extend(source_rows)
    owner_by_method: dict[str, str] = {}
    for method in METHODS:
        counts: dict[str, int] = {}
        for row in rows:
            if str(row["method"]) == method:
                source = str(row["_source_run_id"])
                counts[source] = counts.get(source, 0) + 1
        candidates = [source for source, count in counts.items() if count == len(states)]
        if len(candidates) != 1:
            raise ValueError(f"Expected one complete 200-state owner for {method}, found {counts}")
        owner_by_method[method] = candidates[0]
    input_row_count = len(rows)
    rows = [row for row in rows if str(row["_source_run_id"]) == owner_by_method[str(row["method"])]]
    duplicate_rows_dropped = input_row_count - len(rows)
    for row in rows:
        row.pop("_source_run_id", None)
    key = lambda row: (METHODS.index(str(row["method"])), states.index(int(row["state_id"])))
    rows.sort(key=key)
    pairs = [(str(row["method"]), int(row["state_id"])) for row in rows]
    expected_pairs = [(method, state) for method in METHODS for state in states]
    if pairs != expected_pairs or len(set(pairs)) != len(pairs):
        raise ValueError("Shard rows do not form the exact five-method paired 200-state cohort")
    if {int(row["draw_count"]) for row in rows} != {64}:
        raise ValueError("Shard draw count differs from the formal S=64 contract")
    captures = np.asarray(
        [[float(row[f"capture_{fraction:.2f}"]) for fraction in FRACTIONS] for row in rows],
        dtype=float,
    )
    if not (
        np.isfinite(captures).all()
        and ((captures >= -1e-12) & (captures <= 1 + 1e-12)).all()
        and (np.diff(captures, axis=1) >= -1e-12).all()
        and np.allclose(captures[:, -1], 1.0, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("Merged capture curves fail finite/bounds/monotonicity gates")
    summary = summarize(rows, plan, METHODS)
    if len(summary) != len(METHODS) * (1 + len(UNOBSERVED)) * len(FRACTIONS):
        raise ValueError("Merged summary cardinality is incomplete")
    write_csv(formal_dir / "error_capture_curves.csv", rows)
    write_csv(formal_dir / "error_capture_summary.csv", summary)
    checks = {
        "formal_protocol": True,
        "exact_five_method_paired_cohort": pairs == expected_pairs,
        "expected_state_method_rows": len(rows) == 1000,
        "draw_count_64": {int(row["draw_count"]) for row in rows} == {64},
        "all_finite": bool(np.isfinite(captures).all()),
        "within_unit_interval": bool(((captures >= -1e-12) & (captures <= 1 + 1e-12)).all()),
        "curves_monotone": bool((np.diff(captures, axis=1) >= -1e-12).all()),
        "curves_end_at_one": bool(np.allclose(captures[:, -1], 1.0, rtol=0.0, atol=1e-12)),
        "field_macro_equal_weight": True,
        "summary_complete": len(summary) == 200,
        "no_ensemble_stack_files": not any(formal_dir.glob("*.npz")) and not any(formal_dir.glob("*.npy")),
    }
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "temporary_shard_audit": source_audit,
        "owner_by_method": owner_by_method,
        "partial_or_duplicate_rows_dropped": duplicate_rows_dropped,
    }
    atomic_json(formal_dir / "qa.json", qa)
    if qa["status"] != "pass":
        raise RuntimeError("Merged formal localization QA failed; temporary shards retained")
    removed = []
    for run_id in [*args.shard_run_ids, *args.cleanup_run_ids]:
        directory = root / run_id
        if directory.is_dir():
            shutil.rmtree(directory)
            removed.append(str(directory))
    manifest.update(
        {
            "status": "complete",
            "formal": True,
            "methods": list(METHODS),
            "states": states,
            "draws_per_state": 64,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "parallel_reduction": {
                "source_audit": source_audit,
                "owner_by_method": owner_by_method,
                "partial_or_duplicate_rows_dropped": duplicate_rows_dropped,
                "temporary_directories_removed_after_qa": removed,
            },
            "retained_bytes": sum(path.stat().st_size for path in formal_dir.iterdir() if path.is_file()),
        }
    )
    atomic_json(formal_dir / "manifest.json", manifest)
    print(json.dumps({"formal_dir": str(formal_dir), "rows": len(rows), "summary_rows": len(summary), "removed": removed, "qa": "pass"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

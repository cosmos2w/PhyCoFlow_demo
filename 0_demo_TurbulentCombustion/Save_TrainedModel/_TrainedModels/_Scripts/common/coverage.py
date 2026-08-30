"""Expected-cache coverage and conservative aggregate-status helpers.

Exporters must not silently summarize a partly written reconstruction cache as
if it represented the complete requested test split.  The sensor plan is the
authoritative record of condition/snapshot coverage and is deliberately read
without loading a dataset or a model.
"""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Iterable


def expected_snapshots_by_condition(manifest: Iterable[dict], conditions: Iterable[str]) -> dict[str, list[int]]:
    """Return sensor-plan snapshot IDs for each requested condition.

    Falls back to IDs already present in a legacy manifest when its saved
    sensor-plan path is unavailable.  This preserves old smoke artifacts while
    keeping modern full-test runs auditable.
    """
    rows = list(manifest)
    wanted = {str(condition) for condition in conditions}
    plan_path = next((Path(row["sensor_plan"]) for row in rows if row.get("sensor_plan") and Path(row["sensor_plan"]).exists()), None)
    values = {condition: set() for condition in wanted}
    if plan_path is not None:
        with plan_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                condition = row.get("condition")
                if condition in values:
                    values[condition].add(int(row["snapshot"]))
    else:
        for row in rows:
            condition = row.get("condition")
            if condition in values:
                try:
                    values[condition].add(int(row["snapshot"]))
                except (KeyError, TypeError, ValueError):
                    continue
    return {condition: sorted(snapshots) for condition, snapshots in values.items()}


def aggregate_status(rows: Iterable[dict], valid_n: int, expected_n: int) -> str:
    """Return ``ok`` only for complete coverage; retain useful failure causes."""
    if expected_n and valid_n >= expected_n:
        return "ok"
    if valid_n:
        return "incomplete cache"
    statuses = Counter(str(row.get("status", "missing cache")) for row in rows)
    for candidate in ("missing dependency", "load error", "inference error", "missing checkpoint", "missing config", "missing directory"):
        if candidate in statuses:
            return candidate
    return next(iter(statuses), "missing cache")

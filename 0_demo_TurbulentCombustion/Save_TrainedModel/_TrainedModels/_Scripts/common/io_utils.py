"""Artifact naming and tabular I/O helpers."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def artifact_name(prefix: str, run_id: str, suffix: str) -> str:
    return f"{prefix}_{run_id}.{suffix.lstrip('.')}"


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
    return path


def latest(directory: Path, prefix: str, suffix: str) -> Path:
    paths = sorted(directory.glob(f"{prefix}_*.{suffix}"))
    if not paths:
        raise FileNotFoundError(f"No {prefix}_*.{suffix} under {directory}")
    return paths[-1]


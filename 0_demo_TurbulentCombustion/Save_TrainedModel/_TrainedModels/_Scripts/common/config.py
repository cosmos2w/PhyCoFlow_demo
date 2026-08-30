"""Configuration, paths, reproducibility, and CLI helpers."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

SCRIPT_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = SCRIPT_DIR.parent
DEMO_DIR = ARCHIVE_DIR.parents[1]
SRC_DIR = DEMO_DIR / "src"
RESULTS_DIR = ARCHIVE_DIR / "_Process_Results"
FIGURES_DIR = ARCHIVE_DIR / "_Process_Figures"
DEFAULT_CONFIG = SCRIPT_DIR / "postprocess_config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the editable YAML configuration and attach resolved internal paths."""
    cfg_path = Path(path or DEFAULT_CONFIG).resolve()
    with cfg_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    cfg["_config_path"] = str(cfg_path)
    cfg["_archive_dir"] = str(ARCHIVE_DIR)
    return cfg


def run_id(value: str | None = None) -> str:
    """Return an explicit run ID or a sortable local timestamp."""
    return value or datetime.now().strftime("%Y%m%d_%H%M%S")


def stable_seed(base: int, *parts: object) -> int:
    payload = "|".join(map(str, (base, *parts))).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) & 0x7FFFFFFF


def short_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def ensure_output_dirs() -> None:
    for rel in (
        "ModelInventory", "SensorPlans", "ReconstructionCache", "FieldL2",
        "JointPDF", "JointPDF_JSD", "Spectral/EnergySpectra", "Spectral/SpectralLSD",
    ):
        (RESULTS_DIR / rel).mkdir(parents=True, exist_ok=True)
    for rel in (
        "_Contours", "FieldL2", "JointPDF", "JointPDF_JSD",
        "ConditionMatrix", "Assembled", "Spectral/EnergySpectra", "Spectral/SpectralLSD", "Spectral/Composite",
    ):
        (FIGURES_DIR / rel).mkdir(parents=True, exist_ok=True)


def method_items(cfg: dict[str, Any], selected: Iterable[str] | None = None):
    wanted = None if selected is None else {str(x).lower() for x in selected}
    for method in cfg["methods"]:
        keys = {method["name"].lower(), method["directory"].lower(), *(a.lower() for a in method.get("aliases", []))}
        if wanted is None or "all" in wanted or keys & wanted:
            yield method


def add_common_args(parser: argparse.ArgumentParser, *, models: bool = True) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Post-processing YAML.")
    parser.add_argument("--run-id", default=None, help="Explicit timestamp/run ID shared by related artifacts.")
    if models:
        parser.add_argument("--models", nargs="+", default=["all"], help="Method names/aliases or 'all'.")


def select_snapshots(length: int, requested: list[int] | None, maximum: int | None) -> list[int]:
    values = list(range(length)) if requested is None else [i for i in requested if 0 <= i < length]
    return values if maximum is None else values[:maximum]
